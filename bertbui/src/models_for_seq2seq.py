# coding: utf-8
# Encoder-decoder model with a vision input for multi task learning
# 2021-12-2
# Taichi Iki
# This script relies on Huggingface's transformers


import dataclasses as DC

import types
import torch
import torchvision
import transformers


@DC.dataclass
class ModelConfig(object):
    
    _desc = 'Configuration for model'
    
    use_t5 : int = DC.field(default=0,
            metadata={'help':'if not zero, model will use t5'})
    t5_path : str = DC.field(default='t5-small',
            metadata={'help':'a path to a pretrained t5'})
    encoder_bert : str = DC.field(default='google/bert_uncased_L-4_H-512_A-8',
            metadata={'help':'a path to a pretrained bert for the encoder'})
    decoder_bert : str = DC.field(default='google/bert_uncased_L-4_H-512_A-8',
            metadata={'help':'a path to a pretrained bert for the decoder'})
    
    input_max_l_tokens_l : int = DC.field(default=512,
            metadata={'help':'maximun token length for language in language only contexts'})
    input_max_l_tokens_vl : int = DC.field(default=432,
            metadata={'help':'maximun token length for language in vision and language contexts'})    
    output_max_tokens : int = DC.field(default=128,
            metadata={'help':'maximun token length for output'})    
    
    image_size : str = DC.field(default='320,224',
            metadata={'help':'expected image size; width,height'})                
    optimize_resnet : int = DC.field(default=0,
            metadata={'help':'whether resnet will be optimize or not (frozen)'})
    
    def load_tokenizer(self):
        if not self.use_t5:
            return transformers.AutoTokenizer.from_pretrained(self.encoder_bert)
        else:
            return transformers.T5Tokenizer.from_pretrained(self.t5_path)


def _resnet_forward_impl(self, x):
    
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
        
    return x


class Normalizer(torch.nn.Module):
    
    def __init__(self, mean, std):
        
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean))
        self.register_buffer('std', torch.tensor(std))
        self.dim = (self.mean.shape[0],)
        
    def forward(self, x):
        
        shape = (1,) * (len(x.shape) - 1) + self.dim
        return (x - self.mean.view(shape)) / self.std.view(shape)
    

class SeqSeqModelWithVision(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.image_size = tuple(int(_) for _ in config.image_size.split(','))
        self.optimize_resnet = bool(config.optimize_resnet)
        self.use_t5 = bool(config.use_t5)
        
        if not self.use_t5:
            self.bert = transformers.EncoderDecoderModel. \
                from_encoder_decoder_pretrained(config.encoder_bert, config.decoder_bert)
            self.bert.config.vocab_size = self.bert.config.decoder.vocab_size
            self.base_dim = self.bert.config.encoder.hidden_size
        else:
            self.bert = transformers.T5ForConditionalGeneration.from_pretrained(config.t5_path)
            self.base_dim = self.bert.config.hidden_size
        
        self.resnet = torchvision.models.resnet18(pretrained=True)
        # Replace forward function to get feature maps
        self.resnet._forward_impl = types.MethodType(_resnet_forward_impl, self.resnet)
        self.resnet.requires_grad_(self.optimize_resnet)
        # Normalization factors are specific to the resnet18 of pytorch
        self.resnet_norm = Normalizer(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # Linear layer to adjust dimensions
        self.resnet_conv = torch.nn.Conv2d(self.resnet.fc.in_features, self.base_dim, 1, 1)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        
        pass
    
    def train(self, mode=True):
        
        super().train(mode)
    
        # Non-trainable modules
        # We set eval mode to fix those modules including batch normalization, etc.
        if not self.optimize_resnet:
            self.resnet.eval()
                
        return self
    
    def adapt_tokenizer(self, tokenizer):
        # Call before training or inference with a tokenizer to be used
        
        self.pad_token_id = self.bert.config.pad_token_id = tokenizer.pad_token_id
        if not self.use_t5:
            self.bert.config.decoder_start_token_id = tokenizer.cls_token_id
            self.eos_token_id = tokenizer.sep_token_id
        else:
            self.eos_token_id = tokenizer.eos_token_id
        
        return self
    
    def _image_embedding(self, x):
        """x: a image tensor (N, height, width, 3)
        The size of the image tensor must match self.image_size
        """
        base_dtype = torch.float16 if torch.is_autocast_enabled() else torch.float32
        mb_size, height, width, n_channel = x.shape
        
        x = x.to(base_dtype) / 255 # uint to float
        x = self.resnet_norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.resnet(x)
        x = self.resnet_conv(x) # (n_total, self.base_dim, height, width)
        x = x.permute(0, 2, 3, 1).reshape(mb_size, -1, self.base_dim)
        
        return x
    
    def _make_inputs_embeds_and_mask(self, input_ids, input_images, input_images_enabled):
        
        mb_size = input_ids.shape[0]
        device = input_ids.device
        is_not_padded = (input_ids != self.pad_token_id)
        
        # goal of inputs_embeds: [I-1] [I-2]  ... [I-WH] [CLS] [L-1] ... [L-N] [SEP]
        # We expect that input_ids starts with [CLS] and ends with [SEP]
        # For the examples where image is not used, [I-1] [I-2]  ... [I-WH] will be omitted.
        
        # language embedding
        if not self.use_t5:
            emb_lang = self.bert.encoder.embeddings.word_embeddings(input_ids)
        else:
            emb_lang = self.bert.shared(input_ids)
        
        inputs_embeds = None
        mask = None
        
        if input_images is None:
            # images are not used.
            inputs_embeds = emb_lang
            mask = is_not_padded
        else:
            # input_images_enabled is None, all images are considered enabled.
            if input_images_enabled is None:
                input_images_enabled = torch.ones((mb_size,), device=device, dtype=torch.bool)
            
            emb_img = self._image_embedding(input_images)
            
            len_emb_img = emb_img.shape[1]
            len_emb_total = input_images_enabled*len_emb_img + is_not_padded.sum(axis=1)
            overall_len = len_emb_total.max().item()
            delta_len = overall_len - len_emb_img
            
            m = input_images_enabled[:, None, None]
            pad = torch.zeros((mb_size, delta_len, emb_img.shape[-1]), device=device)
            inputs_embeds = torch.cat([m*emb_img,  pad], axis=1)
            # inputs_embeds: (mb_size, overall_len)
            
            inputs_embeds[:, :emb_lang.shape[1]] += (~m)*emb_lang # w/o image sequences
            inputs_embeds[:, -delta_len:] +=m*(emb_lang[:, :delta_len]) # w/ image sequences
            
            mask = torch.arange(overall_len, device=device)[None] < len_emb_total[:, None]
        
        return inputs_embeds, mask
    
    def _prepare_decoder_input_ids_for_generation(self, input_tensor, decoder_start_token_id=None, bos_token_id=None):
        decoder_start_token_id = self.bert._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
        decoder_input_ids = (
            torch.ones((input_tensor.shape[0], 1), dtype=torch.long, device=input_tensor.device) * decoder_start_token_id
        )
        return decoder_input_ids
    
    def generate(self, input_ids=None, input_images=None, input_images_enabled=None, **kwargs):
        
        assert 'inputs_embeds' not in kwargs, 'This model does not accept inputs_embeds directly'
        assert 'attention_mask' not in kwargs, 'This model does not accept attention_mask directly'
        
        inputs_embeds, mask = \
                self._make_inputs_embeds_and_mask(input_ids, input_images, input_images_enabled)
        
        kwargs['inputs_embeds'] = inputs_embeds
        kwargs['attention_mask'] = mask
        kwargs['decoder_input_ids'] = self._prepare_decoder_input_ids_for_generation(inputs_embeds)
        if 'eos_token_id' not in kwargs:
            kwargs['eos_token_id'] = self.eos_token_id
        if 'max_length' not in kwargs:
            kwargs['max_length'] = self.config.output_max_tokens
        
        return self.bert.generate(**kwargs)  
    
    def forward(self, input_ids=None, input_images=None, input_images_enabled=None, labels=None):
        
        inputs_embeds, mask = \
                self._make_inputs_embeds_and_mask(input_ids, input_images, input_images_enabled)
        
        # to change the ids of padded tokens into ignore_index for the loss function. 
        labels.masked_fill_(labels == self.pad_token_id, -100)
        outputs = self.bert(inputs_embeds=inputs_embeds, attention_mask=mask, labels=labels)
        return outputs
