# coding: utf-8

import dataclasses as DC

import types
import torch
import torchvision
from browserlm import BertModel


@DC.dataclass
class ModelConfig:

    _desc = 'Configuration for model'

    bert_path: str = DC.field(default='models/uncased_L-4_H-512_A-8_torch',
            metadata={'help':'a path to a pretrained bert.'})

    tokenizer_path: str = DC.field(default='models/uncased_L-4_H-512_A-8_torch/tokenizer.json',
            metadata={'help':'path to a pretrained tokenizer.'})

    max_tokens: int = DC.field(default=256,
            metadata={'help':'maximum steps in an example.'})
    
    use_double_bert: int = DC.field(default=0,
            metadata={'help':'if set 1, the model will use two bert mode.'})

    loss_ignored_index: int = DC.field(default=-1,
            metadata={'help':'ignore id used when calculationg cross entropy loss'})
    
    image_input_size: str = DC.field(default='',
            metadata={'help':'if set two numbers separated by comma, model will use image'})
    
    optimize_resnet: int = DC.field(default=0,
            metadata={'help':'if set 1, resnet will be optimied when trianing.'})


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

    
class BertClassifier(torch.nn.Module):

    def __init__(self, config, n_classes):

        super().__init__()

        self.n_classes = n_classes
        self.config = config
        self.use_image_input = bool(config.image_input_size != '')
        self.image_size = (0, 0)
        if self.use_image_input:
            self.image_size = tuple(int(_) for _ in config.image_input_size.split(','))
        self.optimize_resnet = bool(config.optimize_resnet)
        self.use_double_bert = bool(config.use_double_bert)
        
        if self.use_double_bert:
            self.bert_encoder = BertModel.from_dir(config.bert_path)
            self.bert_encoder.requires_grad_(False)
        
        self.bert = BertModel.from_dir(config.bert_path)
        self.base_dim = self.bert.config.hidden_size
        
        if self.use_image_input:
            # ResNet for screenshots
            self.resnet = torchvision.models.resnet18(pretrained=True)
            # Replace forward function to get feature maps
            self.resnet._forward_impl = types.MethodType(_resnet_forward_impl, self.resnet)
            self.resnet.requires_grad_(self.optimize_resnet)
            # linear layer to adjust dimensions
            self.resnet_norm = Normalizer(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.resnet_conv = torch.nn.Conv2d(self.resnet.fc.in_features, self.base_dim, 1, 1)
        
        self.output_linear = torch.nn.Linear(self.base_dim, self.n_classes, bias=True)

        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=config.loss_ignored_index, reduction='mean')
        self.reset_parameters()
        
    def train(self, mode=True):
        
        super().train(mode)
        
        # Non-trainable modules
        # We set eval mode to fix those modules including batch normalization, etc.
        if self.use_image_input and (not self.optimize_resnet):
            self.resnet.eval()
        
        if self.use_double_bert:
            self.bert_encoder.eval()
        
        return self

    def reset_parameters(self):

        pass
    
    def _image_embedding(self, x):
        """images: (mb_size, height, width, channel)
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
    
    def forward(self,
            token_ids,
            images=None,
            requires_loss=False,
            labels=None,
        ):
        
        if not self.use_image_input:
            assert images is None, 'images are input, but use_image is False.'
        
        mask = (token_ids != 0)
        
        if self.use_double_bert:
            inputs_embeds = self.bert_encoder(token_ids, attention_mask=mask).last_hidden_state
        else:
            inputs_embeds = self.bert.embeddings.word_embeddings(token_ids)
        
        if self.use_image_input and (images is not None):
            # we place image seq before [cls] for simple logic
            inputs_images = self._image_embedding(images)
            inputs_embeds = torch.cat([ inputs_images, inputs_embeds], axis=1)
            mb_size, len_image_seq, _ = inputs_images.shape
            image_mask = torch.ones((mb_size, len_image_seq), dtype=torch.bool, device=mask.device)
            mask = torch.cat([image_mask, mask], axis=1)
            del image_mask
        
        last_hidden_state = self.bert(inputs_embeds=inputs_embeds, attention_mask=mask).last_hidden_state
        class_logits = self.output_linear(last_hidden_state[:, 0])

        outputs = {}
        outputs['class_logits'] = class_logits

        if requires_loss:
            loss_reults = self.loss_fn(class_logits, labels)
            outputs['loss'] = loss_reults

        return outputs


class BertExtractor(torch.nn.Module):

    def __init__(self, config):

        super().__init__()
        
        self.n_classes = 2
        self.config = config
        self.use_double_bert = config.use_double_bert
        
        if self.use_double_bert:
            self.bert_encoder = BertModel.from_dir(config.bert_path)
            self.bert_encoder.requires_grad_(False)
        
        self.bert = BertModel.from_dir(config.bert_path)
        self.base_dim = self.bert.config.hidden_size
        
        self.output_linear = torch.nn.Linear(self.base_dim, self.n_classes, bias=True)
        
        self.reset_parameters()
    
    def train(self, mode=True):
        
        super().train(mode)
        if self.use_double_bert:
            self.bert_encoder.eval()
        return self

    def reset_parameters(self):

        pass

    def forward(self,
            token_ids,
            images=None,
            requires_loss=False,
            labels=None,
        ):
        
        assert images is None, 'Not implemented image input for BertExtractor'
        
        mask = (token_ids != 0)
        
        if self.use_double_bert:
            inputs_embeds = self.bert_encoder(token_ids, attention_mask=mask).last_hidden_state
        else:
            inputs_embeds = self.bert.embeddings.word_embeddings(token_ids)
        
        last_hidden_state = self.bert(inputs_embeds=inputs_embeds, attention_mask=mask).last_hidden_state
        start_logits, end_logits = self.output_linear(last_hidden_state).split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = {}
        outputs['start_logits'] = start_logits
        outputs['end_logits'] = end_logits
        
        if requires_loss:
                
            ignored_index = start_logits.shape[1]
            
            start_labels, end_labels = labels
            start_labels = start_labels.clamp(0, ignored_index)
            end_labels = end_labels.clamp(0, ignored_index)
            
            loss_func = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)
            
            start_loss = loss_func(start_logits, start_labels)
            end_loss = loss_func(end_logits, end_labels)
            loss_results = start_loss + end_loss
            outputs['loss'] = loss_results
        
        return outputs

