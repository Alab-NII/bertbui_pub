# coding: utf-8


import dataclasses as DC

import types
import math
import time
import PIL.Image, PIL
import numpy as np
import json
import os

import torch
import torchvision
import tokenizers

from bertbui import BertModel, BertConfig, ModelAction


@DC.dataclass
class ModelConfig:
    
    _desc = 'Configuration for model'
    
    bert_path: str = DC.field(default='models/uncased_L-4_H-512_A-8_torch',
            metadata={'help':'a path to a pretrained bert.'})
    
    bert_layers: str = DC.field(default='',
            metadata={'help':'Empty string or number,{top/bottom}. If set a value, top/bottom k layers are used for the lang bert'})
    
    lang_bert_path: str = DC.field(default='models/uncased_L-4_H-512_A-8_torch',
            metadata={'help':'a path to a pretrained bert for language encoder.'})
    
    lang_bert_layers: str = DC.field(default='',
            metadata={'help':'Empty string or number,{top/bottom}. If set a value, top/bottom k layers are used for the lang bert'})
    
    tokenizer_path: str = DC.field(default='models/uncased_L-4_H-512_A-8_tokenizer.json', 
            metadata={'help':'path to a pretrained tokenizer.'})
        
    memory_size: int = DC.field(default=64, 
            metadata={'help':'the amount of embeddings used for memory.'})
    
    max_tokens: int = DC.field(default=300, 
            metadata={'help':'maximum tokens in a screen.'})
    
    tie_weight: int = DC.field(default=1, 
            metadata={'help':'If 1 output weights are tied with input embeddings.'})
    
    use_language_bert: int = DC.field(default=1, 
            metadata={'help':'If 1 we use fixed bert for sentence encoding.'})
    
    add_fusion_embedding: int = DC.field(default=1, 
            metadata={'help':'If 1 we add fusion embedding to the embedding of the language bert.'})
    
    window_size: tuple = DC.field(default=(640, 448), 
            metadata={'help':'window size, width and height, in px.'})
    
    input_window_size: tuple = DC.field(default=(320, 224), 
            metadata={'help':'window size, width and height, in px, to be input into resnet.'})
    
    actions_str: str = DC.field(default=' '.join(ModelAction.NAMES),
            metadata={'help':'defined actions'})
    
    loss_ignored_index: int = DC.field(default=-1, 
            metadata={'help':'ignore id used when calculationg cross entropy loss'})
    
    additive_coordinate: int = DC.field(default=1, 
            metadata={'help':'If 1, coordinate embeddings are added to token ones'})
        
    coordinate_depth: int = DC.field(default=3, 
            metadata={'help':'The depth of the MLP for coordinate embeddings'})
    
    @property
    def requires_resize(self):
        return not (
            self.window_size[0]==self.input_window_size[0] and \
            self.window_size[1]==self.input_window_size[1]
        )
    
    @property
    def actions(self):
        return self.actions_str.split()
    
    @property
    def action_to_id(self):
        actions = self.actions_str.split()
        return {k:i for i, k in enumerate(actions)}
    
    def get_id_x(self, v):
        # We use 0 as pad, so we shift non-pad value by adding one
        # 0 -> 1, 1->2, ... width-1 -> width
        # we truncate negative values to 0 and excess width-1 to width-1.
        if v is None:
            return 0
        v = int(round(v))
        
        lb = 0 # included, mapped to 1
        ub = self.window_size[0] - 1# included
        v = lb if v < lb else (v if v < ub else ub)
        return v + (-lb + 1)
        
    def get_id_y(self, v):
        if v is None:
            return 0
        v = int(round(v))
        
        lb = 0 # included, mapped to 1
        ub = self.window_size[1] - 1# included
        v = lb if v < lb else (v if v < ub else ub)
        return v + (-lb + 1)


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

    
class CoordinateEmbedder(torch.nn.Module):
    
    def __init__(self, d_base, nx, ny, depth=1):
        
        super().__init__()
        layers = [torch.nn.Linear(4, d_base, bias=(depth==1))]
        for d in range(depth - 1):
            layers.append(torch.nn.BatchNorm1d(d_base))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(d_base, d_base, bias=(d==depth-2)))
        self.linear = torch.nn.Sequential(*layers)
        self.register_buffer('size_factor', torch.tensor([nx, ny, nx, ny])[None])
        self.base_dim = d_base

    def forward(self, x):
        """
        x: tensor with the shape [mb_size, steps, 2 or 4 (cx cy w h)]
        """
        mb_size, n_steps, dim = x.shape
        if dim == 2:
            # considered that the input is (x, y)
            # add (width, height)
            x = torch.cat([x, torch.ones_like(x)], axis=-1)
        y = self.linear((x.view(mb_size*n_steps, 4) - 1) / self.size_factor)
        y = y.view(mb_size, n_steps, self.base_dim)
        
        # filter padded elements
        y.masked_fill_((x <= 0).any(dim=-1, keepdim=True), 0)
        
        return y


class ReaderModule(torch.nn.Module):
    
    ACTION_KEYS = ('name', 'token', 'x', 'y')
    
    def __init__(self, config, pretrained_model_path=None):
        
        super().__init__()
        
        # Configuration
        self.config = config
        self.tie_weight = bool(config.tie_weight)
        self.use_language_bert = bool(config.use_language_bert)
        self.add_fusion_embedding = bool(config.add_fusion_embedding)
        tokenizer_path = config.tokenizer_path
        if pretrained_model_path:
            tokenizer_path = os.path.join(pretrained_model_path, 'tokenizer.json')
        self.tokenizer = tokenizers.Tokenizer.from_file(tokenizer_path)
        self.action_to_id = self.config.action_to_id
        self._get_id_x = config.get_id_x
        self._get_id_y = config.get_id_y
        
        self.additive_coordinate = bool(config.additive_coordinate)
        
        self.action_size = len(config.actions)
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.nx, self.ny = config.window_size
        self.input_window_size = config.input_window_size
        self.output_size = len(self.ACTION_KEYS)
        
        self.memory_size = config.memory_size
        self.half_memory_size = self.memory_size // 2
        self.use_memory = self.memory_size > 0
        
        self.loss_ignored_index = config.loss_ignored_index
        
        # ToDo, change to not depend on config.bert_path after training
        # BERT for tokens
        if self.use_language_bert:
            _config = None
            if pretrained_model_path:
                _config = BertConfig(**json.load(open(os.path.join(pretrained_model_path, 'lang_bert_config.json'))))
            self.lang_bert = self.load_bert_custom(config.lang_bert_path, config.lang_bert_layers, _config)
            self.lang_bert.requires_grad_(False)
        
        # BERT for action genenration
        _config = None
        if pretrained_model_path:
            _config = BertConfig(**json.load(open(os.path.join(pretrained_model_path, 'bert_config.json'))))
        self.fusion_bert = self.load_bert_custom(config.bert_path, config.bert_layers, _config)
        self.fusion_bert.requires_grad_(True)
        self.base_dim = self.fusion_bert.config.hidden_size
        
        # ResNet for screenshots
        self.resnet = torchvision.models.resnet18(pretrained=True)
        # Replace forward function to get feature maps
        self.resnet._forward_impl = types.MethodType(_resnet_forward_impl, self.resnet)
        # Make parameters frozen
        self.resnet.requires_grad_(False)
        # linear layer to adjust dimensions
        self.resnet_norm = Normalizer(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.resnet_conv = torch.nn.Conv2d(self.resnet.fc.in_features, self.base_dim, 1, 1)
        
        # Embeddings
        self.emb_action = torch.nn.Embedding(self.action_size, self.base_dim, padding_idx=0)
        # we use 1,2,...,max_of_an_axis for coordinates, 0 is reserved for padding
        # Make sure that the values of x, y, width, height that ranged (0, max-1) are incremented by one.  
        self.emb_coordinate = CoordinateEmbedder(
            self.base_dim, self.nx, self.ny, 
            config.coordinate_depth
        )
        
        # Memory
        if self.use_memory:
            self.inital_memory = torch.nn.Parameter(torch.empty((1, self.half_memory_size, self.base_dim)))
            self.memory_write_block = torch.nn.Parameter(torch.empty((1, self.half_memory_size, self.base_dim)))
        
        # Outputs
        self.output_tokens = torch.nn.Parameter(torch.empty((1, self.output_size, self.base_dim)))
        self.output_linears = torch.nn.ModuleDict({
            'name': torch.nn.Linear(self.base_dim, self.action_size, bias=True),
            'token': torch.nn.Linear(self.base_dim, self.vocab_size, bias=True),
            'x': torch.nn.Linear(self.base_dim, self.nx+1, bias=True),
            'y': torch.nn.Linear(self.base_dim, self.ny+1, bias=True),
        })
        if self.tie_weight:
            self.output_linears['name'].weight = self.emb_action.weight
            self.output_linears['token'].weight = self.fusion_bert.embeddings.word_embeddings.weight
        
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=config.loss_ignored_index, reduction='mean')
        self.reset_parameters()
    
    def base_bert_config(self, model_path):
        
        with open(os.path.join(model_path, 'bert_config.json'), 'w') as f:
            json.dump(self.fusion_bert.config.__dict__,  f)
        
        if self.use_language_bert:
            with open(os.path.join(model_path, 'lang_bert_config.json'), 'w') as f:
                json.dump(self.lang_bert.config.__dict__,  f)
    
    @staticmethod
    def load_bert_custom(bert_path, new_num_layers, config):
        from collections import OrderedDict
        import re
        
        if config:
            bert = BertModel(config)
        else:
            bert = BertModel.from_dir(bert_path)
        
        if len(new_num_layers) == 0:
            return bert
        
        spec = new_num_layers.split(',')
        new_num_layers = int(spec[0])
        order = 'bottom' if len(spec) == 1 else spec[1]
        
        old_num_layers = bert.config.num_hidden_layers
        assert new_num_layers <= old_num_layers, \
            'new_num_layers should not be larger than old_num_layers'
        
        delete_prefix = []
        rewrite_prefix = {}
        if order == 'top':
            for l in range(old_num_layers):
                old_prefix = f'encoder.layer.{l}.'
                new_l = l - (old_num_layers - new_num_layers)
                new_prefix = f'encoder.layer.{new_l}.' if 0 <= new_l and new_l < new_num_layers else None
                if new_prefix is None:
                    delete_prefix.append(old_prefix)
                else:
                    rewrite_prefix[old_prefix] = new_prefix
        elif order == 'bottom':
            for l in range(old_num_layers):
                old_prefix = f'encoder.layer.{l}.'
                if l >= new_num_layers:
                    delete_prefix.append(old_prefix)
        else:
            raise RuntimeError(f'unknown order for bert loading {order}')
        
        re_delete = None
        if delete_prefix:
            re_delete = re.compile('|'.join(re.escape(_) for _ in delete_prefix))
        
        def key_map(kv):
            k, v = kv
            if delete_prefix and re_delete.match(k):
                return None
            for p, pn in rewrite_prefix.items():
                if k.startswith(p):
                    k = k.replace(p, pn, 1)
                    break
            return k, v
        state_dict = OrderedDict(
            kv for kv in map(key_map, bert.state_dict().items()) if kv is not None
        )
        
        new_bert = BertModel(BertConfig(
            **dict(bert.config.__dict__, **{'num_hidden_layers':new_num_layers}))
        )
        new_bert.load_state_dict(state_dict)
        return new_bert
    
    def train(self, mode=True):
        super().train(mode)
        # Non-trainable modules
        # We set eval mode to fix those modules including batch normalization, etc.
        self.resnet.eval()
        if self.use_language_bert:
            self.lang_bert.eval()
        return self
    
    def reset_parameters(self):
        
        if not self.tie_weight:
            self.output_linears['name'].weight.data = self.emb_action.weight.data.detach().clone()
            self.output_linears['token'].weight.data = \
                self.fusion_bert.embeddings.word_embeddings.weight.data.detach().clone()
        
        weight_data = self.fusion_bert.embeddings.word_embeddings.weight.data
        cls_emb = weight_data[self.tokenizer.token_to_id('[CLS]')][None, None]
        mask_emb = weight_data[self.tokenizer.token_to_id('[MASK]')][None, None]
        self.output_tokens.data = torch.cat([cls_emb, mask_emb, cls_emb, cls_emb], axis=1).detach().clone()
        
        if self.use_memory:
            self.inital_memory.data = torch.zeros((1, self.half_memory_size, self.base_dim), dtype=weight_data.dtype, device=weight_data.device)
            self.memory_write_block.data = cls_emb.repeat(1, self.half_memory_size, 1).detach().clone()
    
    def get_device(self):
        
        return self.emb_action.weight.device
    
    def _seq_last_action(self, x, required_labels, base_dtype):
        
        # x: (mb_id, step_id, [name, token, x, y])
        mb_size, n_steps = x.shape[:2]
        n_total = mb_size * n_steps
        
        labels= None
        if required_labels:
            n_steps -= 1
            n_total -= mb_size
            labels = x[:, 1:].detach().clone().reshape(n_total, self.output_size)
            x = x[:, :-1]
        x = x.reshape(n_total, 1, self.output_size)
        
        seq_action_name = self.emb_action(x[:, :, 0])
        
        y = x[:, :, 1]
        seq_action_word = self.fusion_bert.embeddings.word_embeddings(y)
        seq_action_word.masked_fill_(y[:, :, None] == 0, 0)
        
        y = x[:, :, 2:]
        seq_action_xy = self.emb_coordinate(y.to(base_dtype))
        
        return seq_action_name, seq_action_word, seq_action_xy, labels
    
    def _seq_screenshot(self, x, base_dtype):
        """
        (batch_size, height, width, channel) -> (batch_size, height//32*width//32, base_dim)
        """
        mb_size, n_steps, height, width, n_channel = x.shape
        n_total = mb_size * n_steps
        
        x = x.to(base_dtype) / 255 # uint to base_dtype
        x = self.resnet_norm(x)
        x = x.permute(0, 1, 4, 2, 3).reshape(n_total, n_channel, height, width)
        x = self.resnet(x)
        x = self.resnet_conv(x) # (n_total, self.base_dim, height, width)
        x = x.permute(0, 2, 3, 1).reshape(n_total, -1, self.base_dim)
        
        return x
    
    def _seq_token(self, x, base_dtype):
        
        mb_size, n_steps, n_tokens = x.shape[:3]
        n_total = mb_size * n_steps
        
        x = x.view(n_total, n_tokens, 5) #  5 accounts for (id, cx, cy, width, height)
        token_id = x[:, :, 0]
        token_mask = (token_id != 0)
        if self.use_language_bert:
            seq_token = self.lang_bert(input_ids=token_id, attention_mask=token_mask).last_hidden_state
            if self.add_fusion_embedding:
                seq_token += self.fusion_bert.embeddings.word_embeddings(token_id)
                seq_token *= 0.5
        else:
            seq_token = self.fusion_bert.embeddings.word_embeddings(token_id)
        coordinate = self.emb_coordinate(x[:, :, 1:].to(base_dtype))
        
        if self.additive_coordinate:
            seq_token += coordinate
        else:
            seq_token = torch.cat((seq_token, coordinate), axis=-1).view(n_total, 2*n_tokens, -1)
            token_mask = token_mask[:,:,None]
            token_mask = torch.cat((token_mask, token_mask), axis=-1).view(n_total, 2*n_tokens)
        
        return seq_token, token_mask
    
    def forward(self, 
            screenshots,
            tokens,
            last_actions,
            memory=None,
            requires_loss=False,
            is_ignored=None,
            is_reset=None,
        ):
        """
        Argments:
            screenshots:
                Tensor (batch_size, time_steps, height, width, channel=(R,G,B))
            tokens: 
                Tensor (batch_size, time_steps, sequence_length, channel=(id, cx, cy, height, width))
            last_actions: 
                When requires_loss is False: Tensor (batch_size, time_steps, channel=(name, token, x, y))
                When requires_loss is True: Tensor (batch_size, time_steps+1, channel=(name, token, x, y)
            memory:
                Tensor (batch_size, memory_number)
                Zero initialized when none is given.
            requires_loss:
                Bool
            is_ignored: 
                Tensor (batch_size, time_steps)
                Not used when requires_loss is False. Zero initialized when none is given.
        Returns:
            
        """
        is_mem_sep = (isinstance(memory, str) and memory == 'separate') # currently not used
        
        base_dtype = torch.float16 if torch.is_autocast_enabled() else torch.float32
        device = self.get_device()
        mb_size, n_steps, max_seq_len = tokens.shape[:3]
        n_total = mb_size*n_steps
        
        seq = []
        # input sequence = next action + last actions + Screenshots + Tokens
        
        if self.use_memory:
            # memory write block
            seq.append(self.memory_write_block.repeat(n_total, 1, 1))
        
        # next action
        output_tokens = self.output_tokens.repeat(n_total, 1, 1)
        seq.append(output_tokens) # Shape:  (n_total, output_size, base_dim)
        
        # last actions
        seq_action_name, seq_action_word, seq_action_xy, labels = \
                self._seq_last_action(last_actions, requires_loss, base_dtype)
        seq.append(seq_action_name) # Shape:  (n_total, 1, base_dim)
        seq.append(seq_action_word) # Shape:  (n_total, 1, base_dim)
        seq.append(seq_action_xy) # Shape:  (n_total, 1, base_dim)
        
        # Screenshots
        seq_screenshot = self._seq_screenshot(screenshots, base_dtype)
        seq.append(seq_screenshot) # Shape: (n_total, (w/32)*(h/32), base_dim)
        
        # Tokens
        seq_token, token_mask = self._seq_token(tokens, base_dtype)
        seq.append(seq_token) # Shape: (n_total, max_seq_len, base_dim)
        
        # concatenate the segments of the input sequence
        seq = torch.cat(seq, axis=1)
        seq = seq.view(mb_size, n_steps, -1, self.base_dim)
        seq = seq.swapaxes(0, 1).contiguous()
        # Shape: (n_steps, mb_size, input_len, base_dim)
        
        len_seq_fixed_part = seq.shape[2] + self.half_memory_size - token_mask.shape[1]
        is_last_action_token = torch.zeros((1, seq.shape[2], 1), dtype=torch.bool, device=device)
        pos_last_action = self.half_memory_size + self.output_size
        is_last_action_token[:, pos_last_action:pos_last_action + 3] = True
        
        token_mask = token_mask.view(mb_size, n_steps, -1)
        token_mask = torch.cat((
            torch.ones((n_steps, mb_size, len_seq_fixed_part), dtype=torch.bool, device=device), 
            token_mask.swapaxes(0, 1)
        ), axis=2)
        # Shape: (n_steps, mb_size, input_len)
        
        if is_reset is None:
            is_reset = torch.zeros((mb_size, n_steps), dtype=torch.bool, device=device)
        is_reset = is_reset.swapaxes(0, 1)[:, :, None, None]
        # Shape: (n_steps, mb_size, 1, 1) # 1s account for memory_axis, dim
        
        # Apply BERT recurrently
        extracted = []
        current_memory = None
        
        if self.use_memory:
            inital_memory = self.inital_memory.repeat(mb_size, 1, 1)
            current_memory = inital_memory if memory is None else memory
            
            for inputs_embeds, mask, is_reset_step in zip(seq, token_mask, is_reset):
                # Handle reset flags
                current_memory = torch.where(is_reset_step, inital_memory, current_memory)
                inputs_embeds = inputs_embeds.masked_fill(is_last_action_token & is_reset_step, 0)
                # Make input
                inputs_embeds = torch.cat((current_memory, inputs_embeds), axis=1)
                last_hidden_state = self.fusion_bert(inputs_embeds=inputs_embeds, attention_mask=mask).last_hidden_state
                extracted.append(last_hidden_state[:, self.memory_size:self.memory_size+self.output_size])
                current_memory = last_hidden_state[:, self.half_memory_size:self.memory_size]
        else:
            for inputs_embeds, mask, is_reset_step in zip(seq, token_mask, is_reset):
                inputs_embeds = inputs_embeds.masked_fill(is_last_action_token & is_reset_step, 0)
                last_hidden_state = self.fusion_bert(inputs_embeds=inputs_embeds, attention_mask=mask).last_hidden_state
                extracted.append(last_hidden_state[:, :self.output_size])
            
        extracted = torch.stack(extracted, axis=1)
        # Shape: (batch_size, time_steps, self.memory_size, self.base_dim)
        
        # Push the last states
        # Note: current implemetation just return values at the last index, not considered mask.
        # While this will work when the batch_size is 1, this may not fit some post procrsses.
        outputs = {}
        outputs['memory'] = current_memory
        ext_last = extracted[:, -1]
        outputs['last_actions'] = {
            key: self.output_linears[key](ext_last[:, i]) 
                for i, key in enumerate(self.ACTION_KEYS)
        }
        del ext_last
        
        # loss calculation if needed
        if requires_loss:
            extracted = extracted.view(n_total, self.output_size, self.base_dim) 
            loss_reults = self._calc_loss(extracted, labels, is_ignored)
            outputs.update(loss_reults)
        
        return outputs
    
    def _calc_loss(self, extracted, labels, is_ignored):
        """
        Args
        extracted: TENSOR(X, output_size, base_dim)
        labels: TENSOR(X, output_size)
        is_ignored: TENSOR(X)
        """
        
        # Mask elements
        labels.masked_fill_(labels==0, self.loss_ignored_index)
        if is_ignored is not None:
            labels.masked_fill_(is_ignored.view(-1, 1), self.loss_ignored_index)
        
        loss = 0.0
        loss_detail ={}
        for i, key in enumerate(self.ACTION_KEYS):
            y = self.output_linears[key](extracted[:, i])
            loss_detail[key] = self.loss_fn(y, labels[:, i].clone())
            loss += loss_detail[key]
        
        return {'loss':loss, 'loss_details':loss_detail}
    
    def step(self, env, observation, memory):
        """
        Apply a new action that is predicted from the current observatiion and memory, 
        to the given env, to make a step forwards. 
        Returns new observation and memory.
        When use_memory is False, memory should be None.
        """
        
        screenshots, tokens, last_actions = self.shape_observation_as_inputs(observation)
        
        device = self.get_device()
        screenshots = screenshots.to(device)
        tokens = tokens.to(device)
        last_actions = last_actions.to(device)
        
        if self.use_memory and memory is not None:
            memory = memory.to(device)
        
        with torch.no_grad():
            outputs = self(screenshots, tokens, last_actions, memory)
        
        if self.use_memory:
            memory = outputs['memory'].cpu()
        
        action = self.shape_outputs_as_action(outputs)
        observation = env.apply(action)
        
        return observation, memory   
    
    def shape_observation_as_inputs(self, observation):
        
        # screenshots:
        #     Tensor (batch_size, time_steps, height, width, channel=(R,G,B))
        screenshots = np.asarray(observation.screenshot.resize(self.input_window_size))
        # (height, width, 3), dtype=uint8
        screenshots = torch.tensor(screenshots).unsqueeze(0).unsqueeze(0) 
        
        # tokens: 
        #     Tensor (batch_size, time_steps, sequence_length, channel=(id, cx, cy, height, width))
        tokens = []
        for word in observation.detected_words:
            cx = self._get_id_x(word.cx)
            cy = self._get_id_y(word.cy)
            w = self._get_id_x(word.width)
            h = self._get_id_y(word.height)
            for token_id in self.tokenizer.encode(word.surface, add_special_tokens=False).ids:
                tokens.append((token_id, cx, cy, w, h))
                if len(tokens) >= self.config.max_tokens:
                    break
            if len(tokens) >= self.config.max_tokens:
                break
        tokens = torch.tensor(tokens).unsqueeze(0).unsqueeze(0)
        if tokens.shape[-1] == 0:
            tokens = tokens.long()
        
        # last_actions: 
        #     When requires_loss is False: Tensor (batch_size, time_steps, channel=(name, token, x, y))
        if observation.last_action is None:
            last_actions = (0, 0, 0, 0)
        else:
            last_actions = observation.last_action.to_numpy(self.config, self.tokenizer)
        last_actions = torch.tensor(last_actions).unsqueeze(0).unsqueeze(0)
        
        return screenshots, tokens, last_actions
    
    def shape_outputs_as_action(self, outputs):
        
        ids = {
                k: outputs['last_actions'][k][0].argmax().item() 
                for k in self.ACTION_KEYS
            } 
        name = self.config.actions[ids['name']]
        string = self.tokenizer.id_to_token(ids['token'])
        pointer_xy = (ids['x']-1, ids['y']-1)
        
        return ModelAction(name, string, pointer_xy)
