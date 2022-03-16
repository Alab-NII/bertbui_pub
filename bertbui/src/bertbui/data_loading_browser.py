# coding: utf-8


import os
import time
import json
import ijson
import numpy as np
from concurrent.futures import ThreadPoolExecutor

import torch
import tokenizers

from . import ModelAction, deserialize_teacher_actions, read_metadata


class CustomBatch:
    
    def __init__(self, size, screenshots, tokens, last_actions, is_ignored, is_reset, time_step):
        
        self.size = size
        self.screenshots = screenshots
        self.tokens = tokens
        self.last_actions = last_actions
        self.is_ignored = is_ignored
        self.is_reset = is_reset
        self.time_step = time_step

    def to_device(self, name):
        
        self.screenshots = self.screenshots.to(name)
        self.tokens = self.tokens.to(name)
        self.last_actions = self.last_actions.to(name)
        self.is_ignored = self.is_ignored.to(name)
        self.is_reset = self.is_reset.to(name)
        
        return self


class ConcatDataset(object):
    
    @staticmethod
    def get_num_actions_from_json(file_path):
        num_actions = 0
        for prefix, event, value in ijson.parse(open(file_path)):
            if not prefix.startswith('actions'):
                continue
            elif prefix == 'actions.item.name' and event == 'string':
                num_actions += 1
            elif prefix == 'actions' and event == 'end_array':
                break
        return num_actions

    @staticmethod
    def get_file_step_list(data_dir):
        
        print('walking', data_dir)
        stime = time.time()
        
        def thread_func(path):
            return (path, ConcatDataset.get_num_actions_from_json(path))

        filter_func = lambda x: not x.startswith('.') and x.endswith('.json')
        
        names = sorted(_ for _ in os.listdir(data_dir) if filter_func(_))
        paths = [os.path.join(data_dir, _) for _ in names]
        
        file_steps = []
        with ThreadPoolExecutor() as executor:
            file_steps.extend(executor.map(thread_func, paths))
        
        etime = time.time()
        print(len(file_steps), 'files in', data_dir, etime - stime)
        
        return file_steps 
    
    @staticmethod
    def get_file_step_list_cache(data_dir):
        
        try:
            metadata = read_metadata(data_dir)
        except Exception as e:
            print(e)
            print('Please recreate metadata by $python -m bertbui metadata')
            import sys
            sys.exit()
        print(data_dir, 'metadata loaded from the cache file')
        # sort metadata by key (filename) to freeze the order
        return [(os.path.join(data_dir, k), v['num_actions']) for k, v in sorted(metadata.items(), key=lambda t:t[0])]
        
    def __init__(self, training_config, model_config, data_dirs, 
                 random_seed=None, max_num=None, sampling_seed=None):
        
        self.training_config = training_config
        self.model_config = model_config
        self.data_dirs = data_dirs
        self.random_seed = random_seed
        if random_seed is not None:
            self.random = np.random.RandomState(random_seed)
        else:
            self.random = None
        
        # collect json paths
        self.file_step_pairs = []
        if max_num is not None:
            _random = np.random.RandomState(sampling_seed) 
        for data_dir in data_dirs:
            file_step_list = self.get_file_step_list_cache(data_dir)
            if max_num is not None and len(file_step_list) > max_num:
                _random.shuffle(file_step_list)
                file_step_list = file_step_list[:max_num]
            self.file_step_pairs.extend(file_step_list)
        
        self.tokenizer = tokenizers.Tokenizer.from_file(model_config.tokenizer_path)
        self.action_to_id = model_config.action_to_id
        self._get_id_x = model_config.get_id_x
        self._get_id_y = model_config.get_id_y
        self.max_tokens = model_config.max_tokens
        self.max_step_num = training_config.max_step_num
        
        self.sequences = []
        
        self.pack()
    
    def pack(self):
        
        if self.random:
            self.random.shuffle(self.file_step_pairs)
        sequences = []
        sequence = []
        cur_pos = 0
        file_path, n_steps = (None, 0)
        _id = len(self.file_step_pairs) - 1
        
        while _id >= 0 or file_path is not None:
            # Fetch a new item
            if file_path is None:
                file_path, n_steps = self.file_step_pairs[_id]
                _id -= 1
            
            # We trancate the steps if the instance is the first one in a line.
            if cur_pos == 0:
                n_steps = min(n_steps, self.max_step_num)
            
            if cur_pos + n_steps <= self.max_step_num:
                # Append a segument to the last of the current line
                sequence.append((file_path, cur_pos, n_steps))
                cur_pos += n_steps
                file_path = None
            else:
                # Left the current segement and slide line to the next
                sequences.append(sequence)
                sequence = []
                cur_pos = 0
        if len(sequence) > 0:
            sequences.append(sequence)
        
        self.sequences = sequences
                
    def __len__(self):
        
        return len(self.sequences)
    
    def __getitem__(self, _id):
        
        _get_id_x = self._get_id_x
        _get_id_y = self._get_id_y
    
        if self.model_config.requires_resize:
            _resize = lambda x: x.resize(self.model_config.input_window_size)
        else:
            _resize = lambda x: x
        
        reset_steps = []
        last_actions = []
        screenshots = []
        tokens = []
        token_lens = []
        last_action_in_prev_seq = None
        
        for i, (path, start_pos, n_steps) in enumerate(self.sequences[_id]):
            
            reset_steps.append(start_pos)
            
            data = json.load(open(path, 'r'))
            _, actions, observations = deserialize_teacher_actions(data)
            # The last action  of observations[0] is a dummy action
            # We need the last action of observations[-1] to complete a sequence,
            # but we dont need observations[-1] itself 
            # a[0]=(dummy) o[0] a[1] o[1] ... a[-1] o[-1]=(not need)
            
            # Dummy action at the sequence head
            if i == 0:
                last_actions.append(observations[0].last_action.to_numpy(self.model_config, self.tokenizer))
            
            for j in range(n_steps):
                
                # State
                obs = observations[j]
                screenshots.append(np.asarray(_resize(obs.screenshot)))
                tokens_local = []
                for word in obs.detected_words:
                    cx = _get_id_x(word.cx)
                    cy = _get_id_y(word.cy)
                    w = _get_id_x(word.width)
                    h = _get_id_y(word.height)
                    for token_id in self.tokenizer.encode(word.surface, add_special_tokens=False).ids:
                        tokens_local.append(np.asarray([token_id, cx, cy, w, h]))
                        if len(tokens_local) >= self.max_tokens:
                            break
                    if len(tokens_local) >= self.max_tokens:
                        break
                tokens.append(tokens_local)
                token_lens.append(len(tokens_local))
                
                # Action
                obs = observations[j+1]
                last_actions.append(obs.last_action.to_numpy(self.model_config, self.tokenizer))
        
        # obtain tensors
        last_actions = np.stack(last_actions, axis=0)
        screenshots = np.stack(screenshots, axis=0)
        
        max_token_len = max(token_lens)
        for token_local, token_len in zip(tokens, token_lens):
            delta = max_token_len - token_len
            if delta > 0 :
                token_local.extend([[0, 0, 0, 0, 0]]*delta)
        tokens = np.stack(tokens, axis=0)
        
        input_step_len = screenshots.shape[0]
        
        return last_actions, screenshots, tokens, max_token_len, input_step_len, reset_steps
    
    @staticmethod
    def collate_fn(items):   
        
        # items: [(last_actions, screenshots, tokens, max_token_len, input_step_len, reset_steps)]
        batch_size = len(items)
        token_len = max(item[3] for item in items)
        input_step_len = max(item[4] for item in items)
        
        last_actions = np.zeros((batch_size, input_step_len+1,) + items[0][0].shape[1:] , dtype=items[0][0].dtype)
        screenshots  = np.zeros((batch_size, input_step_len,) + items[0][1].shape[1:] , dtype=items[0][1].dtype)
        tokens  = np.zeros((batch_size, input_step_len, token_len,) + items[0][2].shape[2:] , dtype=items[0][2].dtype)
        is_ignored = np.zeros((batch_size, input_step_len,), dtype=np.bool)
        is_reset = np.zeros((batch_size, input_step_len,), dtype=np.bool)
        
        for i, (_last_actions, _screenshots, 
                _tokens, _max_token_len, 
                _input_step_len, reset_steps) in enumerate(items):
            
            last_actions[i, :_input_step_len+1] = _last_actions
            screenshots[i, :_input_step_len] =  _screenshots
            tokens[i, :_input_step_len, :_max_token_len] = _tokens
            is_ignored[i, _input_step_len:] = True
            for j in reset_steps:
                is_reset[i, j] = True
        
        screenshots = torch.tensor(screenshots)
        tokens = torch.tensor(tokens)
        last_actions = torch.tensor(last_actions)
        is_ignored = torch.tensor(is_ignored)
        is_reset = torch.tensor(is_reset)
        
        return CustomBatch(batch_size, screenshots, tokens, last_actions, is_ignored, is_reset, input_step_len)
