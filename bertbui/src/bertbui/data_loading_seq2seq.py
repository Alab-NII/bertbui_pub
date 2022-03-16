# coding: utf-8


import os
import json
import numpy as np
import torch
import PIL.Image

#from . import DatasetProvider


class CustomBatch:
    
    def __init__(self, size, token_ids, labels, images=None, images_enabled=None):
        
        self.size = size
        self.token_ids = token_ids
        self.labels = labels
        self.images = images
        self.images_enabled = images_enabled
        
    def to_device(self, name):
        
        self.token_ids = self.token_ids.to(name)
        if isinstance(self.labels, tuple):
            self.labels = tuple(_.to(name) for _ in self.labels)
        else:
            self.labels = self.labels.to(name)
        if self.images is not None:
            self.images = self.images.to(name)
        if self.images_enabled is not None:
            self.images_enabled = self.images_enabled.to(name)
        
        return self


class Seq2SeqDataset(object):
    
    @staticmethod
    def normalize_text(text):
        
        return text.strip().lower()    
    
    def open_and_resize_image(self, path):
        
        image = PIL.Image.open(path)
        w, h = image.size
        bw, bh = self.image_size
        sx = bw/w
        sy = bh/h
        
        if sx <= sy:
            nw = bw
            nh = int(h*sx)
            ox = 0
            oy = int(0.5*(bh - nh))
        else:
            nw = int(w*sy)
            nh = bh
            ox = int(0.5*(bw - nw))
            oy = 0
        
        base = PIL.Image.new(mode='RGB', size=self.image_size)
        base.paste(image.resize((nw, nh)), (ox, oy))
        return base
    
    def __init__(self, model_config, dataset_names, data_dir='', disable_images=False):
        
        self.sep_instruction_and_content = ':'
        self.sep_key_and_value = '='
        
        # Tokenizer
        self.tokenizer = model_config.load_tokenizer()
        self.input_max_l_tokens_l = model_config.input_max_l_tokens_l
        self.input_max_l_tokens_vl = model_config.input_max_l_tokens_vl
        self.output_max_tokens = model_config.output_max_tokens
        self.pad_token_id = self.tokenizer.pad_token_id
        self.cls_token_id = None
        if hasattr(self.tokenizer, 'cls_token_id'):
            self.cls_token_id = self.tokenizer.cls_token_id
        
        # Image
        self.disable_images = bool(disable_images)
        self.image_size = model_config.image_size
        if self.image_size:      
            self.image_size = tuple(int(_) for _ in self.image_size.split(','))
        
        # Dataset
        self.data_dir = data_dir
        self.dataset_names = dataset_names
        self.data = None
        self.is_single_dataset = len(dataset_names) == 1
        self.metadata = {}
        
        # pre-compile data with the above settings
        self.pre_compile_data()
    
    def pre_compile_data(self):
        
        dp = DatasetProvider()
        
        data = []
        
        for dataset_name in self.dataset_names:
            data_dict = dp(dataset_name)
            seq2seq_task_type = data_dict.get('seq2seq_task_type')
            assert seq2seq_task_type is not None, f'{dataset_name} does not support seq2seq setting'
                        
            instruction = seq2seq_task_type+'. '+data_dict['seq2seq_instruction']
            
            # specify main content sources
            if seq2seq_task_type == 'visual question answering':
                sentence_columns = ['question']
            elif seq2seq_task_type == 'question and answering':
                sentence_columns = ['paragraph', 'question']
            elif seq2seq_task_type == 'single choice classification':
                sentence_columns = data_dict.get('sentence_columns', None)
            else:
                raise RuntimeError(f'not defined task type {seq2seq_task_type}')
            single_sentence = len(sentence_columns) == 1
            
            # surface modification for the classification tasks
            label_renaming_dict = data_dict.get('renamed_labels', None)
            col_renaming_dict = data_dict.get('renamed_sentence_columns', None)
            
            if self.is_single_dataset:
                self.metadata['task_type'] = seq2seq_task_type
                if seq2seq_task_type == 'single choice classification':
                    if label_renaming_dict:
                        self.metadata['label_mapping'] = {v : k for k, v in label_renaming_dict.items()}
                    else:
                        self.metadata['label_mapping'] = {k : k for k in data_dict['labels']}
            
            n_show = 3
            for raw_entry in data_dict['data']:
                
                # format the main content
                if single_sentence:
                    main_text = raw_entry[sentence_columns[0]]
                else:
                    texts = []
                    for col in sentence_columns:
                        col_surface = col
                        if col_renaming_dict:
                            col_surface = col_renaming_dict[col]
                        content = raw_entry[col]
                        texts.append( f'{col_surface} {self.sep_key_and_value} {content}')
                    main_text = ' '.join(texts)
                
                # format the answer
                if 'answer' in raw_entry:
                    label_text = raw_entry['answer']
                else:
                    label_text = raw_entry['label']
                    if label_renaming_dict:
                        label_text = label_renaming_dict[label_text]
                
                # make image path if provided
                if 'image_path' in raw_entry:
                    image_path = os.path.join(self.data_dir, raw_entry['image_path'].strip(os.path.sep))
                    input_max_tokens =self.input_max_l_tokens_vl 
                else:
                    image_path = None
                    input_max_tokens = self.input_max_l_tokens_l 
                
                # add entry to data
                # tokenization is different according to models
                # - t5 ... [s/]
                # - bert [cls] ... [sep]
                # for input tokens, both styles are ok
                # for output tokens, we remove first special token to match the style that generate function expects
                input_text = self.normalize_text(f'{instruction} {self.sep_instruction_and_content} {main_text}')
                input_ids = self.tokenizer(input_text, max_length=input_max_tokens, truncation=True).input_ids
                label_text = self.normalize_text(label_text)
                label_ids = self.tokenizer(label_text, max_length=self.output_max_tokens, truncation=True).input_ids
                if label_ids[0] == self.cls_token_id:
                    label_ids.pop(0)
                entry = {
                    'input_ids' : input_ids,
                    'label_ids' : label_ids,
                    'image_path' : image_path,
                }
                data.append(entry)
                
                # outputs for debug
                if n_show > 0:
                    print('- dataset', dataset_name)
                    print('input_text', input_text)
                    print('label_text', label_text)
                    print('entry', entry)
                    n_show -= 1
            
            self.data = data
    
    def __len__(self):
        
        return len(self.data)
    
    def __getitem__(self, _id):
        
        entry = self.data[_id]
        input_ids = entry['input_ids']
        label_ids = entry['label_ids']
        image_path = entry['image_path']
        if self.disable_images or (image_path is None):
            image_pil = None
        else:
            image_pil = self.open_and_resize_image(image_path)
        
        return input_ids, label_ids, image_pil, self.pad_token_id, self.image_size
    
    @staticmethod
    def collate_fn(items):  
        # items:
        #     0:input_ids, 1:label_ids, 2:image_pil, 3:pad_token_id, 4:image_size
        
        pad_token_id = items[0][3]
        image_size = items[0][4]
        
        image_pils = [_[2] for _ in items] 
        has_no_image = [_ is None for _ in image_pils]
        
        if all(has_no_image):
            images_enabled = None
            images = None
        else:
            images_enabled = ~torch.tensor(has_no_image)
            
            pad_image = PIL.Image.new(mode='RGB', size=image_size)
            images = torch.tensor(np.stack([np.asarray(pad_image if _ is None else _) for _ in image_pils]))
            # dtype = uint8, shape = (batchsize, height, width, channel)
        
        return CustomBatch(
                size=len(items), 
                token_ids=_fill_1d(items, 0, pad_token_id), 
                labels=_fill_1d(items, 1, pad_token_id), 
                images=images, images_enabled=images_enabled
        )

    
def _fill_1d(items, axis, pad_token_id):
        
    max_len = max(len(item[axis]) for item in items)
    ids = []
    for item in items:
        _ids = item[axis]
        delta = max_len - len(_ids)
        if delta > 0:
            _ids.extend([pad_token_id]*delta)
        ids.append(_ids)
    return torch.tensor(ids)
