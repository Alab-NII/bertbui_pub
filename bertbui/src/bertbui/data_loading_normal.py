# coding: utf-8


import os
import json
import numpy as np
import torch
import tokenizers
import PIL.Image

#from . import DatasetProvider


class CustomBatchBert:
    
    def __init__(self, size, token_ids, labels, images=None):
        
        self.size = size
        self.token_ids = token_ids
        self.labels = labels
        self.images = images
        
    def to_device(self, name):
        
        self.token_ids = self.token_ids.to(name)
        if isinstance(self.labels, tuple):
            self.labels = tuple(_.to(name) for _ in self.labels)
        else:
            self.labels = self.labels.to(name)
        if self.images is not None:
            self.images = self.images.to(name)
        
        return self


class OneStepClassification(object):
    
    def __init__(self, model_config, dataset_name, cls_token='[CLS]', sep_token='[SEP]'):
        
        self.dataset_name = dataset_name
        self.data_dict = data_dict = DatasetProvider()(dataset_name)
        self.data = data_dict['data']
        self.labels = data_dict['labels']
        self.sentence_keys = data_dict['sentence_columns']
        self.from_label_to_id = {l:i for i, l in enumerate(self.labels)}
        
        self.tokenizer = tokenizers.Tokenizer.from_file(model_config.tokenizer_path)
        
        self.max_tokens = model_config.max_tokens
        self.cls_token = cls_token
        self.sep_token = sep_token
    
    def __getitem__(self, _id):
        
        entry = self.data[_id]
        label_id = self.from_label_to_id[entry['label']]
        
        text = self.cls_token
        for key in self.sentence_keys:
            displayed_name = self.data_dict.displayed_name(sentence=key)
            if displayed_name:
                text += '%s: %s%s' % (displayed_name, entry[key], self.sep_token)
            else:
                text += '%s%s' % (entry[key], self.sep_token)
        
        token_ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        token_ids = token_ids[:self.max_tokens]
        
        return token_ids, label_id
    
    def __len__(self):
        
        return len(self.data)
    
    @staticmethod
    def collate_fn(items):  
        
        max_seq_len = max(len(item[0]) for item in items)
        
        token_ids = []
        labels = []
        for item in items:
            _token_ids = item[0]
            delta = max_seq_len - len(_token_ids)
            if delta > 0:
                _token_ids.extend([0]*delta)
            token_ids.append(_token_ids)
            labels.append(item[1])
        
        token_ids = torch.tensor(token_ids)
        labels = torch.tensor(labels)
        
        return CustomBatchBert(size=len(items), token_ids=token_ids, labels=labels)


class OneStepVQA(object):
    
    def __init__(self, model_config, dataset_name, 
            answer_vocab='', data_dir='', ommit_images=False,
            cls_token='[CLS]', sep_token='[SEP]'):
        
        self.ommit_images = ommit_images
        self.dataset_name = dataset_name
        self.data_dict = data_dict = DatasetProvider()(dataset_name)
        self.data = data_dict['data']
        self.labels = json.load(open(answer_vocab))
        self.from_label_to_id = {l:i for i, l in enumerate(self.labels)}
        
        self.tokenizer = tokenizers.Tokenizer.from_file(model_config.tokenizer_path)
        
        self.max_tokens = model_config.max_tokens
        self.cls_token = cls_token
        self.sep_token = sep_token
        
        self.image_size = model_config.image_input_size
        if self.image_size:      
            self.image_size = tuple(int(_) for _ in self.image_size.split(','))
        self.data_dir = data_dir
        
        self.compile_data()
    
    def normalize(self, text):
        
        return text.strip().lower()
    
    def compile_data(self):
        
        cls_token_id = self.tokenizer.token_to_id(self.cls_token)
        sep_token_id = self.tokenizer.token_to_id(self.sep_token)
        answer_oov_id = self.from_label_to_id['[OOV]']
        
        count_oov = 0
        
        for entry in self.data:
            answer = self.normalize(entry['answer'])
            if answer not in self.from_label_to_id:
                entry['label_id'] = answer_oov_id
                count_oov += 1
            else:
                entry['label_id'] = self.from_label_to_id[answer]
            ids = self.tokenizer.encode(entry['question'], add_special_tokens=False).ids
            # image sequence will be inserted between the cls and sep tokens
            entry['token_ids'] = ([cls_token_id, sep_token_id] + ids + [sep_token_id])[:self.max_tokens]
            entry['image_path'] = os.path.join(self.data_dir, entry['image_path'].strip(os.path.sep))
        
        print('# of oov', count_oov)
        
    
    def get_resized_image(self, path):
        
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
    
    def __getitem__(self, _id):
        
        entry = self.data[_id]
        token_ids = entry['token_ids']
        label_id = entry['label_id']
        if self.ommit_images:
            image = None
        else:
            image = self.get_resized_image(entry['image_path'])
        
        return token_ids, image, label_id
    
    def __len__(self):
        
        return len(self.data)
    
    @staticmethod
    def collate_fn(items):  
        
        max_seq_len = max(len(item[0]) for item in items)
        
        token_ids = []
        labels = []
        for item in items:
            _token_ids = item[0]
            delta = max_seq_len - len(_token_ids)
            if delta > 0:
                _token_ids.extend([0]*delta)
            token_ids.append(_token_ids)
            labels.append(item[2])
        
        token_ids = torch.tensor(token_ids)
        labels = torch.tensor(labels)
        
        if items[0][1] is not None:
            image_size = items[0][1].size
            images = torch.tensor(np.stack([np.asarray(_[1]) for _ in items]))
            # image: 
            # dtype = uint8, shape = (batchsize, height, width, channel)
        else:
            images = None
        
        return CustomBatchBert(size=len(items), token_ids=token_ids, labels=labels, images=images)


class OneStepExtraction(object):

    def __init__(self, model_config, dataset_name, cls_token='[CLS]', sep_token='[SEP]'):

        self.dataset_name = dataset_name
        data_dict = DatasetProvider()(dataset_name)
        self.data = data_dict['data']
        self.context_surface = data_dict['context_surface']
        self.question_surface = data_dict['question_surface']
        
        self.tokenizer = tokenizers.Tokenizer.from_file(model_config.tokenizer_path)
        
        self.max_tokens = model_config.max_tokens
        self.cls_token = cls_token
        self.sep_token = sep_token

        self._data_preprosess()
    
    def _data_preprosess(self):
        
        cls_token_id = [self.tokenizer.token_to_id(self.cls_token)]
        sep_token_id = [self.tokenizer.token_to_id(self.sep_token)]
        
        cs_ids = self.tokenizer.encode(self.context_surface+':', add_special_tokens=False).ids
        qs_ids = self.tokenizer.encode(self.question_surface+':', add_special_tokens=False).ids
        
        offset_token_len = len(cs_ids+cls_token_id)

        prosessed = []
        for entry in self.data:
            p = self.tokenizer.encode(entry['paragraph'], add_special_tokens=False)
            q = self.tokenizer.encode(entry['question'], add_special_tokens=False)
            inputs = (cls_token_id + cs_ids + p.ids + sep_token_id + qs_ids + q.ids + sep_token_id)[:self.max_tokens]
            if entry['unanswerable']:
                start_labels = end_labels = 0
            else:
                s_char = entry['answer_start']
                e_char = s_char + len(entry['answer']) - 1
                start_labels = end_labels = 0
                for i, (s, e) in enumerate(p.offsets):
                    if start_labels == 0 and s <= s_char and s_char < e:
                        start_labels = i + offset_token_len #
                    
                    if end_labels == 0 and s < e_char and e_char < e:
                        end_labels = i + offset_token_len #
                        break
            data = {
                '_id': entry['_id'],
                'token_ids': inputs,
                'start_labels': start_labels,
                'end_labels': end_labels,
            }
            prosessed.append(data)

        self.data = prosessed
    
    def __getitem__(self, _id):

        return self.data[_id]
    
    def __len__(self):

        return len(self.data)

    @staticmethod
    def collate_fn(items):

        max_seq_len = max(len(item['token_ids']) for item in items)
        
        token_ids = []
        start_labels = []
        end_labels = []
        for item in items:
            _token_ids = item['token_ids']
            delta = max_seq_len - len(_token_ids)
            if delta > 0:
                _token_ids.extend([0]*delta)
            token_ids.append(_token_ids)
            start_labels.append(item['start_labels'])
            end_labels.append(item['end_labels'])

        token_ids = torch.tensor(token_ids)
        start_labels = torch.tensor(start_labels)
        end_labels = torch.tensor(end_labels)

        return CustomBatchBert(size=len(items), token_ids=token_ids, labels=(start_labels, end_labels))
