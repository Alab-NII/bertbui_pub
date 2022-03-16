# coding: utf-8


from tasksvr import DatasetWrapper, TVisualQuestionAnswering
import os
import re
import json


URLS = [
    'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip',
    'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip',
    'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip',
    'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip',
]


SPLIT_SETTINGS = {
    'train': {
        'q': 'v2_OpenEnded_mscoco_train2014_questions.json', 
        'a': 'v2_mscoco_train2014_annotations.json',
        'i': '/coco/train2014/COCO_train2014_%012d.jpg',
    },
    'validation': {
        'q': 'v2_OpenEnded_mscoco_val2014_questions.json',
        'a': 'v2_mscoco_val2014_annotations.json',
        'i': '/coco/val2014/COCO_val2014_%012d.jpg'
    },
}


class Wrapper(DatasetWrapper):
    """
    splits: train, validation, test
    features: ['sentence', 'label', 'idx']
    """
    
    name = 'vqa_v2'
    
    handler_class = TVisualQuestionAnswering
    
    requires_handlers = ['coco']
    
    exposed_splits = {'train': 'train', 'valid': 'validation'}
    
    instructions = [
        'Visual Question Answering Task\n'
        'See the picture below and answer the following question.',
    ]
    
    context_surfaces = ['']
    question_surfaces = ['Question:']
    unanswerable_surfaces = ['']
    
    # For seq2seq setting
    seq2seq_task_type = 'visual question answering'
    seq2seq_instruction = 'See the picture and answer the following question.'
    
    def load(self):
        data = {}
        for split_name, setting in SPLIT_SETTINGS.items():
            q_path = os.path.join(self.data_dir, self.name, setting['q'])
            a_path = os.path.join(self.data_dir, self.name, setting['a'])
            image_path_tmp = setting['i']
            
            annotation_dict = {_['question_id']:_ for _ in  json.load(open(a_path))['annotations']}
            questions = json.load(open(q_path))['questions']
            
            entries = []
            for q in questions:
                entries.append({
                    'image_path': image_path_tmp%q['image_id'],
                    'question': q['question'],
                    'question_id': q['question_id'],
                    'answer': annotation_dict[q['question_id']]['multiple_choice_answer'],
                })
            data[split_name] = entries
        
        return data
    
    def make_data(self):
        return self.load()
    
    def make_metadata(self):
        return {
            'name': self.name,
            'instructions': self.instructions,
            'context_surfaces': self.context_surfaces,
            'question_surfaces': self.question_surfaces,
            'unanswerable_surfaces': self.unanswerable_surfaces,
        }
    
    def make_task_handlers(self):
        
        data = self.make_data()
        metadata = self.make_metadata()
        handlers = [
            (r'/%s/%s' % (dest, self.name), self.handler_class, {'data': data[src], 'metadata': metadata})
            for dest, src in self.exposed_splits.items()
        ]
        return handlers, set(self.requires_handlers)
    
    def setup(self):
        local_dir = os.path.join(self.data_dir, self.name)
        if not os.path.exists(local_dir):
            os.mkdir(local_dir)
        for url in URLS:
            path = os.path.join(local_dir, os.path.basename(url))
            print('downloading', url)
            self._download_file(url, path)
            print('extracting', path)
            self._extract_zip(local_dir, path)
