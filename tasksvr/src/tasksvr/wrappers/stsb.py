# coding: utf-8


from tasksvr import WrapperForDatasets


class Wrapper(WrapperForDatasets):
    """
    splits: train, validation, test
    features: ['sentence', 'label', 'idx']
    """
    
    name = 'stsb'
    key = ('glue', 'stsb')
    
    exposed_splits = {'train': 'train', 'valid': 'validation'}
    
    requires_handlers = []
    
    instructions = [
        'Text Classification Task\n'
        'Rate how similar the following two sentences are '
        'on a scale from 0 to 5 '
        '(0 being the least similar and  being the most similar 5).',
    ]
    
    seq2seq_task_type = 'single choice classification'
    seq2seq_instructions = [
        'Rate how similar the following two sentences are '
        'on a scale from 0 to 5 '
        '(0 being the least similar and  being the most similar 5).',
    ]
    
    sentence_features = ['sentence1', 'sentence2']
    feature_name_rules = [
        {},
    ]
    
    labels = [0, 1, 2, 3, 4, 5]
    label_rules = [
        {},
    ]
    
    @staticmethod
    def _transform_example(example):
        new_example = example.copy()
        new_example['label'] = int(round(new_example['label']))
        return new_example
    
    def make_data(self):
        data = self.load()
        data.map(self._transform_example)
        return data
    