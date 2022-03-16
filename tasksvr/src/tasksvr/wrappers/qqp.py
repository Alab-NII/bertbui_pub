# coding: utf-8


from tasksvr import WrapperForDatasets


class Wrapper(WrapperForDatasets):
    """
    splits: train, validation, test
    features: ['sentence', 'label', 'idx']
    """
    
    name = 'qqp'
    key = ('glue', 'qqp')
    
    exposed_splits = {'train': 'train', 'valid': 'validation'}
    
    requires_handlers = []
    
    instructions = [
        'Text Classification Task\n'
        'If the next two questions have the same meaning, '
        'press "DUPLICATE". If they are not, press "NOT DUPLICATE".',
    ]
    
    seq2seq_task_type = 'single choice classification'
    seq2seq_instructions = [
        'If the next two questions have the same meaning, '
        'answer "DUPLICATE". If they are not, answer "NOT DUPLICATE".',
    ]
    
    sentence_features = ['question1', 'question2']
    feature_name_rules = [
        {'question1': 'Q1', 'question2': 'Q2'},
    ]
    
    labels = [0, 1]
    label_rules = [
        {0:'NOT DUPLICATE', 1:'DUPLICATE'},
    ]
    