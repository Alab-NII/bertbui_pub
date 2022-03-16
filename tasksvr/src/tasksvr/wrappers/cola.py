# coding: utf-8


from tasksvr import WrapperForDatasets


class Wrapper(WrapperForDatasets):
    """
    splits: train, validation, test
    features: ['sentence', 'label', 'idx']
    """
    
    name = 'cola'
    key = ('glue', 'cola')
    
    exposed_splits = {'train': 'train', 'valid': 'validation'}
    
    requires_handlers = []
    
    instructions = [
        'Text Classification Task\n'
        'If the following sentence is acceptable as an English sentence, '
        'press the acceptable button; if not, press the unacceptable button.',
    ]
    
    seq2seq_task_type = 'single choice classification'
    seq2seq_instructions = [
        'If the following sentence is acceptable as an English sentence, '
        'answer acceptable; if not, answer unacceptable.',
    ]
    
    sentence_features = ['sentence']
    feature_name_rules = [
        {'sentence': ''},
    ]
    
    labels = [0, 1]
    label_rules = [
        {0:'unacceptable', 1:'acceptable'},
    ]
    