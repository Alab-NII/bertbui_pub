# coding: utf-8


from tasksvr import WrapperForDatasets


class Wrapper(WrapperForDatasets):
    """
    splits: train, validation, test
    features: ['sentence', 'label', 'idx']
    """
    
    name = 'sst2'
    key = ('glue', 'sst2')
    
    exposed_splits = {'train': 'train', 'valid': 'validation'}
    
    requires_handlers = []
    
    instructions = [
        'Text Classification Task\n'
        'Predict the emotion of the sentence (positive / negative).',
    ]
    
    seq2seq_task_type = 'single choice classification'
    seq2seq_instructions = [
        'Predict the emotion of the sentence (positive / negative).'
    ]
    
    sentence_features = ['sentence']
    feature_name_rules = [
        {},
    ]
    
    labels = [0, 1]
    label_rules = [
        {0:'negative', 1:'positive'},
    ]
    