# coding: utf-8


from tasksvr import WrapperForDatasets


class Wrapper(WrapperForDatasets):
    """
    splits: train, validation, test
    features: ['sentence', 'label', 'idx']
    """
    
    name = 'mrpc'
    key = ('glue', 'mrpc')
    
    exposed_splits = {'train': 'train', 'valid': 'validation'}
    
    requires_handlers = []
    
    instructions = [
        'Text Classification Task\n'
        'Answer whether the following pairs of sentences are semantically equivalent. '
        'If they are equivalent, click on equivalent; if not, click on not equivalent.',
    ]
    
    seq2seq_task_type = 'single choice classification'
    seq2seq_instructions = [
        'Answer whether the following pairs of sentences are semantically equivalent. '
        'If they are equivalent, answer equivalent; if not, answer not equivalent.',
    ]
    
    sentence_features = ['sentence1', 'sentence2']
    feature_name_rules = [
        {},
    ]
    
    labels = [0, 1]
    label_rules = [
        {0:'not equivalent', 1:'equivalent'},
    ]
