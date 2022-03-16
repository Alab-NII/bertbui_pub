# coding: utf-8


from tasksvr import WrapperForDatasets


class Wrapper(WrapperForDatasets):
    """
    splits: train, validation, test
    features: ['sentence', 'label', 'idx']
    """
    
    name = 'wnli'
    key = ('glue', 'wnli')
    
    exposed_splits = {'train': 'train', 'valid': 'validation'}
    
    requires_handlers = []
    
    instructions = [
        'Text Classification Task\n'
        'Read the following two sentences and answer they entail or not.',
    ]
    
    seq2seq_task_type = 'single choice classification'
    seq2seq_instructions = [
        'Read the following two sentences and answer their relationship: '
        'entailment or not entailment.',
    ]
    
    sentence_features = ['sentence1', 'sentence2']
    feature_name_rules = [
        {},
    ]
    
    labels = [0, 1]
    label_rules = [
        {0:'not entailment', 1:'entailment'}
    ]
    