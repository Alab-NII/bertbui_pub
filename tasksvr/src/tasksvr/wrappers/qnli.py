# coding: utf-8


from tasksvr import WrapperForDatasets


class Wrapper(WrapperForDatasets):
    """
    splits: train, validation, test
    features: ['sentence', 'label', 'idx']
    """
    
    name = 'qnli'
    key = ('glue', 'qnli')
    
    exposed_splits = {'train': 'train', 'valid': 'validation'}
    
    requires_handlers = []
    
    instructions = [
        'Text Classification Task\n'
        'Read the following question-sentence pair and click answerable '
        'if the sentence contains the answer to the question, '
        'and unanswerable if it does not.',
    ]
    
    seq2seq_task_type = 'single choice classification'
    seq2seq_instructions = [
        'Read the following question-sentence pair and answer answerable '
        'if the sentence contains the answer to the question, '
        'and unanswerable if it does not.',
    ]
    
    sentence_features = ['question', 'sentence']
    feature_name_rules = [
        {},
    ]
    
    labels = [0, 1]
    label_rules = [
       {0:'answerable', 1:'unanswerable'},
       {0:'entailment', 1:'not_entailment'},
    ]
