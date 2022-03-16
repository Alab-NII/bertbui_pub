# coding: utf-8


from tasksvr import WrapperForDatasets


class Wrapper(WrapperForDatasets):
    """
    splits: train, validation, test
    features: ['sentence', 'label', 'idx']
    """
    
    name = 'mnli'
    key = ('glue', 'mnli')
    
    exposed_splits = {'train': 'train', 'valid': 'validation_matched'}
    
    requires_handlers = []
    
    instructions = [
        'Text Classification Task\n'
        'For the following premise and hypothesis statements, '
        'click entailment if the premise entails the hypothesis, '
        'contradiction if it contradicts the hypothesis, or neutral if neither.',
    ]
    
    seq2seq_task_type = 'single choice classification'
    seq2seq_instructions = [
        'For the following premise and hypothesis statements, '
        'answer entailment if the premise entails the hypothesis, '
        'contradiction if it contradicts the hypothesis, or neutral if neither.',
    ]
    
    sentence_features = ['premise', 'hypothesis']
    feature_name_rules = [
        {'premise': 'Premise', 'hypothesis': 'Hypothesis'},
    ]
    
    labels = [0, 1, 2]
    label_rules = [
        {0:'entailment', 1:'neutral', 2:'contradiction'}
    ]
    
"""    
# coding: utf-8

from collections import OrderedDict
import json
from browserlm import ClassificationDatasetBuilder


class Builder(ClassificationDatasetBuilder):

    name = 'mnli'
    download_url = 'https://dl.fbaipublicfiles.com/glue/data/MNLI.zip'
    local_dir = 'MNLI'

    splits = OrderedDict([
        ('train', ['train.tsv']),
        ('dev_matched', ['dev_matched.tsv']),
        ('dev_mismatched', ['dev_mismatched.tsv']),
    ])

    base_config = {
        # Basic
        'has_header': True,
        'columns': None,
        'label_column': 'gold_label',
        'ignored_labels': ['-'],
        'labels': ['entailment', 'neutral', 'contradiction'],
        # Custom
        'instruction':
        'sentence_columns': ['sentence1', 'sentence2'],
        'renamed_sentence_columns': {'sentence1':'Premise', 'sentence2':'Hypothesis'},
        'renamed_labels': None,
        # For seq2seq setting
        'seq2seq_task_type' : 'single choice classification',
        'seq2seq_instruction' : 
    }
"""
