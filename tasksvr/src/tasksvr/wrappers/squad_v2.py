# coding: utf-8


from tasksvr import WrapperForDatasets, TTextExtraction


class Wrapper(WrapperForDatasets):
    """
    splits: train, validation, test
    features: ['sentence', 'label', 'idx']
    """
    
    name = 'squad_v2'
    key = ('squad_v2',)
    handler_class = TTextExtraction
    
    exposed_splits = {'train': 'train', 'valid': 'validation'}
    
    requires_handlers = []
    
    instructions = [
        'Reading Comprehension Task\n'
        'Read the next paragraph and answer the following question. '
        'Enter your answer in the textbox '
        'or check the unanswerable box when you think the question is unanswerable.',
    ]
    
    seq2seq_task_type = 'question and answering'
    seq2seq_instructions = [
        'Read the next paragraph and answer the following question. '
        'answer an empty string when you think the question is unanswerable.',
    ]
    
    context_surfaces = ['Paragraph']
    question_surfaces = ['Question']
    unanswerable_surfaces = ['unanswerable']
    
    def make_metadata(self):
        return {
            'name': self.name,
            'instructions': self.instructions,
            'context_surfaces': self.context_surfaces,
            'question_surfaces': self.question_surfaces,
            'unanswerable_surfaces': self.unanswerable_surfaces,
        }
