# coding: utf-8


from tasksvr import DatasetWrapper, TPretraining
import os
import re
import json
import numpy as np


WIKITEXT_URL = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip'
EXTRACTED_FILE_PATH = 'wikitext-103/wiki.train.tokens'

SPLIT_SETTINGS = {
    'train': {'seed': 937823, 'numbers': [15000]*4},
    'validation':  {'seed': 283988, 'numbers': [500]*4},
}

ORDINALS = ['zeroth', 'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth']


class Wrapper(DatasetWrapper):
    """
    splits: train, validation, test
    features: ['sentence', 'label', 'idx']
    """
    
    name = 'pta'
    
    handler_class = TPretraining
    
    requires_handlers = []
    
    exposed_splits = {'train': 'train', 'valid': 'validation'}
    
    def load(self):
        data = {}
        for split_name in SPLIT_SETTINGS.keys():
            path = os.path.join(self.data_dir, self.name, split_name+'.json')
            data[split_name] = json.load(open(path))
        return data
    
    def make_data(self):
        return self.load()
    
    def make_metadata(self):
        return {
            'name': self.name,
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
        path = os.path.join(local_dir, os.path.basename(WIKITEXT_URL))
        print('downloading', WIKITEXT_URL)
        self._download_file(WIKITEXT_URL, path)
        print('extracting')
        self._extract_zip(local_dir, path)
        print('collecting vocabulary')
        vocab_path = os.path.join(local_dir, 'vocabulary.tsv')
        make_vocab_file(
            os.path.join(local_dir, EXTRACTED_FILE_PATH), 
            vocab_path
        )
        
        with open(vocab_path, 'r') as f:
            seed_words = [line.split('\t')[0] for line in f]
        if len(seed_words) <= 0:
            raise RuntimeException('The length of seed words should be larger than zero.')
        
        print('sampling')
        for split_name, setting in SPLIT_SETTINGS.items():
            data = make_data(seed_words, setting)
            save_path = os.path.join(local_dir, split_name+'.json')
            with open(save_path, 'w') as f:
                json.dump(data, f)
    

def make_vocab_file(input_path, output_path):
        
    def normalize(word):
        # normalize all words by lowering them
        return word.lower()
    
    def is_included(word):
        # we just use alphabets and numbers 
        if re.search('[^a-zA-Z0-9]', word):
            return False
        return True
    
    def get_word_count(path):
        # return a dict whose key and value correspond to word and count
        # we do not use count this time.
        # we save the count as well as word though.
        words = []
        with open(path, 'r') as f:
            for line in f:
                words.extend(_ for _ in line.split() if len(_.strip()) > 0)

        count = {}
        for word in words:
            norm_word = normalize(word)
            if is_included(norm_word):
                count[norm_word] = count.get(norm_word, 0) + 1
        
        return count
    
    count = get_word_count(input_path)
    lines = [f'{w}\t{c}'  for w, c in sorted(count.items(), key=lambda t:-t[1])]
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


def make_data(seed_words, setting):
    
    tasks = [TaskCursor, TaskButton, TaskText, TaskArea]
    random = np.random.RandomState(setting['seed'])
    
    seed_words_state = {
        'pos': 0,
        'ids': list(range(len(seed_words))),
    }
    random.shuffle(seed_words_state['ids'])
    
    def get_words(n):
        pos = seed_words_state['pos']
        ids = seed_words_state['ids'][pos:pos+n]
        seed_words_state['pos'] += n
        while len(ids) < n:
            random.shuffle(seed_words_state['ids'])
            pos = seed_words_state['pos'] = 0
            d = n - len(ids)
            ids += seed_words_state['ids'][pos:pos+d]
            seed_words_state['pos'] += d
        return [seed_words[_] for _ in ids]
    
    context = {
        'random': random,
        'seed_words': seed_words,
        'get_words': get_words,
    }
    
    data = []
    for task, n in zip(tasks, setting['numbers']):
        data.extend(task.build(context) for _ in range(n))
            
    return data
    

class TaskCursor(object):
    """keys:
    instruction answer_str
    box_cx box_cy box_width box_height
    """
    
    @classmethod
    def build(cls, context):
        builder = cls._build_0
        return builder(context, {'task_type': cls.__name__})
    
    @classmethod
    def _build_0(cls, context, data):
        
        random = context['random']
        
        instructions = [
            'Move the cursor in the box.',
            'Point to the box with the cursor.'
        ]
        
        data['instruction'] = random.choice(instructions)
        data['box_width'] = w = 30 # px
        data['box_height'] = h = 30 # px
        data['box_cx'] = cx = random.randint(0, 101 - 10) # percentage
        data['box_cy'] = cy = random.randint(20, 101 - 10) # percentage
        data['answer_str'] = json.dumps({
            'task': 'cursor',
            'answer_type': 'str',
            'answer': 'correct_box',
        })
        return data


class TaskButton(object):
    """keys:
    instruction answer_str
    buttons correct_id n_rows n_columns
    """
    
    lambda_exp = 1.0
    verbs = ['Click', 'Push', 'Press', 'Choose', 'Select']
    
    @classmethod
    def build(cls, context):
        builder = cls._build_0
        return builder(context, {'task_type': cls.__name__})
    
    @classmethod
    def _build_0(cls, context, data):
        
        random = context['random']
        
        nx, ny = [int(round(_)) for _ in (1+ random.exponential(cls.lambda_exp, size=(2,)))]
        n_buttons = nx*ny
        button_labels = context['get_words'](n_buttons)
        correct_id = random.randint(0, n_buttons)
        correct_label =  button_labels[correct_id]
        
        verb = random.choice(cls.verbs)
        instruction_tmp = random.choice(['%s the button labelled %s.', '%s the %s button.'])
        
        data['instruction'] = instruction_tmp % (verb, correct_label)
        data['buttons'] = button_labels
        data['correct_id'] = correct_id
        data['n_rows'] = ny
        data['n_columns'] = nx
        data['answer_str'] = json.dumps({
            'task': 'button',
            'answer_type': 'str',
            'answer': correct_label,
        })
        return data


class TaskText(object):
    """keys:
    instruction answer_str
    correct_texts n_rows n_columns
    """
    
    lambda_exp = 0.5
    n_words = 2
    verbs = ['Type', 'Enter', 'Input']
    
    @classmethod
    def build(cls, context):
        builder = cls._build_0
        return builder(context, {'task_type': cls.__name__})
    
    @classmethod
    def _build_0(cls, context, data):
        
        random = context['random']
        
        nx, ny = [int(round(_)) for _ in (1+ random.exponential(cls.lambda_exp, size=(2,)))]
        n_text_boxes = nx*ny
        words = context['get_words'](n_text_boxes*cls.n_words)
        correct_texts = []
        for i in range(0, len(words), cls.n_words):
            correct_texts.append(' '.join(words[i:i+cls.n_words]))
        
        verb = random.choice(cls.verbs)
        instruction_tmp = '%s the string to the left of it in each text box. Click the submit button at last.'
        
        data['instruction'] = instruction_tmp % (verb)
        data['correct_texts'] = correct_texts
        data['n_rows'] = ny
        data['n_columns'] = nx
        data['answer_str'] = json.dumps({
            'task': 'text',
            'answer_type': 'str',
            'answer': '+'.join(correct_texts),
        })
        return data


class TaskArea(object):
    """Keys:
    instruction answer_str
    buttons correct_id n_rows n_columns v_offset
    """
    
    lambda_exp = 1.0
    verbs = ['click', 'push', 'press', 'choose', 'select']
    
    @classmethod
    def build(cls, context):
        builder = cls._build_0
        return builder(context, {'task_type': cls.__name__})
    
    @classmethod
    def _build_0(cls, context, data):
        
        random = context['random']
        
        nx, ny = [int(round(_)) for _ in (1+ random.exponential(cls.lambda_exp, size=(2,)))]
        n_buttons = nx*ny
        button_labels = context['get_words'](n_buttons)
        correct_id = random.randint(0, n_buttons)
        correct_label =  button_labels[correct_id]
        
        verb = random.choice(cls.verbs)
        instruction_tmp = random.choice([
            'Scroll down until the buttons appear and %s the button labelled %s.', 
            'Scroll down until the buttons appear and  %s the %s button.'
        ])
        
        data['instruction'] = instruction_tmp % (verb, correct_label)
        data['buttons'] = button_labels
        data['correct_id'] = correct_id
        data['n_rows'] = ny
        data['n_columns'] = nx
        data['v_offset'] = random.uniform(1, 1.5)
        data['answer_str'] = json.dumps({
            'task': 'area',
            'answer_type': 'str',
            'answer': correct_label,
        })
        return data
