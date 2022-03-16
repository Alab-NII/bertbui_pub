# coding: utf-8


from tasksvr import DatasetWrapper, TSearchAndAnswerInnerPage, TSearchAndAnswer
import os
import re
import json
import numpy as np
import datasets
from collections import OrderedDict


SPLIT_SETTINGS = {
    'train': {
        'num_examples': 100,
        'num_contexts_per_example': (100, 100),
        'num_instructions_per_example': 500,
        'seed': 3434568,
        'path_list': [
            ['squad', 'squad_v2/train'],
            ['vqa', ('vqa_v2/v2_OpenEnded_mscoco_train2014_questions.json', 'vqa_v2/v2_mscoco_train2014_annotations.json')],
        ],
        'tasks': ['qid', 'question', 'answer', 'num_qs'],
        'file_name': 'sa_train.json',
    },
    'validation': {
        'num_examples': 10,
        'num_contexts_per_example': (100, 100),
        'num_instructions_per_example': 500,
        'seed': 1265092,
        'path_list': [
            ['squad', 'squad_v2/validation'],
            ['vqa', ('vqa_v2/v2_OpenEnded_mscoco_val2014_questions.json', 'vqa_v2/v2_mscoco_val2014_annotations.json')],
        ],
        'tasks': ['qid', 'question', 'answer', 'num_qs'],
        'file_name': 'sa_dev.json',
    }
}


class Wrapper(DatasetWrapper):
    """
    splits: train, validation, test
    features: ['sentence', 'label', 'idx']
    """
    
    name = 'sa'
    
    handler_class = TSearchAndAnswer
    
    requires_handlers = ['coco']
    
    exposed_splits = {'train': 'train', 'valid': 'validation'}
    
    def load(self):
        data = {}
        for split_name, setting in SPLIT_SETTINGS.items():
            path = os.path.join(self.data_dir, self.name, setting['file_name'])
            entries = []
            for eid, example in enumerate(json.load(open(path))):
                for task in example['tasks']:
                    entries.append({
                        'home_url': f'/sa?sid={split_name}&eid={eid}',
                        'question': task['question'],
                        'answer': task['answer'],
                        'unanswerable': task['unanswerable'],
                        'gold': task['gold'],
                        'task_type': task['task_type'],
                    })
            data[split_name] = entries
        return data
    
    def make_data(self):
        return self.load()
    
    def make_metadata(self):
        return {
            'name': self.name,
        }
    
    def make_task_handlers(self):
        
        handlers = []
        
        # inner pages
        json_paths = {
            k: os.path.join(self.data_dir, self.name, v['file_name']) 
            for k, v in SPLIT_SETTINGS.items()
        }
        handlers += [
            (r'/sa', TSearchAndAnswerInnerPage, 
            {'datasets': {k: json.load(open(v)) for k, v in json_paths.items()}}), 
        ]
        
        data = self.make_data()
        metadata = self.make_metadata()
        handlers += [
            (r'/%s/%s' % (dest, self.name), self.handler_class, {'data': data[src], 'metadata': metadata})
            for dest, src in self.exposed_splits.items()
        ]
        return handlers, set(self.requires_handlers)
    
    def setup(self):
        local_dir = os.path.join(self.data_dir, self.name)
        if not os.path.exists(local_dir):
            os.mkdir(local_dir)
        
        def join_dir(base_dir, path_list):
            new_list = []
            for name, paths in path_list:
                if isinstance(paths, str):
                    new_list.append([name, os.path.join(base_dir, paths)])
                else:
                    new_list.append([name, [os.path.join(base_dir, _) for _ in paths]])
            return new_list
        
        for split_name, setting in SPLIT_SETTINGS.items():
            setting['path_list'] = join_dir(self.data_dir, setting['path_list'])
            file_path = os.path.join(local_dir, setting['file_name'])
            examples = sample_examples(setting)
            json.dump(examples, open(file_path, 'w'), indent=2)
            print(file_path, 'saved')
    

def load_contexts(path_list):
    """load contexts and their related questions from path_list
    returns a list of dicts whose keys are {'context_type', 'text', 'image_path', 'questions'}.
    a dict contains either text or image_path according to its context_type ('text' / 'image')
    questions consist of a list of question represented as a dict {'question':str, 'answer':str, 'unanswerable':bool}
    """
    
    contexts = []
    for source_type, path in path_list:
        if source_type == 'squad':
            contexts.append(load_contexts_squad(path))
        elif source_type == 'vqa':
            contexts.append(load_contexts_vqa(path))
        else:
            raise RuntimeError(f'Unknown source: {source_type}')
    return contexts


def normalize_q(text):

    return re.sub('\s+', ' ', text).strip()


def load_contexts_squad(path):
    """load the paragraphs and their questions from squad v2 as contexts"""
    
    contexts = OrderedDict()
    for d in datasets.load_from_disk(path):
        context = d['context'].strip()
        if context not in contexts:
            contexts[context] = {
                'context_type': 'text',
                'text': context,
                'questions': [],
            }
        has_answers = len(d['answers']['text']) > 0
        contexts[context]['questions'].append({
            'question': normalize_q(d['question']),
            'answer': d['answers']['text'][0].strip() if has_answers else '',
            'unanswerable': not has_answers,
        })
    contexts = [v for k, v in contexts.items()]
    return contexts


def load_contexts_vqa(path):
    """load the images and their questions from vqa v2 as contexts"""
    
    q_path, a_path = path
    
    annotation_dict = {_['question_id']:_ for _ in  json.load(open(a_path))['annotations']}
    questions = json.load(open(q_path))['questions']
         
    context_dict = {}
    for q in questions:
        image_id = q['image_id']
        context_dict.setdefault(image_id, []).append({
            'question': normalize_q(q['question']), 
            'answer': annotation_dict[q['question_id']]['multiple_choice_answer'], 
            'unanswerable': False,
        })
    
    if 'val2014' in q_path:
        image_path_tmp = '/coco/val2014/COCO_val2014_%012d.jpg'
    elif 'train2014':
        image_path_tmp = '/coco/train2014/COCO_train2014_%012d.jpg'
    else:
        raise RuntimeError('Unknown split: %s' % q_path)
    
    contexts = []
    for image_id, questions in context_dict.items():
        contexts.append({
            'context_type': 'image',
            'image_path': image_path_tmp % image_id,
            'questions': questions,
        })
    
    return contexts


def sample_examples(spec):
    """Sample examples according to the spec dict
    for the detail of spec dict see the settings block below.
    """
    
    random = np.random.RandomState(spec['seed'])
    
    # Get contexts from datasets
    context_sources = load_contexts(spec['path_list'])
    num_context_sources = len(context_sources)
    
    # Sampling contexts
    ids = {i: list(range(len(s))) for i, s in enumerate(context_sources)}
    for _ids in ids.values():
        random.shuffle(_ids)
    
    num_contexts_per_example = spec['num_contexts_per_example']
    example_seeds = []
    for _ in range(spec['num_examples']):
        contexts = []
        for i, s in enumerate(context_sources):
            _ids = ids[i]
            for j in range(num_contexts_per_example[i]):
                contexts.append(s[_ids.pop(0)])
        example_seeds.append({'contexts': contexts})
    
    # Adding IDs for the contexts and questions
    def add_search_index_to_question(contexts, question):
        context = contexts[question['cid']]
        question['index'] = ('%s %s %s %s' % (
            question['qid'],
            question['cid'],
            context['context_type'],
            question['question'],
        )).lower()
        
    get_cid_str = lambda cid: 'C%05d' % cid
    get_qid_str = lambda qid: 'Q%05d' % qid
    
    examples = []
    for example_seed in example_seeds:
        contexts = {}
        questions = []
        # giving cid
        random.shuffle(example_seed['contexts'])
        for cid, context in enumerate(example_seed['contexts']):
            context['cid'] = get_cid_str(cid)
            contexts[context['cid']] = context
            for question in context['questions']:
                question['cid'] = context['cid']
            questions.extend(context['questions'])
        
        # giving qid
        random.shuffle(questions)
        for qid, question in enumerate(questions):
            question['qid'] = get_qid_str(qid)
            add_search_index_to_question(contexts, question)
        
        # complie a dict
        examples.append({
            'num_cid': cid,
            'num_qid': qid,
            'contexts': contexts,
            'questions': questions,
        })
    
    # Making tasks
    max_num_task = spec['num_instructions_per_example']
    target_tasks = spec['tasks']
    for example in examples:
        add_tasks(example, target_tasks)
        random.shuffle(example['tasks'])
        example['tasks'] = example['tasks'][:max_num_task]
    
    return examples


def add_tasks(example, target_tasks):
    """Add tasks to each example"""
    
    def take_head_words(text, n=3):
        re_not_allowed = re.compile('[^a-zA-Z0-9?."\'\-\_&,=–%#()+:;$/{}!’]')
        all_words = text.split()
        
        # first, try to get n words
        words = []
        for w in all_words[:n]:
            if re_not_allowed.search(w):
                break
            words.append(w)
        if len(words) == n:
            return words
        
        # second, get the longest word from allowed words
        w = sorted([_ for _ in all_words if re_not_allowed.search(_) is None], key=lambda _:-len(_))
        if len(w) == 0:
            raise RuntimeError('Bad text: %s' % text)
        #print('second method', text, ':', w[0])
        return [w[0]]
        
    tasks_about_question = {
        'qid': ('What is the QID of the question "%s" ?', 'question', 'qid', False),
        'question': ('What is the question of %s?', 'qid', 'question', False),
        'answer': ('Answer the question of %s.', 'qid', 'answer', None),
    }
    
    tasks = []
    for target_task in target_tasks:
        if target_task == 'num_qs':
            q_template = 'How many questions are related to %s?'
            for cid, context in example['contexts'].items():
                q = q_template % cid
                a = '%d' % (len(context['questions']))
                u = False
                g = [('query', cid.lower())]
                tasks.append({'question':q, 'answer':a, 'unanswerable': u, 'gold': g, 'task_type': target_task})
        
        if target_task in tasks_about_question:
            q_template, q_key, a_key, unanswerable = tasks_about_question[target_task]
            for question in example['questions']:
                _q = question[q_key]
                q = q_template % _q
                a = question[a_key]
                u = question['unanswerable'] if unanswerable is None else unanswerable
                
                if target_task == 'qid':
                    g = [('query', ' '.join(take_head_words(_q)).lower())]
                else:
                    g = [('query', _q.lower())]
                
                if target_task == 'answer':
                    g.append(('click', '#show_%s' % _q))
                elif target_task == 'qid':
                    g.append(('move', '#qid_%s' % question['qid']))
                
                tasks.append({'question':q, 'answer':a, 'unanswerable': u, 'gold': g, 'task_type': target_task})
        
    example['tasks'] = tasks
