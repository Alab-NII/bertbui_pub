# coding: utf-8
# Script for evaluation
# 2021-11-24
# Taichi Iki
#


import os
import re
import json
from tasksvr import Wrappers, get_default_data_dir


# Metrics
from sklearn.metrics import classification_report, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr
import subprocess # For external scripts
import tempfile


# Supported Tasks
supported_tasks = {
    'wnli': {'metrics': ['classification', 'exact_match'], 'cast':'int'},
    'rte': {'metrics': ['classification', 'exact_match'], 'cast':'int'},
    'mrpc': {'metrics': ['classification', 'exact_match'], 'cast':'int'},
    'stsb': {'metrics': ['classification', 'pearson', 'spearman', 'exact_match'], 'cast':'int'},
    'cola': {'metrics': ['classification', 'matthews', 'exact_match'], 'cast':'int'},
    'sst2': {'metrics': ['classification', 'exact_match'], 'cast':'int'},
    'qnli': {'metrics': ['classification', 'exact_match'], 'cast':'int'},
    'qqp': {'metrics': ['classification', 'exact_match'], 'cast':'int'},
    'mnli': {'metrics': ['classification', 'exact_match'], 'cast':'int'},
    'squad_v2': {'metrics': ['squad_script', 'exact_match'], 'cast':'str'},
    'vqa_v2': {'metrics': ['vqa_script', 'exact_match'], 'cast':'str'},
    'pta': {'metrics': ['exact_match'], 'args': {'exact_match_key': 'task_type'}, 'cast':'str'},
    'sa': {'metrics': ['exact_match'], 'args': {'exact_match_key': 'task_type'}, 'cast':'str'},
}
all_supported_tasks = list(supported_tasks.keys())


class Metrics(object):
    
    @staticmethod
    def num_answered(is_answered, ommit_detail=True, *args, **kwargs):
        n_total = len(is_answered)
        n_answered = sum(is_answered)
        n_not_answred = n_total - n_answered
        lines = []
        lines += ['total examples: %d' % n_total]
        lines += ['num not answred: %d (%.5f)' % (n_not_answred, n_not_answred / n_total)]
        if not ommit_detail:
            lines += ['ids: '+str([i for i, mask in enumerate(is_answered) if not mask])]
        return '\n'.join(lines)
    
    @staticmethod
    def classification(y_true, y_pred, labels, target_names, digits=3, *args, **kwargs):
        return classification_report(y_true, y_pred, labels=labels, target_names=target_names, digits=digits)
    
    @staticmethod
    def matthews(y_true, y_pred, *args, **kwargs):
        return matthews_corrcoef(y_true, y_pred)
    
    @staticmethod
    def exact_match(y_true, y_pred, is_answered, y_key=None, ommit_detail=True, *args, **kwargs):
        normalize = lambda text: str(text).strip().lower()
        
        lines = []
        
        n_total_sub = {}
        n_correct_sub = {}
        n_answered_sub = {}
        if y_key is not None:
            for k, t, p, m in zip(y_key, y_true, y_pred, is_answered):
                is_correct = normalize(t)==normalize(p)
                n_total_sub[k] = n_total_sub.get(k, 0) + 1
                n_correct_sub[k] = n_correct_sub.get(k, 0) + int(is_correct)
                n_answered_sub[k] = n_answered_sub.get(k, 0) + int(m)
            lines += ['\t'.join(['key', 'total', 'no_ans', 'ans', 'correct', 'acc'])]
            for k in sorted(n_total_sub):
                n_total = n_total_sub[k]
                n_correct = n_correct_sub[k]
                n_not_answered = n_total - n_answered_sub[k]
                n_answered = n_answered_sub[k]
                lines += [f'{k}\t{n_total}\t{n_not_answered}\t{n_answered}\t{n_correct}\t{n_correct / n_total}']
        else:
            n_total_sub[None] = len(y_true)
            n_correct_sub[None] = sum([normalize(t)==normalize(p) for t, p in zip(y_true, y_pred)])
            n_answered_sub[None] = sum(is_answered)
        n_total = sum(n_total_sub.values())
        n_correct = sum(n_correct_sub.values())
        n_not_answred = n_total - sum(n_answered_sub.values())
        n_answered = n_total - n_not_answred        

        lines += [f'total_num: {n_total}']
        lines += [f'correct_num: {n_correct}']
        lines += [f'acuracy (total): {n_correct/n_total}']
        lines += ['num not answred: %d (%.5f)' % (n_not_answred, n_not_answred / n_total)]
        lines += ['num answred: %d (%.5f)' % (n_answered, n_answered / n_total)]
        lines += [f'acuracy (answered): {n_correct/(n_answered if n_answered > 0 else 1)}']
        if not ommit_detail:
            lines += ['ids: '+str([i for i, mask in enumerate(is_answered) if not mask])]
        return '\n'.join(lines)
    
    @staticmethod
    def pearson(s_true, s_pred, *args, **kwargs):
        return pearsonr(s_true, s_pred)

    @staticmethod
    def spearman(s_true, s_pred, *args, **kwargs):
        return spearmanr(s_true, s_pred)
    
    @staticmethod
    def vqa_script(pred_list, annotation_path, question_path, *args, **kwargs):
        
        # make a temporal file to pass the squad script
        fd, pred_file_path = tempfile.mkstemp(text=True)
        json.dump(pred_list, open(pred_file_path, 'w'))
        
        # required a dict {task_key: answer in string}
        script_path = os.path.join(os.path.dirname(__file__), 'eval_vqa.py')
        cmd = f'python {script_path} --annotation_path {annotation_path} ' \
            + f'--question_path {question_path} --prediction_path {pred_file_path}'
        output = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, check=False)
        
        os.remove(pred_file_path)
        
        lines = []
        lines += [cmd]
        lines += [output.stdout.decode()]
        return '\n'.join(lines)
    
    @staticmethod
    def squad_script(pred_dict, data_file_path, *args, **kwargs):
        
        # make a temporal file to pass the squad script
        fd, pred_file_path = tempfile.mkstemp(text=True)
        json.dump(pred_dict, open(pred_file_path, 'w'))
        
        # required a dict {task_key: answer in string}
        script_path = os.path.join(os.path.dirname(__file__), 'evaluate-v2.0.py')
        cmd = f'python {script_path} {data_file_path} {pred_file_path}'
        output = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, check=False)
        
        os.remove(pred_file_path)
        
        lines = []
        lines += [cmd]
        lines += [output.stdout.decode()]
        return '\n'.join(lines)


def compile_result_list_vqa(result_path, question_path):

    result_dict = json.load(open(result_path))
    prefix = result_dict['task_path']+'?id='
    pred = result_dict['predictions']

    result_list = []
    for i, q in enumerate(json.load(open(question_path))['questions']):
        answer = pred[prefix+str(i)][0]
        answer = answer or ''
        r = {'question_id':q['question_id'], 'answer':answer}
        result_list.append(r)

    return result_list


def eval_from_json(json_path, ommit_detail=True):
    
    print('='*10)
    
    json_dict = json.load(open(json_path, 'r'))
    print(json_path)
    
    if not ('name' in json_dict and 'predictions' in json_dict):
        print('prediction files required the name key', json_path)
        return
    
    if  not ('task_path' in json_dict and 'predictions' in json_dict):
        print('prediction files required the task_path key', json_path)
        return
    
    _, exposed_split, wrapper_name = json_dict['task_path'].split('/')
    wrapper = wrappers[wrapper_name]
    data_split = wrapper.exposed_splits[exposed_split]
    data = wrapper.load()[data_split]
    
    if wrapper_name not in all_supported_tasks:
        print(f'not supported task: {wrapper_name}')
        return
    
    task_metadata = supported_tasks[wrapper_name]
    metrics = task_metadata['metrics']
    cast_type =  task_metadata['cast']
    if cast_type == 'int':
        def cast_func(_):
            if _ is None:
                return -100
            return int(_)
    elif cast_type == 'str':
        def cast_func(_):
            if _ is None:
                return '[NOT_ANSWERED]'
            return str(_)
    else:
        def cast_func(_):
            return _
    
    predictions = json_dict['predictions']
    if isinstance(predictions, dict):
        # convert a dict to a list
        task_path = json_dict['task_path']
        # change keys
        r = re.compile('\?id\=([0-9]+)')
        predictions = {int(r.findall(k)[0]):v for k, v in predictions.items()}
        prediction_list = []
        for i in range(len(data)):
            prediction_list.append([i] + predictions[i])
        predictions = prediction_list
    # predictions format: [id, answer, answer, state, step]
    
    if len(data) != len(predictions):
        print(f'num of examples not mached.')
        return
    
    # Extract Ground Truth
    def get_ground_truth(d):
        if 'label' in d:
            return d['label']
        elif 'answer' in d:
            return d['answer']
        elif 'answer_str' in d:
            x = json.loads(d['answer_str'])
            if isinstance(x, dict):
                return x['answer']
            return str(x)
        elif 'answers' in d:
            return d['answers']['text']
        raise RuntimeError(f'answer not found in {d}')
    ground_truth_seq = [cast_func(get_ground_truth(_)) for _ in data]
    
    # Extract predictions
    id_answer_raw = 1
    if cast_type == 'int':
        def _extract(p):
            return cast_func(p[id_answer_raw])
    elif answer_type == 'str':
        def _extract(p):
            x = p[id_answer_raw]
            if isinstance(x, dict):
                x = x['answer']
            return cast_func(x)
    is_answered = [(_[id_answer_raw] is not None) for _ in predictions]
    prediction_seq = [_extract(_) for _ in predictions]
    
    args = {}
    
    # Append arguments
    args['y_pred'] = prediction_seq
    args['y_true'] = ground_truth_seq
    args['is_answered'] = is_answered
    
    if 'classification' in metrics:
        labels = wrapper.labels
        args['labels'] = labels
        label_rule = wrapper.label_rules[0]
        args['target_names'] = [label_rule.get(_, str(_)) for _ in labels]
    
    if 'pearson' in metrics or 'spearman' in metrics:
        # remove not answered examples
        args['s_pred'] = [float(_) for mask, _ in zip(is_answered, prediction_seq) if mask]
        args['s_true'] = [_['label'] for mask, _ in zip(is_answered, data) if mask]
    
    if 'squad_script' in metrics:
        args['data_file_path'] = os.path.join(wrapper.data_dir, wrapper.name, data_split)
        args['pred_dict'] = {d['id']:p for p, d in zip(prediction_seq, data)}
    
    if 'vqa_script' in metrics:
        if data_split == 'validation':
            args['annotation_path'] = os.path.join(wrapper.data_dir, wrapper.name, 'v2_mscoco_val2014_annotations.json')
            args['question_path'] = os.path.join(wrapper.data_dir, wrapper.name, 'v2_OpenEnded_mscoco_val2014_questions.json')
        else:
            print('unknown split', task_split)
            return
        args['pred_list'] = [{'question_id':d['question_id'], 'answer':p} for p, d in zip(prediction_seq, data)]
    
    if 'exact_match_key' in task_metadata.get('args', {}):
        key = task_metadata['args'].pop('exact_match_key')
        args['y_key'] = [_[key] for _ in data]
    
    args.update(task_metadata.get('args', {}))
    args['ommit_detail'] = ommit_detail
    
    # Calculation
    for metric in metrics:
        results = getattr(Metrics, metric)(**args)
        print('#', metric)
        print(str(results).rstrip() + '\n')
        

def eval_from_dir(dir_path, ommit_detail):

    path_list = []

    for root, dirs, files in os.walk(dir_path):
        if os.path.basename(root) == 'predictions':
            for fname in files:
                path_list.append(os.path.join(root, fname))

    for fpath in sorted(path_list):
        eval_from_json(fpath, ommit_detail)

        
def main():
    
    import argparse

    parser = argparse.ArgumentParser(description='evaluate predictions files')
    parser.add_argument('paths', metavar='P', type=str, nargs='+',
                    help='paths to be evaluated')
    parser.add_argument('--data_dir', type=str, default=None,
                    help='paths to a data directory')
    parser.add_argument('--ommit_detail', type=int, default=1,
                    help='ommit long detail if not zero')

    args = parser.parse_args()
    
    global wrappers
    data_dir = args.data_dir
    if data_dir is None:
        data_dir = get_default_data_dir()
    wrappers = Wrappers.enumerate(data_dir)
    
    ommit_detail = args.ommit_detail
    
    for path in args.paths:
        if os.path.isdir(path):
            eval_from_dir(path, ommit_detail)
        else:
            eval_from_json(path, ommit_detail)

if __name__ == '__main__':
    main()
