# coding: utf-8
# Prediction using browser models
# 2021-11-22
# Taichi Iki

import dataclasses as DC

import multiprocessing as mp
import multiprocessing.pool as mp_pool
import time

import logging
logger = mp.log_to_stderr()
logger.setLevel(logging.DEBUG)

import json
import os
import torch

from bertbui import (
    ModelAction, EnvSTFirefox ,get_all_ids, parse_from_dataclass, read_metadata
)
from bertbui.models_for_browser import ModelConfig, ReaderModule


GLUE_TASKS = ['wnli', 'rte', 'mrpc', 'stsb', 'cola', 'sst2', 'qnli', 'qqp', 'mnli']
QA_TASKS = ['squad_v2', 'vqa_v2']
INTERACTIVE_TASKS = ['sa']
PTA_TASKS = ['pta']

DEFAULT_FIREFOX_BINARY = os.path.join(os.environ.get('VIRTUAL_ENV', ''), 'firefox')

@DC.dataclass
class PredictionConfig:
    
    _desc = 'Configuration for prediction'
    
    model_path: str = DC.field(metadata={'help':'path to a model directory.'})
    task_path: str = DC.field(metadata={'help':'key to a task, such as /valid/wnli'})
    
    weight: str = DC.field(default=None, metadata={'help':'specific weight if not given pytorch_model.bin will be used.'})
    time_steps_scale: float = DC.field(default=1.5, metadata={'help':'if positive, steps are determined from metadata by multiplying this value'})
    max_time_steps: int = DC.field(default=200, metadata={'help':'max time steps before time out.'})
    overwrite: int = DC.field(default=0, metadata={'help':'If not zero, prediction file will be overwrite'})
    num_block_examples: int = DC.field(default=1000, metadata={'help': 'it will save prediction each num_block_examples'})
    port: str = DC.field(default='9973', metadata={'help':'port of the task server.'})
    n_workers: int = DC.field(default=4, metadata={'help':'number of worker threads'})
    data_dir: str = DC.field(default='data', metadata={'help':'path to a data directory'})
    static_dir: str = DC.field(default='static', metadata={'help':'path to a static directory'})
    device_name: str = DC.field(default='cpu', metadata={'help':'device for model'})
    firefix: str = DC.field(default=DEFAULT_FIREFOX_BINARY, metadata={'help':'device for model'})
        
    def get_base_url(self):
        return 'http://localhost:%s'%(self.port)
    
    def decode_dataset_key(self):
        """dataset key decoding
        returns three strings: name, split, task_path
        """
        split, name = self.task_path.split('/')[-2:]
        metadata_path = f'{self.static_dir}/{split}/{name}'
        
        return name, split, self.task_path, metadata_path


sub_context = None
def sub_init(config):
    """Initialize sub process"""
    
    cur_proc = mp.current_process()
    
    # Initialize models
    model_weight = 'pytorch_model.bin'
    if config.weight is not None:
        model_weight = os.path.join('weights', config.weight)
    config_json = json.load(open(os.path.join(config.model_path, 'model_config.json'), 'r'))
    model_config= ModelConfig(**config_json)
    model = ReaderModule(model_config, config.model_path)
    state_dict = torch.load(os.path.join(config.model_path,  model_weight), map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.to(config.device_name)
    model.eval()
    
    # Initialize environment
    env = EnvSTFirefox(
        url_base=config.get_base_url(), 
        tokenizer=model.tokenizer, 
        window_size=model.config.window_size,
        binary_location=config.firefix,
    )
    
    # Expose
    global sub_context
    sub_context = {
        'model': model,
        'env': env,
    }


def sub_main(x):
    """make a prediction for the task x"""
    _id, max_time_steps = x
    model = sub_context['model']
    env = sub_context['env']
    
    observation = env.reset(url=_id)
    state = (observation, None)
    submission = None
    for time_step in range(max_time_steps):
        state = model.step(env, *state)
        status = env.get_status()
        if status:
            submission = env.get_submitted()
            break
    
    return {
        '_id': _id, 
        'status': status if status else 'timeout',
        'submission': submission,
        'loop_num': time_step+1, 
    }


_worker = mp_pool.worker
def worker(inqueue, outqueue, initializer=None, initargs=(), maxtasks=None, wrap_exception=False):
    _worker(inqueue, outqueue, initializer, initargs, maxtasks, wrap_exception)
    del sub_context['model']
    sub_context['env']._unset_driver()
    del sub_context['env']
    print('sub_context deleted')


def predict(config):
    """
    - preparing dataset
    - calculate instances to be handled
    - prepating models and browser in each process
    - prediction
    """
    print(config)
    
    mp_pool.worker = worker

    dataset_name, dataset_split, \
            task_path, metadata_path = config.decode_dataset_key()
    
    # Determine save path 
    predictions_dir = os.path.join(config.model_path, 'predictions')
    if not os.path.exists(predictions_dir):
        os.mkdir(predictions_dir)
    save_name = f'{dataset_name}.{dataset_split}'
    if config.weight:
        save_name = config.weight + '.' + save_name
    save_path = os.path.join(predictions_dir, save_name)
    
    # Calculate targets
    all_ids = get_all_ids(config.get_base_url(), task_path)
    if (not bool(config.overwrite)) and os.path.exists(save_path):
        output = json.load(open(save_path))
        all_ids = [_ for _ in all_ids if _ not in output['predictions']]
    else:
        output = {
            'name':save_name,
            'task_path':task_path,
            'predictions':{},
        }
    
    if len(all_ids) == 0:
        print('All prediction already exist. ')
        print('If you want to overwrite the past prediction, please add "--overwrite 1".')
        return
    
    # Decide max steps
    if config.time_steps_scale > 0:
        metadata = read_metadata(metadata_path, validate=False)
        id_step_list = []
        for _id in all_ids:
            key = _id.split('?id=')[-1] + '.json'
            num_actions = metadata[key]['num_actions']
            max_time_steps = int(round(config.time_steps_scale*num_actions))
            id_step_list.append((_id, max_time_steps))
    else:
        # max_time_steps for the all samples
        id_step_list = [(_, config.max_time_steps) for _ in all_ids]
    
    # Post process
    def post_process(result):
        # the third column is compatibility
        return result['submission'], result['status'], result['loop_num']
    
    print('This execution will handle %d examples with %d workers' \
          % (len(id_step_list), config.n_workers)
    )
    
    stime = time.time()
    with mp.Pool(config.n_workers, initializer=sub_init, initargs=(config,)) as p:
        try:
            for i in range(0, len(id_step_list), config.num_block_examples):
                block_results = p.map(sub_main, id_step_list[i:i+config.num_block_examples])
                output['predictions'].update([(_['_id'], post_process(_)) for _ in block_results])
                json.dump(output, open(save_path, 'w'), indent=1)
                delta_time = int(time.time() - stime)
                n_completed = i+len(block_results)
                print('completed', n_completed,
                      'total elapsed', delta_time, 'per sample',  delta_time / n_completed)
        finally:
            p.close()
            p.join()
            print('pool closed')

    print('All done')
    

if __name__ == '__main__':
    
    config = parse_from_dataclass(PredictionConfig, 'configuration for prediction')
    predict(config)

