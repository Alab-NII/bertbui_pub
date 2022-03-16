# coding: utf-8


import dataclasses as DC

import time
import os
import shutil
import json

import numpy as np
import torch
#torch.autograd.set_detect_anomaly(True)

from bertbui import (
        parse_from_dataclass, 
        ConcatDataset
    )
import bertbui.models_for_browser as models

DATASET_AND_MODEL_CLASS = {
        'classification': (ConcatDataset, models.ReaderModule),
        'extraction': (ConcatDataset, models.ReaderModule),
    }


@DC.dataclass
class TrainingConfig:
    
    _desc = 'Configuration for model training'
    
    model_path: str = DC.field(metadata={'help':'path to a directory where the weights will be saved.'})
    
    mode: str = DC.field(default='train', 
            metadata={'help':'script mode.'})
    
    task_type: str = DC.field(default='classification',
            metadata={'help':'classification or extraction'})
    
    train_datasets: str = DC.field(default='static/train/wnli',
            metadata={'help':'dataset name for training, comma splitted.'})
    
    train_pack_seed: int = DC.field(default=123,
            metadata={'help':'random seed for shuffle training data'})

    valid_datasets: str = DC.field(default='static/valid/wnli',
            metadata={'help':'dataset name for validation, comma splitted.'})
    
    valid_max_num: int = DC.field(default=None, 
            metadata={'help':'maximum number of valid examples per data source.'})
        
    valid_sampling_seed: int = DC.field(default=123, 
            metadata={'help':'a random seed for example sampling.'})
    
    minibatch_size: int = DC.field(default=1, 
            metadata={'help':'examples in a step.'})
        
    iters_to_accumulate: int = DC.field(default=2,
            metadata={'help':'the number of *steps* whose gradients will be acumurated.'})    
    
    interval_to_show_loss: int = DC.field(default=-1, 
            metadata={'help':'show average loss every those *steps*.'})
    
    epochs_to_save: float = DC.field(default=1, 
            metadata={'help':'save model weights every those *epochs*.'})
    
    max_epoch: int = DC.field(default=10, 
            metadata={'help':'The maximum of epoch.'})
    
    random_seed: int = DC.field(default=123, 
            metadata={'help':'a random seed for minibatch sampling.'})
        
    amp_enabled: int = DC.field(default=0, 
            metadata={'help':'When 1, we use autocast of torch'})
    
    device_name: str = DC.field(default='cuda:0', 
            metadata={'help':'When 1, we use autocast of torch'})
    
    optimizer_lr: float = DC.field(default=5e-5, 
            metadata={'help':'Learning rate for the adam optimizer'})
    
    weight_seed: str =  DC.field(default='', 
            metadata={'help':'If set, the model weight is initialized with this weight.'})
    
    max_step_num: int = DC.field(default=50, 
            metadata={'help':'trancuate steps excesses this value.'})

    @property
    def train_log_path(self):
        
        return os.path.join(self.model_path, 'train_log.txt')
    
    @property
    def valid_history_path(self):

        return os.path.join(self.model_path, 'valid_history.json')

    @property
    def weight_dir(self):
        
        return os.path.join(self.model_path, 'weights')
    
    @property
    def amp_enabled_bool(self):
        
        return self.amp_enabled != 0

    def join_path(self, *args):
        
        return os.path.join(self.model_path, *args)


def seed_worker(worker_id):
    
    worker_seed = (torch.initial_seed()+worker_id) % 2**32
    np.random.seed(worker_seed)
    #random.seed(worker_seed)

    
def train(config):
    
    print('training starts with the configuration:')
    print(config)
    
    torch.manual_seed(config['train'].random_seed)
    seed_worker(-1)
    
    os.makedirs(config['train'].model_path, exist_ok=True)
    os.makedirs(config['train'].weight_dir, exist_ok=True)
    
    
    with open(config['train'].join_path('training_config.json'), 'w') as f:
        json.dump(config['train'].__dict__,  f)
    
    with open(config['train'].join_path('model_config.json'), 'w') as f:
        json.dump(config['model'].__dict__, f)
    
    shutil.copyfile(models.__file__, config['train'].join_path('models.py'))
    shutil.copyfile(config['model'].tokenizer_path, config['train'].join_path('tokenizer.json'))
    with open(config['train'].train_log_path, 'a') as f:
        print(config, file=f)
    
    dataset_class, model_class = DATASET_AND_MODEL_CLASS[config['train'].task_type]

    data_loaders = {}
    data_loaders['train'] = torch.utils.data.DataLoader(
        dataset_class(config['train'], config['model'], config['train'].train_datasets.split(','), config['train'].train_pack_seed),
        collate_fn=dataset_class.collate_fn, 
        batch_size=config['train'].minibatch_size, shuffle=True, 
        num_workers=4, drop_last=False,
        worker_init_fn=seed_worker,
        prefetch_factor=2
    )
    print('train examples:', len(data_loaders['train'].dataset))
    
    data_loaders['valid'] = torch.utils.data.DataLoader(
        dataset_class(config['train'], config['model'], config['train'].valid_datasets.split(','),
                 max_num=config['train'].valid_max_num,
                 sampling_seed=config['train'].valid_sampling_seed
            ),
        collate_fn=dataset_class.collate_fn, 
        batch_size=config['train'].minibatch_size, shuffle=False, 
        num_workers=4, drop_last=False,
        prefetch_factor=2,
    )
    print('valid examples:', len(data_loaders['valid'].dataset))
    
    # model creation
    model = model_class(config['model'])
    # save base bert config
    model.base_bert_config(config['train'].model_path)
    
    if config['train'].weight_seed:
        model.load_state_dict(torch.load(config['train'].weight_seed, map_location='cpu'))
    device = torch.device(config['train'].device_name)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train'].optimizer_lr)
    scaler = torch.cuda.amp.GradScaler(enabled=config['train'].amp_enabled_bool)
    
    # Collecting context
    context = {
        'config': config,
        'task_type': config['train'].task_type,
        'device': device,
        'model': model,
        'optimizer': optimizer,
        'scaler': scaler,
        'steps_to_save': int(len(data_loaders['train'])*config['train'].epochs_to_save),
        'save_hook': lambda c: save_hook(c, data_loaders['valid']),
        'global_stat': {
            'epoch': 0,
            'n_total_training_steps': 0,
            'n_updates': 0,
            'n_total_examples_used_for_updates': 0,
        },
        'valid_history': [],
    }
    print('will save every %d steps' % context['steps_to_save'])
    
    # training loop
    for epoch_id in range(config['train'].max_epoch):
        
        context['global_stat']['epoch'] += 1
        
        model.train()
        feed_ids(context, data_loaders['train'], is_training=True)
    
    # best model
    save_name = None
    best_valid_loss = float('inf')
    for record in context['valid_history']:
        bl = float(record['bl'])
        if bl < best_valid_loss:
            best_valid_loss = bl
            save_name = record['save_name']
    if save_name:
        print(f'best validation loss: {best_valid_loss} @ {save_name}')
        shutil.copyfile(
            os.path.join(config['train'].weight_dir, save_name), 
            config['train'].join_path('pytorch_model.bin')
        )


def save_hook(context, data_loader_valid):
    """save and validate model"""
        
    model = context['model']
    save_name = 's%d'%(context['global_stat']['n_total_training_steps'])
    save_path = os.path.join(context['config']['train'].weight_dir, save_name)
    torch.save(model.state_dict(), save_path)
    
    print(f'model saved: {save_name}')
    
    with torch.no_grad():
        model.eval()
        results = feed_ids(context, data_loader_valid, is_training=False)
    
    results['save_name'] = save_name
    
    context['valid_history'].append(results)
    with open(context['config']['train'].valid_history_path, 'w') as f:
        json.dump(context['valid_history'], f)


def feed_ids(context, data_loader, is_training):
    
    # Obtain context
    config = context['config']
    task_type = context['task_type']
    device = context['device']
    model = context['model']
    optimizer = context['optimizer']
    scaler = context['scaler']
    save_hook = context['save_hook']
    steps_to_save = context['steps_to_save']
    global_stat = context['global_stat']
    
    amp_enabled = config['train'].amp_enabled_bool
    max_iters_to_accumulate = config['train'].iters_to_accumulate
    
    interval_to_show_loss = config['train'].interval_to_show_loss
    if interval_to_show_loss < 0:
        interval_to_show_loss = max(1, int(len(data_loader)/20))
    
    memory_mode = None
    
    # Initialize statistic
    stat = {}
    def clear_stat():
        stat['n_processed'] = 0
        stat['loss_sum'] = 0
        stat['loss_details_sum'] = {}
        stat['n_sum'] = 0
        stat['max_step_sum'] = 0
    
    def print_stat(local_n_processed, elapsed_time):
        n_sum = stat['n_sum']
        loss_avg = stat['loss_sum'] / n_sum
        loss_details_avg = {k: v/n_sum for k, v in stat['loss_details_sum'].items()}
        data = [
            'ep=%d'%global_stat['epoch'],
            'gs=%d'%global_stat['n_total_training_steps'],
            'gu=%d'%global_stat['n_updates'],
            'ge=%d'%global_stat['n_total_examples_used_for_updates'],
            'lm=%s'%('T' if is_training else 'V'),
            'lp=%d'%local_n_processed,
            'bp=%d'%stat['n_processed'], 
            'bt=%.3f'%elapsed_time, 
            'bl=%.6f'%loss_avg,
        ]
        data.extend('bl%s=%.4f'%_ for _ in loss_details_avg.items())
        text = ' '.join(data)
        print(text)
        with open(config['train'].train_log_path, 'a') as f:
            print(text, file=f)
        return data
    
    # initialization
    last_time = time.time()
    local_n_processed = 0
    n_accum_steps = 0
    n_left_steps = len(data_loader)
    clear_stat()
    
    model.train(is_training)
    if is_training:
        optimizer.zero_grad()
    
    for batch_id, batch in enumerate(data_loader):
        
        # Batch step contains:
        #     1. Status updation before model calculation
        #     2. Model forward calculation
        #     3. Model backward calculation
        #     4. Status updation after model calculation
        #     5. Saving data
        
        #
        # 1. Status updation before model calculation
        #
        # Determine the next number of accumulation
        show_loss = is_training and (stat['n_sum'] >= interval_to_show_loss)
        save_model = is_training and (global_stat['n_total_training_steps'] + 1) % steps_to_save == 0
        # +1 accounts for this step
        
        if n_accum_steps == 0:
            n_accum_examples = 0
            if is_training:
                iters_to_accumulate = min(
                    max_iters_to_accumulate,
                    steps_to_save - (global_stat['n_total_training_steps'] % steps_to_save),
                    n_left_steps
                )
            else:
                iters_to_accumulate = 1
        do_update = is_training and (n_accum_steps + 1 >= iters_to_accumulate)
        # +1 accounts for this step
        
        #
        # 2. Model forward calculation
        #
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            
            batch = batch.to_device(device)
            
            outputs = model(
                screenshots=batch.screenshots,
                tokens=batch.tokens,
                last_actions=batch.last_actions, 
                memory=memory_mode,
                requires_loss=True,
                is_ignored=batch.is_ignored, 
                is_reset=batch.is_reset,
            )
            outputs['loss_details'] = {k:v.item() for k, v in outputs['loss_details'].items()}
            loss = outputs['loss'] / iters_to_accumulate
            n_accum_steps += 1
            n_accum_examples += batch.size
        
        #
        # 3. Model backward calculation
        #
        if is_training:
            scaler.scale(loss).backward()
            
            global_stat['n_total_training_steps'] += 1
        
        if do_update:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            global_stat['n_updates'] += 1
            global_stat['n_total_examples_used_for_updates'] += n_accum_examples
            n_accum_steps = 0
        
        #
        # Status updation after model calculation
        #
        local_n_processed += batch.size
        n_left_steps -= 1
        
        stat['n_sum'] += 1
        stat['n_processed'] += batch.size
        stat['max_step_sum'] += batch.time_step
        stat['loss_sum'] += outputs['loss'].item()
        for k, v in outputs['loss_details'].items():
            stat['loss_details_sum'][k] = stat['loss_details_sum'].get(k, 0) + v
        
        #
        # 5. Saving data
        #
        if show_loss:
            current_time = time.time()
            print_stat(local_n_processed, current_time - last_time)
            clear_stat()
            last_time = current_time
        
        if save_model:
            save_hook(context)
            model.train(is_training)
    
    if not is_training:
        current_time = time.time()
        results = print_stat(local_n_processed, current_time - last_time)
        return dict(_.split('=') for _ in results)

    return None


if __name__ == "__main__":
    
    config = parse_from_dataclass(
        {'train':TrainingConfig, 'model':models.ModelConfig}, 'configuration for training')
    
    if config['train'].mode == 'train':
        train(config)
    
    print('done')

