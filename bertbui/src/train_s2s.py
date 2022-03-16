# coding: utf-8


import dataclasses as DC

import time
import os
import shutil
import json

import numpy as np
import torch

from browserlm import (
        parse_from_dataclass_dict, 
        Seq2SeqDataset, 
    )
import models_for_seq2seq as models
os.environ['TOKENIZERS_PARALLELISM'] = '0'


@DC.dataclass
class TrainingConfig:
    
    _desc = 'Configuration for seq2seq model training'
    
    model_path: str = DC.field(metadata={'help':'path to a directory where the weights will be saved.'})
    
    mode: str = DC.field(default='predict',
            metadata={'help':'train|predict'})
    
    train_datasets: str = DC.field(default='',
            metadata={'help':'dataset name for training; accept comma spliting'})

    valid_datasets: str = DC.field(default='',
            metadata={'help':'dataset name for validation; accept comma spliting'})

    predict_dataset: str = DC.field(default='',
            metadata={'help':'dataset name for prediction'})
    
    data_dir: str = DC.field(default='data', 
            metadata={'help':'path to data dir.'})
    
    disable_images: int = DC.field(default=0, 
            metadata={'help':'If true, image input for VQA is omitted'})
    
    # 128 batch / update
    minibatch_size: int = DC.field(default=128, 
            metadata={'help':'examples in a step.'})
        
    iters_to_accumulate: int = DC.field(default=1,
            metadata={'help':'the number of *steps* whose gradients will be acumurated.'})    
    
    interval_to_show_loss: int = DC.field(default=-1, 
            metadata={'help':'show average loss every those *steps*.'})
    
    interval_in_epoch_to_save_model: float = DC.field(default=1, 
            metadata={'help':'save model weights every those *epochs*.'})
    
    max_epoch: int = DC.field(default=10, 
            metadata={'help':'The maximum of epoch.'})
    
    random_seed: int = DC.field(default=123, 
            metadata={'help':'a random seed for minibatch sampling.'})
    
    amp_enabled: int = DC.field(default=1, 
            metadata={'help':'When 1, we use autocast of torch'})
    
    optimizer_lr: float = DC.field(default=5e-5, 
            metadata={'help':'Learning rate for the adam optimizer'})
    
    weight_seed: str =  DC.field(default='', 
            metadata={'help':'If set, the model weight is initialized with this weight.'})
        
    num_workers: int = DC.field(default=4, 
            metadata={'help':'number of the workers for batch creation'})
    
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
    def predictions_dir(self):
        
        return os.path.join(self.model_path, 'predictions')
        
    @property
    def model_config_path(self):
        
        return os.path.join(self.model_path, 'model_config.json')
    
    @property
    def amp_enabled_bool(self):
        
        return self.amp_enabled != 0

    def join_path(self, *args):
        
        return os.path.join(self.model_path, *args)


def seed_worker(worker_id):
    
    worker_seed = (torch.initial_seed()+worker_id) % 2**32
    np.random.seed(worker_seed)
    #random.seed(worker_seed)


def predict(config):

    print('prediction starts with the configuration:')
    print(config['train'])
    
    if not os.path.exists(config['train'].predictions_dir):
        os.mkdir(config['train'].predictions_dir)
    
    dataset_class = Seq2SeqDataset
    model_class = models.SeqSeqModelWithVision
    dataset_params = {
        'data_dir': config['train'].data_dir,
        'disable_images': config['train'].disable_images,
    }
    
    model_config = models.ModelConfig(**json.load(open(config['train'].model_config_path)))
    print('model config was loaded:', model_config)
    
    data_loaders = {}
    data_loaders['predict'] = torch.utils.data.DataLoader(
        dataset_class(model_config, [config['train'].predict_dataset], **dataset_params),
        collate_fn=dataset_class.collate_fn,
        batch_size=config['train'].minibatch_size, shuffle=False,
        num_workers=config['train'].num_workers, drop_last=False,
        prefetch_factor=2,
    )
    print('predict examples:', len(data_loaders['predict'].dataset))
    label_mapping = data_loaders['predict'].dataset.metadata.get('label_mapping', {})
    if label_mapping:
        print('label_mapping detected. submission will be rewrited', label_mapping)
    
    # model creation
    tokenizer = model_config.load_tokenizer()
    model = model_class(model_config).adapt_tokenizer(tokenizer)
    
    if config['train'].weight_seed:
        weight_path = config['train'].weight_seed
        if not os.path.exists(weight_path):
            weight_path = os.path.join(config['train'].weight_dir, config['train'].weight_seed)
    else:
        weight_path = os.path.join(config['train'].model_path, 'pytorch_model.bin')
    model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    print('loaded model config', model_config)
    print(f'weight loaded from {weight_path}')
    
    model = model.cuda()
    
    predict_dataset_name = os.path.basename(config['train'].predict_dataset)
    save_path = os.path.join(config['train'].predictions_dir, predict_dataset_name)
    print(save_path)
    
    # Start prediction
    predictions = []
    model.eval()
    for batch in data_loaders['predict']:
        batch = batch.to_device('cuda:0')
        outputs = model.generate(
            input_ids=batch.token_ids, 
            input_images=batch.images,
            input_images_enabled=batch.images_enabled,
        )
        sequences = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(sequences)
        # genenrated label
    
    # change format
    # id, original_label, submission, status, timestep
    if label_mapping:
        def map_func(i, seq):
            mapped = label_mapping.get(seq, None)
            status = 'out_of_label' if mapped is None else 'submitted'
            return (i, mapped, seq, status, 1)
    else:
        def map_func(i, seq):
            return  (i, seq, seq, 'submitted', 1)
    predictions = [map_func(i, _) for i, _ in enumerate(predictions)] 
    
    with open(save_path, 'w') as f:
        json.dump({'name':predict_dataset_name, 'predictions':predictions}, f, indent=1)


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
    with open(config['train'].train_log_path, 'a') as f:
        print(config, file=f)
    
    dataset_class = Seq2SeqDataset
    model_class = models.SeqSeqModelWithVision
    dataset_params = {
        'data_dir': config['train'].data_dir,
        'disable_images': config['train'].disable_images,
    }

    data_loaders = {}
    data_loaders['train'] = torch.utils.data.DataLoader(
        dataset_class(config['model'], config['train'].train_datasets.split(','), **dataset_params), 
        collate_fn=dataset_class.collate_fn, 
        batch_size=config['train'].minibatch_size, shuffle=True, 
        num_workers=config['train'].num_workers, drop_last=False,
        worker_init_fn=seed_worker,
        prefetch_factor=2
    )
    print('train examples:', len(data_loaders['train'].dataset))
    
    data_loaders['valid'] = torch.utils.data.DataLoader(
        dataset_class(config['model'], config['train'].valid_datasets.split(','), **dataset_params), 
        collate_fn=dataset_class.collate_fn, 
        batch_size=config['train'].minibatch_size, shuffle=False, 
        num_workers=config['train'].num_workers, drop_last=False,
        prefetch_factor=2,
    )
    print('valid examples:', len(data_loaders['valid'].dataset))
    
    # model creation
    tokenizer = config['model'].load_tokenizer()
    model = model_class(config['model']).adapt_tokenizer(tokenizer)
    
    if config['train'].weight_seed:
        model.load_state_dict(torch.load(config['train'].weight_seed, map_location='cpu'))
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train'].optimizer_lr)
    scaler = torch.cuda.amp.GradScaler(enabled=config['train'].amp_enabled_bool)
    
    # Collecting context
    context = {
        'config': config,
        'model': model,
        'optimizer': optimizer,
        'scaler': scaler,
        'interval_to_save': int(len(data_loaders['train'])*config['train'].interval_in_epoch_to_save_model),
        'save_hook': lambda c: save_hook(c, data_loaders['valid']),
        'global_stat': {
            'epoch': 0,
            'processed': 0,
            'step': 0,
            'update': 0,
        },
        'valid_history': [],
    }
    
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
        model_path = config['train'].model_path
        text = f'best validation loss: {best_valid_loss} @ {save_name} for {model_path}'
        print(text)
        with open(config['train'].train_log_path, 'a') as f:
            print(text, file=f)
        shutil.copyfile(
            os.path.join(config['train'].weight_dir, save_name), 
            config['train'].join_path('pytorch_model.bin')
        )


def save_hook(context, data_loader_valid):
    """save and validate model"""
        
    model = context['model']
    save_name = 's%d'%(context['global_stat']['step'])
    save_path = os.path.join(context['config']['train'].weight_dir, save_name)
    torch.save(model.state_dict(), save_path)
        
    with torch.no_grad():
        model.eval()
        results = feed_ids(context, data_loader_valid, is_training=False)
    
    results['save_name'] = save_name
    
    context['valid_history'].append(results)
    with open(context['config']['train'].valid_history_path, 'w') as f:
        json.dump(context['valid_history'], f)


def feed_ids(context, data_loader, is_training):
    
    # expanding the context
    config = context['config']
    model = context['model']
    optimizer = context['optimizer']
    scaler = context['scaler']
    save_hook = context['save_hook']
    
    amp_enabled = config['train'].amp_enabled_bool
    max_iters_to_accumulate = config['train'].iters_to_accumulate
    len_data_loader = len(data_loader)
    interval_to_show_loss = config['train'].interval_to_show_loss
    if interval_to_show_loss < 0:
        # auto prediction for the interval
        interval_to_show_loss = max(1, int(len_data_loader/10))
    
    # Statistic
    stat = {}
    def clear_stat():
        stat['n_processed'] = 0
        stat['loss_sum'] = 0
        stat['n_sum'] = 0
        
    def print_stat(local_n_processed, elapsed_time):
        n_sum = stat['n_sum']
        loss_avg = stat['loss_sum'] / n_sum
        data = [
            'gp=%d'%context['global_stat']['processed'],
            'gs=%d'%context['global_stat']['step'],
            'gu=%d'%context['global_stat']['update'],
            'ge=%d'%context['global_stat']['epoch'],
            'lm=%s'%('T' if is_training else 'V'),
            'lp=%d'%local_n_processed,
            'bp=%d'%stat['n_processed'], 
            'bt=%.3f'%elapsed_time, 
            'bl=%.6f'%loss_avg,
        ]
        text = ' '.join(data)
        print(text)
        with open(config['train'].train_log_path, 'a') as f:
            f.write(text)
            f.write('\n')
        return data
    
    clear_stat()
    last_time = time.time()
    local_n_processed = 0
    
    # initialize accumulation
    optimizer.zero_grad()
    n_accum = 0
    iters_to_accumulate = min(max_iters_to_accumulate, len_data_loader)
    model.train(is_training)

    for batch_id, batch in enumerate(data_loader):
        
        local_n_processed += batch.size
        
        # calculation
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            
            batch = batch.to_device('cuda:0')
            
            outputs = model(
                input_ids=batch.token_ids,
                input_images=batch.images,
                input_images_enabled=batch.images_enabled,
                labels=batch.labels,
            )
            loss = outputs.loss / iters_to_accumulate
            n_accum += 1
        
        # Update
        if is_training:
            scaler.scale(loss).backward()
            
            if n_accum >= iters_to_accumulate:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                n_accum = 0
                iters_to_accumulate = min(
                    max_iters_to_accumulate, len_data_loader - batch_id - 1)
                
                context['global_stat']['update'] += 1
                
            context['global_stat']['step'] += 1
            context['global_stat']['processed'] += batch.size
        
        # Staticstic
        stat['n_processed'] += batch.size
        stat['loss_sum'] += outputs['loss'].item()
        stat['n_sum'] += 1
        
        if is_training and (stat['n_sum'] >= interval_to_show_loss):
            current_time = time.time()
            print_stat(local_n_processed, current_time - last_time)
            clear_stat()
            last_time = current_time
        
        # save
        if is_training and (context['global_stat']['step'] % context['interval_to_save']) == 0:
            
            save_hook(context)
            model.train(is_training)
    
    if not is_training:
        current_time = time.time()
        results = print_stat(local_n_processed, current_time - last_time)
        return dict(_.split('=') for _ in results)

    return None


if __name__ == "__main__":
    
    config = parse_from_dataclass_dict(
        {'train':TrainingConfig, 'model':models.ModelConfig}, 'configuration for training')
    
    mode = config['train'].mode
    if mode == 'noop':
        print('noop')
    elif mode == 'train':
        train(config)
    elif mode == 'predict':
        predict(config)
    else:
        raise RuntimeError(f'unknown mode: {config.mode}')

    print('done')
