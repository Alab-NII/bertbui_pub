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
        OneStepClassification, 
        OneStepExtraction,
        OneStepVQA,
    )
import models_for_normal as models

DATASET_MODEL_PARAMS = {
        'classification': (OneStepClassification, models.BertClassifier, []),
        'extraction': (OneStepExtraction, models.BertExtractor, []),
        'vqa': (OneStepVQA, models.BertClassifier, ['answer_vocab', 'data_dir', 'ommit_images']),
    }


@DC.dataclass
class TrainingConfig:
    
    _desc = 'Configuration for model training'
    
    model_path: str = DC.field(metadata={'help':'path to a directory where the weights will be saved.'})
    
    mode: str = DC.field(default='predict',
            metadata={'help':'train|predict'})
    
    task_type: str = DC.field(default='classification',
            metadata={'help':'classification or extraction'})
    
    train_dataset: str = DC.field(default='mnli.train',
            metadata={'help':'dataset name for training'})

    valid_dataset: str = DC.field(default='mnli.dev_matched',
            metadata={'help':'dataset name for validation'})

    predict_dataset: str = DC.field(default='',
            metadata={'help':'dataset name for prediction'})
    
    # 128 batch / update
    minibatch_size: int = DC.field(default=128, 
            metadata={'help':'examples in a step.'})
        
    iters_to_accumulate: int = DC.field(default=1,
            metadata={'help':'the number of *steps* whose gradients will be acumurated.'})    
    
    interval_to_show_loss: int = DC.field(default=100, 
            metadata={'help':'show average loss every those *steps*.'})
    
    interval_in_epoch_to_save_model: float = DC.field(default=1, 
            metadata={'help':'save model weights every those *epochs*.'})
    
    max_epoch: int = DC.field(default=10, 
            metadata={'help':'The maximum of epoch.'})
    
    random_seed: int = DC.field(default=123, 
            metadata={'help':'a random seed for minibatch sampling.'})
    
    amp_enabled: int = DC.field(default=1, 
            metadata={'help':'When 1, we use autocast of torch'})
    
    optimizer_lr: float = DC.field(default=1e-4, 
            metadata={'help':'Learning rate for the adam optimizer'})
    
    weight_seed: str =  DC.field(default='', 
            metadata={'help':'If set, the model weight is initialized with this weight.'})
    
    # For OneStepVQA
    answer_vocab: str =  DC.field(default='', 
            metadata={'help':'vocabulary of answer for VQA'})
    
    data_dir: str = DC.field(default='data', 
            metadata={'help':'path to data dir.'})
    
    ommit_images: int = DC.field(default=0, 
            metadata={'help':'If true, image input for VQA is omitted'})
    
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
    print(config)

    if not os.path.exists(config['train'].predictions_dir):
        os.mkdir(config['train'].predictions_dir)
    
    dataset_class, model_class, param_keys = DATASET_MODEL_PARAMS[config['train'].task_type]
    dataset_params = {k: getattr(config['train'], k) for k in param_keys}
    
    model_config = models.ModelConfig(**json.load(open(config['train'].model_config_path)))
    print('model config was loaded:', model_config)
    
    data_loaders = {}
    data_loaders['predict'] = torch.utils.data.DataLoader(
        dataset_class(model_config, config['train'].predict_dataset, **dataset_params),
        collate_fn=dataset_class.collate_fn,
        batch_size=config['train'].minibatch_size, shuffle=False,
        num_workers=4, drop_last=False,
        prefetch_factor=2,
    )
    print('predict examples:', len(data_loaders['predict'].dataset))

    # model creation
    if config['train'].task_type in ['classification', 'vqa']:
        n_classes = len(data_loaders['predict'].dataset.labels)
        model = model_class(model_config, n_classes)
    else:
        model = model_class(model_config)
    
    if config['train'].weight_seed:
        weight_path = config['train'].weight_seed
        if not os.path.exists(weight_path):
            weight_path = os.path.join(config['train'].weight_dir, config['train'].weight_seed)
    else:
        weight_path = os.path.join(config['train'].model_path, 'pytorch_model.bin')
    model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    print(f'weight loaded from {weight_path}')
    
    model = model.cuda()
    
    predict_dataset_name = os.path.basename(config['train'].predict_dataset)
    save_path = os.path.join(config['train'].predictions_dir, predict_dataset_name)
    print(save_path)
    
    if config['train'].task_type in ['classification', 'vqa']:

        # Start prediction
        predictions = []
        model.eval()
        for batch in data_loaders['predict']:
            batch = batch.to_device('cuda:0')
            outputs = model(token_ids=batch.token_ids, images=batch.images)
            label_ids = outputs['class_logits'].data.argmax(1)
            predictions.extend(label_ids.cpu().numpy())
    
        # add index & convert to labels
        # format: id, prediction, (prediction)
        labels = data_loaders['predict'].dataset.labels
        predictions = [(i, labels[label_id]) for i, label_id in enumerate(predictions)]
        
        # add renamed labels if exists
        if hasattr(data_loaders['predict'].dataset.data_dict, 'displayed_name'):
            displayed_name = data_loaders['predict'].dataset.data_dict.displayed_name
            predictions = [_+(displayed_name(label=_[-1]),) for _ in predictions]
        
        with open(save_path, 'w') as f:
            json.dump({'name':predict_dataset_name, 'predictions':predictions}, f, indent=1)
    
    elif config['train'].task_type == 'extraction':
        
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(config['model'].tokenizer_path)
        
        # Start prediction
        predictions = []
        model.eval()
        for batch in data_loaders['predict']:
            batch = batch.to_device('cuda:0')
            outputs = model(token_ids=batch.token_ids)
            start_ids = outputs['start_logits'].data.argmax(1)
            end_ids = outputs['end_logits'].data.argmax(1)
            for token_ids, sid, eid in zip(batch.token_ids, start_ids, end_ids):
                text = ''
                if sid.item() > 0 and eid >= sid:
                     text = tokenizer.decode(token_ids[sid:eid+1].cpu().numpy())
                predictions.append(text)
        
        # format: id, prediction, prediction, status='submitted', step='1'
        predictions = [
            (entry['_id'], text, text, 'submitted', 1)
            for entry, text in zip(data_loaders['predict'].dataset.data, predictions)]
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
    
    dataset_class, model_class, param_keys  = DATASET_MODEL_PARAMS[config['train'].task_type]
    dataset_params = {k: getattr(config['train'], k) for k in param_keys}

    data_loaders = {}
    data_loaders['train'] = torch.utils.data.DataLoader(
        dataset_class(config['model'], config['train'].train_dataset, **dataset_params), 
        collate_fn=dataset_class.collate_fn, 
        batch_size=config['train'].minibatch_size, shuffle=True, 
        num_workers=8, drop_last=False,
        worker_init_fn=seed_worker,
        prefetch_factor=2
    )
    print('train examples:', len(data_loaders['train'].dataset))
    
    data_loaders['valid'] = torch.utils.data.DataLoader(
        dataset_class(config['model'], config['train'].valid_dataset, **dataset_params), 
        collate_fn=dataset_class.collate_fn, 
        batch_size=config['train'].minibatch_size, shuffle=False, 
        num_workers=8, drop_last=False,
        prefetch_factor=2,
    )
    print('valid examples:', len(data_loaders['valid'].dataset))
    
    # model creation
    if config['train'].task_type in ['classification', 'vqa']:
        n_classes = len(data_loaders['valid'].dataset.labels)
        model = model_class(config['model'], n_classes)
    else:
        model = model_class(config['model'])
    
    if config['train'].weight_seed:
        model.load_state_dict(torch.load(config['train'].weight_seed, map_location='cpu'))
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train'].optimizer_lr)
    scaler = torch.cuda.amp.GradScaler(enabled=config['train'].amp_enabled_bool)
    
    # Collecting context
    context = {
        'config': config,
        'task_type': config['train'].task_type,
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
        text = f'best validation loss: {best_valid_loss} @ {save_name}'
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
    task_type = context['task_type']
    model = context['model']
    optimizer = context['optimizer']
    scaler = context['scaler']
    save_hook = context['save_hook']
    
    amp_enabled = config['train'].amp_enabled_bool
    max_iters_to_accumulate = config['train'].iters_to_accumulate
    len_data_loader = len(data_loader)
    interval_to_show_loss = config['train'].interval_to_show_loss
    if interval_to_show_loss < 0:
        # Auto prediction
        interval_to_show_loss = max(1, int(len_data_loader/10))
    
    # Statistic
    stat = {}
    def clear_stat():
        stat['n_processed'] = 0
        stat['loss_sum'] = 0
        stat['n_sum'] = 0
        if not is_training:
            stat['n_correct'] = 0
            stat['n_label'] = 0
    
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
        if not is_training:
            acc = stat['n_correct'] / stat['n_label']
            data.append('acc=%.3f'%(acc))
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
                token_ids=batch.token_ids,
                images=batch.images,
                requires_loss=True,
                labels=batch.labels,
            )
            loss = outputs['loss'] / iters_to_accumulate
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
        
        if not is_training:
            if task_type in ['classification', 'vqa']:    
                ids = outputs['class_logits'].data.argmax(1)
                stat['n_correct'] += (ids == batch.labels).sum()
                stat['n_label'] += (batch.labels != -1).sum()
            elif task_type == 'extraction':
                for i, key in enumerate(['start_logits', 'end_logits']):
                    ids = outputs[key].data.argmax(1)
                    labels = batch.labels[i]
                    stat['n_correct'] += (ids == labels).sum()
                    stat['n_label'] += ((0 <= labels) & (labels < outputs[key].data.shape[1])).sum()
        
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

