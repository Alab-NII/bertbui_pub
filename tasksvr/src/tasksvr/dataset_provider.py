# coding: utf-8


import logging
logger = logging.getLogger(__name__)


import os
import sys
from collections import OrderedDict
import json
import importlib.util
import numpy as np


class DatasetProvider(object):
    
    @staticmethod
    def get_default_adapters_path():

        return os.path.join(os.path.dirname(__file__), 'adapters')
    
    @staticmethod
    def enumerate_adapters(adapters_path=None):
        
        adapters_path = adapters_path or DatasetProvider.get_default_adapters_path()

        out_py_files = ['__init__.py']
        adapters = {}
        
        for filename in os.listdir(adapters_path):
            if filename.endswith('.py') and filename not in out_py_files:
                module_name = 'adapter_%s' % filename.replace('.py', '')
                filepath = os.path.join(adapters_path, filename)
                
                spec = importlib.util.spec_from_file_location(module_name, filepath)
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                
                assert hasattr(module, 'Builder'), f'Builder not found in {filepath}'
                module_name = module.Builder.name
                assert module_name not in adapters, 'module already exists: %s' % module_name
                adapters[module_name] = module.Builder
        
        return adapters
    
    def __init__(self, data_path='data', adapters_path=None):
        
        adapters_path = adapters_path or self.get_default_adapters_path()
        
        self.datasets = {}
        self.data_path = data_path
        self.adapters = self.enumerate_adapters(adapters_path)
        self.adapters_path = adapters_path
    
    def __call__(self, _id):
        
        if _id in self.datasets:
            return self.datasets[_id]
        
        return self.load_dataset(_id)
    
    @staticmethod
    def _param_to_dict(param):
        
        return dict(kv.split('=') for kv in param.split('&'))
    
    def load_dataset(self, _id):
        
        assert _id not in self.datasets, 'already we have: %s'%_id
        
        def _load(_id):
            
            name, split = _id.split('.', maxsplit=1)
            adapter = self.adapters[name]
            assert split in adapter.splits, 'not found in %s: %s' % (name, split)
            
            dataset = adapter.build(split, self.data_path)
            dataset['name'] = name
            dataset['split'] = split
            
            return dataset
        
        logger.info('loading %s ...' % _id)
        
        if '?' in _id:
            _id, param = _id.split('?', maxsplit=1)
            dataset = _load(_id)
            
            dataset['param'] = param = self._param_to_dict(param)
            if 'r' in param:
                seed = int(param['r'])
                random = np.random.RandomState(seed)
                random.shuffle(dataset['data'])
            
            if 'n' in param:
                max_num = int(param['n'])
                dataset['data'] = dataset['data'][:max_num]
            
        else:
            dataset = _load(_id)
        
        self.datasets[_id] = dataset
        return dataset
        
    def zeros(self, _id, n):
        
        assert _id not in self.datasets, 'already we have: %s'%_id
        
        self.datasets[_id] = {'data': [{} for _ in range(n)]}
        return self.datasets[_id]


class DatasetBuilder(object):
    
    name = None
    download_url = None
    local_dir = None

    splits = OrderedDict([
        ('train', ['train.tsv']),
        ('dev', ['dev.tsv']),
        ('test', ['test.tsv']),
    ])

    local_dir = None

    base_config = {
        'instruction': None,
    }

    @classmethod
    def build(cls, split, data_path):
        
        file_specs = []
        for file_spec in cls.splits[split]:
            if isinstance(file_spec, str):
                file_spec = {'file':file_spec}
            elif isinstance(file_spec, (tuple, list)):
                file_spec = {'files':file_spec}
            else:
                file_spec = file_spec.copy()
            
            if 'file' in file_spec and 'files' in file_spec:
                raise RuntimeError('file spec can have either of file or files key.')
            if 'file' in file_spec:
                file_spec['file'] = os.path.join(data_path, cls.local_dir, file_spec['file'])
            if 'files' in file_spec:
                file_spec['files'] = tuple(os.path.join(data_path, cls.local_dir, _) for _ in file_spec['files'])
            file_specs.append(file_spec)
        return cls._build_from_files(file_specs, split=split)
    
    @classmethod
    def _build_from_files(cls, file_specs, split=None):

        dataset = cls.base_config.copy()
        
        dataset['split'] = split
        dataset['accept_no_labels'] = False
        if hasattr(cls, 'non_labeled_splits'):
            dataset['accept_no_labels'] = split in cls.non_labeled_splits
        
        data = []
        for file_spec in file_specs:
            key = 'file' if 'file' in file_spec else 'files'
            entries = cls._load_file(file_spec[key], dataset)
            
            # This implementation is not efficient 
            # since we should read the same file for different splits, such as train and dev.
            # It might be needed to cache or to do something.
            if 'seed' in file_spec:
                np.random.RandomState(file_spec['seed']).shuffle(entries)
            
            if 'slice' in file_spec:
                p_start, p_end = file_spec['slice']
                len_entries = len(entries)
                i_start = int(round(p_start*len_entries))
                i_end = int(round(p_end*len_entries))
                entries = entries[i_start:i_end]

            data.extend(entries)
        dataset['data'] = data
        
        return dataset
    
    @classmethod
    def _load_file(cls, path, dataset):
        raise NotImplementedError()

    @classmethod
    def download(cls, data_path, overwrite=False):
        
        download_urls = cls.download_url
        if download_urls is None:
            raise RuntimeError('download_url is not specified. Please download manually.')
        
        if isinstance(download_urls, str):
            download_urls = [download_urls]
        
        dest_dir = os.path.join(data_path, cls.local_dir)
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)
        
        for url in download_urls:
            if isinstance(url, str):
                name = url.split('/')[-1]
            else:
                url, name = url
            dest_path = os.path.join(dest_dir, name)
            
            if not overwrite and os.path.exists(dest_path):
                logger.info(f'will use existing file: {dest_path}')
            else:
                logger.info(f'downloading: {url} -> {dest_path}')
                cls._download_file(url, dest_path)
            
            if name.endswith('.zip'):
                cls._extract_zip(dest_path)
                logger.info(f'unzip done: {dest_path}')
    
    @classmethod
    def _download_file(cls, url, dest_path):
        
        import requests
        data = requests.get(url)
        with open(dest_path, 'wb') as f:
            f.write(data.content)
        return dest_path

    @classmethod
    def _extract_zip(cls, zip_file_path):
        
        from zipfile import ZipFile
        import re

        dest_dir = os.path.dirname(zip_file_path)
        
        with ZipFile(zip_file_path, 'r') as myzip:
            for info in myzip.infolist():
                filename = info.filename
                is_dir = info.is_dir()
                parts = re.split('|'.join([re.escape(_) for _ in '/\\']), filename)
                if parts[0] == cls.local_dir:
                    parts.pop(0)
                if parts[-1] == '':
                    parts.pop(-1)
                if parts:
                    dest_path = os.path.join(dest_dir, *parts)
                    if dest_path == zip_file_path:
                        logger.warn(f'zip file with the same name skipped: {dest_path}')
                    elif is_dir:
                        if not os.path.exists(dest_path):
                            os.mkdir(dest_path)
                            logger.info(f'dir made: {dest_path}')
                    else:
                        with myzip.open(info, 'r') as src:
                            with open(dest_path, 'wb') as f:
                                f.write(src.read())
                        logger.info(f'file extracted: {dest_path}')


class ClassificationDataset(dict):

    def displayed_name(self, label=None, sentence=None):

        if label is not None and sentence is None:
            rename_map = self.get('renamed_labels', None)
            return rename_map[label] if rename_map else label
        elif label is None and sentence is not None:
            rename_map = self.get('renamed_sentence_columns', None)
            return rename_map[sentence] if rename_map else sentence
            
        raise RuntimeError('give either label or sentence')
    
    def get_displayed_labels(self):
        
        rename_map = self.get('renamed_labels', None)
        return [rename_map[_] for _ in self['labels']] if rename_map else  self['labels']


class ClassificationDatasetBuilder(DatasetBuilder):
    

    base_config = {
        'has_header': False,
        'columns': ['sentence', 'label'],
        'label_column': 'label',
        'labels': None,
        'instruction': None,
    }

    @classmethod
    def _build_from_files(cls, *args, **kwargs):
        
        dataset = super()._build_from_files(*args, **kwargs)
        
        # Automatic label completion
        if dataset.get('labels') is None:
            dataset['labels'] = sorted(set(_['label'] for _ in data))
        
        return ClassificationDataset(dataset)
    
    @classmethod
    def _load_file(cls, filepath, dataset):
        
        ext = os.path.splitext(filepath)[1]
        
        if ext in ('.tsv', '.txt'):
            return cls._load_file_tsv(filepath, dataset)
        
        raise RuntimeError(f'Unknown extension {ext}: {filepath}')
    
    @classmethod
    def _load_file_tsv(cls, filepath, dataset):

        has_header = dataset['has_header']
        columns = dataset.get('columns')
        if columns:
            col_id = {k:i for i, k in enumerate(columns)}
        else:
            col_id = None
        label_column = dataset.get('label_column', 'label')
        sentence_columns = dataset['sentence_columns']
        ignored_labels = dataset.get('ignored_labels', [])
        accept_no_labels = dataset['accept_no_labels']
        
        assert has_header or col_id is not None, 'no header information found'
        
        entries = []
        with open(filepath, 'r') as f:
            for line_id, line in enumerate(f.readlines()):
                cols = line.strip().split('\t')
                
                if line_id == 0 and has_header:
                    if col_id is None:
                        col_id = {k.strip():i for i, k in enumerate(cols)}
                    continue
                
                label = None
                try:
                    label = cols[col_id[label_column]]
                except:
                    if not accept_no_labels:
                        raise

                if label not in ignored_labels:
                    entry = {}
                    entry['label'] = label
                    for col_name in sentence_columns:
                        entry[col_name] = cols[col_id[col_name]]
                    entries.append(entry)
        
        return entries

