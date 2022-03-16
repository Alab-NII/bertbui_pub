# coding: utf-8


import os
import datasets
from .dataset_wrapper import DatasetWrapper
from tasksvr import TTextClassification


class WrapperForDatasets(DatasetWrapper):

    key = ()
    handler_class = TTextClassification
        
    def get_local_dir(self):
        local_name = '_'.join(str(_) for _ in self.key)
        return os.path.join(self.data_dir, local_name)
    
    def setup(self):
        assert self.key, 'key is empty'
        
        cache_dir = os.path.join(self.data_dir, 'cache')
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        ds = datasets.load_dataset(*self.key, cache_dir=cache_dir, download_mode='force_redownload')
        ds.save_to_disk(self.get_local_dir())
        ds.cleanup_cache_files()
    
    def load(self):
        assert self.key, 'key is empty'
        
        return datasets.load_from_disk(self.get_local_dir())
    
    def make_data(self):
        return self.load()
    
    def make_metadata(self):
        return {
            'name': self.name,
            'instructions': self.instructions,
            'feature_name_rules': self.feature_name_rules,
            'sentence_features': self.sentence_features,
            'label_rules': self.label_rules,
            'labels': self.labels,
        }
    
    def make_task_handlers(self):
        
        data = self.make_data()
        metadata = self.make_metadata()
        handlers = [
            (r'/%s/%s' % (dest, self.name), self.handler_class, {'data': data[src], 'metadata': metadata})
            for dest, src in self.exposed_splits.items()
        ]
        return handlers, set(self.requires_handlers)
