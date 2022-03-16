# coding: utf-8


from tasksvr import DatasetWrapper
import os
import tornado.web


URLS = [
    'http://images.cocodataset.org/zips/train2014.zip',
    'http://images.cocodataset.org/zips/val2014.zip',
]

SPLIT_SETTINGS = {
    'train': {'dir': 'train2014'},
    'validation': {'dir': 'train2014'},
}


class Wrapper(DatasetWrapper):
    """"""
    
    name = 'coco'
    
    exposed_splits = {'train': 'train', 'valid': 'validation'}
    
    def load(self):
        # enumerate file paths
        data = {}
        for split_name, setting in SPLIT_SETTINGS.items():
            split_dir = os.path.join(self.data_dir, self.name, setting['dir'])
            data[split_name] = [os.path.join(split_dir, _) for _ in sorted(os.listdir(split_dir))]
        return data
    
    def make_task_handlers(self):
        url = '/'+self.name+'/(.+)'
        handlers = [
            (url, tornado.web.StaticFileHandler, {'path': os.path.join(self.data_dir, self.name)})
        ]
        return handlers, set()
    
    def setup(self):
        local_dir = os.path.join(self.data_dir, self.name)
        if not os.path.exists(local_dir):
            os.mkdir(local_dir)
        for url in URLS:
            path = os.path.join(local_dir, os.path.basename(url))
            print('downloading', url)
            self._download_file(url, path)
            print('extracting', path)
            self._extract_zip(local_dir, path)
