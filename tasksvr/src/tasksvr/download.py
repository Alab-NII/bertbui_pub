# coding: utf-8

import dataclasses
from tasksvr import parse_from_dataclass, get_default_data_dir, Wrappers


@dataclasses.dataclass
class DownloadConfig:
    data_dir: str = None


SETUP_LIST = [
    'cola', 'mnli', 'mrpc', 'stsb', 'qnli', 'qqp', 'rte', 'sst2', 'wnli',
    'squad_v2',
    'pta',
    'coco', 'vqa_v2',
    'sa', 
]
#SETUP_LIST = ['wnli']

def main():
    
    download_config = parse_from_dataclass(DownloadConfig)
    data_dir = download_config.data_dir
    if data_dir is None:
        data_dir = get_default_data_dir()
    wrappers = Wrappers.enumerate(data_dir)
    
    for key in SETUP_LIST:
        print(key)
        if key in wrappers:
            wrappers[key].setup()
        else:
            print('missing wrapper, ignored')
