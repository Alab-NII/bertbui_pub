# coding: utf-8

# Make static data
# 2021-10-26
#
# This sciprt makes static dataset based on the data from server.py.
# Make sure that you can access task_server.
# It may take a long time. 

import dataclasses as DC

import json
import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor

from bertbui import EnvSTFirefox, ModelAction, parse_from_dataclass
from tokenizers import Tokenizer

DEFAULT_FIREFOX_BINARY = os.path.join(os.environ.get('VIRTUAL_ENV', ''), 'firefox')


@DC.dataclass
class DataConfig:
    
    _desc = 'Configuration for the static data creation'
    
    targets: str = DC.field(metadata={'help':'target urls (comma splitted)'})
    
    output_root_path: str = DC.field(default='static',
        metadata={'help':'path to a directory for outputs'})
    
    max_examples: float = DC.field(default=float('inf'),
        metadata={'help':'the number of the max examples for a split'})
    
    n_workers: int = DC.field(default=8,
        metadata={'help':'the number of the workers'})
    
    tokenizer_path: str = DC.field(default='models/uncased_L-4_H-512_A-8_tokenizer.json',
        metadata={'help':'path to a tokenizer'})
    
    window_size: str = DC.field(default='640,448',
        metadata={'help':'browser window size (comma splitted)'})

    binary_location: str = DC.field(default=DEFAULT_FIREFOX_BINARY,
        metadata={'help':'path to a browser executable binary.'})
        
    url_base: str = DC.field(default='http://localhost:9973',
        metadata={'help':'URL of the task page server'})
    
    overwrite: int = DC.field(default=0,
        metadata={'help':'if not zero, existing files are replaced.'})

        
def run(data_config, target_path):
    
    output_dir = os.path.join(data_config.output_root_path, target_path.strip(os.path.sep))
    os.makedirs(output_dir, exist_ok=True)
    
    def thread_func(ids):
        try:
            env = EnvSTFirefox(
                binary_location=data_config.binary_location,
                url_base=data_config.url_base,
                tokenizer=Tokenizer.from_file(data_config.tokenizer_path),
                window_size=tuple(int(_) for _ in data_config.window_size.split(',')),
            )
        except Exception as e:
            import traceback
            print(f'exception when loading environment: {e}')
            print(traceback.print_exc())
            raise e
        
        for _id in ids:
            try:
                data = env.serialize_teacher_actions('%s?id=%d'%(target_path, _id))
            except Exception as e:
                print(e, _id)
                raise e
            file_path = os.path.join(output_dir, '%d.json'%_id)
            with open(file_path, 'w') as f:
                json.dump(data, f)
    
    n_examples = int(requests.get(data_config.url_base + target_path).text)
    n_examples = min(n_examples, data_config.max_examples)
    
    flat_id_list = list(range(n_examples))
    if data_config.overwrite == 0:
        # remove existing ids
        flat_id_list = [_id for _id in flat_id_list if 
            not os.path.exists(os.path.join(output_dir, '%d.json'%_id))]
        n_examples = len(flat_id_list)
    
    n_workers = data_config.n_workers
    ids_list = [[] for _ in range(n_workers)]
    for i, _id in enumerate(flat_id_list):
        ids_list[i % n_workers].append(_id)

    print('starts',  target_path, n_examples, '->', output_dir)
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        executor.map(thread_func, ids_list)
    print('done', time.time() - t_start)
    print()


def main():
    
    config = parse_from_dataclass(DataConfig, 'configuration for the static data creation')
    
    print('config:', config)
    
    targets = [url.strip() for url in config.targets.split(',')]
    for target in targets:
        run(config, target)    


if __name__ == '__main__':
    
    main()
