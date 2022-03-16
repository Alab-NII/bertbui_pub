# coding: utf-8


import dataclasses as DC

import os
import time
import ijson, json
from concurrent.futures import ThreadPoolExecutor

from .utils import parse_from_dataclass


@DC.dataclass
class DataConfig:
    
    _desc = 'Configuration for metada creation for the static files'
    
    static_dir: str = DC.field(metadata={'help':'path to a directory of static files'})
    
    check_dir: str = DC.field(default=None,
        metadata={'help':'If provided, check the validity of metadata in the directory'})


def read_num_actions_from_json(file_path):
    """partially parse a json file and read the number of actions"""
    
    num_actions = 0
    for prefix, event, value in ijson.parse(open(file_path)):
        if not prefix.startswith('actions'):
            continue
        elif prefix == 'actions.item.name' and event == 'string':
            num_actions += 1
        elif prefix == 'actions' and event == 'end_array':
            break
    return num_actions


def is_taget_json_file(filename):
    """"if finename matches target rule return True otherwise return False"""
    
    return (not filename.startswith('.')) and filename.endswith('.json')


def get_meta_data(path_static_dir, root, files):
    """get mata data dict based on files in the root dir"""
    
    def thread_func(filename):
        filepath = os.path.join(root, filename)
        mtime = os.stat(filepath).st_mtime
        num_actions = read_num_actions_from_json(filepath)
        return {'mtime':mtime, 'num_actions':num_actions}
    
    filenames = [_ for _ in files if is_taget_json_file(_)]
    
    with ThreadPoolExecutor() as executor:
        return dict(zip(filenames, executor.map(thread_func, filenames)))


def make_meta_files(path_static_dir):
    """walk path_static_dir and make  all meta files"""
    
    for root, dirs, files in os.walk(path_static_dir):
        if root != path_static_dir:
            # check any json file exists
            any_json_file = False
            for filename in files:
                if is_taget_json_file(filename):
                    any_json_file = True
                    break
            if not any_json_file:
                continue
            
            stime = time.time()
            # make meta data for this dir and save it
            meta_data = get_meta_data(path_static_dir, root, files)
            meta_file_path = os.path.join(root, 'metadata')
            json.dump(meta_data, open(meta_file_path, 'w'))
            etime = time.time()
            print(root, 'done', len(meta_data), 'files', int(etime - stime), 'elapsed')
            

def read_metadata(dir_path, validate=True):
    """check if the meta file of key is valid or not based on st_mtime
    and return metadata
    """

    meta_data = json.load(open(os.path.join(dir_path, 'metadata')))

    if not validate:
        return meta_data

    num_target_files = 0
    for filename in os.listdir(dir_path):
        if is_taget_json_file(filename):
            filepath = os.path.join(dir_path, filename)
            if meta_data[filename]['mtime'] != os.stat(filepath).st_mtime:
                raise RuntimeError('%s: metadata mismatch mtime on %s' % (dir_path, filename))
            num_target_files += 1
    if num_target_files != len(meta_data):
        msg_tmp = '%s: medata mismatch number of files; %d in metadata, %d actual'
        raise RuntimeError(msg_tmp % (dir_path, len(meta_data), num_target_files))

    return meta_data


def main():
    
    config = parse_from_dataclass(DataConfig, 'Configuration for metada creation for the static files')
    
    print('config:', config)
    
    if config.check_dir:
        read_metadata(config.check_dir)
        print(config.check_dir, 'valid metadata')
    else:
        make_meta_files(config.static_dir) 


if __name__ == '__main__':
    
    main()
