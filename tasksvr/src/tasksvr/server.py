# coding: utf-8


import logging
logger = logging.getLogger(__name__)

import dataclasses as DC
from tasksvr import parse_from_dataclass, get_default_data_dir, Wrappers

import os
import json

import tornado.ioloop
import tornado.web


@DC.dataclass
class ServerConfig:
    
    use: str = DC.field(
            metadata={'help':'specify wrappers\' name to be used; comma splitted'})
    
    ip_address: str = DC.field(default='0.0.0.0', 
            metadata={'help':'ip address to listen'})
    ip_port: int = DC.field(default=9973, 
            metadata={'help':'port to listen'})
    
    random_seed: int = DC.field(default=123, 
            metadata={'help':'an `initial` random seed for training splits'})
    
    data_dir: str = DC.field(default=None, 
            metadata={'help':'a path to data directorry'})
    
    split_names: str = DC.field(default='train,valid,test', 
            metadata={'help':'names recognized as split names'})


def build_handlers(config):
    
    data_dir = config.data_dir
    if data_dir is None:
        data_dir = get_default_data_dir()
    
    wrappers = Wrappers().enumerate(data_dir)
    task_handlers = []
    use_wrappers = config.use.split(',')
    imported_wrappers = set()
    max_trial = 5
    
    def import_wrapper(wrapper_name, d):
        if d < 1:
            raise RuntimeError('reached the max recurence.')
        hs, rws = wrappers[wrapper_name].make_task_handlers()
        for rw_name in rws:
            if rw_name not in imported_wrappers:
                import_wrapper(rw_name, d - 1)
        task_handlers.extend(hs)
        imported_wrappers.add(wrapper_name)
    
    for wrapper_name in use_wrappers:
        wrapper_name = wrapper_name.strip()
        import_wrapper(wrapper_name, max_trial)
    
    handlers = []
    handlers.append((r'/shutdown', ShutdownHandler))
    handlers.extend(make_split_info_handlers(config, task_handlers))
    handlers.extend(task_handlers)
    
    return handlers


def make_split_info_handlers(config, task_handlers):
    
    handlers = []
    
    for split_name in config.split_names.split(','):
        if not split_name.startswith('/'):
            split_name = '/'+split_name
        tasks = [url for url, _, _ in task_handlers if url.startswith(split_name)]
        handlers.append((split_name, SplitInfoHandler, {'tasks': tasks}))
    
    return handlers


class ShutdownHandler(tornado.web.RequestHandler):
    
    WAIT_SECONDS_BEFORE_SHUTDOWN = 3
    
    def get(self):       
        
        self.write('accepted')
        
        io_loop = tornado.ioloop.IOLoop.current()
        
        def stop_func():
            
            server.stop()
            io_loop.stop()
        
        io_loop.call_later(self.WAIT_SECONDS_BEFORE_SHUTDOWN, stop_func)


class SplitInfoHandler(tornado.web.RequestHandler):
    
    def initialize(self, tasks):
            
        self.tasks = tasks
    
    def get(self):
        
        self.write(' '.join(self.tasks))


def main():

    config = parse_from_dataclass(
        ServerConfig, 
        'Configuration for a task server'
    )
    print('config:', config)
    
    handlers = build_handlers(config)
    app = tornado.web.Application(handlers)
    global server
    server = app.listen(config.ip_port, config.ip_address)
    
    print('io loop starts')
    tornado.ioloop.IOLoop.current().start()

    
if __name__ == "__main__":
    main()
