# coding: utf-8
# test_env
# Take screenshot of tasks using the environment for testing
# 2021-11-23
# Taichi Iki


from bertbui import selenium_env, ModelAction, parse_from_dataclass
from tokenizers import Tokenizer
import re
import os
import sys


import dataclasses
@dataclasses.dataclass
class Config:
    tokenizer: str = dataclasses.field(default='models/uncased_L-4_H-512_A-8_tokenizer.json',
            metadata={'help':'a json file thas defines tokenizer'})
    port: int = dataclasses.field(default=9973, 
            metadata={'help':'port to tasksvr'})
    firefox: str = dataclasses.field(default='venv/lib/firefox/firefox', 
            metadata={'help':'path to the firefox binary file'})

        
def main():
    config = parse_from_dataclass(Config)
    
    # params
    tokenizer_path = config.tokenizer
    task_server_url = f'http://127.0.0.1:{config.port}'
    firefox_binary = config.firefox
    window_size = (640,448)
    examples = [
        '/valid/sa?id=0',
        '/valid/vqa_v2?id=0',
        '/valid/cola?id=0', 
        '/valid/squad_v2?id=0',
    ]
    output_dir = 'check_results'
    print('Test environment for')
    print('-', tokenizer_path)
    print('-', task_server_url)
    print('-', firefox_binary)
    print('-', window_size)
    print('-', examples)
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # initialization
    tokenizer = Tokenizer.from_file(tokenizer_path)
    env = selenium_env.EnvSTFirefox(task_server_url, tokenizer, firefox_binary, window_size=window_size)

    # make gold actions and save as gif files
    for example in examples:
        split, task, num = re.findall('/([^/]+)/([^?]+)\?id=([0-9]+)', example)[0]
        save_path = os.path.join(output_dir, f'{split}_{task}_{num}.gif')
    
        _, observations = env.replay_teacher_actions(example)
        images = [o.screenshot_with_ocr for o in observations]
        
        images[0].save(save_path,
               save_all=True, append_images=images[1:], optimize=False, duration=100, loop=0)
        print('done', example, '->', save_path)

if __name__ == '__main__':
    
    main()