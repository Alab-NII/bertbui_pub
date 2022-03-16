# coding: utf-8
# reading_env_v2
# module that provides the reading-with-actions enviroment


from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

from selenium.webdriver import chrome
from selenium.webdriver import firefox

import requests

import PIL.Image, PIL.ImageDraw
import base64
import io
import time
import math
import os
import json
import numpy as np

from .action_and_state import Word, ModelAction, ModelObservation


def get_all_ids(url_base, split):
    """Get all the ids in a given split."""
    
    def fetch(s):
        return requests.get(url_base+s).text
    
    if split.count('/') > 1:
        task_urls = [split]    
    else:
        task_urls =  fetch(split).split()
    
    ids = []
    for task_url in task_urls:
        ids.extend('%s?id=%d'%(task_url, _id) for _id in range(0, int(fetch(task_url))))
        
    return ids


class ActionChainsEx(ActionChains):
    
    def __init__(self, device):
        
        super().__init__(device)
        
        self.w3c_actions._pointer_action.source.DEFAULT_MOVE_DURATION = 100
    
    def move_to_location(self, x, y):
        """
        Moving the mouse to an offset from current mouse position.
        :Args:
         - xoffset: X offset to move to, as a positive or negative integer.
         - yoffset: Y offset to move to, as a positive or negative integer.
        """

        self.w3c_actions.pointer_action.move_to_location(x, y)
        self.w3c_actions.key_action.pause()

        return self


class EnvWithSelenium(object):
    """Browser independent abstract class"""
    
    KEY_MAPPING = {
        'up':       Keys.ARROW_UP,
        'down':  Keys.ARROW_DOWN,
        'left':    Keys.ARROW_LEFT,
        'right':   Keys.ARROW_RIGHT,
        'backspace': Keys.BACKSPACE,
        'space': Keys.SPACE,
        'enter': Keys.ENTER,
    }
    
    def _set_driver(self):
        
        raise NotImplementedError()
    
    def _unset_driver(self):
        
        raise NotImplementedError()
    
    def _is_frame_scrolled_at_end(self, iframe_id):
        
        return self.driver.execute_script(f'return is_frame_scrolled_at_end("{iframe_id}")')
    
    def _is_in_screen(self, iframe_id, query):
        
        return self.driver.execute_script(f'return is_in_screen("{iframe_id}", "{query}")')
    
    def _get_latest_update(self):
        
        return self.driver.execute_script('return get_latest_update()')
        
    def _get_displayed_words(self):
        
        return self.driver.execute_script('return get_displayed_words()')
    
    def get_view_rect(self):
        
        return self.driver.execute_script('b=document.body; return [b.scrollLeft, b.scrollTop, b.clientWidth, b.clientHeight];')
    
    def _meta(self, name):
        return self.driver.execute_script(f'return document.head.querySelector("meta[name=\'{name}\']").content')
    
    def _inner_text(self, target):
        _type = target[0]
        _ident = target[1:]
        if _type == '#':
            return self.driver.execute_script(f'return document.getElementById("{_ident}").innerText')
        elif _type == '.':
            return self.driver.execute_script(f'return document.getElementsByClassName("{_ident}")[0].innerText')
        raise RuntimeError(f'unknown type {_type}: {target}')
    
    def _attr(self, name, target):
        _type = target[0]
        _ident = target[1:]
        if _type == '#':
            return self.driver.execute_script(f'return document.getElementById("{_ident}").getAttribute("{name}")')
        elif _type == '.':
            return self.driver.execute_script(f'return document.getElementsByClassName("{_ident}")[0].getAttribute("{name}")')
        raise RuntimeError(f'unknown type {_type}: {target}')
    
    def _prop(self, name, target):
        _type = target[0]
        _ident = target[1:]
        if _type == '#':
            return self.driver.execute_script(f'return document.getElementById("{_ident}").{name}')
        elif _type == '.':
            return self.driver.execute_script(f'return document.getElementsByClassName("{_ident}")[0].{name}')
        raise RuntimeError(f'unknown type {_type}: {target}')
    
    def _rel_rect(self, target):
        _type = target[0]
        _ident = target[1:]
        if _type == '#':
            return self.driver.execute_script(f'return document.getElementById("{_ident}").getBoundingClientRect()')
        elif _type == '.':
            return self.driver.execute_script(f'return document.getElementsByClassName("{_ident}")[0].getBoundingClientRect()')
        raise RuntimeError(f'unknown type {_type}: {target}')
    
    def __init__(self, 
            binary_location=None,
            open_ports_dir='open_ports',
            window_size=(640, 320), 
            sec_for_waiting=0.20,
            draw_pointer=True,
            *args, **kwargs
        ):
        
        self.binary_location = binary_location
        self.open_ports_dir = open_ports_dir
        self.window_size = tuple(window_size)
        self.sec_for_waiting = sec_for_waiting
        self.draw_pointer = draw_pointer
        
        self.driver = None
        self._set_driver()
        
        self.driver.get('about:blank')
        time.sleep(0.1)
        self._adjust_window_size()
        self._size_yet_adjusted = False

        self.page_cache = {}
        self.clear_page_cache()
        
        self.view_cache = {}
        self.clear_view_cache()
        
        self.custom_state = {}
        self.clear_custom_state()
    
    def clear_page_cache(self):
        self.page_cache = {
            'displayed_words': None,
            'timestamp': None,
        }
    
    def clear_view_cache(self):
        self.view_cache = {
            'screenshot': None,
        }
    
    def clear_custom_state(self):
        self.custom_state = {
            'pointer_xy': None,
            'last_action': None,
        }
        
    def __del__(self):
        
        self._unset_driver()

    def truncate_xy(self, xy):
        
        buffer = [0, 0]
        for i in range(2):
            u = self.window_size[i]
            x = xy[i]
            buffer[i] = 0 if x < 0 else (x if x < u else u-1)
        return tuple(buffer)
            
    def reset(self, return_observation=True, dummy_action=None):
        
        self.clear_page_cache()
        self.clear_view_cache()
        
        if dummy_action is not None:
            self.custom_state['pointer_xy'] = self.truncate_xy(dummy_action.pointer_xy)
            self.custom_state['last_action'] = dummy_action
        else:
            self.custom_state['pointer_xy'] = (0, 0)
            self.custom_state['last_action'] = ModelAction('[PAD]', '[PAD]', (0, 0))
        
        if return_observation:
            return self._get_observation(instantly=False)
        
    def _get_observation(self, instantly):
        
        if self._size_yet_adjusted:
            self._adjust_window_size()
            self._size_yet_adjusted = False
        
        if (not instantly) and self.sec_for_waiting > 0:
            time.sleep(self.sec_for_waiting)
        
        detected_words=self.get_visible_words()
        screenshot=self.get_screenshot()
        
        x = ModelObservation(
            screenshot=screenshot,
            detected_words=detected_words, 
            pointer_xy=self.custom_state['pointer_xy'],
            last_action=self.custom_state['last_action'],
        )
        return x
    
    def _adjust_window_size(self):
        
        screenshot = PIL.Image.open(io.BytesIO(self.driver.get_screenshot_as_png()))
        w, h = screenshot.size
        dx = self.window_size[0] - w
        dy = self.window_size[1] - h
        if dx != 0 or dy !=0 :
            self.driver.set_window_size(self.window_size[0]+dx, self.window_size[1]+dy)
            self.clear_page_cache()
            self.clear_view_cache()
    
    def get_visible_words(self):
        
        # update cache
        latest_timestamp = int(self._get_latest_update())
        if self.page_cache['timestamp'] is None or self.page_cache['timestamp'] < latest_timestamp:
            self.page_cache['displayed_words'] = [
                Word(_['s'], _['x'] + 0.5*_['w'], _['y']+0.5*_['h'], _['w'], _['h']) for _ in self._get_displayed_words()
            ]
            self.page_cache['timestamp'] = latest_timestamp
    
        return self.page_cache['displayed_words']
    
    def get_screenshot(self):
        screenshot = self.view_cache['screenshot']
        if screenshot is None or self.page_cache['timestamp'] < int(self._get_latest_update()):
            screenshot = self.update_screenshot()
        
        image = screenshot.copy()
        if self.draw_pointer:
            draw = PIL.ImageDraw.Draw(image)
            px, py = self.custom_state['pointer_xy']
            draw.ellipse([(px-3, py-3), (px+3, py+3)], fill=(0, 0, 0))
            del draw
        return image   
    
    def update_screenshot(self):
        
        screenshot = PIL.Image.open(io.BytesIO(self.driver.get_screenshot_as_png()))
        self.view_cache['screenshot'] = screenshot = screenshot.convert(mode='RGB')
        return screenshot
    
    def _modify_surface(self, surface, is_continuous):
        
        if surface.startswith('##'):
            return surface[2:]
        elif is_continuous:
            return ' '+surface 
        return surface
    
    def apply(self, action):
        """
        action: ModelAction
        returns ModelObservation
        """
        def update_action_and_get_observation(instantly):
            self.custom_state['last_action'] = action.copy()
            x = self._get_observation(instantly=instantly)
            return x
        
        if action.stop_propagation or action.name == '[PAD]':
            return update_action_and_get_observation(True)
        
        else:
            self.clear_view_cache()
            
            if action.name == 'move_to':
                self.custom_state['pointer_xy'] = self.truncate_xy(action.pointer_xy)
                x, y = self.custom_state['pointer_xy']
                ActionChainsEx(self.driver).move_to_location(x, y).perform()
                return update_action_and_get_observation(instantly=False)
        
            elif action.name == 'click':
                ActionChainsEx(self.driver).click().perform()
                return update_action_and_get_observation(instantly=False)
            
            elif action.name in self.KEY_MAPPING:
                keys = self.KEY_MAPPING[action.name]
                ActionChainsEx(self.driver).send_keys(keys).perform()
                return update_action_and_get_observation(instantly=False)
        
            elif action.name == 'token':
                #last_action = self.custom_state['last_action']
                #is_continuous = last_action.name == 'token'
                is_continuous = False
                keys = self._modify_surface(action.string, is_continuous)
                ActionChainsEx(self.driver).send_keys(keys).perform()
                return update_action_and_get_observation(instantly=False)
            
            else:
                raise ValueError(f'unknown action name {action.name}')
    
    def get_status(self):
        
        return self._meta('status')
    
    def get_answer(self):
    
        return self._meta('answer')
    
    def get_submitted(self):
    
        return json.loads(self._meta('submitted'))
    
    def serialize_teacher_actions(self, url, dummy_action=None):
        
        actions, observations = self.replay_teacher_actions(url, dummy_action)
        
        actions = [_.as_dict() for _ in actions]
        observations = [_.as_dict() for _ in observations]
        
        config = {
            'window_size': self.window_size,
            'sec_for_waiting': self.sec_for_waiting,
            'draw_pointer': self.draw_pointer,
        }
        
        return {
            'config': config,
            'actions': actions,
            'observations': observations,
        }
        
    @classmethod
    def deserialize_teacher_actions(cls, data):
        
        actions = [ModelAction.from_dict(_) for _ in data['actions']]
        observations = [ModelObservation.from_dict(_) for _ in data['observations']]
        return data['config'], actions, observations


class EnvST(EnvWithSelenium):
    
    def __init__(self, url_base, tokenizer, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        self.url_base = url_base
        self.tokenizer = tokenizer
        
    def reset(self, url, *args, **kwargs):

        super().reset(return_observation=False, *args, **kwargs)
        
        self.driver.get(self.url_base + url)
                
        x = self._get_observation(instantly=False)
        return x

    def get_all_ids(self, split):
        
        return get_all_ids(self.url_base, split)
    
    def get_rectangle(self, elem):
        rect = elem.rect
        return rect['x'], rect['y'], rect['width'], rect['height']

    def replay_teacher_actions(self, url=None, dummy_action=None):
        
        if url is not None:
            
            self.reset(url, dummy_action=dummy_action)
        
        teacher_actions = self._meta('teacher_actions').split()
        
        actions = []
        observations = [self._get_observation(instantly=True)]
        
        for line in teacher_actions:
            cells = line.split('+')
            handler_name = '_handler_' + cells[0]
            if not hasattr(self, handler_name):
                raise ValueError(f'handler not found for {handler_name}')
            handler = getattr(self, handler_name)
            _actions, _observations = handler(*cells[1:])
            actions.extend(_actions)
            observations.extend(_observations)
        
        return actions, observations
    
    def _handler_click(self):
        
        action = ModelAction('click')
        observation = self.apply(action)
        return [action], [observation]
    
    def _handler_iframe_move_to(self, frame_id, query, direction='down', max_trial=50):
        
        actions = []
        observations = []
        
        left_trials = max_trial
        elem_rect = self._is_in_screen(frame_id, query)
        while (not elem_rect['displayed_all']) and (left_trials > 0):
            action = ModelAction(direction)
            actions.append(action)
            observations.append(self.apply(action))
            left_trials -= 1
            elem_rect = self._is_in_screen(frame_id, query)
        
        x = int(elem_rect['x'] + elem_rect['w']*0.5)
        y = int(elem_rect['y'] + elem_rect['h']*0.5)
        action = ModelAction('move_to', '', (x, y))
        actions.append(action)
        observations.append(self.apply(action))
        
        return actions, observations
    
    def _handler_iframe_scroll_down(self, frame_id, max_trial=50):
        
        actions = []
        observations = []
        
        left_trials = max_trial
        
        while (not self._is_frame_scrolled_at_end(frame_id)) and (left_trials > 0):
            action = ModelAction('down')
            actions.append(action)
            observations.append(self.apply(action))
            left_trials -= 1
        
        return actions, observations
    
    def _handler_move_to(self, elem_class, position='center', max_trial=50, alpha=5):
        
        actions = []
        observations = []
        
        # move view until elem appears in the screen
        for t in range(max_trial):
            
            elem_rect = self._rel_rect(elem_class)
            
            del_left = elem_rect['x']
            del_right = self.window_size[0] - elem_rect['right']
            del_top = elem_rect['y']
            del_bottom = self.window_size[1] - elem_rect['bottom']
            if del_left >= -alpha and del_right >= -alpha and \
                    del_top >= -alpha and del_bottom >= -alpha:
                break
            
            if del_left < -alpha:
                name = 'left'
            elif del_right < -alpha:
                name = 'right'
            elif del_top < -alpha:
                name = 'up'
            elif del_bottom < -alpha:
                name = 'down'
            else:
                # Why reached here?
                raise Exception(f'unreachable point: ({abs_x}, {abs_y})')
            
            action = ModelAction(name)
            actions.append(action)
            observations.append(self.apply(action))
        
        elem_rect = self._rel_rect(elem_class)
        is_centered = position == 'center'
        x = int(elem_rect['x'] + elem_rect['width']*0.5*is_centered)
        y = int(elem_rect['y'] + elem_rect['height']*0.5*is_centered)
        action = ModelAction('move_to', '', (x, y))
        actions.append(action)
        observations.append(self.apply(action))
        
        return actions, observations
    
    def _handler_key_stroke(self, meta_name):
        
        text = self._meta(meta_name)
        tokens = []
        for word in text.split():
            tokens.extend(self.tokenizer.encode(word, add_special_tokens=False).tokens)
            tokens.append(None)
        tokens.pop()
        
        actions = []
        observations = []
        
        for token in tokens:
            if token:
                action = ModelAction('token', token)
            else:
                action = ModelAction('space')
            actions.append(action)
            observations.append(self.apply(action))
        
        return actions, observations
    
    def _handler_backspace(self, elem_class):
        
        text = self._inner_text(elem_class)
        if len(text) == 0:
            text = self._prop(target=elem_class, name='value')
        n = len(text)
        
        actions = []
        observations = []
        
        for _ in range(n):
            action = ModelAction('backspace')
            actions.append(action)
            observations.append(self.apply(action))
        
        return actions, observations



class EnvSTFirefox(EnvST):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        if self.binary_location is None:
            self.binary_location = 'venv/lib/firefox/firefox'

    def _set_driver(self):

        options = firefox.options.Options()
        options.binary_location = self.binary_location

        options.add_argument('-headless')
        options.add_argument('-width=%s'%self.window_size[0])
        options.add_argument('-height=%s'%self.window_size[1])
        # To avoid poor appearance of elements when using the Linux systems
        options.set_preference('widget.disable-native-theme-for-content', True)
        # To disable quick find, 
        # which can be called by typing some keys and changes screenshot size
        options.set_preference('accessibility.typeaheadfind.manual', False)
        options.set_preference('accessibility.typeaheadfind', False)
        options.set_preference('accessibility.typeaheadfind.autostart', False)

        self.driver = webdriver.Firefox(firefox_options=options)

    def _unset_driver(self):
        
        if self.driver is not None:
            self.driver.quit()
            self.driver = None
