# coding: utf-8


import PIL.Image, PIL.ImageDraw
import base64
import io
import numpy as np


class Word(object):

    def __init__(self, surface, cx, cy, width, height):
        
        self.surface = surface
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
    
    def __repr__(self):
        
        return f'{self.surface}@({self.cx}, {self.cy}, {self.width}, {self.height})'
    
    @classmethod
    def from_word_rect(cls, rect):
         
        w = rect['width']
        h = rect['height']
        return cls(rect['surface'], rect['x']+w*0.5,  rect['y']+h*0.5, w, h)
    
    def as_dict(self):
        
        return {
            'surface': self.surface,
            'cx': self.cx,
            'cy': self.cy,
            'width': self.width,
            'height': self.height,
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    
class ModelAction(object):
    
    NAMES = ['[PAD]', 'move_to', 'token', 'click', 'left', 'right', 'up', 'down', 'space', 'backspace', 'enter']
    PAD_STRING = '[PAD]'
    # for compatibility
    FUNCTIONAL_STRINGS = ['[CLICK]', '[LEFT]', '[RIGHT]', '[UP]', '[DOWN]', '[SPACE]', '[BACKSPACE]', '[ENTER]']
    
    def __init__(self, name, string='', pointer_xy=(0, 0), stop_propagation=False):
        
        self.name = name
        self.string = string
        self.pointer_xy = pointer_xy
        self.stop_propagation = stop_propagation

    def __repr__(self):
        
        return f'({self.name}, "{self.string}", {self.pointer_xy}, {self.stop_propagation})'
    
    def show(self):
        
        if self.name == 'move_to':
            return f'action({self.name}, {self.pointer_xy})'
        elif self.name == 'token':
            return f'action({self.name}, "{self.string}")'
        elif self.name in self.NAMES:
            return f'action({self.name})'
        # unknown action
        return f'action({self.name}, "{self.string}", {self.pointer_xy})'
    
    def copy(self):
        
        return ModelAction(self.name, self.string, self.pointer_xy, self.stop_propagation)
    
    def as_dict(self):
        
        return {
            'name': self.name,
            'string': self.string,
            'pointer_xy': self.pointer_xy,
            'stop_propagation': self.stop_propagation,
        }
    
    def to_numpy(self, model_config, tokenizer):
        
        if self.name == '[PAD]':
            return np.zeros((4,), dtype=np.int)
            
        name_id = model_config.action_to_id[self.name]
        token_id = x_id = y_id = 0
        
        if self.name == 'move_to':
            x_id = model_config.get_id_x(self.pointer_xy[0])
            y_id = model_config.get_id_y(self.pointer_xy[1])
        elif self.name == 'token':
            token_id = tokenizer.token_to_id(self.string)
        
        return np.asarray([name_id, token_id, x_id, y_id], dtype=np.int)
    
    @classmethod
    def from_dict(cls, data):
        # for compatibility
        if data['string'] in cls.FUNCTIONAL_STRINGS:
            return cls(data['string'][1:-1].lower(), '', data['pointer_xy'], data['stop_propagation'])
        elif data['name'] == 'string':
            return cls('token', data['string'], data['pointer_xy'], data['stop_propagation'])
        # 
        return cls(**data)
    

class ModelObservation(object):
    
    def __init__(self, screenshot, detected_words, pointer_xy, last_action):
        
        self.screenshot = screenshot
        self.detected_words = detected_words
        self.pointer_xy = pointer_xy
        self.last_action = last_action
    
    @staticmethod
    def _encode_base64(image, format_type='png'):
        buffer = io.BytesIO()
        image.save(buffer, format=format_type)
        return base64.b64encode(buffer.getvalue()).decode("ascii")
    
    @staticmethod
    def _decode_base64(data):
        return PIL.Image.open(io.BytesIO(base64.b64decode(data)))
    
    @classmethod
    def from_dict(cls, data):
        screenshot = cls._decode_base64(data['screenshot'])
        detected_words = [Word.from_dict(w) for w in data['detected_words']]
        last_action = ModelAction.from_dict(data['last_action'])
        pointer_xy = data['pointer_xy']
        return cls(screenshot, detected_words, pointer_xy, last_action)
    
    def as_dict(self):
        
        return {
            'screenshot': self._encode_base64(self.screenshot),
            'detected_words': [w.as_dict() for w in self.detected_words],
            'pointer_xy': self.pointer_xy,
            'last_action': self.last_action.as_dict(),
        }
    
    def _add_ocr_frames(self, words=None):
        image = self.screenshot.copy()
        draw = PIL.ImageDraw.ImageDraw(image)
        for w in words or self.detected_words:
            x = int(w.cx - 0.5*w.width)
            y = int(w.cy - 0.5*w.height)
            xx = int(w.cx + 0.5*w.width)
            yy = int(w.cy + 0.5*w.height)
            draw.rectangle((x, y, xx, yy), outline='black')
        del draw
        return image
    
    @property
    def screenshot_with_ocr(self):
        return self._add_ocr_frames()

def deserialize_teacher_actions(data_dict):

    actions = [ModelAction.from_dict(_) for _ in data_dict['actions']]
    observations = [ModelObservation.from_dict(_) for _ in data_dict['observations']]
    return data_dict['config'], actions, observations
