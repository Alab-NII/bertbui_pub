# coding: utf-8


import dataclasses
import argparse


def parse_from_dataclass(cls_or_cls_dict, description=''):
    
    return_dict = isinstance(cls_or_cls_dict, dict)
    argument_mapping = {}
    parser = argparse.ArgumentParser(description=description)
    
    def add_arguments(cls, dict_key=None):
        
        if cls in argument_mapping:
            raise RuntimeError('%s has already registered' % str(cls))
        
        prefix = '[%s] ' % dict_key if dict_key else ''
        arguments = argument_mapping[cls] = []

        for _name, field in cls.__dataclass_fields__.items():
            
            name = '--'+_name
            kwargs = {}
            kwargs['type'] = field.type
            
            if not isinstance(field.default, dataclasses._MISSING_TYPE):
                kwargs['default'] = field.default
            else:
                kwargs['required'] = True
             
            if 'help' in field.metadata:
                kwargs['help'] = field.metadata['help']
                if 'default'in kwargs:
                    kwargs['help'] = '%s%s (default=%s)' % (prefix, kwargs['help'], kwargs['default'])
            
            parser.add_argument(name, **kwargs)
            arguments.append(_name)
    
    if not return_dict:
        cls = cls_or_cls_dict
        add_arguments(cls)
        namespace = vars(parser.parse_args())
        return cls(**{a: namespace[a] for a in argument_mapping[cls]})
    else:
        for dict_key, cls in cls_or_cls_dict.items():
            add_arguments(cls, dict_key)
        namespace = vars(parser.parse_args())
        return {
                key: cls(**{a: namespace[a] for a in argument_mapping[cls]}) 
                for key, cls in cls_or_cls_dict.items()
            }
