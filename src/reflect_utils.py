# coding: utf-8
# Copyright (c) Antfin, Inc. All rights reserved.
import importlib


def get_class(module_class):
    '''
    Get the class using import_module
    Args:
        module_class: module class name, full path a.b.class_name
    '''
    names = module_class.strip().split('.')
    module_name = '.'.join(names[:-1])
    module = importlib.import_module(module_name)

    # Use : to split inner class
    class_names = names[-1].split(":")
    cls_obj = getattr(module, class_names[0])
    for inner_class_name in class_names[1:]:
        cls_obj = getattr(cls_obj, inner_class_name)
    return cls_obj


def new_instance(module_class, *args, **kwargs):
    '''
    Create a new instance using import_module
    Args:
        module_class: module class name, full path a.b.class_name
        args: passed to the constructor of class
        kwargs: passed to the constructor of class
    '''
    instance = get_class(module_class)(*args, **kwargs)
    return instance


def get_class_defined_in_module(module_name, clazz):
    import inspect
    module = importlib.import_module(module_name)
    members = [
        m[1] for m in inspect.getmembers(module, lambda o: inspect.isclass(o) and issubclass(o, clazz))
        if m[1].__module__ == module_name
    ]
    return members


def get_object_defined_in_module(module_name, clazz, name=None):
    import inspect
    module = importlib.import_module(module_name)
    members = {m[0]: m[1] for m in inspect.getmembers(module, lambda o: isinstance(o, clazz))}
    if name:
        return members.get(name, None)
    else:
        return list(members.values())
