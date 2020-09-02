import importlib

def import_class(class_str):
    class_str = class_str.split('.')
    module_name = '.'.join(class_str[:-1])
    module = importlib.import_module(module_name)
    return getattr(module, class_str[-1])

if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.abspath('.'))

    model = import_class('model.recurrent_norm.Model')
    print(model)
