from .model import Model

def get_model(config):
    funcs = {

    }

    name = config.MODEL.NAME
    
    if name in funcs:
        func = funcs[name]
    else:
        func = globals()[name]
    
    print('get model: {}'.format(name))
    return func()

