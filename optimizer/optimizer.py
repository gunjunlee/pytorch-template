import torch
import torch.optim as optim


def rmsprop(params, config):
    momentum = config.OPTIMIZER.MOMENTUM
    weight_decay = config.OPTIMIZER.WEIGHT_DECAY
    lr = config.OPTIMIZER.LR
    return optim.RMSprop(params, 
                         lr=lr, 
                         momentum=momentum, 
                         weight_decay=weight_decay)


def adam(params, config):
    weight_decay = config.OPTIMIZER.WEIGHT_DECAY
    lr = config.OPTIMIZER.LR
    return optim.Adam(params, 
                      lr=lr, 
                      weight_decay=weight_decay)


def sgd(params, config):
    momentum = config.OPTIMIZER.MOMENTUM
    weight_decay = config.OPTIMIZER.WEIGHT_DECAY
    lr = config.OPTIMIZER.LR
    return optim.SGD(params, 
                     lr=lr, 
                     momentum=momentum, 
                     weight_decay=weight_decay)


def get_optimizer(params, config):
    funcs = {
        'SGD': sgd,
        'ADAM': adam,
        'RMSPROP': rmsprop,
    }

    name = config.OPTIMIZER.NAME

    if name in funcs:
        func = funcs[name]
    else:
        func = globals()[name]

    print('get optimizer: {}'.format(name))
    return func(params, config)
