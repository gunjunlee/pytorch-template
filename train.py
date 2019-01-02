import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import numpy as np
import toml

import random
import os
import os.path as osp


from models import Model, get_model
from optimizer import get_optimizer
from loss import get_loss
from metric import get_metric
from scheduler import get_scheduler
from dataloader import BaseDataset, BaseDataLoader
from utils import parse_config
from transform import get_seg_transform

def arg_parse():
    import argparse
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('--config', default='config/config.toml', help='path to config file')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = arg_parse()
    config = parse_config(args.config)

    random.seed(1000) # random
    np.random.seed(1000) # np.random
    torch.manual_seed(1000) # torch cpu
    torch.cuda.manual_seed_all(1000) # torch gpu
    # torch.backends.cudnn.deterministic = True # cudnn. *CAUTION* this will make it slow to learn.

    if osp.isdir(config['LOGDIR']):
        chk = input('log dir already exists. are you sure? (y/n): ')
        assert chk.lower() == 'y'
    else:
        os.makedirs(config['LOGDIR'])
    with open(osp.join(config['LOGDIR'], 'config.toml'), 'w') as f:
        toml.dump(config, f)

    '''suggested format
    dataset = CustomDataset()
    dataloader = BaseDataLoader()
    len_train = len(dataset) * 8 // 10
    len_val = len(dataset) - len_train
    train_dataloader, val_dataloader = dataloader.split([len_train, len_val])

    net = get_model()
    if config.GPU:
        net = net.cuda()
        net = nn.DataParallel(net)
    
    criterion = get_loss(config)
    metric = get_metric(config)
    optimizer = get_optimizer(filter(lambda p:p.requires_grad, net.parameters()), config)
    scheduler = get_scheduler(optimizer, config)

    model = Model(net)
    model.summary(input_size=(3, 1024, 1024), use_gpu=config.GPU)
    model.compile(optimizer, criterion, metric, scheduler)
    model.fit(train_dataloader=train_dataloader,
              val_dataloader=val_dataloader,
              epoch=config.EPOCHS,
              use_gpu=config.GPU,
              pth='ckpt/models.pth',
              log=config.LOGDIR)
    '''