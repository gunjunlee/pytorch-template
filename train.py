import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import random

from models import Model
from loss import Loss
from metric import Metric
from dataloader import BaseDataset, BaseDataLoader


def arg_parse():
    import argparse
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('--data', default='./data/train', help='path to training data')
    add_arg('--batch_size', default=32, type=int, help='batch size')
    add_arg('--lr', default=1e-3, type=float, help='learning rate')
    add_arg('--num_workers', default=8, type=int, help='num of workers')
    add_arg('--gpu', default=True, help='using gpu')
    add_arg('--epochs', default=100, type=int, help='num of epochs')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = arg_parse()
    random.seed(1000) # random 을 쓰는 경우
    np.random.seed(1000) # np.random 을 쓰는 경우
    torch.manual_seed(1000) # torch cpu
    torch.cuda.manual_seed_all(1000) # torch gpu
    torch.backends.cudnn.deterministic = True # cudnn 을 쓰는 경우

    '''suggested format
    dataset = Dataset()
    dataloader = Dataloader()
    len_train = len(dataset) * 8 // 10
    len_val = len(dataset) - len_train
    train_dataloader, val_dataloader = dataloader.split([len_train, len_val])

    net = Net()
    if args.gpu:
        net = net.cuda()
        net = nn.DataParallel(net)
    
    criterion = Loss()
    metric = Metric()
    optimizer = optim.SGD(filter(lambda p:p.requires_grad, net.parameters()), 
                          lr=args.lr, 
                          momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model = Model(net)
    model.summary(input_size=(3, 1024, 1024), use_gpu=args.gpu)
    model.compile(optimizer, criterion, metric, scheduler)
    model.fit(train_dataloader=train_dataloader,
              val_dataloader=val_dataloader,
              epoch=args.epochs,
              use_gpu=args.gpu)
    '''