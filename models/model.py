import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from tensorboardX import SummaryWriter

import os
import os.path as osp


class Model(nn.Module):
    def __init__(self, net):
        super(Model, self).__init__()
        self.net = net
        self.optimizer = None
        self.criterion = None
        self.metric = None
        self.scheduler = None
        
    def forward(self, x):
        return self.net(x)
    
    def compile(self, optimizer, criterion, metric=None, scheduler=None):
        if optimizer is None:
            raise ValueError('optimizer is None!')
        if criterion is None:
            raise ValueError('criterion is None!')
        
        self.optimizer = optimizer
        self.criterion = criterion
        self.metric = metric
        self.scheduler = scheduler

    def summary(self, input_size=(3, 224, 224), use_gpu=False):
        from torchsummary import summary
        if use_gpu:
            summary(self.net.cuda(), input_size=input_size)
        else:
            summary(self.net.cpu(), input_size=input_size, device='cpu')

    def fit(self, train_dataloader, val_dataloader,
            epoch=100, use_gpu=True, pth='ckpt/model.pth', log='logs/first'):
        if self.optimizer is None:
            raise RuntimeError('optimizer is not defined!'
                               'Compile the model before fitting!')
        if self.criterion is None:
            raise RuntimeError('criterion is not defined!'
                               'Compile the model before fitting!')
        if not osp.isdir(osp.dirname(pth)):
            os.makedirs(osp.dirname(pth))

        import shutil

        writer = SummaryWriter(log)

        def padding(arg, width, pad=' '):
            if isinstance(arg, float):
                return '{:.6f}'.format(arg).center(width, pad)
            elif isinstance(arg, int):
                return '{:6d}'.format(arg).center(width, pad)
            elif isinstance(arg, str):
                return arg.center(width, pad)
            elif isinstance(arg, tuple):
                if len(arg) != 2:
                    raise ValueError('Unknown type: {}'.format(type(arg), arg))
                if not isinstance(arg[1], str):
                    raise ValueError('Unknown type: {}'
                                     .format(type(arg[1]), arg[1]))
                return colored(padding(arg[0], width, pad=pad), arg[1])
            else:
                raise ValueError('Unknown type: {}'.format(type(arg), arg))

        def print_row(kwarg_list=[], pad=' '):
            len_kwargs = len(kwarg_list)
            term_width = shutil.get_terminal_size().columns
            width = min((term_width-1-len_kwargs)*9//10, 150) // len_kwargs
            row = '|{}' * len_kwargs + '|'
            columns = []
            for kwarg in kwarg_list:
                columns.append(padding(kwarg, width, pad=pad))
            print(row.format(*columns))

        from termcolor import colored
        from time import time

        kwarg_list = ['epoch', 'loss', 'metric',
                      'val loss', 'val metric', 'time']

        print(colored('model training start!', 'green'))

        print_row(kwarg_list=['']*len(kwarg_list), pad='-')
        print_row(kwarg_list=kwarg_list, pad=' ')
        print_row(kwarg_list=['']*len(kwarg_list), pad='-')

        min_val_loss = 1e+8
        for ep in range(1, epoch+1):
            start_time = time()
            train_loss = None
            val_loss = None
            train_metric = None
            val_metric = None

            for phase in ['train', 'val']:
                running_loss = 0
                running_metric = 0

                if phase == 'train':
                    self.net = self.net.train()
                    dataloader = train_dataloader
                elif phase == 'val':
                    self.net = self.net.eval()
                    dataloader = val_dataloader

                for batch_x, batch_y in tqdm(dataloader, leave=False):
                    self.optimizer.zero_grad()

                    if use_gpu:
                        batch_x = batch_x.cuda()
                        batch_y = batch_y.cuda()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.net(batch_x)
                        loss = self.criterion(outputs, batch_y)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    running_loss += loss.item() * batch_x.size(0)
                    running_metric += self.metric(outputs, batch_y).item() * batch_x.size(0)

                running_loss = running_loss / len(dataloader.dataset)
                running_metric = running_metric / len(dataloader.dataset)

                if phase == 'train':
                    train_loss = running_loss
                    train_metric = running_metric
                elif phase == 'val':
                    val_loss = running_loss
                    val_metric = running_metric

            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            elapsed_time = time()-start_time

            writer.add_scalar('loss/train', train_loss, ep)
            writer.add_scalar('loss/val', val_loss, ep)
            writer.add_scalar('metric/train', train_metric, ep)
            writer.add_scalar('metric/val', val_metric, ep)

            if min_val_loss > val_loss:
                min_val_loss = val_loss
                self.save_model(pth)
                ep = (str(ep)+'(saved)', 'blue')

            print_row(kwarg_list=[ep, train_loss, train_metric,
                                  val_loss, val_metric, elapsed_time], pad=' ')
            print_row(kwarg_list=['']*len(kwarg_list), pad='-')

    def save_model(self, pth):
        if hasattr(self.net, 'module'):
            state_dict = self.net.module.state_dict()
        else:
            state_dict = self.net.state_dict()

        torch.save(state_dict, pth)
