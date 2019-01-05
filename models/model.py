import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from tensorboardX import SummaryWriter
from termcolor import colored
from torchsummary import summary

import os
import os.path as osp
import sys
import copy
from time import time

sys.path.append('../')

from utils import print_row, WeightAverage, Accumulator

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

        kwarg_list = ['epoch', 'loss', 'metric',
                      'val loss', 'val metric', 
                      'avg_val_loss', 'avg_val_metric', 'time']

        print(colored('model training start!', 'green'))

        print_row(kwarg_list=['']*len(kwarg_list), pad='-')
        print_row(kwarg_list=kwarg_list, pad=' ')
        print_row(kwarg_list=['']*len(kwarg_list), pad='-')

        min_val_loss = 1e+8
        min_val_metric = -1e+8
        weight_average = WeightAverage(len=5)
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
                elif phase == 'avg_val':
                    state_dict = copy.deepcopy(self.net.state_dict())
                    weight_average.append(state_dict)
                    self.net.load_state_dict(weight_average.mean())
                    dataloader = val_dataloader

                batch_accum = 0
                pbar = tqdm(dataloader, leave=False)
                for batch_x, batch_y in pbar:
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
                    batch_accum += batch_x.size(0)
                    pbar.set_description('loss:{:.4f}, metric:{:.4f}'.format(running_loss/batch_accum, running_metric/batch_accum))

                running_loss = running_loss / len(dataloader.dataset)
                running_metric = running_metric / len(dataloader.dataset)

                if phase == 'train':
                    train_loss = running_loss
                    train_metric = running_metric
                elif phase == 'val':
                    val_loss = running_loss
                    val_metric = running_metric
                elif phase == 'avg_val':
                    avg_val_loss = running_loss
                    avg_val_metric = running_metric
                    self.net.load_state_dict(state_dict)


            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            elapsed_time = time()-start_time

            writer.add_scalar('loss/train', train_loss, ep)
            writer.add_scalar('loss/val', val_loss, ep)
            writer.add_scalar('loss/avg_val', avg_val_loss, ep)
            writer.add_scalar('metric/train', train_metric, ep)
            writer.add_scalar('metric/val', val_metric, ep)
            writer.add_scalar('metric/avg_val', avg_val_metric, ep)

            if min_val_metric < val_metric:
                min_val_metric = val_metric
                self.save_model(pth)
                ep = (str(ep)+'(saved)', 'blue')
            if min_val_metric < avg_val_metric:
                min_val_metric = avg_val_metric
                self.save_model(pth, weight_average)
                print(isinstance(ep, tuple))
                if not isinstance(ep, tuple):
                    ep = (str(ep)+'(saved)', 'blue')

            print_row(kwarg_list=[ep, train_loss, train_metric,
                                  val_loss, val_metric, avg_val_loss, 
                                  avg_val_metric, elapsed_time], pad=' ')
            print_row(kwarg_list=['']*len(kwarg_list), pad='-')

    def extract_state_dict(self, model, weight_average=None):
        def get(model):
            if hasattr(model, 'module'):
                return self.net.module.state_dict()
            else:
                return self.net.state_dict()

        if weight_average is None:
            state_dict = get(model)
        else:
            temp = model.state_dict()
            state_dict = weight_average.mean()
            model.load_state_dict(state_dict)
            state_dict = get(model)
            model.load_state_dict(temp)
        return state_dict

    def save_model(self, pth, weight_average=None):
        state_dict = self.extract_state_dict(self.net, weight_average)
        torch.save(state_dict, pth)
