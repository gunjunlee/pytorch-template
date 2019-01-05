import toml
from easydict import EasyDict as edict
from termcolor import colored

import copy
import shutil


class Accumulator:
    def __init__(self):
        self.val = 0
        self.len = 0

    def __add__(self, val):
        self.val += val
        self.len += 1
        return self

    def mean(self):
        return self.val / self.len

    def init(self):
        self.val = 0
        self.len = 0
        return self


class WeightAverage:
    def __init__(self, len):
        self.state_dicts = []
        self.len = len

    def append(self, state_dict):
        self.state_dicts.append(state_dict)
        if len(self.state_dicts) > self.len:
            self.state_dicts.pop(0)
        return self
    
    def mean(self):
        keys = self.state_dicts[0].keys()
        avg_state_dict = dict()
        for key in keys:
            avg_state_dict[key] = sum(state_dict[key] for state_dict in self.state_dicts)
            avg_state_dict[key] = avg_state_dict[key] / len(self.state_dicts)
        return avg_state_dict


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


def parse_config(config_path):
    return edict(toml.load(config_path))