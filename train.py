from models import Model
from loss import Loss
from metric import Metric


def arg_parse():
    import argparse
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('--data', default='./data/train', help='path to training data')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = arg_parse()
