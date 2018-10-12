from models import Model
from loss import Loss
from metric import Metric


def arg_parse():
    import argparse
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('--data', default='./data/train', help='path to training data')
    add_arg('--batch_size', default=32, type=int, help='batch size')
    add_arg('--lr', default=1e-3, type=float, help='learning rate')
    add_arg('--num_workers', default=8, type=int, help='num of workers')
    add_arg('--gpu', default=True, help='using gpu')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = arg_parse()
