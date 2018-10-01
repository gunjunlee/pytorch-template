def arg_parse():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/train', help='path to training data')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = arg_parse()
