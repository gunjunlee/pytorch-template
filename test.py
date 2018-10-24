def arg_parse():
    import argparse
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('--data', default='./data/test', help='path to test data')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = arg_parse()
