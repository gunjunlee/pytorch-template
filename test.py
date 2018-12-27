def arg_parse():
    import argparse
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('--config', default='config/config.toml', help='path to config file')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = arg_parse()
