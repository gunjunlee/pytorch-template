import toml
from easydict import EasyDict as edict

def parse_config(config_path):
    return edict(toml.load(config_path))
