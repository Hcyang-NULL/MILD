
import os
import sys
sys.path.insert(0, os.getcwd())

import json
import argparse
from easydict import EasyDict as Edict


def parse_args():
    parser = argparse.ArgumentParser(description='MILD Training')
    parser.add_argument('-c', '--config', required=True, type=str, help='path to training configuration file')
    parser.add_argument('--suffix', dest='suffix', default='suffix string append to the name of saving directory')
    parser.add_argument('--no-val', dest='no_val', action='store_true', default=False, help='do not split validation')
    args = parser.parse_args()
    return args


def load_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    config = Edict(config)
    return config


def main():
    args = parse_args()
    config = load_config(args.config)
    config.title += args.suffix
    config.suffix = args.suffix

    if args.no_val:
        config.dataset.split_val = False

    if config.trainer == 'MILD':
        from framework.mild.trainer import train
        train(config)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
