'''
config.py
Written by Li Jiang
'''

import argparse
import yaml
import os

def get_parser():
    parser = argparse.ArgumentParser(description='Point Cloud Segmentation')
    parser.add_argument('--config', type=str, default='config/pointgroup_default_scannet.yaml', help='path to config file')

    ### pretrain
    parser.add_argument('--pretrain', type=str, default='', help='path to pretrain model')

    args_cfg = parser.parse_args()
    assert args_cfg.config is not None
    with open(args_cfg.config, 'r') as f:
        config = yaml.load(f)
    for key in config:
        for k, v in config[key].items():
            setattr(args_cfg, k, v)

    return args_cfg

# cfg = get_parser()
# setattr(cfg, 'exp_path', os.path.join('exp', cfg.dataset, cfg.model_name, cfg.config.split('/')[-1][:-5]))


cfg = None
def get_parser_notebook(cfg_file=None, pretrain_path=None):
    parser = argparse.ArgumentParser(description='Point Cloud Segmentation')
    parser.add_argument('--config', type=str, default='config/pointgroup_default_scannet.yaml', help='path to config file')

    ### pretrain
    parser.add_argument('--pretrain', type=str, default='', help='path to pretrain model')

    input_list = []
    if cfg_file:
        input_list += ['--config', cfg_file]
    if pretrain_path:
        input_list += ['--pretrain', pretrain_path]

    global cfg
    cfg = parser.parse_args(input_list)
    assert cfg.config is not None

    with open(cfg.config, 'r') as f:
        config = yaml.load(f)
    for key in config:
        for k, v in config[key].items():
            setattr(cfg, k, v)

    setattr(cfg, 'exp_path', os.path.join('exp', cfg.dataset, cfg.model_name, cfg.config.split('/')[-1][:-5]))
