import os
import sys
from urllib.parse import urlparse

import torch
from torch.utils.model_zoo import _download_url_to_file
import requests
from maskrcnn_benchmark.config.defaults import _C as default_cfg

from dl_ext.vision_ext.models.maskrcnn import maskrcnn


def loadurl(url, model_dir=None, map_location=None, progress=True):
    if model_dir is None:
        torch_home = os.path.expanduser('~/.dl_ext')
        model_dir = os.path.join(torch_home, 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        _download_url_to_file(url, cached_file, None, progress=progress)
    return torch.load(cached_file, map_location=map_location)


def loadconfig(url, config_dir=None):
    if config_dir is None:
        torch_home = os.path.expanduser('~/.dl_ext')
        config_dir = os.path.join(torch_home, 'configs')
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(config_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        with open(cached_file, 'w') as f:
            f.write(requests.get(url).text)
    return default_cfg.merge_from_file(cached_file)


def main():
    d = loadurl('https://download.pytorch.org/models/maskrcnn/e2e_faster_rcnn_R_50_C4_1x.pth')
    yaml_url = 'https://raw.githubusercontent.com/facebookresearch/maskrcnn-benchmark/master/configs/e2e_faster_rcnn_R_50_C4_1x.yaml'
    cfg = loadconfig(yaml_url)
    model = maskrcnn(cfg)
    state_dict = {k.replace('module.', ''): v for k, v in d['model'].items()}
    model.load_state_dict(state_dict)
    print()


if __name__ == '__main__':
    main()
