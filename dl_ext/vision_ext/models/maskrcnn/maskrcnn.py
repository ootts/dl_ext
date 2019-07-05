from maskrcnn_benchmark.modeling.detector import build_detection_model
import os
import sys
from urllib.parse import urlparse

import torch
from torch.utils.model_zoo import _download_url_to_file
import requests
from maskrcnn_benchmark.config.defaults import _C as default_cfg


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


def load_pretrained_state_dict(url, model_dir=None, map_location=None, progress=True):
    d = loadurl(url, model_dir, map_location, progress)
    state_dict = {k.replace('module.', ''): v for k, v in d['model'].items()}
    return state_dict


def load_config(url, config_dir=None):
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
    cfg = default_cfg.clone()
    cfg.merge_from_file(cached_file)
    cfg.freeze()
    return cfg


config_url = {
    'R_50_C4_FasterRCNN': 'https://raw.githubusercontent.com/facebookresearch/maskrcnn-benchmark/master/configs/e2e_faster_rcnn_R_50_C4_1x.yaml',
    'R_50_FPN_FasterRCNN': 'https://raw.githubusercontent.com/facebookresearch/maskrcnn-benchmark/master/configs/e2e_faster_rcnn_R_50_FPN_1x.yaml',
    'R_101_FPN_FasterRCNN': 'https://raw.githubusercontent.com/facebookresearch/maskrcnn-benchmark/master/configs/e2e_faster_rcnn_R_101_FPN_1x.yaml',
    'X_101_32x8d_FPN_FasterRCNN': 'https://raw.githubusercontent.com/facebookresearch/maskrcnn-benchmark/master/configs/e2e_faster_rcnn_X_101_32x8d_FPN_1x.yaml',
    'R_50_C4_MaskRCNN': 'https://raw.githubusercontent.com/facebookresearch/maskrcnn-benchmark/master/configs/e2e_mask_rcnn_R_50_C4_1x.yaml',
    'R_50_FPN_MaskRCNN': 'https://raw.githubusercontent.com/facebookresearch/maskrcnn-benchmark/master/configs/e2e_mask_rcnn_R_50_FPN_1x.yaml',
    'R_101_FPN_MaskRCNN': 'https://raw.githubusercontent.com/facebookresearch/maskrcnn-benchmark/master/configs/e2e_mask_rcnn_R_101_FPN_1x.yaml',
    'X_101_32x8d_FPN_MaskRCNN': 'https://raw.githubusercontent.com/facebookresearch/maskrcnn-benchmark/master/configs/e2e_mask_rcnn_X_101_32x8d_FPN_1x.yaml',
    'fbnet_chamv1a_FasterRCNN': 'https://raw.githubusercontent.com/facebookresearch/maskrcnn-benchmark/master/configs/e2e_faster_rcnn_fbnet_chamv1a_600.yaml',
    'fbnet_default_FasterRCNN': 'https://raw.githubusercontent.com/facebookresearch/maskrcnn-benchmark/master/configs/e2e_faster_rcnn_fbnet_600.yaml',
    'fbnet_xirb16d_MaskRCNN': 'https://raw.githubusercontent.com/facebookresearch/maskrcnn-benchmark/master/configs/e2e_mask_rcnn_fbnet_xirb16d_dsmask.yaml',
    'fbnet_default_MaskRCNN': 'https://raw.githubusercontent.com/facebookresearch/maskrcnn-benchmark/master/configs/e2e_mask_rcnn_fbnet_xirb16d_dsmask_600.yaml',
}
pretrained_model_url = {
    'R_50_C4_FasterRCNN': 'https://download.pytorch.org/models/maskrcnn/e2e_faster_rcnn_R_50_C4_1x.pth',
    'R_50_FPN_FasterRCNN': 'https://download.pytorch.org/models/maskrcnn/e2e_faster_rcnn_R_50_FPN_1x.pth',
    'R_101_FPN_FasterRCNN': 'https://download.pytorch.org/models/maskrcnn/e2e_faster_rcnn_R_101_FPN_1x.pth',
    'X_101_32x8d_FPN_FasterRCNN': 'https://download.pytorch.org/models/maskrcnn/e2e_faster_rcnn_X_101_32x8d_FPN_1x.pth',
    'R_50_C4_MaskRCNN': 'https://download.pytorch.org/models/maskrcnn/e2e_mask_rcnn_R_50_C4_1x.pth',
    'R_50_FPN_MaskRCNN': 'https://download.pytorch.org/models/maskrcnn/e2e_mask_rcnn_R_50_FPN_1x.pth',
    'R_101_FPN_MaskRCNN': 'https://download.pytorch.org/models/maskrcnn/e2e_mask_rcnn_R_101_FPN_1x.pth',
    'X_101_32x8d_FPN_MaskRCNN': 'https://download.pytorch.org/models/maskrcnn/e2e_mask_rcnn_X_101_32x8d_FPN_1x.pth',
    'fbnet_chamv1a_FasterRCNN': 'https://download.pytorch.org/models/maskrcnn/e2e_faster_rcnn_fbnet_chamv1a_600.pth',
    'fbnet_default_FasterRCNN': 'https://download.pytorch.org/models/maskrcnn/e2e_faster_rcnn_fbnet_600.pth',
    'fbnet_xirb16d_MaskRCNN': 'https://download.pytorch.org/models/maskrcnn/e2e_mask_rcnn_fbnet_xirb16d_dsmask.pth',
    'fbnet_default_MaskRCNN': 'https://download.pytorch.org/models/maskrcnn/e2e_mask_rcnn_fbnet_600.pth',
}


def MaskRCNN(cfg=None):
    if cfg is None:
        cfg = default_cfg
    return build_detection_model(cfg)


def _build_maskrcnn(model_name, pretrained):
    cfg = load_config(config_url[model_name])
    model = MaskRCNN(cfg)
    if pretrained:
        model.load_state_dict(load_pretrained_state_dict(pretrained_model_url[model_name]))
    return model


def R_50_C4_FasterRCNN(pretrained=False):
    return _build_maskrcnn('R_50_C4_FasterRCNN', pretrained)


def R_50_FPN_FasterRCNN(pretrained=False):
    return _build_maskrcnn('R_50_FPN_FasterRCNN', pretrained)


def R_101_FPN_FasterRCNN(pretrained=False):
    return _build_maskrcnn('R_101_FPN_FasterRCNN', pretrained)


def X_101_32x8d_FPN_FasterRCNN(pretrained=False):
    return _build_maskrcnn('X_101_32x8d_FPN_FasterRCNN', pretrained)


def R_50_C4_MaskRCNN(pretrained=False):
    return _build_maskrcnn('R_50_C4_MaskRCNN', pretrained)


def R_50_FPN_MaskRCNN(pretrained=False):
    return _build_maskrcnn('R_50_FPN_MaskRCNN', pretrained)


def R_101_FPN_MaskRCNN(pretrained=False):
    return _build_maskrcnn('R_101_FPN_MaskRCNN', pretrained)


def X_101_32x8d_FPN_MaskRCNN(pretrained=False):
    return _build_maskrcnn('X_101_32x8d_FPN_MaskRCNN', pretrained)


def fbnet_chamv1a_FasterRCNN(pretrained=False):
    return _build_maskrcnn('fbnet_chamv1a_FasterRCNN', pretrained)


def fbnet_default_FasterRCNN(pretrained=False):
    return _build_maskrcnn('fbnet_default_FasterRCNN', pretrained)


def fbnet_xirb16d_MaskRCNN(pretrained=False):
    return _build_maskrcnn('fbnet_xirb16d_MaskRCNN', pretrained)


def fbnet_default_MaskRCNN(pretrained=False):
    return _build_maskrcnn('fbnet_default_MaskRCNN', pretrained)
