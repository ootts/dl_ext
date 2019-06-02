from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.config.defaults import _C as default_cfg


def maskrcnn(cfg=None):
    if cfg is None:
        cfg = default_cfg
    return build_detection_model(cfg)
