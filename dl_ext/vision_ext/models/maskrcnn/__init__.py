# try:
# import maskrcnn_benchmark
from .maskrcnn import R_50_C4_FasterRCNN, R_50_FPN_FasterRCNN, R_101_FPN_FasterRCNN, X_101_32x8d_FPN_FasterRCNN, \
    R_50_C4_MaskRCNN, R_50_FPN_MaskRCNN, R_101_FPN_MaskRCNN, X_101_32x8d_FPN_MaskRCNN, fbnet_chamv1a_FasterRCNN, \
    fbnet_default_FasterRCNN

# fbnet_xirb16d_MaskRCNN, fbnet_default_MaskRCNN

# except:
#     import warnings
#
#     warnings.warn(
#         'maskrcnn is not available! Please install maskrcnn_benchmark according to '
#         'https://github.com/facebookresearch/maskrcnn-benchmark')
__all__ = ["R_50_C4_FasterRCNN", "R_50_FPN_FasterRCNN", "R_101_FPN_FasterRCNN", "X_101_32x8d_FPN_FasterRCNN",
           "R_50_C4_MaskRCNN", "R_50_FPN_MaskRCNN", "R_101_FPN_MaskRCNN", "X_101_32x8d_FPN_MaskRCNN",
           # "fbnet_xirb16d_MaskRCNN", "fbnet_default_MaskRCNN"
           "fbnet_chamv1a_FasterRCNN", "fbnet_default_FasterRCNN",

           ]
