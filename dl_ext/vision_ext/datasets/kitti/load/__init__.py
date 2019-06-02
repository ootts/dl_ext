"""
This file contains loading functions for kitti object dataset.
To use these functions, you should have kitti dataset arranging like following:
kitti
    --object
        --training
            --image_2
            --image_3
            --calib
            --label_2
        --testing
            ...
Currently, these functions support calib,image_2,image_3 and label_2
Note that these functions support overloading by dispatch.
Parameter explanation:
    imgid: int or str, for example: 2333 or '002333'
    kitti_root: write to kitti. not to object.

"""
from .calib import *
from .image_2 import *
from .image_3 import *
from .label_2 import *
