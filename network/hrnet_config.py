# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN


config = CN()

# common params for NETWORK
config.MODEL = CN()
config.MODEL.NAME = 'seg_hrnet'
config.MODEL.PRETRAINED = ''
config.MODEL.ALIGN_CORNERS = True
config.MODEL.NUM_OUTPUTS = 1
config.MODEL.EXTRA = CN(new_allowed=True)


config.MODEL.OCR = CN()
config.MODEL.OCR.MID_CHANNELS = 512
config.MODEL.OCR.KEY_CHANNELS = 256
config.MODEL.OCR.DROPOUT = 0.05
config.MODEL.OCR.SCALE = 1