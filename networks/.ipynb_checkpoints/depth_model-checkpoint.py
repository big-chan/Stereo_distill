# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import cv2
import numpy as np
from layers import *
from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from .resnet_encoder import ResnetEncoder
from .depth_decoder import DepthDecoder
from models.cfnet import cfnet
class DepthGenerator(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self,opt):
        super(DepthGenerator, self).__init__()
        
        self.encoder=ResnetEncoder(18, True)

        self.decoder= DepthDecoder(self.encoder.num_ch_enc, [0,1,2,3])
        
    def forward(self, inputs):
        
        return self.decoder(self.encoder) #self.output
