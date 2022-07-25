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

        self.decoder= DepthDecoder(self.encoder.num_ch_enc, [1,2,3])
        
        self.backproject_depth = BackprojectDepth(opt.batch_size, int(opt.height/2), int(opt.width/2))
        self.project_3d = Project3D(opt.batch_size, int(opt.height/2), int(opt.width/2)) 
        self.stereo_model=cfnet(192,use_concat_volume=True)
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.convs = OrderedDict()
        self.convs[("upconv", 2, 0)] = ConvBlock(12, 256).cuda()
        self.convs[("upconv", 2, 1)] = ConvBlock(256+64+3, 64).cuda()
        self.convs[("upconv", 1, 0)] = ConvBlock(64, 64).cuda()
        self.convs[("upconv", 1, 1)] = ConvBlock(64+64, 32).cuda()
        self.convs[("upconv", 0, 0)] = ConvBlock(32, 16).cuda()
        self.convs[("upconv", 0, 1)] = ConvBlock(16, 16).cuda()
        self.num_output_channels=1
        self.num_ch_dec2 = np.array([16, 32, 64, 128, 256])
        for s in [0,1,2]:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec2[s], self.num_output_channels).cuda()
        self.sigmoid = nn.Sigmoid()
    def forward(self, inputs):
        self.output={}
        source_scale=1
        
        T = inputs["stereo_T"].clone()
        T[:,0, 3]*=-1 
        left_image_high=inputs["color_aug", 0, 0]
        left_image_low=inputs["color_aug", 0, 1]
        outputs=self.decoder(self.encoder(left_image_high))
        self.output[("disp", 3)] =outputs[("disp",3)]
        self.output[("disp", 2)] =outputs[("disp",2)]
        disp=outputs[("disp",1)]
        _, depth= disp_to_depth(disp,0.1,80)
        self.output["depth_bf"]=depth
#         import pdb;pdb.set_trace()
        cam_points = self.backproject_depth(depth, inputs[("inv_K", source_scale)])
        pix_coords = self.project_3d(
            cam_points, inputs[("K", source_scale)], T)
        right_image_low = F.grid_sample(
                    left_image_low,
                    pix_coords,
                    padding_mode="border")
        self.output["right_image_low"]=right_image_low
#         cv2.imwrite("right_imagereal.png",inputs["color", "s", 2][0].cpu().detach().numpy().transpose((1,2,0))*255)
#         cv2.imwrite("right_image2.png",right_image_low[0].cpu().detach().numpy().transpose((1,2,0))*255)
#         cv2.imwrite("left_image2.png",left_image_low[0].cpu().detach().numpy().transpose((1,2,0))*255)
#         import pdb;pdb.set_trace()
        output_stereo=self.stereo_model(left_image_low,right_image_low)
        ################
        self.output["stereo_disp"]=output_stereo[:-1]#*0.54*720.36/5.4
        
        ################
#         cost_volume=output_stereo[-1]
#         cost_volume = self.convs[("upconv", 2, 0)](cost_volume)
# #         import pdb;pdb.set_trace()
#         cost_volume_up=upsample(cost_volume)
#         cost_cat1=torch.cat([cost_volume_up,outputs[('skip_feature', 1)],output_stereo[-2].unsqueeze(1),output_stereo[-3].unsqueeze(1),output_stereo[-4].unsqueeze(1)],dim=1)
# #         cost_cat1=cost_volume_up #torch.cat([cost_volume_up,outputs[('skip_feature', 1)]],dim=1)
#         cost_cat1 = self.convs[("upconv", 2, 1)](cost_cat1) # 64+32 64
#         self.output[("disp", 2)] = self.sigmoid(self.convs[("dispconv", 2)](cost_cat1))
#         #########################################
#         cost_cat1 = self.convs[("upconv", 1, 0)](cost_cat1) # 64 32
#         cost_cat1_up=upsample(cost_cat1)  # 32 
#         cost_cat2=torch.cat([cost_cat1_up,outputs[('skip_feature', 0)]],dim=1)
# #         cost_cat2=cost_cat1_up #torch.cat([cost_cat1_up,outputs[('skip_feature', 0)]],dim=1)
#         cost_cat2 = self.convs[("upconv", 1, 1)](cost_cat2) # 64+32 64
#         self.output[("disp", 1)] = self.sigmoid(self.convs[("dispconv", 1)](cost_cat2))
#         #########################################
#         cost_cat2 = self.convs[("upconv", 0, 0)](cost_cat2) # 64 32
#         cost_cat2_up=upsample(cost_cat2)  # 32 
#         cost_cat3 = self.convs[("upconv", 0, 1)](cost_cat2_up) # 32 16
#         self.output[("disp", 0)] = self.sigmoid(self.convs[("dispconv", 0)](cost_cat3))
        
        return self.output
