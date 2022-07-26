# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *
import cv2
import datasets
import networks
from IPython import embed
import random
import tarfile
# from psmmodels import *
from models.cfnet import cfnet
from models.gwcnet import GwcNet
from pwcnet import Network as pwcnet
from evaluate_depth import evaluate_with_train

def set_random_seed(seed):
    if seed >= 0:
        print("Set random seed@@@@@@@@@@@@@@@@@@@@")
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
class silog_loss(nn.Module):
    def __init__(self, variance_focus=0.85):
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt):
        d = torch.log(depth_est) - torch.log(depth_gt)
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0
class absrel_loss(nn.Module):
    def __init__(self):
        super(absrel_loss, self).__init__()
        self.test = 0

    def forward(self, depth_est, depth_gt):
        abs_rel = torch.mean(torch.abs(depth_gt - depth_est) / depth_gt)
        return abs_rel  
class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"
#         if not self.opt.debug:
#             wandb.init(project="ICRA2023_stereo", entity="bigchan")
#             wandb.run.name =self.opt.model_name
        #######################
        set_random_seed(42)
        if os.path.isdir(self.log_path) is False:
            os.makedirs(self.log_path)
        tar = tarfile.open( os.path.join(self.log_path, 'sources.tar'), 'w' )
#         if not self.opt.debug: 
        tar.add( 'networks' )
        tar.add( 'trainer.py' )
        tar.add( 'train.py' )
        tar.add( 'train.sh' )
        tar.add( 'utils.py' )
        tar.add( 'datasets' )
        tar.add( 'layers.py' )
        tar.add( 'options.py' )
        tar.close()
        #######################
        self.models = {}
        self.parameters_to_train = []
        self.silog_criterion= silog_loss()
        self.abs_criterion= absrel_loss()
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")
        if self.opt.model=="stereo":
#             self.models["depth"] = GwcNet(192,use_concat_volume=True)
            self.models["depth"] = pwcnet()
        elif self.opt.model=="distill":
            self.models["depth_stereo"] =  pwcnet() #GwcNet(192,use_concat_volume=True)
            self.models["depth_stereo"].to(self.device)
            self.parameters_to_train += list(self.models["depth_stereo"].parameters())

            self.models["depth"] = networks.DepthGenerator()
        else:
            self.models["depth"] = networks.DepthGenerator()
        
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        self.model_optimizer = optim.AdamW(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)
        
#         path = "sceneflow_pretraining.ckpt"
#         model_dict = self.models["depth"].state_dict()["model"]
#         pretrained_dict = torch.load(path)
#         pretrained_dict = {k.replace("module.",""): v for k, v in pretrained_dict.items() if k.replace("module.","") in model_dict}
#         model_dict.update(pretrained_dict)
#         self.models["depth"].load_state_dict(model_dict)
        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("trainv2"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size*2 if self.opt.model=="distill" else self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size*2 if self.opt.model=="distill" else self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)
            
        self.backproject_depth[-1] = BackprojectDepth(self.opt.batch_size, 375, 1242)
        self.backproject_depth[-1].to(self.device)
        self.project_3d[-1] = Project3D(self.opt.batch_size, 375, 1242)
        self.project_3d[-1].to(self.device)
        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()
                if self.opt.model=="distill":
                    eval_metric=evaluate_with_train(self.opt,self.models["depth"],self.models["depth_stereo"])
                else:
                    eval_metric=evaluate_with_train(self.opt,self.models["depth"])
#                 self.log_eval(eval_metric)
    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()
            self.batch_idx=batch_idx
            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)
                self.log("train", inputs, outputs, losses)
#                 self.val()

            self.step += 1
    def apply_disparity(self, img, disp):
        batch_size, _, height, width = img.size()

        # Original coordinates of pixels
        x_base = torch.linspace(0, 1, width).repeat(batch_size,
                    height, 1).type_as(img)
        y_base = torch.linspace(0, 1, height).repeat(batch_size,
                    width, 1).transpose(1, 2).type_as(img)

        # Apply shift in X direction
        x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
        flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
        
        # In grid_sample coordinates are assumed to be between -1 and 1
        output = F.grid_sample(img, 2*flow_field - 1, mode='bilinear',
                               padding_mode='zeros')
        
        return output,x_base + x_shifts
    def occlusion_left(self,disp_left,disp_right):
        disp_left = disp_left.unsqueeze(1) /640
        disp_right = disp_right.unsqueeze(1) /640
        disp_right_warped, flow_field_r = self.apply_disparity(-disp_right, -disp_left) #right->left 이므로 disp_left에 - 부호를 붙여줘야함
        
        occ_map_left = (torch.abs(disp_left + disp_right_warped) >= (0.1*(torch.abs(disp_left) + torch.abs(disp_right_warped))+0.5)).type(torch.LongTensor).cuda()
        
        occ_map_left[(flow_field_r.unsqueeze(1)<0)]=1
        occ_map_left[(flow_field_r.unsqueeze(1)>1)]=1
        disp_left = disp_left /(1/640)
        disp_right = disp_right /(1/640)
        
        return occ_map_left
    def occlusion_right(self,disp_left,disp_right):
        disp_left = disp_left.unsqueeze(1) /640
        disp_right = disp_right.unsqueeze(1) /640
        disp_left_warped, flow_field_l = self.apply_disparity(-disp_left, disp_right) #right->left 이므로 disp_left에 - 부호를 붙여줘야함
        
        occ_map_right = (torch.abs(disp_right + disp_left_warped) >= (0.1*(torch.abs(disp_right) + torch.abs(disp_left_warped))+0.5)).type(torch.LongTensor).cuda()

        occ_map_right[(flow_field_l.unsqueeze(1)<0)]=1
        occ_map_right[(flow_field_l.unsqueeze(1)>1)]=1
        
        disp_left = disp_left /(1/640)
        disp_right = disp_right /(1/640)
        
        return occ_map_right
    
    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            f_u=640/5.4
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            if self.opt.model=="stereo":
                ######################################
                left_color=inputs["color_aug", 0, 0].clone()
                right_color=inputs["color_aug", "s", 0].clone()
                ############################################################
                inputs["color_aug", 0, 0]=torch.cat((left_color,right_color),dim=0)
                inputs["color_aug", "s", 0]=torch.cat((right_color,left_color),dim=0)
                ##################################
                left_color=inputs["color", 0, 0].clone()
                right_color=inputs["color", "s", 0].clone()
                inputs["color", 0, 0]=torch.cat((left_color,right_color),dim=0)
                inputs["color", "s", 0]=torch.cat((right_color,left_color),dim=0)
                right_T=inputs["stereo_T"].clone()
                right_T[:,0,3]*=-1
                inputs["stereo_T"]=torch.cat((inputs["stereo_T"],right_T),dim=0)
                inputs[("inv_K",0)]=torch.cat((inputs[("inv_K",0)],inputs[("inv_K",0)]),dim=0)
                inputs[("K",0)]=torch.cat((inputs[("K",0)],inputs[("K",0)]),dim=0)
                ######################################
                outputs={}
                outputs["stereo_disp"]= [self.models["depth"](inputs["color_aug", 0, 0], inputs["color_aug", "s", 0])]#[:-1]
#                 import pdb;pdb.set_trace()
                outputs["occlusion_left"],outputs["occlusion_right"]=self.make_occ_map(outputs["stereo_disp"][-1][:self.opt.batch_size],outputs["stereo_disp"][-1][self.opt.batch_size:]) #.cuda()
                
                outputs["occlusion"]=torch.cat((outputs["occlusion_left"].cuda(),outputs["occlusion_right"].cuda()),dim=0)


            elif self.opt.model=="distill":
                left_color=inputs["color_aug", 0, 0].clone()
                right_color=inputs["color_aug", "s", 0].clone()
                ############################################################
                inputs["color_aug", 0, 0]=torch.cat((left_color,right_color),dim=0)
                inputs["color_aug", "s", 0]=torch.cat((right_color,left_color),dim=0)
                ##################################
                left_color=inputs["color", 0, 0].clone()
                right_color=inputs["color", "s", 0].clone()
                inputs["color", 0, 0]=torch.cat((left_color,right_color),dim=0)
                inputs["color", "s", 0]=torch.cat((right_color,left_color),dim=0)
                right_T=inputs["stereo_T"].clone()
                right_T[:,0,3]*=-1
                inputs["stereo_T"]=torch.cat((inputs["stereo_T"],right_T),dim=0)
                inputs[("inv_K",0)]=torch.cat((inputs[("inv_K",0)],inputs[("inv_K",0)]),dim=0)
                inputs[("K",0)]=torch.cat((inputs[("K",0)],inputs[("K",0)]),dim=0)
                outputs = self.models["depth"](inputs["color_aug", 0, 0])
                
                outputs["stereo_disp"]= [self.models["depth_stereo"](inputs["color_aug", 0, 0], inputs["color_aug", "s", 0])] #[:-1]
                outputs["occlusion_left"],outputs["occlusion_right"]=self.make_occ_map(outputs["stereo_disp"][-1][:self.opt.batch_size],outputs["stereo_disp"][-1][self.opt.batch_size:]) #.cuda()
                
                outputs["occlusion"]=torch.cat((outputs["occlusion_left"].cuda(),outputs["occlusion_right"].cuda()),dim=0)
            else:
                outputs = self.models["depth"](inputs["color_aug", 0, 0])

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))
        
        if self.opt.model=="stereo":
            self.generate_images_pred_stereo_mono1(inputs, outputs)
            losses = self.compute_losses_stereo(inputs, outputs)
        elif self.opt.model=="distill":
            self.generate_images_pred_stereo_mono1(inputs, outputs)
            losses = self.compute_losses_stereo(inputs, outputs)
            loss_distills=0
            for i in self.opt.scales:
                pred_disp_mono=outputs[("disp",i)]
                pred_disp_mono=F.interpolate(
                        pred_disp_mono, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                loss_distill=torch.log(1+torch.abs(pred_disp_mono-outputs["stereo_disp"][-1].detach()))*(1-outputs["occlusion"])
                loss_distills+=loss_distill.sum()/(1-outputs["occlusion"]).sum()
#                 loss_distills+=loss_distill.mean()
#                 import pdb;pdb.set_trace()
            losses["loss"]+=loss_distills
            losses["loss_distill_mono"]=loss_distills
#             self.generate_images_pred(inputs, outputs)
#             losses_mono = self.compute_losses(inputs, outputs)
            
#             losses["loss"]+=losses_mono["loss"]
#             mask=outputs["stereo_disp_filter_loss"]>outputs["mono_disp_filter_loss"]
#             depth=outputs["stereo_disp_filter"].clone()
#             depth[mask]=outputs["mono_disp_filter"][mask]
#             outputs["depth_label"]=depth
#             outputs["depth_label_mask"]=mask
#             losses_distill_stereo=self.compute_depth_losses_stereo(outputs)
#             losses_distill_mono=self.compute_depth_losses_mono(outputs)
    
# #             losses_distill=self.compute_disp_losses(outputs)
#             if self.epoch>=1:
# #                 import pdb;pdb.set_trace()
                
#                 losses["loss"]+=(losses_distill_stereo["loss"]+losses_distill_mono["loss"])*0.1
#                 losses["loss_distill_stereo"]=losses_distill_stereo["loss"]
#                 losses["loss_distill_mono"]=losses_distill_mono["loss"]
        else:
            self.generate_images_pred(inputs, outputs)
            losses = self.compute_losses(inputs, outputs)
        
        return outputs, losses

    def make_warp(self,img,depth,stereo_T,inv_K,K):
        T = stereo_T
        source_scale=0
        cam_points = self.backproject_depth[source_scale](depth, inv_K)
        pix_coords = self.project_3d[source_scale](cam_points, K, T)

        warped_img = F.grid_sample(img,pix_coords,padding_mode="border")
        return warped_img
    
    def make_occ_map(self, disp_left, disp_right):
        
        depth_left=0.54*0.58/(disp_left)
        depth_right=0.54*0.58/(disp_right)

        disp_right_warped, flow_field_r = self.apply_disparity(-depth_left, -disp_left) 
        disp_left_warped, flow_field_l = self.apply_disparity(-depth_right, disp_right) 

        occ_map_left = (torch.abs(depth_left + disp_right_warped) >= 3.0).type(torch.LongTensor).cuda()
        occ_map_right = (torch.abs(depth_right + disp_left_warped) >= 3.0).type(torch.LongTensor).cuda()

        occ_map_left[(flow_field_r<0).unsqueeze(1)]=1
        occ_map_left[(flow_field_r>1).unsqueeze(1)]=1
        
        
        occ_map_right[(flow_field_l<0).unsqueeze(1)]=1
        occ_map_right[(flow_field_l>1).unsqueeze(1)]=1
        ######
        return occ_map_left ,occ_map_right
    
    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)


            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()
        
    def generate_images_pred_stereo_mono1(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        st_output = outputs["stereo_disp"]
        scale=0
        for i in range(len(st_output)):
            disp = st_output[i] #.unsqueeze(1)
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                try:
                    disp = F.interpolate(
                        disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                except:
                    import pdb;pdb.set_trace()
                source_scale = 0
            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", 0, scale,"stereo",i)] = depth
            for ii, frame_id in enumerate(self.opt.frame_ids[1:]):
                T = inputs["stereo_T"]                
                T_=T[:,0,3].clone().repeat((self.opt.width,self.opt.height,1,1)).permute((3,2,1,0))*10
                outputs[("color", frame_id, scale,"stereo",i)],_=self.apply_disparity(inputs[("color", frame_id, source_scale)], T_*disp)
                
    def generate_images_pred_stereo(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        st_output = outputs["stereo_disp"]
        scale=0
        for i in range(len(st_output)):
#             disp = st_output[i].uqnsqueeze(1)
            disp = st_output[i] #.unsqueeze(1)
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                try:
                    disp = F.interpolate(
                        disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                except:
                    import pdb;pdb.set_trace()
                source_scale = 0
            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
#             depth=0.54*720.36/disp/5.4

            outputs[("depth", 0, scale,"stereo",i)] = depth

            for ii, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("color", frame_id, scale,"stereo",i)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    pix_coords,
                    padding_mode="border")


                    
    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0
            if self.opt.model=="stereo":
                depth=disp
            else:
                _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]
                    
    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0
        loss_mask_min=torch.zeros_like(outputs[("disp", 0)]).cuda()+5
        depth_min=torch.zeros_like(outputs[("disp", 0)]).cuda()+5
        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape, device=self.device) * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss
            
            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)
            

            if not self.opt.disable_automasking:

                outputs["identity_selection/{}".format(scale)] = ((idxs > identity_reprojection_loss.shape[1] - 1)).float()
            if self.opt.model=="distill" :


                mask_idxs=loss_mask_min>to_optimise.unsqueeze(1)
                
                depth_min[mask_idxs]=outputs[("depth", 0, scale)][mask_idxs]
                loss_mask_min[mask_idxs]=to_optimise.unsqueeze(1)[mask_idxs]
            loss += to_optimise.mean()
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            try:
                if self.opt.model=="stereo":
                    smooth_loss = get_smooth_loss(norm_disp, target)
                elif self.opt.model=="distill":
                    if scale!=0:
                        smooth_loss = get_smooth_loss(norm_disp, torch.cat((color,inputs[("color", "s", scale)]),dim=0))
                    else:
                        smooth_loss = get_smooth_loss(norm_disp,color)
                else:
                    smooth_loss = get_smooth_loss(norm_disp,color)
#                     import pdb;pdb.set_trace()
                    
            except:
                    import pdb;pdb.set_trace()
            
            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss
        if self.opt.model=="distill" :
            outputs["mono_disp_filter"]=depth_min
            outputs["mono_disp_filter_loss"]=loss_mask_min
        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses
    
    def compute_losses_stereo(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0
        scale=0
        st_output = outputs["stereo_disp"]

        try:
#             loss_mask_min=torch.zeros_like(st_output[0].unsqueeze(1)).cuda()+5
#             depth_min=torch.zeros_like(st_output[0].unsqueeze(1)).cuda()+5
            loss_mask_min=torch.zeros_like(st_output[0]).cuda()+5
            depth_min=torch.zeros_like(st_output[0]).cuda()+5
        except:
            import pdb;pdb.set_trace()
        for i in range(len(st_output)):
            loss = 0
            reprojection_losses = []

            source_scale = 0

#             disp = st_output[i].unsqueeze(1) #outputs[("disp", scale)]
            disp = st_output[i] #.unsqueeze(1) #outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]
            
            for frame_id in self.opt.frame_ids[1:]:
                
                pred = outputs[("color", frame_id, scale,"stereo",i)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape, device=self.device) * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss
            
            
            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)
            if self.opt.model=="distill" :


                mask_idxs=loss_mask_min>to_optimise.unsqueeze(1)
                depth_min[mask_idxs]=outputs[("depth", 0, scale,"stereo",i)][mask_idxs]
                loss_mask_min[mask_idxs]=to_optimise.unsqueeze(1)[mask_idxs]
            

            to_optimise=to_optimise*(1- outputs["occlusion"][:,0,:,:])
            loss += to_optimise.mean()
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            try:
                if self.opt.model=="stereo":
                    smooth_loss = get_smooth_loss(norm_disp, target)
                else:
                    smooth_loss = get_smooth_loss(norm_disp, color)
            except:
                    import pdb;pdb.set_trace()
            
            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            
            total_loss += loss
            losses["loss_stereo/{}".format(i)] = loss

#         total_loss += loss
        total_loss /= len(st_output)#+1
        if self.opt.model=="distill" :
            outputs["stereo_disp_filter"]=depth_min
            outputs["stereo_disp_filter_loss"]=loss_mask_min
        losses["loss"] = total_loss
        return losses
    
    def compute_disp_losses(self, inputs,mask_over_floor=None):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        losses = {}
        loss_total=0
        loss_smooth_l1=0
        loss_SSIM=0
        gt_disp=(inputs["stereo_disp_filter"]*5.4)/(0.54*720.36)
        if self.opt.using_detach:
            gt_disp=gt_disp.detach()

        for i in self.opt.scales:
            
            disp_pred = inputs[("disp", i)]
            gt_disp
            h,w=gt_disp.shape[-2:]
            disp_pred = F.interpolate(disp_pred, [h, w], mode="bilinear", align_corners=False)
            
            loss_log+=self.silog_criterion(depth_gt,depth_pred)*0.1
#             loss_smooth_l1+=F.smooth_l1_loss(disp_pred, gt_disp, size_average=True)
            loss_SSIM+=self.ssim(disp_pred, gt_disp).mean()
        loss_total+=loss_smooth_l1+loss_SSIM
        return loss_total/len(self.opt.scales)

    def compute_depth_losses_stereo(self, inputs,mask_over_floor=None):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        losses = {}
        loss_total=0
        loss_log=0
        loss_abs=0
        loss_SSIM=0
        gt_depth=inputs["depth_label"].detach() 
        for i in range(len(inputs["stereo_disp"])):
            
            depth_pred = inputs[("depth", 0, 0,"stereo",i)]
            loss_log+=self.silog_criterion(gt_depth,depth_pred)#*0.1
            loss_abs+=self.abs_criterion(gt_depth,depth_pred)
#             loss_SSIM+=self.ssim(disp_pred, gt_disp).mean()
        loss_total+=loss_log+loss_abs #+loss_SSIM
        losses["loss"] = loss_total/len(self.opt.scales)
        return losses
    def compute_depth_losses_mono(self, inputs,mask_over_floor=None):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        losses = {}
        loss_total=0
        loss_log=0
        loss_abs=0
        loss_SSIM=0
        gt_depth=inputs["depth_label"].detach()
        for i in self.opt.scales:
            
            depth_pred = inputs[("depth", 0, i)]
            mask=(1-inputs["occlusion"]).type(torch.BoolTensor).cuda()
            
            loss_log+=self.silog_criterion(gt_depth[mask],depth_pred[mask])#*0.1
            loss_abs+=self.abs_criterion(gt_depth[mask],depth_pred[mask])
        loss_total+=loss_log+loss_abs #+loss_SSIM
        losses["loss"] = loss_total/len(self.opt.scales)
        return losses
    
    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))
    def log_eval(self, evalresult):
        writer = self.writers["val"]
        for l, v in evalresult.items():
            writer.add_scalar({"{}".format(l), v, self.step})
            
    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)
        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
#             writer.add_image(
#                     "depht_gt/{}".format(j),
#                     normalize_image(inputs["depth_gt"][j]), self.step)
#             ################################################################################
            if self.opt.model=="stereo" :
                frame_id="s"
                s=0
                writer.add_image(
                            "occlusion/{}".format(j),
                            outputs["occlusion"][j][0][None, ...], self.step)
                writer.add_image(
                            "occlusion_left/{}".format(j),
                            outputs["occlusion"][j][0][None, ...], self.step)
                writer.add_image(
                            "occlusion_right/{}".format(j),
                            outputs["occlusion_right"][j][0][None, ...], self.step)
                for ii in range(len(outputs["stereo_disp"])):
                    try:
                        writer.add_image(
                                    "color_pred_stereo_{}_{}/{}".format(frame_id, s, j),
                                    outputs[("color", "s", 0,"stereo",ii)][j].data, self.step)
                    except:
                        import pdb;pdb.set_trace()
#                     writer.add_image(
#                         "disp_{}_stereo_{}/{}".format(s,ii, j),
#                         normalize_image(outputs["stereo_disp"][ii][j].unsqueeze(0)), self.step)
                    writer.add_image(
                        "disp_{}_stereo_{}/{}".format(s,ii, j),
                        normalize_image(outputs["stereo_disp"][ii][j]), self.step)
                    writer.add_image(
                        "disp_{}_stereo_{}_right/{}".format(s,ii, j),
                        normalize_image(outputs["stereo_disp"][ii][j+ self.opt.batch_size]), self.step)
#             ################################################################################
            elif self.opt.model=="distill" :
                frame_id="s"
                s=0
#                 import pdb;pdb.set_trace()
                writer.add_image(
                            "occlusion_left/{}".format(j),
                            outputs["occlusion"][j][0][None, ...], self.step)
                writer.add_image(
                            "occlusion_right/{}".format(j),
                            outputs["occlusion_right"][j][0][None, ...], self.step)
#                 writer.add_image(
#                         "disp_good_stereo/{}".format(j),
#                         normalize_image(outputs["stereo_disp_filter"][j]), self.step)
#                 writer.add_image(
#                         "disp_good_mono/{}".format(j),
#                         normalize_image(outputs["mono_disp_filter"][j]), self.step)
#                 writer.add_image(
#                         "disp_good/{}".format(j),
#                         normalize_image(outputs["depth_label"][j]), self.step)
                
                for ii in range(len(outputs["stereo_disp"])):
                    try:
                        writer.add_image(
                                    "color_pred_stereo_{}_{}/{}".format(frame_id, s, j),
                                    outputs[("color", "s", 0,"stereo",ii)][j].data, self.step)
                        
                    except:
                        import pdb;pdb.set_trace()
#                     writer.add_image(
#                         "disp_{}_stereo_{}/{}".format(s,ii, j),
#                         normalize_image(outputs["stereo_disp"][ii][j].unsqueeze(0)), self.step)
                    writer.add_image(
                        "disp_{}_stereo_{}/{}".format(s,ii, j),
                        normalize_image(outputs["stereo_disp"][ii][j]), self.step)
                    writer.add_image(
                        "disp_{}_stereo_{}_right/{}".format(s,ii, j),
                        normalize_image(outputs["stereo_disp"][ii][j+ self.opt.batch_size]), self.step)
                for s in self.opt.scales:
#                     for frame_id in self.opt.frame_ids:
#                         writer.add_image(
#                             "color_{}_{}/{}".format(frame_id, s, j),
#                             inputs[("color", frame_id, s)][j].data, self.step)
#                         if s == 0 and frame_id != 0:
#                             writer.add_image(
#                                 "color_pred_{}_{}/{}".format(frame_id, s, j),
#                                 outputs[("color", frame_id, s)][j].data, self.step)
                    writer.add_image(
                        "disp_{}/{}".format(s, j),
                        normalize_image(outputs[("disp", s)][j]), self.step)
#             ################################################################################
            else:
                for s in self.opt.scales:
                    for frame_id in self.opt.frame_ids:
                        writer.add_image(
                            "color_{}_{}/{}".format(frame_id, s, j),
                            inputs[("color", frame_id, s)][j].data, self.step)
                        if s == 0 and frame_id != 0:
                            writer.add_image(
                                "color_pred_{}_{}/{}".format(frame_id, s, j),
                                outputs[("color", frame_id, s)][j].data, self.step)

                    writer.add_image(
                        "disp_{}/{}".format(s, j),
                        normalize_image(outputs[("disp", s)][j]), self.step)
                
#                 writer.add_image("mask_gt_{}/{}".format( s,j),outputs["mask/{}".format(s)][j].float(),self.step)
                    if self.opt.predictive_mask:
                        for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                            writer.add_image(
                                "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                                outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                                self.step)

                    elif not self.opt.disable_automasking:
                        writer.add_image(
                            "automask_{}/{}".format(s, j),
                            outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")