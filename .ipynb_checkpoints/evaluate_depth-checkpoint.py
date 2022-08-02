from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks
# from psmmodels import *
from models.cfnet import cfnet
from models.gwcnet import GwcNet
from pwcnet import Network as pwcnet
cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

#         filenames = readlines(os.path.join(splits_dir, opt.eval_split, "train_files.txt"))
        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

#         encoder_dict = torch.load(encoder_path)

        dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                           opt.height, opt.width,
                                           [0,"s"], 4, is_train=False,img_ext=".png")
        dataloader = DataLoader(dataset, 8, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)
        if opt.model=="stereo":
#             depth_decoder =  cfnet(192,use_concat_volume=True)
            depth_decoder =  pwcnet()
            
        elif opt.model=="distill":
            depth_decoder =  networks.DepthGenerator()
            depth_decoder_stereo =  pwcnet()
            depth_decoder_stereo.load_state_dict(torch.load( os.path.join(opt.load_weights_folder, "depth_stereo.pth")))
            depth_decoder_stereo.cuda()
            depth_decoder_stereo.eval()
        else:
            depth_decoder =  networks.DepthGenerator()
        depth_decoder.load_state_dict(torch.load(decoder_path))

        depth_decoder.cuda()
        depth_decoder.eval()

        pred_disps = []
        pred_disps_stereos = []
        gt_disps = []
        from tqdm import tqdm
        with torch.no_grad():
            for idx,data in enumerate(tqdm(dataloader)):
                for key, ipt in data.items():
                    data[key] = ipt.cuda()
                
                input_color = data[("color", 0, 0)].cuda()
                
                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)
                if opt.model=="stereo":
                    output = depth_decoder(data[("color", 0, 0)].cuda(),data[("color", "s", 0)].cuda())
#                     pred_disp=output[-2].unsqueeze(1).cpu()[:, 0].numpy() #.shape
                    pred_disp=output.cpu()[:, 0].numpy() #.shape
#                     import pdb;pdb.set_trace()
#                     pred_disp, _ = disp_to_depth(output, opt.min_depth, opt.max_depth)
#                     pred_disp =pred_disp.cpu()[:, 0].numpy()
                elif opt.model=="distill":
                    output = depth_decoder_stereo(data[("color", 0, 0)].cuda(),data[("color", "s", 0)].cuda())
                    pred_disp_stereo=output.cpu()[:, 0].numpy() #.shape
                    
                    pred_disps_stereos.append(pred_disp_stereo)
                    output = depth_decoder(data[("color", 0, 0)].cuda())
#                     pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
#                     pred_disp =pred_disp.cpu()[:, 0].numpy()
                    pred_disp=output[("disp", 0)].cpu()[:, 0].numpy() #.shape
                    import pdb;pdb.set_trace()
                    
                else:
                    output = depth_decoder(data[("color", 0, 0)].cuda())
                    pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                    pred_disp =pred_disp.cpu()[:, 0].numpy()
                
#                 pred_disp = pred_disp.cpu()[:, 0].numpy()
                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[0].cpu().numpy(), torch.flip(pred_disp,[3])[0].cpu().numpy())
                pred_disps.append(pred_disp)
        if opt.model=="distill":
            pred_disps_stereos = np.concatenate(pred_disps_stereos)
        pred_disps = np.concatenate(pred_disps)


    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1',allow_pickle=True)["data"]

    print("-> Evaluating")

    if opt.eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    errors_stereo = []
    ratios = []

    for i in tqdm(range(pred_disps.shape[0])):

        gt_depth =gt_depths[i]

        gt_height, gt_width = gt_depth.shape[:2]
        if opt.model=="distill":
            pred_disp_stereo = pred_disps_stereos[i]
            pred_disp_stereo = cv2.resize(pred_disp_stereo, (gt_width, gt_height))
        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))


        if opt.model=="stereo":
#             pred_depth =  0.54*720.36/pred_disp/5.4
#             import pdb;pdb.set_trace()
            pred_depth =  0.54*0.58*640/(pred_disp*640)/5.4
#             pred_depth =  1/pred_disp
        elif opt.model=="distill":
#             pred_disp_stereo =  0.54*720.36/pred_disp_stereo/5.4
#             pred_depth =  1/pred_disp
            pred_disp_stereo =  0.54*0.58*640/(pred_disp_stereo*640)/5.4
            pred_depth =  0.54*0.58*640/(pred_disp*640)/5.4
            
        else:
            pred_depth =  1/pred_disp
            
        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        else:
            mask = gt_depth > 0

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]
        if opt.model=="distill":
            pred_disp_stereo =  pred_disp_stereo[mask]
            pred_disp_stereo *= opt.pred_depth_scale_factor
            pred_disp_stereo[pred_disp_stereo < MIN_DEPTH] = MIN_DEPTH
            pred_disp_stereo[pred_disp_stereo > MAX_DEPTH] = MAX_DEPTH
            errors_stereo.append(compute_errors(gt_depth, pred_disp_stereo))
        pred_depth *= opt.pred_depth_scale_factor


        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))
    if opt.model=="distill":
        mean_errors = np.array(errors_stereo).mean(0)
        print("\n-> ###################################################### Stereo")
        print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
        print("\n-> ###################################################### Mono")
    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
