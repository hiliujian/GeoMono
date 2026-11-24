# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import copy

import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json
import cv2

import gcidc

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from networks import mpvit, pose_decoder, resnet_encoder

# The two helpers below are copied verbatim from evaluate_depth.py so that the
# in-training evaluation and the standalone script compute the same numbers.
def eval_compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def eval_batch_post_process_disparity(l_disp, r_disp):
    """Fuse flipped and unflipped disparities with the Monodepth border weighting."""
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.opt.depth_supervision = "gcidc_three_frame_contact_cuda"
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)
        self.splits_dir = os.path.join(os.path.dirname(__file__), "splits")
        self.eval_available = True
        self.eval_flops_cache = None

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")      

        self.models["encoder"] = mpvit.mpvit_small()
        self.models["encoder"].num_ch_enc = [64,128,216,288,288]
        self.models["encoder"].to(self.device)
 
        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                pose_layers = int(getattr(self.opt, "pose_num_layers", 18))
                self.models["pose_encoder"] = resnet_encoder.ResnetEncoder(
                    pose_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)

                self.models["pose_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = pose_decoder.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())


        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Predict a separate mask for each source frame with the depth decoder architecture.
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        # Keep parameter groups in checkpoint order.
        self.params = [
            {"params": self.parameters_to_train, "lr": 1e-4},
            {"params": list(self.models["encoder"].parameters()), "lr": self.opt.learning_rate},
        ]
        self.model_optimizer = optim.AdamW(self.params)
        self.model_lr_scheduler = optim.lr_scheduler.ExponentialLR(self.model_optimizer, gamma=0.9)

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

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if getattr(self.opt, 'png', False) else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext, load_depth=False)
        from gcdid.dataset import MotionDataset, motion_collate
        cache = self.opt.gcdid_cache
        train_dataset = MotionDataset(train_dataset, cache, self.opt.gcidc_road_cache, self.opt.gcidc_flow_cache)
        print("GCDID cache:", cache)
        print("Depth supervision: GCIDC (CUDA)")
        print("GCIDC road cache:", self.opt.gcidc_road_cache)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True,
            collate_fn=motion_collate)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext, load_depth=False)
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
        if not self.opt.disable_auto_blur:
            assert self.opt.receptive_field_of_auto_blur % 2 == 1, \
                'receptive_field_of_auto_blur should be an odd number'
            self.auto_blur = networks.AutoBlurModule(
                self.opt.receptive_field_of_auto_blur,
                hf_pixel_thresh=self.opt.hf_pixel_thresh,
                hf_area_percent_thresh=self.opt.hf_area_percent_thresh,
            )
            self.auto_blur.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

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

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):
            
            before_op_time = time.time()
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

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

        self.maybe_evaluate_kitti()

        self.model_lr_scheduler.step()

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            if torch.is_tensor(ipt):
                inputs[key] = ipt.to(self.device)
        if 'gcidc_cache_sizes' in inputs:
            from gcdid.dataset import prepare_gcidc_cells
            prepare_gcidc_cells(inputs)

        if not self.opt.disable_auto_blur:
            for scale in self.opt.scales:
                for f_i in self.opt.frame_ids:
                    inputs[('raw_color', f_i, scale)] = inputs[('color', f_i, scale)]
                    inputs[('color', f_i, scale)] = self.auto_blur(
                        inputs[('color', f_i, scale)])

        if self.opt.pose_model_type == "shared":
            # Encode all frames when depth and pose share an encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            features = self.models["encoder"](inputs["color_aug", 0, 0])
            outputs = self.models["depth"](features)


        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        self.generate_images_pred(inputs, outputs)
        # Cached motion selection is already aligned with the augmented image.
        losses = self.compute_losses(inputs, outputs)
        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
          
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)

                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = next(self.val_iter)

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]

            if self.opt.native_scale_reprojection:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for frame_id in self.opt.frame_ids[1:]:

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
                    padding_mode="border",
                    align_corners=True)

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

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.native_scale_reprojection:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)] if self.opt.disable_ambiguity_mask \
                else inputs[('raw_color', 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:     # [0,-1,1]
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
                if not self.opt.native_scale_reprojection:
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

            if not self.opt.disable_ambiguity_mask:
                ambiguity_mask = self.compute_ambiguity_mask(
                    inputs, outputs, reprojection_loss, scale)

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape, device=self.device) * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined[:, 0]
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_ambiguity_mask:
                to_optimise = to_optimise * ambiguity_mask

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            if "dynamic_mask" in inputs:
                dynamic_mask = inputs["dynamic_mask"]
                if dynamic_mask.shape[-2:] != to_optimise.shape[-2:]:
                    dynamic_mask = F.interpolate(
                        dynamic_mask.float().unsqueeze(1),
                        size=to_optimise.shape[-2:],
                        mode="nearest").squeeze(1) > 0.5
                # Average each image over its own unmasked pixels, then sum the batch.
                valid = (~dynamic_mask).to(to_optimise.dtype)          # [B,H,W]
                per_image_valid = valid.sum(dim=(1, 2)).clamp_min(1)   # [B]
                per_image_loss = (to_optimise * valid).sum(dim=(1, 2)) / per_image_valid  # [B]
                photometric_loss = per_image_loss.sum()
                if scale == 0:
                    active = torch.ones_like(dynamic_mask)
                    if not self.opt.disable_automasking:
                        active = idxs >= identity_reprojection_loss.shape[1]
                    if not self.opt.disable_ambiguity_mask:
                        active = active & ambiguity_mask.bool()
                    outputs["gcdid/photo_raw"] = to_optimise.detach()
                    outputs["gcdid/photo_excluded"] = dynamic_mask.detach()
                    outputs["gcdid/photo_active"] = active.detach()
                    outputs["gcdid/photo_removed"] = (active & dynamic_mask).detach()
                    losses["gcdid/selected_pixel_fraction"] = dynamic_mask.float().mean().detach()
                    losses["gcdid/removed_active_photo_fraction"] = ((active & dynamic_mask).sum().float() / active.sum().clamp_min(1)).detach()
                    losses["gcdid/retained_active_photo_pixels"] = (active & ~dynamic_mask).sum().float().detach()
                    if "gcdid_counts" in inputs:
                        for i,name in enumerate(["candidate_instances", "selected_instances"]):
                            losses["gcdid/" + name] = inputs["gcdid_counts"][:,i].sum().float().detach()
            else:
                photometric_loss = to_optimise.mean()
            loss = loss + photometric_loss
            losses["loss/photometric/{}".format(scale)] = photometric_loss.detach()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            smooth_term = self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            loss = loss + smooth_term
            losses["loss/smooth/{}".format(scale)] = smooth_term.detach()

            if scale == 0 and "gcidc_cells" in inputs:
                # Evaluate GCIDC at scale zero, the resolution of the cached instance grid.
                gcidc_term = self.compute_gcidc_loss(
                    inputs, outputs[("depth", 0, scale)], outputs)
                loss = loss + gcidc_term
                losses["loss/gcidc/{}".format(scale)] = gcidc_term
            total_loss = total_loss + loss
            losses["loss/{}".format(scale)] = loss

        total_loss = total_loss/self.num_scales
        losses["loss"] = total_loss
        return losses

    # ------------------------------------------ Ground Contact Inverse Depth Constraint (GCIDC)
    def compute_gcidc_loss(self, inputs, depth, outputs=None):
        """Sum per-image GCIDC losses from detached targets.

        Each image averages its valid target pixels. Images without targets
        contribute zero.
        """
        depth = depth[:, 0] if depth.dim() == 4 else depth
        if "dynamic_mask" not in inputs:
            return depth.sum() * 0.0
        B, H, W = depth.shape
        if inputs["dynamic_mask"].shape != depth.shape:
            raise ValueError("GCIDC mask must match the current training resolution")
        # The first sample supplies the batch eligibility flag and spatial camera rays.
        moving = inputs["dynamic_mask"].bool() & inputs["gcidc_batch_enabled"]
        K = inputs[("K", 0)][:1, :3, :3].to(dtype=depth.dtype).expand(B, -1, -1)
        road = inputs["gcidc_road"].bool()
        road = torch.where(inputs["gcidc_flipped"][:, None, None], road.flip(-1), road)
        if outputs is not None and all(("cam_T_cam",0,s) in outputs for s in (-1,1)):
            poses=torch.stack([outputs[("cam_T_cam",0,s)].detach() for s in (-1,1)], dim=1)
        else:
            # No temporal pose evidence in single-frame or stereo-only calls.
            poses=depth.new_zeros((B,2,4,4))
        comp = gcidc.build_targets(
            depth.detach(), inputs["gcdid_grid"].long(), moving, K,
            inputs["gcidc_cells"], inputs["gcidc_cell_valid"],
            inputs["gcidc_batch_index"], road, inputs["gcidc_flow"], inputs["gcidc_intrinsics"],
            poses,
            inputs["gcidc_flipped"])
        target, mask = comp["u_t"], comp["mask"]
        per_image = gcidc.loss_per_image(depth, target, mask)
        if outputs is not None:
            # TensorBoard reuses the exact training targets; no second fit.
            outputs["gcidc/u_t"] = target
            outputs["gcidc/mask"] = mask
            outputs["gcidc/original_mask"] = comp["original_mask"]
        return per_image.sum()

    @staticmethod
    def extract_ambiguity(ipt):
        grad_r = ipt[:, :, :, :-1] - ipt[:, :, :, 1:]
        grad_b = ipt[:, :, :-1, :] - ipt[:, :, 1:, :]

        grad_l = F.pad(grad_r, (1, 0))
        grad_r = F.pad(grad_r, (0, 1))

        grad_t = F.pad(grad_b, (0, 0, 1, 0))
        grad_b = F.pad(grad_b, (0, 0, 0, 1))

        is_u_same_sign = ((grad_l * grad_r) > 0).any(dim=1, keepdim=True)
        is_v_same_sign = ((grad_t * grad_b) > 0).any(dim=1, keepdim=True)
        is_same_sign = torch.logical_or(is_u_same_sign, is_v_same_sign)

        grad_u = (grad_l.abs() + grad_r.abs()).sum(1, keepdim=True) / 2
        grad_v = (grad_t.abs() + grad_b.abs()).sum(1, keepdim=True) / 2
        grad = torch.sqrt(grad_u ** 2 + grad_v ** 2)

        ambiguity = grad * is_same_sign
        return ambiguity

    def compute_ambiguity_mask(self, inputs, outputs,
                               reprojection_loss, scale):
        src_scale = scale if self.opt.native_scale_reprojection else 0
        min_reproj, min_idx = torch.min(reprojection_loss, dim=1)

        target_ambiguity = self.extract_ambiguity(inputs[("color", 0, src_scale)])

        reproj_ambiguities = []
        for f_i in self.opt.frame_ids[1:]:
            src_ambiguity = self.extract_ambiguity(inputs[("color", f_i, src_scale)])

            reproj_ambiguity = F.grid_sample(
                src_ambiguity, outputs[("sample", f_i, scale)],
                padding_mode="border", align_corners=True)
            reproj_ambiguities.append(reproj_ambiguity)

        reproj_ambiguities = torch.cat(reproj_ambiguities, dim=1)
        reproj_ambiguity = torch.gather(reproj_ambiguities, 1, min_idx.unsqueeze(1))

        synthetic_ambiguity, _ = torch.cat(
            [target_ambiguity, reproj_ambiguity], dim=1).max(dim=1)

        if self.opt.ambiguity_by_negative_exponential:
            ambiguity_mask = torch.exp(-self.opt.negative_exponential_coefficient
                                       * synthetic_ambiguity)
        else:
            ambiguity_mask = synthetic_ambiguity < self.opt.ambiguity_thresh
        return ambiguity_mask

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def maybe_evaluate_kitti(self):
        """Evaluate the current weights after every single epoch

        No cadence knob: one epoch, one evaluation on the full test split.  A
        failing evaluation must never kill a long run, so the exception is
        reported once and the evaluation is switched off for the rest of it.
        """
        if getattr(self.opt, "no_eval", False):
            return

        if not self.eval_available:
            return

        try:
            self.evaluate_kitti()
        except Exception:
            import traceback
            traceback.print_exc()
            self.eval_available = False

    def evaluate_kitti(self):
        """Evaluate the current weights on the eigen test split, in process

        This is the same protocol as LiteMonoMov/trainer.py and
        evaluate_depth.py, so a number seen mid-training is directly comparable
        with a standalone evaluation of the saved weights:

          * test split  splits/<eval_split>/test_files.txt  (697 images for the
            eigen split)
          * ground truth splits/<eval_split>/gt_depths.npz  (native resolution)
          * input ("color", 0, 0), never the augmented ("color_aug", 0, 0)
          * post-processing exactly as in evaluate_depth.py: the batch is
            doubled with a horizontal flip and the two disparities are blended
            with batch_post_process_disparity, not a plain average
          * per-image Garg/Eigen crop, per-image median scaling, clamping to
            [1e-3, 80], then per-image metrics averaged (compute_errors)

        Evaluation uses the encoder and depth decoder. GCDID and GCIDC operate
        during training. The metrics are computed per image, so the evaluation batch size
        only trades speed for memory: it is kept at 4 because this runs inside
        the training process, where the optimizer state is still resident and
        the GPUs are shared with other jobs.
        """
        split = self.opt.eval_split
        test_path = os.path.join(self.splits_dir, split, "test_files.txt")
        gt_path = os.path.join(self.splits_dir, split, "gt_depths.npz")

        if not (os.path.isfile(test_path) and os.path.isfile(gt_path)):
            print("eval: skipped, {} or {} is missing".format(
                test_path, gt_path))
            self.eval_available = False
            return {}

        filenames = readlines(test_path)

        # img_ext must match the training loader: this dataset defaults to
        # '.png' while the local KITTI copy only has '.jpg'
        img_ext = '.png' if getattr(self.opt, 'png', False) else '.jpg'
        dataset = datasets.KITTIRAWDataset(
            self.opt.data_path, filenames, self.opt.height, self.opt.width,
            [0], 4, is_train=False, img_ext=img_ext)
        dataloader = DataLoader(
            dataset, 4, shuffle=False, num_workers=self.opt.num_workers,
            pin_memory=True, drop_last=False)

        pred_disps = []

        print("-> Computing predictions with size {}x{}".format(
            self.opt.width, self.opt.height))

        was_training = self.models["depth"].training
        self.set_eval()
        try:
            if self.device.type == "cuda":
                # give the caching allocator a clean slate: the training loop
                # keeps its own blocks alive and this pass needs a large
                # contiguous one
                torch.cuda.empty_cache()
            with torch.no_grad():
                for data in dataloader:
                    input_color = data[("color", 0, 0)].to(self.device)

                    if self.opt.post_process:
                        # post-processed results require two forward passes
                        input_color = torch.cat(
                            (input_color, torch.flip(input_color, [3])), 0)

                    output = self.models["depth"](
                        self.models["encoder"](input_color))

                    pred_disp, _ = disp_to_depth(
                        output[("disp", 0)], self.opt.min_depth,
                        self.opt.max_depth)
                    pred_disp = pred_disp.cpu()[:, 0].numpy()

                    if self.opt.post_process:
                        N = pred_disp.shape[0] // 2
                        pred_disp = eval_batch_post_process_disparity(
                            pred_disp[:N], pred_disp[N:, :, ::-1])

                    pred_disps.append(pred_disp)
        finally:
            # the training loop continues after this call
            if was_training:
                self.set_train()

        pred_disps = np.concatenate(pred_disps)

        gt_depths = np.load(
            gt_path, fix_imports=True, encoding='latin1',
            allow_pickle=True)["data"]

        MIN_DEPTH = 1e-3
        MAX_DEPTH = 80

        print("-> Evaluating")
        print("   Mono evaluation - using median scaling")

        errors = []
        ratios = []

        for i in range(pred_disps.shape[0]):
            gt_depth = gt_depths[i]
            gt_height, gt_width = gt_depth.shape[:2]

            pred_disp = pred_disps[i]
            pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
            pred_depth = 1 / pred_disp

            if split == "eigen":
                mask = np.logical_and(
                    gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

                crop = np.array([0.40810811 * gt_height,
                                 0.99189189 * gt_height,
                                 0.03594771 * gt_width,
                                 0.96405229 * gt_width]).astype(np.int32)
                crop_mask = np.zeros(mask.shape)
                crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                mask = np.logical_and(mask, crop_mask)
            else:
                mask = gt_depth > 0

            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]

            pred_depth *= self.opt.pred_depth_scale_factor
            if not self.opt.disable_median_scaling:
                ratio = np.median(gt_depth) / np.median(pred_depth)
                ratios.append(ratio)
                pred_depth *= ratio

            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
            errors.append(eval_compute_errors(gt_depth, pred_depth))

        if not self.opt.disable_median_scaling:
            ratios = np.array(ratios)
            med = np.median(ratios)
            print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(
                med, np.std(ratios / med)))

        mean_errors = np.array(errors).mean(0)

        print("\n  " + ("{:>8} | " * 7).format(
            "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
        print("\n  " + self.eval_flops())
        print("\n-> Done!")

        metrics = dict(zip(
            ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"],
            [float(v) for v in mean_errors]))
        self.log_kitti_eval(split, len(filenames), metrics, ratios)
        return metrics

    def eval_flops(self):
        """Model complexity line, computed once and cached

        evaluate_depth.py prints this after every prediction pass.  thop is an
        optional dependency here and profiling mpvit can fail on its custom
        ops, so any failure degrades to n/a instead of killing the run.
        """
        if self.eval_flops_cache is not None:
            return self.eval_flops_cache

        try:
            from thop import clever_format, profile

            # thop registers total_ops / total_params as *buffers* on every
            # submodule it touches, and buffers end up in state_dict().  Profiling
            # the live models therefore polluted save_model(): the epoch-0
            # weights_0/depth.pth got 534 extra keys (Unexpected key(s):
            # total_ops, total_params, ...) and any strict reload of that
            # checkpoint failed. Profile throw-away copies instead so the
            # training weights stay clean.
            # NOTE: MPViT overrides train() without returning self, so eval()
            # returns None -- never chain ".eval()" onto a copy of the encoder.
            sample = torch.zeros(
                1, 3, self.opt.height, self.opt.width, device=self.device)
            enc_ref = copy.deepcopy(self.models["encoder"])
            dep_ref = copy.deepcopy(self.models["depth"])
            enc_ref.eval()
            dep_ref.eval()
            with torch.no_grad():
                features = enc_ref(sample)
                flops_e, params_e = profile(
                    enc_ref, inputs=(sample,), verbose=False)
                flops_d, params_d = profile(
                    dep_ref, inputs=(features,), verbose=False)
            del enc_ref, dep_ref, features

            flops, params = clever_format(
                [flops_e + flops_d, params_e + params_d], "%.3f")
            flops_e, params_e = clever_format([flops_e, params_e], "%.3f")
            flops_d, params_d = clever_format([flops_d, params_d], "%.3f")
            self.eval_flops_cache = (
                "flops: {0}, params: {1}, flops_e: {2}, params_e:{3}, "
                "flops_d:{4}, params_d:{5}".format(
                    flops, params, flops_e, params_e, flops_d, params_d))
        except Exception as e:
            self.eval_flops_cache = (
                "flops: n/a, params: n/a ({})".format(type(e).__name__))

        return self.eval_flops_cache

    def log_kitti_eval(self, split, num_images, metrics, ratios=None):
        """Persist the evaluation quietly to tensorboard, jsonl and csv

        The terminal output is produced by evaluate_kitti itself and matches
        evaluate_depth.py, so nothing is printed here.  Numbers go to
        eval_metrics.jsonl / eval_metrics.csv instead of results.txt.
        """
        writer = self.writers["val"]
        for name, value in metrics.items():
            writer.add_scalar("eval/{}".format(name), value, self.epoch)
        writer.flush()

        if ratios is not None and len(ratios) > 0:
            ratios = np.array(ratios)
            median_ratio = np.median(ratios)
            writer.add_scalar("eval/scaling_median", median_ratio, self.epoch)

        record = {"epoch": self.epoch, "split": split,
                  "num_images": num_images, "time": time.time()}
        record.update(metrics)

        with open(os.path.join(self.log_path, "eval_metrics.jsonl"), "a") as f:
            f.write(json.dumps(record) + "\n")

        csv_path = os.path.join(self.log_path, "eval_metrics.csv")
        write_header = not os.path.isfile(csv_path)
        with open(csv_path, "a") as f:
            if write_header:
                f.write("epoch," + ",".join(metrics.keys()) + "\n")
            values = ",".join("{:.6f}".format(v) for v in metrics.values())
            f.write("{},{}\n".format(self.epoch, values))

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

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        from gcdid.visualize import log_gcidc_panel, log_panels, panel_order

        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)
        if mode == "train":
            # Monitor saturation using the existing training forward pass.
            z = outputs[("depth", 0, 0)].detach().flatten(1)
            q10, q90 = torch.quantile(z, z.new_tensor([0.1, 0.9]), dim=1)
            writer.add_scalar("train_depth/near_min_fraction",
                              (z <= 1.01 * self.opt.min_depth).float().mean(), self.step)
            writer.add_scalar("train_depth/median_p90_p10", (q90 / q10).median(), self.step)

        order = panel_order(inputs)
        for slot, j in enumerate(order):
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    key = ("color", frame_id, s)
                    rgb = inputs.get(("raw_color", frame_id, s), inputs[key]) if frame_id == 0 else inputs[key]
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, slot),
                        rgb[j].detach(), self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, slot),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, slot),
                    normalize_image(outputs[("disp", s)][j]), self.step)
                if s<3:
                    writer.add_image(
                        "delta_{}/{}".format(s, slot),
                        colormap(torch.abs(outputs[("delta", s)][j]).mean(dim=0,keepdim=True)), self.step)
                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, slot),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, slot),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

        log_panels(writer, inputs, outputs, self.step, order=order)
        if "dynamic_mask" in inputs:
            for slot, sample_index in enumerate(order):
                log_gcidc_panel(writer, inputs, outputs, sample_index, slot, self.step)

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
            # Drop thop profiler buffers if a model was ever profiled: they are
            # not weights, and any strict load_state_dict() on the reload path
            # would reject them as unexpected keys.
            to_save = {
                k: v for k, v in model.state_dict().items()
                if not (k.endswith("total_ops") or k.endswith("total_params"))}
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
