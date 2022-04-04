import sys
import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from fabricflownet.utils import remove_occluded_knots, Flow, plot_flow

class FlowDataset(Dataset):
    def __init__(self, cfg, ids, camera_params, stage='train'):
        self.cfg = cfg
        self.camera_params = camera_params
        self.transform = T.Compose([T.ToTensor()])
        
        self.data_path = f'{cfg.base_path}/{cfg.train_name}' if stage == 'train' else f'{cfg.base_path}/{cfg.val_name}'
        self.ids = ids
        self.flow = Flow()
        self.stage = stage

    def __len__(self):
        return len(self.ids)

    def _get_uv_pre(self, depth_pre, id):
        coords_pre = np.load(f'{self.data_path}/coords/{id}_coords_before.npy')
        uv_pre_float = np.load(f'{self.data_path}/knots/{id}_knots_before.npy')
        
        # Remove occluded points and save
        depth_pre_resized = cv2.resize(depth_pre, (720, 720))
        uv_pre = remove_occluded_knots(self.camera_params, uv_pre_float, coords_pre, depth_pre_resized, 
                                        zthresh=0.005, debug_viz=self.cfg.debug_viz.remove_occlusions)
        np.save(f'{self.data_path}/knots/{id}_visibleknots_before.npy', uv_pre)
        return uv_pre

    def __getitem__(self, index):
        id = self.ids[index]
        # Load depth before action, cloth mask, knots
        depth_pre = np.load(f'{self.data_path}/rendered_images/{id}_depth_before.npy')
        cloth_mask = (depth_pre != 0).astype(float) # 200 x 200
        if not os.path.exists(f'{self.data_path}/knots/{id}_visibleknots_before.npy'):
            uv_pre = self._get_uv_pre(depth_pre, id)
        else:
            uv_pre = np.load(f'{self.data_path}/knots/{id}_visibleknots_before.npy', allow_pickle=True)
        depth_pre = self.transform(depth_pre)
        cloth_mask = self.transform(cloth_mask)

        # Load depth after action and knots
        depth_post = np.load(f'{self.data_path}/rendered_images/{id}_depth_after.npy')
        uv_post_float = np.load(f'{self.data_path}/knots/{id}_knots_after.npy')
        depth_post = self.transform(depth_post)

        # Spatial augmentation
        if self.stage == 'train' and torch.rand(1) < self.cfg.spatial_aug:
            depth_pre, depth_post, cloth_mask, uv_pre, uv_post_float = \
                self._spatial_aug(depth_pre, depth_post, cloth_mask, uv_pre, uv_post_float)

        # Remove out of bounds
        uv_pre[uv_pre < 0] = float('NaN')
        uv_pre[uv_pre >= 720] = float('NaN')

        # Get flow image
        flow_gt = self.flow.get_flow_image(uv_pre, uv_post_float)
        flow_gt = self.transform(flow_gt)

        # Get loss mask
        loss_mask = torch.zeros((flow_gt.shape[1], flow_gt.shape[2]), dtype=torch.float32)
        non_nan_idxs = np.rint(uv_pre[~np.isnan(uv_pre).any(axis=1)]/719*199).astype(int)
        loss_mask[non_nan_idxs[:, 1], non_nan_idxs[:, 0]] = 1
        loss_mask = loss_mask.unsqueeze(0)

        # Construct sample
        depths = torch.cat([depth_pre, depth_post], axis=0)
        sample = {'depths': depths, 'flow_gt': flow_gt, 'loss_mask': loss_mask, 'cloth_mask': cloth_mask}

        # Debug plotting
        if self.cfg.debug_viz.data_sample and self.stage == 'train':
            depth_pre_np = depth_pre.squeeze().numpy()
            depth_post_np = depth_post.squeeze().numpy()
            flow_gt_np = flow_gt.permute(1, 2, 0).numpy()
            loss_mask_np = loss_mask.squeeze().numpy()
            self._plot(depth_pre_np, depth_post_np, flow_gt_np, loss_mask_np)
        return sample
    
    def _aug_uv(self, uv, angle, dx, dy):
        rad = np.deg2rad(-angle)
        R = np.array([
            [np.cos(rad), -np.sin(rad)],
            [np.sin(rad), np.cos(rad)]])
        uv -= 719 / 2
        uv = np.dot(R, uv.T).T
        uv += 719 / 2
        uv[:, 0] += dx
        uv[:, 1] += dy
        return uv

    def _spatial_aug(self, depth_pre, depth_post, cloth_mask, uv_pre, uv_post_float):
        spatial_rot = self.cfg.spatial_rot
        spatial_trans = self.cfg.spatial_trans
        angle = torch.randint(low=-spatial_rot, high=spatial_rot+1, size=(1,), dtype=torch.float32)
        dx = np.random.randint(-spatial_trans, spatial_trans+1)
        dy = np.random.randint(-spatial_trans, spatial_trans+1)        
        depth_pre = TF.affine(depth_pre, angle=angle.item(), translate=(dx, dy), scale=1.0, shear=0)
        depth_post = TF.affine(depth_post, angle=angle.item(), translate=(dx, dy), scale=1.0, shear=0)
        cloth_mask = TF.affine(cloth_mask, angle=angle.item(), translate=(dx, dy), scale=1.0, shear=0)
        uv_pre = self._aug_uv(uv_pre, -angle, dx/199*719, dy/199*719)
        uv_post_float = self._aug_uv(uv_post_float, -angle, dx/199*719, dy/199*719)
        return depth_pre, depth_post, cloth_mask, uv_pre, uv_post_float

    def _plot(self, depth_pre, depth_post, flow_gt, loss_mask):
        fig, ax = plt.subplots(1, 4, figsize=(12, 3))
        ax[0].set_title("depth before")
        ax[0].imshow(depth_pre)
        ax[1].set_title("depth after")
        ax[1].imshow(depth_post)
        ax[2].set_title("ground-truth flow")
        plot_flow(ax[2], flow_gt)
        ax[3].set_title("loss mask")
        ax[3].imshow(loss_mask)
        plt.tight_layout()
        plt.show()
