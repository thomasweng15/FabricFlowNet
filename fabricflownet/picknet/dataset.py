import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from PIL import Image
from fabricflownet.utils import remove_occluded_knots, flow_affinewarp, Flow, plot_flow
import cv2
from copy import deepcopy
import os.path as osp
import os
import matplotlib.pyplot as plt
from fabricflownet.flownet.models import FlowNet

class PickNetDataset(Dataset):
    def __init__(self, camera_params, config, ids, mode='train', pick_pt=True):
        self.cfg = config
        self.mode = mode
        self.ids = ids
        self.camera_params = camera_params
        self.pick_pt = pick_pt

        if mode == 'train':
            self.data_path = f"{self.cfg.base_path}/{self.cfg.train_name}"
        else:
            self.data_path = f"{self.cfg.base_path}/{self.cfg.val_name}"

        if self.cfg.flow == 'gt':
            self.gt_flow = True
            self.flow = Flow()
            if not osp.exists(osp.join(self.data_path, "flow_gt")):
                os.mkdir(osp.join(self.data_path, "flow_gt"))
        else:
            self.gt_flow = False
            self.flow = FlowNet(input_channels=2)
            checkpt = torch.load(self.cfg.flow)
            self.flow.load_state_dict(checkpt['state_dict'])
            self.flow.eval()
            if not osp.exists(osp.join(self.data_path, "flow_pred")):
                os.mkdir(osp.join(self.data_path, "flow_pred"))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id = self.ids[index]
        action = np.load(f'{self.data_path}/actions/{id}_action.npy')
        pick_uv1 = action[0]
        place_uv1 = action[1]
        pick_uv2 = action[2]
        place_uv2 = action[3]
        
        depth_post = np.load(f'{self.data_path}/rendered_images/{id}_depth_after.npy')
        depth_pre = np.load(f'{self.data_path}/rendered_images/{id}_depth_before.npy')

        if self.gt_flow:
            uv_post_float = np.load(f'{self.data_path}/knots/{id}_knots_after.npy')

            # Compute and save flow if it does not exist already
            if not osp.exists(osp.join(self.data_path, "flow_gt", f"{id}_flow.npy")):
                coords_pre = np.load(f'{self.data_path}/coords/{id}_coords_before.npy')
                uv_pre_float = np.load(f'{self.data_path}/knots/{id}_knots_before.npy')

                # Remove occlusions
                depth_pre_rs = cv2.resize(depth_pre, (720, 720))
                uv_pre, _ = remove_occluded_knots(self.camera_params, uv_pre_float, coords_pre, depth_pre_rs, 
                    zthresh=0.005, debug_viz=self.cfg.debug_viz.remove_occlusions)

                # Remove out of bounds
                uv_pre[uv_pre < 0] = float('NaN')
                uv_pre[uv_pre >= 720] = float('NaN')

                # Get flow image
                flow_im = self.flow.get_image(uv_pre, uv_post_float)

                # Save the flow
                np.save(osp.join(self.data_path, "flow_gt", f"{id}_flow.npy"), flow_im)
            else:
                flow_im = np.load(osp.join(self.data_path, "flow_gt", f"{id}_flow.npy"))
        else:
            if not osp.exists(osp.join(self.data_path, "flow_pred", f"{id}_flow.npy")):
                inp = torch.stack([torch.FloatTensor(depth_pre), torch.FloatTensor(depth_post)]).unsqueeze(0)
                flow_im = self.flow(inp)
                flow_im = flow_im.squeeze().cpu()
                np.save(osp.join(self.data_path, "flow_pred", f"{id}_flow.npy"), flow_im.detach().numpy())
            else:
                flow_im = np.load(osp.join(self.data_path, "flow_pred", f"{id}_flow.npy"), allow_pickle=True)

        depth_pre = torch.FloatTensor(depth_pre).unsqueeze(0)
        depth_post = torch.FloatTensor(depth_post).unsqueeze(0)

        if self.gt_flow:
            flow_im = flow_im.transpose([2,0,1])

        if not isinstance(flow_im, torch.Tensor):
            flow_im = torch.FloatTensor(flow_im)

        # mask flow
        flow_im[0,:,:][depth_pre[0] == 0] = 0
        flow_im[1,:,:][depth_pre[0] == 0] = 0

        if self.cfg.augment:
            angle = np.random.randint(-5, 6)
            dx = np.random.randint(-5, 6)
            dy = np.random.randint(-5, 6)
            depth_pre, depth_post, pick_uv1, pick_uv2, place_uv1, place_uv2 \
                = self.spatial_aug(depth_pre, depth_post, pick_uv1, pick_uv2, place_uv1, place_uv2, angle, dx, dy)
            flow_im = flow_im.permute(1, 2, 0).detach().numpy()
            flow_im = flow_affinewarp(flow_im, -angle, 0, 0)
            flow_im = torch.FloatTensor(flow_im).permute(2, 0, 1)

        if self.pick_pt:
            uv1, uv2 = pick_uv1, pick_uv2
        else:
            uv1, uv2 = place_uv1, place_uv2

        if self.cfg.debug_viz.data_sample:
            self.plot(depth_pre, depth_post, flow_im, uv1, uv2)

        return depth_pre, depth_post, flow_im, uv1, uv2

    def spatial_aug(self, depth_pre, depth_post, pick_uv1, pick_uv2, place_uv1, place_uv2, angle, dx, dy):
        depth_pre = TF.affine(depth_pre, angle=angle, translate=(dx, dy), scale=1.0, shear=0)
        depth_post = TF.affine(depth_post, angle=angle, translate=(dx, dy), scale=1.0, shear=0)
        pick_uv1 = self.aug_uv(pick_uv1.astype(np.float64)[None,:], -angle, dx, dy, size=199)
        pick_uv1 = pick_uv1.squeeze().astype(int)
        pick_uv2 = self.aug_uv(pick_uv2.astype(np.float64)[None,:], -angle, dx, dy, size=199)
        pick_uv2 = pick_uv2.squeeze().astype(int)
        place_uv1 = self.aug_uv(place_uv1.astype(np.float64)[None,:], -angle, dx, dy, size=199)
        place_uv1 = place_uv1.squeeze().astype(int)
        place_uv2 = self.aug_uv(place_uv2.astype(np.float64)[None,:], -angle, dx, dy, size=199)
        place_uv2 = place_uv2.squeeze().astype(int)
        return depth_pre, depth_post, pick_uv1, pick_uv2, place_uv1, place_uv2

    def aug_uv(self, uv, angle, dx, dy, size=719):
        rad = np.deg2rad(-angle)
        R = np.array([
            [np.cos(rad), -np.sin(rad)],
            [np.sin(rad), np.cos(rad)]])
        uv -= size / 2
        uv = np.dot(R, uv.T).T
        uv += size / 2
        uv[:, 0] += dx
        uv[:, 1] += dy
        uv = np.clip(uv, 0, size)
        return uv

    def plot(self, depth_pre, depth_post, flow, uv1, uv2):
        depth_pre = depth_pre.squeeze(0).numpy()
        depth_post = depth_post.squeeze(0).numpy()
        flow = flow.detach().permute(1, 2, 0).numpy() 
        fig, ax = plt.subplots(1, 3, figsize=(12, 3))
        ax[0].set_title("depth before")
        ax[0].imshow(depth_pre)
        ax[0].scatter(uv1[0], uv1[1], label='pick_uv1' if self.pick_pt else 'place_uv1')
        ax[0].scatter(uv2[0], uv2[1], label='pick_uv2' if self.pick_pt else 'place_uv2')
        ax[1].set_title("depth after")
        ax[1].imshow(depth_post)
        ax[2].set_title(f"{'ground_truth' if self.gt_flow else 'predicted'} flow")
        plot_flow(ax[2], flow, skip=0.05)
        plt.tight_layout()
        plt.show()

class Goals(Dataset):
    def __init__(self, cloth_type='square_towel'):
        self.clothtype_path = f'{os.path.abspath(os.path.dirname(__file__))}/../../data/goals/{cloth_type}' 

        # Load start state
        self.coords_start = np.load(f'{self.clothtype_path}/start.npy')
        
        # Load goals
        self.goals = []

        # Load single step goals
        singlestep_path = f'{self.clothtype_path}/single_step'
        goal_names = sorted([x.replace('.png', '') for x in os.listdir(f'{singlestep_path}/rgb')], key=lambda x: int(x.split('_')[-1]))
        for goal_name in goal_names:
            goal_im = cv2.imread(f'{singlestep_path}/depth/{goal_name}_depth.png')[:, :, 0] / 255
            coords_post = np.load(f'{singlestep_path}/coords/{goal_name}.npy')
            self.goals.append([
                {
                    "goal_name": goal_name,
                    "goal_im": torch.FloatTensor(goal_im),
                    "coords_pre": self.coords_start,  
                    "coords_post": coords_post,
                }
            ])

        # Load multi step goals
        multistep_path = f'{self.clothtype_path}/multi_step'
        goal_names = sorted([x.replace('.png', '') for x in os.listdir(f'{multistep_path}/rgb')])
        curr_goals = []
        curr_multistep_goal = goal_names[0][:-2] # Get multi-step goal name without step number at end 
        for goal_name in goal_names:
            multistep_goal = goal_name[:-2]
            goal_im = cv2.imread(f'{multistep_path}/depth/{goal_name}_depth.png')[:, :, 0] / 255
            coords_post = np.load(f'{multistep_path}/coords/{goal_name}.npy')
            goal = {
                "goal_name": goal_name,
                "goal_im": torch.FloatTensor(goal_im),
                "coords_pre": self.coords_start,  
                "coords_post": coords_post,
            }

            if curr_multistep_goal == multistep_goal: # add new goal to curr_goals
                curr_goals.append(goal)
            else: # switch to new goal set
                self.goals.append(curr_goals)
                curr_goals = [goal]
                curr_multistep_goal = multistep_goal
        self.goals.append(curr_goals)

    def __len__(self):
        return len(self.goals)

    def __getitem__(self, index):
        goal_sequence = self.goals[index]
        return goal_sequence
