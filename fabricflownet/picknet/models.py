import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
import matplotlib.pyplot as plt
from fabricflownet.utils import plot_flow
import cv2

class FlowPickSplit(nn.Module):
    def __init__(self, inchannels, im_w, second=False):
        super(FlowPickSplit, self).__init__()
        self.trunk = nn.Sequential(nn.Conv2d(inchannels, 32, 5, 2),
                                    nn.ReLU(True),
                                    nn.Conv2d(32,32, 5, 2),
                                    nn.ReLU(True),
                                    nn.Conv2d(32,32, 5, 2),
                                    nn.ReLU(True),
                                    nn.Conv2d(32,32, 5, 1),
                                    nn.ReLU(True))
        self.head  = nn.Sequential(nn.Conv2d(32,32, 3, 1),
                                    nn.ReLU(True),
                                    nn.UpsamplingBilinear2d(scale_factor=2),
                                    nn.Conv2d(32,1, 3, 1))

        self.im_w = im_w
        self.second = second
        self.upsample = nn.Upsample(size=(20,20), mode="bilinear")

    def forward(self, x):
        x = self.trunk(x)
        out = self.head(x)
        out = self.upsample(out)
        return out

class FlowPickSplitModel(pl.LightningModule):
    def __init__(self,
                lr=0.0001,
                input_mode='flowonly',
                model_type='split',
                min_loss=True,
                pick=True,
                s_pick_thres = 30,
                a_len_thres = 10,
                im_width=200):
        super(FlowPickSplitModel,self).__init__()
        self.automatic_optimization = False
        self.lr = lr
        self.model_type = model_type
        self.im_width = im_width
        self.min_loss = min_loss
        self.pick = pick
        self.input_mode = input_mode
        self.s_pick_thres = s_pick_thres
        self.a_len_thres = a_len_thres
        self.init_models()
    
    def init_models(self):
        if self.model_type == 'split':
            self.first = FlowPickSplit(2, self.im_width)
            self.second = FlowPickSplit(3, self.im_width, second=True)
        else:
            raise NotImplementedError
 
    def nearest_to_mask(self, u, v, depth):
        mask_idx = np.argwhere(depth)
        nearest_idx = mask_idx[((mask_idx - [u,v])**2).sum(1).argmin()]
        return nearest_idx
        
    def get_flow_place_pt(self, u,v, flow):
        """ compute place point using flow
        """
        flow_u_idxs = np.argwhere(flow[0,:,:])
        flow_v_idxs = np.argwhere(flow[1,:,:])
        nearest_u_idx = flow_u_idxs[((flow_u_idxs - [u,v])**2).sum(1).argmin()]
        nearest_v_idx = flow_v_idxs[((flow_v_idxs - [u,v])**2).sum(1).argmin()]

        flow_u = flow[0,nearest_u_idx[0],nearest_u_idx[1]]
        flow_v = flow[1,nearest_v_idx[0],nearest_v_idx[1]]

        new_u = np.clip(u + flow_u, 0, 199)
        new_v = np.clip(v + flow_v, 0, 199)

        return new_u,new_v
    
    def get_action(self, flow, depth_pre, depth_post):
        pick_uv1, pick_uv2, info = self.forward(flow, depth_pre, depth_post)
        depth_pre_np = depth_pre.detach().squeeze().cpu().numpy()
        pick1 = self.nearest_to_mask(pick_uv1[0], pick_uv1[1], depth_pre_np)
        pick2 = self.nearest_to_mask(pick_uv2[0], pick_uv2[1], depth_pre_np)
        pickmask_u1, pickmask_v1 = pick1
        pickmask_u2, pickmask_v2 = pick2

        flow_np = flow.detach().squeeze(0).cpu().numpy()
        place_u1,place_v1 = self.get_flow_place_pt(pickmask_u1,pickmask_v1,flow_np)
        place_u2,place_v2 = self.get_flow_place_pt(pickmask_u2,pickmask_v2,flow_np)
        place1 = np.array([place_u1, place_v1])
        place2 = np.array([place_u2, place_v2])

        pred_1 = np.array(pick_uv1)
        pred_2 = np.array(pick_uv2)

        # single action threshold
        if self.s_pick_thres > 0 and np.linalg.norm(pick1-pick2) < self.s_pick_thres:
            pick2 = np.array([0,0])
            place2 = np.array([0,0])

        # action size threshold
        if self.a_len_thres > 0 and np.linalg.norm(pick1-place1) < self.a_len_thres:
            pick1 = np.array([0,0])
            place1 = np.array([0,0])

        if self.a_len_thres > 0 and np.linalg.norm(pick2-place2) < self.a_len_thres:
            pick2 = np.array([0,0])
            place2 = np.array([0,0])

        return np.array([pick1, place1, pick2, place2]), np.array([pred_1, pred_2])

    def forward(self, flow, depth_pre, depth_post):
        logits1 = self.first(flow)
        u1, v1 = self.get_pt(logits1)
        pick1_gau = self.get_gaussian(u1,v1)

        if self.input_mode == 'flowonly':
            x2 = torch.cat([flow, pick1_gau], dim=1)
        else:
            # x2 = torch.cat([depth_pre.detach().clone(), depth_post.detach().clone(), pick1_gau.detach().clone()], dim=1)
            raise NotImplementedError

        logits2 = self.second(x2)
        u2, v2 = self.get_pt(logits2)
        info = {
            'logits1': logits1,
            'logits2': logits2,
            'pick1_gau': pick1_gau
        }
        return [u1, v1], [u2, v2], info

    def get_pt(self, logits):
        N = logits.size(0)
        W = logits.size(2)

        prdepth_pre = torch.sigmoid(logits)
        prdepth_pre = prdepth_pre.view(N,1,W*W)
        val,idx = torch.max(prdepth_pre[:,0], 1)

        # u = (idx % 20) * 10
        # v = (idx // 20) * 10
        u = (idx // 20) * 10
        v = (idx % 20) * 10
        return u.item(),v.item()

    def get_gaussian(self, u, v, sigma=5, size=None):
        if size is None:
            size = self.im_width

        x0, y0 = torch.Tensor([u]).cuda(), torch.Tensor([v]).cuda()
        x0 = x0[:, None]
        y0 = y0[:, None]
        # sigma = torch.tensor(sigma, dtype=torch.float32, device=self.device)

        # N = u.size(0)
        N = 1 # u.size(0)
        num = torch.arange(size).float()
        x, y = torch.vstack([num]*N).to(self.device), torch.vstack([num]*N).to(self.device)
        gx = torch.exp(-(x-x0)**2/(2*sigma**2))
        gy = torch.exp(-(y-y0)**2/(2*sigma**2))
        g = torch.einsum('ni,no->nio', gx, gy)

        gmin = g.amin(dim=(1,2))
        gmax = g.amax(dim=(1,2))
        g = (g - gmin[:,None,None])/(gmax[:,None,None] - gmin[:,None,None])
        g = g.unsqueeze(1)
        return g

    def loss(self, logits1, logits2, pick_uv1, pick_uv2):
        N = logits1.size(0)
        W = logits1.size(2)

        pick_uv1 = pick_uv1.cuda()
        pick_uv2 = pick_uv2.cuda()
        label_a = self.get_gaussian(pick_uv1[:,0] // 10, pick_uv1[:,1] // 10, sigma=2, size=20)
        label_b = self.get_gaussian(pick_uv2[:,0] // 10, pick_uv2[:,1] // 10, sigma=2, size=20)

        if self.min_loss:
            loss_1a = torch.mean(F.binary_cross_entropy_with_logits(logits1, label_a, reduction='none'), dim=(1,2,3))
            loss_1b = torch.mean(F.binary_cross_entropy_with_logits(logits1, label_b, reduction='none'), dim=(1,2,3))
            loss_2a = torch.mean(F.binary_cross_entropy_with_logits(logits2, label_a, reduction='none'), dim=(1,2,3))
            loss_2b = torch.mean(F.binary_cross_entropy_with_logits(logits2, label_b, reduction='none'), dim=(1,2,3))

            loss1 = torch.where((loss_1a + loss_2b) < (loss_1b + loss_2a), loss_1a, loss_1b).mean()
            loss2 = torch.where((loss_1a + loss_2b) < (loss_1b + loss_2a), loss_2b, loss_2a).mean()
        else:
            loss1 = F.binary_cross_entropy_with_logits(logits1, label_a)
            loss2 = F.binary_cross_entropy_with_logits(logits2, label_b)

        return loss1, loss2

    def training_step(self, batch, batch_idx):
        opt1, opt2 = self.optimizers()
        depth_pre, depth_post, flow, pick_uv1, pick_uv2 = batch
        if self.input_mode == 'flowonly':
            x1 = flow
        else:
            raise NotImplementedError

        uv1, uv2, info = self.forward(x1, depth_pre, depth_post)
        loss1, loss2 = self.loss(info['logits1'], info['logits2'], pick_uv1, pick_uv2)
        
        opt1.zero_grad()
        self.manual_backward(loss1)
        opt1.step()
        opt2.zero_grad()
        self.manual_backward(loss2)
        opt2.step()

        if batch_idx == 0:
            self.plot(batch, uv1, uv2, info, stage='train')
        self.log_dict({"loss1/train": loss1, "loss2/train": loss2}, on_step=False, on_epoch=True, prog_bar=False)

    def validation_step(self, batch, batch_idx, log=True):
        depth_pre, depth_post, flow, pick_uv1, pick_uv2 = batch
        if self.input_mode == 'flowonly':
            x1 = flow
        else:
            raise NotImplementedError

        uv1, uv2, info = self.forward(x1, depth_pre, depth_post)
        loss1, loss2 = self.loss(info['logits1'], info['logits2'], pick_uv1, pick_uv2)
        
        if batch_idx == 0:
            self.plot(batch, uv1, uv2, info, stage='val')
        self.log_dict({"loss1/val": loss1, "loss2/val": loss2}, on_step=False, on_epoch=True, prog_bar=False)

    def configure_optimizers(self):
        if self.model_type == 'split':
            opt1 = torch.optim.Adam(self.first.parameters(), lr=self.lr)
            opt2 = torch.optim.Adam(self.second.parameters(), lr=self.lr)
            return opt1, opt2

    def plot(self, batch, pred_uv1, pred_uv2, info, stage):
        fig, ax = plt.subplots(2, 3, figsize=(16, 8))

        # Row 1
        depth_pre, depth_post, flow, gt_uv1, gt_uv2 = batch
        depth_pre = depth_pre[0].squeeze().cpu().numpy()
        depth_post = depth_post[0].squeeze().cpu().numpy()
        flow = flow[0].squeeze().permute(1, 2, 0).cpu().numpy() 
        gt_uv1 = gt_uv1[0].squeeze().cpu().numpy()
        gt_uv2 = gt_uv2[0].squeeze().cpu().numpy()
        ax[0][0].set_title("depth before")
        ax[0][0].imshow(depth_pre)
        ax[0][0].scatter(gt_uv1[0], gt_uv1[1], label='pick_uv1' if self.pick else 'place_uv1')
        ax[0][0].scatter(gt_uv2[0], gt_uv2[1], label='pick_uv2' if self.pick else 'place_uv2')
        ax[0][0].legend()
        ax[0][1].set_title("depth after")
        ax[0][1].imshow(depth_post)
        ax[0][2].set_title("flow")
        plot_flow(ax[0][2], flow, skip=0.05)

        # Row 2
        logits1 = info['logits1'][0].detach().squeeze().cpu().numpy()
        logits1 = cv2.resize(logits1, (200, 200))
        pick1_gau = info['pick1_gau'][0].detach().cpu().squeeze().numpy()
        logits2 = info['logits2'][0].detach().cpu().squeeze().numpy()
        logits2 = cv2.resize(logits2, (200, 200))
        ax[1][0].set_title("logits1")
        ax[1][0].imshow(logits1)
        ax[1][1].set_title("pick1_gaussian")
        ax[1][1].imshow(pick1_gau)
        ax[1][2].set_title("logits2")
        ax[1][2].imshow(logits2)
        plt.tight_layout()
        self.logger[1].experiment.add_figure(stage, fig, self.global_step) # tensorboard
        plt.close()
