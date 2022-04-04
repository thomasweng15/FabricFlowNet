import torch
import torch.nn as nn
from torch.nn import init

import math
import numpy as np
import pytorch_lightning as pl

import matplotlib as mpl
import matplotlib.pyplot as plt
from fabricflownet.utils import plot_flow

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )

def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=True)

def deconv(in_planes, out_planes, ksize=3):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=ksize, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1,inplace=True)
    )

class FlowNet(pl.LightningModule):
    def __init__(self, 
                 input_channels = 2, 
                 batchNorm=True, 
                 lr=0.0001, 
                 weight_decay=0.0001):
        super(FlowNet,self).__init__()

        self.lr = lr
        self.weight_decay = weight_decay

        fs = [8, 16, 32, 64, 128] # filter sizes
        self.batchNorm = batchNorm
        self.conv1   = conv(self.batchNorm, input_channels, fs[0], kernel_size=7, stride=2) # 384 -> (384 - 7 + 2*3)/2 + 1 = 377
        self.conv2   = conv(self.batchNorm, fs[0], fs[1], kernel_size=5, stride=2)
        self.conv3   = conv(self.batchNorm, fs[1], fs[2], kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, fs[2], fs[2])
        self.conv4   = conv(self.batchNorm, fs[2], fs[3], stride=2)
        self.conv4_1 = conv(self.batchNorm, fs[3], fs[3])
        self.conv5   = conv(self.batchNorm, fs[3], fs[3], stride=2)
        self.conv5_1 = conv(self.batchNorm, fs[3], fs[3])
        self.conv6   = conv(self.batchNorm, fs[3], fs[4], stride=2)
        self.conv6_1 = conv(self.batchNorm, fs[4], fs[4])

        self.deconv5 = deconv(fs[4],fs[3])
        self.deconv4 = deconv(fs[3]+fs[3]+2,fs[2])
        self.deconv3 = deconv(fs[3]+fs[2]+2,fs[1])
        self.deconv2 = deconv(fs[2]+fs[1]+2,fs[0], ksize=4)

        self.predict_flow6 = predict_flow(fs[4])
        self.predict_flow5 = predict_flow(fs[3]+fs[3]+2)
        self.predict_flow4 = predict_flow(fs[3]+fs[2]+2)
        self.predict_flow3 = predict_flow(fs[2]+fs[1]+2)
        self.predict_flow2 = predict_flow(fs[1]+fs[0]+2)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 3, 2, 1, bias=False) # (H_in-1)*stride - 2*padding + (kernel-1) + 1
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 3, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 3, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6       = self.predict_flow6(out_conv6)
        flow6_up    = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)

        concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)
        flow5       = self.predict_flow5(concat5)
        flow5_up    = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)
        
        concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)
        flow4       = self.predict_flow4(concat4)
        flow4_up    = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)

        concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
        flow3       = self.predict_flow3(concat3)
        flow3_up    = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)

        concat2 = torch.cat((out_conv2,out_deconv2,flow3_up),1)
        flow2 = self.predict_flow2(concat2)

        out = self.upsample1(flow2)
        return out

    def loss(self, input_flow, target_flow, mask):
        b, c, h, w = input_flow.size()
        diff_flow = torch.reshape(target_flow - input_flow*mask, (b, c, h*w))
        mask = torch.reshape(mask, (b, h*w))
        norm_diff_flow = torch.linalg.norm(diff_flow, ord=2, dim=1) # B x 40000 get norm of flow vector diff
        mean_norm_diff_flow = norm_diff_flow.sum(dim=1) / mask.sum(dim=1) # B x 1 get average norm for each image
        batch_mean_diff_flow = mean_norm_diff_flow.mean() # mean over the batch
        return batch_mean_diff_flow

    def training_step(self, batch, batch_idx):
        depth_input = batch['depths']
        flow_gt = batch['flow_gt']
        loss_mask = batch['loss_mask']
        flow_out = self.forward(depth_input)
        train_loss = self.loss(flow_out, flow_gt, loss_mask)
        loss = train_loss

        if batch_idx == 0:
            self.plot(depth_input, loss_mask, flow_gt, flow_out, stage="train")
        self.log('loss/train', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        depth_input = batch['depths']
        flow_gt = batch['flow_gt']
        loss_mask = batch['loss_mask']
        flow_out = self.forward(depth_input)
        loss = self.loss(flow_out, flow_gt, loss_mask)

        if batch_idx == 0:
            self.plot(depth_input, loss_mask, flow_gt, flow_out, stage="val")
        self.log('loss/val', loss)
        return {'loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def plot(self, depth_input, loss_mask, flow_gt, flow_out, stage):
        im1 = depth_input[0, 0].detach().cpu().numpy()
        im2 = depth_input[0, 1].detach().cpu().numpy()
        loss_mask = loss_mask[0].detach().squeeze().cpu().numpy()
        flow_gt = flow_gt[0].detach().permute(1, 2, 0).cpu().numpy()
        flow_out = flow_out[0].detach().permute(1, 2, 0).cpu().numpy()

        fig, ax = plt.subplots(1, 5, figsize=(12, 3), dpi=300)
        ax[0].set_title("depth before")
        ax[0].imshow(im1)
        ax[1].set_title("depth after")
        ax[1].imshow(im2)

        ax[2].set_title("ground-truth flow")
        plot_flow(ax[2], flow_gt)

        ax[3].set_title("predicted flow (masked)")
        flow_out[loss_mask == 0, :] = 0
        plot_flow(ax[3], flow_out)

        ax[4].set_title("loss mask")
        ax[4].imshow(loss_mask)

        plt.tight_layout()
        self.logger[1].experiment.add_figure(stage, fig, self.global_step) # tensorboard
        plt.close()
