from fabricflownet.picknet.models import FlowPickSplitModel
from fabricflownet.picknet.dataset import PickNetDataset

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
from collections import namedtuple
import time

import hydra
import pytorch_lightning.utilities.seed as seed_utils
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

Experience = namedtuple('Experience', ('obs', 'goal', 'act', 'rew', 'nobs', 'done'))

@hydra.main(config_name="config")
def main(cfg):
    seed_utils.seed_everything(cfg.seed)
    with open('.hydra/command.txt', 'w') as f:
        f.write('python ' + ' '.join(sys.argv))

    # Get training samples
    trainpath = f'{cfg.base_path}/{cfg.train_name}'
    trainfs = sorted(['_'.join(fn.split('_')[0:2])
                for fn in os.listdir(f'{trainpath}/actions')])
    if cfg.max_buf != None:
        trainfs = trainfs[:cfg.max_buf]
        print(f"Max training set: {len(trainfs)}")

    # Get validation samples
    valpath = f'{cfg.base_path}/{cfg.val_name}'
    valfs = sorted(['_'.join(fn.split('_')[0:2])
                for fn in os.listdir(f'{valpath}/actions')])
    if cfg.max_buf != None:
        valfs = valfs[:cfg.max_buf]
        print(f"Max val set: {len(valfs)}")

    # buf_train = torch.load(f"{cfg.base_path}/{cfg.train_name}/{cfg.train_name}_idx.buf")
    # if cfg.max_buf is not None:
    #     buf_train = buf_train[:cfg.max_buf]

    # buf_test = torch.load(f"{cfg.base_path}/{cfg.val_name}/{cfg.val_name}_idx.buf")
    # if cfg.max_buf is not None:
    #     buf_test = buf_test[:cfg.max_buf]

    # Get camera params
    train_camera_params = np.load(f"{cfg.base_path}/{cfg.train_name}/camera_params.npy", allow_pickle=True)[()]
    val_camera_params = np.load(f"{cfg.base_path}/{cfg.val_name}/camera_params.npy", allow_pickle=True)[()]
    np.testing.assert_equal(val_camera_params, train_camera_params)
    camera_params = train_camera_params

    train_data = PickNetDataset(camera_params, cfg, trainfs, mode='train', pick_pt=cfg.net_cfg.pick)
    val_data = PickNetDataset(camera_params, cfg, valfs, mode='test', pick_pt=cfg.net_cfg.pick)
    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers, persistent_workers=cfg.workers>0)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=cfg.workers, persistent_workers=cfg.workers>0)

    # Init model
    if cfg.net_cfg.model_type == 'split':
        model = FlowPickSplitModel(**cfg.net_cfg)
    else:
        raise NotImplementedError

    csv_logger = pl_loggers.CSVLogger(save_dir=cfg.csv_log_dir)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=cfg.tboard_log_dir)
    chkpt_cb = ModelCheckpoint(monitor='loss1/val', save_last=True, save_top_k=-1, every_n_val_epochs=10)
    trainer = pl.Trainer(gpus=[0],
                         logger=[csv_logger, tb_logger],
                         max_epochs=cfg.epochs,
                         check_val_every_n_epoch=cfg.check_val_every_n_epoch, # TODO change to every k steps
                         log_every_n_steps=len(train_loader) if len(train_loader) < 50 else 50,
                         callbacks=[chkpt_cb])
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    main()

