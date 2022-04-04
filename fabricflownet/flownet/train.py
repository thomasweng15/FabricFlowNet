from fabricflownet.flownet.dataset import FlowDataset

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from torch.utils.data import DataLoader
import hydra
import pytorch_lightning.utilities.seed as seed_utils
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from fabricflownet.flownet.models import FlowNet

def get_flownet_dataloaders(cfg):
    # Get training samples
    trainpath = f'{cfg.base_path}/{cfg.train_name}'
    trainfs = sorted(['_'.join(fn.split('_')[0:2])
                for fn in os.listdir(f'{trainpath}/actions')])
    if cfg['max_train_samples'] != None:
        trainfs = trainfs[:cfg['max_train_samples']]
        print(f"Max training set: {len(trainfs)}")

    # Get validation samples
    valpath = f'{cfg.base_path}/{cfg.val_name}'
    valfs = sorted(['_'.join(fn.split('_')[0:2])
                for fn in os.listdir(f'{valpath}/actions')])
    if cfg['max_val_samples'] != None:
        valfs = valfs[:cfg['max_val_samples']]
        print(f"Max val set: {len(valfs)}")
    
    # Get camera params
    train_camera_params = np.load(f"{trainpath}/camera_params.npy", allow_pickle=True)[()]
    val_camera_params = np.load(f"{valpath}/camera_params.npy", allow_pickle=True)[()]
    np.testing.assert_equal(val_camera_params, train_camera_params)
    camera_params = train_camera_params

    # Get datasets
    train_data = FlowDataset(cfg, trainfs, camera_params, stage='train')
    val_data = FlowDataset(cfg, valfs, camera_params, stage='val')
    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers, persistent_workers=cfg.workers>0)
    val_loader = DataLoader(val_data, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers, persistent_workers=cfg.workers>0)
    return train_loader, val_loader

@hydra.main(config_name="config")
def main(cfg):
    seed_utils.seed_everything(cfg.seed)
    with open('.hydra/command.txt', 'w') as f:
        f.write('python ' + ' '.join(sys.argv))

    train_loader, val_loader = get_flownet_dataloaders(cfg)
    csv_logger = pl_loggers.CSVLogger(save_dir=cfg.csv_log_dir)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=cfg.tboard_log_dir)
    chkpt_cb = ModelCheckpoint(monitor='loss/val', save_last=True, save_top_k=-1, every_n_val_epochs=10)
    trainer = pl.Trainer(gpus=[0],
                         logger=[csv_logger, tb_logger],
                         max_epochs=cfg.epochs,
                         check_val_every_n_epoch=cfg.check_val_every_n_epoch,
                         log_every_n_steps=len(train_loader) if len(train_loader) < 50 else 50,
                         callbacks=[chkpt_cb])

    flownet = FlowNet(**cfg.net_cfg)
    trainer.fit(flownet, train_loader, val_loader)

if __name__ == '__main__':
    main()
