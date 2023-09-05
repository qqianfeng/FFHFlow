import os
import argparse
import shutil
from tqdm import tqdm
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from ffhflow.configs import get_config
from ffhflow.datasets import FFHDataModule
from ffhflow.ffhflow import FFHFlow
from ffhflow.ffhflow_normal import FFHFlowNormal
from ffhflow.ffhflow_pos_enc import FFHFlowPosEnc
from ffhflow.ffhflow_normal_pos_enc import FFHFlowNormalPosEnc

parser = argparse.ArgumentParser(description='Probabilistic skeleton lifting training code')
# parser.add_argument('--model_cfg', type=str, default='ffhflow/configs/prohmr.yaml', help='Path to config file')
parser.add_argument('--model_cfg', type=str, default='models/ffhflow_normal_flow_continue_train_complete/hparams.yaml', help='Path to config file')
parser.add_argument('--root_dir', type=str, default='checkpoints', help='Directory to save logs and checkpoints')
parser.add_argument('--ckpt_path', type=str, default='models/ffhflow_normal_flow_continue_train_complete/epoch=12-step=154391.ckpt', help='Directory to save logs and checkpoints')

args = parser.parse_args()

# Set up cfg
cfg = get_config(args.model_cfg)

# copy the config file to save_dir
fname = os.path.join(args.root_dir, 'hparams.yaml')
if not os.path.isfile(fname):
    shutil.copy(args.model_cfg, fname)

# Setup Tensorboard logger
logger = TensorBoardLogger(os.path.join(args.root_dir, 'tensorboard'), name='', version='', default_hp_metric=False)

# Set up model
# model = FFHFlowNormalPosEnc(cfg)
# model = FFHFlowNormal(cfg)

# Setup checkpoint saving
checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=
                        os.path.join(args.root_dir, 'tensorboard'),
                        every_n_epochs=1)
# configure dataloader
ffh_datamodule = FFHDataModule(cfg)

# Setup PyTorch Lightning Trainer
ckpt_path = args.ckpt_path

model = FFHFlowNormal.load_from_checkpoint(ckpt_path, cfg=cfg)
model.eval()

val_loader = ffh_datamodule.val_dataloader()

# Go over the images in the dataset.
with torch.no_grad():
    for i, batch in enumerate(val_loader):
        out = model.sample(batch['bps_object'][0],num_samples=100)
        model.show_grasps(batch['pcd_path'][0], out)
