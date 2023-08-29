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
parser.add_argument('--model_cfg', type=str, default='ffhflow/configs/prohmr.yaml', help='Path to config file')
parser.add_argument('--root_dir', type=str, default='checkpoints', help='Directory to save logs and checkpoints')

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
ckpt_path = '/home/yb/workspace/ffhflow/checkpoints/epoch=2-step=24931.ckpt'

model = FFHFlowNormalPosEnc.load_from_checkpoint(ckpt_path, cfg=cfg)
model.eval()

val_loader = ffh_datamodule.val_dataloader()

# Go over the images in the dataset.
for i, batch in enumerate(tqdm(val_loader)):
    with torch.no_grad():
        out = model(batch)
        model.show_grasps(batch, out)
