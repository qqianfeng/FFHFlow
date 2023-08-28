import os
import argparse
import shutil

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
model = FFHFlowNormalPosEnc(cfg)
# model = FFHFlowNormal(cfg)

# Setup checkpoint saving
checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=
                        os.path.join(args.root_dir, 'tensorboard'),
                        every_n_epochs=1)

# configure dataloader
ffh_datamodule = FFHDataModule(cfg)

# Setup PyTorch Lightning Trainer
ckpt_path = '/home/qf/workspace/ffhflow/checkpoints/tensorboard/epoch=7-step=97261.ckpt'

trainer = pl.Trainer(default_root_dir=args.root_dir,
                     logger=logger,
                     gpus=1,
                     limit_val_batches=1,
                     num_sanity_val_steps=0,
                     log_every_n_steps=cfg.GENERAL.LOG_STEPS,
                     flush_logs_every_n_steps=cfg.GENERAL.LOG_STEPS,
                     val_check_interval=cfg.GENERAL.VAL_STEPS,
                     progress_bar_refresh_rate=1,
                     precision=16,
                     max_steps=cfg.GENERAL.TOTAL_STEPS,
                     move_metrics_to_cpu=True,
                     callbacks=[checkpoint_callback])
                    #  resume_from_checkpoint=ckpt_path)

# Train the model
trainer.fit(model, datamodule=ffh_datamodule)
