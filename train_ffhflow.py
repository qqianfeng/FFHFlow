import os
import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from ffhflow.configs import get_config
from ffhflow.datasets import FFHDataModule
from ffhflow.ffhflow import FFHFlow
parser = argparse.ArgumentParser(description='Probabilistic skeleton lifting training code')
parser.add_argument('--model_cfg', type=str, default='ffhflow/configs/prohmr.yaml', help='Path to config file')
parser.add_argument('--root_dir', type=str, default='/tmp/', help='Directory to save logs and checkpoints')


args = parser.parse_args()

# Set up cfg
cfg = get_config(args.model_cfg)

# Setup Tensorboard logger
logger = TensorBoardLogger(os.path.join(args.root_dir, 'tensorboard'), name='', version='', default_hp_metric=False)

# Set up model
model = FFHFlow(cfg)

# Setup checkpoint saving
checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=
                        os.path.join(args.root_dir, 'checkpoints'),
                        every_n_train_steps=cfg.GENERAL.CHECKPOINT_STEPS)

# configure dataloader
ffh_datamodule = FFHDataModule(cfg)

# Setup PyTorch Lightning Trainer
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

# Train the model
trainer.fit(model, datamodule=ffh_datamodule)
