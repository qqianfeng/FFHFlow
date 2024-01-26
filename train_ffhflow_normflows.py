import os
import argparse
import shutil

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch import seed_everything

seed_everything(42, workers=True)

# sets seeds for numpy, torch and python.random.
import sys
sys.path.insert(0,os.path.join(os.path.expanduser('~'),'workspace/normalizing-flows'))

from ffhflow.configs import get_config
from ffhflow.datasets import FFHDataModule
from ffhflow.normflows_ffhflow_pos_enc_with_transl import NormflowsFFHFlowPosEncWithTransl, NormflowsFFHFlowPosEncWithTransl_Grasp


parser = argparse.ArgumentParser(description='Probabilistic skeleton lifting training code')
parser.add_argument('--model_cfg', type=str, default='ffhflow/configs/prohmr.yaml', help='Path to config file')
parser.add_argument('--root_dir', type=str, default='checkpoints', help='Directory to save logs and checkpoints')
parser.add_argument('--resume_ckp', type=str, default=None, help='Directory to checkpoints to be resumed')

args = parser.parse_args()

# Set up cfg
cfg = get_config(args.model_cfg)
print(f"cfg: {cfg}")

# Setup Tensorboard logger
logger = TensorBoardLogger(os.path.join(args.root_dir, cfg['NAME']), name='', version='', default_hp_metric=False)

# Set up model
# model = FFHFlow(cfg)
# model = NormflowsFFHFlowPosEncWithTransl(cfg)
model = NormflowsFFHFlowPosEncWithTransl_Grasp(cfg)

# Setup checkpoint saving
save_folder = os.path.join(args.root_dir, cfg['NAME'])
checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=save_folder,
                        every_n_train_steps=10000,
                        save_top_k=-1)

# copy the config file to save_dir
if not os.path.exists(save_folder):
    os.mkdir(save_folder)
else: # remove tf files only for better monitoring in tensorboard
    files = os.listdir(save_folder)
    for f in files:
        if "events.out.tfevents" in f:
            os.remove(os.path.join(save_folder, f))

fname = os.path.join(save_folder, 'hparams.yaml')
if os.path.isfile(fname):
    os.remove(fname)
shutil.copy(args.model_cfg, fname)

# configure dataloader
ffh_datamodule = FFHDataModule(cfg)

# Setup PyTorch Lightning Trainer
ckpt_path = args.resume_ckp # 'checkpoints/test_lvm1/epoch=12-step=149999.ckpt'

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
                     callbacks=[checkpoint_callback],
                     resume_from_checkpoint=ckpt_path,
                     deterministic=True)

# Train the model
trainer.fit(model, datamodule=ffh_datamodule)
