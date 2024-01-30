import os, time
import argparse
import shutil

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

pl.seed_everything(42, workers=True)

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

def crete_logging_file(log_folder):
    # time_str = time.ctime().replace(" ", "")
    time_str = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = f"{log_folder}/{time_str}.txt"
    os.system(f"echo 'Host:' $(hostname) | tee -a {log_file}")
    os.system(f"echo 'Conda:' $(which conda) | tee -a {log_file}")
    os.system(f"echo $(pwd) | tee -a {log_file}")
    os.system(f"echo 'Version:' $(VERSION) | tee -a {log_file}")
    os.system(f"echo 'Git diff:'| tee -a {log_file}")
    os.system(f"git diff | tee -a {log_file}")
    os.system(f"nvidia-smi| tee -a {log_file}")
    log_file = f"{log_folder}/{time_str}.log"
    return log_file

# Set up cfg
cfg = get_config(args.model_cfg)
print(f"cfg: {cfg}")

# Setup Tensorboard logger
log_folder = os.path.join(args.root_dir, cfg['NAME'])
logger = TensorBoardLogger(log_folder, name='', version='', default_hp_metric=False)
os.makedirs(log_folder, exist_ok=True) 
crete_logging_file(log_folder)
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
