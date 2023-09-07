import argparse
import torch

from ffhflow.configs import get_config
from ffhflow.datasets import FFHDataModule
from ffhflow.ffhflow import FFHFlow
from ffhflow.ffhflow_normal import FFHFlowNormal
from ffhflow.ffhflow_pos_enc import FFHFlowPosEnc
from ffhflow.ffhflow_normal_pos_enc import FFHFlowNormalPosEnc

parser = argparse.ArgumentParser(description='Probabilistic skeleton lifting training code')
# parser.add_argument('--model_cfg', type=str, default='ffhflow/configs/prohmr.yaml', help='Path to config file')
parser.add_argument('--model_cfg', type=str, default='models/ffhflow_bpsmlp_normal_flow_pos_enc_localinn_data_norm/hparams.yaml', help='Path to config file')
parser.add_argument('--root_dir', type=str, default='checkpoints', help='Directory to save logs and checkpoints')
parser.add_argument('--ckpt_path', type=str, default='models/ffhflow_bpsmlp_normal_flow_pos_enc_localinn_data_norm/epoch=16-step=199955.ckpt', help='Directory to save logs and checkpoints')

args = parser.parse_args()

# Set up cfg
cfg = get_config(args.model_cfg)

# Set up model
# model = FFHFlowNormalPosEnc(cfg)
# model = FFHFlowNormal(cfg)

# configure dataloader
ffh_datamodule = FFHDataModule(cfg)

# Setup PyTorch Lightning Trainer
ckpt_path = args.ckpt_path

model = FFHFlowPosEnc.load_from_checkpoint(ckpt_path, cfg=cfg)
model.eval()

val_loader = ffh_datamodule.val_dataloader()

# Go over the images in the dataset.
with torch.no_grad():
    for i, batch in enumerate(val_loader):
        out = model.sample(batch['bps_object'][0], num_samples=100)
        model.show_grasps(batch['pcd_path'][0], out, i)
        filtered_out = model.filter_grasps(out, perc=0.5)
        model.show_grasps(batch['pcd_path'][0], filtered_out, i+100)
        filtered_out = model.filter_grasps(out, perc=0.1)
        model.show_grasps(batch['pcd_path'][0], filtered_out, i+200)
        # model.show_gt_grasps(batch['pcd_path'][0], batch, i)
