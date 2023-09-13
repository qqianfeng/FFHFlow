import argparse
import torch
import os
import numpy as np

from ffhflow.configs import get_config
from ffhflow.datasets import FFHDataModule
from ffhflow.ffhflow_pos_enc import FFHFlowPosEnc
from ffhflow.ffhflow_pos_enc_with_transl import FFHFlowPosEncWithTransl
from ffhflow.utils.metrics import maad_for_grasp_distribution
from ffhflow.utils.grasp_data_handler import GraspDataHandlerVae

parser = argparse.ArgumentParser(description='Probabilistic skeleton lifting training code')
parser.add_argument('--model_cfg', type=str, default='models/ffhflow_bpsmlp_normal_flow_pos_enc_localinn_data_norm_both_transl_rot_with_joint_conf/hparams.yaml', help='Path to config file')
parser.add_argument('--root_dir', type=str, default='checkpoints', help='Directory to save logs and checkpoints')
parser.add_argument('--ckpt_path', type=str, default='models/ffhflow_bpsmlp_normal_flow_pos_enc_localinn_data_norm_both_transl_rot_with_joint_conf/epoch=16-step=199955.ckpt', help='Directory to save logs and checkpoints')

args = parser.parse_args()

# Set up cfg
cfg = get_config(args.model_cfg)

# configure dataloader
ffh_datamodule = FFHDataModule(cfg)

# Setup PyTorch Lightning Trainer
ckpt_path = args.ckpt_path

model = FFHFlowPosEncWithTransl.load_from_checkpoint(ckpt_path, cfg=cfg)
model.eval()

val_loader = ffh_datamodule.val_dataloader()
val_dataset = ffh_datamodule.val_dataset()

base_path = '/home/yb/Documents/ffhflow_grasp'
# # Go over the images in the dataset.
# with torch.no_grad():
#     for i, batch in enumerate(val_loader):
#         if i <2:
#             continue
#         out = model.sample(batch['bps_object'][0], num_samples=100)
#         model.show_grasps(batch['pcd_path'][0], out, i)
#         # filtered_out = model.sort_and_filter_grasps(out, perc=0.5)
#         # model.show_grasps(batch['pcd_path'][0], filtered_out, i+100)
#         filtered_out = model.sort_and_filter_grasps(out, perc=0.1, return_arr=False)
#         # model.show_grasps(batch['pcd_path'][0], filtered_out, i+200, base_path, save=False)
#         # model.show_gt_grasps(batch['pcd_path'][0], batch, i)

# MAAD Metrics
grasp_data_path = os.path.join(cfg.DATASETS.PATH, cfg.DATASETS.GRASP_DATA_NANE)
grasp_data = GraspDataHandlerVae(grasp_data_path)
with torch.no_grad():
    for i, batch in enumerate(val_loader):
        out = model.sample(batch['bps_object'][0], num_samples=100)

        palm_poses, joint_confs, num_pos = grasp_data.get_grasps_for_object(obj_name=batch['obj_name'][0],outcome='positive')
        grasps_gt = val_dataset.get_grasps_from_pcd_path(batch['pcd_path'][0])

        maad_for_grasp_distribution(out, grasps_gt)
        model.show_grasps(batch['pcd_path'][0], out, i)
        # # filtered_out = model.sort_and_filter_grasps(out, perc=0.5)
        # # model.show_grasps(batch['pcd_path'][0], filtered_out, i+100)
        # filtered_out = model.sort_and_filter_grasps(out, perc=0.1, return_arr=False)
        # # model.show_grasps(batch['pcd_path'][0], filtered_out, i+200, base_path, save=False)
        model.show_gt_grasps(batch['pcd_path'][0], grasps_gt, i+300)