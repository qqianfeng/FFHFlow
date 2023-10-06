import argparse
import torch
import os

from ffhflow.configs import get_config
from ffhflow.datasets import FFHDataModule
from ffhflow.ffhflow_pos_enc import FFHFlowPosEnc
from ffhflow.ffhflow_pos_enc_with_transl import FFHFlowPosEncWithTransl
from ffhflow.utils.metrics import maad_for_grasp_distribution
from ffhflow.utils.grasp_data_handler import GraspDataHandlerVae
from ffhflow.ffhflow_pos_enc_neg_grasp import FFHFlowPosEncNegGrasp

def save_batch_to_file(batch):
    torch.save(batch, "eval_batch.pth")

def load_batch(path):
    return torch.load(path)

parser = argparse.ArgumentParser(description='Probabilistic skeleton lifting training code')
parser.add_argument('--model_cfg', type=str, default='models/ffhflow_best_hparam_neg_grasp_only/hparams.yaml', help='Path to config file')
parser.add_argument('--root_dir', type=str, default='checkpoints', help='Directory to save logs and checkpoints')
parser.add_argument('--ckpt_path', type=str, default='models/ffhflow_best_hparam_neg_grasp_only/epoch=24-step=299999.ckpt', help='Directory to save logs and checkpoints')

args = parser.parse_args()

cfg = get_config(args.model_cfg)

# Setup PyTorch Lightning Trainer
ckpt_path = args.ckpt_path

neg_model = FFHFlowPosEncWithTransl.load_from_checkpoint(ckpt_path, cfg=cfg)
neg_model.eval()

parser = argparse.ArgumentParser(description='Probabilistic skeleton lifting training code')
parser.add_argument('--model_cfg', type=str, default='models/ffhflow_flow_pos_enc_res_depth4_epoch25/hparams.yaml', help='Path to config file')
parser.add_argument('--root_dir', type=str, default='checkpoints', help='Directory to save logs and checkpoints')
parser.add_argument('--ckpt_path', type=str, default='models/ffhflow_flow_pos_enc_res_depth4_epoch25/epoch=25-step=299983.ckpt', help='Directory to save logs and checkpoints')

args = parser.parse_args()

# Set up cfg
cfg = get_config(args.model_cfg)

# configure dataloader
ffh_datamodule = FFHDataModule(cfg)

# Setup PyTorch Lightning Trainer
ckpt_path = args.ckpt_path

pos_model = FFHFlowPosEncWithTransl.load_from_checkpoint(ckpt_path, cfg=cfg)
pos_model.eval()

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

transl_loss_sum = 0
rot_loss_sum = 0
joint_loss_sum = 0
print(len(val_loader))
with torch.no_grad():
    batch = load_batch('eval_batch.pth')
    for idx in range(len(batch['obj_name'])):
        palm_poses, joint_confs, num_pos = grasp_data.get_grasps_for_object(obj_name=batch['obj_name'][idx],outcome='negative')
        grasps_gt = val_dataset.get_grasps_from_pcd_path(batch['pcd_path'][idx])

        # out = model.sample(batch['bps_object'][idx], num_samples=grasps_gt['rot_matrix'].shape[0])
        out = pos_model.sample(batch['bps_object'][idx], num_samples=100)
        # transl_loss, rot_loss, joint_loss = maad_for_grasp_distribution(out, grasps_gt)
        # transl_loss_sum += transl_loss
        # rot_loss_sum += rot_loss
        # joint_loss_sum += joint_loss

        # model.show_grasps(batch['pcd_path'][idx], out, idx)
        neg_log_prob = neg_model.predict_log_prob(batch['bps_object'][idx], out)
        out['log_prob'] = out['log_prob'] - neg_log_prob

        filtered_out = pos_model.sort_and_filter_grasps(out, perc=0.1, return_arr=False)

        # # model.show_grasps(batch['pcd_path'][0], filtered_out, i+100)
        # filtered_out = model.sort_and_filter_grasps(out, perc=0.1, return_arr=False)
        # # model.show_grasps(batch['pcd_path'][0], filtered_out, i+200, base_path, save=False)
        # model.show_gt_grasps(batch['pcd_path'][idx], grasps_gt, idx+300)

    print('transl_loss_sum:', transl_loss_sum)
    print('rot_loss_sum:', rot_loss_sum)
    print('joint_loss_sum:', joint_loss_sum)
