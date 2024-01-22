import argparse
import torch
import os
import pickle

from ffhflow.configs import get_config
from ffhflow.datasets import FFHDataModule
from ffhflow.utils.metrics import maad_for_grasp_distribution
from ffhflow.utils.grasp_data_handler import GraspDataHandlerVae

from ffhflow.normflows_ffhflow_pos_enc_with_transl import NormflowsFFHFlowPosEncWithTransl, NormflowsFFHFlowPosEncWithTransl_Grasp

def save_batch_to_file(batch):
    torch.save(batch, "eval_batch.pth")

def load_batch(path):
    return torch.load(path, map_location="cuda:0")

parser = argparse.ArgumentParser(description='Probabilistic skeleton lifting training code')
parser.add_argument('--model_cfg', type=str, default='checkpoints/normflow_affine_old_best_param/hparams.yaml', help='Path to config file')
# parser.add_argument('--root_dir', type=str, default='checkpoints', help='Directory to save logs and checkpoints')
parser.add_argument('--ckpt_path', type=str, default='checkpoints/normflow_affine_old_best_param/epoch=24-step=299999.ckpt', help='Directory to save logs and checkpoints')

args = parser.parse_args()
Visualization = True

# Set up cfg
cfg = get_config(args.model_cfg)

# configure dataloader
ffh_datamodule = FFHDataModule(cfg)

# Setup PyTorch Lightning Trainer
ckpt_path = args.ckpt_path

# model = NormflowsFFHFlowPosEncWithTransl.load_from_checkpoint(ckpt_path, cfg=cfg)
model = NormflowsFFHFlowPosEncWithTransl_Grasp.load_from_checkpoint(ckpt_path, cfg=cfg)
model.eval()

val_loader = ffh_datamodule.val_dataloader()
val_dataset = ffh_datamodule.val_dataset()

save_path = '/home/yb/Documents/ffhflow_grasp'
grasp_data_path = os.path.join(cfg.DATASETS.PATH, cfg.DATASETS.GRASP_DATA_NANE)
grasp_data = GraspDataHandlerVae(grasp_data_path)

##### MAAD Metrics #######
import math

transl_loss_sum = 0
rot_loss_sum = 0
joint_loss_sum = 0
print(len(val_loader))
with torch.no_grad():
    batch = load_batch('eval_batch.pth')
    for idx in range(len(batch['obj_name'])):
        palm_poses, joint_confs, num_pos = grasp_data.get_grasps_for_object(obj_name=batch['obj_name'][idx],outcome='positive')
        grasps_gt = val_dataset.get_grasps_from_pcd_path(batch['pcd_path'][idx])

        # out = model.sample(batch['bps_object'][idx], num_samples=100)
        out = model.sample(batch, idx, num_samples=100)


        transl_loss, rot_loss, joint_loss = maad_for_grasp_distribution(out, grasps_gt)
        if not math.isnan(transl_loss):
            transl_loss_sum += transl_loss
        if not math.isnan(rot_loss):
            rot_loss_sum += rot_loss
        if not math.isnan(joint_loss):
            joint_loss_sum += joint_loss
    print('transl_loss_sum:', transl_loss_sum)
    print('rot_loss_sum:', rot_loss_sum)
    print('joint_loss_sum:', joint_loss_sum)

###########################


#### VISUALIZATION #####
def pth_correction(old_pth):
    new_path = old_pth.replace("/data/hdd1/qf/hithand_data/ffhnet-data/eval/pcd/", "/data/net/userstore/qf/hithand_data/data/ffhnet-data/eval/pcd/")
    return new_path

if Visualization:
    print(f"len(val_loader): {len(val_loader)}")
    with torch.no_grad():
        batch = load_batch('eval_batch.pth')
        for idx in range(len(batch['obj_name'])):
            # if idx < 5:
            #     continue
            palm_poses, joint_confs, num_pos = grasp_data.get_grasps_for_object(obj_name=batch['obj_name'][idx],outcome='positive')
            grasps_gt = val_dataset.get_grasps_from_pcd_path(batch['pcd_path'][idx])

            # out = model.sample(batch['bps_object'][idx], num_samples=grasps_gt['rot_matrix'].shape[0])
            out = model.sample(batch, idx, num_samples=100)

            # If we need to save the results for FFHEvaluator
            # with open('flow_grasps.pkl', 'wb') as fp:
            #     pickle.dump(out, fp, protocol=2)
            # with open('data.pkl', 'wb') as fp:
            #     pickle.dump([batch['bps_object'][idx], batch['pcd_path'][idx], batch['obj_name'][idx]], fp, protocol=2)

            # model.show_grasps(batch['pcd_path'][idx], out, idx)
            filtered_out = model.sort_and_filter_grasps(out, perc=0.5)
            corrected_pth = pth_correction(batch['pcd_path'][idx])
            model.show_grasps(corrected_pth, filtered_out, idx+100)
            # filtered_out = model.sort_and_filter_grasps(out, perc=0.1, return_arr=False)
            # # model.show_grasps(batch['pcd_path'][0], filtered_out, i+200, save_path, save=False)
            # model.show_gt_grasps(batch['pcd_path'][idx], grasps_gt, idx+300)


# #### VISUALIZATION for POS + Neg Grasps#####
# print(len(val_loader))
# with torch.no_grad():
#     batch = load_batch('eval_batch.pth')
#     for idx in range(len(batch['obj_name'])):
#         if idx < 1:
#             continue
#         palm_poses, joint_confs, num_pos = grasp_data.get_grasps_for_object(obj_name=batch['obj_name'][idx],outcome='negative')
#         grasps_gt_pos = val_dataset.get_grasps_from_pcd_path(batch['pcd_path'][idx],label='positive')
#         grasps_gt_neg = val_dataset.get_grasps_from_pcd_path(batch['pcd_path'][idx],label='negative')
#         model.show_gt_grasps_pos_neg(batch['pcd_path'][idx], [grasps_gt_pos, grasps_gt_neg], idx+300)
