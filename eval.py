import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
import pickle

from ffhflow.configs import get_config
from ffhflow.datasets import FFHDataModule
from ffhflow.ffhflow_cnf import FFHFlowCNF
from ffhflow.ffhflow_lvm import FFHFlowLVM
from ffhflow.utils.grasp_data_handler import GraspDataHandlerVae
from ffhflow.utils.metrics import maad_for_grasp_distribution


def save_batch_to_file(batch):
    torch.save(batch, "data/eval_batch.pth")

def load_batch(path):
    return torch.load(path, map_location="cuda:0")

parser = argparse.ArgumentParser(description='Probabilistic skeleton lifting training code')
parser.add_argument('--model_cfg', type=str, default='/data/net/userstore/qf/ffhflow_models/ffhflow_lvm/flow_lvm_lr1e-4_RND1/hparams.yaml', help='Path to config file')
parser.add_argument('--ckpt_path', type=str, default='/data/net/userstore/qf/ffhflow_models/ffhflow_lvm/flow_lvm_lr1e-4_RND1/epoch=16-step=199999.ckpt', help='Directory to save logs and checkpoints')
parser.add_argument('--num_samples', type=float, default=100, help='Number of grasps to be generated for evaluation.')

args = parser.parse_args()
Visualization = True
MAAD = False

# Set up cfg
cfg = get_config(args.model_cfg)

# configure dataloader
ffh_datamodule = FFHDataModule(cfg)

# Setup PyTorch Lightning Trainer
ckpt_path = args.ckpt_path

if "cnf" in args.model_cfg:
    model = FFHFlowCNF.load_from_checkpoint(ckpt_path, cfg=cfg)
else:
    model = FFHFlowLVM.load_from_checkpoint(ckpt_path, cfg=cfg)

model.eval()

# val_loader = ffh_datamodule.val_dataloader(shuffle=True)
val_loader = ffh_datamodule.val_dataloader()
val_dataset = ffh_datamodule.val_dataset()

# path to save results
grasp_data_path = os.path.join(cfg.DATASETS.PATH, cfg.DATASETS.GRASP_DATA_NANE)
grasp_data = GraspDataHandlerVae(grasp_data_path)


if MAAD:
    ##### MAAD Metrics #######

    transl_loss_sum = 0
    rot_loss_sum = 0
    joint_loss_sum = 0
    coverage_sum = 0
    print(f"len(val_loader): {len(val_loader)}")
    num_nan_out = 0
    num_nan_transl = 0
    num_nan_rot = 0
    num_nan_joint = 0
    tmp_transl_sum = 2.717126583509403
    tmp_rot_sum = 6101.889077039017
    tmp_joint_sum = 5175.546305659608

    loss_per_item = {
        'kit_BakingSoda':[0,0,0],
        # 'kit_BathDetergent':[0,0,0],
        'kit_BroccoliSoup':[0,0,0],
        'kit_CoughDropsLemon':[0,0,0],
        'kit_Curry':[0,0,0],
        'kit_FizzyTabletsCalcium':[0,0,0],
        # 'kit_InstantSauce':[0,0,0],
        'kit_NutCandy':[0,0,0],
        'kit_PotatoeDumplings':[0,0,0],
        # 'kit_Sprayflask':[0,0,0],
        'kit_TomatoSoup':[0,0,0],
        'kit_YellowSaltCube2':[0,0,0],
        'kit_Peanuts':[0,0,0]
    }

    with torch.no_grad():
        batch = load_batch('data/eval_batch.pth')
        # batch = load_batch('data/eval_batch_correct_eval.pth')
        print(batch['obj_name'])
        for idx in range(len(batch['obj_name'])):
            palm_poses, joint_confs, num_pos = grasp_data.get_grasps_for_object(obj_name=batch['obj_name'][idx],outcome='positive')
            grasps_gt = val_dataset.get_grasps_from_pcd_path(batch['pcd_path'][idx])

            # out = model.sample(batch['bps_object'][idx], num_samples=100)
            out = model.sample(batch, idx, num_samples=args.num_samples)

            transl_loss, rot_loss, joint_loss, coverage = maad_for_grasp_distribution(out, grasps_gt,L1=False)
            if not math.isnan(transl_loss) and not math.isnan(rot_loss) and not math.isnan(joint_loss):
                transl_loss_sum += transl_loss
                rot_loss_sum += rot_loss
                joint_loss_sum += joint_loss
            else:
                if math.isnan(transl_loss):
                    num_nan_transl += 1
                if math.isnan(rot_loss):
                    num_nan_rot += 1
                if math.isnan(joint_loss):
                    num_nan_joint += 1
                num_nan_out += 1
            coverage_sum += coverage
            loss_per_item[batch['obj_name'][idx]][0] += transl_loss #/tmp_transl_sum
            loss_per_item[batch['obj_name'][idx]][1] += rot_loss  #/tmp_rot_sum
            loss_per_item[batch['obj_name'][idx]][2] += joint_loss  #/tmp_joint_sum

        coverage_mean = coverage_sum / len(batch['obj_name'])
        num_grasp = args.num_samples * len(batch['obj_name'])
        print(f'transl_loss_sum: {transl_loss_sum:.3f}')
        print(f'transl_loss_mean per grasp (m): {transl_loss_sum/num_grasp:.6f}')
        print(f'rot_loss_sum: {rot_loss_sum:.3f}')
        print(f'rot_loss_mean per grasp (rad): {rot_loss_sum/num_grasp:.3f}')
        print(f'joint_loss_sum: {joint_loss_sum:.3f}')
        print(f'joint_loss_mean per grasp (rad^2): {joint_loss_sum/num_grasp:.3f}')
        print(f'coverage: {coverage_mean:.3f}')
        # for k, v in loss_per_item.items():
        #     print(k,v)
        transl_list = []
        rot_list = []
        joint_list = []
        for k, v in loss_per_item.items():
            print(k,v)
            transl_list.append(v[0])
            rot_list.append(v[1])
            joint_list.append(v[2])
        transl_list_np = np.std(transl_list)
        rot_list_np = np.std(rot_list)
        joint_list_np = np.std(joint_list)
        print(transl_list_np)
        print(rot_list_np)
        print(joint_list_np)
        print(f'invalid output is: {num_nan_out}/{len(batch["obj_name"])}')
        print(f'invalid transl output is: {num_nan_transl}/{len(batch["obj_name"])}')
        print(f'invalid rot output is: {num_nan_rot}/{len(batch["obj_name"])}')
        print(f'invalid joint output is: {num_nan_joint}/{len(batch["obj_name"])}')

    ###########################


#### VISUALIZATION #####
num_samples = 100
if Visualization:
    print(f"len(val_loader): {len(val_loader)}")
    with torch.no_grad():
        batch = load_batch('data/eval_batch.pth')
        for idx in range(len(batch['obj_name'])):
            # if idx < 5:
            #     continue
            palm_poses, joint_confs, num_pos = grasp_data.get_grasps_for_object(obj_name=batch['obj_name'][idx],outcome='positive')
            grasps_gt = val_dataset.get_grasps_from_pcd_path(batch['pcd_path'][idx])
            # num_gt_grasps = grasps_gt['transl'].shape[0]
            # out = model.sample(batch['bps_object'][idx], num_samples=grasps_gt['rot_matrix'].shape[0])
<<<<<<< HEAD
            out = model.sample(batch, idx, num_samples=num_gt_grasps, posterior_score="neg_kl") # posterior_score: "neg_kl", "log_prob" or "neg_var"
=======
            out = model.sample(batch, idx, num_samples=num_samples, posterior_score="log_prob") # posterior_score: "log_prob" or "neg_var"
>>>>>>> e2e00e601febd3f9c870c48f3f19e8b94c911fd2
            print('visualize',batch['obj_name'][idx] )
            # out = model.sort_and_filter_grasps(out, perc=0.99, return_arr=False)

            # # plot value distribution to show multi-modality
            # if torch.is_tensor(out['rot_matrix']):
            #     out_np = {}
            #     for key, value in out.items():
            #         out_np[key] = value.cpu().data.numpy()
            # X = np.linspace(-5.0, 5.0, out['pred_angles'].shape[0])
            # fig, ax = plt.subplots()
            # ax.set_title("PDF from Template")
            # # ax.hist(data, density=True, bins=100)
            # ax.hist(out_np['pred_pose_transl'][:,0], label='1')
            # ax.hist(out_np['pred_pose_transl'][:,1], label='2')
            # ax.hist(out_np['pred_pose_transl'][:,2], label='3')
            # ax.legend()
            # fig.show()

            # rng = np.random.RandomState(10)  # deterministic random data
            # a = np.hstack((rng.normal(size=1000),
            #             rng.normal(loc=5, scale=2, size=1000)))
            # _ = plt.hist(a, bins='auto')  # arguments are passed to np.histogram
            # plt.title("Histogram with 'auto' bins")
            # Text(0.5, 1.0, "Histogram with 'auto' bins")
            # plt.show()

            # If we need to save the results for FFHEvaluator
            with open('/data/net/userstore/qf/test/flow_grasps.pkl', 'wb') as fp:
                pickle.dump(out, fp, protocol=2)
            with open('/data/net/userstore/qf/test/data.pkl', 'wb') as fp:
                pickle.dump([batch['bps_object'][idx], batch['pcd_path'][idx], batch['obj_name'][idx]], fp, protocol=2)

            a = input('wait for evaluator')

            with open('/data/net/userstore/qf/test/filtered_grasps.pkl', 'rb') as fp:
                filtered_grasps = pickle.load(fp)

            # original vis
            # model.show_grasps(batch['pcd_path'][idx], out, idx,frame_size=0.015, obj_name=batch['obj_name'][idx])

            # vis with evaluator score
            prob = filtered_grasps['score']
            prob_min = prob.min()
            prob_max = prob.max()
            prob = (prob - prob_min) / (prob_max - prob_min) + 0.1
            original_path = batch['pcd_path'][idx]
            parts = original_path.split('/')
            new_parts = ['/data', 'net', 'userstore','qf','hithand_data','data'] + parts[5:]
            new_parts = '/'.join(new_parts)
            model.show_grasps(new_parts, filtered_grasps, idx, prob=prob)

            # # vis with probablity
            # prob = out['log_prob'].cpu().data.numpy()
            # prob_min = prob.min()
            # prob_max = prob.max()
            # prob = (prob - prob_min) / (prob_max - prob_min) + 0.1
            # original_path = batch['pcd_path'][idx]
            # parts = original_path.split('/')
            # new_parts = ['/data', 'net', 'userstore','qf','hithand_data','data'] + parts[5:]
            # new_parts = '/'.join(new_parts)

            # model.show_grasps(new_parts, out, idx, prob=prob)

            # filtered_out = model.sort_and_filter_grasps(out, perc=0.5)
            # model.show_grasps(batch['pcd_path'][idx], filtered_out, idx+100)
            # filtered_out = model.sort_and_filter_grasps(out, perc=0.1, return_arr=False)
            # gt index till 22
            # model.show_gt_grasps(batch['pcd_path'][idx], grasps_gt, idx+300,frame_size=0.015, obj_name=batch['obj_name'][idx])
