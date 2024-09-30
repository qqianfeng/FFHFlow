import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from tqdm import tqdm
import pickle
import transforms3d

from ffhflow.configs import get_config
from ffhflow.datasets import FFHDataModule
from ffhflow.ffhflow_cnf import FFHFlowCNF
from ffhflow.ffhflow_lvm import FFHFlowLVM
from ffhflow.utils import utils
from ffhflow.utils.grasp_data_handler import GraspDataHandlerVae
from ffhflow.utils.metrics import maad_for_grasp_distribution

def save_batch_to_file(batch):
    torch.save(batch, "data/eval_batch.pth")


def load_batch(path):
    return torch.load(path, map_location="cuda:0")


def compute_maad(batch):
    transl_loss_sum = 0
    rot_loss_sum = 0
    joint_loss_sum = 0
    coverage_sum = 0
    print(f"len(val_loader): {len(val_loader)}")
    num_nan_out = 0
    num_nan_transl = 0
    num_nan_rot = 0
    num_nan_joint = 0
    # tmp_transl_sum = 2.717126583509403
    # tmp_rot_sum = 6101.889077039017
    # tmp_joint_sum = 5175.546305659608

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
        

def plot_transl_dist(pred_pose_transl):
    if torch.is_tensor(pred_pose_transl):
        pred_pose_transl = pred_pose_transl.cpu().data.numpy
    X = np.linspace(-5.0, 5.0, pred_pose_transl.shape[0])
    fig, ax = plt.subplots()
    ax.set_title("PDF from Template")
    # ax.hist(data, density=True, bins=100)
    ax.hist(pred_pose_transl[:,0], label='1')
    ax.hist(pred_pose_transl[:,1], label='2')
    ax.hist(pred_pose_transl[:,2], label='3')
    ax.legend()
    fig.show()

def viz_grasps_with_scores(model, predicted_grasps, idx, batch):
    prob = predicted_grasps['log_prob']
    prob_min = prob.min()
    prob_max = prob.max()
    prob = (prob - prob_min) / (prob_max - prob_min) + 0.1
    # original_path = batch['pcd_path'][idx]
    # parts = original_path.split('/')
    # new_parts = ['/data', 'net', 'userstore','qf','hithand_data','data'] + parts[5:]
    # new_parts = '/'.join(new_parts)
    model.show_grasps(batch['pcd_path'][idx], predicted_grasps, idx, prob=prob)
    
def get_grasps_from_pcd_path(pcd_path, label, val_dataset):
    grasps = val_dataset.get_grasps_from_pcd_path(pcd_path, label=label)
    pred_angles = []
    joint_confs = []
    for i in range(grasps['rot_matrix'].shape[0]):
        pred_angles.append(transforms3d.euler.mat2euler(grasps['rot_matrix'][i]))
        joint_confs.append(utils.reduce_joint_conf(grasps['joint_conf'][i])) 

    pred_angles = np.array(pred_angles).reshape(-1, 3)
    joint_confs = np.array(joint_confs).reshape(-1, 15)
    pred_pose_transl = grasps['transl']
    grasps = np.concatenate([pred_angles, pred_pose_transl, joint_confs], axis=1)

    return grasps

def get_scores_per_item(model, posterior_score, batch, obj_idx, val_dataset):
    pos_grasps = get_grasps_from_pcd_path(batch['pcd_path'][obj_idx], 'positive', val_dataset)
    neg_grasps = get_grasps_from_pcd_path(batch['pcd_path'][obj_idx], 'negative', val_dataset)
    neg_grasps = torch.tensor(neg_grasps).to('cuda')
    pos_grasps = torch.tensor(pos_grasps).to('cuda')
    bps_tensor = batch['bps_object'][obj_idx].to(dtype=torch.float64).to('cuda')
    bps_tensor = bps_tensor.view(1,-1)
    neg_scores = model.compute_grasp_score(neg_grasps, posterior_score, bps_tensor).detach().cpu().numpy()
    pos_scores = model.compute_grasp_score(pos_grasps, posterior_score, bps_tensor).detach().cpu().numpy()
    return pos_scores, neg_scores

def get_grasps_score_hist(val_loader, model, val_dataset, posterior_score):
    print(f"len(val_loader): {len(val_loader)}")
    scores_per_item = {
        'kit_BakingSoda':{"pos":[], "neg":[]},
        'kit_BathDetergent':{"pos":[], "neg":[]},
        'kit_BroccoliSoup':{"pos":[], "neg":[]},
        'kit_CoughDropsLemon':{"pos":[], "neg":[]},
        'kit_Curry':{"pos":[], "neg":[]},
        'kit_FizzyTabletsCalcium':{"pos":[], "neg":[]},
        'kit_InstantSauce':{"pos":[], "neg":[]},
        'kit_NutCandy':{"pos":[], "neg":[]},
        'kit_PotatoeDumplings':{"pos":[], "neg":[]},
        'kit_Sprayflask':{"pos":[], "neg":[]},
        'kit_TomatoSoup':{"pos":[], "neg":[]},
        'kit_YellowSaltCube2':{"pos":[], "neg":[]},
        'kit_Peanuts':{"pos":[], "neg":[]}
    }
    for i, batch in tqdm(enumerate(val_loader)):
        num_obj = len(batch['obj_name'])
        for obj_idx in tqdm(range(num_obj)):
            obj_name = batch['obj_name'][obj_idx]
            pos_scores, neg_scores  = get_scores_per_item(model, posterior_score, batch, obj_idx, val_dataset)
            scores_per_item[obj_name]["pos"].extend(pos_scores)
            scores_per_item[obj_name]["neg"].extend(neg_scores)

    return scores_per_item

def visualize(batch, mode, num_samples=100, grasp_flow_n_samples=30, posterior_score=None, val_dataset=None):
    # print(f"len(val_loader): {len(val_loader)}")
    with torch.no_grad():
        num_obj = len(batch['obj_name'])
        neg_scores, pos_scores = [], []
        for obj_idx in range(num_obj):
            if mode == "viz_transl_dist":
                predicted_grasps = model.sample(batch, 
                                                obj_idx, 
                                                num_samples=num_samples,
                                                grasp_flow_n_samples=grasp_flow_n_samples,
                                                posterior_score=posterior_score)
                predicted_grasps = model.sort_and_filter_grasps(predicted_grasps, perc=0.99, return_arr=False)
                plot_transl_dist(predicted_grasps['pred_pose_transl']) 
            elif mode == "viz_grasps_wo_scores":
                predicted_grasps = model.sample(batch, 
                                                obj_idx, 
                                                num_samples=num_samples,
                                                avg_grasps=False,
                                                grasp_flow_n_samples=grasp_flow_n_samples,
                                                posterior_score=posterior_score)
                # predicted_grasps = model.sort_and_filter_grasps(predicted_grasps, perc=0.5)
                model.show_grasps(batch['pcd_path'][obj_idx], predicted_grasps, -1)
                # show gt grasps
                # grasps_gt = val_dataset.get_grasps_from_pcd_path(batch['pcd_path'][obj_idx])
                # model.show_gt_grasps(batch['pcd_path'][obj_idx], grasps_gt, obj_idx+300,frame_size=0.015, obj_name=batch['obj_name'][obj_idx])
            elif mode == "viz_grasps_with_scores":
                predicted_grasps = model.sample(batch, 
                                                obj_idx, 
                                                num_samples=num_samples,
                                                avg_grasps=True,
                                                grasp_flow_n_samples=grasp_flow_n_samples,
                                                posterior_score=posterior_score)
                viz_grasps_with_scores(model, predicted_grasps, obj_idx, batch)
            elif mode == "viz_neg_pos_hist":
                posterior_score = "neg_kl"
                pos_scores, neg_scores = get_scores_per_item(model, posterior_score, batch, obj_idx)
                plt.hist(neg_scores, density=True,  histtype='barstacked', label=f'neg_grasps({neg_scores.shape[0]})', rwidth=0.5)
                plt.hist(pos_scores, density=True,  histtype='barstacked', label=f'pos_grasps({pos_scores.shape[0]})', rwidth=0.5)
                plt.title(f"Negative and Positive Grasps {posterior_score} of {batch['obj_name'][obj_idx]}")
                plt.legend()
                plt.show()
            elif mode == "filter_with_eval":
                with open('/data/net/userstore/qf/test/flow_grasps.pkl', 'wb') as fp:
                    pickle.dump(predicted_grasps, fp, protocol=2)
                with open('/data/net/userstore/qf/test/data.pkl', 'wb') as fp:
                    pickle.dump([batch['bps_object'][obj_idx], batch['pcd_path'][obj_idx], batch['obj_name'][obj_idx]], fp, protocol=2)

                a = input('wait for evaluator')

                with open('/data/net/userstore/qf/test/filtered_grasps.pkl', 'rb') as fp:
                    filtered_grasps = pickle.load(fp)

                # vis with evaluator score
                prob = filtered_grasps['score']
                prob_min = prob.min()
                prob_max = prob.max()
                prob = (prob - prob_min) / (prob_max - prob_min) + 0.1
                original_path = batch['pcd_path'][obj_idx]
                parts = original_path.split('/')
                new_parts = ['/data', 'net', 'userstore','qf','hithand_data','data'] + parts[5:]
                new_parts = '/'.join(new_parts)
                model.show_grasps(new_parts, filtered_grasps, obj_idx, prob=prob)
            elif mode == "filter_with_prob":
                # vis with probablity
                prob = predicted_grasps['log_prob'].cpu().data.numpy()
                prob_min = prob.min()
                prob_max = prob.max()
                prob = (prob - prob_min) / (prob_max - prob_min) + 0.1
                original_path = batch['pcd_path'][obj_idx]
                parts = original_path.split('/')
                new_parts = ['/data', 'net', 'userstore','qf','hithand_data','data'] + parts[5:]
                new_parts = '/'.join(new_parts)

                model.show_grasps(new_parts, predicted_grasps, obj_idx, prob=prob)
            else:
                print("Please specify the mode for visualization.")
                break

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Probabilistic skeleton lifting training code')
    parser.add_argument('--model_cfg', type=str, default='/data/net/userstore/qf/ffhflow_models/ffhflow_lvm/hparams.yaml', help='Path to config file')
    parser.add_argument('--ckpt_path', type=str, default='/data/net/userstore/qf/ffhflow_models/ffhflow_lvm/epoch=16-step=199999.ckpt', help='Directory to save logs and checkpoints')
    parser.add_argument('--num_samples', type=float, default=100, help='Number of grasps to be generated for evaluation.')

    args = parser.parse_args()
    Visualization = True
    MAAD = False
    Draw_grasp_score_hist = False

    # Set up cfg
    cfg = get_config(args.model_cfg)

    # configure dataloader
    # cfg["TRAIN"]["BATCH_SIZE"] = 1024*8
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
    trn_loader = ffh_datamodule.train_dataloader()
    val_dataset = ffh_datamodule.val_dataset()
    trn_dataset = ffh_datamodule.train_dataset()

    # path to save results
    grasp_data_path = os.path.join(cfg.DATASETS.PATH, cfg.DATASETS.GRASP_DATA_NANE)
    grasp_data = GraspDataHandlerVae(grasp_data_path)

    if Draw_grasp_score_hist:
        posterior_score = "neg_kl"
        scores_per_item = get_grasps_score_hist(val_loader, model, val_dataset, posterior_score)
        # pickle.dump(scores_per_item, open('data/scores_per_item.pkl', 'wb'))
        for k, v in scores_per_item.items():
            pos_scores, neg_scores = v['pos'], v['neg']
            plt.hist(neg_scores, density=True,  histtype='barstacked', label=f'neg_grasps({neg_scores.shape[0]})', rwidth=0.5)
            plt.hist(pos_scores, density=True,  histtype='barstacked', label=f'pos_grasps({pos_scores.shape[0]})', rwidth=0.5)
            plt.title(f"Negative and Positive Grasps {posterior_score} of {k}")
            plt.legend()
            plt.show()

    if Visualization:
        val_fn = 'data/eval_batch.pth'
        for i, batch in enumerate(val_loader):
            print(f"Saving val batch: {i}")
            torch.save(batch, val_fn)
            break

        print(f"Reading val batch from file: {val_fn}")
        first_batch = torch.load(val_fn, map_location="cuda:0") 
        # posterior_score: None, "log_prob", "ent", "neg_kl", "pred_pose_transl_var", "pred_log_var", "pred_pose_angle_var"
        # mode: "viz_transl_dist", "viz_grasps_wo_scores", "viz_grasps_with_scores", "viz_neg_pos_hist", "filter_with_eval", "filter_with_prob"
        visualize(first_batch, mode="viz_grasps_with_scores", num_samples=20, grasp_flow_n_samples=30, posterior_score="pred_pose_transl_var")
    
    if MAAD:
        val_fn = 'data/eval_batch.pth'
        for i, batch in enumerate(val_loader):
            print(f"Saving val batch: {i}")
            torch.save(batch, val_fn)
            break

        print(f"Reading val batch from file: {val_fn}")
        first_batch = torch.load(val_fn, map_location="cuda:0") 
        compute_maad(first_batch)
