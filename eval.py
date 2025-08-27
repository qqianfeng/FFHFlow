import argparse
import math
import os, h5py

import matplotlib.pyplot as plt
import numpy as np
import torch
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


def compute_maad(batch, val_dataset):
    transl_loss_sum = 0
    rot_loss_sum = 0
    joint_loss_sum = 0
    coverage_sum = 0
    # print(f"len(val_loader): {len(val_loader)}")
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
            # palm_poses, joint_confs, num_pos = grasp_data.get_grasps_for_object(obj_name=batch['obj_name'][idx],outcome='positive')
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


def visualize(batch, mode, latent_flow_n_samples=100, grasp_flow_n_samples=30, posterior_score=None, o3d=True, avg_grasps=False):
    # print(f"len(val_loader): {len(val_loader)}")
    with torch.no_grad():
        num_obj = len(batch['obj_name'])
        neg_scores, pos_scores = [], []
        for obj_idx in range(num_obj):
            if mode == "viz_transl_dist":
                predicted_grasps = model.sample(batch, 
                                                obj_idx, 
                                                num_samples=latent_flow_n_samples,
                                                grasp_flow_n_samples=grasp_flow_n_samples,
                                                posterior_score=posterior_score)
                predicted_grasps = model.sort_and_filter_grasps(predicted_grasps, perc=0.99, return_arr=False)
                plot_transl_dist(predicted_grasps['pred_pose_transl']) 
            elif mode == "viz_grasps_wo_scores":
                predicted_grasps = model.sample(batch, 
                                                idx=obj_idx, 
                                                num_samples=latent_flow_n_samples,
                                                avg_grasps=avg_grasps,
                                                grasp_flow_n_samples=grasp_flow_n_samples,
                                                posterior_score=posterior_score)
                # predicted_grasps = model.sort_and_filter_grasps(predicted_grasps, perc=0.5)
                model.show_grasps(batch['pcd_path'][obj_idx], predicted_grasps, -1)
                # show gt grasps
                # grasps_gt = val_dataset.get_grasps_from_pcd_path(batch['pcd_path'][obj_idx])
                # model.show_gt_grasps(batch['pcd_path'][obj_idx], grasps_gt, obj_idx+300,frame_size=0.015, obj_name=batch['obj_name'][obj_idx])
            elif mode == "viz_grasps_w_hands":
                predicted_grasps = model.sample(batch, 
                                                idx=obj_idx, 
                                                num_samples=latent_flow_n_samples,
                                                avg_grasps=avg_grasps,
                                                grasp_flow_n_samples=grasp_flow_n_samples,
                                                posterior_score=posterior_score)
                model.show_grasps(batch['pcd_path'][obj_idx], predicted_grasps, o3d=o3d, w_hands=True)
            elif mode == "viz_grasps_w_scores":
                predicted_grasps = model.sample(batch, 
                                                idx=obj_idx, 
                                                num_samples=latent_flow_n_samples,
                                                avg_grasps=avg_grasps,
                                                grasp_flow_n_samples=grasp_flow_n_samples,
                                                posterior_score=posterior_score)
                model.show_grasps(batch['pcd_path'][obj_idx], predicted_grasps, w_hands=False)
            elif mode == "viz_neg_pos_hist":
                posterior_score = "neg_kl"
                pos_scores, neg_scores = get_scores_per_item(model, posterior_score, batch, obj_idx)
                plt.hist(neg_scores, density=True,  histtype='barstacked', label=f'neg_grasps({neg_scores.shape[0]})', rwidth=0.5)
                plt.hist(pos_scores, density=True,  histtype='barstacked', label=f'pos_grasps({pos_scores.shape[0]})', rwidth=0.5)
                plt.title(f"Negative and Positive Grasps {posterior_score} of {batch['obj_name'][obj_idx]}")
                plt.legend()
                plt.show()
            elif mode == "filter_with_eval":
                predicted_grasps = model.sample(batch, 
                                                idx=obj_idx, 
                                                num_samples=latent_flow_n_samples,
                                                avg_grasps=avg_grasps,
                                                grasp_flow_n_samples=grasp_flow_n_samples,
                                                posterior_score=posterior_score)
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
                model.show_grasps(original_path, filtered_grasps, obj_idx, prob=prob)
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


def load_bps_data(bps_data_path):    
    for obj_folder in tqdm(os.listdir(bps_data_path)):
        print(f"Loading {obj_folder}")
        for npy_file in os.listdir(os.path.join(bps_data_path, obj_folder)):
            bps_data = np.load(os.path.join(bps_data_path, obj_folder, npy_file)).reshape(1, -1)
            if 'bps_data_array' in locals():
                bps_data_array = np.concatenate([bps_data_array, bps_data], axis=0)
            else:
                bps_data_array = bps_data
    return bps_data_array

def compute_posterior_from_pcd(model, bps_data_array, posterior_score, num_samples, batch_size=128):
    bps_data_array = torch.tensor(bps_data_array).to('cuda')
    res_scores_flow2_max = np.zeros(bps_data_array.shape[0])
    res_scores_flow2 = np.zeros(bps_data_array.shape[0])
    res_scores_flow1 = np.zeros(bps_data_array.shape[0])
    for i in range(0, bps_data_array.shape[0], batch_size):
        if i + batch_size > bps_data_array.shape[0]:
            bps_data_array_tmp = bps_data_array[i:]
        else:
            bps_data_array_tmp = bps_data_array[i:i+batch_size]
        # print(f"bps_data_array.shape: {bps_data_array_tmp.shape}")
        batch = {'bps_object': bps_data_array_tmp}
        out = model.sample(batch, num_samples=num_samples, posterior_score=posterior_score)
        if i + batch_size > bps_data_array.shape[0]:
            remaining = bps_data_array.shape[0] - i
            res_scores_flow2[i:] = out['log_prob'].reshape((remaining, -1)).mean(axis=1)
            res_scores_flow1[i:] = out['prior_log_prob'].reshape((remaining, -1)).mean(axis=1)
        else:
            res_scores_flow2[i:i+batch_size] = out['log_prob'].reshape((batch_size, -1)).mean(axis=1)
            res_scores_flow1[i:i+batch_size] = out['prior_log_prob'].reshape(batch_size, -1).mean(axis=1)

    return res_scores_flow1, res_scores_flow2


def save_grasp_pickle(batch):
    pcd_data = h5py.File(path2pcd_trans, 'r')
    res = {'method': 'FFHFlow-lvm',
            'desc': 'grasp generation using FFHFlow-lvm',
            'sample_grasp': {},
            }
    with torch.no_grad():
        for idx in range(len(batch['obj_name'])):
            out = model.sample(batch, idx=idx, num_samples=100)
            obj_trans = pcd_data[batch['obj_name'][idx]]
            id_str = batch['pcd_path'][idx][-7:-4]
            trans_name = batch['obj_name'][idx]+'_pcd' + id_str + '_mesh_to_centroid'
            centroid_T_mesh = obj_trans[trans_name]
            centroid_T_mesh = utils.hom_matrix_from_pos_quat_list(centroid_T_mesh)
            mesh_T_centroid = np.linalg.inv(centroid_T_mesh)

            grasps_mesh = np.zeros((out['pred_pose_transl'].shape[0],4,4))
            joints = np.zeros((out['pred_pose_transl'].shape[0],15))
            log_probs = np.zeros((out['log_prob'].shape[0],1))
            prior_log_probs = np.zeros((out['prior_log_prob'].shape[0],1))
            for i in range(args.num_samples):
                grasp_tmp = np.zeros((4,4))
                grasp_tmp[:3,:3] = out['rot_matrix'][i]
                grasp_tmp[:3,3] = out['transl'][i]
                grasp_tmp[-1,-1] = 1
                grasp_mesh = np.matmul(mesh_T_centroid, grasp_tmp)
                joints[i] = out['joint_conf'][i]
                grasps_mesh[i] = grasp_mesh
                log_probs[i] = out['log_prob'][i]
                prior_log_probs[i] = out['prior_log_prob'][i]

            if batch['obj_name'][idx] not in res['sample_grasp']:
                res['sample_grasp'][batch['obj_name'][idx]] = {}
                res['sample_grasp'][batch['obj_name'][idx]]['grasp'] = grasps_mesh
                res['sample_grasp'][batch['obj_name'][idx]]['joint'] = joints
                res['sample_grasp'][batch['obj_name'][idx]]['log_probs'] = log_probs
                res['sample_grasp'][batch['obj_name'][idx]]['prior_log_probs'] = prior_log_probs

            else:
                res['sample_grasp'][batch['obj_name'][idx]]['grasp'] = np.concatenate((res['sample_grasp'][batch['obj_name'][idx]]['grasp'],grasps_mesh),axis=0)
                res['sample_grasp'][batch['obj_name'][idx]]['joint'] = np.concatenate((res['sample_grasp'][batch['obj_name'][idx]]['joint'],joints),axis=0)
                res['sample_grasp'][batch['obj_name'][idx]]['log_probs'] = np.concatenate((res['sample_grasp'][batch['obj_name'][idx]]['log_probs'],log_probs),axis=0)
                res['sample_grasp'][batch['obj_name'][idx]]['prior_log_probs'] = np.concatenate((res['sample_grasp'][batch['obj_name'][idx]]['prior_log_probs'],prior_log_probs),axis=0)
            # model.show_grasps(batch['pcd_path'][idx], out, idx)

    pickle.dump(res, open('res_flowlvm_wPriorFlow.pkl', 'wb'))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Probabilistic skeleton lifting training code')
    parser.add_argument('--model_cfg', type=str, default='/data/net/userstore/qf/ffhflow_models/ffhflow_lvm/hparams.yaml', help='Path to config file')
    parser.add_argument('--ckpt_path', type=str, default='/data/net/userstore/qf/ffhflow_models/ffhflow_lvm/epoch=16-step=199999.ckpt', help='Directory to save logs and checkpoints')
    parser.add_argument('--num_samples', type=float, default=100, help='Number of grasps to be generated for evaluation.')

    args = parser.parse_args()
    Visualization, MAAD, Grasps_Score, Shapes_Score, SaveGrasp = False, False, False, False, True
    cfg = get_config(args.model_cfg)

    path2pcd_trans = '/home/qf/Downloads/ffhnet-data/pcd_transforms.h5'

    # configure dataloader
    cfg["TRAIN"]["BATCH_SIZE"] = 64 # *8
    ffh_datamodule = FFHDataModule(cfg)

    # Setup PyTorch Lightning Trainer
    ckpt_path = args.ckpt_path
    if "cnf" in args.model_cfg:
        model = FFHFlowCNF.load_from_checkpoint(ckpt_path, cfg=cfg)
    else:
        model = FFHFlowLVM.load_from_checkpoint(ckpt_path, cfg=cfg)
    model.eval()

    # trn_loader = ffh_datamodule.train_dataloader()
    # trn_dataset = ffh_datamodule.train_dataset()
    # val_loader = ffh_datamodule.val_dataloader(shuffle=True)
    kit_val_loader = ffh_datamodule.val_dataloader()
    kit_val_dataset = ffh_datamodule.val_dataset()

    # path to save results
    grasp_data_path = os.path.join(cfg.DATASETS.PATH, cfg.DATASETS.GRASP_DATA_NANE)
    grasp_data = GraspDataHandlerVae(grasp_data_path)

    if Shapes_Score:
        num_samples = 100
        bs = 32
        posterior_score = None # None, "neg_kl", 'mutual_info'
        kit_bps_data_path = os.path.join(cfg.DATASETS.PATH, "eval", "bps")
        ycb_bps_data_path = os.path.join(cfg.DATASETS.PATH, "ycb_eval", "bps")
        similar_bps_array, novel_bps_array = load_bps_data(kit_bps_data_path), load_bps_data(ycb_bps_data_path)
        similar_scores1, similar_scores2 = compute_posterior_from_pcd(model, similar_bps_array, posterior_score, num_samples, batch_size=bs)
        novel_scores1, novel_scores2 = compute_posterior_from_pcd(model, novel_bps_array, posterior_score, num_samples, batch_size=bs)

        n_bins = 50
        plt.rc('legend',fontsize='xx-large')
        plt.subplot(3, 1, 1)
        # plt.style.use('seaborn-whitegrid') # nice and clean grid
        plt.hist(similar_scores1, density=True, bins=n_bins, alpha=0.5,facecolor = '#2ab0ff', edgecolor='#169acf', linewidth=0.5, label='Similar Patial Object Point Clouds') 
        plt.hist(novel_scores1, density=True,  bins=n_bins, alpha=0.5, facecolor='red', edgecolor='red', linewidth=0.5, label='Novel Patial Object Point Clouds') 
        plt.xlabel('Prior Flow Log Prob') 
        plt.legend()
        plt.title(f"Histogram of prior_flow_log_prob for Similar and Novel Shapes (averaged over {num_samples} latent samples)", fontsize="xx-large")

        plt.subplot(3, 1, 2)
        plt.hist(similar_scores2, density=True, bins=n_bins, alpha=0.5, facecolor = '#2ab0ff', edgecolor='#169acf', linewidth=0.5, label='Similar Patial Object Point Clouds') 
        plt.hist(novel_scores2, density=True, bins=n_bins, alpha=0.5, facecolor='red', edgecolor='red', linewidth=0.5, label='Novel Patial Object Point Clouds') 
        plt.grid(False)
        plt.xlabel('Grasp Flow Log Prob') 
        plt.title(f"Histogram of grasp_flow_log_prob for Similar and Novel Shapes (averaged over {num_samples} grasps)", fontsize="xx-large")
        plt.legend()
        
        plt.subplot(3, 1, 3)
        # plt.style.use('seaborn-whitegrid') # nice and clean grid
        plt.hist(similar_scores1+similar_scores2, density=True, bins=n_bins, alpha=0.5,facecolor = '#2ab0ff', edgecolor='#169acf', linewidth=0.5, label='Similar Patial Object Point Clouds') 
        plt.hist(novel_scores1+novel_scores2, density=True,  bins=n_bins, alpha=0.5, facecolor='red', edgecolor='red', linewidth=0.5, label='Novel Patial Object Point Clouds') 
        plt.xlabel('Prior Flow Log Prob + Grasp Flow Log Prob') 
        plt.legend()
        plt.title(f"Histogram of prior_flow_log_prob+grasp_flow_log_prob for Similar and Novel Shapes", fontsize="xx-large")

        plt.show()

    if Grasps_Score:
        # posterior_score: None, "log_prob", "ent", "neg_kl", "pred_transl_var", "pred_log_var", "pred_pose_angle_var"，"pred_post_pose_var", "pred_pose_var"
        posterior_score = "neg_kl"
        scores_per_item = get_grasps_score_hist(kit_val_loader, model, kit_val_dataset, posterior_score)
        # pickle.dump(scores_per_item, open('data/scores_per_item.pkl', 'wb'))
        for k, v in scores_per_item.items():
            pos_scores, neg_scores = v['pos'], v['neg']
            plt.hist(neg_scores, density=True,  histtype='barstacked', label=f'neg_grasps({neg_scores.shape[0]})', rwidth=0.5)
            plt.hist(pos_scores, density=True,  histtype='barstacked', label=f'pos_grasps({pos_scores.shape[0]})', rwidth=0.5)
            plt.title(f"Negative and Positive Grasps {posterior_score} of {k}")
            plt.legend()
            plt.show()

    if Visualization:
        save_first_batch = False
        val_fn = 'data/eval_batch.pth'
        if save_first_batch:
            for i, batch in enumerate(kit_val_loader):
                print(f"Saving val batch: {i}")
                torch.save(batch, val_fn)
                break

        print(f"Reading val batch from file: {val_fn}")
        first_batch = torch.load(val_fn, map_location="cuda:0") 
        # posterior_score: None, "log_prob", "ent", "neg_kl", "pred_transl_var", "pred_log_var", "pred_pose_angle_var"，"pred_post_pose_var", "pred_pose_var"
        # mode: "viz_transl_dist", "viz_grasps_wo_scores", "viz_grasps_w_scores", "viz_neg_pos_hist", "filter_with_eval", "filter_with_prob"
        visualize(first_batch, 
                  mode="viz_grasps_w_hands", 
                  latent_flow_n_samples=5, 
                  grasp_flow_n_samples=1, 
                  posterior_score=None, 
                  o3d=True,
                  avg_grasps=False)

    if MAAD:
        val_fn = 'data/eval_batch.pth'
        for i, batch in enumerate(kit_val_loader):
            print(f"Saving val batch: {i}")
            torch.save(batch, val_fn)
            break

        print(f"Reading val batch from file: {val_fn}")
        first_batch = torch.load(val_fn, map_location="cuda:0") 
        compute_maad(first_batch, kit_val_dataset)

    if SaveGrasp:   
        val_fn = 'data/eval_batch.pth'
        print(f"Reading val batch from file: {val_fn}")
        first_batch = torch.load(val_fn, map_location="cuda:0") 
        save_grasp_pickle(first_batch)
