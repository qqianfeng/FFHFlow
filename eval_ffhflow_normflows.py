import argparse
import torch
import os
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

import sys
# clone this repo https://github.com/qianbot/nflows
sys.path.insert(0,os.path.join(os.path.expanduser('~'),'workspace/normalizing-flows'))

from ffhflow.configs import get_config
from ffhflow.datasets import FFHDataModule
from ffhflow.utils.metrics import maad_for_grasp_distribution, maad_for_grasp_distribution_reversed
from ffhflow.utils.grasp_data_handler import GraspDataHandlerVae

from ffhflow.normflows_ffhflow_pos_enc_with_transl import NormflowsFFHFlowPosEncWithTransl, NormflowsFFHFlowPosEncWithTransl_LVM

def save_batch_to_file(batch):
    torch.save(batch, "eval_batch.pth")

def load_batch(path):
    return torch.load(path, map_location="cuda:0")

parser = argparse.ArgumentParser(description='Probabilistic skeleton lifting training code')
parser.add_argument('--model_cfg', type=str, default='checkpoints/flow_lvm_lr1e-4_best/hparams.yaml', help='Path to config file')
# parser.add_argument('--root_dir', type=str, default='checkpoints', help='Directory to save logs and checkpoints')
parser.add_argument('--ckpt_path', type=str, default='checkpoints/flow_lvm_lr1e-4_best/epoch=16-step=199999.ckpt', help='Directory to save logs and checkpoints')

args = parser.parse_args()
Visualization = False
MAAD = True
run_t_sne = False
load_offline_t_sne = False
# Set up cfg
cfg = get_config(args.model_cfg)

# configure dataloader
ffh_datamodule = FFHDataModule(cfg)

# Setup PyTorch Lightning Trainer
ckpt_path = args.ckpt_path

if "cnf" in args.model_cfg:
    model = NormflowsFFHFlowPosEncWithTransl.load_from_checkpoint(ckpt_path, cfg=cfg)
else:
    model = NormflowsFFHFlowPosEncWithTransl_LVM.load_from_checkpoint(ckpt_path, cfg=cfg)

model.eval()

# val_loader = ffh_datamodule.val_dataloader(shuffle=True)
val_loader = ffh_datamodule.val_dataloader()
val_dataset = ffh_datamodule.val_dataset()

save_path = '/home/yb/Documents/ffhflow_grasp'
grasp_data_path = os.path.join(cfg.DATASETS.PATH, cfg.DATASETS.GRASP_DATA_NANE)
grasp_data = GraspDataHandlerVae(grasp_data_path)

if run_t_sne:
    if not load_offline_t_sne:
        ##### Run evaluation to get t-SNE features
        obj_name_list = []
        cond_feat_list = []
        print(len(val_loader))

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                # start = time()
                for idx in range(len(batch['obj_name'])):
                    # palm_poses, joint_confs, num_pos = grasp_data.get_grasps_for_object(obj_name=batch['obj_name'][idx],outcome='negative')
                    # grasps_gt = val_dataset.get_grasps_from_pcd_path(batch['pcd_path'][idx])

                    out, cond_feat = model.sample(batch, idx, num_samples=100, return_arr=True, return_cond_feat=True)
                    cond_feat = cond_feat.cpu().numpy()
                    obj_name_list.append(np.asarray([batch['obj_name'][idx]])[:, np.newaxis])
                    cond_feat_list.append(cond_feat)
                # print('take time one batch', time()-start)
                if i > 100: #100
                    break

        obj_names = np.concatenate(obj_name_list, axis=0)
        cond_feats = np.concatenate(cond_feat_list, axis=0).astype(np.float64)

        print('generating t-SNE plot...')
        tsne = TSNE(random_state=0)
        tsne_output = tsne.fit_transform(cond_feats)
        print('finish tsne fit transform')
        np.save('tsne_output.npy',tsne_output)
        np.save('obj_names.npy',obj_names)
    else:
        tsne_output = np.load('tsne_output.npy')
        obj_names = np.load('obj_names.npy')
    # scale and move the coordinates so they fit [0; 1] range
    def scale_to_01_range(x):
        # compute the distribution range
        value_range = (np.max(x) - np.min(x))

        # move the distribution so that it starts from zero
        # by extracting the minimal value from all its values
        starts_from_zero = x - np.min(x)

        # make the distribution fit [0; 1] by dividing by its range
        return starts_from_zero / value_range

    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne_output[:, 0]
    ty = tsne_output[:, 1]
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # initialize a matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors_per_name = {
        'kit_BakingSoda':[150,30,230],
        'kit_BathDetergent':[155,155,155],
        'kit_BroccoliSoup':[255,0,0],
        'kit_CoughDropsLemon':[0,255,0],
        'kit_Curry':[0,0,255],
        'kit_FizzyTabletsCalcium':[0,255,255],
        'kit_InstantSauce':[255,0,255],
        'kit_NutCandy':[255,255,0],
        'kit_PotatoeDumplings':[0,100,100],
        'kit_Sprayflask':[100,0,100],
        'kit_TomatoSoup':[0,100,200],
        'kit_YellowSaltCube2':[200,200,100],
    }
    # for every class, we'll add a scatter plot separately
    for obj_name, color in colors_per_name.items():
        # find the samples of the current class in the data
        indices = []
        for i, l in enumerate(obj_names):
            if l[0] == obj_name:
                indices.append(i)
        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        # convert the class color to matplotlib format
        color = [i/255. for i in color]

        # add a scatter plot with the corresponding color and label
        ax.scatter(current_tx, current_ty, c=np.array([color]), label=obj_name)

    # build a legend using the labels we set previously
    ax.legend(loc='best')
    plt.show()
    #####################


if MAAD:
    ##### MAAD Metrics #######
    import math

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
        batch = load_batch('eval_batch.pth')
        # batch = load_batch('eval_batch_correct_eval.pth')
        print(batch['obj_name'])
        for idx in range(len(batch['obj_name'])):
            palm_poses, joint_confs, num_pos = grasp_data.get_grasps_for_object(obj_name=batch['obj_name'][idx],outcome='positive')
            grasps_gt = val_dataset.get_grasps_from_pcd_path(batch['pcd_path'][idx])

            # out = model.sample(batch['bps_object'][idx], num_samples=100)
            out = model.sample(batch, idx, num_samples=100)

            transl_loss, rot_loss, joint_loss, coverage = maad_for_grasp_distribution(out, grasps_gt)
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
        print('transl_loss_sum:', transl_loss_sum)
        print('rot_loss_sum:', rot_loss_sum)
        print('joint_loss_sum:', joint_loss_sum)
        print('coverage', coverage_mean)
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
            num_gt_grasps = grasps_gt['transl'].shape[0]
            # out = model.sample(batch['bps_object'][idx], num_samples=grasps_gt['rot_matrix'].shape[0])
            out = model.sample(batch, idx, num_samples=num_gt_grasps)
            print('visualize',batch['obj_name'][idx] )

            # plot value distribution to show multi-modality
            if torch.is_tensor(out['rot_matrix']):
                out_np = {}
                for key, value in out.items():
                    out_np[key] = value.cpu().data.numpy()
            import matplotlib.pyplot as plt
            X = np.linspace(-5.0, 5.0, out['pred_angles'].shape[0])
            fig, ax = plt.subplots()
            ax.set_title("PDF from Template")
            # ax.hist(data, density=True, bins=100)
            ax.hist(out_np['pred_pose_transl'][:,0], label='1')
            ax.hist(out_np['pred_pose_transl'][:,1], label='2')
            ax.hist(out_np['pred_pose_transl'][:,2], label='3')
            ax.legend()
            fig.show()

            # rng = np.random.RandomState(10)  # deterministic random data
            # a = np.hstack((rng.normal(size=1000),
            #             rng.normal(loc=5, scale=2, size=1000)))
            # _ = plt.hist(a, bins='auto')  # arguments are passed to np.histogram
            # plt.title("Histogram with 'auto' bins")
            # Text(0.5, 1.0, "Histogram with 'auto' bins")
            # plt.show()

            # If we need to save the results for FFHEvaluator
            # with open('flow_grasps.pkl', 'wb') as fp:
            #     pickle.dump(out, fp, protocol=2)
            # with open('data.pkl', 'wb') as fp:
            #     pickle.dump([batch['bps_object'][idx], batch['pcd_path'][idx], batch['obj_name'][idx]], fp, protocol=2)

            model.show_grasps(batch['pcd_path'][idx], out, idx,frame_size=0.015, obj_name=batch['obj_name'][idx])
            # filtered_out = model.sort_and_filter_grasps(out, perc=0.5)
            # model.show_grasps(batch['pcd_path'][idx], filtered_out, idx+100)
            # filtered_out = model.sort_and_filter_grasps(out, perc=0.1, return_arr=False)
            # gt index till 22
            # model.show_gt_grasps(batch['pcd_path'][idx], grasps_gt, idx+300,frame_size=0.015, obj_name=batch['obj_name'][idx])


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
