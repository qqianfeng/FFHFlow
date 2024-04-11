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
parser.add_argument('--model_cfg', type=str, default='/data/net/userstore/qf/ffhflow_models/ffhflow_lvm/hparams.yaml', help='Path to config file')
# parser.add_argument('--root_dir', type=str, default='checkpoints', help='Directory to save logs and checkpoints')
parser.add_argument('--ckpt_path', type=str, default='/data/net/userstore/qf/ffhflow_models/ffhflow_lvm/epoch=16-step=199999.ckpt', help='Directory to save logs and checkpoints')

args = parser.parse_args()

load_offline_t_sne = False
# Set up cfg
cfg = get_config(args.model_cfg)

# configure dataloader
ffh_datamodule = FFHDataModule(cfg)

# Setup PyTorch Lightning Trainer
ckpt_path = args.ckpt_path

# model = NormflowsFFHFlowPosEncWithTransl.load_from_checkpoint(ckpt_path, cfg=cfg)
model = NormflowsFFHFlowPosEncWithTransl_LVM.load_from_checkpoint(ckpt_path, cfg=cfg)
model.eval()

# val_loader = ffh_datamodule.val_dataloader(shuffle=True)
val_loader = ffh_datamodule.val_dataloader()
val_dataset = ffh_datamodule.val_dataset()

grasp_data_path = os.path.join(cfg.DATASETS.PATH, cfg.DATASETS.GRASP_DATA_NANE)
grasp_data = GraspDataHandlerVae(grasp_data_path)


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

                out, cond_feat = model.sample_in_experiment(bps, idx, num_samples=100,return_cond_feat=True)
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
