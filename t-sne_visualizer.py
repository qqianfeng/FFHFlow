import argparse
import torch
import os
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import pickle

import open3d as o3d
import sys
sys.path.insert(0,os.path.join(os.path.expanduser('~'),'workspace/normalizing-flows'))
import subprocess

from ffhflow.configs import get_config
from ffhflow.datasets import FFHDataModule
from ffhflow.utils.metrics import maad_for_grasp_distribution, maad_for_grasp_distribution_reversed
from ffhflow.utils.grasp_data_handler import GraspDataHandlerVae

from ffhflow.ffhflow_cnf import FFHFlowCNF
from ffhflow.ffhflow_lvm import FFHFlowLVM

def save_batch_to_file(batch):
    torch.save(batch, "data/eval_batch.pth")

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

if "cnf" in args.model_cfg:
    model = FFHFlowCNF.load_from_checkpoint(ckpt_path, cfg=cfg)
else:
    model = FFHFlowLVM.load_from_checkpoint(ckpt_path, cfg=cfg)
model.eval()

# val_loader = ffh_datamodule.val_dataloader(shuffle=True)
val_loader = ffh_datamodule.val_dataloader()
val_dataset = ffh_datamodule.val_dataset()

grasp_data_path = os.path.join(cfg.DATASETS.PATH, cfg.DATASETS.GRASP_DATA_NANE)
grasp_data = GraspDataHandlerVae(grasp_data_path)

path2dict = '/data/net/userstore/qf/ffhflow_real_world_exp/real_world_data_dict.pkl'
with open(path2dict, 'rb') as f:
    real_world_data_dict = pickle.load(f)

def load_obj():
    pass

if not load_offline_t_sne:
    ##### Run evaluation to get t-SNE features
    obj_name_list = []
    cond_feat_list = []
    print(len(val_loader))

    with torch.no_grad():
        for name, paths in real_world_data_dict.items():
            print('process ', name)
            for path in paths:
                obj_pcd = o3d.io.read_point_cloud(path)
                pc_tensor = torch.from_numpy(np.asarray(obj_pcd.points))
                pc_center = obj_pcd.get_center()
                obj_pcd.translate(-pc_center)
                pc_tensor.to('cuda')

                torch.save(pc_tensor, '/home/qf/pc_tensor.pt')
                subprocess.run(['/home/qf/workspace/ffhflow/scripts/bps_encode.sh',], shell=True)
                enc_dict = torch.load('/home/qf/enc_dict.pt')
                bps = enc_dict['dists'] # bps as tensor

                out, cond_feat = model.sample_in_experiment(bps, num_samples=100,return_cond_feat=True)
                cond_feat = cond_feat.cpu().numpy()
                obj_name_list.append(np.asarray([name])[:, np.newaxis])
                cond_feat_list.append(cond_feat)
            # print('take time one batch', time()-start)


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
    'sugar_box':[255,0,0],
    'apple':[0,255,0],
    'tomato_soup_can':[0,0,255],
    'pudding_box':[0,255,255],
    'metal_mug':[255,0,255],
    'mustard_container':[255,255,0],
    'baseball':[155,155,155],
    'foam_brick':[200,200,200]
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
