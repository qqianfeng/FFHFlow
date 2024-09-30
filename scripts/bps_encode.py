# Simple script to run bps encoding because its dependency is in conflict with ffhflow pacakge.

import bps_torch.bps as b_torch
import os
import numpy as np
import torch

# Init BPS-Encoder
# modify path there where basis_point_set.npy is located in dataset.
bps_path = '/data/net/userstore/qf/hithand_data/data/ffhnet-data/basis_point_set.npy'
bps_np = np.load(bps_path)
bps = b_torch.bps_torch(custom_basis=bps_np)

# path to save tensor of point cloud "pc_tensor.pt" and encoded dist from bps "enc_dict.pt"
path2bps = '/home/qf/pc_tensor.pt'
path2enc_dict = '/home/qf/enc_dict.pt'
pc_tensor = torch.load(path2bps)
enc_dict = bps.encode(pc_tensor)

torch.save(enc_dict, path2enc_dict)