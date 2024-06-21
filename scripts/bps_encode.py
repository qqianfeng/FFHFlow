import bps_torch.bps as b_torch
import os
import numpy as np
import torch
from time import time

# Init BPS-Encoder
bps_path = '/data/net/userstore/qf/hithand_data/data/ffhnet-data/basis_point_set.npy'
bps_np = np.load(bps_path)
bps = b_torch.bps_torch(custom_basis=bps_np)

start = time()
pc_tensor = torch.load('/home/qf/pc_tensor.pt')
enc_dict = bps.encode(pc_tensor)
print('bps takes from script', time()-start)

torch.save(enc_dict, '/home/qf/enc_dict.pt')