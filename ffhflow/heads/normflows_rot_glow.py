""" Grasp Glow based on normflows package
"""

# Import required packages
import torch
import torchvision as tv
import numpy as np
import sys
sys.path.insert(0,'/home/yb/workspace/normalizing-flows')
print(sys.path)
import normflows as nf
from yacs.config import CfgNode

from matplotlib import pyplot as plt
from tqdm import tqdm

# Set up model
class ConditionalGlow():
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 flow_layers,
                 res_num_layers,
                 context_features,
                 ) -> None:

        # Define flows
        torch.manual_seed(0)
        hidden_channels = hidden_dim
        split_mode = 'channel'
        # True is affine / False is addctive
        scale = True

        # # Construct flow model with the multiscale architecture
        # model = nf.MultiscaleFlow(q0, flows, merges)
        q0 = []
        flows = []
        for j in range(flow_layers):
            flows += [nf.flows.ConditionalGlowBlock(input_dim=input_dim, # 4
                                                        hidden_dim=hidden_channels,
                                                        context_feature=context_features,
                                                        num_blocks=res_num_layers,
                                                        split_mode=split_mode, scale=scale)]

        q0 = nf.distributions.GaussianMixture(n_modes=1,dim=input_dim,
                                                loc=np.zeros((1,input_dim)),trainable=True)
        # q0 = nf.distributions.StandardNormal(shape=input_dim)
        # Construct flow model with the multiscale architecture
        model = nf.ConditionalNormalizingFlow(q0, flows)
        # Move model on GPU if available
        enable_cuda = True
        device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
        self.model = model.to(device)
