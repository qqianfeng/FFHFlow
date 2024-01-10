""" Grasp Glow based on normflows package
"""

# Import required packages
import torch
from torch import nn
import torchvision as tv
import numpy as np
import sys
import os
sys.path.insert(0,os.path.join(os.path.expanduser('~'),'workspace/normalizing-flows'))

import normflows as nf
from normflows.distributions import BaseDistribution

# torchutils from nflows package
def sum_except_batch(x, num_batch_dims=1):
    """Sums all elements of `x` except for the first `num_batch_dims` dimensions."""
    reduce_dims = list(range(num_batch_dims, x.ndimension()))
    return torch.sum(x, dim=reduce_dims)

def split_leading_dim(x, shape):
    """Reshapes the leading dim of `x` to have the given shape."""
    new_shape = torch.Size(shape) + x.shape[1:]
    return torch.reshape(x, new_shape)
        
# Set up model
class ConditionalGlow():
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 flow_layers,
                 res_num_layers,
                 context_features,
                 base,
                 gmm_mode=1,
                 gmm_trainable=False,
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

        if base == "gmm":
            q0 = nf.distributions.GaussianMixture(n_modes=gmm_mode,
                                                  dim=input_dim,
                                                loc=np.zeros((1,input_dim)),trainable=gmm_trainable)
        elif base == "cond_gaussian":
            context_encoder = nn.Linear(context_features, int(input_dim*2))
            context_encoder.cuda()
            q0 = nf.distributions.ConditionalDiagGaussian(shape=input_dim, context_encoder=context_encoder)
        else:
            q0 = nf.distributions.StandardNormal(shape=input_dim)

        # Construct flow model with the multiscale architecture
        model = nf.ConditionalNormalizingFlow(q0, flows)
        
        # Move model on GPU if available
        enable_cuda = True
        device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
        self.model = model.to(device)
