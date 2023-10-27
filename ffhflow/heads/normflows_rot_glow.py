""" Grasp Glow based on normflows package
"""

# Import required packages
import torch
import torchvision as tv
import numpy as np
import sys
sys.path.insert(0,'/home/qf/workspace/normalizing-flows')
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
                 context_feature,
                 ) -> None:

        # Define flows
        num_multi_scale_layer = 3 # multi-scale layer
        torch.manual_seed(0)
        input_shape = (1, input_dim)
        n_dims = np.prod(input_shape)
        channels = 1
        hidden_channels = hidden_dim
        split_mode = 'channel'
        scale = True

        # Set up flows, distributions and merge operations
        q0 = []
        merges = []
        flows = []
        for i in range(num_multi_scale_layer):
            flows_ = []
            for j in range(flow_layers):
                flows_ += [nf.flows.ConditionalGlowBlock(channels=channels * 2 ** (num_multi_scale_layer + 1 - i),
                                                         hidden_channels=hidden_channels,
                                                         context_feature=context_feature,
                                                         num_blocks=res_num_layers,
                                                         split_mode=spli t_mode, scale=scale)]
            flows_ += [nf.flows.Squeeze()]
            flows += [flows_]
            if i > 0:
                merges += [nf.flows.Merge()]
                latent_shape = (input_shape[0] * 2 ** (num_multi_scale_layer - i), input_shape[1] // 2 ** (num_multi_scale_layer - i))
            else:
                latent_shape = (input_shape[0] * 2 ** (num_multi_scale_layer + 1), input_shape[1] // 2 ** num_multi_scale_layer)
            # q0 += [nf.distributions.ClassCondDiagGaussian(latent_shape, num_classes)]
            # This should be the same as standard norm if we set fixed mean and variance.
            q0 += [nf.distributions.DiagGaussian(latent_shape,trainable=False)]

        # Construct flow model with the multiscale architecture
        model = nf.MultiscaleFlow(q0, flows, merges)

        # Move model on GPU if available
        enable_cuda = True
        device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
        self.model = model.to(device)

"""
optimizer = torch.optim.Adamax(model.parameters(), lr=1e-3, weight_decay=1e-5)

for i in tqdm(range(max_iter)):
    try:
        x, y = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        x, y = next(train_iter)
    optimizer.zero_grad()
    loss = model.forward_kld(x.to(device), y.to(device))

    if ~(torch.isnan(loss) | torch.isinf(loss)):
        loss.backward()
        optimizer.step()

    loss_hist = np.append(loss_hist, loss.detach().to('cpu').numpy())
    del(x, y, loss)
"""