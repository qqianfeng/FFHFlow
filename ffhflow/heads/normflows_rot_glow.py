""" Grasp Glow based on normflows package
"""

# Import required packages
import torch
import torchvision as tv
import numpy as np
import normflows as nf
from yacs.config import CfgNode

from matplotlib import pyplot as plt
from tqdm import tqdm

# Set up model
class GraspGlow():
    def __init__(self,
                 channels,
                 hidden_channels,
                 num_layers,
                 ) -> None:

        # Define flows
        num_layers = 3
        K = 16
        torch.manual_seed(0)
        input_shape = (3, 32, 32)
        n_dims = np.prod(input_shape)
        channels = 3
        hidden_channels = 256
        split_mode = 'channel'
        scale = True

        # Set up flows, distributions and merge operations
        q0 = []
        merges = []
        flows = []
        for i in range(num_layers):
            flows_ = []
            for j in range(K):
                flows_ += [nf.flows.GlowBlock(channels * 2 ** (num_layers + 1 - i), hidden_channels,
                                            split_mode=split_mode, scale=scale)]
            flows_ += [nf.flows.Squeeze()]
            flows += [flows_]
            if i > 0:
                merges += [nf.flows.Merge()]
                latent_shape = (input_shape[0] * 2 ** (num_layers - i), input_shape[1] // 2 ** (num_layers - i),
                                input_shape[2] // 2 ** (num_layers - i))
            else:
                latent_shape = (input_shape[0] * 2 ** (num_layers + 1), input_shape[1] // 2 ** num_layers,
                                input_shape[2] // 2 ** num_layers)

            q0 += [nf.distributions.(latent_shape, num_classes)]


        # Construct flow model with the multiscale architecture
        model = nf.MultiscaleFlow(q0, flows, merges)

        # Move model on GPU if available
        enable_cuda = True
        device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
        self.model = model.to(device)


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
