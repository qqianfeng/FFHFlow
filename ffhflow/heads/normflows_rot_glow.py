""" Grasp Glow based on normflows package
"""

# Import required packages
import torch
from torch import nn
import numpy as np
import sys
import os
sys.path.insert(0,os.path.join(os.path.expanduser('~'),'workspace/normalizing-flows'))

import normflows as nf
from normflows.distributions import BaseDistribution
from normflows.flows import AffineCouplingBlock, InvertibleAffine, ActNorm
from normflows import nets

# torchutils from nflows package
def sum_except_batch(x, num_batch_dims=1):
    """Sums all elements of `x` except for the first `num_batch_dims` dimensions."""
    reduce_dims = list(range(num_batch_dims, x.ndimension()))
    return torch.sum(x, dim=reduce_dims)

def split_leading_dim(x, shape):
    """Reshapes the leading dim of `x` to have the given shape."""
    new_shape = torch.Size(shape) + x.shape[1:]
    return torch.reshape(x, new_shape)
        

def merge_leading_dims(x, num_dims):
    """Reshapes the tensor `x` such that the first `num_dims` dimensions are merged to one."""
    if num_dims > x.dim():
        raise ValueError(
            "Number of leading dims can't be greater than total number of dims."
        )
    new_shape = torch.Size([-1]) + x.shape[num_dims:]
    return torch.reshape(x, new_shape)


def repeat_rows(x, num_reps):
    """Each row of tensor `x` is repeated `num_reps` times along leading dimension."""
    shape = x.shape
    x = x.unsqueeze(1)
    x = x.expand(shape[0], num_reps, *shape[1:])
    return merge_leading_dims(x, num_dims=2)

class CondResampledGaussian(BaseDistribution):
    """
    Multivariate Gaussian distribution with diagonal covariance matrix,
    resampled according to a acceptance probability determined by a neural network conditioning on the contexts,
    see arXiv 1810.11428
    """
    def __init__(self, d, a, T, eps, trainable=True, bs_factor=1):
        """
        Constructor
        :param d: Dimension of Gaussian distribution
        :param a: Function returning the acceptance probability
        :param T: Maximum number of rejections
        :param eps: Discount factor in exponential average of Z
        :param bs_factor: Factor to increase the batch size during sampling
        """
        super().__init__()
        self.d = d
        self.a = a
        self.T = T
        self.eps = eps
        self.bs_factor = bs_factor
        self.register_buffer("Z", torch.tensor(-1.))
        if trainable:
            self.loc = nn.Parameter(torch.zeros(1, self.d))
            self.log_scale = nn.Parameter(torch.zeros(1, self.d))
        else:
            self.register_buffer("loc", torch.zeros(1, self.d))
            self.register_buffer("log_scale", torch.zeros(1, self.d))

    def forward(self, context=None, num_samples=1):
        t = 0
        eps = torch.zeros(num_samples, self.d, dtype=self.loc.dtype, device=self.loc.device)
        s = 0
        n = 0
        Z_sum = 0
        for i in range(self.T // self.bs_factor + 1):
            eps_ = torch.randn((num_samples * self.bs_factor, self.d),
                               dtype=self.loc.dtype, device=self.loc.device)
            acc = self.a(eps_)
            if self.training or self.Z < 0.:
                Z_sum = Z_sum + torch.sum(acc).detach()
                n = n + num_samples * self.bs_factor
            dec = torch.rand_like(acc) < acc
            for j, dec_ in enumerate(dec[:, 0]):
                if dec_ or t == self.T - 1:
                    eps[s, :] = eps_[j, :]
                    s = s + 1
                    t = 0
                else:
                    t = t + 1
                if s == num_samples:
                    break
            if s == num_samples:
                break
        z = self.loc + torch.exp(self.log_scale) * eps
        log_p_gauss = - 0.5 * self.d * np.log(2 * np.pi) \
                      - torch.sum(self.log_scale, 1)\
                      - torch.sum(0.5 * torch.pow(eps, 2), 1)
        acc = self.a(eps)
        if self.training or self.Z < 0.:
            eps_ = torch.randn((num_samples, self.d), dtype=self.loc.dtype, device=self.loc.device)
            Z_batch = torch.mean(self.a(eps_))
            Z_ = (Z_sum + Z_batch.detach() * num_samples) / (n + num_samples)
            if self.Z < 0.:
                self.Z = Z_
            else:
                self.Z = (1 - self.eps) * self.Z + self.eps * Z_
            Z = Z_batch - Z_batch.detach() + self.Z
        else:
            Z = self.Z
        alpha = (1 - Z) ** (self.T - 1)
        log_p = torch.log((1 - alpha) * acc[:, 0] / Z + alpha) + log_p_gauss
        return z, log_p

    def log_prob(self, z, context=None):
        eps = (z - self.loc) / torch.exp(self.log_scale)
        log_p_gauss = - 0.5 * self.d * np.log(2 * np.pi) \
                      - torch.sum(self.log_scale, 1) \
                      - torch.sum(0.5 * torch.pow(eps, 2), 1)
        acc = self.a(eps)
        if self.training or self.Z < 0.:
            eps_ = torch.randn_like(z)
            Z_batch = torch.mean(self.a(eps_))
            if self.Z < 0.:
                self.Z = Z_batch.detach()
            else:
                self.Z = (1 - self.eps) * self.Z + self.eps * Z_batch.detach()
            Z = Z_batch - Z_batch.detach() + self.Z
        else:
            Z = self.Z
        alpha = (1 - Z) ** (self.T - 1)
        log_p = torch.log((1 - alpha) * acc[:, 0] / Z + alpha) + log_p_gauss
        return log_p

    def estimate_Z(self, num_samples, num_batches=1):
        """
        Estimate Z via Monte Carlo sampling
        :param num_samples: Number of samples to draw per batch
        :param num_batches: Number of batches to draw
        """
        with torch.no_grad():
            self.Z = self.Z * 0.
            # Get dtype and device
            dtype = self.Z.dtype
            device = self.Z.device
            for i in range(num_batches):
                eps = torch.randn((num_samples, self.d), dtype=dtype,
                                  device=device)
                acc_ = self.a(eps)
                Z_batch = torch.mean(acc_)
                self.Z = self.Z + Z_batch.detach() / num_batches


class ResampledDistribution(BaseDistribution):
    """
    Resampling of a general distribution
    """
    def __init__(self, dist, a, T, eps, bs_factor=1):
        """
        Constructor
        :param dist: Distribution to be resampled
        :param a: Function returning the acceptance probability
        :param T: Maximum number of rejections
        :param eps: Discount factor in exponential average of Z
        :param bs_factor: Factor to increase the batch size during sampling
        """
        super().__init__()
        self.dist = dist
        self.a = a
        self.T = T
        self.eps = eps
        self.bs_factor = bs_factor
        self.register_buffer("Z", torch.tensor(-1.))

    def forward(self,  num_samples, context=None):
        t = 0
        z = None
        log_p_dist = None
        s = 0
        n = 0
        Z_sum = 0
        for i in range(self.T // self.bs_factor + 1):
            z_, log_prob_ = self.dist(num_samples * self.bs_factor, context=context)
            if i == 0:
                z = torch.zeros_like(z_[:num_samples])
                log_p_dist = torch.zeros_like(log_prob_[:num_samples])
            acc = self.a(z_)
            if self.training or self.Z < 0.:
                Z_sum = Z_sum + torch.sum(acc).detach()
                n = n + num_samples * self.bs_factor
            dec = torch.rand_like(acc) < acc
            for j, dec_ in enumerate(dec[:, 0]):
                if dec_ or t == self.T - 1:
                    z[s, ...] = z_[j, ...]
                    log_p_dist[s] = log_prob_[j]
                    s = s + 1
                    t = 0
                else:
                    t = t + 1
                if s == num_samples:
                    break
            if s == num_samples:
                break
        acc = self.a(z)
        if self.training or self.Z < 0.:
            z_, _ = self.dist(num_samples, context=context)
            Z_batch = torch.mean(self.a(z_))
            Z_ = (Z_sum + Z_batch.detach() * num_samples) / (n + num_samples)
            if self.Z < 0.:
                self.Z = Z_
            else:
                self.Z = (1 - self.eps) * self.Z + self.eps * Z_
            Z = Z_batch - Z_batch.detach() + self.Z
        else:
            Z = self.Z
        alpha = (1 - Z) ** (self.T - 1)
        log_p = torch.log((1 - alpha) * acc[:, 0] / Z + alpha) + log_p_dist
        return z, log_p

    def log_prob(self, z, context=None):
        log_p_dist = self.dist.log_prob(z, context)
        acc = self.a(z)
        if self.training or self.Z < 0.:
            z_, _ = self.dist(len(z), context=context)
            Z_batch = torch.mean(self.a(z_))
            if self.Z < 0.:
                self.Z = Z_batch.detach()
            else:
                self.Z = (1 - self.eps) * self.Z + self.eps * Z_batch.detach()
            Z = Z_batch - Z_batch.detach() + self.Z
        else:
            Z = self.Z
        alpha = (1 - Z) ** (self.T - 1)
        log_p = torch.log((1 - alpha) * acc[:, 0] / Z + alpha) + log_p_dist
        return log_p

    def estimate_Z(self, num_samples, context=None, num_batches=1):
        """
        Estimate Z via Monte Carlo sampling
        :param num_samples: Number of samples to draw per batch
        :param num_batches: Number of batches to draw
        """
        with torch.no_grad():
            self.Z = self.Z * 0.
            for i in range(num_batches):
                z, _ = self.dist(num_samples, context=context)
                acc_ = self.a(z)
                Z_batch = torch.mean(acc_)
                self.Z = self.Z + Z_batch.detach() / num_batches


class GlowBlock(nf.flows.Flow):
    """Glow: Generative Flow with Invertible 1x1 Convolutions, [arXiv: 1807.03039](https://arxiv.org/abs/1807.03039)

    One Block of the Glow model, comprised of

    - MaskedAffineFlow (affine coupling layer)
    - Invertible1x1Conv (dropped if there is only one channel)
    - ActNorm (first batch used for initialization)s
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_blocks,
        scale=True,
        scale_map="sigmoid",
        split_mode="channel",
        use_lu=True,
    ):
        super().__init__()
        self.flows = nn.ModuleList([])

        # Coupling layer -> TODO: Do we need this???
        num_param = 2 if scale else 1
        channels = input_dim
        hidden_channels = hidden_dim
        if "channel" == split_mode:
            channels_ = ((channels + 1) // 2,) + 2 * (hidden_channels,)
            channels_ += (num_param * (channels // 2),)
        elif "channel_inv" == split_mode:
            channels_ = (channels // 2,) + 2 * (hidden_channels,)
            channels_ += (num_param * ((channels + 1) // 2),)
        elif "checkerboard" in split_mode:
            channels_ = (channels,) + 2 * (hidden_channels,)
            channels_ += (num_param * channels,)
        else:
            raise NotImplementedError("Mode " + split_mode + " is not implemented.")
        # Because we don't use multi scale so first value from chennels_ is enough for us to use
        param_map = nets.ResidualNet(in_features=channels_[0], 
                                     out_features=input_dim, 
                                     hidden_features=hidden_dim,
                                     num_blocks=num_blocks)

        self.flows += [AffineCouplingBlock(param_map, scale, scale_map, split_mode)]
        # Invertible 1x1 convolution
        self.flows += [InvertibleAffine(input_dim, use_lu)]
        # Activation normalization
        # Modified according to glow in nflow
        self.flows += [ActNorm([input_dim])]

    def forward(self, z, context=None):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        if z.shape[0] != context.shape[0]:
            context = repeat_rows(context, num_reps=z.shape[0]//context.shape[0])
        for flow in self.flows:
            if str(flow) == 'ActNorm()':
                z, log_det = flow(z)
            elif str(flow) == 'InvertibleAffine()':
                z, log_det = flow(z)
            else:
                z, log_det = flow(z, context)
            log_det_tot += log_det
        return z, log_det_tot

    def inverse(self, z, context=None):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for i in range(len(self.flows) - 1, -1, -1):
            if str(self.flows[i]) == 'ActNorm()':
                z, log_det = self.flows[i].inverse(z)
            elif str(self.flows[i]) == 'InvertibleAffine()':
                z, log_det = self.flows[i].inverse(z)
            else:
                z, log_det = self.flows[i].inverse(z, context)
            log_det_tot += log_det
        return z, log_det_tot


# Set up model
class Glow():
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 flow_layers,
                 res_num_layers,
                 flow_config=None
                 ) -> None:

        if flow_config is None:
            base = "gaussian"
        else:
            gmm_mode=flow_config.GMM_MODE,
            gmm_trainable=flow_config.GMM_TRAINABLE

        # Define flows
        torch.manual_seed(0)
        split_mode = 'channel'
        # True is affine / False is addctive
        scale = True

        # # Construct flow model with the multiscale architecture
        # model = nf.MultiscaleFlow(q0, flows, merges)
        flows = []
        for _ in range(flow_layers):
            flows += [GlowBlock(input_dim=input_dim,
                                hidden_dim=hidden_dim,
                                num_blocks=res_num_layers,
                                split_mode=split_mode, 
                                scale=scale)]
            
        # gmm base
        if base == "gmm":
            q0 = nf.distributions.GaussianMixture(n_modes=gmm_mode,
                                                  dim=input_dim,
                                                  loc=np.zeros((1,input_dim)),
                                                  trainable=gmm_trainable)
        elif base == "gaussian":
            q0 = nf.distributions.StandardNormal(shape=input_dim)
        else:
            raise NotImplementedError(f"{base} is NOT implemented, only gaussian and gmm are available.")

        # Construct flow model with the multiscale architecture
        model = nf.NormalizingFlow(q0, flows)
        
        # Move model on GPU if available
        enable_cuda = True
        device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
        self.model = model.to(device)
                
class ConditionalGlow():
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 flow_layers,
                 res_num_layers,
                 context_features,
                 flow_config,
                 ) -> None:

        if flow_config is None:
            base = "gaussian"
        else:
            base = flow_config.BASE
            gmm_mode=flow_config.GMM_MODE,
            gmm_trainable=flow_config.GMM_TRAINABLE
            rsb_T = flow_config.RSB_T
            rsb_eps = flow_config.RSB_EPS
            rsb_acc_hidden = flow_config.RSB_ACC_HIDDEN   
            rsb_acc_layers = flow_config.RSB_ACC_LAYES

        # Define flows
        torch.manual_seed(0)
        hidden_channels = hidden_dim
        split_mode = 'channel'
        # True is affine / False is addctive
        scale = True

        # # Construct flow model with the multiscale architecture
        # model = nf.MultiscaleFlow(q0, flows, merges)
        flows = []
        for _ in range(flow_layers):
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
        elif base == "cond_rsb":   
            context_encoder = nn.Linear(context_features, int(input_dim*2))
            context_encoder.cuda()
            layers = [input_dim]
            layers += (rsb_acc_hidden, ) * rsb_acc_layers
            layers += [1]
            acceptance_func = nf.nets.MLP(layers, output_fn='sigmoid')
            acceptance_func.cuda()
            proposal_dist = nf.distributions.ConditionalDiagGaussian(shape=input_dim, context_encoder=context_encoder)
            q0 = ResampledDistribution(dist=proposal_dist, a=acceptance_func, T=rsb_T, eps=rsb_eps)
        elif base == "gaussian":
            q0 = nf.distributions.StandardNormal(shape=input_dim)
        else:
            raise NotImplementedError(f"The base distribution of {base} has not been implemented.")

        # Construct flow model with the multiscale architecture
        model = nf.ConditionalNormalizingFlow(q0, flows)
        
        # Move model on GPU if available
        enable_cuda = True
        device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
        self.model = model.to(device)
