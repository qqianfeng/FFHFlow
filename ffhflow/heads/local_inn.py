import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from nflows.flows import ConditionalGlow
from torch import Tensor
from yacs.config import CfgNode

from ffhflow.utils.utils import rot_matrix_from_ortho6d


class PositionalEncoding(nn.Module):
    """
    Original from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self,  d_model: int):
        super().__init__()
        self.d_model = d_model
        self.div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[3, batch_size]``
        """
        batch_size = x.shape[1]
        pe = torch.zeros(3, batch_size, self.d_model)
        for i in range(batch_size):
            pe[:, i, 0::2] = torch.sin(x[:,i].reshape(3,1) * self.div_term)

        return pe

class LocalInnFlow(nn.Module):
    """
    """
    def __init__(self, cfg: CfgNode):
        """
        Args:
            cfg (CfgNode): Model config as yacs CfgNode.
        """
        super(LocalInnFlow, self).__init__()
        self.cfg = cfg
        # No conditional inputs for local_inn
        self.flow = ConditionalGlow(cfg.MODEL.FLOW.DIM, cfg.MODEL.FLOW.LAYER_HIDDEN_FEATURES,
                                    cfg.MODEL.FLOW.NUM_LAYERS, cfg.MODEL.FLOW.LAYER_DEPTH,
                                    context_features=None)
        self.pos_enc = PositionalEncoding(d_model=4)

    def log_prob(self, batch: Dict) -> Tuple:
        """
        Compute the log-probability of a set of samples given a batch of images.
        Args:
            smpl_params (Dict): Dictionary containing a set of SMPL parameters.
            feats (torch.Tensor): Conditioning features of shape (N, C).
        Returns:
            log_prob (torch.Tensor): Log-probability of the samples with shape (B, N).
            z (torch.Tensor): The Gaussian latent corresponding to each sample with shape (B, N, 144).
        """

        # feats = feats.float()  # same as to(torch.float32)  [64,1024]

        samples = batch['rot_matrix']  # [batch_size,3,3]
        batch_size = samples.shape[0]

        grasp_samples = samples.reshape(batch_size, -1).to(feats.dtype)
        log_prob, z = self.flow.log_prob(grasp_samples)
        log_prob = log_prob.reshape(batch_size, 1)
        z = z.reshape(batch_size, 1, -1)
        return log_prob, z

    def forward(self, batch, num_samples: Optional[int] = None, z: Optional[torch.Tensor] = None) -> Tuple:
        """
        Run a forward pass of the model.
        If z is not specified, then the model randomly draws num_samples samples for each image in the batch.
        Otherwise the batch of latent vectors z is transformed using the Conditional Normalizing Flows model.
        Args:
            feats (torch.Tensor): Conditioning features of shape (N, C).
            num_samples (int): Number of samples to draw per image.
            z (torch.Tensor): A batch of latent vectors of shape (B, N, 144).
        Returns:
            pred_smpl_params (Dict): Dictionary containing the predicted set of SMPL parameters.
            pred_cam (torch.Tensor): Predicted camera parameters with shape (B, N, 3).
            log_prob (torch.Tensor): Log-probability of the samples with shape (B, N).
            z (torch.Tensor): Either the input z or the randomly drawn batch of latent Gaussian vectors.
            pred_pose_6d (torch.Tensor): Predicted pose vectors in the 6-dimensional representation.
        """
        # feats = feats.float()

        batch_size = batch[0].shape[0]

        # Generates samples from the distribution together with their log probability.
        samples, log_prob, z = self.flow.sample_and_log_prob(num_samples, noise=z)

        z = z.reshape(batch_size, num_samples, -1)
        pred_params = samples.reshape(batch_size, num_samples, -1)

        pred_pose = pred_params[:, :, :6]
        pred_pose_6d = pred_pose.clone()
        pred_pose = rot_matrix_from_ortho6d(pred_pose.reshape(batch_size * num_samples, -1))

        # pred_pose_transl = pred_params[:, :, 6:]
        return log_prob, z, pred_pose_6d #, pred_pose_transl
