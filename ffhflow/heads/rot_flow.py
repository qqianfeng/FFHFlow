import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from typing import Optional, Dict, Tuple
from nflows.flows import ConditionalGlow
from yacs.config import CfgNode

# from prohmr.utils.geometry import rot6d_to_rotmat


class RotFlow(nn.Module):

    def __init__(self, cfg: CfgNode):
        """
        Probabilistic SMPL head using Normalizing Flows.
        Args:
            cfg (CfgNode): Model config as yacs CfgNode.
        """
        super(RotFlow, self).__init__()
        self.cfg = cfg
        # TODO: context_feature???
        self.flow = ConditionalGlow(cfg.MODEL.FLOW.DIM, cfg.MODEL.FLOW.LAYER_HIDDEN_FEATURES,
                                    cfg.MODEL.FLOW.NUM_LAYERS, cfg.MODEL.FLOW.LAYER_DEPTH,
                                    context_features=cfg.MODEL.FLOW.CONTEXT_FEATURES)

    def log_prob(self, batch: Dict, feats: torch.Tensor) -> Tuple:
        """
        Compute the log-probability of a set of smpl_params given a batch of images.
        Args:
            smpl_params (Dict): Dictionary containing a set of SMPL parameters.
            feats (torch.Tensor): Conditioning features of shape (N, C).
        Returns:
            log_prob (torch.Tensor): Log-probability of the samples with shape (B, N).
            z (torch.Tensor): The Gaussian latent corresponding to each sample with shape (B, N, 144).
        """

        # feats = feats.float()  # same as to(torch.float32)  [64,1024]
        batch_size = feats.shape[0]

        samples = batch['rot_matrix'] #[64,3,3]

        feats = feats.reshape(batch_size, 1, -1).repeat(1, 1, 1)

        log_prob, z = self.flow.log_prob(samples.reshape(batch_size, -1).to(feats.dtype), feats.reshape(batch_size, -1))
        log_prob = log_prob.reshape(batch_size, 1)
        z = z.reshape(batch_size, 1, -1)
        return log_prob, z

    def forward(self, feats: torch.Tensor, num_samples: Optional[int] = None, z: Optional[torch.Tensor] = None) -> Tuple:
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
        feats = feats.float()

        batch_size = feats.shape[0]

        if z is None:
            # Generates samples from the distribution together with their log probability.
            samples, log_prob, z = self.flow.sample_and_log_prob(num_samples, context=feats)
            z = z.reshape(batch_size, num_samples, -1)
            pred_params = samples.reshape(batch_size, num_samples, -1)
        else:
            num_samples = z.shape[1]
            samples, log_prob, z = self.flow.sample_and_log_prob(num_samples, context=feats, noise=z)
            pred_params = samples.reshape(batch_size, num_samples, -1)

        # TODO: fix from here
        pred_pose = pred_params[:, :, :self.npose]
        pred_pose_6d = pred_pose.clone()
        pred_pose = rot6d_to_rotmat(pred_pose.reshape(batch_size * num_samples, -1)).view(batch_size, num_samples, self.cfg.SMPL.NUM_BODY_JOINTS+1, 3, 3)
        pred_smpl_params = {'global_orient': pred_pose[:, :, [0]],
                             'body_pose': pred_pose[:, :, 1:]}

        return log_prob, z, pred_pose_6d
