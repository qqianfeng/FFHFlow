import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from typing import Optional, Dict, Tuple
from nflows.flows import ConditionalGlow
from yacs.config import CfgNode

from ffhflow.utils.utils import rot_matrix_from_ortho6d
from .local_inn import PositionalEncoding

class GraspFlowNormalPosEnc(nn.Module):
    """
    Normalizing flow in a "normal" way that only forward direction is computed for loss so 'grasp' -> 'z'.
    """
    def __init__(self, cfg: CfgNode):
        """
        Args:
            cfg (CfgNode): Model config as yacs CfgNode.
        """
        super().__init__()
        self.cfg = cfg
        self.flow = ConditionalGlow(cfg.MODEL.FLOW.DIM, cfg.MODEL.FLOW.LAYER_HIDDEN_FEATURES,
                                    cfg.MODEL.FLOW.NUM_LAYERS, cfg.MODEL.FLOW.LAYER_DEPTH,
                                    context_features=cfg.MODEL.FLOW.CONTEXT_FEATURES)
        self.pe = PositionalEncoding()

    def log_prob(self, batch: Dict, feats: torch.Tensor) -> Tuple:
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
        batch_size = feats.shape[0]

        # # input of rot matrix
        # samples = batch['rot_matrix']  # [batch_size,3,3]

        # input of positional encoded angles
        angles = batch['angle_vector']  # [batch_size,3,3]
        angles = self.pe.forward_localinn(angles)
        angles = angles.reshape(batch_size, -1).to(feats.dtype)

        transl = batch['transl'].reshape(batch_size, -1).to(feats.dtype)
        samples = torch.cat([angles, transl],dim=1)

        feats = feats.reshape(batch_size, -1)
        log_prob, z = self.flow.log_prob(samples, feats)
        log_prob = log_prob.reshape(batch_size, 1)
        z = z.reshape(batch_size, 1, -1)
        return log_prob, z

    def forward(self, batch, feats: torch.Tensor, num_samples: Optional[int] = None, z: Optional[torch.Tensor] = None, train=True) -> Tuple:
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

        batch_size = feats.shape[0]

        # we always sample z from prior
        assert z is None

        # Generates samples from the distribution together with their log probability.
        # samples, log_prob, z = self.flow.sample_and_log_prob(num_samples, context=feats)

        if train is False:
            samples, log_prob, z = self.flow.sample_and_log_prob(num_samples, context=feats)
            z = z.reshape(batch_size, num_samples, -1)
            pred_params = samples.reshape(batch_size, num_samples, -1)

            # decode
            pred_params = pred_params.reshape(batch_size, num_samples, 3,-1)
            # pred_pose_transl = pred_params[:, :, 6:]
            pred_angles = self.pe.backward(pred_params)

            pred_transl = pred_params[:,:,-3:]
            return log_prob, z, pred_angles, pred_transl

        else:
            angles = batch['angle_vector']  # [batch_size,3,3]
            angles = self.pe.forward_localinn(angles)
            angles = angles.reshape(batch_size, -1).to(feats.dtype)

            transl = batch['transl'].reshape(batch_size, -1).to(feats.dtype)
            samples = torch.cat([angles, transl],dim=1)
            # grasp -> z
            log_prob, z = self.flow.log_prob(samples, feats)

            return log_prob, z#, pred_pose_6d, pred_pose_transl
