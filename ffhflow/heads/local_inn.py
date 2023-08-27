import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from nflows.flows import ConditionalGlow
from torch import Tensor
from yacs.config import CfgNode

from ffhflow.utils.utils import rot_matrix_from_ortho6d

# class PositionalEncodingLocalINN(nn.Module):
#     def __init__(self,  d_model: int):


class PositionalEncoding(nn.Module):
    """
    Original from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    grasp pose rotational matrix -> euler angle [alpha, beta, gamma] has range of [-pi, pi], [-pi/3,pi/2], [-pi, pi]
    """

    def __init__(self):
        super().__init__()

    def forward(self, angle_vec, d=4) -> Tensor:
        """_summary_

        Args:
            angle_vec (_type_): [batch_size, 3]
            n (int, optional): _description_. Defaults to 10000.
            d (int, optional): _description_. Defaults to 4.

        Returns:
            Tensor: _description_
        """
        n = torch.Tensor([10]).to(angle_vec.get_device())
        P = torch.zeros((angle_vec.shape[0], angle_vec.shape[1], d)).to(angle_vec.get_device())
        for k in range(angle_vec.shape[1]):
            for i in torch.arange(int(d/2)):
                denominator = torch.pow(n, 2*i/d) # n^0=1 ,n^0.5
                P[:, k, 2*i] = torch.sin(angle_vec[:,k]/denominator)
                P[:, k, 2*i+1] = torch.cos(angle_vec[:,k]/denominator)

        return P

    def backward(self, P: Tensor) -> Tensor:
        """_summary_

        Args:
            P (Tensor): [batch_size, 3, 4]

        Returns:
            Tensor: _description_
        """
        batch_size = P.shape[0]
        angle_vec = torch.zeros([batch_size,3]).to(P.get_device())
        for i in range(3):
            angle_vec[:,i] = torch.atan2(P[:,i,0], P[:,i,1])
        return angle_vec

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

if __name__ == "__main__":
    pe = PositionalEncoding()
    angle_vector = torch.tensor([[0.5,1.0,1.5],[2.0,2.5,3]])
    encoded_angle = pe.forward(angle_vector)
    decoded_angle = pe.backward(encoded_angle)
    print(torch.allclose(decoded_angle, angle_vector,atol=1e-09))
