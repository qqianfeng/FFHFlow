import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
import normflows as nf
from .normflows_rot_glow import ConditionalGlow, Glow
from yacs.config import CfgNode

from .local_inn import PositionalEncoding

class PriorFlow(nn.Module):
    def __init__(self, 
                 cfg: CfgNode):
        super().__init__()
        self.prior_flow_cond = cfg.MODEL.BACKBONE.PRIOR_FLOW_COND 

        if self.prior_flow_cond:
            cond_glow = ConditionalGlow(input_dim=cfg.MODEL.FLOW.CONTEXT_FEATURES,
                                    hidden_dim=cfg.MODEL.BACKBONE.PRIOR_FLOW_LAYER_HIDDEN_FEATURES,
                                    flow_layers=cfg.MODEL.BACKBONE.PRIOR_FLOW_NUM_LAYERS,
                                    res_num_layers=cfg.MODEL.BACKBONE.PRIOR_FLOW_LAYER_DEPTH,
                                    context_features=cfg.MODEL.BACKBONE.PCD_ENC_HIDDEN_DIM,
                                    base_dist=cfg.MODEL.BACKBONE.PRIOR_FLOW_BASE,
                                    cfg=cfg)
            self.flow = cond_glow.model
        else:
            glow = Glow(input_dim=cfg.MODEL.FLOW.CONTEXT_FEATURES,
                        hidden_dim=cfg.MODEL.BACKBONE.PRIOR_FLOW_LAYER_HIDDEN_FEATURES,
                        flow_layers=cfg.MODEL.BACKBONE.PRIOR_FLOW_NUM_LAYERS,
                        res_num_layers=cfg.MODEL.BACKBONE.PRIOR_FLOW_LAYER_DEPTH,
                        cfg=None)
            self.flow = glow.model

    def log_prob(self, latents: torch.Tensor, cond_feats: torch.Tensor = None) -> Tuple:
        # batch_size = cond_feats.shape[0]
        # cond_feats = cond_feats.reshape(batch_size, -1)
        # grasp -> z
        if cond_feats is None:
            log_prob = self.flow.log_prob(latents)
            return log_prob
        else:
            log_prob, z = self.flow.log_prob(latents, cond_feats)
            # log_prob = log_prob.reshape(batch_size, 1)
            # z = z.reshape(batch_size, 1, -1)
            return log_prob, z
    
    def sample(self, cond_feats: torch.Tensor, num_samples: Optional[int] = None,):
        samples, log_prob = self.flow.sample(num_samples, context=cond_feats)
        return samples, log_prob



class NormflowsGraspFlowPosEncWithTransl(nn.Module):
    """
    Normalizing flow implemented according to PROHMR paper. The forward and backward direction are both computed for loss.
    So 'grasp' -> 'z' and also 'z' -> 'grasp'.
    """
    def __init__(self, cfg: CfgNode):
        """
        Args:
            cfg (CfgNode): Model config as yacs CfgNode.
        """
        super().__init__()
        self.cfg = cfg
        self.positional_encoding = cfg.MODEL.FLOW.DIM > 22
        glow = ConditionalGlow(input_dim=cfg.MODEL.FLOW.DIM,
                                hidden_dim=cfg.MODEL.FLOW.LAYER_HIDDEN_FEATURES,
                                flow_layers=cfg.MODEL.FLOW.NUM_LAYERS,
                                res_num_layers=cfg.MODEL.FLOW.LAYER_DEPTH,
                                context_features=cfg.MODEL.FLOW.CONTEXT_FEATURES,
                                base_dist=cfg.MODEL.FLOW.BASE,
                                cfg=cfg)
        self.flow = glow.model
        print(f"positional_encoding is {self.positional_encoding}")
        if self.positional_encoding:
            self.pe = PositionalEncoding()
        else:
            self.pe = None

    def angle_trans_no_pe(self, batch: Dict, feats: torch.Tensor):
        batch_size = feats.shape[0]

        # input of positional encoded angles
        angles = batch['angle_vector']  # [batch_size,3,3]
        angles = angles.reshape(batch_size, -1).to(feats.dtype)

        transl = batch['transl']
        transl = transl.reshape(batch_size, -1).to(feats.dtype)

        return angles, transl

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
        if self.positional_encoding:
            # input of positional encoded angles
            angles = batch['angle_vector']  # [batch_size,3,3]
            angles = self.pe.forward_localinn(angles)
            angles = angles.reshape(batch_size, -1).to(feats.dtype)

            transl = batch['transl']
            transl = self.pe.forward_transl(transl)
            transl = transl.reshape(batch_size, -1).to(feats.dtype)
        else:
            angles, transl = self.angle_trans_no_pe(batch, feats)

        joint_conf = batch['joint_conf']
        joint_conf = joint_conf.reshape(batch_size, -1).to(feats.dtype)
        samples = torch.cat([angles, transl, joint_conf],dim=1)

        feats = feats.reshape(batch_size, -1)
        # grasp -> z
        log_prob, z = self.flow.log_prob(samples, feats)
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
        # feats = feats.float()

        batch_size = feats.shape[0]

        # we always sample z from prior
        assert z is None
        # Generates samples from the distribution together with their log probability.
        samples, log_prob = self.flow.sample(num_samples, context=feats)
        pred_params = samples.reshape(batch_size, num_samples, -1)
        
        if self.positional_encoding:

            pred_pose = pred_params[:, :, :60]
            # decode
            pred_pose = pred_pose.reshape(batch_size, num_samples, 3, -1)
            pred_angles = self.pe.backward(pred_pose)

            pred_pose_transl = pred_params[:, :, 60:120]
            pred_pose_transl = pred_pose_transl.reshape(batch_size, num_samples, 3, -1)
            pred_pose_transl = self.pe.backward_transl(pred_pose_transl)
            pred_joint_conf = pred_params[:, :, 120:]
        else:
            pred_angles = pred_params[:, :, :3]
            pred_pose_transl = pred_params[:, :, 3:6]
            pred_joint_conf = pred_params[:, :, 6:]

        return log_prob, pred_angles, pred_pose_transl, pred_joint_conf
