import torch
import numpy as np
import pytorch_lightning as pl
from typing import Any, Dict, Tuple

from prohmr.models import SMPL
from yacs.config import CfgNode

from prohmr.utils import SkeletonRenderer
from prohmr.utils.geometry import aa_to_rotmat, perspective_projection
from prohmr.optimization import OptimizationTask
from .backbones import PointNetfeat
from .heads import RotFlow
from .utils import utils


def kl_divergence(mu, logvar, device="cpu"):
    """
      Computes the kl divergence for batch of mu and logvar.
    """
    return torch.mean(-.5 * torch.sum(1. + logvar - mu**2 - torch.exp(logvar), dim=-1))


def transl_rot_6D_l2_loss(pred_transl_rot_6D,
                          gt_transl_rot_matrix,
                          torch_l2_loss_fn,
                          device='cpu'):
    """ Takes in the 3D translation and 6D rotation prediction and computes l2 loss to ground truth 3D translation
    and 3x3 rotation matrix by translforming the 6D rotation to 3x3 rotation matrix.
    """
    pred_rot_matrix = utils.rot_matrix_from_ortho6d(pred_transl_rot_6D['rot_6D'],device=device)  #batch_size*3*3
    pred_rot_matrix = pred_rot_matrix.view(pred_rot_matrix.shape[0], -1)  #batch_size*9
    gt_rot_matrix = gt_transl_rot_matrix['rot_matrix']
    # l2 loss on rotation matrix
    l2_loss_rot = torch_l2_loss_fn(pred_rot_matrix, gt_rot_matrix)
    # l2 loss on translation
    l2_loss_transl = torch_l2_loss_fn(pred_transl_rot_6D['transl'], gt_transl_rot_matrix['transl'])

    return l2_loss_transl, l2_loss_rot


class FFHFlow(pl.LightningModule):

    def __init__(self, cfg: CfgNode):
        """
        Setup ProHMR model
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode
        """
        super().__init__()

        self.cfg = cfg
        # Create backbone feature extractor
        # self.backbone = PointNetfeat(cfg)
        self.backbone = PointNetfeat(global_feat=True, feature_transform=False)

        self.flow = RotFlow(cfg)

        self.optimizer = torch.optim.Adam(self.backbone.parameters(),
                                            lr=cfg.TRAIN.LR,
                                            betas=(cfg.TRAIN.BETA1, 0.999),
                                            weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        self.kl_loss = kl_divergence
        self.rec_pose_loss = transl_rot_6D_l2_loss
        self.L2_loss = torch.nn.MSELoss(reduction='mean')

        self.initialized = False

    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        """
        Setup model and distriminator Optimizers
        Returns:
            Tuple[torch.optim.Optimizer, torch.optim.Optimizer]: Model and discriminator optimizers
        """
        optimizer = torch.optim.AdamW(params=list(self.backbone.parameters()) + list(self.flow.parameters()),
                                        lr=self.cfg.TRAIN.LR,
                                        betas=(self.cfg.TRAIN.BETA1, 0.999),
                                        weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)

        return optimizer

    def initialize(self, batch: Dict, conditioning_feats: torch.Tensor):
        """
        Initialize ActNorm buffers by running a dummy forward step
        Args:
            batch (Dict): Dictionary containing batch data
            conditioning_feats (torch.Tensor): Tensor of shape (N, C) containing the conditioning features extracted using thee backbonee
        """
        # Get ground truth SMPL params, convert them to 6D and pass them to the flow module together with the conditioning feats.
        # Necessary to initialize ActNorm layers.

        # TODO:
        # smpl_params = {k: v.clone() for k,v in batch['smpl_params'].items()}
        # batch_size = smpl_params['body_pose'].shape[0]
        # has_smpl_params = batch['has_smpl_params']['body_pose'] > 0
        # smpl_params['body_pose'] = aa_to_rotmat(smpl_params['body_pose'].reshape(-1, 3)).reshape(batch_size, -1, 3, 3)[:, :, :, :2].permute(0, 1, 3, 2).reshape(batch_size, 1, -1)[has_smpl_params]
        # smpl_params['global_orient'] = aa_to_rotmat(smpl_params['global_orient'].reshape(-1, 3)).reshape(batch_size, -1, 3, 3)[:, :, :, :2].permute(0, 1, 3, 2).reshape(batch_size, 1, -1)[has_smpl_params]
        # smpl_params['betas'] = smpl_params['betas'].unsqueeze(1)[has_smpl_params]
        # conditioning_feats = conditioning_feats[has_smpl_params]
        with torch.no_grad():
            _, _ = self.flow.log_prob(batch, conditioning_feats)
            self.initialized = True

    def forward_step(self, batch: Dict, train: bool = False) -> Dict:
        """
        Run a forward step of the network
        Args:
            batch (Dict): Dictionary containing batch data
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            Dict: Dictionary containing the regression output
        """
        if train:
            num_samples = self.cfg.TRAIN.NUM_TRAIN_SAMPLES
        else:
            num_samples = self.cfg.TRAIN.NUM_TEST_SAMPLES


        x = batch["pcd_arr"]
        batch_size = x.shape[0]

        # Compute keypoint features using the backbone
        conditioning_feats, _, _ = self.backbone(x)

       # If ActNorm layers are not initialized, initialize them
       # TODO:
        if not self.initialized:
            self.initialize(batch, conditioning_feats)

        # If validation draw num_samples - 1 random samples and the zero vector
        if num_samples > 1:
            log_prob, _, pred_pose_6d = self.flow(conditioning_feats, num_samples=num_samples-1)
            z_0 = torch.zeros(batch_size, 1, self.cfg.MODEL.FLOW.DIM, device=x.device)
            log_prob_mode, _,  pred_pose_6d_mode = self.flow(conditioning_feats, z=z_0)
            log_prob = torch.cat((log_prob_mode, log_prob), dim=1)
            pred_pose_6d = torch.cat((pred_pose_6d_mode, pred_pose_6d), dim=1)
        else:
            z_0 = torch.zeros(batch_size, 1, self.cfg.MODEL.FLOW.DIM, device=x.device)
            log_prob, _,  pred_pose_6d = self.flow(conditioning_feats, z=z_0)

    def compute_loss(self, batch: Dict, output: Dict, train: bool = True) -> torch.Tensor:
        """
        Compute losses given the input batch and the regression output
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            torch.Tensor : Total loss for current batch
        """

        # KL loss
        kl_loss_val = self.kl_loss(output["mu"], output["logvar"])

        # reconstruction loss
        gt_transl_rot_matrix = {
            # 'transl': self.FFHGenerator.transl,
            'rot_matrix': self.FFHGenerator.rot_matrix
        }
        transl_loss_val, rot_loss_val = self.rec_pose_loss(output, gt_transl_rot_matrix,
                                                        self.L2_loss, self.device)

        # TODO: Compute NLL loss
        # Add some noise to annotations at training time to prevent overfitting
        if train:
            smpl_params = {k: v + self.cfg.TRAIN.SMPL_PARAM_NOISE_RATIO * torch.randn_like(v) for k, v in smpl_params.items()}
        if smpl_params['body_pose'].shape[0] > 0:
            log_prob, _ = self.flow.log_prob(smpl_params, conditioning_feats[has_smpl_params])
        else:
            log_prob = torch.zeros(1, device=device, dtype=dtype)
        loss_nll = -log_prob.mean()

        # TODO: Compute orthonormal loss on 6D representations
        pred_pose_6d = pred_pose_6d.reshape(-1, 2, 3).permute(0, 2, 1)
        loss_pose_6d = ((torch.matmul(pred_pose_6d.permute(0, 2, 1), pred_pose_6d) - torch.eye(2, device=pred_pose_6d.device, dtype=pred_pose_6d.dtype).unsqueeze(0)) ** 2)
        loss_pose_6d = loss_pose_6d.reshape(batch_size, num_samples, -1)
        loss_pose_6d_mode = loss_pose_6d[:, 0].mean()
        loss_pose_6d_exp = loss_pose_6d[:, 1:].mean()

        # combine all the losses
        loss = self.cfg.LOSS_WEIGHTS['NLL'] * loss_nll+\
               self.cfg.LOSS_WEIGHTS['ORTHOGONAL'] * (loss_pose_6d_exp+loss_pose_6d_mode)
            #    sum([loss_smpl_params_exp[k] * self.cfg.LOSS_WEIGHTS[(k+'_EXP').upper()] for k in loss_smpl_params_exp])+\
            #    sum([loss_smpl_params_mode[k] * self.cfg.LOSS_WEIGHTS[(k+'_MODE').upper()] for k in loss_smpl_params_mode])


    def forward(self, batch: Dict) -> Dict:
        """
        Run a forward step of the network in val mode
        Args:
            batch (Dict): Dictionary containing batch data
        Returns:
            Dict: Dictionary containing the regression output
        """
        return self.forward_step(batch, train=False)

    def training_step(self, joint_batch: Dict, batch_idx: int) -> Dict:
        """
        Run a full training step
        batch: 'rot_matrix','transl','joint_conf','bps_object','pcd_path','obj_name'
        Args:
            joint_batch (Dict): Dictionary containing image and mocap batch data
            batch_idx (int): Unused.
            batch_idx (torch.Tensor): Unused.
        Returns:
            Dict: Dictionary containing regression output.
        """
        batch = joint_batch
        optimizer = self.optimizers(use_pl_optimizer=True)
        output = self.forward_step(batch, train=True)

        pred_smpl_params = output['pred_smpl_params']
        num_samples = pred_smpl_params['body_pose'].shape[1]
        pred_smpl_params = output['pred_smpl_params']
        loss = self.compute_loss(batch, output, train=True)

        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

        if self.global_step > 0 and self.global_step % self.cfg.GENERAL.LOG_STEPS == 0:
            self.tensorboard_logging(batch, output, self.global_step, train=True)

        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> Dict:
        """
        Run a validation step and log to Tensorboard
        Args:
            batch (Dict): Dictionary containing batch data
            batch_idx (int): Unused.
        Returns:
            Dict: Dictionary containing regression output.
        """
        output = self.forward_step(batch, train=False)
        pred_smpl_params = output['pred_smpl_params']
        num_samples = pred_smpl_params['body_pose'].shape[1]
        loss = self.compute_loss(batch, output, train=False)
        output['loss'] = loss
        self.tensorboard_logging(batch, output, self.global_step, train=False)

        return output