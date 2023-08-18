import torch
import numpy as np
import pytorch_lightning as pl
from typing import Any, Dict, Tuple

from yacs.config import CfgNode

from .backbones import PointNetfeat, FFHGenerator, BPSMLP
from .heads import GraspFlowNormal
from .utils import utils


def kl_divergence(mu, logvar, device="cpu"):
    """
      Computes the kl divergence for batch of mu and logvar.
    """
    return torch.mean(-.5 * torch.sum(1. + logvar - mu**2 - torch.exp(logvar), dim=-1))


def transl_l2_loss(pred_transl,
                   gt_transl,
                   torch_l2_loss_fn,
                   device='cpu'):
    return torch_l2_loss_fn(pred_transl, gt_transl)


def rot_6D_l2_loss(pred_rot_6D,
                    gt_rot_matrix,
                    torch_l2_loss_fn,
                    device='cpu'):
    """Takes in the 3D translation and 6D rotation prediction and computes l2 loss to ground truth 3D translation
    and 3x3 rotation matrix by translforming the 6D rotation to 3x3 rotation matrix.

    Args:
        pred_transl_rot_6D (_type_): rotation representation of 6 D
        gt_transl_rot_matrix (_type_): roration matrix [3,3] + translation
        torch_l2_loss_fn (_type_): _description_
        device (str, optional): _description_. Defaults to 'cpu'.

    Returns:
        _type_: _description_
    """
    pred_rot_matrix = utils.rot_matrix_from_ortho6d(pred_rot_6D, device=device)  # batch_size*3*3
    pred_rot_matrix = pred_rot_matrix.view(pred_rot_matrix.shape[0], -1)  # batch_size*9
    gt_rot_matrix = gt_rot_matrix.view(pred_rot_matrix.shape[0], -1)  #
    # l2 loss on rotation matrix
    l2_loss_rot = torch_l2_loss_fn(pred_rot_matrix, gt_rot_matrix)

    return l2_loss_rot


class FFHFlowNormal(pl.LightningModule):

    def __init__(self, cfg: CfgNode):
        """
        Setup ProHMR model
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode
        """
        super().__init__()

        self.cfg = cfg

        # Create backbone feature extractor
        # self.backbone = PointNetfeat(global_feat=True, feature_transform=False)
        self.backbone = BPSMLP()

        # # free param in backbone
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

        self.flow = GraspFlowNormal(cfg)

        self.optimizer = torch.optim.Adam(self.backbone.parameters(),
                                            lr=cfg.TRAIN.LR,
                                            betas=(cfg.TRAIN.BETA1, 0.999),
                                            weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        self.kl_loss = kl_divergence
        self.rot_6D_l2_loss = rot_6D_l2_loss
        self.transl_l2_loss = transl_l2_loss

        self.L2_loss = torch.nn.MSELoss(reduction='mean')

        self.initialized = False
        self.automatic_optimization = False

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

        # Compute keypoint features using the ffhgenerator encoder -> {'mu': mu, 'logvar': logvar}, each of [5,]
        conditioning_feats = self.backbone(batch)

        # If ActNorm layers are not initialized, initialize them
        if not self.initialized:
            self.initialize(batch, conditioning_feats)

        log_prob, _ = self.flow(batch, conditioning_feats)

        output = {}
        output['log_prod'] = log_prob
        # output['pred_pose_rot'] = pred_pose_rot
        # output['pred_pose_transl'] = pred_pose_transl
        output['conditioning_feats'] = conditioning_feats

        return output

    def compute_loss(self, batch: Dict, output: Dict, train: bool = True) -> torch.Tensor:
        """
        Compute losses given the input batch and the regression output
        If we want to fintune the model with additional loss (recons_loss and orth_loss), modify it here
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            torch.Tensor : Total loss for current batch
        """

        # 1. Reconstruction loss
        # num_samples = output['pred_pose_rot'].shape[1]
        # pred_pose_6d = output['pred_pose_rot'].view(-1,6)
        # pred_pose_transl = output['pred_pose_transl'].view(-1,3)
        # batch_size = output['pred_pose_rot'].shape[0]
        # gt_rot_matrix = batch['rot_matrix']  # [batch_size, 3,3]
        # gt_transl = batch['transl']

        # rot_loss = self.rot_6D_l2_loss(pred_pose_6d, gt_rot_matrix, self.L2_loss, self.device)
        # transl_loss = self.transl_l2_loss(pred_pose_transl, gt_transl, self.L2_loss, self.device)
        # # TODO: add joint as loss

        # Additional reconstruction loss part for finetune
        num_samples = self.cfg.TRAIN.NUM_TEST_SAMPLES
        conditioning_feats = output['conditioning_feats']

        log_prob, _, pred_pose_rot, pred_pose_transl = self.flow(batch, conditioning_feats, num_samples,train=False)
        pred_pose_6d = pred_pose_rot.view(-1,6)
        pred_pose_transl = pred_pose_transl.view(-1,3)
        batch_size = self.cfg.TRAIN.BATCH_SIZE
        gt_rot_matrix = batch['rot_matrix']  # [batch_size, 3,3]
        gt_transl = batch['transl']

        rot_loss = self.rot_6D_l2_loss(pred_pose_6d, gt_rot_matrix, self.L2_loss, self.device)
        transl_loss = self.transl_l2_loss(pred_pose_transl, gt_transl, self.L2_loss, self.device)

        # 2. Compute NLL loss
        conditioning_feats = output['conditioning_feats']
        log_prob = output['log_prod']

        loss_nll = -log_prob.mean()

        # 3: Compute orthonormal loss on 6D representations
        pred_pose_6d = pred_pose_6d.reshape(-1, 2, 3).permute(0, 2, 1)
        loss_pose_6d = ((torch.matmul(pred_pose_6d.permute(0, 2, 1), pred_pose_6d) - torch.eye(2, device=pred_pose_6d.device, dtype=pred_pose_6d.dtype).unsqueeze(0)) ** 2)
        loss_pose_6d = loss_pose_6d.reshape(batch_size, num_samples, -1).mean()

        # combine all the losses
        loss = self.cfg.LOSS_WEIGHTS['NLL'] * loss_nll +\
               self.cfg.LOSS_WEIGHTS['ORTHOGONAL'] * loss_pose_6d +\
               self.cfg.LOSS_WEIGHTS['ROT'] * rot_loss +\
               self.cfg.LOSS_WEIGHTS['TRANSL'] * transl_loss

        losses = dict(loss=loss.detach(),
                    loss_nll=loss_nll.detach(),
                    loss_pose_6d=loss_pose_6d.detach(),
                    rot_loss=rot_loss.detach(),
                    transl_loss=transl_loss.detach())

        output['losses'] = losses

        return loss

    def compute_val_loss(self, batch: Dict, output: Dict) -> torch.Tensor:
        """
        Compute losses given the input batch and the regression output
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            torch.Tensor : Total loss for current batch
        """

        # 1. Reconstruction loss
        num_samples = self.cfg.TRAIN.NUM_TEST_SAMPLES
        conditioning_feats = output['conditioning_feats']

        log_prob, _, pred_pose_rot, pred_pose_transl = self.flow(batch, conditioning_feats, num_samples,train=False)
        pred_pose_6d = pred_pose_rot.view(-1,6)
        pred_pose_transl = pred_pose_transl.view(-1,3)
        batch_size = self.cfg.TRAIN.BATCH_SIZE
        gt_rot_matrix = batch['rot_matrix']  # [batch_size, 3,3]
        gt_transl = batch['transl']

        rot_loss = self.rot_6D_l2_loss(pred_pose_6d, gt_rot_matrix, self.L2_loss, self.device)
        transl_loss = self.transl_l2_loss(pred_pose_transl, gt_transl, self.L2_loss, self.device)
        # # TODO: add joint as loss

        # 2. Compute NLL loss
        conditioning_feats = output['conditioning_feats']

        # # Add some noise to annotations at training time to prevent overfitting
        # if train:
        #     smpl_params = {k: v + self.cfg.TRAIN.SMPL_PARAM_NOISE_RATIO * torch.randn_like(v) for k, v in smpl_params.items()}

        log_prob, _ = self.flow.log_prob(batch, conditioning_feats)
        loss_nll = -log_prob.mean()

        # 3: Compute orthonormal loss on 6D representations
        pred_pose_6d = pred_pose_6d.reshape(-1, 2, 3).permute(0, 2, 1)
        loss_pose_6d = ((torch.matmul(pred_pose_6d.permute(0, 2, 1), pred_pose_6d) - torch.eye(2, device=pred_pose_6d.device, dtype=pred_pose_6d.dtype).unsqueeze(0)) ** 2)
        loss_pose_6d = loss_pose_6d.reshape(batch_size, num_samples, -1).mean()

        # combine all the losses
        loss = self.cfg.LOSS_WEIGHTS['NLL'] * loss_nll +\
               self.cfg.LOSS_WEIGHTS['ORTHOGONAL'] * loss_pose_6d +\
               self.cfg.LOSS_WEIGHTS['ROT'] * rot_loss +\
               self.cfg.LOSS_WEIGHTS['TRANSL'] * transl_loss

        losses = dict(loss=loss.detach(),
                    loss_nll=loss_nll.detach(),
                    loss_pose_6d=loss_pose_6d.detach(),
                    rot_loss=rot_loss.detach(),
                    transl_loss=transl_loss.detach())

        output['losses'] = losses

        return loss

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
        loss = self.compute_val_loss(batch, output)
        output['loss'] = loss
        self.tensorboard_logging(batch, output, self.global_step, train=False)

        return output

    def tensorboard_logging(self, batch: Dict, output: Dict, step_count: int, train: bool = True) -> None:
        """
        Log results to Tensorboard
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            step_count (int): Global training step count
            train (bool): Flag indicating whether it is training or validation mode
        """

        mode = 'train' if train else 'val'
        summary_writer = self.logger.experiment

        losses = output['losses']

        for loss_name, val in losses.items():
            summary_writer.add_scalar(mode + '/' + loss_name, val.detach().item(), step_count)
