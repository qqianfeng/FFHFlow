import os
from typing import Any, Dict, Tuple

import numpy as np
import torch
import transforms3d
from yacs.config import CfgNode
import normflows as nf
from time import time


# from ffhflow.utils.train_utils import clip_grad_norm
from ffhflow.utils.visualization import show_generated_grasp_distribution

from . import Metaclass
from .backbones import BPSMLP, ResNet_3layer
from .heads import GraspFlowGenerator, LatentFlowPrior
from .utils.losses import gaussian_nll, gaussian_ent
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

class FFHFlowCNF(Metaclass):

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
        self.backbone = BPSMLP(cfg)

        # # free param in backbone
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

        self.flow = GraspFlowGenerator(cfg)

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
        trainable_params = list(self.backbone.parameters()) + \
                           list(self.flow.parameters())

        if self.cfg.TRAIN.OPT == "AdamW":
            optimizer = torch.optim.AdamW(params=trainable_params,
                                            lr=self.cfg.TRAIN.LR,
                                            betas=(self.cfg.TRAIN.BETA1, 0.999),
                                            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        elif self.cfg.TRAIN.OPT == "SGD":
            optimizer = torch.optim.SGD(params=trainable_params,
                                    lr=self.cfg.TRAIN.LR,
                                    momentum=0.9)

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

        # change dim of joint conf to 16 for split
        if self.cfg['BASE_PACKAGE'] == 'normflows':
            new_joint_conf = torch.zeros((batch['joint_conf'].shape[0],16)).to('cuda')
            new_joint_conf[:,:15] = batch['joint_conf']
            batch['joint_conf'] = new_joint_conf

        # Compute keypoint features using the ffhgenerator encoder -> {'mu': mu, 'logvar': logvar}, each of [5,]
        conditioning_feats = self.backbone(batch)

        # If ActNorm layers are not initialized, initialize them
        if not self.initialized:
            self.initialize(batch, conditioning_feats)

        # z -> grasp
        log_prob, pred_angles, pred_pose_transl, pred_joint_conf = self.flow(conditioning_feats, num_samples)

        output = {}
        output['log_prod'] = log_prob
        output['pred_angles'] = pred_angles
        output['pred_pose_transl'] = pred_pose_transl
        output['pred_joint_conf'] = pred_joint_conf
        output['conditioning_feats'] = conditioning_feats
        return output

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

        # 1. Reconstruction loss
        pred_angles = output['pred_angles'].view(-1,3)
        pred_pose_transl = output['pred_pose_transl'].view(-1,3)
        pred_joint_conf = output['pred_joint_conf'].view(-1,16)
        pred_joint_conf = pred_joint_conf[:,:15]
        gt_angles = batch['angle_vector']  # [batch_size, 3,3]
        gt_transl = batch['transl']
        gt_joint_conf = batch['joint_conf'][:,:15]
        rot_loss = self.transl_l2_loss(pred_angles, gt_angles, self.L2_loss, self.device)
        transl_loss = self.transl_l2_loss(pred_pose_transl, gt_transl, self.L2_loss, self.device)
        joint_conf_loss = self.transl_l2_loss(pred_joint_conf, gt_joint_conf, self.L2_loss, self.device)

        # TODO: add joint as loss

        # 2. Compute NLL loss
        conditioning_feats = self.backbone(batch)

        # grasp -> z
        if self.cfg['BASE_PACKAGE'] == 'normflows' and batch['joint_conf'].shape[1]%2 != 0:
            padding_zero = torch.zeros([batch['joint_conf'].shape[0], 1]).to('cuda')
            batch['joint_conf'] = torch.cat([batch['joint_conf'], padding_zero], dim=1)
        log_prob, _ = self.flow.log_prob(batch, conditioning_feats)
        loss_nll = -log_prob.mean()

        # 3: Compute orthonormal loss on 6D representations
        # pred_pose_6d = pred_pose_6d.reshape(-1, 2, 3).permute(0, 2, 1)
        # loss_pose_6d = ((torch.matmul(pred_pose_6d.permute(0, 2, 1), pred_pose_6d) - torch.eye(2, device=pred_pose_6d.device, dtype=pred_pose_6d.dtype).unsqueeze(0)) ** 2)
        # loss_pose_6d = loss_pose_6d.reshape(batch_size, num_samples, -1).mean()

        # combine all the losses
        # loss = self.cfg.LOSS_WEIGHTS['NLL'] * grasp_nll
        # self.cfg.LOSS_WEIGHTS['ROT'] * rot_loss
        #    self.cfg.LOSS_WEIGHTS['ORTHOGONAL'] * loss_pose_6d +\
        #    self.cfg.LOSS_WEIGHTS['TRANSL'] * transl_loss
        loss = self.cfg.LOSS_WEIGHTS['NLL'] * loss_nll
        # self.cfg.LOSS_WEIGHTS['ROT'] * rot_loss
        #    self.cfg.LOSS_WEIGHTS['ORTHOGONAL'] * loss_pose_6d +\
        #    self.cfg.LOSS_WEIGHTS['TRANSL'] * transl_loss

        losses = dict(loss=loss.detach(),
                    loss_nll=loss_nll.detach(),
                    rot_loss=rot_loss.detach(),
                    joint_conf_loss=joint_conf_loss.detach(),
                    transl_loss=transl_loss.detach())
                    # loss_pose_6d=loss_pose_6d.detach(),

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
        # clip_grad_norm(optimizer, max_norm=100)
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
        loss = self.compute_loss(batch, output, train=False)
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

    def predict_log_prob(self, bps, grasps):
        """_summary_

        Args:
            batch (_type_): ["bps_object"]: bps
                            ['angle_vector']  # [batch_size,3,3]
                            ['transl']
                            ['joint_conf']
        Returns:
            _type_: _description_
        """
        bps_tensor = bps.to('cuda')
        self.backbone.to('cuda')
        self.flow.to('cuda')

        batch = {}
        batch_size = grasps['pred_angles'].shape[0]
        print('bps tensor size', bps_tensor.size())
        if bps_tensor.dim() == 1:
            bps_tensor = bps_tensor.unsqueeze(0)
        assert bps_tensor.dim() == 2
        batch['bps_object'] = bps_tensor.repeat(batch_size, 1)
        batch['angle_vector'] = grasps['pred_angles']
        batch['transl'] = grasps['pred_pose_transl']
        batch['joint_conf'] = grasps['joint_conf']

        conditioning_feats = self.backbone(batch)
        log_prob, _ = self.flow.log_prob(batch, conditioning_feats)

        return log_prob

    """ TODO: maybe sample should be like this if not working
    def sample(self, bps, num_samples,return_arr=False):
        bps_tensor = bps.to('cuda')
    """
    def sample(self, batch, idx, num_samples):
        """ generate number of grasp samples

        Args:
            bps (torch.Tensor): one bps object
            num_samples (int): _description_

        Returns:
            tensor: _description_
        """
        # move data to cuda
        bps_tensor = batch['bps_object'][idx].to(dtype=torch.float64).to('cuda') # old: bps.to('cuda')
        bps_tensor = bps_tensor.view(1,-1)

        batch = {'bps_object': bps_tensor}
        self.backbone.to('cuda')
        self.flow.to('cuda')

        conditioning_feats = self.backbone(batch)
        log_prob, pred_angles, pred_pose_transl, pred_joint_conf = self.flow(conditioning_feats, num_samples)
        log_prob = log_prob.view(-1)
        pred_angles = pred_angles.view(-1,3)
        pred_pose_transl = pred_pose_transl.view(-1,3)
        pred_joint_conf = pred_joint_conf.view(-1, 16)
        pred_joint_conf = pred_joint_conf[:,:15]

        output = {}
        output['log_prob'] = log_prob
        output['pred_angles'] = pred_angles
        output['pred_pose_transl'] = pred_pose_transl
        output['pred_joint_conf'] = pred_joint_conf

        # convert position encoding to original format of matrix or vector
        output = self.convert_output_to_grasp_mat(output, return_arr=False)

        return output

    def sort_and_filter_grasps(self, samples: Dict, perc: float = 0.5, return_arr: bool = False):

        num_samples = samples['log_prob'].shape[0]
        filt_num = num_samples * perc
        sorted_score, indices = samples['log_prob'].sort(descending=True)
        thresh = sorted_score[int(filt_num)-1]
        indices = indices[sorted_score > thresh]
        sorted_score = sorted_score[sorted_score > thresh]

        filt_grasps = {}

        for k, v in samples.items():
            # so far no output as pred_joint_conf
            index = indices.clone()
            dim = 0

            # Dynamically adjust dimensions for sorting
            while len(v.shape) > len(index.shape):
                dim += 1
                index = index[..., None]
                index = torch.cat(v.shape[dim] * (index, ), dim)

            # Sort grasps
            filt_grasps[k] = torch.gather(input=v, dim=0, index=index)

        # Cast to python (if required)
        if return_arr:
            filt_grasps = {k: v.cpu().detach().numpy() for k, v in filt_grasps.items()}

        return filt_grasps

    def save_to_path(self, np_arr, name, base_path):
        np.save(os.path.join(base_path,name), np_arr)

    def convert_output_to_grasp_mat(self, samples, return_arr=True):
        """_summary_

        Args:
            samples (dict): pred_angles, pred_pose_transl can be of two types.
            One is after positional encoding, one is original format (mat or vec)
            but ['rot_matrix'] and ['transl'] must be mat or vec for same interface.
            return_arr (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        num_samples = samples['pred_angles'].shape[0]
        pred_rot_matrix = np.zeros((num_samples,3,3))
        pred_transl_all = np.zeros((num_samples,3))

        for idx in range(num_samples):
            pred_angles = samples['pred_angles'][idx].cpu().data.numpy()
            # rescale rotation prediction back
            pred_angles = pred_angles * 2 * np.pi - np.pi
            pred_angles[pred_angles < -np.pi] += 2 * np.pi

            alpha, beta, gamma = pred_angles
            mat = transforms3d.euler.euler2mat(alpha, beta, gamma)
            pred_rot_matrix[idx] = mat

            # rescale transl prediction back

            palm_transl_min = -0.3150945039775345
            palm_transl_max = 0.2628828995958964
            pred_transl = samples['pred_pose_transl'][idx].cpu().data.numpy()
            value_range = palm_transl_max - palm_transl_min
            pred_transl = pred_transl * (palm_transl_max - palm_transl_min) + palm_transl_min

            pred_transl[pred_transl < -value_range / 2] += value_range
            pred_transl[pred_transl > value_range / 2] -= value_range
            pred_transl_all[idx] = pred_transl

        if return_arr:
            samples['rot_matrix'] = pred_rot_matrix
            samples['transl'] = pred_transl_all
            samples['joint_conf'] = samples['pred_joint_conf'].cpu().data.numpy()

        else:
            samples['rot_matrix'] = torch.from_numpy(pred_rot_matrix).cuda()
            samples['transl'] = torch.from_numpy(pred_transl_all).cuda()
            samples['joint_conf'] = samples['pred_joint_conf']
        return samples

    def show_grasps(self, pcd_path, samples: Dict, i: int = 0, base_path: str = '', save: bool = False):
        """Visualization of grasps

        Args:
            pcd_path (str): _description_
            samples (Dict): with of tensor
            i (int): index of sample. If i = -1, no images will be triggered to ask for save
        """

        if torch.is_tensor(samples['rot_matrix']):
            samples_copy = {}
            for key, value in samples.items():
                samples_copy[key] = value.cpu().data.numpy()
            # samples_copy['rot_matrix'] = samples['rot_matrix'].cpu().data.numpy()
            # samples_copy['transl'] = samples['transl'].cpu().data.numpy()
            # samples_copy['pred_joint_conf'] = samples['pred_joint_conf'].cpu().data.numpy()
        else:
            samples_copy = samples
        show_generated_grasp_distribution(pcd_path, samples_copy, save_ix=i)

        if save:
            i = 0
            self.save_to_path(pcd_path, 'pcd_path.npy', base_path)

            centr_T_palm = np.zeros((4,4))
            centr_T_palm[:3,:3] = samples['rot_matrix'][i]
            centr_T_palm[:3,-1] = samples['transl'][i]
            self.save_to_path(centr_T_palm, 'centr_T_palm.npy', base_path)

            # self.save_to_path(grasps['joint_conf'][i], 'joint_conf.npy', base_path)

