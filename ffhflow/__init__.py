import torch
import numpy as np
import pytorch_lightning as pl
from typing import Any, Dict, Tuple

from yacs.config import CfgNode

from .backbones import PointNetfeat, FFHGenerator, BPSMLP
from .heads import GraspFlow
from .utils import utils
from ffhflow.utils.visualization import show_generated_grasp_distribution

class Metaclass(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def sample(self, bps, num_samples):
        """ generate number of grasp samples

        Args:
            bps (torch.Tensor): one bps object
            num_samples (int): _description_

        Returns:
            _type_: _description_
        """
        bps = torch.tile(bps, (1,1))
        # move data to cuda
        bps_tensor = torch.tensor(bps).to('cuda')
        batch = {'bps_object': bps_tensor}
        self.backbone.to('cuda')
        self.flow.to('cuda')

        conditioning_feats = self.backbone(batch)
        log_prob, _, pred_pose_rot, pred_pose_transl = self.flow(batch, conditioning_feats, num_samples,train=False)
        pred_pose_6d = pred_pose_rot.view(-1,6)
        pred_pose_transl = pred_pose_transl.view(-1,3)

        output = {}
        output['pred_pose_6d'] = pred_pose_6d
        output['pred_pose_transl'] = pred_pose_transl
        return output

    def show_grasps(self, pcd_path, samples: Dict, i: int):
        """Visualization of grasps

        Args:
            pcd_path (str): _description_
            samples (Dict): _description_
            i (int): index of sample
        """
        num_samples = samples['pred_pose_6d'].shape[0]
        pred_rot_matrix = utils.rot_matrix_from_ortho6d(samples['pred_pose_6d'])
        pred_rot_matrix = pred_rot_matrix.reshape((num_samples, 3, 3))
        pred_transl = samples['pred_pose_transl'].cpu().data.numpy()

        grasps = {'rot_matrix': pred_rot_matrix.cpu().data.numpy(), 'transl': pred_transl}
        show_generated_grasp_distribution(pcd_path, grasps, save_ix=i)