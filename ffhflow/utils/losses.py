import torch 
import numpy as np
from .utils import utils

def kl_divergence(mu, logvar, device="cpu"):
    """
      Computes the kl divergence for batch of mu and logvar.
    """
    return torch.mean(-.5 * torch.sum(1. + logvar - mu**2 - torch.exp(logvar), dim=-1))

def gaussian_nll(samples, mus, logvars):
    return 0.5 * torch.add(
        torch.sum(logvars + ((samples - mus)**2 / torch.exp(logvars))) / samples.shape[0],
        np.log(2.0 * np.pi) * samples.shape[1]
    )

def gaussian_ent(logvars):
    return 0.5 * torch.add(logvars.shape[1] * (1.0 + np.log(2.0 * np.pi)), logvars.sum(1).mean())


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

