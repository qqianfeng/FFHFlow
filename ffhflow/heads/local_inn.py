import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from nflows.flows import ConditionalGlow
from torch import Tensor
from yacs.config import CfgNode
import numpy as np
from copy import deepcopy
# from ffhflow.utils.utils import rot_matrix_from_ortho6d



class PositionalEncoding(nn.Module):
    """
    Original from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    grasp pose rotational matrix -> euler angle [alpha, beta, gamma] has range of [-pi, pi], [-pi/2,pi/2], [-pi, pi]
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
        n = torch.Tensor([10]).to(angle_vec.device)
        P = torch.zeros((angle_vec.shape[0], angle_vec.shape[1], d)).to(angle_vec.device)
        for k in range(angle_vec.shape[1]):
            for i in torch.arange(int(d/2)):
                denominator = torch.pow(n, torch.Tensor([2*i/d]).to(angle_vec.device)) # n^0=1 ,n^0.5
                P[:, k, 2*i] = torch.sin(angle_vec[:,k]/denominator)
                P[:, k, 2*i+1] = torch.cos(angle_vec[:,k]/denominator)

        return P

    def forward_localinn(self, angle_vec, d=20) -> Tensor:
        """_summary_

        Args:
            angle_vec (_type_): [batch_size, 3]
            n (int, optional): _description_. Defaults to 10000.
            d (int, optional): _description_. Defaults to 4.

        Returns:
            Tensor: _description_
        """
        P = torch.zeros((angle_vec.shape[0], angle_vec.shape[1], d)).to(angle_vec.device)
        for k in range(angle_vec.shape[1]):
            for i in torch.arange(int(d/2)):
                denominator = torch.Tensor([2**i]).to(angle_vec.device)  # n^0=1 ,n^0.5
                pi = torch.from_numpy(np.array([2*np.pi])).to(angle_vec.device)
                P[:, k, 2*i] = torch.sin(angle_vec[:,k] * denominator * pi)
                P[:, k, 2*i+1] = torch.cos(angle_vec[:,k] * denominator * pi)

        return P

    def backward(self, P: Tensor) -> Tensor:
        """_summary_

        Args:
            P (Tensor): [batch_size, 3, 4]

        Returns:
            Tensor: _description_
        """
        # from sample function, batch size is 1 and P has shape of [1,num_samples,3,20]
        if P.dim() == 4:
            P = P[0,:,:,:]
        batch_size = P.shape[0]
        angle_vec = torch.zeros([batch_size,3]).to(P.device)
        pi = torch.from_numpy(np.array([2*np.pi])).to(angle_vec.device)

        for i in range(3):
            angle_vec[:,i] = torch.atan2(P[:,i,0], P[:,i,1]) / pi

        # # Test backward pass
        # angle_vec_test = torch.zeros([batch_size,3]).to(P.device)
        # for i in range(3):
        #     angle_vec_test[:,i] = torch.atan2(P[:,i,2], P[:,i,3]) * torch.pow(torch.Tensor([10]).to(angle_vec.device), torch.Tensor([0.5]).to(angle_vec.device))
        # print(torch.allclose(angle_vec, angle_vec_test, atol=1e-09))

        return angle_vec

    def backward2(self, P: Tensor) -> Tensor:
        """_summary_

        Args:
            P (Tensor): [batch_size, 3, 4]

        Returns:
            Tensor: _description_
        """
        # from sample function, batch size is 1 and P has shape of [1,num_samples,3,20]
        if P.dim() == 4:
            P = P[0,:,:,:]
        batch_size = P.shape[0]
        angle_vec = torch.zeros([batch_size,3]).to(P.device)
        denominator = torch.Tensor([2**1]).to(P.device)
        for i in range(3):
            angle_vec[:,i] = torch.atan2(P[:,i,2], P[:,i,3]) / denominator
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

        # pred_pose = pred_params[:, :, :6]
        # pred_pose_6d = pred_pose.clone()
        # pred_pose = rot_matrix_from_ortho6d(pred_pose.reshape(batch_size * num_samples, -1))

        # pred_pose_transl = pred_params[:, :, 6:]
        return log_prob, z, pred_pose_6d #, pred_pose_transl


if __name__ == "__main__":
    pe = PositionalEncoding()
    angle_vector = torch.tensor([[-0.6154, -0.1396, -2.9588],
        [-2.9998, -0.1904,  1.4876],
        [ 0.5587, -0.3360,  1.2632],
        [-0.8274,  0.4835,  2.9523],
        [-1.0283, -0.9779, -2.9851],
        [-3.0229, -0.3011,  0.0157],
        [-2.8174,  0.4655,  0.2709],
        [-2.9825,  0.8776,  0.1904],
        [ 2.2218,  0.7480,  2.6583],
        [-1.1349,  0.4133,  2.7053],
        [ 2.8937, -0.3148,  1.5909],
        [-2.9633, -0.9354, -0.6648],
        [ 1.6407, -0.2167,  1.6164],
        [ 2.8057, -0.6806,  0.0401],
        [-2.8803, -0.0648,  0.2265],
        [-0.7335, -0.1495, -3.1232],
        [-0.8460, -0.5731,  3.1134],
        [ 2.7425, -0.1326,  0.1035],
        [-2.2209, -0.4443,  1.4374],
        [-2.9735, -0.6370, -0.2321],
        [-0.6740, -0.3343,  1.5985],
        [ 2.5635, -0.1939,  1.5283],
        [-0.6194, -0.5937,  1.0297],
        [-2.6090, -0.2300,  1.5648],
        [-2.2115, -0.2953,  1.7447],
        [-1.7352, -1.2529, -2.0672],
        [-1.0203,  0.2174,  2.9435],
        [-2.6192, -0.5148,  1.7574],
        [ 1.8217, -0.3006,  1.5263],
        [-0.0624, -0.2909,  1.6260],
        [ 2.9030, -0.7467,  1.4144],
        [-2.2557, -0.4346,  1.6270],
        [-2.1944,  1.0961,  1.3689],
        [-1.1343, -1.1316, -2.5927],
        [-3.0614, -0.3313,  1.5335],
        [-0.7529, -1.3716, -2.9479],
        [-1.5070, -1.3445, -1.6168],
        [-1.0722, -1.2446, -2.3524],
        [ 1.6423, -0.0880,  1.7376],
        [ 0.1122, -0.3033,  1.7356],
        [-0.7854, -0.4743, -3.0053],
        [-2.1391, -0.3432,  1.5907],
        [-0.7667, -0.5808,  3.0788],
        [ 0.5750, -0.3188,  1.6127],
        [-1.7729, -1.1738, -1.3514],
        [-2.9754,  0.3503, -0.0059],
        [ 1.6286, -0.8194,  1.3705],
        [ 1.8944, -0.2774,  1.5707],
        [-2.9414, -0.2306, -0.2958],
        [ 0.7513, -0.6300,  1.9068],
        [-1.0796,  0.6809,  2.7834],
        [-3.1282, -0.2981,  1.5454],
        [ 2.3604, -0.3592,  1.6080],
        [-1.8079,  0.7735,  1.8610],
        [-1.0607, -0.3402,  1.7830],
        [-2.9246,  0.1922,  0.2942],
        [-2.5403,  0.9839,  1.3718],
        [ 3.0537, -0.2071,  1.3226],
        [-2.2396,  1.1143,  1.2166],
        [-2.9918,  0.7664,  0.6178],
        [-1.5151, -0.2733,  1.6056],
        [ 3.0073, -0.8204, -0.2505],
        [-0.1926,  0.6698,  1.0936],
        [-0.4912, -0.2851,  1.5580]], device='cuda:0')
    angle_vector_origin = deepcopy(angle_vector)
    angle_vector[:,0] = (angle_vector[:,0] + np.pi) / 2 / np.pi
    angle_vector[:,1] = (angle_vector[:,1] + np.pi) / 2 / np.pi
    angle_vector[:,2] = (angle_vector[:,2] + np.pi) / 2 / np.pi

    encoded_angle = pe.forward_localinn(angle_vector)
    decoded_angle = pe.backward(encoded_angle)
    decoded_angle2 = pe.backward2(encoded_angle)

    decoded_angle[:,0] = decoded_angle[:,0] * 2 * np.pi - np.pi
    decoded_angle[:,1] = decoded_angle[:,1] * 2 * np.pi - np.pi
    decoded_angle[:,2] = decoded_angle[:,2] * 2 * np.pi - np.pi


    decoded_angle[decoded_angle<-np.pi] += 2*np.pi
    # a = decoded_angle[:,1]
    # a[a<-np.pi/2] += np.pi/2
    # decoded_angle[:,1] = a
    print(angle_vector_origin - decoded_angle)
    print(torch.allclose(decoded_angle, angle_vector_origin, atol=1e-05))
