import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
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

    def forward_transl(self, transl_vec, d=20) -> Tensor:
        """_summary_

        Args:
            transl_vec (_type_): [batch_size, 3]
            n (int, optional): _description_. Defaults to 10000.
            d (int, optional): _description_. Defaults to 4.

        Returns:
            Tensor: _description_
        """
        P = torch.zeros((transl_vec.shape[0], transl_vec.shape[1], d)).to(transl_vec.device)
        for k in range(transl_vec.shape[1]):
            for i in torch.arange(int(d/2)):
                denominator = torch.Tensor([2**i]).to(transl_vec.device)  # n^0=1 ,n^0.5
                pi = torch.from_numpy(np.array([2*np.pi])).to(transl_vec.device)
                P[:, k, 2*i] = torch.sin(transl_vec[:,k] * denominator * pi)
                P[:, k, 2*i+1] = torch.cos(transl_vec[:,k] * denominator * pi)

        return P

    def backward(self, P: Tensor) -> Tensor:
        """_summary_

        Args:
            P (Tensor): [batch_size, 3, 20]

        Returns:
            Tensor: _description_
        """
        # from sample function, batch size is 1 and P has shape of [1,num_samples,3,20]
        P = torch.squeeze(P)
        assert P.dim() == 3
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

    def backward_transl(self, P: Tensor) -> Tensor:
        """_summary_

        Args:
            P (Tensor): [batch_size, 3, 20]

        Returns:
            Tensor: _description_
        """
        # from sample function, batch size is 1 and P has shape of [1,num_samples,3,20]
        P = torch.squeeze(P)
        assert P.dim() == 3
        batch_size = P.shape[0]
        transl_vec = torch.zeros([batch_size,3]).to(P.device)
        pi = torch.from_numpy(np.array([2*np.pi])).to(transl_vec.device)

        for i in range(3):
            transl_vec[:,i] = torch.atan2(P[:,i,0], P[:,i,1]) / pi

        # # Test backward pass
        # transl_vec_test = torch.zeros([batch_size,3]).to(P.device)
        # for i in range(3):
        #     transl_vec_test[:,i] = torch.atan2(P[:,i,2], P[:,i,3]) * torch.pow(torch.Tensor([10]).to(transl_vec.device), torch.Tensor([0.5]).to(transl_vec.device))
        # print(torch.allclose(transl_vec, transl_vec_test, atol=1e-09))

        return transl_vec

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
    # angle_vector_origin = deepcopy(angle_vector)
    # angle_vector[:,0] = (angle_vector[:,0] + np.pi) / 2 / np.pi
    # angle_vector[:,1] = (angle_vector[:,1] + np.pi) / 2 / np.pi
    # angle_vector[:,2] = (angle_vector[:,2] + np.pi) / 2 / np.pi

    # encoded_angle = pe.forward_localinn(angle_vector)
    # decoded_angle = pe.backward(encoded_angle)
    # decoded_angle2 = pe.backward2(encoded_angle)

    # decoded_angle[:,0] = decoded_angle[:,0] * 2 * np.pi - np.pi
    # decoded_angle[:,1] = decoded_angle[:,1] * 2 * np.pi - np.pi
    # decoded_angle[:,2] = decoded_angle[:,2] * 2 * np.pi - np.pi


    # decoded_angle[decoded_angle<-np.pi] += 2*np.pi
    # # a = decoded_angle[:,1]
    # # a[a<-np.pi/2] += np.pi/2
    # # decoded_angle[:,1] = a
    # print(angle_vector_origin - decoded_angle)
    # print(torch.allclose(decoded_angle, angle_vector_origin, atol=1e-05))



    transl_vector = torch.tensor([[-0.0593, -0.1598,  0.1257],
        [-0.0414, -0.1202, -0.0064],
        [ 0.1284, -0.0347,  0.0137],
        [ 0.0242, -0.1697,  0.0112],
        [-0.0213, -0.1556, -0.0123],
        [ 0.0150,  0.0248, -0.1444],
        [-0.0775, -0.1016,  0.0624],
        [-0.0672, -0.1618,  0.0168],
        [ 0.0531, -0.1510, -0.0088],
        [ 0.0756, -0.0943,  0.2091],
        [-0.0385, -0.1291, -0.0102],
        [-0.1179, -0.0546,  0.0336],
        [ 0.0538,  0.0035, -0.0660],
        [ 0.1627,  0.0158,  0.0032],
        [ 0.1127, -0.1102,  0.0348],
        [-0.1301, -0.0810, -0.0622],
        [ 0.1379,  0.0309,  0.0068],
        [-0.0136, -0.1331, -0.0283],
        [-0.0047, -0.2068, -0.0542],
        [ 0.0988, -0.0435,  0.1191],
        [-0.0333, -0.1529, -0.0317],
        [-0.0123, -0.1299, -0.0237],
        [-0.0104, -0.0575,  0.1561],
        [-0.1040, -0.0649,  0.0504],
        [-0.0331, -0.1111, -0.0241],
        [ 0.0381, -0.1470, -0.0417],
        [ 0.0174, -0.1722,  0.0136],
        [-0.0215, -0.0368, -0.1013],
        [-0.0390, -0.1727, -0.0006],
        [ 0.0971, -0.0981,  0.0902],
        [-0.0374, -0.1589, -0.0476],
        [-0.0162, -0.1662, -0.0148],
        [-0.0884, -0.0222, -0.1012],
        [-0.1155, -0.0527,  0.0666],
        [-0.0886, -0.0789,  0.1813],
        [-0.1064, -0.0397, -0.0282],
        [ 0.0225, -0.1687, -0.0297],
        [ 0.0113, -0.1460, -0.0157],
        [-0.0993, -0.0388,  0.0960],
        [-0.0345, -0.1369, -0.0106],
        [-0.0140, -0.1048, -0.0189],
        [ 0.1258, -0.0467,  0.0741],
        [-0.0781, -0.1058,  0.0243],
        [ 0.0078, -0.0720,  0.1023],
        [ 0.0457, -0.1165, -0.0381],
        [ 0.0732, -0.0902, -0.0118],
        [ 0.0411, -0.1215, -0.0363],
        [-0.0050, -0.1212, -0.0268],
        [-0.0084, -0.1569, -0.0659],
        [ 0.0217, -0.0691,  0.1240],
        [ 0.0522,  0.0235, -0.1106],
        [ 0.1348, -0.0142, -0.0231],
        [ 0.0081, -0.1791, -0.0047],
        [ 0.0688, -0.0401, -0.0845],
        [ 0.0515, -0.1131, -0.0853],
        [ 0.1156, -0.0418,  0.0735],
        [ 0.0581, -0.0722,  0.1499],
        [ 0.0642, -0.1585, -0.0407],
        [ 0.0208, -0.2308,  0.0191],
        [ 0.0266, -0.1803, -0.0285],
        [-0.0941, -0.0607,  0.0600],
        [-0.0251, -0.0319, -0.0668],
        [ 0.0087, -0.1553,  0.0297],
        [-0.0166, -0.1582, -0.0058]], device='cuda:0')

    transl_vector_origin = deepcopy(transl_vector)

    palm_transl_min = -0.3150945039775345
    palm_transl_max = 0.2628828995958964
    value_range = palm_transl_max - palm_transl_min
    print(value_range)
    transl_vector = (transl_vector - palm_transl_min) / (palm_transl_max - palm_transl_min)

    encoded_angle = pe.forward_transl(transl_vector)
    decoded_angle = pe.backward_transl(encoded_angle)

    decoded_angle = decoded_angle * (palm_transl_max - palm_transl_min) + palm_transl_min

    decoded_angle[decoded_angle < -value_range / 2] += value_range
    decoded_angle[decoded_angle > value_range / 2] -= value_range
    # a = decoded_angle[:,1]
    # a[a<-np.pi/2] += np.pi/2
    # decoded_angle[:,1] = a
    print(transl_vector_origin, decoded_angle)
    print(torch.allclose(decoded_angle, transl_vector_origin, atol=1e-05))
