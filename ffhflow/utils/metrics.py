import numpy as np
import torch

def geodesic_distance_rotmats_pairwise_tf(r1s, r2s):
    """TensorFlow version of `geodesic_distance_rotmats_pairwise_np`."""
    # These are the traces of R1^T R2
    trace = torch.einsum('aij,bij->ab', r1s, r2s)
    return torch.acos(torch.clip_by_value((trace - 1.0) / 2.0, -1.0, 1.0))


def geodesic_distance_rotmats_pairwise_np(r1s, r2s):
    """Computes pairwise geodesic distances between two sets of rotation matrices.

    Args:
      r1s: [N, 3, 3] numpy array
      r2s: [M, 3, 3] numpy array

    Returns:
      [N, M] angular distances.
    """
    rot_rot_transpose = np.einsum('aij,bkj->abik', r1s, r2s, optimize=True) #[N,M,3,3]
    tr = np.trace(rot_rot_transpose, axis1=-2, axis2=-1) #[N,M]
    return np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))


def euclidean_distance_points_pairwise_np(pt1, pt2):
    """_summary_

    Args:
        pt1 (_type_): [N, 3] numpy array, predicted grasp translation
        pts (_type_): [M, 3] numpy array, ground truth grasp translation

    Returns:
        dist_mat _type_: [N,M]
    """
    dist_mat = np.zeros((pt1.shape[0],pt2.shape[0]))
    for idx in range(pt1.shape[0]):
        deltas = pt2 - pt1[idx]
        dist_2 = np.einsum('ij,ij->i', deltas, deltas)
        dist_mat[idx] = dist_2
    return dist_mat


def compute_spread(rotations, rotations_gt):
    """Measures the spread of a distribution (or mode) around ground truth(s).

    In the case of one ground truth, this is the expected angular error.
    When there are multiple ground truths, the only related quantity that makes
    sense is the expected angular error to the nearest ground truth.

    Args:
      rotations: The grid of rotation matrices for which the probabilities were
        evaluated.
      probabilities: The probability for each rotation.
      rotations_gt: The set of ground truth rotation matrices.
    Returns:
      A scalar, the spread (in radians).
    """
    dists = geodesic_distance_rotmats_pairwise_np(rotations, rotations_gt)
    min_distance_to_gt = np.min(dists, axis=1)
    return (min_distance_to_gt).sum()


def maad_for_grasp_distribution(grasp1, grasp2):
    if torch.is_tensor(grasp1['rot_matrix']):
        grasp1['rot_matrix'] = grasp1['rot_matrix'].cpu().data.numpy()
        grasp1['transl'] = grasp1['transl'].cpu().data.numpy()
        grasp1['pred_joint_conf'] = grasp1['pred_joint_conf'].cpu().data.numpy()
    transl_dist_mat = euclidean_distance_points_pairwise_np(grasp1['transl'], grasp2['transl'])
    rot_dist_mat = geodesic_distance_rotmats_pairwise_np(grasp1['rot_matrix'], grasp2['rot_matrix'])
    transl_loss = np.mean(transl_dist_mat, axis=1)  # [N,1]
    rot_loss = np.zeros_like(transl_loss)
    for idx in range(transl_loss.shape[0]):
        rot_loss[idx] = rot_dist_mat[idx, np.argmin(transl_dist_mat[idx])]

    return transl_loss, rot_loss



if __name__ == "__main__":
    a = np.zeros((1,3,3))
    b = np.ones((1,3,3))
    geodesic_distance_rotmats_pairwise_np(a,b)