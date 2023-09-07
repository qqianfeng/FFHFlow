import numpy as np

def geodesic_distance_rotmats_pairwise_np(r1s, r2s):
  """Computes pairwise geodesic distances between two sets of rotation matrices.

  Args:
    r1s: [N, 3, 3] numpy array
    r2s: [M, 3, 3] numpy array

  Returns:
    [N, M] angular distances.
  """
  rot_rot_transpose = np.einsum('aij,bkj->abik', r1s, r2s, optimize=True)
  tr = np.trace(rot_rot_transpose, axis1=-2, axis2=-1)
  return np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))

def maad_two_grasp_distribution(grasp1, grasp2):
    pass
