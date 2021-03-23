import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix
import multiprocessing as mp
from joblib import Parallel, delayed
import time

def query_from_kd_tree_mp(kd_tree_z, x_query_point, d2_thres):
  """query from KD Tree in multiprocessing
    Input:
      kd_tree_z     - a KD Tree built by points in Z point cloud 
      x_query_point - X[x_idx, 0:3]
      d2_thres      - search radius
  """
  # use d2_thres as search radius and find index of points within radius
  match_idx = kd_tree_z.query_ball_point(x_query_point, r=d2_thres)

  return match_idx


def se_kernel_geo(X, Z, l=0.075, s2=0.01, sp_thres=0.0006):
  """Compute function angle from LiDAR scans geometric only
    Input: 
      A nx4 numpy array of homogeneous points (x, y, z, 1).
      l = length scal, default is ell=0.1
      s2 = sigma * sigma
    Returns: 
      A matrix
  """
  # TODO: GPU function

  # smaller point cloud for testing...
  # X = X[:100, :]
  # Z = Z[:100, :]

  # Initialize list for later creating A sparse matrix
  A_row = []
  A_col = []
  A_value = []

  # convert k threshold to d2 threshold (so that we only need to calculate k when needed)
  d2_thres = -2.0 * l * l * np.log(sp_thres/s2)

  # KD Tree initialize with (x,y,z) from Z point cloud
  kd_tree_z = KDTree(Z[:,0:3])

  # loop through points in X point cloud in multiprocessing, and find matches in KDTree
  start = time.time()
  pool = mp.Pool(mp.cpu_count())
  match_indexes = [pool.apply(query_from_kd_tree_mp, args=(kd_tree_z, X[x_idx, 0:3], d2_thres)) for x_idx in range(X.shape[0])]
  pool.close()
  stop = time.time()

  # print('match_indexes', match_indexes)
  print('time_query', stop-start)

  start = time.time()
  # loop through matched points
  for x_idx in range(X.shape[0]):
    for z_idx in match_indexes[x_idx]:
      d2 = np.linalg.norm(X[x_idx, 0:3] - Z[z_idx, 0:3])  # euclidean distrance square
      k = 0
      
      if d2 < d2_thres:
        # geometric only kernel
        k = s2 * np.exp(-d2 / (2 * l * l))

        # TODO: add color, intensity, and semantics

        # print('x_idx = %d, z_idx = %d, d2 = %.4f, k = %.4f' % (x_idx, z_idx, d2, k))

        if k > sp_thres:
          # add to A matrix
          A_row.append(x_idx)
          A_col.append(z_idx)
          A_value.append(k)
  
  stop = time.time()
  print('time_compute', stop-start)

  # build sparse A matrix
  A = csr_matrix((A_value, (A_row, A_col)), shape=(X.shape[0], Z.shape[0])).toarray()
  A_valid = np.count_nonzero(A)
  A_sum = np.sum(A)

  # print('A\n', A, '\nA_sum =', A_sum, 'A_valid =', A_valid)

  return A_sum, A_valid