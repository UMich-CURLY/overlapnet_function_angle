#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: This script generate the overlap and orientation combined mapping file.

try: from utils import *
except: from utils import *
from tqdm import tqdm


def com_overlap_yaw_oxford(database_scan_paths, database_poses, query_scan_paths, query_poses, leg_output_width=360):
  """compute the overlap and yaw ground truth from the ground truth poses,
     which is used for OverlapNet training and testing.
     Args:
       scan_paths: paths of all raw LiDAR scans
       poses: ground-truth poses either given by the dataset or generated by SLAM or odometry
       frame_idx: the current frame index
     Returns:
       ground_truth_mapping: the ground truth overlap and yaw used for training OverlapNet,
                             where each row contains [current_frame_idx, reference_frame_idx, overlap, yaw]
  """
  # init ground truth overlap and yaw
  print('Start to compute ground truth overlap and yaw ...')
  print('database_scan_paths', len(database_scan_paths), 'database_poses', database_poses.shape, \
        'query_scan_paths', len(query_scan_paths), 'query_poses', query_poses.shape)
  frame_idx_1 = []
  frmae_idx_2 = []
  overlaps = []
  yaw_idxs = []
  yaw_resolution = leg_output_width


  for frame_idx in tqdm(range(len(query_scan_paths))):
  
    # we calculate the ground truth for one given frame only
    # generate range projection for the given frame
    current_points = load_oxford_vertex(query_scan_paths[frame_idx])
    current_range, project_points, _, _ = range_projection(current_points)
    visible_points = project_points[current_range > 0]
    valid_num = len(visible_points)
    current_pose = query_poses[frame_idx]

    for reference_idx in range(len(database_scan_paths)):
      frame_idx_1.append(frame_idx)
      frmae_idx_2.append(reference_idx)
      # generate range projection for the reference frame
      reference_pose = database_poses[reference_idx]
      reference_points = load_oxford_vertex(database_scan_paths[reference_idx])
      reference_points_world = reference_pose.dot(reference_points.T).T
      reference_points_in_current = np.linalg.inv(current_pose).dot(reference_points_world.T).T
      reference_range, _, _, _ = range_projection(reference_points_in_current)
      
      # calculate overlap
      overlap = np.count_nonzero(
      abs(reference_range[reference_range > 0] - current_range[reference_range > 0]) < 5) / valid_num
      overlaps.append(overlap)
      
      # calculate yaw angle
      relative_transform = np.linalg.inv(current_pose).dot(reference_pose)
      relative_rotation = relative_transform[:3, :3]
      _, _, yaw = euler_angles_from_rotation_matrix(relative_rotation)

      # discretize yaw angle and shift the 0 degree to the center to make the network easy to lean
      yaw_element_idx = int(- (yaw / np.pi) * yaw_resolution//2 + yaw_resolution//2)
      yaw_idxs.append(yaw_element_idx)

  # ground truth format: each row contains [current_frame_idx, reference_frame_idx, overlap, yaw]
  ground_truth_mapping = np.zeros((len(overlaps), 4))
  ground_truth_mapping[:, 0] = frame_idx_1
  ground_truth_mapping[:, 1] = frmae_idx_2
  ground_truth_mapping[:, 2] = overlaps
  ground_truth_mapping[:, 3] = yaw_idxs
  
  print('Finish generating ground_truth_mapping!')
  
  return ground_truth_mapping
