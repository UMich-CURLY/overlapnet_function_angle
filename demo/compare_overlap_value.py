#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: This demo shows how to generate the overlap and yaw ground truth files for training and testing.

import yaml
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/utils'))
from utils import *
from com_overlap_yaw import com_overlap_yaw
from normalize_data import normalize_data
from split_train_val import split_train_val
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


def compute_overlap_with_different_angles(scan_path, angles, yaw_resolution=360):
  """compute the overlap value with different yaw angles.
     Args:
       scan_path: one path to the raw LiDAR scan
       angles: differnet yaw angles
     Returns:
       overlap_values: overlap with different yaw angles,
                             where each row contains [angle, overlap, yaw]
  """
  # init ground truth overlap and yaw
  print('Start to compute overlap with different yaw angles ...')
  overlaps = []
  yaw_idxs = []

  current_points = load_vertex(scan_path)
  reference_points = load_vertex(scan_path)

  current_range, project_points, _, _ = range_projection(current_points)
  visible_points = project_points[current_range > 0]
  valid_num = len(visible_points)
  
  relative_transform = np.eye(4)

  for angle in tqdm(angles): 
    # construct rotation matrix based on the yaw angle
    relative_rotation = R.from_euler('z', angle, degrees=True)
    relative_transform[:3, :3] = relative_rotation.as_matrix()
    
    # transform the points based on the yaw angle
    reference_points_in_current = relative_transform.dot(reference_points.T).T
    reference_range, _, _, _ = range_projection(reference_points_in_current)

    # calculate overlap
    overlap = np.count_nonzero(
      abs(reference_range[reference_range > 0] - current_range[reference_range > 0]) < 1) / valid_num
    overlaps.append(overlap)
    
    # calculate yaw angle
    _, _, yaw = euler_angles_from_rotation_matrix(relative_rotation.as_matrix())

    # discretize yaw angle and shift the 0 degree to the center to make the network easy to learn
    yaw_element_idx = int(- (yaw / np.pi) * yaw_resolution//2 + yaw_resolution//2)
    yaw_idxs.append(yaw_element_idx)

  overlap_values = np.zeros((len(overlaps), 3))
  overlap_values[:, 0] = angles
  overlap_values[:, 1] = overlaps
  overlap_values[:, 2] = yaw_idxs

  
  print('Finish generating overlap_values!')
  
  return overlap_values

def vis_overlap_angle(overlap_values, save_path):
  """Visualize the overlap value versus yaw angle values"""

  plt.plot(overlap_values[:,0], overlap_values[:,1])
  plt.xlabel('Yaw Angle Difference')
  plt.ylabel('Overlap Values')
  plt.title('Yaw Angle vs. Overlap Values')
  plt.savefig(save_path)


if __name__ == '__main__':
  config_filename = 'config/demo.yml'
  
  if len(sys.argv) > 1:
    config_filename = sys.argv[1]
  
  # load the configuration file
  config = yaml.load(open(config_filename))
  
  # set the related parameters
  poses_file = config['Demo4']['poses_file']
  calib_file = config['Demo4']['calib_file']
  scan_folder = config['Demo4']['scan_folder']
  dst_folder = config['Demo4']['dst_folder']
  
  # load scan paths
  scan_paths = load_files(scan_folder)

  # load calibrations
  T_cam_velo = load_calib(calib_file)
  T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
  T_velo_cam = np.linalg.inv(T_cam_velo)

  # load poses
  poses = load_poses(poses_file)
  pose0_inv = np.linalg.inv(poses[0])

  # for KITTI dataset, we need to convert the provided poses 
  # from the camera coordinate system into the LiDAR coordinate system  
  poses_new = []
  for pose in poses:
    poses_new.append(T_velo_cam.dot(pose0_inv).dot(pose).dot(T_cam_velo))
  poses = np.array(poses_new)

  angles = np.arange(-180, 180, 0.1)

  # generate overlap and yaw ground truth array
  overlap_values = compute_overlap_with_different_angles(scan_paths[0], angles)
  
  
  # specify the goal folder
  dst_folder = os.path.join(dst_folder, 'overlap_value_with_different_yaw_angle')
  try:
    os.stat(dst_folder)
    print('generating depth data in: ', dst_folder)
  except:
    print('creating new depth folder: ', dst_folder)
    os.mkdir(dst_folder)

  # save csv file
  output_path = os.path.join(dst_folder, 'overlap_value_yaw_angle.csv')
  np.savetxt(output_path, overlap_values, delimiter=",")
  
  print('Finish saving the overlap values for different yaw angles at: ', dst_folder)
  
  # visualize the overlap value versus yaw angles
  plot_path = os.path.join(dst_folder, 'overlap_value_versus_yaw_angle_detail.png')
  vis_overlap_angle(overlap_values, plot_path)
  
  





