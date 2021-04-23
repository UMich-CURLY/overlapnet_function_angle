#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: This demo shows how to generate the overlap and yaw ground truth files for training and testing.

import yaml
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/utils'))
from utils import *
from com_function_angle import com_function_angle, read_function_angle_com_yaw, read_function_angle_com_overlap_yaw
from normalize_data import normalize_data
from normalize_balance import normalize_balance
from split_train_val import split_train_val
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm


def vis_gt(xys, ground_truth_mapping):
  """Visualize the overlap value on trajectory"""
  # set up plot
  fig, ax = plt.subplots()
  norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True)
  mapper = cm.ScalarMappable(norm=norm)  # cmap="magma"
  mapper.set_array(ground_truth_mapping[:, 2])
  colors = np.array([mapper.to_rgba(a) for a in ground_truth_mapping[:, 2]])
  
  # sort according to overlap
  indices = np.argsort(ground_truth_mapping[:, 2])
  xys = xys[indices]
  
  ax.scatter(xys[:, 0], xys[:, 1], c=colors[indices], s=10)
  
  ax.axis('square')
  ax.set_xlabel('X [m]')
  ax.set_ylabel('Y [m]')
  ax.set_title('Demo 4: Generate ground truth for training')
  cbar = fig.colorbar(mapper, ax=ax)
  cbar.set_label('cos(function angle)', rotation=270, weight='bold')
  plt.show()


if __name__ == '__main__':
  config_filename = 'config/demo.yml'
  
  if len(sys.argv) > 1:
    config_filename = sys.argv[1]
  
  # load the configuration file
  config = yaml.load(open(config_filename))
  
  # set the related parameters
  seq_idx = config['Demo4']['seq']
  poses_file = config['Demo4']['poses_file']
  calib_file = config['Demo4']['calib_file']
  scan_folder = config['Demo4']['scan_folder']
  dst_folder = config['Demo4']['dst_folder']
  funcangle_file = config['Demo4']['function_angle_file']
  dst_folder = os.path.join(dst_folder, 'ground_truth_both_balance') # ground_truth_both_nfa, ground_truth_function_angle_no, ground_truth_overlap_nfa

  # specify the goal folder
  try:
    os.stat(dst_folder)
    print('generating depth data in: ', dst_folder)
  except:
    print('creating new depth folder: ', dst_folder)
    os.mkdir(dst_folder)

  if config['Demo4']['precomputed_file'] is not None:
    ground_truth_mapping = np.load(config['Demo4']['precomputed_file'])
  else:
  
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

    # generate overlap and yaw ground truth array
    # ground_truth_mapping = com_function_angle(scan_paths, poses, frame_idx=0)
    ground_truth_mapping = read_function_angle_com_overlap_yaw(scan_paths, poses, funcangle_file)

    # save this before normalization
    numpy_output_path = os.path.join(dst_folder, 'ground_truth_mapping.npy')
    np.save(numpy_output_path, ground_truth_mapping)
  
  # check yaw angle distribution before normalization
  normalize_idx = 4
  gt_map = ground_truth_mapping
  bin_0_9 = gt_map[np.where(gt_map[:, normalize_idx] < 36)]
  bin_10_19 = gt_map[(gt_map[:, normalize_idx] < 72) & (gt_map[:, normalize_idx] >= 36)]
  bin_20_29 = gt_map[(gt_map[:, normalize_idx] < 108) & (gt_map[:, normalize_idx] >= 72)]
  bin_30_39 = gt_map[(gt_map[:, normalize_idx] < 144) & (gt_map[:, normalize_idx] >= 108)]
  bin_40_49 = gt_map[(gt_map[:, normalize_idx] < 180) & (gt_map[:, normalize_idx] >= 144)]
  bin_50_59 = gt_map[(gt_map[:, normalize_idx] < 216) & (gt_map[:, normalize_idx] >= 180)]
  bin_60_69 = gt_map[(gt_map[:, normalize_idx] < 252) & (gt_map[:, normalize_idx] >= 216)]
  bin_70_79 = gt_map[(gt_map[:, normalize_idx] < 288) & (gt_map[:, normalize_idx] >= 252)]
  bin_80_89 = gt_map[(gt_map[:, normalize_idx] < 324) & (gt_map[:, normalize_idx] >= 288)]
  bin_90_100 = gt_map[(gt_map[:, normalize_idx] <= 360) & (gt_map[:, normalize_idx] >= 324)]

  # print the distribution
  distribution = [len(bin_0_9), len(bin_10_19), len(bin_20_29), len(bin_30_39), len(bin_40_49),
                  len(bin_50_59), len(bin_60_69), len(bin_70_79), len(bin_80_89), len(bin_90_100)]
  print('\nyaw angle distribution before normalization\n', distribution)
  
  print('ground_truth_mapping\n', ground_truth_mapping.shape)
  # normalize the distribution of ground truth data -both
  # use overlap value to normalize = 2, use function angle value to normalize = 3
  dist_norm_data_both = normalize_data(ground_truth_mapping, 2) 

  # check result for another value
  normalize_idx = 3
  gt_map = dist_norm_data_both
  bin_0_9 = gt_map[np.where(gt_map[:, normalize_idx] < 0.1)]
  bin_10_19 = gt_map[(gt_map[:, normalize_idx] < 0.2) & (gt_map[:, normalize_idx] >= 0.1)]
  bin_20_29 = gt_map[(gt_map[:, normalize_idx] < 0.3) & (gt_map[:, normalize_idx] >= 0.2)]
  bin_30_39 = gt_map[(gt_map[:, normalize_idx] < 0.4) & (gt_map[:, normalize_idx] >= 0.3)]
  bin_40_49 = gt_map[(gt_map[:, normalize_idx] < 0.5) & (gt_map[:, normalize_idx] >= 0.4)]
  bin_50_59 = gt_map[(gt_map[:, normalize_idx] < 0.6) & (gt_map[:, normalize_idx] >= 0.5)]
  bin_60_69 = gt_map[(gt_map[:, normalize_idx] < 0.7) & (gt_map[:, normalize_idx] >= 0.6)]
  bin_70_79 = gt_map[(gt_map[:, normalize_idx] < 0.8) & (gt_map[:, normalize_idx] >= 0.7)]
  bin_80_89 = gt_map[(gt_map[:, normalize_idx] < 0.9) & (gt_map[:, normalize_idx] >= 0.8)]
  bin_90_100 = gt_map[(gt_map[:, normalize_idx] <= 1) & (gt_map[:, normalize_idx] >= 0.9)]

  # print the distribution
  distribution = [len(bin_0_9), len(bin_10_19), len(bin_20_29), len(bin_30_39), len(bin_40_49),
                  len(bin_50_59), len(bin_60_69), len(bin_70_79), len(bin_80_89), len(bin_90_100)]
  print('\nanother distribution after normalization\n', distribution)

  # check result for yaw angle
  normalize_idx = 4
  gt_map = dist_norm_data_both
  bin_0_9 = gt_map[np.where(gt_map[:, normalize_idx] < 36)]
  bin_10_19 = gt_map[(gt_map[:, normalize_idx] < 72) & (gt_map[:, normalize_idx] >= 36)]
  bin_20_29 = gt_map[(gt_map[:, normalize_idx] < 108) & (gt_map[:, normalize_idx] >= 72)]
  bin_30_39 = gt_map[(gt_map[:, normalize_idx] < 144) & (gt_map[:, normalize_idx] >= 108)]
  bin_40_49 = gt_map[(gt_map[:, normalize_idx] < 180) & (gt_map[:, normalize_idx] >= 144)]
  bin_50_59 = gt_map[(gt_map[:, normalize_idx] < 216) & (gt_map[:, normalize_idx] >= 180)]
  bin_60_69 = gt_map[(gt_map[:, normalize_idx] < 252) & (gt_map[:, normalize_idx] >= 216)]
  bin_70_79 = gt_map[(gt_map[:, normalize_idx] < 288) & (gt_map[:, normalize_idx] >= 252)]
  bin_80_89 = gt_map[(gt_map[:, normalize_idx] < 324) & (gt_map[:, normalize_idx] >= 288)]
  bin_90_100 = gt_map[(gt_map[:, normalize_idx] <= 360) & (gt_map[:, normalize_idx] >= 324)]

  # print the distribution
  distribution = [len(bin_0_9), len(bin_10_19), len(bin_20_29), len(bin_30_39), len(bin_40_49),
                  len(bin_50_59), len(bin_60_69), len(bin_70_79), len(bin_80_89), len(bin_90_100)]
  print('\nyaw angle distribution after normalization\n', distribution)
  
  # # we only want function angle data this time
  # dist_norm_data_both = np.hstack((dist_norm_data_both[:, :2], dist_norm_data_both[:, 3:]))

  # # we only want overlap data this time
  # dist_norm_data_both = np.hstack((dist_norm_data_both[:, :3], dist_norm_data_both[:, 4].reshape(-1,1)))

  # split ground truth for training and validation
  train_data, validation_data = split_train_val(dist_norm_data_both)
    
  # training data
  train_seq = np.empty((train_data.shape[0], 2), dtype=object)
  train_seq[:] = seq_idx # add sequence label to the data and save them as npz files 
  np.savez_compressed(dst_folder + '/train_set', overlaps=train_data, seq=train_seq)
  
  # validation data
  validation_seq = np.empty((validation_data.shape[0], 2), dtype=object)
  validation_seq[:] = seq_idx
  np.savez_compressed(dst_folder + '/validation_set', overlaps=validation_data, seq=validation_seq)
  
  # raw ground truth data, fully mapping, could be used for testing
  ground_truth_seq = np.empty((ground_truth_mapping.shape[0], 2), dtype=object)
  ground_truth_seq[:] = seq_idx
  np.savez_compressed(dst_folder + '/ground_truth_overlap_yaw', overlaps=ground_truth_mapping, seq=ground_truth_seq)
  
  print('Finish saving the ground truth data for training and testing at: ', dst_folder)
  
  # visualize the raw ground truth mapping
  # vis_gt(poses[:, :2, 3], ground_truth_mapping)

