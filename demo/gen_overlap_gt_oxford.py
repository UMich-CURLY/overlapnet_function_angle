#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: This demo shows how to generate the overlap and yaw ground truth files for training and testing.

import yaml
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/utils'))
from utils import *
from com_overlap_yaw_oxford import com_overlap_yaw_oxford
from normalize_data import normalize_data
from split_train_val import split_train_val
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import csv

def vis_pose(database_pose, query_pose):
  """Visualize two trajectories of database and quert"""
  # set up plot
  plt.figure()
  plt.scatter(database_pose[:,0,3], database_pose[:,1,3], c='blue')
  plt.title('Database Trajectory')
  plt.savefig('database_trajectory.png')
  plt.figure()
  plt.scatter(query_pose[:,0,3], query_pose[:,1,3], c='red')
  plt.title('Query Trajectory')
  plt.savefig('query_trajectory.png')

def Rx(theta):
  return np.matrix([[ 1, 0           , 0           ],
                   [ 0, np.cos(theta),-np.sin(theta)],
                   [ 0, np.sin(theta), np.cos(theta)]])
  
def Ry(theta):
  return np.matrix([[ np.cos(theta), 0, np.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-np.sin(theta), 0, np.cos(theta)]])
  
def Rz(theta):
  return np.matrix([[ np.cos(theta), -np.sin(theta), 0 ],
                   [ np.sin(theta), np.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])

def getGlobalPoses(ins_file, ldmrs_timestamps):
  # ins_file: ins.cvs file for each sequence, headings are 
  #           0-timestamp 1-ins_status 2-latitude 3-longitude	4-altitude	
  #           5-northing 6-easting 7-down 8-utm_zone 9-velocity_north
  #           10-velocity_east	11-velocity_down 12-roll 13-pitch 14-yaw

  # timestamps: ldms.timestamps for each sequence

  # read ins_file data from csv file
  ins_data = []

  with open(ins_file, newline='') as csvfile:
    ins_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(ins_reader)
    for row in ins_reader:
      ins_data.append(row)

  ins_data = np.array(ins_data)
  # print('ins_data', ins_data.shape)
  ins_timestamps = ins_data[:,0]
  ins_timestamps = ins_timestamps.astype(int)
  # print('ins_timestamps', ins_timestamps.shape, ins_timestamps)

  # read ldmrs_timestamps from file
  pose_timestamps = np.loadtxt(ldmrs_timestamps, dtype=str)
  pose_timestamps = pose_timestamps[:,0]
  pose_timestamps = pose_timestamps.astype(int)
  # print('pose_timestamps', pose_timestamps.shape)

  # find interpolate poses
  gt_poses = []
  for time in pose_timestamps:
    # find two closest indexes
    # the last lower index
    lower_index = np.where(ins_timestamps <= time)[0]
    if len(lower_index) == 0:
      lower_index = -1
    else:
      lower_index = lower_index[len(lower_index)-1]
      last_pose = np.eye(4)
      last_pose[0,3] = ins_data[lower_index, 5] # northing
      last_pose[1,3] = ins_data[lower_index, 6] # easting
      last_pose[2,3] = ins_data[lower_index, 7] # down
      phi = float(ins_data[lower_index, 12]) # roll
      theta = float(ins_data[lower_index, 13]) # pitch
      psi = float(ins_data[lower_index, 14]) # yaw
      last_pose[0:3,0:3] = Rz(psi) * Ry(theta) * Rx(phi)
    
    # the first upper index
    upper_index = np.where(ins_timestamps > time)[0]
    if len(upper_index) == 0:
      upper_index = -1
    else:
      upper_index = upper_index[0]
      next_pose = np.eye(4)
      next_pose[0,3] = ins_data[upper_index, 5] # northing
      next_pose[1,3] = ins_data[upper_index, 6] # easting
      next_pose[2,3] = ins_data[upper_index, 7] # down
      phi = float(ins_data[upper_index, 12]) # roll
      theta = float(ins_data[upper_index, 13]) # pitch
      psi = float(ins_data[upper_index, 14]) # yaw
      next_pose[0:3,0:3] = Rz(psi) * Ry(theta) * Rx(phi)

    scale = (time - ins_timestamps[lower_index]) / (ins_timestamps[upper_index] - ins_timestamps[lower_index])
    # print('current', time, 'lower', ins_timestamps[lower_index], 'upper', ins_timestamps[upper_index], 'scale', scale)


    # interpolate poses

    if lower_index == -1:
      current_pose = next_pose
    elif upper_index == -1:    
      current_pose = last_pose
    else:
      current_pose = np.eye(4)
      current_pose[0:3,3] = last_pose[0:3,3] + scale * (next_pose[0:3,3] - last_pose[0:3,3])
      phi = float(ins_data[lower_index, 12]) + scale * (float(ins_data[upper_index, 12]) - float(ins_data[lower_index, 12])) # roll
      theta = float(ins_data[lower_index, 13]) + scale * (float(ins_data[upper_index, 13]) - float(ins_data[lower_index, 13])) # pitch
      psi = float(ins_data[lower_index, 14]) + scale * (float(ins_data[upper_index, 14]) - float(ins_data[lower_index, 14])) # yaw
      current_pose[0:3,0:3] = Rz(psi) * Ry(theta) * Rx(phi)
    
    gt_poses.append(current_pose)

  print('gt_poses', len(gt_poses))

  return gt_poses


if __name__ == '__main__':
  config_filename = 'config/oxford.yml'
  
  if len(sys.argv) > 1:
    config_filename = sys.argv[1]
  
  # load the configuration file
  config = yaml.load(open(config_filename), Loader=yaml.FullLoader)
  
  # set the related parameters
  database_poses_file = config['database_poses_file']
  query_poses_file = config['query_poses_file']
  database_scan_folder = config['database_scan_folder']
  query_scan_folder = config['query_scan_folder']
  database_timestamp_file = config['database_timestamp_file']
  query_timestamp_file = config['query_timestamp_file']
  dst_folder = config['dst_folder']

  # add sequence label to the data and save them as npz files
  seq_idx = config['query_seq']

  # specify the goal folder
  dst_folder = os.path.join(dst_folder, 'ground_truth_overlap')
  try:
    os.stat(dst_folder)
    print('generating depth data in: ', dst_folder)
  except:
    print('creating new depth folder: ', dst_folder)
    os.mkdir(dst_folder)

  if 'precomputed_file' in config:
    ground_truth_mapping = np.load(config['precomputed_file'])
  else:
  
    # load scan paths
    database_scan_paths = load_files(database_scan_folder)
    query_scan_paths = load_files(query_scan_folder)

    # load poses

    # database_poses = []
    # try:
    #   with open(database_poses_file, 'r') as csvfile:
    #     posereader = csv.reader(csvfile, delimiter=',')
    #     next(posereader)
    #     for line in posereader:
    #       current_pose = np.eye(4)
    #       # print(line)
    #       current_pose[0,3] = float(line[1])
    #       current_pose[1,3] = float(line[2])
    #       database_poses.append(current_pose)    
    # except FileNotFoundError:
    #     print('Ground truth poses are not avaialble.')

    database_poses = getGlobalPoses(database_poses_file, database_timestamp_file)
    database_poses = np.array(database_poses)
    # np.savetxt(os.path.join(database_scan_folder, 'ldmrs_pose.txt'), database_poses.reshape(-1, 16))
    # pose0_inv = np.linalg.inv(database_poses[0])

    # poses_new = []
    # for pose in database_poses:
    #   poses_new.append(pose0_inv.dot(pose))
    # database_poses = np.array(poses_new)


    # query_poses = []
    # try:
    #   with open(query_poses_file, 'r') as csvfile:
    #     posereader = csv.reader(csvfile, delimiter=',')
    #     next(posereader)
    #     for line in posereader:
    #       current_pose = np.eye(4)
    #       # print(line)
    #       current_pose[0,3] = float(line[1])
    #       current_pose[1,3] = float(line[2])
    #       query_poses.append(current_pose)    
    # except FileNotFoundError:
    #     print('Ground truth poses are not avaialble.')
  
    query_poses = getGlobalPoses(query_poses_file, query_timestamp_file)
    query_poses = np.array(query_poses)
    # pose0_inv = np.linalg.inv(query_poses[0])

    # poses_new = []
    # for pose in query_poses:
    #   poses_new.append(pose0_inv.dot(pose))
    # query_poses = np.array(poses_new)

    # visualize database and query poses
    # vis_pose(database_poses, query_poses)

    # generate overlap and yaw ground truth array
    ground_truth_mapping = com_overlap_yaw_oxford(database_scan_paths, database_poses, query_scan_paths, query_poses)
    
    # save this before normalization
    numpy_output_path = os.path.join(dst_folder, 'ground_truth_mapping_overlap.npy')
    np.save(numpy_output_path, ground_truth_mapping)
    # np.savetxt(os.path.join(query_scan_folder, 'ldmrs_pose.txt'), database_poses.reshape(-1, 16))
  
  # normalize the distribution of ground truth data
  dist_norm_data = normalize_data(ground_truth_mapping)
  
  # split ground truth for training and validation
  train_data, validation_data = split_train_val(dist_norm_data)
      
  # training data
  train_seq = np.empty((train_data.shape[0], 2), dtype=object)
  train_seq[:] = seq_idx
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
  