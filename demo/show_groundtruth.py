import numpy as np

import yaml
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/utils'))
from utils import *
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
from com_function_angle import read_function_angle
import subprocess
import ffmpeg

def vis_gt(xys, ground_truth_mapping):
  """Visualize the overlap value on trajectory"""
  # set up plot

  plt.rcParams.update({'font.size': 20, 'figure.figsize': (10,8), 'lines.markersize': 5})
  # plt.rcParams["figure.figsize"] = (10,8)
  # plt.rcParams['lines.markersize'] = 5

  fig, ax = plt.subplots()
  norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True)
  mapper = cm.ScalarMappable(norm=norm, cmap="magma")  # cmap="magma" mpl.cm.cool
  mapper.set_array(ground_truth_mapping[:, 2])
  colors = np.array([mapper.to_rgba(a) for a in ground_truth_mapping[:, 2]])
  
  # sort according to overlap
  indices = np.argsort(ground_truth_mapping[:, 2])
  xys2 = xys[indices]
  
  ax.scatter(xys2[:, 0], xys2[:, 1], c=colors[indices]) # , s=20
  ax.plot(np.array([-50, 600, 600, -50]), np.array([300, 300, -350, -350]), markersize=0, color='w') 
  ax.plot(xys[ground_truth_mapping.shape[0]-1, 0], xys[ground_truth_mapping.shape[0]-1, 1], marker="*", markersize=18, color='c')
  ax.set_xlim(xmin=-50, xmax=600)
  ax.set_ylim(ymin=-350, ymax=300)
  
  ax.axis('square')
  ax.set_aspect('auto')
  ax.set_xlabel('X [m]')
  ax.set_ylabel('Y [m]')
  # ax.set_title('Ground truth function angle value')
  cbar = fig.colorbar(mapper, ax=ax)
  cbar.set_label('Predicted Similarity Measure', labelpad=5, weight='bold')

  savename = str(ground_truth_mapping.shape[0]-1).zfill(6)+".png"

  plt.tight_layout()
  plt.savefig("pics/seq00_test/"+savename)


def readfiles():
  # overlap_file = '/home/cel/CURLY/code/OverlapNet/data/07/ground_truth/ground_truth_overlap_yaw/overlaps.npy'

  # overlap_file_0 = '/home/cel/CURLY/code/OverlapNet/data/preprocess_data_demo/ground_truth/ground_truth_overlap_yaw/overlaps.npy'

  # function_angle_file_0 = '/home/cel/CURLY/code/OverlapNet/data/preprocess_data_demo/ground_truth/ground_truth_overlap_yaw/function_angles.npy'
  
  # ground_truth_mapping = np.load(overlap_file)

  # frame_idx_1 = ground_truth_mapping[:, 0]
  # frame_idx_2 = ground_truth_mapping[:, 1]
  # function_angles = ground_truth_mapping[:, 2]
  # yaw_idxs = ground_truth_mapping[:, 3]

  # print('\noverlap\n', ground_truth_mapping)
  # print('\nShape:', ground_truth_mapping.shape)
  # print('\nframe_idx_1\n', frame_idx_1)
  # print('\nframe_idx_2\n', frame_idx_2)
  # print('\nfunction_angles\n', function_angles)
  # print('\nyaw_idxs\n', yaw_idxs)

  # overlap_values = np.load(overlap_file_0)[:, 2]
  # function_angles = np.load(function_angle_file_0)[:, 0]

  # print('overlap_values', overlap_values.shape)
  # print('overlap_values', overlap_values)
  # print('function_angles', function_angles.shape)
  # print('function_angles', function_angles)
  

  # # set up plot
  # fig = plt.figure()
  # frame_index = np.arange(1101)
  # print('frame_index', frame_index)
  # plt.plot(frame_index, overlap_values, label='overlap')
  # plt.plot(frame_index, function_angles, label='function_angle')
  
  # plt.xlabel('frame index')
  # plt.ylabel('overlap / cos(function angle)')
  # plt.title('Overlap value & function angle comparison\n seq 07 frame 0 with respect to other frames')
  # plt.yticks(np.arange(0, 1, 0.1))
  # plt.legend()
  # plt.show()

  # plt.figure()
  # depth_data = np.load('/home/cel/data/kitti/sequences/07/depth/000020.npy')
  # # plt.imshow(depth_data)
  # # plt.show()
  # # plt.axis('off')
  # # plt.savefig("pics/depth_data.png", bbox_inches='tight')
  # # depth_data = (depth_data / np.max(depth_data) * 255).asarray(int)
  # plt.imsave('pics/depth_data2.png', depth_data)

  # plt.figure()
  # normal_data = np.load('/home/cel/data/kitti/sequences/07/normal/000020.npy')
  # # plt.imshow(normal_data)
  # # plt.show()
  # # plt.axis('off')
  # # plt.savefig("pics/normal_data.png", bbox_inches='tight')
  # normal_data = (normal_data + 1) / 2
  # plt.imsave('pics/normal_data2.png', normal_data)

  config_filename = 'config/demo.yml'
  
  if len(sys.argv) > 1:
    config_filename = sys.argv[1]
  
  # load the configuration file
  config = yaml.load(open(config_filename))
  
  # set the related parameters
  poses_file = "/home/cel/DockerFolder/data/kitti/sequences/00/poses.txt"
  calib_file = "/home/cel/DockerFolder/data/kitti/sequences/00/calib.txt"
  scan_folder = "/home/cel/DockerFolder/data/kitti/sequences/00/velodyne"
  dst_folder = "pics/"
  
  # src_file = ["/home/cel/DockerFolder/data/kitti/sequences/00/function_angle/00_0_1399.csv", 
  #             "/home/cel/DockerFolder/data/kitti/sequences/00/function_angle/00_1400_1999.csv", 
  #             "/home/cel/DockerFolder/data/kitti/sequences/00/function_angle/00_2000_2399.csv", 
  #             "/home/cel/DockerFolder/data/kitti/sequences/00/function_angle/00_2400_2799.csv", 
  #             "/home/cel/DockerFolder/data/kitti/sequences/00/function_angle/00_2800_3099.csv", 
  #             "/home/cel/DockerFolder/data/kitti/sequences/00/function_angle/00_3100_3199.csv", 
  #             "/home/cel/DockerFolder/data/kitti/sequences/00/function_angle/00_3200_3299.csv", 
  #             "/home/cel/DockerFolder/data/kitti/sequences/00/function_angle/00_3300_3499.csv", 
  #             "/home/cel/DockerFolder/data/kitti/sequences/00/function_angle/00_3500_3699.csv", 
  #             "/home/cel/DockerFolder/data/kitti/sequences/00/function_angle/00_3700.csv", 
  #             "/home/cel/DockerFolder/data/kitti/sequences/00/function_angle/00_3800.csv", 
  #             "/home/cel/DockerFolder/data/kitti/sequences/00/function_angle/00_3900.csv", 
  #             "/home/cel/DockerFolder/data/kitti/sequences/00/function_angle/00_4000.csv", 
  #             "/home/cel/DockerFolder/data/kitti/sequences/00/function_angle/00_4100.csv", 
  #             "/home/cel/DockerFolder/data/kitti/sequences/00/function_angle/00_4200.csv", 
  #             "/home/cel/DockerFolder/data/kitti/sequences/00/function_angle/00_4300.csv", 
  #             "/home/cel/DockerFolder/data/kitti/sequences/00/function_angle/00_4400.csv", 
  #             "/home/cel/DockerFolder/data/kitti/sequences/00/function_angle/00_4500.csv"]

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


  # ground_truth_mapping = read_function_angle(src_file)

  # np.save("/home/cel/DockerFolder/data/kitti/sequences/00/ground_truth_function_angle/temp.npy", ground_truth_mapping)

  # ground_truth_mapping = np.load("/home/cel/DockerFolder/data/kitti/sequences/00/ground_truth_function_angle/temp.npy")
  # ground_truth_mapping_overlap = np.load("/home/cel/DockerFolder/data/kitti/sequences/00/ground_truth_overlap/ground_truth_mapping_overlap.npy")

  # test result
  prediction = np.empty((0,4))
  test_result_filenames = ["/home/cel/DockerFolder/code/overlapnet_function_angle/result/seq00_test_all/seq00_test_function_angle_scratch_900k/validation_results.npz",
                           "/home/cel/DockerFolder/code/overlapnet_function_angle/result/seq00_test_all/seq00_test_function_angle_scratch_900k_1800k/validation_results.npz",
                           "/home/cel/DockerFolder/code/overlapnet_function_angle/result/seq00_test_all/seq00_test_function_angle_scratch_1800k_2700k/validation_results.npz",
                           "/home/cel/DockerFolder/code/overlapnet_function_angle/result/seq00_test_all/seq00_test_function_angle_scratch_2700k_3600k/validation_results.npz",
                           "/home/cel/DockerFolder/code/overlapnet_function_angle/result/seq00_test_all/seq00_test_function_angle_scratch_3600k_4500k/validation_results.npz",
                           "/home/cel/DockerFolder/code/overlapnet_function_angle/result/seq00_test_all/seq00_test_function_angle_scratch_4500k_5500k/validation_results.npz",
                           "/home/cel/DockerFolder/code/overlapnet_function_angle/result/seq00_test_all/seq00_test_function_angle_scratch_5500k_6500k/validation_results.npz",
                           "/home/cel/DockerFolder/code/overlapnet_function_angle/result/seq00_test_all/seq00_test_function_angle_scratch_6500k_7500k/validation_results.npz",
                           "/home/cel/DockerFolder/code/overlapnet_function_angle/result/seq00_test_all/seq00_test_function_angle_scratch_7500k_8500k/validation_results.npz",
                           "/home/cel/DockerFolder/code/overlapnet_function_angle/result/seq00_test_all/seq00_test_function_angle_scratch_8500k_end/validation_results.npz"                           
                          ]
  for test_result_filename in test_result_filenames:
    with np.load(test_result_filename) as data:
      prediction = np.vstack((prediction, data['arr_0.npy']))

  print('read file')
  return prediction, poses

def find_max(ground_truth_mapping, poses):
  
  ground_truth_mapping_max = np.zeros((4541, 3))
  for idx in range(4541):
    ground_truth_mapping_mask = ground_truth_mapping[ground_truth_mapping[:,0]==idx, :]
    ground_truth_mapping_mask2 = ground_truth_mapping_mask[ground_truth_mapping_mask[:,1]<=idx-50, :]
    if ground_truth_mapping_mask2.shape[0] > 0:
      max_idx = np.argmax(ground_truth_mapping_mask2[:,2])
      ground_truth_mapping_max[idx, 0] = idx
      ground_truth_mapping_max[idx, 1] = ground_truth_mapping_mask2[max_idx, 1]
      ground_truth_mapping_max[idx, 2] = ground_truth_mapping_mask2[max_idx, 2]

    vis_gt(poses[:, :2, 3], ground_truth_mapping_max[:idx+1,:])

  print('finished calculattion')
  return ground_truth_mapping_max


if __name__ == '__main__':
  ground_truth_mapping, poses = readfiles()
  ground_truth_mapping_max = find_max(ground_truth_mapping, poses)
  np.save("pics/seq00_test/place_recognize_max.npy", ground_truth_mapping_max)
  np.save("pics/seq00_test/place_recognize_max.npy", ground_truth_mapping_max)
  os.chdir("pics/seq00_test/")
  subprocess.call([
      'ffmpeg', '-framerate', '200', '-i', '%06d.png', '-r', '200', '-pix_fmt', 'yuv420p',
      'seq00_test_function_angle_with_robot.mp4'
  ])



