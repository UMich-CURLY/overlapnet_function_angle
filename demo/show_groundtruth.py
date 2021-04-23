import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm


if __name__ == '__main__':
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

  plt.figure()
  depth_data = np.load('/home/cel/data/kitti/sequences/07/depth/000020.npy')
  # plt.imshow(depth_data)
  # plt.show()
  # plt.axis('off')
  # plt.savefig("pics/depth_data.png", bbox_inches='tight')
  # depth_data = (depth_data / np.max(depth_data) * 255).asarray(int)
  plt.imsave('pics/depth_data2.png', depth_data)

  plt.figure()
  normal_data = np.load('/home/cel/data/kitti/sequences/07/normal/000020.npy')
  # plt.imshow(normal_data)
  # plt.show()
  # plt.axis('off')
  # plt.savefig("pics/normal_data.png", bbox_inches='tight')
  normal_data = (normal_data + 1) / 2
  plt.imsave('pics/normal_data2.png', normal_data)





