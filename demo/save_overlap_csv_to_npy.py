import numpy as np
import csv


if __name__ == '__main__':
  function_angles = []
  # read function_angle from saved csv file
  with open('/home/cel/CURLY/code/DockerFolder/data/kitti/sequences/07_overlap/preprocess_data_demo/07_0.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
      _, function_angle = row
      function_angles.append(float(function_angle))

  # to numpy array
  function_angles = np.array(function_angles).reshape(-1,1)

  # save to npy file
  np.save('/home/cel/CURLY/code/OverlapNet/data/preprocess_data_demo/ground_truth/ground_truth_overlap_yaw/function_angles.npy', function_angles)