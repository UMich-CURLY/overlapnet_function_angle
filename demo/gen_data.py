#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: a demo to generate data and
#        visualize different cues generated from LiDAR scans as images

import yaml
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/utils'))
import matplotlib.pyplot as plt
from utils import *
import gen_depth_data as gen_depth
import gen_normal_data as gen_normal
import gen_intensity_data as gen_intensity
import gen_semantic_data as gen_semantics


def show_images(depth_data, normal_data):
  """ This function is used to visualize different types of data
      generated from the LiDAR scan, including depth, normal, intensity and semantics.
  """
  fig, axs = plt.subplots(2, figsize=(6, 4))
  axs[0].set_title('range_data')
  axs[0].imshow(depth_data)
  axs[0].set_axis_off()
  
  axs[1].set_title('normal_data')
  axs[1].imshow(normal_data)
  axs[1].set_axis_off()

  plt.suptitle('Preprocessed data from the LiDAR scan')
  plt.show()


def gen_data(scan_folder, dst_folder, visualize=True):
  """ This function is used to generate different types of data
      from the LiDAR scan, including depth, normal, intensity and semantics.
  """
  range_data = gen_depth.gen_depth_data(scan_folder, dst_folder)[0]
  normal_data = gen_normal.gen_normal_data(scan_folder, dst_folder)[0]

  if visualize:
    show_images(range_data, normal_data)


if __name__ == '__main__':
  # load config file
  config_filename = 'config/gen_data.yml'
  if len(sys.argv) > 1:
    config_filename = sys.argv[1]
  
#   config = yaml.load(open(config_filename))
  
#   # set the related parameters
#   scan_folder = config["scan_folder"]
# #   semantic_folder = config["semantic_folder"]
#   dst_folder = config["dst_folder"]

  for seq in ["2014-11-14-16-34-33", "2014-12-09-13-21-02", "2014-12-16-09-14-09",
              "2015-02-10-11-58-05", "2015-03-10-14-18-10", "2015-06-09-15-06-29", 
              "2015-08-14-14-54-57", "2015-11-12-11-22-05", "2014-11-18-13-20-12", 
              "2014-12-10-18-10-50", "2014-12-16-18-44-24", "2015-02-13-09-16-26", 
              "2015-03-17-11-08-44", "2015-08-12-15-04-18", "2015-08-28-09-50-22", 
              "2015-11-13-10-28-08", "2014-12-02-15-30-08", "2014-12-12-10-45-15",
              "2015-02-03-08-45-10", "2015-02-17-14-42-12", "2015-05-19-14-06-38", 
              "2015-08-13-16-02-58", "2015-10-30-13-52-14"]:

    scan_folder = "/home/cel/data/oxford_test/"+seq+"/pointcloud_20m"
    dst_folder = "/home/cel/data/oxford_test/"+seq

    # start the demo1 to generate different types of data from LiDAR scan
    gen_data(scan_folder, dst_folder, False)
