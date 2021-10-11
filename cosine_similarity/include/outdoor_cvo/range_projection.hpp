#pragma once
#include <string>
#include <fstream>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <map>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
using namespace std;
using namespace boost::filesystem;

pcl::PointCloud<pcl::PointXYZI>::Ptr range_projection(pcl::PointCloud<pcl::PointXYZI>::Ptr pc_in){

  // set up parameters
  float fov_up = 3.0;
  float fov_down = -25.0;
  int proj_H = 64;
  int proj_W = 900; 
  float max_range = 50.0;

  // laser parameters
  fov_up = fov_up / 180.0 * M_PI;  // field of view up in radians
  fov_down = fov_down / 180.0 * M_PI;  // field of view down in radians
  float fov = abs(fov_down) + abs(fov_up);  // get field of view total in radians

  // loop through points and keep track of depth
  std::vector<pair<float, int>> depth_index;

  for (int i = 0; i < pc_in->size(); i++) {
    // get scan components
    float scan_x = pc_in->points[i].x;
    float scan_y = pc_in->points[i].y;
    float scan_z = pc_in->points[i].z;

    // get depth of all points
    float depth = sqrt(scan_x * scan_x + scan_y * scan_y + scan_z * scan_z);
    if (depth <= 0 || depth >= max_range){
      continue;
    }

    // save depth and index, sort it afterwards
    depth_index.push_back(make_pair(depth, i)); 
  }

  // order in increasing depth
  sort(depth_index.begin(), depth_index.end()); 
    
  // loop from the heighest depth
  std::map<pair<int, int>, int> map_projection;

  for (int j = depth_index.size() - 1; j >= 0; j--) {
    float depth = depth_index[j].first;
    int idx = depth_index[j].second;

    // get scan components
    float scan_x = pc_in->points[idx].x;
    float scan_y = pc_in->points[idx].y;
    float scan_z = pc_in->points[idx].z;
    float intensity = pc_in->points[idx].intensity;

    // get angles of all points
    float yaw = -atan2(scan_y, scan_x);
    float pitch = asin(scan_z / depth);

    // get projections in image coords and scale to image size using angular resolution
    int proj_x = int((0.5 * (yaw / M_PI + 1.0)) * proj_W);  // in [0.0, 1.0] -> [0.0, W]
    int proj_y = int((1.0 - (pitch + abs(fov_down)) / fov) * proj_H);  // in [0.0, 1.0] -> [0.0, H]

    // round and clamp for use as index
    proj_x = min(proj_W - 1, proj_x);
    proj_x = max(0, proj_x);  // in [0,W-1]

    proj_y = min(proj_H - 1, proj_y);
    proj_y = max(0, proj_y);  // in [0,H-1]

    // save proj_x and proj_y in hash table, replace far points by near points
    map_projection[make_pair(proj_x, proj_y)] = idx;
  }

  // keep selected points
  pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out (new pcl::PointCloud<pcl::PointXYZI>);

  std::map<pair<int, int>, int>::iterator it;
  for (it = map_projection.begin(); it != map_projection.end(); it++)
  {
    int idx = it->second;

    pcl::PointXYZI point;

    // get scan components
    point.x = pc_in->points[idx].x;
    point.y = pc_in->points[idx].y;
    point.z = pc_in->points[idx].z;
    point.intensity = pc_in->points[idx].intensity;

    // save selected points
    pc_out->push_back(point); 
  }

  std::cout<<"Point cloud after range projection with "<<pc_out->size()<<" points"<<std::endl;

  return pc_out;
}