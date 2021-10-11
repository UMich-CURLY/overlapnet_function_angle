#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <boost/filesystem.hpp>
#include "utils/CvoPointCloud.hpp"
#include "cvo/CvoGPU.hpp"
#include "range_projection.hpp"
#include "load_data.hpp"
using namespace std;
using namespace boost::filesystem;


int main(int argc, char *argv[]) {
  // set current frame
  string frame_index_start(argv[5]);
  int frame_idx_start = std::stoi (frame_index_start);

  // set the related parameters and sepecify paths  
  // string poses_file = "/home/cel/media/data/kitti/sequences/07/poses.txt";  // path of poses  
  // string calib_file = "/home/cel/media/data/kitti/sequences/07/calib.txt";  // path of calibration file of the dataset  
  // string scan_folder = "/home/cel/media/data/kitti/sequences/07/velodyne/";  // path of raw LiDAR scan folder  
  string poses_file(argv[1]);
  string calib_file(argv[2]);
  string scan_folder(argv[3]);

  // output file
  // string output_filename = "07_function_angle_200.csv";
  string output_filename(argv[4]);
  std::ofstream innerproduct_file(output_filename); 
  std::cout<<"The output file has been created\n";

  // set cvo align
  cvo::CvoGPU cvo_align("cvo_params/cvo_geometric_params_gpu.yaml");
  cvo::CvoParams & init_param = cvo_align.get_params();
  std::cout<<"ell_init = "<< init_param.ell_init <<std::endl;

  // load scan paths
  std::vector<string> scan_paths;
  scan_paths = load_scan_paths(scan_folder);

  // load calibrations
  Eigen::Matrix4f T_cam_velo;
  T_cam_velo = load_calib(calib_file);

  // load poses
  std::vector<Eigen::Matrix4f> poses;
  poses = load_poses(poses_file, T_cam_velo);


  // generate overlap and yaw ground truth array
  // init ground truth overlap and yaw
  std::cout << "Start to compute ground truth overlap and yaw ..." << std::endl;
  std::vector<float> overlaps;
  std::vector<int> yaw_idxs;
  int yaw_resolution = 360;

  // we calculate the ground truth for one given frame only
  // generate range projection for the given frame
  for (int frame_idx = frame_idx_start; frame_idx < scan_paths.size(); frame_idx++){
    // int frame_idx = 0; // current frame idx (want to find loop with respect to this)
    string current_scan_path = scan_paths[frame_idx];

    // load point cloud
    pcl::PointCloud<pcl::PointXYZI>::Ptr current_points (new pcl::PointCloud<pcl::PointXYZI>);
    current_points = load_data(current_scan_path);

    // range projection
    // pcl::PointCloud<pcl::PointXYZI>::Ptr current_points_after_projection = range_projection(current_points);
    pcl::PointCloud<pcl::PointXYZI>::Ptr current_points_world (new pcl::PointCloud<pcl::PointXYZI>);

    Eigen::Matrix4f current_pose = poses[frame_idx];
    pcl::transformPointCloud (*current_points, *current_points_world, current_pose);

    // create a CvoPointCloud
    cvo::CvoPointCloud current_cvo_pointcloud(current_points_world);

    Eigen::Matrix4f identity_pose = Eigen::Matrix4f::Identity();

    // loop through reference point clouds
    for (int refrence_idx = 0; refrence_idx <= frame_idx; refrence_idx++){
      std::cout<<"\n frame "<<refrence_idx<<" / "<<frame_idx<<std::endl;
      // generate range projection for the reference frame
      Eigen::Matrix4f refrence_pose = poses[refrence_idx];

      // Their way, changing the pose of current frame point cloud
      // pcl::PointCloud<pcl::PointXYZI>::Ptr reference_points;
      // pcl::transformPointCloud (*current_points_world, *reference_points, refrence_pose.inverse());
      // reference_points_after_projection = range_projection(reference_points);

      // Our way, load a new point cloud
      // initialize
      pcl::PointCloud<pcl::PointXYZI>::Ptr reference_points (new pcl::PointCloud<pcl::PointXYZI>);
      // pcl::PointCloud<pcl::PointXYZI>::Ptr reference_points_after_projection (new pcl::PointCloud<pcl::PointXYZI>);
      pcl::PointCloud<pcl::PointXYZI>::Ptr reference_points_world (new pcl::PointCloud<pcl::PointXYZI>);
      // reference path, points, and range projection
      string reference_scan_path = scan_paths[refrence_idx];
      reference_points = load_data(reference_scan_path);
      // reference_points_after_projection = range_projection(reference_points);
      pcl::transformPointCloud (*reference_points, *reference_points_world, refrence_pose);

      // create a CvoPointCloud
      cvo::CvoPointCloud reference_cvo_pointcloud(reference_points_world);

      // calculate inner product
      float inner_product = cvo_align.function_angle(current_cvo_pointcloud, reference_cvo_pointcloud, identity_pose);
      innerproduct_file << frame_idx << " " << refrence_idx << " " << inner_product<<"\n"<<std::flush;
      std::cout<<"inner product = "<<inner_product<<std::endl;
    }
  }
  innerproduct_file.close();
  
  return 0;
}
