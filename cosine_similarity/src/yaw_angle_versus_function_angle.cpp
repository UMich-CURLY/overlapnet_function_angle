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

#include <chrono>
using namespace std::chrono;

int main(int argc, char *argv[]) {
  string poses_file(argv[1]);
  string calib_file(argv[2]);
  string scan_folder(argv[3]);

  // output file
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
  int frame_idx = 0;
  string current_scan_path = scan_paths[frame_idx];

  Eigen::Matrix4f identity_pose = Eigen::Matrix4f::Identity();
    
  // load point cloud
  pcl::PointCloud<pcl::PointXYZI>::Ptr current_points (new pcl::PointCloud<pcl::PointXYZI>);
  current_points = load_data(current_scan_path);

  // range projection
  pcl::PointCloud<pcl::PointXYZI>::Ptr current_points_after_projection = range_projection(current_points);
  
  // create a CvoPointCloud
  cvo::CvoPointCloud current_cvo_pointcloud(current_points_after_projection);
  
  // we calculate the ground truth for one given frame only
  auto start = high_resolution_clock::now();
  int computation_count = 0;
  for (float angle = -180; angle < 180; angle+=0.5){
    computation_count += 1;
    std::cout<<"\n angle "<<angle<<std::endl;
    // new point cloud
    pcl::PointCloud<pcl::PointXYZI>::Ptr reference_points_transformed (new pcl::PointCloud<pcl::PointXYZI>);
    
    // construct transformation matrix
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    float theta = M_PI/180*angle; // The angle of rotation in radians
    transform.rotate (Eigen::AngleAxisf (theta, Eigen::Vector3f::UnitX()));

    // Executing the transformation
    pcl::transformPointCloud (*current_points_after_projection, *reference_points_transformed, transform);

    // create a CvoPointCloud
    cvo::CvoPointCloud reference_cvo_pointcloud(reference_points_transformed);

    // calculate inner product
    float inner_product = cvo_align.function_angle(current_cvo_pointcloud, reference_cvo_pointcloud, identity_pose);
    innerproduct_file << angle << " " << inner_product<<"\n"<<std::flush;
    std::cout<<"inner product = "<<inner_product<<std::endl;

  }
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<seconds>(stop - start);
  cout << "duration = " << duration.count() << "seconds for " << computation_count << "times of computation" << endl;
  innerproduct_file.close();
  
  return 0;
}
