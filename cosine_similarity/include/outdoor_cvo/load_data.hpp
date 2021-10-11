#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <boost/filesystem.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
using namespace std;
using namespace boost::filesystem;


std::vector<string> load_scan_paths(string scan_folder){
  std::vector<string> scan_names;
  std::vector<string> scan_paths;
  for (const auto & p : directory_iterator(scan_folder)) {
    if (is_regular_file(p.path())) {
      string curr_file = p.path().filename().string();
      size_t last_ind = curr_file.find_last_of(".");
      string raw_name = curr_file.substr(0, last_ind);
      scan_names.push_back(raw_name);
    }
  }
  sort(scan_names.begin(), scan_names.end());

  // add back folder name
  for (int i=0; i < scan_names.size(); i++){
    scan_paths.push_back(scan_folder + scan_names[i] + ".bin");
  }
  return scan_paths;
}

Eigen::Matrix4f load_calib(string calib_file){
  Eigen::Matrix4f T_cam_velo = Eigen::Matrix4f::Identity();

  std::ifstream calib_infile(calib_file);
  if (calib_infile.is_open()) {
    string header;
    float t11, t12 ,t13, t14, t21, t22, t23, t24, t31, t32, t33, t34;
    while(calib_infile >> header >> t11 >> t12 >> t13 >> t14 >> t21 >> t22 >> t23 >> t24 >> t31 >> t32 >> t33 >> t34) {  
      if (header == "Tr:"){
        T_cam_velo(0,0) = t11;
        T_cam_velo(0,1) = t12;
        T_cam_velo(0,2) = t13;
        T_cam_velo(0,3) = t14;
        T_cam_velo(1,0) = t21;
        T_cam_velo(1,1) = t22;
        T_cam_velo(1,2) = t23;
        T_cam_velo(1,3) = t24;
        T_cam_velo(2,0) = t31;
        T_cam_velo(2,1) = t32;
        T_cam_velo(2,2) = t33;
        T_cam_velo(2,3) = t34;
        // std::cout<<"T_cam_velo: \n"<< t11 <<" "<< t12 <<" "<< t13 <<" "<< t14 <<" "<< t21 <<" "<< t22 <<" "<< t23 <<" "<< t24 <<" "<< t31 <<" "<< t32 <<" "<< t33 <<" "<< t34 << std::endl;
      }
    }
    if (!calib_infile.eof()) {
    }
    calib_infile.close();
  } else {
      std::cerr<<"Calibrations are not avaialble.\n";
  }
  return T_cam_velo;
}

std::vector<Eigen::Matrix4f> load_poses(string poses_file, Eigen::Matrix4f T_cam_velo){
  std::vector<Eigen::Matrix4f> poses;
  Eigen::Matrix4f pose0_inv = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f T_velo_cam = T_cam_velo.inverse();

  std::ifstream pose_infile(poses_file);
  if (pose_infile.is_open()) {
    float t11, t12 ,t13, t14, t21, t22, t23, t24, t31, t32, t33, t34;
    while(pose_infile >> t11 >> t12 >> t13 >> t14 >> t21 >> t22 >> t23 >> t24 >> t31 >> t32 >> t33 >> t34) {
      Eigen::Matrix4f TF = Eigen::Matrix4f::Identity();
      TF(0,0) = t11;
      TF(0,1) = t12;
      TF(0,2) = t13;
      TF(0,3) = t14;
      TF(1,0) = t21;
      TF(1,1) = t22;
      TF(1,2) = t23;
      TF(1,3) = t24;
      TF(2,0) = t31;
      TF(2,1) = t32;
      TF(2,2) = t33;
      TF(2,3) = t34;

      // for KITTI dataset, we need to convert the provided poses 
      // from the camera coordinate system into the LiDAR coordinate system  
      if (poses.size() == 0){  // first pose
        pose0_inv = TF.inverse();
        TF = Eigen::Matrix4f::Identity();
        // std::cout<<"first TF: \n"<< t11 <<" "<< t12 <<" "<< t13 <<" "<< t14 <<" "<< t21 <<" "<< t22 <<" "<< t23 <<" "<< t24 <<" "<< t31 <<" "<< t32 <<" "<< t33 <<" "<< t34 << std::endl;
      }
      else{
        TF = T_velo_cam * pose0_inv * TF * T_cam_velo;
      }
      
      poses.push_back(TF);
      // std::cout<<"last TF: \n"<< t11 <<" "<< t12 <<" "<< t13 <<" "<< t14 <<" "<< t21 <<" "<< t22 <<" "<< t23 <<" "<< t24 <<" "<< t31 <<" "<< t32 <<" "<< t33 <<" "<< t34 << std::endl;
    }
    if (!pose_infile.eof()) {
    }
    pose_infile.close();
  } else {
      std::cerr<<"Poses are not avaialble.\n";
  }
  return poses;
}



pcl::PointCloud<pcl::PointXYZI>::Ptr load_data(string current_scan_path){
  pcl::PointCloud<pcl::PointXYZI>::Ptr current_points (new pcl::PointCloud<pcl::PointXYZI>);

  std::ifstream scan_infile(current_scan_path, std::ios::in | std::ios::binary);
	if(!scan_infile.good()){
		std::cerr << "Could not read velodyne file: " << current_scan_path << std::endl;
		exit(EXIT_FAILURE);
	}
	scan_infile.seekg(0, ios::beg);

	for (int i = 0; scan_infile.good() && !scan_infile.eof(); i++) {
		pcl::PointXYZI point;
		scan_infile.read((char *) &point.x, 3*sizeof(float));
		scan_infile.read((char *) &point.intensity, sizeof(float));
		current_points->push_back(point);
	}
	scan_infile.close();

  std::cout<<"Loaded point cloud from bin file with "<<current_points->size()<<" points"<<std::endl;

  return current_points;
}