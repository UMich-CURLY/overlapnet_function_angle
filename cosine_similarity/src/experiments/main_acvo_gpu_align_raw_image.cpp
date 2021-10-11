#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <boost/filesystem.hpp>
//#include <opencv2/opencv.hpp>
#include "dataset_handler/KittiHandler.hpp"
#include "graph_optimizer/Frame.hpp"
#include "utils/Calibration.hpp"
#include "utils/CvoPointCloud.hpp"
#include "cvo/AdaptiveCvoGPU.hpp"
#include "cvo/Cvo.hpp"
#include "cvo/CvoParams.hpp"
using namespace std;
using namespace boost::filesystem;


int main(int argc, char *argv[]) {
  // list all files in current directory.
  //You could put any file path in here, e.g. "/home/me/mwah" to list that directory
  cvo::KittiHandler kitti(argv[1], 0);
  int total_iters = kitti.get_total_number();
  string cvo_param_file(argv[2]);
  string calib_file;
  calib_file = string(argv[1] ) +"/cvo_calib.txt"; 
  cvo::Calibration calib(calib_file);
  std::ofstream accum_output(argv[3]);
  int start_frame = std::stoi(argv[4]);
  kitti.set_start_index(start_frame);
  int max_num = std::stoi(argv[5]);
  
  
  cvo::AdaptiveCvoGPU cvo_align(cvo_param_file );
  cvo::CvoParams init_param = cvo_align.get_params();
  cvo::CvoParams first_frame_param = init_param;
  first_frame_param.ell_init = 0.95;
  first_frame_param.ell_max = 1.0;
  cvo_align.write_params(&first_frame_param);

  std::cout<<"write ell! ell init is "<<cvo_align.get_params().ell_init<<std::endl;

  //cvo::cvo cvo_align_cpu("/home/rayzhang/outdoor_cvo/cvo_params/cvo_params.txt");
  Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();  // from source frame to the target frame
  init_guess(2,3)=0;
  Eigen::Affine3f init_guess_cpu = Eigen::Affine3f::Identity();
  init_guess_cpu.matrix()(2,3)=0;
  Eigen::Matrix4f accum_mat = Eigen::Matrix4f::Identity();
  // start the iteration

  cv::Mat source_left, source_right;
  //std::vector<float> semantics_source;
  //kitti.read_next_stereo(source_left, source_right, 19, semantics_source);
  kitti.read_next_stereo(source_left, source_right);
  std::shared_ptr<cvo::Frame> source(new cvo::Frame(start_frame, source_left, source_right,
                                                    //19, semantics_source, 
                                                    calib));
  //0.2));
  
  for (int i = start_frame; i<min(total_iters, start_frame+max_num)-1 ; i++) {
    
    // calculate initial guess
    std::cout<<"\n\n\n\n============================================="<<std::endl;
    std::cout<<"Aligning "<<i<<" and "<<i+1<<" with GPU "<<std::endl;

    kitti.next_frame_index();
    cv::Mat left, right;
    //sdt::vector<float> semantics_target;
    if (kitti.read_next_stereo(left, right) != 0) {
      std::cout<<"finish all files\n";
      break;
    }


    std::shared_ptr<cvo::Frame> target(new cvo::Frame(i+1, left, right, calib));

    // std::cout<<"reading "<<files[cur_kf]<<std::endl;
    auto source_fr = source->points();
    auto target_fr = target->points();

    Eigen::Matrix4f result, init_guess_inv;
    init_guess_inv = init_guess.inverse();
    printf("Start align... num_fixed is %d, num_moving is %d\n", source_fr.num_points(), target_fr.num_points());
    std::cout<<std::flush;
    cvo_align.align(source_fr, target_fr, init_guess_inv, result);
    
    // get tf and inner product from cvo getter
    double in_product = cvo_align.inner_product(source_fr, target_fr, result);
    //double in_product_normalized = cvo_align.inner_product_normalized();
    //int non_zeros_in_A = cvo_align.number_of_non_zeros_in_A();
    std::cout<<"The gpu inner product between "<<i-1 <<" and "<< i <<" is "<<in_product<<"\n";
    //std::cout<<"The normalized inner product between "<<i-1 <<" and "<< i <<" is "<<in_product_normalized<<"\n";
    std::cout<<"Transform is "<<result <<"\n\n";

    // append accum_tf_list for future initialization
    init_guess = result;
    accum_mat = accum_mat * result;
    std::cout<<"accum tf: \n"<<accum_mat<<std::endl;
    
    
    // log accumulated pose

    accum_output << accum_mat(0,0)<<" "<<accum_mat(0,1)<<" "<<accum_mat(0,2)<<" "<<accum_mat(0,3)<<" "
                <<accum_mat(1,0)<<" " <<accum_mat(1,1)<<" "<<accum_mat(1,2)<<" "<<accum_mat(1,3)<<" "
                <<accum_mat(2,0)<<" " <<accum_mat(2,1)<<" "<<accum_mat(2,2)<<" "<<accum_mat(2,3);
    accum_output<<"\n";
    accum_output<<std::flush;
    
    std::cout<<"\n\n===========next frame=============\n\n";
   
    source = target;
    if (i == start_frame) {
      cvo_align.write_params(&init_param);
      
    }


  }


  accum_output.close();

  return 0;
}
