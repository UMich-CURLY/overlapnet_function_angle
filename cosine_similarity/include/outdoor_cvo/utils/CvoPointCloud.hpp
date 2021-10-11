#pragma once
#include <string>
#include <memory>
#include <vector>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include "utils/data_type.hpp"
#include "utils/RawImage.hpp"
#include "utils/Calibration.hpp"
#include "utils/LidarPointType.hpp"
#include "utils/PointSegmentedDistribution.hpp"
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>


namespace semantic_bki {
  class SemanticBKIOctoMap;
}


namespace cvo {

  class CvoPointCloud{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    //const int pixel_pattern[8][2] = {{0,0}, {-2, 0},{-1,-1}, {-1,1}, {0,2},{0,-2},{1,1},{2,0} };
    const int pixel_pattern[8][2] = {{0,0}, {-1, 0},{-1,-1}, {-1,1}, {0,1},{0,-1},{1,1},{1,0} };
    
    CvoPointCloud(const RawImage & left_raw_image,
                  const cv::Mat & right_image,
                  const Calibration &calib);

    CvoPointCloud(const RawImage & rgb_raw_image,
                  const cv::Mat & depth_image,
                  const Calibration &calib,
                  const bool& is_using_rgbd);
    
    CvoPointCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr pc,
                  int target_num_points,
                  int beam_num);

    CvoPointCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr pc_intensity,
                  int beam_num=64);

    CvoPointCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr pc, 
                  const std::vector<int> & semantics,
                  int num_classes=19,
                  int target_num_points = 5000,
                  int beam_num =64);

    CvoPointCloud(pcl::PointCloud<pcl::PointXYZIR>::Ptr pc,
                  int target_num_points = 5000
                  );

    CvoPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc);

    CvoPointCloud(pcl::PointCloud<pcl::PointXYZIR>::Ptr pc, 
                  const std::vector<int> & semantics,
                  int num_classes=19,
                  int target_num_points = 5000
                  );
    
    
    CvoPointCloud(const semantic_bki::SemanticBKIOctoMap * map,
                  int num_semantic_class);

    // CvoPointCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr pc);

    CvoPointCloud();

    CvoPointCloud(const std::string & filename);
    
    ~CvoPointCloud();

    int read_cvo_pointcloud_from_file(const std::string & filename, int feature_dim=5);
    
    static void transform(const Eigen::Matrix4f& pose,
                          const CvoPointCloud & input,
                          CvoPointCloud & output);

    // getters
    int num_points() const {return num_points_;}
    int num_classes() const {return num_classes_;}
    int feature_dimensions() const {return feature_dimensions_;}
    const ArrayVec3f & positions() const {return positions_;}
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> & labels() const { return labels_;}
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> & features() const {return features_;}
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> & normals() const {return normals_;}
    //const Eigen::Matrix<float, Eigen::Dynamic, 9> & covariance() const {return covariance_;}
    const pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals() const {return cloud_with_normals_;}
    const Eigen::Matrix<float, Eigen::Dynamic, 2> & types() const {return types_;}

    const std::vector<float> & covariance()  const {return covariance_;}
    const std::vector<float> & eigenvalues() const {return eigenvalues_;}

    // for visualization via pcl_viewer
    void write_to_color_pcd(const std::string & name) const;
    void write_to_label_pcd(const std::string & name) const;
    void write_to_pcd(const std::string & name) const;
    void write_to_txt(const std::string & name) const;
    void write_to_intensity_pcd(const std::string & name) const;
   
  private:
    int num_points_;
    int num_classes_;
    int feature_dimensions_;
    
    ArrayVec3f positions_;  // points position. x,y,z
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> features_;   // rgb, gradient in [0,1]
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> normals_;  // surface normals
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> labels_; // number of points by number of classes
    //Eigen::Matrix<float, Eigen::Dynamic, 9> covariance_;

    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals_;
    Eigen::Matrix<float, Eigen::Dynamic, 2> types_; // type of the point using loam point selector, edge=(1,0), surface=(0,1)
    cv::Vec3f avg_pixel_color_pattern(const cv::Mat & raw, int u, int v, int w);
    

    //thrust::device_vector<float> covariance_;
    std::vector<float> covariance_;
    std::vector<float> eigenvalues_;
    //thrust::device_vector<float> eigenvalues_;
    //perl_registration::cuPointCloud<CvoPoint>::SharedPtr pc_gpu;
    //void compute_covarianes(pcl::PointCloud<pcl::PointXYZI> & pc_raw);
    //void compute_covariance(const pcl::PointCloud<pcl::PointXYZI> & pc_input,
    //                        // outputs
    //                        std::vector<float>& covariance_all,
    //                        std::vector<float>& eigenvalues_all) const;


  };
  // for historical reasons
  typedef CvoPointCloud point_cloud;

  void write_all_to_label_pcd(const std::string name,
                          const pcl::PointCloud<pcl::PointXYZI> & pc,
                          int num_class,
                          const std::vector<int> & semantic);
}
