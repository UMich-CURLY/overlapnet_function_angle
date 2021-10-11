#pragma once
#include <iostream>
#include <cmath>
#include <memory>
#include <thrust/device_vector.h>
namespace cvo {

  inline
  __device__
  void skew_gpu(Eigen::Vector3f * v, Eigen::Matrix3f * M) {
    // Convert vector to skew-symmetric matrix
    (*M) << 0, -(*v)[2], (*v)[1],
      (*v)[2], 0, -(*v)[0], 
      -(*v)[1], (*v)[0], 0;
  }


  template<typename T>
  T * thrust_raw_arr(std::shared_ptr<thrust::device_vector<T>> v_thrust) {
    auto  dv = v_thrust.get();
    return thrust::raw_pointer_cast( &(*dv)[0]);
  }

  template<typename T>
  __device__ T dot(T * a, T* b, int dim) {
    T result = 0;
    for (int i = 0; i < dim; i++) {
      result += a[i] * b[i];
    }
    return result;
  }


  template<typename T>
  __device__ T squared_dist(T * a, T* b, int dim) {
    T result = 0;
    for (int i = 0; i < dim; i++) {
      T tmp = (a[i] - b[i]);
      result += tmp * tmp;
    }
    return result;
  }

        //float cov_inv[9];
      //Eigen::Matrix3f cov_curr_inv = Eigen::Map<Eigen::Matrix3f>(cov_inv);
      //cov_curr_inv = cov_curr.inverse().eval();

  __host__ __device__ __forceinline__ Eigen::Matrix3f Inverse(
                                                              const Eigen::Matrix3f& m) {
    float det = m(0, 0) * (m(1, 1) * m(2, 2) - m(2, 1) * m(1, 2)) -
      m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0)) +
      m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0));

    float invdet = 1 / det;

    Eigen::Matrix3f minv;  // inverse of matrix m
    minv(0, 0) = (m(1, 1) * m(2, 2) - m(2, 1) * m(1, 2)) * invdet;
    minv(0, 1) = (m(0, 2) * m(2, 1) - m(0, 1) * m(2, 2)) * invdet;
    minv(0, 2) = (m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1)) * invdet;
    minv(1, 0) = (m(1, 2) * m(2, 0) - m(1, 0) * m(2, 2)) * invdet;
    minv(1, 1) = (m(0, 0) * m(2, 2) - m(0, 2) * m(2, 0)) * invdet;
    minv(1, 2) = (m(1, 0) * m(0, 2) - m(0, 0) * m(1, 2)) * invdet;
    minv(2, 0) = (m(1, 0) * m(2, 1) - m(2, 0) * m(1, 1)) * invdet;
    minv(2, 1) = (m(2, 0) * m(0, 1) - m(0, 0) * m(2, 1)) * invdet;
    minv(2, 2) = (m(0, 0) * m(1, 1) - m(1, 0) * m(0, 1)) * invdet;

    return minv;
  }




  template<typename T>
  __device__ float squared_dist(T & a, T & b) {
    auto diff_x = (a.x-b.x);
    auto diff_y = (a.y-b.y);
    auto diff_z = (a.z-b.z);
    return  diff_x * diff_x + diff_y * diff_y + diff_z * diff_z ;
  }

  template <typename T>
  __device__ void subtract(T*a, T*b, T*result, int num_elem) {
    for (int i = 0; i < num_elem; i++) {
      result[i] = a[i] - b[i];
      
    }
  }

  template<typename T>
  __device__ void cross3(T *a, T *b, T *result) {
    result[0] = a[2] * b[3] - a[3] * b[2];
    result[1] = a[3] * b[1] - a[1] * b[3];
    result[2] = a[1] * b[2] - a[2] * b[1];
  }

  
  template<typename T>
  __device__ T square_norm(T *a, int dim) {
    T result = 0;
    for (int j = 0; j < dim ;j++) {
      result += a[j] * a[j];
      
    }
    return result;
  }

    
  template<typename T>
  __device__ void scalar_multiply (T *a, T scalar, int dim) {

    for (int j = 0; j < dim ;j++) {
      a[j] = a[j] *scalar ;
      
    }


  }


  template <typename T>
  __device__ void vec_mul_mat(T * v, T* M, int rows, int cols, T* result ) {
    for (int c = 0; c < cols; c++ ) {
      result[c] = 0;
      for (int r = 0; r < rows; r++)
        result[c] += v[r] * M[r * cols + c];
    }
    
  }
  

}
