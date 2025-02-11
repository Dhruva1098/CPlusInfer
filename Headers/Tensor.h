//
// Created by Dhruva Sharma on 10/2/25.
//

#ifndef TENSOR_CLASS_H
#define TENSOR_CLASS_H
#include <vector>

class Tensor {
  // No templates type, will use double throughout
  // Constructor, vector<int>;vector<int>(2)

  Tensor(std::vector<size_t>& shape);  // Tensor t1({1,1,1}) // all garbage values
  Tensor(std::vector<size_t>& shape, double* initial_value);  // Tensor t1({1,1,1}, 40})

  // destructor
  ~Tensor();

  // accessors: shape, dim
  std::vector<size_t> shape(const Tensor& tensor) const;
  std::vector<size_t> dim(const std::vector<size_t>& tensor_internal) const;
  // arr[3]
  // operator overload [] (single dimensional tensor)
  double operator[](const size_t tensor_index) const;
  // operator overlad () (multi dimensional tensor)
  double& operator[](const std::vector<size_t>& tensor_index); const
  // op overload +, -, /, *
  Tensor operator+(const Tensor& other) const; // t1 + t2 == t1(operator+(t2))
  // private
  // check_dim
  // calculate_index arr[6] == arr + 6*sizeof(int);
};
#endif //TENSOR_CLASS_H


