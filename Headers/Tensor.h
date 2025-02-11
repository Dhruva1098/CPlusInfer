// Copyright 2025 Dhruva Sharma

#ifndef HEADERS_TENSOR_H_
#define HEADERS_TENSOR_H_

#include <vector>

class Tensor {
  // No templates type, will use double throughout
  // Constructor, vector<int>;vector<int>(2)
  explicit Tensor(const std::vector<size_t>& shape);  // Tensor t1({1,1,1}) // all garbage values
  Tensor(const std::vector<size_t>& shape, double* initialValue);  // Tensor t1({1,1,1}, 40})

  // destructor
  ~Tensor();

  // accessors: shape, dim, size
  size_t size() const;
  size_t dim(size_t index) const;
  std::vector<size_t> shape() const;

  // element access [] (single dimensional tensor)
  double& operator[](size_t index);  // Creating different for both const and non const
  const double& operator[](size_t index) const;
  // operator overlad () (multi dimensional tensor)
  double& operator()(const std::vector<size_t>& indices);
  const double& operator()(const std::vector<size_t>& indices) const;
  // op overload +, -, /, *
  Tensor operator+(const Tensor& other) const;  // t1 + t2 == t1(operator+(t2))

 private:
    std::vector<size_t> _shape;
    std::vector<double> _data;
    size_t computeIndex(const std::vector<size_t>& indices) const;
    bool checkShapeCompatibility(const Tensor& other) const;
};

#endif  // HEADERS_TENSOR_H_


