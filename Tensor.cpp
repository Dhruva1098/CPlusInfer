#include "Headers/tensor.h"

// Constructor
Tensor::Tensor(const std::vector<size_t>& shape)
    : shape_(shape), data_(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>())) {}

// Constructor with initial value
Tensor::Tensor(const std::vector<size_t> &shape, double initialValue)
    : shape_(shape), data_(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>()), initialValue) {}

// Destructor
Tensor::~Tensor() {}

// Accessors
size_t Tensor::size() const {
  return data_.size();
}

size_t Tensor::dim(size_t index) const {
  if (index >= shape_.size()) {
    throw std::out_of_range("dimension index out of range");
  }
  return shape_[index];
}

std::vector<size_t> Tensor::shape() const {
  return shape_;
}

