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

// Access Operator overloads
double& Tensor::operator[](size_t index) {
  if (index >= data_.size()) {
    throw std::out_of_range("index out of range");
  }
  return data_[index];
}
const double& Tensor::operator[](size_t index) const {
  if (index >= data_.size()) {
    throw std::out_of_range("index out of range");
  }
  return data_[index];
}

double& Tensor::operator()(const std::vector<size_t>& indices) {
  return data_[computeIndex(indices)];
}
const double& Tensor::operator()(const std::vector<size_t>& indices) const {
  return data_[computeIndex(indices)];
}

//Operator overloading
// + : Element wise
Tensor Tensor::operator+(const Tensor& other) const {
  if (!checkShapeCompatibility(other)) {
    throw std::invalid_argument("Tensors must have the same shape for addition.");
  }
  Tensor result(shape_);  // Create a new tensor with the same shape
  for (size_t i = 0; i < data_.size(); i++) {
    result.data_[i] = this->data_[i] + other.data_[i];  // Element-wise addition
  }
  return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
  if (!checkShapeCompatibility(other)) {
    throw std::invalid_argument("Tensors must have the same shape for subtraction.");
  }
  Tensor result(shape_);  //shape of the object that called the function
  for(size_t i = 0; i < data_.size(); i++) {
    result.data_[i] = this->data_[i] - other.data_[i];
  }
  return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
  if(!checkShapeCompatibility(other)) {
    throw std::invalid_argument("Tensors must have the same shape for scalar multiplication.");
  }
  Tensor result(shape_);
  for(size_t i = 0; i < data_.size(); i++) {
    result.data_[i] = this->data_[i] * other.data_[i];
  }
  return result;
}
Tensor Tensor::operator/(const Tensor& other) const {
  if (!checkShapeCompatibility(other)) {
    throw std::invalid_argument("Tensors must have the same shape for addition.");
  }
  Tensor result(shape_);  // Create a new tensor with the same shape
  for (size_t i = 0; i < data_.size(); i++) {
    if(other.data_[i] == 0) {
      throw std::invalid_argument("Division by zero.");
    }
    result.data_[i] = this->data_[i] / other.data_[i];  // Element-wise addition
  }
  return result;
}
//function to do matrix multipliacaiton canges by jyoti {
//  //WE NEED TO FETCH DIMENSION AND CHECK IF SAME COL! AND ROW 2
//



// Helper functions
// Use this to check if two tensors are compatible for operations
bool Tensor::checkShapeCompatibility(const Tensor& other) const {
  return shape_ == other.shape_;
}

// this is pretty simple. To jump to 2nd row, we get address by --> initital address * no of elements in 1 row.
// Similar concept here
size_t Tensor::computeIndex(const std::vector<size_t>& indices) const {
  if (indices.size() != shape_.size()) {
    throw std::invalid_argument("number of indeces must equal to dimension");
  }
  size_t index = 0;
  size_t stride = 1;
  for (int i = indices.size() - 1; i >= 0; i--) {
    if (indices[i] >= shape_[i]) {
      throw std::out_of_range("index out of range");
    }
    stride *= shape_[i];
    index += indices[i] * stride;
  } return index;
}