#include "Headers/Tensor.h"

// Constructors
template <typename T>
Tensor<T>::Tensor(const std::vector<size_t>& shape)
    :shape_( shape), data_(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>())) {}

template <typename T>
Tensor<T>::Tensor(const std::vector<size_t> &shape, T initialValue)
    : shape_(shape), data_(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>()), initialValue) {}

template <typename T>
Tensor<T>::Tensor(const Tensor& other) : shape_(other.shape_), data_(other.data_) {  // Deep Copy
  std::cout << "Copy Constructor called" << std::endl;
}

template <typename T>
Tensor<T>::Tensor(Tensor&& other) noexcept  // Shallow copy
  : shape_(other.shape_), data_(std::move(other.data_)) {
  std::cout << "Move Constructor called" << std::endl;
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor& other) {  // Copy assignment -deep
  std::cout << "Copy Assignment operator called" << std::endl;
  if (this != &other) {  // Prevent self assignment
    shape_ = other.shape_;
    data_ = other.data_;
  }
  return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(Tensor&& other) noexcept {  // Move assignment - shallow copy ansd steal
  std::cout << "Move Assignment operator called" << std::endl;
  if (this != &other) {
    shape_ = std::move(other.shape_);
    data_ = std::move(other.data_);
  }
  return *this;
}

// Destructor
template <typename T>
Tensor<T>::~Tensor() {
  std::cout << "Destructor called" << std::endl;
}

// Accessors
template <typename T>
size_t Tensor<T>::size() const {
  return data_.size();
}

template <typename T>
size_t Tensor<T>::dim(size_t index) const {
  if (index >= shape_.size()) {
    throw std::out_of_range("dimension index out of range");
  }
  return shape_[index];
}

template <typename T>
std::vector<size_t> Tensor<T>::shape() const {
  return shape_;
}

// Access Operator overloads
template <typename T>
T& Tensor<T>::operator[](size_t index) {
  if (index >= data_.size()) {
    throw std::out_of_range("index out of range");
  }
  return data_[index];
}

template <typename T>
const T& Tensor<T>::operator[](size_t index) const {
  if (index >= data_.size()) {
    throw std::out_of_range("index out of range");
  }
  return data_[index];
}

template <typename T>
T& Tensor<T>::operator()(const std::vector<size_t>& indices) {
  return data_[computeIndex(indices)];
}
template <typename T>
const T& Tensor<T>::operator()(const std::vector<size_t>& indices) const {
  return data_[computeIndex(indices)];
}

//Operator overloading
// + : Element wise
template <typename T>
Tensor<T> Tensor<T>::operator+(const Tensor<T>& other) const {
  if (!checkShapeCompatibility(other)) {
    throw std::invalid_argument("Tensors must have the same shape for addition.");
  }
  Tensor<T> result(shape_);  // Create a new tensor with the same shape
  for (size_t i = 0; i < data_.size(); i++) {
    result.data_[i] = this->data_[i] + other.data_[i];  // Element-wise addition
  }
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::operator-(const Tensor<T>& other) const {
  if (!checkShapeCompatibility(other)) {
    throw std::invalid_argument("Tensors must have the same shape for subtraction.");
  }
  Tensor<T> result(shape_);  //shape of the object that called the function
  for(size_t i = 0; i < data_.size(); i++) {
    result.data_[i] = this->data_[i] - other.data_[i];
  }
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::operator*(const Tensor<T>& other) const {
  if(!checkShapeCompatibility(other)) {
    throw std::invalid_argument("Tensors must have the same shape for scalar multiplication.");
  }
  Tensor<T> result(shape_);
  for(size_t i = 0; i < data_.size(); i++) {
    result.data_[i] = this->data_[i] * other.data_[i];
  }
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::operator/(const Tensor<T>& other) const {
  if (!checkShapeCompatibility(other)) {
    throw std::invalid_argument("Tensors must have the same shape for addition.");
  }
  Tensor<T> result(shape_);  // Create a new tensor with the same shape
  for (size_t i = 0; i < data_.size(); i++) {
    if(other.data_[i] == static_cast<T>(0)) {
      throw std::invalid_argument("Division by zero.");
    }
    result.data_[i] = this->data_[i] / other.data_[i];  // Element-wise addition
  }
  return result;
}

//function to do matrix multiplication
// WE NEED TO FETCH DIMENSION AND CHECK IF SAME COL! AND ROW 2

// great comment lol, this will stay

template <typename T>
Tensor<T> Tensor<T>::reLU() const {
  Tensor<T> output(shape_);
  for (size_t i = 0; i < data_.size(); i++) {
    output.data_[i] = std::max(data_[i], static_cast<T>(0));
  }
  return output;
 }

// reshape
template <typename T>
Tensor<T> Tensor<T>::reshape(const std::vector<size_t>& newShape) const {
  size_t newSize = std::accumulate(newShape.begin(), newShape.end(), 1, std::multiplies<size_t>());
  if (newSize != data_.size()) {
    throw std::invalid_argument("Total size of tensor must remain same after reshape.");
  }
  Tensor<T> result(newSize);
  result.data_ = data_;
  return result;
}

// transpose
template<typename T>
Tensor<T> Tensor<T>::transpose() const {
  if (shape_.size() != 2) {
    throw std::invalid_argument("Transpose is valid for 2d tensors only");
  }
  size_t rows = shape_[0];
  size_t cols = shape_[1];
  Tensor<T> result({cols, rows});  // swap dimensions
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      result({j,i}) = (*this)({i,j});
    }
  }
  return result;
}

// Helper functions
// Use this to check if two tensors are compatible for operations
template <typename T>
bool Tensor<T>::checkShapeCompatibility(const Tensor<T>& other) const {
  return shape_ == other.shape_;
}

// this is pretty simple.
// I wrote this is pretty simple and made this column major
// To jump to 2nd row, we get address by --> initial address * no of elements in 1 row.
// Similar concept here
template <typename T>
size_t Tensor<T>::computeIndex(const std::vector<size_t>& indices) const {
  if (indices.size() != shape_.size()) {
    throw std::invalid_argument("number of indices must equal to dimension");
  }
  size_t index = 0;
  size_t stride = 1;
  for (int i = indices.size() - 1; i >= 0; i--) {
    if (indices[i] >= shape_[i]) {
      throw std::out_of_range("index out of range");
    }
    index += indices[i] * stride;
    stride *= shape_[i]; // this comes after wards for row major
  } return index;
}