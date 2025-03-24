// Implementation of Tensor class methods
#include "Tensor.h"

template <typename T>
Tensor<T>::Tensor(const std::vector<size_t>& shape)
    : shape_(shape) {
    size_t total_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    data_.resize(total_size);
}

template <typename T>
Tensor<T>::Tensor(const std::vector<size_t>& shape, T initialValue)
    : shape_(shape) {
    size_t total_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    data_.resize(total_size, initialValue);
}

template <typename T>
Tensor<T>::Tensor(const Tensor& other)
    : shape_(other.shape_), data_(other.data_) {
}

template <typename T>
Tensor<T>::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)), data_(std::move(other.data_)) {
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor& other) {
    if (this != &other) {
        shape_ = other.shape_;
        data_ = other.data_;
    }
    return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        shape_ = std::move(other.shape_);
        data_ = std::move(other.data_);
    }
    return *this;
}

template <typename T>
Tensor<T>::~Tensor() {
    // Vector will clean up automatically
}

template <typename T>
size_t Tensor<T>::size() const {
    return data_.size();
}

template <typename T>
size_t Tensor<T>::dim(size_t index) const {
    if (index >= shape_.size()) {
        throw std::out_of_range("Index out of range in dim()");
    }
    return shape_[index];
}

template <typename T>
std::vector<size_t> Tensor<T>::shape() const {
    return shape_;
}

template <typename T>
T& Tensor<T>::operator[](size_t index) {
    if (index >= data_.size()) {
        throw std::out_of_range("Index out of range in operator[]");
    }
    return data_[index];
}

template <typename T>
const T& Tensor<T>::operator[](size_t index) const {
    if (index >= data_.size()) {
        throw std::out_of_range("Index out of range in operator[]");
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

template <typename T>
size_t Tensor<T>::computeIndex(const std::vector<size_t>& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("Number of indices does not match tensor dimensions");
    }

    size_t index = 0;
    size_t stride = 1;

    for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
        if (indices[i] >= shape_[i]) {
            throw std::out_of_range("Index out of bounds in operator()");
        }
        index += indices[i] * stride;
        stride *= shape_[i];
    }

    return index;
}

template <typename T>
bool Tensor<T>::checkShapeCompatibility(const Tensor& other) const {
    return shape_ == other.shape_;
}

template <typename T>
Tensor<T> Tensor<T>::operator+(const Tensor& other) const {
    if (!checkShapeCompatibility(other)) {
        throw std::invalid_argument("Incompatible tensor shapes for addition");
    }

    Tensor<T> result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }
    return result;
}

template <typename T>
Tensor<T> Tensor<T>::operator-(const Tensor& other) const {
    if (!checkShapeCompatibility(other)) {
        throw std::invalid_argument("Incompatible tensor shapes for subtraction");
    }

    Tensor<T> result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] - other.data_[i];
    }
    return result;
}

template <typename T>
Tensor<T> Tensor<T>::operator*(const Tensor& other) const {
    if (!checkShapeCompatibility(other)) {
        throw std::invalid_argument("Incompatible tensor shapes for element-wise multiplication");
    }

    Tensor<T> result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * other.data_[i];
    }
    return result;
}

template <typename T>
Tensor<T> Tensor<T>::operator/(const Tensor& other) const {
    if (!checkShapeCompatibility(other)) {
        throw std::invalid_argument("Incompatible tensor shapes for division");
    }

    Tensor<T> result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        if (other.data_[i] == 0) {
            throw std::domain_error("Division by zero");
        }
        result.data_[i] = data_[i] / other.data_[i];
    }
    return result;
}

template <typename T>
Tensor<T> Tensor<T>::operator+(int x) const {
    Tensor<T> result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] + x;
    }
    return result;
}

template <typename T>
Tensor<T> Tensor<T>::operator-(int x) const {
    Tensor<T> result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] - x;
    }
    return result;
}

template <typename T>
Tensor<T> Tensor<T>::operator*(int x) const {
    Tensor<T> result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * x;
    }
    return result;
}

template <typename T>
Tensor<T> Tensor<T>::reLU() const {
    Tensor<T> result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = std::max(static_cast<T>(0), data_[i]);
    }
    return result;
}

template <typename T>
Tensor<T> Tensor<T>::reshape(const std::vector<size_t>& newShape) const {
    size_t newSize = std::accumulate(newShape.begin(), newShape.end(), 1, std::multiplies<size_t>());
    if (newSize != data_.size()) {
        throw std::invalid_argument("New shape must have same total number of elements");
    }

    Tensor<T> result(newShape);
    std::copy(data_.begin(), data_.end(), result.data_.begin());
    return result;
}

template <typename T>
Tensor<T> Tensor<T>::transpose() const {
    // For now, only implement 2D transpose
    if (shape_.size() != 2) {
        throw std::invalid_argument("Transpose currently only supports 2D tensors");
    }

    std::vector<size_t> newShape = {shape_[1], shape_[0]};
    Tensor<T> result(newShape);

    for (size_t i = 0; i < shape_[0]; ++i) {
        for (size_t j = 0; j < shape_[1]; ++j) {
            result({j, i}) = (*this)({i, j});
        }
    }

    return result;
}

template <typename T>
void Tensor<T>::print() const {
    if (shape_.empty()) {
        std::cout << "Empty tensor" << std::endl;
        return;
    }

    // For now, just print the first few elements
    std::cout << "Tensor shape: [";
    for (size_t i = 0; i < shape_.size(); ++i) {
        std::cout << shape_[i];
        if (i < shape_.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Data (first 10 elements): ";
    for (size_t i = 0; i < std::min(data_.size(), size_t(10)); ++i) {
        std::cout << data_[i] << " ";
    }
    std::cout << std::endl;
}