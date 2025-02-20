// Copyright 2025 Dhruva Sharma

#ifndef HEADERS_TENSOR_H_
#define HEADERS_TENSOR_H_

#include <vector>
#include <stdexcept>
#include <iostream>
#include <numeric>

template <typename T>
class Tensor {
public:

    // Constructors
    explicit Tensor(const std::vector<size_t>& shape);  // Tensor t1({1,1,1}) // all garbage values
    Tensor(const std::vector<size_t>& shape, T initialValue);  // Tensor t1({1,1,1}, 40})
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;

    // Assignment Operators
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;

    ~Tensor();

    // accessors: shape, dim, size
    size_t size() const;
    size_t dim(size_t index) const;
    std::vector<size_t> shape() const;

    // element access [] (single dimensional tensor)
    T& operator[](size_t index);  // Creating different for both const and non const
    const T& operator[](size_t index) const;

    // operator overload () (multi dimensional tensor)
    T& operator()(const std::vector<size_t>& indices);
    const T& operator()(const std::vector<size_t>& indices) const;


    // op overload +, -, /, *
    Tensor operator+(const Tensor& other) const;  // t1 + t2 == t1(operator+(t2))
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;

    //op overloading for scalar value
    Tensor operator+(int x) const;
    Tensor operator-(int x) const;
    Tensor operator*(int x) const;

    // ReLU
    Tensor reLU() const;

    // Reshape
    Tensor<T> reshape(const std::vector<size_t>& newShape) const;

    // Transpose
    Tensor transpose() const;

    // Print
    void print() const;

private:
    std::vector<size_t> shape_;
    std::vector<T> data_;
    size_t computeIndex(const std::vector<size_t>& indices) const;
    bool checkShapeCompatibility(const Tensor& other) const;
};

#endif  // HEADERS_TENSOR_H_


