//
// Created by Dhruva Sharma on 6/2/25.
//
#include <iostream>
#include <vector>
#include "headers/Tensor.h"
#include "Tensor.cpp"


int main() {
    // Test Constructors
    Tensor<double> t1({2, 3});
    std::cout << "Tensor t1 (default constructor):" << std::endl;
    t1.print();

    Tensor<int> t2({2, 2}, 42);
    std::cout << "Tensor t2 (initial value constructor):" << std::endl;
    t2.print();

    Tensor<int> t3(t2);  // Copy constructor
    std::cout << "Tensor t3 (copy constructor from t2):" << std::endl;
    t3.print();

    // Test Assignment Operators
    Tensor<int> t5({2, 2}, 1);
    t5 = t2;  // Copy assignment
    std::cout << "Tensor t5 (copy assignment from t2):" << std::endl;
    t5.print();

    Tensor<int> t6({1, 4}, 5);
    t6 = std::move(t5);  // Move assignment
    std::cout << "Tensor t6 (move assignment from t5):" << std::endl;
    t6.print();

    // Test Accessors
    std::cout << "Size of t2: " << t2.size() << std::endl;
    std::cout << "Dimension 0 of t2: " << t2.dim(0) << std::endl;
    std::cout << "Shape of t2: ";
    std::vector<size_t> shape = t2.shape();
    for (size_t dim : shape) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;

    // Test Element Access
    t2({0, 0}) = 100;
    std::cout << "t2 after setting t2[0, 0] = 100:" << std::endl;
    t2.print();
    std::cout << "Value at t2[0, 0]: " << t2({0, 0}) << std::endl;

    // Test Operators
    Tensor<int> t7({2, 2}, 2);
    Tensor<int> t8 = t2 + t7;
    std::cout << "t8 (t2 + t7):" << std::endl;
    t8.print();

    Tensor<int> t9 = t2 - t7;
    std::cout << "t9 (t2 - t7):" << std::endl;
    t9.print();

    Tensor<int> t10 = t2 * t7;
    std::cout << "t10 (t2 * t7):" << std::endl;
    t10.print();

    Tensor<int> t11 = t2 / t7;
    std::cout << "t11 (t2 / t7):" << std::endl;
    t11.print();

    //Test operator overloading for scalar
    Tensor<int> t18=t7+3;
    std::cout << "t18 (t7+3):" << std::endl;
    t18.print();

    Tensor<int> t19=t7-3;
    std::cout << "t18 (t7-3):" << std::endl;
    t19.print();

    Tensor<int> t20=t7*0;
    std::cout << "t18 (t7*0):" << std::endl;
    t20.print();
    return 0;

    // Test ReLU
    Tensor<int> t12({2, 2}, -1);
    t12({0,0}) = -5;
    t12({0,1}) = 0;
    t12({1,0}) = 3;
    t12({1,1}) = -2;
    Tensor<int> t13 = t12.reLU();
    std::cout << "t12 before ReLU:" << std::endl;
    t12.print();
    std::cout << "t13 (ReLU of t12):" << std::endl;
    t13.print();

    // Test Reshape
    Tensor<int> t14({2, 3}, 1);
    t14({0,0}) = 1;
    t14({0,1}) = 2;
    t14({0,2}) = 3;
    t14({1,0}) = 4;
    t14({1,1}) = 5;
    t14({1,2}) = 6;

    std::cout << "t14 before Reshape:" << std::endl;
    t14.print();
    Tensor<int> t15 = t14.reshape({3, 2});
    std::cout << "t15 (Reshape of t14):" << std::endl;
    t15.print();

    // Test Transpose
    Tensor<int> t16({2, 3}, 1);
    t16({0,0}) = 1;
    t16({0,1}) = 2;
    t16({0,2}) = 3;
    t16({1,0}) = 4;
    t16({1,1}) = 5;
    t16({1,2}) = 6;
    std::cout << "t16 before Transpose:" << std::endl;
    t16.print();

    Tensor<int> t17 = t16.transpose();
    std::cout << "t17 (Transpose of t16):" << std::endl;
    t17.print();



}