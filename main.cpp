//
// Created by Dhruva Sharma on 6/2/25.
//
#include <iostream>
#include <vector>
#include "Tensor.cpp"
#include "Headers/Tensor.h"


int main() {
    std::vector<size_t> shape;
    shape.push_back(2);
    shape.push_back(2);
    Tensor A(shape, 5);
    Tensor B(shape,5);
    Tensor C = A * B;
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            std::cout << C({i,j}) << std::endl;
        }
    }
 }