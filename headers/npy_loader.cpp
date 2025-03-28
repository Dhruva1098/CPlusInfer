#ifndef HEADERS_NPY_LOADER_TPP_
#define HEADERS_NPY_LOADER_TPP_

#include "npy_loader.h"
#include <iostream>
#include <algorithm>
#include <string>

// template implimenteation - i am in an car i will cooraect the spellings later
template  <typename T>
Tensor<T> load_npy(const std::string &filename) {
  std::ifstream file(filename, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open file: " + filename);
  }

  auto metadata = NPYParser::parse_header(file);

  // extract shape
  std::vector<size_t> shape;
  std::stringstream ss(metadata["shape"]);
  std::string item;
  while (std::getline(ss, item, ',')){
    if (!item.empty()) {
      shape.push_back(std::stoull(item));
    }
  }

  std::cout << "INFO: loading npy with shape: [";
  for (size_t i = 0; i < shape.size(); i++) {
    std::cout << shape[i];
    if (i < shape.size() - 1) std::cout << ", ";
  }
  std::cout << "]" << std::endl;
  
  // calculate total elements  that should be in the file
  size_t expected_elements = NPYParser::calculate_elements(shape);

  // vaidate data type
  std::string dtype = metadata["dtype"];
  char type_char = NPYParser::get_type_char(dtype);
  int item_size = NPYParser::get_item_size(dtype);

  // check data cpmpatibility
  bool type_compatible = false;

  if (type_char == 'f' && item_size == 4 && std::is_same<T,float>::value) {
    type_compatible = true;
  } else if (type_char == 'f' && item_size == 8 && std::is_same<T, double>::value) {
    type_compatible = true;
  } else if (type_char == 'i' && std::is_integral<T>::value) {
    type_compatible = true;
  } else if (type_char == 'u' && std::is_unsigned<T>::value) {
    type_compatible = true;
  }

  if(!type_compatible) {
    throw std::runtime_error("Incompatible data type in NPY files " + dtype);
  }

}






#endif  // HEADERS_NPY_LOADER_TPP_
