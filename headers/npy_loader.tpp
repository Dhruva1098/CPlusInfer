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

  // Read data if compatible
  std::vector<T> data(expected_elements);
  file.read(reinterpret_cast<char*>(data.data()),expected_elements * sizeof(T));
  
  // check if we actyally read data we are suposed to 
  if (file.gcount() != expected_elements * sizeof(T)) {
    std::cerr << "WARNING : read " << file.gcount() << " bytes, expected "
      << expected_elements*sizeof(T) << " bytes" << std::endl;
  }

  // check for endianess
  if(NPYParser::needs_byteswap(dtype)) {
    NPYParser::byteswap(data.data(), expected_elements);
  }

  // now create a tensor with right shape
  Tensor<T> tensor(shape);
  for(size_t i = 0; i < expected_elements && i < data.size(); i++){
    tensor[i] = data[i];
  }

  // id array was stored in fortran order for god knows why, transpose it
  if(metadata["fortran_order"] == "true" && shape.size() > 1){
    tensor = tensor.transpose();
  }

  return tensor;
}

// implement save_npy
template <typename T>
void save_npy(const Tensor<T>& tensor, const std::string& filename) {
  std::ofstream file(filename, std::ios::binary);
  if(!file) {
    throw std::runtime_error("Failed ot open file for writing" + filename);
  }

  // write magic string, vetsion etc.
  file.write("\x93NUMPY", 6);
  //ver 1.0
  file.write("\x01\x00", 2);

  // construct header
  std::stringstream header_ss;
  header_ss << "{'descr: '}";

  // endianess and type
#if defined(__LITTLE_ENDIAN__) || (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __LITTLE_ENDIAN__)
  header_ss << "<";
#else
  header_ss << ">";
#endif
  
  // type code and size
  if (std::is_same<T,float>::value) {
    header_ss << "f4";
  } else if (std::is_same<T, double>::value) {
    header_ss << "f8";
  } else if (std::is_same<T, int8_t>::value) {
    header_ss << "i1";
  } else if (std::is_same<T, int16_t>::value) {
    header_ss << "i2";
  } else if (std::is_same<T, int32_t>::value) {
    header_ss << "i4";
  } else if (std::is_same<T, int64_t>::value) {
    header_ss << "i8";
  } else if (std::is_same<T, uint8_t>::value) {
    header_ss << "u1";
  } else if (std::is_same<T, uint16_t>::value) {
    header_ss << "u2";
  } else if (std::is_same<T, uint32_t>::value) {
    header_ss << "u4";
  } else if (std::is_same<T, uint64_t>::value) {
    header_ss << "u8";
  } else {
    throw std::runtime_error("Unsupported data type for NPY export");
  }

  // shape information
  header_ss << " ', 'fortran_order': False, 'shape' : (";

  const std::vector<size_t>& shape = tensor.shape();
  for(size_t i = 0; i < shape.size(); i++){
    header_ss << shape[i];
    if (i < shape.size() -1) {
      header_ss << ", ";
    }
  }

  header_ss << "),}";

  // pad header with spaces to ensore it is properly aligned
  std::string header = header_ss.str();
  while((header.size() + 10) % 16 != 0) {
    header += ' ';
  }
  header += '\n';

  // write header length LE
  uint16_t header_len = static_cast<uint16_t>(header.size());
  file.write(reinterpret_cast<char*>(&header_len), 2);

  // writing header
  file.write(header.c_str(), header.size());

  // write data
  size_t size = tensor.size();
  for (size_t i = 0; i < size; i++) {
    T value = tensor[i];
    file.write(reinterpret_cast<const char*>(&value), sizeof(T));
  }
}

// template specialization for byteswap
template <typename T>
void NPYParser::byteswap(T* data, size_t elements) {
  for(size_t i = 0; i < elements; i++) {
    char* bytes = reinterpret_cast<char*>(&data[i]);
    for (size_t j = 0; j < sizeof(T) / 2; j++) {
      std::swap(bytes[j], bytes[sizeof(T) - 1 - j]);
    }
  }
}

#endif  // HEADERS_NPY_LOADER_TPP_
