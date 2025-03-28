#ifndef HEADERS_NPY_LOADER_H_
#define HEADERS_NPY_LOADER_H_

#include "Tensor.h"
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <regex>
#include <stdexcept>
#include <cstring>
#include <type_traits>

class NPYParser {
public:
  // parse NPY files header and return metadata
  static std::map<std::string, std::string> parse_header(std::istream& file);
  static std::vector<size_t> parse_shape(const std::string& header);
  static std::string parse_dtype(const std::string& header);
  static bool parse_fortran_order(const std::string& header);
  static size_t calculate_elements(const std::vector<size_t>& shape);
  static bool needs_byteswap(const std::string& dtype);
  
  template<typename T>
  static void byteswap(T* data, size_t elements);

  // get item from dtype string
  static int get_item_size(const std::string& dtype);

  // get type character from dtype string
  static char get_type_char(const std::string& dtype);
};

// NPY to Tensor (template declaration)
template <typename T>
Tensor<T> load_npy(const std::string& filename);

// function to save tensor to NPY file (template declaration)
template <typename T>
void save_npy(const Tensor<T>& tensor, const std::string& filename);

//rempove below comment later
#include "npy_loader.tpp"

#endif  // HEADERS_NPY_LOADER_H_
