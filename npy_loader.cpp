#include "headers/npy_loader.h"
#include <iostream>
#include <sstream>
#include <regex>
#include <stdexcept>

// parse NPY file header and return metaata
std::map<std::string, std::string> NPYParser::parse_header(std::istream& file) {
  std::map<std::string, std::string> metadata;

  // read magic string
  char magic[6];
  file.read(magic, 6);
  if (std::string(magic, 6) != "\x93NUMPY") {
    throw std::runtime_error("Unvalid NPY file format");
  }
}
