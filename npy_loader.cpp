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

  // read version 
  uint8_t major_version, minor_version;
  file.read(reinterpret_cast<char*>(&major_version), 1);
  file.read(reinterpret_cast<char*>(&minor_version), 1);

  // store version
  metadata["major_version"] = std::to_string(major_version);
  metadata["minor_version"] = std::to_string(minor_version);

  // read header length
  uint16_t header_len = 0;
  uint32_t header_len_big = 0;
  
  if (major_version == 1) {
    file.read(reinterpret_cast<char*>(&header_len), 2);
  } else if (major_version == 2){
    file.read(reinterpret_cast<char*>(&header_len_big), 4);
    header_len = static_cast<uint16_t>(header_len_big);
  } else {
    throw std::runtime_error("Unsupported NPY version " + std::to_string(major_version));
  }




























}
