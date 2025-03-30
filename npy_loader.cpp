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
  
  // read header 
  std::vector<char> header_buf(header_len);
  file.read(header_buf.data(), header_len);
  std::string header(header_buf.begin(), header_buf.end());

  // store raw header
  metadata["header"] = header;
  
  // parse shape
  std::vector<size_t> shape = parse_shape(header);
  std::stringstream shape_ss;
  for(size_t i = 0; i < shape.size(); i++) {
    shape_ss << shape[i];
    if(i < shape.size()-1) shape_ss << ",";
  }
  metadata["shape"] = shape.ss.str();

  // parse dtype
  metadata["dtype"] = parse_dtype(header);

  // parse fortran_order
  metadata["fortran_order"] = parse_fortran_order(header) ? "true" : "false";

  return metadata;
}

// helper
std::vector<size_t> NPYParser::parse_shape(const std::String& header) {
  std::vector<size_t> shape;
  
  // find the shape section in header, using regex for now
  std::regex shape_regex(" 'shape':\\s*\\(([^\\])*)\\)");
  std::smatch shape_match;
}



























