#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <regex>
#include <iostream>
#include <cstring>
#include <memory>
#include <map>
#include "headers/Tensor.h"

// Forward declarations
template <typename T>
Tensor<T> load_npy(const std::string& filename);

// Class to handle NPY file parsing
class NPYParser {
public:
    // Parse NPY file header and return metadata
    static std::map<std::string, std::string> parse_header(std::istream& file) {
        std::map<std::string, std::string> metadata;
        
        // Read magic string
        char magic[6];
        file.read(magic, 6);
        if (std::string(magic, 6) != "\x93NUMPY") {
            throw std::runtime_error("Invalid NPY file format");
        }
        
        // Read version
        uint8_t major_version, minor_version;
        file.read(reinterpret_cast<char*>(&major_version), 1);
        file.read(reinterpret_cast<char*>(&minor_version), 1);
        
        // Store version
        metadata["major_version"] = std::to_string(major_version);
        metadata["minor_version"] = std::to_string(minor_version);
        
        // Read header length
        uint16_t header_len = 0;
        uint32_t header_len_big = 0;
        
        if (major_version == 1) {
            file.read(reinterpret_cast<char*>(&header_len), 2);
        } else if (major_version == 2) {
            file.read(reinterpret_cast<char*>(&header_len_big), 4);
            header_len = static_cast<uint16_t>(header_len_big);
        } else {
            throw std::runtime_error("Unsupported NPY version: " + std::to_string(major_version));
        }
        
        // Read header
        std::vector<char> header_buf(header_len);
        file.read(header_buf.data(), header_len);
        std::string header(header_buf.begin(), header_buf.end());
        
        // Store raw header
        metadata["header"] = header;
        
        // Parse shape
        std::vector<size_t> shape = parse_shape(header);
        std::stringstream shape_ss;
        for (size_t i = 0; i < shape.size(); i++) {
            shape_ss << shape[i];
            if (i < shape.size() - 1) shape_ss << ",";
        }
        metadata["shape"] = shape_ss.str();
        
        // Parse dtype
        metadata["dtype"] = parse_dtype(header);
        
        // Parse fortran_order
        metadata["fortran_order"] = parse_fortran_order(header) ? "true" : "false";
        
        return metadata;
    }
    
    // Helper function to parse shape from the header string
    static std::vector<size_t> parse_shape(const std::string& header) {
        std::vector<size_t> shape;
        
        // Find the shape section in the header
        std::regex shape_regex("'shape':\\s*\\(([^\\)]*)\\)");
        std::smatch shape_match;
        
        if (std::regex_search(header, shape_match, shape_regex) && shape_match.size() > 1) {
            std::string shape_str = shape_match[1];
            
            // Split by comma
            std::stringstream ss(shape_str);
            std::string item;
            
            while (std::getline(ss, item, ',')) {
                // Remove whitespace
                item.erase(0, item.find_first_not_of(" "));
                item.erase(item.find_last_not_of(" ") + 1);
                
                if (!item.empty()) {
                    shape.push_back(std::stoul(item));
                }
            }
            
            // Special case: empty tuple means scalar
            if (shape.empty() && shape_str.find_first_not_of(" ,") == std::string::npos) {
                shape.push_back(1);
            }
        }
        
        return shape;
    }
    
    // Helper function to parse dtype from the header string
    static std::string parse_dtype(const std::string& header) {
        std::regex dtype_regex("'descr':\\s*'([^']*)'");
        std::smatch dtype_match;
        
        if (std::regex_search(header, dtype_match, dtype_regex) && dtype_match.size() > 1) {
            return dtype_match[1];
        }
        
        throw std::runtime_error("Could not parse dtype from NPY header");
    }
    
    // Helper function to parse fortran_order
    static bool parse_fortran_order(const std::string& header) {
        std::regex fortran_regex("'fortran_order':\\s*(\\w+)");
        std::smatch fortran_match;
        
        if (std::regex_search(header, fortran_match, fortran_regex) && fortran_match.size() > 1) {
            return fortran_match[1] == "True";
        }
        
        return false;
    }
    
    // Calculate total elements from shape
    static size_t calculate_elements(const std::vector<size_t>& shape) {
        size_t total = 1;
        for (size_t dim : shape) {
            total *= dim;
        }
        return total;
    }
    
    // Check if the data type needs byte-swapping
    static bool needs_byteswap(const std::string& dtype) {
        bool is_little_endian = dtype[0] == '<';
        bool is_big_endian = dtype[0] == '>';
        
#if defined(__LITTLE_ENDIAN__) || (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __LITTLE_ENDIAN)
        bool system_is_little_endian = true;
#else
        bool system_is_little_endian = false;
#endif
        
        return (is_little_endian && !system_is_little_endian) || 
               (is_big_endian && system_is_little_endian);
    }
    
    // Perform byte-swapping on a buffer
    template <typename T>
    static void byteswap(T* data, size_t elements) {
        for (size_t i = 0; i < elements; i++) {
            char* bytes = reinterpret_cast<char*>(&data[i]);
            for (size_t j = 0; j < sizeof(T) / 2; j++) {
                std::swap(bytes[j], bytes[sizeof(T) - 1 - j]);
            }
        }
    }
    
    // Get item size from dtype string
    static int get_item_size(const std::string& dtype) {
        return std::stoi(dtype.substr(2));
    }
    
    // Get type character from dtype string
    static char get_type_char(const std::string& dtype) {
        return dtype[1];
    }
};

// Function to load NPY file into Tensor
template <typename T>
Tensor<T> load_npy(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    
    // Parse header
    auto metadata = NPYParser::parse_header(file);
    
    // Extract shape
    std::vector<size_t> shape;
    std::stringstream ss(metadata["shape"]);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            shape.push_back(std::stoul(item));
        }
    }
    
    // Validate data type
    std::string dtype = metadata["dtype"];
    char type_char = NPYParser::get_type_char(dtype);
    int item_size = NPYParser::get_item_size(dtype);
    
    // Check data type compatibility
    bool type_compatible = false;
    
    if (type_char == 'f' && item_size == 4 && std::is_same<T, float>::value) {
        type_compatible = true;
    } else if (type_char == 'f' && item_size == 8 && std::is_same<T, double>::value) {
        type_compatible = true;
    } else if (type_char == 'i' && std::is_integral<T>::value) {
        // We could add more precise size checking here
        type_compatible = true;
    } else if (type_char == 'u' && std::is_unsigned<T>::value) {
        type_compatible = true;
    }
    
    if (!type_compatible) {
        throw std::runtime_error("Incompatible data type in NPY file: " + dtype);
    }
    
    // Calculate total elements
    size_t total_elements = NPYParser::calculate_elements(shape);
    
    // Read the data
    std::vector<T> data(total_elements);
    file.read(reinterpret_cast<char*>(data.data()), total_elements * sizeof(T));
    
    // Check if we need to convert endianness
    if (NPYParser::needs_byteswap(dtype)) {
        NPYParser::byteswap(data.data(), total_elements);
    }
    
    // Create tensor with correct shape
    Tensor<T> tensor(shape);
    
    // Fill tensor with data
    for (size_t i = 0; i < total_elements; i++) {
        tensor[i] = data[i];
    }
    
    // If the array was stored in Fortran order (column-major), we need to transpose it
    if (metadata["fortran_order"] == "true" && shape.size() > 1) {
        tensor = tensor.transpose();
    }
    
    return tensor;
}

// Function to save Tensor to NPY file
template <typename T>
void save_npy(const Tensor<T>& tensor, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }
    
    // Write magic string, version, etc.
    file.write("\x93NUMPY", 6);
    
    // Version 1.0
    file.write("\x01\x00", 2);
    
    // Construct header
    std::stringstream header_ss;
    header_ss << "{'descr': '";
    
    // Add endianness and type
#if defined(__LITTLE_ENDIAN__) || (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __LITTLE_ENDIAN)
    header_ss << "<";
#else
    header_ss << ">";
#endif
    
    // Type code and size
    if (std::is_same<T, float>::value) {
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
    
    // Add shape information
    header_ss << "', 'fortran_order': False, 'shape': (";
    
    const std::vector<size_t>& shape = tensor.shape();
    for (size_t i = 0; i < shape.size(); i++) {
        header_ss << shape[i];
        if (i < shape.size() - 1) {
            header_ss << ", ";
        }
    }
    
    header_ss << "), }";
    
    // Pad header with spaces to ensure it's properly aligned
    std::string header = header_ss.str();
    while ((header.size() + 10) % 16 != 0) {
        header += ' ';
    }
    header += '\n';
    
    // Write header length (little endian)
    uint16_t header_len = static_cast<uint16_t>(header.size());
    file.write(reinterpret_cast<char*>(&header_len), 2);
    
    // Write header
    file.write(header.c_str(), header.size());
    
    // Write data
    const std::vector<size_t>& tensor_shape = tensor.shape();
    size_t size = tensor.size();
    for (size_t i = 0; i < size; i++) {
        T value = tensor[i];
        file.write(reinterpret_cast<const char*>(&value), sizeof(T));
    }
}

// Example function to load model weights and biases
void load_model_weights(const std::string& weights_file, const std::string& biases_file) {
    try {
        std::cout << "Loading weights from: " << weights_file << std::endl;
        std::cout << "Loading biases from: " << biases_file << std::endl;
        
        // Load weights and biases
        Tensor<float> weights = load_npy<float>(weights_file);
        Tensor<float> biases = load_npy<float>(biases_file);
        
        // Print shape information
        std::cout << "\nWeights shape: [";
        for (size_t i = 0; i < weights.shape().size(); i++) {
            std::cout << weights.shape()[i];
            if (i < weights.shape().size() - 1) std::cout << ", ";
        }
        std::cout << "] (total elements: " << weights.size() << ")" << std::endl;
        
        std::cout << "Biases shape: [";
        for (size_t i = 0; i < biases.shape().size(); i++) {
            std::cout << biases.shape()[i];
            if (i < biases.shape().size() - 1) std::cout << ", ";
        }
        std::cout << "] (total elements: " << biases.size() << ")" << std::endl;
        
        // Print first few values
        const size_t num_preview = 5;
        std::cout << "\nFirst " << num_preview << " weight values:" << std::endl;
        for (size_t i = 0; i < num_preview && i < weights.size(); i++) {
            std::cout << weights[i] << " ";
        }
        std::cout << std::endl;
        
        std::cout << "First " << num_preview << " bias values:" << std::endl;
        for (size_t i = 0; i < num_preview && i < biases.size(); i++) {
            std::cout << biases[i] << " ";
        }
        std::cout << std::endl;
        
        // Example of saving back to NPY
        std::cout << "\nDemonstrating round-trip conversion (save and reload):" << std::endl;
        save_npy(weights, "weights_copy.npy");
        save_npy(biases, "biases_copy.npy");
        
        std::cout << "Files saved to weights_copy.npy and biases_copy.npy" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <weights_file.npy> <biases_file.npy>" << std::endl;
        return 1;
    }
    
    load_model_weights(argv[1], argv[2]);
    return 0;
}