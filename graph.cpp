#include "headers/graph.h"

#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <unordered_map>

using json = nlohmann::json;

void Graph::addLayer(std::shared_ptr<Layer> layer){
  layers_.push_back(layer);
}

Tensor<float> Graph::forward(const Tensor<float>& input) {
  Tensor<float> output = input;

  // proceed through each layer sequentually
  for(const auto& layer : layers_){
    std::cout << "Processing lauer: " << layer->name() << std::endl;
    output = layer->forward(output);
  }
  return output;
}

// Load model from architecture and weights

Graph Graph::loadFromJSON(const std::string &architecture_file, const std::string &params_file, const std::string &weights_dir) {
  Graph graph;
  try {
    // Parse json
    std::ifstream arch_file(architecture_file);
    if(!arch_file.is_open()) {
      throw std::runtime_error("failed to open architecture file: " + architecture_file);
    }

    json arch_json;
    arch_file >> arch_json;

    // parse
    std::ifstream param_file(params_file);
    if(!param_file.is_open()) {
      throw std::runtime_error("failed to open paramater file:" + params_file);
    }

    json params_json;
    param_file >> params_json;

    // extract layers from architecture
    const auto& layers = arch_json["architecture"]["layers"];

    // map to store layer objects by name
    std::unordered_map<std::string, std::shared_ptr<Layer>> layer_map;

    // First pass - create lauer objects
    for (const auto& layer : layers) {
      const std::string layer_type = layer["type"];
      const std::string layer_name = layer["name"];
      std::shared_ptr<Layer> layer_obj;

      
      // WILL DO THIS LATER I HAVE TO START FROM
      // CREATING LAYERS
      std::cout << "Creating Layer: " << layer_name << " (Type: " << layer_type << ")" << std::endl;
      if (layer_type == "Linear") {
        const size_t in_features = layer["parameters"]["in_features"];
        const size_t out_features = layer["parameters"]["out_features"];
      
        layer_obj = std::make_shared<LinearLayer>(layer_name, in_features, out_features);
      }
      else if (layer_type == "ReLU") {
        layer_obj = std::make_shared<ReLULayer>(layer_name);
      }
      // Add more layer types as needed
  
    if (layer_obj) {
      layer_map[layer_name] = layer_obj;
      graph.addLayer(layer_obj);
    }
  }
  // Second pass: load weights for application layers
  for (const auto& [param_name, paran_info] : params_json.items()){
    // parse layer name from parameter name
    std::string layer_name = param_name.substr(0, param_name.find('.'));
    std::string param_type = param_name.substr(param_name.find('.') + 1);

    // Find corresponding layer
    if (layer_map.find(layer_name) != layer_map.end()) {
      auto layer = layer_map[layer_name];

      // if it is a linear layer that needs weights/biases
      if (auto linear_layer = std::dynamic_pointer_cast<LinearLayer>(layer)){
        // LOad weights and biases
        if (param_type == "weight") {
          std::string weight_file = weights_dir + "/" + param_info["file_path"].get<std::string>();
          std::string bias_file = weights_dir + "/" + params_json[layer_name + ".bias"]["file_path"].get<std::string>();
          std::cout << "Loading weights for " << layer_name << " from " << weight_file << std::sndl;
          std::cout << "Loading biases for " << layer_name << " from " << bias_file << std::endl;

          // use our NPY parser to load weights
          Tensor<float> weights = load_npy<float>(weight_file);
          Tensor<float> biases = load_npy<float>(bias_file);

          // set layer weights
          linear_layer->loadWeights(weights, biases);
        }
      }
    }
  }
} catch (const std::exception& e) {
    std::cerr << "Error loading model: " << e.what() << std::endl;
  }
return graph;
}

