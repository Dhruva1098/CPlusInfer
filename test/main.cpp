
//This is just to check structure and data Store.
// Here Struct is storing some data of mode NOT ALL, we can change those things as required
//Ths is not getting activation function name but we can just add the variavble in the struct and get it.

#include <iostream>
#include <fstream>
#include <vector>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Structure to store Layer information
struct Layer {
    std::string name;
    std::string type;
    int units = -1;  // Default: -1 (only relevant for Dense layers)


    void print() const {
        std::cout << "Layer: " << name << " (Type: " << type << ")";
        if (type == "Dense") {
            std::cout << ", Units: " << units;
        }
        std::cout << std::endl;
    }
};

// Structure to store the entire model
struct ModelConfig {
    std::string modelName;
    std::vector<Layer> layers;
    std::string lossFunction;
    std::vector<std::string> metrics;

    // Function to print model details
    void print() const {
        std::cout << "Model Name: " << modelName << std::endl;
        std::cout << "Layers:\n";
        for (const auto& layer : layers) {
            layer.print();
        }
        std::cout << "Loss Function: " << lossFunction << std::endl;
        std::cout << "Metrics: ";
        for (const auto& metric : metrics) {
            std::cout << metric << " ";
        }
        std::cout << std::endl;
    }
};
ModelConfig parseModelConfig(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Could not open JSON file!");
    }

    json modelData;
    file >> modelData;

    ModelConfig model;
    model.modelName = modelData["config"]["name"];

    // Parse layers
    auto layersJson = modelData["config"]["layers"];
    for (const auto& layerJson : layersJson) {
        Layer layer;
        layer.name = layerJson["config"]["name"];
        layer.type = layerJson["class_name"];

        // If it's a Dense layer, extract units
        if (layer.type == "Dense") {
            layer.units = layerJson["config"]["units"];
        }

        model.layers.push_back(layer);
    }

    // Parse compile configuration
    model.lossFunction = modelData["compile_config"]["loss"];
    for (const auto& metric : modelData["compile_config"]["metrics"]) {
        model.metrics.push_back(metric);
    }

    return model;
}
int main() {
    try {
        ModelConfig model = parseModelConfig("model_info.json");
        model.print();  // Print the extracted data
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}
