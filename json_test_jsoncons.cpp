#include <iostream>
#include <fstream>
#include "jsoncons/json.hpp"

using json = jsoncons::json;

int main() {
    std::ifstream is("extra/model_info.json");
    if (!is) {
        std::cerr << "Error: Cannot open input file!" << std::endl;
        return 1;
    }

    std::ofstream out("out.txt");
    if (!out) {
        std::cerr << "Error: Cannot open output file!" << std::endl;
        return 1;
    }

    json model = json::parse(is);

    for (const auto& i : model.object_range()) {
        std::string s = model["class_name"].as<std::string>();
        out << i.key() << ": " << i.value() << std::endl;
    }

    std::cout << "Output written to out.txt" << std::endl;

    return 0;
}