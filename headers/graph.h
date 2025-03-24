#ifndef HEADERS_GRAPH_H_
#define HEADERS_GRAPH_H_

#include "layer.h"
#include "Tensor.h"

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>

// Computation graph
class Graph {
public:
  Graph() = default;

  // add lauer
  void addLayer(std::shared_ptr<Layer> layer);

  // Inference on input
  Tensor<float> forward(const Tensor<float>& input);

  // Load model arch. and weights
  static Graph loadFromJSON(const std::string& architecture_file, const std::string& params_file, const std::string& weights_dir);

private:
  std::vector<std::shared_ptr<Layer>> layers_;

};

#endif  // HEADERS_GRAPH_H_
