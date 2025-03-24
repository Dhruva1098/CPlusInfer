#ifndef HEADERS_LAYER_H_
#define HEADERS_LAYER_H_

#include "Tensor.h"
#include <string>
#include <memory> 
#include <vector>


class Graph;

// Base Layer class
class Layer {
public:
  Layer(const std::string& name) : name_(name) {}
  virtual ~Layer() = default;

  // Forward Pass
  virtual Tensor<float> forward(const Tensor<float>& input) = 0;
  
  const std::string& name() const { return name_; }

protected:
  std::string name_;
};

// Linear Fully COnnected layer
class LinearLayer : public Layer {
public:
  LinearLayer(const std::string& name, size_t in_features, size_t out_features)
    : Layer(name), in_features_(in_features), out_features_(out_features) {}

  // Load weights and biases
  void loadWeights(const Tensor<float>& weights, const Tensor<float>& biases);

  // Forward implementation for linear (weights*input + bias)
  Tensor<float> forward(const Tensor<float>& input) override;

private:
  size_t in_features_;
  size_t out_features_;
  Tensor<float> weights_;
  Tensor<float> biases_;
};

// ReLU
class ReLULayer : public Layer {
public:
  ReLULayer(const std::string& name) : Layer(name) {}
  
  // forward
  Tensor<float> forward(const Tensor<float>& input) override;
};







#endif  // Headers_GRAPH_H_ 
