#include "Headers/layer.h"

#include <iostream>
#include <stdexcept>

// Linear Layer
void LinearLayer::loadWeights(const Tensor<float>& weights, const Tensor<float>& biases) {
  if(weights.shape().size() != 2 || 
     weights.shape()[0] != out_features_ ||
     weights.shape()[1] != in_features_) {
    throw std::runtime_error("Weight dimensions don't match layer configuration");
  }

  if(biases.shape().size() != 1 ||
     biases.shape()[0] != out_features_) {
    throw std::runtime_error("Bias dimensions dont match layer configurations");
  }

  weights_ = weights;
  biases_ = biases;
  
  std::cout << "Loaded WEights for " << name_ << " with shape [" <<
  weights_.shape()[0] << ", " << weights_.shape()[1] << "]" << std::endl;
}

Tensor<float> LinearLayer::forward(const Tensor<float>& input) {
  if (input.shape().size() < 1) {
    throw std::runtime_error("Input tensor has invalid shape");
  } 
  // Get bathc size and reshape input if needed
  std::vector<size_t> input_shape = input.shape();
  size_t batch_size = 1;
  for(size_t i = 0; i < input_shape.size() - 1; i++){
    batch_size *= input_shape[i];
  }

  // check dimensions
  if (input_shape.back() != in_features_){
    throw std::runtime_error("Input feature dimension doesn't match layer config");
  }

  // Reshape input to : [batch_size, in_features]
  Tensor<float> reshaped_input;
  if(input_shape.size() > 2) {
    reshaped_input = input.reshape({batch_size, in_features_});
  } else {
    reshaped_input = input;
  }

  // output tensor [batch_size, out_features_]
  std::vector<size_t> output_shape = {batch_size, out_features_};
  Tensor<float> output(output_shape, 0.0f);

  // matrix mul and bias addition
  for (size_t b = 0; b < batch_size; b++){
    for (size_t o = 0; o < out_features_; o++){
      float sum = 0.0f;

      // dot prod
      for (size_t i = 0; i < in_features_; i++) {
        sum += reshaped_input({b, i}) * weights_({o, i});
      }

      // add bias
      sum += biases_[0];

      // result
      output({b, o}) = sum;
    }
  }

  // reshape to orignal shape
  if(input_shape.size() > 2) {
    std::vector<size_t> final_shape(input_shape.begin(), input_shape.end() - 1);
    final_shape.push_back(out_features_);
    return output.reshape(final_shape);
  }
  return output;
}

//ReLU
Tensor<float> ReLULayer::forward(const Tensor<float>& input) {
  return input.reLU();
}

