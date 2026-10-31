#ifndef LAYER_H
#define LAYER_H
#include <vector>
#include "tensor.h"
class Layer {
public:
    std::vector<int> output_dims; // to pass into the next layer so it can form its parameter tensor. Used in build function of layers.
    //for back prop
    tensor last_output;
    tensor last_input;

    virtual void build(const std::vector<int>& prev_dim) = 0; ///accepts dims vector of prev tensor

    virtual tensor backward(tensor& grad_op) = 0;

    virtual void layerCompute(tensor &input_vals) = 0;

    virtual void update(double learning_rate) = 0;

    virtual ~Layer() = default;
};
#endif