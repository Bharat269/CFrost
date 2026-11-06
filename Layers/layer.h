#pragma once
#include <vector>
#include <Tensor/tensor.h>
#include <Optimizers/OptimizerBaseClass.h>
class Layer {
public:
    std::vector<int> output_dims; // to pass into the next layer so it can form its parameter tensor. Used in build function of layers.
    //for back prop
    tensor last_output;
    tensor last_input;

    virtual void build(const std::vector<int>& prev_dim) = 0; ///accepts dims vector of prev tensor

    virtual tensor backward(tensor& grad_op) = 0;

    virtual void layerCompute(tensor &input_vals) = 0;

    virtual void update(OptimizerBaseClass *optimizer, int layer_idx) = 0; //layer_idx is for stateful optims to know which layer's state to access

    virtual ~Layer() = default;
};
