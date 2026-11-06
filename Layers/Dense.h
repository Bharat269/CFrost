#pragma once
#include <Tensor/tensor.h>
#include <Activations\ActivationClasses.h>
#include "layer.h"
#include <memory>
#include <Activations\ActivationClasses.h>
class Dense: public Layer{
    public:
    int size;  //neuron count
    inline static std::mt19937 generator{std::random_device{}()}; // random generator
    
    //layer data
    tensor weights; //2d tensor by size x input
    tensor bias;  //1 d tensor by size
    //std::vector<int> output_dims; inherited from layer

    //forward pass(preds)
    tensor last_raw_output;
    //tensor last_output; inherited from layer
    // tensor last_input; inherited from layer
    
    //gradient descent cache
    tensor grad_weights;
    tensor grad_bias;
    tensor grad_input;
    
    //base class activation pointer. inherited class will be assigned later.
    std::unique_ptr<ActivationBaseClass> activation;
    
    Dense(int Size, std::string Activation="linear");
    
    void build(const std::vector<int> &prev_dim) override ; // resize and initialize all parameters. Called sequentially for all layers.
    //backward pass of the dense layer type
    tensor backward(tensor &grad_op) override;
    //forward pass calc of the dense layer type
    void layerCompute(tensor &input) override ;
    void update(OptimizerBaseClass* optimizer, int layer_idx) override ;
};