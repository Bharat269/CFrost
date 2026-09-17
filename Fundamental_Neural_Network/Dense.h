#ifndef DENSE_H
#define DENSE_H
#include "tensor.h"
#include <string>
#include "CustomMath_CPU.h"
#include <memory>
#include <random>
#include "layer.h"
class Dense: public Layer{
    public:
    int size;  //neuron count
    inline static std::mt19937 generator{std::random_device{}()}; // random generator
    //layer data
    tensor weights; //2d tensor by size x input
    tensor bias;  //1 d tensor by size
    std::string activation;
    //forward pass(preds)
    tensor last_raw_output;
    //gradient descent cache
    tensor grad_weights;
    tensor grad_bias;
    tensor grad_input;
    Dense(int Size, std::string Activation="Linear"){
        size=Size;
        output_dims={};
        output_dims.push_back(size);
        activation=Activation;
        last_output.setDim({size});
        last_raw_output.setDim({size});
        output_dims={size};
    }
    
    void build(const std::vector<int> &prev_dim) override { // resize and initialize all matrix
        int in = prev_dim[0];
        weights.setDim({size,in}); //input being scalar here is okk because dense will only have 1D
        grad_weights.setDim({size,in});
        bias.setDim({size});
        grad_bias.setDim({size});
        last_input.setDim({in});
        grad_input.setDim({in});
        weights.randomizeVal(size*in);
    }
    tensor backward(tensor &grad_op) override{
        // f'(raw output)
        tensor delta = activationSpecificDerivative(activation,last_raw_output);
        
        // convert it into dL/dZ aka theta(delta here)
        delta = multiply(delta,grad_op);

        //dL/dW
        grad_weights= outerProduct(delta,last_input);
        //dL/dBias
        grad_bias = delta;
        //dL/dinput
        grad_input = transposeMatMul(delta, weights);
        return grad_input;
    }
    void layerCompute(tensor &input)override { //forward pass calc
        last_input.copyTensor(input);
        last_raw_output=add(matMul( weights,input),bias);
        last_output = performActivation(activation, last_raw_output);

    }
    void update(double learning_rate) override {
        updateWeights(weights, grad_weights, learning_rate);
        updateBias(bias, grad_bias, learning_rate);
    }

};
#endif