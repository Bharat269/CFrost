#ifndef DENSE_H
#define DENSE_H
#include "tensor.h"
#include <string>
#include <memory>
#include <random>
#include "layer.h"
#include "ActivationClasses.h"
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
    
    Dense(int Size, std::string Activation="linear"){
        size=Size;

        last_output.setDim({size});
        last_raw_output.setDim({size});
        output_dims={size};

        activation = assignActivation(Activation);
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
        tensor delta = activation->returnDerivative(last_raw_output);
        
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
        last_output = activation->performActivation(last_raw_output);

    }
    void update(double learning_rate) override {
        updateWeights(weights, grad_weights, learning_rate);
        updateBias(bias, grad_bias, learning_rate);
    }

};
#endif