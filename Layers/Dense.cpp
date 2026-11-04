#include "Dense.h"
#include <MathModules/custommath.h>
#include <string>
#include <stdexcept>

#include <Activations\ActivationClasses.h>
#include <random>

//MEMBER vars AND INHERITED MEMEBER vars
    // public:
    // int size;  //neuron count
    // inline static std::mt19937 generator{std::random_device{}()}; // random generator

    // //layer data
    // tensor weights; //2d tensor by size x input
    // tensor bias;  //1 d tensor by size
    // //std::vector<int> output_dims; inherited from layer

    // //forward pass(preds)
    // tensor last_raw_output;
    // //tensor last_output; inherited from layer
    // // tensor last_input; inherited from layer

    // //gradient descent cache
    // tensor grad_weights;
    // tensor grad_bias;
    // tensor grad_input;

    // //base class activation pointer. inherited class will be assigned later.
    // std::unique_ptr<ActivationBaseClass> activation;

//Constructor
Dense::Dense(int Size, std::string Activation){
    size=Size;

    last_output.setDim({size});
    last_raw_output.setDim({size});
    output_dims={size};
    activation = assignActivation(Activation);
}
    
void Dense::build(const std::vector<int> &prev_dim) { // resize and initialize all parameters. Called sequentially for all layers.
    if(prev_dim.empty())throw std::invalid_argument("In Dense::build(), previous layer has no output dimensions.");
    if(prev_dim.size()!=1) throw std::invalid_argument("In Dense::build(), output dimensions of the layer before a dense layer must be 1D."); //in case someone tries fitting a conv2d or pool layer without flattening ig
    int in = prev_dim[0];
    weights.setDim({size,in}); //input being scalar here is okk because dense will only have 1D
    grad_weights.setDim({size,in});
    bias.setDim({size});
    grad_bias.setDim({size});
    last_input.setDim({in});
    grad_input.setDim({in});
    weights.randomizeVal(size*in);
}
//backward pass of the dense layer type
tensor Dense::backward(tensor &grad_op) {
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
void Dense::layerCompute(tensor &input)  { //forward pass calc
    last_input.copyTensor(input);
    last_raw_output=add(matMul( weights,input),bias);
    last_output = activation->performActivation(last_raw_output);

}
void Dense::update(OptimizerBaseClass* optimizer)  {
    optimizer->update(weights,grad_weights);
    optimizer->update(bias,grad_bias);
}
