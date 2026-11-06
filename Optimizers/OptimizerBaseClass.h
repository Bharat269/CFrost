#pragma once
#include <Tensor/tensor.h>
#include <Models/ModelBaseClass.h>
class OptimizerBaseClass{
    public:
    virtual void initialize_state(ModelBaseClass*)=0; //adjust the optimizers state array dimnsion to match the layers/params in the model. irrelevant for sgd, needed for adam
    virtual ~OptimizerBaseClass() = default;
    virtual void update(tensor &weights, tensor &gradients, int &layer_idx)=0 ; //perform update based on the optimizer type
    // layer_idx to know which layer of the model is being updated.
    // Irrelevant for stateless optim like SGD, needed for stateful otpim like Adam
};

