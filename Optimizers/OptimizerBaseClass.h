#pragma once
#include <Tensor/tensor.h>
class OptimizerBaseClass{
    public:
    virtual ~OptimizerBaseClass() = default;
    virtual void update(tensor &weights, tensor &gradients)=0 ;
};

