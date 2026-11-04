#pragma once
#include "OptimizerBaseClass.h"
class SGD:public OptimizerBaseClass{
    double learning_rate;
    public:
    SGD();
    SGD(double opt_learning_rate);
    void update(tensor &weights, tensor &gradients) override;
};