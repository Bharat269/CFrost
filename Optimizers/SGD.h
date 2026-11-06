#pragma once
#include "OptimizerBaseClass.h"
class SGD:public OptimizerBaseClass{
    double learning_rate;
    public:
    SGD();
    SGD(double opt_learning_rate);
    void initialize_state(ModelBaseClass*) override; //not needed for SGD, just implmenting the virtul func
    void update(tensor &weights, tensor &gradients, int &layer_idx) override; //layer_idx wont be used
};