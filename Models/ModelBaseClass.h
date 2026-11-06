#pragma once
#include <vector>
#include <LossClasses/LossClasses.h>
class OptimizerBaseClass;//to avoid circular dependency of model and optimizer
class ModelBaseClass{
    public:
    std::unique_ptr<LossBaseClass> loss;
    OptimizerBaseClass *optimizer;
    std::vector<int> input_dim; //will* be specified to an inherited class in constructor during run time
    bool compiled = false;

    virtual void compile()=0; // resize weight matrices of each layer
    virtual double onePassUpdate(tensor &input_vals, tensor &target)=0;  //perform one pass through the neural net(fwd computation, back propogation, parameter update)
    virtual double test(tensor &input_vals, tensor &target)=0;
    virtual std::vector<std::pair<double,double>> fit(std::vector<tensor> &inputArray, std::vector<tensor> &targetArray, int epochs, bool shuffle=true, float split=0.8)=0;
    virtual tensor pred(tensor input_vals)=0;
};
