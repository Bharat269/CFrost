#pragma once
#include <vector>
#include <LossClasses/LossClasses.h>
#include <Layers/layer.h>
#include <Optimizers/OptimizerBaseClass.h>
class Seq{
    private:
    std::vector<std::unique_ptr<Layer>> layers;
    std::unique_ptr<LossBaseClass> loss;
    OptimizerBaseClass *optimizer;
    std::vector<int> input_dim; //will* be specified to an inherited class in constructor during run time
    bool compiled = false;
    public:
    Seq(std::vector<int> single_input_dim, OptimizerBaseClass *model_optimizer, std::string Loss="mse");
    void add(std::unique_ptr<Layer> layer);
    void compile(); // resize weight matrices of each layer
    double onePassUpdate(tensor &input_vals, tensor &target);  //perform one pass through the neural net(fwd computation, back propogation, parameter update)
    double test(tensor &input_vals, tensor &target);
    std::vector<std::pair<double,double>> fit(std::vector<tensor> &inputArray, std::vector<tensor> &targetArray, int epochs, bool shuffle=true, float split=0.8);
    tensor pred(tensor input_vals);
};
