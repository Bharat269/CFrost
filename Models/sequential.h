#pragma once
#include "ModelBaseClass.h"
#include <Layers/layer.h>
class Seq: public ModelBaseClass{
    private:
    std::vector<std::unique_ptr<Layer>> layers;
    // std::unique_ptr<LossBaseClass> loss;
    // OptimizerBaseClass *optimizer;
    // std::vector<int> input_dim; //will* be specified to an inherited class in constructor during run time
    // bool compiled = false;
    public:
    Seq(std::vector<int> single_input_dim, OptimizerBaseClass *model_optimizer, std::string Loss="mse");
    void add(std::unique_ptr<Layer> layer);
    void compile() override; // resize weight matrices of each layer
    double onePassUpdate(tensor &input_vals, tensor &target) override;  //perform one pass through the neural net(fwd computation, back propogation, parameter update)
    double test(tensor &input_vals, tensor &target) override;
    std::vector<std::pair<double,double>> fit(std::vector<tensor> &inputArray, std::vector<tensor> &targetArray, int epochs, bool shuffle=true, float split=0.8) override;
    tensor pred(tensor input_vals)override;
};
