#pragma once
#include <vector>
#include <LossClasses/LossClasses.h>
#include <Layers/layer.h>
class Seq{
    private:
    std::vector<std::unique_ptr<Layer>> layers;
    std::unique_ptr<LossBaseClass> loss;
    std::vector<int> input_dim; //will be specified to an inherited class in constructor during run time
    bool compiled = false;
    public:
    Seq(std::vector<int> single_input_dim, std::string Loss="mse");
    void add(std::unique_ptr<Layer> layer);
    void compile(); // resize weight matrices of each layer
    double onePassUpdate(tensor &input_vals, tensor &target, double &learning_rate);  //perform one pass through the neural net(fwd computation, back propogation, parameter update)
    double test(tensor &input_vals, tensor &target);
    std::vector<std::pair<double,double>> fit(std::vector<tensor> &inputArray, std::vector<tensor> &targetArray, int epochs, double learning_rate = 0.0001,bool shuffle=true, float split=0.8);
    tensor pred(tensor input_vals);
};
