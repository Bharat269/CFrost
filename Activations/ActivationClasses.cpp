#include "ActivationClasses.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <string>
//ReLu implementations
tensor relu::performActivation(const tensor& input){
    tensor result(input.getDims());
    for (int i = 0; i < input.getSize(); i++){
        double x = input.getValArray()[i];
        result.getValArray()[i] = std::max(0.0, x);
    }
    return result;
}
tensor relu::returnDerivative(const tensor &input){
    tensor result(input.getDims());
    for(int i=0;i<input.getSize();i++){
        double x = input.getValArray()[i];
        result.getValArray()[i] = x > 0.0 ? 1.0 : 0.0;
    }
    return result;
}
//Sigmoid implementations
tensor sigmoid::performActivation(const tensor& input) {
    tensor result(input.getDims());
    for (int i = 0; i < input.getSize(); i++){
        double x = input.getValArray()[i];
        result.getValArray()[i] = 1.0 / (1.0 +std::exp(-x));
    }
    return result;
}
tensor sigmoid::returnDerivative(const tensor &input) {
    tensor result(input.getDims());
    for(int i=0;i<input.getSize();i++){
        double x = input.getValArray()[i];
        double s = 1.0 / (1.0 + std::exp(-x));
        result.getValArray()[i] = s * (1.0 - s);
    }
    return result;
}
//Linear implementations
tensor linear::performActivation(const tensor& input) {
    tensor result(input.getDims());
    for (int i = 0; i < input.getSize(); i++){
        double x = input.getValArray()[i];
        result.getValArray()[i] = x;
    }
    return result;
}
tensor linear::returnDerivative(const tensor &input) {
    tensor result(input.getDims());
    for(int i=0;i<input.getSize();i++){
        result.getValArray()[i] = 1.0;
    }
    return result;
}
std::unique_ptr<ActivationBaseClass> assignActivation(std::string activation){
    if(activation=="relu") return std::make_unique<relu>();
    else if(activation=="linear") return std::make_unique<linear>();
    else if(activation=="sigmoid") return std::make_unique<sigmoid>();
    else throw std::invalid_argument("In ActivationClasses::assignActivation, the activation type specified does not exist.");
}
