#pragma once
#include <Tensor/tensor.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <MathModules/custommath.h>
class LossBaseClass{
    public:
    virtual double getError(const tensor& target,const tensor& prediction)=0;
    virtual tensor getGradient(const tensor &pred, const tensor &target)=0;
    virtual ~LossBaseClass() = default;
};
class mse: public LossBaseClass{
    public:
    double getError(const tensor& target,const tensor& prediction) override;
    tensor getGradient(const tensor &pred, const tensor &target) override;
};
//helper function to alott inherited class instead of base class
std::unique_ptr<LossBaseClass> assignLoss(std::string loss);
