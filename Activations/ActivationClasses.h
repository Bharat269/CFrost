#pragma once
#include <Tensor/tensor.h>
#include <memory>
class ActivationBaseClass{ //to enforce all classes and maintain consistency
    public:
    virtual ~ActivationBaseClass() = default;
    virtual tensor performActivation(const tensor& input)=0;
    virtual tensor returnDerivative(const tensor& input) = 0;
};
//ReLu definitions
class relu: public ActivationBaseClass{
    public:
    tensor performActivation(const tensor& input) override;
    tensor returnDerivative(const tensor &input) override;
};

//Sigmoid definitions
class sigmoid: public ActivationBaseClass{
    public:
    tensor performActivation(const tensor& input) override;
    tensor returnDerivative(const tensor &input) override;
};
//linear definitons
class linear: public ActivationBaseClass{
    public:
    tensor performActivation(const tensor& input) override;
    tensor returnDerivative(const tensor &input) override;
};
std::unique_ptr<ActivationBaseClass> assignActivation(std::string activation);
