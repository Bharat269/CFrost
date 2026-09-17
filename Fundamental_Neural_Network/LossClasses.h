#ifndef LOSS_CLASSES
#define LOSS_CLASSES
#include "tensor.h"
#include <memory>
#include <stdexcept>
#include <string>
class LossBaseClass{
    public:
    virtual double getError(const tensor& target,const tensor& prediction)=0;
    ~LossBaseClass() = default;
};
class mse: public LossBaseClass{
    public:
    double getError(const tensor& target,const tensor& prediction) override{
        if (target.getDims() != prediction.getDims()) throw std::invalid_argument("Target and prediction dimensions must match");
        double sum = 0.0;

        for (int i = 0; i < target.getSize(); i++){
            double difference =prediction.getValArray()[i] -target.getValArray()[i];
            sum += difference * difference;
        }
        return sum / static_cast<double>(target.getSize());
    }
};

//helper function to alott inherited class instead of base class
std::unique_ptr<LossBaseClass> assignLoss(std::string loss){
    if(loss=="mse"||loss=="MSE") return std::make_unique<mse>();
    else throw std::invalid_argument("In LossClasses::assignLoss, the loss type specified does not exist.");
}
#endif 
