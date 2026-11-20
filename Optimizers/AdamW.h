#pragma once
#include "OptimizerBaseClass.h"
struct AdamState{
    tensor first_moment;
    tensor second_moment;
    int step; // the t value. increased after performing the update. same for the whole model in one update
};
class AdamW: public OptimizerBaseClass{
    float learning_rate;
    float beta1;//first moment decay
    float beta2;//second moment decay
    float epsilon;//numerical stability constant
    float lambda;//weight decay
    
    public:
    AdamW::AdamW(float learningRate=0.001, float Beta1=0.9, float Beta2=0.999, float Epsilon=0.00000001, float Lambda=0.01);
    void initialize_state(ModelBaseClass* model) override;
    void update(tensor &weights, tensor &gradients, int &layer_idx) override;
    
};