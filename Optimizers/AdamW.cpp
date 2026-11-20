#include "AdamW.h"
AdamW::AdamW(float learningRate, float Beta1, float Beta2, float Epsilon, float Lambda){
    learning_rate=learningRate;
    beta1 = Beta1;
    beta2 = Beta2;
    epsilon = Epsilon;
    lambda = Lambda;
}
void AdamW::initialize_state(ModelBaseClass *model){
    
}