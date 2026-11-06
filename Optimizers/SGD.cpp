#include "SGD.h"
#include <MathModules/custommath.h>
#include <stdexcept>

SGD::SGD(){
    learning_rate=0.0005;
}
SGD::SGD(double opt_learning_rate){
    learning_rate=opt_learning_rate;
}
void SGD::update(tensor &weights, tensor &gradients, int &layer_idx)  {
    //sgd update is severely underoptimized
    //NOTE-- change the math module to adjust this
    if (weights.getDims() != gradients.getDims()) throw std::invalid_argument("In SGD::update(), Weight and gradient dimensions must match");
    subtract_in_place(weights, tensorMulScalar(gradients,learning_rate));
}
void SGD::initialize_state(ModelBaseClass*){
    return;
}
