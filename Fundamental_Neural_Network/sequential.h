#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H 
#include <vector>
#include <string>
#include "custommathcpp.h"
#include <memory>
#include <random>
#include <algorithm>

class dense {
    public:
    int size;
    inline static std::mt19937 generator{std::random_device{}()}; // random generator
    //layer data
    std::vector<std::vector<double>> weights;
    std::vector<double> bias;
    std::string activation;
    //forward pass(preds)
    std::vector<double> last_raw_output;
    std::vector<double> last_output;
    std::vector<double> last_input;
    //gradient descent cache
    std::vector<std::vector<double>> grad_weights;
    std::vector<double> grad_bias;
    std::vector<double> grad_input;
    dense(int Size, std::string Activation="Linear"){
        size=Size;
        activation=Activation;
        grad_weights.resize(size);
        
    }

    void resize(int in) {
            weights.resize(size);
            bias.assign(size, 0.0);
            
            // Properly seeded Xavier/Glorot uniform initialization
            double limit = std::sqrt(6.0 / (in + size));
            std::uniform_real_distribution<double> distribution(-limit, limit);
            for (auto &row : weights) {
                row.resize(in);
                for (double &w : row) {
                    w = distribution(generator);
                }
            }
        }
    std::vector<double> backward(std::vector<double> &grad_op){
        // f'(raw output)
        std::vector<double> delta = activationSpecificDerivative(activation,last_raw_output);
        
        // convert it into dL/dZ aka theta(delta here)
        delta = vectorPointWiseMult(delta,grad_op);

        //dL/dW
        grad_weights= vecTransposeMultVector(delta,last_input);
        //dL/dBias
        grad_bias = delta;
        //dL/dinput
        grad_input = vecTransposeXmatrix(delta, weights);
        return grad_input;
    }
    void layerCompute(std::vector<double> &input){
        last_input=input;
        last_raw_output = addBias(vecXmatrix(input, weights),bias);
        last_output = performActivation(activation, last_raw_output);
    }

};
class Seq{
    public:
    std::vector<std::unique_ptr<dense>> layers;
    std::string error;
    int input_dim;
    public:
    Seq(int ip_size,int first_layer_dim, std::string activation = "Linear", std::string Error="mse"){
        layers.push_back(std::make_unique<dense>(first_layer_dim, activation));
        error=Error;
        input_dim=ip_size;
    }
    void add(int Size,std::string activation = "Linear"){
        layers.push_back(std::make_unique<dense>(Size, activation));
    }
    void compile(){ // resize weight matrices of each layer
        layers[0]->resize(input_dim);
        for(int i = 1;i<layers.size();i++){
            layers[i]->resize(layers[i-1]->size);
        }
    }
   
    double onePassUpdate(std::vector<double> &input_vals, std::vector<double> &target, double &learning_rate){  //perform one pass through the neural net
        //forward pass
        layers[0]->layerCompute(input_vals);
        for(int i = 1;i<layers.size();i++){
            layers[i]->layerCompute(layers[i-1]->last_output);
        }
        double l = getError(target,layers[layers.size()-1]->last_output,error);
        //backward pass
        std::vector<double> gradient;
        gradient = vectorSubtract(layers[layers.size()-1]->last_output,target);
        double scale = 2.0/target.size();
        for(auto &g:gradient) g*=scale;
        for(int i = layers.size() - 1; i >= 0; i--) {
            gradient = layers[i]->backward(gradient);
        }
        //update weights and bias
        for(int i = 0;i<layers.size();i++){
            updateWeights(layers[i]->weights,layers[i]->grad_weights,learning_rate);
            updateBias(layers[i]->bias,layers[i]->grad_bias,learning_rate);
        }
        return l;
    }
    double test(std::vector<double> &input_vals, std::vector<double> &target){
        layers[0]->layerCompute(input_vals);
        for(int i = 1;i<layers.size();i++){
            layers[i]->layerCompute(layers[i-1]->last_output);
        }
        double l = getError(target,layers[layers.size()-1]->last_output,error);
        return l;
    }
    std::vector<std::pair<double,double>> fit(std::vector<std::vector<double>> &inputArray, std::vector<std::vector<double>> &targetArray, int epochs, double learning_rate = 0.0001,bool shuffle=true, float split=0.8){
        std::vector<std::pair<double,double>> history;
        int train=split*inputArray.size();
        std::vector<int> indices(inputArray.size());
        std::random_device rd;
        std::default_random_engine rng(rd());
        for (size_t i = 0; i < indices.size(); ++i) indices[i] = i;
        if(shuffle){
            // Shuffle indices
            std::shuffle(indices.begin(), indices.end(), rng);
        }
        for(int _ = 0;_<epochs;_++){
            double train_avg_error=0;
            int i;
            for(i = 0;i<train;i++){
                int idx = indices[i];
                train_avg_error += onePassUpdate(inputArray[idx],targetArray[idx],learning_rate);
            }
            train_avg_error/=train;
            double test_avg_error=0;
            for(;i<inputArray.size();i++){
                int idx = indices[i];
                test_avg_error+=test(inputArray[idx],targetArray[idx]);
            }
            test_avg_error/=(inputArray.size()-train);
            history.push_back({train_avg_error,test_avg_error});
        }
        return history;
    }
    std::vector<double> pred(std::vector<double> input_vals){
        layers[0]->layerCompute(input_vals);
        for(int i = 1;i<layers.size();i++){
            layers[i]->layerCompute(layers[i-1]->last_output);
        }
        return layers[layers.size()-1]->last_output;
    }
    
};
#endif
