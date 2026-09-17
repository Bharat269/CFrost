#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H 
#include <vector>
#include <string>
#include <algorithm>
#include "Dense.h"

class Seq{
    private:
    std::vector<std::unique_ptr<Layer>> layers;
    std::string error;
    std::vector<int> input_dim;
    bool compiled = false;
    public:
    Seq(std::vector<int> single_input_dim, std::string Error="mse"){
        error=Error;
        input_dim=single_input_dim;
    }
    void addDense(int Size,std::string activation = "Linear"){ // add layers to the seq model
        layers.push_back(std::make_unique<Dense>(Size, activation));
    }
    void compile(){ // resize weight matrices of each layer
        if(layers.size()==0) throw std::invalid_argument("No layers in sequential model.");
        layers[0]->build(input_dim);
        for(int i = 1;i<layers.size();i++){
            layers[i]->build(layers[i-1]->output_dims);
        }
        compiled=true;
    }

    double onePassUpdate(tensor &input_vals, tensor &target, double &learning_rate){  //perform one pass through the neural net(fwd computation, back propogation, parameter update)
        if(!compiled) throw std::invalid_argument("Model not compiled yet.");
        //forward pass
        layers[0]->layerCompute(input_vals);
        for(int i = 1;i<layers.size();i++){
            layers[i]->layerCompute(layers[i-1]->last_output);
        }
        double l = mse(target,layers[layers.size()-1]->last_output);
        //backward pass
        tensor gradient = subtract(layers[layers.size()-1]->last_output,target);
        tensorMulScalar_inplace(gradient,(2.0/target.getSize()));
        for(int i = layers.size() - 1; i >= 0; i--) {
            gradient = layers[i]->backward(gradient);
        }
        //update weights and bias
        for(auto &x:layers){
            x->update(learning_rate);
        }
        return l;
    }
    double test(tensor &input_vals, tensor &target){
        if(!compiled) throw std::invalid_argument("Model not compiled yet.");
        layers[0]->layerCompute(input_vals);
        for(int i = 1;i<layers.size();i++){
            layers[i]->layerCompute(layers[i-1]->last_output);
        }
        double l = mse(target,layers[layers.size()-1]->last_output);
        return l;
    }
    std::vector<std::pair<double,double>> fit(std::vector<tensor> &inputArray, std::vector<tensor> &targetArray, int epochs, double learning_rate = 0.0001,bool shuffle=true, float split=0.8){
        if(!compiled) throw std::invalid_argument("Model not compiled yet.");
        if(inputArray.size()!=targetArray.size()) throw std::invalid_argument("Number of rows in input data and target data are not equal");
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
                tensor ip = inputArray[i];
                tensor op = targetArray[i];
                if(ip.getDims()!=input_dim) throw std::invalid_argument("Given input dimensions and model input dimensions dont match at index "+ std::to_string(idx));
                train_avg_error += onePassUpdate(ip,op,learning_rate);
            }
            train_avg_error/=train;
            double test_avg_error=0;
            for(;i<inputArray.size();i++){
                int idx = indices[i];
                tensor ip = inputArray[i];
                tensor op = targetArray[i];
                if(ip.getDims()!=input_dim) throw std::invalid_argument("Given input dimensions and model input dimensions dont match at index "+ std::to_string(idx));
                test_avg_error+=test(ip,op);
            }
            test_avg_error/=(inputArray.size()-train);
            history.push_back({train_avg_error,test_avg_error});
        }
        return history;
    }
    tensor pred(tensor input_vals){ 
        if(!compiled) throw std::invalid_argument("Model not compiled yet.");
        if(input_vals.getDims()!=input_dim) throw std::invalid_argument("Given input dimensions and model input dimensions dont match");
        layers[0]->layerCompute(input_vals);
        for(int i = 1;i<layers.size();i++){
            layers[i]->layerCompute(layers[i-1]->last_output);
        }
        return layers[layers.size()-1]->last_output;
    }

    
};
#endif
