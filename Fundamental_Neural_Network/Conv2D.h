#ifndef CONV2D_H
#define CONV2D_H
#include "layer.h"
#include "CustomMath_CPU.h"
#include "ActivationClasses.h"
#include <stdexcept>

class Conv2D: public Layer{ //Input must be a tensor of 3D. Number of images(original or convolved), image length, image breadth. Intial input will be (count,l,b)
    public:
    /*
    Classic 2D greyscale Conv. 
    The prev layer gives n feature maps, one each from its n filters.
    The current layer has m filters. A filter isnt a matrix, but a set of matrices called kernels. 
    Each filter applies a unique matrix to each filter maps. 
    Ergo, one filter applies n matrices, one each for the n feature maps from prev output.
    NOT one single matrix per filter across all the n feature maps.
    So, n feature maps, m filters, n*m matrices.
    But bias is const across all kernels in a filter.
    */
    tensor filters; 
    tensor bias;// one for each filter, applied across all kernels of that filter
    std::unique_ptr<ActivationBaseClass> activation;
    //std::vector<int> output_dims; inherited from layer
    
    tensor last_raw_output;
    //tensor last_output; inherited from layer
    // tensor last_input; inherited from layer

    int step;
    /*
    filter_dims:{filter_count, kernel_height, kernel_width}
    The number of input channels is not known yet.
    It is determined when build() receives the previous
    layer's output dimensions.
    */
    Conv2D(std::vector<int> filter_dims,int step_size = 1,std::string Activation = "linear") { //filter dims should be {kernel count, kernel heigth, kernel width}
        if(filter_dims.size() != 3)throw std::invalid_argument( "In Conv2D::Conv2D(), filter dimensions must be {filter_count, kernel_height, kernel_width}.");
        if(filter_dims[0] <= 0 ||filter_dims[1] <= 0 ||filter_dims[2] <= 0)throw std::invalid_argument("In Conv2D::Conv2D(), filter dimensions must be ""positive integers.");
        if(step_size <= 0)throw std::invalid_argument("In Conv2D::Conv2D(), step size must be positive.");
        filters.setDim(filter_dims);
        bias.setDim({filter_dims[0]}); //number of biases = number of filters. One bias is constant across all cells of all kernels of one filter.
        activation = assignActivation(Activation);
        step = step_size;
    }
    /*
    Receives the dimensions of the previous layer. {input_channels, input_height, input_width}
    Before build() filters dimensions are {filter_count, kernel_height, kernel_width}
    After build()filter dimensions are { filter_count, input_channels, kernel_height, kernel_width}
    */
    void build(const std::vector<int>& prev_dim) override {

        if(prev_dim.size() != 3) throw std::invalid_argument("In Conv2D::build(), the previous layer must output a 3D tensor {input_channels, height, width}.");
        // not treating the previous layers input feature count as part of output dims because its irrelevant.
        if(prev_dim[0] <= 0 ||prev_dim[1] <= 0 ||prev_dim[2] <= 0)throw std::invalid_argument("In Conv2D::build(), the previous layer has at least one dimension less than or equal to zero.");
        //convert current filter to 4D. While initializing with constructor, we didnt have the previous layers data, i.e. the output count. So we are adding a dimension in build()
        std::vector<int> filter_dims = filters.getDims();
        int filter_count   = filter_dims[0];
        int kernel_height  = filter_dims[1];
        int kernel_width   = filter_dims[2];
        int input_channels = prev_dim[0];
        int input_height   = prev_dim[1];
        int input_width    = prev_dim[2];
        if(kernel_height > input_height || kernel_width > input_width) throw std::invalid_argument("In Conv2D::build(), kernel dimensions cannot be greater than the corresponding input dimensions.");

        filters.setDim({filter_count, input_channels, kernel_height, kernel_width});
        output_dims = { filter_count, (input_height - kernel_height) / step + 1, (input_width - kernel_width) / step + 1 };
        //output_dims was declared in base class. Output dims will be passed into the next layer in Seq::compile()
        last_output.setDim(output_dims);
        last_input.setDim({input_channels,input_height,input_width});
    }
    tensor backward(tensor& grad_op) override{

    }
    void layerCompute(tensor& input_vals) override {
        last_input.copyTensor(input_vals);
        last_raw_output = performConv2D(input_vals, filters, step);
        last_raw_output = addBias(last_raw_output, bias);
        last_output = activation->performActivation(last_raw_output);
    }
    void update(double learning_rate) override{

    }

};

#endif