#ifndef CONV2D_H
#define CONV2D_H
#include "layer.h"
#include "CustomMath_CPU.h"
#include <stdexcept>

class Conv2D: public Layer{
    tensor filter;
    std::vector<int> input_dims;
    int step;
    Conv2D(std::vector<int> filter_dims, std::vector<int> ip_dims,int step_size=2, std::string activation="linear"){
        if(ip_dims.size()!=2) throw std::invalid_argument("In COnv2D::Conv2D, the given input dimension must be 2D. Conv2D only processes greyscale images.");
        if(filter_dims.size()!=2) throw std::invalid_argument("In COnv2D::Conv2D, the given filter dimension must be 2D. Conv2D only processes greyscale images.");
        if(filter_dims[0]>ip_dims[0]||filter_dims[1]>ip_dims[1]) throw std::invalid_argument("In Conv2D::Conv2D, filter dimension lengths cannot be greater than corresponding input dimension lengths.");
        filter.setDim(filter_dims);
        input_dims=ip_dims;
    }
};

#endif