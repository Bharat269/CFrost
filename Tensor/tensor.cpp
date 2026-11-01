#include "tensor.h"
#include <stdexcept>
#include <vector>
#include <cmath>
#include <random>


//constructors
std::mt19937 tensor::generator{std::random_device{}()};
tensor::tensor(){
    dims={};
    vals={};
}
tensor::tensor(const std::vector<int>& dimensions) {
    for (const int &x:dimensions) if(x<=0) throw std::invalid_argument("In Tensor::tensor(const std::vector<int>& dimensions), negative or zero dimensions are invalid.");
    dims = dimensions;
    vals.resize(getSize());
}
tensor::tensor(const std::vector<int>& dimensions,const std::vector<double>& values){
    for (const int &x:dimensions) if(x<=0) throw std::invalid_argument("In Tensor::tensor(const std::vector<int>& dimensions,const std::vector<double>& values), negative or zero dimensions are invalid.");
    dims = dimensions;
    if (values.size() != getSize()){
        throw std::invalid_argument("Tensor data size does not match dimensions");
    }
    vals = values;
}

int tensor::getSize() const{ //number of elements accoridng to dim
    int prod = 1;
    for (const int& x : dims)prod *= x;
    return prod;
}
int tensor::noDims(){//return number of dimensions
    return dims.size();
}
int tensor::getIndex(const std::vector<int>& indices){ //return flat index based on dimension indices
    if (indices.size() != dims.size())
    {
        throw std::out_of_range("In tensor::getIndex, Tensor dimensions count do not match the number of given indices");
    }
    int index = 0;
    int stride = 1;
    // Row-major indexing
    for (int i = static_cast<int>(dims.size()) - 1;i >= 0;--i){
        if(indices[i] < 0 || indices[i] >= dims[i]){
            throw std::out_of_range("In tensor::getIndex, Tensor index out of bounds" );
        }
        index += indices[i] * stride;
        stride *= dims[i];
    }
    return index;
}
double& tensor::getVal(const std::vector<int>& indices){ //return value at given indices(dimension based not flat)
    return vals[getIndex(indices)];
}
std::vector<int>& tensor::getDims(){  //get dimensions array
    return dims;
}
const std::vector<int>& tensor::getDims()const { //get dimensions array (const overload)
    return dims;
}
std::vector<double>& tensor::getValArray(){ //get flat value array 
    return vals;
}
const std::vector<double>& tensor::getValArray() const { //get flat value array (const overload)
    return vals;
}
void tensor::setDim(std::vector<int> dimensions){ //set dimensions and resize vals by flattened(dim)
    for (int &x:dimensions) if(x<=0) throw std::invalid_argument("In Tensor::setDim(), negative or zero dimensions are invalid.");
    dims =dimensions;
    vals.resize(getSize());
}
void tensor::setVal(double x){ //assign vals via single scalar
    for (double &y:vals) y=x;
}
void tensor::setVal(std::vector<double> &temp){ //assign vals by copying a flat vector
    if(temp.size()!=getSize()) throw std::invalid_argument("In tensor::setVal, input data size does not match dimensions");
    vals = temp;
}
void tensor::copyTensor(tensor &temp){ //copy a tensor
    dims=temp.dims;
    vals=temp.vals;
}
void tensor::randomizeVal(int length){ //assign random values to
    double limit = std::sqrt(6.0 / (length));
    std::uniform_real_distribution<double> distribution(-limit, limit);
    for(double &x:vals) x = distribution(generator);
}

