#pragma once
#include <vector>
#include <random>
class tensor
{
private:
    std::vector<int> dims;
    std::vector<double> vals;
    static std::mt19937 generator;
public:
    //constructors
    tensor();
    explicit tensor(const std::vector<int>& dimensions);
    tensor(const std::vector<int>& dimensions,const std::vector<double>& values);
    
    int getSize() const ; //number of elements accoridng to dim
    int noDims(); //return number of dimensions
    int getIndex(const std::vector<int>& indices); //return flat index based on dimension indices

    double& getVal(const std::vector<int>& indices); //return value at given indices(dimension based not flat)

    std::vector<int>& getDims();  //get dimensions array
    const std::vector<int>& getDims() const; //get dimensions array (const overload)
    std::vector<double>& getValArray(); //get flat value array 
    const std::vector<double>& getValArray() const; //get flat value array (const overload)

    void setDim(std::vector<int> dimensions); //set dimensions and resize vals by flattened(dim)
    void setVal(double x); //assign vals via single scalar
    void setVal(std::vector<double> &temp); //assign vals by copying a flat vector
    void copyTensor(tensor &temp); //copy a tensor
    void randomizeVal(int length); //assign random values to
};
