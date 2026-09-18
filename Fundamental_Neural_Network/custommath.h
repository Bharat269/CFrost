//this is a template for the background math
#ifndef CUSTOMMATH_H
#define CUSTOMMATH_H
#include "tensor.h"
#include <string>

/*
CREATE A HELPER FUNCTION TO CONVERT TO AND FROM CUSTOM TENSOR A CPU OR GPU BASED BUILT-IN TENSOR FOR BETTER ABSTRACTION
*/
// LINEAR ALGEBRA


// matrix x vector
// a = [out, in]
// b = [in]
// result = [out]
tensor matMul(const tensor& a, const  tensor& b);

// outer product
// a = [out]
// b = [in]
// result = [out, in]
tensor outerProduct(const tensor& a, const tensor& b);

// Backward gradient projection: grad_input = W^T * delta
// W^T × delta
//
// W     = [out, in]
// delta = [out]
// result = [in]
tensor transposeMatMul(const tensor& delta,const tensor& weights);

// Element-wise Operations
tensor add(const tensor& a, const tensor& b);

tensor subtract(const tensor& a,const tensor& b);

//multiply corresponding elements
tensor multiply(const tensor& a,const tensor& b);

void tensorMulScalar_inplace(tensor& a, double b);

// Parameter updates
void updateWeights(tensor& weights,const tensor& gradWeights,double learningRate);

void updateBias(tensor& bias,const tensor& gradBias,double learningRate);

//conv2D operations
tensor conv2D(const tensor& input,const tensor& filters,int step);
#endif
