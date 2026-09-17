#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include "tensor.h"
#include <unsupported/Eigen/CXX11/Tensor>
//using aliases for eigen tensors
using EigenTensor1D = Eigen::Tensor<double, 1, Eigen::RowMajor>;
using EigenTensor2D = Eigen::Tensor<double, 2, Eigen::RowMajor>;
using EigenTensor3D = Eigen::Tensor<double, 3, Eigen::RowMajor>;
using EigenTensor4D = Eigen::Tensor<double, 4, Eigen::RowMajor>;


//custom tensor to eigen::tensor
template <int Rank>
auto toEigen(const tensor& t)
{
    const auto& dims = t.getDims();
    if (dims.size() != Rank)
    {
        throw std::invalid_argument("Tensor rank does not match requested Eigen rank");
    }
    Eigen::DSizes<Eigen::DenseIndex, Rank> dimensions;

    for (int i = 0; i < Rank; ++i) dimensions[i] = dims[i];

    Eigen::Tensor<double, Rank, Eigen::RowMajor> result(dimensions);

    std::copy(t.getValArray().begin(), t.getValArray().end(),result.data());
    return result;
}
//eigen::tensor to custom tensor
template <typename Scalar, int Rank, int Options, typename IndexType>
tensor fromEigen(const Eigen::Tensor<Scalar, Rank, Options, IndexType>& t)
{
    std::vector<int> dims(Rank);
    std::vector<double> vals(t.size());
    for (int i = 0; i < Rank; ++i)dims[i] = static_cast<int>(t.dimension(i));
    std::copy(t.data(), t.data() + t.size(), vals.begin());
    return tensor(dims, vals);
}


// LINEAR ALGEBRA


// matrix x vector
// a = [out, in]
// b = [in]
// result = [out]
tensor matMul(const tensor& a, const  tensor& b)
{
    auto &aDims = a.getDims();
    auto &bDims = b.getDims();

    if (aDims.size() != 2 || bDims.size() != 1)
        throw std::invalid_argument(
            "In CustomMathCPU.matMul(), matMul expects a 2D tensor and a 1D tensor"
        );

    if (aDims[1] != bDims[0])
        throw std::invalid_argument(
            "In CustomMathCPU.matMul(), matMul dimension mismatch"
        );

    EigenTensor2D A = toEigen<2>(a);
    EigenTensor1D B = toEigen<1>(b);

    Eigen::Tensor<double, 1, Eigen::RowMajor> result =
        A.contract(
            B,
            Eigen::array<Eigen::IndexPair<int>, 1>{
                Eigen::IndexPair<int>(1, 0)
            }
        );

    return fromEigen(result);
}

// outer product
// a = [out]
// b = [in]
// result = [out, in]
tensor outerProduct(const tensor& a, const tensor& b)
{
    const auto& aDims = a.getDims();
    const auto& bDims = b.getDims();

    if (aDims.size() != 1 || bDims.size() != 1)
        throw std::invalid_argument(
            "In CustomMathCPU.outerProduct(), outerProduct expects two 1D tensors"
        );

    EigenTensor1D A = toEigen<1>(a);
    EigenTensor1D B = toEigen<1>(b);

    EigenTensor2D result =
        A.contract(
            B,
            Eigen::array<Eigen::IndexPair<int>, 0>{}
        );

    return fromEigen(result);
}

// Backward gradient projection: grad_input = W^T * delta
// W^T × delta
//
// W     = [out, in]
// delta = [out]
// result = [in]
tensor transposeMatMul(const tensor& delta,const tensor& weights)
{
    const auto& deltaDims = delta.getDims();
    const auto& weightDims = weights.getDims();

    if (deltaDims.size() != 1 || weightDims.size() != 2) throw std::invalid_argument( "In CustomMathCPU.transposeMatMul(),transposeMatMul expects a 1D delta and 2D weights" );
    if (weightDims[0] != deltaDims[0])throw std::invalid_argument("In CustomMathCPU.transposeMatMul(), transposeMatMul dimension mismatch" );

    EigenTensor1D D = toEigen<1>(delta);
    EigenTensor2D W = toEigen<2>(weights);
    EigenTensor1D result =
        W.contract(
            D,
            Eigen::array<Eigen::IndexPair<int>, 1>{
                Eigen::IndexPair<int>(0, 0)
            }
        );
    return fromEigen(result);
}


// Element-wise Operations


tensor add(const tensor& a, const tensor& b)
{
    if (a.getDims() != b.getDims()) throw std::invalid_argument("In CustomMathCPU.add(), Tensor dimensions must match for addition" );
    tensor result(a.getDims());
    for (int i = 0; i < a.getSize(); ++i)result.getValArray()[i] = a.getValArray()[i] + b.getValArray()[i];
    return result;
}

tensor subtract(const tensor& a,const tensor& b)
{
    if(a.getDims() != b.getDims())throw std::invalid_argument("In CustomMathCPU.subtract(),Tensor dimensions must match for subtraction");
    tensor result(a.getDims());
    for (int i = 0; i < a.getSize(); ++i)result.getValArray()[i] =a.getValArray()[i] - b.getValArray()[i];
    return result;
}

//multiply corresponding elements
tensor multiply(const tensor& a,const tensor& b)
{
    if (a.getDims() != b.getDims())throw std::invalid_argument( "In CustomMathCPU.multiply(),Tensor dimensions must match for multiplication");
    tensor result(a.getDims());
    for (int i = 0; i < a.getSize(); ++i)result.getValArray()[i] = a.getValArray()[i] * b.getValArray()[i];
    return result;
}

void tensorMulScalar_inplace(tensor& a, double b){
    for(double& x : a.getValArray())
        x *= b;
}
// Parameter updates
void updateWeights(tensor& weights,const tensor& gradWeights,double learningRate)
{
    if (weights.getDims() != gradWeights.getDims()) throw std::invalid_argument("In CustomMathCPU.updateWeights(), Weight and gradient dimensions must match");
    for (int i = 0; i < weights.getSize(); ++i){
        weights.getValArray()[i] -=learningRate * gradWeights.getValArray()[i];
    }
}

void updateBias(tensor& bias,const tensor& gradBias,double learningRate)
{
    if (bias.getDims() != gradBias.getDims())throw std::invalid_argument("In CustomMathCPU.updateBias(), Bias and gradient dimensions must match");
    for (int i = 0; i < bias.getSize(); ++i)
    {
        bias.getValArray()[i] -=learningRate * gradBias.getValArray()[i];
    }
}

