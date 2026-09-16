#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <stdexcept>

// ============================================================================
// Internal Helpers for Data Layout Conversion
// ============================================================================

// Converts std::vector<std::vector<double>> (2D array) into Eigen::MatrixXd
static Eigen::MatrixXd toEigenMatrix(const std::vector<std::vector<double>> &vec2d) {
    if (vec2d.empty()) return Eigen::MatrixXd(0, 0);
    size_t rows = vec2d.size();
    size_t cols = vec2d[0].size();
    Eigen::MatrixXd mat(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            mat(i, j) = vec2d[i][j];
        }
    }
    return mat;
}

// Converts Eigen::MatrixXd into std::vector<std::vector<double>>
static std::vector<std::vector<double>> toStdVector2D(const Eigen::MatrixXd &mat) {
    std::vector<std::vector<double>> vec2d(mat.rows(), std::vector<double>(mat.cols()));
    for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
            vec2d[i][j] = mat(i, j);
        }
    }
    return vec2d;
}

// ============================================================================
// Matrix Operations
// ============================================================================

// Forward computation: Output = W * Input
std::vector<double> vecXmatrix(std::vector<double> input_vector, std::vector<std::vector<double>> &weights) {
    if (weights.empty() || input_vector.empty()) return {};

    Eigen::Map<const Eigen::VectorXd> x(input_vector.data(), input_vector.size());
    Eigen::MatrixXd W = toEigenMatrix(weights);

    Eigen::VectorXd res = W * x;
    return std::vector<double>(res.data(), res.data() + res.size());
}

// Gradient weight outer product: dL/dW = delta (outer) input_vals
std::vector<std::vector<double>> vecTransposeMultVector(std::vector<double> &a, std::vector<double> &b) {
    if (a.empty() || b.empty()) return {};

    Eigen::Map<const Eigen::VectorXd> vecA(a.data(), a.size());
    Eigen::Map<const Eigen::VectorXd> vecB(b.data(), b.size());

    Eigen::MatrixXd outer = vecA * vecB.transpose();
    return toStdVector2D(outer);
}

// Backward gradient projection: grad_input = W^T * delta
std::vector<double> vecTransposeXmatrix(const std::vector<double> &delta, const std::vector<std::vector<double>> &weights) {
    if (weights.empty() || delta.empty()) return {};

    Eigen::Map<const Eigen::VectorXd> d(delta.data(), delta.size());
    Eigen::MatrixXd W = toEigenMatrix(weights);

    Eigen::VectorXd res = W.transpose() * d;
    return std::vector<double>(res.data(), res.data() + res.size());
}

// ============================================================================
// Vector Arithmetic
// ============================================================================

std::vector<double> addBias(std::vector<double> input_vector, std::vector<double> &bias) {
    Eigen::Map<const Eigen::VectorXd> in(input_vector.data(), input_vector.size());
    Eigen::Map<const Eigen::VectorXd> b(bias.data(), bias.size());

    Eigen::VectorXd res = in + b;
    return std::vector<double>(res.data(), res.data() + res.size());
}

std::vector<double> vectorPointWiseMult(std::vector<double> &a, std::vector<double> &b) {
    Eigen::Map<const Eigen::VectorXd> vecA(a.data(), a.size());
    Eigen::Map<const Eigen::VectorXd> vecB(b.data(), b.size());

    Eigen::VectorXd res = vecA.cwiseProduct(vecB);
    return std::vector<double>(res.data(), res.data() + res.size());
}

std::vector<double> vectorSubtract(std::vector<double> &a, std::vector<double> &b) {
    Eigen::Map<const Eigen::VectorXd> vecA(a.data(), a.size());
    Eigen::Map<const Eigen::VectorXd> vecB(b.data(), b.size());

    Eigen::VectorXd res = vecA - vecB;
    return std::vector<double>(res.data(), res.data() + res.size());
}

// ============================================================================
// Parameter Optimizers (SGD)
// ============================================================================

void updateWeights(std::vector<std::vector<double>> &weights, std::vector<std::vector<double>> &grad_weights, double &learning_rate) {
    Eigen::MatrixXd W = toEigenMatrix(weights);
    Eigen::MatrixXd dW = toEigenMatrix(grad_weights);

    W -= learning_rate * dW;
    weights = toStdVector2D(W);
}

void updateBias(std::vector<double> &bias, std::vector<double> &grad_bias, double &learning_rate) {
    Eigen::Map<Eigen::VectorXd> b(bias.data(), bias.size());
    Eigen::Map<const Eigen::VectorXd> db(grad_bias.data(), grad_bias.size());

    b -= learning_rate * db;
}

// ============================================================================
// Activations & Derivatives
// ============================================================================

std::vector<double> performActivation(std::string &activation, std::vector<double> input_vector) {
    Eigen::Map<const Eigen::VectorXd> in(input_vector.data(), input_vector.size());
    Eigen::VectorXd out(in.size());

    if (activation == "ReLU" || activation == "relu") {
        out = in.cwiseMax(0.0);
    } else if (activation == "Sigmoid" || activation == "sigmoid") {
        out = in.unaryExpr([](double x) { return 1.0 / (1.0 + std::exp(-x)); });
    } else if (activation == "GELU" || activation == "gelu") {
        out = in.unaryExpr([](double x) {
            return 0.5 * x * (1.0 + std::tanh(std::sqrt(2.0 / M_PI) * (x + 0.044715 * std::pow(x, 3))));
        });
    } else { // Linear / Default
        out = in;
    }
    return std::vector<double>(out.data(), out.data() + out.size());
}

std::vector<double> activationSpecificDerivative(std::string activation, std::vector<double> &last_raw_output) {
    Eigen::Map<const Eigen::VectorXd> in(last_raw_output.data(), last_raw_output.size());
    Eigen::VectorXd der(in.size());

    if (activation == "ReLU" || activation == "relu") {
        der = in.unaryExpr([](double x) { return x > 0.0 ? 1.0 : 0.0; });
    } else if (activation == "Sigmoid" || activation == "sigmoid") {
        der = in.unaryExpr([](double x) {
            double s = 1.0 / (1.0 + std::exp(-x));
            return s * (1.0 - s);
        });
    } else { // Linear / Default
        der = Eigen::VectorXd::Ones(in.size());
    }
    return std::vector<double>(der.data(), der.data() + der.size());
}

// ============================================================================
// Error & Loss Calculations
// ============================================================================

double mse(std::vector<double> &target, std::vector<double> pred) {
    Eigen::Map<const Eigen::VectorXd> t(target.data(), target.size());
    Eigen::Map<const Eigen::VectorXd> p(pred.data(), pred.size());
    return (p - t).squaredNorm() / static_cast<double>(target.size());
}

double mse(double target, std::vector<double> pred) {
    if (pred.empty()) return 0.0;
    return (pred[0] - target) * (pred[0] - target);
}

double getError(double target, double pred, std::string error) {
    return (pred - target) * (pred - target);
}

double getError(std::vector<double> &target, std::vector<double> pred, std::string error) {
    return mse(target, pred);
}