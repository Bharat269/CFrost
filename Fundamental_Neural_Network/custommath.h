//this is a template for the background math
#ifndef CUSTOMMATH_H
#define CUSTOMMATH_H
#include <vector>
#include <string>
#include <Eigen/Dense>
//normal matrix vector mult 
std::vector<double> vecXmatrix(std::vector<double> input_vector, std::vector<std::vector<double>> &weights);

std::vector<std::vector<double>> vecTransposeMultVector(std::vector<double> &a, std::vector<double> &b);

std::vector<double> vecTransposeXmatrix(const std::vector<double> &delta, const std::vector<std::vector<double>> &weights);

//point wise math
std::vector<double> addBias(std::vector<double> input_vector, std::vector<double> &bias);

std::vector<double> vectorPointWiseMult(std::vector<double> &a, std::vector<double> &b);

std::vector<double> vectorSubtract(std::vector<double> &a, std::vector<double> &b);

//parameter updates
void updateWeights(std::vector<std::vector<double>> &weights, std::vector<std::vector<double>> &grad_weights, double learning_rate = 0.05);

void updateBias(std::vector<double> &bias, std::vector<double> &grad_bias, double learning_rate);

//Activations and Derviations
std::vector<double> performActivation(std::string &activation,std::vector<double> input_vector);

std::vector<double> activationSpecificDerivative(std::string activation, std::vector<double> &last_raw_output);

//loss calc
double mse(std::vector<double> &target, std::vector<double> pred);

double mse(double target, std::vector<double> pred);

double getError(double target, double pred, std::string error);

double getError(std::vector<double> &target, std::vector<double> pred, std::string error);

#endif
