#include <iostream>
#include <armadillo>
#include <math.h>

#include "../include/Sigmoid.h"

using namespace arma;

double Sigmoid(double z){
  return 1./(1.+exp(-z));
}

arma::Mat<double> Sigmoid(arma::Mat<double> z){
  return 1./(1.+exp(-z));
}

