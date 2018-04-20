#include<armadillo>
#include<iostream>

#include "../include/Sigmoid.h"
#include "../include/SigmoidGradient.h"

using namespace std;
using namespace arma;

double SigmoidGradient(double z){

  return Sigmoid(z) * (1-Sigmoid(z));
  
}

Mat<double> SigmoidGradient(Mat<double> z){

  return Sigmoid(z) % (1-Sigmoid(z));
  
}
