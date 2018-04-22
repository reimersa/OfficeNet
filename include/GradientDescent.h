#pragma once
#include<iostream>
#include<armadillo>

std::vector<arma::Mat<double>> GradientDescent(arma::Mat<double> Theta1, arma::Mat<double> Theta2, arma::Mat<double> Theta3, arma::Mat<double> X, arma::Mat<double> y, double lambda, double alpha, int niter);
