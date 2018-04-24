#pragma once
#include<iostream>
#include<armadillo>

std::vector<arma::Mat<double>> GradientDescent(arma::Mat<double> Theta1, arma::Mat<double> Theta2, arma::Mat<double> Theta3, arma::Mat<double> X, arma::Mat<double> X_test, arma::Mat<double> X_cv, arma::Mat<double> y, arma::Mat<double> y_test, arma::Mat<double> y_cv, arma::Mat<double> EventWeight, arma::Mat<double> EventWeight_test, arma::Mat<double> EventWeight_cv, double lambda, double alpha, int niter);
