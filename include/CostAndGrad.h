#include<armadillo>

double Cost(arma::Mat<double> X, arma::Mat<double> Theta1, arma::Mat<double> Theta2, arma::Mat<double> Theta3, arma::Mat<double> y, double lambda);

arma::Mat<double> Gradient(arma::Mat<double> Theta1, arma::Mat<double> Theta2, arma::Mat<double> Theta3, arma::Mat<double> X, arma::Mat<double> y, double lambda);
