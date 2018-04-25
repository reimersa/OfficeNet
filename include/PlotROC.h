#pragma once
#include<armadillo>
#include<TString.h>

void PlotROC(arma::Mat<double> h, arma::Mat<double> y, arma::Mat<double> EventWeight, TString plotname, TString tag);
