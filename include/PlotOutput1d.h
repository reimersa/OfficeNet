#pragma once
#include<TString.h>
#include<armadillo>

void PlotOutput1d(arma::Mat<double> h, arma::Mat<double> y, arma::Mat<double> EventWeight, double frac_low, double frac_high, TString plotname, TString tag);
