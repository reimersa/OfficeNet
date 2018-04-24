#include<iostream>
#include<armadillo>
#include<TGraph.h>
#include<TCanvas.h>
#include<TAxis.h>
#include<TStyle.h>
#include<TLegend.h>

#include "../include/GradientDescent.h"
#include "../include/CostAndGrad.h"

using namespace std;
using namespace arma;

vector<Mat<double>> GradientDescent(Mat<double> Theta1, Mat<double> Theta2, Mat<double> Theta3, Mat<double> X, Mat<double> X_test, Mat<double> X_cv, Mat<double> y, Mat<double> y_test, Mat<double> y_cv, Mat<double> EventWeight, Mat<double> EventWeight_test, Mat<double> EventWeight_cv, double lambda, double alpha, int niter){

  bool debug = true;

  vector<Mat<double>> thetas = {Theta1, Theta2, Theta3};
  vector<double> costhistory, costhistory_test, costhistory_cv;
  vector<double> trainingiter;
  
  for(int i=0; i<niter; i++){
    vector<Mat<double>> gradients = Gradient(thetas[0], thetas[1], thetas[2], X, y, EventWeight, lambda);
  if(gradients.size() != thetas.size()) throw runtime_error("Error in GradientDescent(): The number of Theta matrices and gradients is not equal.");

  // For monitoring
  costhistory.emplace_back(Cost(X, EventWeight, thetas[0], thetas[1], thetas[2], y, lambda));
  costhistory_test.emplace_back(Cost(X_test, EventWeight_test, thetas[0], thetas[1], thetas[2], y_test, lambda));
  costhistory_cv.emplace_back(Cost(X_cv, EventWeight_cv, thetas[0], thetas[1], thetas[2], y_cv, lambda));
  trainingiter.emplace_back(i);
  
    for(unsigned int j=0; j<thetas.size(); j++){
      if(debug){
	//cout << "Theta[" << j << "] before step " << i << ": " << endl << thetas[j] << endl;
	//cout << "Grad[" << j << "] for step " << i << ": " << endl << gradients[j] << endl;
      }
      thetas[j] = thetas[j] - gradients[j] * alpha;
      if(debug){
	//cout << "Theta[" << j << "] after step " << i << ": " << endl << thetas[j] << endl;
	if(i%10==0 && j==0)cout << "Cost after step " << i << ": " << costhistory[i] << endl;
	if(i%10==0 && j==0)cout << "Test-Cost after step " << i << ": " << costhistory_test[i] << endl;
	if(i%10==0 && j==0)cout << "Cross validation-Cost after step " << i << ": " << costhistory_cv[i] << endl;
      }
    }
  }


  // Monitoring: Produce graph of cost vs. niter
  TGraph* costvsiter = new TGraph(costhistory.size(), &trainingiter[0], &costhistory[0]);
  TGraph* costvsiter_test = new TGraph(costhistory_test.size(), &trainingiter[0], &costhistory_test[0]);
  TGraph* costvsiter_cv = new TGraph(costhistory_cv.size(), &trainingiter[0], &costhistory_cv[0]);

  double maximum = max(max(costhistory[0], costhistory_test[0]), costhistory_cv[0]) * 1.2;
  double minimum = min(min(costhistory[niter-1], costhistory_test[niter-1]), costhistory_cv[niter-1]) * 0.8;

  
  unique_ptr<TCanvas> c;
  c.reset(new TCanvas());
  costvsiter->GetXaxis()->SetTitle("Number of iterations");
  costvsiter->GetYaxis()->SetTitle("Cost");
  costvsiter->SetTitle("");
  costvsiter->SetMarkerColor(kBlack);
  costvsiter->SetLineColor(kBlack);
  costvsiter_test->SetMarkerColor(kRed+1);
  costvsiter_test->SetLineColor(kRed+1);
  costvsiter_cv->SetMarkerColor(kBlue+1);
  costvsiter_cv->SetLineColor(kBlue+1);
  costvsiter->Draw("APL");
  costvsiter_test->Draw("PL SAME");
  costvsiter_cv->Draw("PL SAME");
  costvsiter->GetYaxis()->SetRangeUser(minimum,maximum);
  c->SetLogy();

  unique_ptr<TLegend> leg;
  leg.reset(new TLegend(0.5,0.6,0.7,0.8));
  leg->SetBorderSize(0);
  leg->SetFillStyle(0);
  leg->AddEntry(costvsiter, "Training set", "L");
  leg->AddEntry(costvsiter_test, "Test set", "L");
  leg->AddEntry(costvsiter_cv, "Cross validation set", "L");

  leg->Draw();
  c->SaveAs("Monitoring/CostVsIterations.eps");
  
  
  return thetas;
  
}
