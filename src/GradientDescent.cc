#include<iostream>
#include<armadillo>
#include<TGraph.h>
#include<TCanvas.h>
#include<TAxis.h>
#include<TStyle.h>

#include "../include/GradientDescent.h"
#include "../include/CostAndGrad.h"

using namespace std;
using namespace arma;

vector<Mat<double>> GradientDescent(Mat<double> Theta1, Mat<double> Theta2, Mat<double> Theta3, Mat<double> X, Mat<double> y, double lambda, double alpha, int niter){

  bool debug = true;

  vector<Mat<double>> thetas = {Theta1, Theta2, Theta3};
  vector<double> costhistory;
  vector<double> trainingiter;
  
  for(int i=0; i<niter; i++){
  vector<Mat<double>> gradients = Gradient(thetas[0], thetas[1], thetas[2], X, y, lambda);
  if(gradients.size() != thetas.size()) throw runtime_error("Error in GradientDescent(): The number of Theta matrices and gradients is not equal.");

  // For monitoring
  costhistory.emplace_back(Cost(X, thetas[0], thetas[1], thetas[2], y, lambda));
  trainingiter.emplace_back(i);
  
    for(unsigned int j=0; j<thetas.size(); j++){
      if(debug){
	//cout << "Theta[" << j << "] before step " << i << ": " << endl << thetas[j] << endl;
	//cout << "Grad[" << j << "] for step " << i << ": " << endl << gradients[j] << endl;
      }
      thetas[j] = thetas[j] - gradients[j] * alpha;
      if(debug){
	//cout << "Theta[" << j << "] after step " << i << ": " << endl << thetas[j] << endl;
	if(i%10==0 && j==0)cout << "Cost after step " << i << ": " << Cost(X, thetas[0], thetas[1], thetas[2], y, lambda) << endl;
      }
    }
  }


  // Monitoring: Produce graph of cost vs. niter
  unique_ptr<TGraph> costvsiter;
  costvsiter.reset(new TGraph(costhistory.size(), &trainingiter[0], &costhistory[0]));
  unique_ptr<TCanvas> c;
  c.reset(new TCanvas());
  costvsiter->GetXaxis()->SetTitle("Number of iterations");
  costvsiter->GetYaxis()->SetTitle("Cost");
  costvsiter->SetTitle("");
  costvsiter->Draw("APL");
  c->SetLogy();
  c->SaveAs("Monitoring/CostVsIterations.eps");
  
  
  return thetas;
  
}
