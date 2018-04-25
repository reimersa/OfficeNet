#include<TString.h>
#include<iostream>
#include<armadillo>
#include<TGraph.h>
#include<TAxis.h>
#include<TCanvas.h>
#include<TLegend.h>

#include "../include/PlotROC.h"

using namespace std;
using namespace arma;

void PlotROC(Mat<double> h, Mat<double> y, Mat<double> EventWeight, TString plotname, TString tag){

  double sig_tot = 0., bkg_tot = 0.;
  vector<double> sig_sel, bkg_sel;
  int nsteps = 1000.;
  for(int i=0; i<nsteps; i++){
    sig_sel.emplace_back(0.);
    bkg_sel.emplace_back(0.);
  }

  for(unsigned int i=0; i<h.n_rows; i++){
    if(y.at(i,0)==1){
      sig_tot += EventWeight.at(i,0);

      // Scan through output values
      for(int j=0; j<nsteps; j++){
	double cut = (double)j/(double)nsteps;
	//cout << "h in signal case: " << h.at(i,0) << ", cut: " << cut << endl;
	if(h.at(i,0) > cut) sig_sel[j] += EventWeight.at(i,0);
      }
    }
    else if(y.at(i,0)==0){
      bkg_tot += EventWeight.at(i,0);

      // Scan through output values
      for(int j=0; j<nsteps; j++){
	double cut = (double)j/(double)nsteps;
	//cout << "h in bgk case: " << h.at(i,0) << ", cut: " << cut << endl;
	if(h.at(i,0) >  cut) bkg_sel[j] += EventWeight.at(i,0);
      }
      
    }
  }
  
  for(int i=0; i<nsteps; i++){
    sig_sel[i] = sig_sel[i] / sig_tot;
    bkg_sel[i] = 1. - (bkg_sel[i] / bkg_tot);
    //if(i%10==0) cout << "sig eff: " << sig_sel[i] << ", bkg eff: " << bkg_sel[i] << endl;
  }
  
  TGraph* ROC = new TGraph(nsteps, &sig_sel[0], &bkg_sel[0]);
  ROC->GetXaxis()->SetRangeUser(0.9,1.1);
  ROC->GetXaxis()->SetTitle("Signal efficiency");
  ROC->GetYaxis()->SetRangeUser(9E-1,1.1);
  ROC->GetYaxis()->SetTitle("Background rejection");
  ROC->SetTitle("");
  ROC->SetLineWidth(2);

  unique_ptr<TCanvas> c;
  c.reset(new TCanvas());
  ROC->Draw("APL");
  //c->SetLogy();

  c->SaveAs("Output/" + plotname + "_" + tag + ".eps");

}
