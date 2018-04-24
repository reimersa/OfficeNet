#include<TString.h>
#include<TH1D.h>
#include<TStyle.h>
#include<TCanvas.h>
#include<TLegend.h>
#include<armadillo>
#include<iostream>

#include "../include/PlotOutput1d.h"

using namespace std;
using namespace arma;

void PlotOutput1d(Mat<double> h, Mat<double> y, Mat<double> EventWeight, double frac_low, double frac_high, TString plotname, TString tag){

  TH1D* output_sig = new TH1D("output_sig", "OfficeNet output;NN output;Events", 40, 0, 1);
  TH1D* output_bkg = new TH1D("output_bkg", "OfficeNet output;NN output;Events", 40, 0, 1);

  // Find out number of signal/background examples
  int n_sig = 0, n_bkg = 0;
  for(unsigned int i=0; i<h.n_rows; i++){
    if(y.at(i,0) == 1) n_sig++;
    else n_bkg++;
  }


  // Always signal at the bottom of y, h, ...
  for(unsigned int i=0; i<h.n_rows; i++){
    if(!(i >= n_bkg*frac_low && i < n_bkg*frac_high) && !(i >= n_bkg + n_sig*frac_low && i < n_bkg + n_sig*frac_high)) continue;
    
    if(y.at(i,0) == 1){
      output_sig->Fill(h.at(i,0), EventWeight.at(i,0));
    }
    else if(y.at(i,0) == 0){
      output_bkg->Fill(h.at(i,0), EventWeight.at(i,0));
    }
  }

  double maximum = max(output_sig->GetBinContent(output_sig->GetMaximumBin()), output_bkg->GetBinContent(output_bkg->GetMaximumBin()));
  output_sig->GetYaxis()->SetRangeUser(1E-3,maximum*1.2);

  output_sig->SetLineColor(kRed+1);
  output_sig->SetMarkerColor(kRed+1);
  output_sig->SetLineWidth(2);
  output_sig->SetTitle("OfficeNet output for sample: " + tag);
  output_bkg->SetLineColor(kBlue+1);
  output_bkg->SetMarkerColor(kBlue+1);
  output_bkg->SetLineWidth(2);

  unique_ptr<TLegend> leg;
  leg.reset(new TLegend(0.5,0.6,0.7,0.8));
  leg->SetBorderSize(0);
  leg->SetFillStyle(0);
  leg->AddEntry(output_sig, "Signal events", "L");
  leg->AddEntry(output_bkg, "Background events", "L");
  
  unique_ptr<TCanvas> c;
  c.reset(new TCanvas());
  output_sig->Draw("E");
  output_sig->Draw("HIST SAME");
  output_bkg->Draw("E SAME");
  output_bkg->Draw("HIST SAME");
  leg->Draw();
  gStyle->SetOptStat(0);
  c->SetLogy();
  c->SaveAs("Output/" + plotname + "_" + tag + ".eps");


  delete output_sig;
  delete output_bkg;
}
