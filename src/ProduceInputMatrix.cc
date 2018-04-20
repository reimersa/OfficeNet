#include<TString.h>
#include<TFile.h>
#include<TTree.h>
#include<iostream>
#include<armadillo>

#include "../include/ProduceInputMatrix.h"

using namespace std;
using namespace arma;

vector<Mat<double>> InputMatrix(){

  // Get pre-processed input rootfiles
  TFile* in_sig = new TFile("../data/Input_LQtoTMuM900.root");
  TFile* in_bkg = new TFile("../data/Input_TTbar.root");

  // Create trees for each file
  TTree* tree_sig = (TTree*)in_sig->Get("AnalysisTree");
  TTree* tree_bkg = (TTree*)in_bkg->Get("AnalysisTree");

  // Declare variables to be read out
  double ST=0., STLep=0.;
  int NBJets_int = 0;

  // Set up raw return matrix
  Mat<double> X, y;

  // Handle signal tree
  tree_sig->SetBranchAddress("ST", &ST);
  tree_sig->SetBranchAddress("STLep", &STLep);
  tree_sig->SetBranchAddress("NBJets", &NBJets_int);

  // Loop over signal tree
  for(unsigned int i=0; i<200; i++){
    tree_sig->GetEntry(i);

    // Normalize the features

    //ST in signal has mean 2000, width 500
    ST = (ST - 2000.) / 500.;

    //STLep: mean 900, width 300
    STLep = (STLep - 900) / 300;

    //NBJets: mean 5, width 1.7
    double NBJets = (NBJets - 5.) / 1.7;
    
    Mat<double> thisev = {1., ST, STLep, NBJets};
    Mat<double> y_init = {1.};
    if(i==0){
      X = thisev;
      y = y_init;
    }
    else{
      X.insert_rows(0, thisev);
      y.insert_rows(0, y_init);
    }
  }

  
  // Loop over background tree
  for(unsigned int i=0; i<200; i++){
    tree_bkg->GetEntry(i);

    // Normalize the features

    //ST in signal has mean 2000, width 500
    ST = (ST - 2000.) / 500.;

    //STLep: mean 900, width 300
    STLep = (STLep - 900) / 300;

    //NBJets: mean 5, width 1.7
    double NBJets = (NBJets - 5.) / 1.7;
    
    Mat<double> thisev = {1., ST, STLep, NBJets};
    Mat<double> y_init = {0.};
    X.insert_rows(0, thisev);
    y.insert_rows(0, y_init);
  }


  vector<Mat<double>> ret = {X, y};
  return ret;
}