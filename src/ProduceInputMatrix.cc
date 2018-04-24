#include<TString.h>
#include<TFile.h>
#include<TTree.h>
#include<iostream>
#include<armadillo>

#include "../include/ProduceInputMatrix.h"

using namespace std;
using namespace arma;

//vector<Mat<double>> ProduceInputMatrix(){
void ProduceInputMatrix(int nevt, double frac_train, double frac_test, double frac_cv){

  // Get pre-processed input rootfiles
  TFile* in_sig = new TFile("../data/Input_LQtoTMuM900.root");
  TFile* in_bkg = new TFile("../data/Input_TTbar.root");

  // Create trees for each file
  TTree* tree_sig = (TTree*)in_sig->Get("AnalysisTree");
  TTree* tree_bkg = (TTree*)in_bkg->Get("AnalysisTree");

  // Declare variables to be read out
  double ST=0., STLep=0., weight=0.;
  int NBJets_int = 0;

  // Set up raw return matrix
  Mat<double> X, X_test, X_cv, y, y_test, y_cv, EventWeight, EventWeight_test, EventWeight_cv;

  // Handle signal tree
  tree_sig->SetBranchAddress("ST", &ST);
  tree_sig->SetBranchAddress("STLep", &STLep);
  tree_sig->SetBranchAddress("NBJets", &NBJets_int);
  tree_sig->SetBranchAddress("EventWeight", &weight);

  int nevt_sig = tree_sig->GetEntries();
  if(nevt != -1) nevt_sig = nevt;
  int nevt_bkg = tree_bkg->GetEntries();
  if(nevt != -1) nevt_bkg = nevt;
  if(nevt_sig > tree_sig->GetEntries()) throw runtime_error("In ProduceInputMatrix(): You want to run over more signal events than there are stored in the tree.");
  if(nevt_bkg > tree_bkg->GetEntries()) throw runtime_error("In ProduceInputMatrix(): You want to run over more background events than there are stored in the tree.");
  
  // Loop over signal tree
  for(unsigned int i=0; i<tree_sig->GetEntries(); i++){
    tree_sig->GetEntry(i);
    if(i%1000==0) cout << "At sig event no. " << i << endl;

    // Normalize the features

    //ST in signal has mean 2000, width 500
    ST = (ST - 2000.) / 500.;

    //STLep: mean 900, width 300
    STLep = (STLep - 900) / 300;

    //NBJets: mean 5, width 1.7
    double NBJets = (NBJets_int - 5.) / 1.7;
    
    Mat<double> thisev = {1., ST, STLep, NBJets};
    Mat<double> y_init = {1.};
    Mat<double> weight_init = {weight};
    //Mat<double> weight_init = {1.};
    if(i==0){
      X = thisev;
      y = y_init;
      EventWeight = weight_init;
    }
    else{
      if(i < frac_train*nevt_sig){
	X.insert_rows(0, thisev);
	y.insert_rows(0, y_init);
	EventWeight.insert_rows(0, weight_init);
      }
      else if(i < (frac_train + frac_test) * nevt_sig){
	X_test.insert_rows(0, thisev);
	y_test.insert_rows(0, y_init);
	EventWeight_test.insert_rows(0, weight_init);
      }
      else if(i < (frac_train + frac_test + frac_cv) * nevt_sig){
	X_cv.insert_rows(0, thisev);
	y_cv.insert_rows(0, y_init);
	EventWeight_cv.insert_rows(0, weight_init);
      }
      else break;
    }
  }

  
  // Loop over background tree
  for(unsigned int i=0; i<tree_bkg->GetEntries(); i++){
    tree_bkg->GetEntry(i);
    if(i%1000==0) cout << "At bkg event no. " << i << endl;


    // Handle background tree
    tree_bkg->SetBranchAddress("ST", &ST);
    tree_bkg->SetBranchAddress("STLep", &STLep);
    tree_bkg->SetBranchAddress("NBJets", &NBJets_int);
    tree_bkg->SetBranchAddress("EventWeight", &weight);

    
    // Normalize the features

    //ST in signal has mean 2000, width 500
    ST = (ST - 2000.) / 500.;

    //STLep: mean 900, width 300
    STLep = (STLep - 900) / 300;

    //NBJets: mean 5, width 1.7
    double NBJets = (NBJets_int - 5.) / 1.7;
    
    Mat<double> thisev = {1., ST, STLep, NBJets};
    Mat<double> y_init = {0.};
    Mat<double> weight_init = {weight};
    //Mat<double> weight_init = {1.};
  
    if(i < frac_train*nevt_bkg){
      X.insert_rows(0, thisev);
      y.insert_rows(0, y_init);
      EventWeight.insert_rows(0, weight_init);
    }
    else if(i < (frac_train + frac_test) * nevt_bkg){
      X_test.insert_rows(0, thisev);
      y_test.insert_rows(0, y_init);
      EventWeight_test.insert_rows(0, weight_init);
    }
    else if(i < (frac_train + frac_test + frac_cv) * nevt_bkg){
      X_cv.insert_rows(0, thisev);
      y_cv.insert_rows(0, y_init);
      EventWeight_cv.insert_rows(0, weight_init);
    }
    else break;
  }

  X.save("../data/X.bin");
  y.save("../data/y.bin");
  EventWeight.save("../data/EventWeight.bin");
  X_test.save("../data/X_test.bin");
  y_test.save("../data/y_test.bin");
  EventWeight_test.save("../data/EventWeight_test.bin");
  X_cv.save("../data/X_cv.bin");
  y_cv.save("../data/y_cv.bin");
  EventWeight_cv.save("../data/EventWeight_cv.bin");

  //vector<Mat<double>> ret = {X, y};
  //return ret;
}
