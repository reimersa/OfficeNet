#include <iostream>
#include <armadillo>
#include <math.h>
#include <TString.h>
#include <vector>

#include "../include/Sigmoid.h"

using namespace std;
using namespace arma;

int main(){

  bool debug = true;

  cout << "Hello from main()! " << endl;

  // Set up Theta1,2,3 randomly
  // --------------------------
  arma_rng::set_seed_random();
  Mat<double> Theta1(3,4,fill::randu), Theta2(3,4,fill::randu), Theta3(1,4,fill::randu);
  if(debug){
    cout << "Randomly initialized Theta matrices: " << endl;
    cout << "Theta 1: " << endl << Theta1 << endl << "Theta 2: " << endl << Theta2 << endl << "Theta 3: " << endl << Theta3 << endl;
  }
  
  // Test preprocessing of files
  //vector<TString> varnames = {"ST", "STLep", "NJets", "NBjets", "EventWeight"};
  //Preprocess("/nfs/dust/cms/user/reimersa/LQToTopMu/Run2_80X_v3/Optimization/35867fb_DibosonNLO/ForML/uhh2.AnalysisModuleRunner.MC.TTbar.root", varnames, "/nfs/dust/cms/user/reimersa/LQToTopMu/Run2_80X_v3/OfficeNet/data/Input_TTbar.root");






  
}
