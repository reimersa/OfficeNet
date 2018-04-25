#include<iostream>
#include<math.h>
#include<TString.h>
#include<TFile.h>
#include<TTreeReader.h>
#include<TTree.h>
#include<TBranch.h>
#include<TTreeReaderValue.h>
#include<TTreeReaderArray.h>
#include<vector>

#include "../include/parameters.h"


using namespace std;

void Preprocess(TString filepath, TString outfilepath, bool debug);

int main(){
  bool debug = true;
  if(debug) cout << "Hello from Preprocess()!" << endl;
  TString inbase = "/nfs/dust/cms/user/reimersa/LQToTopMu/Run2_80X_v3/Optimization/35867fb_DibosonNLO/ForML/uhh2.AnalysisModuleRunner.MC.";
  TString outbase = "/nfs/dust/cms/user/reimersa/LQToTopMu/Run2_80X_v3/OfficeNet/data/Input_";
  if(debug) cout << "Defined in- and outbase." << endl;

  vector<TString> samplenames = {"TTbar", "DYJets", "LQtoTMuM200", "LQtoTMuM300", "LQtoTMuM400", "LQtoTMuM500", "LQtoTMuM600", "LQtoTMuM700", "LQtoTMuM800", "LQtoTMuM900", "LQtoTMuM1000", "LQtoTMuM1200", "LQtoTMuM1400", "LQtoTMuM1700", "LQtoTMuM2000"};

  for(unsigned int i=0; i<samplenames.size(); i++) Preprocess(inbase+samplenames[i]+".root", outbase+samplenames[i]+".root", debug);

  Preprocess(inbase+"TTbar.root", outbase+"TTbar.root", debug);
  Preprocess(inbase+"LQtoTMuM900.root", outbase+"LQtoTMuM900.root", debug);
  if(debug) cout << "After Preprocess." << endl;
  return 0;

}

void Preprocess(TString filepath, TString outfilepath, bool debug){
  // Get the file
  //unique_ptr<TFile> infile;
  //infile.reset(new TFile(filepath, "READ"));
  TFile* infile = new TFile(filepath, "READ");
  
  // Get AnalysisTree 
  TTree* tree = (TTree*)infile->Get("AnalysisTree");

  // Get variables
  tree->SetBranchStatus("*",0);
  for(unsigned int i=0; i<varnames.size(); i++) tree->SetBranchStatus(varnames[i],1); 

  // Create output file
  TFile* outfile = new TFile(outfilepath, "RECREATE");

  // Write branch
  TTree* newtree = (TTree*)tree->CloneTree();
  newtree->Write();
  outfile->Close();
  if(debug) cout << "Closed outfile." << endl;



  delete outfile;
  delete infile;
  
}
