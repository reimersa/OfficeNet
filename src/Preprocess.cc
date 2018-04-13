#include<iostream>
#include<armadillo>
#include<math.h>
#include<TString.h>
#include<TFile.h>
#include<TTreeReader.h>
#include<TTree.h>
#include<TBranch.h>
#include<TTreeReaderValue.h>
#include<TTreeReaderArray.h>
#include<vector>

#include "../include/Preprocess.h"

using namespace std;
using namespace arma;

void Preprocess(TString filepath, vector<TString> varnames, TString outfilepath){

  // Get the file
  //unique_ptr<TFile> infile;
  //infile.reset(new TFile(filepath, "READ"));
  TFile* infile = new TFile(filepath, "READ");
  
  // Get AnalysisTree 
  TTreeReader reader("AnalysisTree", infile);
  vector<TTreeReaderValue<double>> varvector;

  // Get variables and add them to a vector of readers
  for(unsigned int i=0; i<varnames.size(); i++){
    TTreeReaderValue<double> var(reader, varnames[i]);
    varvector.emplace_back(var);
  }

  // Define placeholders for final variables
  vector<double> vars;
  vector<vector<double>> arrs;
  for(unsigned int i=0; i<varnames.size(); i++) vars.emplace_back(0.);

  // Declare tree and link branches to values in 'vars'
  TTree tree("Input","Input variables");
  for(unsigned int i=0; i<varnames.size(); i++){
    TString name = varnames[i];
    TString nametype = varnames[i] + "/F";
    tree.Branch(name, &vars[i], nametype);
  }

  // Loop over events: fill variables with correct values and update branch
  while(reader.Next()){
    for(unsigned int i=0; i<varnames.size(); i++){
      vars[i] = *(varvector[i]);
    }

    // Fill branch
    tree.Fill();
  }

  // Close input file
  infile->Close();

  // Create output file
  unique_ptr<TFile> outfile;
  outfile.reset(new TFile(outfilepath, "RECREATE"));

  // Write branch
  outfile->cd();
  tree.Write();
  outfile->Close();






  
}
