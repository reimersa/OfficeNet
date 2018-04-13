{

  //TFile* inf = new TFile("/home/arne/OfficeNet/data/uhh2.AnalysisModuleRunner.MC.QCDPt15to7000_pythia8_AK4CHS_Flat_RunBCD.root", "READ");
  TFile* inf = new TFile("/home/arne/OfficeNet/data/uhh2.AnalysisModuleRunner.MC.TTbar.root", "READ");

  //get branch
  TTree* tree = (TTree*)inf->Get("AnalysisTree");

  int weight = 0.;

  tree->SetBranchAddress("event", &weight);

  int nentries = tree->GetEntries();
  cout << "nentries: " << nentries << endl;

  for(int i=0; i<nentries; i++){
    tree->GetEntry(i);
    if(i%10000) cout << "weight for this event: " << weight << endl;
    
  }

  
}
