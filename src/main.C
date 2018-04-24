#include <iostream>
#include <armadillo>
#include <math.h>
#include <TString.h>
#include <vector>

#include "../include/Sigmoid.h"
#include "../include/SigmoidGradient.h"
#include "../include/ProduceInputMatrix.h"
#include "../include/Prediction.h"
#include "../include/CostAndGrad.h"
#include "../include/GradientDescent.h"
#include "../include/PlotOutput1d.h"

using namespace std;
using namespace arma;

int main(){

  bool debug = false;
  double frac_train = 0.6;
  double frac_test = 0.2;
  double frac_cv = 0.2;
  if(frac_train + frac_test + frac_cv > 1) throw runtime_error("In main(): Fractions for training, testing, and cross validation in summation exceed 1.");

  cout << "Hello from main()! " << endl;

  // Set up Theta1,2,3 randomly
  // --------------------------
  
  arma_rng::set_seed_random();
  double init_epsilon = 0.12;
  Mat<double> Theta1(3,4,fill::randu), Theta2(3,4,fill::randu), Theta3(1,4,fill::randu);
  Theta1 = Theta1*2*init_epsilon-init_epsilon;
  Theta2 = Theta2*2*init_epsilon-init_epsilon;
  Theta3 = Theta3*2*init_epsilon-init_epsilon;
  
  if(debug){
    cout << "Randomly initialized Theta matrices: " << endl;
    cout << "Theta 1: " << endl << Theta1 << endl << "Theta 2: " << endl << Theta2 << endl << "Theta 3: " << endl << Theta3 << endl;
  }

  // Produce input matrices from training data
  // -----------------------------------------
  ProduceInputMatrix(5000, frac_train, frac_test, frac_cv);
  
  // Get training data
  // -----------------

  cout << "Getting training data..." << endl;
  Mat<double> X, X_test, X_cv, y, y_test, y_cv, EventWeight, EventWeight_test, EventWeight_cv;
  if(X.load("../data/X.bin") && y.load("../data/y.bin") && EventWeight.load("../data/EventWeight.bin")) cout << "Successfully loaded matrices X, y, and EventWeight." << endl;
  else throw runtime_error("Error when loading input matrices X, y, and EventWeight.");
  if(X_test.load("../data/X_test.bin") && y_test.load("../data/y_test.bin") && EventWeight_test.load("../data/EventWeight_test.bin")) cout << "Successfully loaded matrices X, y, and EventWeight." << endl;
  else throw runtime_error("Error when loading input matrices X_test, y_test, and EventWeight_test.");
  if(X_cv.load("../data/X_cv.bin") && y_cv.load("../data/y_cv.bin") && EventWeight_cv.load("../data/EventWeight_cv.bin")) cout << "Successfully loaded matrices X, y, and EventWeight." << endl;
  else throw runtime_error("Error when loading input matrices X_cv, y_cv, and EventWeight_cv.");


  if(debug){
    cout << "X: " << endl << X << endl;
    cout << "y: " << endl << y << endl;
    cout << "EventWeight: " << endl << EventWeight << endl;
    cout << "X_test: " << endl << X_test << endl;
    cout << "y_test: " << endl << y_test << endl;
    cout << "EventWeight_test: " << endl << EventWeight_test << endl;
    cout << "X_cv: " << endl << X_cv << endl;
    cout << "y_cv: " << endl << y_cv << endl;
    cout << "EventWeight_cv: " << endl << EventWeight_cv << endl;
  }
  
  // Compute cost function
  double lambda = 0.0;
  double cost = Cost(X, EventWeight, Theta1, Theta2, Theta3, y, lambda);
  if(debug) cout << "Cost: " << cost << endl;
  
  if(debug){
    cout << "SigmoidGradient for [-999, 0., 999] should be [0, 0.25, 0]: " << SigmoidGradient(-999) << ", " << SigmoidGradient(0) << ", " << SigmoidGradient(999) << endl;
    Mat<double> test = {-999, 0., 999};
    cout << "SigmoidGradient on a matrix for [-999, 0., 999] should be [0, 0.25, 0]: " << SigmoidGradient(test) << endl;
  }
  
  if(debug){
    vector<Mat<double>> gradients = Gradient(Theta1, Theta2, Theta3, X, y, EventWeight, lambda);
    Mat<double> grad1 = gradients[0];
    Mat<double> grad2 = gradients[1];
    Mat<double> grad3 = gradients[2];
    cout << "Gradient 1: " << endl << grad1 << endl << "Gradient 2: " << endl << grad2 << endl << "Gradient 3: " << endl << grad3 << endl;
  }


  // Train OfficeNet and store final Thetas
  // --------------------------------------

  
  vector<Mat<double>> thetas = GradientDescent(Theta1, Theta2, Theta3, X, X_test, X_cv, y, y_test, y_cv, EventWeight, EventWeight_test, EventWeight_cv, lambda, 30.0, 1000);

  thetas[0].save("../data/Theta1.bin");
  thetas[1].save("../data/Theta2.bin");
  thetas[2].save("../data/Theta3.bin");
  
  
  // Read trained Theta matrices
  // ---------------------------
  
  
  /*
  vector<Mat<double>> thetas;
  Mat<double> theta_tmp;
  theta_tmp.load("../data/Theta1.bin");
  thetas.emplace_back(theta_tmp);
  theta_tmp.load("../data/Theta2.bin");
  thetas.emplace_back(theta_tmp);
  theta_tmp.load("../data/Theta3.bin");
  thetas.emplace_back(theta_tmp);
  */
  
  Mat<double> h = Prediction(thetas[0], thetas[1], thetas[2], X);
  if(debug){
    double bkg_in = 0, sig_in = 0, bkg_right = 0, sig_right = 0;
    
    
    for(unsigned int i=0; i<y.n_rows; i++){
      if(y.at(i,0) == 0) bkg_in += EventWeight.at(i,0);
      else if(y.at(i,0) == 1) sig_in += EventWeight.at(i,0);
      else throw runtime_error("There are different labels in y than '0' or '1'.");
      
      if(h.at(i,0) < 0.5 && y.at(i,0) == 0) bkg_right += EventWeight.at(i,0);
      else if(h.at(i,0) >= 0.5 && y.at(i,0) == 1) sig_right += EventWeight.at(i,0);
    }
    
    cout << "Out of " << bkg_in << " background events, OfficeNet classified " << bkg_right/bkg_in * 100 << "% of the training events correctly. Number of correctly predicted events: " << bkg_right << endl;
    cout << "Out of " << sig_in << " signal events, OfficeNet classified " << sig_right/sig_in * 100 << "% of the training events correctly. Number of correctly predicted events: " << sig_right << endl;
  }

  PlotOutput1d(h, y, EventWeight, 0., frac_train, "NNOutput", "Training");
  PlotOutput1d(h, y, EventWeight, frac_train, frac_train+frac_test, "NNOutput", "Test");
  PlotOutput1d(h, y, EventWeight, frac_train+frac_test, 1., "NNOutput", "CV");
  //cout << "h: " << endl << h << endl;
  
}

  

