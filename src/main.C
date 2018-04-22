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

using namespace std;
using namespace arma;

int main(){

  bool debug = false;

  cout << "Hello from main()! " << endl;

  // Set up Theta1,2,3 randomly
  // --------------------------
  
  arma_rng::set_seed_random();
  double init_epsilon = 0.2;
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
  //ProduceInputMatrix();
  

  // Get training data
  // -----------------

  cout << "Getting training data..." << endl;
  Mat<double> X, y;
  if(X.load("../data/X.bin") && y.load("../data/y.bin")) cout << "Successfully loaded matrices X and y." << endl;
  else throw runtime_error("Error when loading input matrices X and y.");

  Mat<double> h = Prediction(Theta1, Theta2, Theta3, X);

  if(debug){
    cout << "X: " << endl << X << endl;
    cout << "y: " << endl << y << endl;
    cout << "h: " << endl << h << endl;
  }


  // Compute cost function
  double lambda = 0.0;
  double cost = Cost(X, Theta1, Theta2, Theta3, y, lambda);
  if(debug) cout << "Cost: " << cost << endl;
  
  if(debug){
    cout << "SigmoidGradient for [-999, 0., 999] should be [0, 0.25, 0]: " << SigmoidGradient(-999) << ", " << SigmoidGradient(0) << ", " << SigmoidGradient(999) << endl;
    Mat<double> test = {-999, 0., 999};
    cout << "SigmoidGradient on a matrix for [-999, 0., 999] should be [0, 0.25, 0]: " << SigmoidGradient(test) << endl;
  }
  
  if(debug){
    vector<Mat<double>> gradients = Gradient(Theta1, Theta2, Theta3, X, y, lambda);
    Mat<double> grad1 = gradients[0];
    Mat<double> grad2 = gradients[1];
    Mat<double> grad3 = gradients[2];
    cout << "Gradient 1: " << endl << grad1 << endl << "Gradient 2: " << endl << grad2 << endl << "Gradient 3: " << endl << grad3 << endl;
  }

  vector<Mat<double>> thetas = GradientDescent(Theta1, Theta2, Theta3, X, y, lambda, 1.0, 5000);

  
}

  

