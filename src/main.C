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

using namespace std;
using namespace arma;

int main(){

  bool debug = false;

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


  // Get training data
  // -----------------

  //test comment master
  
  vector<Mat<double>> inputs = InputMatrix();
  Mat<double> X = inputs[0];
  Mat<double> y = inputs[1];
  
  Mat<double> h = Prediction(Theta1, Theta2, Theta3, X);

  if(debug){
    cout << "X: " << endl << X << endl;
    cout << "y: " << endl << y << endl;
    cout << "h: " << endl << h << endl;
  }


  // Compute cost function
  double lambda = 0.1;
  double cost = Cost(X, Theta1, Theta2, Theta3, y, lambda);
  if(debug) cout << "Cost: " << cost << endl;
  
  if(debug){
    cout << "SigmoidGradient for [-999, 0., 999] should be [0, 0.25, 0]: " << SigmoidGradient(-999) << ", " << SigmoidGradient(0) << ", " << SigmoidGradient(999) << endl;
    Mat<double> test = {-999, 0., 999};
    cout << "SigmoidGradient on a matrix for [-999, 0., 999] should be [0, 0.25, 0]: " << SigmoidGradient(test) << endl;
  }

  Mat<double> test = Gradient(Theta1, Theta2, Theta3, X, y, lambda);
    
}
