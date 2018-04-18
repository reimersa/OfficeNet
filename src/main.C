#include <iostream>
#include <armadillo>
#include <math.h>
#include <TString.h>
#include <vector>

#include "../include/Sigmoid.h"
#include "../include/ProduceInputMatrix.h"
#include "../include/Prediction.h"

using namespace std;
using namespace arma;

int main(){

  bool debug = true;

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
  
  Mat<double> X = InputMatrix(3);
  
  Prediction(Theta1, Theta2, Theta3, X);





  
}
