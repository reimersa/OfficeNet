#include<iostream>
#include<armadillo>

#include "../include/CostAndGrad.h"
#include "../include/SigmoidGradient.h"
#include "../include/Sigmoid.h"
#include "../include/Prediction.h"

using namespace std;
using namespace arma;

double Cost(Mat<double> X, Mat<double> Theta1, Mat<double> Theta2, Mat<double> Theta3, Mat<double> y, double lambda){

  // Get predicted values
  Mat<double> h = Prediction(Theta1, Theta2, Theta3, X);

  // Compare prediction and truth to build the cost
  Mat<double> unity = ones<mat>(y.n_rows,1);
  Mat<double> unregmat = - (y % log(h)) - ((unity - y) % (log(unity - h)));
  double cost = accu(unregmat);

  Mat<double> Th1unreg = Theta1.submat(span(0,Theta1.n_rows-1), span(1,Theta1.n_cols-1));
  Mat<double> Th2unreg = Theta2.submat(span(0,Theta2.n_rows-1), span(1,Theta2.n_cols-1));
  Mat<double> Th3unreg = Theta3.submat(span(0,Theta3.n_rows-1), span(1,Theta3.n_cols-1));
  // Add regularization
  double reg = accu(Th1unreg % Th1unreg) + accu(Th2unreg % Th2unreg) + accu(Th3unreg % Th3unreg);

  cost += reg;
  return cost;
}

//This is backpropagation
Mat<double> Gradient(Mat<double> Theta1, Mat<double> Theta2, Mat<double> Theta3, Mat<double> X, Mat<double> y, double lambda){

  bool debug = true;

  // Compute z and a for all layers first
  // ------------------------------------
  
  Mat<double> z2 = Theta1*X.t();
  if(debug){
    cout << "z2: " << endl << z2 << endl;
    cout << "size of z2: " << size(z2) << endl;
  }

  // Compute output of this layer (a)
  Mat<double> a2 = Sigmoid(z2);
  Mat<double> insert2(1,a2.n_cols,fill::ones);
  a2.insert_rows(0, insert2);
  
  //Transpose a to have the same format as for X
  a2 = a2.t();
  if(debug){
    cout << "a2: " << endl << a2 << endl;
    cout << "size of a2: " << size(a2) << endl;
  }

  // Second hidden layer
  // -------------------

  // Compute "inputs" for this layer (z)
  Mat<double> z3 = Theta2*a2.t();
  if(debug){
    cout << "z3: " << endl << z3 << endl;
    cout << "size of z3: " << size(z3) << endl;
  }

  // Compute output of this layer (a)
  Mat<double> a3 = Sigmoid(z3);
  Mat<double> insert3(1,a3.n_cols,fill::ones);
  a3.insert_rows(0, insert3);
  
  //Transpose a to have the same format as for X
  a3 = a3.t();
  
  Mat<double> h = Prediction(Theta1, Theta2, Theta3, X);

  // Output layer (4)
  // ----------------
  Mat<double> d4 = h - y;

  // Second hidden layer (3)
  // -----------------------
  z3.insert_rows(0, insert3);
  if(debug){
    cout << "size d4: " << size(d4) << ", Theta3: " << size(Theta3) << ", g(z3): " << size(SigmoidGradient(z3)) << endl; 
  }


  
  //just a dummy return so far...TODO!
  return z3;
}
