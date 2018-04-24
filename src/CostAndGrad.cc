#include<iostream>
#include<armadillo>
#include <iomanip>

#include "../include/CostAndGrad.h"
#include "../include/SigmoidGradient.h"
#include "../include/Sigmoid.h"
#include "../include/Prediction.h"

using namespace std;
using namespace arma;

double Cost(Mat<double> X, Mat<double> EventWeight, Mat<double> Theta1, Mat<double> Theta2, Mat<double> Theta3, Mat<double> y, double lambda){

  // Get predicted values
  Mat<double> h = Prediction(Theta1, Theta2, Theta3, X);

  // Compare prediction and truth to build the cost
  Mat<double> unity = ones<mat>(y.n_rows,1);
  Mat<double> unregmat = (- (y % log(h)) - ((unity - y) % (log(unity - h)))) % EventWeight;
  double cost = accu(unregmat) / X.n_rows;

  Mat<double> Th1unreg = Theta1.submat(span(0,Theta1.n_rows-1), span(1,Theta1.n_cols-1));
  Mat<double> Th2unreg = Theta2.submat(span(0,Theta2.n_rows-1), span(1,Theta2.n_cols-1));
  Mat<double> Th3unreg = Theta3.submat(span(0,Theta3.n_rows-1), span(1,Theta3.n_cols-1));
  // Add regularization
  double reg = ( accu(Th1unreg % Th1unreg) + accu(Th2unreg % Th2unreg) + accu(Th3unreg % Th3unreg)) * lambda/(2*X.n_rows);

  cost += reg;
  return cost;
}

//This is backpropagation
vector<Mat<double>> Gradient(Mat<double> Theta1, Mat<double> Theta2, Mat<double> Theta3, Mat<double> X, Mat<double> y, Mat<double> EventWeight, double lambda){

  bool debug = false;

  // Compute z and a for all layers first
  // ------------------------------------
  
  Mat<double> z2 = Theta1*X.t();
  if(debug){
    //cout << "z2: " << endl << z2 << endl;
    cout << "size of z2: " << size(z2) << endl;
  }

  // Compute output of this layer (a)
  Mat<double> a2 = Sigmoid(z2);
  Mat<double> insert2(1,a2.n_cols,fill::ones);
  a2.insert_rows(0, insert2);
  
  //Transpose a to have the same format as for X
  a2 = a2.t();
  if(debug){
    //cout << "a2: " << endl << a2 << endl;
    cout << "size of a2: " << size(a2) << endl;
  }

  // Second hidden layer
  // -------------------

  // Compute "inputs" for this layer (z)
  Mat<double> z3 = Theta2*a2.t();
  if(debug){
    //cout << "z3: " << endl << z3 << endl;
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
  Mat<double> d4 = (h - y) % EventWeight;

  // Second hidden layer (3)
  // -----------------------
  z3.insert_rows(0, insert3);
  if(debug){
    cout << "size d4: " << size(d4) << ", Theta3: " << size(Theta3) << ", g(z3): " << size(SigmoidGradient(z3)) << endl; 
  }
  Mat<double> d3 = (Theta3.t() * d4.t()) % SigmoidGradient(z3);

  //remove bias unit contribution
  d3 = d3.submat(span(1,d3.n_rows-1), span(0,d3.n_cols-1));
  d3 = d3.t();
  if(debug){
    //cout << "d3: " << endl << d3 << endl << "size of d3: " << size(d3) << endl;
  }

  // First hidden layer (2)
  // ----------------------
  z2.insert_rows(0, insert2);
  if(debug){
    cout << "size d3: " << size(d3) << ", Theta2: " << size(Theta2) << ", g(z2): " << size(SigmoidGradient(z2)) << endl; 
  }
  Mat<double> d2 = (Theta2.t() * d3.t()) % SigmoidGradient(z2);

    //remove bias unit contribution
  d2 = d2.submat(span(1,d2.n_rows-1), span(0,d2.n_cols-1));
  d2 = d2.t();
  if(debug){
    //cout << "d2: " << endl << d2 << endl << "size of d2: " << size(d2) << endl;
  }


  // Accumulate gradient from deltas
  // -------------------------------

  //D3 = D3 + d4*a3
  if(debug){
    cout << "size d4: " << size(d4) << ", size a3: " << size(a3) << ", size Theta3: " << size(Theta3) << ", number of training examples: " << X.n_rows << endl;
  }
  Mat<double> D3 = (d4.t() * a3) / X.n_rows;
  if(debug){
    //cout << "D3: " << endl << D3 << endl << "size D3: " << size(D3) << ", size Theta3: " << size(Theta3) << endl;
  }

  
  //D2 = D2 + d3*a2
  if(debug){
    cout << "size d3: " << size(d3) << ", size a2: " << size(a2) << ", size Theta2: " << size(Theta2) << ", number of training examples: " << X.n_rows << endl;
  }
  Mat<double> D2 =(d3.t() * a2) / X.n_rows;
  if(debug){
    //cout << "D2: " << endl << D2 << endl << "size D2: " << size(D2) << ", size Theta2: " << size(Theta2) << endl;
  }

  
  //D1 = D1 + d2*a1
  if(debug){
    cout << "size d2: " << size(d2) << ", size X: " << size(X) << ", size Theta1: " << size(Theta1) << ", number of training examples: " << X.n_rows << endl;
  }
  Mat<double> D1 = (d2.t() * X) / X.n_rows;
  if(debug){
    //cout << "D1: " << endl << D1 << endl << "size D1: " << size(D1) << ", size Theta1: " << size(Theta1) << endl;
  }

  
  // Regularization
  // --------------

  // Remove bias weights
  Mat<double> Theta1_forreg = Theta1.submat(span(0,Theta1.n_rows-1), span(1,Theta1.n_cols-1));
  Mat<double> Theta2_forreg = Theta2.submat(span(0,Theta2.n_rows-1), span(1,Theta2.n_cols-1));
  Mat<double> Theta3_forreg = Theta3.submat(span(0,Theta3.n_rows-1), span(1,Theta3.n_cols-1));

  // Insert zeros to not regularize the bias weights
  Theta1_forreg.insert_cols(0,1,true);
  Theta2_forreg.insert_cols(0,1,true);
  Theta3_forreg.insert_cols(0,1,true);
  
  D1 = D1 + (Theta1_forreg * lambda / X.n_rows);
  D2 = D2 + (Theta2_forreg * lambda / X.n_rows);
  D3 = D3 + (Theta3_forreg * lambda / X.n_rows);

  vector<Mat<double>> gradients;
  gradients.emplace_back(D1);
  gradients.emplace_back(D2);
  gradients.emplace_back(D3);


  // Gradient checking
  // -----------------

  // D3 (1,1)

  if(debug){
    double eps = 1E-5;
    Mat<double> numgrad_D3(D3.n_rows, D3.n_cols, fill::zeros);
    for(unsigned int i=0; i<Theta3.n_rows; i++){
      for(unsigned int j=0; j<Theta3.n_cols; j++){
	Mat<double> epsmat_D3(Theta3.n_rows, Theta3.n_cols, fill::zeros);
	epsmat_D3.at(i,j) = eps;
	double cost_D3_up = Cost(X, EventWeight, Theta1, Theta2, Theta3 + epsmat_D3, y, lambda);
	double cost_D3_down = Cost(X, EventWeight, Theta1, Theta2, Theta3 - epsmat_D3, y, lambda);
      
	//cout << "cost_D3_up: " << cost_D3_up << ", down: " << cost_D3_down << ", diff/(2e): " << (cost_D3_up - cost_D3_down) / (2*eps) << endl;
  
	numgrad_D3.at(i,j) = (cost_D3_up - cost_D3_down) / (2*eps);
      }
    }

    cout << "Gradient determined by function: " << endl << D3 << endl << "Numerical gradient: " << endl << numgrad_D3 << endl;
  }

  
  return gradients;
}
