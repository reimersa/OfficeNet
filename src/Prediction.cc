#include<iostream>
#include<armadillo>
#include<TString.h>

#include "../include/Prediction.h"
#include "../include/Sigmoid.h"

using namespace std;
using namespace arma;

void Prediction(Mat<double> Theta1, Mat<double> Theta2, Mat<double> Theta3, Mat<double> X){

  bool debug = true;
  if(debug){
    cout << "Theta1: " << endl << Theta1 << endl;
    cout << "X: " << endl << X << endl;
  }

  // First hidden layer
  // -----------------

  // Compute "inputs" for this layer (z)
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
  if(debug){
    cout << "a3: " << endl << a3 << endl;
    cout << "size of a3: " << size(a3) << endl;
  }


  // Output layer
  // ------------

  // Compute "inputs" for this layer (z)
  Mat<double> z4 = Theta3*a3.t();
  if(debug){
    cout << "z4: " << endl << z4 << endl;
    cout << "size of z4: " << size(z4) << endl;
  }
  
  // Compute output of this layer (a)
  Mat<double> a4 = Sigmoid(z4);
  
  //Transpose a to have the same format as for X
  a4 = a4.t();
  if(debug){
    cout << "a4: " << endl << a4 << endl;
    cout << "size of a4: " << size(a4) << endl;
  }




}
