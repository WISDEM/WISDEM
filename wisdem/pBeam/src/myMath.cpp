//
//  myMath.cpp
//  pbeam
//
//  Created by Andrew Ning on 2/4/12.
//  Copyright (c) 2012 NREL. All rights reserved.
//

//#include <Accelerate/Accelerate.h>
#include <cstdlib> // for malloc
#include <algorithm> // for max/min
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "myMath.h"

using namespace std;

namespace myMath {

  // MARK: ------------- EIGENVALUES --------------

        
  // solves generalized eigenvalue problem Ax = lambda * Bx
  int generalizedEigenvalues(bool cmpVec, const Matrix &A, const Matrix &B, Vector &eig, Matrix &eig_vec){

    int flag = cmpVec ? Eigen::ComputeEigenvectors : Eigen::EigenvaluesOnly;

    Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> es(A, B, flag);

    eig = es.eigenvalues().real();
    if (cmpVec) eig_vec = es.eigenvectors().real();
    return es.info(); // != Eigen::Success) abort();
   }

  int generalizedEigenvalues(const Matrix &A, const Matrix &B, Vector &eig){
    Matrix empty(0, 0);
    return generalizedEigenvalues(false, A, B, eig, empty);
  }
    
  int generalizedEigenvalues(const Matrix &A, const Matrix &B, Vector &eig, Matrix &eig_vec){
    return generalizedEigenvalues(true, A, B, eig, eig_vec);
  }


  // -------------------------------------



  // MARK: -------------- LINEAR SYSTEM SOLVER -----------------


  int solveSPDBLinearSystem(const Matrix &A, const Vector &b, Vector &x){
    //x = A.llt().solve(b);
    x = A.ldlt().solve(b);
    return 0;
  }

  //TODO: add unit test for this
  int solveLinearSystem(const Matrix &A, const Vector &b, Vector &x){
    x = A.householderQr().solve(b);
    return 0;
  }


  // -------------------------------------


  // MARK: ------------ INTEGRATION -----------------
  // TODO: add unit tests
    
  void cumtrapz(const Vector &f, const Vector &x, Vector &y){
        
    y(0) = 0.0;
        
    for (int i = 1; i < x.size(); i++) {
      y(i) = y(i-1) + 0.5*(f(i)+f(i-1)) * (x(i)-x(i-1));
    }
  }
    
  double trapz(const Vector &f, const Vector &x){
        
    int n = (int) x.size();
    Vector y(n);
    cumtrapz(f, x, y);
        
    return y(n-1);
  }
    

  void vectorFromArray(double x[], Vector &v){

    for (int i = 0; i < v.size(); i++) {
      v(i) = x[i];
    }
        
  }

}
