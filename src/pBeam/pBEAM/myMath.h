//
//  myMath.h
//  pbeam
//
//  Created by Andrew Ning on 2/4/12.
//  Copyright (c) 2012 NREL. All rights reserved.
//


#ifndef pbeam_myMath_h
#define pbeam_myMath_h

#include <Eigen/Dense>

typedef Eigen::MatrixXd Matrix;
typedef Eigen::VectorXd Vector;



/**
 This class contains only static methods useful for mathematical analysis and 
 defining matrix/vector data.
 
 **/


namespace myMath {
        
    
    /**
     Solves the generalized eigenvalue problem Ax = lambda Bx
     This particular solver assumes that A and B are both banded and symmetric.  
     In addition B must be positive definite.
     Because A and B are both symmetric only data in the upper triangular portion needs to be passed in.
     This is just a wrapper to a fortran call of LAPACK's dsbgv()
     
     Arguments:
     A - matrix on left hand side of generalized eigenvalue problem (must be size N x N)
     B - matrix on right hand side of generalized eigenvalue problem (must be size N x N)
     
     Out:
     eig - vector of eigenvalues sorted in ascending order (must be of length N)
     eig_vec (optional) - optionally compute associated eigenvectors (stored in columns)
     
     Returns:
     int - an integer describing exit condition.  0 is successful.  For other errors see dsbgv documentation.
     
     **/
    int generalizedEigenvalues(const Matrix &A, const Matrix &B, Vector &eig);
    int generalizedEigenvalues(const Matrix &A, const Matrix &B, Vector &eig, Matrix &eig_vec);
    
    /**
     Solves the linear system Ax = b
     The first assumes A is banded, symmetric, and positive definite.  (only upper triangular
     portion need be passed in).
     The second is for any general A.
          
     Arguments:
     A - matrix in linear system (N x N)
     b - right hand side vector (length N)

     Out:
     x - least squares solution to linear system (length N)
     
     Returns:
     int - an integer describing exit condition.  0 is successful.  For other errors see dpbsv or dgesv documentation.
     
     **/
    int solveSPDBLinearSystem(const Matrix &A, const Vector &b, Vector &x);
    int solveLinearSystem(const Matrix &A, const Vector &b, Vector &x);
    
    
    
    void cumtrapz(const Vector &f, const Vector &x, const Vector &y);
    double trapz(const Vector &f, const Vector &x);
    
    
    /**
     Populates vector with data from native array
     
     Arguments:
     x - source data.  must be same length as vector v
     
     Out:
     v - source data copied into vector container
     
     **/
    void vectorFromArray(double x[], Vector &v);
 
}


#endif
