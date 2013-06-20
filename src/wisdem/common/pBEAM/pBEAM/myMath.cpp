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

#include "myMath.h"

using namespace std;

extern "C"{

    
    int dsbgv_(char *jobz, char *uplo, int *n, int *ka, 
               int *kb, double *ab, int *ldab, double *bb, int *
               ldbb, double *w, double *z__, int *ldz, double *work, 
               int *info);
    
    int dpbsv_(char *uplo, int *n, int *kd, int *
               nrhs, double *ab, int *ldab, double *b, int *ldb, 
               int *info);
    
    int dsymv_(char *uplo, int *n, double *alpha, 
               double *a, int *lda, double *x, int *incx, double 
               *beta, double *y, int *incy);
    
    int dgesv_(int *n, int *nrhs, double *a, int *lda,
               int *ipiv, double *b, int *ldb, int *info);
    
    
}


namespace myMath {

    // MARK: ------------- EIGENVALUES --------------

        
    // solves generalized eigenvalue problem Ax = lambda * Bx
    int generalizedEigenvalues(bool cmpVec, const Matrix &A, const Matrix &B, int superDiag, Vector &eig, Matrix &eig_vec){
        
        int N = (int) A.size1();
         
        char jobz = 'N'; // compute eigenvalues only
        if (cmpVec) {
            jobz = 'V'; // compute eigenvalues and eigenvectors
        }

        char uplo = 'U'; // symmetric matrix stored in upper triangular (actually I've written full matrix but for efficiency could be rewritten to only store in upper tri)
        
        int ka = superDiag;
        int lda = ka + 1;
        int ldz = N;
        
        double *AB = (double *) malloc( lda*N*sizeof(double) );
        double *BB = (double *) malloc( lda*N*sizeof(double) );
        double *work = (double *) malloc( 3*N*sizeof(double) );
        double *vec = NULL;
        if (cmpVec) {
            vec = (double *) malloc( ldz*N*sizeof(double) );
            eig_vec.resize(ldz, N);
        }
        
        
        // zero out all entries
        for (int i = 0; i < lda*N; i++) {
            AB[i] = 0.0;
            BB[i] = 0.0;
        }
        
        // from fortran LAPACK doc: if UPLO = 'U', AB(ka+1+i-j,j) = A(i,j) for max(1,j-ka)<=i<=j;
        // but fortran uses row major order so A_c[i][j] = A_f[j*lda + i] and also need to adjust
        // for 0 based index in c and 1 based in fortran
        for (int j = 0; j < N; j++) {
            for (int i = max(0, j-ka); i <= j; i++) {
                AB[j*lda + (ka + i - j)] = A(i, j);
                BB[j*lda + (ka + i - j)] = B(i, j);
            }
        }
        
        int info;
        
        double *eig_data = &(eig.data()[0]);
        
        dsbgv_(&jobz, &uplo, &N, &ka, &ka, AB, &lda, BB, &lda, eig_data, vec, &ldz, work, &info);
        
        free(AB);
        free(BB);
        free(work);
                
        if (cmpVec) {
            // unpack eigenvectors
            // fortran uses row major order so A_c[i][j] = A_f[j*lda + i]
            for (int j = 0; j < N; j++) {
                for (int i = 0; i < ldz; i++) {
                    eig_vec(i, j) = vec[j*ldz + i];                                
                }
            }
            free(vec);
        }
        
        
        return info;
        
    }

    int generalizedEigenvalues(const Matrix &A, const Matrix &B, int superDiag, Vector &eig){
        Matrix empty(0, 0);
        return generalizedEigenvalues(false, A, B, superDiag, eig, empty);
    }
    
    int generalizedEigenvalues(const Matrix &A, const Matrix &B, int superDiag, Vector &eig, Matrix &eig_vec){
        return generalizedEigenvalues(true, A, B, superDiag, eig, eig_vec);
    }


    // -------------------------------------



    // MARK: -------------- LINEAR SYSTEM SOLVER -----------------


    int solveSPDBLinearSystem(const Matrix &A, const Vector &b, int superDiag, Vector &x){
        
        int N = (int) A.size1();
        
        char uplo = 'U'; // symmetric matrix stored in upper triangular
        int kd = superDiag;
        int lda = kd + 1;
        int rhs = 1;
        
        double *AB;
        AB = (double *) malloc( lda * N * sizeof(double) );
        
        // zero out all entries
        for (int i = 0; i < lda*N; i++) {
            AB[i] = 0.0;
        }
        
        // from fortran LAPACK doc: if UPLO = 'U', AB(ka+1+i-j,j) = A(i,j) for max(1,j-ka)<=i<=j;
        // but fortran uses row major order so A_c[i][j] = A_f[j*lda + i] and also need to adjust
        // for 0 based index in c and 1 based in fortran
        for (int j = 0; j < N; j++) {
            for (int i = max(0, j-kd); i <= j; i++) {
                AB[j*lda + (kd + i - j)] = A(i,j);
            }
        }
        
        for (int i = 0; i < N; i++) {
            x[i] = b[i]; // copy over because lapack returns solution in place
        }
        
        int info;
        
        double *x_data = &(x.data()[0]);
        
        dpbsv_(&uplo, &N, &kd, &rhs, AB, &lda, x_data, &N, &info);
        
        free(AB);
        
        return info;
        
    }

    //TODO: add unit test for this
    int solveLinearSystem(const Matrix &A, const Vector &b, Vector &x){
        
        int N = (int) A.size1();
        
        int lda = N;
        int ldb = N;
        int rhs = 1;
        
        double *AA;
        AA = (double *) malloc( lda * N * sizeof(double) );
        
        // zero out all entries
        for (int i = 0; i < lda*N; i++) {
            AA[i] = 0.0;
        }
        
        // fortran uses row major order so A_c[i][j] = A_f[j*lda + i]
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                AA[j*lda + i] = A(i,j);
            }
        }
        
        for (int i = 0; i < N; i++) {
            x[i] = b[i]; // copy over because lapack returns solution in place
        }
        
        int info;
        
        double *x_data = &(x.data()[0]);
        int *iPivot = (int *) malloc( N * sizeof(int) );
        
        dgesv_(&N, &rhs, AA, &lda, iPivot, x_data, &ldb, &info);
        
        free(AA);
        free(iPivot);
        
        return info;
        
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
    


    // MARK: -------------- UBLAS VECTOR/MATRIX CONVENIENCE METHODS --------------
        

    void vectorFromArray(double x[], Vector &v){

        for (int i = 0; i < v.size(); i++) {
            v(i) = x[i];
        }
        
    }

  

// -------------- MATRIX VECTOR MULTIPLY -----------------

//// A is symmetric
//int myMath::matrixVectortorMultiply(Matrix &A, Vector &x, Vector &y){
//    
//    int N = (int) A.size();
//    
//    char uplo = 'U'; // symmetric matrix stored in upper triangular
//    double alpha = 1.0;
//    double beta = 0.0;
//    int lda = N;
//    int incx = 1;
//    int incy = 1;
//    
//    
//    double *AA;
//    AA = (double *) malloc( lda * N * sizeof(double) );
//    
//    // fortran uses row major order so A_c[i][j] = A_f[j*lda + i] and also need to adjust
//    // for 0 based index in c and 1 based in fortran
//    for (int j = 0; j < N; j++) {
//        for (int i = 0; i < N; i++) {
//            AA[j*lda + i] = A[i][j];
//        }
//    }
//    
//    
//    int result = dsymv_(&uplo, &N, &alpha, AA, &lda, x.data(), &incx, &beta, y.data(), &incy);
//    
//    free(AA);
//    
//    
//    return result;
//    
//    
//}



// -------------------------------------------------------


}