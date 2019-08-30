//
//  testMath.cpp
//  pbeam
//
//  Created by Andrew Ning on 2/6/12.
//  Copyright (c) 2012 NREL. All rights reserved.
//

#include "catch.hpp"

#include "myMath.h"

using namespace myMath;

TEST_CASE( "eig" ){
    
  //using namespace boost;
    
    int N = 3;
    
    Matrix A(N, N);
    A.setZero();
    
    A(0, 0) = 1.0;   A(0, 1) = 2.0;
    A(1, 0) = 2.0;   A(1, 1) = 5.0;   A(1, 2) = 3.0;
                     A(2, 1) = 3.0;   A(2, 2) = 7.0;
    
    Matrix B(N, N);
    B.setZero();
    
    B(0, 0) = 4.0;   B(0, 1) = 1.1;
    B(1, 0) = 1.1;   B(1, 1) = 2.2;   B(1, 2) = 3.3;
                     B(2, 1) = 3.3;   B(2, 2) = 9.0;
    

    Vector eig(N);

    int result = generalizedEigenvalues(A, B, eig);
    
    REQUIRE(result == 0);
    
    double tol = 1e-8;
    REQUIRE(eig(0) == Approx(-0.024733114462918).epsilon(tol));
    REQUIRE(eig(1) == Approx(0.771999972563689).epsilon(tol));
    REQUIRE(eig(2) == Approx(4.232127081293167).epsilon(tol));
    
    
}






TEST_CASE( "linear_solve" ){
    

    Matrix A(3, 3);
    A.setZero();
    
    A(0, 0) = 4.0;   A(0, 1) = 1.1;
    A(1, 0) = 1.1;   A(1, 1) = 2.2;   A(1, 2) = 3.3;
                     A(2, 1) = 3.3;   A(2, 2) = 9.0;

    Vector b(3);
    b(0) = 12.0;
    b(1) = -5.0;
    b(2) = 3.3;
    
    Vector x(3);
    
    int result = solveSPDBLinearSystem(A, b, x);
    
    REQUIRE(result == 0);
    
    double tol = 1e-8;
    REQUIRE(x(0) == Approx(6.803999999999998).epsilon(tol));
    REQUIRE(x(1) == Approx(-13.832727272727267).epsilon(tol));
    REQUIRE(x(2) == Approx(5.438666666666664).epsilon(tol));
    
    
}
