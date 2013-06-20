//
//  testMath.cpp
//  pbeam
//
//  Created by Andrew Ning on 2/6/12.
//  Copyright (c) 2012 NREL. All rights reserved.
//

#define BOOST_TEST_DYN_LINK
#ifdef STAND_ALONE
#define BOOST_TEST_MODULE pbeamTests
#endif
#include <boost/test/unit_test.hpp>

#include "myMath.h"

BOOST_AUTO_TEST_SUITE( math )


using namespace myMath;

BOOST_AUTO_TEST_CASE( eig ){
    
    using namespace boost;
    
    int N = 3;
    
    Matrix A(N, N);
    
    A(0, 0) = 1.0;   A(0, 1) = 2.0;
    A(1, 0) = 2.0;   A(1, 1) = 5.0;   A(1, 2) = 3.0;
                     A(2, 1) = 3.0;   A(2, 2) = 7.0;
    
    Matrix B(N, N);
    
    B(0, 0) = 4.0;   B(0, 1) = 1.1;
    B(1, 0) = 1.1;   B(1, 1) = 2.2;   B(1, 2) = 3.3;
                     B(2, 1) = 3.3;   B(2, 2) = 9.0;
    

    Vector eig(N);

    int result = generalizedEigenvalues(A, B, 1, eig);
    
    BOOST_CHECK(result == 0);
    
    double tol = 1e-8;
    BOOST_CHECK_CLOSE(eig(0), -0.024733114462918, tol);
    BOOST_CHECK_CLOSE(eig(1), 0.771999972563689, tol);
    BOOST_CHECK_CLOSE(eig(2), 4.232127081293167, tol);
    
    
}






BOOST_AUTO_TEST_CASE( linear_solve ){
    

    Matrix A(3, 3);
    
    A(0, 0) = 4.0;   A(0, 1) = 1.1;
    A(1, 0) = 1.1;   A(1, 1) = 2.2;   A(1, 2) = 3.3;
                     A(2, 1) = 3.3;   A(2, 2) = 9.0;

    Vector b(3);
    b(0) = 12.0;
    b(1) = -5.0;
    b(2) = 3.3;
    
    Vector x(3);
    int superDiag = 1;
    
    int result = solveSPDBLinearSystem(A, b, superDiag, x);
    
    BOOST_CHECK(result == 0);
    
    double tol = 1e-8;
    BOOST_CHECK_CLOSE(x(0), 6.803999999999998, tol);
    BOOST_CHECK_CLOSE(x(1), -13.832727272727267, tol);
    BOOST_CHECK_CLOSE(x(2), 5.438666666666664, tol);
    
    
}


BOOST_AUTO_TEST_SUITE_END()
