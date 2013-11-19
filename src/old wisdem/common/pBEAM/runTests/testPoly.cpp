//
//  testPoly.cpp
//  runTests
//
//  Created by Andrew Ning on 2/4/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#define BOOST_TEST_DYN_LINK
#ifdef STAND_ALONE
#define BOOST_TEST_MODULE pbeamTests
#endif
#include <boost/test/unit_test.hpp>

#include "Poly.h"


BOOST_AUTO_TEST_SUITE( poly )


BOOST_AUTO_TEST_CASE( paren ){
    
    double x1[3] = {1.9, 2.4, 5.2};
    
    Poly p1(3, x1);
    
    double tol = 1e-8;
    BOOST_CHECK_CLOSE(p1(0), 1.9, tol);
    
    p1(0) = 1.8;
    BOOST_CHECK_CLOSE(p1(0), 1.8, tol);    
    
}

BOOST_AUTO_TEST_CASE( val ){
    
    Poly p1(4, 1.2, 8.1, 6.8, 3.5);
    
    double tol = 1e-8;
    BOOST_CHECK_CLOSE(p1.eval(2.3), 76.589399999999998, tol);
          
    Poly p2(1, 1.1);
                      
    BOOST_CHECK_CLOSE(p2.eval(2.3), 1.1, tol);    
    
}



BOOST_AUTO_TEST_CASE( mult_const ){
    
    double x1[3] = {1.9, 2.4, 5.2};

    Poly p1(3, x1);
    
    Poly p2 = p1 * 2.0;
    
    double tol = 1e-8;
    BOOST_CHECK_CLOSE(p2(0), 3.8, tol);
    BOOST_CHECK_CLOSE(p2(1), 4.8, tol);
    BOOST_CHECK_CLOSE(p2(2), 10.4, tol);

}

BOOST_AUTO_TEST_CASE( mult ){
    
    Poly p1(3, 1.9, 2.4, 5.2);
    Poly p2(4, 3.2, 3.1, 6.8, 1.5);
    
    Poly p3 = p1 * p2;
    
    double tol = 1e-8;
    BOOST_CHECK_CLOSE(p3(0), 6.0800, tol);
    BOOST_CHECK_CLOSE(p3(1), 13.5700, tol);
    BOOST_CHECK_CLOSE(p3(2), 37.0000, tol);
    BOOST_CHECK_CLOSE(p3(3), 35.2900, tol);
    BOOST_CHECK_CLOSE(p3(4), 38.9600, tol);
    BOOST_CHECK_CLOSE(p3(5), 7.8000, tol);
    
}

BOOST_AUTO_TEST_CASE( add ){
    
    double x1[] = {1.9, 2.4, 5.2};
    double x2[] = {3.2, 3.1, 6.8, 1.5};
    Poly p1(3, x1);
    Poly p2(4, x2);
    
    Poly p3 = p1 + p2;
    
    double tol = 1e-8;
    BOOST_CHECK_CLOSE(p3(0), 3.2, tol);
    BOOST_CHECK_CLOSE(p3(1), 5.0, tol);
    BOOST_CHECK_CLOSE(p3(2), 9.2, tol);
    BOOST_CHECK_CLOSE(p3(3), 6.7, tol);
    
}


BOOST_AUTO_TEST_CASE( integrate )
{
    double x1[] = {1.2, 8.1, 6.8, 3.5};
    Poly p1(4, x1);
    
    Poly p2 = p1.integrate();
    
    double tol = 1e-8;
    BOOST_CHECK_EQUAL(p2.length(), 5);
    BOOST_CHECK_CLOSE(p2(0), 0.3, tol);
    BOOST_CHECK_CLOSE(p2(1), 2.7, tol);    
    BOOST_CHECK_CLOSE(p2(2), 3.4, tol);
    BOOST_CHECK_CLOSE(p2(3), 3.5, tol);
    BOOST_CHECK_CLOSE(p2(4), 0.0, tol);
    
}


BOOST_AUTO_TEST_CASE( integrate_w_limits ){
    
    double x1[] = {1.2, 8.1, 6.8, 3.5};
    Poly p1(4, x1);
    double lower = 1.1;
    double upper = 3.3;
    
    double tol = 1e-8;    
    BOOST_CHECK_CLOSE(p1.integrate(lower,upper), 169.186600, tol);
}


BOOST_AUTO_TEST_CASE( deriv ){
    
    double x1[] = {1.2, 8.1, 6.8, 3.5};
    Poly p1(4, x1);

    Poly p2 = p1.differentiate();
    
    double tol = 1e-8;    
    BOOST_CHECK_CLOSE(p2(0), 3.6, tol);
    BOOST_CHECK_CLOSE(p2(1), 16.2, tol);
    BOOST_CHECK_CLOSE(p2(2), 6.8, tol);
}



BOOST_AUTO_TEST_SUITE_END()


