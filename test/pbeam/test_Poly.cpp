//
//  testPoly.cpp
//  runTests
//
//  Created by Andrew Ning on 2/4/12.
//  gbarter removed boost 6/2018

//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "catch.hpp"

#include "Poly.h"

TEST_CASE( "paren" ){
    
    double x1[3] = {1.9, 2.4, 5.2};
    
    Poly p1(3, x1);
    
    double tol = 1e-8;
    REQUIRE(p1(0) == Approx(1.9).epsilon(tol));
    
    p1(0) = 1.8;
    REQUIRE(p1(0) == Approx(1.8).epsilon(tol));    
    
}

TEST_CASE( "val" ){
    
    Poly p1(4, 1.2, 8.1, 6.8, 3.5);
    
    double tol = 1e-8;
    REQUIRE(p1.eval(2.3) == Approx(76.589399999999998).epsilon(tol));
          
    Poly p2(1, 1.1);
                      
    REQUIRE(p2.eval(2.3) == Approx(1.1).epsilon(tol));    
    
}



TEST_CASE( "mult_const" ){
    
    double x1[3] = {1.9, 2.4, 5.2};

    Poly p1(3, x1);
    
    Poly p2 = p1 * 2.0;
    
    double tol = 1e-8;
    REQUIRE(p2(0) == Approx(3.8).epsilon(tol));
    REQUIRE(p2(1) == Approx(4.8).epsilon(tol));
    REQUIRE(p2(2) == Approx(10.4).epsilon(tol));

}

TEST_CASE( "mult" ){
    
    Poly p1(3, 1.9, 2.4, 5.2);
    Poly p2(4, 3.2, 3.1, 6.8, 1.5);
    
    Poly p3 = p1 * p2;
    
    double tol = 1e-8;
    REQUIRE(p3(0) == Approx(6.0800).epsilon(tol));
    REQUIRE(p3(1) == Approx(13.5700).epsilon(tol));
    REQUIRE(p3(2) == Approx(37.0000).epsilon(tol));
    REQUIRE(p3(3) == Approx(35.2900).epsilon(tol));
    REQUIRE(p3(4) == Approx(38.9600).epsilon(tol));
    REQUIRE(p3(5) == Approx(7.8000).epsilon(tol));
    
}

TEST_CASE( "add" ){
    
    double x1[] = {1.9, 2.4, 5.2};
    double x2[] = {3.2, 3.1, 6.8, 1.5};
    Poly p1(3, x1);
    Poly p2(4, x2);
    
    Poly p3 = p1 + p2;
    
    double tol = 1e-8;
    REQUIRE(p3(0) == Approx(3.2).epsilon(tol));
    REQUIRE(p3(1) == Approx(5.0).epsilon(tol));
    REQUIRE(p3(2) == Approx(9.2).epsilon(tol));
    REQUIRE(p3(3) == Approx(6.7).epsilon(tol));
    
}


TEST_CASE( "integrate" ){
  
    double x1[] = {1.2, 8.1, 6.8, 3.5};
    Poly p1(4, x1);
    
    Poly p2 = p1.integrate();
    
    double tol = 1e-8;
    REQUIRE(p2.length()== 5);
    REQUIRE(p2(0) == Approx(0.3).epsilon(tol));
    REQUIRE(p2(1) == Approx(2.7).epsilon(tol));    
    REQUIRE(p2(2) == Approx(3.4).epsilon(tol));
    REQUIRE(p2(3) == Approx(3.5).epsilon(tol));
    REQUIRE(p2(4) == Approx(0.0).epsilon(tol));
    
}


TEST_CASE( "integrate_w_limits" ){
    
    double x1[] = {1.2, 8.1, 6.8, 3.5};
    Poly p1(4, x1);
    double lower = 1.1;
    double upper = 3.3;
    
    double tol = 1e-8;    
    REQUIRE(p1.integrate(lower,upper) == Approx(169.186600).epsilon(tol));
}


TEST_CASE( "deriv" ){
    
    double x1[] = {1.2, 8.1, 6.8, 3.5};
    Poly p1(4, x1);

    Poly p2 = p1.differentiate();
    
    double tol = 1e-8;    
    REQUIRE(p2(0) == Approx(3.6).epsilon(tol));
    REQUIRE(p2(1) == Approx(16.2).epsilon(tol));
    REQUIRE(p2(2) == Approx(6.8).epsilon(tol));
}


