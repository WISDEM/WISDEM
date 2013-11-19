//
//  Poly.h
//  pbeam
//
//  Created by Andrew Ning on 2/4/12.
//  Copyright (c) 2012 NREL. All rights reserved.
//

#ifndef pbeam_Poly_h
#define pbeam_Poly_h


/**
 This class represents a polynomial.

 Polynomials are represented by their coefficients in descending powers.
 For example: [3.0, 5.0, 7.0] = 3x^2 + 5x + 7
 whereas [9.0 2.0 1.1 7.6] = 9x^3 + 2x^2 + 1.1 + 7.6

 Methods are provided for common operators such as addition and multiplication
 as well as analytic integration and differentiation.
 
 **/



class Poly {
    int n;
    double *p;
    
public:

    
// MARK: ------------ CONSTRUCTORS/DESCTRUCTOR ---------------
    
    // empty constructor
    Poly();
    
    /**
     Construct from array of coefficients and length of array x.
     
     Arguments:
     x - coefficient array
     length - length of x
     
     Coefficients start from highest power and descend to a constant term.
     for example {3.0, 5.0, 7.0} == 3x^2 + 5x + 7
     whereas {9.0, 2.0, 1.1, 7.6} == 9x^3 + 2x^2 + 1.1x + 7.6
     
     **/
    Poly(int length, double x[]);
    
    
    /**
     Construct from a variable number of arguments of type double.
     
     Arguments:
     length - number of entries following
     ... - variable number of arguments
     
     for example: Poly(3, 1.1, 2.2, 3.3) == 1.1x^2 + 2.2x + 3.3
     
     **/
    Poly(int length, ...);
    
    
    // copy constructor
    Poly(const Poly &pSource);
    
    // assignment operator
    Poly& operator= (const Poly &pSource);
    
    // destructor
    ~Poly();
    
    
// MARK: -----------  ARRAY-LIKE INTERACTION ---------------
    
    /**
     Allows polynomial coefficients to be acccessed by index.
     i.e. p(4)
     
     They can be accessed from index 0 up to length()-1
     
     **/
    double& operator() (int idx);
    double  operator() (int idx) const;
    
    // returns length of polynomial
    int length() const;


    
// MARK: ----------------- POLY MULTIPLICATION ------------------
    
    
    /**
     Multiplies two polynomials together.
     
     Arguments:
     y - second polynomial
     
     Returns:
     x * y
     
     **/
    Poly operator* (const Poly &y) const;
    
         
    /**
     multiplication/division-assignment
     
     Arguments:
     right - polynomial to multiply/divide (or constant k)
     
     Returns: reference (multiply/divide in place)
     p *= or /= right 
     
     **/
    
    Poly& operator*=(const Poly& right);
    Poly& operator*=(const double k);
    Poly& operator/=(const double k);

     
     
// MARK: ------------- POLY ADDITION / NEGATION --------------
     
    /**
     Adds/subtracts two polynomials together.
     
     Arguments:
     y - second polynomial
     
     Returns:
     x +/- y

     **/
    Poly operator+ (const Poly &y) const;
    Poly operator- (const Poly &y) const;
    
    /**
     addition/subtraction-assignment
     
     Arguments:
     right - polynomial to add/subtract (or constant k)
     
     Returns: reference (adds/subtracts in place)
     p +=/-= right 
     
     **/
    
    Poly& operator+=(const Poly& right);
    Poly& operator+=(const double k);
    Poly& operator-=(const Poly& right);
    Poly& operator-=(const double k);
    
    /**
     Negation.
     
     Returns:
     the negative of the polynomial
     
     **/
    Poly operator- () const;
    
    
    
    
// MARK: ---------------  POLY EVALUATION ---------------
    
    /**
     Evaluate polynomial at a point
     
     Arguments:
     x - point at which to evaluate polynomial
     
     Returns:
     p(x) polynomial evaluted at x
     
     **/
    double eval(double x) const;
    
    
    
    
// MARK: ---------------  POLY INT/DERIV ---------------
    
    /**
     Analytic indefinite integral of a polynomial.
     
     Returns:
     polynomial - integral of p
     
     **/
    Poly integrate() const;
    
    
    /**
     Analytic integral of a polynomial within definite limits
     
     Arguments:
     lower - lower bound of integration
     upper - upper bound of integration
     
     Returns:
     definite integral of p from lower to upper
     
     **/
    double integrate(double lower, double upper) const;    
    
    
    
    /**
     Useful when polynomial is specified in one direction,
     but integration must be done in the other direction.
     Returns integrated polynomial in the original direction
     (not the direction of integration).
     
     Returns:
     polynomial - integral of p in opposite direction of specification
     
     **/
    Poly backwardsIntegrate() const;
    
    
    /**
     Analytic derivative of a polynomial
     
     Returns:
     result - derivative of polynomial
     **/
    Poly differentiate() const;
    
};


//MARK: ------- NON-MEMBER FUNCTIONS -----------------


/**
 Multiplies or divides polynomial by a constant.
 
 Arguments:
 k - constant to multiply or divide polynomial by
 
 Returns:
 left * / k or k * / right
 
 **/
Poly operator* (const Poly &left, const double k);
Poly operator* (const double k, const Poly &right);
Poly operator/ (const Poly &left, const double k);


/**
 Adds/subtracts a constant to a polynomial
 
 Arguments:
 k - constant to add/subtract to polynomial
 
 Returns
 left +/- k or k +/- right
 
 **/
Poly operator+ (const Poly &left, const double k);
Poly operator+ (const double k, const Poly &right);
Poly operator- (const Poly &left, const double k);



// overload output stream
std::ostream& operator<<(std::ostream& output, const Poly& p);

#endif
