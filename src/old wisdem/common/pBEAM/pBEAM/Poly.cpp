//
//  Poly.cpp
//  pbeam
//
//  Created by Andrew Ning on 2/4/12.
//  Copyright (c) 2012 NREL. All rights reserved.
//

#include <algorithm>
#include <cmath>
#include <iostream>
#include <cstdarg>  // needed for some compilers for var args

#include "Poly.h"

using namespace std;


/**
 Polynomials are represented internally by arrays of the polynomial coefficients in descending powers.
 For example: [3.0, 5.0, 7.0] = 3x^2 + 5x + 7
 whereas [9.0 2.0 1.1 7.6] = 9x^3 + 2x^2 + 1.1 + 7.6
 **/


// MARK: ------------ CONSTRUCTORS/DESCTRUCTOR ---------------

Poly::Poly(){
    n = 0;
    p = new double[n];
}


Poly::Poly(int length, double x[]){
    n = length;
    p = new double[n];
    for (int i = 0; i < length; i++) {
        p[i] = x[i];
    }
    
}

Poly::Poly(int length, ...){
    
    va_list values;
    va_start(values, length);
    
    n = length;
    p = new double[n];
    for (int i = 0; i < n; i++) {
        p[i] = va_arg(values, double);
    }
    
    va_end(values);
    
}

// copy constructor
Poly::Poly(const Poly &pSource){
    
    n = pSource.n;
    if (pSource.p){
        p = new double[n];
        
        for (int i = 0; i < n; i++) {
            p[i] = pSource.p[i];
        }
    } else{
        p = 0;
    }
    
}


// assignment overloading
Poly& Poly::operator= (const Poly &pSource){
    
    if (this != &pSource){
        
        delete [] p; 
        
        n = pSource.n;
        if (pSource.p){
            p = new double[n];
            
            for (int i = 0; i < n; i++) {
                p[i] = pSource.p[i];
            }
        } else{
            p = 0;
        }
        
    }
    
    return *this;
}


Poly::~Poly(){
    delete [] p;
}



    


// MARK: -----------  ARRAY-LIKE INTERACTION ---------------

int Poly::length() const{
    return n;
}


double& Poly::operator() (int idx){
    
    return p[idx];
}

double Poly::operator() (int idx) const{
    
    return p[idx];
}



// MARK: ---------------  POLY EVALUATION ---------------


// evaluate polynomial at value x
double Poly::eval(double x) const{
    
    double value = 0.0;
    
    for (int i = 0; i < n; i++) {
        value += p[n-1-i] * pow(x, i);
    }
    
    return value;
}



// --------------------------------------------




// MARK: ----------------- POLY MULTIPLICATION ------------------

// multiply polynomial by a constant
Poly operator*(const Poly &left, const double k){

    int n = left.length();
    double x[n];
    for (int i = 0; i < n; i++) {
        x[i] = left(i)*k;
    }
    
    return Poly(n, x);
}

Poly operator*(const double k, const Poly &right){
    
    return right * k;
}

// divide polynomial by a constant
Poly operator/(const Poly &left, const double k){
 
    return left * (1.0/k);
}


// multiply polynomial by another polynomial
Poly Poly::operator*(const Poly &y) const{
    
    // compute length of resulting polynomial
    int nz = max(max(n + y.n - 1, n), y.n);

    double z[nz];
    
    // uses discrete convolution algorithm
    for (int i = 0; i < nz; i++) {
        z[i] = 0.0;
        
        for (int j = 0; j < n; j++) {
            int ij = i - j;
            if (ij >= 0 && ij < y.n) {
                z[i] += p[j] * y.p[ij];
            }
        }
    }

    
    return Poly(nz, z);
}

// multiply self by polynomial
Poly& Poly::operator*=(const Poly& right){
    
    *this = *this * right;
    
    return *this;
}

// multiply self by constant
Poly& Poly::operator*=(const double k){
    
    *this = *this * k;
    
    return *this;
}

// divide self by constant
Poly& Poly::operator/=(const double k){
    
    *this = *this / k;
    
    return *this;
}

// -----------------------------------------



// MARK: ------------- POLY ADDITION / NEGATION --------------

// add two polynomials
Poly Poly::operator+ (const Poly &y) const{
    
    int nz = max(n, y.n);
    double z_array[nz];
    Poly z(nz, z_array);
    
    int i;
    
    if (n >= y.n) {
        
        // first copy larger one over
        for (i = 0; i < n; i++) {
            z.p[i] = p[i];
        }
        
        // add end of other polynomial
        for (i = 0; i < y.n; i++) {
            z.p[nz-1-i] += y.p[y.n-1-i];
        }
        
    } else{
        
        // first copy larger one over
        for (i = 0; i < y.n; i++) {
            z.p[i] = y.p[i];
        }
        
        // add end of other polynomial
        for (i = 0; i < n; i++) {
            z.p[z.n-1-i] += p[n-1-i];
        }
        
    }

    return z;
}

// subtracts two polynomials
Poly Poly::operator- (const Poly &y) const{

    return *this + (-y);
}

// add polynomial and constant
Poly operator+ (const Poly &left, const double k){
    
    int n = left.length();
    double x[n];
    for (int i = 0; i < n; i++) {
        x[i] = left(i);
    }
    
    x[n-1] += k;
    
    return Poly(n, x);
}

// add polynomial and constant
Poly operator+ (const double k, const Poly &right){
    
    return right + k;

}

// subtract constant from polynomial
Poly operator- (const Poly &left, const double k){
    
    return left + (-k);
}


// add constant to self
Poly& Poly::operator+=(const double k){
    
    *this = *this + k;
    
    return *this;
}

// add polynomial to self
Poly& Poly::operator+=(const Poly& right){
    
    *this = *this + right;
    
    return *this;
}

// subtract constant from self
Poly& Poly::operator-=(const double k){
    
    *this = *this - k;
    
    return *this;
}

// subtract polynomial from self
Poly& Poly::operator-=(const Poly& right){
    
    *this = *this - right;
    
    return *this;
}

// negate
Poly Poly::operator- () const{
    
    double x[n];
    
    for (int i = 0; i < n; i++) {
        x[i] = -p[i];
    }
    
    return Poly(n, x);
}

// --------------------------------------------



// MARK: ---------------  POLY INT/DERIV ---------------


// integrate polynomial p, return result pout of length: np+1 
Poly Poly::integrate() const{
    
    double pout[n+1];
    
    for (int i = 0; i < n; i++) {
        pout[i] = p[i] / (n-i);
    }
    pout[n] = 0.0;
    
    return Poly(n+1, pout);
}



// integrate polynomial from lower to upper
double Poly::integrate(double lower, double upper) const{
    
    // perform integration
    Poly integral = this->integrate();
    
    return integral.eval(upper) - integral.eval(lower);
}


Poly Poly::backwardsIntegrate() const{
    
    Poly p = this->integrate();

    double const_term = 0.0;
    
    for (int i = 0; i < p.length() - 1; i++) {
        const_term += p(i);
        p(i) = -p(i);
    }
    
    p += const_term;
        
    return p;
}






// analytic derivative of polynomial p, result if of length np-1
Poly Poly::differentiate() const{
    
    double result[n-1];
    
    for (int i = 0; i < n-1; i++) {
        result[i] = p[i]*(n-1-i);
    }    
    
    return Poly(n-1, result);
}


// --------------------------------------------


// MARK: ------------- OUTPUT STREAM ------------------


ostream& operator<<(ostream& output, const Poly& p) {

    output << "poly = (";
    
    if (p.length() > 0) {
        output << p(0);
    }
    
    for (int i = 1; i < p.length(); i++) {
        output << "," << p(i);
    }
    
    output << ")";
    
    return output;  
}
