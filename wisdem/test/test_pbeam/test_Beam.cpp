//
//  testBeam.cpp
//  pbeam
//
//  Created by Andrew Ning on 2/7/12.
//  Copyright (c) 2012 NREL. All rights reserved.
//

#include "catch.hpp"

#include <iostream>
#include <math.h>
#include "Beam.h"

// Test data from "Finite Element Structural Analysis", Yang, pg. 145
TEST_CASE( "cantilever_deflection" ){


    double E = 2.0;
    double I = 3.0;
    double L = 4.0;
    double p0 = 5.0;

    int nodes = 2;

    Vector Px(nodes);
    Vector Py(nodes);
    Vector Pz(nodes);

    Px(0) = -p0;
    Px(1) = 0.0;

    Py.setZero();
    Pz.setZero();

    Loads loads = Loads(Px, Py, Pz);

    TipData tip;

    BaseData base;
    for (int i = 0; i < 6; i++) {
        base.rigid[i] = true;
    }

    Vector z(2);
    z(0) = 0.0;
    z(1) = L;

    Vector EIx(2);
    EIx(0) = E*I;
    EIx(1) = E*I;

    Vector EIy(2);
    EIy(0) = E*I;
    EIy(1) = E*I;

    Vector EA(2);
    EA(0) = E*1.0;
    EA(1) = E*1.0;

    Vector GJ(2);
    GJ(0) = 1.0;
    GJ(1) = 1.0;

    Vector rhoA(2);
    rhoA(0) = 1.0;
    rhoA(1) = 1.0;

    Vector rhoJ(2);
    rhoJ(0) = 1.0;
    rhoJ(1) = 1.0;

    SectionData sec = SectionData(z, EA, EIx, EIy, GJ, rhoA, rhoJ);

    Beam beam = Beam(sec, loads, tip, base);

    Vector dx(2);
    Vector dy(2);
    Vector dz(2);
    Vector dtheta_x(2);
    Vector dtheta_y(2);
    Vector dtheta_z(2);

    beam.computeDisplacement(dx, dy, dz, dtheta_x, dtheta_y, dtheta_z);

    double tol = 1e-8;
    REQUIRE(dx(0) == Approx(0.0).epsilon(tol));
    REQUIRE(dx(1) == Approx(-p0*pow(L,3)/E/I*L/30).epsilon(tol));
    REQUIRE(dtheta_y(0) == Approx(0.0).epsilon(tol));
    REQUIRE(dtheta_y(1) == Approx(-p0*pow(L,3)/E/I*1.0/24).epsilon(tol));

}


// Test data from "Finite Element Structural Analysis", Yang, pg. 180
TEST_CASE( "tapered_deflections" ){


    //double r = 8.0;
    double E = 2.0;
    double I = 3.0;
    double L = 4.0;
    double P = 5.0;

    int nodes = 3;

    Vector Px(nodes);
    Vector Py(nodes);
    Vector Pz(nodes);

    Px.setZero();
    Py.setZero();
    Pz.setZero();

    Loads loads = Loads(Px, Py, Pz);

    TipData tip;
    tip.Fx = -P;

    BaseData base;
    for (int i = 0; i < 6; i++) {
        base.rigid[i] = true;
    }

    Vector z(nodes);
    z(0) = 0.0;
    z(1) = L/2.0;
    z(2) = L;

    Vector EIx(nodes);
    EIx(0) = E*9*I;
    EIx(1) = E*5*I;
    EIx(2) = E*I;

    Vector EIy(nodes);
    EIy(0) = E*9*I;
    EIy(1) = E*5*I;
    EIy(2) = E*I;

    Vector EA(nodes);
    EA(0) = E*1.0;
    EA(1) = E*1.0;
    EA(2) = E*1.0;

    Vector GJ(nodes);
    GJ(0) = 1.0;
    GJ(1) = 1.0;
    GJ(2) = 1.0;

    Vector rhoA(nodes);
    rhoA(0) = 1.0;
    rhoA(1) = 1.0;
    rhoA(2) = 1.0;

    Vector rhoJ(nodes);
    rhoJ(0) = 1.0;
    rhoJ(1) = 1.0;
    rhoJ(2) = 1.0;

    SectionData sec = SectionData(z, EA, EIx, EIy, GJ, rhoA, rhoJ);

    Beam beam = Beam(sec, loads, tip, base);

    Vector dx(nodes);
    Vector dy(nodes);
    Vector dz(nodes);
    Vector dtheta_x(nodes);
    Vector dtheta_y(nodes);
    Vector dtheta_z(nodes);

    beam.computeDisplacement(dx, dy, dz, dtheta_x, dtheta_y, dtheta_z);

    double tol_pct_1 = 0.17;
    double tol_pct_2 = 0.77;
    REQUIRE(dx(0) == Approx(0.0).epsilon(1e-8));
    REQUIRE(dx(nodes-1) == Approx(-0.051166*P*pow(L,3)/E/I).epsilon(tol_pct_1));
    REQUIRE(dtheta_y(0) == Approx(0.0).epsilon(1e-8));
    REQUIRE(dtheta_y(nodes-1) == Approx(-0.090668*P*L*L/E/I).epsilon(tol_pct_2));

}


// Test data from "Consistent Mass Matrix for Distributed Mass Systmes", John Archer,
// Journal of the Structural Division Proceedings of the American Society of Civil Engineers,
// pg. 168
TEST_CASE( "freq_free_free_beam_n_1" ){

    double E = 2.0;
    double I = 3.0;
    double L = 4.0;
    double A = 5.0;
    double rho = 6.0;

    int n = 1;

    int nodes = n+1;

    Vector Px(nodes);
    Vector Py(nodes);
    Vector Pz(nodes);
    Px.setZero();
    Py.setZero();
    Pz.setZero();

    Loads loads = Loads(Px, Py, Pz);

    TipData tip;

    BaseData base;

    Vector z(nodes);
    for (int i = 0; i < nodes; i++) {
        z(i) = L*i;
    }

    Vector EIx(nodes);
    for (int i = 0; i < nodes; i++) {
        EIx(i) = E * I;
    }

    Vector EIy(nodes);
    for (int i = 0; i < nodes; i++) {
        EIy(i) = E * I;
    }

    Vector EA(nodes);
    for (int i = 0; i < nodes; i++) {
        EA(i) = 1.0;
    }

    Vector GJ(nodes);
    for (int i = 0; i < nodes; i++) {
        GJ(i) = 1.0;
    }

    Vector rhoA(nodes);
    for (int i = 0; i < nodes; i++) {
        rhoA(i) = rho * A;
    }

    Vector rhoJ(nodes);
    for (int i = 0; i < nodes; i++) {
        rhoJ(i) = 1.0;
    }

    SectionData sec = SectionData(z, EA, EIx, EIy, GJ, rhoA, rhoJ);

    Beam beam = Beam(sec, loads, tip, base);


    int nFreq = 100;
    Vector freq(nFreq);

    beam.computeNaturalFrequencies(nFreq, freq);

    double m = rho * A;
    double alpha = m * pow(n*L, 4) / (840.0 * E * I);

    double tol_pct = 5e-6*100;
    REQUIRE(freq(1) == Approx(sqrt(0.85714 / alpha) / (2*M_PI)).epsilon(tol_pct));
    REQUIRE(freq(2) == Approx(sqrt(0.85714 / alpha) / (2*M_PI)).epsilon(tol_pct));

    REQUIRE(freq(4) == Approx(sqrt(10.0 / alpha) / (2*M_PI)).epsilon(tol_pct));
    REQUIRE(freq(5) == Approx(sqrt(10.0 / alpha) / (2*M_PI)).epsilon(tol_pct));


}


// Test data from "Consistent Mass Matrix for Distributed Mass Systmes", John Archer,
// Journal of the Structural Division Proceedings of the American Society of Civil Engineers,
// pg. 168
TEST_CASE( "freq_free_free_beam_n_2" ){

    double E = 2.0;
    double I = 3.0;
    double L = 4.0;
    double A = 5.0;
    double rho = 6.0;

    int n = 2;

    int nodes = n+1;

    Vector Px(nodes);
    Vector Py(nodes);
    Vector Pz(nodes);
    Px.setZero();
    Py.setZero();
    Pz.setZero();

    Loads loads = Loads(Px, Py, Pz);

    TipData tip;

    BaseData base;

    Vector z(nodes);
    for (int i = 0; i < nodes; i++) {
        z(i) = L*i;
    }

    Vector EIx(nodes);
    for (int i = 0; i < nodes; i++) {
        EIx(i) = E * I;
    }

    Vector EIy(nodes);
    for (int i = 0; i < nodes; i++) {
        EIy(i) = E * I;
    }

    Vector EA(nodes);
    for (int i = 0; i < nodes; i++) {
        EA(i) = 1.0;
    }

    Vector GJ(nodes);
    for (int i = 0; i < nodes; i++) {
        GJ(i) = 1.0;
    }

    Vector rhoA(nodes);
    for (int i = 0; i < nodes; i++) {
        rhoA(i) = rho * A;
    }

    Vector rhoJ(nodes);
    for (int i = 0; i < nodes; i++) {
        rhoJ(i) = 1.0;
    }

    SectionData sec = SectionData(z, EA, EIx, EIy, GJ, rhoA, rhoJ);

    Beam beam = Beam(sec, loads, tip, base);

    int nFreq = 100;
    Vector freq(nFreq);

    beam.computeNaturalFrequencies(nFreq, freq);

    double m = rho * A;
    double alpha = m * pow(n*L, 4) / (840.0 * E * I);

    double tol_pct = 6e-6*100;
    REQUIRE(freq(1) == Approx(sqrt(0.59858 / alpha) / (2*M_PI)).epsilon(tol_pct));
    REQUIRE(freq(2) == Approx(sqrt(0.59858 / alpha) / (2*M_PI)).epsilon(tol_pct));

    REQUIRE(freq(5) == Approx(sqrt(5.8629 / alpha) / (2*M_PI)).epsilon(tol_pct));
    REQUIRE(freq(6) == Approx(sqrt(5.8629 / alpha) / (2*M_PI)).epsilon(tol_pct));

    REQUIRE(freq(8) == Approx(sqrt(36.659 / alpha) / (2*M_PI)).epsilon(tol_pct));
    REQUIRE(freq(9) == Approx(sqrt(36.659 / alpha) / (2*M_PI)).epsilon(tol_pct));

    REQUIRE(freq(10) == Approx(sqrt(93.566 / alpha) / (2*M_PI)).epsilon(tol_pct));
    REQUIRE(freq(11) == Approx(sqrt(93.566 / alpha) / (2*M_PI)).epsilon(tol_pct));


}



// Test data from "Consistent Mass Matrix for Distributed Mass Systmes", John Archer,
// Journal of the Structural Division Proceedings of the American Society of Civil Engineers,
// pg. 168
TEST_CASE( "freq_free_free_beam_n_3" ){

    double E = 2.0;
    double I = 3.0;
    double L = 4.0;
    double A = 5.0;
    double rho = 6.0;

    int n = 3;

    int nodes = n+1;

    Vector Px(nodes);
    Vector Py(nodes);
    Vector Pz(nodes);
    Px.setZero();
    Py.setZero();
    Pz.setZero();

    Loads loads = Loads(Px, Py, Pz);

    TipData tip;

    BaseData base;

    Vector z(nodes);
    for (int i = 0; i < nodes; i++) {
        z(i) = L*i;
    }

    Vector EIx(nodes);
    for (int i = 0; i < nodes; i++) {
        EIx(i) = E * I;
    }

    Vector EIy(nodes);
    for (int i = 0; i < nodes; i++) {
        EIy(i) = E * I;
    }

    Vector EA(nodes);
    for (int i = 0; i < nodes; i++) {
        EA(i) = 1.0;
    }

    Vector GJ(nodes);
    for (int i = 0; i < nodes; i++) {
        GJ(i) = 1.0;
    }

    Vector rhoA(nodes);
    for (int i = 0; i < nodes; i++) {
        rhoA(i) = rho * A;
    }

    Vector rhoJ(nodes);
    for (int i = 0; i < nodes; i++) {
        rhoJ(i) = 1.0;
    }

    SectionData sec = SectionData(z, EA, EIx, EIy, GJ, rhoA, rhoJ);

    Beam beam = Beam(sec, loads, tip, base);

    int nFreq = 100;
    Vector freq(nFreq);

    beam.computeNaturalFrequencies(nFreq, freq);

    double m = rho * A;
    double alpha = m * pow(n*L, 4) / (840.0 * E * I);

    double tol_pct = 6e-6*100;
    REQUIRE(freq(1) == Approx(sqrt(0.59919 / alpha) / (2*M_PI)).epsilon(tol_pct));
    REQUIRE(freq(2) == Approx(sqrt(0.59919 / alpha) / (2*M_PI)).epsilon(tol_pct));

    REQUIRE(freq(5) == Approx(sqrt(4.5750 / alpha) / (2*M_PI)).epsilon(tol_pct));
    REQUIRE(freq(6) == Approx(sqrt(4.5750 / alpha) / (2*M_PI)).epsilon(tol_pct));

    REQUIRE(freq(8) == Approx(sqrt(22.010 / alpha) / (2*M_PI)).epsilon(.00105));
    REQUIRE(freq(9) == Approx(sqrt(22.010 / alpha) / (2*M_PI)).epsilon(.00105));

    REQUIRE(freq(11) == Approx(sqrt(70.920 / alpha) / (2*M_PI)).epsilon(tol_pct));
    REQUIRE(freq(12) == Approx(sqrt(70.920 / alpha) / (2*M_PI)).epsilon(tol_pct));

    REQUIRE(freq(14) == Approx(sqrt(265.91 / alpha) / (2*M_PI)).epsilon(0.00073));
    REQUIRE(freq(15) == Approx(sqrt(265.91 / alpha) / (2*M_PI)).epsilon(0.00073));

    REQUIRE(freq(16) == Approx(sqrt(402.40 / alpha) / (2*M_PI)).epsilon(tol_pct));
    REQUIRE(freq(17) == Approx(sqrt(402.40 / alpha) / (2*M_PI)).epsilon(tol_pct));


}


// unit test data from Euler's buckling formula for a clamped/free beam
TEST_CASE( "buckling_euler" ){

    double E = 2.0;
    double I = 3.0;
    double L = 4.0;

    int n = 3;

    int nodes = n+1;

    Vector z(nodes);
    for (int i = 0; i < nodes; i++) {
        z(i) = L*i;
    }

    Vector EIx(nodes);
    for (int i = 0; i < nodes; i++) {
        EIx(i) = E * I;
    }

    Vector EIy(nodes);
    for (int i = 0; i < nodes; i++) {
        EIy(i) = E * I;
    }

    Vector EA(nodes);
    for (int i = 0; i < nodes; i++) {
        EA(i) = 1.0;
    }

    Vector GJ(nodes);
    for (int i = 0; i < nodes; i++) {
        GJ(i) = 1.0;
    }

    Vector rhoA(nodes);
    for (int i = 0; i < nodes; i++) {
        rhoA(i) = 1.0;
    }

    Vector rhoJ(nodes);
    for (int i = 0; i < nodes; i++) {
        rhoJ(i) = 1.0;
    }

    Vector Px(nodes);
    Vector Py(nodes);
    Vector Pz(nodes);
    Px.setZero();
    Py.setZero();
    Pz.setZero();

    Loads loads = Loads(Px, Py, Pz);

    TipData tip;

    BaseData base;
    for (int i = 0; i < 6; i++) {
        base.rigid[i] = true;
    }

    SectionData sec = SectionData(z, EA, EIx, EIy, GJ, rhoA, rhoJ);

    Beam beam = Beam(sec, loads, tip, base);

    double Pcr_x, Pcr_y;

    beam.computeMinBucklingLoads(Pcr_x, Pcr_y);

    double Ltotal = n*L;


    double tol_pct = 0.011;
    REQUIRE(Pcr_x == Approx(E * I * pow(M_PI/2.0/Ltotal, 2)).epsilon(tol_pct));
    REQUIRE(Pcr_y == Approx(E * I * pow(M_PI/2.0/Ltotal, 2)).epsilon(tol_pct));

}


// private method used for test below
double computeCriticalFactorForTaperedBeamBuckling(int nEI, int nq, double m_start, double m_finish, double m_delta){


    double E = 2.0;
    double L = 9.0;
    double b0 = 2.5;
    double h0 = 1.8;

    double I0 = b0 * pow(h0, 3) / 12.0;
    double I0_other = h0 * pow(b0, 3) / 12.0;

    int n = 30;
    int nodes = n+1;

    TipData tip;

    BaseData base;
    for (int i = 0; i < 6; i++) {
        base.rigid[i] = true;
    }

    Vector z(nodes);
    for (int i = 0; i < nodes; i++) {
        z(i) = L*(double)i/(nodes-1);
    }


    PolyVec EIx(nodes-1);
    for (int i = 0; i < nodes-1; i++) {

        if (nEI == 0){

            EIx[i] = Poly(1, E*I0_other);

        } else if (nEI == 1){

            double EI1 = (1 - z(i)/L) * E*I0_other;
            double EI2 = (1 - z(i+1)/L) * E*I0_other;
            EIx[i] = Poly(2, EI2-EI1, EI1);
        }
    }


    PolyVec EIy(nodes-1);
    for (int i = 0; i < nodes-1; i++) {

        if (nEI == 0){

            EIy[i] = Poly(1, E*I0);

        } else if (nEI == 1){

            double EI1 = (1 - z(i)/L) * E*I0;
            double EI2 = (1 - z(i+1)/L) * E*I0;
            EIy[i] = Poly(2, EI2-EI1, EI1);
        }
    }

    PolyVec EA(nodes-1);
    for (int i = 0; i < nodes-1; i++) {
        EA[i] = Poly(1, 1.0);
    }

    PolyVec GJ(nodes-1);
    for (int i = 0; i < nodes-1; i++) {
        GJ[i] = Poly(1, 1.0);
    }

    PolyVec rhoA(nodes-1);
    for (int i = 0; i < nodes-1; i++) {
        rhoA[i] = Poly(1, 1.0);
    }

    PolyVec rhoJ(nodes-1);
    for (int i = 0; i < nodes-1; i++) {
        rhoJ[i] = Poly(1, 1.0);
    }

    PolyVec Px(nodes-1);
    for (int i = 0; i < nodes-1; i++) {
        Px[i] = Poly(1, 0.0);
    }

    PolyVec Py(nodes-1);
    for (int i = 0; i < nodes-1; i++) {
        Py[i] = Poly(1, 0.0);
    }


    Vector Fx(nodes), Fy(nodes), Fz(nodes), Mx(nodes), My(nodes), Mz(nodes);
    Fx.setZero();
    Fy.setZero();
    Fz.setZero();
    Mx.setZero();
    My.setZero();
    Mz.setZero();

    double Pcr_x, Pcr_y;
    double Pcr_x_prev;

    double m;

    for (m = m_start; m <= m_finish; m += m_delta){

        double q0;
        if (nq == 0){
            q0 = -m * E * I0 / L / L / L;

        } else if (nq == 1){
            q0 = -m * E * I0 / L / L / L * 2.0;
        }

        PolyVec Pz(nodes-1);
        for (int i = 0; i < nodes-1; i++) {


            if (nq == 0){

                Pz[i] = Poly(1, q0);

            } else if (nq == 1){

                double Pz1 = (1 - z(i)/L) * q0;
                double Pz2 = (1 - z(i+1)/L) * q0;
                Pz[i] = Poly(2, Pz2-Pz1, Pz1);
            }
        }

        PolynomialSectionData sec = PolynomialSectionData(z, EA, EIx, EIy, GJ, rhoA, rhoJ);
        PolynomialLoads loads = PolynomialLoads(Px, Py, Pz, Fx, Fy, Fz, Mx, My, Mz);
        Beam beam = Beam(sec, loads, tip, base);



        beam.computeMinBucklingLoads(Pcr_x, Pcr_y);

        if (Pcr_x < 0){ // buckled
            break;
        }

        Pcr_x_prev = Pcr_x;
    }

    double m_prev = m - m_delta;
    double m_cr = m_prev + (0.0 - Pcr_x_prev)/(Pcr_x-Pcr_x_prev) * (m - m_prev);

    return m_cr;

}



// Test data from "Theory of Elastic Stability", Timoshenko, pg. 138-139
// for a tapered beam buckling under self-weight
TEST_CASE( "buckling_tapered" ){

    int nEI = 0;
    int nq = 0;
    double q0_start = 7.5;
    double q0_finish = 8.0;
    double q0_delta = 0.01;
    double m = computeCriticalFactorForTaperedBeamBuckling(nEI, nq, q0_start, q0_finish, q0_delta);

    REQUIRE(m == Approx(7.84).epsilon(0.06));


    nEI = 1;
    nq = 0;
    q0_start = 5.5;
    q0_finish = 6.0;
    q0_delta = 0.01;
    m = computeCriticalFactorForTaperedBeamBuckling(nEI, nq, q0_start, q0_finish, q0_delta);

    REQUIRE(m == Approx(5.78).epsilon(0.06));

    nEI = 0;
    nq = 1;
    q0_start = 15.5;
    q0_finish = 16.5;
    q0_delta = 0.01;
    m = computeCriticalFactorForTaperedBeamBuckling(nEI, nq, q0_start, q0_finish, q0_delta);

    REQUIRE(m == Approx(16.1).epsilon(0.06));


    nEI = 1;
    nq = 1;
    q0_start = 12.0;
    q0_finish = 14.0;
    q0_delta = 0.01;
    m = computeCriticalFactorForTaperedBeamBuckling(nEI, nq, q0_start, q0_finish, q0_delta);

    REQUIRE(m == Approx(13.0).epsilon(0.1));

}


// Test data from "Mechanical of Materials", Gere, 6th ed., pg. 273
// cantilevered beam with linear distributed load
TEST_CASE( "shear_bending_simple" ){

    double L = 10.0;
    double q0 = 3.0;

    int n = 1;

    int nodes = n+1;

    Vector z(nodes);
    for (int i = 0; i < nodes; i++) {
        z(i) = (double)i/(nodes-1) * L;
    }

    Vector EIx(nodes);
    for (int i = 0; i < nodes; i++) {
        EIx(i) = 1.0;
    }

    Vector EIy(nodes);
    for (int i = 0; i < nodes; i++) {
        EIy(i) = 1.0;
    }

    Vector EA(nodes);
    for (int i = 0; i < nodes; i++) {
        EA(i) = 1.0;
    }

    Vector GJ(nodes);
    for (int i = 0; i < nodes; i++) {
        GJ(i) = 1.0;
    }

    Vector rhoA(nodes);
    for (int i = 0; i < nodes; i++) {
        rhoA(i) = 1.0;
    }

    Vector rhoJ(nodes);
    for (int i = 0; i < nodes; i++) {
        rhoJ(i) = 1.0;
    }

    Vector Px(nodes);
    Vector Py(nodes);
    Vector Pz(nodes);
    Px.setZero();
    Py.setZero();
    Pz.setZero();

    for (int i = 0; i < nodes; i++) {
        Px(i) = q0*(1 - z(i)/L);
    }

    Loads loads = Loads(Px, Py, Pz);

    TipData tip;

    BaseData base;
    for (int i = 0; i < 6; i++) {
        base.rigid[i] = true;
    }

    SectionData sec = SectionData(z, EA, EIx, EIy, GJ, rhoA, rhoJ);

    Beam beam = Beam(sec, loads, tip, base);

    PolyVec Vx, Vy, Fz, Mx, My, Tz;

    beam.shearAndBending(Vx, Vy, Fz, Mx, My, Tz);


    double tol_pct = 1e-8;
    REQUIRE(Vx[0](0) == Approx(q0*L/2.0).epsilon(tol_pct));
    REQUIRE(Vx[0](1) == Approx(-q0*L).epsilon(tol_pct));
    REQUIRE(Vx[0](2) == Approx(q0*L/2.0).epsilon(tol_pct));

    REQUIRE(Mx[0](0) == Approx(-q0*L*L/6.0).epsilon(tol_pct));
    REQUIRE(Mx[0](1) == Approx(3.0*q0*L*L/6.0).epsilon(tol_pct));
    REQUIRE(Mx[0](2) == Approx(-3.0*q0*L*L/6.0).epsilon(tol_pct));
    REQUIRE(Mx[0](3) == Approx(q0*L*L/6.0).epsilon(tol_pct));


}



// Test data from "Mechanical of Materials", Gere, 6th ed., pg. 288
// cantilevered beam with two point loads
TEST_CASE( "shear_bending_simple_pt" ){

    double L = 10.0;
    double P1 = 2.0;
    double P2 = 3.0;

    int n = 3;

    int nodes = n+1;

    Vector z(nodes);
    for (int i = 0; i < nodes; i++) {
        z(i) = (double)i/(nodes-1) * L;
    }

    Vector EIx(nodes);
    for (int i = 0; i < nodes; i++) {
        EIx(i) = 1.0;
    }

    Vector EIy(nodes);
    for (int i = 0; i < nodes; i++) {
        EIy(i) = 1.0;
    }

    Vector EA(nodes);
    for (int i = 0; i < nodes; i++) {
        EA(i) = 1.0;
    }

    Vector GJ(nodes);
    for (int i = 0; i < nodes; i++) {
        GJ(i) = 1.0;
    }

    Vector rhoA(nodes);
    for (int i = 0; i < nodes; i++) {
        rhoA(i) = 1.0;
    }

    Vector rhoJ(nodes);
    for (int i = 0; i < nodes; i++) {
        rhoJ(i) = 1.0;
    }

    Vector ESx(nodes), ESy(nodes), EIxy(nodes);
    ESx.setZero();
    ESy.setZero();
    EIxy.setZero();

    Vector Px(nodes);
    Vector Py(nodes);
    Vector Pz(nodes);
    Px.setZero();
    Py.setZero();
    Pz.setZero();

    Vector Fx_pt(nodes), Fy_pt(nodes), Fz_pt(nodes), Mx_pt(nodes), My_pt(nodes), Mz_pt(nodes);
    Fx_pt.setZero();
    Fy_pt.setZero();
    Fz_pt.setZero();
    Mx_pt.setZero();
    My_pt.setZero();
    Mz_pt.setZero();

    Fx_pt(1) = -P2;
    Fx_pt(3) = -P1;

    Loads loads = Loads(Px, Py, Pz, Fx_pt, Fy_pt, Fz_pt, Mx_pt, My_pt, Mz_pt);

    TipData tip;

    BaseData base;
    for (int i = 0; i < 6; i++) {
        base.rigid[i] = true;
    }

    SectionData sec = SectionData(z, EA, EIx, EIy, GJ, rhoA, rhoJ);

    Beam beam = Beam(sec, loads, tip, base);

    PolyVec Vx, Vy, Fz, Mx, My, Tz;

    beam.shearAndBending(Vx, Vy, Fz, Mx, My, Tz);


    double tol_pct = 1e-8;
    REQUIRE(Vx[0].length()== 3);
    REQUIRE(Vx[0](0) == Approx(0.0).epsilon(tol_pct));
    REQUIRE(Vx[0](1) == Approx(0.0).epsilon(tol_pct));
    REQUIRE(Vx[0](2) == Approx(-P1-P2).epsilon(tol_pct));

    REQUIRE(Vx[1](0) == Approx(0.0).epsilon(tol_pct));
    REQUIRE(Vx[1](1) == Approx(0.0).epsilon(tol_pct));
    REQUIRE(Vx[1](2) == Approx(-P1).epsilon(tol_pct));

    REQUIRE(Vx[2](0) == Approx(0.0).epsilon(tol_pct));
    REQUIRE(Vx[2](1) == Approx(0.0).epsilon(tol_pct));
    REQUIRE(Vx[2](2) == Approx(-P1).epsilon(tol_pct));

    double b = L/3.0;
    double a = 2.0/3.0*L;

    REQUIRE(Mx[0](0) == Approx(0.0).epsilon(tol_pct));
    REQUIRE(Mx[0](1) == Approx(0.0).epsilon(tol_pct));
    REQUIRE(Mx[0](2) == Approx(-P1*a + P1*L + P2*b).epsilon(tol_pct));
    REQUIRE(Mx[0](3) == Approx(-P1*L - P2*b).epsilon(tol_pct));

    REQUIRE(Mx[1](0) == Approx(0.0).epsilon(tol_pct));
    REQUIRE(Mx[1](1) == Approx(0.0).epsilon(tol_pct));
    REQUIRE(Mx[1](2) == Approx(-0.5*P1*a + P1*a).epsilon(tol_pct));
    REQUIRE(Mx[1](3) == Approx(-P1*a).epsilon(tol_pct));

    REQUIRE(Mx[2](0) == Approx(0.0).epsilon(tol_pct));
    REQUIRE(Mx[2](1) == Approx(0.0).epsilon(tol_pct));
    REQUIRE(Mx[2](2) == Approx(0.5*P1*a).epsilon(tol_pct));
    REQUIRE(Mx[2](3) == Approx(-0.5*P1*a).epsilon(tol_pct));


}


// Test data from Rick Damiani's ANSYS model
// tower with springs at base, and offset mass
TEST_CASE( "ricks_tower" ){

    double dmp = 6.2;
    double dtw_base = 6.424;
    double dtw_top = dtw_base * 0.55;

    double E = 210.0e9;
    double G = 80.769e9;
    double rho = 8502.0;

    double E_grout = 3.9e10;
    double rho_grout = 2500.0;
    double G_grout = 1.5e10;

    Vector z(100);
    z(0) = 0.0;
    PolyVec EIx(100);
    PolyVec EIy(100);
    PolyVec EA(100);
    PolyVec GJ(100);
    PolyVec rhoA(100);
    PolyVec rhoJ(100);



    int idx = 0;

    // monopile bottom
    int n_mp_bot = 2;
    double I = M_PI / 8.0 * pow(dmp, 3) * dmp/76.938;
    double A = M_PI * dmp * dmp/76.938;
    for (int i = 1; i <= n_mp_bot; i++){
        z(idx+i) = z(idx) + (double)i/n_mp_bot*(2.0*dmp);
        EIx[idx+i-1] = Poly(1, E*I);
        EIy[idx+i-1] = Poly(1, E*I);
        EA[idx+i-1] = Poly(1, E*A);
        GJ[idx+i-1] = Poly(1, G*(I+I));
        rhoA[idx+i-1] = Poly(1, rho*A);
        rhoJ[idx+i-1] = Poly(1, rho*(I+I));
    }
    idx += n_mp_bot;

    // monopile bottom
    int n_mp_top = 2;
    I = M_PI / 8.0 * pow(dmp, 3) * dmp/100.0;
    A = M_PI * dmp * dmp/100.0;
    for (int i = 1; i <= n_mp_top; i++){
        z(idx+i) = z(idx) + (double)i/n_mp_top*(25.0 + 5.0 - 0.5*dmp - 2.0*dmp);
        EIx[idx+i-1] = Poly(1, E*I);
        EIy[idx+i-1] = Poly(1, E*I);
        EA[idx+i-1] = Poly(1, E*A);
        GJ[idx+i-1] = Poly(1, G*(I+I));
        rhoA[idx+i-1] = Poly(1, rho*A);
        rhoJ[idx+i-1] = Poly(1, rho*(I+I));
    }
    idx += n_mp_top;

    // overlap
    int n_ov = 1;
    double I1 = M_PI / 8.0 * pow(dtw_base, 3) * 0.062;
    double I2 = M_PI / 8.0 * pow(dmp, 3) * dmp/100.0;
    double I3 = M_PI / 64.0 * (pow(dtw_base-0.062, 4) - pow(dmp+dmp/100.0, 4));
    double A1 = M_PI * dtw_base * 0.062;
    double A2 = M_PI * dmp * dmp/100.0;
    double A3 = M_PI / 4.0 * (pow(dtw_base-0.062, 2) - pow(dmp+dmp/100.0, 2));
    for (int i = 1; i <= n_ov; i++){
        z(idx+i) = z(idx) + (double)i/n_ov*(0.5*dmp);
        EIx[idx+i-1] = Poly(1, E*I1 + E*I2 + E_grout*I3);
        EIy[idx+i-1] = Poly(1, E*I1 + E*I2 + E_grout*I3);
        EA[idx+i-1] = Poly(1, E*A1 + E*A2 + E_grout*A3);
        GJ[idx+i-1] = Poly(1, G*2*I1 + G*2*I2 + G_grout*2*I3);
        rhoA[idx+i-1] = Poly(1, rho*A1 + rho*A2 + rho_grout*A3);
        rhoJ[idx+i-1] = Poly(1, rho*2*I1 + rho*2*I2 + rho_grout*2*I3);
    }
    idx += n_ov;

    // transition
    int n_ts = 4;
    I = M_PI / 8.0 * pow(dtw_base, 3) * 0.062;
    A = M_PI * dtw_base * 0.062;
    for (int i = 1; i <= n_ts; i++){
        z(idx+i) = z(idx) + (double)i/n_ts*(18.05 - 0.5*dmp);
        EIx[idx+i-1] = Poly(1, E*I);
        EIy[idx+i-1] = Poly(1, E*I);
        EA[idx+i-1] = Poly(1, E*A);
        GJ[idx+i-1] = Poly(1, G*(I+I));
        rhoA[idx+i-1] = Poly(1, rho*A);
        rhoJ[idx+i-1] = Poly(1, rho*(I+I));
    }
    idx += n_ts;

    // tower
    int n_tw = 16;
    for (int i = 1; i <= n_tw; i++){
        z(idx+i) = z(idx) + (double)i/n_tw*(88.37);
        double d_bot = dtw_base + (double)(i-1)/n_tw*(dtw_top-dtw_base);
        double d_top = dtw_base + (double)(i)/n_tw*(dtw_top-dtw_base);
        Poly d = Poly(2, d_top-d_bot, d_bot);
        Poly t = d / 120.0;
        Poly Itw = M_PI / 8.0 * d * d * d* t;
        Poly Atw = M_PI * d * t;
        EIx[idx+i-1] = E * Itw;
        EIy[idx+i-1] = E * Itw;
        EA[idx+i-1] = E * Atw;
        GJ[idx+i-1] = G * 2*Itw;
        rhoA[idx+i-1] = rho * Atw;
        rhoJ[idx+i-1] = rho * 2*Itw;
    }
    idx += n_tw;

    int nodes = n_mp_bot + n_mp_top + n_ov + n_ts + n_tw + 1;
    z.resize(nodes);
    EIx.resize(nodes-1);
    EIy.resize(nodes-1);
    EA.resize(nodes-1);
    GJ.resize(nodes-1);
    rhoA.resize(nodes-1);
    rhoJ.resize(nodes-1);

    PolyVec Px(nodes-1);
    PolyVec Py(nodes-1);
    PolyVec Pz(nodes-1);
    //Px.setZero();
    //Py.setZero();
    //Pz.setZero();

    Vector Fx_pt(nodes), Fy_pt(nodes), Fz_pt(nodes), Mx_pt(nodes), My_pt(nodes), Mz_pt(nodes);
    Fx_pt.setZero();
    Fy_pt.setZero();
    Fz_pt.setZero();
    Mx_pt.setZero();
    My_pt.setZero();
    Mz_pt.setZero();

    double m = 5.7380e5;
    double Ixx = 86.579e6;
    double Iyy = 53.530e6;
    double Izz = 58.112e6;
    double Itip[] = {Ixx, Iyy, Izz, 0.0, 0.0, 0.0};
    double cm[] = {0.0, 0.0, 2.34};
    double F[] = {2000.0e3, 0.0, 0.0};
    double M[] = {0.0, 0.0, 0.0};

//    m = 0.0;
//    Itip[0] = 0.0;
//    Itip[1] = 0.0;
//    Itip[2] = 0.0;

    TipData tip(m, cm, Itip, F, M);

    double kx = 4.72e8;
    double ktx = 1.27e11;
    BaseData base(kx, ktx, kx, ktx, 999, 999, 999);
    //BaseData base(999, 999, 999, 999, 999, 999, 999);

    PolynomialSectionData sec = PolynomialSectionData(z, EA, EIx, EIy, GJ, rhoA, rhoJ);
    PolynomialLoads loads = PolynomialLoads(Px, Py, Pz, Fx_pt, Fy_pt, Fz_pt, Mx_pt, My_pt, Mz_pt);
    Beam beam = Beam(sec, loads, tip, base);

    int nFreq = 100;
    Vector freq(nFreq);

//    beam.computeNaturalFrequencies(nFreq, freq);
//    double mass = beam.computeMass();
//    double Pcr_x, Pcr_y;
//    beam.computeMinBucklingLoads(Pcr_x, Pcr_y);

//    std::cout << mass << std::endl;
//    for (int i = 0; i < 20; i++){
//        std::cout << freq(i) << std::endl;
//    }
//    std::cout << Pcr_x << std::endl;



//    beam.computeAxialStress(<#Vector &x#>, <#Vector &y#>, <#Vector &z#>, <#Vector &E#>, <#Vector &sigma_axial#>)

//    double tol_pct = 6e-6*100;
    //REQUIRE(freq(1), sqrt(0.59919 / alpha) / (2*M_PI)).epsilon(tol_pct));

}
