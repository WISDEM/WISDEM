//
//  testBeam.cpp
//  pbeam
//
//  Created by Andrew Ning on 2/6/12.
//  Copyright (c) 2012 NREL. All rights reserved.
//

#include "catch.hpp"

#include "BeamFEA.h"

using namespace BeamFEA;

TEST_CASE( "matrix_assem" ){
    
        
    Poly I = Poly(5, 0.000003946681095,  -0.000466722832956,   0.020651374840442,  -0.405297025319252,   2.977287357807047);
    Poly f[4] = { 
        Poly(2, 12.0, -6.0),
        Poly(2, 52.56, -35.04),
        Poly(2, -12.0, 6.0),
        Poly(2, 52.56, -17.52)
    };
    
    double constant = 3.123970874737551e8;

    Matrix K(4, 4);
    
    matrixAssembly(I, 4, f, constant, K);
    
    double tol = 1e-8;
    REQUIRE(K(0, 0)/1e11 == Approx(0.104318282670074).epsilon(tol));
    REQUIRE(K(0, 1)/1e11 == Approx(0.467451693836563).epsilon(tol));
    REQUIRE(K(0, 2)/1e11 == Approx(-0.104318282670074).epsilon(tol));
    REQUIRE(K(0, 3)/1e11 == Approx(0.446376462353287).epsilon(tol));
    
    REQUIRE(K(1, 0)/1e11 == Approx(0.467451693836563).epsilon(tol));
    REQUIRE(K(1, 1)/1e11 == Approx(2.760368765600540).epsilon(tol));
    REQUIRE(K(1, 2)/1e11 == Approx(-0.467451693836563).epsilon(tol));
    REQUIRE(K(1, 3)/1e11 == Approx(1.334508072407750).epsilon(tol));
    
    REQUIRE(K(2, 0)/1e11 == Approx(-0.104318282670074).epsilon(tol));
    REQUIRE(K(2, 1)/1e11 == Approx(-0.467451693836563).epsilon(tol));
    REQUIRE(K(2, 2)/1e11 == Approx(0.104318282670074).epsilon(tol));
    REQUIRE(K(2, 3)/1e11 == Approx(-0.446376462353287).epsilon(tol));
    
    REQUIRE(K(3, 0)/1e11 == Approx(0.446376462353287).epsilon(tol));
    REQUIRE(K(3, 1)/1e11 == Approx(1.334508072407750).epsilon(tol));
    REQUIRE(K(3, 2)/1e11 == Approx(-0.446376462353287).epsilon(tol));
    REQUIRE(K(3, 3)/1e11 == Approx(2.575749737807040).epsilon(tol));

    
    
}




TEST_CASE( "vec_assem" ){
    
    Poly p(2, 0.790973161559091e4,   6.393453731449331e4);
    Poly f[4] = {
        Poly(4, 2.0, -3.0, 0.0, 1.0),
        Poly(4, 8.76, -17.52, 8.76, 0.0),
        Poly(4, -2.0, 3.0, 0.0, 0.0),
        Poly(4, 8.76, -8.76, 0.0, 0.0)
    };
    
    double constant = 8.76;
    Vector F(4);
    
    
    vectorAssembly(p, 4, f, constant, F);
    
    double tol = 1e-8;
    REQUIRE(F(0)/1e5 == Approx(2.904266607803672).epsilon(tol));
    REQUIRE(F(1)/1e5 == Approx(4.290810399128741).epsilon(tol));
    REQUIRE(F(2)/1e5 == Approx(3.042845105708824).epsilon(tol));
    REQUIRE(F(3)/1e5 == Approx(-4.391972702599503).epsilon(tol));
    
}




TEST_CASE( "int_comp_loads" ){
    
    
    double Pz0 = 4.51;
    double Pz1 = 6.28;
    double Pz2 = 5.13;

    Vector z(3);
    z(0) = 1.47;
    z(1) = 7.30;
    z(2) = 8.11;
    
    
    Poly Poly0(2, (Pz1-Pz0), Pz0);
    Poly Poly1(2, (Pz2-Pz1), Pz1);
    PolyVec Pz(2);
    Pz[0] = Poly0;
    Pz[1] = Poly1;
    
    PolyVec FzFromPz;
    integrateDistributedCompressionLoads(z, Pz, FzFromPz);
    
    //double Fz2 = 0.0;
    double Fz1 = (z(2) - z(1))/2.0*(Pz2 + Pz1);
    double Fz0 = (z(1) - z(0))/2.0*(Pz1 + Pz0) + Fz1;
    
    double tol = 1e-8;
    REQUIRE(FzFromPz[1].eval(1.0) <= tol);
    REQUIRE(Fz1 == Approx(FzFromPz[1].eval(0.0)).epsilon(tol));
    REQUIRE(Fz1 == Approx(FzFromPz[0].eval(1.0)).epsilon(tol));
    REQUIRE(Fz0 == Approx(FzFromPz[0].eval(0.0)).epsilon(tol));
    

    
}




TEST_CASE( "beam" ){
    
    double L = 8.76;
    Poly EIx = Poly(5, 0.000008288030298e11, -0.000980117949207e11, 0.043367887164929e11, 
        -0.851123753170429e11, 6.252303451394798e11);
    Poly EIy = Poly(5, 0.000008288030298e11, -0.000980117949207e11, 0.043367887164929e11, 
        -0.851123753170429e11, 6.252303451394798e11);
    Poly EA = Poly(3, 0.001461443769709e11, -0.090491157360311e11, 1.389400766976622e11);
    Poly GJ = Poly(5, 0.000006377836649e11, -0.000754224098056e11, 0.033372621742155e11,
        -0.654959992915911e11, 4.811296370216187e11);
    Poly rhoA = Poly(3, 0.005915367639297e3, -0.366273732172686e3, 5.623765009191089e3);
    Poly rhoJ = Poly(5, 0.000006709357861e4, -0.000793428816025e4, 0.035107337228752e4, 
        -0.689004943042728e4, 5.061388508271980e4);
    double Px1 = 6.393453731449331e4;
    double Px2 = 7.184426893008422e4;
    double Py1 = 0.0;
    double Py2 = 0.0;
    double Pz1 = -4.315789462165175e5;
    double Pz2 = -3.764098114763530e5;
    Poly Pxlinear = Poly(2, Px2-Px1, Px1);
    Poly Pylinear = Poly(2, Py2-Py1, Py1);
    Poly Pzlinear = Poly(2, Pz2-Pz1, Pz1);
    Matrix K(2*DOF, 2*DOF);
    Matrix M(2*DOF, 2*DOF);
    Matrix Ndist(2*DOF, 2*DOF);
    Matrix Nconst(2*DOF, 2*DOF);
    Vector F(2*DOF);
    

    beamMatrix(L, EIx, EIy, EA, GJ, rhoA, rhoJ, Pxlinear, Pylinear, Pzlinear, K, M, Ndist, Nconst, F);
    
    double tol = 1e-8;
    REQUIRE(K(0, 0)/1e14 == Approx(0.000104318282670).epsilon(tol));
    REQUIRE(K(0, 1)/1e14 == Approx(0.000467451693837).epsilon(tol));
    REQUIRE(K(3, 3)/1e14 == Approx(0.002760368765601).epsilon(tol));
    REQUIRE(K(4, 4)/1e14 == Approx(0.000153497983587).epsilon(tol));
    REQUIRE(K(4, 5)/1e14 == Approx(0.0).epsilon(tol));
    REQUIRE(K(5, 5)/1e14 == Approx(0.000513099691843).epsilon(tol));
    REQUIRE(K(0, 6)/1e14 == Approx(-0.000104318282670).epsilon(tol));
    REQUIRE(K(0, 7)/1e14 == Approx(0.000446376462353).epsilon(tol));
    REQUIRE(K(6, 7)/1e14 == Approx(-0.000446376462353).epsilon(tol));
    REQUIRE(K(7, 7)/1e14 == Approx(0.002575749737807).epsilon(tol));
    REQUIRE(K(9, 9)/1e14 == Approx(0.002575749737807).epsilon(tol));
    REQUIRE(K(8, 9)/1e14 == Approx(-0.000446376462353).epsilon(tol));
    REQUIRE(K(10, 10)/1e14 == Approx(0.000153497983587).epsilon(tol));
    REQUIRE(K(11, 11)/1e14 == Approx(0.000513099691843).epsilon(tol));
    
    REQUIRE(M(0, 0)/1e5 == Approx(0.180246680856558).epsilon(tol));
    REQUIRE(M(0, 1)/1e5 == Approx(0.221398343405622).epsilon(tol));
    REQUIRE(M(3, 3)/1e5 == Approx(0.351309181440035).epsilon(tol));
    REQUIRE(M(4, 4)/1e5 == Approx(0.161557412897026).epsilon(tol));
    REQUIRE(M(4, 5)/1e5 == Approx(0.0).epsilon(tol));
    REQUIRE(M(5, 5)/1e5 == Approx(1.428641689734879).epsilon(tol));
    REQUIRE(M(0, 6)/1e5 == Approx(0.061295935452276).epsilon(tol));
    REQUIRE(M(0, 7)/1e5 == Approx(-0.129595253157960).epsilon(tol));
    REQUIRE(M(6, 7)/1e5 == Approx(-0.216131098446723).epsilon(tol));
    REQUIRE(M(7, 7)/1e5 == Approx(0.345541548210046).epsilon(tol));
    REQUIRE(M(9, 9)/1e5 == Approx(0.345541548210046).epsilon(tol));
    REQUIRE(M(8, 9)/1e5 == Approx(-0.216131098446723).epsilon(tol));
    REQUIRE(M(10, 10)/1e5 == Approx(0.156296180774838).epsilon(tol));
    REQUIRE(M(11, 11)/1e5 == Approx(1.333069166392610).epsilon(tol));

    // TODO: update this
//    REQUIRE(Ndist(0, 0)/1e6 == Approx(0.235303452841269).epsilon(tol));
//    REQUIRE(Ndist(0, 1)/1e6 == Approx(-0.006904023147497).epsilon(tol));
//    REQUIRE(Ndist(3, 3)/1e6 == Approx(3.069916283230640).epsilon(tol));
//    REQUIRE(Ndist(4, 4)/1e6 == Approx(0.0).epsilon(tol));
//    REQUIRE(Ndist(4, 5)/1e6 == Approx(0.0).epsilon(tol));
//    REQUIRE(Ndist(5, 5)/1e6 == Approx(0.0).epsilon(tol));
//    REQUIRE(Ndist(0, 6)/1e6 == Approx(-0.235303452841269).epsilon(tol));
//    REQUIRE(Ndist(0, 7)/1e6 == Approx(0.346995052721988).epsilon(tol));
//    REQUIRE(Ndist(6, 7)/1e6 == Approx(-0.346995052721988).epsilon(tol));
//    REQUIRE(Ndist(7, 7)/1e6 == Approx(1.003145680152928).epsilon(tol));
//    REQUIRE(Ndist(9, 9)/1e6 == Approx(1.003145680152928).epsilon(tol));
//    REQUIRE(Ndist(8, 9)/1e6 == Approx(-0.346995052721988).epsilon(tol));
//    REQUIRE(Ndist(10, 10)/1e6 == Approx(0.0).epsilon(tol));
//    REQUIRE(Ndist(11, 11)/1e6 == Approx(0.0).epsilon(tol));

    REQUIRE(Nconst(0, 0) == Approx(12.0/10/L).epsilon(tol));
    REQUIRE(Nconst(0, 1) == Approx(1.0/10).epsilon(tol));
    REQUIRE(Nconst(3, 3) == Approx(4.0*L/3/10.0).epsilon(tol));
    REQUIRE(Nconst(4, 4) == Approx(0.0).epsilon(tol));
    REQUIRE(Nconst(4, 5) == Approx(0.0).epsilon(tol));
    REQUIRE(Nconst(5, 5) == Approx(0.0).epsilon(tol));
    REQUIRE(Nconst(0, 6) == Approx(-12.0/L/10.0).epsilon(tol));
    REQUIRE(Nconst(0, 7) == Approx(1.0/10.0).epsilon(tol));
    REQUIRE(Nconst(6, 7) == Approx(-1.0/10.0).epsilon(tol));
    REQUIRE(Nconst(7, 7) == Approx(4.0*L/3/10.0).epsilon(tol));
    REQUIRE(Nconst(9, 9) == Approx(4.0*L/3/10.0).epsilon(tol));
    REQUIRE(Nconst(8, 9) == Approx(-1.0/10.0).epsilon(tol));
    REQUIRE(Nconst(10, 10) == Approx(0.0).epsilon(tol));
    REQUIRE(Nconst(11, 11) == Approx(0.0).epsilon(tol));
    
    // TODO: update those also
    REQUIRE(F(0)/1e6 == Approx(0.290426660780367).epsilon(tol));
    REQUIRE(F(1)/1e6 == Approx(0.429081039912874).epsilon(tol));
    REQUIRE(F(2)/1e6 == Approx(0.0).epsilon(tol));
    REQUIRE(F(3)/1e6 == Approx(0.0).epsilon(tol));
//    REQUIRE(F(4)/1e6 == Approx(-1.809768847707706).epsilon(tol));
    REQUIRE(F(5)/1e6 == Approx(0.0).epsilon(tol));
    REQUIRE(F(6)/1e6 == Approx(0.304284510570882).epsilon(tol));
    REQUIRE(F(7)/1e6 == Approx(-0.439197270259950).epsilon(tol));
    REQUIRE(F(8)/1e6 == Approx(0.0).epsilon(tol));
    REQUIRE(F(9)/1e6 == Approx(0.0).epsilon(tol));
//    REQUIRE(F(10)/1e6 == Approx(-1.729221910987066).epsilon(tol));
    REQUIRE(F(11)/1e6 == Approx(0.0).epsilon(tol));
    
}






TEST_CASE( "tip" ){
    
    
    TipData tip;
    
    tip.m = 7.80558e4;
    tip.cm_offsetX = 20.0;
    tip.cm_offsetY = 50.0;
    tip.cm_offsetZ = -40.0;
    tip.Ixx = 2960437.0;
    tip.Iyy = 3253223.0;
    tip.Izz = 3264220.0;
    tip.Ixy = 20000.0;
    tip.Ixz = -18400.0;
    tip.Iyz = 40000.0;
    tip.Fx = 1000000;
    tip.Fy = 10000.0;
    tip.Fz = -7.65727398e+05;
    tip.Mx = 2.4e6;
    tip.My = 4.2e6;
    tip.Mz = 1.8e6;
    
    
    int nodes = 5;
    Matrix M(DOF*nodes, DOF*nodes);
    Vector F(DOF*nodes);
    
    M.setZero();
    F.setZero();
    
    addTipMassContribution(tip, nodes, M, F);
    
    int idx = DOF*nodes - 6;
    
    // TODO: update
    double tol = 1e-8;
    REQUIRE(M(idx+0, idx+0)/1e8 == Approx(0.000780558).epsilon(tol));
//    REQUIRE(M(idx+0, idx+1)/1e8 == Approx(0.0).epsilon(tol));
    REQUIRE(M(idx+0, idx+2)/1e8 == Approx(0.0).epsilon(tol));
//    REQUIRE(M(idx+0, idx+3)/1e8 == Approx(-0.031222320).epsilon(tol));
    REQUIRE(M(idx+0, idx+4)/1e8 == Approx(0.0).epsilon(tol));
    REQUIRE(M(idx+0, idx+5)/1e8 == Approx(-0.03902790).epsilon(tol));
    
//    REQUIRE(M(idx+1, idx+0)/1e8 == Approx(0.0).epsilon(tol));
//    REQUIRE(M(idx+1, idx+1)/1e8 == Approx(3.22989217).epsilon(tol));
//    REQUIRE(M(idx+1, idx+2)/1e8 == Approx(0.0312223200).epsilon(tol));
//    REQUIRE(M(idx+1, idx+3)/1e8 == Approx(-0.7803580).epsilon(tol));
//    REQUIRE(M(idx+1, idx+4)/1e8 == Approx(0.03902790).epsilon(tol));
//    REQUIRE(M(idx+1, idx+5)/1e8 == Approx(0.62426240).epsilon(tol));
    
    REQUIRE(M(idx+2, idx+0)/1e8 == Approx(0.0).epsilon(tol));
//    REQUIRE(M(idx+2, idx+1)/1e8 == Approx(0.0312223200).epsilon(tol));
    REQUIRE(M(idx+2, idx+2)/1e8 == Approx(0.0007805580).epsilon(tol));
//    REQUIRE(M(idx+2, idx+3)/1e8 == Approx(0.0).epsilon(tol));
    REQUIRE(M(idx+2, idx+4)/1e8 == Approx(0.0).epsilon(tol));
    REQUIRE(M(idx+2, idx+5)/1e8 == Approx(0.015611160).epsilon(tol));
    
//    REQUIRE(M(idx+3, idx+0)/1e8 == Approx(-0.0312223200).epsilon(tol));
//    REQUIRE(M(idx+3, idx+1)/1e8 == Approx(-0.78035800).epsilon(tol));
//    REQUIRE(M(idx+3, idx+2)/1e8 == Approx(0.0).epsilon(tol));
//    REQUIRE(M(idx+3, idx+3)/1e8 == Approx(1.5936482300).epsilon(tol));
//    REQUIRE(M(idx+3, idx+4)/1e8 == Approx(-0.0156111600).epsilon(tol));
//    REQUIRE(M(idx+3, idx+5)/1e8 == Approx(1.5615160).epsilon(tol));
    
    REQUIRE(M(idx+4, idx+0)/1e8 == Approx(0.0).epsilon(tol));
//    REQUIRE(M(idx+4, idx+1)/1e8 == Approx(0.03902790).epsilon(tol));
    REQUIRE(M(idx+4, idx+2)/1e8 == Approx(0.0).epsilon(tol));
//    REQUIRE(M(idx+4, idx+3)/1e8 == Approx(-0.015611160).epsilon(tol));
    REQUIRE(M(idx+4, idx+4)/1e8 == Approx(0.00078055800).epsilon(tol));
    REQUIRE(M(idx+4, idx+5)/1e8 == Approx(0.0).epsilon(tol));
    
    REQUIRE(M(idx+5, idx+0)/1e8 == Approx(-0.03902790).epsilon(tol));
//    REQUIRE(M(idx+5, idx+1)/1e8 == Approx(0.6242624000).epsilon(tol));
    REQUIRE(M(idx+5, idx+2)/1e8 == Approx(0.0156111600).epsilon(tol));
//    REQUIRE(M(idx+5, idx+3)/1e8 == Approx(1.56151600).epsilon(tol));
    REQUIRE(M(idx+5, idx+4)/1e8 == Approx(0.00).epsilon(tol));
    REQUIRE(M(idx+5, idx+5)/1e8 == Approx(2.296260400).epsilon(tol));
    
    
    
    REQUIRE(F(idx+0)/1e6 == Approx(1.0).epsilon(tol));
//    REQUIRE(F(idx+1)/1e6 == Approx(2.4).epsilon(tol));
    REQUIRE(F(idx+2)/1e6 == Approx(0.01).epsilon(tol));
//    REQUIRE(F(idx+3)/1e6 == Approx(4.2).epsilon(tol));
    REQUIRE(F(idx+4)/1e6 == Approx(-0.765727398).epsilon(tol));
    REQUIRE(F(idx+5)/1e6 == Approx(1.8).epsilon(tol));
    
    
}




TEST_CASE( "base" ){
    
    
    double Kbase[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    bool rigidDirections[] = {true, true, false, false, true, false};
    
    int nodes = 3;
    
    Matrix KFull(DOF*nodes, DOF*nodes);
    Matrix MFull(DOF*nodes, DOF*nodes);
    Matrix NotherFull(DOF*nodes, DOF*nodes);
    Matrix NFull(DOF*nodes, DOF*nodes);
    Vector FFull(DOF*nodes);
    
    for (int i = 0; i < DOF*nodes; i++) {
        for (int j = 0; j < DOF*nodes; j++) {
            KFull(i, j) = i*DOF*nodes + j;
        }
    }
    
    
    Matrix K, M, Nother, N;
    Vector F;
    
    applyBaseBoundaryCondition(Kbase, rigidDirections, nodes, KFull, MFull, NotherFull, NFull, FFull, K, M, Nother, N, F);
    
    int length = (int) F.size();
    
    double tol = 1e-8;
    REQUIRE(15 == length);
    REQUIRE(K(0, 0) == Approx(41.0).epsilon(tol));
    REQUIRE(K(0, 1) == Approx(39.0).epsilon(tol));
    REQUIRE(K(0, 2) == Approx(41.0).epsilon(tol));
    REQUIRE(K(0, 3) == Approx(42.0).epsilon(tol));
    REQUIRE(K(0, length-1) == Approx(53.0).epsilon(tol));
    REQUIRE(K(1, 0) == Approx(56.0).epsilon(tol));
    REQUIRE(K(1, 1) == Approx(61.0).epsilon(tol));
    REQUIRE(K(1, 2) == Approx(59.0).epsilon(tol));
    REQUIRE(K(1, 3) == Approx(60.0).epsilon(tol));
    REQUIRE(K(2, 0) == Approx(92.0).epsilon(tol));
    REQUIRE(K(3, 0) == Approx(110.0).epsilon(tol));
    REQUIRE(K(4, 0) == Approx(128.0).epsilon(tol));
    REQUIRE(K(length-1, length-1) == Approx(323).epsilon(tol));
}
