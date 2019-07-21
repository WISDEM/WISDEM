//
//  Beam.cpp
//  pbeam
//
//  Created by Andrew Ning on 2/6/12.
//  Copyright (c) 2012 NREL. All rights reserved.
//

#define _USE_MATH_DEFINES
#include <cmath> // for pi
#include <cstdio> // fprintf
#include <vector>
#include <iostream>
#include "Beam.h"

// MARK: ------------------- CONSTRUCTORS ----------------------------


// init with circular cross section data
Beam::Beam(const Vector &z_node, const Vector &d_node, const Vector &t_node,
           const Loads& loads, const IsotropicMaterial &mat,
           const TipData &tip, const BaseData &base){

    // get nodes
    nodes = (int) z_node.size();

    // save data
    this->z_node = z_node;
    Fx_node = loads.Fx;
    Fy_node = loads.Fy;
    Fz_node = loads.Fz;
    Mx_node = loads.Mx;
    My_node = loads.My;
    Mz_node = loads.Mz;

    this->tip = tip;
    this->base = base;

    // resize vectors
    EIxx.resize(nodes-1);
    EIyy.resize(nodes-1);
    EA.resize(nodes-1);
    GJ.resize(nodes-1);
    rhoA.resize(nodes-1);
    rhoJ.resize(nodes-1);
    Px.resize(nodes-1);
    Py.resize(nodes-1);
    Pz.resize(nodes-1);

    for (int i = 0; i < nodes-1; i++) {

        // linear variation in thickness and diameter
        Poly d = Poly(2, d_node(i+1) - d_node(i), d_node(i));
        Poly t = Poly(2, t_node(i+1) - t_node(i), t_node(i));

        // compute area polynomial
        Poly A = M_PI * d * t;

        // compute moment of inertia polynomial
        Poly I = M_PI/8.0 * t * d * d * d;

        // J = Ix + Iy (parallel axis theorem)
        Poly J = I + I;

        // setup section properties
        EIxx[i] = mat.E * I;
        EIyy[i] = mat.E * I;
        EA[i] = mat.E * A;
        GJ[i] = mat.G * J;
        rhoA[i] = mat.rho * A;
        rhoJ[i] = mat.rho * J;

        // linear variation in distributed loads
        Px[i] = Poly(2, loads.Px(i+1) - loads.Px(i), loads.Px(i));
        Py[i] = Poly(2, loads.Py(i+1) - loads.Py(i), loads.Py(i));
        Pz[i] = Poly(2, loads.Pz(i+1) - loads.Pz(i), loads.Pz(i));

    }

    translateFromGlobalToFEACoordinateSystem();
    assembleMatrices();

}


// init with general data
Beam::Beam(const SectionData& sec, const Loads& loads,
           const TipData &tip, const BaseData &base){

    nodes = sec.nodes;

    z_node = sec.z;
    Fx_node = loads.Fx;
    Fy_node = loads.Fy;
    Fz_node = loads.Fz;
    Mx_node = loads.Mx;
    My_node = loads.My;
    Mz_node = loads.Mz;

    this->tip = tip;
    this->base = base;

    // resize vectors
    EA.resize(nodes-1);
    EIxx.resize(nodes-1);
    EIyy.resize(nodes-1);
    GJ.resize(nodes-1);
    rhoA.resize(nodes-1);
    rhoJ.resize(nodes-1);
    Px.resize(nodes-1);
    Py.resize(nodes-1);
    Pz.resize(nodes-1);


    for (int i = 0; i < nodes-1; i++) {

        // define linear variation in properties
        EIxx[i] = Poly(2, sec.EIxx(i+1) - sec.EIxx(i), sec.EIxx(i));
        EIyy[i] = Poly(2, sec.EIyy(i+1) - sec.EIyy(i), sec.EIyy(i));
        EA[i] = Poly(2, sec.EA(i+1) - sec.EA(i), sec.EA(i));
        GJ[i] = Poly(2, sec.GJ(i+1) - sec.GJ(i), sec.GJ(i));
        rhoA[i] = Poly(2, sec.rhoA(i+1) - sec.rhoA(i), sec.rhoA(i));
        rhoJ[i] = Poly(2, sec.rhoJ(i+1) - sec.rhoJ(i), sec.rhoJ(i));
        Px[i] = Poly(2, loads.Px(i+1) - loads.Px(i), loads.Px(i));
        Py[i] = Poly(2, loads.Py(i+1) - loads.Py(i), loads.Py(i));
        Pz[i] = Poly(2, loads.Pz(i+1) - loads.Pz(i), loads.Pz(i));
    }

    translateFromGlobalToFEACoordinateSystem();

    assembleMatrices();
}




// init directly with polynomial data
Beam::Beam(const PolynomialSectionData& sec, const PolynomialLoads& loads,
           const TipData &tip, const BaseData &base)
  : nodes(sec.nodes), EA(sec.EA), 
    EIxx(sec.EIxx), EIyy(sec.EIyy), GJ(sec.GJ),
    rhoA(sec.rhoA), rhoJ(sec.rhoJ), z_node(sec.z),
    Px(loads.Px), Py(loads.Py), Pz(loads.Pz),
    Fx_node(loads.Fx), Fy_node(loads.Fy), Fz_node(loads.Fz),
    Mx_node(loads.Mx), My_node(loads.My), Mz_node(loads.Mz),
    tip(tip), base(base)
{

    translateFromGlobalToFEACoordinateSystem();
    assembleMatrices();

}



//Beam::Beam(const Beam &b, int nothing)
//: nodes(b.nodes), z_node(b.z_node), EA(b.EA),
//EIxx(b.EIxx), EIyy(b.EIyy), GJ(b.GJ),
//rhoA(b.rhoA), rhoJ(b.rhoJ), Px(b.Px), Py(b.Py), Pz(b.Pz),
//Fx_node(b.Fx_node), Fy_node(b.Fy_node), Fz_node(b.Fz_node),
//Mx_node(b.Mx_node), My_node(b.My_node), Mz_node(b.Mz_node),
//tip(b.tip), base(b.base), length(b.length),
//K(b.K), M(b.M), N(b.N), Nother(b.Nother), F(b.F)
//{
//    // do nothing
//}

void Beam::translateFromGlobalToFEACoordinateSystem(){

    // coordinate system is the same,
    // however moments of inertia are defined differently (Ixx = int(x^2 dA) as opposed to standard Ixx = int(y^2 dA))
    // moments are also defined differently. positive in FEA is positive bending direction for 2D beam.
    // where as global system uses positive about 3D axis directions.
    // Mx_local = My_global
    // My_local = -Mx_global

    PolyVec tempP;
    Vector tempV;
    double temp;

    // moments of inertia
    tempP = EIxx;
    EIxx = EIyy;
    EIyy = tempP;

    temp = tip.Ixx;
    tip.Ixx = tip.Iyy;
    tip.Iyy = temp;


    // moments
    tempV = Mx_node;
    Mx_node = My_node;
    My_node = -tempV;

    temp = tip.Mx;
    tip.Mx = tip.My;
    tip.My = -temp;

    // spring constants (swap k_tx and k_ty)
    temp = base.k[1];
    base.k[1] = base.k[3];
    base.k[3] = temp;

    temp = base.rigid[1];
    base.rigid[1] = base.rigid[3];
    base.rigid[3] = temp;
}


// private method
void Beam::assembleMatrices(){

    using namespace BeamFEA;

    Matrix KFull(DOF*nodes, DOF*nodes);
    Matrix MFull(DOF*nodes, DOF*nodes);
    Matrix NotherFull(DOF*nodes, DOF*nodes);
    Matrix NFull(DOF*nodes, DOF*nodes);
    Vector FFull(DOF*nodes);


    // assemble FEM matrices
    FEMAssembly(nodes, z_node, EIxx, EIyy, EA, GJ, rhoA, rhoJ, Px, Py, Pz, KFull, MFull, NotherFull, NFull, FFull);

    // add point loads
    addPointLoads(nodes, FFull, Fx_node, Fy_node, Fz_node, Mx_node, My_node, Mz_node);

    // add contribution from tip mass
    addTipMassContribution(tip, nodes, MFull, FFull);

    // apply base b.c.
    length = applyBaseBoundaryCondition(base.k, base.rigid, nodes, KFull, MFull, NotherFull, NFull, FFull, K, M, Nother, N, F);

}



int Beam::getNumNodes() const{
    return nodes;
}


// MARK: ------------------- COMPUTATIONS ----------------------------


double Beam::computeMass() const{

    double m = 0.0;

    for (int i = 0; i < nodes-1; i++) {

        double L = z_node(i+1) - z_node(i);
        m += L * rhoA[i].integrate(0.0, 1.0);
    }

    return m;

}



double Beam::computeOutOfPlaneMomentOfInertia() const{

    double I = 0.0;

    for (int i = 0; i < nodes-1; i++) {

        double L = z_node(i+1) - z_node(i);
        Poly zpoly = Poly(2, z_node(i+1) - z_node(i), z_node(i));
        Poly Iseg = rhoA[i] * zpoly * zpoly;
        I += L * Iseg.integrate(0.0, 1.0);
    }

    return I;

}

void Beam::computeNaturalFrequencies(int n, Vector &freq) const{
    Matrix empty(0,0);
    naturalFrequencies(false, n, freq, empty);
}

void Beam::computeNaturalFrequencies(int n, Vector &freq, Matrix &vec) const{
    naturalFrequencies(true, n, freq, vec);
}

// private method
void Beam::naturalFrequencies(bool cmpVec, int n, Vector &freq, Matrix &vec) const{

    // compute eigenvalues (omega^2)
    Vector eig(length);
    Matrix eig_vec(0, 0);

    if (cmpVec) {
        myMath::generalizedEigenvalues(K, M, eig, eig_vec);
    } else {
        myMath::generalizedEigenvalues(K, M, eig);
    }


    // resize vector to largest possible
    std::vector<double> tmpFreq(length);
    vec.resize(length, length);
    vec.setZero();

    int idx = 0;
    Vector idx_save(length);
    for (int i = 0; i < length; i++) {

        if (eig(i) > 1e-6) { // don't save rigid modes
            tmpFreq[idx] = sqrt(eig(i)) / (2.0*M_PI);
            idx_save(idx) = i;
            idx++;
        }

        if (idx == n) break;
    }

    // resize vector to actual length (after removing rigid modes and accounting for user input)
    freq.resize(idx);
    for (int k=0; k<idx; k++) freq(k) = tmpFreq[k];

    // parse eigenvectors
    if (cmpVec) {

        vec.resize(6*nodes, idx);
        for (int i = 0; i < idx; i++) {

            Vector dx, dy, dz, dtx, dty, dtz;
	    Vector eig_vec_i(eig_vec.rows());
	    for (int k=0; k<eig_vec.rows(); k++) eig_vec_i[k] = eig_vec(idx_save(i),k);
            computeDisplacementComponentsFromVector(eig_vec_i, dx, dy, dz, dtx, dty, dtz);

            Vector colVec(6*nodes);
	    for(int k=0; k<nodes; k++) {
	      colVec[0*nodes + k] = dx[k];
	      colVec[1*nodes + k] = dy[k];
	      colVec[2*nodes + k] = dz[k];
	      colVec[3*nodes + k] = dtx[k];
	      colVec[4*nodes + k] = dty[k];
	      colVec[5*nodes + k] = dtz[k];
	    }

	    for (int k=0; k<vec.rows(); k++) vec(k,i) = colVec[k];
        }
    }
}




void Beam::computeDisplacement(Vector &dx, Vector &dy, Vector&dz, Vector &dtheta_x, Vector &dtheta_y, Vector &dtheta_z) const{

    // solve linear system
    Vector q(length);

    myMath::solveSPDBLinearSystem(K, F, q);

    computeDisplacementComponentsFromVector(q, dx, dy, dz, dtheta_x, dtheta_y, dtheta_z);
}

// private method
void Beam::computeDisplacementComponentsFromVector(const Vector &q, Vector &dx, Vector &dy, Vector&dz,
                                                   Vector &dtheta_x, Vector &dtheta_y, Vector &dtheta_z) const{

    // resize vectors
    dx.resize(nodes);
    dy.resize(nodes);
    dz.resize(nodes);
    dtheta_x.resize(nodes);
    dtheta_y.resize(nodes);
    dtheta_z.resize(nodes);

    // add in rigid directions
    int i;
    int idx = 0;
    Vector qFull(DOF*nodes);

    for (int i = 0; i < DOF*nodes; i++) {
        if (i < DOF && base.rigid[i]) {
            qFull(i) = 0.0;
        } else{
            qFull(i) = q(idx++);
        }
    }

    // separate by DOF
    for (i = 0; i < nodes; i++) {
        dx(i) = qFull(0 + i*DOF);
        dtheta_x(i) = qFull(1 + i*DOF);
        dy(i) = qFull(2 + i*DOF);
        dtheta_y(i) = qFull(3 + i*DOF);
        dz(i) = qFull(4 + i*DOF);
        dtheta_z(i) = qFull(5 + i*DOF);
    }

    // translate back to global coordinates
    Vector temp = dtheta_x;
    dtheta_x = -dtheta_y;
    dtheta_y = temp;

}





void Beam::computeMinBucklingLoads(double &Pcr_x, double &Pcr_y) const{


    // buckling analysis
    double total_Fz = tip.Fz;
    for (int i = 0; i < nodes; i++) {
        total_Fz += Fz_node(i);
    }

    Pcr_x = estimateCriticalBucklingLoad(total_Fz, 0, 1);
    Pcr_y = estimateCriticalBucklingLoad(total_Fz, 2, 3);

}





// private method
double Beam::estimateCriticalBucklingLoad(double FzExisting, int ix1, int ix2) const{

    // determine length of indicies
    int length = 2*nodes;
    if (base.rigid[ix1]) length--;
    if (base.rigid[ix2]) length--;

    // determine how many directions are removed in total
    int subtract = 0;
    int subtract1 = 0;
    int subtract2 = 0;
    for (int i = 0; i < DOF; i++) {
        if (base.rigid[i]) {
            subtract++;

            if (i < ix1) subtract1++;
            if (i < ix2) subtract2++;
        }
    }

    // need to find indicies corresponidng only to x-direction
    int idx = 0;
    std::vector<int> indices(length);
    if (!base.rigid[ix1]) indices[idx++] = ix1 - subtract1;
    if (!base.rigid[ix2]) indices[idx++] = ix2 - subtract2;
    for (int i = 1; i < nodes; i++) {
        indices[idx++] = ix1 + i*DOF - subtract;
        indices[idx++] = ix2 + i*DOF - subtract;
    }

    // copy relavant portions of matrix over
    Matrix Kbend(length,length); // = project(K, indices, indices);
    Matrix Nbend(length,length); // = project(N, indices, indices);
    Matrix Nbend_other(length,length); // = project(Nother, indices, indices);
    for (int ii=0; ii<length; ii++) {
      for (int jj=0; jj<length; jj++) {
	Kbend(ii,jj) = K(indices[ii], indices[jj]);
	Nbend(ii,jj) = N(indices[ii], indices[jj]);
	Nbend_other(ii,jj) = Nother(indices[ii], indices[jj]);
      }
    }
    

    // solve eigenvalue problem
    Vector eig(length);
    int result = myMath::generalizedEigenvalues(Kbend - Nbend_other, Nbend, eig);

    if (result != 0){
        printf("error in eigenvalue analysis (error code: %d)\n", result);
    }

    // determine minimum critical load (LAPACK should return a sorted array - unless there was an error).
    double Pcr = eig(0);

    // subtracting off existing compressive force (or add if it was a tensile force)
    Pcr += FzExisting;

    return Pcr;
}




// using FEA coordinate system
void Beam::shearAndBending(PolyVec &Vx, PolyVec &Vy, PolyVec &Fz, PolyVec &Mx, PolyVec &My, PolyVec &Tz) const{

    Vx.resize(nodes-1);
    Vy.resize(nodes-1);
    Fz.resize(nodes-1);
    Mx.resize(nodes-1);
    My.resize(nodes-1);
    Tz.resize(nodes-1);

    // integrate from tip downward (b.c. forces are known there)
    double Vx_prev = tip.Fx + Fx_node(nodes-1);
    double Vy_prev = tip.Fy + Fy_node(nodes-1);
    double Fz_prev = tip.Fz + Fz_node(nodes-1);

    double Mx_prev = tip.Mx + Mx_node(nodes-1);
    double My_prev = tip.My + My_node(nodes-1);
    double Tz_prev = tip.Mz + Mz_node(nodes-1);

    for (int i = nodes-2; i >= 0; i--) {
        double L = z_node(i+1) - z_node(i);

        // shear
        Vx[i] = L * Px[i].backwardsIntegrate() + Vx_prev;
        Vy[i] = L * Py[i].backwardsIntegrate() + Vy_prev;

        // axial
        Fz[i] = L * Pz[i].backwardsIntegrate() + Fz_prev;

        // moments
        Mx[i] = L * Vx[i].backwardsIntegrate() + Mx_prev;
        My[i] = L * Vy[i].backwardsIntegrate() + My_prev;

        // torsion (no distributed torsion loads yet)
        Tz[i] = Poly(1, Tz_prev);


        // save value for front of previous element
        Vx_prev = Vx[i].eval(0.0) + Fx_node[i];
        Vy_prev = Vy[i].eval(0.0) + Fy_node[i];
        Fz_prev = Fz[i].eval(0.0) + Fz_node[i];

        Mx_prev = Mx[i].eval(0.0) + Mx_node[i];
        My_prev = My[i].eval(0.0) + My_node[i];
        Tz_prev = Tz[i].eval(0.0) + Mz_node[i];

    }


}


// x and y should be in same coordinate system (centered at elastic axis, in principal directions)
// void Beam::computeAxialStress(Vector &x, Vector &y, Vector &z, Vector &E, Vector &sigma_axial) const{
void Beam::computeAxialStrain(Vector &x, Vector &y, Vector &z, Vector &epsilon_axial) const{

    // get distributed forces/moments
    PolyVec Vx, Vy, Fz, Mx, My, Tz;
    shearAndBending(Vx, Vy, Fz, Mx, My, Tz);

    // find location in structure
    int idx;
    double distance;

    for (int i = 0; i < z.size(); i++) {

        // check for out of bounds
        if (z(i) < z_node(0) || z(i) > z_node(nodes-1)) {
            epsilon_axial(i) = 0.0;
            continue;
        }

        // find location in structure (could do something faster like binary search)
        for (int j = 1; j < nodes; j++) {
            if (z(i) <= z_node(j)) {
                idx = j-1;
                distance = (z(i) - z_node(j-1)) / (z_node(j) - z_node(j-1));
                break;
            }
        }

        // evalute section properties
        double EA_sec = EA[idx].eval(distance);
        double EIxx_sec = EIxx[idx].eval(distance);
        double EIyy_sec = EIyy[idx].eval(distance);

        // evalute section forces/moments
        double Fz_sec = Fz[idx].eval(distance);
        double Mx_sec = Mx[idx].eval(distance);
        double My_sec = My[idx].eval(distance);

        epsilon_axial(i) = Fz_sec/EA_sec - Mx_sec/EIxx_sec*x(i) - My_sec/EIyy_sec*y(i);


//        // compute axial strain
//        Matrix A(3, 3);
//        A(0, 0) = EA_sec;       A(0, 1) = -ESx_sec;     A(0, 2) = -ESy_sec;
//        A(1, 0) = -ESx_sec;     A(1, 1) = EIxx_sec;     A(1, 2) = EIxy_sec;
//        A(2, 0) = -ESy_sec;     A(2, 1) = EIxy_sec;    A(2, 2) = EIyy_sec;
//
//        Vector f(3);
//        f(0) = Fz_sec;
//        f(1) = Mx_sec;
//        f(2) = My_sec;
//
//        Vector v(3);
//        myMath::solveLinearSystem(A, f, v);
//
//        Vector p(3);
//        p(0) = 1.0;
//        p(1) = -x(i);
//        p(2) = -y(i);
//
//        double eps_zz = boost::numeric::ublas::inner_prod(v, p);
//
//        // compute axial stress
//        sigma_axial(i) = E(i) * eps_zz;

    }
}


// x and y should be in same coordinate system as used in inputs EIxx, EIxy, etc.

//void Beam::computeShearStressForThinShellSection(Vector &x, Vector &y, Vector &t, Vector &E, Vector &dEdz, double z, double shear_stress) const{
//
//    // get distributed forces/moments
//    PolyVec Vx, Vy, Fz, Mx, My, Tz;
//    shearAndBending(Vx, Vy, Fz, Mx, My, Tz);
//
//    // find location in structure
//    int idx;
//    double distance;
//
//    // check for out of bounds
//    if (z < z_node(0) || z > z_node(nodes-1)) {
//        shear_stress = 0.0;
//        return;
//    }
//
//    // find location in structure (could do something faster like binary search)
//    for (int j = 1; j < nodes; j++) {
//        if (z <= z_node(j)) {
//            idx = j-1;
//            distance = (z - z_node(j-1)) / (z_node(j) - z_node(j-1));
//        }
//    }
//
//    // evalute section properties
//    double EA_sec = EA[idx].eval(distance);
//    double EIxx_sec = EIxx[idx].eval(distance);
//    double EIyy_sec = EIyy[idx].eval(distance);
//
//    // evalute section forces/moments
//    double Fz_sec = Fz[idx].eval(distance);
//    double Mx_sec = Mx[idx].eval(distance);
//    double My_sec = My[idx].eval(distance);
//    double Tz_sec = Tz[idx].eval(distance);
//
//    // derivative of stiffness properties
//    double dEA_dz = EA[idx].differentiate().eval(distance);
//    double dESx_dz = ESx[idx].differentiate().eval(distance);
//    double dESy_dz = ESy[idx].differentiate().eval(distance);
//    double dEIxx_dz = EIxx[idx].differentiate().eval(distance);
//    double dEIyy_dz = EIyy[idx].differentiate().eval(distance);
//    double dEIxy_dz = EIxy[idx].differentiate().eval(distance);
//
//    // derivative of section forces/moments
//    double dFz_dz = Fz[idx].differentiate().eval(distance);
//    double dMx_dz = Mx[idx].differentiate().eval(distance);
//    double dMy_dz = My[idx].differentiate().eval(distance);
//
//    // compute axial strain
//    Matrix A(3, 3);
//    A(0, 0) = EA_sec;       A(0, 1) = -ESx_sec;     A(0, 2) = -ESy_sec;
//    A(1, 0) = -ESx_sec;     A(1, 1) = EIxx_sec;     A(1, 2) = EIxy_sec;
//    A(2, 0) = -ESy_sec;     A(2, 1) = EIxy_sec;    A(2, 2) = EIyy_sec;
//
//    Vector f(3);
//    f(0) = Fz_sec;
//    f(1) = Mx_sec;
//    f(2) = My_sec;
//
//    Matrix dA_dz(3, 3);
//    dA_dz(0, 0) = dEA_dz;   dA_dz(0, 1) = -dESx_dz; dA_dz(0, 2) = -dESy_dz;
//    dA_dz(1, 0) = -dESx_dz; dA_dz(1, 1) = dEIxx_dz; dA_dz(1, 2) = dEIxy_dz;
//    dA_dz(2, 0) = -dESy_dz; dA_dz(2, 1) = dEIxy_dz; dA_dz(2, 2) = dEIyy_dz;
//
//    Vector df_dz(3);
//    df_dz(0) = dFz_dz;
//    df_dz(1) = dMx_dz;
//    df_dz(2) = dMy_dz;
//
//    Vector v(3);
//    myMath::solveLinearSystem(A, f, v);
//
//    Vector dv_dz(3);
//    myMath::solveLinearSystem(A, df_dz-prod(dA_dz,v), dv_dz);
//
//    // path integral
//    int n = x.size();
//    Vector s(n);
//    s(0) = 0.0;
//    for (int i = 1; i < n; i++) {
//        s(i) = s(i-1) + sqrt(x(i)*x(i) + y(i)*y(i));
//    }
//
//    Vector p0(n), p1(n), p2(n);
//    myMath::cumtrapz(element_prod(E,t), s, p0);
//    myMath::cumtrapz(-element_prod(element_prod(E,t), x), s, p1);
//    myMath::cumtrapz(-element_prod(element_prod(E,t), y), s, p2);
//
//    // cut component of flexural shear flow
//    Vector qs_prime = dv_dz(0)*p0 + dv_dz(1)*p1 + dv_dz(2)*p2;
//
//    // TODO: I have omitted dEdz term
//
//    // TODO: compute shear center
//    double xc = 0.0;
//    double yc = 0.0;
//
//    // compute moment due to first component of shear flow
//    Vector r(n);
//    for (int i = 0; i < n; i++) {
//        r(i) = sqrt(pow(x(i)-xc, 2) + pow(y(i)-yc, 2));
//    }
//    double Mq = myMath::trapz(element_prod(qs_prime, r), s);
//
//    // compute Abar
//    double Abar = 0.5*(myMath::trapz(x, y) - myMath::trapz(y, x));
//
//    // constant shear flow
//    double q0 = (Tz_sec - Mq) / (2.0*Abar);
//
//    // total shear flow
//    Vector q = qs_prime;
//    for (int i = 0; i < n; i++) {
//        q(i) += q0;
//    }
//
//    // shear stress
//    Vector tau = element_prod(q, t);
//
//}
