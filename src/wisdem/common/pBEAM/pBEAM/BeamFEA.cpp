//
//  BeamFEA.cpp
//  pbeam
//
//  Created by Andrew Ning on 2/6/12.
//  Copyright (c) 2012 NREL. All rights reserved.
//



#include "BeamFEA.h"

namespace BeamFEA {

// MARK: --------- MATRIX ASSEMBLY METHODS ------------------



// creates matrix s.t. K_{ij} = constant * integral( I * f_i * f_j )
// where I is a polynomial and f[i] represents the ith shape function as a polynomial
void matrixAssembly(const Poly &I, int ns, const Poly f[], double constant, Matrix &K){
    
    for (int i = 0; i < ns; i++) {
        for (int j = i; j < ns; j++) {
            
            K(i, j) = constant * (I * f[i] * f[j]).integrate(0.0, 1.0);
            
            // integration is symmetric
            if (i != j) {
                K(j, i) = K(i, j);
            }
        }
    }
    
}



// creates a vector s.t. F_i = constant * integral (p * f_i)
// where p is a polynomial and f[i] is the ith shape funciton as a polynomial
void vectorAssembly(const Poly &p, int ns, const Poly f[], double constant, Vector &F){
    
    
    for (int i = 0; i < ns; i++) {
        F(i) = constant * (p * f[i]).integrate(0.0, 1.0);
    }
    
}


    
void integrateDistributedCompressionLoads(const Vector &z_node, const PolyVec &Pz, PolyVec &Fz){
    
    int nodes = (int) z_node.size();
    
    Fz.resize(nodes-1);

    // integrate from tip downward (b.c. forces are known at free end)
    double Fz_prev = 0.0;
        
    for (int i = nodes-2; i >= 0; i--) {
        double L = z_node(i+1) - z_node(i);
        
        Fz(i) = L * Pz(i).backwardsIntegrate() + Fz_prev;

        Fz_prev = Fz(i).eval(0.0);
    }
    
}


// computes FEM matrices for one 12-dof beam element
void beamMatrix(double L, const Poly &EIx, const Poly &EIy, const Poly &EA, 
                        const Poly &GJ, const Poly &rhoA, const Poly &rhoJ,
                        const Poly &Px, const Poly &Py, const Poly &FzfromPz,
                        Matrix &K, Matrix &M, Matrix &Ndist, Matrix &Nconst, Vector &F){
    
    using namespace boost::numeric::ublas;
    
    // initialize
    K.clear();
    M.clear();
    Ndist.clear();
    Nconst.clear();
    F.clear();
    
    int i;
    
    // ------- bending (x-dir) ---------------
    const int ns = 4; // number of shape functions
    
    // define the shape functions    
    Poly f[ns] = {
        Poly(4, 2.0, -3.0, 0.0, 1.0),
        Poly(4, 1.0*L, -2.0*L, 1.0*L, 0.0*L),
        Poly(4, -2.0, 3.0, 0.0, 0.0),
        Poly(4, 1.0*L, -1.0*L, 0.0*L, 0.0*L)
    };
    
    Poly fp[ns];
    Poly fpp[ns];
    
    for (i = 0; i < ns; i++) {
        fp[i] = f[i].differentiate();
        fpp[i] = fp[i].differentiate();
    }
    
    
    // stiffness matrix
    Matrix KbendX(ns, ns);
    matrixAssembly(EIx, ns, fpp, 1.0/pow(L,3), KbendX);
   
    // inertia matrix
    Matrix Mbend(ns, ns);
    matrixAssembly(rhoA, ns, f, L, Mbend);
    
    // incremental stiffness matrix from distributed loads
    Matrix Nbend_dist(ns, ns);
    matrixAssembly(-FzfromPz, ns, fp, 1.0/L, Nbend_dist); // compression loads positive
    
    // incremental stiffness matrix from constant loads    
    Matrix Nbend_const(ns, ns);
    Poly one(1, 1.0);
    matrixAssembly(one, ns, fp, 1.0/L, Nbend_const);

    // distributed applied loads    
    Vector FbendX(ns);
    vectorAssembly(Px, ns, f, L, FbendX);
    
    // put into global matrix
    indirect_array<> idx(4);
    idx(0) = 0; idx(1) = 1; idx(2) = 6; idx(3) = 7;
    
    project(K, idx, idx) = KbendX;
    project(M, idx, idx) = Mbend;
    project(Ndist, idx, idx) = Nbend_dist;
    project(Nconst, idx, idx) = Nbend_const;
    project(F, idx) = FbendX;
    
    
    // ---------- bending (y-dir) ---------------
    
    // stiffness matrix
    Matrix KbendY(ns, ns);
    matrixAssembly(EIy, ns, fpp, 1.0/pow(L,3), KbendY);

    
    // distributed applied loads
    Vector FbendY(ns);
    vectorAssembly(Py, ns, f, L, FbendY);

    // put into global matrix (mass and incremental stiffness are same in x an y)
    idx(0) = 2; idx(1) = 3; idx(2) = 8; idx(3) = 9;
    
    project(K, idx, idx) = KbendY;
    project(M, idx, idx) = Mbend;
    project(Ndist, idx, idx) = Nbend_dist;
    project(Nconst, idx, idx) = Nbend_const;
    project(F, idx) = FbendY;
    
    
    // ----------- axial ----------------
    const int nsz = 2; // number of shape functions
    
    Poly fz[nsz] = {
        Poly(2, -1.0, 1.0),
        Poly(2, 1.0, 0.0)
    };
    
    // derivatives of the shape function
    Poly fzp [nsz];
    Poly fzpp [nsz];
    
    for (i = 0; i < nsz; i++) {
        fzp[i] = fz[i].differentiate();
        fzpp[i] = fzp[i].differentiate();
    }
    
    // stiffness matrix
    Matrix Kaxial(nsz, nsz);
    matrixAssembly(EA, nsz, fzp, 1.0/L, Kaxial);

    
    // inertia matrix
    Matrix Maxial(nsz, nsz);
    matrixAssembly(rhoA, nsz, fz, L, Maxial);
    
    // axial loads already given (work equivalent approach not appropriate for distributed axial loads)
    Vector Faxial(nsz);
    Faxial(0) = FzfromPz.eval(0.0);
    Faxial(1) = FzfromPz.eval(1.0);
    
    // put into global matrix
    indirect_array<> idx_z(2);
    idx_z(0) = 4; idx_z(1) = 10;
    
    project(K, idx_z, idx_z) = Kaxial;
    project(M, idx_z, idx_z) = Maxial;
    project(F, idx_z) = Faxial;
    
    // --------- torsion -------------
    // same shape functions as axial
    
    // stiffness matrix
    Matrix Ktorsion(nsz, nsz);
    matrixAssembly(GJ, nsz, fzp, 1.0/L, Ktorsion);

    // inertia matrix
    Matrix Mtorsion(nsz, nsz);
    matrixAssembly(rhoJ, nsz, fz, L, Mtorsion);
    
    // put into global matrix
    idx_z(0) = 5; idx_z(1) = 11;
    
    project(K, idx_z, idx_z) = Ktorsion;
    project(M, idx_z, idx_z) = Mtorsion;
    
}




// assembles FEA matrices for the various elements into global matrices for the structure
// EIx, EA etc. are arrays of length nodes-1
// matrix are of size DOF*nodes x DOF*nodes
void FEMAssembly(int nodes, const Vector &z, const PolyVec &EIx, const PolyVec &EIy, 
                          const PolyVec &EA, const PolyVec &GJ, const PolyVec &rhoA, const PolyVec &rhoJ, 
                          const PolyVec &Px, const PolyVec &Py, const PolyVec &Pz,
                          Matrix &K, Matrix &M, Matrix &Nother, Matrix &N, Vector &F){
    
    using namespace boost::numeric::ublas;
    
    // initialize
    K.clear();
    M.clear();
    Nother.clear();
    N.clear();
    F.clear();
    
    Matrix Ksub(2*DOF, 2*DOF);
    Matrix Msub(2*DOF, 2*DOF);
    Matrix Ndist_sub(2*DOF, 2*DOF);
    Matrix Nconst_sub(2*DOF, 2*DOF);
    Vector Fsub(2*DOF, 2*DOF);
        
    range r;
    
    
    // integrate distributed axial loads
    PolyVec FzFromPz;
    integrateDistributedCompressionLoads(z, Pz, FzFromPz);
    
    for (int i = 0; i < nodes-1; i++) {
        
        double L = z[i+1] - z[i];
        
        // compute submatrix
        beamMatrix(L, EIx(i), EIy(i), EA(i), GJ(i), rhoA(i), rhoJ(i), Px(i), Py(i), FzFromPz(i),
                   Ksub, Msub, Ndist_sub, Nconst_sub, Fsub);

        // insert into global matrix
        r = range(i*DOF, i*DOF + 2*DOF);
        
        project(K, r, r) += Ksub;
        project(M, r, r) += Msub;
        project(Nother, r, r) += Ndist_sub;
        project(N, r, r) += Nconst_sub;
        project(F, r) += Fsub;
        
    }
       
}





// MARK: --------- BOUNDARY CONDITIONS ------------------


// size of matrix is DOF*nodes x DOF*nodes, Vector is DOF*nodes
void addTipMassContribution(const TipData &tip, int nodes, Matrix &M, Vector &F){
    
    using namespace boost::numeric::ublas;
    
    double m = tip.m;
    
    double x = tip.cm_offsetX;
    double y = tip.cm_offsetY;
    double z = tip.cm_offsetZ;
    
    double Ixx = tip.Ixx;
    double Iyy = tip.Iyy;
    double Izz = tip.Izz;
    double Ixy = tip.Ixy;
    double Ixz = tip.Ixz;
    double Iyz = tip.Iyz;
    
//    double MtipArray[DOF][DOF] = {
//        {m,     0.0,                    0.0,    m*z,                    0.0,    -m*y},
//        {0.0,   Ixx + m*(y*y + z*z),    -m*z,   Ixy - m*x*y,            m*y,    Ixz - m*x*z},
//        {0.0,   -m*z,                   m,      0.0,                    0.0,    m*x},
//        {m*z,   Ixy - m*x*y,            0.0,    Iyy + m*(x*x + z*z),    -m*x,   Iyz - m*y*z},
//        {0.0,   m*y,                    0.0,    -m*x,                   m,      0},
//        {-m*y,  Ixz - m*x*z,            m*x,    Iyz - m*y*z,            0.0,    Izz + m*(x*x + y*y)},
//    };
    double MtipArray[DOF][DOF] = {
        {m,     m*z,                    0.0,    0.0,                    0.0,    -m*y},
        {m*z,   Ixx + m*(x*x + z*z),    0.0,    -Ixy + m*x*y,           -m*x,   Iyz - m*y*z},
        {0.0,   0.0,                    m,      m*z,                    0.0,    m*x},
        {0.0,   -Ixy + m*x*y,           m*z,    Iyy + m*(y*y + z*z),    -m*y,   -Ixz + m*x*z},
        {0.0,   -m*x,                    0.0,   -m*y,                   m,      0},
        {-m*y,  Iyz - m*y*z,            m*x,    -Ixz + m*x*z,           0.0,    Izz + m*(x*x + y*y)},
    };
    Matrix Mtip(DOF, DOF);
    
    // copy over into matrix
    for (int i = 0; i < DOF; i++) {
        for (int j = 0; j < DOF; j++) {
            Mtip(i, j) = MtipArray[i][j];
        }
    }
    
    Vector Ftip(DOF);
    Ftip(0) = tip.Fx;
    Ftip(1) = tip.Mx;
    Ftip(2) = tip.Fy;
    Ftip(3) = tip.My;
    Ftip(4) = tip.Fz;
    Ftip(5) = tip.Mz;
    
    // add at end of matrix
    range r = range(DOF*(nodes-1), DOF*nodes);
    
    project(M, r, r) += Mtip;
    project(F, r) += Ftip;
    
}


int __computeReducedSize(const bool rigidDirections[DOF], int nodes);
    
// intended to be private method.  Computes the new size of the global matrices after removing rigid directions
// due to base boundary conditions.  

int __computeReducedSize(const bool rigidDirections[DOF], int nodes){
    
    int remove = 0;
    for (int i = 0; i < DOF; i++) {
        if (rigidDirections[i]) {
            remove += 1;
        }
    }
    
    return DOF*nodes - remove;
    
}    
    
    

// full is DOF*nodes x DOF*nodes
// others have length given by reduction
int applyBaseBoundaryCondition(const double Kbase[], const bool rigidDirections[], int nodes, 
                                         Matrix &KFull, const Matrix &MFull, const Matrix &NotherFull, 
                                         const Matrix &NFull, const Vector &FFull, 
                                         Matrix &K, Matrix &M, Matrix &Nother, Matrix &N, Vector &F){

    using namespace boost::numeric::ublas;

    // resize matrices and clear data
    int length = __computeReducedSize(rigidDirections, nodes);
    K.resize(length, length, false);
    M.resize(length, length, false);
    Nother.resize(length, length, false);
    N.resize(length, length, false);
    F.resize(length, false);
    
    K.clear();
    M.clear();
    Nother.clear();
    N.clear();
    F.clear();
    
    
    // add base stiffness
    for (int i = 0; i < DOF; i++) {
        KFull(i, i) += Kbase[i];
    }
    
    
    // copy into a reduced set of arrays  
    indirect_array<> idx(length);
    int j = 0;
    for (int i = 0; i < DOF*nodes; i++) {
        if (i >= DOF || !rigidDirections[i]){
            idx(j++) = i;
        }
    }
    
    K = project(KFull, idx, idx);
    M = project(MFull, idx, idx);
    Nother = project(NotherFull, idx, idx);
    N = project(NFull, idx, idx);
    F = project(FFull, idx);
        
    
    return length;
}




// F is legnth DOF*nodes
// others are length nodes
void addPointLoads(int nodes, Vector &F,
                   const Vector &FxPoint, const Vector &FyPoint, const Vector &FzPoint,
                   const Vector &MxPoint, const Vector &MyPoint, const Vector &MzPoint){
    
    
    for (int i = 0; i < nodes; i++) {
        F[0 + i*DOF] += FxPoint[i];
        F[1 + i*DOF] += MxPoint[i];
        F[2 + i*DOF] += FyPoint[i];
        F[3 + i*DOF] += MyPoint[i];
        F[4 + i*DOF] += FzPoint[i];
        F[5 + i*DOF] += MzPoint[i];
    }
    
    
}


}