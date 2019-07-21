//
//  FEAData.h
//  pbeam
//
//  Created by aning on 2/7/12.
//  Copyright (c) 2012 NREL. All rights reserved.
//

#ifndef pbeam_FEAData_h
#define pbeam_FEAData_h

/**
 This header defines a few pieces of data used throughout the finite element analysis.
 
 **/
#include <vector>
#include "Poly.h"

#define DOF 6

typedef std::vector<Poly> PolyVec;

struct TipData{
    
    /**
     mass - mass of tip object
     cm - offset from tip of beam to mass
     Iij - moments of inertia about beam tip
     F, M - forces and moments applied to beam tip
     
     **/
    
    double m;
    double cm_offsetX;
    double cm_offsetY;
    double cm_offsetZ;
    double Ixx, Iyy, Izz, Ixy, Ixz, Iyz;
    double Fx, Fy, Fz;
    double Mx, My, Mz;

    TipData(): 
    m(0.0), cm_offsetX(0.0), cm_offsetY(0.0), cm_offsetZ(0.0),
    Ixx(0.0), Iyy(0.0), Izz(0.0), Ixy(0.0), Ixz(0.0), Iyz(0.0),
    Fx(0.0), Fy(0.0), Fz(0.0), Mx(0.0), My(0.0), Mz(0.0) { 
        //nothing else
    }
    
    
    TipData(double m, double cm[3], double I[6], double F[3], double M[3]): 
    m(m), cm_offsetX(cm[0]), cm_offsetY(cm[1]), cm_offsetZ(cm[2]),
    Ixx(I[0]), Iyy(I[1]), Izz(I[2]), Ixy(I[3]), Ixz(I[4]), Iyz(I[5]),
    Fx(F[0]), Fy(F[1]), Fz(F[2]), Mx(M[0]), My(M[1]), Mz(M[2]) { 
        // nothing else
    }
    
    
    
};

typedef struct TipData TipData;






struct BaseData{
    
    /**
     k - spring constants at base
     rigid - true if completely rigid in that direction (i.e. k_i = infinity)
     
     **/
    
    double k[DOF]; // kxx, ktxtx, kyy, ktyty, kzz, ktztz
    bool rigid[DOF]; // same order
    
    
    BaseData(){ // default is free end
        
        for (int i = 0; i < DOF; i++){
            k[i] = 0.0;
            rigid[i] = false;
        }
    }
    
    BaseData(double kxx, double ktxtx, double kyy, double ktyty, double kzz, double ktztz, double infinity){
        k[0] = kxx;
        k[1] = ktxtx;
        k[2] = kyy;
        k[3] = ktyty;
        k[4] = kzz;
        k[5] = ktztz;
        
        for (int i = 0; i < DOF; i++) {
            rigid[i] = (k[i] == infinity);
        }

    }
    
    
};

typedef struct BaseData BaseData;



struct IsotropicMaterial{
    
    // E, G, rho - material properties.  elastic modulus, shear modulus, and density.
    double E, G, rho;  
    
    IsotropicMaterial():
    E(0.0), G(0.0), rho(0.0){
       // nothing else
    }
    
    IsotropicMaterial(double E, double G, double rho):
    E(E), G(G), rho(rho){
        //nothing else
    }
    
    
    
};

struct SectionData {
    
    int nodes;
    Vector z, EA, EIxx, EIyy, GJ, rhoA, rhoJ;
    
    // default constructor
    SectionData(){
        nodes = 0;
    }
    
    // see main constructor,
    // but omitting ESx, ESy, EIxy which are not needed for finite element analysis 
    // (only needed for axial stress)
    SectionData(const Vector &z, const Vector &EA, 
                const Vector &EIxx, const Vector &EIyy, 
                const Vector &GJ, const Vector &rhoA, const Vector &rhoJ):
    nodes((int)z.size()), z(z), EA(EA), EIxx(EIxx), EIyy(EIyy), GJ(GJ), rhoA(rhoA), rhoJ(rhoJ){
    }
    
    
    /** 
     main constructor
     
     z - axial position of each section
     EA - axial stiffness for each section
     ESx - moment of stiffness about the x-axis (int(E y, dA))
     ESy - moment of stiffness about the y-axis (int(E y, dA))
     EIxx - bending stiffness about x-axis for each section (int(E*y*y, dA))
     EIyy - bending stiffness about y-axis for each section
     EIxy - moment of centrifugal stiffness (int(E*x*y, dA))
     GJ - torsional stiffness of each section
     rhoA - mass per unit length for each section
     rhoJ - polar moment of inertia per unit length for each section
     
     **/
//    SectionData(const Vector &z, const Vector &EA,
//                const Vector &EIxx, const Vector &EIyy,
//                const Vector &GJ, const Vector &rhoA, const Vector &rhoJ):
//    nodes(z.size()), z(z), EA(EA), ESx(ESx), ESy(ESy), EIxx(EIxx), EIyy(EIyy), EIxy(EIxy),
//    GJ(GJ), rhoA(rhoA), rhoJ(rhoJ){
//        
//    }
};


struct PolynomialSectionData {

    int nodes;
    Vector z;
    PolyVec EA, EIxx, EIyy, GJ, rhoA, rhoJ;
    
    // default constructor
    PolynomialSectionData(){
        nodes = 0;
    }
    
    PolynomialSectionData(const Vector &z, const PolyVec &EA, 
                const PolyVec &EIxx, const PolyVec &EIyy, 
                const PolyVec &GJ, const PolyVec &rhoA, const PolyVec &rhoJ):
    nodes((int) z.size()), z(z), EA(EA), EIxx(EIxx), EIyy(EIyy), GJ(GJ), rhoA(rhoA), rhoJ(rhoJ){
    }
};

struct Loads {
    
    int nodes;
    Vector Px, Py, Pz, Fx, Fy, Fz, Mx, My, Mz;
    
    // default constructor
    Loads(){
        nodes = 0;
    }
    
    // zero loading constructor
    Loads(int n){
        nodes = n;
        
        Px.resize(nodes);
        Py.resize(nodes);
        Pz.resize(nodes);
        Fx.resize(nodes);
        Fy.resize(nodes);
        Fz.resize(nodes);
        Mx.resize(nodes);
        My.resize(nodes);
        Mz.resize(nodes);

	for (int k=0; k<nodes; k++) {
	  Px[k] = 0.0;
	  Py[k] = 0.0;
	  Pz[k] = 0.0;
	  Fx[k] = 0.0;
	  Fy[k] = 0.0;
	  Fz[k] = 0.0;
	  Mx[k] = 0.0;
	  My[k] = 0.0;
	  Mz[k] = 0.0;
	}
        
    }
    
    // see main constructor
    // but with point loads initialized to zero.
    Loads(const Vector &Px, const Vector &Py, const Vector &Pz):
    nodes((int)Px.size()), Px(Px), Py(Py), Pz(Pz){
        
        Fx.resize(nodes);
        Fy.resize(nodes);
        Fz.resize(nodes);
        Mx.resize(nodes);
        My.resize(nodes);
        Mz.resize(nodes);
        
	for (int k=0; k<nodes; k++) {
	  Fx[k] = 0.0;
	  Fy[k] = 0.0;
	  Fz[k] = 0.0;
	  Mx[k] = 0.0;
	  My[k] = 0.0;
	  Mz[k] = 0.0;
	}
    }
    
    /**
     main constructor
     
     Px_node, Py_node, Pz_node - distributed loads at each section
     Fx_node, Fy_node, Fz_node - applied point force at each node
     Mx_node, My_node, Mz_node - applied point moment at each node
     
     **/
    Loads(const Vector &Px, const Vector &Py, const Vector &Pz, 
          const Vector &Fx, const Vector &Fy, const Vector &Fz, 
          const Vector &Mx, const Vector &My, const Vector &Mz):
    nodes((int)Px.size()), Px(Px), Py(Py), Pz(Pz), Fx(Fx), Fy(Fy), Fz(Fz), Mx(Mx), My(My), Mz(Mz){
    }
};

struct PolynomialLoads {
    
    int nodes;
    PolyVec Px, Py, Pz;
    Vector Fx, Fy, Fz, Mx, My, Mz;

    
    // default constructor
    PolynomialLoads(){
        nodes = 0;
    }
    
     /**
     main constructor
     
     Px_node, Py_node, Pz_node - distributed loads at each section
     Fx_node, Fy_node, Fz_node - applied point force at each node
     Mx_node, My_node, Mz_node - applied point moment at each node
     
     **/
    PolynomialLoads(const PolyVec &Px, const PolyVec &Py, const PolyVec &Pz, 
          const Vector &Fx, const Vector &Fy, const Vector &Fz, 
          const Vector &Mx, const Vector &My, const Vector &Mz):
    nodes((int)Fx.size()), Px(Px), Py(Py), Pz(Pz), Fx(Fx), Fy(Fy), Fz(Fz), Mx(Mx), My(My), Mz(Mz){
    }
};

#endif
