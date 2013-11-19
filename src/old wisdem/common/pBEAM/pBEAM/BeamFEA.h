//
//  BeamFEA.h
//  pbeam
//
//  Created by Andrew Ning on 2/6/12.
//  Copyright (c) 2012 NREL. All rights reserved.
//

#ifndef pbeam_BeamFEA_h
#define pbeam_BeamFEA_h

#include "myMath.h"
#include "Poly.h"
#include "FEAData.h"


/**
 This class contians only methods, useful for finite element analysis of beam elements
 represented by polynomial data between nodes.
 
 **/





namespace BeamFEA {

    
    
    
    // MARK: -------------------- FEA MATRIX ASSEMBLY METHODS ----------------------------
    
    
    /**
     Creates a matrix used in finite element analyses. (K_{ij} = constant  integral( I  f_i  f_j ))
     The form is generic and can be used for stiffness matrix, inertia matrix, incremental stiffness, etc.
     
     Arguments:
     I - a polynomial in the integrand
     ns - number of shape functions
     f[] - contains the shape functions.  f[i] is a polynomial describing the ith shape function.
     constant - a constant multiplier for each term in the matrix
     
     Returns:
     K the resulting matrix.  K must be of size ns x ns
     
     **/
    void matrixAssembly(const Poly &I, int ns, const Poly f[], double constant, Matrix &K); 
    
    
    
    /**
     Creates a vector used in finite element analyses. (F_i = constant  integral (p  f_i))
     
     Arguments:
     
     p - a polynomial in the integrand
     ns - number of shape functions
     f[] - contains the shape functions.  f[i] is a polynomial describing the ith shape function.
     constant - a constant multiplier for each term in the vector
     
     Returns:
     F the resulting vector.  F must be of length ns
     
     **/
    void vectorAssembly(const Poly &p, int ns, const Poly f[], double constant, Vector &F);
    
    
    
    /** 
     A convenience method that translates polynomials of distributed loads to 
     polynomials of axial loads.  For axial loads the load from elements above
     contribute to the current element (above in the sense of opposite of integration direction).
     
     Arguments:
     z_node - vector of axial locations
     Pz - distributed load polynomials in z direction (positive in positve z direction)
     
     Our:
     Fz - axial force polynomials
     
     **/
    void integrateDistributedCompressionLoads(const Vector &z_node, const PolyVec &Pz, PolyVec &Fz);
    
    
    /**
     Computes finite element matrices for a 12-dof beam element. z is the azial direction.
     The dof are in the order: (x, theta_x, y, theta_y, z, theta_z)_bottom then repeated for the top
     
     Arguments:
     L - length of element
     the following arrays all represent polynomials in normalized coordinates (0 to 1) from bottom to top of the element. (z up)
     EIx - bending stiffness in x direction 
     EIy - bending stiffness in y direction
     EA - axial stiffness 
     GJ - torsional stiffness
     rhoA - mass per unit length
     rhoJ - polar mass moment of inertia per unit length
     Px - distributed force (force per unit length) in x direction (polynomial)
     Py - distributed force in y direction
     FzFromPz - nodal axial forces from distributed force in z-direction
     
     Returns:
     K - stiffness matrix.  size: 2*DOF x 2*DOF
     M - inertia matrix. size: 2*DOF x 2*DOF
     Ndist - the incremental stiffness matrix due to the distributed (and axial) loads. size: 2*DOF x 2*DOF
     Nconst - the incremental stiffness matrix for a unit constant load. size: 2*DOF x 2*DOF
     F - work-equivalent nodal forces/moments. length: 2*DOF
     
     **/
    void beamMatrix(double L, const Poly &EIx, const Poly &EIy, const Poly &EA, 
                    const Poly &GJ, const Poly &rhoA, const Poly &rhoJ,
                    const Poly &Px, const Poly &Py, const Poly &FzfromPz,
                    Matrix &K, Matrix &M, Matrix &Ndist, Matrix &Nconst, Vector &F);
    
    
    /**
     Assembes the FEA matrices from each element into the global matrix for the structure.
     See beam() for order of the DOFs
     
     Arguments:
     nodes - number of nodes at which data is supplied
     z - axial location of each node
     EIx - x-dir bending stiffness matrix.  EIx[i] is a polynomial describing variation in bending stiffness
     from node i to node i+1.  length: nodes-1
     EIy - y-dir bending stiffness.  length: nodes-1
     EA - axial stiffness.  length: nodes-1
     GJ - torsional stiffness.  length: nodes-1
     rhoA - mass per unit length.  length: nodes-1
     rhoJ - mass moment of inertia per unit length.  length: nodes-1
     Px - distributed loads (force/length) in x-direction.  length: nodes-1  
     Px[i] is a polynomial describe distributed loads in element normalized coordinates
     Py - distributed loads in y-direction.  length: nodes-1
     Pz - distributed loads in z-direction.  length: nodes-1
     
     Returns:
     global matrices for
     K - stiffness matrix.  size: DOF*nodes x DOF*nodes
     M - inertia matrix.  size: DOF*nodes x DOF*nodes
     Ndist - the incremental stiffness matrix due to the distributed (and axial) loads.  size: DOF*nodes x DOF*nodes
     Nconst - the incremental stiffness matrix for a unit constant load.  size: DOF*nodes x DOF*nodes
     F - work-equivalent nodal forces/moments.  length: DOF*nodes
     
     **/
    void FEMAssembly(int nodes, const Vector &z, const PolyVec &EIx, const PolyVec &EIy, 
                            const PolyVec &EA, const PolyVec &GJ, const PolyVec &rhoA, const PolyVec &rhoJ, 
                            const PolyVec &Px, const PolyVec &Py, const PolyVec &Pz,
                            Matrix &K, Matrix &M, Matrix &Nother, Matrix &N, Vector &F);
  
    
    
    
    // MARK: -------------------- BOUNDARY CONDITION METHODS  ----------------------
    
    
    
    /**
     Computes contribution of tip mass to the inertia matrix and force vector
     It is assumed that data is ordered such that base is index 0 and tip at the end.
     
     Arguments:
     tipData - struct containing tip mass/force information (see FEAData.h)
     nodes - number of nodes in beam
     
     In/Out:
     M - inertia matrix modified in place.  size: DOF*nodes x DOF*nodes
     F - nodal force vector modified in place.  length: DOF*nodes
     
     **/
    void addTipMassContribution(const TipData &tip, int nodes, Matrix &M, Vector &F);
    
    
    

    
    /**
     Removes rows/columns from full global matrices corresponding to directions that are
     considered rigid.  For those that are not, a stiffness value can be input.
     
     Arguments:
     KBase - equivalent spring stiffness of the base in each DOF.  order: (x, theta_x, y, theta_y, z, theta_z)
     rigidDirection - if true the direction has infinite stiffness and corresponding rows/columns are removed from global matrices
     nodes - number of nodes in beam
     KFull, MFull, NotherFull, NFull, FFull - the original finite element matrices.  size: DOF*nodes x DOF*nodes
     
     Out:
     K,M,Nother,N,F - the new reduced finite element matrices.  
     
     Return:
     legnth - the new length of the reduced finite element matricies.  i.e. size(K) = length x length
     **/
    int applyBaseBoundaryCondition(const double Kbase[], const bool rigidDirections[], int nodes, 
                                           Matrix &KFull, const Matrix &MFull, const Matrix &NotherFull, 
                                           const Matrix &NFull, const Vector &FFull, 
                                           Matrix &K, Matrix &M, Matrix &Nother, Matrix &N, Vector &F);
    
    
    
    /**
     Add point loads at the nodes
     
     Arguments:
     nodes - number of nodes
     Fx,Fy,Fz,Mx,My,Mz - external point loads at each node.  length: nodes
     
     In/Out:
     F - finite element force vector.  length: DOF*nodes
     
     **/
    void addPointLoads(int nodes, Vector &F,
                              const Vector &FxPoint, const Vector &FyPoint, const Vector &FzPoint,
                              const Vector &MxPoint, const Vector &MyPoint, const Vector &MzPoint);
    


   
}
    
    

#endif
