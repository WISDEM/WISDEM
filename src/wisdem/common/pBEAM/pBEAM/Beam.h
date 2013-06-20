//
//  Beam.h
//  pbeam
//
//  Created by Andrew Ning on 2/6/12.
//  Copyright (c) 2012 NREL. All rights reserved.
//

#ifndef pbeam_Beam_h
#define pbeam_Beam_h


#include "myMath.h"
#include "Poly.h"
#include "BeamFEA.h"

/**
 pBEAM = Polynomial Beam Element Analysis Module

 This is the main class which represents a structure composed of beam finite elements.
 Sectional properties vary between nodes as polynomials of any order.  Input data should
 be centered at the elastic center with axes in the principal directions.

 Though this code can stand alone it is meant to be used as a module as part of a larger program.
 A Python interface has been created to use the API of this class (see pyBeam.cpp).

 **/


class Beam {

    int nodes;

    PolyVec EA;
    PolyVec EIxx, EIyy;
    PolyVec GJ;
    PolyVec rhoA;
    PolyVec rhoJ;
    PolyVec Px;
    PolyVec Py;
    PolyVec Pz;

    Vector z_node;
    Vector Fx_node, Fy_node, Fz_node;
    Vector Mx_node, My_node, Mz_node;

    TipData tip;
    BaseData base;

    int length;
    Matrix K, M, N, Nother;
    Vector F;



public:

//MARK: ------------------ CONSTRUCTORS --------------------

    /**
     Constructor for beam with circular shell cross sections.

     Arguments:
     z_node - axial position of each section
     d_node - diameter of each section (assumed to vary linearly between nodes)
     t_node - wall thickness of each section (assumed to vary linearly between nodes)
     loads - applied loads (see BeamFEA.h)
     mat - material properties (see BeamFEA.h)
     tip - struct containing tip data (see BeamFEA.h)
     base - struct containing base b.c. (see BeamFEA.h)

     **/
    Beam(const Vector &z_node, const Vector &d_node, const Vector &t_node,
         const Loads& loads, const IsotropicMaterial &mat,
         const TipData &tip, const BaseData &base);


    /**
     Constructor for beam with general section data.
     All properties are assumed to vary linearly between sections.

     sec - sectional stiffness/inertial properties EIxx, GJ, etc. (see BeamFEA.h)
     loads - applied loads (see BeamFEA.h)
     tip - struct containing tip data (see BeamFEA.h)
     base - struct containing base b.c. (see BeamFEA.h)

     **/
     Beam(const SectionData& sec, const Loads& loads,
         const TipData &tip, const BaseData &base);


    /**
     Constructor for beam with general polynomial data.  Inputs are vectors
     of polynomials which describe the section property as a polynomial
     varying across the element in normalized coordinates.  Each PolyVec
     should be of length: nodes-1

     Arguments:
     z_node - axial position of each section
     EA_node - axial stiffness polynomial of each element
     EIxx - bending stiffness polynomial about x-axis of each element. (int(E*y*y, dA))
     EIyy - bending stiffness polynomial about y-axis of each element.
     GJ - torsional stiffness polynomial of each element
     rhoA - mass per unit length polynomial of each element
     rhoJ - polar moment of inertia per unit length polynomial of each element
     Px, Py, Pz - distributed loads polynomial of each element
     Fx_node, Fy_node, Fz_node - applied point force at each node
     Mx_node, My_node, Mz_node - applied point moment at each node
     tip - struct containing tip data (see BeamFEA.h)
     base - struct containing base b.c. (see BeamFEA.h)

     **/
    Beam(const PolynomialSectionData& sec, const PolynomialLoads& loads,
         const TipData &tip, const BaseData &base);


//    Beam(const Beam &b, int nothing);

    /**
     Returns number of nodes in structure

     **/
    int getNumNodes() const;


//MARK: ----------------- COMPUTATIONS ----------------


    /**
     Compute the mass of the structure

     Returns:
     mass

     **/
    double computeMass() const;


    /**
     Compute the out-of-plane mass moment of inertia of the structure
     int(rho z^2, dV)

     Returns:
     mass moment of inertia

     **/
    double computeOutOfPlaneMomentOfInertia() const;

    /**
     Compute the natural frequencies of the structure

     Arguments:
     n - number of natural frequencies to return (unless n exceeds the total DOF of the structure,
         in which case all computed natural frequencies will be returned.
         (currently there is no speed advantage to requesting less frequencies, as all of them
          are computed anyway.  However, future versions may use reduction methods)

     Out:
     freq - a vector which will be resized if necessary

     **/
    void computeNaturalFrequencies(int n, Vector &freq) const;
    void computeNaturalFrequencies(int n, Vector &freq, Matrix &vec) const;



    /**
     Compute the displacements of the structure (in FEA coordinate system)

     Out:
     displacements in x, y, z, theta_x, theta_y, theta_z at each node.

     **/
    void computeDisplacement(Vector &dx, Vector &dy, Vector&dz, Vector &dtheta_x, Vector &dtheta_y, Vector &dtheta_z) const;



    /**
     Estimates the minimum critical buckling loads due to axial loading in addition to any existing input loads.

     Our:
     Pcr_x - critical buckling load in the x-direction
     Pcr_y - critical buckling load in the y-direction

     **/
    void computeMinBucklingLoads(double &Pcr_x, double &Pcr_y) const;



    /**
     Computes the shear forces, axial forces, bending loads, and torsion loads as polynomials between nodes.
     Uses FEA coordinate system.

     Out:
     Vx - shear in x-direction.
     Vy - shear in y-direction
     Fz - axial force
     Mx - moment in x-z plane
     My - moment in y-z plane
     Tz - axial torsion

     **/
    void shearAndBending(PolyVec &Vx, PolyVec &Vy, PolyVec &Fz, PolyVec &Mx, PolyVec &My, PolyVec &Tz) const;


    /**
     Computes the axial strain along the structure at given locations.

     In:
     Provide vector of points to evalute strain (x,y,z) with elastic modulus E
     x(i) - x distance from elastic center for point i
     y(i) - y distance from elastic center for point i
     z(i) - z axial location of point i

     Out:
     epsilon_axial - axial strain at point (x(i), y(i), z(i)) due to axial loads and bi-directional bending

     **/
    void computeAxialStrain(Vector &x, Vector &y, Vector &z, Vector &epsilon_axial) const;


    // void computeShearStressForThinShellSections(Vector &x, Vector &y, Vector &z, Vector &E, Vector &sigma_axial) const;

private:

    // translation of definitions
    void translateFromGlobalToFEACoordinateSystem();

    // matrix assembly for the various constructors
    void assembleMatrices();

    // estimate natural frequencies and associated eigenvectors
    void naturalFrequencies(bool cmpVec, int n, Vector &freq, Matrix &vec) const;

    void computeDisplacementComponentsFromVector(const Vector &q, Vector &dx, Vector &dy, Vector&dz,
                                                 Vector &dtheta_x, Vector &dtheta_y, Vector &dtheta_z) const;

    // estimating critical buckling loads
    double estimateCriticalBucklingLoad(double FzExisting, int ix1, int ix2) const;
};



#endif
