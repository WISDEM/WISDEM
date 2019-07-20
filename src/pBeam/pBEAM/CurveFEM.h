//
//  CurveFEM.h
//  pbeam
//
//  Created by Andrew Ning on 2/6/12.
//  Copyright (c) 2012 NREL. All rights reserved.
//

#ifndef pbeam_CurveFEM_h
#define pbeam_CurveFEM_h

#include <vector>
#include "myMath.h"

/**
   CurveFEM

   This is a C++ port of the Andrew Ning modification of the original Scott Larwood Fortran90 CurveFEM code

   This has been included in pBEAM because of the similarity of purpose and function.  
   Additional work could better integrate the two codes further

**/


class CurveFEM {

  // Number of nodes, elements (nodes-1), and DOF (nodes*6)
  int nnode, ne, ndof;

  // Bandwidth of stiffness matrix
  int nb = 12; 
  
  // Orientation of element, True if vertical
  std::vector<bool> vertical_flag;

  // Rotational rate about global y-axis
  double omega;
  
  // Global node number, loc(i,j), jth corner of element i
  Eigen::MatrixX2i loc;

  // Element angular alignment
  Vector alpha;
  
  // Element lineal density
  Vector mu;

  // x, y, z coordinates of nodes
  Vector cx, cy, cz;

  // Set values for fixed degrees of freedom
  std::vector<int> kfix;


private:
  /*
    Transformation of a symmetric, banded matrix into a full matrix
    Outputs:
    Aout     | Full matrix (direct return)
  */
  Matrix band2full(Matrix &Ain);

  
  /*
    Calculates axial force for spinning finite elements. Coordinates are in CurveFAST Lj system.
    Outputs:
    f1       ! Element axial force (direct return)
  */
  Vector taper_axial_force();

  
  /* 
     Builds banded matrices for structure with tapered space frame elements in a rotating reference frame.
     Outputs:
     gyro     ! Global gyroscopic matrix (via pointer)
     cf       ! Global centrifugal force matrix (via pointer)
     kspin    ! Global gyroscopic matrix (via pointer)
  */
  void taper_frame_spin(Matrix &gyro, Matrix &cf, Matrix &kspin);

  
  /* 
     Builds banded mass and stiffness matrices for structure with space frame elements in double precision
     Inputs:
     ea       ! Element EA extensional stiffness
     eix      ! Element EIxx bending stiffness
     eiy      ! Element EIyy bending stiffness
     gj       ! Element GJ torsional stiffness
     jprime   ! Element lineal rotational mass moment of inertia
     Outputs:
     gm       ! Global mass matrix
     gs       ! Global stiffness matrix
  */
  void taper_mass_stiff(const Vector &ea, const Vector &eix, const Vector &eiy, const Vector &gj, const Vector &jprime,
			Matrix &gm, Matrix &gs);



  
public:

  /*
    Constructor

    Inputs:
    omegaRPM  ! Rotational speed, RPM
    StrcTwst  ! Structural twist
    BldR      ! Radial positions of nodes
    PrecrvRef ! Blade out of plane curvature
    PreswpRef ! Blade in-plane sweep
    BMassDen  ! Blade lineal density
    rootFix   | BC to hold root node fixed in all 6 DOF
  */
  CurveFEM(const double omegaRPM, const Vector &StrcTwst, const Vector &BldR, const Vector &PrecrvRef,
	   const Vector &PreswpRef, const Vector &BMassDen, const bool rootFix);


  /*
    Compute natural frequencies of structure (Hz)
    Inputs:
    ea    ! Blade extensional stiffness
    eix   ! Blade edgewise stiffness
    eiy   ! Blade flapwise stiffness
    gj    ! Blade torsional stiffness
    rhoJ  ! Blade axial mass moment of inertia per unit length

    Outputs:
    freqs       ! Global mass matrix (direct return)
  */  
  void frequencies(const Vector &ea, const Vector &eix, const Vector &eiy, const Vector &gj, const Vector &rhoJ, Vector &freqs, Matrix &eig_vec);
};



#endif
