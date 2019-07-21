//
//  CurveFEM.cpp
//  pbeam
//
//  Created by Garrett Barter on 7/11/18.
//  Copyright (c) 2018 NREL. All rights reserved.
//

#include <vector>
#include <cmath>
#include <iostream>
#include "CurveFEM.h"

#define _USE_MATH_DEFINES


CurveFEM::CurveFEM(const double omegaRPM, const Vector &StrcTwst, const Vector &BldR,
		   const Vector &PrecrvRef, const Vector &PreswpRef, const Vector &BMassDen,
		   const bool rootFix) {
  /****************************
    Constructor

    Inputs:
    omegaRPM  ! Rotational speed, RPM
    StrcTwst  ! Structural twist
    BldR      ! Radial positions of nodes
    PrecrvRef ! Blade out of plane curvature
    PreswpRef ! Blade in-plane sweep
    BMassDen  ! Blade lineal density
  */
  
  // Number of nodes, elements, and DOF
  nnode = BldR.size();
  ne    = nnode - 1;
  ndof  = 6 * nnode;
  
  // Convert to radians
  omega = omegaRPM * (2.0 * M_PI / 60.0);

  // Set values of global node numbers
  loc.resize(ne,2);
  for (int i=0; i<ne; i++) {
    // Number of inboard node
    loc(i,0) = i;

    // Number of outboard node
    loc(i,1) = i + 1;
  }

  // Set values for alpha, whose array size is the number of elements (tip not used)
  alpha = StrcTwst.array() * (M_PI / 180.0);

  // Set orientation of elements to horizontal
  vertical_flag.resize(ne);
  std::fill(vertical_flag.begin(), vertical_flag.end(), false);

  // Set coordinates
  cx = PrecrvRef.replicate(1,1);
  cy = PreswpRef.replicate(1,1);
  cz = BldR.replicate(1,1);

  // Mass properties
  mu = BMassDen;

  // Boundary conditions
  if (rootFix) kfix = {0, 1, 2, 3, 4, 5};
  else kfix = {};
}



Matrix CurveFEM::band2full(Matrix &Ain) {
  /****************************
  Purpose:
    Transformation of a symmetric, banded matrix into a full matrix
 
  Record of revisions:
     Date        Programmer      Description of change
     ====       =============   ==================================
  01/04/2008    S. Larwood      Original code
  07/11/2018    G. Barter       C++ code
  */

  Matrix Aout = Ain.replicate(1,1);
  
  // Build upper triangular
  for (int i=0; i<ndof; i++)
    for (int j=0; j<nb; j++)
      if (i+j < ndof)
	Aout(i,i+j) = Ain(i,j);

  // Symmetric terms
  for (int i=0; i<ndof; i++)
    for (int j=0; j<ndof; j++)
      Aout(j,i) = Aout(i,j);

  return Aout;
}




Vector CurveFEM::taper_axial_force() {
  /****************************
  Purpose:
    Calculates axial force for spinning finite elements. Coordinates are in CurveFAST Lj system.
 
  Reference:
    for transformation matrices-
    Rao, S. S. (2005). "The Finite Element Method in Engineering"
    Fourth Edition, Pergamon Press, Section 9.4
 
    and
 
    for axial force calculations-
    Leung, A. Y. T. and T. C. Fung (1988). "Spinning Finite Elements."
    Journal of Sound and Vibration 125(3): pp. 523-537.
 
  Record of revisions:
     Date        Programmer      Description of change
     ====       =============   ==================================
  01/18/2008    S. Larwood      Original code based on axial_force from 01/07/2008
  07/19/2013    S. Andrew Ning  commented out unused variables
  07/11/2018    G. Barter       C++ code

  Output: f1, Element axial force (Vector(ne))
  */

  // Data dictionary: declare local variable types & definitions
  Vector al(ne);  // Element length
  Vector b32(ne); // Element 1,1 of transformation matrix
  Vector b33(ne); // Element 1,3 of transformation matrix

  
  // Calculate required direction cosines and lengths for each element
  for (int i=0; i<ne; i++) {

    // Find element node number
    int ie = loc(i,0); // First corner of element
    int je = loc(i,1); // Second corner of element

    // Compute length
    al(i) = sqrt( pow(cx(je)-cx(ie), 2.0) + pow(cy(je)-cy(ie), 2.0) +
		  pow(cz(je)-cz(ie), 2.0) );

    // Compute elements of (3x3)transformation matrix
    // If element is vertical there is special handling
    if (vertical_flag[i]) {

      b32(i) = 0.0;
      b33(i) = 0.0;
      
    } else {

      b32(i) = (cy(je) - cy(ie))/al(i);
      b33(i) = (cz(je) - cz(ie))/al(i);
      
    }
  }


  // Initialize axial force values
  Vector f1(ne);
  f1.setZero();

  for (int i=0; i<ne; i++) {

    // Add up contribution from outer elements, except for last element
    if (i < ne-1) {

      for (int j=i+1; j<ne; j++) {

	// Find element node number
	int ie = loc(j,0); // First corner of element
	int je = loc(j,1); // Second corner of element

	f1(i) += pow(omega,2.0) * al(j)/2.0 * ( (b32(i) * cy(ie) + b33(i) * cz(ie)) *
						(mu(ie) + mu(je))  +
						al(j) * (b32(i) * b32(j) + b33(i) * b33(j)) 
						* (mu(ie) + 2.0 * mu(je))/3.0   );
	
      }

    }

    // Find element node number
    int ie = loc(i,0); // First corner of element
    int je = loc(i,1); // Second corner of element

    // Add contribution of current element to outer elements (if any)
    f1(i) += pow(omega,2.0) * al(i)/2.0 * ( (b32(i) * cy(ie) + b33(i) * cz(ie) ) *
					    (mu(ie) + mu(je)) 
					    + al(i) * (pow(b32(i), 2.0) + pow(b33(i), 2.0)) 
					    * (mu(ie) + 2.0 * mu(je))/3.0   );

  }

  return f1;
}





void CurveFEM::taper_frame_spin(Matrix &gyro, Matrix &cf, Matrix &kspin) {
  /****************************
  Purpose:
    Builds banded matrices for structure with tapered space frame elements in a rotating reference frame.
 
  Reference:
    Rao, S. S. (2005). "The Finite Element Method in Engineering"
    Fourth Edition, Pergamon Press, Section 9.4
 
    and
 
    Leung, A. Y. T. and T. C. Fung (1988). "Spinning Finite Elements."
    Journal of Sound and Vibration 125(3): pp. 523-537.
 
  Record of revisions:
     Date        Programmer      Description of change
     ====       =============   ==================================
  01/25/2008    S. Larwood      Original code based on frame_spin 01/16/2008
  07/19/2013    S. Andrew Ning  Removed matmul to use intrinsic MATMUL.  commented out unused vars
  07/11/2018    G. Barter       C++ code

   Outputs:
   gyro     ! Global gyroscopic matrix (via pointer)
   cf       ! Global centrifugal force matrix (via pointer)
   kspin    ! Global gyroscopic matrix (via pointer)
  */
  
  // Initialize global matrices
  cf.setZero();
  gyro.setZero();
  kspin.setZero();

  // Calculate axial forces for the elements
  Vector f1 = taper_axial_force();

  for (int ii=0; ii<ne; ii++) {

    // Initialize local element matrices
    Matrix cf_e(12, 12); cf_e.setZero(); // Element local/global centrifugal force matrix
    Matrix gyro_e(12, 12); gyro_e.setZero(); // Element local/global gyroscopic matrix
    Matrix kspin_e(12, 12); kspin_e.setZero(); // Element local/global spin stiffness matrix

    // Find node element node number
    int ie = loc(ii,0);
    int je = loc(ii,1);

    // Compute length
    double al = sqrt( pow(cx(je)-cx(ie), 2.0) + pow(cy(je)-cy(ie), 2.0) +
		      pow(cz(je)-cz(ie), 2.0) );
    double al2 = pow(al, 2.0);

    // Compute slopes
    double alz = (cx(je) - cx(ie))/al;
    double amz = (cy(je) - cy(ie))/al;
    double anz = (cz(je) - cz(ie))/al;

    // Compute elements of (3x3)transformation matrix
    // If element is vertical there is special handling
    Matrix b(3,3);
    double cs = cos(alpha(ii));
    double ss = sin(alpha(ii));
    if (vertical_flag[ii]) {
      
      b(0,0) = 0.0;
      b(0,1) = -ss;
      b(0,2) = -cs;
      b(1,0) = 0.0;
      b(1,1) = cs;
      b(1,2) = -ss;
      b(2,0) = 1.0;
      b(2,1) = 0.0;
      b(2,2) = 0.0;

    } else {

      double d = sqrt( pow(amz, 2.0) + pow(anz, 2.0) );
      double a11 = (pow(amz, 2.0) + pow(anz, 2.0)) / d;
      double a12 = -(alz * amz) /d;
      double a13 = -(alz * anz) /d;
      double a21 = 0.0;
      double a22 = anz/d;
      double a23 = -amz/d;
      b(0,0) = a11 * cs - a21 * ss;
      b(0,1) = a12 * cs - a22 * ss;
      b(0,2) = a13 * cs - a23 * ss;
      b(1,0) = a11 * ss + a21 * cs;
      b(1,1) = a12 * ss + a22 * cs;
      b(1,2) = a13 * ss + a23 * cs;
      b(2,0) = alz;
      b(2,1) = amz;
      b(2,2) = anz;
    }


    // Compute various parameters
    double a0  = pow(omega,2.0)*(b(2,1) * cy(ie) + b(2,2) * cz(ie));
    double b0  = pow(omega,2.0)*(pow(b(2,1), 2.0) + pow(b(2,2), 2.0));
    double c0  = mu(ie) * a0;
    double d0  = ((mu(je) - mu(ie)) * a0 + mu(ie) * b0 * al )/ al;
    double e0  = (mu(je) - mu(ie)) * b0 / al;
    double a11 = pow(b(0,1), 2.0) + pow(b(0,2), 2.0);
    double a12 = b(0,1)*b(1,1) + b(0,2)*b(1,2);
    double a13 = b(0,1)*b(2,1) + b(0,2)*b(2,2);
    double a22 = pow(b(1,1), 2.0) + pow(b(1,2), 2.0);
    double a23 = b(1,1)*b(2,1) + b(1,2)*b(2,2);
    double a33 = pow(b(2,1), 2.0) + pow(b(2,2), 2.0);
    double b1  = b(0,2)*b(1,1) - b(0,1)*b(1,2);
    double b2  = b(0,2)*b(2,1) - b(0,1)*b(2,2);
    double b3  = b(1,2)*b(2,1) - b(1,1)*b(2,2);

    // Build centrifugal force matrix
    cf_e(0,0)   =   (f1(ii)/(30.0*al)    * 36.0          
		     -c0/60.0            * 36.0          
		     -d0*al/420.0        * 72.0          
		     -e0*al2/2520.0    * 180.0);

    cf_e(0,4)   =   (f1(ii)/(30.0*al)    * 3.0*al        
		     -c0/60.0            * 6.0*al        
		     -d0*al/420.0        * 15.0*al       
		     -e0*al2/2520.0    * 42.0*al);

    cf_e(0,6)   =   (f1(ii)/(30.0*al)    * -36.0         
		     -c0/60.0            * -36.0         
		     -d0*al/420.0        * -72.0         
		     -e0*al2/2520.0    * -180.0);

    cf_e(0,10)  =   (f1(ii)/(30.0*al)    * 3.0*al        
		     -c0/60.0            * 0.0*al        
		     -d0*al/420.0        * -6.0*al       
		     -e0*al2/2520.0    * -30.0*al);

    cf_e(1,1)   =   (f1(ii)/(30.0*al)    * 36.0          
		     -c0/60.0            * 36.0          
		     -d0*al/420.0        * 72.0          
		     -e0*al2/2520.0    * 180.0);

    cf_e(1,3)   =   (f1(ii)/(30.0*al)    * -3.0*al       
		     -c0/60.0            * -6.0*al       
		     -d0*al/420.0        * -15.0*al      
		     -e0*al2/2520.0    * -42.0*al);

    cf_e(1,7)   =   (f1(ii)/(30.0*al)    * -36.0         
		     -c0/60.0            * -36.0         
		     -d0*al/420.0        * -72.0         
		     -e0*al2/2520.0    * -180.0);

    cf_e(1,9)  =   (f1(ii)/(30.0*al)    * -3.0*al       
		    -c0/60.0            * 0.0*al        
		    -d0*al/420.0        * 6.0*al        
		    -e0*al2/2520.0    * 30.0*al);

    cf_e(3,3)   =   (f1(ii)/(30.0*al)    * 4.0*al2     
		     -c0/60.0            * 2.0*al2     
		     -d0*al/420.0        * 4.0*al2     
		     -e0*al2/2520.0    * 11.0*al2);

    cf_e(3,7)   =   (f1(ii)/(30.0*al)    * 3.0*al        
		     -c0/60.0            * 6.0*al        
		     -d0*al/420.0        * 15.0*al       
		     -e0*al2/2520.0    * 42.0*al);

    cf_e(3,9)  =   (f1(ii)/(30.0*al)    * -al2        
		    -c0/60.0            * -al2        
		    -d0*al/420.0        * -3.0*al2    
		    -e0*al2/2520.0    * -11.0*al2);

    cf_e(4,4)   =   (f1(ii)/(30.0*al)    * 4.0*al2     
		     -c0/60.0            * 2.0*al2     
		     -d0*al/420.0        * 4.0*al2     
		     -e0*al2/2520.0    * 11.0*al2);

    cf_e(4,6)   =   (f1(ii)/(30.0*al)    * -3.0*al       
		     -c0/60.0            * -6.0*al       
		     -d0*al/420.0        * -15.0*al      
		     -e0*al2/2520.0    * -42.0*al);

    cf_e(4,10)  =   (f1(ii)/(30.0*al)    * -al2        
		     -c0/60.0            * -al2        
		     -d0*al/420.0        * -3.0*al2    
		     -e0*al2/2520.0    * -11.0*al2);

    cf_e(6,6)   =   (f1(ii)/(30.0*al)    * 36.0          
		     -c0/60.0            * 36.0          
		     -d0*al/420.0        * 72.0          
		     -e0*al2/2520.0    * 180.0);

    cf_e(6,10)  =   (f1(ii)/(30.0*al)    * -3.0*al       
		     -c0/60.0            * 0.0*al        
		     -d0*al/420.0        * 6.0*al        
		     -e0*al2/2520.0    * 30.0*al);

    cf_e(7,7)   =   (f1(ii)/(30.0*al)    * 36.0          
		     -c0/60.0            * 36.0          
		     -d0*al/420.0        * 72.0          
		     -e0*al2/2520.0    * 180.0);

    cf_e(7,9)  =   (f1(ii)/(30.0*al)    * 3.0*al        
		    -c0/60.0            * 0.0*al        
		    -d0*al/420.0        * -6.0*al       
		    -e0*al2/2520.0    * -30.0*al);

    cf_e(9,9) =   (f1(ii)/(30.0*al)    * 4.0*al2     
		   -c0/60.0            * 6.0*al2     
		   -d0*al/420.0        * 18.0*al2        
		   -e0*al2/2520.0    * 65.0*al2);

    cf_e(10,10) =   (f1(ii)/(30.0*al)    * 4.0*al2     
		     -c0/60.0            * 6.0*al2     
		     -d0*al/420.0        * 18.0*al2    
		     -e0*al2/2520.0    * 65.0*al2);

    // Add symmetric terms
    for (int i=0; i<12; i++)
      for (int j=0; j<12; j++)
	cf_e(j,i) = cf_e(i,j);

    // Build local gyroscopic matrix
    gyro_e(0,1)     =  (mu(ie)/420.0            * 156.0*b1      + 
			(mu(je)-mu(ie))/840.0   * 72.0*b1);

    gyro_e(0,2)     =  (mu(ie)/420.0            * 147.0*b2      + 
			(mu(je)-mu(ie))/840.0   * 70.0*b2);

    gyro_e(0,3)     =  (mu(ie)/420.0            * -22.0*al*b1   + 
			(mu(je)-mu(ie))/840.0   * -14.0*al*b1);

    gyro_e(0,7)     =  (mu(ie)/420.0            * 54.0*b1       + 
			(mu(je)-mu(ie))/840.0   * 54.0*b1);

    gyro_e(0,8)     =  (mu(ie)/420.0            * 63.0*b2       + 
			(mu(je)-mu(ie))/840.0   * 56.0*b2);

    gyro_e(0,9)    =  (mu(ie)/420.0            * 13.0*al*b1    + 
		       (mu(je)-mu(ie))/840.0   * 12.0*al*b1);

    gyro_e(1,2)     =  (mu(ie)/420.0            * 147.0*b3      + 
			(mu(je)-mu(ie))/840.0   * 70.0*b3);

    gyro_e(1,4)     =  (mu(ie)/420.0            * -22.0*al*b1   + 
			(mu(je)-mu(ie))/840.0   * -14.0*al*b1);

    gyro_e(1,6)     =  (mu(ie)/420.0            * -54.0*b1      + 
			(mu(je)-mu(ie))/840.0   * -54.0*b1);

    gyro_e(1,8)     =  (mu(ie)/420.0            * 63.0*b3       + 
			(mu(je)-mu(ie))/840.0   * 56.0*b3);

    gyro_e(1,10)    =  (mu(ie)/420.0            * 13.0*al*b1    + 
			(mu(je)-mu(ie))/840.0   * 12.0*al*b1);

    gyro_e(2,3)     =  (mu(ie)/420.0            * 21.0*al*b3    + 
			(mu(je)-mu(ie))/840.0   * 14.0*al*b3);

    gyro_e(2,4)     =  (mu(ie)/420.0            * -21.0*al*b2   + 
			(mu(je)-mu(ie))/840.0   * -14.0*al*b2);

    gyro_e(2,6)     =  (mu(ie)/420.0            * -63.0*b2      + 
			(mu(je)-mu(ie))/840.0   * -70.0*b2);

    gyro_e(2,7)     =  (mu(ie)/420.0            * -63.0*b3      + 
			(mu(je)-mu(ie))/840.0   * -70.0*b3);

    gyro_e(2,9)    =  (mu(ie)/420.0            * -14.0*al*b3   + 
		       (mu(je)-mu(ie))/840.0   * -14.0*al*b3);

    gyro_e(2,10)    =  (mu(ie)/420.0            * 14.0*al*b2    + 
			(mu(je)-mu(ie))/840.0   * 14.0*al*b2);

    gyro_e(3,4)     =  (mu(ie)/420.0            * 4.0*al2*b1  + 
			(mu(je)-mu(ie))/840.0   * 3.0*al2*b1);

    gyro_e(3,6)     =  (mu(ie)/420.0            * 13.0*al*b1    + 
			(mu(je)-mu(ie))/840.0   * 14.0*al*b1);

    gyro_e(3,8)     =  (mu(ie)/420.0            * -14.0*al*b3   + 
			(mu(je)-mu(ie))/840.0   * -14.0*al*b3);

    gyro_e(3,10)    =  (mu(ie)/420.0            * -3.0*al2*b1 + 
			(mu(je)-mu(ie))/840.0   * -3.0*al2*b1);

    gyro_e(4,7)     =  (mu(ie)/420.0            * 13.0*al*b1    + 
			(mu(je)-mu(ie))/840.0   * 14.0*al*b1);

    gyro_e(4,8)     =  (mu(ie)/420.0            * 14.0*al*b2    + 
			(mu(je)-mu(ie))/840.0   * 14.0*al*b2);

    gyro_e(4,9)    =  (mu(ie)/420.0            * 3.0*al2*b1  + 
		       (mu(je)-mu(ie))/840.0   * 3.0*al2*b1);

    gyro_e(6,7)     =  (mu(ie)/420.0            * 156.0*b1      + 
			(mu(je)-mu(ie))/840.0   * 240.0*b1);

    gyro_e(6,8)     =  (mu(ie)/420.0            * 147.0*b2      + 
			(mu(je)-mu(ie))/840.0   * 224.0*b2);

    gyro_e(6,9)    =  (mu(ie)/420.0            * 22.0*al*b1    + 
		       (mu(je)-mu(ie))/840.0   * 30.0*al*b1);

    gyro_e(7,8)     =  (mu(ie)/420.0            * 147.0*b3      + 
			(mu(je)-mu(ie))/840.0   * 224.0*b3);

    gyro_e(7,10)    =  (mu(ie)/420.0            * 22.0*al*b1    + 
			(mu(je)-mu(ie))/840.0   * 30.0*al*b1);

    gyro_e(8,9)    =  (mu(ie)/420.0            * -21.0*al*b3   + 
		       (mu(je)-mu(ie))/840.0   * -22.0*al*b3);

    gyro_e(8,10)    =  (mu(ie)/420.0            * 21.0*al*b2    + 
			(mu(je)-mu(ie))/840.0   * 22.0*al*b2);

    gyro_e(9,10)   =  (mu(ie)/420.0            * 4.0*al2*b1  + 
		       (mu(je)-mu(ie))/840.0   * 5.0*al2*b1);


    // Add skew-symmetric terms
    for (int i=0; i<12; i++)
      for (int j=0; j<12; j++)
	gyro_e(j,i) = -gyro_e(i,j);

    // Multiply matrices by common factors
    gyro_e *= al * omega;

    // Build local spin stiffness matrix
    kspin_e(0,0)    =  (mu(ie)/420.0            * 156.0*a11         + 
			(mu(je)-mu(ie))/840.0   * 72.0*a11);

    kspin_e(0,1)    =  (mu(ie)/420.0            * 156.0*a12         + 
			(mu(je)-mu(ie))/840.0   * 72.0*a12);

    kspin_e(0,2)    =  (mu(ie)/420.0            * 147.0*a13         + 
			(mu(je)-mu(ie))/840.0   * 70.0*a13);

    kspin_e(0,3)    =  (mu(ie)/420.0            * -22.0*al*a12      + 
			(mu(je)-mu(ie))/840.0   * -14.0*al*a12);

    kspin_e(0,4)    =  (mu(ie)/420.0            * 22.0*al*a11       + 
			(mu(je)-mu(ie))/840.0   * 14.0*al*a11);

    kspin_e(0,6)    =  (mu(ie)/420.0            * 54.0*a11          + 
			(mu(je)-mu(ie))/840.0   * 54.0*a11);

    kspin_e(0,7)    =  (mu(ie)/420.0            * 54.0*a12          + 
			(mu(je)-mu(ie))/840.0   * 54.0*a12);

    kspin_e(0,8)    =  (mu(ie)/420.0            * 63.0*a13          + 
			(mu(je)-mu(ie))/840.0   * 56.0*a13);

    kspin_e(0,9)   =  (mu(ie)/420.0            * 13.0*al*a12       + 
		       (mu(je)-mu(ie))/840.0   * 12.0*al*a12);

    kspin_e(0,10)   =  (mu(ie)/420.0            * -13.0*al*a11      + 
			(mu(je)-mu(ie))/840.0   * -12.0*al*a11);

    kspin_e(1,1)    =  (mu(ie)/420.0            * 156.0*a22         + 
			(mu(je)-mu(ie))/840.0   * 72.0*a22);

    kspin_e(1,2)    =  (mu(ie)/420.0            * 147.0*a23         + 
			(mu(je)-mu(ie))/840.0   * 70.0*a23);

    kspin_e(1,3)    =  (mu(ie)/420.0            * -22.0*al*a22      + 
			(mu(je)-mu(ie))/840.0   * -14.0*al*a22);

    kspin_e(1,4)    =  (mu(ie)/420.0            * 22.0*al*a12       + 
			(mu(je)-mu(ie))/840.0   * 14.0*al*a12);

    kspin_e(1,6)    =  (mu(ie)/420.0            * 54.0*a12          + 
			(mu(je)-mu(ie))/840.0   * 54.0*a12);

    kspin_e(1,7)    =  (mu(ie)/420.0            * 54.0*a22          + 
			(mu(je)-mu(ie))/840.0   * 54.0*a22);

    kspin_e(1,8)    =  (mu(ie)/420.0            * 63.0*a23          + 
			(mu(je)-mu(ie))/840.0   * 56.0*a23);

    kspin_e(1,9)   =  (mu(ie)/420.0            * 13.0*al*a22       + 
		       (mu(je)-mu(ie))/840.0   * 12.0*al*a22);

    kspin_e(1,10)   =  (mu(ie)/420.0            * -13.0*al*a12      + 
			(mu(je)-mu(ie))/840.0   * -12.0*al*a12);

    kspin_e(2,2)    =  (mu(ie)/420.0            * 140.0*a33         + 
			(mu(je)-mu(ie))/840.0   * 70.0*a33);

    kspin_e(2,3)    =  (mu(ie)/420.0            * -21.0*al*a23      + 
			(mu(je)-mu(ie))/840.0   * -14.0*al*a23);

    kspin_e(2,4)    =  (mu(ie)/420.0            * 21.0*al*a13       + 
			(mu(je)-mu(ie))/840.0   * 14.0*al*a13);

    kspin_e(2,6)    =  (mu(ie)/420.0            * 63.0*a13          + 
			(mu(je)-mu(ie))/840.0   * 70.0*a13);

    kspin_e(2,7)    =  (mu(ie)/420.0            * 63.0*a23          + 
			(mu(je)-mu(ie))/840.0   * 70.0*a23);

    kspin_e(2,8)    =  (mu(ie)/420.0            * 70.0*a33          + 
			(mu(je)-mu(ie))/840.0   * 70.0*a33);

    kspin_e(2,9)   =  (mu(ie)/420.0            * 14.0*al*a23       + 
		       (mu(je)-mu(ie))/840.0   * 14.0*al*a23);

    kspin_e(2,10)   =  (mu(ie)/420.0            * -14.0*al*a13      + 
			(mu(je)-mu(ie))/840.0   * -14.0*al*a13);

    kspin_e(3,3)    =  (mu(ie)/420.0            * 4.0*al2*a22     + 
			(mu(je)-mu(ie))/840.0   * 3.0*al2*a22);

    kspin_e(3,4)    =  (mu(ie)/420.0            * -4.0*al2*a12    + 
			(mu(je)-mu(ie))/840.0   * -3.0*al2*a12);

    kspin_e(3,6)    =  (mu(ie)/420.0            * -13.0*al*a12      + 
			(mu(je)-mu(ie))/840.0   * -14.0*al*a12);

    kspin_e(3,7)    =  (mu(ie)/420.0            * -13.0*al*a22      + 
			(mu(je)-mu(ie))/840.0   * -14.0*al*a22);

    kspin_e(3,8)    =  (mu(ie)/420.0            * -14.0*al*a23      + 
			(mu(je)-mu(ie))/840.0   * -14.0*al*a23);

    kspin_e(3,9)   =  (mu(ie)/420.0            * -3.0*al2*a22    + 
		       (mu(je)-mu(ie))/840.0   * -3.0*al2*a22);

    kspin_e(3,10)   =  (mu(ie)/420.0            * 3.0*al2*a12     + 
			(mu(je)-mu(ie))/840.0   * 3.0*al2*a12);

    kspin_e(4,4)    =  (mu(ie)/420.0            * 4.0*al2*a11     + 
			(mu(je)-mu(ie))/840.0   * 3.0*al2*a11);

    kspin_e(4,6)    =  (mu(ie)/420.0            * 13.0*al*a11       + 
			(mu(je)-mu(ie))/840.0   * 14.0*al*a11);

    kspin_e(4,7)    =  (mu(ie)/420.0            * 13.0*al*a12       + 
			(mu(je)-mu(ie))/840.0   * 14.0*al*a12);

    kspin_e(4,8)    =  (mu(ie)/420.0            * 14.0*al*a13       + 
			(mu(je)-mu(ie))/840.0   * 14.0*al*a13);

    kspin_e(4,9)   =  (mu(ie)/420.0            * 3.0*al2*a12     + 
		       (mu(je)-mu(ie))/840.0   * 3.0*al2*a12);

    kspin_e(4,10)   =  (mu(ie)/420.0            * -3.0*al2*a11    + 
			(mu(je)-mu(ie))/840.0   * -3.0*al2*a11);

    kspin_e(6,6)    =  (mu(ie)/420.0            * 156.0*a11         + 
			(mu(je)-mu(ie))/840.0   * 240.0*a11);

    kspin_e(6,7)    =  (mu(ie)/420.0            * 156.0*a12         + 
			(mu(je)-mu(ie))/840.0   * 240.0*a12);

    kspin_e(6,8)    =  (mu(ie)/420.0            * 147.0*a13         + 
			(mu(je)-mu(ie))/840.0   * 224.0*a13);

    kspin_e(6,9)   =  (mu(ie)/420.0            * 22.0*al*a12       + 
		       (mu(je)-mu(ie))/840.0   * 30.0*al*a12);

    kspin_e(6,10)   =  (mu(ie)/420.0            * -22.0*al*a11      + 
			(mu(je)-mu(ie))/840.0   * -30.0*al*a11);

    kspin_e(7,7)    =  (mu(ie)/420.0            * 156.0*a22         + 
			(mu(je)-mu(ie))/840.0   * 240.0*a22);

    kspin_e(7,8)    =  (mu(ie)/420.0            * 147.0*a23         + 
			(mu(je)-mu(ie))/840.0   * 224.0*a23);

    kspin_e(7,9)   =  (mu(ie)/420.0            * 22.0*al*a22       + 
		       (mu(je)-mu(ie))/840.0   * 30.0*al*a22);

    kspin_e(7,10)   =  (mu(ie)/420.0            * -22.0*al*a12      + 
			(mu(je)-mu(ie))/840.0   * -30.0*al*a12);

    kspin_e(8,8)    =  (mu(ie)/420.0            * 140.0*a33         + 
			(mu(je)-mu(ie))/840.0   * 210.0*a33);

    kspin_e(8,9)   =  (mu(ie)/420.0            * 21.0*al*a23       + 
		       (mu(je)-mu(ie))/840.0   * 22.0*al*a23);

    kspin_e(8,10)   =  (mu(ie)/420.0            * -21.0*al*a13      + 
			(mu(je)-mu(ie))/840.0   * -22.0*al*a13);

    kspin_e(9,9)  =  (mu(ie)/420.0            * 4.0*al2*a22     + 
		      (mu(je)-mu(ie))/840.0   * 5.0*al2*a22);

    kspin_e(9,10)  =  (mu(ie)/420.0            * -4.0*al2*a12    + 
		       (mu(je)-mu(ie))/840.0   * -5.0*al2*a12);

    kspin_e(10,10)  =  (mu(ie)/420.0            * 4.0*al2*a11     + 
			(mu(je)-mu(ie))/840.0   * 5.0*al2*a11);

    // Add symmetric terms
    for (int i=0; i<12; i++)
      for (int j=0; j<12; j++)
	kspin_e(j,i) = kspin_e(i,j);

    // Multiply matrices by common factors
    kspin_e *= ( al * pow(omega, 2.0) );

    // Initialize transformation matrix
    Matrix lambda(12,12);
    lambda.setZero();

    // Build transformation matrix
    for (int i=0; i<3; i++) {
      for (int j=0; j<3; j++) {

	// Set index values
	int i3 = i + 3;
	int j3 = j + 3;
	int i6 = i + 6;
	int j6 = j + 6;
	int i9 = i + 9;
	int j9 = j + 9;

	// Build 12x12 lambda matrix from 3x3 b matrix
	lambda(i,j) = b(i,j);
	lambda(i3,j3) = b(i,j);
	lambda(i6,j6) = b(i,j);
	lambda(i9,j9) = b(i,j);
      }
    }
    Matrix lambdat = lambda.transpose();
    
    // Multiply local centrifugal force matrix by transformation
    // Multiply previous result by transpose of transformation to obtain
    // element global centrifugal force matrix
    cf_e = lambdat * (cf_e * lambda);

    // Multiply local gyroscopic matrix by transformation
    // Multiply previous result by transpose of transformation to obtain
    // element global gyroscopic matrix
    gyro_e = lambdat * (gyro_e * lambda);

    // Multiply local spin stiffness matrix by transformation
    // Multiply previous result by transpose of transformation to obtain
    // element global spin stiffness matrix
    kspin_e = lambdat * (kspin_e * lambda);

    // Steps to assemble global stiffness matrices
    double n[12];
    for (int i=0; i<6; i++) {
      n[i]= 6*ie + i;
      n[i+6] = 6*je + i;
    }

    // Place this elements contribution into the global stiffness matrix
    for (int i=0; i<12; i++) {
      for (int j=0; j<12; j++) {
	
	int ik = n[i];
	int jk = n[j];
	int in = jk - ik;

	if (in >= 0) {
	  cf(ik,in) += cf_e(i,j);
	  gyro(ik,in) += gyro_e(i,j);
	  kspin(ik,in) += kspin_e(i,j);
	}
      }
    }


  }
}



void CurveFEM::taper_mass_stiff(const Vector &ea, const Vector &eix, const Vector &eiy, const Vector &gj, const Vector &jprime,
				Matrix &gm, Matrix &gs) {
  /****************************
  Purpose:
    Builds banded mass and stiffness matrices for structure with space frame elements in double precision
 
  Reference:
    Rao, S. S. (2005). "The Finite Element Method in Engineering"
    Fourth Edition, Pergamon Press, Section 12.3.2, p. 428
 
  Record of revisions:
     Date        Programmer      Description of change
     ====       =============   ==================================
  09/05/2008    S. Larwood      Fixed errors in ml(3,3) and ml(6,6)
  01/25/2008    S. Larwood      Original code based on taper_mass 09/19/2007
  07/19/2013    S. Andrew Ning  Removed matmul to use intrinsic MATMUL.  commented out unused vars.
  07/11/2018    G. Barter       C++ code

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

  // Initialize global stiffness matrix
  gm.setZero();
  gs.setZero();

  for (int ii=0; ii<ne; ii++) {

    // Initialize local element matrices
    Matrix ml(12, 12); ml.setZero();
    Matrix kl(12, 12); kl.setZero();

    // Find node element node number
    int ie = loc(ii,0);
    int je = loc(ii,1);

    // Compute length
    double al = sqrt( pow(cx(je)-cx(ie), 2.0) + pow(cy(je)-cy(ie), 2.0) +
		      pow(cz(je)-cz(ie), 2.0) );
    double al2 = pow(al, 2.0);
    double al3 = pow(al, 3.0);

    // Compute slopes
    double alz = (cx(je) - cx(ie))/al;
    double amz = (cy(je) - cy(ie))/al;
    double anz = (cz(je) - cz(ie))/al;

    // Build local stiffness matrix
    ml(0,0)     = al/35.0 * (10.0 * mu(ie) + 3.0 * mu(je) );
    ml(0,4)     = al2/420.0 * (17.0 * mu(ie) + 5.0 * mu(je) );
    ml(0,6)     = 9.0 * al/140.0 * (mu(ie) + mu(je) );
    ml(0,10)    = -al2/420.0 * (7.0 * mu(ie) + 6.0 * mu(je) );
    ml(1,1)     = al/35.0 * (10.0 * mu(ie) + 3.0 * mu(je) );
    ml(1,3)     = -al2/420.0 * (17.0 * mu(ie) + 5.0 * mu(je) );
    ml(1,7)     = 9.0 * al/140.0 * (mu(ie) + mu(je) );
    ml(1,9)     = al2/420.0 * (7.0 * mu(ie) + 6.0 * mu(je) );
    ml(2,2)     = al/12.0 * (3.0 * mu(ie) + mu(je) );
    ml(2,8)     = al/12.0 * (mu(ie) + mu(je) );
    ml(3,3)     = al3/840.0 * (5.0 * mu(ie) + 3.0 * mu(je) );
    ml(3,7)     = -al2/420.0 * (6.0 * mu(ie) + 7.0 * mu(je) );
    ml(3,9)     = -al3/280.0 * (mu(ie) + mu(je) );
    ml(4,4)     = al3/840.0 * (5.0 * mu(ie) + 3.0 * mu(je) );
    ml(4,6)     = al2/420.0 * (6.0 * mu(ie) + 7.0 * mu(je) );
    ml(4,10)    = -al3/280.0 * (mu(ie) + mu(je) );
    ml(5,5)     = al/12.0 * (3.0 * jprime(ie) + jprime(je) );
    ml(5,11)    = al/12.0 * (jprime(ie) + jprime(je) );
    ml(6,6)     = al/35.0 * (3.0 * mu(ie) + 10.0 * mu(je) );
    ml(6,10)    = -al2/420.0 * (7.0 * mu(ie) + 15.0 * mu(je) );
    ml(7,7)     = al/35.0 * (3.0 * mu(ie) + 10.0 * mu(je) );
    ml(7,9)     = al2/420.0 * (7.0 * mu(ie) + 15.0 * mu(je) );
    ml(8,8)     = al/4.0 * (mu(ie)/3.0 + mu(je) );
    ml(9,9)     = al3/840.0 * (3.0 * mu(ie) + 5.0 * mu(je) );
    ml(10,10)   = al3/840.0 * (3.0 * mu(ie) + 5.0 * mu(je) );
    ml(11,11)   = al/4.0 * (jprime(ie)/3.0 + jprime(je) );

    // Build local stiffness matrix- terms in upper triangle
    kl(0,0)     = 6.0/al3 * (eiy(ie) + eiy(je));
    kl(0,4)     = 2.0/al2 * (2.0 * eiy(ie) + eiy(je));
    kl(0,6)     = -6.0/al3 * (eiy(ie) + eiy(je));
    kl(0,10)    = 2.0/al2 * (eiy(ie) + 2.0 * eiy(je));
    kl(1,1)     = 6.0/al3 * (eix(ie) + eix(je));
    kl(1,3)     = -2.0/al2 * (2.0* eix(ie) + eix(je));
    kl(1,7)     = -6.0/al3 * (eix(ie) + eix(je));
    kl(1,9)     = -2.0/al2 * (eix(ie) + 2.0 * eix(je));
    kl(2,2)     = (ea(ie) + ea(je))/(2.0 * al);
    kl(2,8)     = -(ea(ie) + ea(je))/(2.0 * al);
    kl(3,3)     = 1.0/al * (3.0 * eix(ie) + eix(je));
    kl(3,7)     = 2.0/al2 * (2.0 * eix(ie) + eix(je));
    kl(3,9)     = 1.0/al * (eix(ie) + eix(je));
    kl(4,4)     = 1.0/al * (3.0 * eiy(ie) + eiy(je));
    kl(4,6)     = -2.0/al2 * (2.0 * eiy(ie) + eiy(je));
    kl(4,10)    = 1.0/al * (eiy(ie) + eiy(je));
    kl(5,5)     = (gj(ie) + gj(je))/(2.0 * al);
    kl(5,11)    = -(gj(ie) + gj(je))/(2.0 * al);
    kl(6,6)     = 6.0/al3 * (eiy(ie) + eiy(je));
    kl(6,10)    = -2.0/al2 * (eiy(ie) + 2.0 * eiy(je));
    kl(7,7)     = 6.0/al3 * (eix(ie) + eix(je));
    kl(7,9)     = 2.0/al2 * (eix(ie) + 2.0 * eix(je));
    kl(8,8)     = (ea(ie) + ea(je))/(2.0 * al);
    kl(9,9)     = 1.0/al * (eix(ie) + 3.0 * eix(je));
    kl(10,10)   = 1.0/al * (eiy(ie) + 3.0 * eiy(je));
    kl(11,11)   = (gj(ie) + gj(je))/(2.0 * al);
    
    // Add symmetric terms
    for (int i=0; i<12; i++) {
      for (int j=0; j<12; j++) {
	ml(j,i) = ml(i,j);
	kl(j,i) = kl(i,j);
      }
    }
    
    // Compute elements of (3x3)transformation matrix
    // If element is vertical there is special handling
    Matrix b(3,3);
    double cs = cos(alpha(ii));
    double ss = sin(alpha(ii));
    if (vertical_flag[ii]) {
      
      b(0,0) = 0.0;
      b(0,1) = -ss;
      b(0,2) = -cs;
      b(1,0) = 0.0;
      b(1,1) = cs;
      b(1,2) = -ss;
      b(2,0) = 1.0;
      b(2,1) = 0.0;
      b(2,2) = 0.0;

    } else {

      double d = sqrt( pow(amz, 2.0) + pow(anz, 2.0) );
      double a11 = (pow(amz, 2.0) + pow(anz, 2.0)) / d;
      double a12 = -(alz * amz) /d;
      double a13 = -(alz * anz) /d;
      double a21 = 0.0;
      double a22 = anz/d;
      double a23 = -amz/d;
      b(0,0) = a11 * cs - a21 * ss;
      b(0,1) = a12 * cs - a22 * ss;
      b(0,2) = a13 * cs - a23 * ss;
      b(1,0) = a11 * ss + a21 * cs;
      b(1,1) = a12 * ss + a22 * cs;
      b(1,2) = a13 * ss + a23 * cs;
      b(2,0) = alz;
      b(2,1) = amz;
      b(2,2) = anz;
    }

    // Initialize transformation matrix
    Matrix lambda(12,12);
    lambda.setZero();

    // Build transformation matrix
    for (int i=0; i<3; i++) {
      for (int j=0; j<3; j++) {

	// Set index values
	int i3 = i + 3;
	int j3 = j + 3;
	int i6 = i + 6;
	int j6 = j + 6;
	int i9 = i + 9;
	int j9 = j + 9;

	// Build 12x12 lambda matrix from 3x3 b matrix
	lambda(i,j) = b(i,j);
	lambda(i3,j3) = b(i,j);
	lambda(i6,j6) = b(i,j);
	lambda(i9,j9) = b(i,j);
      }
    }
    Matrix lambdat = lambda.transpose();

    // Multiply local mass/stiffness by transformation
    // element global mass/stiffness matrix
    Matrix mg = lambdat * (ml * lambda);
    Matrix kg = lambdat * (kl * lambda);

    // Steps to assemble global stiffness matrices
    double n[12];
    for (int i=0; i<6; i++) {
      n[i]= 6*ie + i;
      n[i+6] = 6*je + i;
    }

    // Place this elements contribution into the global stiffness matrix
    for (int i=0; i<12; i++) {
      for (int j=0; j<12; j++) {
	
	int ik = n[i];
	int jk = n[j];
	int in = jk - ik;

	if (in >= 0) {
	  gm(ik,in) += mg(i,j);
	  gs(ik,in) += kg(i,j);
	}
      }
    }

  }

  
  // Incorporate boundary conditions
  for(auto const &ix: kfix) {
    gm(ix,0) *= 1e6;
    gs(ix,0) *= 1e6;
  }
}


void CurveFEM::frequencies(const Vector &ea, const Vector &eix, const Vector &eiy, const Vector &gj, const Vector &rhoJ,
  Vector &freqs, Matrix &eig_vec) {
  /****************************
  Purpose:
    Compute natural frequencies of structure (Hz)

  Record of revisions:
     Date        Programmer      Description of change
     ====       =============   ==================================
  07/18/2013    S. Andrew Ning  Simplified version of S. Larwood's main routine and read_input
  07/11/2018    G. Barter       C++ code

   Inputs:
    ea    ! Blade extensional stiffness
    eix   ! Blade edgewise stiffness
    eiy   ! Blade flapwise stiffness
    gj    ! Blade torsional stiffness
    rhoJ  ! Blade axial mass moment of inertia per unit length

   Outputs:
   freqs       ! Global mass matrix (direct return)
  */
  
  // Initialize matrices
  Matrix cf(ndof, ndof);    // Global centrifugal force matrix
  Matrix gm(ndof, ndof);    // Generalized mass matrix
  Matrix gs(ndof, ndof);    // Global stiffness matrix
  Matrix gyro(ndof, ndof);  // Global gyroscopic matrix
  Matrix kspin(ndof, ndof); // Global spin-stiffness matrix
  
  // Buil the mass and stiffness matrices
  // First avoid some singularities
  double eps = 9.999E-4;
  Vector myrhoJ = rhoJ.array() + eps;
  taper_mass_stiff(ea, eix, eiy, gj, myrhoJ, gm, gs);
  
  // Build the gyroscopic matrices
  taper_frame_spin(gyro, cf, kspin);
  
  // Convert various matrices from banded to full
  gs    = band2full(gs);
  gm    = band2full(gm);
  cf    = band2full(cf);
  kspin = band2full(kspin);

  // solve eigenvalues
  Vector eigs(ndof);
  // Matrix eig_vec(ndof, ndof);
  myMath::generalizedEigenvalues(gs + cf - kspin, gm, eigs, eig_vec);

  int idx = 0;
  std::vector<double> tmpFreq(ndof);
  for (int k=0; k<ndof; k++) {
    if (eigs(k) < 1e-6) continue;
    tmpFreq[idx] = sqrt(eigs(k)) / (2.0*M_PI);
    idx++;
  }

  // resize vector to actual length (after removing rigid modes and accounting for user input)
  // Vector freqs(idx);
  for (int k=0; k<idx; k++) freqs(k) = tmpFreq[k];
  
  // return freqs;
}
