/*
 This file is part of FRAME3DD:
 Static and dynamic structural analysis of 2D and 3D frames and trusses with
 elastic and geometric stiffness.
 ---------------------------------------------------------------------------
 http://frame3dd.sourceforge.net/
 ---------------------------------------------------------------------------
 Copyright (C) 1992-2009  Henri P. Gavin
 
    FRAME3DD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    FRAME3DD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with FRAME3DD.  If not, see <http://www.gnu.org/licenses/>.
*//**
	@file
	Coordinate transformations for the FRAME3DD solver API.
*/
#ifndef FRAME_COORDTRANS_H
#define FRAME_COORDTRANS_H

#include "microstran/vec3.h"

#include "common.h" 
#include "HPGutil.h" 


/**
	COORD_TRANS -  evaluate the 3D coordinate transformation coefficients 1dec04
	Default order of coordinate rotations...  typical for Y as the vertical axis
	1. rotate about the global Z axis
	2. rotate about the global Y axis
	3. rotate about the local  x axis --- element 'roll'

	If Zvert is defined as 1, then the order of coordinate rotations is typical
	for Z as the vertical axis
	1. rotate about the global Y axis
	2. rotate about the global Z axis
	3. rotate about the local  x axis --- element 'roll'

	Q=TF;   U=TD;   T'T=I;   Q=kU;   TF=kTD;   T'TF=T'kTD;   T'kT = K;   F=KD
*/
void coord_trans (
	vec3 *xyz, 			// XYZ coordinate of all nodes
	double L, 			// length of all beam elements
	int n1, int n2, 		// node connectivity
	double *t1, double *t2, double *t3, double *t4, double *t5, 
	double *t6, double *t7, double *t8, double *t9, // coord transformation
	float p				// the roll angle (radians) 
);

/**  
  ATMA - carry out the coordinate transformation
*/
void atma (
        double t1, double t2, double t3,
        double t4, double t5, double t6,
        double t7, double t8, double t9,
        double **m, float r1, float r2
);

#endif /* FRAME_COORDTRANS_H */

