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
*/

#include <math.h>

#include "coordtrans.h"
#include "NRutil.h"

/* -------------------------------------------------------------------------
COORD_TRANS - calculate the 9 elements of the block-diagonal 12-by-12
coordinate transformation matrix, t1, t2, ..., t9.  

These coordinate transformation factors are used to:
* transform frame element end forces from the element (local) coordinate system 
to the structral (global) coordinate system.  
* transfrom end displacements from the structural (global) coordinate system
to the element (local) coordinate system,
* transform the frame element stiffness and mass matrices
from element (local) coordinates to structral (global) coordinates.

Element matrix coordinate transformations are carried out by function ATMA
in frame3dd.c

Currently coordinate transformations do not consider the effect of 
finite node sizes ... this needs work, and could require a substantial
re-write of much of the code.  

Currently the effect of finite node sizes is used only in the calculation
of the element stiffness matrices.  
------------------------------------------------------------------------- */
void coord_trans(
		vec3 *xyz,
		double L,
		int n1, int n2,
		double *t1, double *t2, double *t3, double *t4, double *t5,
		double *t6, double *t7, double *t8, double *t9,
		float p			/**< the roll angle (radians) */
){
	double	Cx, Cy, Cz, den,		/* direction cosines	*/
		Cp, Sp;			/* cosine and sine of roll angle */

	Cx = (xyz[n2].x - xyz[n1].x) / L;
	Cy = (xyz[n2].y - xyz[n1].y) / L;
	Cz = (xyz[n2].z - xyz[n1].z) / L;

	*t1 = *t2 = *t3 = *t4 = *t5 = *t6 = *t7 = *t8 = *t9 = 0.0;

	Cp = cos(p);
	Sp = sin(p);

#if Zvert				// the global Z axis is vertical

	if ( fabs(Cz) == 1.0 ) {
		*t3 =  Cz;
		*t4 = -Cz*Sp;
		*t5 =  Cp;
		*t7 = -Cz*Cp;
		*t8 = -Sp;
	} else {

		den = sqrt ( 1.0 - Cz*Cz );

		*t1 = Cx;
	   	*t2 = Cy;
		*t3 = Cz;

		*t4 = (-Cx*Cz*Sp - Cy*Cp)/den;    
 		*t5 = (-Cy*Cz*Sp + Cx*Cp)/den;
		*t6 = Sp*den;

		*t7 = (-Cx*Cz*Cp + Cy*Sp)/den;
		*t8 = (-Cy*Cz*Cp - Cx*Sp)/den;
	   	*t9 = Cp*den;
	}

#else					// the global Y axis is vertical

	if ( fabs(Cy) == 1.0 ) {
		*t2 =  Cy;
		*t4 = -Cy*Cp;
		*t6 =  Sp;
		*t7 =  Cy*Sp;
		*t9 =  Cp;
	} else {

		den = sqrt ( 1.0 - Cy*Cy );

		*t1 = Cx;
	   	*t2 = Cy;
		*t3 = Cz;

		*t4 = (-Cx*Cy*Cp - Cz*Sp)/den;    
		*t5 = den*Cp;
 		*t6 = (-Cy*Cz*Cp + Cx*Sp)/den;

		*t7 = (Cx*Cy*Sp - Cz*Cp)/den;
	   	*t8 = -den*Sp;
		*t9 = (Cy*Cz*Sp + Cx*Cp)/den;
	}

#endif

	return;
}


/* ------------------------------------------------------------------------------
 * ATMA  -  perform the coordinate transformation from local to global     6jan96
 *	  include effects of a finite node radii, r1 and r2.	    9dec04
 *	  ------------------------------------------------------------------------------*/
void atma(
	double t1, double t2, double t3,
	double t4, double t5, double t6,
	double t7, double t8, double t9,
	double **m, float r1, float r2
){
	double  **a, **ma, **dmatrix();
	int     i,j,k;

	a  = dmatrix(1,12,1,12);
	ma = dmatrix(1,12,1,12);

	for (i=1; i<=12; i++)
	    for (j=i; j<=12; j++) 
		ma[j][i] = ma[i][j] = a[j][i] = a[i][j] = 0.0;

	for (i=0; i<=3; i++) {
		a[3*i+1][3*i+1] = t1;
		a[3*i+1][3*i+2] = t2;
		a[3*i+1][3*i+3] = t3;
		a[3*i+2][3*i+1] = t4;
		a[3*i+2][3*i+2] = t5;
		a[3*i+2][3*i+3] = t6;
		a[3*i+3][3*i+1] = t7;
		a[3*i+3][3*i+2] = t8;
		a[3*i+3][3*i+3] = t9;
	}


/*  effect of finite node radius on coordinate transformation  ... */
/*  this needs work ... */
/*
 	a[5][1] =  r1*t7; 
  	a[5][2] =  r1*t8; 
  	a[5][3] =  r1*t9; 
  	a[6][1] = -r1*t4; 
  	a[6][2] = -r1*t5; 
  	a[6][3] = -r1*t6; 
 
  	a[11][7] = -r2*t7; 
  	a[11][8] = -r2*t8; 
  	a[11][9] = -r2*t9; 
  	a[12][7] =  r2*t4; 
  	a[12][8] =  r2*t5; 
  	a[12][9] =  r2*t6; 
*/

#ifdef MATRIX_DEBUG
	save_dmatrix( "aa", a, 1,12, 1,12, 0, "w");	 /*  save cord xfmtn */
#endif

	for (j=1; j <= 12; j++)			 /*  MT = M T     */
	    for (i=1; i <= 12; i++)
		for (k=1; k <= 12; k++)    ma[i][j] += m[i][k] * a[k][j];

#ifdef MATRIX_DEBUG
	save_dmatrix( "ma", ma, 1,12, 1,12, 0, "w");	/*  partial transformation */
#endif

	for (i=1; i<=12; i++)   for (j=i; j<=12; j++)   m[j][i] = m[i][j] = 0.0;

	for (j=1; j <= 12; j++)			 /*  T'MT = T' MT */
	    for (i=1; i <= 12; i++)
		for (k=1; k <= 12; k++)    m[i][j] += a[k][i] * ma[k][j];

#ifdef MATRIX_DEBUG
	save_dmatrix( "atma", m, 1,12, 1,12, 0, "w");	       /*  debug atma */
#endif

	free_dmatrix(a, 1,12,1,12);
	free_dmatrix(ma,1,12,1,12);
}
