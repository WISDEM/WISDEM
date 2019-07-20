/*
 This file is part of FRAME3DD:
 Static and dynamic structural analysis of 2D and 3D frames and trusses with
 elastic and geometric stiffness.
 ---------------------------------------------------------------------------
 http://frame3dd.sourceforge.net/
 ---------------------------------------------------------------------------
 Copyright (C) 1992-2014  Henri P. Gavin
 
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
	Main functions of the FRAME3DD solver API
*/

#ifndef FRAME_PY_FRAME_H
#define FRAME_PY_FRAME_H

/* for Micro-Stran compatability, structure for cartesian vectors */
#include "microstran/vec3.h"

/* maximum number of load cases */
#define _NL_ 32


/** form the global stiffness matrix */
void assemble_K(
	double **K,		/**< stiffness matrix			*/
	int DoF,		/**< number of degrees of freedom	*/
	int nE,			/**< number of frame elements		*/
	int nN,			/**< number of frame nodes		*/
	vec3 *xyz,		/**< XYZ locations of every node	*/
	float *r,		/**< rigid radius of every node	*/
	double *L, double *Le,	/**< length of each frame element, effective */
	int *N1, int *N2,	/**< node connectivity			*/
	float *Ax, float *Asy, float *Asz,	/**< section areas	*/
	float *Jx, float *Iy, float *Iz,	/**< section inertias	*/
	float *E, float *G,	/**< elastic and shear moduli		*/
	float *p,		/**< roll angle, radians		*/
	int shear,		/**< 1: include shear deformation, 0: don't */
	int geom,		/**< 1: include goemetric stiffness, 0: don't */
	double **Q,		/**< frame element end forces		*/
	int debug,		/**< 1: write element stiffness matrices*/
	float *EKx, float *EKy, float *EKz,  // extra nodal stiffness
	float *EKtx, float *EKty, float *EKtz
		);


/** solve {F} =   [K]{D} via L D L' decomposition */
void solve_system(
	double **K,	/**< stiffness matrix for the restrained frame	*/
	double *D,	/**< displacement vector to be solved		*/
	double *F,	/**< external load vector			*/
	double *R,	/**< reaction vector				*/
	int DoF,	/**< number of degrees of freedom		*/
	int *q,		/**< 1: not a reaction; 0: a reaction coordinate */
	int *r,		/**< 0: not a reaction; 1: a reaction coordinate */
	int *ok,	/**< indicates positive definite stiffness matrix */
	int verbose,	/**< 1: copious screen output; 0: none		*/
	double *rms_resid /**< the RMS error of the solution residual */
);


/*
 * COMPUTE_REACTION_FORCES : R(r) = [K(r,q)]*{D(q)} + [K(r,r)]*{D(r)} - F(r)
 * reaction forces satisfy equilibrium in the solved system
 * 2012-10-12  , 2014-05-16
 */
void compute_reaction_forces(
	double *R, 	/**< computed reaction forces			*/
	double *F,	/**< vector of equivalent external loads	*/
	double **K,	/**< stiffness matrix for the solved system	*/
	double *D,	/**< displacement vector for the solved system	*/
	int DoF,	/**< number of structural coordinates		*/
	int *r		/**< 0: not a reaction; 1: a reaction coordinate */
);


/* add_feF :  add fixed end forces to internal element forces 
 * removed reaction calculations on 2014-05-14 
 *  
void add_feF(	
	vec3 *xyz,		//< XYZ locations of each node
	double *L,		//< length of each frame element, effective
	int *N1, int *N2,	//< node connectivity	
	float *p,		//< roll angle, radians	
	double **Q,		//< frame element end forces 
	double **eqF_temp,	//< temp. equiv.end forces for all frame elements
	double **eqF_mech,	//< mech. equiv.end forces for all frame elements 
	int nE,			//< number of frame elements
	int DoF,		//< number of degrees of freedom
	int verbose		//< 1: copious screen output; 0: none
);
*/


/*
 * EQUILBRIUM_ERROR - compute {dF} = {F} - [K_qq]{D_q} - [K_qr]{D_r}
 * and return ||dF|| / ||F||
 * use only the upper trianlge of [K_qq]
 */
double equilibrium_error(
	double *dF,	/**< equilibrium error  {dF} = {F} - [K]{D}	*/
	double *F,	/**< load vector                                */
	double **K,	/**< stiffness matrix for the restrained frame  */
	double *D,	/**< displacement vector to be solved           */
	int DoF,	/**< number of degrees of freedom               */
	int *q,		/**< 1: not a reaction; 0: a reaction coordinate */
	int *r		/**< 0: not a reaction; 1: a reaction coordinate */
);


/** evaluate the member end forces for every member */
void element_end_forces(
	double **Q,	/**< frame element end forces			*/
	int nE,		/**< number of frame elements			*/
	vec3 *xyz,	/** XYZ locations of each node			*/
	double *L, double *Le,	/**< length of each frame element, effective */
	int *N1, int *N2,	/**< node connectivity			*/
	float *Ax, float *Asy, float *Asz,	/**< section areas	*/
	float *Jx, float *Iy, float *Iz,	/**< section area inertias */
	float *E, float *G,	/**< elastic and shear moduli		*/
	float *p,		/**< roll angle, radians		*/
	double **eqF_temp, /**< equivalent temp loads on elements, global */
	double **eqF_mech, /**< equivalent mech loads on elements, global */
	double *D,	/**< displacement vector			*/
	int shear,	/**< 1: include shear deformation, 0: don't	*/
	int geom,	/**< 1: include goemetric stiffness, 0: don't	*/
	int *axial_strain_warning /** < 0: strains < 0.001         */ 
);



/** assemble global mass matrix from element mass & inertia */
void assemble_M(
	double **M,	/**< mass matrix				*/
	int DoF,	/**< number of degrees of freedom		*/
	int nN, int nE,	/**< number of nodes, number of frame elements	*/
	vec3 *xyz,	/** XYZ locations of each node			*/
	float *r,	/**< rigid radius of every node		*/
	double *L,	/**< length of each frame element, effective	*/
	int *N1, int *N2, /**< node connectivity			*/
	float *Ax,	/**< node connectivity				*/
	float *Jx, float *Iy, float *Iz,	/**< section area inertias*/
	float *p,	/**< roll angle, radians			*/
	float *d,	/**< frame element density			*/
	float *EMs,	/**< extra frame element mass			*/
	float *NMs,	/**< node mass					*/
	float *NMx, float *NMy, float *NMz,	/**< node inertias	*/
	int lump,	/**< 1: lumped mass matrix, 0: consistent mass	*/
	int debug	/**< 1: write element mass matrices	 	*/
);


/** static condensation of stiffness matrix from NxN to nxn */
void static_condensation(
	double **A,	/**< a square matrix				*/
	int N,		/**< the dimension of the matrix		*/
	int *q,		/**< list of matrix indices to retain		*/
	int n,		/**< the dimension of the condensed matrix	*/
	double **Ac,	/**< the condensed matrix			*/
	int verbose	/**< 1: copious screen output; 0: none		*/
);


/**     
 	Paz condensation of mass and stiffness matrices
	matches the response at a particular frequency, sqrt(L)/2/pi
        Paz M. Dynamic condensation. AIAA J 1984;22(5):724-727.
*/
void paz_condensation(
	double **M, double **K,	/**< mass and stiffness matrices	*/
	int N,			/**< dimension of the matrices, DoF	*/
	int *q,			/**< list of degrees of freedom to retain */
	int n,			/**< dimension of the condensed matrices */
	double **Mc, double **Kc,	/**< the condensed matrices	*/
	double w2,		/**< matched value of frequency squared	*/
	int verbose	/**< 1: copious screen output; 0: none		*/
);


/**
	dynamic condensation of mass and stiffness matrices
	matches the response at a set of frequencies

	@NOTE Kc and Mc may be ill-conditioned, and xyzsibly non-positive def.
*/
void modal_condensation(
	double **M, double **K,	/**< mass and stiffness matrices	*/
	int N,			/**< dimension of the matrices, DoF	*/
	int *R,		/**< R[i]=1: DoF i is fixed, R[i]=0: DoF i is free */
	int *p,		/**< list of primary degrees of freedom		*/
	int n,		/**< the dimension of the condensed matrix	*/
	double **Mc, double **Kc,	/**< the condensed matrices	*/
	double **V, double *f,	/**< mode shapes and natural frequencies*/
	int *m,		/**< list of modes to match in the condensed model */
	int verbose	/**< 1: copious screen output; 0: none		*/
);


/**
	release allocated memory
*/
void deallocate( 
	int nN, int nE, int nL, int *nF, int *nU, int *nW, int *nP, int *nT, int DoF,
	int modes,
	vec3 *xyz, float *rj, double *L, double *Le,
	int *N1, int *N2, int *q, int *r,
	float *Ax, float *Asy, float *Asz,
	float *Jx, float *Iy, float *Iz,
	float *E, float *G,
	float *p,
	float ***U, float ***W, float ***P, float ***T,
	float **Dp,
	double **F_mech, double **F_temp,
	double ***eqF_mech, double ***eqF_temp, double *F, double *dF, 
	double **K, double **Q,
	double *D, double *dD,
	double *R, double *dR,
	float *d, float *EMs,
	float *NMs, float *NMx, float *NMy, float *NMz,
	double **M, double *f, double **V, 
	int *c, int *m, 
	double **pkNx, double **pkVy, double **pkVz, double **pkTx, double **pkMy, double **pkMz,
	double **pkDx, double **pkDy, double **pkDz, double **pkRx, double **pkSy, double **pkSz,
	float *EKx, float *EKy, float *EKz, float *EKtx, float *EKty, float *EKtz
	
);


#endif /* FRAME_PY_FRAME_H */

