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
*//** @file
	Main FRAME3DD program driver
*//** @mainpage
FRAME3DD: a program for static and dynamic structural analysis of 2D and 3D
frames and trusses with elastic and geometric stiffness.

Also included is a system for parsing Microstran .arc 'Archive' files and
for parsing calculated force and displacement output files (.p1 format) from
Microstran. See @ref mstranp. It is intended that ultimately the .arc format
be an alternative method of inputting data to the FRAME3DD program,
but currently these two parts of the code are distinct.

For more information go to http://frame3dd.sourceforge.net/

The input file format for FRAME is defined in doc/user_manual.html

Henri P. Gavin hpgavin@duke.edu (main FRAME3DD code) 
John Pye john.pye@anu.edu.au (Microstran parser and viewer)

For compilation/installation, see README.txt.

*/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"
#include "frame3dd.h"
#include "frame3dd_io.h"
#include "eig.h"
#include "HPGmatrix.h"
#include "HPGutil.h"
#include "NRutil.h"


// compile the Frame3DD analysis into another code, such as a GUI
#ifdef WITH_GLOBALS	
 int run_kernel ( int argc, char *argv[] ) {
#endif

// compile Frame3DD to run as a stand-alone code through the terminal
#ifndef WITH_GLOBALS
 int main ( int argc, char *argv[] ) {
#endif

	char	IN_file[FILENMAX],	// the input  data filename
		OUT_file[FILENMAX],	// the output data filename
		title[MAXL],		// the title of the analysis
		errMsg[MAXL],		// the text of an error message
		meshpath[FRAME3DD_PATHMAX] = "EMPTY_MESH", // mesh data path
		plotpath[FRAME3DD_PATHMAX] = "EMPTY_PLOT", // plot file path
		infcpath[FRAME3DD_PATHMAX] = "EMPTY_INFC", // int  file path
		modepath[FRAME3DD_PATHMAX] = "EMPTY_MODE", // mode data path
		strippedInputFile[FRAME3DD_PATHMAX] = "EMPTY_TEMP"; // temp data path

	FILE	*fp;		// input and output file pointer

	vec3	*xyz;		// X,Y,Z node coordinates (global)

	float	*rj = NULL,	// node size radius, for finite sizes
		*Ax,*Asy, *Asz,	// cross section areas, incl. shear
		*Jx,*Iy,*Iz,	// section inertias		
		*E=NULL, *G=NULL,// elastic modulus and shear moduli
		*p=NULL,	// roll of each member, radians	
		***U=NULL,	// uniform distributed member loads
		***W=NULL,	// trapizoidal distributed member loads
		***P=NULL,	// member concentrated loads	
		***T=NULL,	// member temperature  loads
		**Dp=NULL,	// prescribed node displacements
		*d, *EMs=NULL,	// member densities and extra inertia
		*NMs=NULL, 	// mass of a node
		*NMx,*NMy,*NMz,	// inertia of a node in global coord	
		gX[_NL_],	// gravitational acceleration in global X 
		gY[_NL_],	// gravitational acceleration in global Y
		gZ[_NL_],	// gravitational acceleration in global Z
		pan=1.0,	// >0: pan during animation; 0: don't
		scale=1.0,	// zoom scale for 3D plotting in Gnuplot
		dx=1.0;		// x-increment for internal force data

	double	**K=NULL,	// equilibrium stiffness matrix
		// **Ks=NULL,	// Broyden secant stiffness matrix	
		traceK = 0.0,	// trace of the global stiffness matrix
		**M = NULL,	// global mass matrix
		traceM = 0.0,	// trace of the global mass matrix
		***eqF_mech=NULL,// equivalent end forces from mech loads global
		***eqF_temp=NULL,// equivalent end forces from temp loads global
		**F_mech=NULL,	// mechanical load vectors, all load cases	
		**F_temp=NULL,	// thermal load vectors, all load cases
		*F  = NULL, 	// total load vectors for a load case
		*R  = NULL,	// total reaction force vector
		*dR = NULL,	// incremental reaction force vector 
		*D  = NULL,	// displacement vector
		*dD = NULL,	// incremental displacement vector
		//dDdD = 0.0,	// dD' * dD
		*dF = NULL,	// equilibrium error in nonlinear anlys
		*L  = NULL,	// node-to-node length of each element
		*Le = NULL,	// effcve lngth, accounts for node size
		**Q = NULL,	// local member node end-forces
		tol = 1.0e-9,	// tolerance for modal convergence
		shift = 0.0,	// shift-factor for rigid-body-modes
		struct_mass,	// mass of structural system	
		total_mass,	// total structural mass and extra mass 
		*f  = NULL,	// resonant frequencies	
		**V = NULL,	// resonant mode-shapes
		rms_resid=1.0,	// root mean square of residual displ. error
		error = 1.0,	// rms equilibrium error and reactions
		Cfreq = 0.0,	// frequency used for Guyan condensation
		**Kc, **Mc,	// condensed stiffness and mass matrices
		exagg_static=10,// exaggerate static displ. in mesh data
		exagg_modal=10;	// exaggerate modal displ. in mesh data

		// peak internal forces, moments, and displacments
		// in each frame element and each load case 
	double	**pkNx, **pkVy, **pkVz, **pkTx, **pkMy, **pkMz,
		**pkDx, **pkDy, **pkDz, **pkRx, **pkSy, **pkSz;

	int	nN=0,		// number of Nodes
		nE=0,		// number of frame Elements
		nL=0, lc=0,	// number of Load cases
		DoF=0, i, j,	// number of Degrees of Freedom
		nR=0,		// number of restrained nodes
		nD[_NL_],	// number of prescribed nodal displ'nts
		nF[_NL_],	// number of loaded nodes
		nU[_NL_],	// number of members w/ unifm dist loads
		nW[_NL_],	// number of members w/ trapz dist loads
		nP[_NL_],	// number of members w/ conc point loads
		nT[_NL_],	// number of members w/ temp. changes
		nI=0,		// number of nodes w/ extra inertia
		nX=0,		// number of elemts w/ extra mass
		nC=0,		// number of condensed nodes
		*N1, *N2,	// begin and end node numbers
		shear=0,	// indicates shear deformation
		geom=0,		// indicates  geometric nonlinearity
		anlyz=1,	// 1: stiffness analysis, 0: data check	
		*q=NULL,*r=NULL,sumR,	// reaction data, total no. of reactions
		nM=0,		// number of desired modes
		Mmethod,	// 1: Subspace Jacobi, 2: Stodola
		nM_calc,	// number of modes to calculate
		lump=1,		// 1: lumped, 0: consistent mass matrix
		iter=0,		// number of iterations	
		ok=1,		// number of (-ve) diag. terms of L D L'
		anim[128],	// the modes to be animated
		Cdof=0,		// number of condensed degrees o freedom
		Cmethod=0,	// matrix condensation method
		*c=NULL,	// vector of DoF's to condense
		*m=NULL,	// vector of modes to condense
		filetype=0,	// 1 if .CSV, 2 if file is Matlab
		debug=0,	// 1: debugging screen output, 0: none
		verbose=1,	// 1: copious screen output, 0: none
		axial_strain_warning = 0, // 0: "ok", 1: strain > 0.001
		ExitCode = 0;	// error code returned by Frame3DD

	int	shear_flag= -1,	//   over-ride input file value	
		geom_flag = -1,	//   over-ride input file value
		anlyz_flag= -1,	//   over-ride input file value
		D3_flag = -1,	//   over-ride 3D plotting check
		lump_flag = -1,	//   over-ride input file value
		modal_flag= -1,	//   over-ride input file value	
		write_matrix=-1,//   write stiffness and mass matrix
		axial_sign=-1,  //   suppress 't' or 'c' in output data
		condense_flag=-1; // over-ride input file value	

	int	sfrv=0;		// *scanf return value for err checking

	double	exagg_flag=-1.0, // over-ride input file value
		tol_flag  =-1.0, // over-ride input file value
		shift_flag=-1.0; // over-ride input file value

	float	pan_flag = -1.0; // over-ride input file value

	char	extn[16];	// Input Data file name extension


	parse_options ( argc, argv, IN_file, OUT_file, 
			&shear_flag, &geom_flag, &anlyz_flag, &exagg_flag, 
			&D3_flag, 
			&lump_flag, &modal_flag, &tol_flag, &shift_flag, 
			&pan_flag, &write_matrix, &axial_sign, &condense_flag,
			&verbose, &debug);

	if ( verbose ) { /*  display program name, version and license type */
		textColor('w','b','b','x');
		fprintf(stdout,"\n FRAME3DD version: %s\n", VERSION);
		fprintf(stdout," Analysis of 2D and 3D structural frames with elastic and geometric stiffness.\n");
		fprintf(stdout," http://frame3dd.sf.net\n");
		fprintf(stdout," GPL Copyright (C) 1992-2014, Henri P. Gavin\n");
		fprintf(stdout," This is free software with absolutely no warranty.\n");
		fprintf(stdout," For details, see the GPL license file, LICENSE.txt\n");
		color(0); fprintf(stdout,"\n");
	}

	/* open the input data file */

	if ((fp = fopen (IN_file, "r")) == NULL) { /* open input data file */
		sprintf (errMsg,"\n ERROR: cannot open input data file '%s'\n", IN_file);
		errorMsg(errMsg); 
		display_help();
		if ( argc == 1 ) {
			fprintf(stderr," Press the 'Enter' key to close.\n");
			(void) getchar();	// clear the buffer ?? 
			while( !getchar() ) ;	// wait for the Enter key 
		}
		exit(11);
	}

	filetype = get_file_ext( IN_file, extn ); /* .CSV or .FMM or other? */

//	temp_file_location("frame3dd.3dd",strippedInputFile,FRAME3DD_PATHMAX);
	output_path("frame3dd.3dd",strippedInputFile,FRAME3DD_PATHMAX,NULL);

	parse_input(fp, strippedInputFile);	/* strip comments from input data */
	fclose(fp);

	if ((fp = fopen (strippedInputFile, "r")) == NULL) { /* open stripped input file */
		sprintf(errMsg,"\n ERROR: cannot open stripped input data file '%s'\n", strippedInputFile);
		errorMsg(errMsg); 
		exit(13);
	}

	frame3dd_getline(fp, title, MAXL);
	if ( verbose ) {	/*  display analysis title */
		textColor('w','g','b','x');
		fprintf(stdout,"\n");
		fprintf(stdout," ** %s ** \n", title );
		color(0);
		fprintf(stdout,"\n");
	}

	sfrv=fscanf(fp, "%d", &nN );		/* number of nodes	*/
	if (sfrv != 1)	sferr("nN value for number of nodes");
	if ( verbose ) {	/* display nN */
		fprintf(stdout," number of nodes ");
		dots(stdout,36);	fprintf(stdout," nN =%4d ",nN);
	}

					/* allocate memory for node data ... */
	rj  =  vector(1,nN);		/* rigid radius around each node */
	xyz = (vec3 *)malloc(sizeof(vec3)*(1+nN));	/* node coordinates */

	read_node_data ( fp, nN, xyz, rj );
	if ( verbose )	printf(" ... complete\n");

	DoF = 6*nN;		/* total number of degrees of freedom	*/

	q   = ivector(1,DoF);	/* allocate memory for reaction data ... */
	r   = ivector(1,DoF);	/* allocate memory for reaction data ... */
	read_reaction_data ( fp, DoF, nN, &nR, q, r, &sumR, verbose );
	if ( verbose )	fprintf(stdout," ... complete\n");

	sfrv=fscanf(fp, "%d", &nE );	/* number of frame elements	*/
	if (sfrv != 1)	sferr("nE value for number of frame elements");
	if ( verbose ) {	/* display nE */
		fprintf(stdout," number of frame elements");
		dots(stdout,28);	fprintf(stdout," nE =%4d ",nE);
	}
	if ( nN > nE + 1) {	/* not enough elements */
		fprintf(stderr,"\n  warning: %d nodes and %d members...", nN, nE );
		fprintf(stderr," not enough elements to connect all nodes.\n");
    	}

				/* allocate memory for frame elements ... */
	L   = dvector(1,nE);	/* length of each element		*/
	Le  = dvector(1,nE);	/* effective length of each element	*/

	N1  = ivector(1,nE);	/* node #1 of each element		*/
	N2  = ivector(1,nE);	/* node #2 of each element		*/

	Ax  =  vector(1,nE);	/* cross section area of each element	*/
	Asy =  vector(1,nE);	/* shear area in local y direction 	*/
	Asz =  vector(1,nE);	/* shear area in local z direction	*/
	Jx  =  vector(1,nE);	/* torsional moment of inertia 		*/
	Iy  =  vector(1,nE);	/* bending moment of inertia about y-axis */
	Iz  =  vector(1,nE);	/* bending moment of inertia about z-axis */

	E   =  vector(1,nE);	/* frame element Young's modulus	*/
	G   =  vector(1,nE);	/* frame element shear modulus		*/
	p   =  vector(1,nE);	/* element rotation angle about local x axis */
	d   =  vector(1,nE);	/* element mass density			*/

	read_frame_element_data( fp, nN, nE, xyz,rj, L, Le, N1, N2,
					Ax, Asy, Asz, Jx, Iy, Iz, E, G, p, d );
	if ( verbose) 	fprintf(stdout," ... complete\n");


	read_run_data ( fp, OUT_file, &shear, shear_flag, &geom, geom_flag,
			meshpath, plotpath, infcpath, 
			&exagg_static, exagg_flag, &scale, &dx,
			&anlyz, anlyz_flag, debug );

	sfrv=fscanf(fp, "%d", &nL );	/* number of load cases		*/
	if (sfrv != 1)	sferr("nL value for number of load cases");
	if ( verbose ) {	/* display nL */
		fprintf(stdout," number of load cases ");
		dots(stdout,31);	fprintf(stdout," nL = %3d \n",nL);
	}

	if ( nL < 1 ) {	/* not enough load cases */
		errorMsg("\n ERROR: the number of load cases must be at least 1\n"); 
		exit(101);
	}
	if ( nL >= _NL_ ) { /* too many load cases */
		sprintf(errMsg,"\n ERROR: maximum of %d load cases allowed\n", _NL_-1);
		errorMsg(errMsg); 
		exit(102);
	}
					/* allocate memory for loads ... */
	U   =  D3matrix(1,nL,1,nE,1,4);    /* uniform load on each member */
	W   =  D3matrix(1,nL,1,10*nE,1,13);/* trapezoidal load on each member */
	P   =  D3matrix(1,nL,1,10*nE,1,5); /* internal point load each member */
	T   =  D3matrix(1,nL,1,nE,1,8);    /* internal temp change each member*/
	Dp  =  matrix(1,nL,1,DoF); /* prescribed displacement of each node */

	F_mech  = dmatrix(1,nL,1,DoF);	/* mechanical load vector	*/
	F_temp  = dmatrix(1,nL,1,DoF);	/* temperature load vector	*/
	F       = dvector(1,DoF);	/* external load vector	*/
	dF	= dvector(1,DoF);	/* equilibrium error {F} - [K]{D} */

	eqF_mech =  D3dmatrix(1,nL,1,nE,1,12); /* eqF due to mech loads */
	eqF_temp =  D3dmatrix(1,nL,1,nE,1,12); /* eqF due to temp loads */

	K   = dmatrix(1,DoF,1,DoF);	/* global stiffness matrix	*/
	Q   = dmatrix(1,nE,1,12);	/* end forces for each member	*/

	D   = dvector(1,DoF);	/* displacments of each node		*/
	dD  = dvector(1,DoF);	/* incremental displ. of each node	*/
	R   = dvector(1,DoF);	/* reaction forces			*/
	dR  = dvector(1,DoF);	/* incremental reaction forces		*/

	EMs =  vector(1,nE);	/* lumped mass for each frame element	*/
	NMs =  vector(1,nN);	/* node mass for each node		*/
	NMx =  vector(1,nN);	/* node inertia about global X axis	*/
	NMy =  vector(1,nN);	/* node inertia about global Y axis	*/
	NMz =  vector(1,nN);	/* node inertia about global Z axis	*/

	c = ivector(1,DoF); 	/* vector of condensed degrees of freedom */
	m = ivector(1,DoF); 	/* vector of condensed mode numbers	*/

	// peak axial forces, shears, torques, and moments along each element
	pkNx = dmatrix(1,nL,1,nE);	
	pkVy = dmatrix(1,nL,1,nE);
	pkVz = dmatrix(1,nL,1,nE);
	pkTx = dmatrix(1,nL,1,nE);
	pkMy = dmatrix(1,nL,1,nE);
	pkMz = dmatrix(1,nL,1,nE); 

	// peak displacements and slopes along each element
	pkDx = dmatrix(1,nL,1,nE);
	pkDy = dmatrix(1,nL,1,nE);
	pkDz = dmatrix(1,nL,1,nE);
	pkRx = dmatrix(1,nL,1,nE);
	pkSy = dmatrix(1,nL,1,nE);
	pkSz = dmatrix(1,nL,1,nE); 

	read_and_assemble_loads( fp, nN, nE, nL, DoF, xyz, L, Le, N1, N2,
			Ax,Asy,Asz, Iy,Iz, E, G, p,
			d, gX, gY, gZ, r, shear,
			nF, nU, nW, nP, nT, nD,
			Q, F_temp, F_mech, F, U, W, P, T,
			Dp, eqF_mech, eqF_temp, verbose );

	if ( verbose ) {	/* display load data complete */
		fprintf(stdout,"                                                     ");
		fprintf(stdout," load data ... complete\n");
	}

	read_mass_data( fp, IN_file, nN, nE, &nI, &nX,
			d, EMs, NMs, NMx, NMy, NMz,
			L, Ax, &total_mass, &struct_mass, &nM,
			&Mmethod, modal_flag, 
			&lump, lump_flag, &tol, tol_flag, &shift, shift_flag,
			&exagg_modal, modepath, anim, &pan, pan_flag, 
			verbose, debug );
	if ( verbose ) {	/* display mass data complete */
		fprintf(stdout,"                                                     ");
		fprintf(stdout," mass data ... complete\n");
	}

	read_condensation_data( fp, nN,nM, &nC, &Cdof, 
			&Cmethod, condense_flag, c,m, verbose );

	if( nC>0 && verbose ) {	/*  display condensation data complete */
		fprintf(stdout,"                                      ");
		fprintf(stdout," matrix condensation data ... complete\n");
	}

	fclose(fp);		/* close the input data file */

	fp = fopen(OUT_file, "a"); /* open the output data file for appending */

	if(fp==NULL) {	/* unable to append to output data file */
		fprintf(stderr,"Unable to append to output data file '%s'!\n",
								OUT_file);
		exit(14);
	}

	write_input_data ( fp, title, nN,nE,nL, nD,nR, nF,nU,nW,nP,nT,
			xyz, rj, N1,N2, Ax,Asy,Asz, Jx,Iy,Iz, E,G, p,
			d, gX, gY, gZ, 
			F_temp, F_mech, Dp, r, U, W, P, T,
			shear, anlyz, geom );


	if ( anlyz ) {			/* solve the problem	*/
	 srand(time(NULL));
	 for (lc=1; lc<=nL; lc++) {	/* begin load case analysis loop */

		if ( verbose ) {	/* display the load case number  */
			fprintf(stdout,"\n");
			textColor('y','g','b','x');
			fprintf(stdout," Load Case %d of %d ... ", lc,nL );
			fprintf(stdout,"                                          ");

			fflush(stdout);
			color(0);
			fprintf(stdout,"\n");
		}

		/*  initialize displacements and displ. increment to {0}  */
		/*  initialize reactions     and react. increment to {0}  */
		for (i=1; i<=DoF; i++)	D[i] = dD[i] = R[i] = dR[i] = 0.0;	

		/*  initialize internal element end forces Q = {0}	*/
		for (i=1; i<=nE; i++)	for (j=1;j<=12;j++)	Q[i][j] = 0.0;

		/*  elastic stiffness matrix  [K({D}^(i))], {D}^(0)={0} (i=0) */
		assemble_K ( K, DoF, nE, xyz, rj, L, Le, N1, N2,
					Ax, Asy, Asz, Jx,Iy,Iz, E, G, p,
					shear, geom, Q, debug );

#ifdef MATRIX_DEBUG
		save_dmatrix ( "Ku", K, 1,DoF, 1,DoF, 0, "w" ); // unloaded stiffness matrix
#endif

		/* first apply temperature loads only, if there are any ... */
		if (nT[lc] > 0) {
			if ( verbose )
				fprintf(stdout," Linear Elastic Analysis ... Temperature Loads\n");

			/*  solve {F_t} = [K({D=0})] * {D_t} */
			solve_system(K,dD,F_temp[lc],dR,DoF,q,r,&ok,verbose,&rms_resid);

			/* increment {D_t} = {0} + {D_t} temp.-induced displ */
			for (i=1; i<=DoF; i++)	if (q[i]) D[i] += dD[i];
			/* increment {R_t} = {0} + {R_t} temp.-induced react */
			for (i=1; i<=DoF; i++)	if (r[i]) R[i] += dR[i];

			if (geom) {	/* assemble K = Ke + Kg */
			 /* compute   {Q}={Q_t} ... temp.-induced forces     */
			 element_end_forces ( Q, nE, xyz, L, Le, N1,N2,
				Ax, Asy,Asz, Jx,Iy,Iz, E,G, p,
				eqF_temp[lc], eqF_mech[lc], D, shear, geom,
				&axial_strain_warning );

			 /* assemble temp.-stressed stiffness [K({D_t})]     */
			 assemble_K ( K, DoF, nE, xyz, rj, L, Le, N1, N2,
						Ax,Asy,Asz, Jx,Iy,Iz, E, G, p,
						shear,geom, Q, debug );
			}
		}

		/* ... then apply mechanical loads only, if there are any ... */
		if ( nF[lc]>0 || nU[lc]>0 || nW[lc]>0 || nP[lc]>0 || nD[lc]>0 || 
		     gX[lc] != 0 || gY[lc] != 0 || gZ[lc] != 0 ) {
			if ( verbose )
				fprintf(stdout," Linear Elastic Analysis ... Mechanical Loads\n");
			/* incremental displ at react'ns = prescribed displ */
			for (i=1; i<=DoF; i++)	if (r[i]) dD[i] = Dp[lc][i];

			/*  solve {F_m} = [K({D_t})] * {D_m}	*/
			solve_system(K,dD,F_mech[lc],dR,DoF,q,r,&ok,verbose,&rms_resid);

			/* combine {D} = {D_t} + {D_m}	*/
			for (i=1; i<=DoF; i++) {
				if (q[i])	D[i] += dD[i];
				else {		D[i]  = Dp[lc][i]; dD[i] = 0.0; }
			}
			/* combine {R} = {R_t} + {R_m} --- for linear systems */
			for (i=1; i<=DoF; i++)	if (r[i]) R[i] += dR[i];
		}


		/*  combine {F} = {F_t} + {F_m} */
		for (i=1; i<=DoF; i++)	F[i] = F_temp[lc][i] + F_mech[lc][i]; 

		/*  element forces {Q} for displacements {D}	*/ 
		element_end_forces ( Q, nE, xyz, L, Le, N1,N2,
				Ax, Asy,Asz, Jx,Iy,Iz, E,G, p, 
				eqF_temp[lc], eqF_mech[lc], D, shear, geom,
				&axial_strain_warning );

		/*  check the equilibrium error	*/
		error = equilibrium_error ( dF, F, K, D, DoF, q,r );

		if ( geom && verbose )
			fprintf(stdout,"\n Non-Linear Elastic Analysis ...\n");

/*
 * 		if ( geom ) { // initialize Broyden secant stiffness matrix, Ks 
 *			Ks  = dmatrix( 1, DoF, 1, DoF );
 *			for (i=1;i<=DoF;i++) {
 *				for(j=i;j<=DoF;j++) {
 *					Ks[i][j]=Ks[j][i]=K[i][j];
 *				}
 *			}
 *		}
 */

		/* quasi Newton-Raphson iteration for geometric nonlinearity  */
		if (geom) { error = 1.0; ok = 0; iter = 0; } /* re-initialize */
		while ( geom && error > tol && iter < 500 && ok >= 0) {

			++iter;

			/*  assemble stiffness matrix [K({D}^(i))]	      */
			assemble_K ( K, DoF, nE, xyz, rj, L, Le, N1, N2,
				Ax,Asy,Asz, Jx,Iy,Iz, E, G, p,
				shear,geom, Q, debug );

			/*  compute equilibrium error, {dF}, at iteration i   */
			/*  {dF}^(i) = {F} - [K({D}^(i))]*{D}^(i)	      */
			/*  convergence criteria = || {dF}^(i) ||  /  || F || */
			error = equilibrium_error ( dF, F, K, D, DoF, q,r );

			/*  Powell-Symmetric-Broyden secant stiffness update  */
			// PSB_update ( Ks, dF, dD, DoF );  /* not helpful?   */

			/*  solve {dF}^(i) = [K({D}^(i))] * {dD}^(i)	      */
			solve_system(K,dD,dF,dR,DoF,q,r,&ok,verbose,&rms_resid);

			if ( ok < 0 ) {	/*  K is not positive definite	      */
				fprintf(stderr,"   The stiffness matrix is not pos-def. \n");
				fprintf(stderr,"   Reduce loads and re-run the analysis.\n");
				ExitCode = 181;
				break;
			}

			/*  increment {D}^(i+1) = {D}^(i) + {dD}^(i)	      */
			for (i=1; i<=DoF; i++)	if ( q[i] )	D[i] += dD[i];

			/*  element forces {Q} for displacements {D}^(i)      */ 
			element_end_forces ( Q, nE, xyz, L, Le, N1,N2,
				Ax, Asy,Asz, Jx,Iy,Iz, E,G, p, 
				eqF_temp[lc], eqF_mech[lc], D, shear, geom,
				&axial_strain_warning );

			if ( verbose ) { /*  display equilibrium error        */
			 fprintf(stdout,"   NR iteration %3d ---", iter);
			 fprintf(stdout," RMS relative equilibrium error = %8.2e \n",error);
			}
		}			/* end quasi Newton-Raphson iteration */


		/*   strain limit failure ... */
		if (axial_strain_warning > 0 && ExitCode == 0)   ExitCode = 182;
		/*   strain limit _and_ buckling failure ... */
		if (axial_strain_warning > 0 && ExitCode == 181) ExitCode = 183;

 		if ( geom )	compute_reaction_forces( R,F,K, D, DoF, r );

		/*  dealocate Broyden secant stiffness matrix, Ks */
		// if ( geom )	free_dmatrix(Ks, 1, DoF, 1, DoF );

		if ( write_matrix )	/* write static stiffness matrix */
			save_ut_dmatrix ( "Ks", K, DoF, "w" );

		/*  display RMS equilibrium error */
		if ( verbose && ok >= 0 ) evaluate ( error, rms_resid, tol, geom );

 		write_static_results ( fp, nN,nE,nL, lc, DoF, N1,N2,
				F,D,R, r,Q, rms_resid, ok, axial_sign );

		if ( filetype == 1 ) {		// .CSV format output
			write_static_csv(OUT_file, title,
			    nN,nE,nL,lc, DoF, N1,N2, F,D,R, r,Q, error, ok );
		}

		if ( filetype == 2 ) {		// .m matlab format output
			write_static_mfile (OUT_file, title, nN,nE,nL,lc, DoF,
					N1,N2, F,D,R, r,Q, error, ok );
		}

/*
 *		if ( verbose ) 
 *		 printf("\n   If the program pauses here for very long,"
 *		 " hit CTRL-C to stop execution, \n"
 *		 "    reduce exagg_static in the Input Data,"
 *		 " and re-run the analysis. \n");
 */

		write_internal_forces ( OUT_file, fp, infcpath, lc, nL, title, dx, xyz,
					Q, nN, nE, L, N1, N2, 
					Ax, Asy, Asz, Jx, Iy, Iz, E, G, p,
					d, gX[lc], gY[lc], gZ[lc],
					nU[lc],U[lc],nW[lc],W[lc],nP[lc],P[lc],
					D, shear, error );

		static_mesh ( IN_file, infcpath, meshpath, plotpath, title,
					nN, nE, nL, lc, DoF,
					xyz, L, N1,N2, p, D,
					exagg_static, D3_flag, anlyz,
					dx, scale );

	 } /* end load case loop */
	} else {		/*  data check only  */
	
	 if ( verbose ) {	/* display data check only */
	 	fprintf(stdout,"\n * %s *\n", title );
	 	fprintf(stdout,"  DATA CHECK ONLY.\n");
	 }
	 static_mesh ( IN_file, infcpath, meshpath, plotpath, title,
			nN, nE, nL, lc, DoF,
			xyz, L, N1,N2, p, D,
			exagg_static, D3_flag, anlyz, dx, scale );
	}


	if ( nM > 0 ) { /* carry out modal analysis */

		if(verbose & anlyz) fprintf(stdout,"\n\n Modal Analysis ...\n");

		nM_calc = (nM+8)<(2*nM) ? nM+8 : 2*nM;		/* Bathe */

		M   = dmatrix(1,DoF,1,DoF);
		f   = dvector(1,nM_calc);
		V   = dmatrix(1,DoF,1,nM_calc);

		assemble_M ( M, DoF, nN, nE, xyz, rj, L, N1,N2,
				Ax, Jx,Iy,Iz, p, d, EMs, NMs, NMx, NMy, NMz,
				lump, debug );

#ifdef MATRIX_DEBUG
		save_dmatrix ( "Mf", M, 1,DoF, 1,DoF, 0, "w" ); /* free mass matrix */
#endif

		for (j=1; j<=DoF; j++) { /*  compute traceK and traceM */
			if ( !r[j] ) {
				traceK += K[j][j];
				traceM += M[j][j];
			}
		}
		for (i=1; i<=DoF; i++) { /*  modify K and M for reactions    */
			if ( r[i] ) {	/* apply reactions to upper triangle */
				K[i][i] = traceK * 1e4;
				M[i][i] = traceM;
				for (j=i+1; j<=DoF; j++)
					K[j][i]=K[i][j]=M[j][i]=M[i][j] = 0.0;
		    }
		}

		if ( write_matrix ) {	/* write Kd and Md matrices */
			save_ut_dmatrix ( "Kd", K, DoF, "w" );/* dynamic stff matx */
			save_ut_dmatrix ( "Md", M, DoF, "w" );/* dynamic mass matx */
		}

		if ( anlyz ) {	/* subspace or stodola methods */
			if( Mmethod == 1 )
				subspace( K, M, DoF, nM_calc, f, V, tol,shift,&iter,&ok, verbose );
			if( Mmethod == 2 )
				stodola ( K, M, DoF, nM_calc, f, V, tol,shift,&iter,&ok, verbose );

			for (j=1; j<=nM_calc; j++) f[j] = sqrt(f[j])/(2.0*PI);

			write_modal_results ( fp, nN,nE,nI, DoF, M,f,V,
					total_mass, struct_mass,
					iter, sumR, nM, shift, lump, tol, ok );
		}
	}

	fprintf(fp,"\n");
	fclose (fp);

	if ( nM > 0 && anlyz ) {	/* write modal analysis results */

		modal_mesh ( IN_file, meshpath, modepath, plotpath, title,
				nN,nE, DoF, nM, xyz, L, N1,N2, p,
				M, f, V, exagg_modal, D3_flag, anlyz );

		animate ( IN_file, meshpath, modepath, plotpath, title,anim,
				nN,nE, DoF, nM, xyz, L, p, N1,N2, f,
				V, exagg_modal, D3_flag, pan, scale );
	}

	if ( nC > 0 ) {		/* matrix condensation of stiffness and mass */

		if ( verbose ) fprintf(stdout,"\n Matrix Condensation ...\n");

		if(Cdof > nM && Cmethod == 3){
			fprintf(stderr,"  Cdof > nM ... Cdof = %d  nM = %d \n",
				 Cdof, nM );
			fprintf(stderr,"  The number of condensed degrees of freedom");
			fprintf(stderr," may not exceed the number of computed modes");
			fprintf(stderr," when using dynamic condensation.\n");
			exit(94);
		}

		Kc = dmatrix(1,Cdof,1,Cdof);
		Mc = dmatrix(1,Cdof,1,Cdof);

		if ( m[1] > 0 && nM > 0 )	Cfreq = f[m[1]];

		if ( Cmethod == 1 && anlyz) {	/* static condensation only */
			static_condensation(K, DoF, c, Cdof, Kc, 0 );
			if ( verbose )
				fprintf(stdout,"   static condensation of K complete\n");
		}
		if ( Cmethod == 2 && anlyz ) {  /*  dynamic condensation  */
			paz_condensation(M, K, DoF, c, Cdof, Mc,Kc, Cfreq, 0 ); 
			if ( verbose ) {
				fprintf(stdout,"   Paz condensation of K and M complete");
				fprintf(stdout," ... dynamics matched at %f Hz.\n", Cfreq );
			}
		}
		if ( Cmethod == 3 && nM > 0 && anlyz ) {
			modal_condensation(M,K, DoF, r, c, Cdof, Mc,Kc, V,f, m, 0 );
			if ( verbose ) 
				fprintf(stdout,"   modal condensation of K and M complete\n");
		}
		save_dmatrix("Kc", Kc, 1,Cdof, 1,Cdof, 0, "w" );
		save_dmatrix("Mc", Mc, 1,Cdof, 1,Cdof, 0, "w" );

		free_dmatrix(Kc, 1,Cdof,1,Cdof );
		free_dmatrix(Mc, 1,Cdof,1,Cdof );
	}


	/* deallocate memory used for each frame analysis variable */
	deallocate ( nN, nE, nL, nF, nU, nW, nP, nT, DoF, nM,
			xyz, rj, L, Le, N1, N2, q,r,
			Ax, Asy, Asz, Jx, Iy, Iz, E, G, p,
			U,W,P,T, Dp, F_mech, F_temp,
			eqF_mech, eqF_temp, F, dF, 
			K, Q, D, dD, R, dR, 
			d,EMs,NMs,NMx,NMy,NMz, M,f,V, c, m, 
			pkNx, pkVy, pkVz, pkTx, pkMy, pkMz,
			pkDx, pkDy, pkDz, pkRx, pkSy, pkSz
	);

	if ( verbose ) fprintf(stdout,"\n");

	if ( argc == 1 ) { /* wait for keyboard entry to close the terminal */
	   fprintf(stderr," The Output Data was appended to %s \n", OUT_file );
	   fprintf(stderr," A Gnuplot script was written to %s \n", plotpath );
	   fprintf(stderr," Press the 'Enter' key to close.\n");
	   (void) getchar();	// clear the buffer ?? 
	   while( !getchar() ) ;	// wait for the Enter key to be hit 
	}
	color(0);

	return( ExitCode );
}
