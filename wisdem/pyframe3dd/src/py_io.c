/*
  S. Andrew Ning
  October 31, 2013
  Mimics frame3d_io.c but allows direct initialization rather than input files
*/

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <assert.h>

#include "common.h"
// #include "frame3dd_io.h"
#include "coordtrans.h"
#include "py_HPGmatrix.h"
#include "HPGutil.h"
#include "NRutil.h"
#include "py_io.h"





/*------------------------------------------------------------------------------
  READ_NODE_DATA  -  read node location data
  Oct 31 2013
  ------------------------------------------------------------------------------*/
int read_node_data(Nodes *nodes, int nN, vec3 *xyz, float *r ){

  int i, j;
  char errMsg[MAXL];

  for (i=1;i<=nN;i++) {       /* read node coordinates    */
    j = nodes->N[i-1];
    if ( j <= 0 || j > nN ) {
      sprintf(errMsg,"\nERROR: in node coordinate data, node number out of range\n(node number %d is <= 0 or > %d)\n", j, nN);
      errorMsg(errMsg);
      return 41;
    }
    xyz[j].x = nodes->x[i-1];
    xyz[j].y = nodes->y[i-1];
    xyz[j].z = nodes->z[i-1];
    r[j] = fabs(nodes->r[i-1]);
  }
  return 0;
}



/*------------------------------------------------------------------------------
  READ_FRAME_ELEMENT_DATA  -  read frame element property data
  Oct 31, 2013
  ------------------------------------------------------------------------------*/
int read_frame_element_data (Elements *elements,
			      int nN, int nE, vec3 *xyz, float *r,
			      double *L, double *Le,
			      int *N1, int *N2,
			      float *Ax, float *Asy, float *Asz,
			      float *Jx, float *Iy, float *Iz, float *E, float *G, float *p, float *d){

  int n1, n2, i, n, b;
  int *epn, epn0=0;   /* vector of elements per node */
  char errMsg[MAXL];

  epn = ivector(1,nN);

  for (n=1;n<=nN;n++) epn[n] = 0;

  for (i=1;i<=nE;i++) {       /* read frame element properties */
    b = elements->EL[i-1];
    if ( b <= 0 || b > nE ) {
      sprintf(errMsg,"\n  error in frame element property data: Element number out of range  \n Frame element number: %d  \n", b);
      errorMsg(errMsg);
      return 51;
    }
    N1[b] = elements->N1[i-1];
    N2[b] = elements->N2[i-1];

    epn[N1[b]] += 1;        epn[N2[b]] += 1;

    if ( N1[b] <= 0 || N1[b] > nN || N2[b] <= 0 || N2[b] > nN ) {
      sprintf(errMsg,"\n  error in frame element property data: node number out of range  \n Frame element number: %d \n", b);
      errorMsg(errMsg);
      return 52;
    }

    Ax[b] = elements->Ax[i-1];
    Asy[b] = elements->Asy[i-1];
    Asz[b] = elements->Asz[i-1];
    Jx[b] = elements->Jx[i-1];
    Iy[b] = elements->Iy[i-1];
    Iz[b] = elements->Iz[i-1];
    E[b] = elements->E[i-1];
    G[b] = elements->G[i-1];
    p[b] = elements->roll[i-1];
    d[b] = elements->density[i-1];

    p[b] = p[b]*PI/180.0;   /* convert from degrees to radians */

    if ( Ax[b] < 0 || Asy[b] < 0 || Asz[b] < 0 ||
	 Jx[b] < 0 ||  Iy[b] < 0 ||  Iz[b] < 0  ) {
      sprintf(errMsg,"\n  error in frame element property data: section property < 0 \n  Frame element number: %d  \n", b);
      errorMsg(errMsg);
      return 53;
    }
    if ( Ax[b] == 0 ) {
      sprintf(errMsg,"\n  error in frame element property data: cross section area is zero   \n  Frame element number: %d  \n", b);
      errorMsg(errMsg);
      return 54;
    }
    if ( (Asy[b] == 0 || Asz[b] == 0) && G[b] == 0 ) {
      sprintf(errMsg,"\n  error in frame element property data: a shear area and shear modulus are zero   \n  Frame element number: %d  \n", b);
      errorMsg(errMsg);
      return 55;
    }
    if ( Jx[b] == 0 ) {
      sprintf(errMsg,"\n  error in frame element property data: torsional moment of inertia is zero   \n  Frame element number: %d  \n", b);
      errorMsg(errMsg);
      return 56;
    }
    if ( Iy[b] == 0 || Iz[b] == 0 ) {
      sprintf(errMsg,"\n  error: cross section bending moment of inertia is zero   \n  Frame element number : %d  \n", b);
      errorMsg(errMsg);
      return 57;
    }
    if ( E[b] <= 0 || G[b] <= 0 ) {
      sprintf(errMsg,"\n  error : material elastic modulus E or G is not positive   \n  Frame element number: %d  \n", b);
      errorMsg(errMsg);
      return 58;
    }
    if ( d[b] <= 0 ) {
      sprintf(errMsg,"\n  error : mass density d is not positive   \n  Frame element number: %d  \n", b);
      errorMsg(errMsg);
      return 59;
    }
  }

  for (b=1;b<=nE;b++) {       /* calculate frame element lengths */
    n1 = N1[b];
    n2 = N2[b];

#define SQ(X) ((X)*(X))
    L[b] =  SQ( xyz[n2].x - xyz[n1].x ) +
      SQ( xyz[n2].y - xyz[n1].y ) +
      SQ( xyz[n2].z - xyz[n1].z );
#undef SQ

    L[b] = sqrt( L[b] );
    Le[b] = L[b] - r[n1] - r[n2];
    if ( n1 == n2 || L[b] == 0.0 ) {
      sprintf(errMsg,
	      " Frame elements must start and stop at different nodes\n  frame element %d  N1= %d N2= %d L= %e\n   Perhaps frame element number %d has not been specified.\n  or perhaps the Input Data file is missing expected data.\n",
	      b, n1,n2, L[b], i );
      errorMsg(errMsg);
      return 60;
    }
    if ( Le[b] <= 0.0 ) {
      sprintf(errMsg, " Node  radii are too large.\n  frame element %d  N1= %d N2= %d L= %e \n  r1= %e r2= %e Le= %e \n",
	      b, n1,n2, L[b], r[n1], r[n2], Le[b] );
      errorMsg(errMsg);
      return 61;
    }
  }

  for ( n=1; n<=nN; n++ ) {
    if ( epn[n] == 0 ) {
      sprintf(errMsg,"node or frame element property data:\n     node number %3d is unconnected. \n", n);
      sferr(errMsg);
      epn0 += 1;
    }
  }

  free_ivector(epn,1,nN);

  if ( epn0 > 0 ) return 42;

  return 0;
}


/*------------------------------------------------------------------------------
  READ_RUN_DATA  -  read information for analysis
  Oct 31, 2013
  ------------------------------------------------------------------------------*/

int read_run_data (OtherElementData *other, int *shear, int *geom, double *exagg_static, float *dx){

  *shear = other->shear;
  *geom = other->geom;
  *exagg_static = other->exagg_static;
  *dx = other->dx;

  if (*shear != 0 && *shear != 1) {
    errorMsg(" Rember to specify shear deformations with a 0 or a 1 \n after the frame element property info.\n");
    return 71;
  }

  if (*geom != 0 && *geom != 1) {
    errorMsg(" Rember to specify geometric stiffness with a 0 or a 1 \n after the frame element property info.\n");
    return 72;
  }

  if ( *exagg_static < 0.0 ) {
    errorMsg(" Remember to specify an exageration factor greater than zero.\n");
    return 73;
  }

  if ( *dx <= 0.0 && *dx != -1 ) {
    errorMsg(" Remember to specify a frame element increment greater than zero.\n");
    return 74;
  }

  return 0;
}



/*------------------------------------------------------------------------------
  READ_REACTION_DATA - Read fixed node displacement boundary conditions
  Oct 31, 2013
  ------------------------------------------------------------------------------*/
int read_reaction_data (Reactions *reactions, int DoF, int nN,
			int *nR, int *q, int *r, int *sumR, int verbose, int geom,
			float *EKx, float *EKy, float *EKz,
			float *EKtx, float *EKty, float *EKtz){

  int i,j;
  char errMsg[MAXL];

  for (i=1; i<=DoF; i++)  r[i] = 0;
  for (i=1; i<=nN; i++){
    EKx[i] = 0.0;
    EKy[i] = 0.0;
    EKz[i] = 0.0;
    EKtx[i] = 0.0;
    EKty[i] = 0.0;
    EKtz[i] = 0.0;
  }

  *nR = reactions->nK;
  if ( verbose ) {
    fprintf(stdout," number of nodes with reactions (or extra stiffness) ");
    dots(stdout,21);
    fprintf(stdout," nR =%4d ", *nR );
  }
  if ( *nR < 0 || *nR > DoF/6 ) {
    fprintf(stderr," number of nodes with reactions (or extras stiffness) ");
    dots(stderr,21);
    fprintf(stderr," nR = %3d ", *nR );
    sprintf(errMsg,"\n  error: valid ranges for nR is 0 ... %d \n", DoF/6 );
    errorMsg(errMsg);
    return 80;
  }


  for (i=1; i <= *nR; i++) {
    j = reactions->N[i-1];

    if ( j > nN ) {
      sprintf(errMsg,"\n  error in reaction data: node number %d is greater than the number of nodes, %d \n", j, nN );
      errorMsg(errMsg);
      return 81;
    }

    // save rigid locations (and extra stiffness if needed)
    if (reactions->Kx[i-1] == reactions->rigid){
      r[6*j-5] = 1;
    } else{
      EKx[j] = reactions->Kx[i-1];
    }
    if (reactions->Ky[i-1] == reactions->rigid){
      r[6*j-4] = 1;
    } else{
      EKy[j] = reactions->Ky[i-1];
    }
    if (reactions->Kz[i-1] == reactions->rigid){
      r[6*j-3] = 1;
    } else{
      EKz[j] = reactions->Kz[i-1];
    }
    if (reactions->Ktx[i-1] == reactions->rigid){
      r[6*j-2] = 1;
    } else{
      EKtx[j] = reactions->Ktx[i-1];
    }
    if (reactions->Kty[i-1] == reactions->rigid){
      r[6*j-1] = 1;
    } else{
      EKty[j] = reactions->Kty[i-1];
    }
    if (reactions->Ktz[i-1] == reactions->rigid){
      r[6*j] = 1;
    } else{
      EKtz[j] = reactions->Ktz[i-1];
    }

  }

  *sumR=0;    for (i=1;i<=DoF;i++)    *sumR += r[i];
  //if ( *sumR < 4 && geom) {
  //sprintf(errMsg,"\n  warning:  geometric stiffness can not be used for unrestrained structure.  set geom=0 or added more reactions.\n");
    //errorMsg(errMsg);
    //return 84;
  //}
  // if ( *sumR < 4 ) {
  //     sprintf(errMsg,"\n  Warning:  un-restrained structure   %d imposed reactions.\n  At least 4 reactions are required to support static loads.\n", *sumR );
  //     errorMsg(errMsg);
  //     /*  return 84; */
  // }
  if ( *sumR >= DoF ) {
    sprintf(errMsg,"\n  error in reaction data:  Fully restrained structure\n   %d imposed reactions >= %d degrees of freedom\n", *sumR, DoF );
    errorMsg(errMsg);
    return 85;
  }

  for (i=1; i<=DoF; i++)  if (r[i]) q[i] = 0; else q[i] = 1;

  return 0;
}


/*------------------------------------------------------------------------------
  READ_AND_ASSEMBLE_LOADS  -
  read load information data, assemble un-restrained load vectors
  09 Sep 2008
  ------------------------------------------------------------------------------*/
int read_and_assemble_loads (
			      LoadCase* loadcases,
			      int nN, int nE, int nL, int DoF,
			      vec3 *xyz,
			      double *L, double *Le,
			      int *J1, int *J2,
			      float *Ax, float *Asy, float *Asz,
			      float *Iy, float *Iz, float *E, float *G,
			      float *p,
			      float *d, float *gX, float *gY, float *gZ,
			      int *r,
			      int shear,
			      int *nF, int *nU, int *nW, int *nP, int *nT, int *nD,
			      double **Q,
			      double **F_temp, double **F_mech, double *Fo,
			      float ***U, float ***W, float ***P, float ***T, float **Dp,
			      double ***feF_mech, double ***feF_temp,
			      int verbose){

  float   hy, hz;         /* section dimensions in local coords */

  float   x1,x2, w1,w2;
  double  Ln, R1o, R2o, f01, f02;

  double  Nx1, Vy1, Vz1, Mx1=0.0, My1=0.0, Mz1=0.0, /* fixed end forces */
    Nx2, Vy2, Vz2, Mx2=0.0, My2=0.0, Mz2=0.0,
    Ksy, Ksz,       /* shear deformatn coefficients */
    a, b,           /* point load locations */
    t1, t2, t3, t4, t5, t6, t7, t8, t9; /* 3D coord Xfrm coeffs */
  int i,j,l, lc, n, n1, n2;
  char errMsg[MAXL];
  LoadCase lcase;
  PointLoads pL;
  UniformLoads uL;
  TrapezoidalLoads tL;
  ElementLoads eL;
  TemperatureLoads tempL;
  PrescribedDisplacements pD;

  for (j=1; j<=DoF; j++) Fo[j] = 0.0;
  for (j=1; j<=DoF; j++)
    for (lc=1; lc <= nL; lc++)
      F_temp[lc][j] = F_mech[lc][j] = 0.0;
  for (i=1; i<=12; i++)
    for (n=1; n<=nE; n++)
      for (lc=1; lc <= nL; lc++)
	feF_mech[lc][n][i] = feF_temp[lc][n][i] = 0.0;

  for (i=1; i<=DoF; i++)  for (lc=1; lc<=nL; lc++) Dp[lc][i] = 0.0;

  for (i=1;i<=nE;i++) for(j=1;j<=12;j++)  Q[i][j] = 0.0;


  for (lc = 1; lc <= nL; lc++) {      /* begin load-case loop */

    lcase = loadcases[lc-1];
    pL = lcase.pointLoads;
    uL = lcase.uniformLoads;
    tL = lcase.trapezoidalLoads;
    eL = lcase.elementLoads;
    tempL = lcase.temperatureLoads;
    pD = lcase.prescribedDisplacements;

    if ( verbose ) {  /*  display the load case number */
      textColor('y','g','b','x');
      fprintf(stdout," load case %d of %d: ", lc, nL );
      fprintf(stdout,"                                            ");
      fflush(stdout);
      color(0);
      fprintf(stdout,"\n");
    }

    /* gravity loads applied uniformly to all frame elements  */
    gX[lc] = lcase.gx;
    gY[lc] = lcase.gy;
    gZ[lc] = lcase.gz;

    for (n=1; n<=nE; n++) {

      n1 = J1[n]; n2 = J2[n];

      coord_trans ( xyz, L[n], n1, n2,
		    &t1, &t2, &t3, &t4, &t5, &t6, &t7, &t8, &t9, p[n] );

      feF_mech[lc][n][1]  = d[n]*Ax[n]*L[n]*gX[lc] / 2.0;
      feF_mech[lc][n][2]  = d[n]*Ax[n]*L[n]*gY[lc] / 2.0;
      feF_mech[lc][n][3]  = d[n]*Ax[n]*L[n]*gZ[lc] / 2.0;

      feF_mech[lc][n][4]  = d[n]*Ax[n]*L[n]*L[n] / 12.0 *
	( (-t4*t8+t5*t7)*gY[lc] + (-t4*t9+t6*t7)*gZ[lc] );
      feF_mech[lc][n][5]  = d[n]*Ax[n]*L[n]*L[n] / 12.0 *
	( (-t5*t7+t4*t8)*gX[lc] + (-t5*t9+t6*t8)*gZ[lc] );
      feF_mech[lc][n][6]  = d[n]*Ax[n]*L[n]*L[n] / 12.0 *
	( (-t6*t7+t4*t9)*gX[lc] + (-t6*t8+t5*t9)*gY[lc] );

      feF_mech[lc][n][7]  = d[n]*Ax[n]*L[n]*gX[lc] / 2.0;
      feF_mech[lc][n][8]  = d[n]*Ax[n]*L[n]*gY[lc] / 2.0;
      feF_mech[lc][n][9]  = d[n]*Ax[n]*L[n]*gZ[lc] / 2.0;

      feF_mech[lc][n][10] = d[n]*Ax[n]*L[n]*L[n] / 12.0 *
	( ( t4*t8-t5*t7)*gY[lc] + ( t4*t9-t6*t7)*gZ[lc] );
      feF_mech[lc][n][11] = d[n]*Ax[n]*L[n]*L[n] / 12.0 *
	( ( t5*t7-t4*t8)*gX[lc] + ( t5*t9-t6*t8)*gZ[lc] );
      feF_mech[lc][n][12] = d[n]*Ax[n]*L[n]*L[n] / 12.0 *
	( ( t6*t7-t4*t9)*gX[lc] + ( t6*t8-t5*t9)*gY[lc] );

    }                 /* end gravity loads */


    nF[lc] = pL.nF;
    if ( verbose ) {
      fprintf(stdout,"  number of loaded nodes ");
      dots(stdout,28);    fprintf(stdout," nF = %3d\n", nF[lc]);
    }
    for (i=1; i <= nF[lc]; i++) { /* ! global structural coordinates ! */
      j = pL.N[i-1];
      if ( j < 1 || j > nN ) {
	sprintf(errMsg,"\n  error in node load data: node number out of range ... Node : %d\n   Perhaps you did not specify %d node loads \n  or perhaps the Input Data file is missing expected data.\n", j, nF[lc] );
	errorMsg(errMsg);
	return 121;
      }

      F_mech[lc][6*j-5] = pL.Fx[i-1];
      F_mech[lc][6*j-4] = pL.Fy[i-1];
      F_mech[lc][6*j-3] = pL.Fz[i-1];
      F_mech[lc][6*j-2] = pL.Mxx[i-1];
      F_mech[lc][6*j-1] = pL.Myy[i-1];
      F_mech[lc][6*j] = pL.Mzz[i-1];

      //if ( F_mech[lc][6*j-5]==0 && F_mech[lc][6*j-4]==0 && F_mech[lc][6*j-3]==0 && F_mech[lc][6*j-2]==0 && F_mech[lc][6*j-1]==0 && F_mech[lc][6*j]==0 )
	//fprintf(stderr,"\n   Warning: All node loads applied at node %d  are zero\n", j );
    }   /* end node point loads  */

        /* uniformly distributed loads */
    nU[lc] = uL.nU;
    if ( verbose ) {
      fprintf(stdout,"  number of uniformly distributed loads ");
      dots(stdout,13);    fprintf(stdout," nU = %3d\n", nU[lc]);
    }
    if ( nU[lc] < 0 || nU[lc] > nE ) {
      fprintf(stderr,"  number of uniformly distributed loads ");
      dots(stderr,13);
      fprintf(stderr," nU = %3d\n", nU[lc]);
      sprintf(errMsg,"\n  error: valid ranges for nU is 0 ... %d \n", nE );
      errorMsg(errMsg);
      return 131;
    }
    for (i=1; i <= nU[lc]; i++) { /* ! local element coordinates ! */
      n = uL.EL[i-1];
      if ( n < 1 || n > nE ) {
	sprintf(errMsg,"\n  error in uniform distributed loads: element number %d is out of range\n",n);
	errorMsg(errMsg);
	return 132;
      }
      U[lc][i][1] = (double) n;
      U[lc][i][2] = uL.Ux[i-1];
      U[lc][i][3] = uL.Uy[i-1];
      U[lc][i][4] = uL.Uz[i-1];

      //if ( U[lc][i][2]==0 && U[lc][i][3]==0 && U[lc][i][4]==0 )
      //fprintf(stderr,"\n   Warning: All distributed loads applied to frame element %d  are zero\n", n );

      Nx1 = Nx2 = U[lc][i][2]*Le[n] / 2.0;
      Vy1 = Vy2 = U[lc][i][3]*Le[n] / 2.0;
      Vz1 = Vz2 = U[lc][i][4]*Le[n] / 2.0;
      Mx1 = Mx2 = 0.0;
      My1 = -U[lc][i][4]*Le[n]*Le[n] / 12.0;  My2 = -My1;
      Mz1 =  U[lc][i][3]*Le[n]*Le[n] / 12.0;  Mz2 = -Mz1;

      n1 = J1[n]; n2 = J2[n];

      coord_trans ( xyz, L[n], n1, n2,
		    &t1, &t2, &t3, &t4, &t5, &t6, &t7, &t8, &t9, p[n] );


      /* {F} = [T]'{Q} */
      feF_mech[lc][n][1]  += ( Nx1*t1 + Vy1*t4 + Vz1*t7 );
      feF_mech[lc][n][2]  += ( Nx1*t2 + Vy1*t5 + Vz1*t8 );
      feF_mech[lc][n][3]  += ( Nx1*t3 + Vy1*t6 + Vz1*t9 );
      feF_mech[lc][n][4]  += ( Mx1*t1 + My1*t4 + Mz1*t7 );
      feF_mech[lc][n][5]  += ( Mx1*t2 + My1*t5 + Mz1*t8 );
      feF_mech[lc][n][6]  += ( Mx1*t3 + My1*t6 + Mz1*t9 );

      feF_mech[lc][n][7]  += ( Nx2*t1 + Vy2*t4 + Vz2*t7 );
      feF_mech[lc][n][8]  += ( Nx2*t2 + Vy2*t5 + Vz2*t8 );
      feF_mech[lc][n][9]  += ( Nx2*t3 + Vy2*t6 + Vz2*t9 );
      feF_mech[lc][n][10] += ( Mx2*t1 + My2*t4 + Mz2*t7 );
      feF_mech[lc][n][11] += ( Mx2*t2 + My2*t5 + Mz2*t8 );
      feF_mech[lc][n][12] += ( Mx2*t3 + My2*t6 + Mz2*t9 );

    }             /* end uniformly distributed loads */

    /* trapezoidally distributed loads */
    nW[lc] = tL.nW;
    if ( verbose ) {
      fprintf(stdout,"  number of trapezoidally distributed loads ");
      dots(stdout,9); fprintf(stdout," nW = %3d\n", nW[lc]);
    }
    if ( nW[lc] < 0 || nW[lc] > 10*nE ) {
      sprintf(errMsg,"\n  error: valid ranges for nW is 0 ... %d \n", 10*nE );
      errorMsg(errMsg);
      return 140;
    }
    for (i=1; i <= nW[lc]; i++) { /* ! local element coordinates ! */
      n = tL.EL[i-1];
      if ( n < 1 || n > nE ) {
	sprintf(errMsg,"\n  error in trapezoidally-distributed loads: element number %d is out of range\n",n);
	errorMsg(errMsg);
	return 141;
      }
      W[lc][i][1] = (double) n;
      W[lc][i][2] = tL.xx1[i-1];
      W[lc][i][3] = tL.xx2[i-1];
      W[lc][i][4] = tL.wx1[i-1];
      W[lc][i][5] = tL.wx2[i-1];
      W[lc][i][6] = tL.xy1[i-1];
      W[lc][i][7] = tL.xy2[i-1];
      W[lc][i][8] = tL.wy1[i-1];
      W[lc][i][9] = tL.wy2[i-1];
      W[lc][i][10] = tL.xz1[i-1];
      W[lc][i][11] = tL.xz2[i-1];
      W[lc][i][12] = tL.wz1[i-1];
      W[lc][i][13] = tL.wz2[i-1];

      Ln = L[n];

      /* error checking */

      if ( W[lc][i][ 4]==0 && W[lc][i][ 5]==0 &&
	   W[lc][i][ 8]==0 && W[lc][i][ 9]==0 &&
	   W[lc][i][12]==0 && W[lc][i][13]==0 ) {
	//fprintf(stderr,"\n   Warning: All trapezoidal loads applied to frame element %d  are zero\n", n );
	//fprintf(stderr,"     load case: %d , element %d , load %d\n ", lc, n, i );
      }

      if ( W[lc][i][ 2] < 0 ) {
	sprintf(errMsg,"\n   error in x-axis trapezoidal loads, load case: %d , element %d , load %d\n  starting location = %f < 0\n",
		lc, n, i , W[lc][i][2]);
	errorMsg(errMsg);
	return 142;
      }
      if ( W[lc][i][ 2] > W[lc][i][3] ) {
	sprintf(errMsg,"\n   error in x-axis trapezoidal loads, load case: %d , element %d , load %d\n  starting location = %f > ending location = %f \n",
		lc, n, i , W[lc][i][2], W[lc][i][3] );
	errorMsg(errMsg);
	return 143;
      }
      if ( W[lc][i][ 3] > Ln ) {
	sprintf(errMsg,"\n   error in x-axis trapezoidal loads, load case: %d , element %d , load %d\n ending location = %f > L (%f) \n",
		lc, n, i, W[lc][i][3], Ln );
	errorMsg(errMsg);
	return 144;
      }
      if ( W[lc][i][ 6] < 0 ) {
	sprintf(errMsg,"\n   error in y-axis trapezoidal loads, load case: %d , element %d , load %d\n starting location = %f < 0\n",
		lc, n, i, W[lc][i][6]);
	errorMsg(errMsg);
	return 142;
      }
      if ( W[lc][i][ 6] > W[lc][i][7] ) {
	sprintf(errMsg,"\n   error in y-axis trapezoidal loads, load case: %d , element %d , load %d\n starting location = %f > ending location = %f \n",
		lc, n, i, W[lc][i][6], W[lc][i][7] );
	errorMsg(errMsg);
	return 143;
      }
      if ( W[lc][i][ 7] > Ln ) {
	sprintf(errMsg,"\n   error in y-axis trapezoidal loads, load case: %d , element %d , load %d\n ending location = %f > L (%f) \n",
		lc, n, i, W[lc][i][7],Ln );
	errorMsg(errMsg);
	return 144;
      }
      if ( W[lc][i][10] < 0 ) {
	sprintf(errMsg,"\n   error in z-axis trapezoidal loads, load case: %d , element %d , load %d\n starting location = %f < 0\n",
		lc, n, i, W[lc][i][10]);
	errorMsg(errMsg);
	return 142;
      }
      if ( W[lc][i][10] > W[lc][i][11] ) {
	sprintf(errMsg,"\n   error in z-axis trapezoidal loads, load case: %d , element %d , load %d\n starting location = %f > ending location = %f \n",
		lc, n, i, W[lc][i][10], W[lc][i][11] );
	errorMsg(errMsg);
	return 143;
      }
      if ( W[lc][i][11] > Ln ) {
	sprintf(errMsg,"\n   error in z-axis trapezoidal loads, load case: %d , element %d , load %d\n ending location = %f > L (%f) \n",lc, n, i, W[lc][i][11], Ln );
	errorMsg(errMsg);
	return 144;
      }

      if ( shear ) {
	Ksy = (12.0*E[n]*Iz[n]) / (G[n]*Asy[n]*Le[n]*Le[n]);
	Ksz = (12.0*E[n]*Iy[n]) / (G[n]*Asz[n]*Le[n]*Le[n]);
      } else  Ksy = Ksz = 0.0;

      /* x-axis trapezoidal loads (along the frame element length) */
      x1 =  W[lc][i][2]; x2 =  W[lc][i][3];
      w1 =  W[lc][i][4]; w2 =  W[lc][i][5];

      Nx1 = ( 3.0*(w1+w2)*Ln*(x2-x1) - (2.0*w2+w1)*x2*x2 + (w2-w1)*x2*x1 + (2.0*w1+w2)*x1*x1 ) / (6.0*Ln);
      Nx2 = ( -(2.0*w1+w2)*x1*x1 + (2.0*w2+w1)*x2*x2  - (w2-w1)*x1*x2 ) / ( 6.0*Ln );

      /* y-axis trapezoidal loads (across the frame element length) */
      x1 =  W[lc][i][6];  x2 = W[lc][i][7];
      w1 =  W[lc][i][8]; w2 =  W[lc][i][9];

      R1o = ( (2.0*w1+w2)*x1*x1 - (w1+2.0*w2)*x2*x2 +
	      3.0*(w1+w2)*Ln*(x2-x1) - (w1-w2)*x1*x2 ) / (6.0*Ln);
      R2o = ( (w1+2.0*w2)*x2*x2 + (w1-w2)*x1*x2 -
	      (2.0*w1+w2)*x1*x1 ) / (6.0*Ln);

      f01 = (  3.0*(w2+4.0*w1)*x1*x1*x1*x1 -  3.0*(w1+4.0*w2)*x2*x2*x2*x2
	       - 15.0*(w2+3.0*w1)*Ln*x1*x1*x1 + 15.0*(w1+3.0*w2)*Ln*x2*x2*x2
	       -  3.0*(w1-w2)*x1*x2*(x1*x1 + x2*x2)
	       + 20.0*(w2+2.0*w1)*Ln*Ln*x1*x1 - 20.0*(w1+2.0*w2)*Ln*Ln*x2*x2
	       + 15.0*(w1-w2)*Ln*x1*x2*(x1+x2)
	       -  3.0*(w1-w2)*x1*x1*x2*x2 - 20.0*(w1-w2)*Ln*Ln*x1*x2 ) / 360.0;

      f02 = (  3.0*(w2+4.0*w1)*x1*x1*x1*x1 - 3.0*(w1+4.0*w2)*x2*x2*x2*x2
	       -  3.0*(w1-w2)*x1*x2*(x1*x1+x2*x2)
	       - 10.0*(w2+2.0*w1)*Ln*Ln*x1*x1 + 10.0*(w1+2.0*w2)*Ln*Ln*x2*x2
	       -  3.0*(w1-w2)*x1*x1*x2*x2 + 10.0*(w1-w2)*Ln*Ln*x1*x2 ) / 360.0;

      Mz1 = -( 4.0*f01 + 2.0*f02 + Ksy*(f01 - f02) ) / ( Ln*Ln*(1.0+Ksy) );
      Mz2 = -( 2.0*f01 + 4.0*f02 - Ksy*(f01 - f02) ) / ( Ln*Ln*(1.0+Ksy) );

      Vy1 =  R1o + Mz1/Ln + Mz2/Ln;
      Vy2 =  R2o - Mz1/Ln - Mz2/Ln;

      /* z-axis trapezoidal loads (across the frame element length) */
      x1 =  W[lc][i][10]; x2 =  W[lc][i][11];
      w1 =  W[lc][i][12]; w2 =  W[lc][i][13];

      R1o = ( (2.0*w1+w2)*x1*x1 - (w1+2.0*w2)*x2*x2 +
	      3.0*(w1+w2)*Ln*(x2-x1) - (w1-w2)*x1*x2 ) / (6.0*Ln);
      R2o = ( (w1+2.0*w2)*x2*x2 + (w1-w2)*x1*x2 -
	      (2.0*w1+w2)*x1*x1 ) / (6.0*Ln);

      f01 = (  3.0*(w2+4.0*w1)*x1*x1*x1*x1 -  3.0*(w1+4.0*w2)*x2*x2*x2*x2
	       - 15.0*(w2+3.0*w1)*Ln*x1*x1*x1 + 15.0*(w1+3.0*w2)*Ln*x2*x2*x2
	       -  3.0*(w1-w2)*x1*x2*(x1*x1 + x2*x2)
	       + 20.0*(w2+2.0*w1)*Ln*Ln*x1*x1 - 20.0*(w1+2.0*w2)*Ln*Ln*x2*x2
	       + 15.0*(w1-w2)*Ln*x1*x2*(x1+x2)
	       -  3.0*(w1-w2)*x1*x1*x2*x2 - 20.0*(w1-w2)*Ln*Ln*x1*x2 ) / 360.0;

      f02 = (  3.0*(w2+4.0*w1)*x1*x1*x1*x1 - 3.0*(w1+4.0*w2)*x2*x2*x2*x2
	       -  3.0*(w1-w2)*x1*x2*(x1*x1+x2*x2)
	       - 10.0*(w2+2.0*w1)*Ln*Ln*x1*x1 + 10.0*(w1+2.0*w2)*Ln*Ln*x2*x2
	       -  3.0*(w1-w2)*x1*x1*x2*x2 + 10.0*(w1-w2)*Ln*Ln*x1*x2 ) / 360.0;

      My1 = ( 4.0*f01 + 2.0*f02 + Ksz*(f01 - f02) ) / ( Ln*Ln*(1.0+Ksz) );
      My2 = ( 2.0*f01 + 4.0*f02 - Ksz*(f01 - f02) ) / ( Ln*Ln*(1.0+Ksz) );

      Vz1 =  R1o - My1/Ln - My2/Ln;
      Vz2 =  R2o + My1/Ln + My2/Ln;

      n1 = J1[n]; n2 = J2[n];

      coord_trans ( xyz, Ln, n1, n2,
		    &t1, &t2, &t3, &t4, &t5, &t6, &t7, &t8, &t9, p[n] );


      /* {F} = [T]'{Q} */
      feF_mech[lc][n][1]  += ( Nx1*t1 + Vy1*t4 + Vz1*t7 );
      feF_mech[lc][n][2]  += ( Nx1*t2 + Vy1*t5 + Vz1*t8 );
      feF_mech[lc][n][3]  += ( Nx1*t3 + Vy1*t6 + Vz1*t9 );
      feF_mech[lc][n][4]  += ( Mx1*t1 + My1*t4 + Mz1*t7 );
      feF_mech[lc][n][5]  += ( Mx1*t2 + My1*t5 + Mz1*t8 );
      feF_mech[lc][n][6]  += ( Mx1*t3 + My1*t6 + Mz1*t9 );

      feF_mech[lc][n][7]  += ( Nx2*t1 + Vy2*t4 + Vz2*t7 );
      feF_mech[lc][n][8]  += ( Nx2*t2 + Vy2*t5 + Vz2*t8 );
      feF_mech[lc][n][9]  += ( Nx2*t3 + Vy2*t6 + Vz2*t9 );
      feF_mech[lc][n][10] += ( Mx2*t1 + My2*t4 + Mz2*t7 );
      feF_mech[lc][n][11] += ( Mx2*t2 + My2*t5 + Mz2*t8 );
      feF_mech[lc][n][12] += ( Mx2*t3 + My2*t6 + Mz2*t9 );

    }         /* end trapezoidally distributed loads */

    /* element point loads  */
    nP[lc] = eL.nP;
    if ( verbose ) {
      fprintf(stdout,"  number of concentrated frame element point loads ");
      dots(stdout,2); fprintf(stdout," nP = %3d\n", nP[lc]);
    }
    if ( nP[lc] < 0 || nP[lc] > 10*nE ) {
      fprintf(stderr,"  number of concentrated frame element point loads ");
      dots(stderr,3);
      fprintf(stderr," nP = %3d\n", nP[lc]);
      sprintf(errMsg,"\n  error: valid ranges for nP is 0 ... %d \n", 10*nE );
      errorMsg(errMsg);
      return 150;
    }
    for (i=1; i <= nP[lc]; i++) { /* ! local element coordinates ! */
      n = eL.EL[i-1];
      if ( n < 1 || n > nE ) {
	sprintf(errMsg,"\n   error in internal point loads: frame element number %d is out of range\n",n);
	errorMsg(errMsg);
	return 151;
      }
      P[lc][i][1] = (double) n;
      P[lc][i][2] = eL.Px[i-1];
      P[lc][i][3] = eL.Py[i-1];
      P[lc][i][4] = eL.Pz[i-1];
      P[lc][i][5] = eL.x[i-1];

      a = P[lc][i][5];    b = L[n] - a;

      if ( a < 0 || L[n] < a || b < 0 || L[n] < b ) {
	sprintf(errMsg,"\n  error in point load data: Point load coord. out of range\n   Frame element number: %d  L: %lf  load coord.: %lf\n",
                n, L[n], P[lc][i][5] );
	errorMsg(errMsg);
	return 152;
      }

      if ( shear ) {
	Ksy = (12.0*E[n]*Iz[n]) / (G[n]*Asy[n]*Le[n]*Le[n]);
	Ksz = (12.0*E[n]*Iy[n]) / (G[n]*Asz[n]*Le[n]*Le[n]);
      } else  Ksy = Ksz = 0.0;

      Ln = L[n];

      Nx1 = P[lc][i][2]*a/Ln;
      Nx2 = P[lc][i][2]*b/Ln;

      Vy1 = (1./(1.+Ksz))    * P[lc][i][3]*b*b*(3.*a + b) / ( Ln*Ln*Ln ) +
	(Ksz/(1.+Ksz)) * P[lc][i][3]*b/Ln;
      Vy2 = (1./(1.+Ksz))    * P[lc][i][3]*a*a*(3.*b + a) / ( Ln*Ln*Ln ) +
	(Ksz/(1.+Ksz)) * P[lc][i][3]*a/Ln;

      Vz1 = (1./(1.+Ksy))    * P[lc][i][4]*b*b*(3.*a + b) / ( Ln*Ln*Ln ) +
	(Ksy/(1.+Ksy)) * P[lc][i][4]*b/Ln;
      Vz2 = (1./(1.+Ksy))    * P[lc][i][4]*a*a*(3.*b + a) / ( Ln*Ln*Ln ) +
	(Ksy/(1.+Ksy)) * P[lc][i][4]*a/Ln;

      Mx1 = Mx2 = 0.0;

      My1 = -(1./(1.+Ksy))  * P[lc][i][4]*a*b*b / ( Ln*Ln ) -
	(Ksy/(1.+Ksy))* P[lc][i][4]*a*b   / (2.*Ln);
      My2 =  (1./(1.+Ksy))  * P[lc][i][4]*a*a*b / ( Ln*Ln ) +
	(Ksy/(1.+Ksy))* P[lc][i][4]*a*b   / (2.*Ln);

      Mz1 =  (1./(1.+Ksz))  * P[lc][i][3]*a*b*b / ( Ln*Ln ) +
	(Ksz/(1.+Ksz))* P[lc][i][3]*a*b   / (2.*Ln);
      Mz2 = -(1./(1.+Ksz))  * P[lc][i][3]*a*a*b / ( Ln*Ln ) -
	(Ksz/(1.+Ksz))* P[lc][i][3]*a*b   / (2.*Ln);

      n1 = J1[n]; n2 = J2[n];

      coord_trans ( xyz, Ln, n1, n2,
		    &t1, &t2, &t3, &t4, &t5, &t6, &t7, &t8, &t9, p[n] );

      /* {F} = [T]'{Q} */
      feF_mech[lc][n][1]  += ( Nx1*t1 + Vy1*t4 + Vz1*t7 );
      feF_mech[lc][n][2]  += ( Nx1*t2 + Vy1*t5 + Vz1*t8 );
      feF_mech[lc][n][3]  += ( Nx1*t3 + Vy1*t6 + Vz1*t9 );
      feF_mech[lc][n][4]  += ( Mx1*t1 + My1*t4 + Mz1*t7 );
      feF_mech[lc][n][5]  += ( Mx1*t2 + My1*t5 + Mz1*t8 );
      feF_mech[lc][n][6]  += ( Mx1*t3 + My1*t6 + Mz1*t9 );

      feF_mech[lc][n][7]  += ( Nx2*t1 + Vy2*t4 + Vz2*t7 );
      feF_mech[lc][n][8]  += ( Nx2*t2 + Vy2*t5 + Vz2*t8 );
      feF_mech[lc][n][9]  += ( Nx2*t3 + Vy2*t6 + Vz2*t9 );
      feF_mech[lc][n][10] += ( Mx2*t1 + My2*t4 + Mz2*t7 );
      feF_mech[lc][n][11] += ( Mx2*t2 + My2*t5 + Mz2*t8 );
      feF_mech[lc][n][12] += ( Mx2*t3 + My2*t6 + Mz2*t9 );
    }                 /* end element point loads */

    /* thermal loads    */
    nT[lc] = tempL.nT;
    if ( verbose ) {
      fprintf(stdout,"  number of temperature changes ");
      dots(stdout,21); fprintf(stdout," nT = %3d\n", nT[lc] );
    }
    if ( nT[lc] < 0 || nT[lc] > nE ) {
      fprintf(stderr,"  number of temperature changes ");
      dots(stderr,21);
      fprintf(stderr," nT = %3d\n", nT[lc] );
      sprintf(errMsg,"\n  error: valid ranges for nT is 0 ... %d \n", nE );
      errorMsg(errMsg);
      return 160;
    }
    for (i=1; i <= nT[lc]; i++) { /* ! local element coordinates ! */
      n = tempL.EL[i-1];
      if ( n < 1 || n > nE ) {
	sprintf(errMsg,"\n  error in temperature loads: frame element number %d is out of range\n",n);
	errorMsg(errMsg);
	return 161;
      }
      T[lc][i][1] = (double) n;
      T[lc][i][2] = tempL.a[i-1];
      T[lc][i][3] = tempL.hy[i-1];
      T[lc][i][4] = tempL.hz[i-1];
      T[lc][i][5] = tempL.Typ[i-1];
      T[lc][i][6] = tempL.Tym[i-1];
      T[lc][i][7] = tempL.Tzp[i-1];
      T[lc][i][8] = tempL.Tzm[i-1];

      a  = T[lc][i][2];
      hy = T[lc][i][3];
      hz = T[lc][i][4];

      if ( hy < 0 || hz < 0 ) {
	sprintf(errMsg,"\n  error in thermal load data: section dimension < 0\n   Frame element number: %d  hy: %f  hz: %f\n", n,hy,hz);
	errorMsg(errMsg);
	return 162;
      }

      Nx2 = (a/4.0)*( T[lc][i][5]+T[lc][i][6]+T[lc][i][7]+T[lc][i][8])*E[n]*Ax[n];
      Nx1 = -Nx2;
      Vy1 = Vy2 = Vz1 = Vz2 = 0.0;
      Mx1 = Mx2 = 0.0;
      My1 =  (a/hz)*(T[lc][i][8]-T[lc][i][7])*E[n]*Iy[n];
      My2 = -My1;
      Mz1 =  (a/hy)*(T[lc][i][5]-T[lc][i][6])*E[n]*Iz[n];
      Mz2 = -Mz1;

      n1 = J1[n]; n2 = J2[n];

      coord_trans ( xyz, L[n], n1, n2,
		    &t1, &t2, &t3, &t4, &t5, &t6, &t7, &t8, &t9, p[n] );

      /* {F} = [T]'{Q} */
      feF_temp[lc][n][1]  += ( Nx1*t1 + Vy1*t4 + Vz1*t7 );
      feF_temp[lc][n][2]  += ( Nx1*t2 + Vy1*t5 + Vz1*t8 );
      feF_temp[lc][n][3]  += ( Nx1*t3 + Vy1*t6 + Vz1*t9 );
      feF_temp[lc][n][4]  += ( Mx1*t1 + My1*t4 + Mz1*t7 );
      feF_temp[lc][n][5]  += ( Mx1*t2 + My1*t5 + Mz1*t8 );
      feF_temp[lc][n][6]  += ( Mx1*t3 + My1*t6 + Mz1*t9 );

      feF_temp[lc][n][7]  += ( Nx2*t1 + Vy2*t4 + Vz2*t7 );
      feF_temp[lc][n][8]  += ( Nx2*t2 + Vy2*t5 + Vz2*t8 );
      feF_temp[lc][n][9]  += ( Nx2*t3 + Vy2*t6 + Vz2*t9 );
      feF_temp[lc][n][10] += ( Mx2*t1 + My2*t4 + Mz2*t7 );
      feF_temp[lc][n][11] += ( Mx2*t2 + My2*t5 + Mz2*t8 );
      feF_temp[lc][n][12] += ( Mx2*t3 + My2*t6 + Mz2*t9 );
    }             /* end thermal loads    */


    for (n=1; n<=nE; n++) {
      n1 = J1[n];    n2 = J2[n];
      for (i=1; i<= 6; i++) F_mech[lc][6*n1- 6+i] += feF_mech[lc][n][i];
      for (i=7; i<=12; i++) F_mech[lc][6*n2-12+i] += feF_mech[lc][n][i];
      for (i=1; i<= 6; i++) F_temp[lc][6*n1- 6+i] += feF_temp[lc][n][i];
      for (i=7; i<=12; i++) F_temp[lc][6*n2-12+i] += feF_temp[lc][n][i];
    }

    nD[lc] = pD.nD;
    if ( verbose ) {
      fprintf(stdout,"  number of prescribed displacements ");
      dots(stdout,16);    fprintf(stdout," nD = %3d\n", nD[lc] );
    }
    for (i=1; i <= nD[lc]; i++) {
      j = pD.N[i-1];

      Dp[lc][6*j-5] = pD.Dx[i-1];
      Dp[lc][6*j-4] = pD.Dy[i-1];
      Dp[lc][6*j-3] = pD.Dz[i-1];
      Dp[lc][6*j-2] = pD.Dxx[i-1];
      Dp[lc][6*j-1] = pD.Dyy[i-1];
      Dp[lc][6*j] = pD.Dzz[i-1];

      for (l=5; l >=0; l--) {
	if ( r[6*j-l] == 0 && Dp[lc][6*j-l] != 0.0 ) {
	  sprintf(errMsg," Initial displacements can be prescribed only at restrained coordinates\n  node: %d  dof: %d  r: %d\n",
		  j, 6-l, r[6*j-l] );
	  errorMsg(errMsg);
	  return 171;
	}
      }
    }

  }                   /* end load-case loop */

  return 0;
}


/*------------------------------------------------------------------------------
  READ_MASS_DATA  -  read element densities and extra inertial mass data
  Oct 31, 2013
  ------------------------------------------------------------------------------*/
int read_mass_data (
		     DynamicData *dynamic, ExtraInertia *extraInertia, ExtraMass *extraMass,
		     int nN, int nE, int *nI, int *nX,
		     float *d, float *EMs,
		     float *NMs, float *NMx, float *NMy, float *NMz,
		     double *L, float *Ax,
		     double *total_mass, double *struct_mass,
		     int *nM, int *Mmethod,
		     int *lump,
		     double *tol, double *shift,
		     double *exagg_modal,
		     int anim[], float *pan,
		     int verbose, int debug){


  int j, jnt, m, b;
  // int i,j, jnt, m, b, nA;
  // int full_len=0, len=0;

  char    errMsg[MAXL];

  *total_mass = *struct_mass = 0.0;

  *nM = dynamic->nM;

  if ( verbose ) {
    fprintf(stdout," number of dynamic modes ");
    dots(stdout,28);    fprintf(stdout," nM = %3d\n", *nM);
  }

  if ( *nM < 1 ) {
    *nM = 0;

    /* calculate the total mass and the structural mass */
    for (b=1; b <= nE; b++) {
      *total_mass  += d[b]*Ax[b]*L[b];
      *struct_mass += d[b]*Ax[b]*L[b];
    }

    return 0;
  }

  *Mmethod = dynamic->Mmethod;

  if ( verbose ) {
    fprintf(stdout," modal analysis method ");
    dots(stdout,30);    fprintf(stdout," %3d ",*Mmethod);
    if ( *Mmethod == 1 ) fprintf(stdout," (Subspace-Jacobi)\n");
    if ( *Mmethod == 2 ) fprintf(stdout," (Stodola)\n");
  }

  *lump = dynamic->lump;
  *tol = dynamic->tol;
  *shift = dynamic->shift;
  *exagg_modal = dynamic->exagg_modal;

  /* number of nodes with extra inertias */
  *nI = extraInertia->nI;
  if ( verbose ) {
    fprintf(stdout," number of nodes with extra lumped inertia ");
    dots(stdout,10);    fprintf(stdout," nI = %3d\n",*nI);
  }
  for (j = 1; j <= nN; j++){
    NMs[j] = 0.0;
    NMx[j] = 0.0;
    NMy[j] = 0.0;
    NMz[j] = 0.0;
  }
  for (j=1; j <= *nI; j++) {
    jnt = extraInertia->N[j-1];
    if ( jnt < 1 || jnt > nN ) {
      sprintf(errMsg,"\n  error in node mass data: node number out of range    Node : %d  \n   Perhaps you did not specify %d extra masses \n   or perhaps the Input Data file is missing expected data.\n",
	      jnt, *nI );
      errorMsg(errMsg);
      return 86;
    }
    NMs[jnt] = extraInertia->EMs[j-1];
    NMx[jnt] = extraInertia->EMx[j-1];
    NMy[jnt] = extraInertia->EMy[j-1];
    NMz[jnt] = extraInertia->EMz[j-1];

    *total_mass += NMs[jnt];

    //if ( NMs[jnt]==0 && NMx[jnt]==0 && NMy[jnt]==0 && NMz[jnt]==0 )
    //  fprintf(stderr,"\n  Warning: All extra node inertia at node %d  are zero\n", jnt );
  }

  /* number of frame elements with extra beam mass */
  *nX = extraMass->nX;
  if ( verbose ) {
    fprintf(stdout," number of frame elements with extra mass ");
    dots(stdout,11);    fprintf(stdout," nX = %3d\n",*nX);
  }

  for (m=1;m<=nE;m++){
    EMs[m] = 0.0;
  }

  for (m=1; m <= *nX; m++) {
    b = extraMass->EL[m-1];
    if ( b < 1 || b > nE ) {
      sprintf(errMsg,"\n  error in element mass data: element number out of range   Element: %d  \n   Perhaps you did not specify %d extra masses \n   or perhaps the Input Data file is missing expected data.\n",
	      b, *nX );
      errorMsg(errMsg);
      return 87;
    }
    EMs[b] = extraMass->EMs[m-1];
  }


  /* calculate the total mass and the structural mass */
  for (b=1; b <= nE; b++) {
    *total_mass  += d[b]*Ax[b]*L[b] + EMs[b];
    *struct_mass += d[b]*Ax[b]*L[b];
  }


  for (m=1;m<=nE;m++) {           /* check inertia data   */
    if ( d[m] < 0.0 || EMs[m] < 0.0 || d[m]+EMs[m] <= 0.0 ) {
      sprintf(errMsg,"\n  error: Non-positive mass or density\n  d[%d]= %f  EMs[%d]= %f\n",m,d[m],m,EMs[m]);
      errorMsg(errMsg);
      return 88;
    }
  }

  if ( verbose ) {
    fprintf(stdout," structural mass ");
    dots(stdout,36);    fprintf(stdout,"  %12.4e\n",*struct_mass);
    fprintf(stdout," total mass ");
    dots(stdout,41);    fprintf(stdout,"  %12.4e\n",*total_mass);
  }

  // left out animation

  // sfrv=fscanf ( fp, "%d", &nA );
  // if (sfrv != 1) sferr("nA value in mode animation data");
  // if ( verbose ) {
  //     fprintf(stdout," number of modes to be animated ");
  //     dots(stdout,21);    fprintf(stdout," nA = %3d\n",nA);
  // }
  // if (nA > 20)
  //   fprintf(stderr," nA = %d, only 20 or fewer modes may be animated\n", nA );
  // for ( m = 0; m < 20; m++ )  anim[m] = 0;
  // for ( m = 0; m < nA; m++ ) {
  //     sfrv=fscanf ( fp, "%d", &anim[m] );
  //     if (sfrv != 1) sferr("mode number in mode animation data");
  // }

  // sfrv=fscanf ( fp, "%f", pan );
  // if (sfrv != 1) sferr("pan value in mode animation data");
  // if ( pan_flag != -0.0 ) *pan = pan_flag;


  // strcpy(base_file,OUT_file);
  // while ( base_file[len++] != '\0' )
  //     /* the length of the base_file */ ;
  // full_len = len;

  // while ( base_file[len--] != '.' && len > 0 )
  //     /* find the last '.' in base_file */ ;
  // if ( len == 0 ) len = full_len;
  // base_file[++len] = '\0';    /* end base_file at the last '.' */

  // while ( base_file[len] != '/' && base_file[len] != '\\' && len > 0 )
  //     len--;  /* find the last '/' or '\' in base_file */
  // i = 0;
  // while ( base_file[len] != '\0' )
  //     mode_file[i++] = base_file[len++];
  // mode_file[i] = '\0';
  // strcat(mode_file,"-m");
  // output_path(mode_file,modepath,FRAME3DD_PATHMAX,NULL);

  return 0;
}


/*------------------------------------------------------------------------------
  READ_CONDENSE   -  read matrix condensation information
  Oct 31, 2013
  ------------------------------------------------------------------------------*/
int read_condensation_data (
			     Condensation *condensation,
			     int nN, int nM,
			     int *nC, int *Cdof,
			     int *Cmethod, int *c, int *m, int verbose){

  int i,j,k,  **cm;
  char errMsg[MAXL];

  *Cmethod = *nC = *Cdof = 0;

  *Cmethod = condensation->Cmethod;

  if ( *Cmethod <= 0 )  {
    *Cmethod = *nC = *Cdof = 0;
    return 0;
  }

  if ( *Cmethod > 3 ) *Cmethod = 1;   /* default */
  if ( verbose ) {
    fprintf(stdout," condensation method ");
    dots(stdout,32);    fprintf(stdout," %d ", *Cmethod );
    if ( *Cmethod == 1 )    fprintf(stdout," (static only) \n");
    if ( *Cmethod == 2 )    fprintf(stdout," (Guyan) \n");
    if ( *Cmethod == 3 )    fprintf(stdout," (dynamic) \n");
  }

  *nC = condensation->nC;

  if ( verbose ) {
    fprintf(stdout," number of nodes with condensed DoF's ");
    dots(stdout,15);    fprintf(stdout," nC = %3d\n", *nC );
  }

  if ( (*nC) > nN ) {
    sprintf(errMsg,"\n  error in matrix condensation data: \n error: nC > nN ... nC=%d; nN=%d;\n The number of nodes with condensed DoF's may not exceed the total number of nodes.\n",
	    *nC, nN );
    errorMsg(errMsg);
    return 90;
  }

  cm = imatrix( 1, *nC, 1,7 );

  for ( i=1; i <= *nC; i++) {
    cm[i][1] = condensation->N[i-1];
    cm[i][2] = condensation->cx[i-1];
    cm[i][3] = condensation->cy[i-1];
    cm[i][4] = condensation->cz[i-1];
    cm[i][5] = condensation->cxx[i-1];
    cm[i][6] = condensation->cyy[i-1];
    cm[i][7] = condensation->czz[i-1];
    if ( cm[i][1] < 1 || cm[i][1] > nN ) {     /* error check */
      sprintf(errMsg,"\n  error in matrix condensation data: \n  condensed node number out of range\n  cj[%d] = %d  ... nN = %d  \n", i, cm[i][1], nN );
      errorMsg(errMsg);
      return 91;
    }
  }

  for (i=1; i <= *nC; i++)  for (j=2; j<=7; j++)  if (cm[i][j]) (*Cdof)++;

  k=1;
  for (i=1; i <= *nC; i++) {
    for (j=2; j<=7; j++) {
      if (cm[i][j]) {
	c[k] = 6*(cm[i][1]-1) + j-1;
	++k;
      }
    }
  }

  for (i=1; i<= *Cdof; i++) {
    m[i] = condensation->m[i-1];
    if ( (m[i] < 0 || m[i] > nM) && *Cmethod == 3 ) {
      sprintf(errMsg,"\n  error in matrix condensation data: \n  m[%d] = %d \n The condensed mode number must be between   1 and %d (modes).\n",
	      i, m[i], nM );
      errorMsg(errMsg);
      return 92;
    }
  }

  free_imatrix(cm,1, *nC, 1,7);
  return 0;
}


/*------------------------------------------------------------------------------
  WRITE_STATIC_RESULTS -  save node displacements and frame element end forces
  Oct 31, 2013
  ------------------------------------------------------------------------------*/
void write_static_results (
			   Displacements* displacements, Forces* forces, ReactionForces* reactionForces,
			   Reactions* reactions, int nR,
			   int nN, int nE, int nL, int lc, int DoF,
			   int *J1, int *J2,
			   double *R, double *D, int *r, double **Q,
			   double err, int ok){

  // double  disp;
  int i,j,n,k;
  char errMsg[MAXL];
  double vals[6];

  if ( ok < 0 ) {
    sprintf(errMsg,"  * The Stiffness Matrix is not positive-definite *\n");
    //errorMsg(errMsg);
    sprintf(errMsg,"    Check that all six rigid-body translations are restrained\n");
    //errorMsg(errMsg);
    sprintf(errMsg,"    If geometric stiffness is included, reduce the loads.\n");
    //errorMsg(errMsg);
  }

  for (j=1; j<= nN; j++) {
    displacements[lc-1].node[j-1] = j;
    displacements[lc-1].x[j-1] = D[6*j-5];
    displacements[lc-1].y[j-1] = D[6*j-4];
    displacements[lc-1].z[j-1] = D[6*j-3];
    displacements[lc-1].xrot[j-1] = D[6*j-2];
    displacements[lc-1].yrot[j-1] = D[6*j-1];
    displacements[lc-1].zrot[j-1] = D[6*j];
  }


  for (n=1; n<= nE; n++) {
    forces[lc-1].element[2*n-2] = n;
    forces[lc-1].node[2*n-2] = J1[n];
    forces[lc-1].Nx[2*n-2] = Q[n][1];
    forces[lc-1].Vy[2*n-2] = Q[n][2];
    forces[lc-1].Vz[2*n-2] = Q[n][3];
    forces[lc-1].Txx[2*n-2] = Q[n][4];
    forces[lc-1].Myy[2*n-2] = Q[n][5];
    forces[lc-1].Mzz[2*n-2] = Q[n][6];

    forces[lc-1].element[2*n-1] = n;
    forces[lc-1].node[2*n-1] = J2[n];
    forces[lc-1].Nx[2*n-1] = Q[n][7];
    forces[lc-1].Vy[2*n-1] = Q[n][8];
    forces[lc-1].Vz[2*n-1] = Q[n][9];
    forces[lc-1].Txx[2*n-1] = Q[n][10];
    forces[lc-1].Myy[2*n-1] = Q[n][11];
    forces[lc-1].Mzz[2*n-1] = Q[n][12];

  }



  for (k=1; k<=nR; k++) {  // iteration through reactions
    j = reactions->N[k-1];  // node number
    reactionForces[lc-1].node[k-1] = j;

    for (i=5; i>=0; i--) {
      if (r[6*j-i]){
	vals[i] = R[6*j-i];
      } else{
	vals[i] = 0.0;
      }
    }

    reactionForces[lc-1].Fx[k-1] = vals[5];
    reactionForces[lc-1].Fy[k-1] = vals[4];
    reactionForces[lc-1].Fz[k-1] = vals[3];
    reactionForces[lc-1].Mxx[k-1] = vals[2];
    reactionForces[lc-1].Myy[k-1] = vals[1];
    reactionForces[lc-1].Mzz[k-1] = vals[0];
  }


  // for (j=1; j<=nN; j++) {

  //     reactionForces[lc-1].node[j-1] = j;

  //     for (i=5; i>=0; i--) {
  //         if (r[6*j-i]){
  //             vals[i] = F[6*j-i];
  //         } else{
  //             vals[i] = 0.0;
  //         }
  //     }

  //     reactionForces[lc-1].Fx[j-1] = vals[5];
  //     reactionForces[lc-1].Fy[j-1] = vals[4];
  //     reactionForces[lc-1].Fz[j-1] = vals[3];
  //     reactionForces[lc-1].Mxx[j-1] = vals[2];
  //     reactionForces[lc-1].Myy[j-1] = vals[1];
  //     reactionForces[lc-1].Mzz[j-1] = vals[0];
  // }

  return;
}



/*------------------------------------------------------------------------------
  WRITE_INTERNAL_FORCES -
  calculate frame element internal forces, Nx, Vy, Vz, Tx, My, Mz
  calculate frame element local displacements, Rx, Dx, Dy, Dz
  write internal forces and local displacements to an output data file
  Oct 31, 2013
  ------------------------------------------------------------------------------*/
void write_internal_forces (
			    InternalForces **internalForces,
			    int lc, int nL, float dx,
			    vec3 *xyz,
			    double **Q, int nN, int nE, double *L, int *J1, int *J2,
			    float *Ax,float *Asy,float *Asz,float *Jx,float *Iy,float *Iz,
			    float *E, float *G, float *p,
			    float *d, float gX, float gY, float gZ,
			    int nU, float **U, int nW, float **W, int nP, float **P,
			    double *D, int shear, double error
			    ){
  double  t1, t2, t3, t4, t5, t6, t7, t8, t9, /* coord transformation */
    u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12; /* displ. */

  double  xx1,xx2, wx1,wx2,   /* trapz load data, local x dir */
    xy1,xy2, wy1,wy2,   /* trapz load data, local y dir */
    xz1,xz2, wz1,wz2;   /* trapz load data, local z dir */

  double  wx=0, wy=0, wz=0, // distributed loads in local coords at x[i]
    wx_=0,wy_=0,wz_=0,// distributed loads in local coords at x[i-1]
    wxg=0,wyg=0,wzg=0,// gravity loads in local x, y, z coord's
    tx=0.0, tx_=0.0;  // distributed torque about local x coord

  double  xp;     /* location of internal point loads */

  double  *x, dx_, dxnx,  /* distance along frame element     */
    *Nx,        /* axial force within frame el.     */
    *Vy, *Vz,   /* shear forces within frame el.    */
    *Tx,        /* torsional moment within frame el.    */
    *My, *Mz,   /* bending moments within frame el. */
    *Sy, *Sz,   /* transverse slopes of frame el.   */
    *Dx, *Dy, *Dz,  /* frame el. displ. in local x,y,z, dir's */
    *Rx;        /* twist rotation about the local x-axis */

  int n, m,       /* frame element number         */
    cU=0, cW=0, cP=0, /* counters for U, W, and P loads */
    i, nx,      /* number of sections alont x axis  */
    n1,n2,i1,i2;    /* starting and stopping node no's  */

  // char    errMsg[MAXL];
  time_t  now;        /* modern time variable type        */

  if (dx == -1.0) return; // skip calculation of internal forces and displ

  (void) time(&now);


  for ( m=1; m <= nE; m++ ) { // loop over all frame elements

    n1 = J1[m]; n2 = J2[m]; // node 1 and node 2 of elmnt m

    nx = floor(L[m]/dx);    // number of x-axis increments
    if (nx < 1) nx = 1; // at least one x-axis increment

    // allocate memory for interior force data for frame element "m"
    x  = dvector(0,nx);
    Nx = dvector(0,nx);
    Vy = dvector(0,nx);
    Vz = dvector(0,nx);
    Tx = dvector(0,nx);
    My = dvector(0,nx);
    Mz = dvector(0,nx);
    Sy = dvector(0,nx);
    Sz = dvector(0,nx);
    Rx = dvector(0,nx);
    Dx = dvector(0,nx);
    Dy = dvector(0,nx);
    Dz = dvector(0,nx);

    // the local x-axis for frame element "m" starts at 0 and ends at L[m]
    for (i=0; i<nx; i++)    x[i] = i*dx;
    x[nx] = L[m];
    dxnx = x[nx]-x[nx-1];   // length of the last x-axis increment

    // find interior axial force, shear forces, torsion and bending moments

    coord_trans ( xyz, L[m], n1, n2,
		  &t1, &t2, &t3, &t4, &t5, &t6, &t7, &t8, &t9, p[m] );

    // distributed gravity load in local x, y, z coordinates
    wxg = d[m]*Ax[m]*(t1*gX + t2*gY + t3*gZ);
    wyg = d[m]*Ax[m]*(t4*gX + t5*gY + t6*gZ);
    wzg = d[m]*Ax[m]*(t7*gX + t8*gY + t9*gZ);

    // add uniformly-distributed loads to gravity load
    for (n=1; n<=nE && cU<nU; n++) {
      if ( (int) U[n][1] == m ) { // load n on element m
	wxg += U[n][2];
	wyg += U[n][3];
	wzg += U[n][4];
	++cU;
      }
    }

    // interior forces for frame element "m" at (x=0)
    Nx[0] = -Q[m][1];   // positive Nx is tensile
    Vy[0] = -Q[m][2];   // positive Vy in local y direction
    Vz[0] = -Q[m][3];   // positive Vz in local z direction
    Tx[0] = -Q[m][4];   // positive Tx r.h.r. about local x axis
    My[0] =  Q[m][5];   // positive My -> positive x-z curvature
    Mz[0] = -Q[m][6];   // positive Mz -> positive x-y curvature

    dx_ = dx;
    for (i=1; i<=nx; i++) { /*  accumulate interior span loads */

      // start with gravitational plus uniform loads
      wx = wxg;
      wy = wyg;
      wz = wzg;

      if (i==1) {
	wx_ = wxg;
	wy_ = wyg;
	wz_ = wzg;
	tx_ = tx;
      }

      // add trapezoidally-distributed loads
      for (n=1; n<=10*nE && cW<nW; n++) {
	if ( (int) W[n][1] == m ) { // load n on element m
	  if (i==nx) ++cW;
	  xx1 = W[n][2];  xx2 = W[n][3];
	  wx1 = W[n][4];  wx2 = W[n][5];
	  xy1 = W[n][6];  xy2 = W[n][7];
	  wy1 = W[n][8];  wy2 = W[n][9];
	  xz1 = W[n][10]; xz2 = W[n][11];
	  wz1 = W[n][12]; wz2 = W[n][13];

	  if ( x[i]>xx1 && x[i]<=xx2 )
	    wx += wx1+(wx2-wx1)*(x[i]-xx1)/(xx2-xx1);
	  if ( x[i]>xy1 && x[i]<=xy2 )
	    wy += wy1+(wy2-wy1)*(x[i]-xy1)/(xy2-xy1);
	  if ( x[i]>xz1 && x[i]<=xz2 )
	    wz += wz1+(wz2-wz1)*(x[i]-xz1)/(xz2-xz1);
	}
      }

      // trapezoidal integration of distributed loads
      // for axial forces, shear forces and torques
      if (i==nx)  dx_ = dxnx;
      Nx[i] = Nx[i-1] - 0.5*(wx+wx_)*dx_;
      Vy[i] = Vy[i-1] - 0.5*(wy+wy_)*dx_;
      Vz[i] = Vz[i-1] - 0.5*(wz+wz_)*dx_;
      Tx[i] = Tx[i-1] - 0.5*(tx+tx_)*dx_;

      // update distributed loads at x[i-1]
      wx_ = wx;
      wy_ = wy;
      wz_ = wz;
      tx_ = tx;

      // add interior point loads
      for (n=1; n<=10*nE && cP<nP; n++) {
	if ( (int) P[n][1] == m ) { // load n on element m
	  if (i==nx) ++cP;
	  xp = P[n][5];
	  if ( x[i] <= xp && xp < x[i]+dx ) {
	    Nx[i] -= P[n][2] * 0.5 * (1.0 - (xp-x[i])/dx);
	    Vy[i] -= P[n][3] * 0.5 * (1.0 - (xp-x[i])/dx);
	    Vz[i] -= P[n][4] * 0.5 * (1.0 - (xp-x[i])/dx);

	  }
	  if ( x[i]-dx <= xp && xp < x[i] ) {
	    Nx[i] -= P[n][2] * 0.5 * (1.0 - (x[i]-dx-xp)/dx);
	    Vy[i] -= P[n][3] * 0.5 * (1.0 - (x[i]-dx-xp)/dx);
	    Vz[i] -= P[n][4] * 0.5 * (1.0 - (x[i]-dx-xp)/dx);
	  }
	}
      }

    }
    // linear correction for bias in trapezoidal integration
    for (i=1; i<=nx; i++) {
      Nx[i] -= (Nx[nx]-Q[m][7])  * i/nx;
      Vy[i] -= (Vy[nx]-Q[m][8])  * i/nx;
      Vz[i] -= (Vz[nx]-Q[m][9])  * i/nx;
      Tx[i] -= (Tx[nx]-Q[m][10]) * i/nx;
    }
    // trapezoidal integration of shear force for bending momemnt
    dx_ = dx;
    for (i=1; i<=nx; i++) {
      if (i==nx)  dx_ = dxnx;
      My[i] = My[i-1] - 0.5*(Vz[i]+Vz[i-1])*dx_;
      Mz[i] = Mz[i-1] - 0.5*(Vy[i]+Vy[i-1])*dx_;

    }
    // linear correction for bias in trapezoidal integration
    for (i=1; i<=nx; i++) {
      My[i] -= (My[nx]+Q[m][11]) * i/nx;
      Mz[i] -= (Mz[nx]-Q[m][12]) * i/nx;
    }

    // find interior transverse displacements

    i1 = 6*(n1-1);  i2 = 6*(n2-1);

    /* compute end deflections in local coordinates */

    u1  = t1*D[i1+1] + t2*D[i1+2] + t3*D[i1+3];
    u2  = t4*D[i1+1] + t5*D[i1+2] + t6*D[i1+3];
    u3  = t7*D[i1+1] + t8*D[i1+2] + t9*D[i1+3];

    u4  = t1*D[i1+4] + t2*D[i1+5] + t3*D[i1+6];
    u5  = t4*D[i1+4] + t5*D[i1+5] + t6*D[i1+6];
    u6  = t7*D[i1+4] + t8*D[i1+5] + t9*D[i1+6];

    u7  = t1*D[i2+1] + t2*D[i2+2] + t3*D[i2+3];
    u8  = t4*D[i2+1] + t5*D[i2+2] + t6*D[i2+3];
    u9  = t7*D[i2+1] + t8*D[i2+2] + t9*D[i2+3];

    u10 = t1*D[i2+4] + t2*D[i2+5] + t3*D[i2+6];
    u11 = t4*D[i2+4] + t5*D[i2+5] + t6*D[i2+6];
    u12 = t7*D[i2+4] + t8*D[i2+5] + t9*D[i2+6];


    // rotations and displacements for frame element "m" at (x=0)
    Dx[0] =  u1;    // displacement in  local x dir  at node J1
    Dy[0] =  u2;    // displacement in  local y dir  at node J1
    Dz[0] =  u3;    // displacement in  local z dir  at node J1
    Rx[0] =  u4;    // rotationin about local x axis at node J1
    Sy[0] =  u6;    // slope in  local y  direction  at node J1
    Sz[0] = -u5;    // slope in  local z  direction  at node J1

    // axial displacement along frame element "m"
    dx_ = dx;
    for (i=1; i<=nx; i++) {
      if (i==nx)  dx_ = dxnx;
      Dx[i] = Dx[i-1] + 0.5*(Nx[i-1]+Nx[i])/(E[m]*Ax[m])*dx_;
    }
    // linear correction for bias in trapezoidal integration
    for (i=1; i<=nx; i++) {
      Dx[i] -= (Dx[nx]-u7) * i/nx;
    }

    // torsional rotation along frame element "m"
    dx_ = dx;
    for (i=1; i<=nx; i++) {
      if (i==nx)  dx_ = dxnx;
      Rx[i] = Rx[i-1] + 0.5*(Tx[i-1]+Tx[i])/(G[m]*Jx[m])*dx_;
    }
    // linear correction for bias in trapezoidal integration
    for (i=1; i<=nx; i++) {
      Rx[i] -= (Rx[nx]-u10) * i/nx;
    }

    // transverse slope along frame element "m"
    dx_ = dx;
    for (i=1; i<=nx; i++) {
      if (i==nx)  dx_ = dxnx;
      Sy[i] = Sy[i-1] + 0.5*(Mz[i-1]+Mz[i])/(E[m]*Iz[m])*dx_;
      Sz[i] = Sz[i-1] + 0.5*(My[i-1]+My[i])/(E[m]*Iy[m])*dx_;
    }
    if ( shear ) {
      for (i=1; i<=nx; i++) {
	Sy[i] += Vy[i]/(G[m]*Asy[m]);
	Sz[i] += Vz[i]/(G[m]*Asz[m]);
      }
    }
    // linear correction for bias in trapezoidal integration
    for (i=1; i<=nx; i++) {
      Sy[i] -= (Sy[nx]-u12) * i/nx;
      Sz[i] -= (Sz[nx]+u11) * i/nx;
    }
    // displacement along frame element "m"
    dx_ = dx;
    for (i=1; i<=nx; i++) {
      if (i==nx)  dx_ = dxnx;
      Dy[i] = Dy[i-1] + 0.5*(Sy[i-1]+Sy[i])*dx_;
      Dz[i] = Dz[i-1] + 0.5*(Sz[i-1]+Sz[i])*dx_;
    }
    // linear correction for bias in trapezoidal integration
    for (i=1; i<=nx; i++) {
      Dy[i] -= (Dy[nx]-u8) * i/nx;
      Dz[i] -= (Dz[nx]-u9) * i/nx;
    }


    // write results to the internal frame element force output data file
    for (i=0; i<=nx; i++) {
      internalForces[lc-1][m-1].x[i] = x[i];
      internalForces[lc-1][m-1].Nx[i] = Nx[i];
      internalForces[lc-1][m-1].Vy[i] = Vy[i];
      internalForces[lc-1][m-1].Vz[i] = Vz[i];
      internalForces[lc-1][m-1].Tx[i] = Tx[i];
      internalForces[lc-1][m-1].My[i] = My[i];
      internalForces[lc-1][m-1].Mz[i] = Mz[i];
      internalForces[lc-1][m-1].Dx[i] = Dx[i];
      internalForces[lc-1][m-1].Dy[i] = Dy[i];
      internalForces[lc-1][m-1].Dz[i] = Dz[i];
      internalForces[lc-1][m-1].Rx[i] = Rx[i];
    }

    // free memory
    free_dvector(x,0,nx);
    free_dvector(Nx,0,nx);
    free_dvector(Vy,0,nx);
    free_dvector(Vz,0,nx);
    free_dvector(Tx,0,nx);
    free_dvector(My,0,nx);
    free_dvector(Mz,0,nx);
    free_dvector(Rx,0,nx);
    free_dvector(Sy,0,nx);
    free_dvector(Sz,0,nx);
    free_dvector(Dx,0,nx);
    free_dvector(Dy,0,nx);
    free_dvector(Dz,0,nx);

  }               // end of loop over all frame elements

}


/*------------------------------------------------------------------------------
  WRITE_MODAL_RESULTS -  save modal frequencies and mode shapes
  Oct 31, 2013
  ------------------------------------------------------------------------------*/
void write_modal_results(
			 MassResults* massR, ModalResults* modalR,
			 int nN, int nE, int nI, int DoF,
			 double **M, double *f, double **V,
			 double total_mass, double struct_mass,
			 int iter, int sumR, int nM,
			 double shift, int lump, double tol, int ok
			 ){

  int i, j, k, m, num_modes;
  double  mpfX, mpfY, mpfZ,   /* mode participation factors   */
    *msX, *msY, *msZ;
  // double  fs;

  msX = dvector(1,DoF);
  msY = dvector(1,DoF);
  msZ = dvector(1,DoF);

  for (i=1; i<=DoF; i++) {
    msX[i] = msY[i] = msZ[i] = 0.0;
    for (j=1; j<=DoF; j+=6) msX[i] += M[i][j];
    for (j=2; j<=DoF; j+=6) msY[i] += M[i][j];
    for (j=3; j<=DoF; j+=6) msZ[i] += M[i][j];
  }

  if ( (DoF - sumR) > nM )    num_modes = nM;
  else    num_modes = DoF - sumR;

  *(massR->total_mass) = total_mass;
  *(massR->struct_mass) = struct_mass;

  for (j=1; j <= nN; j++) {
    k = 6*(j-1);
    massR->N[j-1] = j;
    massR->xmass[j-1] = M[k+1][k+1];
    massR->ymass[j-1] = M[k+2][k+2];
    massR->zmass[j-1] = M[k+3][k+3];
    massR->xinrta[j-1] = M[k+4][k+4];
    massR->yinrta[j-1] = M[k+5][k+5];
    massR->zinrta[j-1] = M[k+6][k+6];
  }

  //TODO use tol
  for (m=1; m<=num_modes; m++) {
    mpfX = 0.0; for (i=1; i<=DoF; i++)    mpfX += V[i][m]*msX[i];
    mpfY = 0.0; for (i=1; i<=DoF; i++)    mpfY += V[i][m]*msY[i];
    mpfZ = 0.0; for (i=1; i<=DoF; i++)    mpfZ += V[i][m]*msZ[i];
    *(modalR[m-1].freq) = f[m];
    *(modalR[m-1].xmpf) = mpfX;
    *(modalR[m-1].ympf) = mpfY;
    *(modalR[m-1].zmpf) = mpfZ;

    for (j=1; j<= nN; j++) {
      modalR[m-1].N[j-1] = j;
      modalR[m-1].xdsp[j-1] = V[6*j-5][m];
      modalR[m-1].ydsp[j-1] = V[6*j-4][m];
      modalR[m-1].zdsp[j-1] = V[6*j-3][m];
      modalR[m-1].xrot[j-1] = V[6*j-2][m];
      modalR[m-1].yrot[j-1] = V[6*j-1][m];
      modalR[m-1].zrot[j-1] = V[6*j][m];
    }
  }

  // fprintf(fp,"M A T R I X    I T E R A T I O N S: %d\n", iter );

  // fs = sqrt(4.0*PI*PI*f[nM]*f[nM] + tol) / (2.0*PI);

  // fprintf(fp,"There are %d modes below %f Hz.", -ok, fs );
  // if ( -ok > nM ) {
  //     fprintf(fp," ... %d modes were not found.\n", -ok-nM );
  //     fprintf(fp," Try increasing the number of modes in \n");
  //     fprintf(fp," order to get the missing modes below %f Hz.\n",fs);
  // } else  fprintf(fp," ... All %d modes were found.\n", nM );


  free_dvector(msX,1,DoF);
  free_dvector(msY,1,DoF);
  free_dvector(msZ,1,DoF);
  return;
}




/*------------------------------------------------------------------------------
  DOTS  -  print a set of dots (periods)
  ------------------------------------------------------------------------------*/
void dots ( FILE *fp, int n ) {
  int i;
  for (i=1; i<=n; i++)    fprintf(fp,".");
}
