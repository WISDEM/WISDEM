/******************************************************************************
 preframe.c  -  interactive data input for frame analysis program 
	to compile:	gcc -O -o preframe preframe.c
	to run:		preframe  output_file
 David Hoang, Duke University, April, 1997
******************************************************************************/

#include <stdio.h>
#include <ctype.h>
#define MAXLINE 80

void file_names(void);
void joint_loads(void);
void distrb_loads(void);
void concen_loads(void);
void temperature(void);
void reactions(void);
void displacements(void);
void modal_files(void);
void inertia(void);

FILE *fpout;

int	nJ=0,	/* number of joints				*/
	nM=0,	/* number of members				*/
	nF=0,	/* number of joint loads			*/
	nW=0,	/* number of distributed member loads		*/
	nP=0,	/* number of concentrated member loads		*/ 
	nT=0,	/* number of thermal loads			*/
	nR=0,	/* number of supported joints (reactions)	*/
	nD=0;	/* number of prescribed displacements		*/

void main(argc, argv)
int	 argc;
char    *argv[];
{ 
 char	ans;
 int	j, m;		/* a joint number, a member number	*/
 float	x, y, z, r;	/* joint coordinates 			*/  

 int    J1, J2;		/* member location joints		*/ 
 float  Ax, Ay, Az;	/* member cross section area properties	*/
 float  Jp, Iy, Iz;	/* member cross section inertia prop's	*/
 float	E,G;		/* member material property constants	*/


 if (argc <= 2) {
    if ((fpout = fopen (argv[1], "w")) == 0) {
	  fprintf (stderr," error: cannot open file '%s'\n", argv[2]);
	  fprintf (stderr," usage: preframe output_file\n");
	  exit(0);
     }
 } else {
	  fprintf (stderr," usage: preframe output_file\n"); 
	  exit(0);
 }

/********* Start Execution  **********/

 do {	 
	printf("How many  joints do you have?  ");	scanf ("%d",&nJ);
	printf("How many members do you have?  ");	scanf ("%d",&nM);
	printf("\n\t%5d joints   %5d members ", nJ, nM );
	printf("\t\t     ... Is this okay? (y/n) "); scanf ("%s", &ans);
 } while ( ans !='y' );

 fprintf(fpout,"%d   %d \n\n",nJ, nM);

/***** Joint Data  *****/

 printf("\nFor each joint, enter its x,y,z coordinates and its radius, r .\n");
 for (j=1; j<=nJ ; j++) {
    do { 
	printf("\n For joint %i, enter coordinate values ... \n", j );
	printf("   input x[%i] : ",j);	scanf ("%f",&x);
	printf("   input y[%i] : ",j);	scanf ("%f",&y);
	printf("   input z[%i] : ",j);	scanf ("%f",&z);
	printf("   input r[%i] : ",j);	scanf ("%f",&r);
	printf("joint      x	        y            z	         r\n",j,j,j,j);
	printf("----- ------------ ------------ ------------ ----------\n");
	printf("%5d %12.4f %12.4f %12.4f %10.4f \n",j, x, y, z, r);
	printf("\t\t\t\t\t\t     ... Is this okay? (y/n) "); scanf ("%s", &ans);
    } while ( ans !='y');
    fprintf(fpout,"%4d %12.4f %12.4f %12.4f %12.4f\n", j, x, y, z, r);
 }

/***** Member Data *****/

 printf("\nFor each member, enter its geometric and material properties.\n");
 printf("Members connect joints 'J1' to 'J2' \n");
 fprintf(fpout,"\n");
 for (m=1; m<=nM ; m++) {
   do {
    printf("\n For member %i, enter values for the ... \n", m);
    printf("   joint number               J1[%i] : ",m); scanf ("%d", &J1);
    printf("   joint number               J2[%i] : ",m); scanf ("%d", &J2);
    printf("   cross section area,        Ax[%i] : ",m); scanf ("%f", &Ax);
    printf("   shear section area,        Ay[%i] : ",m); scanf ("%f", &Ay);
    printf("   shear section area,        Az[%i] : ",m); scanf ("%f", &Az);
    printf("   torsion moment of inertia, Jp[%i] : ",m); scanf("%f", &Jp);
    printf("   bending moment of inertia, Iy[%i] : ",m); scanf("%f", &Iy);
    printf("   bending moment of inertia, Iz[%i] : ",m); scanf("%f", &Iz);
    printf("   Young's elastic modulus,    E[%i] : ",m); scanf ("%f", &E);
    printf("    shear  elastic modulus,    G[%i] : ",m); scanf ("%f", &G);
    printf(" J1   J2    Ax    Ay    Az     Jp     Iy     Iz     E       G \n");
    printf("---- ---- ----- ----- ----- ------ ------ ------ ------- -------");
    printf("\n%4d %4d %5.1f %5.1f %5.1f %6.1f %6.1f %6.1f %7.1f %7.1f \n", 
					J1, J2, Ax, Ay, Az, Jp, Iy, Iz, E, G );
    printf("\t\t\t\t\t\t     ... Is this okay? (y/n) "); scanf ("%s", &ans);
   } while( ans != 'y' || J1 > nJ || J2 > nJ || E == 0 ); 
 
   fprintf(fpout,"%4d %4d %4d %7.1f %7.1f %7.1f", m, J1, J2, Ax, Ay, Az );
   fprintf(fpout,"%7.1f %7.1f %7.1f %7.1f %7.1f\n", Jp, Iy, Iz, E, G );
 } 

 file_names();
 joint_loads();
 distrb_loads();
 concen_loads();
 temperature();
 reactions();
 displacements();
 modal_files();
 inertia();

 fclose(fpout);

} /*end main */


/***** Mesh, shear, analysis, and annotation file name *****/
void file_names(void)
{ float exagg		     /*exagg mesh deformations*/;

 int	shear,			/*include shear deformations*/
	anlyz;			/*1:stiff analysis 0:data check only*/

 char	ans,		   /*for use in yes/no procedure    */
	mesh_file[MAXLINE],		  /* mesh file name */
	ann_file[MAXLINE];

 do {
	printf("\nEnter \"1\" to include shear deformation.  ");
	printf(" Otherwise, enter \"0\" : ");
	scanf ("%i",&shear);}
 while (shear !=1 && shear != 0);

 do {
	printf("\nWhat is the mesh file name?  ");	scanf ("%s", mesh_file);
	printf(" The file name you input is %s", mesh_file );
	printf("  ... Is this okay? (y/n) ");	scanf ("%s", &ans);
 } while (ans != 'y');

 do {
	printf("\nWhat is the annotation file name?  ");
	scanf ("%s", ann_file);
	printf(" The file name you input is %s", ann_file);
	printf("  ... Is this okay? (y/n) ");	scanf ("%s", &ans);
 } while (ans != 'y');

 do {
	printf("\nBy what factor do you want to exaggerate mesh deformation? ");
	scanf ("%f", &exagg);
	printf(" The number you input is %f ", exagg );
	printf("  ... Is this okay? (y/n) ");	scanf ("%s", &ans);
 } while ( ans != 'y');

 do {
	printf("\nEnter \"1\" to include stiffness analysis. ");
	printf("Enter \"0\" to data check only. ");	scanf ("%i",&anlyz);
 } while ( anlyz != 1 && anlyz != 0 );

 fprintf(fpout,"\n%d\n",shear);
 fprintf(fpout,"%s   %s   %f\n", mesh_file, ann_file, exagg);
 fprintf(fpout,"%d\n", anlyz);
}

/***** Loaded Joints *****/
void joint_loads(void)
{
 char	ans;
 int	j, f;
 float	Fx, Fy, Fz, Mxx, Myy, Mzz; /*loaded joints*/

 printf ("\nYour frame may have concentrated loads at the joints.\n");
 do {
	printf (" Input the number of joint loads : "); scanf ("%d",&nF);
	printf(" %5d joint loads", nF );
	printf("\t\t\t\t     ... Is this okay? (y/n) ");
	scanf("%s", &ans);
 } while( ans != 'y' || nF < 0 );

 fprintf(fpout,"\n%d\n", nF);

 for ( f=1; f<=nF ; f++ ) {
    do {
	printf(" Enter the joint number for joint load number %d : ", f);
	scanf("%d", &j);
	printf(" For joint %d, input values for the ... \n", j);
	printf("   point force in the x-direction, Fx[%d] : ",j);
	scanf ("%f",&Fx);
	printf("   point force in the y-direction, Fy[%d] : ",j);
	scanf ("%f",&Fy);
	printf("   point force in the z-direction, Fz[%d] : ",j);
	scanf ("%f",&Fz);
	printf("   moment about the x-axis,       Mxx[%d] : ",j);
	scanf ("%f",&Mxx);
	printf("   moment about the y-axis,       Myy[%d] : ",j);
	scanf ("%f",&Myy);
	printf("   moment about the z-axis,       Mzz[%d] : ",j);
	scanf ("%f",&Mzz);
	printf("joint    Fx       Fy       Fz       Mxx      Myy      Mzz \n");
	printf("----- -------- -------- -------- -------- -------- --------\n");
	printf("%5d %8.3f %8.3f %8.3f %8.2f %8.2f %8.2f\n",
						j, Fx, Fy, Fz, Mxx, Myy, Mzz );
	printf("\t\t\t\t\t\t     ... Is this okay? (y/n) "); scanf("%s", &ans);
    } while( ans != 'y' || j > nJ );
    fprintf(fpout,"%4d  %f  %f  %f  %f  %f  %f\n", j, Fx,Fy,Fz, Mxx,Myy,Mzz );
 }  /*****End  "for" Loaded Joints ****/

} /* end Function joint_loads */


/***** Uniform Distributed Loads *****/
void distrb_loads(void)
{
 int	m, f;
 char	ans;
 float	Wx, Wy, Wz;    /*Uniform Distributed loads*/

 printf ("\nYour frame may have distributed loads on the members.\n");
 do {
	printf (" Input the number of distributed loads : "); scanf ("%d",&nW);
	printf(" %5d distributed loads", nW );
	printf("\t\t\t     ... Is this okay? (y/n) ");
	scanf("%s", &ans);
 } while( ans != 'y' || nW < 0 );

 fprintf(fpout,"\n%d\n", nW); 

 for (f=1; f<=nW ; f++) {
    do {
	printf(" Enter the member number for distributed load number %d : ", f);
	scanf("%d", &m);
	printf(" For member %d, input values for the ... \n", m);
	printf("   distributed load in the x-direction, Wx[%d] : ",m);
	scanf ("%f",&Wx);
	printf("   distributed load in the y-direction, Wy[%d] : ",m);
	scanf ("%f",&Wy);
	printf("   distributed load in the z-direction, Wz[%d] : ",m);
	scanf ("%f",&Wz);
	printf("member      Wx           Wy           Wz     \n");
	printf("------ ------------ ------------ ------------\n");
	printf("%6d %12.4f %12.4f %12.4f", m, Wx, Wy, Wz );
	printf("        ... Is this okay? (y/n) "); scanf ("%s", &ans);
    } while( ans != 'y' || m > nM );
    fprintf(fpout,"%4d   %.4f   %.4f   %.4f\n",m, Wx, Wy, Wz);
 }  /*****End  "for" Loaded Members ****/

} /* end Function distrb_loads */


/***** Concentrated Point Loads On Members *****/
void concen_loads (void)
{
 char	ans;
 int	m=0,f=0;
 float	Px, Py, Pz, x;  

 printf ("\nYour frame may have point loads on the members.\n");
 printf("Point loads are specified by their values in member coordinates");
 printf(" (Px, Py, Pz),\n");
 printf("and a distance (x) along the member from joint J1.\n");

 do {
	printf (" Input the number of point loads : "); scanf ("%d",&nP);
	printf(" %5d concentrated point loads", nP );
	printf("\t\t\t     ... Is this okay? (y/n) ");
	scanf("%s", &ans);
 } while( ans != 'y' || nP < 0 );

 fprintf(fpout,"\n%d\n", nP); 

 for (f=1; f<=nP ; f++) {
    do {
	printf(" Enter the member number for concentrated load number %d : ",f);
	scanf("%d", &m);
	printf(" For member %d, input values for the ... \n", m);
	printf("   concentrated load in the x-direction, Px[%d] : ",m);
	scanf ("%f",&Px);
	printf("   concentrated load in the y-direction, Py[%d] : ",m);
	scanf ("%f",&Py);
	printf("   concentrated load in the z-direction, Pz[%d] : ",m);
	scanf ("%f",&Pz);
	printf("   distance from joint J1,                x[%d] : ",m);
	scanf ("%f",&x);
	printf("member      Px           Py           Pz          x    \n");
	printf("------ ------------ ------------ ------------ -------- \n");
	printf("%6d %12.4f %12.4f %12.4f %8.3f\n", m, Px, Py, Pz, x );
	printf("\t\t\t\t\t\t     ... Is this okay? (y/n) "); scanf("%s", &ans);
    } while( ans != 'y' || m < 1 || m > nM );
    fprintf(fpout,"%4d  %f  %f  %f  %f\n",m, Px, Py, Pz, x);
 }  
} /* end Function concen_loads */


/***** Temperature Changes *****/
void temperature(void)
{
 int	m, t;
 char	ans;
 float	a, hy, hz,			/* member properties	*/
	Typls, Tymin, Tzpls, Tzmin ;	/* temperature changes	*/

 printf("\nYou may specify temperature changes or thermal gradients ");
 printf("for any member.\n");
 printf("These are specified by the coef. of thermal expansion (a), \n");
 printf("member depths (hy,hz) and surface temperatures ");
 printf("(Ty+, Ty-, Tz+, Tz-).\n");

 do {
	printf (" Input the number of members with temperature changes : ");
	scanf ("%d", &nT);
	printf(" %5d members with temperature changes", nT );
	printf("\t\t     ... Is this okay? (y/n) ");	scanf("%s", &ans);
 } while( ans != 'y' || nT < 0 || nT > nM );

 fprintf(fpout,"\n%d\n", nT); 

 for ( t=1; t <= nT; t++ ) {
    do {
	printf(" Enter the member number for temperature change %d : ",t);
	scanf("%d", &m);
	printf(" For member %d, input values for the ... \n", m);
	printf("   coefficient of thermal expansion,        a[%d] : ",m);
	scanf ("%f", &a);
	printf("   depth in the local y direction,         hy[%d] : ",m);
	scanf ("%f", &hy);
	printf("   depth in the local z direction,         hy[%d] : ",m);
	scanf ("%f", &hz);
	printf("   temperature on the positive y surface, Ty+[%d] : ",m);
	scanf ("%f", &Typls);
	printf("   temperature on the negative y surface, Ty-[%d] : ",m);
	scanf ("%f", &Tymin);
	printf("   temperature on the positive z surface, Tz+[%d] : ",m);
	scanf ("%f", &Tzpls);
	printf("   temperature on the negative z surface, Tz-[%d] : ",m);
	scanf ("%f", &Tzmin);
	printf("member    a       hy      hz     Ty+    Ty-    Tz+    Tz-  \n");
	printf("------ ------- ------- ------- ------ ------ ------ ------ \n");
	printf("%6d %7.1e %7.3f %7.3f %6.1f %6.1f %6.1f %6.1f \n",
				m, a, hy, hz, Typls, Tymin, Tzpls, Tzmin );
	printf("\t\t\t\t\t\t     ... Is this okay? (y/n) "); scanf("%s", &ans);
    } while( ans != 'y' || m < 1 || m > nM );
    fprintf(fpout,"%4d %.4e %.3f %.3f %.3f %.3f %.3f %.3f \n",
				m, a, hy, hz, Typls, Tymin, Tzpls, Tzmin );
 }  
} /* end function temperature */

/***** Reactions *****/
void reactions(void)
{
 int	j, r;
 char	ans;
 int	Rx, Ry, Rz, Rxx, Ryy, Rzz; /*Restrained Joints */

 printf("\nYou must specify enough reactions ");
 printf("to restrain your frame in all six directions.\n");
 do {
	printf (" Input the number of restrained joints : "); scanf ("%d",&nR);
	printf(" %5d restrained joints", nR );
	printf("\t\t\t     ... Is this okay? (y/n) ");	scanf("%s", &ans);
 } while( ans != 'y' || nR < 1 || nR > nJ );

 fprintf(fpout,"\n%d\n", nR);

 for (r=1; r <= nR; r++) {
   do {  
	printf(" Enter the joint number for reaction number %d : ", r);
	scanf ("%d", &j);
	printf(" For joint %d, input values for the ... \n", j);
	printf("   x- direction  reaction (1:fixed 0:free),  Rx[%i] : ",j);
	scanf ("%d", &Rx);
	printf("   y- direction  reaction (1:fixed 0:free),  Ry[%i] : ",j);
	scanf ("%d", &Ry);
	printf("   z- direction  reaction (1:fixed 0:free),  Rz[%i] : ",j);
	scanf ("%d", &Rz);
	printf("   x-axis moment reaction (1:fixed 0:free), Rxx[%i] : ",j);
	scanf ("%d", &Rxx);
	printf("   y-axis moment reaction (1:fixed 0:free), Ryy[%i] : ",j);
	scanf ("%d", &Ryy);
	printf("   z-axis moment reaction (1:fixed 0:free), Rzz[%i] : ",j);
	scanf ("%d", &Rzz);

	printf("joint  Rx  Ry  Rz Rxx Ryy Rzz  \n");
	printf("----- --- --- --- --- --- --- \n");
	printf("%5d %3d %3d %3d %3d %3d %3d", j, Rx, Ry, Rz, Rxx, Ryy, Rzz );
	printf("\t\t\t    ... Is this okay? (y/n) ");
	scanf("%s", &ans);
   } while( tolower(ans) != 'y' || j < 1 || j > nJ ); 
   fprintf(fpout,"%4d %3d %3d %3d %3d %3d %3d \n", j, Rx,Ry,Rz, Rxx,Ryy,Rzz );
 }
}

/***** Prescribed Displacements *****/
void displacements(void)
{
 int	j, d;
 char	ans;
 float	Dx, Dy, Dz, Dxx, Dyy, Dzz; 

 printf("\nYou may prescribe a displacement ");
 printf("at any coordinate that has a reaction.\n");
 do {
	printf (" Input the number of joints with prescribed displacements : ");
	scanf ("%d",&nD);
	printf(" %5d prescribed displacements", nD );
	printf("\t\t\t     ... Is this okay? (y/n) ");	scanf("%s", &ans);
 } while( ans != 'y' || nD < 0 || nD > nR );

 fprintf(fpout,"\n%d\n", nD);

 for (d=1; d <= nD; d++) {
   do {  
	printf(" Enter the joint number for displacement number %d : ", d);
	scanf ("%d", &j);
	printf(" For joint %d, input values for the ... \n", j);
	printf("   x-direction displacement Dx[%i] : ", j); scanf ("%f", &Dx);
	printf("   y-direction displacement Dy[%i] : ", j); scanf ("%f", &Dy);
	printf("   z-direction displacement Dz[%i] : ", j); scanf ("%f", &Dz);
	printf("   x- axis rotation        Dxx[%i] : ", j); scanf ("%f", &Dxx );
	printf("   y- axis rotation        Dyy[%i] : ", j); scanf ("%f", &Dyy );
	printf("   z- axis rotation        Dzz[%i] : ", j); scanf ("%f", &Dzz );

	printf("joint     Dx      Dy       Dz       Dxx      Dyy      Dzz  \n");
	printf("----- -------- -------- -------- -------- -------- --------\n");
	printf("%5d %8.5f %8.5f %8.5f %8.6f %8.6f %8.6f \n",
						j, Dx, Dy, Dz, Dxx, Dyy, Dzz );
	printf("\t\t\t\t\t\t     ... Is this okay? (y/n) "); scanf("%s", &ans);
   } while( tolower(ans) == 'n' || j < 1 || j > nJ ); 
   fprintf(fpout,"%4d %8.5f %8.5f %8.5f %8.6f %8.6f %8.6f \n",
						j, Dx, Dy, Dz, Dxx, Dyy, Dzz );
 }
}

/***** DYNAMIC INERTIA DATA INPUT *****/
void modal_files(void)
{
 char	ans, mode_file[MAXLINE];
 int	lump, modes;
 float	tol;

 printf("\nYou may compute dynamic vibrational properties for your frame.\n");
 do {
     printf(" Input the number of desired modes         : ");
     scanf("%d", &modes);
     do {
	printf(" Input 0 for consistent mass, 1 for lumped : ");
	scanf("%d",&lump);
     } while( lump!=0 && lump!=1 );
     printf(" Input the mode shape data file name       : ");
     scanf("%s", mode_file);
     printf(" Input the convergence tolerance           : ");
     scanf("%f",&tol);
     printf("modes  lump    mode file name       tol \n");
     printf("-----  ---- -------------------- --------\n"); 
     printf("%5d  %4d %20s %8.6f", modes, lump, mode_file, tol );
     printf("            ... Is this okay? (y/n) "); scanf("%s", &ans);
 } while( tolower(ans) != 'y' || modes > nJ || tol <= 0 );

 fprintf(fpout,"\n%d \n%d \n %.80s \n %f \n",  modes, lump, mode_file, tol );
}			   /* ^^^-THIS IS THE FIX ... don't ask me why ... */

/***** Member Density and extra masses, not including self masses *****/
void inertia(void)
{
 int	m;
 char	ans;
 float	d, Ms;
	 
 printf(" You must specify density and lumped masses for each member\n");
 
 for (m=1; m<=nM ; m++) {
    do {
	printf(" For member %i, input values for the ...\n", m);
	printf("   mass density,  d[%i] : ",m);	scanf ("%f",&d );
	printf("   lumped mass,  Ms[%i] : ",m);	scanf ("%f",&Ms );
	printf("member	     d             Ms      \n", m, m);
	printf("------  ------------  ------------ \n"); 
	printf("%6d  %12.8f  %12.8f", m, d, Ms );
	printf("\t\t     ... Is this okay? (y/n) "); scanf("%s", &ans);
    } while ( tolower(ans) != 'y' || (Ms <= 0 && d <= 0) );
    fprintf(fpout,"%4d   %f    %f \n", m, d, Ms );
 }
}
