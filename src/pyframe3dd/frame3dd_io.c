/*
 This file is part of FRAME3DD:
 Static and dynamic structural analysis of 2D and 3D frames and trusses with
 elastic and geometric stiffness.
 ---------------------------------------------------------------------------
 http://frame3dd.sourceforge.net/
 ---------------------------------------------------------------------------
 Copyright (C) 1992-2015  Henri P. Gavin

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
	Input/output routines for FRAME.

	@note The file format for FRAME is defined in doc/user_manual.html.
*/

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <assert.h>

#include "common.h"
#include "frame3dd_io.h"
#include "coordtrans.h"
#include "HPGmatrix.h"
#include "HPGutil.h"
#include "NRutil.h"

/* #define MASSDATA_DEBUG */

/* forward decls */

static void getline_no_comment(
	FILE *fp,    /**< pointer to the file from which to read */
	char *s,     /**< pointer to the string to which to write */
	int lim      /**< the longest anticipated line length  */
);


/*
 * PARSE_OPTIONS -  parse command line options		
 * command line options over-ride values in the input data file	 	
 * 04 Mar 2009, 22 Sep 2009
 */
void parse_options (
	int argc, char *argv[], 
	char IN_file[], char OUT_file[], 
	int *shear_flag,
	int *geom_flag,
	int *anlyz_flag,
	double *exagg_flag,
	int *D3_flag,
	int *lump_flag,
	int *modal_flag,
	double *tol_flag, 
	double *shift_flag,
	float *pan_flag,
	int *write_matrix,
	int *axial_sign,
	int *condense_flag,
	int *verbose,
	int *debug
){

	char	option;
	char	errMsg[MAXL];
	int	sfrv=0;		/* *scanf return value	*/

	/* default values */

	*shear_flag = *geom_flag  = *anlyz_flag = *lump_flag = *modal_flag = -1;
	*exagg_flag = *tol_flag = *shift_flag = -1.0;
	*D3_flag = 0;
	*pan_flag = -1.0;
	*condense_flag = -1;
	*write_matrix = 0;
	*axial_sign = 1;
	*debug = 0; *verbose = 1;

	strcpy(  IN_file , "\0" );
	strcpy( OUT_file , "\0" );

	/* set up file names for the the input data and the output data */

	switch ( argc ) {
	 case 1: {
 		fprintf(stderr,"\n Frame3DD version: %s\n", VERSION);
		fprintf(stderr," Analysis of 2D and 3D structural frames with elastic and geometric stiffness.\n");
		fprintf(stderr," http://frame3dd.sourceforge.net\n\n");
		fprintf (stderr," Please enter the  input data file name: ");
		sfrv=scanf("%s", IN_file );
		if (sfrv != 1) sferr("IN_file");
		fprintf (stderr," Please enter the output data file name: ");
		sfrv=scanf("%s", OUT_file );
		if (sfrv != 1) sferr("OUT_file");
		return;
	 }
	 case 3: {
		if ( argv[1][0] != '-' ) {
			strcpy(  IN_file , argv[1] );
			strcpy( OUT_file , argv[2] );
			return;
		}
	 }
	}

	// remaining unused flags ... b j k n u y 

	while ((option=getopt(argc,argv, "i:o:acdhqvwxzs:e:f:g:l:m:p:r:t:")) != -1){
		switch ( option ) {
			case 'i':		/* input data file name */
				strcpy(IN_file,optarg);
				break;
			case 'o':		/* output data file name */
				strcpy(OUT_file,optarg);
				break;
			case 'h':		/* help	*/
				display_help();
				exit(0);
			case 'v':		/* version */
				display_version();
				exit(0);
			case 'a':		/* about */
				display_version_about();
				exit(0);
			case 'q':		/* quiet */
				*verbose = 0;
				break;
			case 'c':		/* data check only */
				*anlyz_flag = 0;
				break;
			case 'd':		/* debug */
				*debug = 1;
				break;
			case 'w':		/* write stiffness and mass */
				*write_matrix = 1;
				break;
			case 'x':		/* write sign of axial forces */
				*axial_sign = 0;
				break;
			case 's':		/* shear deformation */
				if (strcmp(optarg,"Off")==0)
					*shear_flag = 0;
				else if (strcmp(optarg,"On")==0)
					*shear_flag = 1;
				else {
				 errorMsg("\n frame3dd command-line error: argument to -s option should be either On or Off\n");
				 exit(3);
				}
				break;
			case 'g':		/* geometric stiffness */
				if (strcmp(optarg,"Off")==0)
					*geom_flag = 0;
				else if (strcmp(optarg,"On")==0)
					*geom_flag = 1;
				else {
				 errorMsg("\n frame3dd command-line error: argument to -g option should be either On or Off\n");
				 exit(4);
				}
				break;
			case 'e':		/* static mesh exagg. factor */
				*exagg_flag = atof(optarg);
				break;
			case 'z':		/* force 3D plotting */
				*D3_flag = 1;
				break;
			case 'l':		/* lumped or consistent mass */
				if (strcmp(optarg,"Off")==0)
					*lump_flag = 0;
				else if (strcmp(optarg,"On")==0)
					*lump_flag = 1;
				else {
				 errorMsg("\n frame3dd command-line error: argument to -l option should be either On or Off\n");
				 exit(5);
				}
				break;
			case 'm':		/* modal analysis method */
				if (strcmp(optarg,"J")==0)
					*modal_flag = 1;
				else if (strcmp(optarg,"S")==0)
					*modal_flag = 2;
				else {
				 errorMsg("\n frame3dd command-line error: argument to -m option should be either J or S\n");
				 exit(6);
				}
				break;
			case 't':		/* modal analysis tolerence */
				*tol_flag = atof(optarg);
				if (*tol_flag == 0.0) {
				 errorMsg("\n frame3dd command-line error: argument to -t option should be a number.\n");
				 exit(7);
				}
				break;
			case 'f':		/* modal analysis freq. shift */
				*shift_flag = atof(optarg);
				if (*shift_flag == 0.0) {
				 errorMsg("\n frame3dd command-line error: argument to -f option should be a number.\n");
				 exit(8);
				}
				break;
			case 'p':		/* pan rate	*/
				*pan_flag = atof(optarg);
				if (*pan_flag < 0.0) {
				 errorMsg("\n frame3dd command-line error: argument to -p option should be a positive number.\n");
				 exit(9);
				}
				break;
			case 'r':		/* matrix condensation method */
				*condense_flag = atoi(optarg);
				if (*condense_flag < 0 || *condense_flag > 3) {
				 errorMsg("\n frame3dd command-line error: argument to -r option should be 0, 1, or 2.\n");
				 exit(10);
				}
				break;
			case '?':
				sprintf(errMsg,"  Missing argument or Unknown option: -%c\n\n", option );
				errorMsg(errMsg);
				display_help();
				exit(2);
		}
	}

	if ( strcmp(IN_file,"\0") == 0 ) {
		fprintf (stderr," Please enter the  input data file name: ");
		sfrv=scanf("%s", IN_file );
		if (sfrv != 1) sferr("IN_file");
		fprintf (stderr," Please enter the output data file name: ");
		sfrv=scanf("%s", OUT_file );
		if (sfrv != 1) sferr("OUT_file");
	}
	if ( strcmp(IN_file,"\0") != 0 && strcmp(OUT_file,"\0") == 0 ) {
		strcpy( OUT_file, IN_file );
		strcat( OUT_file, ".out" );
	}
}


/*
 * DISPLAY_HELP -  display help information to stderr	
 * 04 Mar 2009, 22 Sep 2009
 */
void display_help()
{
 textColor('g','x','x','x');
 fprintf(stderr,"\n Frame3DD version: %s\n", VERSION);
 fprintf(stderr," Analysis of 2D and 3D structural frames with elastic and geometric stiffness.\n");
 fprintf(stderr," http://frame3dd.sourceforge.net\n\n");
/* fprintf(stderr,"  Usage: frame3dd -i<input> -o<output> [-hvcqz] [-s<On|Off>] [-g<On|Off>] [-e<value>] [-l<On|Off>] [-f<value>] [-m J|S] [-t<value>] [-p<value>] \n");
 */
 fprintf(stderr,"  Frame3DD may be run with interactive prompting for file names by typing ...\n");
 fprintf(stderr,"       frame3dd \n\n");
 fprintf(stderr,"  Frame3DD may be run without command-line options by typing ...\n");
 fprintf(stderr,"       frame3dd <InFile> <OutFile> \n\n");

 fprintf(stderr,"  Frame3DD may be run with command-line options by typing ...\n");
 fprintf(stderr,"       frame3dd -i <InFile> -o <OutFile> [OPTIONS] \n\n");

 fprintf(stderr," ... where [OPTIONS] over-rides values in the input data file and includes\n");
 fprintf(stderr,"     one or more of the following:\n\n");

 fprintf(stderr," -------------------------------------------------------------------------\n");
 fprintf(stderr,"  -i  <InFile>  the  input data file name --- described in the manual\n");
 fprintf(stderr,"  -o <OutFile>  the output data file name\n");
 fprintf(stderr,"  -h            print this help message and exit\n");
 fprintf(stderr,"  -v            display program version, website, brief help info and exit\n");
 fprintf(stderr,"  -a            display program version, website and exit\n");
 fprintf(stderr,"  -c            data check only - the output data reviews the input data\n");
 fprintf(stderr,"  -w            write stiffness and mass matrices to files named Ks Kd Md\n");
 fprintf(stderr,"  -x            suppress writing of 't' or 'c' for sign of axial forces\n");
 fprintf(stderr,"  -q            suppress screen output except for warning messages\n");
 fprintf(stderr,"  -s  On|Off    On: include shear deformation or Off: neglect ...\n");
 fprintf(stderr,"  -g  On|Off    On: include geometric stiffness or Off: neglect ...\n");
 fprintf(stderr,"  -e <value>    static deformation exaggeration factor for Gnuplot output\n");
 fprintf(stderr,"  -z            force X-Y-Z plotting\n");
 fprintf(stderr,"  -l  On|Off    On: lumped mass matrix or Off: consistent mass matrix\n");
 fprintf(stderr,"  -f <value>    modal frequency shift for unrestrained structures\n");
 fprintf(stderr,"  -m   J|S      modal analysis method: J=Jacobi-Subspace or S=Stodola\n");
 fprintf(stderr,"  -t <value>    convergence tolerance for modal analysis\n");
 fprintf(stderr,"  -p <value>    pan rate for mode shape animation\n");
 fprintf(stderr,"  -r <value>    matrix condensation method: 0, 1, 2, or 3 \n");
 fprintf(stderr," -------------------------------------------------------------------------\n");
 color(0);

}


/*
 * DISPLAY_USAGE -  display usage information to stderr	
 * 04 Mar 2009
 */
void display_usage()
{
 fprintf(stderr,"\n Frame3DD version: %s\n", VERSION);
 fprintf(stderr," Analysis of 2D and 3D structural frames with elastic and geometric stiffness.\n");
 fprintf(stderr," http://frame3dd.sourceforge.net\n\n");
/* fprintf(stderr,"  Usage: frame3dd -i<input> -o<output> [-hvcqz] [-s<On|Off>] [-g<On|Off>] [-e<value>] [-l<On|Off>] [-f<value>] [-m J|S] [-t<value>] [-p<value>] \n");
 */
 fprintf(stderr,"  Usage: frame3dd -i <input> -o <output> [OPTIONS] \n\n");

 fprintf(stderr,"  Type ...   frame3dd -h   ... for additional help information.\n\n");

}

/*
 * DISPLAY_VERSION_HELP -  display version, website, and brief help info. to stderr
 * 04 Mar 2009
 */
void display_version()
{
 fprintf(stderr,"\n Frame3DD version: %s\n", VERSION);
 fprintf(stderr," Analysis of 2D and 3D structural frames with elastic and geometric stiffness.\n");
 fprintf(stderr," http://frame3dd.sourceforge.net\n\n");

 fprintf(stderr,"  Usage: frame3dd -i <input> -o <output> [OPTIONS] \n\n");

 fprintf(stderr,"  Type ...   frame3dd -h   ... for additional help information.\n\n");
}


/*
 * DISPLAY_VERSION_ABOUT-  display version and website to stderr for 
 * running as a background process 
 * 22 Sep 2009
 * Contributed by Barry Sanford, barry.sanford@trimjoist.com
 */
void display_version_about()
{
 fprintf(stderr," Frame3DD version: %s\n", VERSION);
 fprintf(stderr," Analysis of 2D and 3D structural frames with elastic and geometric stiffness\n");
 fprintf(stderr," http://frame3dd.sourceforge.net\n");
 fprintf(stderr," GPL Copyright (C) 1992-2015, Henri P. Gavin \n");
 fprintf(stderr," Frame3DD is distributed in the hope that it will be useful");
 fprintf(stderr," but with no warranty.\n");
 fprintf(stderr," For details see the GNU Public Licence:");
 fprintf(stderr," http://www.fsf.org/copyleft/gpl.html\n");
}


/*
 * READ_NODE_DATA  -  read node location data		
 * 04 Jan 2009
 */
void read_node_data( FILE *fp, int nN, vec3 *xyz, float *r )
{
	int	i, j,
		sfrv=0;		/* *scanf return value	*/
	char	errMsg[MAXL];

	for (i=1;i<=nN;i++) {		/* read node coordinates	*/
		sfrv=fscanf(fp, "%d", &j );
		if (sfrv != 1) sferr("node number in node data");
		if ( j <= 0 || j > nN ) {
		    sprintf(errMsg,"\nERROR: in node coordinate data, node number out of range\n(node number %d is <= 0 or > %d)\n", j, nN);
		    errorMsg(errMsg);
		    exit(41);
		}
		sfrv=fscanf(fp, "%lf %lf %lf %f", &xyz[j].x, &xyz[j].y, &xyz[j].z, &r[j]);
		if (sfrv != 4) sferr("node coordinates in node data");
		/* fprintf(stderr,"\nj = %d, pos = (%lf, %lf, %lf), r = %f", j, xyz[j].x, xyz[j].y, xyz[j].z, r[j]); */
		r[j] = fabs(r[j]);
	}
	return;
}


/*
 * READ_FRAME_ELEMENT_DATA  -  read frame element property data		
 * 04 Jan 2009
 */
void read_frame_element_data (
	FILE *fp,
	int nN, int nE, vec3 *xyz, float *r,
	double *L, double *Le,
	int *N1, int *N2,
	float *Ax, float *Asy, float *Asz,
	float *Jx, float *Iy, float *Iz, float *E, float *G, float *p, float *d
){
	int	n1, n2, i, n, b;
	int	*epn, epn0=0;	/* vector of elements per node */
	int	sfrv=0;		/* *scanf return value */
	char	errMsg[MAXL];

	epn = ivector(1,nN);

	for (n=1;n<=nN;n++)	epn[n] = 0;

	for (i=1;i<=nE;i++) {		/* read frame element properties */
		sfrv=fscanf(fp, "%d", &b );
		if (sfrv != 1) sferr("frame element number in element data");
		if ( b <= 0 || b > nE ) {
		    sprintf(errMsg,"\n  error in frame element property data: Element number out of range  \n Frame element number: %d  \n", b);
		    errorMsg(errMsg);
		    exit(51);
		}
	      	sfrv=fscanf(fp, "%d %d", &N1[b], &N2[b] );

		epn[N1[b]] += 1;        epn[N2[b]] += 1;

		if (sfrv != 2) sferr("node numbers in frame element data");
		if ( N1[b] <= 0 || N1[b] > nN || N2[b] <= 0 || N2[b] > nN ) {
		    sprintf(errMsg,"\n  error in frame element property data: node number out of range  \n Frame element number: %d \n", b);
		    errorMsg(errMsg);
		    exit(52);
		}
		sfrv=fscanf(fp, "%f %f %f", &Ax[b], &Asy[b], &Asz[b] );
		if (sfrv != 3) sferr("section areas in frame element data");
		sfrv=fscanf(fp, "%f %f %f", &Jx[b],  &Iy[b],  &Iz[b] );
		if (sfrv != 3) sferr("section inertias in frame element data");
		sfrv=fscanf(fp, "%f %f", &E[b], &G[b] );
		if (sfrv != 2) sferr("material moduli in frame element data");
		sfrv=fscanf(fp, "%f", &p[b]);
		if (sfrv != 1) sferr("roll angle in frame element data");

		p[b] = p[b]*PI/180.0;	/* convert from degrees to radians */

		sfrv=fscanf(fp, "%f",  &d[b]);
		if (sfrv != 1) sferr("mass density in frame element data");

		if ( Ax[b] < 0 || Asy[b] < 0 || Asz[b] < 0 ||
		     Jx[b] < 0 ||  Iy[b] < 0 ||  Iz[b] < 0	) {
		 sprintf(errMsg,"\n  error in frame element property data: section property < 0 \n  Frame element number: %d  \n", b);
		 errorMsg(errMsg);
		 exit(53);
		}
		if ( Ax[b] == 0 ) {
		 sprintf(errMsg,"\n  error in frame element property data: cross section area is zero   \n  Frame element number: %d  \n", b);
		 errorMsg(errMsg);
		 exit(54);
		}
		if ( (Asy[b] == 0 || Asz[b] == 0) && G[b] == 0 ) {
		 sprintf(errMsg,"\n  error in frame element property data: a shear area and shear modulus are zero   \n  Frame element number: %d  \n", b);
		 errorMsg(errMsg);
		 exit(55);
		}
		if ( Jx[b] == 0 ) {
		 sprintf(errMsg,"\n  error in frame element property data: torsional moment of inertia is zero   \n  Frame element number: %d  \n", b);
		 errorMsg(errMsg);
		 exit(56);
		}
		if ( Iy[b] == 0 || Iz[b] == 0 ) {
		 sprintf(errMsg,"\n  error: cross section bending moment of inertia is zero   \n  Frame element number : %d  \n", b);
		 errorMsg(errMsg);
		 exit(57);
		}
		if ( E[b] <= 0 || G[b] <= 0 ) {
		 sprintf(errMsg,"\n  error : material elastic modulus E or G is not positive   \n  Frame element number: %d  \n", b);
		 errorMsg(errMsg);
		 exit(58);
		}
		if ( d[b] <= 0 ) {
		 sprintf(errMsg,"\n  error : mass density d is not positive   \n  Frame element number: %d  \n", b);
		 errorMsg(errMsg);
		 exit(59);
		}
	}

	for (b=1;b<=nE;b++) {		/* calculate frame element lengths */
		n1 = N1[b];
		n2 = N2[b];

#define SQ(X) ((X)*(X))
		L[b] =	SQ( xyz[n2].x - xyz[n1].x ) +
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
		   exit(60);
		}
		if ( Le[b] <= 0.0 ) {
		   sprintf(errMsg, " Node  radii are too large.\n  frame element %d  N1= %d N2= %d L= %e \n  r1= %e r2= %e Le= %e \n",
		   b, n1,n2, L[b], r[n1], r[n2], Le[b] );
		   errorMsg(errMsg);
		   exit(61);
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

	if ( epn0 > 0 ) exit(42);

	return;
}


/*
 * READ_RUN_DATA  -  read information for analysis
 * 29 Dec 2008
 */
void read_run_data (
	FILE	*fp, 
	char	*OUT_file,	/* output data file name */
	int	*shear,
	int	shear_flag,
	int	*geom,
	int	geom_flag,
	char	*meshpath,
	char	*plotpath,
	char	*infcpath,
	double	*exagg_static,
	double	exagg_flag,
	float   *scale,
	float	*dx,
	int	*anlyz,
	int	anlyz_flag,
	int	debug
){
	int	full_len=0, len=0, i;
	char	base_file[96] = "EMPTY_BASE";
	char	mesh_file[96] = "EMPTY_MESH";
	int	sfrv=0;		/* *scanf return value */

	strcpy(base_file,OUT_file);	
	while ( base_file[len++] != '\0' )
		/* the length of the base_file */ ;
	full_len = len;
	while ( base_file[len--] != '.' && len > 0 )
		/* find the last '.' in base_file */ ;
	if ( len == 0 )	len = full_len;
	base_file[++len] = '\0';	/* end base_file at the last '.' */

	strcpy(plotpath,base_file);
	strcat(plotpath,".plt");

	strcpy(infcpath,base_file);
	strcat(infcpath,".if");

	while ( base_file[len] != '/' && base_file[len] != '\\' && len > 0 )
		len--;	/* find the last '/' or '\' in base_file */ 
	i = 0;
	while ( base_file[len] != '\0' )
		mesh_file[i++] = base_file[len++];
	mesh_file[i] = '\0';
	strcat(mesh_file,"-msh");
	output_path(mesh_file,meshpath,FRAME3DD_PATHMAX,NULL);

	if ( debug) {
		fprintf(stderr,"OUT_FILE  = %s \n", OUT_file);
		fprintf(stderr,"BASE_FILE = %s \n", base_file);
		fprintf(stderr,"PLOTPATH  = %s \n", plotpath);
		fprintf(stderr,"MESH_FILE = %s \n", mesh_file);
		fprintf(stderr,"MESHPATH  = %s \n", meshpath);
	}

	sfrv=fscanf( fp, "%d %d %lf %f %f", shear,geom, exagg_static,scale,dx );
	if (sfrv != 5) sferr("shear, geom, exagg_static, scale, or dx variables");

	if (*shear != 0 && *shear != 1) {
	    errorMsg(" Rember to specify shear deformations with a 0 or a 1 \n after the frame element property info.\n");
	    exit(71);
	}

	if (*geom != 0 && *geom != 1) {
	    errorMsg(" Rember to specify geometric stiffness with a 0 or a 1 \n after the frame element property info.\n");
	    exit(72);
	}

	if ( *exagg_static < 0.0 ) {
	    errorMsg(" Remember to specify an exageration factor greater than zero.\n");
	    exit(73);
	}

	if ( *dx <= 0.0 && *dx != -1 ) {
	    errorMsg(" Remember to specify a frame element increment greater than zero.\n");
	    exit(74);
	}
	

	/* over-ride values from input data file with command-line options */
	if ( shear_flag != -1   )	*shear = shear_flag;
	if ( geom_flag  != -1   )	*geom = geom_flag;
	if ( exagg_flag != -1.0 )	*exagg_static = exagg_flag;
	if ( anlyz_flag != -1.0 )	*anlyz = anlyz_flag;


	return;
}


/*
 * FRAME3DD_GETLINE -  get line into a character string. from K&R        03feb94
 */
int frame3dd_getline (
FILE	*fp,
char    *s,
int     lim
){
    int     c=0, i=0;

    while (--lim > 0 && (c=getc(fp)) != EOF && c != '\n' )
	s[i++] = c;
/*      if (c == '\n')  s[i++] = c;	*/
    s[i] = '\0';
    return i;
}


/* platform-dependent path sperator character ... */

#if defined(WIN32) || defined(DJGPP)
static const char sep = '\\';
#else
static const char sep = '/';
#endif

/*
 * TEMP_DIR
 * return platform-specific temp file location -- 
 * John Pye, Feb 2009
 */
static const char *temp_dir(){
#if defined(WIN32) || defined(DJGPP)
	char *tmp;
	tmp = getenv("TEMP");
	if ( tmp==NULL ) {
		fprintf(stderr,
"ERROR: Environment Variables %%TEMP%% and %%FRAME3DD_OUTDIR%% are not set.  "
"At least one of these variables must be set so that Frame3DD knows where to "
"write its temporary files.  Set one of these variable, then re-run Frame3DD."
"The Frame3DD on-line documentation provides help on this issue.");
		exit(15);
	} 
#else
	const char *tmp = "/tmp";	/* Linux, Unix, OS X	*/
#endif
	return tmp;
}


/*
 * OUTPUT_PATH
 * return path for output files using either current directory, or FRAME3DD_OUTDIR
 * if specified. -- 
 * John Pye, Feb 2009.
 */
void output_path(const char *fname, char fullpath[], const int len, const char *default_outdir) {
	int res;
	assert(fname!=NULL);

	/*			deprecated code, January 15 2010 ...
	if ( fname[0]==sep ) {	in Win32 absolute path starts with C:\ not \ ??
		// absolute output path specified 
//		res = snprintf(fullpath,len,"%s",fname);
		res = sprintf(fullpath,"%s",fname);
	} else {
	*/

//		fprintf(stderr,"Generating output path for file '%s'\n",fname);
		const char *outdir;
		outdir = getenv("FRAME3DD_OUTDIR");
		if (outdir==NULL) {
			if (default_outdir==NULL)
				outdir = temp_dir();
			else
				outdir = default_outdir;
		}
//		res = snprintf(fullpath,len,"%s%c%s",outdir,sep,fname);
		res = sprintf(fullpath,"%s%c%s",outdir,sep,fname);

	/*			closing bracket for deprecated code "if"
	}
	*/

	if ( res > len ) {
		errorMsg("ERROR: unable to construct output filename: overflow.\n");
		exit(16);
	}
//	printf("Output file path generated: %s\n",fullpath); /* debug */
}


/*
 * PARSE_INPUT                                                            
 * strip comments from the input file, and write a stripped input file
 * 07 May 2003
 */
void parse_input(FILE *fp, const char *tpath){
	FILE	*fpc;		/* stripped input file pointer	*/
	char	line[256];
	char	errMsg[MAXL];

	if ((fpc = fopen (tpath, "w")) == NULL) {
		sprintf (errMsg,"\n  error: cannot open parsed input data file: '%s' \n", tpath );
		errorMsg(errMsg);
		exit(12);
	}

	do {
		getline_no_comment(fp, line, 256);
		fprintf(fpc, "%s \n", line );
	} while ( line[0] != '_' && line[0] != EOF );

	fclose(fpc);

}


/*
 * GETLINE_NO_COMMENT                                                
 * get a line into a character string. from K&R
 * get the line only up to one of the following characters:  \n  %  #  ? 
 * ignore all comma (,) characters
 * ignore all double quote (") characters
 * ignore all semi-colon (;) characters
 * 09 Feb 2009
 */
void getline_no_comment (
	FILE *fp,   /**< pointer to the file from which to read */
	char *s,    /**< pointer to the string to which to write */
	int lim    /**< the longest anticipated line length  */
){
	int     c=0, i=0;

	while (--lim > 0 && (c=getc(fp)) != EOF && 
		c != '\n' && c != '%' && c != '#' && c != '?' ) {
		if (c != ',' && c != '"' && c != ';')
			s[i++] = c;
		else
			s[i++] = ' ';
	/*      if (c == '\n')  s[i++] = c;     */
	}
	s[i] = '\0';
	if (c != '\n')
		while (--lim > 0 && (c=getc(fp)) != EOF && c != '\n')
		/* read the rest of the line, otherwise do nothing */ ;

	if ( c == EOF ) s[0] = EOF;

	return;
}


/*
 * READ_REACTION_DATA - Read fixed node displacement boundary conditions
 * 29 Dec 2009
 */
void read_reaction_data (
	FILE *fp, int DoF, int nN, int *nR, int *q, int *r, int *sumR, int verbose
){
	int	i,j,l;
	int	sfrv=0;		/* *scanf return value */
	char	errMsg[MAXL];

	for (i=1; i<=DoF; i++)	r[i] = 0;

	sfrv=fscanf(fp,"%d", nR );	/* read restrained degrees of freedom */
	if (sfrv != 1) sferr("number of reactions in reaction data");
	if ( verbose ) {
		fprintf(stdout," number of nodes with reactions ");
		dots(stdout,21);
		fprintf(stdout," nR =%4d ", *nR );
	}
	if ( *nR < 0 || *nR > DoF/6 ) {
		fprintf(stderr," number of nodes with reactions ");
		dots(stderr,21);
		fprintf(stderr," nR = %3d ", *nR );
		sprintf(errMsg,"\n  error: valid ranges for nR is 0 ... %d \n", DoF/6 );
		errorMsg(errMsg);
		exit(80);
	}

	for (i=1; i <= *nR; i++) {
	    sfrv=fscanf(fp,"%d", &j);
	    if (sfrv != 1) sferr("node number in reaction data");
	    for (l=5; l >=0; l--) {

		sfrv=fscanf(fp,"%d", &r[6*j-l] );
		if (sfrv != 1) sferr("reaction value in reaction data");

		if ( j > nN ) {
		    sprintf(errMsg,"\n  error in reaction data: node number %d is greater than the number of nodes, %d \n", j, nN );
		    errorMsg(errMsg);
		    exit(81);
		}
		if ( r[6*j-l] != 0 && r[6*j-l] != 1 ) {
		    sprintf(errMsg,"\n  error in reaction data: Reaction data must be 0 or 1\n   Data for node %d, DoF %d is %d\n", j, 6-l, r[6*j-l] );
		    errorMsg(errMsg);
		    exit(82);
		}
	    }
	    *sumR = 0;
	    for (l=5; l >=0; l--) 	*sumR += r[6*j-l];
	    if ( *sumR == 0 ) {
		sprintf(errMsg,"\n  error: node %3d has no reactions\n   Remove node %3d from the list of reactions\n   and set nR to %3d \n",
		j, j, *nR-1 );
		errorMsg(errMsg);
		exit(83);
	    }
	}
	*sumR=0;	for (i=1;i<=DoF;i++)	*sumR += r[i];
	if ( *sumR < 4 ) {
	    sprintf(errMsg,"\n  Warning:  un-restrained structure   %d imposed reactions.\n  At least 4 reactions are required to support static loads.\n", *sumR );
	    errorMsg(errMsg);
	    /*	exit(84); */
	}
	if ( *sumR >= DoF ) {
	    sprintf(errMsg,"\n  error in reaction data:  Fully restrained structure\n   %d imposed reactions >= %d degrees of freedom\n", *sumR, DoF );
	    errorMsg(errMsg);
	    exit(85);
	}

	for (i=1; i<=DoF; i++)	if (r[i]) q[i] = 0;	else q[i] = 1;

	return;
}


/*
 * READ_AND_ASSEMBLE_LOADS  
 * Read load information data, assemble load vectors in global coordinates
 * Returns vector of equivalent loadal forces F_temp and F_mech and 
 * a matrix of equivalent element end forces eqF_temp and eqF_mech from 
 * distributed internal and temperature loadings.  
 * eqF_temp and eqF_mech are computed for the global coordinate system 
 * 2008-09-09, 2015-05-15
 */
void read_and_assemble_loads (
		FILE *fp,
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
		double ***eqF_mech, // equivalent mech loads, global coord 
		double ***eqF_temp, // equivalent temp loads, global coord 
		int verbose
){
	float	hy, hz;			/* section dimensions in local coords */

	float	x1,x2, w1,w2;
	double	Ln, R1o, R2o, f01, f02; 

	/* equivalent element end forces from distributed and thermal loads */
	double	Nx1, Vy1, Vz1, Mx1=0.0, My1=0.0, Mz1=0.0,
		Nx2, Vy2, Vz2, Mx2=0.0, My2=0.0, Mz2=0.0;
	double	Ksy, Ksz, 		/* shear deformatn coefficients	*/
		a, b,			/* point load locations */
		t1, t2, t3, t4, t5, t6, t7, t8, t9;	/* 3D coord Xfrm coeffs */
	int	i,j,l, lc, n, n1, n2;
	int	sfrv=0;		/* *scanf return value */

	char	errMsg[MAXL];

	/* initialize load data vectors and matrices to zero */
	for (j=1; j<=DoF; j++)	Fo[j] = 0.0;
	for (j=1; j<=DoF; j++)
		for (lc=1; lc <= nL; lc++)
			F_temp[lc][j] = F_mech[lc][j] = 0.0;
	for (i=1; i<=12; i++)
		for (n=1; n<=nE; n++)
			for (lc=1; lc <= nL; lc++)
				eqF_mech[lc][n][i] = eqF_temp[lc][n][i] = 0.0;

	for (i=1; i<=DoF; i++)	for (lc=1; lc<=nL; lc++) Dp[lc][i] = 0.0;

	for (i=1;i<=nE;i++)	for(j=1;j<=12;j++)	Q[i][j] = 0.0;

	for (lc = 1; lc <= nL; lc++) {		/* begin load-case loop */

	  if ( verbose ) {	/*  display the load case number */
		textColor('y','g','b','x');
		fprintf(stdout," load case %d of %d: ", lc, nL );
		fprintf(stdout,"                                            ");
		fflush(stdout);
		color(0);
		fprintf(stdout,"\n");
	  }

	  /* gravity loads applied uniformly to all frame elements ------- */
	  sfrv=fscanf(fp,"%f %f %f", &gX[lc], &gY[lc], &gZ[lc] );
	  if (sfrv != 3) sferr("gX gY gZ values in load data");

	  for (n=1; n<=nE; n++) {

		n1 = J1[n];	n2 = J2[n];

		coord_trans ( xyz, L[n], n1, n2,
			&t1, &t2, &t3, &t4, &t5, &t6, &t7, &t8, &t9, p[n] );

		eqF_mech[lc][n][1]  = d[n]*Ax[n]*L[n]*gX[lc] / 2.0;
		eqF_mech[lc][n][2]  = d[n]*Ax[n]*L[n]*gY[lc] / 2.0;
		eqF_mech[lc][n][3]  = d[n]*Ax[n]*L[n]*gZ[lc] / 2.0;

		eqF_mech[lc][n][4]  = d[n]*Ax[n]*L[n]*L[n] / 12.0 *
			( (-t4*t8+t5*t7)*gY[lc] + (-t4*t9+t6*t7)*gZ[lc] );
		eqF_mech[lc][n][5]  = d[n]*Ax[n]*L[n]*L[n] / 12.0 *
			( (-t5*t7+t4*t8)*gX[lc] + (-t5*t9+t6*t8)*gZ[lc] );
		eqF_mech[lc][n][6]  = d[n]*Ax[n]*L[n]*L[n] / 12.0 *
			( (-t6*t7+t4*t9)*gX[lc] + (-t6*t8+t5*t9)*gY[lc] );

		eqF_mech[lc][n][7]  = d[n]*Ax[n]*L[n]*gX[lc] / 2.0;
		eqF_mech[lc][n][8]  = d[n]*Ax[n]*L[n]*gY[lc] / 2.0;
		eqF_mech[lc][n][9]  = d[n]*Ax[n]*L[n]*gZ[lc] / 2.0;

		eqF_mech[lc][n][10] = d[n]*Ax[n]*L[n]*L[n] / 12.0 *
			( ( t4*t8-t5*t7)*gY[lc] + ( t4*t9-t6*t7)*gZ[lc] );
		eqF_mech[lc][n][11] = d[n]*Ax[n]*L[n]*L[n] / 12.0 *
			( ( t5*t7-t4*t8)*gX[lc] + ( t5*t9-t6*t8)*gZ[lc] );
		eqF_mech[lc][n][12] = d[n]*Ax[n]*L[n]*L[n] / 12.0 *
			( ( t6*t7-t4*t9)*gX[lc] + ( t6*t8-t5*t9)*gY[lc] );

		/* debugging ... check eqF data
		printf("n=%d ", n);
		for (l=1;l<=12;l++) {
			if (eqF_mech[lc][n][l] != 0)
			   printf(" eqF %d = %9.2e ", l, eqF_mech[lc][n][l] );
		}
		printf("\n");
		*/
	  }					/* end gravity loads */

	  /* node point loads -------------------------------------------- */
	  sfrv=fscanf(fp,"%d", &nF[lc] );
	  if (sfrv != 1) sferr("nF value in load data");
	  if ( verbose ) {
		fprintf(stdout,"  number of loaded nodes ");
	  	dots(stdout,28);	fprintf(stdout," nF = %3d\n", nF[lc]);
	  }
	  for (i=1; i <= nF[lc]; i++) {	/* ! global structural coordinates ! */
		sfrv=fscanf(fp,"%d", &j);
		if (sfrv != 1) sferr("node value in point load data");
		if ( j < 1 || j > nN ) {
		    sprintf(errMsg,"\n  error in node load data: node number out of range ... Node : %d\n   Perhaps you did not specify %d node loads \n  or perhaps the Input Data file is missing expected data.\n", j, nF[lc] );
		    errorMsg(errMsg);
		    exit(121);
		}

		for (l=5; l>=0; l--) {
			sfrv=fscanf(fp,"%lf", &F_mech[lc][6*j-l] );
			if (sfrv != 1) sferr("force value in point load data");
		}

		if ( F_mech[lc][6*j-5]==0 && F_mech[lc][6*j-4]==0 && F_mech[lc][6*j-3]==0 && F_mech[lc][6*j-2]==0 && F_mech[lc][6*j-1]==0 && F_mech[lc][6*j]==0 )
		    fprintf(stderr,"\n   Warning: All node loads applied at node %d  are zero\n", j );
	  }					/* end node point loads  */

	  /* uniformly distributed loads --------------------------------- */
	  sfrv=fscanf(fp,"%d", &nU[lc] );
	  if (sfrv != 1) sferr("nU value in uniform load data");
	  if ( verbose ) {
		fprintf(stdout,"  number of uniformly distributed loads ");
	  	dots(stdout,13);	fprintf(stdout," nU = %3d\n", nU[lc]);
	  }
	  if ( nU[lc] < 0 || nU[lc] > nE ) {
		fprintf(stderr,"  number of uniformly distributed loads ");
	  	dots(stderr,13);
	  	fprintf(stderr," nU = %3d\n", nU[lc]);
		sprintf(errMsg,"\n  error: valid ranges for nU is 0 ... %d \n", nE );
		errorMsg(errMsg);
		exit(131);
	  }
	  for (i=1; i <= nU[lc]; i++) {	/* ! local element coordinates ! */
		sfrv=fscanf(fp,"%d", &n );
	  	if (sfrv != 1) sferr("frame element number in uniform load data");
		if ( n < 1 || n > nE ) {
		    sprintf(errMsg,"\n  error in uniform distributed loads: element number %d is out of range\n",n);
		    errorMsg(errMsg); 
		    exit(132);
		}
		U[lc][i][1] = (double) n;
		for (l=2; l<=4; l++) {
			sfrv=fscanf(fp,"%f", &U[lc][i][l] );
	  		if (sfrv != 1) sferr("load value in uniform load data");
		}

		if ( U[lc][i][2]==0 && U[lc][i][3]==0 && U[lc][i][4]==0 )
		    fprintf(stderr,"\n   Warning: All distributed loads applied to frame element %d  are zero\n", n );

		Nx1 = Nx2 = U[lc][i][2]*Le[n] / 2.0;
		Vy1 = Vy2 = U[lc][i][3]*Le[n] / 2.0;
		Vz1 = Vz2 = U[lc][i][4]*Le[n] / 2.0;
		Mx1 = Mx2 = 0.0;
		My1 = -U[lc][i][4]*Le[n]*Le[n] / 12.0;	My2 = -My1;
		Mz1 =  U[lc][i][3]*Le[n]*Le[n] / 12.0;	Mz2 = -Mz1;

		/* debugging ... check end force values
		 * printf("n=%d Vy=%9.2e Vz=%9.2e My=%9.2e Mz=%9.2e\n",
		 *				n, Vy1,Vz1, My1,Mz1 );
		 */

		n1 = J1[n];	n2 = J2[n];

		coord_trans ( xyz, L[n], n1, n2,
			&t1, &t2, &t3, &t4, &t5, &t6, &t7, &t8, &t9, p[n] );

		/* debugging ... check coordinate transform coefficients
		printf("t1=%5.2f t2=%5.2f t3=%5.2f \n", t1, t2, t3 );
		printf("t4=%5.2f t5=%5.2f t6=%5.2f \n", t4, t5, t6 );
		printf("t7=%5.2f t8=%5.2f t9=%5.2f \n", t7, t8, t9 );
		*/

		/* {F} = [T]'{Q} */
		eqF_mech[lc][n][1]  += ( Nx1*t1 + Vy1*t4 + Vz1*t7 );
		eqF_mech[lc][n][2]  += ( Nx1*t2 + Vy1*t5 + Vz1*t8 );
		eqF_mech[lc][n][3]  += ( Nx1*t3 + Vy1*t6 + Vz1*t9 );
		eqF_mech[lc][n][4]  += ( Mx1*t1 + My1*t4 + Mz1*t7 );
		eqF_mech[lc][n][5]  += ( Mx1*t2 + My1*t5 + Mz1*t8 );
		eqF_mech[lc][n][6]  += ( Mx1*t3 + My1*t6 + Mz1*t9 );

		eqF_mech[lc][n][7]  += ( Nx2*t1 + Vy2*t4 + Vz2*t7 );
		eqF_mech[lc][n][8]  += ( Nx2*t2 + Vy2*t5 + Vz2*t8 );
		eqF_mech[lc][n][9]  += ( Nx2*t3 + Vy2*t6 + Vz2*t9 );
		eqF_mech[lc][n][10] += ( Mx2*t1 + My2*t4 + Mz2*t7 );
		eqF_mech[lc][n][11] += ( Mx2*t2 + My2*t5 + Mz2*t8 );
		eqF_mech[lc][n][12] += ( Mx2*t3 + My2*t6 + Mz2*t9 );

		/* debugging ... check eqF values
		printf("n=%d ", n);
		for (l=1;l<=12;l++) {
			if (eqF_mech[lc][n][l] != 0)
			   printf(" eqF %d = %9.2e ", l, eqF_mech[lc][n][l] );
		}
		printf("\n");
		*/ 
	  }				/* end uniformly distributed loads */

	  /* trapezoidally distributed loads ----------------------------- */
	  sfrv=fscanf(fp,"%d", &nW[lc] );
	  if (sfrv != 1) sferr("nW value in load data");
	  if ( verbose ) {
		fprintf(stdout,"  number of trapezoidally distributed loads ");
	  	dots(stdout,9);	fprintf(stdout," nW = %3d\n", nW[lc]);
	  }
	  if ( nW[lc] < 0 || nW[lc] > 10*nE ) {
		sprintf(errMsg,"\n  error: valid ranges for nW is 0 ... %d \n", 10*nE );
		errorMsg(errMsg);
		exit(140);
	  }
	  for (i=1; i <= nW[lc]; i++) {	/* ! local element coordinates ! */
		sfrv=fscanf(fp,"%d", &n );
	  	if (sfrv != 1) sferr("frame element number in trapezoidal load data");
		if ( n < 1 || n > nE ) {
		    sprintf(errMsg,"\n  error in trapezoidally-distributed loads: element number %d is out of range\n",n);
		    errorMsg(errMsg);
		    exit(141);
		}
		W[lc][i][1] = (double) n;
		for (l=2; l<=13; l++) {
			sfrv=fscanf(fp,"%f", &W[lc][i][l] );
			if (sfrv != 1) sferr("value in trapezoidal load data");
		}

		Ln = L[n];

		/* error checking */

		if ( W[lc][i][ 4]==0 && W[lc][i][ 5]==0 &&
		     W[lc][i][ 8]==0 && W[lc][i][ 9]==0 &&
		     W[lc][i][12]==0 && W[lc][i][13]==0 ) {
		  fprintf(stderr,"\n   Warning: All trapezoidal loads applied to frame element %d  are zero\n", n );
		  fprintf(stderr,"     load case: %d , element %d , load %d\n ", lc, n, i );
		}

		if ( W[lc][i][ 2] < 0 ) {
		  sprintf(errMsg,"\n   error in x-axis trapezoidal loads, load case: %d , element %d , load %d\n  starting location = %f < 0\n",
		  lc, n, i , W[lc][i][2]);
		  errorMsg(errMsg);
		  exit(142);
		}
		if ( W[lc][i][ 2] > W[lc][i][3] ) {
		  sprintf(errMsg,"\n   error in x-axis trapezoidal loads, load case: %d , element %d , load %d\n  starting location = %f > ending location = %f \n", 
		  lc, n, i , W[lc][i][2], W[lc][i][3] );
		  errorMsg(errMsg);
		  exit(143);
		}
		if ( W[lc][i][ 3] > Ln ) {
		  sprintf(errMsg,"\n   error in x-axis trapezoidal loads, load case: %d , element %d , load %d\n ending location = %f > L (%f) \n",
		  lc, n, i, W[lc][i][3], Ln );
		  errorMsg(errMsg);
		  exit(144);
		}
		if ( W[lc][i][ 6] < 0 ) {
		  sprintf(errMsg,"\n   error in y-axis trapezoidal loads, load case: %d , element %d , load %d\n starting location = %f < 0\n",
		  lc, n, i, W[lc][i][6]);
		  errorMsg(errMsg);
		  exit(142);
		}
		if ( W[lc][i][ 6] > W[lc][i][7] ) {
		  sprintf(errMsg,"\n   error in y-axis trapezoidal loads, load case: %d , element %d , load %d\n starting location = %f > ending location = %f \n",
		  lc, n, i, W[lc][i][6], W[lc][i][7] );
		  errorMsg(errMsg);
		  exit(143);
		}
		if ( W[lc][i][ 7] > Ln ) {
		  sprintf(errMsg,"\n   error in y-axis trapezoidal loads, load case: %d , element %d , load %d\n ending location = %f > L (%f) \n",
		  lc, n, i, W[lc][i][7],Ln );
		  errorMsg(errMsg);
		  exit(144);
		}
		if ( W[lc][i][10] < 0 ) {
		  sprintf(errMsg,"\n   error in z-axis trapezoidal loads, load case: %d , element %d , load %d\n starting location = %f < 0\n",
		  lc, n, i, W[lc][i][10]);
		  errorMsg(errMsg);
		  exit(142);
		}
		if ( W[lc][i][10] > W[lc][i][11] ) {
		  sprintf(errMsg,"\n   error in z-axis trapezoidal loads, load case: %d , element %d , load %d\n starting location = %f > ending location = %f \n",
		  lc, n, i, W[lc][i][10], W[lc][i][11] );
		  errorMsg(errMsg);
		  exit(143);
		}
		if ( W[lc][i][11] > Ln ) {
		  sprintf(errMsg,"\n   error in z-axis trapezoidal loads, load case: %d , element %d , load %d\n ending location = %f > L (%f) \n",lc, n, i, W[lc][i][11], Ln );
		  errorMsg(errMsg);
		  exit(144);
		}

		if ( shear ) {
			Ksy = (12.0*E[n]*Iz[n]) / (G[n]*Asy[n]*Le[n]*Le[n]);
			Ksz = (12.0*E[n]*Iy[n]) / (G[n]*Asz[n]*Le[n]*Le[n]);
		} else	Ksy = Ksz = 0.0;

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

		/* debugging ... check internal force values
		printf("n=%d\n Nx1=%9.3f\n Nx2=%9.3f\n Vy1=%9.3f\n Vy2=%9.3f\n Vz1=%9.3f\n Vz2=%9.3f\n My1=%9.3f\n My2=%9.3f\n Mz1=%9.3f\n Mz2=%9.3f\n",
				n, Nx1,Nx2,Vy1,Vy2,Vz1,Vz2, My1,My2,Mz1,Mz2 );
		*/

		n1 = J1[n];	n2 = J2[n];

		coord_trans ( xyz, Ln, n1, n2,
			&t1, &t2, &t3, &t4, &t5, &t6, &t7, &t8, &t9, p[n] );

		/* debugging ... check coordinate transformation coefficients
		printf("t1=%5.2f t2=%5.2f t3=%5.2f \n", t1, t2, t3 );
		printf("t4=%5.2f t5=%5.2f t6=%5.2f \n", t4, t5, t6 );
		printf("t7=%5.2f t8=%5.2f t9=%5.2f \n", t7, t8, t9 );
		*/

		/* {F} = [T]'{Q} */
		eqF_mech[lc][n][1]  += ( Nx1*t1 + Vy1*t4 + Vz1*t7 );
		eqF_mech[lc][n][2]  += ( Nx1*t2 + Vy1*t5 + Vz1*t8 );
		eqF_mech[lc][n][3]  += ( Nx1*t3 + Vy1*t6 + Vz1*t9 );
		eqF_mech[lc][n][4]  += ( Mx1*t1 + My1*t4 + Mz1*t7 );
		eqF_mech[lc][n][5]  += ( Mx1*t2 + My1*t5 + Mz1*t8 );
		eqF_mech[lc][n][6]  += ( Mx1*t3 + My1*t6 + Mz1*t9 );

		eqF_mech[lc][n][7]  += ( Nx2*t1 + Vy2*t4 + Vz2*t7 );
		eqF_mech[lc][n][8]  += ( Nx2*t2 + Vy2*t5 + Vz2*t8 );
		eqF_mech[lc][n][9]  += ( Nx2*t3 + Vy2*t6 + Vz2*t9 );
		eqF_mech[lc][n][10] += ( Mx2*t1 + My2*t4 + Mz2*t7 );
		eqF_mech[lc][n][11] += ( Mx2*t2 + My2*t5 + Mz2*t8 );
		eqF_mech[lc][n][12] += ( Mx2*t3 + My2*t6 + Mz2*t9 );

		/* debugging ... check eqF data
		for (l=1;l<=13;l++) printf(" %9.2e ", W[lc][i][l] );
		printf("\n"); 
		printf("n=%d ", n);
		for (l=1;l<=12;l++) {
			if (eqF_mech[lc][n][l] != 0)
			   printf(" eqF %d = %9.3f ", l, eqF_mech[lc][n][l] );
		}
		printf("\n");
		*/
	  }			/* end trapezoidally distributed loads */

	  /* internal element point loads -------------------------------- */
	  sfrv=fscanf(fp,"%d", &nP[lc] );
	  if (sfrv != 1) sferr("nP value load data");
	  if ( verbose ) {
	  	fprintf(stdout,"  number of concentrated frame element point loads ");
	  	dots(stdout,2);	fprintf(stdout," nP = %3d\n", nP[lc]);
	  }
	  if ( nP[lc] < 0 || nP[lc] > 10*nE ) {
	  	fprintf(stderr,"  number of concentrated frame element point loads ");
	  	dots(stderr,3);
	  	fprintf(stderr," nP = %3d\n", nP[lc]);
		sprintf(errMsg,"\n  error: valid ranges for nP is 0 ... %d \n", 10*nE );
		errorMsg(errMsg);
		exit(150);
	  }
	  for (i=1; i <= nP[lc]; i++) {	/* ! local element coordinates ! */
		sfrv=fscanf(fp,"%d", &n );
		if (sfrv != 1) sferr("frame element number value point load data");
		if ( n < 1 || n > nE ) {
		    sprintf(errMsg,"\n   error in internal point loads: frame element number %d is out of range\n",n);
		    errorMsg(errMsg);
		    exit(151);
		}
		P[lc][i][1] = (double) n;
		for (l=2; l<=5; l++) { 
			sfrv=fscanf(fp,"%f", &P[lc][i][l] );
			if (sfrv != 1) sferr("value in point load data");
		}
		a = P[lc][i][5];	b = L[n] - a;

		if ( a < 0 || L[n] < a || b < 0 || L[n] < b ) {
		    sprintf(errMsg,"\n  error in point load data: Point load coord. out of range\n   Frame element number: %d  L: %lf  load coord.: %lf\n",
		    n, L[n], P[lc][i][5] );
		    errorMsg(errMsg);
		    exit(152);
		}

		if ( shear ) {
			Ksy = (12.0*E[n]*Iz[n]) / (G[n]*Asy[n]*Le[n]*Le[n]);
			Ksz = (12.0*E[n]*Iy[n]) / (G[n]*Asz[n]*Le[n]*Le[n]);
		} else	Ksy = Ksz = 0.0;

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

		n1 = J1[n];	n2 = J2[n];

		coord_trans ( xyz, Ln, n1, n2,
			&t1, &t2, &t3, &t4, &t5, &t6, &t7, &t8, &t9, p[n] );

		/* {F} = [T]'{Q} */
		eqF_mech[lc][n][1]  += ( Nx1*t1 + Vy1*t4 + Vz1*t7 );
		eqF_mech[lc][n][2]  += ( Nx1*t2 + Vy1*t5 + Vz1*t8 );
		eqF_mech[lc][n][3]  += ( Nx1*t3 + Vy1*t6 + Vz1*t9 );
		eqF_mech[lc][n][4]  += ( Mx1*t1 + My1*t4 + Mz1*t7 );
		eqF_mech[lc][n][5]  += ( Mx1*t2 + My1*t5 + Mz1*t8 );
		eqF_mech[lc][n][6]  += ( Mx1*t3 + My1*t6 + Mz1*t9 );

		eqF_mech[lc][n][7]  += ( Nx2*t1 + Vy2*t4 + Vz2*t7 );
		eqF_mech[lc][n][8]  += ( Nx2*t2 + Vy2*t5 + Vz2*t8 );
		eqF_mech[lc][n][9]  += ( Nx2*t3 + Vy2*t6 + Vz2*t9 );
		eqF_mech[lc][n][10] += ( Mx2*t1 + My2*t4 + Mz2*t7 );
		eqF_mech[lc][n][11] += ( Mx2*t2 + My2*t5 + Mz2*t8 );
		eqF_mech[lc][n][12] += ( Mx2*t3 + My2*t6 + Mz2*t9 );
	  }					/* end element point loads */

	  /* thermal loads ----------------------------------------------- */
	  sfrv=fscanf(fp,"%d", &nT[lc] );
	  if (sfrv != 1) sferr("nT value in load data");
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
		exit(160);
	  }
	  for (i=1; i <= nT[lc]; i++) {	/* ! local element coordinates ! */
		sfrv=fscanf(fp,"%d", &n );
		if (sfrv != 1) sferr("frame element number in temperature load data");
		if ( n < 1 || n > nE ) {
		    sprintf(errMsg,"\n  error in temperature loads: frame element number %d is out of range\n",n);
		    errorMsg(errMsg);
		    exit(161);
		}
		T[lc][i][1] = (double) n;
		for (l=2; l<=8; l++) {
			sfrv=fscanf(fp,"%f", &T[lc][i][l] );
			if (sfrv != 1) sferr("value in temperature load data");
		}
		a  = T[lc][i][2];
		hy = T[lc][i][3];
		hz = T[lc][i][4];

		if ( hy < 0 || hz < 0 ) {
		    sprintf(errMsg,"\n  error in thermal load data: section dimension < 0\n   Frame element number: %d  hy: %f  hz: %f\n", n,hy,hz);
		    errorMsg(errMsg);
		    exit(162);
		}

		Nx2 = a*(1.0/4.0)*( T[lc][i][5]+T[lc][i][6]+T[lc][i][7]+T[lc][i][8])*E[n]*Ax[n];
		Nx1 = -Nx2;
		Vy1 = Vy2 = Vz1 = Vz2 = 0.0;
		Mx1 = Mx2 = 0.0;
		My1 =  (a/hz)*(T[lc][i][8]-T[lc][i][7])*E[n]*Iy[n];
		My2 = -My1;
		Mz1 =  (a/hy)*(T[lc][i][5]-T[lc][i][6])*E[n]*Iz[n];
		Mz2 = -Mz1;

		n1 = J1[n];	n2 = J2[n];

		coord_trans ( xyz, L[n], n1, n2,
			&t1, &t2, &t3, &t4, &t5, &t6, &t7, &t8, &t9, p[n] );

		/* {F} = [T]'{Q} */
		eqF_temp[lc][n][1]  += ( Nx1*t1 + Vy1*t4 + Vz1*t7 );
		eqF_temp[lc][n][2]  += ( Nx1*t2 + Vy1*t5 + Vz1*t8 );
		eqF_temp[lc][n][3]  += ( Nx1*t3 + Vy1*t6 + Vz1*t9 );
		eqF_temp[lc][n][4]  += ( Mx1*t1 + My1*t4 + Mz1*t7 );
		eqF_temp[lc][n][5]  += ( Mx1*t2 + My1*t5 + Mz1*t8 );
		eqF_temp[lc][n][6]  += ( Mx1*t3 + My1*t6 + Mz1*t9 );

		eqF_temp[lc][n][7]  += ( Nx2*t1 + Vy2*t4 + Vz2*t7 );
		eqF_temp[lc][n][8]  += ( Nx2*t2 + Vy2*t5 + Vz2*t8 );
		eqF_temp[lc][n][9]  += ( Nx2*t3 + Vy2*t6 + Vz2*t9 );
		eqF_temp[lc][n][10] += ( Mx2*t1 + My2*t4 + Mz2*t7 );
		eqF_temp[lc][n][11] += ( Mx2*t2 + My2*t5 + Mz2*t8 );
		eqF_temp[lc][n][12] += ( Mx2*t3 + My2*t6 + Mz2*t9 );
	  }				/* end thermal loads	*/

	  /* debugging ...  check eqF's prior to asembly 
	  for (n=1; n<=nE; n++) {	
		printf("n=%d ", n);
		for (l=1;l<=12;l++) {
			if (eqF_mech[lc][n][l] != 0)
			   printf(" eqF %d = %9.2e ", l, eqF_mech[lc][n][l] );
		}
		printf("\n"); 
	  }
	  */

	  // assemble all element equivalent loads into 
	  // separate load vectors for mechanical and thermal loading
	  for (n=1; n<=nE; n++) {
	     n1 = J1[n];	n2 = J2[n];
	     for (i=1; i<= 6; i++) F_mech[lc][6*n1- 6+i] += eqF_mech[lc][n][i];
	     for (i=7; i<=12; i++) F_mech[lc][6*n2-12+i] += eqF_mech[lc][n][i];
	     for (i=1; i<= 6; i++) F_temp[lc][6*n1- 6+i] += eqF_temp[lc][n][i];
	     for (i=7; i<=12; i++) F_temp[lc][6*n2-12+i] += eqF_temp[lc][n][i];
	  }

	  /* prescribed displacements ------------------------------------ */
	  sfrv=fscanf(fp,"%d", &nD[lc] );
	  if (sfrv != 1) sferr("nD value in load data");
	  if ( verbose ) {
	  	fprintf(stdout,"  number of prescribed displacements ");
	  	dots(stdout,16);	fprintf(stdout," nD = %3d\n", nD[lc] );
	  }
	  for (i=1; i <= nD[lc]; i++) {
		sfrv=fscanf(fp,"%d", &j);
		if (sfrv != 1) sferr("node number value in prescribed displacement data");
		for (l=5; l >=0; l--) {
			sfrv=fscanf(fp,"%f", &Dp[lc][6*j-l] );
			if (sfrv != 1) sferr("prescribed displacement value");
			if ( r[6*j-l] == 0 && Dp[lc][6*j-l] != 0.0 ) {
			    sprintf(errMsg," Initial displacements can be prescribed only at restrained coordinates\n  node: %d  dof: %d  r: %d\n",
			    j, 6-l, r[6*j-l] );
			    errorMsg(errMsg);
			    exit(171);
			}
		}
	  }

	}					/* end load-case loop */

	return;
}


/*
 * READ_MASS_DATA  -  read element densities and extra inertial mass data	16aug01 
 */
void read_mass_data (
		FILE *fp,
		char *OUT_file, 
		int nN, int nE, int *nI, int *nX, 
		float *d, float *EMs,
		float *NMs, float *NMx, float *NMy, float *NMz,
		double *L, float *Ax,
		double *total_mass, double *struct_mass,
		int *nM, int *Mmethod, int modal_flag, 
		int *lump, int lump_flag, 
		double *tol, double tol_flag, double *shift, double shift_flag,
		double *exagg_modal, 
		char modepath[],
		int anim[], float *pan, float pan_flag,
		int verbose, int debug
){
/*	double	ms = 0.0; */
	int	i,j, jnt, m, b, nA;
	int	full_len=0, len=0;
	int	sfrv=0;		/* *scanf return value	*/

	char	base_file[96] = "EMPTY_BASE";
	char	mode_file[96] = "EMPTY_MODE";
	char	errMsg[MAXL];

	*total_mass = *struct_mass = 0.0;

	sfrv=fscanf ( fp, "%d", nM );
	if (sfrv != 1) sferr("nM value in mass data");

	if ( verbose ) {
		fprintf(stdout," number of dynamic modes ");
		dots(stdout,28);	fprintf(stdout," nM = %3d\n", *nM);
	}

	if ( *nM < 1 || sfrv != 1 ) {
		*nM = 0;
		return;
	}

	sfrv=fscanf( fp, "%d", Mmethod );
	if (sfrv != 1) sferr("Mmethod value in mass data");
	if ( modal_flag != -1 )	*Mmethod = modal_flag;

	if ( verbose ) {
		fprintf(stdout," modal analysis method ");
		dots(stdout,30);	fprintf(stdout," %3d ",*Mmethod);
		if ( *Mmethod == 1 ) fprintf(stdout," (Subspace-Jacobi)\n");
		if ( *Mmethod == 2 ) fprintf(stdout," (Stodola)\n");
	}


#ifdef MASSDATA_DEBUG
	FILE	*mf;				// mass data file
	mf = fopen("MassData.txt","w");		// open mass data file
	if ((mf = fopen ("MassData.txt", "w")) == NULL) {
	  errorMsg("\n  error: cannot open mass data debug file: 'MassData.txt' \n");
	  exit(29);
	}
	fprintf(mf,"%% structural mass data \n");
	fprintf(mf,"%% element\tAx\t\tlength\t\tdensity\t\tmass \n");
#endif

	sfrv=fscanf( fp, "%d", lump );
	if (sfrv != 1) sferr("lump value in mass data");
	sfrv=fscanf( fp, "%lf", tol );
	if (sfrv != 1) sferr("tol value in mass data");
	sfrv=fscanf( fp, "%lf", shift );
	if (sfrv != 1) sferr("shift value in mass data");
	sfrv=fscanf( fp, "%lf", exagg_modal );
	if (sfrv != 1) sferr("exagg_modal value in mass data");

	if (  lump_flag != -1   )	*lump = lump_flag;
	if (  tol_flag  != -1.0 )	*tol  = tol_flag;
	if ( shift_flag != -1.0 )	*shift = shift_flag;


	/* number of nodes with extra inertias */
	sfrv=fscanf(fp,"%d", nI );
	if (sfrv != 1) sferr("nI value in mass data");
	if ( verbose ) {
		fprintf(stdout," number of nodes with extra lumped inertia ");
		dots(stdout,10);	fprintf(stdout," nI = %3d\n",*nI);
	}
	for (j=1; j <= *nI; j++) {
		sfrv=fscanf(fp, "%d", &jnt );
		if (sfrv != 1) sferr("node value in extra node mass data");
		if ( jnt < 1 || jnt > nN ) {
	    		sprintf(errMsg,"\n  error in node mass data: node number out of range    Node : %d  \n   Perhaps you did not specify %d extra masses \n   or perhaps the Input Data file is missing expected data.\n",
			jnt, *nI );
			errorMsg(errMsg);
	    		exit(86);
		}
		sfrv=fscanf(fp, "%f %f %f %f",
			&NMs[jnt], &NMx[jnt], &NMy[jnt], &NMz[jnt] );
		if (sfrv != 4) sferr("node inertia in extra mass data");
		*total_mass += NMs[jnt];

		if ( NMs[jnt]==0 && NMx[jnt]==0 && NMy[jnt]==0 && NMz[jnt]==0 )
	    	fprintf(stderr,"\n  Warning: All extra node inertia at node %d  are zero\n", jnt );
	}

	/* number of frame elements with extra beam mass */
	sfrv=fscanf(fp,"%d", nX );
	if (sfrv != 1) sferr("nX value in mass data");
	if ( verbose ) {
		fprintf(stdout," number of frame elements with extra mass ");
		dots(stdout,11);	fprintf(stdout," nX = %3d\n",*nX);
		if (sfrv != 1) sferr("element value in extra element mass data");
	}
	for (m=1; m <= *nX; m++) {
		sfrv=fscanf(fp, "%d", &b );
		if (sfrv != 1) sferr("element number in extra element mass data");
		if ( b < 1 || b > nE ) {
			sprintf(errMsg,"\n  error in element mass data: element number out of range   Element: %d  \n   Perhaps you did not specify %d extra masses \n   or perhaps the Input Data file is missing expected data.\n", 
			b, *nX ); 
			errorMsg(errMsg);
	    		exit(87);
		}
		sfrv=fscanf(fp, "%f", &EMs[b] );
		if (sfrv != 1) sferr("extra element mass value in mass data");
	}


	/* calculate the total mass and the structural mass */
	for (b=1; b <= nE; b++) {
		*total_mass  += d[b]*Ax[b]*L[b] + EMs[b];
		*struct_mass += d[b]*Ax[b]*L[b];
#ifdef MASSDATA_DEBUG
		fprintf(mf," %4d\t\t%12.5e\t%12.5e\t%12.5e\t%12.5e \n",
		 b, Ax[b], L[b], d[b], d[b]*Ax[b]*L[b] );
#endif
	}

#ifdef MASSDATA_DEBUG
	fclose(mf);
#endif

	for (m=1;m<=nE;m++) {			/* check inertia data	*/
	    if ( d[m] < 0.0 || EMs[m] < 0.0 || d[m]+EMs[m] <= 0.0 ) {
		sprintf(errMsg,"\n  error: Non-positive mass or density\n  d[%d]= %f  EMs[%d]= %f\n",m,d[m],m,EMs[m]);
		errorMsg(errMsg);
		exit(88);
	    }
	}
/*	for (m=1;m<=nE;m++) ms += EMs[m]; // consistent mass doesn't agree  */
/*	if ( ms > 0.0 )	    *lump = 1;    // with concentrated masses, EMs  */

	if ( verbose ) {
		fprintf(stdout," structural mass ");
		dots(stdout,36);	fprintf(stdout,"  %12.4e\n",*struct_mass);
		fprintf(stdout," total mass ");
		dots(stdout,41);	fprintf(stdout,"  %12.4e\n",*total_mass);
	}
	sfrv=fscanf ( fp, "%d", &nA );
	if (sfrv != 1) sferr("nA value in mode animation data");
	if ( verbose ) {
		fprintf(stdout," number of modes to be animated ");
		dots(stdout,21);	fprintf(stdout," nA = %3d\n",nA);
	}
	if (nA > 20)
	  fprintf(stderr," nA = %d, only 100 or fewer modes may be animated\n", nA );
	for ( m = 0; m < 20; m++ )	anim[m] = 0;
	for ( m = 1; m <= nA; m++ ) {
		sfrv=fscanf ( fp, "%d", &anim[m] );
		if (sfrv != 1) sferr("mode number in mode animation data");
	}

	sfrv=fscanf ( fp, "%f", pan );
	if (sfrv != 1) sferr("pan value in mode animation data");
	if ( pan_flag != -1.0 )	*pan = pan_flag;

	if ( verbose ) {
		fprintf(stdout," pan rate ");
		dots(stdout,43); fprintf(stdout," %8.3f\n", *pan);
	}

	strcpy(base_file,OUT_file);	
	while ( base_file[len++] != '\0' )
		/* the length of the base_file */ ;
	full_len = len;

	while ( base_file[len--] != '.' && len > 0 )
		/* find the last '.' in base_file */ ;
	if ( len == 0 )	len = full_len;
	base_file[++len] = '\0';	/* end base_file at the last '.' */

	while ( base_file[len] != '/' && base_file[len] != '\\' && len > 0 )
		len--;	/* find the last '/' or '\' in base_file */ 
	i = 0;
	while ( base_file[len] != '\0' )
		mode_file[i++] = base_file[len++];
	mode_file[i] = '\0';
	strcat(mode_file,"-m");
	output_path(mode_file,modepath,FRAME3DD_PATHMAX,NULL);

	return;
}


/* 
 * READ_CONDENSE   -  read matrix condensation information 	        30aug01
 */
void read_condensation_data (
		FILE *fp,
		int nN, int nM,
		int *nC, int *Cdof,
		int *Cmethod, int condense_flag, int *c, int *m, int verbose
){
	int	i,j,k,  **cm;
	int	sfrv=0;		/* *scanf return value */
	char	errMsg[MAXL];

	*Cmethod = *nC = *Cdof = 0;

	if ( (sfrv=fscanf ( fp, "%d", Cmethod )) != 1 )   {
		*Cmethod = *nC = *Cdof = 0;
		if ( verbose )
			fprintf(stdout," missing matrix condensation data \n");
		return;
	}

	if ( condense_flag != -1 )	*Cmethod = condense_flag;

	if ( *Cmethod <= 0 )  {
		if ( verbose )
			fprintf(stdout," Cmethod = %d : no matrix condensation \n", *Cmethod );
		*Cmethod = *nC = *Cdof = 0;
		return;
	}

	if ( *Cmethod > 3 ) *Cmethod = 1;	/* default */
	if ( verbose ) {
		fprintf(stdout," condensation method ");
		dots(stdout,32);	fprintf(stdout," %d ", *Cmethod );
		if ( *Cmethod == 1 )	fprintf(stdout," (static only) \n");
		if ( *Cmethod == 2 )	fprintf(stdout," (Guyan) \n");
		if ( *Cmethod == 3 )	fprintf(stdout," (dynamic) \n");
	}

	if ( (sfrv=fscanf ( fp, "%d", nC )) != 1 )  {
		*Cmethod = *nC = *Cdof = 0;
		if ( verbose )
			fprintf(stdout," missing matrix condensation data \n");
		return;
	}

	if ( verbose ) {
		fprintf(stdout," number of nodes with condensed DoF's ");
		dots(stdout,15);	fprintf(stdout," nC = %3d\n", *nC );
	}

	if ( (*nC) > nN ) {
	  sprintf(errMsg,"\n  error in matrix condensation data: \n error: nC > nN ... nC=%d; nN=%d;\n The number of nodes with condensed DoF's may not exceed the total number of nodes.\n", 
	  *nC, nN );
	  errorMsg(errMsg);
	  exit(90);
	}

	cm = imatrix( 1, *nC, 1,7 );

	for ( i=1; i <= *nC; i++) {
	 sfrv=fscanf( fp, "%d %d %d %d %d %d %d",
	 &cm[i][1],
	 &cm[i][2], &cm[i][3], &cm[i][4], &cm[i][5], &cm[i][6], &cm[i][7]);
	 if (sfrv != 7) sferr("DoF numbers in condensation data");
	 if ( cm[i][1] < 1 || cm[i][1] > nN ) {		/* error check */
	  sprintf(errMsg,"\n  error in matrix condensation data: \n  condensed node number out of range\n  cj[%d] = %d  ... nN = %d  \n", i, cm[i][1], nN );
	  errorMsg(errMsg);
	  exit(91);
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
	 sfrv=fscanf( fp, "%d", &m[i] );
	 if (sfrv != 1 && *Cmethod == 3) {
		sferr("mode number in condensation data");
		sprintf(errMsg,"condensed mode %d = %d", i, m[i] );
		errorMsg(errMsg);
	 }
	 if ( (m[i] < 0 || m[i] > nM) && *Cmethod == 3 ) {
	  sprintf(errMsg,"\n  error in matrix condensation data: \n  m[%d] = %d \n The condensed mode number must be between   1 and %d (modes).\n", 
	  i, m[i], nM );
	  errorMsg(errMsg);
	  exit(92);
	 }
	}

	free_imatrix(cm,1, *nC, 1,7);
	return;
}


/* 
 * WRITE_INPUT_DATA  -  save input data					07nov02
 */
void write_input_data (
	FILE *fp,
	char *title, int nN, int nE, int nL,
	int *nD, int nR,
	int *nF, int *nU, int *nW, int *nP, int *nT,
	vec3 *xyz, float *r,
	int *J1, int *J2,
	float *Ax, float *Asy, float *Asz, float *Jx, float *Iy, float *Iz,
	float *E, float *G, float *p, float *d, 
	float *gX, float *gY, float *gZ, 
	double **Ft, double **Fm, float **Dp,
	int *R,
	float ***U, float ***W, float ***P, float ***T,
	int shear, int anlyz, int geom
){
	int	i,j,n, lc;
	time_t  now;		/* modern time variable type	*/

	(void) time(&now);

	for (i=1; i<=80; i++)	fprintf(fp,"_");
  	fprintf(fp,"\nFrame3DD version: %s ", VERSION );
	fprintf(fp,"              http://frame3dd.sf.net/\n");
	fprintf(fp,"GPL Copyright (C) 1992-2015, Henri P. Gavin \n");
	fprintf(fp,"Frame3DD is distributed in the hope that it will be useful");
	fprintf(fp," but with no warranty.\n");
	fprintf(fp,"For details see the GNU Public Licence:");
	fprintf(fp," http://www.fsf.org/copyleft/gpl.html\n");
	for (i=1; i<=80; i++)	fprintf(fp,"_"); fprintf(fp,"\n\n");
	fprintf(fp,"%s\n",title);
	fprintf(fp, "%s", ctime(&now) );

	for (i=1; i<=80; i++)	fprintf(fp,"_");	fprintf(fp,"\n");

	fprintf(fp,"In 2D problems the Y-axis is vertical.  ");
#if Zvert
	fprintf(fp,"In 3D problems the Z-axis is vertical.\n");
#else
	fprintf(fp,"In 3D problems the Y-axis is vertical.\n");
#endif

	for (i=1; i<=80; i++)	fprintf(fp,"_");	fprintf(fp,"\n");

	fprintf(fp,"%5d NODES          ", nN ); 
	fprintf(fp,"%5d FIXED NODES    ", nR );
	fprintf(fp,"%5d FRAME ELEMENTS ", nE ); 
	fprintf(fp,"%3d LOAD CASES   \n", nL );

	for (i=1; i<=80; i++)	fprintf(fp,"_"); fprintf(fp,"\n");

	fprintf(fp,"N O D E   D A T A       ");
	fprintf(fp,"                                    R E S T R A I N T S\n");
	fprintf(fp,"  Node       X              Y              Z");
	fprintf(fp,"         radius  Fx Fy Fz Mx My Mz\n");
	for (i=1; i<=nN; i++) {
	 j = 6*(i-1);
	 fprintf(fp,"%5d %14.6f %14.6f %14.6f %8.3f  %2d %2d %2d %2d %2d %2d\n",
		i, xyz[i].x, xyz[i].y, xyz[i].z, r[i],
			R[j+1], R[j+2], R[j+3], R[j+4], R[j+5], R[j+6] );
	}
	fprintf(fp,"F R A M E   E L E M E N T   D A T A\t\t\t\t\t(local)\n");
	fprintf(fp,"  Elmnt  J1    J2     Ax   Asy   Asz    ");
	fprintf(fp,"Jxx     Iyy     Izz       E       G roll  density\n");
	for (i=1; i<= nE; i++) {
		fprintf(fp,"%5d %5d %5d %6.1f %5.1f %5.1f",
					i, J1[i],J2[i], Ax[i], Asy[i], Asz[i] );
		fprintf(fp," %6.1f %7.1f %7.1f %8.1f %7.1f %3.0f %8.2e\n",
			Jx[i], Iy[i], Iz[i], E[i], G[i], p[i]*180.0/PI, d[i] );
	}
	if ( shear )	fprintf(fp,"  Include shear deformations.\n");
	else		fprintf(fp,"  Neglect shear deformations.\n");
	if ( geom )	fprintf(fp,"  Include geometric stiffness.\n");
	else		fprintf(fp,"  Neglect geometric stiffness.\n");

	for (lc = 1; lc <= nL; lc++) {		/* start load case loop */

	  fprintf(fp,"\nL O A D   C A S E   %d   O F   %d  ... \n\n", lc,nL);
	  fprintf(fp,"   Gravity X = ");
	  if (gX[lc] == 0) fprintf(fp," 0.0 "); else fprintf(fp," %.3f ", gX[lc]);
	  fprintf(fp,"   Gravity Y = ");
	  if (gY[lc] == 0) fprintf(fp," 0.0 "); else fprintf(fp," %.3f ", gY[lc]);
	  fprintf(fp,"   Gravity Z = ");
	  if (gZ[lc] == 0) fprintf(fp," 0.0 "); else fprintf(fp," %.3f ", gZ[lc]);
	  fprintf(fp,"\n");
	  fprintf(fp," %3d concentrated loads\n", nF[lc] );
	  fprintf(fp," %3d uniformly distributed loads\n", nU[lc]);
	  fprintf(fp," %3d trapezoidally distributed loads\n", nW[lc]);
	  fprintf(fp," %3d concentrated point loads\n", nP[lc] );
	  fprintf(fp," %3d temperature loads\n", nT[lc] );
	  fprintf(fp," %3d prescribed displacements\n", nD[lc] );
	  if ( nF[lc] > 0 || nU[lc] > 0 || nW[lc] > 0 || nP[lc] > 0 || nT[lc] > 0 ) {
	    fprintf(fp," N O D A L   L O A D S");
	    fprintf(fp,"  +  E Q U I V A L E N T   N O D A L   L O A D S  (global)\n");
	    fprintf(fp,"  Node        Fx          Fy          Fz");
	    fprintf(fp,"          Mxx         Myy         Mzz\n");
	    for (j=1; j<=nN; j++) {
		i = 6*(j-1);
		if ( Fm[lc][i+1]!=0.0 || Fm[lc][i+2]!=0.0 || Fm[lc][i+3]!=0.0 ||
		     Fm[lc][i+4]!=0.0 || Fm[lc][i+5]!=0.0 || Fm[lc][i+6]!=0.0 ) {
			fprintf(fp, " %5d", j);
			for (i=5; i>=0; i--) fprintf(fp, " %11.3f", Fm[lc][6*j-i] );
			fprintf(fp, "\n");
		}
	    }
	  }

	  if ( nU[lc] > 0 ) {
	    fprintf(fp," U N I F O R M   L O A D S");
	    fprintf(fp,"\t\t\t\t\t\t(local)\n");
	    fprintf(fp,"  Elmnt       Ux               Uy               Uz\n");
	    for (n=1; n<=nU[lc]; n++) {
		fprintf(fp, " %5d", (int) (U[lc][n][1]) );
		for (i=2; i<=4; i++) fprintf(fp, " %16.8f", U[lc][n][i] );
		fprintf(fp, "\n");
	    }
	  }

	  if ( nW[lc] > 0 ) {
	    fprintf(fp," T R A P E Z O I D A L   L O A D S");
	    fprintf(fp,"\t\t\t\t\t(local)\n");
	    fprintf(fp,"  Elmnt       x1               x2               W1               W2\n");
	    for (n=1; n<=nW[lc]; n++) {
		fprintf(fp, " %5d", (int) (W[lc][n][1]) );
		for (i=2; i<=5; i++) fprintf(fp, " %16.8f", W[lc][n][i] );
		fprintf(fp, "  (x)\n");
		fprintf(fp, " %5d", (int) (W[lc][n][1]) );
		for (i=6; i<=9; i++) fprintf(fp, " %16.8f", W[lc][n][i] );
		fprintf(fp, "  (y)\n");
		fprintf(fp, " %5d", (int) (W[lc][n][1]) );
		for (i=10; i<=13; i++) fprintf(fp, " %16.8f", W[lc][n][i] );
		fprintf(fp, "  (z)\n");
	    }
	  }

	  if ( nP[lc] > 0 ) {
	    fprintf(fp," C O N C E N T R A T E D   P O I N T   L O A D S");
	    fprintf(fp,"\t\t\t\t(local)\n");
	    fprintf(fp,"  Elmnt       Px          Py          Pz          x\n");
	    for (n=1; n<=nP[lc]; n++) {
		fprintf(fp, " %5d", (int) (P[lc][n][1]) );
		for (i=2; i<=5; i++) fprintf(fp, " %11.3f", P[lc][n][i] );
		fprintf(fp, "\n");
	    }
	  }

	  if ( nT[lc] > 0 ) {
	    fprintf(fp," T E M P E R A T U R E   C H A N G E S");
	    fprintf(fp,"\t\t\t\t\t(local)\n");
	    fprintf(fp,"  Elmnt     coef      hy        hz");
	    fprintf(fp,"        Ty+       Ty-       Tz+       Tz-\n");
	    for (n=1; n<=nT[lc]; n++) {
		fprintf(fp, " %5d", (int) (T[lc][n][1]) );
		fprintf(fp, " %9.2e", T[lc][n][2] );
		for (i=3; i<=8; i++) fprintf(fp, " %9.3f", T[lc][n][i] );
		fprintf(fp, "\n");
	    }
	  }

	  if ( nD[lc] > 0 ) {
	    fprintf(fp,"\n P R E S C R I B E D   D I S P L A C E M E N T S");
	    fprintf(fp,"                        (global)\n");
	    fprintf(fp,"  Node        Dx          Dy          Dz");
	    fprintf(fp,"          Dxx         Dyy         Dzz\n");
	    for (j=1; j<=nN; j++) {
		i = 6*(j-1);
		if ( Dp[lc][i+1]!=0.0 || Dp[lc][i+2]!=0.0 || Dp[lc][i+3]!=0.0 ||
		     Dp[lc][i+4]!=0.0 || Dp[lc][i+5]!=0.0 || Dp[lc][i+6]!=0.0 ){
			fprintf(fp, " %5d", j);
			for (i=5; i>=0; i--) fprintf(fp, " %11.3f",
							Dp[lc][6*j-i] );
			fprintf(fp, "\n");
		}
	    }
	  }

	}					/* end load case loop	*/

	if (anlyz) {
	 fprintf(fp,"\nE L A S T I C   S T I F F N E S S   A N A L Y S I S");
	 fprintf(fp,"   via  L D L'  decomposition\n\n");
	}
	else		fprintf(fp,"D A T A   C H E C K   O N L Y\n");
	fflush(fp);
	return;
}


/*
 * WRITE_STATIC_RESULTS -  save node displacements and frame element end forces
 * 09 Sep 2008 , 2015-05-15
 */
void write_static_results (
		FILE *fp,
		int nN, int nE, int nL, int lc, int DoF,
		int *J1, int *J2,
		double *F, double *D, double *R, int *r, double **Q,
		double err, int ok, int axial_sign
){
	double	disp;
	int	i,j,n;

	if ( ok < 0 ) {
	 fprintf(fp,"  * The Stiffness Matrix is not positive-definite *\n");
	 fprintf(fp,"    Check that all six rigid-body translations are restrained\n");
	 fprintf(fp,"    If geometric stiffness is included, reduce the loads.\n");
/*	 return; */
	}

	fprintf(fp,"\nL O A D   C A S E   %d   O F   %d  ... \n\n", lc, nL);

	fprintf(fp,"N O D E   D I S P L A C E M E N T S  ");
	fprintf(fp,"\t\t\t\t\t(global)\n");
	fprintf(fp,"  Node    X-dsp       Y-dsp       Z-dsp");
	fprintf(fp,"       X-rot       Y-rot       Z-rot\n");
	for (j=1; j<= nN; j++) {
	    disp = 0.0;
	    for ( i=5; i>=0; i-- ) disp += fabs( D[6*j-i] );
	    if ( disp > 0.0 ) {
		fprintf(fp," %5d", j);
		for ( i=5; i>=0; i-- ) {
			if ( fabs(D[6*j-i]) < 1.e-8 )
				fprintf (fp, "    0.0     ");
			else    fprintf (fp, " %11.6f",  D[6*j-i] );
		}
		fprintf(fp,"\n");
	    }
	}
	fprintf(fp,"F R A M E   E L E M E N T   E N D   F O R C E S");
	fprintf(fp,"\t\t\t\t(local)\n");
	fprintf(fp,"  Elmnt  Node       Nx          Vy         Vz");
	fprintf(fp,"        Txx        Myy        Mzz\n");
	for (n=1; n<= nE; n++) {
		fprintf(fp," %5d  %5d", n, J1[n]);
		if ( fabs(Q[n][1]) < 0.0001 )
			fprintf (fp, "      0.0   ");
		else    fprintf (fp, " %10.3f", Q[n][1] );
		if ( Q[n][1] >=  0.0001 && axial_sign) fprintf(fp, "c");
		if ( Q[n][1] <= -0.0001 && axial_sign) fprintf(fp, "t");
		if (!axial_sign) fprintf(fp," ");
		for (i=2; i<=6; i++) {
			if ( fabs(Q[n][i]) < 0.0001 )
				fprintf (fp, "      0.0  ");
			else    fprintf (fp, " %10.3f", Q[n][i] );
		}
		fprintf(fp,"\n");
		fprintf(fp," %5d  %5d", n, J2[n]);
		if ( fabs(Q[n][7]) < 0.0001 )
			fprintf (fp, "      0.0   ");
		else    fprintf (fp, " %10.3f", Q[n][7] );
		if ( Q[n][7] >=  0.0001 && axial_sign) fprintf(fp, "t");
		if ( Q[n][7] <= -0.0001 && axial_sign) fprintf(fp, "c");
		if (!axial_sign) fprintf(fp," ");
		for (i=8; i<=12; i++) {
			if ( fabs(Q[n][i]) < 0.0001 )
				fprintf (fp, "      0.0  ");
			else    fprintf (fp, " %10.3f", Q[n][i] );
		}
		fprintf(fp,"\n");
	}
	fprintf(fp,"R E A C T I O N S\t\t\t\t\t\t\t(global)\n");
	fprintf(fp,"  Node        Fx          Fy          Fz");
	fprintf(fp,"         Mxx         Myy         Mzz\n");
	for (j=1; j<=nN; j++) {
		i = 6*(j-1);
		if ( r[i+1] || r[i+2] || r[i+3] ||
		     r[i+4] || r[i+5] || r[i+6] ) {
			fprintf(fp, " %5d", j);
			for (i=5; i>=0; i--) {
			    if ( r[6*j-i] ) fprintf (fp, " %11.3f", R[6*j-i] );
			    else		fprintf (fp, "       0.0  ");
			}
			fprintf(fp, "\n");
		}
	}
	fprintf(fp,"R M S    R E L A T I V E    E Q U I L I B R I U M    E R R O R: %9.3e\n", err );
	fflush(fp);
	return;
}


/*
 * CSV_filename - return the file name for the .CSV file and 
 * whether the file should be written or appended (wa)
 * 1 Nov 2015
 */
void CSV_filename( char CSV_file[], char wa[], char OUT_file[], int lc )
{
	int i,j;

	i=0;
	j=0;
	while (i<FILENMAX) {
		CSV_file[j] = OUT_file[i];
		if ( CSV_file[j] == '+' ||
		     CSV_file[j] == '-' ||
		     CSV_file[j] == '*' ||
		     CSV_file[j] == '^' ||
		     CSV_file[j] == '.' ||
		     CSV_file[j] == '\0') {
			CSV_file[j] = '_';
			break;
		}
		i++;
		j++;
	}
	CSV_file[++j] = '\0';
	strcat(CSV_file,"out.CSV");

	wa[0] = 'a';
	if (lc == 1) wa[0] = 'w';
	wa[1] = '\0';

//	fprintf(stderr," 1. CSV_file = %s  wa = %s \n", CSV_file, wa );
}


/*
 * WRITE_STATIC_CSV -  save node displacements and frame element end forces
 * 31 Dec 2008
 */
void write_static_csv (
		char OUT_file[],
		char title[],
		int nN, int nE, int nL, int lc, int DoF,
		int *J1, int *J2,
		double *F, double *D, double *R, int *r, double **Q,
		double err, int ok
){
	FILE	*fpcsv;
	int	i,j,n;
	char	wa[4];
	char	CSV_file[FILENMAX];
	time_t  now;		/* modern time variable type	*/
	char	errMsg[MAXL];

	(void) time(&now);

	CSV_filename( CSV_file, wa, OUT_file, lc );

	if ((fpcsv = fopen (CSV_file, wa)) == NULL) {
	  sprintf (errMsg,"\n  error: cannot open CSV output data file: %s \n", CSV_file);
	  errorMsg(errMsg);
	  exit(17);
	}


	if ( lc == 1 ) {
  	 fprintf(fpcsv,"\" Frame3DD version: %s ", VERSION );
	 fprintf(fpcsv,"              http://frame3dd.sf.net/\"\n");
	 fprintf(fpcsv,"\"GPL Copyright (C) 1992-2015, Henri P. Gavin \"\n");
	 fprintf(fpcsv,"\"Frame3DD is distributed in the hope that it will be useful");
	 fprintf(fpcsv," but with no warranty.\"\n");
	 fprintf(fpcsv,"\"For details see the GNU Public Licence:");
	 fprintf(fpcsv," http://www.fsf.org/copyleft/gpl.html\"\n");
	 fprintf(fpcsv,"\" %s \"\n",title);
	 fprintf(fpcsv,"\" %s \"\n", ctime(&now) );

	 fprintf(fpcsv,"\" .CSV formatted results of Frame3DD analysis \"\n");
	 fprintf(fpcsv,"\n , Load Case , Displacements , End Forces , Reactions , Internal Forces \n");
	 for (i = 1; i <= nL; i++) {
	 	fprintf(fpcsv," First Row , %d , %d , %d , %d  , %d  \n",
			i,
			15+(i-1)*(nN*2+nE*4+13) + 2*nL,
			17+(i-1)*(nN*2+nE*4+13) + 2*nL + nN,
			19+(i-1)*(nN*2+nE*4+13) + 2*nL + nN + 2*nE,
			23+(i-1)*(nN*2+nE*4+13) + 2*nL + 2*nN + 2*nE );
	 	fprintf(fpcsv," Last Row , %d , %d , %d , %d , %d \n",
			i,
			15+(i-1)*(nN*2+nE*4+13) + 2*nL + nN - 1,
			17+(i-1)*(nN*2+nE*4+13) + 2*nL + nN + 2*nE - 1,
			19+(i-1)*(nN*2+nE*4+13) + 2*nL + 2*nN + 2*nE - 1, 
			23+(i-1)*(nN*2+nE*4+13) + 2*nL + 2*nN + 4*nE - 1 );
	 }

	}


	if ( ok < 0 ) {
	 fprintf(fpcsv,"\"  * The Stiffness Matrix is not positive-definite * \"\n");
	 fprintf(fpcsv,"\" Check that all six rigid-body translations are restrained\"\n");
	 fprintf(fpcsv,"\" If geometric stiffness is included, reduce the loads.\"\n");
/*	 return; */
	}


	fprintf(fpcsv,"\n\"L O A D   C A S E   %d   O F   %d  ... \"\n\n", lc, nL);

	fprintf(fpcsv,"\"N O D E   D I S P L A C E M E N T S");
	fprintf(fpcsv,"    (global)\"\n");
	fprintf(fpcsv,"Node  ,  X-dsp   ,   Y-dsp  ,    Z-dsp");
	fprintf(fpcsv," ,     X-rot  ,    Y-rot   ,   Z-rot\n");
	for (j=1; j<= nN; j++) {
		fprintf(fpcsv," %5d,", j);
		for ( i=5; i>=0; i-- ) {
			if ( fabs(D[6*j-i]) < 1.e-8 )
				fprintf (fpcsv, "    0.0,    ");
			else    fprintf (fpcsv, " %12.5e,",  D[6*j-i] );
		}
		fprintf(fpcsv,"\n");
	}
	fprintf(fpcsv,"\"F R A M E   E L E M E N T   E N D   F O R C E S");
	fprintf(fpcsv,"  (local)\"\n");
	fprintf(fpcsv,"Elmnt , Node  ,    Nx     ,    Vy   ,     Vz");
	fprintf(fpcsv,"   ,     Txx   ,    Myy  ,     Mzz\n");
	for (n=1; n<= nE; n++) {
		fprintf(fpcsv," %5d, %5d,", n, J1[n]);
		if ( fabs(Q[n][1]) < 0.0001 )
			fprintf (fpcsv, "      0.0,  ");
		else    fprintf (fpcsv, " %12.5e,", Q[n][1] );
		for (i=2; i<=6; i++) {
			if ( fabs(Q[n][i]) < 0.0001 )
				fprintf (fpcsv, "      0.0, ");
			else    fprintf (fpcsv, " %12.5e,", Q[n][i] );
		}
		fprintf(fpcsv,"\n");
		fprintf(fpcsv," %5d, %5d,", n, J2[n]);
		if ( fabs(Q[n][7]) < 0.0001 )
			fprintf (fpcsv, "      0.0,  ");
		else    fprintf (fpcsv, " %12.5e,", Q[n][7] );
		for (i=8; i<=12; i++) {
			if ( fabs(Q[n][i]) < 0.0001 )
				fprintf (fpcsv, "      0.0, ");
			else    fprintf (fpcsv, " %12.5e,", Q[n][i] );
		}
		fprintf(fpcsv,"\n");
	}
	fprintf(fpcsv,"\"R E A C T I O N S  (global)\"\n");
	fprintf(fpcsv," Node   ,    Fx      ,   Fy   ,      Fz");
	fprintf(fpcsv,"   ,     Mxx    ,    Myy    ,    Mzz\n");
	for (j=1; j<=nN; j++) {
		i = 6*(j-1);
		fprintf(fpcsv, " %5d,", j);
		for (i=5; i>=0; i--) {
			if ( r[6*j-i] ) fprintf (fpcsv, " %12.5e,", R[6*j-i] );
			else	fprintf (fpcsv, "       0.0, ");
		}
		fprintf(fpcsv, "\n");
	}
	fprintf(fpcsv,"\"R M S    R E L A T I V E    E Q U I L I B R I U M    E R R O R:\", %9.3e\n", err );

	fclose(fpcsv);

	return;
}


/*
 * WRITE_VALUE - write a value in %f or %e notation depending on numerical values
 * and the number of available significant figures
 * 12 Dec 2009
 */
/*
void write_value ( 
		FILE *fp, 
		int sig_figs, 
		float threshold, 
		char *spaces,
		double x
){
	int nZspaces;

	nZspaces = (int) strlen(*spaces);

	if ( fabs(x) < threshold ) fprintf ( fp, "0.0 \n");

}
*/


/*
 * WRITE_STATIC_MFILE -  						
 * save node displacements and frame element end forces in an m-file
 * this function interacts with frame_3dd.m, an m-file interface to frame3dd
 * 09 Sep 2008
 */
void write_static_mfile (
		char *OUT_file, char *title,
		int nN, int nE, int nL, int lc, int DoF,
		int *J1, int *J2,
		double *F, double *D, double *R, int *r, double **Q,
		double err, int ok
){
	FILE	*fpm;
	int	i,j,n;
	char	*wa;
	char	M_file[FILENMAX];
	time_t  now;	/* modern time variable type	*/
	char	errMsg[MAXL];

	(void) time(&now);

	i=0;
	j=0;
	while (i<FILENMAX) {
		M_file[j] = OUT_file[i];
		if ( M_file[j] == '+' ||
		     M_file[j] == '-' ||
		     M_file[j] == '*' ||
		     M_file[j] == '^' ||
		     M_file[j] == '.' ||
		     M_file[j] == '\0') {
			M_file[j] = '_';
			break;
		}
		i++;
		j++;
	}
	M_file[++j] = '\0';
	strcat(M_file,"out.m");

	wa  = "a";
	if (lc == 1) wa = "w";

	if ((fpm = fopen (M_file, wa)) == NULL) {
	  sprintf (errMsg,"\n  error: cannot open Matlab output data file: %s \n", M_file );
	  errorMsg(errMsg);
	  exit(18);
	}

	if ( lc == 1 ) {
  	 fprintf(fpm,"%% Frame3DD version: %s ", VERSION );
	 fprintf(fpm,"              http://frame3dd.sf.net/\n");
	 fprintf(fpm,"%%GPL Copyright (C) 1992-2015, Henri P. Gavin \n");
	 fprintf(fpm,"%%Frame3DD is distributed in the hope that it will be useful");
	 fprintf(fpm," but with no warranty.\n");
	 fprintf(fpm,"%%For details see the GNU Public Licence:");
	 fprintf(fpm," http://www.fsf.org/copyleft/gpl.html\n");
	 fprintf(fpm,"%% %s\n",title);
	 fprintf(fpm, "%% %s", ctime(&now) );

	 fprintf(fpm,"%% m-file formatted results of frame3dd analysis\n");
	 fprintf(fpm,"%% to be read by frame_3dd.m\n");
	}


	if ( ok < 0 ) {
	 fprintf(fpm,"%%  The Stiffness Matrix is not positive-definite *\n");
	 fprintf(fpm,"%%  Check that all six rigid-body translations are restrained\n");
	 fprintf(fpm,"%%  If geometric stiffness is included, reduce the loads.\n");
/*	 return; */
	}

	fprintf(fpm,"\n%% L O A D   C A S E   %d   O F   %d  ... \n\n", lc, nL);

	fprintf(fpm,"%% N O D E   D I S P L A C E M E N T S  ");
	fprintf(fpm,"\t\t(global)\n");
	fprintf(fpm,"%%\tX-dsp\t\tY-dsp\t\tZ-dsp\t\tX-rot\t\tY-rot\t\tZ-rot\n");
	fprintf(fpm,"D%d=[",lc);
	for (j=1; j<= nN; j++) {
		for ( i=5; i>=0; i-- ) {
			if ( fabs(D[6*j-i]) < 1.e-8 )
				fprintf (fpm, "\t0.0\t");
			else    fprintf (fpm, "\t%13.6e",  D[6*j-i] );
		}
		if ( j < nN )	fprintf(fpm," ; \n");
		else		fprintf(fpm," ]'; \n\n");
	}

	fprintf(fpm,"%% F R A M E   E L E M E N T   E N D   F O R C E S");
	fprintf(fpm,"\t\t(local)\n");
	fprintf(fpm,"%%\tNx_1\t\tVy_1\t\tVz_1\t\tTxx_1\t\tMyy_1\t\tMzz_1\t");
	fprintf(fpm,"  \tNx_2\t\tVy_2\t\tVz_2\t\tTxx_2\t\tMyy_2\t\tMzz_2\n");
	fprintf(fpm,"F%d=[",lc);
	for (n=1; n<= nE; n++) {
		if ( fabs(Q[n][1]) < 0.0001 )
			fprintf (fpm, "\t0.0\t");
		else    fprintf (fpm, "\t%13.6e", Q[n][1] );
		for (i=2; i<=6; i++) {
			if ( fabs(Q[n][i]) < 0.0001 )
				fprintf (fpm, "\t0.0\t");
			else    fprintf (fpm, "\t%13.6e", Q[n][i] );
		}
		if ( fabs(Q[n][7]) < 0.0001 )
			fprintf (fpm, "\t0.0\t");
		else    fprintf (fpm, "\t%13.6e", Q[n][7] );
		for (i=8; i<=12; i++) {
			if ( fabs(Q[n][i]) < 0.0001 )
				fprintf (fpm, "\t0.0\t");
			else    fprintf (fpm, "\t%13.6e", Q[n][i] );
		}
		if ( n < nE )	fprintf(fpm," ; \n");
		else		fprintf(fpm," ]'; \n\n");
	}

	fprintf(fpm,"%% R E A C T I O N S\t\t\t\t(global)\n");
	fprintf(fpm,"%%\tFx\t\tFy\t\tFz\t\tMxx\t\tMyy\t\tMzz\n");
	fprintf(fpm,"R%d=[",lc);
	for (j=1; j<=nN; j++) {
		i = 6*(j-1);
		for (i=5; i>=0; i--) {
			if ( !r[6*j-i] || fabs(R[6*j-i]) < 0.0001 )
				fprintf (fpm, "\t0.0\t");
			else    fprintf (fpm, "\t%13.6e", R[6*j-i] );
		}
		if ( j < nN )	fprintf(fpm," ; \n");
		else		fprintf(fpm," ]'; \n\n");
	}

	fprintf(fpm,"%% R M S    R E L A T I V E    E Q U I L I B R I U M    E R R O R: %9.3e\n", err );
	fprintf(fpm,"\n\n  load Ks \n\n");

	fclose(fpm);

	return;
}


/*
 * PEAK_INTERNAL_FORCES
 * calculate frame element internal forces, Nx, Vy, Vz, Tx, My, Mz
 * calculate frame element local displacements, Rx, Dx, Dy, Dz
 * return the peak values of the internal forces, moments, slopes, and displacements
 * 18jun13
 */
void peak_internal_forces (
		int lc, 	// load case number
		int nL, 	// total number of load cases
		vec3 *xyz, 	// node locations
		double **Q, int nN, int nE, double *L, int *N1, int *N2, 
		float *Ax,float *Asy,float *Asz,float *Jx,float *Iy,float *Iz,
		float *E, float *G, float *p,
		float *d, float gX, float gY, float gZ,
		int nU, float **U, int nW, float **W, int nP, float **P,
		double *D, int shear, 
		float dx,	// x-axis increment along frame element

		// vectors of peak forces, moments, displacements and slopes 
		// for each frame element, for load case "lc" 
		double **pkNx, double **pkVy, double **pkVz, 
		double **pkTx, double **pkMy, double **pkMz,
		double **pkDx, double **pkDy, double **pkDz,
		double **pkRx, double **pkSy, double **pkSz
){
	double	t1, t2, t3, t4, t5, t6, t7, t8, t9, /* coord transformation */
		u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12; /* displ. */

	double	xx1,xx2, wx1,wx2,	/* trapz load data, local x dir */
		xy1,xy2, wy1,wy2,	/* trapz load data, local y dir */
		xz1,xz2, wz1,wz2;	/* trapz load data, local z dir */

	double	wx=0, wy=0, wz=0, // distributed loads in local coords at x[i] 
		wx_=0,wy_=0,wz_=0,// distributed loads in local coords at x[i-1]
		wxg=0,wyg=0,wzg=0,// gravity loads in local x, y, z coord's
		tx=0.0, tx_=0.0;  // distributed torque about local x coord 

	double	xp;		/* location of internal point loads	*/

	double	x, 		/* distance along frame element		*/

		// underscored "_" variables correspond to x=(i-1)*dx;
		// non-underscored variables correspond to x=i*dx;
		Nx_, Nx,	/* axial force within frame el.		*/
		Vy_, Vy, Vz_, Vz,/* shear forces within frame el.	*/
		Tx_, Tx,		/* torsional moment within frame el.	*/
		My_, My, Mz_, Mz, /* bending moments within frame el.	*/
		Sy_, Sy, Sz_, Sz,	/* transverse slopes of frame el.	*/
		Dx, Dy, Dz,	/* frame el. displ. in local x,y,z, dir's */
		Rx;		/* twist rotation about the local x-axis */

	int	n, m,		// frame element number	
		nx=1000,	// number of sections alont x axis
		cU=0, cW=0, cP=0, // counters for U, W, and P loads
		i,		// counter along x axis from node N1 to node N2
		n1,n2,i1,i2;	// starting and stopping node numbers

	if (dx == -1.0)	return;	// skip calculation of internal forces and displ

	for ( m=1; m <= nE; m++ ) {	// initialize peak values to zero
		pkNx[lc][m] = pkVy[lc][m] = pkVz[lc][m] = 0.0; 
		pkTx[lc][m] = pkMy[lc][m] = pkMz[lc][m] = 0.0;
		pkDx[lc][m] = pkDy[lc][m] = pkDz[lc][m] = 0.0;
		pkRx[lc][m] = pkSy[lc][m] = pkSz[lc][m] = 0.0;
	}

	for ( m=1; m <= nE; m++ ) {	// loop over all frame elements

		n1 = N1[m];	n2 = N2[m]; // node 1 and node 2 of elmnt m

		dx = L[m] / (float) nx; // x-axis increment, same for each element

	// no need to allocate memory for interior force or displacement data 

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
		Nx_ = Nx = -Q[m][1];	// positive Nx is tensile
		Vy_ = Vy = -Q[m][2];	// positive Vy in local y direction
		Vz_ = Vz = -Q[m][3];	// positive Vz in local z direction
		Tx_ = Tx = -Q[m][4];	// positive Tx r.h.r. about local x axis
		My_ = My =  Q[m][5];	// positive My -> positive x-z curvature
		Mz_ = Mz = -Q[m][6];	// positive Mz -> positive x-y curvature

		i1 = 6*(n1-1);	i2 = 6*(n2-1);

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
		Dx =  u1;	// displacement in  local x dir  at node N1
		Dy =  u2;	// displacement in  local y dir  at node N1
		Dz =  u3;	// displacement in  local z dir  at node N1
		Rx =  u4;	// rotationin about local x axis at node N1
		Sy_ = Sy =  u6;	// slope in  local y  direction  at node N1
		Sz_ = Sz = -u5;	// slope in  local z  direction  at node N1

		// accumulate interior span loads, forces, moments, slopes, and displacements
		// all in a single loop  

		for (i=1; i<=nx; i++) {

			x = i*dx;	// location from node N1 along the x-axis

			// start with gravitational plus uniform loads
			wx = wxg; wy = wyg; wz = wzg;

			if (i==1) { wx_ = wxg; wy_ = wyg; wz_ = wzg; tx_ = tx; }

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

				if ( x>xx1 && x<=xx2 )
				    wx += wx1+(wx2-wx1)*(x-xx1)/(xx2-xx1);
				if ( x>xy1 && x<=xy2 )
				    wy += wy1+(wy2-wy1)*(x-xy1)/(xy2-xy1);
				if ( x>xz1 && x<=xz2 )
				    wz += wz1+(wz2-wz1)*(x-xz1)/(xz2-xz1);
			    }
			}

			// trapezoidal integration of distributed loads 
			// for axial forces, shear forces and torques

			Nx = Nx - 0.5*(wx+wx_)*dx;
			Vy = Vy - 0.5*(wy+wy_)*dx;
			Vz = Vz - 0.5*(wz+wz_)*dx;
			Tx = Tx - 0.5*(tx+tx_)*dx;

			// update distributed loads at x = (i-1)*dx
			wx_ = wx;
			wy_ = wy;
			wz_ = wz;
			tx_ = tx;
			
			// add interior point loads 
			for (n=1; n<=10*nE && cP<nP; n++) {
			    if ( (int) P[n][1] == m ) { // load n on element m
				if (i==nx) ++cP;
				xp = P[n][5];
				if ( x <= xp && xp < x+dx ) {
					Nx -= P[n][2] * 0.5 * (1.0 - (xp-x)/dx);
					Vy -= P[n][3] * 0.5 * (1.0 - (xp-x)/dx);
					Vz -= P[n][4] * 0.5 * (1.0 - (xp-x)/dx);
					
				}
				if ( x-dx <= xp && xp < x ) {
					Nx -= P[n][2] * 0.5 * (1.0 - (x-dx-xp)/dx);
					Vy -= P[n][3] * 0.5 * (1.0 - (x-dx-xp)/dx);
					Vz -= P[n][4] * 0.5 * (1.0 - (x-dx-xp)/dx);
				}
			    }
			}

			// trapezoidal integration of shear force for bending momemnt
			My = My - 0.5*(Vz_ + Vz)*dx;
			Mz = Mz - 0.5*(Vy_ + Vy)*dx;

			// displacement along frame element "m"
			Dx = Dx + 0.5*(Nx_ + Nx)/(E[m]*Ax[m])*dx;

			// torsional rotation along frame element "m"
			Rx = Rx + 0.5*(Tx_+Tx)/(G[m]*Jx[m])*dx;

			// transverse slope along frame element "m"
			Sy = Sy + 0.5*(Mz_ + Mz)/(E[m]*Iz[m])*dx;
			Sz = Sz + 0.5*(My_ + My)/(E[m]*Iy[m])*dx;
		
			if ( shear ) {
				Sy += Vy/(G[m]*Asy[m]);
				Sz += Vz/(G[m]*Asz[m]);
			}

			// displacement along frame element "m"
			Dy = Dy + 0.5*(Sy_+Sy)*dx;
			Dz = Dz + 0.5*(Sz_+Sz)*dx;

			// update forces, moments, and slopes at x = (i-1)*dx
			Nx_ = Nx;
			Vy_ = Vy;
			Vz_ = Vz;
			Tx_ = Tx;
			My_ = My;
			Mz_ = Mz;
			Sy_ = Sy;
			Sz_ = Sz;

			// update the peak forces, moments, slopes and displacements
			// and their locations along the frame element

			pkNx[lc][m] = (fabs(Nx) > pkNx[lc][m]) ? fabs(Nx) : pkNx[lc][m];
			pkVy[lc][m] = (fabs(Vy) > pkVy[lc][m]) ? fabs(Vy) : pkVy[lc][m];
			pkVz[lc][m] = (fabs(Vz) > pkVz[lc][m]) ? fabs(Vz) : pkVz[lc][m];

			pkTx[lc][m] = (fabs(Tx) > pkTx[lc][m]) ? fabs(Tx) : pkTx[lc][m];
			pkMy[lc][m] = (fabs(My) > pkMy[lc][m]) ? fabs(My) : pkMy[lc][m];
			pkMz[lc][m] = (fabs(Mz) > pkMz[lc][m]) ? fabs(Mz) : pkMz[lc][m];


			pkDx[lc][m] = (fabs(Dx) > pkDx[lc][m]) ? fabs(Dx) : pkDx[lc][m];
			pkDy[lc][m] = (fabs(Dy) > pkDy[lc][m]) ? fabs(Dy) : pkDy[lc][m];
			pkDz[lc][m] = (fabs(Dz) > pkDz[lc][m]) ? fabs(Dz) : pkDz[lc][m];

			pkRx[lc][m] = (fabs(Rx) > pkRx[lc][m]) ? fabs(Rx) : pkRx[lc][m];
			pkSy[lc][m] = (fabs(Sy) > pkSy[lc][m]) ? fabs(Sy) : pkSy[lc][m];
			pkSz[lc][m] = (fabs(Sz) > pkSz[lc][m]) ? fabs(Sz) : pkSz[lc][m];

		}			// end of loop along element "m"

		// at the end of this loop,
		// the variables Nx; Vy; Vz; Tx; My; Mz; Dx; Dy; Dz; Rx; Sy; Sz;
		// contain the forces, moments, displacements, and slopes 
		// at node N2 of element "m"

		// comparing the internal forces and displacements at node N2
		// to the values conmputed via trapezoidal rule could give an estimate 
		// of the accuracy of the trapezoidal rule, (how small "dx" needs to be)
			
		// linear correction for bias in trapezoidal integration 
		// is not implemented, the peak values are affected by accumulation
		// of round-off error in trapezoidal rule integration.   
		// round-off errors are larger in the peak displacements than in the peak forces


	}				// end of loop over all frame elements

	// DEBUG --- write output to terminal
	fprintf(stderr,"P E A K   F R A M E   E L E M E N T   I N T E R N A L   F O R C E S");
	fprintf(stderr,"\t(local)\n");
	fprintf(stderr,"  Elmnt       Nx          Vy         Vz");
	fprintf(stderr,"        Txx        Myy        Mzz\n");
	for (m=1; m<=nE; m++) 
		fprintf(stderr," %5d  %10.3f  %10.3f %10.3f %10.3f %10.3f %10.3f\n",
			m, pkNx[lc][m], pkVy[lc][m], pkVz[lc][m], pkTx[lc][m], pkMy[lc][m], pkMz[lc][m] );
	
	fprintf(stderr,"\n P E A K   I N T E R N A L   D I S P L A C E M E N T S");
	fprintf(stderr,"\t\t\t(local)\n");
  	fprintf(stderr,"  Elmnt  X-dsp       Y-dsp       Z-dsp       X-rot       Y-rot       Z-rot\n");
	for (m=1; m<=nE; m++) 
		fprintf(stderr," %5d %10.6f  %10.6f  %10.6f  %10.6f  %10.6f  %10.6f\n",
			m, pkDx[lc][m], pkDy[lc][m], pkDz[lc][m], pkRx[lc][m], pkSy[lc][m], pkSz[lc][m] );

}


/*
 * WRITE_INTERNAL_FORCES - 
 * calculate frame element internal forces, Nx, Vy, Vz, Tx, My, Mz
 * calculate frame element local displacements, Rx, Dx, Dy, Dz
 * write internal forces and local displacements to an output data file
 * 4jan10, 7mar11, 21jan14
 */
void write_internal_forces (
		char OUT_file[],
		FILE *fp, char infcpath[], int lc, int nL, char title[], float dx,
		vec3 *xyz, 
		double **Q, int nN, int nE, double *L, int *J1, int *J2, 
		float *Ax,float *Asy,float *Asz,float *Jx,float *Iy,float *Iz,
		float *E, float *G, float *p,
		float *d, float gX, float gY, float gZ,
		int nU, float **U, int nW, float **W, int nP, float **P,
		double *D, int shear, double error
){
	double	t1, t2, t3, t4, t5, t6, t7, t8, t9, /* coord transformation */
		u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12; /* displ. */

	double	xx1,xx2, wx1,wx2,	/* trapz load data, local x dir */
		xy1,xy2, wy1,wy2,	/* trapz load data, local y dir */
		xz1,xz2, wz1,wz2;	/* trapz load data, local z dir */

	double	wx=0, wy=0, wz=0, // distributed loads in local coords at x[i] 
		wx_=0,wy_=0,wz_=0,// distributed loads in local coords at x[i-1]
		wxg=0,wyg=0,wzg=0,// gravity loads in local x, y, z coord's
		tx=0.0, tx_=0.0;  // distributed torque about local x coord 

	double	xp;		/* location of internal point loads	*/

	double	*x, dx_, dxnx,	/* distance along frame element		*/
		*Nx,		/* axial force within frame el.		*/
		*Vy, *Vz,	/* shear forces within frame el.	*/
		*Tx,		/* torsional moment within frame el.	*/
		*My, *Mz, 	/* bending moments within frame el.	*/
		*Sy, *Sz,	/* transverse slopes of frame el.	*/
		*Dx, *Dy, *Dz,	/* frame el. displ. in local x,y,z, dir's */
		*Rx;		/* twist rotation about the local x-axis */

	double	maxNx, maxVy, maxVz, 	/*  maximum internal forces	*/
		maxTx, maxMy, maxMz,	/*  maximum internal moments	*/
		maxDx, maxDy, maxDz,	/*  maximum element displacements */
		maxRx, maxSy, maxSz;	/*  maximum element rotations	*/

	double	minNx, minVy, minVz, 	/*  minimum internal forces	*/
		minTx, minMy, minMz,	/*  minimum internal moments	*/
		minDx, minDy, minDz,	/*  minimum element displacements */
		minRx, minSy, minSz;	/*  minimum element rotations	*/

	int	n, m,		/* frame element number			*/
		cU=0, cW=0, cP=0, /* counters for U, W, and P loads	*/
		i, nx,		/* number of sections alont x axis	*/
		n1,n2,i1,i2;	/* starting and stopping node no's	*/

	char	fnif[FILENMAX];/* file name    for internal force data	*/
	char	CSV_file[FILENMAX];
	char	errMsg[MAXL];
	char	wa[4];          /* indicate 'write' or 'append' to file */
	FILE	*fpif,		/* file pointer for internal force data */
		*fpcsv;         /* file pointer to .CSV output data file */
	time_t  now;		/* modern time variable type		*/

	if (dx == -1.0)	return;	// skip calculation of internal forces and displ

	(void) time(&now);

 
	CSV_filename( CSV_file, wa, OUT_file, lc );

	if ((fpcsv = fopen (CSV_file, "a")) == NULL) {
	  sprintf (errMsg,"\n  error: cannot open CSV output data file: %s \n", CSV_file);
	  errorMsg(errMsg);
	  exit(17);
	}  

	/* file name for internal force data for load case "lc" */
	sprintf(fnif,"%s%02d",infcpath,lc);
	
	/* open the interior force data file */
	if ((fpif = fopen (fnif, "w")) == NULL) {
         sprintf (errMsg,"\n  error: cannot open interior force data file: %s \n",fnif);
	 errorMsg(errMsg);
         exit(19);
	}

	fprintf(fpif,"# FRAME3DD ANALYSIS RESULTS  http://frame3dd.sf.net/");
	fprintf(fpif," VERSION %s \n", VERSION);
	fprintf(fpif,"# %s\n", title );
	fprintf(fpif,"# %s\n", fnif);
	fprintf(fpif,"# %s", ctime(&now) );
	fprintf(fpif,"# L O A D  C A S E   %d  of   %d \n", lc, nL );
	fprintf(fpif,"# F R A M E   E L E M E N T   I N T E R N A L   F O R C E S (local)\n");
	fprintf(fpif,"# F R A M E   E L E M E N T   T R A N S V E R S E   D I S P L A C E M E N T S (local)\n\n");

	// write header information for each frame element to txt output data file 
	fprintf(fp,"\nP E A K   F R A M E   E L E M E N T   I N T E R N A L   F O R C E S");
	fprintf(fp,"(local)\", \n");
	fprintf(fp,"  Elmnt   .         Nx          Vy         Vz");
	fprintf(fp,"        Txx        Myy        Mzz\n");

	// write header information for each frame element to CSV output data file
	fprintf(fpcsv,"\n\"P E A K   F R A M E   E L E M E N T   I N T E R N A L   F O R C E S ");
	fprintf(fpcsv,"   (local)\",\n");
	fprintf(fpcsv," \"Elmnt\",  \".\", \"Nx\", \"Vy\", \"Vz\", ");
	fprintf(fpcsv," \"Txx\", \"Myy\", \"Mzz\", \n");


/*	fprintf(fp,"\n P E A K   I N T E R N A L   D I S P L A C E M E N T S");
 *	fprintf(fp,"\t\t\t(local)\n");
 * 	fprintf(fp,"  Elmnt  X-dsp       Y-dsp       Z-dsp       X-rot       Y-rot       Z-rot\n");
*/

	for ( m=1; m <= nE; m++ ) {	// loop over all frame elements

		n1 = J1[m];	n2 = J2[m]; // node 1 and node 2 of elmnt m

		nx = floor(L[m]/dx);	// number of x-axis increments
		if (nx < 1) nx = 1;	// at least one x-axis increment

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
		for (i=0; i<nx; i++)	x[i] = i*dx;	
		x[nx] = L[m];		
		dxnx = x[nx]-x[nx-1];	// length of the last x-axis increment


	// write header information for each frame element

		fprintf(fpif,"#\tElmnt\tN1\tN2        \tX1        \tY1        \tZ1        \tX2        \tY2        \tZ2\tnx\n");
		fprintf(fpif,"# @\t%5d\t%5d\t%5d\t%14.6e\t%14.6e\t%14.6e\t%14.6e\t%14.6e\t%14.6e\t%5d\n",m, n1, n2, xyz[n1].x, xyz[n1].y, xyz[n1].z, xyz[n2].x, xyz[n2].y, xyz[n2].z, nx+1 );

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
		Nx[0] = -Q[m][1];	// positive Nx is tensile
		Vy[0] = -Q[m][2];	// positive Vy in local y direction
		Vz[0] = -Q[m][3];	// positive Vz in local z direction
		Tx[0] = -Q[m][4];	// positive Tx r.h.r. about local x axis
		My[0] =  Q[m][5];	// positive My -> positive x-z curvature
		Mz[0] = -Q[m][6];	// positive Mz -> positive x-y curvature

		dx_ = dx;
		for (i=1; i<=nx; i++) {	/*  accumulate interior span loads */

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
			if (i==nx)	dx_ = dxnx;
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
		// linear correction of forces for bias in trapezoidal integration
		for (i=1; i<=nx; i++) {
			Nx[i] -= (Nx[nx]-Q[m][7])  * i/nx;
			Vy[i] -= (Vy[nx]-Q[m][8])  * i/nx;
			Vz[i] -= (Vz[nx]-Q[m][9])  * i/nx;
			Tx[i] -= (Tx[nx]-Q[m][10]) * i/nx;
		}
		// trapezoidal integration of shear force for bending momemnt
		dx_ = dx;
		for (i=1; i<=nx; i++) {
			if (i==nx)	dx_ = dxnx;
			My[i] = My[i-1] - 0.5*(Vz[i]+Vz[i-1])*dx_;
			Mz[i] = Mz[i-1] - 0.5*(Vy[i]+Vy[i-1])*dx_;

		}
		// linear correction of moments for bias in trapezoidal integration
		for (i=1; i<=nx; i++) {
			My[i] -= (My[nx]+Q[m][11]) * i/nx;
			Mz[i] -= (Mz[nx]-Q[m][12]) * i/nx;
		}

	// find interior transverse displacements 

		i1 = 6*(n1-1);	i2 = 6*(n2-1);

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
		Dx[0] =  u1;	// displacement in  local x dir  at node N1
		Dy[0] =  u2;	// displacement in  local y dir  at node N1
		Dz[0] =  u3;	// displacement in  local z dir  at node N1
		Rx[0] =  u4;	// rotationin about local x axis at node N1
		Sy[0] =  u6;	// slope in  local y  direction  at node N1
		Sz[0] = -u5;	// slope in  local z  direction  at node N1

		// axial displacement along frame element "m"
		dx_ = dx;
		for (i=1; i<=nx; i++) {
			if (i==nx)	dx_ = dxnx;
			Dx[i] = Dx[i-1] + 0.5*(Nx[i-1]+Nx[i])/(E[m]*Ax[m])*dx_;
		}
		// linear correction of axial displacement for bias in trapezoidal integration
		for (i=1; i<=nx; i++) {
			Dx[i] -= (Dx[nx]-u7) * i/nx;
		}
		
		// torsional rotation along frame element "m"
		dx_ = dx;
		for (i=1; i<=nx; i++) {
			if (i==nx)	dx_ = dxnx;
			Rx[i] = Rx[i-1] + 0.5*(Tx[i-1]+Tx[i])/(G[m]*Jx[m])*dx_;
		}
		// linear correction of torsional rot'n for bias in trapezoidal integration
		for (i=1; i<=nx; i++) {
			Rx[i] -= (Rx[nx]-u10) * i/nx;
		}
		
		// transverse slope along frame element "m"
		dx_ = dx;
		for (i=1; i<=nx; i++) {
			if (i==nx)	dx_ = dxnx;
			Sy[i] = Sy[i-1] + 0.5*(Mz[i-1]+Mz[i])/(E[m]*Iz[m])*dx_;
			Sz[i] = Sz[i-1] + 0.5*(My[i-1]+My[i])/(E[m]*Iy[m])*dx_;
		}
		// linear correction for bias in trapezoidal integration
		for (i=1; i<=nx; i++) {
			Sy[i] -= (Sy[nx]-u12) * i/nx;
			Sz[i] -= (Sz[nx]+u11) * i/nx;
		}
		if ( shear ) {		// add-in slope due to shear deformation
			for (i=0; i<=nx; i++) {
				Sy[i] += Vy[i]/(G[m]*Asy[m]);
				Sz[i] += Vz[i]/(G[m]*Asz[m]);
			}
		}
		// displacement along frame element "m"
		dx_ = dx;
		for (i=1; i<=nx; i++) {
			if (i==nx)	dx_ = dxnx;
			Dy[i] = Dy[i-1] + 0.5*(Sy[i-1]+Sy[i])*dx_;
			Dz[i] = Dz[i-1] + 0.5*(Sz[i-1]+Sz[i])*dx_;
		}
		// linear correction for bias in trapezoidal integration
 		for (i=1; i<=nx; i++) {
			Dy[i] -= (Dy[nx]-u8) * i/nx;
			Dz[i] -= (Dz[nx]-u9) * i/nx;
		}

	// initialize the maximum and minimum element forces and displacements 
		maxNx = minNx = Nx[0]; maxVy = minVy = Vy[0]; maxVz = minVz = Vz[0];  	//  maximum internal forces
		maxTx = minTx = Tx[0]; maxMy = minMy = My[0]; maxMz = minMz = Mz[0]; 	//  maximum internal moments
		maxDx = minDx = Dx[0]; maxDy = minDy = Dy[0]; maxDz = minDz = Dz[0]; 	//  maximum element displacements
		maxRx =	minRx = Rx[0]; maxSy = minSy = Sy[0]; maxSz = minSz = Sz[0];	//  maximum element rotations

	// find maximum and minimum internal element forces
		for (i=1; i<=nx; i++) {
			maxNx = (Nx[i] > maxNx) ? Nx[i] : maxNx;
			minNx = (Nx[i] < minNx) ? Nx[i] : minNx;
			maxVy = (Vy[i] > maxVy) ? Vy[i] : maxVy;
			minVy = (Vy[i] < minVy) ? Vy[i] : minVy;
			maxVz = (Vz[i] > maxVz) ? Vz[i] : maxVz;
			minVz = (Vz[i] < minVz) ? Vz[i] : minVz;

			maxTx = (Tx[i] > maxTx) ? Tx[i] : maxTx;
			minTx = (Tx[i] < minTx) ? Tx[i] : minTx;
			maxMy = (My[i] > maxMy) ? My[i] : maxMy;
			minMy = (My[i] < minMy) ? My[i] : minMy;
			maxMz = (Mz[i] > maxMz) ? Mz[i] : maxMz;
			minMz = (Mz[i] < minMz) ? Mz[i] : minMz;
		}

	// find maximum and minimum internal element displacements
		for (i=1; i<=nx; i++) {
			maxDx = (Dx[i] > maxDx) ? Dx[i] : maxDx;
			minDx = (Dx[i] < minDx) ? Dx[i] : minDx;
			maxDy = (Dy[i] > maxDy) ? Dy[i] : maxDy;
			minDy = (Dy[i] < minDy) ? Dy[i] : minDy;
			maxDz = (Dz[i] > maxDz) ? Dz[i] : maxDz;
			minDz = (Dz[i] < minDz) ? Dz[i] : minDz;
			maxRx = (Rx[i] > maxRx) ? Rx[i] : maxRx;
			minRx = (Rx[i] < minRx) ? Rx[i] : minRx;
			maxSy = (Sy[i] > maxSy) ? Sy[i] : maxSy;
			minSy = (Sy[i] < minSy) ? Sy[i] : minSy;
			maxSz = (Sz[i] > maxSz) ? Sz[i] : maxSz;
			minSz = (Sz[i] < minSz) ? Sz[i] : minSz;
		}

	// write max and min element forces to the internal frame element force output data file
		fprintf(fpif,"#                \tNx        \tVy        \tVz        \tTx        \tMy        \tMz        \tDx        \tDy        \tDz         \tRx\t*\n");
		fprintf(fpif,"# MAXIMUM\t%14.6e\t%14.6e\t%14.6e\t%14.6e\t%14.6e\t%14.6e\t%14.6e\t%14.6e\t%14.6e\t%14.6e\n", maxNx,maxVy,maxVz,maxTx,maxMy,maxMz,maxDx,maxDy,maxDz,maxRx );
		fprintf(fpif,"# MINIMUM\t%14.6e\t%14.6e\t%14.6e\t%14.6e\t%14.6e\t%14.6e\t%14.6e\t%14.6e\t%14.6e\t%14.6e\n", minNx,minVy,minVz,minTx,minMy,minMz,minDx,minDy,minDz,minRx );


	// write results to the internal frame element force output data file
		fprintf(fpif,"#.x                \tNx        \tVy        \tVz        \tTx       \tMy        \tMz        \tDx        \tDy        \tDz        \tRx\t~\n");
		for (i=0; i<=nx; i++) {
			fprintf(fpif,"%14.6e\t", x[i] );
			fprintf(fpif,"%14.6e\t%14.6e\t%14.6e\t",
						Nx[i], Vy[i], Vz[i] );
			fprintf(fpif,"%14.6e\t%14.6e\t%14.6e\t",
						Tx[i], My[i], Mz[i] );
			fprintf(fpif,"%14.6e\t%14.6e\t%14.6e\t%14.6e\n",
						Dx[i], Dy[i], Dz[i], Rx[i] );
		}
		fprintf(fpif,"#---------------------------------------\n\n\n");

	// write max and min element forces to the Frame3DD text output data file
		fprintf(fp," %5d   max  %10.3f  %10.3f %10.3f %10.3f %10.3f %10.3f\n",
				m, maxNx, maxVy, maxVz, maxTx, maxMy, maxMz );
		fprintf(fp," %5d   min  %10.3f  %10.3f %10.3f %10.3f %10.3f %10.3f\n",
				m, minNx, minVy, minVz, minTx, minMy, minMz );

	// write max and min element forces to the Frame3DD CSV output data file
		fprintf(fpcsv," %5d, \"max\", %10.3f,  %10.3f, %10.3f, %10.3f, %10.3f, %10.3f\n",
				m, maxNx, maxVy, maxVz, maxTx, maxMy, maxMz );
		fprintf(fpcsv," %5d, \"min\", %10.3f,  %10.3f, %10.3f, %10.3f, %10.3f, %10.3f\n",
				m, minNx, minVy, minVz, minTx, minMy, minMz );
	
/*
		fprintf(fp," %5d %10.6f  %10.6f  %10.6f  %10.6f  %10.6f  %10.6f\n",
				m, maxDx, maxDy, maxDz, maxRx, maxSy, maxSz );
		fprintf(fp," %5d %10.6f  %10.6f  %10.6f  %10.6f  %10.6f  %10.6f\n",
				m, minDx, minDy, minDz, minRx, minSy, minSz );
*/

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

	}				// end of loop over all frame elements

	fclose(fpif);
	fclose(fpcsv);
}


/*
 * WRITE_MODAL_RESULTS -  save modal frequencies and mode shapes	
 * 16 Aug 2001
 */
void write_modal_results(
		FILE *fp,
		int nN, int nE, int nI, int DoF,
		double **M, double *f, double **V,
		double total_mass, double struct_mass,
		int iter, int sumR, int nM,
		double shift, int lump, double tol, int ok
){
	int	i, j, k, m, num_modes;
	double	mpfX, mpfY, mpfZ,	/* mode participation factors	*/
		*msX, *msY, *msZ;
	double	fs;

	msX = dvector(1,DoF);
	msY = dvector(1,DoF);
	msZ = dvector(1,DoF);

	for (i=1; i<=DoF; i++) {
		msX[i] = msY[i] = msZ[i] = 0.0;
		for (j=1; j<=DoF; j+=6) msX[i] += M[i][j];
		for (j=2; j<=DoF; j+=6) msY[i] += M[i][j];
		for (j=3; j<=DoF; j+=6) msZ[i] += M[i][j];
	}

	if ( (DoF - sumR) > nM )	num_modes = nM;
	else	num_modes = DoF - sumR;

	fprintf(fp,"\nM O D A L   A N A L Y S I S   R E S U L T S\n");
	fprintf(fp,"  Total Mass:  %e   ", total_mass );
	fprintf(fp,"  Structural Mass:  %e \n", struct_mass );
	fprintf(fp,"N O D A L   M A S S E S");
	fprintf(fp,"\t(diagonal of the mass matrix)\t\t\t(global)\n");
	fprintf(fp,"  Node  X-mass      Y-mass      Z-mass");
	fprintf(fp,"      X-inrta     Y-inrta     Z-inrta\n");
	for (j=1; j <= nN; j++) {
		k = 6*(j-1);
		fprintf(fp," %5d", j);
		for ( i=1; i<=6; i++ )
			fprintf (fp, " %11.5e", M[k+i][k+i] );
		fprintf(fp,"\n");
	}
	if ( lump )	fprintf(fp,"  Lump masses at nodes.\n");
	else		fprintf(fp,"  Use consistent mass matrix.\n");
	fprintf(fp,"N A T U R A L   F R E Q U E N C I E S   & \n");
	fprintf(fp,"M A S S   N O R M A L I Z E D   M O D E   S H A P E S \n");
	fprintf(fp," convergence tolerance: %.3e \n", tol);
	for (m=1; m<=num_modes; m++) {
	    mpfX = 0.0;	for (i=1; i<=DoF; i++)    mpfX += V[i][m]*msX[i];
	    mpfY = 0.0;	for (i=1; i<=DoF; i++)    mpfY += V[i][m]*msY[i];
	    mpfZ = 0.0;	for (i=1; i<=DoF; i++)    mpfZ += V[i][m]*msZ[i];
	    fprintf(fp,"  MODE %5d:   f= %lf Hz,  T= %lf sec\n",m,f[m],1./f[m]);
	    fprintf(fp,"\t\tX- modal participation factor = %12.4e \n", mpfX);
	    fprintf(fp,"\t\tY- modal participation factor = %12.4e \n", mpfY);
	    fprintf(fp,"\t\tZ- modal participation factor = %12.4e \n", mpfZ);

	    fprintf(fp,"  Node    X-dsp       Y-dsp       Z-dsp");
	    fprintf(fp,"       X-rot       Y-rot       Z-rot\n");
	    for (j=1; j<= nN; j++) {
		fprintf(fp," %5d", j);
		for ( i=5; i>=0; i-- )	fprintf (fp, " %11.3e", V[6*j-i][m] );
		fprintf(fp,"\n");
	    }
	}

	fprintf(fp,"M A T R I X    I T E R A T I O N S: %d\n", iter );

	fs = sqrt(4.0*PI*PI*f[nM]*f[nM] + tol) / (2.0*PI);

	fprintf(fp,"There are %d modes below %f Hz.", -ok, fs );
	if ( -ok > nM ) {
		fprintf(fp," ... %d modes were not found.\n", -ok-nM );
		fprintf(fp," Try increasing the number of modes in \n");
		fprintf(fp," order to get the missing modes below %f Hz.\n",fs);
	} else  fprintf(fp," ... All %d modes were found.\n", nM );


	free_dvector(msX,1,DoF);
	free_dvector(msY,1,DoF);
	free_dvector(msZ,1,DoF);
	fflush(fp);
	return;
}


/*
 * STATIC_MESH  - create mesh data of deformed and undeformed mesh  22 Feb 1999 
 * use gnuplot	
 * useful gnuplot options: unset xtics ytics ztics border view key
 * This function illustrates how to read the internal force output data file.
 * The internal force output data file contains all the information required 
 * to plot deformed meshes, internal axial force, internal shear force, internal
 * torsion, and internal bending moment diagrams.
 */
void static_mesh(
		char IN_file[],
		char infcpath[], char meshpath[], char plotpath[],
		char *title, int nN, int nE, int nL, int lc, int DoF,
		vec3 *xyz, double *L,
		int *N1, int *N2, float *p, double *D, 
		double exagg_static, int D3_flag, int anlyz, float dx, float scale
){
	FILE	*fpif=NULL, *fpm=NULL;
	double	mx, my, mz; /* coordinates of the frame element number labels */
	char	fnif[FILENMAX], meshfl[FILENMAX],
		D2='#', D3='#',	/* indicates plotting in 2D or 3D	*/
		errMsg[MAXL],
		ch = 'a';
	int	sfrv=0,		/* *scanf return value			*/
		frel, nx,	/* frame element number, number of increments */
		n1, n2;		/* node numbers			*/
	float	x1, y1, z1,	/* coordinates of node n1		*/
		x2, y2, z2;	/* coordinates of node n2		*/
	int	j=0, m=0, n=0,
		X=0, Y=0, Z=0,
		lw = 1;		/*  line width of deformed mesh		*/
	time_t  now;		/* modern time variable type		*/

	(void) time(&now);

	// write gnuplot plotting script commands

	for ( j=1; j<=nN; j++ ) { // check for three-dimensional frame 
		if (xyz[j].x != 0.0) X=1;
		if (xyz[j].y != 0.0) Y=1;
		if (xyz[j].z != 0.0) Z=1;
	}
	if ( (X && Y && Z) || D3_flag ) {
		D3 = ' '; D2 = '#';
	} else {
		D3 = '#'; D2 = ' ';
	}

	if (lc <= 1) {	// open plotting script file for writing
	    if ((fpm = fopen (plotpath, "w")) == NULL) {
		sprintf (errMsg,"\n  error: cannot open gnuplot script file: %s \n", plotpath);
		errorMsg(errMsg);
		exit(23);
	    }
	} else {	// open plotting script file for appending
	    if ((fpm = fopen (plotpath, "a")) == NULL) {
		sprintf (errMsg,"\n  error: cannot open gnuplot script file: %s \n", plotpath);
		errorMsg(errMsg);
		exit(24);
	    }
	}

	// file name for deformed mesh data for load case "lc" 
	if ( lc >= 1 && anlyz )	sprintf( meshfl, "%sf.%03d", meshpath, lc );

	// write header, plot-setup cmds, node label, and element label data

	if (lc <= 1) {	// header & node number & element number labels

	 fprintf(fpm,"# FRAME3DD ANALYSIS RESULTS  http://frame3dd.sf.net/");
	 fprintf(fpm," VERSION %s \n", VERSION);
	 fprintf(fpm,"# %s\n", title );
	 fprintf(fpm,"# %s", ctime(&now) );
	 fprintf(fpm,"# G N U P L O T   S C R I P T   F I L E \n");
	 /* fprintf(fpm,"#  X=%d , Y=%d , Z=%d, D3=%d  \n", X,Y,Z,D3_flag); */

	 fprintf(fpm,"set autoscale\n");
	 fprintf(fpm,"unset border\n");
	 fprintf(fpm,"set pointsize 1.0\n");
	 fprintf(fpm,"set xtics; set ytics; set ztics; \n");
	 fprintf(fpm,"unset zeroaxis\n");
	 fprintf(fpm,"unset key\n");
	 fprintf(fpm,"unset label\n");
	 fprintf(fpm,"set size ratio -1    # 1:1 2D axis scaling \n");	
	 fprintf(fpm,"# set view equal xyz # 1:1 3D axis scaling \n");	

 	 fprintf(fpm,"# NODE NUMBER LABELS\n");
	 for (j=1; j<=nN; j++)
		fprintf(fpm,"set label ' %d' at %12.4e, %12.4e, %12.4e\n",
					j, xyz[j].x,xyz[j].y,xyz[j].z );

	 fprintf(fpm,"# ELEMENT NUMBER LABELS\n");
	 for (m=1; m<=nE; m++) {
		n1 = N1[m];	n2 = N2[m];
		mx = 0.5 * ( xyz[n1].x + xyz[n2].x );
		my = 0.5 * ( xyz[n1].y + xyz[n2].y );
		mz = 0.5 * ( xyz[n1].z + xyz[n2].z );
		fprintf(fpm,"set label ' %d' at %12.4e, %12.4e, %12.4e\n",
								m, mx, my, mz );
	 }

	 // 3D plot setup commands

	 fprintf(fpm,"%c set parametric\n", D3 );
	 fprintf(fpm,"%c set view 60, 70, %5.2f \n", D3, scale );
	 fprintf(fpm,"%c set view equal xyz # 1:1 3D axis scaling \n", D3 );
	 fprintf(fpm,"%c unset key\n", D3 );
	 fprintf(fpm,"%c set xlabel 'x'\n", D3 );
	 fprintf(fpm,"%c set ylabel 'y'\n", D3 );
	 fprintf(fpm,"%c set zlabel 'z'\n", D3 );
//	 fprintf(fpm,"%c unset label\n", D3 );

	} 

	// different plot title for each load case

	fprintf(fpm,"set title \"%s\\n", title );
	fprintf(fpm,"analysis file: %s ", IN_file );
	if ( anlyz ) {
		fprintf(fpm,"  deflection exaggeration: %.1f ", exagg_static );
		fprintf(fpm,"  load case %d of %d \"\n", lc, nL );
	} else {
		fprintf(fpm,"  data check only \"\n");
	}
	fprintf(fpm,"unset clip; \nset clip one; set clip two\n");
	fprintf(fpm,"set xyplane 0 \n"); // requires Gnuplot >= 4.6

	// 2D plot command

	fprintf(fpm,"%c plot '%s' u 2:3 t 'undeformed mesh' w lp ",
								D2, meshpath);
	if (!anlyz) fprintf(fpm,"lw %d lt 1 pt 6 \n", lw );
	else fprintf(fpm,"lw 1 lt 5 pt 6, '%s' u 1:2 t 'load case %d of %d' w l lw %d lt 3\n", meshfl, lc, nL, lw );

	// 3D plot command

	fprintf(fpm,"%c splot '%s' u 2:3:4 t 'load case %d of %d' w lp ",
							D3, meshpath, lc, nL );
	if (!anlyz) fprintf(fpm," lw %d lt 1 pt 6 \n", lw );
	else fprintf(fpm," lw 1 lt 5 pt 6, '%s' u 1:2:3 t 'load case %d of %d' w l lw %d lt 3\n",meshfl, lc, nL, lw );

	if ( lc < nL && anlyz )	fprintf(fpm,"pause -1\n");

	fclose(fpm);

	// write undeformed mesh data

	if (lc <= 1) {
	 // open the undeformed mesh data file for writing
	 if ((fpm = fopen (meshpath, "w")) == NULL) {
		sprintf (errMsg,"\n  error: cannot open gnuplot undeformed mesh data file: %s\n", meshpath );
		errorMsg(errMsg);
		exit(21);
	 }

	 fprintf(fpm,"# FRAME3DD ANALYSIS RESULTS  http://frame3dd.sf.net/");
	 fprintf(fpm," VERSION %s \n", VERSION);
	 fprintf(fpm,"# %s\n", title );
	 fprintf(fpm,"# %s", ctime(&now) );
	 fprintf(fpm,"# U N D E F O R M E D   M E S H   D A T A   (global coordinates)\n");
	 fprintf(fpm,"# Node        X            Y            Z \n");

	 for (m=1; m<=nE; m++) {
		n = N1[m];	// i = 6*(n-1);
		fprintf (fpm,"%5d %12.4e %12.4e %12.4e \n",
					n , xyz[n].x , xyz[n].y , xyz[n].z );
		n = N2[m];	// i = 6*(n-1);
		fprintf (fpm,"%5d %12.4e %12.4e %12.4e",
					n , xyz[n].x , xyz[n].y , xyz[n].z );
		fprintf (fpm,"\n\n\n");
	 }
	 fclose(fpm);
	}

	if (!anlyz) return; 	// no deformed mesh

	// write deformed mesh data

	// open the deformed mesh data file for writing 
	if ((fpm = fopen (meshfl, "w")) == NULL) {
		sprintf (errMsg,"\n  error: cannot open gnuplot deformed mesh data file %s \n", meshfl );
		errorMsg(errMsg);
		exit(22);
	}

	fprintf(fpm,"# FRAME3DD ANALYSIS RESULTS  http://frame3dd.sf.net/");
	fprintf(fpm," VERSION %s \n", VERSION);
	fprintf(fpm,"# %s\n", title );
	fprintf(fpm,"# L O A D  C A S E   %d  of   %d \n", lc, nL );
	fprintf(fpm,"# %s", ctime(&now) );
	fprintf(fpm,"# D E F O R M E D   M E S H   D A T A ");
	fprintf(fpm,"  deflection exaggeration: %.1f\n", exagg_static );
	fprintf(fpm,"#       X-dsp        Y-dsp        Z-dsp\n");
	
	// open the interior force data file for reading 
	if ( dx > 0.0 && anlyz ) {
	 // file name for internal force data for load case "lc" 
	 sprintf( fnif, "%s%02d", infcpath, lc );
	 if ((fpif = fopen (fnif, "r")) == NULL) {
          sprintf (errMsg,"\n  error: cannot open interior force data file: %s \n",fnif);
	  errorMsg(errMsg);
          exit(20);
	 }
	}

	for (m=1; m<=nE; m++) {	// write deformed shape data for each element

		ch = 'a'; 

		fprintf( fpm, "\n# element %5d \n", m );
		if ( dx < 0.0 && anlyz ) {
			cubic_bent_beam ( fpm,
				N1[m],N2[m], xyz, L[m],p[m], D, exagg_static );
		} 
		if ( dx > 0.0 && anlyz ) {
			while ( ch != '@' )	ch = getc(fpif);
			sfrv=fscanf(fpif,"%d %d %d %f %f %f %f %f %f %d",
			 &frel, &n1, &n2, &x1, &y1, &z1, &x2, &y2, &z2, &nx);
			if (sfrv != 10) sferr(fnif);
			if ( frel != m || N1[m] != n1 || N2[m] != n2 ) {
			 fprintf(stderr," error in static_mesh parsing\n");
			 fprintf(stderr,"  frel = %d; m = %d; nx = %d \n", frel,m,nx );
			}
			/* debugging ... check mesh data 
			printf("  frel = %3d; m = %3d; n1 =%4d; n2 = %4d; nx = %3d L = %f \n", frel,m,n1,n2,nx,L[m] );
			*/
			while ( ch != '~' )	ch = getc(fpif);
			force_bent_beam ( fpm, fpif, fnif, nx, 
				N1[m],N2[m], xyz, L[m],p[m], D, exagg_static );
		}

	}

	if ( dx > 0.0 && anlyz ) fclose(fpif);

	fclose(fpm);

	return;
}


/*
 * MODAL_MESH  -  create mesh data of the mode-shape meshes, use gnuplot	19oct98
 * useful gnuplot options: unset xtics ytics ztics border view key
 */
void modal_mesh(
		char IN_file[], char meshpath[], char modepath[],
		char plotpath[], char *title,
		int nN, int nE, int DoF, int nM,
		vec3 *xyz, double *L,
		int *J1, int *J2, float *p,
		double **M, double *f, double **V,
		double exagg_modal, int D3_flag, int anlyz
){
	FILE	*fpm;
	double mpfX, mpfY, mpfZ;	/* mode participation factors	*/
	double *msX, *msY, *msZ;
	double *v;		/* a mode-shape vector */

	int	i, j, m,n, X=0, Y=0, Z=0;
	int	lw = 1;		/*  line thickness of deformed mesh	*/
	char	D2='#', D3 = '#',	/* indicate 2D or 3D frame	*/
		modefl[FILENMAX],
		errMsg[MAXL];


	msX = dvector(1,DoF);
	msY = dvector(1,DoF);
	msZ = dvector(1,DoF);
	v   = dvector(1,DoF);

	for (i=1; i<=DoF; i++) {	/* modal participation factors */
		msX[i] = msY[i] = msZ[i] = 0.0;
		for (j=1; j<=DoF; j+=6) msX[i] += M[i][j];
		for (j=2; j<=DoF; j+=6) msY[i] += M[i][j];
		for (j=3; j<=DoF; j+=6) msZ[i] += M[i][j];
	}

	if (!anlyz) exagg_modal = 0.0;

	for (m=1; m<=nM; m++) {

		sprintf( modefl,"%s-%02d-", modepath, m );

		if ((fpm = fopen (modefl, "w")) == NULL) {
			sprintf (errMsg,"\n  error: cannot open gnuplot modal mesh file: %s \n", modefl);
			errorMsg(errMsg);
			exit(27);
		}

		fprintf(fpm,"# FRAME3DD ANALYSIS RESULTS  http://frame3dd.sf.net/");
		fprintf(fpm," VERSION %s \n", VERSION);
		fprintf(fpm,"# %s\n", title );
		fprintf(fpm,"# M O D E   S H A P E   D A T A   F O R   M O D E");
		fprintf(fpm,"   %d\t(global coordinates)\n", m );
		fprintf(fpm,"# deflection exaggeration: %.1f\n\n", exagg_modal );
		mpfX = 0.0;	for (i=1; i<=DoF; i++)    mpfX += V[i][m]*msX[i];
		mpfY = 0.0;	for (i=1; i<=DoF; i++)    mpfY += V[i][m]*msY[i];
		mpfZ = 0.0;	for (i=1; i<=DoF; i++)    mpfZ += V[i][m]*msZ[i];
		fprintf(fpm,"# MODE %5d:   f= %lf Hz, T= %lf sec\n", m,f[m],1./f[m]);
		fprintf(fpm,"#\t\tX- modal participation factor = %12.4e \n", mpfX);
		fprintf(fpm,"#\t\tY- modal participation factor = %12.4e \n", mpfY);
		fprintf(fpm,"#\t\tZ- modal participation factor = %12.4e \n", mpfZ);

		for(i=1; i<=DoF; i++)	v[i] = V[i][m];

		fprintf(fpm,"#      X-dsp       Y-dsp       Z-dsp\n\n");

		for(n=1; n<=nE; n++) {
			fprintf( fpm, "\n# element %5d \n", n );
			cubic_bent_beam ( fpm, J1[n], J2[n], xyz, L[n], p[n], v, exagg_modal );
		}

		fclose(fpm);

		for ( j=1; j<=nN; j++ ) { // check for three-dimensional frame
			if (xyz[j].x != 0.0) X=1;
			if (xyz[j].y != 0.0) Y=1;
			if (xyz[j].z != 0.0) Z=1;
		}

		if ( (X && Y && Z) || D3_flag ) {
			D3 = ' '; D2 = '#';
		} else {
			D3 = '#'; D2 = ' ';
		}


		if ((fpm = fopen (plotpath, "a")) == NULL) {
			sprintf (errMsg,"\n  error: cannot append gnuplot script file: %s \n",plotpath);
			errorMsg(errMsg);
			exit(25);
		}

		fprintf(fpm,"pause -1\n");

		if (m==1) {
			fprintf(fpm,"unset label\n");
			fprintf(fpm,"%c unset key\n", D3 );
		}

		fprintf(fpm,"set title '%s     mode %d     %lf Hz'\n",IN_file,m,f[m]);

		// 2D plot command

		fprintf(fpm,"%c plot '%s' u 2:3 t 'undeformed mesh' w l ", D2, meshpath );
		if (!anlyz) fprintf(fpm," lw %d lt 1 \n", lw );
		else fprintf(fpm," lw 1 lt 5 , '%s' u 1:2 t 'mode-shape %d' w l lw %d lt 3\n", modefl, m, lw );

		// 3D plot command 

		fprintf(fpm,"%c splot '%s' u 2:3:4 t 'undeformed mesh' w l ",
								D3, meshpath);
		if (!anlyz) fprintf(fpm," lw %d lt 1 \n", lw );
		else fprintf(fpm," lw 1 lt 5 , '%s' u 1:2:3 t 'mode-shape %d' w l lw %d lt 3\n", modefl, m, lw );

		fclose(fpm);

	}

	free_dvector(msX,1,DoF);
	free_dvector(msY,1,DoF);
	free_dvector(msZ,1,DoF);
	free_dvector(v,1,DoF);
}


/*
 * ANIMATE -  create mesh data of animated mode-shape meshes, use gnuplot	16dec98
 * useful gnuplot options: unset xtics ytics ztics border view key
 * mpeg movie example:   % convert mesh_file-03-f-*.ps mode-03.mpeg
 * ... requires ImageMagick and mpeg2vidcodec packages
 */
void animate(
	char IN_file[], char meshpath[], char modepath[], char plotpath[],
	char *title,
	int anim[],
	int nN, int nE, int DoF, int nM,
	vec3 *xyz, double *L, float *p,
	int *J1, int *J2, double *f, double **V,
	double exagg_modal, int D3_flag, 
	float pan,		/* pan rate for animation	     */
	float scale		/* inital zoom scale in 3D animation */
){
	FILE	*fpm;

	float	x_min = 0.0, x_max = 0.0,
		y_min = 0.0, y_max = 0.0,
		z_min = 0.0, z_max = 0.0,
		Dxyz = 0.0, 		/* "diameter" of the structure	*/
		rot_x_init  =  70.0,	/* inital x-rotation in 3D animation */
		rot_x_final =  60.0,	/* final  x-rotation in 3D animation */
		rot_z_init  = 100.0,	/* inital z-rotation in 3D animation */
		rot_z_final = 120.0,	/* final  z-rotation in 3D animation */
		zoom_init  = 1.0*scale,	/* init.  zoom scale in 3D animation */
		zoom_final = 1.1*scale, /* final  zoom scale in 3D animation */
		frames = 25;		/* number of frames in animation */

	double	ex=10,		/* an exageration factor, for animation */
		*v;

	int	fr, i,j, m,n, X=0, Y=0, Z=0, c, CYCLES=3,
		frame_number = 0,
		lw = 1,		/*  line thickness of deformed mesh	*/
		total_frames;	/* total number of frames in animation	*/

	char	D2 = '#', D3 = '#',	/* indicate 2D or 3D frame	*/
		Movie = '#',	/* use '#' for no-movie  -OR-  ' ' for movie */
		modefl[FILENMAX], framefl[FILENMAX];
	char	errMsg[MAXL];

	for (j=1; j<=nN; j++) {		// check for three-dimensional frame
		if (xyz[j].x != 0.0) X=1;
		if (xyz[j].y != 0.0) Y=1;
		if (xyz[j].z != 0.0) Z=1;
		if (j==1) {
			x_min = x_max = xyz[j].x;
			y_min = y_max = xyz[j].y;
			z_min = z_max = xyz[j].z;
		}
		if (xyz[j].x < x_min ) x_min = xyz[j].x;
		if (xyz[j].y < y_min ) y_min = xyz[j].y;
		if (xyz[j].z < z_min ) z_min = xyz[j].z;
		if ( x_max < xyz[j].x ) x_max = xyz[j].x;
		if ( y_max < xyz[j].y ) y_max = xyz[j].y;
		if ( z_max < xyz[j].z ) z_max = xyz[j].z;
	}
	if ( (X && Y && Z) || D3_flag ) {
		D3 = ' '; D2 = '#';
	} else {
		D3 = '#'; D2 = ' ';
	}

	Dxyz = sqrt( (x_max-x_min)*(x_max-x_min) + (y_max-y_min)*(y_max-y_min) + (z_max-z_min)*(z_max-z_min) );


	if ((fpm = fopen (plotpath, "a")) == NULL) {
		sprintf (errMsg,"\n  error: cannot append gnuplot script file: %s \n",plotpath);
		errorMsg(errMsg);
		exit(26);
	}
	i = 1;
	while ( (m = anim[i]) != 0 && i < 100) {
	 if ( i==1 ) {

	   fprintf(fpm,"\n# --- M O D E   S H A P E   A N I M A T I O N ---\n");
	   fprintf(fpm,"# rot_x_init  = %7.2f\n", rot_x_init ); 
	   fprintf(fpm,"# rot_x_final = %7.2f\n", rot_x_final ); 
	   fprintf(fpm,"# rot_z_init  = %7.2f\n", rot_z_init ); 
	   fprintf(fpm,"# rot_z_final = %7.2f\n", rot_z_final ); 
	   fprintf(fpm,"# zoom_init   = %7.2f\n", zoom_init ); 
	   fprintf(fpm,"# zoom_final  = %7.2f\n", zoom_init );
	   fprintf(fpm,"# pan rate    = %7.2f \n", pan );
	   fprintf(fpm,"set autoscale\n");
	   fprintf(fpm,"unset border\n");
	   fprintf(fpm,"%c unset xlabel \n", D3 );
	   fprintf(fpm,"%c unset ylabel \n", D3 );
	   fprintf(fpm,"%c unset zlabel \n", D3 );
	   fprintf(fpm,"%c unset label \n", D3 );
	   fprintf(fpm,"unset key\n");
	   fprintf(fpm,"%c set parametric\n", D3 );

	   fprintf(fpm,"# x_min = %12.5e     x_max = %12.5e \n", x_min, x_max);
	   fprintf(fpm,"# y_min = %12.5e     y_max = %12.5e \n", y_min, y_max);
	   fprintf(fpm,"# z_min = %12.5e     z_max = %12.5e \n", z_min, z_max);
	   fprintf(fpm,"# Dxyz = %12.5e \n", Dxyz );
	   fprintf(fpm,"set xrange [ %lf : %lf ] \n",
			x_min-0.2*Dxyz, x_max+0.1*Dxyz );
	   fprintf(fpm,"set yrange [ %lf : %lf ] \n",
			y_min-0.2*Dxyz, y_max+0.1*Dxyz );
	   fprintf(fpm,"set zrange [ %lf : %lf ] \n",
			z_min-0.2*Dxyz, z_max+0.1*Dxyz );

/*
 *	   if ( x_min != x_max )
 *		fprintf(fpm,"set xrange [ %lf : %lf ] \n",
 *	 		x_min-0.2*(x_max-x_min), x_max+0.2*(x_max-x_min) );
 *	   else fprintf(fpm,"set xrange [ %lf : %lf ] \n",
 *			x_min-exagg_modal, x_max+exagg_modal );
 *	   if (y_min != y_max)
 *		fprintf(fpm,"set yrange [ %lf : %lf ] \n",
 *	 		y_min-0.2*(y_max-y_min), y_max+0.2*(y_max-y_min) );
 *	   else fprintf(fpm,"set yrange [ %lf : %lf ] \n",
 *			y_min-exagg_modal, y_max+exagg_modal );
 *	   if (z_min != z_max)
 *	   	fprintf(fpm,"set zrange [ %lf : %lf ] \n",
 *			z_min-0.2*(z_max-z_min), z_max+0.2*(z_max-z_min) );
 *	   else fprintf(fpm,"set zrange [ %lf : %lf ] \n",
 *			z_min-exagg_modal, z_max+exagg_modal );
 */

	   fprintf(fpm,"unset xzeroaxis; unset yzeroaxis; unset zzeroaxis\n");
	   fprintf(fpm,"unset xtics; unset ytics; unset ztics; \n");
	   fprintf(fpm,"%c set view 60, 70, %5.2f \n", D3, scale );
	   fprintf(fpm,"set size ratio -1    # 1:1 2D axis scaling \n");	
	   fprintf(fpm,"%c set view equal xyz # 1:1 3D axis scaling \n", D3 );

	 }

	 fprintf(fpm,"pause -1 \n");
	 fprintf(fpm,"set title '%s     mode %d      %lf Hz'\n",IN_file,m,f[m]);

	 frame_number = 0;
	 total_frames = 2*CYCLES*frames;
	 for ( c=1; c <= CYCLES; c++ ) {

	  for ( fr=0; fr<=frames; fr++ ) {

	    ++frame_number;

	    sprintf(modefl,"%s-%02d.%03d", modepath, m, fr  );
	    sprintf(framefl,"%s-%02d-f-%03d.ps", modepath, m, fr  );

	    fprintf(fpm,"%c plot '%s' u 2:3 w l lw 1 lt 5, ", D2,meshpath );
	    fprintf(fpm," '%s' u 1:2 w l lw %d lt 3 ; \n", modefl, lw );
	    if ( pan != 0.0 )
	     fprintf(fpm,"%c set view %7.2f, %7.2f, %5.3f # pan = %f\n", D3,
		rot_x_init + pan*(rot_x_final-rot_x_init)*frame_number/total_frames,
		rot_z_init + pan*(rot_z_final-rot_z_init)*frame_number/total_frames,
		zoom_init + pan*(zoom_final-zoom_init)*frame_number/total_frames, pan );
	    fprintf(fpm,"%c splot '%s' u 2:3:4 w l lw 1 lt 5, ",D3,meshpath);
            fprintf(fpm," '%s' u 1:2:3 w l lw %d lt 3;", modefl, lw );

	    if ( fr==0 && c==1 )	fprintf(fpm,"  pause 1.5 \n");
	    else			fprintf(fpm,"  pause 0.05 \n");
	    fprintf(fpm,"%c  load 'saveplot';\n",Movie);
	    fprintf(fpm,"%c  !mv my-plot.ps %s\n", Movie, framefl );
	  }
	  for ( fr = frames-1; fr > 0; fr-- ) {

	    ++frame_number;

	    sprintf(modefl,"%s-%02d.%03d", modepath, m, fr  );
	    sprintf(framefl,"%s-%02d-f-%03d.ps", modepath, m, fr  );

	    fprintf(fpm,"%c plot '%s' u 2:3 w l lw 1 lt 5, ", D2,meshpath );
	    fprintf(fpm," '%s' u 1:2 w l lw %d lt 3; \n", modefl, lw );
	    if ( pan != 0.0 )
	     fprintf(fpm,"%c set view %7.2f, %7.2f, %5.3f # pan = %f\n", D3,
		rot_x_init + pan*(rot_x_final-rot_x_init)*frame_number/total_frames,
		rot_z_init + pan*(rot_z_final-rot_z_init)*frame_number/total_frames,
		zoom_init + pan*(zoom_final-zoom_init)*frame_number/total_frames, pan );
	    fprintf(fpm,"%c splot '%s' u 2:3:4 w l lw 1 lt 5, ",D3,meshpath);
	    fprintf(fpm," '%s' u 1:2:3 w l lw %d lt 3;", modefl, lw );
	    fprintf(fpm,"  pause 0.05 \n");
	    fprintf(fpm,"%c  load 'saveplot';\n",Movie);
	    fprintf(fpm,"%c  !mv my-plot.ps %s\n", Movie, framefl );
	  }

	 }
	 fr = 0;

	 sprintf(modefl,"%s-%02d.%03d", modepath, m, fr  );

	 fprintf(fpm,"%c plot '%s' u 2:3 w l lw %d lt 5, ", D2, meshpath, lw );
	 fprintf(fpm," '%s' u 1:2 w l lw 3 lt 3 \n", modefl );
	 fprintf(fpm,"%c splot '%s' u 2:3:4 w l lw %d lt 5, ",D3,meshpath, lw );
	 fprintf(fpm," '%s' u 1:2:3 w l lw 3 lt 3 \n", modefl );

	 i++;
	}
	fclose(fpm);

	v = dvector(1,DoF);

	i = 1;
	while ( (m = anim[i]) != 0 ) {
	  for ( fr=0; fr<=frames; fr++ ) {

	    sprintf(modefl,"%s-%02d.%03d", modepath, m, fr  );

	    if ((fpm = fopen (modefl, "w")) == NULL) {
		sprintf (errMsg,"\n  error: cannot open gnuplot modal mesh data file: %s \n", modefl);
		errorMsg(errMsg);
		exit(28);
	    }

	    ex = exagg_modal*cos( PI*fr/frames );

	    fprintf(fpm,"# FRAME3DD ANALYSIS RESULTS  http://frame3dd.sf.net/");
	    fprintf(fpm," VERSION %s \n", VERSION);
	    fprintf(fpm,"# %s\n", title );
	    fprintf(fpm,"# A N I M A T E D   M O D E   S H A P E   D A T A \n");
	    fprintf(fpm,"# deflection exaggeration: %.1f\n", ex );
	    fprintf(fpm,"# MODE %5d: f= %lf Hz  T= %lf sec\n\n",m,f[m],1./f[m]);

	    for (j=1; j<=DoF; j++)	v[j] = V[j][m];		/* mode "m" */

	    fprintf(fpm,"#      X-dsp       Y-dsp       Z-dsp\n\n");

	    for (n=1; n<=nE; n++) {
		fprintf( fpm, "\n# element %5d \n", n );
		cubic_bent_beam ( fpm, J1[n], J2[n], xyz, L[n], p[n], v, ex );
	    }

	    fclose(fpm);
	  }
	  i++;
	}

	free_dvector(v,1,DoF);

	return;
}


/*
 * CUBIC_BENT_BEAM  -  computes cubic deflection functions from end deflections
 * and end rotations.  Saves deflected shapes to a file.  These bent shapes
 * are exact for mode-shapes, and for frames loaded at their nodes.
 * 15 May 2009
 */
void cubic_bent_beam(
	FILE *fpm, int n1, int n2, vec3 *xyz,
	double L, float p, double *D, double exagg
){
	double	t1, t2, t3, t4, t5, t6, t7, t8, t9, 	/* coord xfmn	*/
		u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12,
		*a, *b, **A,
		s, v, w, dX, dY, dZ;
	int	i1, i2, pd;
	char	errMsg[MAXL];

	A = dmatrix(1,4,1,4);
	a = dvector(1,4);
	b = dvector(1,4);

	coord_trans ( xyz, L, n1, n2,
			&t1, &t2, &t3, &t4, &t5, &t6, &t7, &t8, &t9, p );

	i1 = 6*(n1-1);	i2 = 6*(n2-1);

		/* compute end deflections in local coordinates */

	u1  = exagg*(t1*D[i1+1] + t2*D[i1+2] + t3*D[i1+3]);
	u2  = exagg*(t4*D[i1+1] + t5*D[i1+2] + t6*D[i1+3]);
	u3  = exagg*(t7*D[i1+1] + t8*D[i1+2] + t9*D[i1+3]);

	u4  = exagg*(t1*D[i1+4] + t2*D[i1+5] + t3*D[i1+6]);
	u5  = exagg*(t4*D[i1+4] + t5*D[i1+5] + t6*D[i1+6]);
	u6  = exagg*(t7*D[i1+4] + t8*D[i1+5] + t9*D[i1+6]);

	u7  = exagg*(t1*D[i2+1] + t2*D[i2+2] + t3*D[i2+3]);
	u8  = exagg*(t4*D[i2+1] + t5*D[i2+2] + t6*D[i2+3]);
	u9  = exagg*(t7*D[i2+1] + t8*D[i2+2] + t9*D[i2+3]);

	u10 = exagg*(t1*D[i2+4] + t2*D[i2+5] + t3*D[i2+6]);
	u11 = exagg*(t4*D[i2+4] + t5*D[i2+5] + t6*D[i2+6]);
	u12 = exagg*(t7*D[i2+4] + t8*D[i2+5] + t9*D[i2+6]);

		/* curve-fitting problem for a cubic polynomial */

	a[1] =  u2;		b[1] =  u3;
	a[2] =  u8;   		b[2] =  u9;
	a[3] =  u6;		b[3] = -u5;
	a[4] =  u12;		b[4] = -u11;

	u7 += L;
	A[1][1] = 1.0;   A[1][2] = u1;   A[1][3] = u1*u1;   A[1][4] = u1*u1*u1;
	A[2][1] = 1.0;   A[2][2] = u7;   A[2][3] = u7*u7;   A[2][4] = u7*u7*u7;
	A[3][1] = 0.0;   A[3][2] = 1.;   A[3][3] = 2.*u1;   A[3][4] = 3.*u1*u1;
	A[4][1] = 0.0;   A[4][2] = 1.;   A[4][3] = 2.*u7;   A[4][4] = 3.*u7*u7;
	u7 -= L;

	lu_dcmp ( A, 4, a, 1, 1, &pd );		/* solve for cubic coef's */

	if (!pd) {
	 sprintf(errMsg," n1 = %d  n2 = %d  L = %e  u7 = %e \n", n1,n2,L,u7);
	 errorMsg(errMsg);
	 exit(30);
	}

	lu_dcmp ( A, 4, b, 0, 1, &pd );		/* solve for cubic coef's */

	// debug ... if deformed mesh exageration is too big, some elements
	// may not be plotted.  
	//fprintf( fpm, "# u1=%e  L+u7=%e, dx = %e \n",
	//				u1, fabs(L+u7), fabs(L+u7-u1)/10.0); 
	for ( s = u1; fabs(s) <= 1.01*fabs(L+u7); s += fabs(L+u7-u1) / 10.0 ) {

			/* deformed shape in local coordinates */
		v = a[1] + a[2]*s + a[3]*s*s + a[4]*s*s*s;
		w = b[1] + b[2]*s + b[3]*s*s + b[4]*s*s*s;

			/* deformed shape in global coordinates */
		dX = t1*s + t4*v + t7*w;
		dY = t2*s + t5*v + t8*w;
		dZ = t3*s + t6*v + t9*w;

		fprintf (fpm," %12.4e %12.4e %12.4e\n",
			xyz[n1].x + dX , xyz[n1].y + dY , xyz[n1].z + dZ );
	}
	fprintf(fpm,"\n\n");

	free_dmatrix(A,1,4,1,4);
	free_dvector(a,1,4);
	free_dvector(b,1,4);

	return;
}


/*
 * FORCE_BENT_BEAM  -  reads internal frame element forces and deflections
 * from the internal force and deflection data file.  
 * Saves deflected shapes to a file.  These bent shapes are exact. 
 * Note: It would not be difficult to adapt this function to plot
 * internal axial force, shear force, torques, or bending moments. 
 * 9 Jan 2010
 */
void force_bent_beam(
	FILE *fpm, FILE *fpif, char fnif[], int nx, int n1, int n2, vec3 *xyz,
	double L, float p, double *D, double exagg
){
	double	t1, t2, t3, t4, t5, t6, t7, t8, t9; 	/* coord xfmn	*/
	double	xi, dX, dY, dZ;
	float	x, Nx, Vy, Vz, Tx, My, Mz, Dx, Dy, Dz, Rx;
	double	Lx, Ly, Lz;
	int	n;
	int	sfrv=0;		/* *scanf return value	*/

	Lx = xyz[n2].x - xyz[n1].x;
	Ly = xyz[n2].y - xyz[n1].y;
	Lz = xyz[n2].z - xyz[n1].z;

	coord_trans ( xyz, L, n1, n2,
			&t1, &t2, &t3, &t4, &t5, &t6, &t7, &t8, &t9, p );

	x = -1.0;
	n = 0;
	for ( xi = 0; xi <= 1.01*L && n < nx; xi += 0.10*L ) {

		while ( x < xi && n < nx ) {
		    /* read the deformed shape in local coordinates */
		    sfrv=fscanf(fpif,"%f %f %f %f %f %f %f %f %f %f %f",
			&x, &Nx, &Vy, &Vz, &Tx, &My, &Mz, &Dx, &Dy, &Dz, &Rx );
//		    printf("x = %12.4f\n", x );		/* debug */
		    if (sfrv != 11) sferr(fnif);
		    ++n;
		} 

		/* exaggerated deformed shape in global coordinates */
		dX = exagg * ( t1*Dx + t4*Dy + t7*Dz );
		dY = exagg * ( t2*Dx + t5*Dy + t8*Dz );
		dZ = exagg * ( t3*Dx + t6*Dy + t9*Dz );

		fprintf (fpm," %12.4e %12.4e %12.4e\n",
					xyz[n1].x + (x/L)*Lx + dX ,
					xyz[n1].y + (x/L)*Ly + dY ,
					xyz[n1].z + (x/L)*Lz + dZ );

//		printf("...  x = %7.3f  n = %3d  Dx = %10.3e   Dy = %10.3e   Dz = %10.3e \n", x,n,Dx,Dy,Dz ); /* debug */
//		printf("                           dX = %10.3e   dY = %10.3e   dZ = %10.3e \n", dX,dY,dZ ); /* debug */

	}

	fprintf(fpm,"\n\n");

	return;
}


/*
 * SFERR  -  Display error message upon an erronous *scanf operation
 *
void sferr ( char s[] ) {
	char	errMsg[MAXL];
	sprintf(errMsg,">> Input Data file error while reading %s\n",s);
	errorMsg(errMsg);
	return;
}
*/


/*
 * MY_ITOA  -  Convert an integer n to charcters in s, from K&R, 1978,   p. 59-60
 * ... specialized for portability between GNU GCC and DJGPP GCC
 */
void my_itoa ( int n, char s[], int k ) {
	int	c, i, j, sign;

	if ((sign = n) < 0) 		/* record sign */
		n = -n;			/* make n positive */
	i = 0;
	do {				/* generate digits in reverse order */
		s[i++] = n % 10 + '0';	/* get next digit */
	} while ((n /= 10) > 0);	/* delete it */
	for (;i<k;)	s[i++] = '0';	/* add leading '0' */
	if (sign < 0)
		s[i++] = '-';
	s[i] = '\0';
					/* reverse order of string s */
	j = 0;
	while ( s[j] != '\0' )	j++;	/* j is length of s - 1 */
	--j;

	for (i = 0; i < j; i++, j--) {
		c = s[i];
		s[i] = s[j];
		s[j] = c;
	}
	return;
}


/*
 * GET_FILE_EXT  -  get the file extension,
 *		return 1 if the extension is ".csv"
 *		return 2 if the extension is ".fmm"
 *		return 0 otherwise
 */
int get_file_ext( char *filename, char *ext )
{
	int	i=0, full_len=0, len=0;

	while ( filename[len++] != '\0' ) /* the length of file filename */ ;
	full_len = len;
	while ( filename[len--] != '.' && len > 0 ) /* the last '.' in filename */ ;
	if ( len == 0 )	len = full_len;
	++len;

	for ( i=0; len < full_len; i++,len++ ) ext[i] = tolower(filename[len]);

	/* debugging ... check file names
	printf(" filename '%s' has length %d and extension = '%s' \n",
							filename, len, ext);
	printf(" Is .CSV? ... = %d \n", !strcmp(ext,".csv") );
	*/

	if ( !strcmp(ext,".csv") ) return (1);
	if ( !strcmp(ext,".fmm") ) return (2);
	return(0);
}


/*
 * DOTS  -  print a set of dots (periods)
 */
void dots ( FILE *fp, int n ) {
	int i;
	for (i=1; i<=n; i++)	fprintf(fp,".");
}


/*
 * EVALUATE -  displays a randomly-generated goodbye message.  
 */
void evaluate ( float error, float rms_resid, float tol, int geom ) 
{
	int r;

	r = rand() % 10;

	color(0);
	fprintf(stdout,"  RMS relative equilibrium error  = %9.3e ", error );
	if ( error < tol ) {
		fprintf(stdout," < tol = %7.1e ", tol );
		(void) fflush(stdout);
		textColor('y','b','b','x');
		fprintf(stdout," ** converged ** ");
	} if ( error > tol ) {
		fprintf(stdout," > tol = %7.1e ", tol );
		(void) fflush(stdout);
		textColor('y','r','b','x');
		fprintf(stdout," !! not converged !! ");
	}
	(void) fflush(stdout);
	color(0);	
	fprintf(stdout,"\n");
	fprintf(stdout,"  RMS residual incremental displ. = %9.3e ", rms_resid);
	dots(stdout,17);
	(void) fflush(stdout);

	if ( rms_resid < 1e-24 ) {

	    textColor('y','b','b','x');
	    switch ( r ) {
		case 0: fprintf(stdout," * brilliant!  * "); break; 
		case 1: fprintf(stdout," *  chuffed!   * "); break; 
		case 2: fprintf(stdout," *  woo-hoo!   * "); break; 
		case 3: fprintf(stdout," *  wicked!    * "); break; 
		case 4: fprintf(stdout," *   beaut!    * "); break; 
		case 5: fprintf(stdout," *   flash!    * "); break; 
		case 6: fprintf(stdout," *  well done! * "); break; 
		case 7: fprintf(stdout," *  priceless! * "); break; 
		case 8: fprintf(stdout," *  sweet as!  * "); break; 
		case 9: fprintf(stdout," *good as gold!* "); break; 
	    }
	    (void) fflush(stdout);
	    color(0);	
	    fprintf(stdout,"\n");
	    return;
	}

	if ( rms_resid < 1e-16 ) {

	    textColor('y','g','b','x');
	    switch ( r ) {
		case 0: fprintf(stdout,"   acceptable!   "); break; 
		case 1: fprintf(stdout,"      bling!     "); break; 
		case 2: fprintf(stdout,"  that will do!  "); break; 
		case 3: fprintf(stdout,"   not shabby!   "); break; 
		case 4: fprintf(stdout,"   reasonable!   "); break; 
		case 5: fprintf(stdout,"   very good!    "); break; 
		case 6: fprintf(stdout,"   up to snuff!  "); break; 
		case 7: fprintf(stdout,"     bully!      "); break; 
		case 8: fprintf(stdout,"      nice!      "); break; 
		case 9: fprintf(stdout,"     choice!     "); break; 
	    }
	    (void) fflush(stdout);
	    color(0);	
	    fprintf(stdout,"\n");
	    return;
	}
	
	if ( rms_resid < 1e-12 ) {

	    textColor('y','c','b','x');
	    switch ( r ) {
		case 0: fprintf(stdout," adequate. "); break; 
		case 1: fprintf(stdout," passable. "); break; 
		case 2: fprintf(stdout," all right. "); break; 
		case 3: fprintf(stdout," ok. "); break; 
		case 4: fprintf(stdout," not bad. "); break; 
		case 5: fprintf(stdout," fine. "); break; 
		case 6: fprintf(stdout," fair. "); break; 
		case 7: fprintf(stdout," respectable. "); break; 
		case 8: fprintf(stdout," tolerable. "); break; 
		case 9: fprintf(stdout," just ok. "); break; 
	    }
	    (void) fflush(stdout);
	    color(0);	
	    fprintf(stdout,"\n");
	    return;
	}

	if ( rms_resid > 1e-12 ) {

	    textColor('y','r','b','x');
	    switch ( r ) {
		case 0: fprintf(stdout," abominable! "); break; 
		case 1: fprintf(stdout," puckeroo! "); break; 
		case 2: fprintf(stdout," atrocious! "); break; 
		case 3: fprintf(stdout," not ok! "); break; 
		case 4: fprintf(stdout," wonky! "); break; 
		case 5: fprintf(stdout," crappy! "); break; 
		case 6: fprintf(stdout," oh noooo! "); break; 
		case 7: fprintf(stdout," abominable! "); break; 
		case 8: fprintf(stdout," munted! "); break; 
		case 9: fprintf(stdout," awful! "); break; 
	    }
	    (void) fflush(stdout);
	    color(0);	
	    fprintf(stdout,"\n");
	    return;
	}

}

