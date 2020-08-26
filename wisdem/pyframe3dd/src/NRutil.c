/** @file
	Memory allocation functions from Numerical Recipes in C, by Press,
	Cambridge University Press, 1988
	http://www.nr.com/public-domain.html
*/


#include "NRutil.h"

#if defined(__STDC__) || defined(ANSI) || defined(NRANSI) /* ANSI */

#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define NR_END 1
#define FREE_ARG char*

void NRerror(char error_text[])
/* Numerical Recipes standard error handler */
{
	fprintf(stderr,"Numerical Recipes run-time error...\n");
	fprintf(stderr,"%s\n",error_text);
	fprintf(stderr,"...now exiting to system...\n");
	exit(1000);
}

float *vector(long nl, long nh)
/* allocate a float vector with subscript range v[nl..nh] */
{
	float *v;

	v=(float *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(float)));
	if (!v) NRerror("allocation failure in vector()");
	return v-nl+NR_END;
}

int *ivector(long nl, long nh)
/* allocate an int vector with subscript range v[nl..nh] */
{
	int *v;

	v=(int *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(int)));
	if (!v) NRerror("allocation failure in ivector()");
	return v-nl+NR_END;
}

unsigned char *cvector(long nl, long nh)
/* allocate an unsigned char vector with subscript range v[nl..nh] */
{
	unsigned char *v;

	v=(unsigned char *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(unsigned char)));
	if (!v) NRerror("allocation failure in cvector()");
	return v-nl+NR_END;
}

unsigned long *lvector(long nl, long nh)
/* allocate an unsigned long vector with subscript range v[nl..nh] */
{
	unsigned long *v;

	v=(unsigned long *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(long)));
	if (!v) NRerror("allocation failure in lvector()");
	return v-nl+NR_END;
}

double *dvector(long nl, long nh)
/* allocate a double vector with subscript range v[nl..nh] */
{
	double *v;

	v=(double *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(double)));
	if (!v) NRerror("allocation failure in dvector()");
	return v-nl+NR_END;
}

float **matrix(long nrl, long nrh, long ncl, long nch)
/* allocate a float matrix with subscript range m[nrl..nrh][ncl..nch] */
{
	long i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
	float **m;

	/* allocate pointers to rows */
	m=(float **) malloc((size_t)((nrow+NR_END)*sizeof(float*)));
	if (!m) NRerror("allocation failure 1 in matrix()");
	m += NR_END;
	m -= nrl;

	/* allocate rows and set pointers to them */
	m[nrl]=(float *) malloc((size_t)((nrow*ncol+NR_END)*sizeof(float)));
	if (!m[nrl]) NRerror("allocation failure 2 in matrix()");
	m[nrl] += NR_END;
	m[nrl] -= ncl;

	for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;

	/* return pointer to array of pointers to rows */
	return m;
}

double **dmatrix(long nrl, long nrh, long ncl, long nch)
/* allocate a double matrix with subscript range m[nrl..nrh][ncl..nch] */
{
	long i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
	double **m;

	/* allocate pointers to rows */
	m=(double **) malloc((size_t)((nrow+NR_END)*sizeof(double*)));
	if (!m) NRerror("allocation failure 1 in matrix()");
	m += NR_END;
	m -= nrl;

	/* allocate rows and set pointers to them */
	m[nrl]=(double *) malloc((size_t)((nrow*ncol+NR_END)*sizeof(double)));
	if (!m[nrl]) NRerror("allocation failure 2 in matrix()");
	m[nrl] += NR_END;
	m[nrl] -= ncl;

	for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;

	/* return pointer to array of pointers to rows */
	return m;
}

int **imatrix(long nrl, long nrh, long ncl, long nch)
/* allocate a int matrix with subscript range m[nrl..nrh][ncl..nch] */
{
	long i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
	int **m;

	/* allocate pointers to rows */
	m=(int **) malloc((size_t)((nrow+NR_END)*sizeof(int*)));
	if (!m) NRerror("allocation failure 1 in matrix()");
	m += NR_END;
	m -= nrl;


	/* allocate rows and set pointers to them */
	m[nrl]=(int *) malloc((size_t)((nrow*ncol+NR_END)*sizeof(int)));
	if (!m[nrl]) NRerror("allocation failure 2 in matrix()");
	m[nrl] += NR_END;
	m[nrl] -= ncl;

	for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;

	/* return pointer to array of pointers to rows */
	return m;
}

float **subMatrix(float **a, long oldrl, long oldrh, long oldcl, long oldch,
	long newrl, long newcl)
/* point a subMatrix [newrl..][newcl..] to a[oldrl..oldrh][oldcl..oldch] */
{
	long i,j,nrow=oldrh-oldrl+1,ncol=oldcl-newcl;
	float **m;

	/* allocate array of pointers to rows */
	m=(float **) malloc((size_t) ((nrow+NR_END)*sizeof(float*)));
	if (!m) NRerror("allocation failure in subMatrix()");
	m += NR_END;
	m -= newrl;

	/* set pointers to rows */
	for(i=oldrl,j=newrl;i<=oldrh;i++,j++) m[j]=a[i]+ncol;

	/* return pointer to array of pointers to rows */
	return m;
}

float **convert_matrix(float *a, long nrl, long nrh, long ncl, long nch)
/* allocate a float matrix m[nrl..nrh][ncl..nch] that points to the matrix
declared in the standard C manner as a[nrow][ncol], where nrow=nrh-nrl+1
and ncol=nch-ncl+1. The routine should be called with the address
&a[0][0] as the first argument. */
{
	long i,j,nrow=nrh-nrl+1,ncol=nch-ncl+1;
	float **m;

	/* allocate pointers to rows */
	m=(float **) malloc((size_t) ((nrow+NR_END)*sizeof(float*)));
	if (!m) NRerror("allocation failure in convert_matrix()");
	m += NR_END;
	m -= nrl;

	/* set pointers to rows */
	m[nrl]=a-ncl;
	for(i=1,j=nrl+1;i<nrow;i++,j++) m[j]=m[j-1]+ncol;
	/* return pointer to array of pointers to rows */
	return m;
}

float ***f3tensor(long nrl, long nrh, long ncl, long nch, long ndl, long ndh)
/* allocate a float 3tensor with range t[nrl..nrh][ncl..nch][ndl..ndh] */
{
	long i,j,nrow=nrh-nrl+1,ncol=nch-ncl+1,ndep=ndh-ndl+1;
	float ***t;

	/* allocate pointers to pointers to rows */
	t=(float ***) malloc((size_t)((nrow+NR_END)*sizeof(float**)));
	if (!t) NRerror("allocation failure 1 in f3tensor()");
	t += NR_END;
	t -= nrl;

	/* allocate pointers to rows and set pointers to them */
	t[nrl]=(float **) malloc((size_t)((nrow*ncol+NR_END)*sizeof(float*)));
	if (!t[nrl]) NRerror("allocation failure 2 in f3tensor()");
	t[nrl] += NR_END;
	t[nrl] -= ncl;

	/* allocate rows and set pointers to them */
	t[nrl][ncl]=(float *) malloc((size_t)((nrow*ncol*ndep+NR_END)*sizeof(float)));
	if (!t[nrl][ncl]) NRerror("allocation failure 3 in f3tensor()");
	t[nrl][ncl] += NR_END;
	t[nrl][ncl] -= ndl;

	for(j=ncl+1;j<=nch;j++) t[nrl][j]=t[nrl][j-1]+ndep;
	for(i=nrl+1;i<=nrh;i++) {
		t[i]=t[i-1]+ncol;
		t[i][ncl]=t[i-1][ncl]+ncol*ndep;
		for(j=ncl+1;j<=nch;j++) t[i][j]=t[i][j-1]+ndep;
	}

	/* return pointer to array of pointers to rows */
	return t;
}

void free_vector(float *v, long nl, long nh)
/* free a float vector allocated with vector() */
{
	free((FREE_ARG) (v+nl-NR_END));
}

void free_ivector(int *v, long nl, long nh)
/* free an int vector allocated with ivector() */
{
	free((FREE_ARG) (v+nl-NR_END));
}

void free_cvector(unsigned char *v, long nl, long nh)
/* free an unsigned char vector allocated with cvector() */
{
	free((FREE_ARG) (v+nl-NR_END));
}

void free_lvector(unsigned long *v, long nl, long nh)
/* free an unsigned long vector allocated with lvector() */
{
	free((FREE_ARG) (v+nl-NR_END));
}

void free_dvector(double *v, long nl, long nh)
/* free a double vector allocated with dvector() */
{
	free((FREE_ARG) (v+nl-NR_END));
}

void free_matrix(float **m, long nrl, long nrh, long ncl, long nch)
/* free a float matrix allocated by matrix() */
{
	free((FREE_ARG) (m[nrl]+ncl-NR_END));
	free((FREE_ARG) (m+nrl-NR_END));
}

void free_dmatrix(double **m, long nrl, long nrh, long ncl, long nch)
/* free a double matrix allocated by dmatrix() */
{
	free((FREE_ARG) (m[nrl]+ncl-NR_END));
	free((FREE_ARG) (m+nrl-NR_END));
}

void free_imatrix(int **m, long nrl, long nrh, long ncl, long nch)
/* free an int matrix allocated by imatrix() */
{
	free((FREE_ARG) (m[nrl]+ncl-NR_END));
	free((FREE_ARG) (m+nrl-NR_END));
}

void free_subMatrix(float **b, long nrl, long nrh, long ncl, long nch)
/* free a subMatrix allocated by subMatrix() */
{
	free((FREE_ARG) (b+nrl-NR_END));
}

void free_convert_matrix(float **b, long nrl, long nrh, long ncl, long nch)
/* free a matrix allocated by convert_matrix() */
{
	free((FREE_ARG) (b+nrl-NR_END));
}

void free_f3tensor(float ***t, long nrl, long nrh, long ncl, long nch,
	long ndl, long ndh)
/* free a float f3tensor allocated by f3tensor() */
{
	free((FREE_ARG) (t[nrl][ncl]+ndl-NR_END));
	free((FREE_ARG) (t[nrl]+ncl-NR_END));
	free((FREE_ARG) (t+nrl-NR_END));
}
fcomplex *Cvector(int nl, int nh)
/* allocate storage for a complex vector	*/
{
	fcomplex *v;

	v=(fcomplex *)malloc((unsigned) (nh-nl+1)*sizeof(fcomplex));
	if (!v) NRerror("allocation failure in Cvector()");
	return v-nl;
}


fcomplex **Cmatrix(int nrl, int nrh, int ncl, int nch)	
/* allocate storage for a Complex matrix	*/
{
	int	i;
	fcomplex **m;

	m=(fcomplex **)malloc((unsigned) (nrh-nrl+1)*sizeof(fcomplex*));
	if (!m) NRerror("allocation failure 1 in Cmatrix()");
	m -= nrl;
	for (i=nrl;i<=nrh;i++) {
		m[i]=(fcomplex *)malloc((unsigned)(nch-ncl+1)*sizeof(fcomplex));
		if (!m[i]) NRerror("allocation failure 2 in Cmatrix()");
		m[i] -= ncl;
	}
	return m;
}


float ***D3matrix(int nrl, int nrh, int ncl, int nch, int nzl, int nzh)
 /* storage for a 3-D matrix */
{
	int     i,j;
	float   ***m;

	m=(float ***) malloc((unsigned) (nrh-nrl+1)*sizeof(float*));
	if (!m) NRerror("alloc failure 1 in 3Dmatrix()");
	m -= nrl;
	for (i=nrl;i<=nrh;i++) {
		m[i]=(float **) malloc((unsigned) (nch-ncl+1)*sizeof(float*));
		if (!m[i]) NRerror("alloc failure 2 in 3Dmatrix()");
		m[i] -= ncl;
		for (j=ncl;j<=nch;j++) {
			m[i][j]=(float *)
				malloc((unsigned) (nzh-nzl+1)*sizeof(float));
			if (!m[i][j]) NRerror("alloc failure 3 in 3Dmatrix()");
			m[i][j] -= nzl;
		}
	}
	return m;
}

double ***D3dmatrix(int nrl, int nrh, int ncl, int nch, int nzl, int nzh)
/* storage for a 3-D matrix */
{
	int     i,j;
	double   ***m;

	m=(double ***) malloc((unsigned) (nrh-nrl+1)*sizeof(double*));
	if (!m) NRerror("alloc failure 1 in 3Ddmatrix()");
	m -= nrl;
	for (i=nrl;i<=nrh;i++) {
		m[i]=(double **) malloc((unsigned) (nch-ncl+1)*sizeof(double*));
		if (!m[i]) NRerror("alloc failure 2 in 3Dmatrix()");
		m[i] -= ncl;
		for (j=ncl;j<=nch;j++) {
			m[i][j]=(double *)
				malloc((unsigned) (nzh-nzl+1)*sizeof(double));
			if (!m[i][j]) NRerror("alloc failure 3 in 3Ddmatrix()");
			m[i][j] -= nzl;
		}
	}
	return m;
}



void free_Cvector(fcomplex *v, int nl, int nh)
{
	free((char*) (v+nl));
}

void free_Cmatrix(fcomplex **m, int nrl, int nrh, int ncl, int nch)
{
	int	i;

	for(i=nrh;i>=nrl;i--) free((char*) (m[i]+ncl));
	free((char*) (m+nrl));
}

void free_D3matrix(float ***m, int nrl, int nrh, int ncl, int nch, int nzl, int nzh)
{
	int     i,j;

	for(i=nrh;i>=nrl;i--) {
		for(j=nch;j>=ncl;j--) {
			free((char*) (m[i][j]+nzl));
		}
	}
}

void free_D3dmatrix(double ***m, int nrl, int nrh, int ncl, int nch, int nzl, int nzh)
{
	int     i,j;

	for(i=nrh;i>=nrl;i--) {
		for(j=nch;j>=ncl;j--) {
			free((char*) (m[i][j]+nzl));
		}
	}
}




#else /* ANSI */
/* traditional - K&R */

#include <stdio.h>
#define NR_END 1
#define FREE_ARG char*

void NRerror(error_text)
char error_text[];
/* Numerical Recipes standard error handler */
{
	void exit();

	fprintf(stderr,"Numerical Recipes run-time error...\n");
	fprintf(stderr,"%s\n",error_text);
	fprintf(stderr,"...now exiting to system...\n");
	exit(1000);
}

float *vector(nl,nh)
long nh,nl;
/* allocate a float vector with subscript range v[nl..nh] */
{
	float *v;

	v=(float *)malloc((unsigned int) ((nh-nl+1+NR_END)*sizeof(float)));
	if (!v) NRerror("allocation failure in vector()");
	return v-nl+NR_END;
}

int *ivector(nl,nh)
long nh,nl;
/* allocate an int vector with subscript range v[nl..nh] */
{
	int *v;

	v=(int *)malloc((unsigned int) ((nh-nl+1+NR_END)*sizeof(int)));
	if (!v) NRerror("allocation failure in ivector()");
	return v-nl+NR_END;
}

unsigned char *cvector(nl,nh)
long nh,nl;
/* allocate an unsigned char vector with subscript range v[nl..nh] */
{
	unsigned char *v;

	v=(unsigned char *)malloc((unsigned int) ((nh-nl+1+NR_END)*sizeof(unsigned char)));
	if (!v) NRerror("allocation failure in cvector()");
	return v-nl+NR_END;
}

unsigned long *lvector(nl,nh)
long nh,nl;
/* allocate an unsigned long vector with subscript range v[nl..nh] */
{
	unsigned long *v;

	v=(unsigned long *)malloc((unsigned int) ((nh-nl+1+NR_END)*sizeof(long)));
	if (!v) NRerror("allocation failure in lvector()");
	return v-nl+NR_END;
}

double *dvector(nl,nh)
long nh,nl;
/* allocate a double vector with subscript range v[nl..nh] */
{
	double *v;

	v=(double *)malloc((unsigned int) ((nh-nl+1+NR_END)*sizeof(double)));
	if (!v) NRerror("allocation failure in dvector()");
	return v-nl+NR_END;
}

float **matrix(nrl,nrh,ncl,nch)
long nch,ncl,nrh,nrl;
/* allocate a float matrix with subscript range m[nrl..nrh][ncl..nch] */
{
	long i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
	float **m;

	/* allocate pointers to rows */
	m=(float **) malloc((unsigned int)((nrow+NR_END)*sizeof(float*)));
	if (!m) NRerror("allocation failure 1 in matrix()");
	m += NR_END;
	m -= nrl;

	/* allocate rows and set pointers to them */
	m[nrl]=(float *) malloc((unsigned int)((nrow*ncol+NR_END)*sizeof(float)));
	if (!m[nrl]) NRerror("allocation failure 2 in matrix()");
	m[nrl] += NR_END;
	m[nrl] -= ncl;

	for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;

	/* return pointer to array of pointers to rows */
	return m;
}

double **dmatrix(nrl,nrh,ncl,nch)
long nch,ncl,nrh,nrl;
/* allocate a double matrix with subscript range m[nrl..nrh][ncl..nch] */
{
	long i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
	double **m;

	/* allocate pointers to rows */
	m=(double **) malloc((unsigned int)((nrow+NR_END)*sizeof(double*)));
	if (!m) NRerror("allocation failure 1 in matrix()");
	m += NR_END;
	m -= nrl;

	/* allocate rows and set pointers to them */
	m[nrl]=(double *) malloc((unsigned int)((nrow*ncol+NR_END)*sizeof(double)));
	if (!m[nrl]) NRerror("allocation failure 2 in matrix()");
	m[nrl] += NR_END;
	m[nrl] -= ncl;

	for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;

	/* return pointer to array of pointers to rows */
	return m;
}

int **imatrix(nrl,nrh,ncl,nch)
long nch,ncl,nrh,nrl;
/* allocate a int matrix with subscript range m[nrl..nrh][ncl..nch] */
{
	long i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
	int **m;

	/* allocate pointers to rows */
	m=(int **) malloc((unsigned int)((nrow+NR_END)*sizeof(int*)));
	if (!m) NRerror("allocation failure 1 in matrix()");
	m += NR_END;
	m -= nrl;


	/* allocate rows and set pointers to them */
	m[nrl]=(int *) malloc((unsigned int)((nrow*ncol+NR_END)*sizeof(int)));
	if (!m[nrl]) NRerror("allocation failure 2 in matrix()");
	m[nrl] += NR_END;
	m[nrl] -= ncl;

	for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;

	/* return pointer to array of pointers to rows */
	return m;
}

float **subMatrix(a,oldrl,oldrh,oldcl,oldch,newrl,newcl)
float **a;
long newcl,newrl,oldch,oldcl,oldrh,oldrl;
/* point a subMatrix [newrl..][newcl..] to a[oldrl..oldrh][oldcl..oldch] */
{
	long i,j,nrow=oldrh-oldrl+1,ncol=oldcl-newcl;
	float **m;

	/* allocate array of pointers to rows */
	m=(float **) malloc((unsigned int) ((nrow+NR_END)*sizeof(float*)));
	if (!m) NRerror("allocation failure in subMatrix()");
	m += NR_END;
	m -= newrl;

	/* set pointers to rows */
	for(i=oldrl,j=newrl;i<=oldrh;i++,j++) m[j]=a[i]+ncol;

	/* return pointer to array of pointers to rows */
	return m;
}

float **convert_matrix(a,nrl,nrh,ncl,nch)
float *a;
long nch,ncl,nrh,nrl;
/* allocate a float matrix m[nrl..nrh][ncl..nch] that points to the matrix
declared in the standard C manner as a[nrow][ncol], where nrow=nrh-nrl+1
and ncol=nch-ncl+1. The routine should be called with the address
&a[0][0] as the first argument. */
{
	long i,j,nrow=nrh-nrl+1,ncol=nch-ncl+1;
	float **m;

	/* allocate pointers to rows */
	m=(float **) malloc((unsigned int) ((nrow+NR_END)*sizeof(float*)));
	if (!m)	NRerror("allocation failure in convert_matrix()");
	m += NR_END;
	m -= nrl;

	/* set pointers to rows */
	m[nrl]=a-ncl;
	for(i=1,j=nrl+1;i<nrow;i++,j++) m[j]=m[j-1]+ncol;
	/* return pointer to array of pointers to rows */
	return m;
}

float ***f3tensor(nrl,nrh,ncl,nch,ndl,ndh)
long nch,ncl,ndh,ndl,nrh,nrl;
/* allocate a float 3tensor with range t[nrl..nrh][ncl..nch][ndl..ndh] */
{
	long i,j,nrow=nrh-nrl+1,ncol=nch-ncl+1,ndep=ndh-ndl+1;
	float ***t;

	/* allocate pointers to pointers to rows */
	t=(float ***) malloc((unsigned int)((nrow+NR_END)*sizeof(float**)));
	if (!t) NRerror("allocation failure 1 in f3tensor()");
	t += NR_END;
	t -= nrl;

	/* allocate pointers to rows and set pointers to them */
	t[nrl]=(float **) malloc((unsigned int)((nrow*ncol+NR_END)*sizeof(float*)));
	if (!t[nrl]) NRerror("allocation failure 2 in f3tensor()");
	t[nrl] += NR_END;
	t[nrl] -= ncl;

	/* allocate rows and set pointers to them */
	t[nrl][ncl]=(float *) malloc((unsigned int)((nrow*ncol*ndep+NR_END)*sizeof(float)));
	if (!t[nrl][ncl]) NRerror("allocation failure 3 in f3tensor()");
	t[nrl][ncl] += NR_END;
	t[nrl][ncl] -= ndl;

	for(j=ncl+1;j<=nch;j++) t[nrl][j]=t[nrl][j-1]+ndep;
	for(i=nrl+1;i<=nrh;i++) {
		t[i]=t[i-1]+ncol;
		t[i][ncl]=t[i-1][ncl]+ncol*ndep;
		for(j=ncl+1;j<=nch;j++) t[i][j]=t[i][j-1]+ndep;
	}

	/* return pointer to array of pointers to rows */
	return t;
}

void free_vector(v,nl,nh)
float *v;
long nh,nl;
/* free a float vector allocated with vector() */
{
	free((FREE_ARG) (v+nl-NR_END));
}

void free_ivector(v,nl,nh)
int *v;
long nh,nl;
/* free an int vector allocated with ivector() */
{
	free((FREE_ARG) (v+nl-NR_END));
}

void free_cvector(v,nl,nh)
long nh,nl;
unsigned char *v;
/* free an unsigned char vector allocated with cvector() */
{
	free((FREE_ARG) (v+nl-NR_END));
}

void free_lvector(v,nl,nh)
long nh,nl;
unsigned long *v;
/* free an unsigned long vector allocated with lvector() */
{
	free((FREE_ARG) (v+nl-NR_END));
}

void free_dvector(v,nl,nh)
double *v;
long nh,nl;
/* free a double vector allocated with dvector() */
{
	free((FREE_ARG) (v+nl-NR_END));
}

void free_matrix(m,nrl,nrh,ncl,nch)
float **m;
long nch,ncl,nrh,nrl;
/* free a float matrix allocated by matrix() */
{
	free((FREE_ARG) (m[nrl]+ncl-NR_END));
	free((FREE_ARG) (m+nrl-NR_END));
}

void free_dmatrix(m,nrl,nrh,ncl,nch)
double **m;
long nch,ncl,nrh,nrl;
/* free a double matrix allocated by dmatrix() */
{
	free((FREE_ARG) (m[nrl]+ncl-NR_END));
	free((FREE_ARG) (m+nrl-NR_END));
}

void free_imatrix(m,nrl,nrh,ncl,nch)
int **m;
long nch,ncl,nrh,nrl;
/* free an int matrix allocated by imatrix() */
{
	free((FREE_ARG) (m[nrl]+ncl-NR_END));
	free((FREE_ARG) (m+nrl-NR_END));
}

void free_subMatrix(b,nrl,nrh,ncl,nch)
float **b;
long nch,ncl,nrh,nrl;
/* free a subMatrix allocated by subMatrix() */
{
	free((FREE_ARG) (b+nrl-NR_END));
}

void free_convert_matrix(b,nrl,nrh,ncl,nch)
float **b;
long nch,ncl,nrh,nrl;
/* free a matrix allocated by convert_matrix() */
{
	free((FREE_ARG) (b+nrl-NR_END));
}

void free_f3tensor(t,nrl,nrh,ncl,nch,ndl,ndh)
float ***t;
long nch,ncl,ndh,ndl,nrh,nrl;
/* free a float f3tensor allocated by f3tensor() */
{
	free((FREE_ARG) (t[nrl][ncl]+ndl-NR_END));
	free((FREE_ARG) (t[nrl]+ncl-NR_END));
	free((FREE_ARG) (t+nrl-NR_END));
}

#endif /* ANSI */

/*
 * SHOW_VECTOR  -  display a vector of dimension [1..n]
 */
void show_vector ( float *A, int n )
{
	int     j;

	for (j=1; j <= n; j++) {
		if (A[j] != 0)	fprintf(stdout,"%14.6e", A[j] );
		else		fprintf(stdout,"   0       ");
	}
	fprintf(stdout," ]';\n\n");
	return;
}

/*
 * SHOW_DVECTOR  -  display a vector of dimension [1..n]
 */
void show_dvector ( double *A, int n )
{
	int     j;

	for (j=1; j <= n; j++) {
		if ( fabs(A[j]) >= 1.e-99)
			fprintf(stdout,"%14.6e", A[j] );
		else	fprintf(stdout,"   0       ");
	}
	fprintf(stdout," ]';\n\n");
	return;
}

/*
 * SHOW_IVECTOR  -  display a vector of integers of dimension [1..n]
 */
void show_ivector ( int *A, int n )
{
	int     j;

	for (j=1; j <= n; j++) {
		if (A[j] != 0)	fprintf(stdout,"%11d", A[j] );
		else	 	fprintf(stdout,"   0       ");
	}
	fprintf(stdout," ]';\n\n");
	return;
}


/*
 * SHOW_MATRIX  -  display a matrix of dimension [1..m][1..n]
 */
void show_matrix ( float **A, int m, int n )
{
	int     i,j;

	for (i=1; i <= m; i++) {
		for (j=1; j <= n; j++) {
			if (A[i][j] != 0) fprintf(stdout,"%14.6e", A[i][j] );
			else		  fprintf(stdout,"   0       ");
		}
		if (i==m)	fprintf(stdout," ];\n\n");
		else		fprintf(stdout," \n");
	}
	return;
}

/*
 * SHOW_DMATRIX  - display a matrix of dimension [1..m][1..n] 
 */
void show_dmatrix ( double **A, int m, int n )
{
	int     i,j;

	for (i=1; i <= m; i++) {
		for (j=1; j <= n; j++) {
			if (fabs(A[i][j]) > 1.e-99) fprintf(stdout,"%11.3e", A[i][j] );
			else		  fprintf(stdout,"   0       ");
		}
		if (i==m)	fprintf(stdout," ];\n\n");
		else		fprintf(stdout," \n");
	}
	return;
}


/*
 * SAVE_VECTOR  -  save a vector of dimension [1..n] to the named file 
 */
void save_vector( char filename[], float *V, int nl, int nh, const char *mode )
{
	FILE    *fp_v;
	int     i;
	void	exit();
	time_t	now;

	if ((fp_v = fopen (filename, mode)) == NULL) {
		printf (" error: cannot open file: '%s' \n", filename );
		exit(1011);
	}
        (void) time(&now);
	fprintf(fp_v,"%% filename: %s - %s", filename, ctime(&now));
	fprintf(fp_v,"%% type: vector\n");
	fprintf(fp_v,"%% rows: %d\n", 1 );
	fprintf(fp_v,"%% columns: %d\n", nh-nl+1 );
	for (i=nl; i <= nh; i++) {
		if (V[i] != 0)	fprintf(fp_v,"%15.6e", V[i] );
		else		fprintf(fp_v,"    0         ");         
		fprintf(fp_v,"\n");
	}
	fclose(fp_v);
	return;
}

/*
 * SAVE_DVECTOR  -  save a vector of dimension [1..n] to the named file 
 */
void save_dvector( char filename[], double *V, int nl, int nh, const char *mode )
{
	FILE    *fp_v;
	int     i;
	void	exit();
	time_t	now;

	if ((fp_v = fopen (filename, mode)) == NULL) {
		printf (" error: cannot open file: '%s' \n", filename );
		exit(1011);
	}
        (void) time(&now);
	fprintf(fp_v,"%% filename: %s - %s", filename, ctime(&now));
	fprintf(fp_v,"%% type: vector\n");
	fprintf(fp_v,"%% rows: %d\n", 1 );
	fprintf(fp_v,"%% columns: %d\n", nh-nl+1 );
	for (i=nl; i <= nh; i++) {
		if (V[i] != 0)	fprintf(fp_v,"%21.12e", V[i] );
		else	        fprintf(fp_v,"    0                ");
		fprintf(fp_v,"\n");
	}
	fclose(fp_v);
	return;
}

/*
 * SAVE_IVECTOR  -  save an integer vector of dimension [1..n] to the named file 
 */
void save_ivector( char filename[], int *V, int nl, int nh, const char *mode )
{
	FILE    *fp_v;
	int     i;
	void	exit();
	time_t	now;

	if ((fp_v = fopen (filename, mode)) == NULL) {
		printf (" error: cannot open file: '%s' \n", filename );
		exit(1012);
	}
        (void) time(&now);
	fprintf(fp_v,"%% filename: %s - %s", filename, ctime(&now));
	fprintf(fp_v,"%% type: vector\n");
	fprintf(fp_v,"%% rows: %d\n", 1 );
	fprintf(fp_v,"%% columns: %d\n", nh-nl+1 );
	for (i=nl; i <= nh; i++) {
		if (V[i] != 0)	fprintf(fp_v,"%15d", V[i] );
		else		fprintf(fp_v,"   0         ");         
		fprintf(fp_v,"\n");
	}
	fclose(fp_v);
	return;
}

/*
 * SAVE_MATRIX  -  save a matrix of dimension [ml..mh][nl..nh] to the named file
 */
void save_matrix ( char filename[], float **A, int ml, int mh, int nl, int nh, int transpose, const char *mode )
{
	FILE    *fp_m;
	int     i,j, rows, cols;
	void	exit();
	time_t	now;

	if ( transpose ) rows = nh-nl+1; else rows = mh-ml+1;
	if ( transpose ) cols = mh-ml+1; else cols = nh-nl+1;

	if ((fp_m = fopen (filename, mode)) == NULL) {
		printf (" error: cannot open file: %s \n", filename );
		exit(1013);
	}
        (void) time(&now);
	fprintf(fp_m,"%% filename: %s - %s", filename, ctime(&now));
	fprintf(fp_m,"%% type: matrix \n");
	fprintf(fp_m,"%% rows: %d\n", rows );
	fprintf(fp_m,"%% columns: %d\n", cols );
	if ( transpose ) {
	    for (j=nl; j <= nh; j++) {
		for (i=ml; i <= mh; i++) {
			if (A[i][j] != 0) fprintf(fp_m,"%15.6e", A[i][j] );
			else		  fprintf(fp_m,"    0          ");
		}
		fprintf(fp_m,"\n");
	    }
	} else {
	    for (i=ml; i <= mh; i++) {
		for (j=nl; j <= nh; j++) {
			if (A[i][j] != 0) fprintf(fp_m,"%15.6e", A[i][j] );
			else		  fprintf(fp_m,"    0          ");
		}
		fprintf(fp_m,"\n");
	    }
	}
	fclose ( fp_m);
	return;
}

/*
 * SAVE_DMATRIX  - save a matrix of dimension [ml..mh][nl..nh] to the named file
 */
void save_dmatrix ( char filename[], double **A, int ml, int mh, int nl, int nh, int transpose, const char *mode )
{
	FILE    *fp_m;
	int     i,j, rows, cols;
	void	exit();
	time_t	now;

	if ( transpose ) rows = nh-nl+1; else rows = mh-ml+1;
	if ( transpose ) cols = mh-ml+1; else cols = nh-nl+1;

	if ((fp_m = fopen (filename, mode)) == NULL) {
		printf (" error: cannot open file: %s \n", filename );
		exit(1014);
	}
        (void) time(&now);
	fprintf(fp_m,"%% filename: %s - %s", filename, ctime(&now));
	fprintf(fp_m,"%% type: matrix \n");
	fprintf(fp_m,"%% rows: %d\n", rows );
	fprintf(fp_m,"%% columns: %d\n", cols );
	if ( transpose ) {
	    for (j=nl; j <= nh; j++) {
		for (i=ml; i <= mh; i++) {
			if (fabs(A[i][j]) > 1.e-99) fprintf(fp_m,"%21.12e", A[i][j] );
			else		            fprintf(fp_m,"    0                ");
		}
		fprintf(fp_m,"\n");
	    }
	} else { 
	    for (i=ml; i <= mh; i++) {
		for (j=nl; j <= nh; j++) {
			if (fabs(A[i][j]) > 1.e-99) fprintf(fp_m,"%21.12e", A[i][j] );
			else		            fprintf(fp_m,"    0                ");
		}
		fprintf(fp_m,"\n");
	    }
	}
	fclose ( fp_m);
	return;
}

/*
 * SAVE_UT_MATRIX  - 						     23apr01 
 * save a symmetric matrix of dimension [1..n][1..n] to the named file 
 *  use only upper-triangular part
 */
void save_ut_matrix ( char filename[], float **A, int n, const char *mode )
{
	FILE    *fp_m;
	int     i,j;
        void	exit();
	time_t	now;

	if ((fp_m = fopen (filename, mode)) == NULL) {
		printf (" error: cannot open file: %s \n", filename );
		exit(1015);
	}
        (void) time(&now);
	fprintf(fp_m,"%% filename: %s - %s", filename, ctime(&now));
	fprintf(fp_m,"%% type: matrix \n");
	fprintf(fp_m,"%% rows: %d\n", n );
	fprintf(fp_m,"%% columns: %d\n", n );
	for (i=1; i <= n; i++) {
	  for (j=1; j <= n; j++) {
		if ( i > j ) {
			if (fabs(A[j][i]) > 1.e-99) fprintf(fp_m,"%15.6e", A[j][i] );
			else		            fprintf(fp_m,"    0          ");
		} else {
			if (fabs(A[i][j]) > 1.e-99) fprintf(fp_m,"%15.6e", A[i][j] );
			else		            fprintf(fp_m,"    0          ");
		}
	  }
	  fprintf(fp_m,"\n");
	}
	fclose ( fp_m);
	return;
}

/*
 * SAVE_UT_DMATRIX  - 						23apr01
 * save a symetric matrix of dimension [1..n][1..n] to the named file 
 * use only upper-triangular part
 */
void save_ut_dmatrix ( char filename[], double **A, int n, const char *mode )
{
	FILE    *fp_m;
	int     i,j;
        void	exit();
	time_t	now;

	if ((fp_m = fopen (filename, mode)) == NULL) {
		printf (" error: cannot open file: %s \n", filename );
		exit(1016);
	}
        (void) time(&now);
	fprintf(fp_m,"%% filename: %s - %s\n", filename, ctime(&now));
	fprintf(fp_m,"%% type: matrix \n");
	fprintf(fp_m,"%% rows: %d\n", n );
	fprintf(fp_m,"%% columns: %d\n", n );
	for (i=1; i <= n; i++) {
	  for (j=1; j <= n; j++) {
		if ( i > j ) {
			if (fabs(A[j][i]) > 1.e-99) fprintf(fp_m,"%21.12e", A[j][i] );
			else		            fprintf(fp_m,"    0                ");
		} else {
			if (fabs(A[i][j]) > 1.e-99) fprintf(fp_m,"%21.12e", A[i][j] );
			else		            fprintf(fp_m,"    0                ");
		}
	  }
	  fprintf(fp_m,"\n");
	}
	fclose ( fp_m);
	return;
}
