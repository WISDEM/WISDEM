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
*//** @file
	Routines to solve the generalized eigenvalue problem

	H.P. Gavin, Civil Engineering, Duke University, hpgavin@duke.edu  1 March 2007
	Bathe, Finite Element Procecures in Engineering Analysis, Prentice Hall, 1982
*/
#ifndef PYFRAME_EIG_H
#define PYFRAME_EIG_H

/**
	Find the lowest m eigenvalues, w, and eigenvectors, V, of the 
	general eigenproblem, K V = w M V, using sub-space / Jacobi iteration.

	@param K is an n by n  symmetric real (stiffness) matrix
	@param M is an n by n  symmetric positive definate real (mass) matrix
	@param w is a diagonal matrix of eigen-values
	@param V is a  rectangular matrix of eigen-vectors
*/
int subspace(
	double **K, double **M,	/**< stiffness and mass matrices	*/
	int n, int m,		/**< DoF and number of required modes	*/
	double *w, double **V,	/**< modal frequencies and mode shapes	*/
	double tol,		/**< covergence tolerence		*/
	double shift,		/**< frequency shift for unrestrained frames */
	int *iter,		/**< number of sub-space iterations	*/
	int *ok,		/**< Sturm check result			*/
	int verbose		/**< 1: copious screen output, 0: none	*/
);


/**
	carry out matrix-matrix-matrix multiplication for symmetric A
	C = X' A X     C is J by J	X is N by J	A is N by N
*/
void xtAx( double **A, double **X, double **C, int N, int J );

/**
	calculate the lowest m eigen-values and eigen-vectors of the
	generalized eigen-problem, K v = w M v, using a matrix
	iteration approach with shifting.

	@param n number of degrees of freedom
	@param m number of required modes
*/
int stodola(
	double **K, double **M,	/**< stiffness and mass matrices	*/
	int n, int m,		/**< DoF and number of required modes	*/
	double *w, double **V,	/**< modal frequencies and mode shapes	*/
	double tol,		/**< covergence tolerence		*/
	double shift,		/**< frequency shift for unrestrained frames */
	int *iter,		/**< number of sub-space iterations	*/
	int *ok,		/**< Sturm check result			*/
	int verbose		/**< 1: copious screen output, 0: none	*/
);

#endif /* FRAME_EIG_H */

