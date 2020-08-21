/*	FRAME3DD: Static and dynamic structural analysis of 2D & 3D frames and trusses
	Copyright (C) 2007-2008 John Pye

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*//**
	@file 
	3D vector type and related functions/methods.
*/
#ifndef MSTRAP_VEC3_H
#define MSTRAP_VEC3_H

#include "config.h"
#include <stdio.h>

#ifdef __cplusplus
extern "C"{
#endif

/**
	3D vector type used by the microstran parser
*/

typedef struct vec3_struct{
	double x, y, z;
} vec3;

MSTRANP_API const vec3 VEC3_ZERO;

MSTRANP_API vec3 vec3_create(double x, double y, double z);
MSTRANP_API double vec3_dot(vec3 A, vec3 B);
MSTRANP_API vec3 vec3_add(vec3 A, vec3 B);
MSTRANP_API vec3 vec3_cross(vec3 A, vec3 B);
MSTRANP_API vec3 vec3_scale(vec3 A, double s);
MSTRANP_API vec3 vec3_norm(vec3 A);
MSTRANP_API double vec3_mod(vec3 A);
MSTRANP_API vec3 vec3_negate(vec3 A);

/**
	Vector difference/subtraction.
	@return the vector (A - B)
*/
MSTRANP_API vec3 vec3_diff(vec3 A, vec3 B);

MSTRANP_API int vec3_print(FILE *f, vec3 A);

MSTRANP_API vec3 vec3_rotate(vec3 A, vec3 axis, double theta);

/**
	Calculate the angle between two vectors, in radians.
*/
MSTRANP_API double vec3_angle(vec3 A, vec3 B);

/**
	Calculate the angle between two vectors, in radians. Also return
	the cross-product of the two vectors, useful with vec3_rotate.
*/
MSTRANP_API double vec3_angle_cross(vec3 A, vec3 B, vec3 *C);

MSTRANP_API char vec3_equal(vec3 A, vec3 B);
MSTRANP_API char vec3_equal_tol(vec3 A, vec3 B, double tol);

char vec3_isnan(const vec3 *A);

#define VEC3_PR(V) (fprintf(stderr,"%s = ",#V), vec3_print(stderr,V), fprintf(stderr,"\n"))

#define VEC3_CHECK_NAN(V) (vec3_isnan(&(V)) ? (VEC3_PR(V), assert(!vec3_isnan(&(V)))) : 0)

#define VEC3_ASSERT_EQUAL_TOL(X,Y,TOL) (vec3_equal_tol((X),(Y),TOL) ? 0 : (VEC3_PR(X), VEC3_PR(Y), assert(vec3_equal_tol((X),(Y),TOL))))

#define VEC3_NOT_EQUAL(X,Y) ((X).x!=(Y).x || (X).y!=(Y).y || (X).z!=(Y).z)

#define VEC3_NOT_ZERO(X) ((X).x!=0 || (X).y!=0 || (X).z!=0)

#ifdef __cplusplus
};
#endif

#endif

