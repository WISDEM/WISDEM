/*	
 This file is part of FRAME3DD: 
 Static and dynamic structural analysis of 2D & 3D frames and trusses
 with elastic and geometric stiffness.
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
*/

#ifndef FRAME_COMMON_H
#define FRAME_COMMON_H

#define FRAME3DD_PATHMAX 512
#ifndef MAXL
#define MAXL    512
#endif

#define FILENMAX 128

#ifndef VERSION
#define VERSION "20140514+"
#endif

#ifndef PI
#define PI 3.14159265358979323846264338327950288419716939937510
#endif

// Zvert=1: Z axis is vertical... rotate about Y-axis, then rotate about Z-axis
// Zvert=0: Y axis is vertical... rotate about Z-axis, then rotate about Y-axis
#define Zvert 1	

#endif /* FRAME_COMMON_H */

