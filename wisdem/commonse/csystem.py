#!/usr/bin/env python
# encoding: utf-8
"""
csystem.py

Created by Andrew Ning on 2/21/2012.
Copyright (c) NREL. All rights reserved.
"""

from __future__ import print_function

import numpy as np


class DirectionVector(object):
    """Handles rotation of direction vectors to appropriate coordinate systems.
    All angles must be in degrees.

    """

    def __init__(self, x, y, z, dx=None, dy=None, dz=None):
        """3-Dimensional vector that depends on direction only (not position).

        Parameters
        ----------
        x : float or ndarray
            x-direction of vector(s)
        y : float or ndarray
            y-direction of vector(s)
        z : float or ndarray
            z-direction of vector(s)

        """

        self.x = np.array(x)
        self.y = np.array(y)
        self.z = np.array(z)

        if dx is None:
            dx = {}
            dx["dx"] = np.ones_like(self.x)
            dx["dy"] = np.zeros_like(self.y)
            dx["dz"] = np.zeros_like(self.z)

            dy = {}
            dy["dx"] = np.zeros_like(self.x)
            dy["dy"] = np.ones_like(self.y)
            dy["dz"] = np.zeros_like(self.z)

            dz = {}
            dz["dx"] = np.zeros_like(self.x)
            dz["dy"] = np.zeros_like(self.y)
            dz["dz"] = np.ones_like(self.z)

        self.dx = dx
        self.dy = dy
        self.dz = dz

    @classmethod
    def fromArray(cls, array):
        """initialize with NumPy array

        Parameters
        ----------
        array : ndarray
            construct DirectionVector using array of size 3

        """

        return cls(array[0], array[1], array[2])

    def toArray(self):
        """convert DirectionVector to NumPy array

        Returns
        -------
        array : ndarray
            NumPy array in order x, y, z containing DirectionVector data

        """

        return np.c_[self.x, self.y, self.z]

    def _rotateAboutZ(self, xstring, ystring, zstring, theta, thetaname, reverse=False):
        """
        x X y = z.  rotate c.s. about z, +theta
        all angles in degrees
        """

        thetaM = 1.0
        if reverse:
            thetaM = -1.0

        x = getattr(self, xstring)
        y = getattr(self, ystring)
        z = getattr(self, zstring)
        dx = getattr(self, "d" + xstring)
        dy = getattr(self, "d" + ystring)
        dz = getattr(self, "d" + zstring)

        theta = np.radians(theta * thetaM)
        c = np.cos(theta)
        s = np.sin(theta)

        xnew = x * c + y * s
        ynew = -x * s + y * c
        znew = z

        angles = []
        for key in dx.keys():
            if not key in ["dx", "dy", "dz"]:
                angles.append(key)

        dxnew = {}
        dxnew["dx"] = dx["dx"] * c + dy["dx"] * s
        dxnew["dy"] = dx["dy"] * c + dy["dy"] * s
        dxnew["dz"] = dx["dz"] * c + dy["dz"] * s
        dxnew["d" + thetaname] = (-x * s + y * c) * np.radians(thetaM)
        for dangle in angles:
            dxnew[dangle] = dx[dangle] * c + dy[dangle] * s

        dynew = {}
        dynew["dx"] = -dx["dx"] * s + dy["dx"] * c
        dynew["dy"] = -dx["dy"] * s + dy["dy"] * c
        dynew["dz"] = -dx["dz"] * s + dy["dz"] * c
        dynew["d" + thetaname] = (-x * c - y * s) * np.radians(thetaM)
        for dangle in angles:
            dynew[dangle] = -dx[dangle] * s + dy[dangle] * c

        dznew = {}
        dznew["dx"] = dz["dx"] * np.ones_like(theta)  # multiply by ones just to get right size in case of float
        dznew["dy"] = dz["dy"] * np.ones_like(theta)
        dznew["dz"] = dz["dz"] * np.ones_like(theta)
        dznew["d" + thetaname] = np.zeros_like(theta)
        for dangle in angles:
            dznew[dangle] = dz[dangle]

        return xnew, ynew, znew, dxnew, dynew, dznew

    def windToInertial(self, beta):
        """Rotates from wind-aligned to inertial

        Parameters
        ----------
        beta : float (deg)
            wind angle

        Returns
        -------
        vector : DirectionVector
            a DirectionVector in the inertial coordinate system

        """
        xw, yw, zw, dxw, dyw, dzw = self._rotateAboutZ("x", "y", "z", beta, "beta", reverse=True)
        return DirectionVector(xw, yw, zw, dxw, dyw, dzw)

    def inertialToWind(self, beta):
        """Rotates from inertial to wind-aligned

        Parameters
        ----------
        beta : float (deg)
            wind angle

        Returns
        -------
        vector : DirectionVector
            a DirectionVector in the wind-aligned coordinate system

        """
        xw, yw, zw, dxw, dyw, dzw = self._rotateAboutZ("x", "y", "z", beta, "beta")
        return DirectionVector(xw, yw, zw, dxw, dyw, dzw)

    def yawToWind(self, Psi):
        """Rotates from yaw-aligned to wind-aligned

        Parameters
        ----------
        Psi : float (deg)
            yaw angle

        Returns
        -------
        vector : DirectionVector
            a DirectionVector in the wind-aligned coordinate system

        """
        xw, yw, zw, dxw, dyw, dzw = self._rotateAboutZ("x", "y", "z", Psi, "yaw", reverse=True)
        return DirectionVector(xw, yw, zw, dxw, dyw, dzw)

    def windToYaw(self, Psi):
        """Rotates from wind-aligned to yaw-aligned

        Parameters
        ----------
        Psi : float (deg)
            yaw angle

        Returns
        -------
        vector : DirectionVector
            a DirectionVector in the yaw-aligned coordinate system

        """
        xy, yy, zy, dxy, dyy, dzy = self._rotateAboutZ("x", "y", "z", Psi, "yaw")
        return DirectionVector(xy, yy, zy, dxy, dyy, dzy)

    def hubToYaw(self, Theta, derivatives=False):
        """Rotates from hub-aligned to yaw-aligned

        Parameters
        ----------
        Theta : float (deg)
            tilt angle

        Returns
        -------
        vector : DirectionVector
            a DirectionVector in the yaw-aligned coordinate system

        """
        zy, xy, yy, dzy, dxy, dyy = self._rotateAboutZ("z", "x", "y", Theta, "tilt", reverse=True)
        return DirectionVector(xy, yy, zy, dxy, dyy, dzy)

    def yawToHub(self, Theta):
        """Rotates from yaw-aligned to hub-aligned

        Parameters
        ----------
        Theta : float (deg)
            tilt angle

        Returns
        -------
        vector : DirectionVector
            a DirectionVector in the hub-aligned coordinate system

        """
        zh, xh, yh, dzh, dxh, dyh = self._rotateAboutZ("z", "x", "y", Theta, "tilt")
        return DirectionVector(xh, yh, zh, dxh, dyh, dzh)

    def hubToAzimuth(self, Lambda):
        """Rotates from hub-aligned to azimuth-aligned

        Parameters
        ----------
        Lambda : float or ndarray (deg)
            azimuth angle

        Returns
        -------
        vector : DirectionVector
            a DirectionVector in the azimuth-aligned coordinate system

        """
        yz, zz, xz, dyz, dzz, dxz = self._rotateAboutZ("y", "z", "x", Lambda, "azimuth")
        return DirectionVector(xz, yz, zz, dxz, dyz, dzz)

    def azimuthToHub(self, Lambda):
        """Rotates from azimuth-aligned to hub-aligned

        Parameters
        ----------
        Lambda : float or ndarray (deg)
            azimuth angle

        Returns
        -------
        vector : DirectionVector
            a DirectionVector in the hub-aligned coordinate system

        """

        yh, zh, xh, dyh, dzh, dxh = self._rotateAboutZ("y", "z", "x", Lambda, "azimuth", reverse=True)
        return DirectionVector(xh, yh, zh, dxh, dyh, dzh)

    def azimuthToBlade(self, Phi):
        """Rotates from azimuth-aligned to blade-aligned

        Parameters
        ----------
        Phi : float (deg)
            precone angle

        Returns
        -------
        vector : DirectionVector
            a DirectionVector in the blade-aligned coordinate system

        """

        zb, xb, yb, dzb, dxb, dyb = self._rotateAboutZ("z", "x", "y", Phi, "precone", reverse=True)
        return DirectionVector(xb, yb, zb, dxb, dyb, dzb)

    def bladeToAzimuth(self, Phi):
        """Rotates from blade-aligned to azimuth-aligned

        Parameters
        ----------
        Phi : float (deg)
            precone angle

        Returns
        -------
        vector : DirectionVector
            a DirectionVector in the azimuth-aligned coordinate system


        """

        za, xa, ya, dza, dxa, dya = self._rotateAboutZ("z", "x", "y", Phi, "precone")
        return DirectionVector(xa, ya, za, dxa, dya, dza)

    def airfoilToBlade(self, theta):
        """Rotates from airfoil-aligned to blade-aligned

        Parameters
        ----------
        theta : float (deg)
            twist angle

        Returns
        -------
        vector : DirectionVector
            a DirectionVector in the blade-aligned coordinate system

        """

        xb, yb, zb, dxb, dyb, dzb = self._rotateAboutZ("x", "y", "z", theta, "theta")
        return DirectionVector(xb, yb, zb, dxb, dyb, dzb)

    def bladeToAirfoil(self, theta):
        """Rotates from blade-aligned to airfoil-aligned

        Parameters
        ----------
        theta : float (deg)
            twist angle

        Returns
        -------
        vector : DirectionVector
            a DirectionVector in the airfoil-aligned coordinate system

        """

        xa, ya, za, dxa, dya, dza = self._rotateAboutZ("x", "y", "z", theta, "theta", reverse=True)
        return DirectionVector(xa, ya, za, dxa, dya, dza)

    def airfoilToProfile(self):
        """Rotates from airfoil-aligned to profile

        Returns
        -------
        vector : DirectionVector
            a DirectionVector in the profile coordinate system

        """

        return DirectionVector(self.y, self.x, self.z, self.dy, self.dx, self.dz)

    def profileToAirfoil(self):
        """Rotates from profile to airfoil-aligned

        Returns
        -------
        vector : DirectionVector
            a DirectionVector in the airfoil-aligned coordinate system

        """

        return DirectionVector(self.y, self.x, self.z, self.dy, self.dx, self.dz)

    def cross(self, other):
        """cross product between two DirectionVectors

        Parameters
        ----------
        other : DirectionVector
            other vector to cross with

        Returns
        -------
        vector : DirectionVector
            vector = self X other

        """
        v1 = np.c_[self.x, self.y, self.z]
        v2 = np.c_[other.x, other.y, other.z]
        v = np.cross(v1, v2)

        if len(v.shape) > 1:
            return DirectionVector(v[:, 0], v[:, 1], v[:, 2])
        else:
            return DirectionVector(v[0], v[1], v[2])

    def cross_deriv(self, other, namea="a", nameb="b"):
        """defined only for floats for now"""

        # c = a X b
        a = self
        b = other

        dx = {}
        dx[namea] = np.r_[0.0, b.z, -b.y]
        dx[nameb] = np.r_[0.0, -a.z, a.y]

        dy = {}
        dy[namea] = np.r_[-b.z, 0.0, b.x]
        dy[nameb] = np.r_[a.z, 0.0, -a.x]

        dz = {}
        dz[namea] = np.r_[b.y, -b.x, 0.0]
        dz[nameb] = np.r_[-a.y, a.x, 0.0]

        return dx, dy, dz

    def cross_deriv_array(self, other, namea="a", nameb="b"):
        # c = a X b
        a = self
        b = other

        dx = {}
        dx["d" + namea + "x"] = np.zeros_like(b.x)
        dx["d" + namea + "y"] = b.z
        dx["d" + namea + "z"] = -b.y
        dx["d" + nameb + "x"] = np.zeros_like(a.x)
        dx["d" + nameb + "y"] = -a.z
        dx["d" + nameb + "z"] = a.y

        dy = {}
        dy["d" + namea + "x"] = -b.z
        dy["d" + namea + "y"] = np.zeros_like(b.y)
        dy["d" + namea + "z"] = b.x
        dy["d" + nameb + "x"] = a.z
        dy["d" + nameb + "y"] = np.zeros_like(a.y)
        dy["d" + nameb + "z"] = -a.x

        dz = {}
        dz["d" + namea + "x"] = b.y
        dz["d" + namea + "y"] = -b.x
        dz["d" + namea + "z"] = np.zeros_like(b.z)
        dz["d" + nameb + "x"] = -a.y
        dz["d" + nameb + "y"] = a.x
        dz["d" + nameb + "z"] = np.zeros_like(a.z)

        return dx, dy, dz

    def __neg__(self):
        """negate direction vector"""

        return DirectionVector(-self.x, -self.y, -self.z)

    def __add__(self, other):
        """add two DirectionVector objects (v1 = v2 + v3)"""

        if isinstance(other, DirectionVector):
            return DirectionVector(self.x + other.x, self.y + other.y, self.z + other.z)
        else:
            return DirectionVector(self.x + other, self.y + other, self.z + other)

    def __sub__(self, other):
        """subtract DirectionVector objects (v1 = v2 - v3)"""

        if isinstance(other, DirectionVector):
            return DirectionVector(self.x - other.x, self.y - other.y, self.z - other.z)
        else:
            return DirectionVector(self.x - other, self.y - other, self.z - other)

    def __iadd__(self, other):
        """add DirectionVector object to self (v1 += v2)"""

        if isinstance(other, DirectionVector):
            self.x += other.x
            self.y += other.y
            self.z += other.z
        else:
            self.x += other
            self.y += other
            self.z += other

        return self

    def __isub__(self, other):
        """subract DirectionVector object from self (v1 -= v2)"""

        if isinstance(other, DirectionVector):
            self.x -= other.x
            self.y -= other.y
            self.z -= other.z
        else:
            self.x -= other
            self.y -= other
            self.z -= other

        return self

    def __mul__(self, other):
        """multiply vector times a scalar or element by element muiltiply times another vector (v1 = alpha * v2 or v1 = v2 * v3)"""

        if isinstance(other, DirectionVector):
            return DirectionVector(self.x * other.x, self.y * other.y, self.z * other.z)
        else:
            return DirectionVector(self.x * other, self.y * other, self.z * other)

    def __truediv__(self, other):
        """divide vector by a scalar or element by element division with another vector (v1 = v2 / alpha or v1 = v2 / v3)"""

        if isinstance(other, DirectionVector):
            return DirectionVector(self.x / other.x, self.y / other.y, self.z / other.z)
        else:
            return DirectionVector(self.x / float(other), self.y / float(other), self.z / float(other))

    def __imul__(self, other):
        """multiply self times a scalar or element by element muiltiply times another vector (v1 *= alpha or v1 *= v2)"""

        if isinstance(other, DirectionVector):
            self.x *= other.x
            self.y *= other.y
            self.z *= other.z
        else:
            self.x *= other
            self.y *= other
            self.z *= other
        return self

    def __str__(self):
        """print string representation"""

        return "{0}, {1}, {2}".format(self.x, self.y, self.z)
