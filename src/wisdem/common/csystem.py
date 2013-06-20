#!/usr/bin/env python
# encoding: utf-8
"""
csystem.py

Created by Andrew Ning on 2/21/2012.
Copyright (c) NREL. All rights reserved.
"""

import numpy as np


# !!!!!!!! NOTE ALL ANGLES SHOULD BE PASSED IN DEGREES !!!!!!!!!!!!!!
def _rotateAboutZ(x, y, z, theta):
    """
    x X y = z.  rotate c.s. about z, +theta
    all angles in degrees
    """

    theta = np.radians(theta)
    c = np.cos(theta)
    s = np.sin(theta)

    xnew = x*c + y*s
    ynew = -x*s + y*c

    return xnew, ynew, z






class DirectionVector(object):
    """Handles rotation of direction vectors to appropriate coordinate systems.
    All angles must be in degrees.

    """

    def __init__(self, x, y, z):
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

        return np.array([self.x, self.y, self.z])


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
        xi, yi, zi = _rotateAboutZ(self.x, self.y, self.z, -beta)
        return DirectionVector(xi, yi, zi)


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
        xw, yw, zw = _rotateAboutZ(self.x, self.y, self.z, beta)
        return DirectionVector(xw, yw, zw)


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
        xw, yw, zw = _rotateAboutZ(self.x, self.y, self.z, -Psi)
        return DirectionVector(xw, yw, zw)


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
        xy, yy, zy = _rotateAboutZ(self.x, self.y, self.z, Psi)
        return DirectionVector(xy, yy, zy)


    def hubToYaw(self, Theta):
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
        zy, xy, yy = _rotateAboutZ(self.z, self.x, self.y, -Theta)
        return DirectionVector(xy, yy, zy)


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
        zh, xh, yh = _rotateAboutZ(self.z, self.x, self.y, Theta)
        return DirectionVector(xh, yh, zh)


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
        yz, zz, xz = _rotateAboutZ(self.y, self.z, self.x, Lambda)
        return DirectionVector(xz, yz, zz)


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
        yh, zh, xh = _rotateAboutZ(self.y, self.z, self.x, -Lambda)
        return DirectionVector(xh, yh, zh)



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


        zb, xb, yb = _rotateAboutZ(self.z, self.x, self.y, -Phi)
        return DirectionVector(xb, yb, zb)



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

        za, xa, ya = _rotateAboutZ(self.z, self.x, self.y, Phi)
        return DirectionVector(xa, ya, za)



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
        xb, yb, zb = _rotateAboutZ(self.x, self.y, self.z, theta)
        return DirectionVector(xb, yb, zb)


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

        xa, ya, za = _rotateAboutZ(self.x, self.y, self.z, -theta)
        return DirectionVector(xa, ya, za)


    def airfoilToProfile(self):
        """Rotates from airfoil-aligned to profile

        Returns
        -------
        vector : DirectionVector
            a DirectionVector in the profile coordinate system

        """

        return DirectionVector(self.y, self.x, self.z)


    def profileToAirfoil(self):
        """Rotates from profile to airfoil-aligned

        Returns
        -------
        vector : DirectionVector
            a DirectionVector in the airfoil-aligned coordinate system

        """

        return DirectionVector(self.y, self.x, self.z)




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

        v1 = np.array([self.x, self.y, self.z]).transpose()
        v2 = np.array([other.x, other.y, other.z]).transpose()
        v = np.cross(v1, v2)

        if len(v.shape) > 1:
            return DirectionVector(v[:, 0], v[:, 1], v[:, 2])
        else:
            return DirectionVector(v[0], v[1], v[2])


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


    def __div__(self, other):
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


    def __idiv__(self, other):
        """divide self by a scalar or element by element division with another vector (v1 /= alpha or v1 /= v2)"""
        if isinstance(other, DirectionVector):
            self.x /= other.x
            self.y /= other.y
            self.z /= other.z
        else:
            self.x /= float(other)
            self.y /= float(other)
            self.z /= float(other)
        return self


    def __str__(self):
        """print string representation"""

        return '{0}, {1}, {2}'.format(self.x, self.y, self.z)





# class Force(DirectionVector):

#     def __init__(self, Fx, Fy, Fz):
#         """Force vector (or any vector that depends only on orientation).

#         Parameters
#         ----------
#         Fx : float or ndarray
#             force in x-direction
#         Fy : float or ndarray
#             force in y-direction
#         Fz : float or ndarray
#             force in z-direction

#         """
#         super(DirectionVector, self).__init__(Fx, Fy, Fz)




# def _position(r):
#     return Position(r.x, r.y, r.z)


# class Position(DirectionVector):

#     def __init__(self, x, y, z):
#         """Position vector.

#         Parameters
#         ----------
#         x : float or ndarray
#             x-location of position(s)
#         y : float or ndarray
#             y-location of position(s)
#         z : float or ndarray
#             z-location of position(s)

#         """
#         super(Position, self).__init__(x, y, z)


#     def __translate(self, x, y, z):
#         return self - DirectionVector(x, y, z)


#     def yawToWind(self, Psi, height):
#         r = self.__translate(0.0, 0.0, -height)
#         rnew = r.yawToWind(Psi)

#         return _position(rnew)


#     def windToYaw(self, Psi, height):
#         r = self.__translate(0.0, 0.0, height)
#         rnew = r.windToYaw(Psi)

#         return _position(rnew)


#     def hubToYaw(self, Theta, length):
#         r = self.__translate(length, 0.0, 0.0)
#         rnew = r.hubToYaw(Theta)

#         return _position(rnew)


#     def yawToHub(self, Theta, length):
#         r = DirectionVector(self.x, self.y, self.z).yawToHub(Theta)
#         rnew = r.__translate(-length, 0.0, 0.0)

#         return _position(rnew)


#     def hubToAzimuth(self, Lambda, dy, dz):
#         r = self.__translate(0.0, dy, dz)
#         rnew = r.hubToAzimuth(Lambda)

#         return _position(rnew)

#     def azimuthToHub(self, Lambda, dy, dz):
#         r = DirectionVector(self.x, self.y, self.z).azimuthToHub(Lambda)
#         rnew = _position(r).__translate(0.0, -dy, -dz)

#         return _position(rnew)


#     def bladeToAzimuth(self, Phi):
#         r = DirectionVector(self.x, self.y, self.z).bladeToAzimuth(Phi)
#         return _position(r)


#     def profileToAirfoil(self, d):
#         r = self.__translate(d, 0.0, 0.0)
#         rnew = r.profileToAirfoil()

#         return _position(rnew)


#     def airfoilToProfile(self, d):
#         r = self.airfoilToProfile()
#         rnew = r.__translate(-d, 0.0, 0.0)

#         return _position(rnew)



# class Velocity(DirectionVector):

#     def __init__(self, Vx, Vy, Vz):
#         """Velocity vector.

#         Parameters
#         ----------
#         Vx : float or ndarray
#             velocity in x-direction
#         Vy : float or ndarray
#             velocity in y-direction
#         Vz : float or ndarray
#             velocity in z-direction

#         """
#         super(DirectionVector, self).__init__(Vx, Vy, Vz)


#     def hubToAzimuth(self, Lambda, Omega, yh, zh):

#         yh = np.array(yh)
#         zh = np.array(zh)
#         r = np.sqrt(yh**2 + zh**2)
#         beta = np.arctan2(zh, -yh)
#         Vy = Omega*r*np.sin(beta)
#         Vz = Omega*r*np.cos(beta)

#         V = self + Velocity(0, Vy, Vz)
#         Vnew = DirectionVector(V.x, V.y, V.z).hubToAzimuth(Lambda)

#         return Velocity(Vnew.x, Vnew.y, Vnew.z)


#     def azimuthToHub(self, Lambda, Omega, yh, zh):

#         yh = np.array(yh)
#         zh = np.array(zh)
#         r = np.sqrt(yh**2 + zh**2)
#         beta = np.arctan2(zh, -yh)
#         Vy = Omega*r*np.sin(beta)
#         Vz = Omega*r*np.cos(beta)

#         V = DirectionVector(self.x, self.y, self.z).azimuthToHub(Lambda)
#         Vnew = V - Velocity(0, Vy, Vz)

#         return Velocity(Vnew.x, Vnew.y, Vnew.z)





# def _FM(f, m):
#     return ForceMoment(f.x, f.y, f.z, m.x, m.y, m.z)


# class ForceMoment:

#     def __init__(self, Fx, Fy, Fz, Mx, My, Mz):
#         """Force vector and a corresponding moment vector.

#         Parameters
#         ----------
#         Fx : float or ndarray
#             x-direction of force(s)
#         Fy : float or ndarray
#             y-direction of force(s)
#         Fz : float or ndarray
#             z-direction of force(s)
#         Mx : float or ndarray
#             x-direction of moment(s)
#         My : float or ndarray
#             y-direction of moment(s)
#         Mz : float or ndarray
#             z-direction of moment(s)

#         """

#         self.F = DirectionVector(Fx, Fy, Fz)
#         self.M = DirectionVector(Mx, My, Mz)


#     def __translate(self, x, y, z):
#         r = DirectionVector(x, y, z)
#         M = self.M - r.cross(self.F)

#         return self.F, M


#     def windToInertial(self, beta):

#         Fnew = self.F.windToInertial(beta)
#         Mnew = self.M.windToInertial(beta)

#         return _FM(Fnew, Mnew)


#     def inertialToWind(self, beta):
#         """docstring"""

#         Fnew = self.F.inertialToWind(beta)
#         Mnew = self.M.inertialToWind(beta)

#         return _FM(Fnew, Mnew)


#     def yawToWind(self, psi, height):
#         """docstring"""

#         F, M = self.__translate(0.0, 0.0, -height)
#         Fnew = F.yawToWind(psi)
#         Mnew = M.yawToWind(psi)

#         return _FM(Fnew, Mnew)


#     def windToYaw(self, psi, height):
#         """docstring"""

#         F, M = self.__translate(0.0, 0.0, height)
#         Fnew = F.windToYaw(psi)
#         Mnew = M.windToYaw(psi)

#         return _FM(Fnew, Mnew)


#     def hubToYaw(self, Theta, length):

#         F, M = self.__translate(length, 0.0, 0.0)
#         print self.F, F
#         print self.M, M
#         Fnew = F.hubToYaw(Theta)
#         Mnew = M.hubToYaw(Theta)

#         return _FM(Fnew, Mnew)


#     def yawToHub(self, Theta, length):

#         F = self.F.yawToHub(Theta)
#         M = self.M.yawToHub(Theta)
#         Fnew, Mnew = _FM(F, M).__translate(-length, 0.0, 0.0)

#         return _FM(Fnew, Mnew)


#     def hubToAzimuth(self, Lambda, yh, zh):

#         F, M = self.__translate(0.0, yh, zh)
#         Fnew = F.hubToAzimuth(Lambda)
#         Mnew = M.hubToAzimuth(Lambda)

#         return _FM(Fnew, Mnew)


#     def azimuthToHub(self, Lambda, yh, zh):

#         F = self.F.azimuthToHub(Lambda)
#         M = self.M.azimuthToHub(Lambda)
#         Fnew, Mnew = _FM(F, M).__translate(0.0, -yh, -zh)

#         return _FM(Fnew, Mnew)


#     def azimuthToBlade(self, Phi):

#         Fnew = self.F.azimuthToBlade(Phi)
#         Mnew = self.M.azimuthToBlade(Phi)

#         return _FM(Fnew, Mnew)


#     def bladeToAzimuth(self, Phi):

#         Fnew = self.F.bladeToAzimuth(Phi)
#         Mnew = self.M.bladeToAzimuth(Phi)

#         return _FM(Fnew, Mnew)


#     def airfoilToBlade(self, theta):

#         Fnew = self.F.airfoilToBlade(theta)
#         Mnew = self.M.airfoilToBlade(theta)

#         return _FM(Fnew, Mnew)


#     def bladeToAirfoil(self, theta):

#         Fnew = self.F.bladeToAirfoil(theta)
#         Mnew = self.M.bladeToAirfoil(theta)

#         return _FM(Fnew, Mnew)


#     def profileToAirfoil(self, d):

#         F, M = self.__translate(d, 0.0, 0.0)
#         Fnew = F.profileToAirfoil()
#         Mnew = M.profileToAirfoil()

#         return _FM(Fnew, Mnew)


#     def airfoilToProfile(self, d):

#         F = self.F.airfoilToProfile()
#         M = self.M.airfoilToProfile()
#         Fnew, Mnew = _FM(F, M).__translate(-d, 0.0, 0.0)

#         return _FM(Fnew, Mnew)


#     def __iadd__(self, other):
#         self.F += other.F
#         self.M += other.M

#         return self

#     def __div__(self, other):
#         F = DirectionVector(self.F.x, self.F.y, self.F.z).__div__(other)
#         M = DirectionVector(self.M.x, self.M.y, self.M.z).__div__(other)

#         return _FM(F, M)


# def example():
#     # from twister.common import DirectionVector

#     # define a thrust load in the rotor-aligned coordinate system
#     T = 1.0e6
#     thrust = DirectionVector(T, 0.0, 0.0)  # thrust load in the x_r direction

#     # define angles (in degrees)
#     tilt = 10.0
#     yaw = 15.0

#     # translate to wind-aligned coordinate system (through an intermediate frame)
#     Fwind = thrust.rotorToYaw(tilt).yawToWind(yaw)

#     # print the components
#     print 'Fwind components (x 10^6 N) =', Fwind.x/1e6, Fwind.y/1e6, Fwind.z/1e6


if __name__ == '__main__':

    x = DirectionVector(4.0, 5.0, 6.0)

    print x

    x = DirectionVector([4.0]*2, [5.0]*2, [6.0]*2)

    print x

    # p = Position(1.0, 2.0, 3.0)

    # x = DirectionVector(4.0, 5.0, 6.0)

    # print p - x

    # y = p - x

    # print p.__class__
    # print x.__class__
    # print y.__class__

    # print p.yawToWind(0.0, 10.0).yawToWind(0.0, 10.0)

    # # example()


