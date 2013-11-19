#!/usr/bin/env python
# encoding: utf-8
"""
wave.py

Created by Andrew Ning on 2012-01-20.
Copyright (c) 2012 NREL. All rights reserved.
"""

import math
import numpy as np
from scipy.optimize import brentq

from zope.interface import implements, Attribute, Interface



class WaveModel(Interface):
    """A nondimensional wave profile that can be querired for
    dimensional velocities and accelerations at specific heights"""

    rho = Attribute("""water density (kg/m**3)""")
    mu = Attribute("""water dynamic viscosity (kg/(m*s))""")
    cm = Attribute("""added mass coefficient""")


    def velocityAtHeight(z, Uc, z_surface, z_floor=0.0):
        """Computes (maximum) wave velocity vector at heights z

        Parameters
        ----------
        z : array_like
            z-locations where velocity should be evaluated (m)
        z_surface : float
            z location of water surface (m)
        z_floor : float, optional
            z location of sea floor (m)

        Returns
        -------
        Vx : ndarray (m/s)
            x-component of velocity in inertial coordinate system
        Vy : ndarray (m/s)
            y-component of velocity in inertial coordinate system
        Vz : ndarray (m/s)
            z-component of velocity in inertial coordinate system

        Notes
        -----
        Uses the inertial coordinate system.
        x is downstream if there is no sideslip angle
        z is vertical and y is given by the RHR

        """


    def accelerationAtHeight(z, Uc, z_surface, z_floor=0.0):
        """Computes (maximum) wave acceleration vector at heights z

        Parameters
        ----------
        z : array_like
            z-locations where acceleration should be evaluated (m)
        z_surface : float
            z location of water surface (m)
        z_floor : float, optional
            z location of sea floor (m)

        Returns
        -------
        Ax : ndarray (m/s^2)
            x-component of acceleration in inertial coordinate system
        Ay : ndarray (m/s^2)
            y-component of acceleration in inertial coordinate system
        Az : ndarray (m/s^2)
            z-component of acceleration in inertial coordinate system

        Notes
        -----
        Uses the inertial coordinate system.
        x is downstream if there is no sideslip angle
        z is vertical and y is given by the RHR

        """



class LinearWaves:
    implements(WaveModel)


    def __init__(self, hs, T, g=9.81, rho=1027.0, mu=1.3351e-3, cm=2.0, beta=0.0):
        """Initializes waves using linear (Airy) wave theory.

        Parameters
        ----------
        hs : float
            significant wave height (crest-to-trough) (m)
        T : float
            period of waves (s)
        uc : float
            mean current speed (m/s)
        g : float, optional
            acceleration of gravity (m/s**2)
        rho : float, optional
            water density (kg/m**3)
        mu : float, optional
            water dynamic viscosity (kg/(m*s))
        cm : float, optional
            mass coefficient
        beta : float, optional
            directionality of current (beta = 0 means water is aligned with x direction) (deg)

        """

        self.hs = hs
        self.T = T
        self.g = g
        self.rho = rho
        self.mu = mu
        self.cm = cm
        self.beta = math.radians(beta)



    def velocityAtHeight(self, z, Uc, z_surface, z_floor=0.0):
        """see interface"""

        z = np.array(z)

        # water depth
        d = z_surface - z_floor

        # design wave height
        h = 1.1*self.hs

        # circular frequency
        omega = 2.0*math.pi/self.T

        # compute wave number from dispersion relationship
        k = brentq(lambda k: omega**2 - self.g*k*math.tanh(d*k), 0, 10*omega**2/self.g)

        # zero at surface
        z_rel = z - z_surface

        # maximum velocity
        Umax = h/2.0*omega*np.cosh(k*(z_rel + d))/math.sinh(k*d) + Uc

        # check heights
        Umax[np.logical_or(z < z_floor, z > z_surface)] = 0

        # get components
        Vx = Umax * math.cos(self.beta)
        Vy = Umax * math.sin(self.beta)
        Vz = 0 * Umax

        return (Vx, Vy, Vz)


    def accelerationAtHeight(self, z, Uc, z_surface, z_floor=0.0):
        """see interface"""

        # compute velocities (could save computed values, but cost is small)
        (Vx, Vy, Vz) = self.velocityAtHeight(z, Uc, z_surface, z_floor)

        # circular frequency
        omega = 2.0*math.pi/self.T

        # compute accelerations
        Ax = Vx * omega
        Ay = Vy * omega
        Az = 0 * Ax

        return (Ax, Ay, Az)
