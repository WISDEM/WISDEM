#!/usr/bin/env python
# encoding: utf-8
"""
soil.py

Created by Andrew Ning on 2/27/2012.
Copyright (c)  NREL. All rights reserved.
"""

import numpy as np

from zope.interface import implements, Attribute, Interface


class SoilModel(Interface):
    """Estimates equivalent spring stiffness at the base of a
    foundation supported in soil."""

    depth = Attribute("""depth of soil (m)""")
    infinity = Attribute("""floating point representation of infinitely stiff directions""")

    def equivalentSpringStiffnessAtBase(r0, t0):
        """computes spring stiffness at base of tower

        Parameters
        ----------
        r0 : float
            radius of base of tower (m)
        t0 : float
            wall thickness of base of tower (m)

        Returns
        -------
        [kx, ktxtx, ky, ktyty, kz, ktztz] : ndarray of floats
            spring stiffness in inertial coordinate system.
            txtx means rotation about the x-axis

        Notes
        -----
        Uses the inertial coordinate system.
        x is downstream if there is no sideslip angle
        z is vertical and y is given by the RHR

        """



class SoilModelCylindricalFoundation():
    implements(SoilModel)
    """Estimates equivalent spring stiffness at the base of a
    foundation supported in soil.

    A cylindrical shell section is assumed.
    Method is based on Chapter 4: Geotechnical Considerations of
    "Design of Structures and Foundations for Vibration Machines".

    """


    def __init__(self, G, nu, h, rigid):
        """computes spring stiffness at base of tower

        Parameters
        ----------
        G : float
            shear modulus of soil (Pa)
        nu : float
            Poisson's ratio of soil
        h : float
            depth of foundation in the soil (m)
        rigid : array_like of int
            list of indices for directions which should be considered
            as infinitely rigid.  order is x, theta_x, y, theta_y, z, theta_z

        """
        self.G = G
        self.nu = nu
        self.depth = h
        self.rigid = rigid

        self.infinity = float('inf')


    def equivalentSpringStiffnessAtBase(self, r0, t0):
        """see interface"""

        G = self.G
        nu = self.nu
        h = self.depth

        # vertical
        eta = 1.0 + 0.6*(1.0-nu)*h/r0
        k_z = 4*G*r0*eta/(1.0-nu)

        # horizontal
        eta = 1.0 + 0.55*(2.0-nu)*h/r0
        k_x = 32.0*(1.0-nu)*G*r0*eta/(7.0-8.0*nu)

        # rocking
        eta = 1.0 + 1.2*(1.0-nu)*h/r0 + 0.2*(2.0-nu)*(h/r0)**3
        k_thetax = 8.0*G*r0**3*eta/(3.0*(1.0-nu))

        # torsional
        k_phi = 16.0*G*r0**3/3.0

        k = np.array([k_x, k_thetax, k_x, k_thetax, k_z, k_phi])
        k[self.rigid] = self.infinity

        return k
