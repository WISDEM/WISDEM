#!/usr/bin/env python
# encoding: utf-8
"""
wind.py

Created by Andrew Ning on 2012-01-20.
Copyright (c) 2012 NREL. All rights reserved.
"""

from math import cos, sin, log, radians
import numpy as np

from zope.interface import implements, Attribute, Interface



class WindModel(Interface):
    """A nondimensional wind profile that can be
    querired for dimensional velocities at specific heights"""

    rho = Attribute("""air density : float (kg/m**3)""")

    mu = Attribute("""air dynamic viscosity : float (kg/(m*s))""")


    def velocityAtHeight(z, Uref, zref, z0=0):
        """Computes the wind velocity vector at specific heihts

        Parameters
        ----------
        z : array_like
            heights where velocity should be evaluted (m)
        Uref : float
            reference velocity (m/s)
        zref : float
            corresponding height for the reference velocity (m)
        z_0 : float, optional
            height of ground/sea (bottom of wind profile) (m)

        Returns
        -------
        Vx : ndarray (m/s)
            x components of velocity in inertial coordinate system
        Vy : ndarray (m/s)
            y components of velocity in inertial coordinate system
        Vz : ndarray (m/s)
            z components of velocity in inertial coordinate system

        """


    # def alpha(self, z):
    #     """effective power-law exponent"""

    #     delta = 0.1*(z-self.z0)

    #     zp = z + delta
    #     zm = z - delta
    #     Up = self.velocityAtHeight([zp])[0]
    #     Um = self.velocityAtHeight([zm])[0]

    #     return (log(Up) - log(Um)) / (log(zp) - log(zm))





class WindWithPowerProfile(object):
    implements(WindModel)


    def __init__(self, alpha=1.0/7, rho=1.225, mu=1.7934e-5, beta=0.0):
        """Power law profile

        Parameters
        ----------
        alpha : float, optional
            exponent used in power law.
            form: U = Uref*((z-z0)/(zref-z0))**alpha
        rho : float, optional
            density of air (kg/m**3)
        mu : float, optional
            dynamic viscosity of air (kg/(m*s))
        beta : float, optional (deg)
            directionality of air.  see inertial and wind-aligned coordinate system.

        """

        self.alpha = alpha
        self.rho = rho
        self.mu = mu
        self.beta = radians(beta)


    def velocityAtHeight(self, z, Uref, zref, z0=0):
        """see interface"""

        # rename local variables
        z = np.array(z)
        beta = self.beta

        # find velocity
        U = np.zeros_like(z)
        idx = [z > z0]
        U[idx] = Uref*((z[idx] - z0)/(zref - z0))**self.alpha

        # resolve into components
        Vx = U * cos(beta)
        Vy = U * sin(beta)
        Vz = 0 * U

        return (Vx, Vy, Vz)





class WindWithLogProfile(object):
    implements(WindModel)

    def __init__(self, z_roughness=10.0, rho=1.225, mu=1.7934e-5, beta=0.0):
        """Logarithmic profile

        Parameters
        ----------
        z_roughness : float, optional
            surface roughness length of the particular terrain (mm)
        rho : float, optional
            density of air (kg/m**3)
        mu : float, optional
            dynamic viscosity of air (kg/(m*s))
        beta : float, optional
            directionality of air.  see inertial and wind-aligned coordinate system.

        """

        self.z_roughness = z_roughness*1e-3
        self.rho = rho
        self.mu = mu
        self.beta = radians(beta)


    def velocityAtHeight(self, z, Uref, zref, z0=0):
        """see interface"""

        # rename local variables
        z = np.array(z)
        beta = self.beta
        z_roughness = self.z_roughness

        # find velocity
        U = np.zeros_like(z)
        idx = [z > z0]
        U[idx] = Uref*(np.log((z[idx] - z0)/z_roughness) / log((zref - z0)/z_roughness))

        # resolve into components
        Vx = U * cos(beta)
        Vy = U * sin(beta)
        Vz = 0 * U
        return (Vx, Vy, Vz)






if __name__ == '__main__':

    wind = WindWithPowerProfile()

    z = np.linspace(0, 12, 100)
    Uref = 5.0
    zref = 10.0
    Vx, Vy, Vz = wind.velocityAtHeight(z, Uref, zref)

    wind2 = WindWithLogProfile()

    Vx2, Vy2, Vz2 = wind2.velocityAtHeight(z, Uref, zref)

    import matplotlib.pyplot as plt
    plt.plot(Vx, z)
    plt.plot(Vx2, z)
    plt.show()

    # import nose
    # nose.main(defaultTest="tests/test_wind.py")
