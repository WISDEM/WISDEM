#!/usr/bin/env python
# encoding: utf-8
"""
toweraero.py

Created by Andrew Ning on 2012-01-20.
Copyright (c) NREL. All rights reserved.
"""

import numpy as np
from math import pi

from twister.common import _akima, DirectionVector

# TODO: rethink the modularity.  especialy wave with cm.


class MonopileAero(object):
    """monopile aero/hydro dynamics"""

    def __init__(self, z, d, z_surface, wind, wave, n=30):
        """z from base to tip"""

        self.z = np.linspace(z[0], z[-1], n)
        self.d = np.interp(self.z, z, d)
        self.z_hub = z[-1]
        self.z_floor = z[0]
        self.z_surface = z_surface
        self.wind = wind
        self.wave = wave


    def distributedLoads(self, Uhub, Uc, yaw):
        """return in yaw-aligned coordinate system"""

        z, Px_wind, Py_wind, Pz_wind, q_wind = self.__distributedWindLoads(Uhub)
        z, Px_wave, Py_wave, Pz_wave, q_wave = self.__distributedWaveLoads(Uc)

        Px = Px_wind + Px_wave
        Py = Py_wind + Py_wave
        Pz = Pz_wind + Pz_wave
        q = q_wind + q_wave

        # rotate to yaw-aligned c.s.
        P = DirectionVector(Px, Py, Pz)

        # rotate to yaw c.s.
        beta = np.degrees(np.arctan2(P.y, P.x))
        Pyaw = P.inertialToWind(beta).windToYaw(yaw)

        return self.z, Pyaw.x, Pyaw.y, Pyaw.z, q


    def __distributedWindLoads(self, Uhub):
        """aerodynamic wind loads

        Parameters
        ----------
        wind : WindModel
            any object that implements the WindModel interface

        Returns
        -------
        z : ndarray (m)
            locations along tower/monopile where loads are evaluated
        Px : ndarray (N/m)
            distributed loads in x-direction
        Py : ndarray (N/m)
            distributed loads in y-direction
        Pz : ndarray (N/m)
            distributed loads in z-direction
        q : ndarray (N/m^2)
            dynamic pressure

        Throws
        ------
        ValueError : if any evaluation point is outside tower bounds (self.z)

        Notes
        -----
        Forces returned in :ref:`inertial coordinate system <inertial_coord>`

        """

        # # ensure not to include values above or below tower
        # if np.any(z > np.amax(self.z)):
        #     raise ValueError('evaluation point outside tower bounds')
        # elif np.any(z < np.amin(self.z)):
        #     raise ValueError('evaluation point outside tower bounds')

        # rename
        rho = self.wind.rho
        mu = self.wind.mu


        # # interpolate
        # d = np.interp(z, self.z, self.d)

        # get velocity magnitude and direction
        (Ux, Uy, Uz) = self.wind.velocityAtHeight(self.z, Uhub, self.z_hub, self.z_surface)
        U = (Ux**2 + Uy**2)**0.5
        psi = np.zeros_like(Ux)
        idx = (Ux != 0)
        psi[idx] = np.arctan(Uy[idx] / Ux[idx])

        # dynamic pressure
        q = 0.5*rho*U**2

        # Reynolds number and drag
        Re = rho*U*self.d/mu
        cd = self.__getDragCoefficient(Re)
        Fp = q*cd*self.d

        # components of distributed loads
        Px = Fp*np.cos(psi)
        Py = Fp*np.sin(psi)
        Pz = 0*Fp

        return self.z, Px, Py, Pz, q



    def __distributedWaveLoads(self, Uc):
        """hydrodynamic wave loads

        Parameters
        ----------
        wave : WaveModel
            any object that implements the WaveModel interface

        Returns
        -------
        z : ndarray (m)
            locations along tower/monopile where loads are evaluated
        Px : ndarray (N/m)
            distributed loads in x-direction
        Py : ndarray (N/m)
            distributed loads in y-direction
        Pz : ndarray (N/m)
            distributed loads in z-direction
        q : ndarray (N/m^2)
            dynamic pressure

        Throws
        ------
        ValueError : if any evaluation point is outside tower bounds (self.z)

        Notes
        -----
        Forces returned in :ref:`inertial coordinate system <inertial_coord>`

        """

        # # ensure not to include values above or below tower
        # if np.any(z > np.amax(self.z)):
        #     raise ValueError('evaluation point outside tower bounds')
        # elif np.any(z < np.amin(self.z)):
        #     raise ValueError('evaluation point outside tower bounds')

        # rename
        rho = self.wave.rho
        mu = self.wave.mu
        cm = self.wave.cm

        # # interpolate
        # d = np.interp(z, self.z, self.d)

        # get velocity and accleration (magnitude and direction)
        (Ux, Uy, Uz) = self.wave.velocityAtHeight(self.z, Uc, self.z_surface, self.z_floor)
        (Ax, Ay, Az) = self.wave.accelerationAtHeight(self.z, Uc, self.z_surface, self.z_floor)

        # convert to magnitude and direction
        U = (Ux**2 + Uy**2)**0.5
        A = (Ax**2 + Ay**2)**0.5
        psi = np.zeros_like(Ux)
        idx = (Ux != 0)
        psi[idx] = np.arctan(Uy[idx] / Ux[idx])

        # dynamic pressure
        q = 0.5*rho*U**2

        # Reynolds number and drag
        Re = rho*U*self.d/mu
        cd = self.__getDragCoefficient(Re)

        # inertial and drag forces
        Fi = rho*cm*pi/4.0*self.d**2*A
        Fd = q*cd*self.d
        Fp = Fi + Fd

        # components of distributed loads
        Px = Fp*np.cos(psi)
        Py = Fp*np.sin(psi)
        Pz = 0*Fp

        return self.z, Px, Py, Pz, q




    def __getDragCoefficient(self, Re):
        """
        Private method which computes drag coefficient over a smooth
        circular cylinder as a function of Reynolds number.

        """

        Re /= 1.0e6

        # "Experiments on the Flow Past a Circular Cylinder at Very High Reynolds Numbers", Roshko
        Re_pt = [0.00001, 0.0001, 0.0010, 0.0100, 0.0200, 0.1220, 0.2000, 0.3000, 0.4000,
                 0.5000, 1.0000, 1.5000, 2.0000, 2.5000, 3.0000, 3.5000, 4.0000, 5.0000, 10.0000]
        cd_pt = [4.0000,  2.0000, 1.1100, 1.1100, 1.2000, 1.2000, 1.1700, 0.9000, 0.5400,
                 0.3100, 0.3800, 0.4600, 0.5300, 0.5700, 0.6100, 0.6400, 0.6700, 0.7000, 0.7000]

        # interpolate
        cd = np.zeros_like(Re)
        cd[Re != 0] = _akima.interpolate(np.log10(Re_pt), cd_pt, np.log10(Re[Re != 0]))


        return cd



if __name__ == '__main__':

    from twister.tower.wind import WindWithPowerProfile
    from wave import LinearWaves



    z = np.array([0.0, 6.0, 12.0, 18.0, 24.0, 30.0, 38.76, 47.52, 56.28,
                  65.04, 73.8, 82.56, 91.32, 100.08, 108.84, 117.6])
    d = np.array([6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 5.787, 5.574, 5.361, 5.148,
                  4.935, 4.722, 4.509, 4.296, 4.083, 3.87])



    Uref = 50.0            # reference wind speed (m/s) - Class I IEC (0.2 X is average for AEP, Ve50 = 1.4*Vref*(z/zhub)^.11 for the blade, max thrust at rated power for tower)
    zref = 117.6  # hubHeight            # height of reference wind speed (m)
                            # (wind speed assumed to be 0 at z_surface)
    z_bottom = 0.0          # reference position of ground (m)
    z_surface = 20.0        # position of water surface (m)
    # alpha = 1.0/7          # power law exponent
    # rho = 1.225             # air density (kg/m^3)
    # mu = rho*1.464e-5       # dynamic viscosity (kg/(m*s))
    # psi = 0.0               # wind direction (deg)

    #cdf = WTPerf.WeibullCDF(10.5, 2.15)

    # Vrated_ref = 11.4
    wind = WindWithPowerProfile()

    # --------------- waves ---------------------

    hs = 7.5   # 7.5 is 10 year extreme             # significant wave height (m)
    T = 19.6                # wave period (s)
    # z_bottom = 0.0          # reference position of ground (m)
    # z_surface = 20.0        # position of water surface (m)
    # g = 9.81                # acceleration due to gravity (m/s^2)
    uc = 1.2  # a               # current speed (m/s)
    # psi = 0.0               # wave direction (deg)
    # rho = 1027.0            # water density (kg/m^3)
    # mu = rho*1.3e-6        # dynamic viscosity (kg/(m*s))
    # cm = 2.0               # hydrodynamic added mass coefficient

    wave = LinearWaves(hs, T)
# -----------------------------------------------


    tower = MonopileAero(z, d, z_surface, wind, wave)

    # z = np.linspace(0, 117.6, 10)
    # Px, Py, Pz, q = tower.distributedWindLoads(z)

    # z = np.linspace(0, 40.0, 100)
    z, Px, Py, Pz, q = tower.distributedLoads(Uref, zref, uc)

    import matplotlib.pyplot as plt
    plt.plot(Px, z)
    # plt.plot(1.2*Px, z)
    plt.show()

