#!/usr/bin/env python
# encoding: utf-8
"""
ccblade_sa.py

Created by S. Andrew Ning on 5/11/2012
Copyright (c) NREL. All rights reserved.

A blade element momentum method using theory detailed in [1]_.  Has the
advantages of guaranteed convergence and at a superlinear rate, and
continuously differentiable output.
_sa == stand-alone version

.. [1] S. Andrew Ning, "A simple solution method for the blade element momentum
equations with guaranteed convergence", Wind Energy, 2013.

"""

import numpy as np
from math import pi, radians
from scipy.optimize import brentq
from scipy.interpolate import RectBivariateSpline
from zope.interface import Interface, implements
import warnings

from airfoilprep import Airfoil
import _bemroutines


try:
    from wisdem.common import cosd, DirectionVector, _akima, RPM2RS, bladePositionAzimuthCS
except ImportError:
    from common.csystem import DirectionVector
    import _akima
    from common.utilities import cosd, RPM2RS, bladePositionAzimuthCS




# ------------------
#  Interfaces
# ------------------


class AirfoilInterface(Interface):
    """Interface for airfoil aerodynamic analysis."""

    def evaluate(alpha, Re):
        """Get lift/drag coefficient at the specified angle of attack and Reynolds number

        Parameters
        ----------
        alpha : float (rad)
            angle of attack
        Re : float
            Reynolds number

        Returns
        -------
        cl : float
            lift coefficient
        cd : float
            drag coefficient

        Notes
        -----
        Any implementation can be used, but to keep the smooth properties
        of CCBlade, the implementation should be C1 continuous.

        """


# ------------------
#  Airfoil Class
# ------------------


class CCAirfoil:
    """A helper class to evaluate airfoil data using a continuously
    differentiable cubic spline"""
    implements(AirfoilInterface)


    def __init__(self, alpha, Re, cl, cd):
        """Setup CCAirfoil from raw airfoil data on a grid.

        Parameters
        ----------
        alpha : array_like (deg)
            angles of attack where airfoil data are defined
            (should be defined from -180 to +180 degrees)
        Re : array_like
            Reynolds numbers where airfoil data are defined
            (can be empty or of length one if not Reynolds number dependent)
        cl : array_like
            lift coefficient 2-D array with shape (alpha.size, Re.size)
            cl[i, j] is the lift coefficient at alpha[i] and Re[j]
        cd : array_like
            drag coefficient 2-D array with shape (alpha.size, Re.size)
            cd[i, j] is the drag coefficient at alpha[i] and Re[j]

        """

        alpha = np.radians(alpha)

        # special case if zero or one Reynolds number (need at least two for bivariate spline)
        if len(Re) < 2:
            Re = [1e1, 1e15]
            cl = np.c_[cl, cl]
            cd = np.c_[cd, cd]

        kx = min(len(alpha)-1, 3)
        ky = min(len(Re)-1, 3)

        # a small amount of smoothing is used to prevent spurious multiple solutions
        self.cl_spline = RectBivariateSpline(alpha, Re, cl, kx=kx, ky=ky, s=0.1)
        self.cd_spline = RectBivariateSpline(alpha, Re, cd, kx=kx, ky=ky, s=0.001)


    @classmethod
    def initFromAerodynFile(cls, aerodynFile):
        """convenience method for initializing with AeroDyn formatted files

        Parameters
        ----------
        aerodynFile : str
            location of AeroDyn style airfoiil file

        Returns
        -------
        af : CCAirfoil
            a constructed CCAirfoil object

        """

        af = Airfoil.initFromAerodynFile(aerodynFile)
        alpha, Re, cl, cd = af.createDataGrid()
        return cls(alpha, Re, cl, cd)


    def evaluate(self, alpha, Re):
        """Get lift/drag coefficient at the specified angle of attack and Reynolds number.

        Parameters
        ----------
        alpha : float (rad)
            angle of attack
        Re : float
            Reynolds number

        Returns
        -------
        cl : float
            lift coefficient
        cd : float
            drag coefficient

        Notes
        -----
        This method uses a spline so that the output is continuously differentiable, and
        also uses a small amount of smoothing to help remove spurious multiple solutions.

        """

        cl = self.cl_spline.ev(alpha, Re)[0]
        cd = self.cd_spline.ev(alpha, Re)[0]

        return cl, cd



# ------------------
#  Main Class: CCBlade
# ------------------


class CCBlade:

    def __init__(self, r, chord, theta, af, Rhub, Rtip, B=3, rho=1.225, mu=1.81206e-5,
                 precone=0.0, tilt=0.0, yaw=0.0, shearExp=0.2, hubHt=80.0, nSector=8,
                 tiploss=True, hubloss=True, wakerotation=True, usecd=True, iterRe=1):
        """Constructor for aerodynamic rotor analysis

        Parameters
        ----------
        r : array_like (m)
            locations defining the blade along a reference axis that
            follows the blade path (values should be increasing).
        chord : array_like (m)
            corresponding chord length at each section
        theta : array_like (deg)
            corresponding :ref:`twist angle <blade_airfoil_coord>` at each section---
            positive twist decreases angle of attack.
        af : list(AirfoilInterface)
            list of :ref:`AirfoilInterface <airfoil-interface-label>` objects at each section
        Rhub : float (m)
            location of hub
        Rtip : float (m)
            location of tip
        B : int, optional
            number of blades
        rho : float, optional (kg/m^3)
            freestream fluid density
        mu : float, optional (kg/m/s)
            dynamic viscosity of fluid
        precone : float or array_like, optional (deg)
            :ref:`hub precone angle <azimuth_blade_coord>`
            can be used for precurve in addition to precone by using an array input (blade length is preserved).
        tilt : float, optional (deg)
            nacelle :ref:`tilt angle <yaw_hub_coord>`
        yaw : float, optional (deg)
            nacelle :ref:`yaw angle<wind_yaw_coord>`
        shearExp : float, optional
            shear exponent for a power-law wind profile across hub
        hubHt : float, optional
            hub height used for power-law wind profile.
            U = Uref*(z/hubHt)**shearExp
        nSector : int, optional
            number of azimuthal sectors to descretize aerodynamic calculation.  automatically set to
            ``1`` if tilt, yaw, and shearExp are all 0.0.  Otherwise set to a minimum of 4.
        tiploss : boolean, optional
            if True, include Prandtl tip loss model
        hubloss : boolean, optional
            if True, include Prandtl hub loss model
        wakerotation : boolean, optional
            if True, include effect of wake rotation (i.e., tangential induction factor is nonzero)
        usecd : boolean, optional
            If True, use drag coefficient in computing induction factors
            (always used in evaluating distributed loads from the induction factors).
            Note that the default implementation may fail at certain points if drag is not included
            (see Section 4.2 in :cite:`Ning2013A-simple-soluti`).  This can be worked around, but has
            not been implemented.
        iterRe : int, optional
            The number of iterations to use to converge Reynolds number.  Generally iterRe=1 is sufficient,
            but for high accuracy in Reynolds number, iterRe=2 iterations can be used.  More than that
            should not be necessary.

        """

        self.r = np.array(r)
        self.chord = np.array(chord)
        self.theta = np.radians(theta)
        self.af = af
        self.Rhub = Rhub
        self.Rtip = Rtip
        self.B = B
        self.rho = rho
        self.mu = mu
        self.tilt = tilt
        self.yaw = yaw
        self.shearExp = shearExp
        self.hubHt = hubHt
        self.bemoptions = dict(usecd=usecd, tiploss=tiploss, hubloss=hubloss, wakerotation=wakerotation)
        self.iterRe = iterRe

        # check if precurve specified
        if isinstance(precone, float) or isinstance(precone, int):
            self.precone = precone*np.ones_like(r)
        else:
            self.precone = np.array(precone)

        # find blade position in azimuthal c.s.
        self.preconefull = np.concatenate(([self.precone[0]], self.precone, [self.precone[-1]]))
        self.rfull = np.concatenate(([Rhub], self.r, [Rtip]))
        blade_az = bladePositionAzimuthCS(self.rfull, self.preconefull)
        self.z_azim_full = blade_az.z

        # actual rotor radius
        self.rotorR = blade_az.z[-1]

        # save intermediate positions (not root and tip)
        self.z_azim = blade_az.z[1:-1]
        self.x_azim = blade_az.x[1:-1]
        self.y_azim = np.zeros_like(self.x_azim)

        # azimuthal discretization
        if self.tilt == 0.0 and self.yaw == 0.0 and self.shearExp == 0.0:
            self.nSector = 1  # no more are necessary
        else:
            self.nSector = max(4, nSector)  # at least 4 are necessary




    def __runBEM(self, phi, r, chord, theta, af, Vx, Vy):
        """private method to run the BEM method.  primarily through Fortran calls."""

        a = 0.0
        ap = 0.0
        for i in range(self.iterRe):

            alpha, W, Re = _bemroutines.relativewind(phi, a, ap, Vx, Vy, self.pitch,
                                                     chord, theta, self.rho, self.mu)

            cl, cd = af.evaluate(alpha, Re)

            fzero, a, ap = _bemroutines.inductionfactors(r, chord, self.Rhub, self.Rtip, phi,
                                                         cl, cd, self.B, Vx, Vy, **self.bemoptions)

        return fzero, a, ap


    def __errorFunction(self, phi, r, chord, theta, af, Vx, Vy):
        """private method.  only want to return residual for Brent's method"""

        fzero, a, ap = self.__runBEM(phi, r, chord, theta, af, Vx, Vy)

        return fzero


    def __evaluateLoads(self, phi, a, ap):
        """private method.  convert solution for local inflow angle and induction factors
        to normal and tangential loads.

        """

        n = len(a)
        cl = np.zeros(n)
        cd = np.zeros(n)
        W = np.zeros(n)

        for i in range(n):

            alpha, W[i], Re = _bemroutines.relativewind(phi[i], a[i], ap[i], self.Vx[i], self.Vy[i],
                                                        self.pitch, self.chord[i], self.theta[i],
                                                        self.rho, self.mu)
            cl[i], cd[i] = self.af[i].evaluate(alpha, Re)

        cn = cl*np.cos(phi) + cd*np.sin(phi)  # these expressions should always contain drag
        ct = cl*np.sin(phi) - cd*np.cos(phi)

        q = 0.5*self.rho*W**2
        Np = cn*q*self.chord
        Tp = ct*q*self.chord

        # loads must go to zero at ends
        Np = np.concatenate(([0.0], Np, [0.0]))
        Tp = np.concatenate(([0.0], Tp, [0.0]))
        theta = np.concatenate(([self.theta[0]], self.theta, [self.theta[-1]]))

        return self.rfull, Tp, Np, np.degrees(theta), self.preconefull



    def distributedAeroLoads(self, Uinf, Omega, pitch, azimuth):
        """Compute distributed aerodynamic loads along blade.

        Parameters
        ----------
        Uinf : float or array_like (m/s)
            hub height wind speed (float).  If desired, an array can be input which specifies
            the velocity at each radial location along the blade (useful for analyzing loads
            behind tower shadow for example).  In either case shear corrections will be applied.
        Omega : float (RPM)
            rotor rotation speed
        pitch : float (deg)
            blade pitch in same direction as :ref:`twist <blade_airfoil_coord>`
            (positive decreases angle of attack)
        azimuth : float (deg)
            the :ref:`azimuth angle <hub_azimuth_coord>` where aerodynamic loads should be computed at

        Returns
        -------
        r : ndarray (m)
            radial stations along blade where force is specified (all the way from hub to tip)
        Tp : ndarray (N/m)
            force per unit length tangential to the section in the direction of rotation
        Np : ndarray (N/m)
            force per unit length normal to the section on downwind side
        theta : ndarray (deg)
            corresponding geometric :ref:`twist angle <blade_airfoil_coord>` (not including pitch)---
            positive twists nose into the wind
        precone : ndarray (deg)
            corresponding :ref:`precone/precurve <azimuth_blade_coord>` angles (these later two outputs
            are provided to facilitate coordinate transformations)

        """

        self.pitch = radians(pitch)

        # get section heights in wind-aligned coordinate system
        heightFromHub = DirectionVector(self.x_azim, self.y_azim, self.z_azim).azimuthToHub(azimuth).hubToYaw(self.tilt).z

        # shear profile
        V = Uinf*np.abs(1 + heightFromHub/self.hubHt)**self.shearExp

        # compute wind and rotation velocity in azimuthal reference frame
        Vwind = DirectionVector(V, 0*V, 0*V).windToYaw(self.yaw).yawToHub(self.tilt).hubToAzimuth(azimuth)
        OmegaV = DirectionVector(Omega*RPM2RS, 0.0, 0.0)
        RV = DirectionVector(self.x_azim, self.y_azim, self.z_azim)
        Vrot = -OmegaV.cross(RV)  # negative sign because relative wind opposite to rotation

        # combine and rotate to local blade frame
        Vtotal = (Vwind + Vrot).azimuthToBlade(self.precone)

        self.Vx = Vtotal.x
        self.Vy = Vtotal.y

        # initialize
        n = len(self.r)
        avec = np.zeros(n)
        apvec = np.zeros(n)
        phivec = np.zeros(n)

        if Omega == 0:  # non-rotating

            phivec = pi/2.0 * np.ones(n)
            avec = np.zeros(n)
            apvec = np.zeros(n)

        else:

            errf = self.__errorFunction

            # ---------------- loop across blade ------------------
            for i in xrange(n):

                # index dependent arguments
                args = (self.r[i], self.chord[i], self.theta[i], self.af[i], self.Vx[i], self.Vy[i])

                # set standard limits
                epsilon = 1e-6
                phi_lower = epsilon
                phi_upper = pi/2

                if errf(phi_lower, *args)*errf(phi_upper, *args) > 0:  # an uncommon but possible case

                    if errf(-pi/4, *args) < 0 and errf(-epsilon, *args) > 0:
                        phi_lower = -pi/4
                        phi_upper = -epsilon
                    else:
                        phi_lower = pi/2
                        phi_upper = pi - epsilon

                try:
                    phi_opt = brentq(errf, phi_lower, phi_upper, args=args)

                except ValueError:

                    warnings.warn('error.  check input values.')
                    phi_opt = 0.0


                phivec[i] = phi_opt
                zero, avec[i], apvec[i] = self.__runBEM(phi_opt, *args)

            # ----------------------------------------------

        # # update distributed loads
        # rload, Tp, Np, theta, precone = self.__evaluateLoads(phivec, avec, apvec)


        # # get gradients
        # delta = 1e-6
        # zero_new = np.zeros(n)

        # phivec += delta
        # ignore, Tp_d, Np_d, ignore, ignore = self.__evaluateLoads(phivec, avec, apvec)
        # for i in range(n):
        #     args = (self.r[i], self.chord[i], self.theta[i], self.af[i], self.Vx[i], self.Vy[i])
        #     zero_new[i], a, ap = self.__runBEM(phivec[i], *args)
        # phivec -= delta

        # dNpdy = (Np_d[1:-1] - Np[1:-1]) / delta
        # dTpdy = (Tp_d[1:-1] - Tp[1:-1]) / delta
        # drdy = (zero_new - zero) / delta

        # dNpdx = np.zeros((3, n))
        # dTpdx = np.zeros((3, n))
        # drdx = np.zeros((3, n))

        # self.r += delta
        # ignore, Tp_d, Np_d, ignore, ignore = self.__evaluateLoads(phivec, avec, apvec)
        # for i in range(n):
        #     args = (self.r[i], self.chord[i], self.theta[i], self.af[i], self.Vx[i], self.Vy[i])
        #     zero_new[i], a, ap = self.__runBEM(phivec[i], *args)
        # self.r -= delta

        # dNpdx[0, :] = (Np_d[1:-1] - Np[1:-1]) / delta
        # dTpdx[0, :] = (Tp_d[1:-1] - Tp[1:-1]) / delta
        # drdx[0, :] = (zero_new - zero) / delta


        # self.chord += delta
        # ignore, Tp_d, Np_d, ignore, ignore = self.__evaluateLoads(phivec, avec, apvec)
        # for i in range(n):
        #     args = (self.r[i], self.chord[i], self.theta[i], self.af[i], self.Vx[i], self.Vy[i])
        #     zero_new[i], a, ap = self.__runBEM(phivec[i], *args)
        # self.chord -= delta

        # dNpdx[1, :] = (Np_d[1:-1] - Np[1:-1]) / delta
        # dTpdx[1, :] = (Tp_d[1:-1] - Tp[1:-1]) / delta
        # drdx[1, :] = (zero_new - zero) / delta

        # self.theta += delta
        # ignore, Tp_d, Np_d, ignore, ignore = self.__evaluateLoads(phivec, avec, apvec)
        # for i in range(n):
        #     args = (self.r[i], self.chord[i], self.theta[i], self.af[i], self.Vx[i], self.Vy[i])
        #     zero_new[i], a, ap = self.__runBEM(phivec[i], *args)
        # self.theta -= delta

        # dNpdx[2, :] = (Np_d[1:-1] - Np[1:-1]) / delta
        # dTpdx[2, :] = (Tp_d[1:-1] - Tp[1:-1]) / delta
        # drdx[2, :] = (zero_new - zero) / delta

        # DTpDx = dTpdx - dTpdy/drdy*drdx
        # DNpDx = dNpdx - dNpdy/drdy*drdx

        # return rload, Tp, Np, theta, precone, DTpDx, DNpDx

        # update distributed loads
        return self.__evaluateLoads(phivec, avec, apvec)



    def evaluate(self, Uinf, Omega, pitch, coefficient=False):
        """Run the aerodynamic analysis at the specified conditions.

        Parameters
        ----------
        Uinf : array_like (m/s)
            hub height wind speed
        Omega : array_like (RPM)
            rotor rotation speed
        pitch : array_like (deg)
            blade pitch setting
        coefficient : bool, optional
            if True, results are returned in nondimensional form

        Returns
        -------
        P or CP : ndarray (W)
            power or power coefficient
        T or CT : ndarray (N)
            thrust or thrust coefficient (magnitude)
        Q or CQ : ndarray (N*m)
            torque or torque coefficient (magnitude)

        Notes
        -----

        CP = P / (q * Uinf * A)

        CT = T / (q * A)

        CQ = Q / (q * A * R)

        note: that the rotor radius R, may not actually be Rtip in the case
            of precone/precurve

        """

        # rename
        B = self.B
        nsec = self.nSector

        # initialize
        Uinf = np.array(Uinf)
        Omega = np.array(Omega)
        pitch = np.array(pitch)

        npts = len(Uinf)
        T = np.zeros(npts)
        Q = np.zeros(npts)
        P = np.zeros(npts)


        for i in range(npts):  # iterate across conditions

            for j in range(nsec):  # integrate across azimuth
                azimuth = 360.0*float(j)/nsec

                # run analysis
                r, Tp, Np, theta, precone = self.distributedAeroLoads(Uinf[i], Omega[i], pitch[i], azimuth)

                thrust = Np*cosd(precone)
                torque = Tp*self.z_azim_full

                # smooth out integration
                oldr = r
                r = np.linspace(oldr[0], oldr[-1], 200)
                thrust = _akima.interpolate(oldr, thrust, r)
                torque = _akima.interpolate(oldr, torque, r)

                # integrate Thrust and Torque
                T[i] += B * np.trapz(thrust, r) / nsec
                Q[i] += B * np.trapz(torque, r) / nsec


        # Power
        P = Q * Omega*RPM2RS

        # normalize if necessary
        if coefficient:
            q = 0.5 * self.rho * Uinf**2
            A = pi * self.rotorR**2
            P /= (q * A * Uinf)
            T /= (q * A)
            Q /= (q * self.rotorR * A)

        return P, T, Q







if __name__ == '__main__':

    # geometry
    Rhub = 1.5
    Rtip = 63.0

    r = np.array([2.8667, 5.6000, 8.3333, 11.7500, 15.8500, 19.9500, 24.0500,
                  28.1500, 32.2500, 36.3500, 40.4500, 44.5500, 48.6500, 52.7500,
                  56.1667, 58.9000, 61.6333])
    chord = np.array([3.542, 3.854, 4.167, 4.557, 4.652, 4.458, 4.249, 4.007, 3.748,
                      3.502, 3.256, 3.010, 2.764, 2.518, 2.313, 2.086, 1.419])
    theta = np.array([13.308, 13.308, 13.308, 13.308, 11.480, 10.162, 9.011, 7.795,
                      6.544, 5.361, 4.188, 3.125, 2.319, 1.526, 0.863, 0.370, 0.106])
    B = 3  # number of blades

    # atmosphere
    rho = 1.225
    mu = 1.81206e-5

    import os
    afinit = CCAirfoil.initFromAerodynFile  # just for shorthand
    basepath = os.path.join('5MW_files', '5MW_AFFiles') + os.path.sep

    # load all airfoils
    airfoil_types = [0]*8
    airfoil_types[0] = afinit(basepath + 'Cylinder1.dat')
    airfoil_types[1] = afinit(basepath + 'Cylinder2.dat')
    airfoil_types[2] = afinit(basepath + 'DU40_A17.dat')
    airfoil_types[3] = afinit(basepath + 'DU35_A17.dat')
    airfoil_types[4] = afinit(basepath + 'DU30_A17.dat')
    airfoil_types[5] = afinit(basepath + 'DU25_A17.dat')
    airfoil_types[6] = afinit(basepath + 'DU21_A17.dat')
    airfoil_types[7] = afinit(basepath + 'NACA64_A17.dat')

    # place at appropriate radial stations
    af_idx = [0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7]

    af = [0]*len(r)
    for i in range(len(r)):
        af[i] = airfoil_types[af_idx[i]]


    tilt = -5.0
    precone = 2.5
    yaw = 0.0
    shearExp = 0.2
    hubHt = 80.0
    nSector = 8

    # create CCBlade object
    aeroanalysis = CCBlade(r, chord, theta, af, Rhub, Rtip, B, rho, mu,
                           precone, tilt, yaw, shearExp, hubHt, nSector)


    # set conditions
    Uinf = 10.0
    tsr = 7.55
    pitch = 0.0
    Omega = Uinf*tsr/Rtip * 30.0/pi  # convert to RPM
    azimuth = 90

    # evaluate distributed loads
    rload, Tp, Np, theta, precone = aeroanalysis.distributedAeroLoads(Uinf, Omega, pitch, azimuth)

    # plot
    import matplotlib.pyplot as plt
    rstar = (rload - rload[0]) / (rload[-1] - rload[0])
    plt.plot(rstar, Tp/1e3, 'k', label='lead-lag')
    plt.plot(rstar, Np/1e3, 'r', label='flapwise')
    plt.xlabel('blade fraction')
    plt.ylabel('distributed aerodynamic loads (kN)')
    plt.legend(loc='upper left')

    CP, CT, CQ = aeroanalysis.evaluate([Uinf], [Omega], [pitch], coefficient=True)

    print CP, CT, CQ


    tsr = np.linspace(2, 14, 50)
    Omega = 10.0 * np.ones_like(tsr)
    Uinf = Omega*pi/30.0 * Rtip/tsr
    pitch = np.zeros_like(tsr)

    CP, CT, CQ = aeroanalysis.evaluate(Uinf, Omega, pitch, coefficient=True)

    plt.figure()
    plt.plot(tsr, CP, 'k')
    plt.xlabel('$\lambda$')
    plt.ylabel('$c_p$')

    plt.show()
