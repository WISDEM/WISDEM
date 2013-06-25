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
equations with guaranteed convergence", Wind Energy, 2013. (in press)

"""

import numpy as np
from math import pi, radians, cos
from scipy.optimize import brentq
from scipy.interpolate import RectBivariateSpline
from zope.interface import Interface, implements

from airfoilprep import Airfoil
import _bemroutines
import _akima
from csystem import DirectionVector


RPM2RS = pi/30.0


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




class CCBlade:

    def __init__(self, r, chord, theta, af, Rhub, Rtip, B=3, rho=1.225, mu=1.81206e-5,
                 precone=0.0, tilt=0.0, yaw=0.0, shearExp=0.2, hubHt=80.0, nSector=8,
                 tiploss=True, hubloss=True, wakerotation=True, usecd=True, iterRe=1):
        """Constructor for aerodynamic rotor analysis

        Parameters
        ----------
        r : array_like (m)
            radial locations where blade is defined (should be increasing)
        chord : array_like (m)
            corresponding chord length at each section
        theta : array_like (deg)
            corresponding :ref:`twist angle <blade_airfoil_coord>` at each section---
            positive twist decreases angle of attack.
        af : list(AirfoilInterface)
            list of :ref:`AirfoilInterface <airfoil-interface-label>` objects at each section
        Rhub : float (m)
            radial location of hub
        Rtip : float (m)
            radial location of tip
        B : int, optional
            number of blades
        rho : float, optional (kg/m^3)
            freestream fluid density
        mu : float, optional (kg/m/s)
            dynamic viscosity of fluid
        precone : float, optional (deg)
            blade :ref:`precone angle <azimuth_blade_coord>`
        tilt : float, optional (deg)
            nacelle :ref:`tilt angle <yaw_hub_coord>`
        yaw : float, optional (deg)
            nacelle :ref:`yaw angle<wind_yaw_coord>`
        shearExp : float, optional
            shear exponent for a power-law wind profile across hub
        hubHt : float, optional
            hub height used for power-law wind profile.
            U = Uref*(z/z0)**shearExp, where z = hubHt + r*cos(azimuth), z0 = hubHt
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
        self.theta = np.radians(np.array(theta))
        self.af = af
        self.Rhub = Rhub
        self.Rtip = Rtip
        self.B = B
        self.rho = rho
        self.mu = mu
        self.precone = precone
        self.tilt = tilt
        self.yaw = yaw
        self.shearExp = shearExp
        self.hubHt = hubHt

        self.iterRe = iterRe
        self.bemoptions = dict(usecd=usecd, tiploss=tiploss, hubloss=hubloss, wakerotation=wakerotation)


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
        Np = np.zeros(n)
        Tp = np.zeros(n)
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

        # loads must go to zero at tips
        r = np.concatenate(([self.Rhub], self.r, [self.Rtip]))
        theta = np.concatenate(([self.theta[0]], self.theta, [self.theta[-1]]))
        Np = np.concatenate(([0.0], Np, [0.0]))
        Tp = np.concatenate(([0.0], Tp, [0.0]))

        return r, theta, Tp, Np



    def distributedAeroLoads(self, Uinf, Omega, pitch, azimuth):
        """Compute distributed aerodynamic loads along blade.

        Parameters
        ----------
        Uinf : float (m/s)
            freestream velocity
        Omega : float (RPM)
            rotor rotation speed
        pitch : float (deg)
            blade pitch in same direction as :ref:`twist <blade_airfoil_coord>`
            (positive decreases angle of attack)
        azimuth : float (deg)
            :ref:`azimuth angle <hub_azimuth_coord>` where aerodynamic loads should be computed at

        Returns
        -------
        r : ndarray (m)
            radial stations where force is specified (should go all the way from hub to tip)
        theta : ndarray (deg)
            corresponding geometric twist angle (not including pitch)---
            positive twists nose into the wind
        Tp : ndarray (N/m)
            force per unit length tangential to the section in the direction of rotation
        Np : ndarray (N/m)
            force per unit length normal to the section on downwind side

        """

        self.pitch = radians(pitch)

        # get section heights in wind-aligned coordinate system
        heightFromHub = DirectionVector(0*self.r, 0*self.r, self.r).bladeToAzimuth(self.precone)\
            .azimuthToHub(azimuth).hubToYaw(self.tilt).z

        # shear profile
        V = Uinf*(1 + heightFromHub/self.hubHt)**self.shearExp

        # convert velocity to blade reference frame
        Vwind = DirectionVector(V, 0*V, 0*V).windToYaw(self.yaw).yawToHub(self.tilt)\
            .hubToAzimuth(azimuth).azimuthToBlade(self.precone)
        self.Vx = Vwind.x*np.ones_like(self.r)
        self.Vy = Vwind.y + Omega*RPM2RS*self.r*cos(radians(self.precone))

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

                if errf(phi_lower, *args)*errf(phi_upper, *args) > 0:  # a rare but possible case

                    if errf(-pi/4, *args) < 0 and errf(-epsilon, *args) > 0:
                        phi_lower = -pi/4
                        phi_upper = -epsilon
                    else:
                        phi_lower = pi/2
                        phi_upper = pi - epsilon

                try:
                    phi_opt = brentq(errf, phi_lower, phi_upper, args=args)

                except ValueError:

                    print 'user error.  check input values.'
                    phi_opt = 0.0


                phivec[i] = phi_opt
                zero, avec[i], apvec[i] = self.__runBEM(phi_opt, *args)

            # ----------------------------------------------


        # update distributed loads
        r, theta, Tp, Np = self.__evaluateLoads(phivec, avec, apvec)

        return r, np.degrees(theta), Tp, Np


        # # conform to coordinate system
        # Px = Np
        # Py = -Tp
        # Pz = 0*Np

        # return r, np.degrees(theta), Px, Py, Pz



    def evaluate(self, Uinf, Omega, pitch, coefficient=False):
        """Run the aerodynamic analysis at the specified conditions.

        Parameters
        ----------
        Uinf : array_like (m/s)
            freestream velocity
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
        Normalization uses Rtip and rho provided in the constructor:

        CP = P / (q * Uinf * A)

        CT = T / (q * A)

        CQ = Q / (q * A * Rtip)

        where A = pi*Rtip**2 and q = 0.5*rho*Uinf**2

        """

        # rename
        cosPrecone = cos(radians(self.precone))
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
                r, theta, Tp, Np = self.distributedAeroLoads(Uinf[i], Omega[i], pitch[i], azimuth)

                # interpolate to help smooth out radial discretization
                oldr = r
                r = np.linspace(oldr[0], oldr[-1], 200)
                Tp = _akima.interpolate(oldr, Tp, r)
                Np = _akima.interpolate(oldr, Np, r)

                # integrate Thrust and Torque
                T[i] += B * np.trapz(Np*cosPrecone, r) / nsec
                Q[i] += B * np.trapz(r*Tp*cosPrecone, r) / nsec


        # Power
        P = Q * Omega*RPM2RS

        # normalize if necessary
        if coefficient:
            q = 0.5 * self.rho * Uinf**2
            A = pi * self.Rtip**2
            P /= (q * A * Uinf)
            T /= (q * A)
            Q /= (q * self.Rtip * A)

        return P, T, Q







if __name__ == '__main__':


    # geometry
    Rhub = 1.5
    Rtip = 63.0
    r = np.array([2.87, 5.60, 8.33, 11.75, 15.85, 19.95, 24.05, 28.15, 32.25,
                  36.35, 40.45, 44.55, 48.65, 52.75, 56.17, 58.9, 61.63])
    chord = np.array([3.48, 3.88, 4.22, 4.50, 4.57, 4.44, 4.28, 4.08, 3.85,
                      3.58, 3.28, 2.97, 2.66, 2.35, 2.07, 1.84, 1.59])
    theta = np.array([13.28, 13.28, 13.28, 13.28, 11.77, 10.34, 8.97, 7.67,
                      6.43, 5.28, 4.20, 3.21, 2.30, 1.50, 0.91, 0.48, 0.09])
    B = 3

    # atmosphere
    rho = 1.225
    mu = 1.81206e-5

    import os
    basepath = os.path.join('5MW_files', '5MW_AFFiles') + os.path.sep

    # load all airfoils
    afinit = CCAirfoil.initFromAerodynFile
    airfoil_types = [0]*8
    airfoil_types[0] = afinit(basepath + 'Cylinder1.dat')
    airfoil_types[1] = afinit(basepath + 'Cylinder2.dat')
    airfoil_types[2] = afinit(basepath + 'DU40_A17.dat')
    airfoil_types[3] = afinit(basepath + 'DU35_A17.dat')
    airfoil_types[4] = afinit(basepath + 'DU30_A17.dat')
    airfoil_types[5] = afinit(basepath + 'DU25_A17.dat')
    airfoil_types[6] = afinit(basepath + 'DU21_A17.dat')
    airfoil_types[7] = afinit(basepath + 'NACA64_A17.dat')

    # place at appropriate radial stations, and convert format
    af_idx = [0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7]

    bem_airfoil = [0]*len(r)
    for i in range(len(r)):
        bem_airfoil[i] = airfoil_types[af_idx[i]]


    precone = 20.0
    tilt = 20.0
    yaw = 20.0
    shearExp = 0.2
    hubHt = 80.0
    nSector = 8

    # create CCBlade object
    aeroanalysis = CCBlade(r, chord, theta, bem_airfoil, Rhub, Rtip, B, rho, mu, precone, tilt, yaw,
                           shearExp, hubHt, nSector)


    # set conditions
    Uinf = 10.0
    tsr = 7.55
    pitch = 0.0
    Omega = Uinf*tsr/Rtip * 30.0/pi  # convert to RPM
    azimuth = 90

    # evaluate distributed loads
    rload, theta, Tp, Np = aeroanalysis.distributedAeroLoads(Uinf, Omega, pitch, azimuth)

    # plot
    import matplotlib.pyplot as plt
    rstar = (rload - rload[0]) / (rload[-1] - rload[0])
    plt.plot(rstar, Tp/1e3, 'k', label='lead-lag')
    plt.plot(rstar, Np/1e3, 'r', label='flapwise')
    plt.xlabel('blade fraction')
    plt.ylabel('distributed aerodynamic loads (kN)')
    plt.legend(loc='upper left')
    # plt.show()

    P, T, Q = aeroanalysis.evaluate([Uinf], [Omega], [pitch])

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

    aeroanalysis = CCBlade(r, chord, theta, bem_airfoil, Rhub, Rtip, B, rho, mu, precone=0, tilt=0, yaw=0, shearExp=0)

    CP, CT, CQ = aeroanalysis.evaluate(Uinf, Omega, pitch, coefficient=True)

    plt.plot(tsr, CP, 'r')


    plt.show()