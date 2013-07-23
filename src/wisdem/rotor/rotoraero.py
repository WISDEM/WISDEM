#!/usr/bin/env python
# encoding: utf-8
"""
rotoraero.py

Created by Andrew Ning on 2012-07-03.
Copyright (c) NREL. All rights reserved.
"""

import numpy as np
from math import pi, sqrt
from scipy import interpolate, optimize
from zope.interface import Interface, Attribute, implements

from wisdem.common import DirectionVector, _akima, RPM2RS, RS2RPM, sind, cosd, bladePositionAzimuthCS



# ------------------
#  Interfaces
# ------------------


class RotorAeroAnalysisInterface(Interface):
    """Interface for evaluating rotor aerodynamic performance used by :class:`RotorAero`."""

    # r = Attribute("""radial positions along blade : float (m)""")

    rotorR = Attribute("""radius of rotor used in normalization of pressure coefficient : float (m)""")

    rho = Attribute("""freestream air density used in normalized of pressure coefficient : float (kg/m**3)""")

    yaw = Attribute(""":ref:`yaw angle <wind_yaw_coord>` about axis through tower.
                    positive CCW rotation when viewed from above : float (deg)""")

    tilt = Attribute(""":ref:`tilt angle <yaw_hub_coord>` about axis parallel to ground.
                     positive tilts rotor up for upstream configuration : float (deg)""")

    precone = Attribute(""":ref:`hub precone angle <azimuth_blade_coord>`, positive tilts blades
                        away from tower for upwind configuration. : ndarray (deg)""")

    # # hubPrecone = Attribute("""precone at hub""")

    nBlade = Attribute("""number of blades : int""")



    def distributedAeroLoads(Uinf, Omega, pitch, azimuth):
        """Compute distributed aerodynamic loads along blade
        in the :ref:`blade-aligned coordinate system <blade_coord>`.

        Parameters
        ----------
        Uinf : float (m/s)
            freestream velocity
        Omega : float (RPM)
            rotor rotation speed
        pitch : float (deg)
            blade pitch setting (see :ref:`twist definition <blade_airfoil_coord>`)
        azimuth: float (deg)
            azimuth angle (see :ref:`azimuth definition <hub_azimuth_coord>`).
            this parameter can be ignored if the analysis only provides an azimuthal-average value

        Returns
        -------
        r : ndarray (m)
            radial stations where force is specified (should go all the way from hub to tip)
        Px : ndarray (N/m)
            force per unit length in :ref:`blade-aligned <blade_coord>` x-direction
            (normal direction, from airfoil lower to upper surface for no twist)
        Py : ndarray (N/m)
            force per unit length in :ref:`blade-aligned <blade_coord>` y-direction
            (opposite to tangential rotation direction)
        Pz : ndarray (N/m)
            force per unit length in :ref:`blade-aligned <blade_coord>` z-direction
            (positive toward blade tip)
        theta : ndarray (deg)
            corresponding geometric twist angle (not including pitch).  positive twists nose into the wind.
        precone : ndarray (deg)
            corresponding precone angles

        """



    def evaluate(Uinf, Omega, pitch, coefficient=False):
        """Run the aerodynamic analysis at the specified conditions.
        Thrust and Torque should both be magnitudes.

        Parameters
        ----------
        Uinf : array_like (m/s)
            freestream velocity
        Omega : array_like (RPM)
            rotor rotation speed
        pitch : array_like (deg)
            blade pitch setting
        coefficient : bool, optional
            If True results are returned in nondimensional form

        Returns
        -------
        P or CP : ndarray (W)
            power or power coefficient
        T or CT : ndarray (N)
            thrust or thrust coefficient (magnitude)
        Q or CQ : ndarray (N*m)
            torque or torque coefficient (magnitude)

        See Also
        --------
        integrateForPower : provides a default implementation that can be used

        Notes
        -----
        Normalization uses rotorR and rho
        defined in the properties::

        CP = P / (q * Uinf * A)

        CT = T / (q * A)

        CQ = Q / (q * A * R)

        where A = pi*rotorR**2 and q = 0.5*rho*Uinf**2

        """


    def hubLoads(Uinf, Omega, pitch):
        """Compute forces and moments at the rotor hub in the :ref:`hub-aligned coordinate system
        <hub_coord>`.

        Parameters
        ----------
        Uinf : float (m/s)
            freestream velocity
        Omega : float (RPM)
            rotor rotation speed
        pitch : float (deg)
            blade pitch setting

        Returns
        -------
        Fx, Fy, Fz, Mx, My, Mz : float (N, N*m)
            Forces and moments in the x, y, and z directions using the
            hub-aligned coordinate system.

        See Also
        --------
        integrateForHubLoads : provides a default implementation that can be used

        """


class DrivetrainEfficiencyInterface(Interface):
    """interface for providing efficiency of drivetrain.  used in rotoraero.py"""

    def efficiency(power, ratedPower):
        """estimate drivetrain efficiency

        Parameters
        ----------
        power : array_like (W)
            array of power conditions to evaluate efficiency at
        ratedPower : float (W)
            rated power of the turbine

        Returns
        -------
        eff : ndarray
            drivetrain efficiency at each power setting

        """




# ------------------
#  Base Classes
# ------------------


class RotorAeroAnalysisBase(object):
    """A base class that provides some default functionality for RotorAeroAnalysisInterface"""
    implements(RotorAeroAnalysisInterface)


    def __init__(self, nAzimuth):
        """Constructor

        Parameters
        ----------
        nAzimuth : int
            number of azimuthal sections to use in integration

        """
        self.nAzimuth = nAzimuth




    def evaluate(self, Uinf, Omega, pitch, coefficient=False):
        """see :meth:`interface <RotorAeroAnalysisInterface.evaluate>`"""

        # rename
        B = self.nBlade
        nsec = self.nAzimuth

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
                r, Px, Py, Pz, theta, precone = self.distributedAeroLoads(Uinf[i], Omega[i], pitch[i], azimuth)

                # swap coordinate directions
                Tp = -Py
                Np = Px
                Rp = Pz

                # thrust and tangential loads
                blade_azim = bladePositionAzimuthCS(r, precone)

                thrust = Np*cosd(precone)-Rp*sind(precone)
                torque = Tp*blade_azim.z

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




    def hubLoads(self, Uinf, Omega, pitch):
        """see :meth:`interface <RotorAeroAnalysisInterface.hubLoads>`"""

        # rename
        B = self.nBlade
        nsec = self.nAzimuth

        Fx = 0.0
        Fy = 0.0
        Fz = 0.0
        Mx = 0.0
        My = 0.0
        Mz = 0.0

        for i in range(nsec):
            azimuth = 360.0*float(i)/nsec

            # run analysis
            r, Px, Py, Pz, theta, precone = self.distributedAeroLoads(Uinf, Omega, pitch, azimuth)

            # loads in azimuthal c.s.
            P = DirectionVector(Px, Py, Pz).bladeToAzimuth(precone)

            # location of loads in azimuth c.s.
            position = bladePositionAzimuthCS(r, precone)

            # distributed bending load in azimuth coordinate ysstem
            Mp = position.cross(P)

            # convert to hub c.s.
            P = P.azimuthToHub(azimuth)
            Mp = Mp.azimuthToHub(azimuth)

            # interpolate to help smooth out radial discretization
            oldr = r
            r = np.linspace(oldr[0], oldr[-1], 200)
            Px = _akima.interpolate(oldr, P.x, r)
            Py = _akima.interpolate(oldr, P.y, r)
            Pz = _akima.interpolate(oldr, P.z, r)
            Mpx = _akima.interpolate(oldr, Mp.x, r)
            Mpy = _akima.interpolate(oldr, Mp.y, r)
            Mpz = _akima.interpolate(oldr, Mp.z, r)


            # integrate across span
            Fx += B * np.trapz(Px, r) / nsec
            Fy += B * np.trapz(Py, r) / nsec
            Fz += B * np.trapz(Pz, r) / nsec

            Mx += B * np.trapz(Mpx, r) / nsec
            My += B * np.trapz(Mpy, r) / nsec
            Mz += B * np.trapz(Mpz, r) / nsec


        if nsec == 1:  # symmetry
            Fy = 0.0
            Fz = 0.0
            My = 0.0
            Mz = 0.0

        return (Fx, Fy, Fz, Mx, My, Mz)







# ----------------------------
#  Main Class
# ----------------------------


class RotorAero(object):
    """Utilizes an aerodynamic model of a wind turbine rotor to generate
    power curves, etc.
    """


    def __init__(self, aeroanalysis, drivetrain, machineType, npts_cp_curve=30):
        """Constructor

        Parameters
        ----------
        aeroanalysis : RotorAeroInterface
            any object that implements RotorAeroInterface
        drivetrain : DrivetrainEfficiencyInterface
            any object that implements DrivetrainEfficiencyInterface
        machineType : dict
            use below dictionary methods to define the machine type
        npts_cp_curve : int
            number of points in cp-tsr curve.  higher npts will yield higher accuracy
            in the power curve, but will require more evaluations of aeroanalysis.

        """

        control = machineType
        self.analysis = aeroanalysis
        self.drivetrain = drivetrain
        self.control = control

        self.varSpeed = control['varSpeed']
        self.varPitch = control['varPitch']
        self.Vin = control['Vin']
        self.Vout = control['Vout']
        self.ratedPower = control['ratedPower']

        # nominal rotation speed (used for tsr sweeps -- small Reynolds number dependence)
        if self.varSpeed:
            omegaNom = 0.5*(control['maxOmega'] + control['minOmega'])
        else:
            omegaNom = control['Omega']

        # rotor radius
        R = aeroanalysis.rotorR

        # get optimal operating conditions
        self.tsr_opt, self.pitch_opt = self.__optimalOperatingConditions(R, omegaNom)

        # generate tsr/cp curve
        self.cpaero = self.__create_cpaero(R, omegaNom, self.tsr_opt,
                                           self.pitch_opt, npts_cp_curve)

        # rated speed
        self.ratedSpeed = self.__findRatedSpeed()




# ------------------
#  Machine Types
# ------------------

    @staticmethod
    def FSFP(Vin, Vout, ratedPower, Omega, pitch):
        """Define fixed-speed fixed-pitch machine.

        Parameters
        ----------
        Vin : float (m/s)
            cut-in speed
        Vout : float (m/s)
            cut-out speed
        ratedPower : float (W)
            rated power is not directly regulated only implicitly through stall.
            but is used in calculation of drivetrain efficiency
        Omega : float (RPM)
            fixed rotation speed (RPM)
        pitch : float (deg)
            fixed pitch setting --- positive same as twist

        Returns
        -------
        machineType : dict
            use in rotor initialization

        """
        varSpeed = False
        varPitch = False
        return locals()


    @staticmethod
    def FSVP(Vin, Vout, ratedPower, Omega, optimization):
        """Define fixed-speed var-pitch machine.

        Parameters
        ----------
        Vin : float (m/s)
            cut-in speed
        Vout : float (m/s)
            cut-out speed
        ratedPower : float (W)
            rated power
        Omega : float (RPM)
            fixed rotation speed (RPM)
        optimization : dict
            optimization method to find optimal pitch (see helper methods)

        Returns
        -------
        machineType : dict
            use in rotor initialization

        """
        varSpeed = False
        varPitch = True
        return locals()


    @staticmethod
    def VSFP(Vin, Vout, ratedPower, minOmega, maxOmega, pitch, optimization):
        """Define var-speed fixed-pitch machine.

        Parameters
        ----------
        Vin : float (m/s)
            cut-in speed
        Vout : float (m/s)
            cut-out speed
        ratedPower : float (W)
            rated power
        minOmega : float (RPM)
            minimum rotation speed (RPM)
        maxOmega : float (RPM)
            maximum rotation speed (RPM)
        pitch : float (deg)
            fixed pitch setting --- positive same as twist
        optimization : dict
            optimization method to find optimal rotation speed (see helper methods)

        Returns
        -------
        machineType : dict
            use in rotor initialization

        """
        varSpeed = True
        varPitch = False
        return locals()


    @staticmethod
    def VSVP(Vin, Vout, ratedPower, minOmega, maxOmega, optimization):
        """Define var-speed var-pitch machine.

        Parameters
        ----------
        Vin : float (m/s)
            cut-in speed
        Vout : float (m/s)
            cut-out speed
        ratedPower : float (W)
            rated power
        minOmega : float (RPM)
            minimum rotation speed (RPM)
        maxOmega : float (RPM)
            maximum rotation speed (RPM)
        optimization : dict
            optimization method to find optimal rotation speed and pitch (see helper methods)

        Returns
        -------
        machineType : dict
            use in rotor initialization

        """
        varSpeed = True
        varPitch = True
        return locals()






# ------------------
#  Optimization Methods
# ------------------

    @staticmethod
    def externalOpt(tsr, pitch):
        """tsr and pitch optimized externally

        Parameters
        ----------
        tsr : float
            current tip speed ratio
        pitch : float (deg)
            current pitch setting

        Returns
        -------
        opt : dict
            properly formatted dictionary for use in PowerRegulator

        """
        method = 'external'
        return locals()


    @staticmethod
    def gradientOpt(tsr=7.0, pitch=0.0):
        """find optimal tsr/pitch using a gradient-based method

        Parameters
        ----------
        tsr : float
            initial guess for optimal tip speed ratio
        pitch : float (deg)
            initial guess for optimal pitch setting

        Returns
        -------
        opt : dict
            properly formatted dictionary for use in PowerRegulator

        Notes
        -----
        For use with analysis methods that are C1 continuous.
        In the current implementation variable-speed varible-pitch machines
        uses a Quasi-Newton method with BFGS Hessian updates is used.
        Fixed-speed variable-pitch or variable-speed fixed-pitch machines
        use Brent's method.  However, the particular algorithm is subject to change.

        """
        method = 'gradient'
        return locals()


    @staticmethod
    def directSearchOpt(tsr=7.0, pitch=0.0):
        """find optimal tsr/pitch using a direct-search method

        Parameters
        ----------
        tsr : float
            initial guess for optimal tip speed ratio
        pitch : float (deg)
            initial guess for optimal pitch setting

        Returns
        -------
        opt : dict
            properly formatted dictionary for use in PowerRegulator

        Notes
        -----
        For use with analysis methods that are not differentiable.
        In the current implementation variable-speed varible-pitch machines
        uses a Nelder-Mead simplex method.
        Fixed-speed variable-pitch or variable-speed fixed-pitch machines
        use Brent's method.  However, the particular algorithm is subject to change.

        """
        method = 'search'
        return locals()


    @staticmethod
    def surrogateOpt(tsr_min=4.0, tsr_max=10.0, n_tsr=10,
                     pitch_min=-5.0, pitch_max=5.0, n_pitch=10):
        """find optimal tsr/pitch using a surrogate

        Parameters
        ----------
        tsr_min, tsr_max : float
            min and max tip speed ratios for constructing surrogate
        n_tsr : int
            number of points in surrogate grid for tsr from tsr_min to tsr_max
        pitch_min, pitch_max : float (deg)
            min and max pitch settings for constructing surrogate
        n_pitch : int
            number of points in surrogate grid for pitch from pitch_min to pitch_max

        Returns
        -------
        opt : dict
            properly formatted dictionary for use in PowerRegulator

        Notes
        -----
        For use with analysis methods that are not differentiable.
        Fits parameter space with cubic splines, then uses
        the gradient approach on the surrogate.

        May be advantageous over the direct search method for analysis
        tools which can evaluate in batch faster, or to achieve higher
        accuracy in the optimal values.

        """
        method = 'surrogate'
        tsr = 0.5*(tsr_min+tsr_max)  # starting points for optimization
        pitch = 0.5*(pitch_min+pitch_max)
        return locals()



# ----------------------------
#  Probability Distributions
# ----------------------------

    @staticmethod
    def WeibullCDF(A, k):
        """Weibull cumulative distribution function

        Arguments
        ---------
        A : float (m/s)
            scale factor
        k : float
            shape or form factor

        Returns
        -------
        f : func
            cumulative distribution function

        """
        return lambda U: 1.0 - np.exp(-(U/A)**k)


    @staticmethod
    def WeibullPDF(A, k):
        """Weibull probability distribution function

        Arguments
        ---------
        A : float (m/s)
            scale factor
        k : float
            shape or form factor

        Returns
        -------
        f : func
            probability distribution function

        """
        return lambda U: k/A*(U/A)**(k-1)*np.exp(-(U/A)**k)


    @staticmethod
    def RayleighCDF(Ubar):
        """Rayleigh cumulative distribution function.

        Arguments
        ---------
        Ubar : float (m/s)
            mean wind speed

        Returns
        -------
        f : func
            cumulative distribution function

        """
        return RotorAero.WeibullCDF(2.0*Ubar/sqrt(pi), 2)


    @staticmethod
    def RayleighPDF(Ubar):
        """Rayleigh probability distribution function.

        Arguments
        ---------
        Ubar : float (m/s)
            mean wind speed

        Returns
        -------
        f : func
            probability distribution function

        """
        return RotorAero.WeibullPDF(2.0*Ubar/sqrt(pi), 2)




# --------------------------------------
#  Private Methods Used In Constructor
# --------------------------------------


    def __optimalOperatingConditions(self, R, omegaNom):
        """return tsr_opt, pitch_opt"""

        analysis = self.analysis
        control = self.control

        if self.varSpeed and self.varPitch:  # 2D optimization

            optimization = control['optimization']
            optMethod = optimization['method']

            # setup objectives for optimization
            if optMethod == 'gradient' or optMethod == 'search':

                # define objective
                def obj(x, *args):
                    tsr = x[0]
                    pitch = x[1]
                    Omega = omegaNom
                    Uinf = Omega*RPM2RS*R/tsr
                    cp, ct, cq = analysis.evaluate([Uinf], [Omega], [pitch], coefficient=True)
                    return -cp[0]  # scalar rather than array of length 1

            elif optMethod == 'surrogate':

                # setup parameter bounds
                n = optimization['n_tsr']
                m = optimization['n_pitch']
                tsr_vec = np.linspace(optimization['tsr_min'], optimization['tsr_max'], n)
                pitch_vec = np.linspace(optimization['pitch_min'], optimization['pitch_max'], m)

                # run analysis on grid
                [T, P] = np.meshgrid(tsr_vec, pitch_vec)
                tsr = np.reshape(T, n*m)
                pitch = np.reshape(P, n*m)
                Omega = omegaNom*np.ones(n*m)
                Uinf = Omega*RPM2RS*R/tsr
                cp, ct, cq = analysis.evaluate(Uinf, Omega, pitch, coefficient=True)

                # create spline
                CP = np.reshape(cp, (m, n))
                spline = interpolate.RectBivariateSpline(tsr_vec, pitch_vec, CP.transpose())

                # define objective
                def obj(x, *args):
                    return -spline.ev(x[0], x[1])

            x0 = [optimization['tsr'], optimization['pitch']]

            if optMethod == 'external':
                xopt = x0
            else:
                if optMethod == 'search':
                    method = 'Nelder-Mead'  # TODO: sometimes this can eval at negative tsr -> then bad things happen
                else:  # gradient and surrogate
                    method = 'BFGS'
                result = optimize.minimize(obj, x0, method=method)
                xopt = result.x

            return xopt[0], xopt[1]




        elif self.varSpeed and not self.varPitch:  # 1D optimization

            optimization = control['optimization']
            optMethod = optimization['method']
            pitch_opt = control['pitch']

            if optMethod == 'gradient' or optMethod == 'search':

                # define objective
                def obj(tsr, *args):
                    pitch = pitch_opt
                    Omega = omegaNom
                    Uinf = Omega*RPM2RS*R/tsr
                    cp, ct, cq = analysis.evaluate([Uinf], [Omega], [pitch], coefficient=True)
                    return -cp[0]  # scalar rather than array of length 1

                tsr_min = optimization['tsr'] - 5  # TODO: make variable
                tsr_max = optimization['tsr'] + 5

            elif optMethod == 'surrogate':

                # setup parameter bounds
                n = optimization['n_tsr']
                tsr_vec = np.linspace(optimization['tsr_min'], optimization['tsr_max'], n)

                # run analysis
                pitch = pitch_opt*np.ones(n)
                Omega = omegaNom*np.ones(n)
                Uinf = Omega*RPM2RS*R/tsr_vec
                cp, ct, cq = analysis.evaluate(Uinf, Omega, pitch, coefficient=True)

                # create spline
                spline = interpolate.interp1d(tsr_vec, cp, kind='cubic')

                # define objective
                def obj(tsr, *args):
                    return -spline(tsr)

                tsr_min = optimization['tsr_min']
                tsr_max = optimization['tsr_max']


            if optMethod == 'external':
                tsr_opt = optimization['tsr']
            else:
                tsr_opt = optimize.brent(obj, brack=(tsr_min, optimization['tsr'], tsr_max))

            return tsr_opt, pitch_opt


        elif not self.varSpeed and self.varPitch:  # 1D optimization  (at omegaNom, TODO: may compute optimal pitch for every speed)

            optimization = control['optimization']
            optMethod = optimization['method']

            if optMethod == 'gradient' or optMethod == 'search':

                # define objective
                def obj(pitch, *args):
                    Omega = omegaNom
                    Uinf = 0.5*(self.Vin + self.Vout)  # TODO: may want to change this later
                    cp, ct, cq = analysis.evaluate([Uinf], [Omega], [pitch], coefficient=True)
                    return -cp[0]  # scalar rather than array of length 1

                pitch_min = optimization['pitch'] - 10  # TODO: may want to change this as well
                pitch_max = optimization['pitch'] + 10

            elif optMethod == 'surrogate':

                # setup parameter bounds
                n = optimization['n_tsr']
                pitch_vec = np.linspace(optimization['pitch_min'], optimization['pitch_max'], n)

                # run analysis
                Omega = omegaNom*np.ones(n)
                Uinf = 0.5*(self.Vin + self.Vout)*np.ones(n)  # TODO: may want to change this later
                cp, ct, cq = analysis.evaluate(Uinf, Omega, pitch, coefficient=True)

                # create spline
                spline = interpolate.interp1d(pitch_vec, cp, kind='cubic')

                # define objective
                def obj(pitch, *args):
                    return -spline(pitch)

                pitch_min = optimization['pitch_min']
                pitch_max = optimization['pitch_max']


            if optMethod == 'external':
                pitch_opt = optimization['pitch']
            else:
                pitch_opt = optimize.brent(obj, brack=(pitch_min, optimization['pitch'], pitch_max))

            return None, pitch_opt

        else:  # fixed speed and pitch

            return None, control['pitch']



    def __create_cpaero(self, R, omegaNom, tsr_opt, pitch_opt, npts_cp_curve):
        """returns function cp(tsr) for aero-only nondimensional power curve"""

        analysis = self.analysis
        control = self.control

        if self.varSpeed:

            # at Vin
            tsr_low_Vin = control['minOmega']*RPM2RS*R/self.Vin
            tsr_high_Vin = control['maxOmega']*RPM2RS*R/self.Vin

            tsr_max = min(max(tsr_opt, tsr_low_Vin), tsr_high_Vin)

            # at Vout
            tsr_low_Vout = control['minOmega']*RPM2RS*R/self.Vout
            tsr_high_Vout = control['maxOmega']*RPM2RS*R/self.Vout

            tsr_min = max(min(tsr_opt, tsr_high_Vout), tsr_low_Vout)

        else:
            tsr_max = control['Omega']*RPM2RS*R/self.Vin
            tsr_min = control['Omega']*RPM2RS*R/self.Vout

        if abs(tsr_min - tsr_max) < 1e-3:
            npts = 1
        else:
            npts = npts_cp_curve

        # compute nominal power curve
        tsr = np.linspace(tsr_min, tsr_max, npts)
        Omega = omegaNom*np.ones(npts)
        pitch = pitch_opt*np.ones(npts)
        Uinf = Omega*RPM2RS*R/tsr
        cp, ct, cq = analysis.evaluate(Uinf, Omega, pitch, coefficient=True)


        # setup spline for tsr vs cp
        if npts == 1:
            spline = lambda tsr_local: cp
        else:
            spline = interpolate.interp1d(tsr, cp, kind='cubic')


        def aero_cp(tsr_local):
            cp_local = 0*tsr_local

            # only valid within tsr limits
            idx = np.logical_and(tsr_local >= tsr_min, tsr_local <= tsr_max)
            cp_local[idx] = spline(tsr_local[idx])

            return cp_local

        return aero_cp



    def __findRatedSpeed(self):
        """returns rated speed.  if rated power not reached,
        returns speed for max power"""

        # generate power curve without power limit
        n = 200
        V = np.linspace(self.Vin, self.Vout, n)
        P = self.powerCurve(V, unregulated=True)

        # find rated speed
        idx = np.argmax(P)
        if (P[idx] <= self.ratedPower):  # check if rated power not reached
            return V[idx]  # speed at maximum power generation for this case
        else:
            return np.interp(self.ratedPower, P[:idx], V[:idx])



# -------------------------
#  RotorAero Methods
# -------------------------



    def powerCurve(self, V, unregulated=False):
        """Computes power curve, or power at given wind speeds

        Parameters
        ----------
        V : array_like (m/s)
            hub height velocities to evaluate power at
        unregulated : bool, optional
            if True don't apply regulation (i.e., limit at rated power)

        Returns
        -------
        P : ndarray (W)
            power at each specified freestream velocity

        """

        R = self.analysis.rotorR
        rho = self.analysis.rho

        V = np.array(V)
        V[V==0] = 1e-6  # avoid divide by zero

        # get tip speed ratio
        if self.varSpeed:
            tsr = self.tsr_opt*np.ones(len(V))
            min_tsr = self.control['minOmega']*RPM2RS*R/V
            max_tsr = self.control['maxOmega']*RPM2RS*R/V
            tsr = np.maximum(tsr, min_tsr)
            tsr = np.minimum(tsr, max_tsr)

        else:
            tsr = self.control['Omega']*RPM2RS*R/V


        # compute nominal power curve
        cp = self.cpaero(tsr)
        P = cp*0.5*rho*V**3*pi*R**2

        # drivetrain efficiency losses
        eff = self.drivetrain.efficiency(P, self.ratedPower)
        P *= eff

        # add effect of control
        if not unregulated and (self.varSpeed or self.varPitch):

            # P[P > self.ratedPower] = self.ratedPower
            P[V > self.ratedSpeed] = self.ratedPower


        return P




    def AEP(self, cdf_func):  # , soiling_losses=0.0, array_losses=0.10, availability=0.95):
        """Estimates annual energy production for a given wind distribution
        input as a cumulative distribution function.

        Parameters
        ----------
        cdf_func : func
            cumulative distribution function of form: cdf = cdf_func(V).  both inputs
            and outputs are ndarrays.  Use helper methods above as examples.

        Returns
        -------
        AEP : float (kWh)
            annual energy production

        """

        n = 200
        V = np.linspace(self.Vin, self.Vout, n)
        Power = self.powerCurve(V)

        # compute AEP
        CDF = cdf_func(V)
        AEP = np.trapz(Power/1e3, CDF*365.0*24.0)  # in kWh

        # # soiling, availability, and wake losses
        # AEP *= (1-soiling_losses)*(1-array_losses)*availability

        return AEP





    def averageRotorSpeed(self, pdf_func):
        """computes a PDF-weighted average rotor speed across operating wind speeds

        Parameters
        ----------
        pdf_func : func
            probability distribution function of form: pdf = pdf_func(V).  both inputs
            and outputs are ndarrays.  Use helper methods above as examples.

        Returns
        -------
        Omega_bar : float (RPM)
            average rotor speed

        Notes
        -----
        Useful for estimating number of cycles for a fatigue analysis.  This calculation
        is an over-estimate for a variable-speed, fixed-pitch machine.  For the other
        machine types the calculation is exact.  The approximation is used, because
        of the additional computational expense of computing the appropriate rotor speed
        in region 3.

        """

        n = 20
        V = np.linspace(0.0, 2*self.Vout, n)  # attempt to capture tails
        R = self.analysis.rotorR

        PDF = pdf_func(V)

        # rotation speed in operating range
        if self.varSpeed:

            Omega = np.zeros_like(V)

            idx = V >= self.ratedSpeed
            Omega[idx] = self.ratedSpeed*self.tsr_opt/R*RS2RPM

            idx = V < self.ratedSpeed
            Omega[idx] = V[idx]*self.tsr_opt/R*RS2RPM

            Omega = np.maximum(Omega, self.control['minOmega'])
            Omega = np.minimum(Omega, self.control['maxOmega'])

        else:
            Omega = self.control['Omega']*np.ones_like(V)

        # truncate for out of operating range
        idx = np.logical_or(V <= self.Vin, V >= self.Vout)
        Omega[idx] = 0.0

        return np.trapz(Omega * PDF, V)



    def conditionsAtRated(self):
        """Relevant parameters at rated speed

        Returns
        -------
        Vrated : float (m/s)
            rated speed
        Omega : float (RPM)
            rotor rotation speed at rated
        pitch : float (deg)
            blade pitch setting at rated speed
        thrust : float (N)
            thrust at rated speed
        torque : float (N*m)
            torque at rated speed

        """

        Omega, pitch = self.__findControlSetting(self.ratedSpeed)

        P, T, Q = self.analysis.evaluate([self.ratedSpeed], [Omega], [pitch])

        T = T[0]  # make them scalars rather than arrays of length 1
        Q = Q[0]

        return (self.ratedSpeed, Omega, pitch, T, Q)



    def thrustAndTorque(self, Uinf, pitch=0.0):
        """thrust and torque as a function of wind speed

        Parameters
        ----------
        Uinf : float (m/s)
            hub height wind speed
        pitch : float, optional (deg)
            this is only used if Uinf is outside of operational range
            (i.e. larger than Vout or less than Vin).  In which case it is assumed that
            the rotor is not rotating, and is at the specified pitch.  This is useful,
            for example, when simulating a 50 year extreme event.

        Notes
        -----
        This method takes much longer to evaluate than evaluating power because
        the power curve can be determined without knowing the corresponding
        rotation speed and pitch setting.  However, computing thrust and torque
        requires finding the correct control settings, which generally requires
        running the aerodynamic simulation multiple times.

        """

        # find control setting
        if Uinf >= self.Vin and Uinf <= self.Vout:
            Omega, pitch = self.__findControlSetting(Uinf)
        else:
            Omega = 0.0

        P, T, Q = self.analysis.evaluate([Uinf], [Omega], [pitch])

        return T[0], Q[0]



    def aeroLoads(self, Uinf, azimuth, pitch=0.0):
        """Compute (azimuthally-averaged) distributed aerodynamic loads along blade
        in the airfoil-aligned coordinate system (:ref:`blade_airfoil_coord`).

        Parameters
        ----------
        Uinf : float (m/s)
            freestream velocity
        pitch : float, optional (deg)
            this is only used if Uinf is outside of operational range
            (i.e. larger than Vout or less than Vin).  In which case it is assumed that
            the rotor is not rotating, and is at the specified pitch.  This is useful,
            for example, when simulating a 50 year extreme event

        Returns
        -------
        r : ndarray (m)
            radial stations where force is specified (should go all the way from hub to tip)
        Px : ndarray (N/m)
            force per unit length in airfoil x-direction (normal to y in airfoil plane, lower to upper surface)
        Py : ndarray (N/m)
            force per unit length in airfoil y-direction (along chord line, positive toward trailing edge)
        Pz : ndarray (N/m)
            force per unit length in airfoil z-direction (direction of increasing radius)
        pitch : float (deg)
            corresponding pitch setting for given speed

        """

        if not (isinstance(Uinf, float) or isinstance(Uinf, int)):  # allow case where velocity along radius is specified
            Uhub = Uinf[0]
        else:
            Uhub = Uinf

        # find control setting
        if Uhub >= self.Vin and Uhub <= self.Vout:
            Omega, pitch = self.__findControlSetting(Uhub)
        else:
            Omega = 0.0

        r, Px, Py, Pz, twist, precone = self.analysis.distributedAeroLoads(Uinf, Omega, pitch, azimuth)

        # rotate to airfoil coordinate system
        theta = twist + pitch
        P = DirectionVector(Px, Py, Pz).bladeToAirfoil(theta)

        return r, P, Omega, pitch, azimuth, self.analysis.tilt, precone



    # def totalLoads(self, rotorstruc, Uinf, tilt, azimuth, pitch, g=9.81, separate=False):


    #     # find control setting
    #     if Uinf >= self.Vin and Uinf <= self.Vout:
    #         Omega, pitch = self.__findControlSetting(Uinf)
    #     else:
    #         Omega = 0.0

    #     r_a, Px_a, Py_a, Pz_a, pitch = self.distributedAeroLoads(Uinf, azimuth, pitch)

    #     r_w, Px_w, Py_w, Pz_w = rotorstruc.weightLoads(tilt, azimuth, pitch, g)

    #     r_c, Px_c, Py_c, Pz_c = rotorstruc.centrifugalLoads(Omega, pitch)


    #     # interpolate aerodynamic loads onto structural grid
    #     Px_a = _akima.interpolate(r_a, Px_a, r_w)
    #     Py_a = _akima.interpolate(r_a, Py_a, r_w)
    #     Pz_a = _akima.interpolate(r_a, Pz_a, r_w)

    #     # combine
    #     aero = DirectionVector(Px_a, Py_a, Pz_a)
    #     weight = DirectionVector(Px_w, Py_w, Pz_w)
    #     cent = DirectionVector(Px_c, Py_c, Pz_c)

    #     total = aero + weight + cent

    #     if separate:
    #         return r_w, aero, weight, cent
    #     else:
    #         return r_w, total







    def hubForcesAndMoments(self, Uinf, pitch=0.0):
        """Compute aerodynamic forces and moments at the rotor hub in the
        hub-aligned coordinate system (:ref:`yaw_hub_coord`).

        Parameters
        ----------
        Uinf : float (m/s)
            freestream velocity
        pitch : float, optional (deg)
            this is only used if Uinf is outside of operational range
            (i.e. larger than Vout or less than Vin).  In which case it is assumed that
            the rotor is not rotating, and is at the specified pitch.  This is useful,
            for example, when simulating a 50 year extreme event

        Returns
        -------
        Fx, Fy, Fz, Mx, My, Mz : float (N, N*m)
            Forces and moments in the x, y, and z directions using the
            hub-aligned coordinate system.


        """

        # TODO: should check if I've already run this velocity

        # find control setting
        if Uinf >= self.Vin and Uinf <= self.Vout:
            Omega, pitch = self.__findControlSetting(Uinf)
        else:
            Omega = 0.0


        return self.analysis.hubLoads(Uinf, Omega, pitch)




    def __findControlSetting(self, Uinf):
        """private method
        finds rotation speed and pitch to achieve the correct power output
        at a given wind speed.
        Uinf must be between Vin and Vout

        """

        R = self.analysis.rotorR


        # region 2/2.5
        if Uinf <= self.ratedSpeed:

            if self.varSpeed:
                Omega = Uinf*self.tsr_opt/R*RS2RPM
                Omega = min(Omega, self.control['maxOmega'])
                Omega = max(Omega, self.control['minOmega'])
            else:
                Omega = self.control['Omega']

            if self.varPitch:
                pitch = self.pitch_opt
            else:
                pitch = self.control['pitch']


        # region 3
        else:

            # fixed speed, fixed pitch
            if not self.varSpeed and not self.varPitch:
                Omega = self.control['Omega']
                pitch = self.control['pitch']

            # variable speed, fixed pitch
            elif self.varSpeed and not self.varPitch:

                pitch = self.control['pitch']

                # choose slowing down branch
                tsr_branch = np.linspace(2, self.tsr_opt, 50)
                cp_branch = self.cpaero(tsr_branch)

                # add drivetrain losses
                P_branch = cp_branch * (0.5*self.analysis.rho*Uinf**3*pi*R**2)
                P_branch *= self.drivetrain.efficiency(P_branch, self.ratedPower)

                tsr_reg = float(np.interp(self.ratedPower, P_branch, tsr_branch))
                Omega = Uinf*tsr_reg/R*RS2RPM


            # fixed speed, variable pitch
            elif not self.varSpeed and self.varPitch:
                Omega = self.control['Omega']
                pitch = self.__getPitchToRegulatePower(Uinf, Omega, self.ratedPower)

            # variable speed, variable pitch
            else:
                Omega = min(self.control['maxOmega'], self.ratedSpeed*self.tsr_opt/R*RS2RPM)
                pitch = self.__getPitchToRegulatePower(Uinf, Omega, self.ratedPower)

        return Omega, pitch



    def __getPitchToRegulatePower(self, Uinf, Omega, P):
        """private method
        finds pitch speed to achieve rated power
        at given wind and rotation speed.

        """

        n = 40
        pitchV = np.linspace(-5, 30, n)

        P_sweep, T, Q = self.analysis.evaluate(Uinf*np.ones(n), Omega*np.ones(n), pitchV)

        # choose pitch to feather branch (and reverse order)
        idx = P_sweep.argmax()
        P_branch = P_sweep[:idx:-1]
        pitch_branch = pitchV[:idx:-1]

        # add drivetrain losses
        P_branch *= self.drivetrain.efficiency(P_branch, self.ratedPower)

        return float(np.interp(P, P_branch, pitch_branch))







if __name__ == '__main__':


    # import nose
    # config = nose.config.Config(verbosity=1)
    # nose.main(defaultTest="tests/test_rotoraero.py", config=config)

    from airfoilprep import Airfoil
    from ccblade import CCBlade
    from wtperf import WTPerf
    from CSMdt import NRELCSMDrivetrain


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
    basepath = os.path.join('5MW_files', '5MW_AFFiles')

    # load all airfoils
    airfoil_types = [0]*8
    airfoil_types[0] = Airfoil.initFromAerodynFile(basepath + os.path.sep + 'Cylinder1.dat')
    airfoil_types[1] = Airfoil.initFromAerodynFile(basepath + os.path.sep + 'Cylinder2.dat')
    airfoil_types[2] = Airfoil.initFromAerodynFile(basepath + os.path.sep + 'DU40_A17.dat')
    airfoil_types[3] = Airfoil.initFromAerodynFile(basepath + os.path.sep + 'DU35_A17.dat')
    airfoil_types[4] = Airfoil.initFromAerodynFile(basepath + os.path.sep + 'DU30_A17.dat')
    airfoil_types[5] = Airfoil.initFromAerodynFile(basepath + os.path.sep + 'DU25_A17.dat')
    airfoil_types[6] = Airfoil.initFromAerodynFile(basepath + os.path.sep + 'DU21_A17.dat')
    airfoil_types[7] = Airfoil.initFromAerodynFile(basepath + os.path.sep + 'NACA64_A17.dat')

    # place at appropriate radial stations, and convert format
    af_idx = [0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7]

    bem_airfoil = [0]*len(r)
    for i in range(len(r)):
        bem_airfoil[i] = airfoil_types[af_idx[i]]


    # create CCBlade object
    aeroanalysis = CCBlade(r, chord, theta, bem_airfoil, Rhub, Rtip, B, rho, mu, precone=20, tilt=20, yaw=20)
    # aeroanalysis = WTPerf(r, chord, theta, bem_airfoil, Rhub, Rtip, B, rho, mu, precone=20, tilt=20, yaw=20)


    # set conditions
    Uinf = 10.0
    tsr = 7.55
    pitch = 0.0
    Omega = Uinf*tsr/Rtip * 30.0/pi  # convert to RPM
    azimuth = 0

    # evaluate distributed loads
    rloads, Px, Py, Pz, theta, precone = aeroanalysis.distributedAeroLoads(Uinf, Omega, pitch, azimuth)

    # plot
    import matplotlib.pyplot as plt
    rstar = (rloads - rloads[0]) / (rloads[-1] - rloads[0])
    plt.plot(rstar, -Py/1e3, 'k', label='edgewise')
    plt.plot(rstar, Px/1e3, 'r', label='flapwise')
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
    # plt.savefig('/Users/sning/Dropbox/NREL/SysEng/CCBlade/docs-dev/images/cp.pdf')
    # plt.show()


    # del aeroanalysis
    # aeroanalysis = WTPerf(r, chord, theta, bem_airfoil, Rhub, Rtip, B, rho, mu, precone=0, tilt=0, yaw=0)

    # CP, CT, CQ = aeroanalysis.evaluate(Uinf, Omega, pitch, coefficient=True)

    # plt.plot(tsr, CP, 'r')

    # plt.show()
    # exit()

    # drivetrain
    drivetrain = NRELCSMDrivetrain('geared')

    Vin = 3.0
    Vout = 25.0
    ratedPower = 5e6
    minOmega = 0.0
    maxOmega = 10.0
    optimization = RotorAero.externalOpt(7.0, 0.0)

    machineType = RotorAero.VSVP(Vin, Vout, ratedPower, minOmega, maxOmega, optimization)
    # machineType = fixedSpeedFixedPitch(Vin, Vout, ratedPower, 12.4, 0.0)
    # machineType = varSpeedFixedPitch(Vin, Vout, ratedPower, minOmega, maxOmega, 0.0, optimization)

    rotor = RotorAero(aeroanalysis, drivetrain, machineType)

    V = np.linspace(Vin, Vout, 200)
    P = rotor.powerCurve(V)

    plt.figure()
    plt.plot(V, P)

    Uinf = 10.0
    Fx, Fy, Fz, Mx, My, Mz = rotor.hubForcesAndMoments(Uinf)
    print Fx, Fy, Fz, Mx, My, Mz


    print rotor.AEP(RotorAero.RayleighCDF(10.0))
    print rotor.averageRotorSpeed(RotorAero.RayleighPDF(10.0))
    print rotor.conditionsAtRated()
    V = np.linspace(Vin, Vout, 40)
    T = np.zeros(40)
    Q = np.zeros(40)
    for i in range(len(V)):
        T[i], Q[i] = rotor.thrustAndTorque(V[i])

    plt.figure()
    plt.plot(V, T)

    plt.figure()
    plt.plot(V, Q)

    P = rotor.powerCurve(V)

    plt.figure()
    plt.plot(V, P/Q*RS2RPM)

    # print rotor.distributedAeroLoads(10.0)



    plt.show()
