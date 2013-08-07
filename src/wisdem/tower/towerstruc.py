#!/usr/bin/env python
# encoding: utf-8
"""
tower.py

Created by Andrew Ning on 2012-01-20.
Copyright (c) 2012 NREL. All rights reserved.
"""

from math import pi, cos, atan2, sqrt
import numpy as np

from wisdem.common import _pBEAM


class MonopileStruc(object):
    """This class represents a wind turbine tower/monopile with cylindrical
    shell sections.

    Wind and water loads can be included and so may be
    appropriate for both onshore and offshore applications.
    Outputs are computed using finite element analysis.

    """

    def __init__(self, z, d, t, n, top, soil, E=210e9, G=80.8e9, rho=8500.0, sigma_y=450.0e6, g=9.81):
        """Constructor for cylindrical shell sections

        Parameters
        ----------
        z : array_like
            vertical positions defining the tower.  Must start from base of tower and go up. (m)
        d : array_like
            diameters at corresponding z locations (m)
        t : array_like
            shell thickness at corresponding z locations (m)
        n : array_like of ints
            number of finite elements to use between z sections.  len(n) must equal len(z) - 1
        wind : WindModel
            wind object implementing interface in wind.py
        wave : WaveModel
            wave object implementing interface in wave.py
        top : dictionary
            defines mass properties and forces from a top mass - typically a rotor-nacelle-assembly (RNA)
            pass None if there is no top mass.
            'm' : float
                mass
            'cm' : array_like
                (x,y,z) defining location of RNA's center of mass relative to tower top
            'I' : array_like
                (Ixx, Iyy, Izz, Ixy, Ixz, Iyz) moments and products of inertia about tower top
            'F' : array_like
                (Fx, Fy, Fz) applied forces from RNA at tower top
            'M' : array_like
                (Mx, My, Mz) applied moments from RNA about tower top
        soil : SoilModel
            soil object implementing interface in soil.py
        material : dictionary, optional
            defines material properties, dictionary keys below.  Assumed isotropic.  Default properties are for
            steel, with an increased density to account for unmodeled components.
            'E' : float
                modulus of elasticity (Pa)
            'G' : float
                shear modulus (Pa)
            'rho' : float
                density (kg/m**3)
        g : float, optional
            acceleration of gravity (m/s**2)

        """

        # compute nodal locations
        self.z = np.array([z[0]])
        for i in range(0, len(n)):
            znode = np.linspace(z[i], z[i+1], n[i]+1)
            self.z = np.r_[self.z, znode[1:]]

        self.nodes = len(self.z)

        # interpolate
        self.d = np.interp(self.z, z, d)
        self.t = np.interp(self.z, z, t)

        # save relevant material properties
        self.E = E
        self.sigma_y = sigma_y

        # d = 0.5*(self.d[:-1] + self.d[1:])
        # t = 0.5*(self.t[:-1] + self.t[1:])

        # A = pi*d*t
        # r = (d - t/2)/(d+t/2)
        # a = 0.54414
        # b = 2.97294
        # c = -1.51899
        # As = A/(a + b*r + c*r**2)

        # I = pi/8*d**3*t
        # J = 2*I


        # # for i in range(len(A)):
        # #     print A[i], As[i], As[i], J[i], I[i], I[i], E, G, 0.0, rho

        # # Ax    Asy     Asz     Jxx     Iyy     Izz       E      G  roll density

        # print top['m']
        # print top['I']

        # exit()

        # compute distributed loads
        # Px_wind, Py_wind, Pz_wind, q_wind = toweraero.distributedWindLoads(self.z, Uhub, zhub)
        # Px_wave, Py_wave, Pz_wave, q_wave = toweraero.distributedWaveLoads(self.z)
        Px_weight, Py_weight, Pz_weight = self.__distributedWeightLoads(rho, g)

        # Px = Px_wind + Px_wave + Px_weight
        # Py = Py_wind + Py_wave + Py_weight
        # Pz = Pz_wind + Pz_wave + Pz_weight

        # dynamic pressure
        # self.q_dyn = q_wind + q_wave

        # evaluate soil stiffness
        z_soil = z[0] + soil.depth
        d_soil = np.interp(z_soil, z, d)
        t_soil = np.interp(z_soil, z, t)
        k = soil.equivalentSpringStiffnessAtBase(d_soil/2.0, t_soil)

        # translate into pBEAM struct
        # loads = _pBEAM.Loads(self.nodes, Px, Py, Pz)
        loads = _pBEAM.Loads(self.nodes)  # no loads
        mat = _pBEAM.Material(E, G, rho)
        # if top is None:
        #     tip = _pBEAM.TipData()
        # else:
        tip = _pBEAM.TipData(top['m'], np.array(top['cm']), np.array(top['I']), np.zeros(3), np.zeros(3))
        baseData = _pBEAM.BaseData(k, soil.infinity)


        # create tower object
        self.tower = _pBEAM.Beam(self.nodes, self.z, self.d, self.t,
                                 loads, mat, tip, baseData)


        # define method to change loading on tower
        def changeLoads(zp, Px, Py, Pz, Ftop, Mtop):

            # interpolate then rotate distributed loads
            Px = np.interp(self.z, zp, Px)
            Py = np.interp(self.z, zp, Py)
            Pz = np.interp(self.z, zp, Pz)

            Px += Px_weight
            Py += Py_weight
            Pz += Pz_weight

            loads = _pBEAM.Loads(self.nodes, Px, Py, Pz)
            tip = _pBEAM.TipData(top['m'], np.array(top['cm']), np.array(top['I']), Ftop, Mtop)

            Ftop[2] -= top['m']*g

            # create tower object
            self.tower = _pBEAM.Beam(self.nodes, self.z, self.d, self.t,
                                     loads, mat, tip, baseData)


        self.changeLoads = changeLoads



    # def changeRNA(self, tip):
    #     """docstring"""

    #     self.tip = _pBEAM.TipData(tip['m'], tip['cm'], tip['I'], tip['F'], tip['M'])

    #     self.tower = _pBEAM.Beam(self.nodes, self.z, self.d, self.t, self.loads,
    #                              self.mat, self.tip, self.baseData)



    # def changeWindSpeed(self, Uref):
    #     """docstring"""

    #     # TODO: check this scaling
    #     self.wind.scaleToWindSpeedAtHeight(Uref, self.heightAboveGround)

    #     Pwind = self.__getDistributedWindLoads(self.wind)
    #     Pwater = self.__getDistributedWaveLoads(self.wave)
    #     Pweight = self.__distributedWeightLoads(self.material['rho'], self.g)
    #     Px = Pwind[0] + Pwater[0] + Pweight[0]
    #     Py = Pwind[1] + Pwater[1] + Pweight[1]
    #     Pz = Pwind[2] + Pwater[2] + Pweight[2]
    #     self.loads = _pBEAM.Loads(self.nodes, Px, Py, Pz)

    #     self.tower = _pBEAM.Beam(self.nodes, self.z, self.d, self.t,
    #         self.loads, self.mat, self.tip, self.baseData)


    # # @property
    # def heightAboveGround(self):
    #     """or water"""
    #     return self.z[-1] - self.wind.z0





    def __distributedWeightLoads(self, rho, g):
        """
        Private method which computes distributed self-weight
        load at each node.

        """

        Pz = -rho*g*(pi*self.d*self.t)

        return (0*Pz, 0*Pz, Pz)




    def getNodes(self):
        """Returns the total number of nodes after discretization."""
        return self.nodes



    def mass(self):
        """Returns the estimated mass of the structure."""
        return self.tower.mass()



    def naturalFrequencies(self, n):
        """
        Computes the first n natural frequencies of the tower.

        Arguments:
        n - number of natural frequencies to return. (currently there is no performance benefit for decreasing n)

        Returns:
        Array of size n unless n is greater than the number of total degrees
         of freedom of the structure.
        Array is sorted from lowest to highest frequency.
        All frequencies are in Hz.

        """
        return self.tower.naturalFrequencies(n)



    def displacement(self, zp, Px, Py, Pz, Ftop, Mtop):
        """
        Computes the displacement of the structure due to the applied loading.

        Returns:
        A tuple of arrays (dx, dy, dz, dtheta_x, dtheta_y, dtheta_z)
        Each array contains the displacement of the structure for the given
        degree of freedom at each node.

        """
        self.changeLoads(zp, Px, Py, Pz, Ftop, Mtop)
        return self.tower.displacement()



    def criticalBucklingLoads(self, zp, Px, Py, Pz, Ftop, Mtop):
        """
        Estimates the critical buckling loads of the tower due to axial loads.

        Returns:
        A tuple (Pcr_x, Pcr_y) with the critical axial buckling load for
        buckling in the x and y direction.
        Critical loads are in addition to any specified applied loads.

        """
        self.changeLoads(zp, Px, Py, Pz, Ftop, Mtop)
        return self.tower.criticalBucklingLoads()



    # def axialStress(self, x, y, z):
    #     """
    #     Computes the axial stress at the given locations in the structure.

    #     Arguments:
    #     x, y, z - location in structure at which to evaluate axial stress

    #     Returns:
    #     axial stress - array of same size as x containing stress at each point

    #     """
    #     return self.material['E']*self.tower.axialStrain(len(x), x, y, z)



    def axialStress(self, zp, Px, Py, Pz, Ftop, Mtop):
        """
        Computes the axial stress at 4 points along the permiter of the
        circular shell at each node of the tower.

        Returns:
        array containing axial stress evaluated at 4 * nodes locations

        """
        self.changeLoads(zp, Px, Py, Pz, Ftop, Mtop)


        # evalute stress at 4 points at each section
        length = 4 * self.nodes
        xvec = np.zeros(length)
        yvec = np.zeros(length)
        zvec = np.zeros(length)

        for i in range(self.nodes):
            xvec[i*4 + 0] = self.d[i] / 2.0
            yvec[i*4 + 0] = 0.0
            zvec[i*4 + 0] = self.z[i]

            xvec[i*4 + 1] = 0.0
            yvec[i*4 + 1] = self.d[i] / 2.0
            zvec[i*4 + 1] = self.z[i]

            xvec[i*4 + 2] = -self.d[i] / 2.0
            yvec[i*4 + 2] = 0.0
            zvec[i*4 + 2] = self.z[i]

            xvec[i*4 + 3] = 0.0
            yvec[i*4 + 3] = -self.d[i] / 2.0
            zvec[i*4 + 3] = self.z[i]

        return self.E*self.tower.axialStrain(length, xvec, yvec, zvec)




    def hoopStress(self, zq, q_dyn, L_reinforced):
        """
        Estimates the hoop stress at 4 points along the permiter of the
        circular shell at each node of the tower.

        Uses the dynamic pressure from the wind/wave loads and an
        emperical method from Eurocode to estimate an equivalent
        axisymmetric pressure distribution.

        Returns:
        array containing hoop stress evaluated at 4 * nodes locations
        """

        r = self.d / 2.0
        t = self.t
        q_dyn = np.interp(self.z, zq, q_dyn)

        # Eurocode method
        C_theta = 1.5
        omega = L_reinforced/np.sqrt(r*t)
        k_w = 0.46*(1.0 + 0.1*np.sqrt(C_theta/omega*r/t))
        k_w = np.maximum(0.65, np.minimum(1.0, k_w))
        Peq = k_w*q_dyn
        sigma_theta = -Peq*r/t

        # same stress around tower
        sigma_theta_tower = np.zeros(4*self.nodes)
        for i in range(self.nodes):
            sigma_theta_tower[i*4 + 0] = sigma_theta[i]
            sigma_theta_tower[i*4 + 1] = sigma_theta[i]
            sigma_theta_tower[i*4 + 2] = sigma_theta[i]
            sigma_theta_tower[i*4 + 3] = sigma_theta[i]

        return sigma_theta_tower


    def shearStress(self, zp, Px, Py, Pz, Ftop, Mtop):

        self.changeLoads(zp, Px, Py, Pz, Ftop, Mtop)

        Vx, Vy, Fz, Mx, My, Tz = self.tower.shearAndBending()

        A = pi * self.d * self.t
        shear_stress_max_x = 2 * Vx / A
        shear_stress_max_y = 2 * Vy / A

        shear_stress = np.zeros(len(A)*4)
        shear_stress[::4] = shear_stress_max_x
        shear_stress[2::4] = shear_stress_max_x
        shear_stress[1::4] = shear_stress_max_y
        shear_stress[3::4] = shear_stress_max_y

        return shear_stress


    def vonMisesStress(self, sigma_z, sigma_t, tau_zt):

        a = ((sigma_z + sigma_t)/2.0)**2
        b = ((sigma_z - sigma_t)/2.0)**2
        c = tau_zt**2
        sigma = np.sqrt(a + 3.0*(b+c))

        return sigma



    def shellBuckling(self, npt, sigma_z, sigma_t, tau_zt, L_reinforced, gamma_f=1.2, gamma_b=1.1):
        """
        Estimate shell buckling constraint along tower.

        Arguments:
        npt - number of locations at each node at which stress is evaluated.
        sigma_z - axial stress at npt*node locations.  must be in order
                      [(node1_pts1-npt), (node2_pts1-npt), ...]
        sigma_t - azimuthal stress given at npt*node locations
        tau_zt - shear stress (z, theta) at npt*node locations
        E - modulus of elasticity
        sigma_y - yield stress
        L_reinforced - reinforcement length - structure is re-discretized with this spacing
        gamma_f - safety factor for stresses
        gamma_b - safety factor for buckling

        Returns:
        z
        an array of shell buckling constraints evaluted at (z[0] at npt locations,
        z[0]+L_reinforced at npt locations, ...).
        Each constraint must be <= 0 to avoid failure.
        """

        # break up into chunks of length L_reinforced
        z_re = np.arange(self.z[0], self.z[-1], L_reinforced)
        if (z_re[-1] != self.z[-1]):
            z_re = np.r_[z_re, self.z[-1]]

        # initialize
        constraint = np.zeros(npt * (len(z_re) - 1))

        # evaluate each line separately
        for j in range(npt):

            # pull off stresses along line
            sigma_z_line = sigma_z[j::npt]
            sigma_t_line = sigma_t[j::npt]
            tau_zt_line = tau_zt[j::npt]

            # interpolate into sections
            d_re = np.interp(z_re, self.z, self.d)
            t_re = np.interp(z_re, self.z, self.t)
            sigma_z_re = np.interp(z_re, self.z, sigma_z_line)
            sigma_t_re = np.interp(z_re, self.z, sigma_t_line)
            tau_zt_re = np.interp(z_re, self.z, tau_zt_line)

            for i in range(len(z_re)-1):
                h = z_re[i+1] - z_re[i]
                r1 = d_re[i] / 2.0
                r2 = d_re[i+1] / 2.0
                t1 = t_re[i]
                t2 = t_re[i+1]
                sigma_z_shell = sigma_z_re[i]  # use base value - should be conservative
                sigma_t_shell = sigma_t_re[i]
                tau_zt_shell = tau_zt_re[i]

                # only compressive stresses matter.
                # also change to magnitudes and add safety factor
                sigma_z_shell = gamma_f*abs(min(sigma_z_shell, 0.0))
                sigma_t_shell = gamma_f*abs(sigma_t_shell)
                tau_zt_shell = gamma_f*abs(tau_zt_shell)

                constraint[i*4 + j] = self.__shellBucklingOneSection(h, r1, r2, t1, t2, gamma_b, sigma_z_shell, sigma_t_shell, tau_zt_shell)

        return z_re[0:-1], constraint



    def __cubicspline(self, ptL, ptR, fL, fR, gL, gR, pts):

        A = np.array([[ptL**3, ptL**2, ptL, 1],
                      [ptR**3, ptR**2, ptR, 1],
                      [3*ptL**2, 2*ptL, 1, 0],
                      [3*ptR**2, 2*ptR, 1, 0]])
        b = np.array([fL, fR, gL, gR])

        coeff = np.linalg.solve(A, b)

        value = coeff[0]*pts**3 + coeff[1]*pts**2 + coeff[2]*pts + coeff[3]

        return value


    def __cxsmooth(self, omega, rovert):

        Cxb = 6.0  # clamped-clamped
        constant = 1 + 1.83/1.7 - 2.07/1.7**2

        ptL1 = 1.7-0.25
        ptR1 = 1.7+0.25

        ptL2 = 0.5*rovert - 1.0
        ptR2 = 0.5*rovert + 1.0

        ptL3 = (0.5+Cxb)*rovert - 1.0
        ptR3 = (0.5+Cxb)*rovert + 1.0


        if omega < ptL1:
            Cx = constant - 1.83/omega + 2.07/omega**2

        elif omega >= ptL1 and omega <= ptR1:

            fL = constant - 1.83/ptL1 + 2.07/ptL1**2
            fR = 1.0
            gL = 1.83/ptL1**2 - 4.14/ptL1**3
            gR = 0.0
            Cx = self.__cubicspline(ptL1, ptR1, fL, fR, gL, gR, omega)

        elif omega > ptR1 and omega < ptL2:
            Cx = 1.0

        elif omega >= ptL2 and omega <= ptR2:

            fL = 1.0
            fR = 1 + 0.2/Cxb*(1-2.0*ptR2/rovert)
            gL = 0.0
            gR = -0.4/Cxb/rovert
            Cx = self.__cubicspline(ptL2, ptR2, fL, fR, gL, gR, omega)

        elif omega > ptR2 and omega < ptL3:
            Cx = 1 + 0.2/Cxb*(1-2.0*omega/rovert)

        elif omega >= ptL3 and omega <= ptR3:

            fL = 1 + 0.2/Cxb*(1-2.0*ptL3/rovert)
            fR = 0.6
            gL = -0.4/Cxb/rovert
            gR = 0.0
            Cx = self.__cubicspline(ptL3, ptR3, fL, fR, gL, gR, omega)

        else:
            Cx = 0.6

        return Cx


    def __sigmasmooth(self, omega, E, rovert):

        Ctheta = 1.5  # clamped-clamped

        ptL = 1.63*rovert*Ctheta - 1
        ptR = 1.63*rovert*Ctheta + 1

        if omega < 20.0*Ctheta:
            offset = (10.0/(20*Ctheta)**2 - 5/(20*Ctheta)**3)
            Cthetas = 1.5 + 10.0/omega**2 - 5/omega**3 - offset
            sigma = 0.92*E*Cthetas/omega/rovert

        elif omega >= 20.0*Ctheta and omega < ptL:

            sigma = 0.92*E*Ctheta/omega/rovert

        elif omega >= ptL and omega <= ptR:

            alpha1 = 0.92/1.63 - 2.03/1.63**4

            fL = 0.92*E*Ctheta/ptL/rovert
            fR = E*(1.0/rovert)**2*(alpha1 + 2.03*(Ctheta/ptR*rovert)**4)
            gL = -0.92*E*Ctheta/rovert/ptL**2
            gR = -E*(1.0/rovert)*2.03*4*(Ctheta/ptR*rovert)**3*Ctheta/ptR**2

            sigma = self.__cubicspline(ptL, ptR, fL, fR, gL, gR, omega)

        else:

            alpha1 = 0.92/1.63 - 2.03/1.63**4
            sigma = E*(1.0/rovert)**2*(alpha1 + 2.03*(Ctheta/omega*rovert)**4)

        return sigma


    def __tausmooth(self, omega, rovert):

        ptL1 = 9
        ptR1 = 11

        ptL2 = 8.7*rovert - 1
        ptR2 = 8.7*rovert + 1

        if omega < ptL1:
            C_tau = sqrt(1.0 + 42.0/omega**3 - 42.0/10**3)

        elif omega >= ptL1 and omega <= ptR1:
            fL = sqrt(1.0 + 42.0/ptL1**3 - 42.0/10**3)
            fR = 1.0
            gL = -63.0/ptL1**4/fL
            gR = 0.0
            C_tau = self.__cubicspline(ptL1, ptR1, fL, fR, gL, gR, omega)

        elif omega > ptR1 and omega < ptL2:
            C_tau = 1.0

        elif omega >= ptL2 and omega <= ptR2:
            fL = 1.0
            fR = 1.0/3.0*sqrt(ptR2/rovert) + 1 - sqrt(8.7)/3
            gL = 0.0
            gR = 1.0/6/sqrt(ptR2*rovert)
            C_tau = self.__cubicspline(ptL2, ptR2, fL, fR, gL, gR, omega)

        else:
            C_tau = 1.0/3.0*sqrt(omega/rovert) + 1 - sqrt(8.7)/3

        return C_tau



    def __shellBucklingOneSection(self, h, r1, r2, t1, t2, gamma_b, sigma_z, sigma_t, tau_zt):
        """
        Estimate shell buckling for one tapered cylindrical shell section.

        Arguments:
        h - height of conical section
        r1 - radius at bottom
        r2 - radius at top
        t1 - shell thickness at bottom
        t2 - shell thickness at top
        E - modulus of elasticity
        sigma_y - yield stress
        gamma_b - buckling reduction safety factor
        sigma_z - axial stress component
        sigma_t - azimuthal stress component
        tau_zt - shear stress component (z, theta)

        Returns:
        buckling_constraint, which must be <= 0 to avoid failure

        """

        E = self.E
        sigma_y = self.sigma_y

        #NOTE: definition of r1, r2 switched from Eurocode document to be consistent with FEM.

        # ----- geometric parameters --------
        beta = atan2(r1-r2, h)
        L = h/cos(beta)
        t = 0.5*(t1+t2)

        # ------------- axial stress -------------
        # length parameter
        le = L
        re = 0.5*(r1+r2)/cos(beta)
        omega = le/sqrt(re*t)
        rovert = re/t

        # compute Cx
        Cx = self.__cxsmooth(omega, rovert)


        # if omega <= 1.7:
        #     Cx = 1.36 - 1.83/omega + 2.07/omega/omega
        # elif omega > 0.5*rovert:
        #     Cxb = 6.0  # clamped-clamped
        #     Cx = max(0.6, 1 + 0.2/Cxb*(1-2.0*omega/rovert))
        # else:
        #     Cx = 1.0

        # critical axial buckling stress
        sigma_z_Rcr = 0.605*E*Cx/rovert

        # compute buckling reduction factors
        lambda_z0 = 0.2
        beta_z = 0.6
        eta_z = 1.0
        Q = 25.0  # quality parameter - high
        lambda_z = sqrt(sigma_y/sigma_z_Rcr)
        delta_wk = 1.0/Q*sqrt(rovert)*t
        alpha_z = 0.62/(1 + 1.91*delta_wk/t)**1.44

        chi_z = self.__buckling_reduction_factor(alpha_z, beta_z, eta_z, lambda_z0, lambda_z)

        # design buckling stress
        sigma_z_Rk = chi_z*sigma_y
        sigma_z_Rd = sigma_z_Rk/gamma_b

        # ---------------- hoop stress ------------------

        # length parameter
        le = L
        re = 0.5*(r1+r2)/(cos(beta))
        omega = le/sqrt(re*t)
        rovert = re/t

        # Ctheta = 1.5  # clamped-clamped
        # CthetaS = 1.5 + 10.0/omega**2 - 5.0/omega**3

        # # critical hoop buckling stress
        # if (omega/Ctheta < 20.0):
        #     sigma_t_Rcr = 0.92*E*CthetaS/omega/rovert
        # elif (omega/Ctheta > 1.63*rovert):
        #     sigma_t_Rcr = E*(1.0/rovert)**2*(0.275 + 2.03*(Ctheta/omega*rovert)**4)
        # else:
        #     sigma_t_Rcr = 0.92*E*Ctheta/omega/rovert

        sigma_t_Rcr = self.__sigmasmooth(omega, E, rovert)

        # buckling reduction factor
        alpha_t = 0.65  # high fabrication quality
        lambda_t0 = 0.4
        beta_t = 0.6
        eta_t = 1.0
        lambda_t = sqrt(sigma_y/sigma_t_Rcr)

        chi_theta = self.__buckling_reduction_factor(alpha_t, beta_t, eta_t, lambda_t0, lambda_t)

        sigma_t_Rk = chi_theta*sigma_y
        sigma_t_Rd = sigma_t_Rk/gamma_b

        # ----------------- shear stress ----------------------

        # length parameter
        le = h
        rho = sqrt((r1+r2)/(2.0*r2))
        re = (1.0 + rho - 1.0/rho)*r2*cos(beta)
        omega = le/sqrt(re*t)
        rovert = re/t

        # if (omega < 10):
        #     C_tau = sqrt(1.0 + 42.0/omega**3)
        # elif (omega > 8.7*rovert):
        #     C_tau = 1.0/3.0*sqrt(omega/rovert)
        # else:
        #     C_tau = 1.0
        C_tau = self.__tausmooth(omega, rovert)

        tau_zt_Rcr = 0.75*E*C_tau*sqrt(1.0/omega)/rovert

        # reduction factor
        alpha_tau = 0.65  # high fabrifaction quality
        beta_tau = 0.6
        lambda_tau0 = 0.4
        eta_tau = 1.0
        lambda_tau = sqrt(sigma_y/sqrt(3)/tau_zt_Rcr)

        chi_tau = self.__buckling_reduction_factor(alpha_tau, beta_tau, eta_tau, lambda_tau0, lambda_tau)

        tau_zt_Rk = chi_tau*sigma_y/sqrt(3)
        tau_zt_Rd = tau_zt_Rk/gamma_b

        # buckling interaction parameters

        k_z = 1.25 + 0.75*chi_z
        k_theta = 1.25 + 0.75*chi_theta
        k_tau = 1.75 + 0.25*chi_tau
        k_i = (chi_z*chi_theta)**2

        # buckling constraint

        buckling_constraint = \
            (sigma_z/sigma_z_Rd)**k_z + \
            (sigma_t/sigma_t_Rd)**k_theta - \
            k_i*(sigma_z*sigma_t/sigma_z_Rd/sigma_t_Rd) + \
            (tau_zt/tau_zt_Rd)**k_tau - 1

        return buckling_constraint



    def __buckling_reduction_factor(self, alpha, beta, eta, lambda_0, lambda_bar):
        """
        Computes a buckling reduction factor used in Eurocode shell buckling formula.
        """

        lambda_p = sqrt(alpha/(1.0-beta))

        ptL = 0.9*lambda_0
        ptR = 1.1*lambda_0

        if (lambda_bar < ptL):
            chi = 1.0

        elif lambda_bar >= ptL and lambda_bar <= ptR:  # cubic spline section

            fracR = (ptR-lambda_0)/(lambda_p-lambda_0)
            fL = 1.0
            fR = 1-beta*fracR**eta
            gL = 0.0
            gR = -beta*eta*fracR**(eta-1)/(lambda_p-lambda_0)

            chi = self.__cubicspline(ptL, ptR, fL, fR, gL, gR, lambda_bar)

        elif lambda_bar > ptR and lambda_bar < lambda_p:
            chi = 1.0 - beta*((lambda_bar-lambda_0)/(lambda_p-lambda_0))**eta

        else:
            chi = alpha/lambda_bar**2



        # if (lambda_bar <= lambda_0):
        #     chi = 1.0
        # elif (lambda_bar >= lambda_p):
        #     chi = alpha/lambda_bar**2
        # else:
        #     chi = 1.0 - beta*((lambda_bar-lambda_0)/(lambda_p-lambda_0))**eta

        return chi



if __name__ == '__main__':

    # import nose
    # config = nose.config.Config(verbosity=1)
    # nose.main(defaultTest="test/test_tower.py", config=config)

    from wind import WindWithPowerProfile
    from wave import LinearWaves
    from soil import SoilModelCylindricalFoundation
    from toweraero import MonopileAero

    # ------------ tower geometry ---------
    z = [0, 30.0, 73.8, 117.6]         # heights starting from bottom of tower to top (m)
    d = [6.0, 6.0, 4.935, 3.87]      # corresponding diameters (m)
    t = [0.027, 0.027, 0.023, 0.019]   # corresponding shell thicknesses (m)
    t = np.array(t)*1.3
    n = [5, 5, 5]                  # number of finite elements per section

    L_reinforced = 30.0        # reinforcement length of cylindrical sections (m)

    z_floor = 0.0          # reference position of sea floor (m)
    z_surface = 20.0        # position of water surface (m)
    # -----------------------------------------------

    # --------------- waves ---------------------

    hs = 7.5   # 7.5 is 10 year extreme   # significant wave height (m)
    T = 19.6                # wave period (s)
    #

    wave = LinearWaves(hs, T)
    # -----------------------------------------------


    # --------------- wind --------------------------
    alpha = 1.0/7          # power law exponent

    wind = WindWithPowerProfile(alpha, beta=20)
    # -----------------------------------------------

    # tower aerodynamics
    toweraero = MonopileAero(z, d, z_surface, wind, wave)

    # # ------------ tower material properties ---------

    # E = 210e9               # elastic modulus (Pa)
    # G = 80.8e9              # shear modulus (Pa)
    # rho = 8500.0            # material density (kg/m^3)
    # sigma_y = 450.0e6      # yield stress (Pa)

    # material = dict(E=E, G=G, rho=rho, sigma_y=sigma_y)
    # # -----------------------------------------------

    # ------------ soil properties ---------

    G = 140e6               # shear modulus of soil (Pa)
    nu = 0.4                # Poissons ratio of soil
    depth = 10              # depth of soil (m)
    rigid = [0, 1, 2, 3, 4, 5]          # indices for degrees of freedom which should be considered infinitely rigid
                            # (order is x, theta_x, y, theta_y, z, theta_z)

    soil = SoilModelCylindricalFoundation(G, nu, depth, rigid)
    # -----------------------------------------------

    # RNA
    # g = 9.81
    tip = dict(m=300000.0,
               cm=np.array([0.0, 0.0, 0.0]),
               I=np.array([2960437.0, 3253223.0, 3264220.0, 0.0, -18400.0, 0.0]))
               # F=np.array([750000.0, 0.0, -300000.0*g]),
               # M=np.array([0.0, 0.0, 0.0])
               # )



    # assemble structures
    Uref = 50.0            # reference wind speed (m/s) - Class I IEC (0.2 X is average for AEP, Ve50 = 1.4*Vref*(z/zhub)^.11 for the blade, max thrust at rated power for tower)
    zref = 120.0            # height of reference wind speed (m)
    Uc = 1.2               # current speed (m/s)

    yaw = 0.0
    rp, Px, Py, Pz, q_dyn = toweraero.distributedLoads(Uref, Uc, yaw)

    Ftop = np.array([750000.0, 0.0, 0.0])
    Mtop = np.array([0.0, 0.0, 0.0])
    loads = (rp, Px, Py, Pz, Ftop, Mtop)

    tower = MonopileStruc(z, d, t, n, tip, soil)

    mass = tower.mass()
    freq = tower.naturalFrequencies(5)
    disp = tower.displacement(*loads)
    buckling = tower.criticalBucklingLoads(*loads)
    axial_stress = tower.axialStress(*loads)
    hoop_stress = tower.hoopStress(rp, q_dyn, L_reinforced)
    shear_stress = tower.shearStress(*loads)

    # gamma_f = 1.2
    # gamma_b = 1.1

    von_mises = tower.vonMisesStress(axial_stress, hoop_stress, shear_stress)

    shell_buckling = tower.shellBuckling(4, axial_stress, hoop_stress, shear_stress, L_reinforced)

    print 'mass = ', mass
    print

    print 'natural freq = ', freq
    print

    print 'disp = '
    for i in range(0, 6):
        print disp[i]
    print

    print 'buckling = ', buckling
    print

    print 'axial stress = ', axial_stress
    print

    print 'hoop stress = ', hoop_stress
    print

    print 'von mises stress = ', von_mises
    print

    print 'shell_buckling = ', shell_buckling


    import matplotlib.pyplot as plt
    plt.plot(tower.z, von_mises[::4])
    plt.plot(tower.z, von_mises[1::4])
    plt.plot(tower.z, von_mises[2::4])
    plt.plot(tower.z, von_mises[3::4])
    plt.figure()
    plt.plot(tower.z, disp[0])

    plt.show()

    # import profile
    #     test = CylindricalTowerModelTests()
    #     profile.run('test.testWindLoads()', 'stats')
    #
    #     import pstats
    #     p = pstats.Stats('stats')
    #     p.strip_dirs().sort_stats('cumulative').print_stats(10)

    # import timeit
    # test = CylindricalTowerModelTests()
    # t = timeit.Timer(test.testWindLoads)
    # print t.timeit(100)
