#!/usr/bin/env python
# encoding: utf-8
"""
UtilizationSupplement.py

Copyright (c) NREL. All rights reserved.
"""

from math import atan2
import numpy as np
from wisdem.commonse.constants import eps
from wisdem.commonse.utilities import CubicSplineSegment, cubic_spline_eval, smooth_max, smooth_min, nodal2sectional, sectional2nodal
from openmdao.api import ExplicitComponent
from scipy.optimize import brentq, minimize_scalar

#-------------------------------------------------------------------------------
# Name:        UtilizationSupplement.py
# Purpose:     It contains functions to calculate utilizations for cylindric members,
#              for instance tower sections and monopile
#
# Author:      ANing/RRD/GBarter
#
# Created:     07/14/2015 - It is based on towerSupplement.py by ANing, 2012.
# Copyright:   (c) rdamiani 2015
# Licence:     <Apache 2015>
#-------------------------------------------------------------------------------



class GeometricConstraints(ExplicitComponent):
    """docstring for OtherConstraints"""
    def initialize(self):
        self.options.declare('nPoints')
        self.options.declare('diamFlag', default=True)
        
    def setup(self):
        nPoints       = self.options['nPoints']
        
        self.add_input('d', np.zeros(nPoints), units='m')
        self.add_input('t', np.zeros(nPoints-1), units='m')
        self.add_input('min_d_to_t', 120.0)
        self.add_input('max_taper', 0.4)

        self.add_output('weldability', np.zeros(nPoints-1))
        self.add_output('manufacturability', np.zeros(nPoints-1))
        self.add_output('slope', np.zeros(nPoints-2))

        # Derivatives
        self.declare_partials('*', '*', method='fd', form='central', step=1e-6)


    def compute(self, inputs, outputs):
        diamFlag = self.options['diamFlag']

        d,_ = nodal2sectional(inputs['d'])
        t = inputs['t']

        # Check if the input was radii instead of diameters and convert if necessary
        if not diamFlag: d *= 2.0
        
        min_d_to_t = inputs['min_d_to_t']
        max_taper = inputs['max_taper']

        outputs['weldability'] = 1.0 - (d/t)/min_d_to_t
        d_ratio = d[1:]/d[:-1]
        manufacturability = np.minimum(d_ratio, 1.0/d_ratio) - max_taper
        outputs['manufacturability'] = np.r_[manufacturability, manufacturability[-1]]
        outputs['slope'] = d_ratio

    # def compute_partials(self, inputs, J):

    #     dw_dd = np.diag(-1.0/self.t/self.min_d_to_t)
    #     dw_dt = np.diag(self.d/self.t**2/self.min_d_to_t)

    #     dw = np.hstack([dw_dd, dw_dt])



    #     dm_dd = np.zeros_like(self.d)
    #     dm_dd[0] = self.d[-1]/self.d[0]**2
    #     dm_dd[-1] = -1.0/self.d[0]

    #     dm = np.hstack([dm_dd, np.zeros(len(self.t))])


def fatigue(M_DEL, N_DEL, d, t, m=4, DC=80.0, eta=1.265, stress_factor=1.0, weld_factor=True):
    """estimate fatigue damage for tower station

    Parmeters
    ---------
    M_DEL : array_like(float) (N*m)
        damage equivalent moment at tower section
    N_DEL : array_like(int)
        corresponding number of cycles in lifetime
    d : array_like(float) (m)
        tower diameter at section
    t : array_like(float) (m)
        tower shell thickness at section
    m : int
        slope of S/N curve
    DC : float (N/mm**2)
        some max stress from a standard
    eta : float
        safety factor
    stress_factor : float
        load_factor * stress_concentration_factor
    weld_factor : bool
        if True include an empirical weld factor

    Returns
    -------
    damage : float
        damage from Miner's rule for this tower section
    """


    # convert to mm
    dvec = np.array(d)*1e3
    tvec = np.array(t)*1e3

    t = sectional2nodal(t)
    nvec = len(d)
    damage = np.zeros(nvec)

    # initialize weld factor (added cubic spline around corner)
    if weld_factor:
        x1 = 24.0
        x2 = 26.0
        spline = CubicSplineSegment(x1, x2, 1.0, (25.0/x2)**0.25, 0.0, 25.0**0.25*-0.25*x2**-1.25)


    for i in range(nvec):

        d = dvec[i]
        t = tvec[i]

        # weld factor
        if not weld_factor or t <= x1:
            weld = 1.0
        elif t >= x2:
            weld = (25.0/t)**0.25
        else:
            weld = spline.eval(t)


        # stress
        r = d/2.0
        I = np.pi*r**3*t
        c = r
        sigma = M_DEL[i]*c/I * stress_factor * 1e3  # convert to N/mm^2

        # maximum allowed stress
        Smax = DC * weld / eta

        # number of cycles to failure
        Nf = (Smax/sigma)**m

        # number of cycles for this load
        N1 = 2e6  # TODO: where does this come from?
        N = N_DEL[i]/N1

        # damage
        damage[i] = N/Nf

    return damage


def vonMisesStressUtilization(axial_stress, hoop_stress, shear_stress, gamma, sigma_y):
    """combine stress for von Mises"""

    # von mises stress
    a = ((axial_stress + hoop_stress)/2.0)**2
    b = ((axial_stress - hoop_stress)/2.0)**2
    c = shear_stress**2
    von_mises = np.sqrt(a + 3.0*(b+c))

    # stress margin
    stress_utilization = gamma * von_mises / sigma_y

    return stress_utilization  # This must be <1 to pass

def hoopStress(d, t, q_dyn):
    r = d/2.0-t/2.0  # radius of cylinder middle surface
    return (-q_dyn * r / t)

def hoopStressEurocode(z, d, t, L_reinforced, q_dyn):
    """default method for computing hoop stress using Eurocode method
       GB 06/21/2018: Ansys comparisons for submerged case suggests this over-compensates for stiffener
                      I'm not even sure the Eurocode is implemented correctly here.  Suggest using the standard
                      hoop stress expression above or API's handling of ring stiffeners below.
    """

    r = d/2.0-t/2.0  # radius of cylinder middle surface
    omega = L_reinforced/np.sqrt(r*t)

    C_theta = 1.5  # clamped-clamped
    k_w = 0.46*(1.0 + 0.1*np.sqrt(C_theta/omega*r/t))
    kw = smooth_max(k_w, 0.65)
    kw = smooth_min(k_w, 1.0)
    Peq = k_w*q_dyn
    return hoopStress(d, t, Peq)


def bucklingGL(d, t, Fz, Myy, tower_height, E, sigma_y, gamma_f=1.2, gamma_b=1.1, gamma_g=1.1):

    # other factors
    alpha = 0.21  # buckling imperfection factor
    beta = 1.0  # bending coefficient
    sk_factor = 2.0  # fixed-free
    tower_height = tower_height * sk_factor

    # geometry
    A = np.pi * d * t
    I = np.pi * (d/2.0)**3 * t
    Wp = I / (d/2.0)

    # applied loads
    Nd = -Fz * gamma_g
    Md = Myy * gamma_f

    # plastic resistance
    Np = A * sigma_y / gamma_b
    Mp = Wp * sigma_y / gamma_b

    # factors
    Ne = np.pi**2 * (E * I) / (1.1 * tower_height**2)
    lambda_bar = np.sqrt(Np * gamma_b / Ne)
    phi = 0.5 * (1 + alpha*(lambda_bar - 0.2) + lambda_bar**2)
    kappa = np.ones_like(d)
    idx = lambda_bar > 0.2
    kappa[idx] = 1.0 / (phi[idx] + np.sqrt(phi[idx]**2 - lambda_bar[idx]**2))
    delta_n = 0.25*kappa*lambda_bar**2
    delta_n = np.minimum(delta_n, 0.1)

    GL_utilization = Nd/(kappa*Np) + beta*Md/Mp + delta_n  #this is utilization must be <1

    return GL_utilization





def shellBucklingEurocode(d, t, sigma_z, sigma_t, tau_zt, L_reinforced, E, sigma_y, gamma_f=1.2, gamma_b=1.1):
    """
    Estimate shell buckling utilization along tower.

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
    EU_utilization: - array of shell buckling utilizations evaluted at (z[0] at npt locations, \n
                      z[0]+L_reinforced at npt locations, ...). \n
                      Each utilization must be < 1 to avoid failure.
    """

    n = len(d)
    EU_utilization = np.zeros(n)
    sigma_z_sh = np.zeros(n)
    sigma_t_sh = np.zeros(n)
    tau_zt_sh = np.zeros(n)

    for i in range(n):
        h = L_reinforced[i]

        r1 = d[i]/2.0 - t[i]/2.0
        r2 = d[i]/2.0 - t[i]/2.0

        sigma_z_shell = sigma_z[i]
        sigma_t_shell = sigma_t[i]
        tau_zt_shell = tau_zt[i]

        # TODO: the following is non-smooth, although in general its probably OK
        # change to magnitudes and add safety factor
        sigma_z_shell = gamma_f*abs(sigma_z_shell)
        sigma_t_shell = gamma_f*abs(sigma_t_shell)
        tau_zt_shell = gamma_f*abs(tau_zt_shell)

        EU_utilization[i] = _shellBucklingOneSection(h, r1, r2, t[i], gamma_b, sigma_z_shell, sigma_t_shell, tau_zt_shell, E[i], sigma_y[i])

        #make them into vectors
        sigma_z_sh[i]=sigma_z_shell
        sigma_t_sh[i]=sigma_t_shell
        tau_zt_sh[i]=tau_zt_shell

    return EU_utilization  # this is utilization must be <1




def _cxsmooth(omega, rovert):

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
        Cx = cubic_spline_eval(ptL1, ptR1, fL, fR, gL, gR, omega)

    elif omega > ptR1 and omega < ptL2:
        Cx = 1.0

    elif omega >= ptL2 and omega <= ptR2:

        fL = 1.0
        fR = 1 + 0.2/Cxb*(1-2.0*ptR2/rovert)
        gL = 0.0
        gR = -0.4/Cxb/rovert
        Cx = cubic_spline_eval(ptL2, ptR2, fL, fR, gL, gR, omega)

    elif omega > ptR2 and omega < ptL3:
        Cx = 1 + 0.2/Cxb*(1-2.0*omega/rovert)

    elif omega >= ptL3 and omega <= ptR3:

        fL = 1 + 0.2/Cxb*(1-2.0*ptL3/rovert)
        fR = 0.6
        gL = -0.4/Cxb/rovert
        gR = 0.0
        Cx = cubic_spline_eval(ptL3, ptR3, fL, fR, gL, gR, omega)

    else:
        Cx = 0.6

    return Cx


def _sigmasmooth(omega, E, rovert):

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

        sigma = cubic_spline_eval(ptL, ptR, fL, fR, gL, gR, omega)

    else:

        alpha1 = 0.92/1.63 - 2.03/1.63**4
        sigma = E*(1.0/rovert)**2*(alpha1 + 2.03*(Ctheta/omega*rovert)**4)

    return sigma


def _tausmooth(omega, rovert):

    ptL1 = 9
    ptR1 = 11

    ptL2 = 8.7*rovert - 1
    ptR2 = 8.7*rovert + 1

    if omega < ptL1:
        C_tau = np.sqrt(1.0 + 42.0/omega**3 - 42.0/10**3)

    elif omega >= ptL1 and omega <= ptR1:
        fL = np.sqrt(1.0 + 42.0/ptL1**3 - 42.0/10**3)
        fR = 1.0
        gL = -63.0/ptL1**4/fL
        gR = 0.0
        C_tau = cubic_spline_eval(ptL1, ptR1, fL, fR, gL, gR, omega)

    elif omega > ptR1 and omega < ptL2:
        C_tau = 1.0

    elif omega >= ptL2 and omega <= ptR2:
        fL = 1.0
        fR = 1.0/3.0*np.sqrt(ptR2/rovert) + 1 - np.sqrt(8.7)/3
        gL = 0.0
        gR = 1.0/6/np.sqrt(ptR2*rovert)
        C_tau = cubic_spline_eval(ptL2, ptR2, fL, fR, gL, gR, omega)

    else:
        C_tau = 1.0/3.0*np.sqrt(omega/rovert) + 1 - np.sqrt(8.7)/3

    return C_tau



def _shellBucklingOneSection(h, r1, r2, t, gamma_b, sigma_z, sigma_t, tau_zt, E, sigma_y):
    """
    Estimate shell buckling for one tapered cylindrical shell section.

    Arguments:
    h - height of conical section
    r1 - radius at bottom
    r2 - radius at top
    t - shell thickness
    E - modulus of elasticity
    sigma_y - yield stress
    gamma_b - buckling reduction safety factor
    sigma_z - axial stress component
    sigma_t - azimuthal stress component
    tau_zt - shear stress component (z, theta)

    Returns:
    EU_utilization, shell buckling utilization which must be < 1 to avoid failure

    """

    #NOTE: definition of r1, r2 switched from Eurocode document to be consistent with FEM.

    # ----- geometric parameters --------
    beta = atan2(r1-r2, h)
    L = h/np.cos(beta)

    # ------------- axial stress -------------
    # length parameter
    le = L
    re = 0.5*(r1+r2)/np.cos(beta)
    omega = le/np.sqrt(re*t)
    rovert = re/t

    # compute Cx
    Cx = _cxsmooth(omega, rovert)


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
    lambda_z = np.sqrt(sigma_y/sigma_z_Rcr)
    delta_wk = 1.0/Q*np.sqrt(rovert)*t
    alpha_z = 0.62/(1 + 1.91*(delta_wk/t)**1.44)

    chi_z = _buckling_reduction_factor(alpha_z, beta_z, eta_z, lambda_z0, lambda_z)

    # design buckling stress
    sigma_z_Rk = chi_z*sigma_y
    sigma_z_Rd = sigma_z_Rk/gamma_b

    # ---------------- hoop stress ------------------

    # length parameter
    le = L
    re = 0.5*(r1+r2)/(np.cos(beta))
    omega = le/np.sqrt(re*t)
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

    sigma_t_Rcr = np.maximum(eps, _sigmasmooth(omega, E, rovert))

    # buckling reduction factor
    alpha_t = 0.65  # high fabrication quality
    lambda_t0 = 0.4
    beta_t = 0.6
    eta_t = 1.0
    lambda_t = np.sqrt(sigma_y/sigma_t_Rcr)

    chi_theta = _buckling_reduction_factor(alpha_t, beta_t, eta_t, lambda_t0, lambda_t)

    sigma_t_Rk = chi_theta*sigma_y
    sigma_t_Rd = sigma_t_Rk/gamma_b

    # ----------------- shear stress ----------------------

    # length parameter
    le = h
    rho = np.sqrt((r1+r2)/(2.0*r2))
    re = (1.0 + rho - 1.0/rho)*r2*np.cos(beta)
    omega = le/np.sqrt(re*t)
    rovert = re/t

    # if (omega < 10):
    #     C_tau = np.sqrt(1.0 + 42.0/omega**3)
    # elif (omega > 8.7*rovert):
    #     C_tau = 1.0/3.0*np.sqrt(omega/rovert)
    # else:
    #     C_tau = 1.0
    C_tau = _tausmooth(omega, rovert)

    tau_zt_Rcr = 0.75*E*C_tau*np.sqrt(1.0/omega)/rovert

    # reduction factor
    alpha_tau = 0.65  # high fabrifaction quality
    beta_tau = 0.6
    lambda_tau0 = 0.4
    eta_tau = 1.0
    lambda_tau = np.sqrt(sigma_y/np.sqrt(3)/tau_zt_Rcr)

    chi_tau = _buckling_reduction_factor(alpha_tau, beta_tau, eta_tau, lambda_tau0, lambda_tau)

    tau_zt_Rk = chi_tau*sigma_y/np.sqrt(3)
    tau_zt_Rd = tau_zt_Rk/gamma_b

    # buckling interaction parameters

    k_z = 1.25 + 0.75*chi_z
    k_theta = 1.25 + 0.75*chi_theta
    k_tau = 1.75 + 0.25*chi_tau
    k_i = (chi_z*chi_theta)**2

    # shell buckling utilization

    utilization = \
        (sigma_z/sigma_z_Rd)**k_z + \
        (sigma_t/sigma_t_Rd)**k_theta - \
        k_i*(sigma_z*sigma_t/sigma_z_Rd/sigma_t_Rd) + \
        (tau_zt/tau_zt_Rd)**k_tau

    return utilization #this is utilization must be <1



def _buckling_reduction_factor(alpha, beta, eta, lambda_0, lambda_bar):
    """
    Computes a buckling reduction factor used in Eurocode shell buckling formula.
    """

    lambda_p = np.sqrt(alpha/(1.0-beta))

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

        chi = cubic_spline_eval(ptL, ptR, fL, fR, gL, gR, lambda_bar)

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




def _TBeamProperties(h_web, t_web, w_flange, t_flange):
    """Computes T-cross section area, CG, and moments of inertia
    See: http://www.amesweb.info/SectionalPropertiesTabs/SectionalPropertiesTbeam.aspx

    INPUTS:
    ----------
    h_web    : float (scalar/vector),  web (T-base) height
    t_web    : float (scalar/vector),  web (T-base) thickness
    w_flange : float (scalar/vector),  flange (T-top) width/height
    t_flange : float (scalar/vector),  flange (T-top) thickness

    OUTPUTS:
    -------
    area : float (scalar/vector),  T-cross sectional area
    y_cg : float (scalar/vector),  Position of CG along y-axis (extending from base up through the T)
    Ixx  : float (scalar/vector),  Moment of intertia around axis parallel to flange, through y_cg
    Iyy  : float (scalar/vector),  Moment of intertia around y-axis
    """
    # Area of T cross section is sum of the two rectangles
    area_web    = h_web * t_web
    area_flange = w_flange * t_flange
    area        = area_web + area_flange
    # Y-position of the center of mass (Yna) measured from the base
    y_cg = ( (h_web + 0.5*t_flange)*area_flange + 0.5*h_web*area_web ) / area
    # Moments of inertia: y-axis runs through base (spinning top),
    # x-axis runs parallel to flange through cg
    Iyy =  (area_web*t_web**2 + area_flange*w_flange**2    ) / 12.0
    Ixx = ((area_web*h_web**2 + area_flange*t_flange**2) / 12.0 +
           area_web*(y_cg - 0.5*h_web)**2 +
           area_flange*(h_web + 0.5*t_flange - y_cg)**2 )
    return area, y_cg, Ixx, Iyy


def _IBeamProperties(h_web, t_web, w_flange, t_flange, w_base, t_base):
    """Computes uneven I-cross section area, CG
    See: http://www.amesweb.info/SectionalPropertiesTabs/SectionalPropertiesTbeam.aspx

    INPUTS:
    ----------
    h_web    : float (scalar/vector),  web (I-stem) height
    t_web    : float (scalar/vector),  web (I-stem) thickness
    w_flange : float (scalar/vector),  flange (I-top) width/height
    t_flange : float (scalar/vector),  flange (I-top) thickness
    w_base   : float (scalar/vector),  base (I-bottom) width/height
    t_base   : float (scalar/vector),  base (I-bottom) thickness

    OUTPUTS:
    -------
    area : float (scalar/vector),  T-cross sectional area
    y_cg : float (scalar/vector),  Position of CG along y-axis (extending from base up through the T)
    """
    # Area of T cross section is sum of the two rectangles
    area_web    = h_web * t_web
    area_flange = w_flange * t_flange
    area_base   = w_base * t_base
    area        = area_web + area_flange + area_base
    # Y-position of the center of mass (Yna) measured from the base
    y_cg = ( (t_base + h_web + 0.5*t_flange)*area_flange + (t_base + 0.5*h_web)*area_web + 0.5*t_base*area_base ) / area
    return area, y_cg


def _compute_applied_axial(R_od, t_wall, m_stack, section_mass):
    """Compute axial stress for spar from z-axis loading

    INPUTS:
    ----------
    params       : dictionary of input parameters
    section_mass : float (scalar/vector),  mass of each spar section as axial loading increases with spar depth

    OUTPUTS:
    -------
    stress   : float (scalar/vector),  axial stress
    """

    R       = R_od - 0.5*t_wall
    # Add in weight of sections above it
    axial_load    = m_stack + np.r_[0.0, np.cumsum(section_mass[:-1])]
    # Divide by shell cross sectional area to get stress
    return (gravity * axial_load / (2.0 * np.pi * R * t_wall))


def _compute_applied_hoop(pressure, R_od, t_wall):
    """Compute hoop stress WITHOUT accounting for stiffener rings

    INPUTS:
    ----------
    pressure : float (scalar/vector),  radial (hydrostatic) pressure
    R_od     : float (scalar/vector),  radius to outer wall of shell
    t_wall   : float (scalar/vector),  shell wall thickness

    OUTPUTS:
    -------
    stress   : float (scalar/vector),  hoop stress with no stiffeners
    """
    return (pressure * R_od / t_wall)

    
def _compute_stiffener_factors(pressure, axial_stress, R_od, t_wall, h_web, t_web, w_flange, t_flange, L_stiffener, E, nu):
    """Compute modifiers to stress due to presence of stiffener rings.

    INPUTS:
    ----------
    params       : dictionary of input parameters
    pressure     : float (scalar/vector),  radial (hydrostatic) pressure
    axial_stress : float (scalar/vector),  axial loading (z-axis) stress

    OUTPUTS:
    -------
    stiffener_factor_KthL : float (scalar/vector),  Stress modifier from stiffeners for local buckling from axial loads
    stiffener_factor_KthG : float (scalar/vector),  Stress modifier from stiffeners for general buckling from external pressure
    """

    # Geometry computations
    R_flange = R_od - h_web # Should have "- t_wall", but not in appendix B
    area_stiff, y_cg, Ixx, Iyy = _TBeamProperties(h_web, t_web, w_flange, t_flange)
    t_stiff  = area_stiff / h_web # effective thickness(width) of stiffener section

    # Compute hoop stress modifiers accounting for stiffener rings
    # This has to be done at midpoint between stiffeners and at stiffener location itself
    # Compute beta (just a local term used here)
    D    = E * t_wall**3 / (12.0 * (1 - nu*nu))
    beta = (0.25 * E * t_wall / R_od**2 / D)**0.25
    # Compute psi-factor (just a local term used here)
    u     = np.minimum(0.5 * beta * L_stiffener, 30.0)
    psi_k = 2.0 * (np.sin(u)*np.cosh(u) + np.cos(u)*np.sinh(u)) / (np.sinh(2*u) + np.sin(2*u)) 

    # Compute a couple of other local terms
    u   = np.minimum(beta * L_stiffener, 30.0)
    k_t = 8 * beta**3 * D * (np.cosh(u) - np.cos(u)) / (np.sinh(u) + np.sin(u))
    k_d = E * t_stiff * (R_od**2 - R_flange**2) / R_od / ((1+nu)*R_od**2 + (1-nu)*R_flange**2)

    # Pressure from axial load
    pressure_sigma = pressure - nu*axial_stress*t_wall/R_od

    # Compute the correction to hoop stress due to the presesnce of ring stiffeners
    stiffener_factor_KthL = 1 - psi_k * (pressure_sigma / pressure) * (k_d / (k_d + k_t))
    stiffener_factor_KthG = 1 -         (pressure_sigma / pressure) * (k_d / (k_d + k_t))
    return stiffener_factor_KthL, stiffener_factor_KthG


def _compute_elastic_stress_limits(R_od, t_wall, h_section, h_web, t_web, w_flange, t_flange,
                                   L_stiffener, E, nu, KthG, loading='hydrostatic'):
    """Compute modifiers to stress due to presence of stiffener rings.

    INPUTS:
    ----------
    params  : dictionary of input parameters
    KthG    : float (scalar/vector),  Stress modifier from stiffeners for general buckling from external pressure
    loading : string (hydrostatic/radial), Parameter that determines a coefficient- is only included for unit testing 
              consistency with API 2U Appdx B and should not be used in practice

    OUTPUTS:
    -------
    elastic_axial_local_FxeL    : float (scalar/vector),  Elastic stress limit for local buckling from axial loads
    elastic_extern_local_FreL   : float (scalar/vector),  Elastic stress limit for local buckling from external pressure loads
    elastic_axial_general_FxeG  : float (scalar/vector),  Elastic stress limit for general instability from axial loads
    elastic_extern_general_FreG : float (scalar/vector),  Elastic stress limit for general instability from external pressure loads
    """

    # Geometry computations
    nsections = R_od.size
    area_stiff, y_cg, Ixx, Iyy = _TBeamProperties(h_web, t_web, w_flange, t_flange)
    area_stiff_bar = area_stiff / L_stiffener / t_wall
    R  = R_od - 0.5*t_wall

    # 1. Local shell mode buckling from axial loads
    # Compute a few parameters that define the curvature of the geometry
    m_x  = L_stiffener / np.sqrt(R * t_wall)
    z_x  = m_x**2 * np.sqrt(1 - nu**2)
    z_m  = 12.0 * z_x**2 / np.pi**4
    # Imperfection factor- empirical fit that converts theory to reality
    a_xL = 9.0 * (300.0 + (2*R/t_wall))**(-0.4)
    # Calculate buckling coefficient
    C_xL = np.sqrt( 1 + 150.0 * a_xL**2 * m_x**4 / (2*R/t_wall) )
    # Calculate elastic and inelastic final limits
    elastic_axial_local_FxeL   = C_xL * np.pi**2 * E * (t_wall/L_stiffener)**2 / 12.0 / (1-nu**2)

    # 2. Local shell mode buckling from external (pressure) loads
    # Imperfection factor- empirical fit that converts theory to reality
    a_thL = np.ones(m_x.shape)
    a_thL[m_x > 5.0] = 0.8
    # Find the buckling mode- closest integer that is root of solved equation
    n   = np.zeros((nsections,))
    maxn = 50
    for k in range(nsections):
        c = L_stiffener[k] / np.pi / R[k]
        myfun = lambda x:((c*x)**2*(1 + (c*x)**2)**4/(2 + 3*(c*x)**2) - z_m[k])
        try:
            n[k] = brentq(myfun, 0, maxn)
        except:
            n[k] = maxn
    # Calculate beta (local term)
    beta  = np.round(n) * L_stiffener / np.pi / R
    # Calculate buckling coefficient
    C_thL = a_thL * ( (1+beta**2)**2/(0.5+beta**2) + 0.112*m_x**4/(1+beta**2)**2/(0.5+beta**2) )
    # Calculate elastic and inelastic final limits
    elastic_extern_local_FreL   = C_thL * np.pi**2 * E * (t_wall/L_stiffener)**2 / 12.0 / (1-nu**2)

    # 3. General instability buckling from axial loads
    # Compute imperfection factor
    a_x = 0.85 / (1 + 0.0025*2*R/t_wall)
    a_xG = a_x
    a_xG[area_stiff_bar>=0.2] = 0.72
    ind = np.logical_and(area_stiff_bar<0.06, area_stiff_bar<0.2)
    a_xG[ind] = (3.6 - 5.0*a_x[ind])*area_stiff_bar[ind]
    # Calculate elastic and inelastic final limits
    elastic_axial_general_FxeG   = 0.605 * a_xG * E * t_wall/R * np.sqrt(1 + area_stiff_bar)

    # 4. General instability buckling from external loads
    # Distance from shell centerline to stiffener cg
    z_r = -(y_cg + 0.5*t_wall)
    # Imperfection factor
    a_thG = 0.8
    # Effective shell width if the outer shell and the T-ring stiffener were to be combined to make an uneven I-beam
    L_shell_effective = 1.1*np.sqrt(2.0*R*t_wall) + t_web
    L_shell_effective[m_x <= 1.56] = L_stiffener[m_x <= 1.56]
    # Get properties of this effective uneven I-beam
    _, yna_eff = _IBeamProperties(h_web, t_web, w_flange, t_flange, L_shell_effective, t_wall)
    Rc = R_od - yna_eff
    # Compute effective shell moment of inertia based on Ir - I of stiffener
    Ier = Ixx + area_stiff*z_r**2*L_shell_effective*t_wall/(area_stiff+L_shell_effective*t_wall) + L_shell_effective*t_wall**3/12.0
    # Lambda- a local constant
    lambda_G = np.pi * R / h_section
    # Coefficient factor listed as 'k' in peG equation
    coeff = 0.5 if loading in ['hydro','h','hydrostatic','static'] else 0.0    
    # Compute pressure leading to elastic failure
    n = np.zeros(R_od.shape)
    pressure_failure_peG = np.zeros(R_od.shape)
    for k in range(nsections):
        peG = lambda x: ( E*lambda_G[k]**4*t_wall[k]/R[k]/(x**2+0.0*lambda_G[k]**2-1)/(x**2 + lambda_G[k]**2)**2 +
                          E*Ier[k]*(x**2-1)/L_stiffener[k]/Rc[k]**2/R_od[k] )
        minout = minimize_scalar(peG, bounds=(2.0, 15.0), method='bounded')
        n[k] = minout.x
        pressure_failure_peG[k] = peG(n[k])
    # Calculate elastic and inelastic final limits
    elastic_extern_general_FreG   = a_thG * pressure_failure_peG * R_od * KthG / t_wall

    return elastic_axial_local_FxeL, elastic_extern_local_FreL, elastic_axial_general_FxeG, elastic_extern_general_FreG




def _plasticityRF(Felastic, yield_stress):
    """Computes plasticity reduction factor for elastic stresses near the yield stress to obtain an inelastic stress
    This is defined in Section 5 of API Bulletin 2U

    INPUTS:
    ----------
    Felastic     : float (scalar/vector),  elastic stress
    yield_stress : float (scalar/vector),  yield stress

    OUTPUTS:
    -------
    Finelastic   : float (scalar/vector),  modified (in)elastic stress
    """
    Fratio = np.array(yield_stress / Felastic)
    eta    = Fratio * (1.0 + 3.75*Fratio**2)**(-0.25)
    Finelastic = np.array(Felastic)
    Finelastic[Felastic > 0.5*yield_stress] *= eta[Felastic > 0.5*yield_stress]
    return Finelastic


def _safety_factor(Ficj, yield_stress):
    """Use the inelastic limits and yield stress to compute required safety factors
    This is defined in Section 9 of API Bulletin 2U

    INPUTS:
    ----------
    Ficj          : float (scalar/vector),  inelastic stress
    yield_stress  : float (scalar/vector),  yield stress

    OUTPUTS:
    -------
    safety_factor : float (scalar/vector),  margin applied to inelastic stress limits
    """
    # Partial safety factor, psi
    psi = np.array(1.4 - 0.4 * Ficj / yield_stress)
    psi[Ficj <= 0.5*yield_stress] = 1.2
    psi[Ficj >= yield_stress] = 1.0
    # Final safety factor is 25% higher to give a margin
    return 1.25*psi


def shellBuckling_withStiffeners(P, sigma_ax, R_od, t_wall, h_section, h_web, t_web, w_flange, t_flange,
                                 L_stiffener, E, nu, sigma_y, loading='hydro'):

    # APPLIED STRESSES (Section 11 of API Bulletin 2U)
    stiffener_factor_KthL, stiffener_factor_KthG = _compute_stiffener_factors(P, sigma_ax, R_od, t_wall, h_web, t_web,
                                                                              w_flange, t_flange, L_stiffener, E, nu)
    hoop_stress_nostiff = _compute_applied_hoop(P, R_od, t_wall)
    hoop_stress_between = hoop_stress_nostiff * stiffener_factor_KthL
    hoop_stress_atring  = hoop_stress_nostiff * stiffener_factor_KthG
    
    # BUCKLING FAILURE STRESSES (Section 4 of API Bulletin 2U)
    (elastic_axial_local_FxeL, elastic_extern_local_FreL,
     elastic_axial_general_FxeG, elastic_extern_general_FreG) = _compute_elastic_stress_limits(R_od, t_wall, h_section, h_web,
                                                                                               t_web, w_flange, t_flange,
                                                                                               L_stiffener, E, nu, stiffener_factor_KthG,
                                                                                               loading=loading)
    inelastic_axial_local_FxcL    = _plasticityRF(elastic_axial_local_FxeL   , sigma_y)
    inelastic_axial_general_FxcG  = _plasticityRF(elastic_axial_general_FxeG , sigma_y)
    inelastic_extern_local_FrcL   = _plasticityRF(elastic_extern_local_FreL  , sigma_y)
    inelastic_extern_general_FrcG = _plasticityRF(elastic_extern_general_FreG, sigma_y)

    # COMBINE AXIAL AND HOOP (EXTERNAL PRESSURE) LOADS TO FIND DESIGN LIMITS
    # (Section 6 of API Bulletin 2U)
    load_per_length_Nph = sigma_ax            * t_wall
    load_per_length_Nth = hoop_stress_nostiff * t_wall
    load_ratio_k        = load_per_length_Nph / load_per_length_Nth
    def solveFthFph(Fxci, Frci, Kth):
        Fphci = np.zeros(Fxci.shape)
        Fthci = np.zeros(Fxci.shape)
        Kph   = 1.0
        c1    = (Fxci + Frci) / sigma_y - 1.0
        c2    = load_ratio_k * Kph / Kth
        for k in range(Fxci.size):
            try:
                Fthci[k] = brentq(lambda x: (c2[k]*x/Fxci[k])**2 - c1[k]*(c2[k]*x/Fxci[k])*(x/Frci[k]) + (x/Frci[k])**2 - 1.0, 0, Fxci[k]+Frci[k], maxiter=20)
            except:
                Fthci[k] = Fxci[k] + Frci[k]
            Fphci[k] = c2[k] * Fthci[k]
        return Fphci, Fthci

    inelastic_local_FphcL, inelastic_local_FthcL = solveFthFph(inelastic_axial_local_FxcL, inelastic_extern_local_FrcL, stiffener_factor_KthL)
    inelastic_general_FphcG, inelastic_general_FthcG = solveFthFph(inelastic_axial_general_FxcG, inelastic_extern_general_FrcG, stiffener_factor_KthG)

    # Use the inelastic limits and yield stress to compute required safety factors
    # and adjust the limits accordingly
    axial_limit_local_FaL     = inelastic_local_FphcL   / _safety_factor(inelastic_local_FphcL  , sigma_y)
    extern_limit_local_FthL   = inelastic_local_FthcL   / _safety_factor(inelastic_local_FthcL  , sigma_y)
    axial_limit_general_FaG   = inelastic_general_FphcG / _safety_factor(inelastic_general_FphcG, sigma_y)
    extern_limit_general_FthG = inelastic_general_FthcG / _safety_factor(inelastic_general_FthcG, sigma_y)

    # Compare limits to applied stresses and use this ratio as a design constraint
    # (Section 9 "Allowable Stresses" of API Bulletin 2U)
    # These values must be <= 1.0
    axial_local_api      = sigma_ax / axial_limit_local_FaL
    axial_general_api    = sigma_ax / axial_limit_general_FaG
    external_local_api   = hoop_stress_between / extern_limit_local_FthL
    external_general_api = hoop_stress_between / extern_limit_general_FthG

    # Compute unification ratios without safety factors in case we want to apply our own later
    axial_local_raw      = sigma_ax / inelastic_local_FphcL
    axial_general_raw    = sigma_ax / inelastic_general_FphcG
    external_local_raw   = hoop_stress_between / inelastic_local_FthcL
    external_general_raw = hoop_stress_between / inelastic_general_FthcG
    
    return (axial_local_api, axial_general_api, external_local_api, external_general_api,
            axial_local_raw, axial_general_raw, external_local_raw, external_general_raw)
