#!/usr/bin/env python
# encoding: utf-8
"""
UtilizationSupplement.py

Copyright (c) NREL. All rights reserved.
"""

import numpy as np
import openmdao.api as om
from wisdem.commonse.utilities import CubicSplineSegment, nodal2sectional

# -------------------------------------------------------------------------------
# Name:        UtilizationSupplement.py
# Purpose:     It contains functions to calculate utilizations for cylindric members,
#              for instance tower sections and monopile
#
# Author:      ANing/RRD/GBarter
#
# Created:     07/14/2015 - It is based on towerSupplement.py by ANing, 2012.
# Copyright:   (c) rdamiani 2015
# Licence:     <Apache 2015>
# -------------------------------------------------------------------------------


class GeometricConstraints(om.ExplicitComponent):
    """
    Compute the minimum diameter-to-thickness ratio and taper constraints.

    Parameters
    ----------
    d : numpy array[nPoints], [m]
        Sectional tower diameters
    t : numpy array[nPoints-1], [m]
        Sectional tower wall thicknesses
    min_d_to_t : float
        Minimum diameter-to-thickness ratio, dictated by ability to roll steel
    max_taper : float
        Maximum taper ratio of tower sections

    Returns
    -------
    constr_d_to_t : numpy array[n_points-1]
        Minimum diameter-to-thickness constraint, must be negative to be feasible
    constr_taper : numpy array[n_points-1]
        Taper ratio constraint, must be positve to be feasible
    slope : numpy array[n_points-2]
        Slope constraint, must be less than 1.0 to be feasible

    """

    def initialize(self):
        self.options.declare("nPoints")
        self.options.declare("diamFlag", default=True)

    def setup(self):
        nPoints = self.options["nPoints"]

        self.add_input("d", np.zeros(nPoints), units="m")
        self.add_input("t", np.zeros(nPoints - 1), units="m")

        self.add_output("constr_d_to_t", np.zeros(nPoints - 1))
        self.add_output("constr_taper", np.zeros(nPoints - 1))
        self.add_output("slope", np.zeros(nPoints - 1))
        if nPoints > 2:
            self.add_output("thickness_slope", np.zeros(nPoints - 2))

        # Derivatives
        self.declare_partials("*", "*", method="fd", form="central", step=1e-6)

    def compute(self, inputs, outputs):
        # Unpack inputs
        d = inputs["d"]
        t = inputs["t"]
        diamFlag = self.options["diamFlag"]

        # Check if the input was radii instead of diameters and convert if necessary
        if not diamFlag:
            d *= 2.0

        dave, _ = nodal2sectional(d)
        d_ratio = d[1:] / d[:-1]
        t_ratio = t[1:] / t[:-1]

        outputs["constr_d_to_t"] = dave / t
        outputs["constr_taper"] = np.minimum(d_ratio, 1.0 / d_ratio)
        outputs["slope"] = d_ratio
        if self.options["nPoints"] > 2:
            outputs["thickness_slope"] = t_ratio

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
    dvec = np.array(d) * 1e3
    tvec = np.array(t) * 1e3
    r = 0.5 * dvec

    nvec = len(d)
    damage = np.zeros(nvec)

    # initialize weld factor (added cubic spline around corner)
    if weld_factor:
        x1 = 24.0
        x2 = 26.0
        spline = CubicSplineSegment(x1, x2, 1.0, (25.0 / x2) ** 0.25, 0.0, 25.0 ** 0.25 * -0.25 * x2 ** -1.25)
        weld = spline.eval(tvec)
        weld[tvec >= x2] = (25.0 / tvec[tvec >= x2]) ** 0.25
        weld[tvec <= x1] = 1.0
    else:
        weld = 1.0

    # stress
    I = np.pi * r ** 3 * tvec
    sigma = M_DEL * r / I * stress_factor * 1e3  # convert to N/mm^2

    # maximum allowed stress
    Smax = DC * weld / eta

    # number of cycles to failure
    Nf = (Smax / sigma) ** m

    # number of cycles for this load
    N1 = 2e6  # TODO: where does this come from?
    N = N_DEL / N1

    # damage
    damage = N / Nf

    return damage


def vonMisesStressUtilization(axial_stress, hoop_stress, shear_stress, gamma, sigma_y):
    """combine stress for von Mises"""

    # von mises stress
    a = ((axial_stress + hoop_stress) / 2.0) ** 2
    b = ((axial_stress - hoop_stress) / 2.0) ** 2
    c = shear_stress ** 2
    von_mises = np.sqrt(a + 3.0 * (b + c))

    # stress margin
    stress_utilization = gamma * von_mises / sigma_y

    return stress_utilization  # This must be <1 to pass


def hoopStress(d, t, q_dyn):
    r = d / 2.0 - t / 2.0  # radius of cylinder middle surface
    return -q_dyn * r / t
