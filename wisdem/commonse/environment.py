#!/usr/bin/env python
# encoding: utf-8
"""
environment.py

Created by Andrew Ning on 2012-01-20.
Copyright (c) NREL. All rights reserved.
"""

from __future__ import print_function

import numpy as np
import openmdao.api as om
from scipy.optimize import brentq

from wisdem.commonse.constants import gravity

# TODO CHECK

# -----------------
#  Base Components
# -----------------


class WindBase(om.ExplicitComponent):
    """
    Base component for wind speed/direction

    Parameters
    ----------
    Uref : float, [m/s]
        reference wind speed (usually at hub height)
    zref : float, [m]
        corresponding reference height
    z : numpy array[npts], [m]
        'heights where wind speed should be computed
    z0 : float, [m]
        bottom of wind profile (height of ground/sea)

    Returns
    -------
    U : numpy array[npts], [m/s]
        magnitude of wind speed at each z location

    """

    def initialize(self):
        self.options.declare("nPoints")

    def setup(self):
        npts = self.options["nPoints"]

        self.add_input("Uref", 0.0, units="m/s")
        self.add_input("zref", 0.0, units="m")
        self.add_input("z", np.zeros(npts), units="m")
        self.add_input("z0", 0.0, units="m")

        self.add_output("U", np.zeros(npts), units="m/s")


class WaveBase(om.ExplicitComponent):
    """
    Base component for wave speed/direction

    Parameters
    ----------
    rho_water : float, [kg/m**3]
        water density
    z : numpy array[npts], [m]
        heights where wave speed should be computed
    z_surface : float, [m]
        vertical location of water surface
    z_floor : float, [m]
        vertical location of sea floor

    Returns
    -------
    U : numpy array[npts], [m/s]
        horizontal wave velocity at each z location
    W : numpy array[npts], [m/s]
        vertical wave velocity at each z location
    V : numpy array[npts], [m/s]
        total wave velocity at each z location
    A : numpy array[npts], [m/s**2]
        horizontal wave acceleration at each z location
    p : numpy array[npts], [N/m**2]
        pressure oscillation at each z location

    """

    def initialize(self):
        self.options.declare("nPoints")

    def setup(self):
        npts = self.options["nPoints"]

        self.add_input("rho_water", 0.0, units="kg/m**3")
        self.add_input("z", np.zeros(npts), units="m")
        self.add_input("z_surface", 0.0, units="m")
        self.add_input("z_floor", 0.0, units="m")

        self.add_output("U", np.zeros(npts), units="m/s")
        self.add_output("W", np.zeros(npts), units="m/s")
        self.add_output("V", np.zeros(npts), units="m/s")
        self.add_output("A", np.zeros(npts), units="m/s**2")
        self.add_output("p", np.zeros(npts), units="N/m**2")

    def compute(self, inputs, outputs):
        """default to no waves"""
        n = len(inputs["z"])
        outputs["U"] = np.zeros(n)
        outputs["W"] = np.zeros(n)
        outputs["V"] = np.zeros(n)
        outputs["A"] = np.zeros(n)
        outputs["p"] = np.zeros(n)
        # outputs['U0'] = 0.
        # outputs['A0'] = 0.


# -----------------------
#  Subclassed Components
# -----------------------


class PowerWind(WindBase):
    """
    Power-law profile wind.
    Any nodes must not cross z0, and if a node is at z0 it must stay at that point.
    Otherwise gradients crossing the boundary will be wrong.

    Parameters
    ----------
    Uref : float, [m/s]
        reference wind speed (usually at hub height)
    zref : float, [m]
        corresponding reference height
    z : numpy array[npts], [m]
        'heights where wind speed should be computed
    z0 : float, [m]
        bottom of wind profile (height of ground/sea)
    shearExp : float
        shear exponent

    Returns
    -------
    U : numpy array[npts], [m/s]
        magnitude of wind speed at each z location

    """

    def setup(self):
        super(PowerWind, self).setup()

        # parameters
        self.add_input("shearExp", 0.0)

        self.declare_partials("U", ["Uref", "z", "zref"])

    def compute(self, inputs, outputs):

        # rename
        z = inputs["z"]
        if isinstance(z, float) or isinstance(z, np.float_):
            z = np.array([z])
        zref = float(inputs["zref"])
        z0 = float(inputs["z0"])

        # velocity
        idx = z > z0
        outputs["U"] = np.zeros(self.options["nPoints"])
        outputs["U"][idx] = inputs["Uref"] * ((z[idx] - z0) / (zref - z0)) ** inputs["shearExp"]

        # # add small cubic spline to allow continuity in gradient
        # k = 0.01  # fraction of profile with cubic spline
        # zsmall = z0 + k*(zref - z0)

        # self.spline = CubicSpline(x1=z0, x2=zsmall, f1=0.0, f2=Uref*k**shearExp,
        #     g1=0.0, g2=Uref*k**shearExp*shearExp/(zsmall - z0))

        # idx = np.logical_and(z > z0, z < zsmall)
        # self.U[idx] = self.spline.eval(z[idx])

        # self.zsmall = zsmall
        # self.k = k

    def compute_partials(self, inputs, J):

        # rename
        z = inputs["z"]
        if isinstance(z, float) or isinstance(z, np.float_):
            z = np.array([z])
        zref = inputs["zref"]
        z0 = inputs["z0"]
        shearExp = inputs["shearExp"]
        idx = z > z0
        npts = self.options["nPoints"]

        U = np.zeros(npts)
        U[idx] = inputs["Uref"] * ((z[idx] - z0) / (zref - z0)) ** inputs["shearExp"]

        # gradients
        dU_dUref = np.zeros(npts)
        dU_dz = np.zeros(npts)
        dU_dzref = np.zeros(npts)

        dU_dUref[idx] = U[idx] / inputs["Uref"]
        dU_dz[idx] = U[idx] * shearExp / (z[idx] - z0)
        dU_dzref[idx] = -U[idx] * shearExp / (zref - z0)

        J["U", "Uref"] = dU_dUref
        J["U", "z"] = np.diag(dU_dz)
        J["U", "zref"] = dU_dzref
        # TODO still missing several partials? This is what was in the original code though...

        # # cubic spline region
        # idx = np.logical_and(z > z0, z < zsmall)

        # # d w.r.t z
        # dU_dz[idx] = self.spline.eval_deriv(z[idx])

        # # d w.r.t. Uref
        # df2_dUref = k**shearExp
        # dg2_dUref = k**shearExp*shearExp/(zsmall - z0)
        # dU_dUref[idx] = self.spline.eval_deriv_inputs(z[idx], 0.0, 0.0, 0.0, df2_dUref, 0.0, dg2_dUref)

        # # d w.r.t. zref
        # dx2_dzref = k
        # dg2_dzref = -Uref*k**shearExp*shearExp/k/(zref - z0)**2
        # dU_dzref[idx] = self.spline.eval_deriv_params(z[idx], 0.0, dx2_dzref, 0.0, 0.0, 0.0, dg2_dzref)


class LogWind(WindBase):
    """
    Logarithmic-profile wind

    Parameters
    ----------
    Uref : float, [m/s]
        reference wind speed (usually at hub height)
    zref : float, [m]
        corresponding reference height
    z : numpy array[npts], [m]
        'heights where wind speed should be computed
    z0 : float, [m]
        bottom of wind profile (height of ground/sea)
    z_roughness : float, [mm]
        surface roughness length

    Returns
    -------
    U : numpy array[npts], [m/s]
        magnitude of wind speed at each z location

    """

    def setup(self):
        super(LogWind, self).setup()

        # parameters
        self.add_input("z_roughness", 1e-3, units="mm")

        self.declare_partials("U", ["Uref", "z", "zref"])

    def compute(self, inputs, outputs):

        # rename
        z = inputs["z"]
        if isinstance(z, float) or isinstance(z, np.float_):
            z = np.array([z])
        zref = float(inputs["zref"])
        z0 = float(inputs["z0"])
        z_roughness = float(inputs["z_roughness"]) / 1e3  # convert to m

        # find velocity
        idx = np.where(z - z0 > z_roughness)[0]
        outputs["U"] = np.zeros_like(z)
        outputs["U"][idx] = inputs["Uref"] * np.log((z[idx] - z0) / z_roughness) / np.log((zref - z0) / z_roughness)

    def compute_partials(self, inputs, J):

        # rename
        z = inputs["z"]
        if isinstance(z, float) or isinstance(z, np.float_):
            z = np.array([z])
        zref = float(inputs["zref"])
        z0 = float(inputs["z0"])
        z_roughness = float(inputs["z_roughness"]) / 1e3
        Uref = float(inputs["Uref"])
        npts = self.options["nPoints"]

        dU_dUref = np.zeros(npts)
        dU_dz_diag = np.zeros(npts)
        dU_dzref = np.zeros(npts)

        idx = np.where(z - z0 > z_roughness)[0]
        lt = np.log((z[idx] - z0) / z_roughness)
        lb = np.log((zref - z0) / z_roughness)
        dU_dUref[idx] = lt / lb
        dU_dz_diag[idx] = Uref / lb / (z[idx] - z0)
        dU_dzref[idx] = -Uref * lt / np.log((zref - z0) / z_roughness) ** 2 / (zref - z0)

        J["U", "Uref"] = dU_dUref
        J["U", "z"] = np.diag(dU_dz_diag)
        J["U", "zref"] = dU_dzref


class LinearWaves(WaveBase):
    """
    Linear (Airy) wave theory

    Parameters
    ----------
    rho_water : float, [kg/m**3]
        water density
    z : numpy array[npts], [m]
        heights where wave speed should be computed
    z_surface : float, [m]
        vertical location of water surface
    z_floor : float, [m]
        vertical location of sea floor
    Hsig_wave : float, [m]
        Maximum wave height (crest-to-trough)
    Tsig_wave : float, [s]
        period of maximum wave height

    Returns
    -------
    U : numpy array[npts], [m/s]
        horizontal wave velocity at each z location
    W : numpy array[npts], [m/s]
        vertical wave velocity at each z location
    V : numpy array[npts], [m/s]
        total wave velocity at each z location
    A : numpy array[npts], [m/s**2]
        horizontal wave acceleration at each z location
    p : numpy array[npts], [N/m**2]
        pressure oscillation at each z location
    phase_speed : float, [m/s]
        Phase speed of wave

    """

    def setup(self):
        super(LinearWaves, self).setup()

        # variables
        self.add_input("Uc", 0.0, units="m/s", desc="mean current speed")

        # parameters
        self.add_input("Hsig_wave", 0.0, units="m")
        self.add_input("Tsig_wave", 0.0, units="s")

        # For Ansys AQWA connection
        self.add_output("phase_speed", val=0.0, units="m/s")

        self.declare_partials("U", ["Uc", "z"])
        self.declare_partials("V", ["Uc", "z"])
        self.declare_partials("W", ["Uc", "z"])
        self.declare_partials("A", ["Uc", "z"])
        self.declare_partials("p", ["Uc", "z"])

    def compute(self, inputs, outputs):
        super(LinearWaves, self).compute(inputs, outputs)

        # water depth
        z_floor = inputs["z_floor"]
        if z_floor > 0.0:
            z_floor *= -1.0
        d = inputs["z_surface"] - z_floor
        # Use zero entries if there is no depth and no water
        if d == 0.0:
            return

        # design wave height
        h = inputs["Hsig_wave"]

        # circular frequency
        omega = 2.0 * np.pi / inputs["Tsig_wave"]

        # compute wave number from dispersion relationship
        k = brentq(lambda k: omega**2 - gravity * k * np.tanh(d * k), 0, 1e3 * omega**2 / gravity, disp=False)
        self.k = k
        outputs["phase_speed"] = omega / k

        # zero at surface
        z_rel = inputs["z"] - inputs["z_surface"]

        # Amplitude
        a = 0.5 * h

        # maximum velocity
        outputs["U"] = a * omega * np.cosh(k * (z_rel + d)) / np.sinh(k * d) + inputs["Uc"]
        outputs["W"] = -a * omega * np.sinh(k * (z_rel + d)) / np.sinh(k * d)
        outputs["V"] = np.sqrt(outputs["U"] ** 2.0 + outputs["W"] ** 2.0)
        # outputs['U0'] = a*omega*np.cosh(k*(0. + d))/np.sinh(k*d) + inputs['Uc']

        # acceleration
        outputs["A"] = (outputs["U"] - inputs["Uc"]) * omega
        # outputs['A0'] = (outputs['U0'] - inputs['Uc']) * omega

        # Pressure oscillation is just sum of static and dynamic contributions
        # Hydrostatic is simple rho * g * z
        # Dynamic is from standard solution to Airy (Potential Flow) Wave theory
        # Full pressure would also include standard dynamic head (0.5*rho*V^2)
        outputs["p"] = inputs["rho_water"] * gravity * (a * np.cosh(k * (z_rel + d)) / np.cosh(k * d) - z_rel)

        # check heights
        idx = np.logical_or(inputs["z"] < z_floor, inputs["z"] > inputs["z_surface"])
        outputs["U"][idx] = 0.0
        outputs["W"][idx] = 0.0
        outputs["V"][idx] = 0.0
        outputs["A"][idx] = 0.0
        outputs["p"][idx] = 0.0

    def compute_partials(self, inputs, J):
        outputs = {}
        self.compute(inputs, outputs)

        # rename
        z_floor = inputs["z_floor"]
        if z_floor > 0.0:
            z_floor *= -1.0
        z = inputs["z"]
        d = inputs["z_surface"] - z_floor
        h = inputs["Hsig_wave"]
        omega = 2.0 * np.pi / inputs["Tsig_wave"]
        k = self.k
        z_rel = z - inputs["z_surface"]

        # Amplitude
        a = 0.5 * h

        # derivatives
        dU_dz = h / 2.0 * omega * np.sinh(k * (z_rel + d)) / np.sinh(k * d) * k
        dU_dUc = np.ones_like(z)
        dW_dz = -h / 2.0 * omega * np.cosh(k * (z_rel + d)) / np.sinh(k * d) * k
        dV_dz = 0.5 / outputs["V"] * (2 * outputs["U"] * dU_dz + 2 * outputs["W"] * dW_dz)
        dV_dUc = 0.5 / outputs["V"] * (2 * outputs["U"] * dU_dUc)
        dA_dz = omega * dU_dz
        dA_dUc = 0.0  # omega*dU_dUc
        dp_dz = inputs["rho_water"] * gravity * (a * np.sinh(k * (z_rel + d)) * k / np.cosh(k * d) - 1.0)

        idx = np.logical_or(z < z_floor, z > inputs["z_surface"])
        dU_dz[idx] = 0.0
        dW_dz[idx] = 0.0
        dV_dz[idx] = 0.0
        dA_dz[idx] = 0.0
        dp_dz[idx] = 0.0
        dU_dUc[idx] = 0.0
        dV_dUc[idx] = 0.0

        # dU0 = np.zeros((1,npts))
        # dA0 = omega * dU0

        J["U", "z"] = np.diag(dU_dz)
        J["U", "Uc"] = dU_dUc
        J["W", "z"] = np.diag(dW_dz)
        J["W", "Uc"] = 0.0
        J["V", "z"] = np.diag(dV_dz)
        J["V", "Uc"] = 0.0
        J["A", "z"] = np.diag(dA_dz)
        J["A", "Uc"] = 0.0
        J["p", "z"] = np.diag(dp_dz)
        J["p", "Uc"] = 0.0
        # J['U0', 'z'] = dU0
        # J['U0', 'Uc'] = 1.0
        # J['A0', 'z'] = dA0
        # J['A0', 'Uc'] = 1.0


class TowerSoil(om.ExplicitComponent):
    """
    Soil stiffness method from Arya, Suresh C., Michael W. O'Neill, and George Pincus.
    Design of structures and foundations for vibrating machines. Gulf Pub Co, 1979.

    Parameters
    ----------
    d0 : float, [m]
        diameter of base of tower
    depth : float, [m]
        depth of foundation in the soil
    G : float, [Pa]
        shear modulus of soil
    nu : float
        Poisson's ratio of soil
    k_usr : numpy array[6], [N/m]
        User overrides of stiffness values. Use positive values and for rigid
        use np.inf. Order is x, theta_x, y, theta_y, z, theta_z

    Returns
    -------
    k : numpy array[6], [N/m]
        Spring stiffness (x, theta_x, y, theta_y, z, theta_z)
    """

    def initialize(self):
        self.options.declare("npts", default=1)

    def setup(self):
        npts = self.options["npts"]

        # variable
        self.add_input("d0", 1.0, units="m")
        self.add_input("depth", 0.0, units="m")

        # inputeter
        self.add_input("G", 140e6, units="Pa")
        self.add_input("nu", 0.4)
        self.add_input("k_usr", -1 * np.ones(6), units="N/m")

        self.add_output("z_k", np.zeros(npts), units="N/m")
        self.add_output("k", np.zeros((npts, 6)), units="N/m")

        self.declare_partials("k", ["d0", "depth"])

    def compute(self, inputs, outputs):

        G = float(inputs["G"])
        nu = float(inputs["nu"])
        depth = float(inputs["depth"])
        h = np.linspace(depth, 0.0, self.options["npts"])
        r0 = 0.5 * float(inputs["d0"])
        # vertical
        eta = 1.0 + 0.6 * (1.0 - nu) * h / r0
        k_z = 4 * G * r0 * eta / (1.0 - nu)

        # horizontal
        eta = 1.0 + 0.55 * (2.0 - nu) * h / r0
        k_x = 32.0 * (1.0 - nu) * G * r0 * eta / (7.0 - 8.0 * nu)

        # rocking
        eta = 1.0 + 1.2 * (1.0 - nu) * h / r0 + 0.2 * (2.0 - nu) * (h / r0) ** 3
        k_thetax = 8.0 * G * r0**3 * eta / (3.0 * (1.0 - nu))

        # torsional
        k_phi = 16.0 * G * r0**3 * np.ones(h.size) / 3.0

        outputs["k"] = np.c_[k_x, k_thetax, k_x, k_thetax, k_z, k_phi]
        outputs["z_k"] = -h
        ind = np.nonzero(inputs["k_usr"] >= 0.0)[0]
        outputs["k"][:, ind] = inputs["k_usr"][np.newaxis, ind]

    def compute_partials(self, inputs, J):

        G = inputs["G"]
        nu = inputs["nu"]
        h = np.linspace(inputs["depth"], 0.0, self.options["npts"])
        r0 = 0.5 * inputs["d0"]

        # vertical
        eta = 1.0 + 0.6 * (1.0 - nu) * h / r0
        deta_dr0 = -0.6 * (1.0 - nu) * h / r0**2
        dkz_dr0 = 4 * G / (1.0 - nu) * (eta + r0 * deta_dr0)

        deta_dh = 0.6 * (1.0 - nu) / r0
        dkz_dh = 4 * G * r0 / (1.0 - nu) * deta_dh

        # horizontal
        eta = 1.0 + 0.55 * (2.0 - nu) * h / r0
        deta_dr0 = -0.55 * (2.0 - nu) * h / r0**2
        dkx_dr0 = 32.0 * (1.0 - nu) * G / (7.0 - 8.0 * nu) * (eta + r0 * deta_dr0)

        deta_dh = 0.55 * (2.0 - nu) / r0
        dkx_dh = 32.0 * (1.0 - nu) * G * r0 / (7.0 - 8.0 * nu) * deta_dh

        # rocking
        eta = 1.0 + 1.2 * (1.0 - nu) * h / r0 + 0.2 * (2.0 - nu) * (h / r0) ** 3
        deta_dr0 = -1.2 * (1.0 - nu) * h / r0**2 - 3 * 0.2 * (2.0 - nu) * (h / r0) ** 3 / r0
        dkthetax_dr0 = 8.0 * G / (3.0 * (1.0 - nu)) * (3 * r0**2 * eta + r0**3 * deta_dr0)

        deta_dh = 1.2 * (1.0 - nu) / r0 + 3 * 0.2 * (2.0 - nu) * (1.0 / r0) ** 3 * h**2
        dkthetax_dh = 8.0 * G * r0**3 / (3.0 * (1.0 - nu)) * deta_dh

        # torsional
        dkphi_dr0 = 16.0 * G * 3 * r0**2 / 3.0
        dkphi_dh = 0.0

        dk_dr0 = np.c_[dkx_dr0, dkthetax_dr0, dkx_dr0, dkthetax_dr0, dkz_dr0, dkphi_dr0]
        # dk_dr0[inputs['rigid']] = 0.0
        dk_dh = np.c_[dkx_dh, dkthetax_dh, dkx_dh, dkthetax_dh, dkz_dh, dkphi_dh]
        # dk_dh[inputs['rigid']] = 0.0

        J["k", "d0"] = 0.5 * dk_dr0
        J["k", "depth"] = dk_dh
        ind = np.nonzero(inputs["k_usr"] >= 0.0)[0]
        J["k", "d0"][:, ind] = 0.0
        J["k", "depth"][:, ind] = 0.0
