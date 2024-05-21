import math

import numpy as np
import openmdao.api as om

from wisdem.commonse.akima import Akima
from wisdem.commonse.csystem import DirectionVector
from wisdem.commonse.utilities import cosd, sind  # , linspace_with_deriv, interp_with_deriv, hstack, vstack
from wisdem.commonse.environment import LogWind, PowerWind, LinearWaves

# -----------------
#  Helper Functions
# -----------------
# "Experiments on the Flow Past a Circular Cylinder at Very High Reynolds Numbers", Roshko
Re_pt = [
    0.00001,
    0.0001,
    0.0010,
    0.0100,
    0.0200,
    0.1220,
    0.2000,
    0.3000,
    0.4000,
    0.5000,
    1.0000,
    1.5000,
    2.0000,
    2.5000,
    3.0000,
    3.5000,
    4.0000,
    5.0000,
    10.0000,
]
cd_pt = [
    4.0000,
    2.0000,
    1.1100,
    1.1100,
    1.2000,
    1.2000,
    1.1700,
    0.9000,
    0.5400,
    0.3100,
    0.3800,
    0.4600,
    0.5300,
    0.5700,
    0.6100,
    0.6400,
    0.6700,
    0.7000,
    0.7000,
]
# For rectangular
# This assumes that the CD only depends on the aspect ratio
# No consideration on the KC number and Reynolds (assuming it's not sensitive to Re bc of sharp corners)
AR_pt = [
    2.0,
    1,
    0.5
]

cd_AR_pt = [
    2.2,
    2.0,
    1.4
]

drag_spline = Akima(np.log10(Re_pt), cd_pt, delta_x=0.0)  # exact akima because control points do not change
drag_AR_spline = Akima(AR_pt, cd_AR_pt, delta_x=0.0)  # exact akima because control points do not change


def cylinderDrag(Re):
    """Drag coefficient for a smooth circular cylinder.

    Parameters
    ----------
    Re : array_like
        Reynolds number

    Returns
    -------
    cd : array_like
        drag coefficient (normalized by cylinder diameter)

    """

    ReN = Re / 1.0e6

    cd = np.zeros_like(Re)
    dcd_dRe = np.zeros_like(Re)
    idx = ReN > 0
    cd[idx], dcd_dRe[idx], _, _ = drag_spline.interp(np.log10(ReN[idx]))
    dcd_dRe[idx] /= Re[idx] * math.log(10)  # chain rule

    return cd, dcd_dRe

def rectangular_cylinderDrag(AR):
    """Drag coefficient for a rectangular cylinder.

    Parameters
    ----------
    AR : array_like
        Aspect ratio

    Returns
    -------
    cd : array_like
        drag coefficient (normalized by frontal project area)

    """

    cd = np.zeros_like(AR)
    dcd_dAR = np.zeros_like(AR)
    idx = AR > 0
    cd[idx], dcd_dAR[idx], _, _ = drag_AR_spline.interp(AR)

    return cd, dcd_dAR


# -----------------
#  Components
# -----------------


class AeroHydroLoads(om.ExplicitComponent):
    """
    Compute summed forces due to wind and wave loads.

    Parameters
    ----------
    windLoads_Px : numpy array[nPoints], [N/m]
        distributed loads, force per unit length in x-direction
    windLoads_Py : numpy array[nPoints], [N/m]
        distributed loads, force per unit length in y-direction
    windLoads_Pz : numpy array[nPoints], [N/m]
        distributed loads, force per unit length in z-direction
    windLoads_qdyn : numpy array[nPoints], [N/m**2]
        dynamic pressure
    windLoads_z : numpy array[nPoints], [m]
        corresponding heights
    windLoads_beta : float, [deg]
        wind/wave angle relative to inertia c.s.
    waveLoads_Px : numpy array[nPoints], [N/m]
        distributed loads, force per unit length in x-direction
    waveLoads_Py : numpy array[nPoints], [N/m]
        distributed loads, force per unit length in y-direction
    waveLoads_Pz : numpy array[nPoints], [N/m]
        distributed loads, force per unit length in z-direction
    waveLoads_qdyn : numpy array[nPoints], [N/m**2]
        dynamic pressure
    waveLoads_z : numpy array[nPoints], [m]
        corresponding heights
    waveLoads_beta : float, [deg]
        wind/wave angle relative to inertia c.s.
    z : numpy array[nPoints], [m]
        locations along cylinder
    yaw : float, [deg]
        yaw angle

    Returns
    -------
    Px : numpy array[nPoints], [N/m]
        force per unit length in x-direction
    Py : numpy array[nPoints], [N/m]
        force per unit length in y-direction
    Pz : numpy array[nPoints], [N/m]
        force per unit length in z-direction
    qdyn : numpy array[nPoints], [N/m**2]
        dynamic pressure

    """

    def initialize(self):
        self.options.declare("nPoints")

    def setup(self):
        nPoints = self.options["nPoints"]

        self.add_input("windLoads_Px", np.zeros(nPoints), units="N/m")
        self.add_input("windLoads_Py", np.zeros(nPoints), units="N/m")
        self.add_input("windLoads_Pz", np.zeros(nPoints), units="N/m")
        self.add_input("windLoads_qdyn", np.zeros(nPoints), units="N/m**2")
        self.add_input("windLoads_z", np.zeros(nPoints), units="m")
        self.add_input("windLoads_beta", 0.0, units="deg")
        self.add_input("waveLoads_Px", np.zeros(nPoints), units="N/m")
        self.add_input("waveLoads_Py", np.zeros(nPoints), units="N/m")
        self.add_input("waveLoads_Pz", np.zeros(nPoints), units="N/m")
        self.add_input("waveLoads_qdyn", np.zeros(nPoints), units="N/m**2")
        self.add_input("waveLoads_z", np.zeros(nPoints), units="m")
        self.add_input("waveLoads_beta", 0.0, units="deg")
        self.add_input("z", np.zeros(nPoints), units="m")
        self.add_input("yaw", 0.0, units="deg")

        self.add_output("Px", np.zeros(nPoints), units="N/m")
        self.add_output("Py", np.zeros(nPoints), units="N/m")
        self.add_output("Pz", np.zeros(nPoints), units="N/m")
        self.add_output("qdyn", np.zeros(nPoints), units="N/m**2")

    def compute(self, inputs, outputs):
        z = inputs["z"]
        windLoads = (
            DirectionVector(inputs["windLoads_Px"], inputs["windLoads_Py"], inputs["windLoads_Pz"])
            .inertialToWind(inputs["windLoads_beta"])
            .windToYaw(inputs["yaw"])
        )
        waveLoads = (
            DirectionVector(inputs["waveLoads_Px"], inputs["waveLoads_Py"], inputs["waveLoads_Pz"])
            .inertialToWind(inputs["waveLoads_beta"])
            .windToYaw(inputs["yaw"])
        )

        Px = np.interp(z, inputs["windLoads_z"], windLoads.x) + np.interp(z, inputs["waveLoads_z"], waveLoads.x)
        Py = np.interp(z, inputs["windLoads_z"], windLoads.y) + np.interp(z, inputs["waveLoads_z"], waveLoads.y)
        Pz = np.interp(z, inputs["windLoads_z"], windLoads.z) + np.interp(z, inputs["waveLoads_z"], waveLoads.z)
        qdyn = np.interp(z, inputs["windLoads_z"], inputs["windLoads_qdyn"]) + np.interp(
            z, inputs["waveLoads_z"], inputs["waveLoads_qdyn"]
        )

        # The following are redundant, at one point we will consolidate them to something that works for both cylinder (not using vartrees) and jacket (still using vartrees)
        outputs["Px"] = Px
        outputs["Py"] = Py
        outputs["Pz"] = Pz
        outputs["qdyn"] = qdyn


# -----------------


class CylinderWindDrag(om.ExplicitComponent):
    """
    Compute drag forces on a cylindrical cylinder due to wind.

    Parameters
    ----------
    U : numpy array[nPoints], [m/s]
        magnitude of wind speed
    z : numpy array[nPoints], [m]
        heights where wind speed was computed
    d : numpy array[nPoints], [m]
        corresponding diameter of cylinder section
    beta_wind : float, [deg]
        corresponding wind angles relative to inertial coordinate system
    rho_air : float, [kg/m**3]
        air density
    mu_air : float, [kg/(m*]
        dynamic viscosity of air
    cd_usr : float
        User input drag coefficient to override Reynolds number based one

    Returns
    -------
    windLoads_Px : numpy array[nPoints], [N/m]
        distributed loads, force per unit length in x-direction
    windLoads_Py : numpy array[nPoints], [N/m]
        distributed loads, force per unit length in y-direction
    windLoads_Pz : numpy array[nPoints], [N/m]
        distributed loads, force per unit length in z-direction
    windLoads_qdyn : numpy array[nPoints], [N/m**2]
        dynamic pressure
    windLoads_z : numpy array[nPoints], [m]
        corresponding heights
    windLoads_beta : float, [deg]
        wind/wave angle relative to inertia c.s.

    """

    def initialize(self):
        self.options.declare("nPoints")

    def setup(self):
        nPoints = self.options["nPoints"]

        # variables
        self.add_input("U", np.zeros(nPoints), units="m/s")
        self.add_input("z", np.zeros(nPoints), units="m")
        self.add_input("d", np.zeros(nPoints), units="m")
        self.add_input("beta_wind", 0.0, units="deg")
        self.add_input("rho_air", 0.0, units="kg/m**3")
        self.add_input("mu_air", 0.0, units="kg/(m*s)")
        self.add_input("cd_usr", -1.0*np.ones(nPoints))
        # self.add_input("ca_usr", -1.0)

        self.add_output("windLoads_Px", np.zeros(nPoints), units="N/m")
        self.add_output("windLoads_Py", np.zeros(nPoints), units="N/m")
        self.add_output("windLoads_Pz", np.zeros(nPoints), units="N/m")
        self.add_output("windLoads_qdyn", np.zeros(nPoints), units="N/m**2")
        self.add_output("windLoads_z", np.zeros(nPoints), units="m")
        self.add_output("windLoads_beta", 0.0, units="deg")

        arange = np.arange(nPoints)
        self.declare_partials("windLoads_Px", "U", rows=arange, cols=arange)
        self.declare_partials("windLoads_Px", "d", rows=arange, cols=arange)

        self.declare_partials("windLoads_Py", "U", rows=arange, cols=arange)
        self.declare_partials("windLoads_Py", "d", rows=arange, cols=arange)
        self.declare_partials(["windLoads_Px", "windLoads_Py"], "cd_usr", method="fd")

        self.declare_partials("windLoads_qdyn", "U", rows=arange, cols=arange)
        self.declare_partials("windLoads_qdyn", "rho_air", method="fd")

        self.declare_partials("windLoads_z", "z", rows=arange, cols=arange, val=1.0)
        self.declare_partials("windLoads_beta", "beta_wind", val=1.0)

    def compute(self, inputs, outputs):
        rho = inputs["rho_air"]
        U = inputs["U"]
        d = inputs["d"]
        mu = inputs["mu_air"]
        beta = inputs["beta_wind"]

        # dynamic pressure
        q = 0.5 * rho * U**2

        # Reynolds number and drag
        if np.any(inputs["cd_usr"] < 0.0):
            Re = rho * U * d / mu
            cd, dcd_dRe = cylinderDrag(Re)
        else:
            cd = inputs["cd_usr"]
            Re = 1.0
            dcd_dRe = 0.0
        Fp = q * cd * d

        # components of distributed loads
        Px = Fp * cosd(beta)
        Py = Fp * sind(beta)
        Pz = 0 * Fp

        # pack data
        outputs["windLoads_Px"] = Px
        outputs["windLoads_Py"] = Py
        outputs["windLoads_Pz"] = Pz
        outputs["windLoads_qdyn"] = q
        outputs["windLoads_z"] = inputs["z"]
        outputs["windLoads_beta"] = beta

    def compute_partials(self, inputs, J):
        # rename
        rho = inputs["rho_air"]
        U = inputs["U"]
        d = inputs["d"]
        mu = inputs["mu_air"]
        beta = inputs["beta_wind"]

        # dynamic pressure
        q = 0.5 * rho * U**2

        # Reynolds number and drag
        if np.any(inputs["cd_usr"] < 0.0):
            Re = rho * U * d / mu
            cd, dcd_dRe = cylinderDrag(Re)
        else:
            cd = inputs["cd_usr"]
            Re = 1.0
            dcd_dRe = 0.0

        # derivatives
        dq_dU = rho * U
        const = (dq_dU * cd + q * dcd_dRe * rho * d / mu) * d
        dPx_dU = const * cosd(beta)
        dPy_dU = const * sind(beta)

        const = (cd + dcd_dRe * Re) * q
        dPx_dd = const * cosd(beta)
        dPy_dd = const * sind(beta)

        J["windLoads_Px", "U"] = dPx_dU
        J["windLoads_Px", "d"] = dPx_dd

        J["windLoads_Py", "U"] = dPy_dU
        J["windLoads_Py", "d"] = dPy_dd

        J["windLoads_qdyn", "U"] = dq_dU


class RectangularCylinderWindDrag(om.ExplicitComponent):
    """
    Compute drag forces on a rectangular cylinder due to wind.

    Parameters
    ----------
    U : numpy array[nPoints], [m/s]
        magnitude of wind speed
    z : numpy array[nPoints], [m]
        heights where wind speed was computed
    a : numpy array[nPoints], [m]
        corresponding side length a of rectangular cylinder section
    b : numpy array[nPoints], [m]
        corresponding side length b of rectangular cylinder section
    beta_wind : float, [deg]
        corresponding wind angles relative to inertial coordinate system
    rho_air : float, [kg/m**3]
        air density
    mu_air : float, [kg/(m*]
        dynamic viscosity of air
    cd_usr : float
        User input drag coefficient to override Reynolds number based one

    Returns
    -------
    windLoads_Px : numpy array[nPoints], [N/m]
        distributed loads, force per unit length in x-direction
    windLoads_Py : numpy array[nPoints], [N/m]
        distributed loads, force per unit length in y-direction
    windLoads_Pz : numpy array[nPoints], [N/m]
        distributed loads, force per unit length in z-direction
    windLoads_qdyn : numpy array[nPoints], [N/m**2]
        dynamic pressure
    windLoads_z : numpy array[nPoints], [m]
        corresponding heights
    windLoads_beta : float, [deg]
        wind/wave angle relative to inertia c.s.

    """

    def initialize(self):
        self.options.declare("nPoints")

    def setup(self):
        nPoints = self.options["nPoints"]

        # variables
        self.add_input("U", np.zeros(nPoints), units="m/s")
        self.add_input("z", np.zeros(nPoints), units="m")
        self.add_input("a", np.zeros(nPoints), units="m")
        self.add_input("b", np.zeros(nPoints), units="m")
        self.add_input("beta_wind", 0.0, units="deg")
        self.add_input("rho_air", 0.0, units="kg/m**3")
        self.add_input("mu_air", 0.0, units="kg/(m*s)")
        self.add_input("cd_usr", -1.0*np.ones(nPoints))
        self.add_input("cdy_usr", -1.0*np.ones(nPoints))

        self.add_output("windLoads_Px", np.zeros(nPoints), units="N/m")
        self.add_output("windLoads_Py", np.zeros(nPoints), units="N/m")
        self.add_output("windLoads_Pz", np.zeros(nPoints), units="N/m")
        self.add_output("windLoads_qdyn", np.zeros(nPoints), units="N/m**2")
        self.add_output("windLoads_z", np.zeros(nPoints), units="m")
        self.add_output("windLoads_beta", 0.0, units="deg")

        arange = np.arange(nPoints)
        self.declare_partials("windLoads_Px", "U", rows=arange, cols=arange)
        self.declare_partials("windLoads_Px", "a", rows=arange, cols=arange)
        self.declare_partials("windLoads_Px", "b", rows=arange, cols=arange)

        self.declare_partials("windLoads_Py", "U", rows=arange, cols=arange)
        self.declare_partials("windLoads_Py", "a", rows=arange, cols=arange)
        self.declare_partials("windLoads_Py", "b", rows=arange, cols=arange)
        self.declare_partials(["windLoads_Px", "windLoads_Py"], "cd_usr", method="fd")

        self.declare_partials("windLoads_qdyn", "U", rows=arange, cols=arange)
        self.declare_partials("windLoads_qdyn", "rho_air", method="fd")

        self.declare_partials("windLoads_z", "z", rows=arange, cols=arange, val=1.0)
        self.declare_partials("windLoads_beta", "beta_wind", val=1.0)

    def compute(self, inputs, outputs):
        rho = inputs["rho_air"]
        U = inputs["U"]
        a = inputs["a"]
        b = inputs["b"]
        beta = inputs["beta_wind"]

        # dynamic pressure
        q = 0.5 * rho * U**2
        # TODO: This is wrong for now. The force can be simply decomposed to two direction. The cd depends on beta
        qx = q* (cosd(beta)**2)
        qy = q* (sind(beta)**2)


        # Aspect ratio and drag
        if np.any(inputs["cd_usr"] < 0.0):
            ARx = b/a
            cdx, dcdx_dARx = rectangular_cylinderDrag(ARx)
        else:
            cdx = inputs["cd_usr"]
            ARx = 1.0
            dcdx_dARx = 0.0

        if np.any(inputs["cdy_usr"] < 0.0):
            ARy = a/b
            cdy, dcdy_dARy = rectangular_cylinderDrag(ARy)
        else:
            cdy = inputs["cdy_usr"]
            ARy = 1.0
            dcdy_dARy = 0.0

        # components of distributed loads
        Px = qx * cdx * b
        Py = qy * cdy * a
        Pz = 0 * q

        # pack data
        outputs["windLoads_Px"] = Px
        outputs["windLoads_Py"] = Py
        outputs["windLoads_Pz"] = Pz
        outputs["windLoads_qdyn"] = q
        outputs["windLoads_z"] = inputs["z"]
        outputs["windLoads_beta"] = beta

    def compute_partials(self, inputs, J):
        # rename
        rho = inputs["rho_air"]
        U = inputs["U"]
        a = inputs["a"]
        b = inputs["b"]
        beta = inputs["beta_wind"]

        # dynamic pressure
        q = 0.5 * rho * U**2
        dq_dU = rho * U
        dqx_dU = dq_dU* (cosd(beta)**2)
        dqy_dU = dq_dU* (sind(beta)**2)

        # Aspect ratio and drag
        if np.any(inputs["cd_usr"] < 0.0):
            ARx = b/a
            dARx_da = -1/a**2
            dARx_db = 1/a
            cdx, dcdx_dARx = rectangular_cylinderDrag(ARx)
        else:
            cdx = inputs["cd_usr"]
            ARx = 1.0
            dcdx_dARx = 0.0

        if np.any(inputs["cdy_usr"] < 0.0):
            ARy = a/b
            dARy_da = 1/b
            dARy_db = -1/b**2
            cdy, dcdy_dARy = rectangular_cylinderDrag(ARy)
        else:
            cdy = inputs["cdy_usr"]
            ARy = 1.0
            dcdy_dARy = 0.0
            
        # components of distributed loads
        # Px = qx * cdx * b
        # Py = qy * cdy * a
        # Pz = 0 * q
        # derivatives
  
        dPx_da = q * b * (dcdx_dARx * dARx_da)
        dPx_db = q * b * (dcdx_dARx * dARx_db)
        dPy_da = q * b * (dcdy_dARy * dARy_da)
        dPy_db = q * b * (dcdy_dARy * dARy_db)
        dPx_dU = dqx_dU * cdx * b
        dPy_dU = dqy_dU * cdy * a


        J["windLoads_Px", "U"] = dPx_dU
        J["windLoads_Px", "a"] = dPx_da
        J["windLoads_Px", "b"] = dPx_db

        

        J["windLoads_Py", "U"] = dPy_dU
        J["windLoads_Py", "a"] = dPy_da
        J["windLoads_Py", "b"] = dPy_db

        J["windLoads_qdyn", "U"] = dq_dU

# -----------------


class CylinderWaveDrag(om.ExplicitComponent):
    """
    Compute drag forces on a cylindrical cylinder due to waves.

    Parameters
    ----------
    U : numpy array[nPoints], [m/s]
        magnitude of wave speed
    A : numpy array[nPoints], [m/s**2]
        magnitude of wave acceleration
    p : numpy array[nPoints], [N/m**2]
        pressure oscillation
    z : numpy array[nPoints], [m]
        heights where wave speed was computed
    d : numpy array[nPoints], [m]
        corresponding diameter of cylinder section
    beta_wave : float, [deg]
        corresponding wave angles relative to inertial coordinate system
    rho_water : float, [kg/m**3]
        water density
    mu_water : float, [kg/(m*]
        dynamic viscosity of water
    ca_usr : float
        added mass coefficient
    cd_usr : float
        User input drag coefficient to override Reynolds number based one

    Returns
    -------
    waveLoads_Px : numpy array[nPoints], [N/m]
        distributed loads, force per unit length in x-direction
    waveLoads_Py : numpy array[nPoints], [N/m]
        distributed loads, force per unit length in y-direction
    waveLoads_Pz : numpy array[nPoints], [N/m]
        distributed loads, force per unit length in z-direction
    waveLoads_qdyn : numpy array[nPoints], [N/m**2]
        dynamic pressure
    waveLoads_pt : numpy array[nPoints], [N/m**2]
        total (static+dynamic) pressure
    waveLoads_z : numpy array[nPoints], [m]
        corresponding heights
    waveLoads_beta : float, [deg]
        wind/wave angle relative to inertia c.s.

    """

    def initialize(self):
        self.options.declare("nPoints")

    def setup(self):
        nPoints = self.options["nPoints"]

        # variables
        self.add_input("U", np.zeros(nPoints), units="m/s")
        self.add_input("A", np.zeros(nPoints), units="m/s**2")
        self.add_input("p", np.zeros(nPoints), units="N/m**2")
        self.add_input("z", np.zeros(nPoints), units="m")
        self.add_input("d", np.zeros(nPoints), units="m")
        self.add_input("beta_wave", 0.0, units="deg")
        self.add_input("rho_water", 0.0, units="kg/m**3")
        self.add_input("mu_water", 0.0, units="kg/(m*s)")
        self.add_input("ca_usr", -1.0*np.ones(nPoints))
        self.add_input("cd_usr", -1.0*np.ones(nPoints))

        self.add_output("waveLoads_Px", np.zeros(nPoints), units="N/m")
        self.add_output("waveLoads_Py", np.zeros(nPoints), units="N/m")
        self.add_output("waveLoads_Pz", np.zeros(nPoints), units="N/m")
        self.add_output("waveLoads_qdyn", np.zeros(nPoints), units="N/m**2")
        self.add_output("waveLoads_pt", np.zeros(nPoints), units="N/m**2")
        self.add_output("waveLoads_z", np.zeros(nPoints), units="m")
        self.add_output("waveLoads_beta", 0.0, units="deg")

        self.declare_partials("*", "rho_water", method="fd")

        arange = np.arange(nPoints)
        self.declare_partials(["waveLoads_Px", "waveLoads_Py"], ["U", "d", "ca_usr", "cd_usr", "beta_wave"], method="fd")
        self.declare_partials("waveLoads_Px", "A", rows=arange, cols=arange)

        self.declare_partials("waveLoads_Py", "A", rows=arange, cols=arange)

        self.declare_partials("waveLoads_qdyn", "U", rows=arange, cols=arange)
        self.declare_partials("waveLoads_pt", "U", rows=arange, cols=arange)
        self.declare_partials("waveLoads_pt", "p", rows=arange, cols=arange, val=1.0)
        self.declare_partials("waveLoads_z", "z", rows=arange, cols=arange, val=1.0)
        self.declare_partials("waveLoads_beta", "beta_wave", val=1.0)

    def compute(self, inputs, outputs):
        # wlevel = inputs['wlevel']
        # if wlevel > 0.0: wlevel *= -1.0

        rho = inputs["rho_water"]
        U = inputs["U"]
        # U0 = inputs['U0']
        d = inputs["d"]
        # zrel= inputs['z']-wlevel
        mu = inputs["mu_water"]
        beta = inputs["beta_wave"]
        # beta0 = inputs['beta0']

        # dynamic pressure
        q = 0.5 * rho * U * np.abs(U)
        # q0= 0.5*rho*U0**2

        # Reynolds number and drag
        if np.any(inputs["cd_usr"] < 0.0):
            Re = rho * U * d / mu
            cd, dcd_dRe = cylinderDrag(Re)
        else:
            cd = inputs["cd_usr"] * np.ones_like(d)
            Re = 1.0
            dcd_dRe = 0.0

        if np.any(inputs["ca_usr"] < 0.0):
            ca = 1.0
        else:
            ca = inputs["ca_usr"]

        # inertial and drag forces
        Fi = rho * ca * math.pi / 4.0 * d**2 * inputs["A"]  # Morrison's equation
        Fd = q * cd * d
        Fp = Fi + Fd

        # components of distributed loads
        Px = Fp * cosd(beta)
        Py = Fp * sind(beta)
        Pz = 0.0 * Fp

        # FORCES [N/m] AT z=0 m
        # idx0 = np.abs(zrel).argmin()  # closest index to z=0, used to find d at z=0
        # d0 = d[idx0]  # initialize
        # cd0 = cd[idx0]  # initialize
        # if (zrel[idx0]<0.) and (idx0< (zrel.size-1)):       # point below water
        #    d0 = np.mean(d[idx0:idx0+2])
        #    cd0 = np.mean(cd[idx0:idx0+2])
        # elif (zrel[idx0]>0.) and (idx0>0):     # point above water
        #    d0 = np.mean(d[idx0-1:idx0+1])
        #    cd0 = np.mean(cd[idx0-1:idx0+1])
        # Fi0 = rho*inputs['cm']*math.pi/4.0*d0**2*inputs['A0']  # Morrison's equation
        # Fd0 = q0*cd0*d0
        # Fp0 = Fi0 + Fd0

        # Px0 = Fp0*cosd(beta0)
        # Py0 = Fp0*sind(beta0)
        # Pz0 = 0.*Fp0

        # Store qties at z=0 MSL
        # outputs['waveLoads_Px0'] = Px0
        # outputs['waveLoads_Py0'] = Py0
        # outputs['waveLoads_Pz0'] = Pz0
        # outputs['waveLoads_qdyn0'] = q0
        # outputs['waveLoads_beta0'] = beta0

        # pack data
        outputs["waveLoads_Px"] = Px
        outputs["waveLoads_Py"] = Py
        outputs["waveLoads_Pz"] = Pz
        outputs["waveLoads_qdyn"] = q
        outputs["waveLoads_pt"] = q + inputs["p"]
        outputs["waveLoads_z"] = inputs["z"]
        outputs["waveLoads_beta"] = beta

    def compute_partials(self, inputs, J):
        # wlevel = inputs['wlevel']
        # if wlevel > 0.0: wlevel *= -1.0

        rho = inputs["rho_water"]
        U = inputs["U"]
        # U0 = inputs['U0']
        d = inputs["d"]
        # zrel= inputs['z']-wlevel
        mu = inputs["mu_water"]
        beta = inputs["beta_wave"]
        # beta0 = inputs['beta0']

        # dynamic pressure
        q = 0.5 * rho * U**2
        # q0= 0.5*rho*U0**2

        # Reynolds number and drag
        if np.any(inputs["cd_usr"] < 0.0):
            cd = inputs["cd_usr"] * np.ones_like(d)
            Re = 1.0
            dcd_dRe = 0.0
        else:
            Re = rho * U * d / mu
            cd, dcd_dRe = cylinderDrag(Re)

        if np.any(inputs["ca_usr"] < 0.0):
            ca = 1.0
        else:
            ca = inputs["ca_usr"]

        # derivatives
        dq_dU = rho * U
        const = (dq_dU * cd + q * dcd_dRe * rho * d / mu) * d
        dPx_dU = const * cosd(beta)
        dPy_dU = const * sind(beta)

        const = (cd + dcd_dRe * Re) * q + rho * ca * math.pi / 4.0 * 2 * d * inputs["A"]
        dPx_dd = const * cosd(beta)
        dPy_dd = const * sind(beta)

        const = rho * ca * math.pi / 4.0 * d**2
        dPx_dA = const * cosd(beta)
        dPy_dA = const * sind(beta)

        J["waveLoads_Px", "A"] = dPx_dA
        J["waveLoads_Py", "A"] = dPy_dA
        J["waveLoads_qdyn", "U"] = dq_dU
        J["waveLoads_pt", "U"] = dq_dU


class RectangularCylinderWaveDrag(om.ExplicitComponent):
    """
    Compute drag forces on a rectangular cylinder due to waves.

    Parameters
    ----------
    U : numpy array[nPoints], [m/s]
        magnitude of wave speed
    A : numpy array[nPoints], [m/s**2]
        magnitude of wave acceleration
    p : numpy array[nPoints], [N/m**2]
        pressure oscillation
    z : numpy array[nPoints], [m]
        heights where wave speed was computed
    a : numpy array[nPoints], [m]
        corresponding side length a of rectangular cylinder section
    b : numpy array[nPoints], [m]
        corresponding side length b of rectangular cylinder section
    beta_wave : float, [deg]
        corresponding wave angles relative to inertial coordinate system
    rho_water : float, [kg/m**3]
        water density
    mu_water : float, [kg/(m*]
        dynamic viscosity of water
    cax, cay : float
        mass coefficient
    cd_usr : float
        User input drag coefficient to override Reynolds number based one

    Returns
    -------
    waveLoads_Px : numpy array[nPoints], [N/m]
        distributed loads, force per unit length in x-direction
    waveLoads_Py : numpy array[nPoints], [N/m]
        distributed loads, force per unit length in y-direction
    waveLoads_Pz : numpy array[nPoints], [N/m]
        distributed loads, force per unit length in z-direction
    waveLoads_qdyn : numpy array[nPoints], [N/m**2]
        dynamic pressure
    waveLoads_pt : numpy array[nPoints], [N/m**2]
        total (static+dynamic) pressure
    waveLoads_z : numpy array[nPoints], [m]
        corresponding heights
    waveLoads_beta : float, [deg]
        wind/wave angle relative to inertia c.s.

    """

    def initialize(self):
        self.options.declare("nPoints")

    def setup(self):
        nPoints = self.options["nPoints"]

        # variables
        self.add_input("U", np.zeros(nPoints), units="m/s")
        self.add_input("A", np.zeros(nPoints), units="m/s**2")
        self.add_input("p", np.zeros(nPoints), units="N/m**2")
        self.add_input("z", np.zeros(nPoints), units="m")
        self.add_input("a", np.zeros(nPoints), units="m")
        self.add_input("b", np.zeros(nPoints), units="m")
        self.add_input("beta_wave", 0.0, units="deg")
        self.add_input("rho_water", 0.0, units="kg/m**3")
        self.add_input("mu_water", 0.0, units="kg/(m*s)")
        self.add_input("ca_usr", -1.0*np.ones(nPoints))
        self.add_input("cay_usr", -1.0*np.ones(nPoints))
        self.add_input("cd_usr", -1.0*np.ones(nPoints))
        self.add_input("cdy_usr", -1.0*np.ones(nPoints))

        self.add_output("waveLoads_Px", np.zeros(nPoints), units="N/m")
        self.add_output("waveLoads_Py", np.zeros(nPoints), units="N/m")
        self.add_output("waveLoads_Pz", np.zeros(nPoints), units="N/m")
        self.add_output("waveLoads_qdyn", np.zeros(nPoints), units="N/m**2")
        self.add_output("waveLoads_pt", np.zeros(nPoints), units="N/m**2")
        self.add_output("waveLoads_z", np.zeros(nPoints), units="m")
        self.add_output("waveLoads_beta", 0.0, units="deg")

        self.declare_partials("*", "rho_water", method="fd")

        arange = np.arange(nPoints)
        self.declare_partials(["waveLoads_Px", "waveLoads_Py"], ["U", "a", "b", "ca_usr", "cay_usr", "cd_usr", "cdy_usr", "beta_wave"], method="fd")
        self.declare_partials("waveLoads_Px", "A", rows=arange, cols=arange)

        self.declare_partials("waveLoads_Py", "A", rows=arange, cols=arange)

        self.declare_partials("waveLoads_qdyn", "U", rows=arange, cols=arange)
        self.declare_partials("waveLoads_pt", "U", rows=arange, cols=arange)
        self.declare_partials("waveLoads_pt", "p", rows=arange, cols=arange, val=1.0)
        self.declare_partials("waveLoads_z", "z", rows=arange, cols=arange, val=1.0)
        self.declare_partials("waveLoads_beta", "beta_wave", val=1.0)

    def compute(self, inputs, outputs):
        # wlevel = inputs['wlevel']
        # if wlevel > 0.0: wlevel *= -1.0

        rho = inputs["rho_water"]
        U = inputs["U"]
        # U0 = inputs['U0']
        a = inputs["a"]
        b = inputs["b"]
        # zrel= inputs['z']-wlevel
        mu = inputs["mu_water"]
        beta = inputs["beta_wave"]
        # beta0 = inputs['beta0']

        # dynamic pressure
        q = 0.5 * rho * U * np.abs(U)
        qx = q * cosd(beta) * np.abs(cosd(beta))
        qy = q * sind(beta) * np.abs(sind(beta))
        # q0= 0.5*rho*U0**2

        # Aspect ratio and drag
        if np.any(inputs["cd_usr"] < 0.0):
            ARx = b/a
            cdx, dcdx_dARx = rectangular_cylinderDrag(ARx)
        else:
            cdx = inputs["cd_usr"]

        if np.any(inputs["cdy_usr"] < 0.0):
            ARy = a/b
            cdy, dcdy_dARy = rectangular_cylinderDrag(ARy)
        else:
            cdy = inputs["cdy_usr"]


        if np.any(inputs["ca_usr"] < 0.0):
            # TODO: add the correct internal calculation
            cax = 2.0
        else:
            cax = inputs["ca_usr"]

        if np.any(inputs["cay_usr"] < 0.0):
            # TODO: add the correct internal calculation
            cay = 2.0
        else:
            cay = inputs["cay_usr"]


        # inertial and drag forces
        Fix = rho * cax * a * b * inputs["A"] * cosd(beta)  # Morrison's equation
        Fiy = rho * cay * a * b * inputs["A"] * sind(beta) # Morrison's equation
        

        # components of distributed loads
        Px = qx * cdx * b + Fix
        Py =  qy * cdy * b + Fiy
        Pz = 0.0 * q

        # pack data
        outputs["waveLoads_Px"] = Px
        outputs["waveLoads_Py"] = Py
        outputs["waveLoads_Pz"] = Pz
        outputs["waveLoads_qdyn"] = q
        outputs["waveLoads_pt"] = q + inputs["p"]
        outputs["waveLoads_z"] = inputs["z"]
        outputs["waveLoads_beta"] = beta

    def compute_partials(self, inputs, J):
        # wlevel = inputs['wlevel']
        # if wlevel > 0.0: wlevel *= -1.0

        rho = inputs["rho_water"]
        U = inputs["U"]
        # U0 = inputs['U0']
        a = inputs["a"]
        b = inputs["b"]
        # zrel= inputs['z']-wlevel
        mu = inputs["mu_water"]
        beta = inputs["beta_wave"]
        # beta0 = inputs['beta0']

        # dynamic pressure
        q = 0.5 * rho * U * np.abs(U)
        if U > 0:
            dq_dU = rho*U
        else:
            dq_dU = - rho*U
    
        dqx_dU = dq_dU * cosd(beta) * np.abs(cosd(beta))
        dqy_dU = dq_dU * sind(beta) * np.abs(sind(beta))

        # Aspect ratio and drag
        if np.any(inputs["cd_usr"] < 0.0):
            ARx = b/a
            dARx_da = -1/a**2
            dARx_db = 1/a
            cdx, dcdx_dARx = rectangular_cylinderDrag(ARx)
        else:
            # 
            cdx = inputs["cd_usr"]
            ARx = 1.0
            dcdx_dARx = 0.0

        if np.any(inputs["cdy_usr"] < 0.0):
            ARy = a/b
            dARy_da = 1/b
            dARy_db = -1/b**2
            cdy, dcdy_dARy = rectangular_cylinderDrag(ARy)
        else:
            # 
            cdy = inputs["cdy_usr"]
            ARy = 1.0
            dcdy_dARy = 0.0

        if np.any(inputs["ca_usr"] < 0.0):
        # TODO: add the correct internal calculation
            cax = 2.0
        else:
            cax = inputs["ca_usr"]

        if np.any(inputs["cay_usr"] < 0.0):
        # TODO: add the correct internal calculation
            cay = 2.0
        else:
            cay = inputs["cay_usr"]

        # inertial and drag forces
        # Fix = rho * inputs["cmx"] * a * b * inputs["A"] * cosd(beta)  # Morrison's equation
        # Fiy = rho * inputs["cmy"] * a * b * inputs["A"] * sind(beta) # Morrison's equation
        

        # components of distributed loads
        # Px = qx * cdx * b + Fix
        # Py =  qy * cdy * b + Fiy
        # Pz = 0.0 * q
            
        dFix_da = rho * cax * b * inputs["A"] * cosd(beta)
        dFix_db = rho * cax * a * inputs["A"] * cosd(beta)
        dFiy_da = rho * cay * b * inputs["A"] * sind(beta)
        dFiy_db = rho * cay * a * inputs["A"] * sind(beta)

        dPx_dU = dqx_dU * cdx * b
        dPy_dU = dqy_dU * cdy * a

        dPx_da = q * b * (dcdx_dARx * dARx_da) + dFix_da
        dPx_db = q * b * (dcdx_dARx * dARx_db) + dFix_db
        dPy_da = q * b * (dcdy_dARy * dARy_da) + dFiy_da
        dPy_db = q * b * (dcdy_dARy * dARy_db) + dFiy_db

        dPx_dA = rho * cax * a * b * cosd(beta)
        dPy_dA = rho * cay * a * b * cosd(beta)

        J["waveLoads_Px", "A"] = dPx_dA
        J["waveLoads_Py", "A"] = dPy_dA
        J["waveLoads_qdyn", "U"] = dq_dU
        J["waveLoads_pt", "U"] = dq_dU

# ___________________________________________#


class CylinderEnvironment(om.Group):
    def initialize(self):
        self.options.declare("wind", default="power")
        self.options.declare("nPoints")
        self.options.declare("water_flag", default=True)
        self.options.declare("member_shape", default="circular")

    def setup(self):
        nPoints = self.options["nPoints"]
        wind = self.options["wind"]
        water_flag = self.options["water_flag"]
        member_shape = self.options["member_shape"]

        self.set_input_defaults("z0", 0.0)
        self.set_input_defaults("cd_usr", -1.0*np.ones(nPoints))
        self.set_input_defaults("yaw", 0.0, units="deg")

        self.set_input_defaults("beta_wind", 0.0, units="deg")
        self.set_input_defaults("rho_air", 1.225, units="kg/m**3")
        self.set_input_defaults("mu_air", 1.81206e-5, units="kg/m/s")
        self.set_input_defaults("shearExp", 0.2)

        if water_flag:
            self.set_input_defaults("beta_wave", 0.0, units="deg")
            self.set_input_defaults("rho_water", 1025.0, units="kg/m**3")
            self.set_input_defaults("mu_water", 1.08e-3, units="kg/m/s")

        # Wind profile and loads
        promwind = ["Uref", "zref", "z", "z0"]
        if wind is None or wind.lower() in ["power", "powerwind", ""]:
            self.add_subsystem("wind", PowerWind(nPoints=nPoints), promotes=promwind + ["shearExp"])

        elif wind.lower() == "logwind":
            self.add_subsystem("wind", LogWind(nPoints=nPoints), promotes=promwind)

        else:
            raise ValueError("Unknown wind type, " + wind)
        
        if member_shape == "circular":
            self.add_subsystem(
                "windLoads",
                CylinderWindDrag(nPoints=nPoints),
                promotes=["cd_usr", "beta_wind", "rho_air", "mu_air", "z", "d"],
            )
        elif member_shape == "rectangular":
            self.add_subsystem(
                "windLoads",
                RectangularCylinderWindDrag(nPoints=nPoints),
                promotes=["cd_usr", "cdy_usr", "beta_wind", "rho_air", "mu_air", "z", "a", "b"],
            )

        # Wave profile and loads
        if water_flag:
            self.add_subsystem(
                "wave",
                LinearWaves(nPoints=nPoints),
                promotes=[
                    "z",
                    "Uc",
                    "Hsig_wave",
                    "Tsig_wave",
                    "rho_water",
                    ("z_floor", "water_depth"),
                    ("z_surface", "z0"),
                ],
            )

            if member_shape == "circular":
                self.add_subsystem(
                    "waveLoads",
                    CylinderWaveDrag(nPoints=nPoints),
                    promotes=["ca_usr", "cd_usr", "beta_wave", "rho_water", "mu_water", "z", "d"],
                )
            elif member_shape == "rectangular":
                self.add_subsystem(
                    "waveLoads",
                    RectangularCylinderWaveDrag(nPoints=nPoints),
                    promotes=["ca_usr", "cay_usr", "cd_usr", "cdy_usr", "beta_wave", "rho_water", "mu_water", "z", "a", "b"],
                ) 

        # Combine all loads
        self.add_subsystem(
            "distLoads", AeroHydroLoads(nPoints=nPoints), promotes=["Px", "Py", "Pz", "qdyn", "yaw", "z"]
        )

        # Connections
        self.connect("wind.U", "windLoads.U")
        if water_flag:
            self.connect("wave.U", "waveLoads.U")
            self.connect("wave.A", "waveLoads.A")
            self.connect("wave.p", "waveLoads.p")

        self.connect("windLoads.windLoads_Px", "distLoads.windLoads_Px")
        self.connect("windLoads.windLoads_Py", "distLoads.windLoads_Py")
        self.connect("windLoads.windLoads_Pz", "distLoads.windLoads_Pz")
        self.connect("windLoads.windLoads_qdyn", "distLoads.windLoads_qdyn")
        self.connect("windLoads.windLoads_beta", "distLoads.windLoads_beta")
        self.connect("windLoads.windLoads_z", "distLoads.windLoads_z")

        if water_flag:
            self.connect("waveLoads.waveLoads_Px", "distLoads.waveLoads_Px")
            self.connect("waveLoads.waveLoads_Py", "distLoads.waveLoads_Py")
            self.connect("waveLoads.waveLoads_Pz", "distLoads.waveLoads_Pz")
            self.connect("waveLoads.waveLoads_pt", "distLoads.waveLoads_qdyn")
            self.connect("waveLoads.waveLoads_beta", "distLoads.waveLoads_beta")
            self.connect("waveLoads.waveLoads_z", "distLoads.waveLoads_z")


def main():
    # initialize problem
    U = np.array([20.0, 25.0, 30.0])
    z = np.array([10.0, 30.0, 80.0])
    d = np.array([5.5, 4.0, 3.0])

    beta = np.array([45.0, 45.0, 45.0])
    rho = 1.225
    mu = 1.7934e-5
    # cd_usr = 0.7

    nPoints = len(z)

    prob = om.Problem(reports=False)

    root = prob.model = om.Group()

    root.add("p1", CylinderWindDrag(nPoints))

    prob.setup()

    prob["p1.U"] = U
    prob["p1.z"] = z
    prob["p1.d"] = d
    prob["p1.beta"] = beta
    prob["p1.rho"] = rho
    prob["p1.mu"] = mu
    # prob['p1.cd_usr'] = cd_usr

    # run
    prob.run_once()

    # out
    Re = prob["p1.rho"] * prob["p1.U"] * prob["p1.d"] / prob["p1.mu"]
    cd, dcd_dRe = cylinderDrag(Re)
    print(cd)
    import matplotlib.pyplot as plt

    plt.plot(prob["p1.windLoads_Px"], prob["p1.windLoads_z"])
    plt.plot(prob["p1.windLoads_Py"], prob["p1.windLoads_z"])
    plt.plot(prob["p1.windLoads_qdyn"], prob["p1.windLoads_z"])
    plt.show()


if __name__ == "__main__":
    main()
