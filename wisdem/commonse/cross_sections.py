from __future__ import print_function
import numpy as np
from wisdem.commonse.utilities import nodal2sectional
import openmdao.api as om


class CylindricalShellProperties(om.ExplicitComponent):
    """
    OpenMDAO wrapper for tube class to obtain cylindrical sheel properties.

    Parameters
    ----------
    d : numpy array[nFull], [m]
        tower diameter at corresponding locations
    t : numpy array[nFull-1], [m]
        shell thickness at corresponding locations

    Returns
    -------
    Az : numpy array[nFull-1], [m**2]
        cross-sectional area
    Asx : numpy array[nFull-1], [m**2]
        x shear area
    Asy : numpy array[nFull-1], [m**2]
        y shear area
    Jz : numpy array[nFull-1], [m**4]
        polar moment of inertia
    Ixx : numpy array[nFull-1], [m**4]
        area moment of inertia about x-axis
    Iyy : numpy array[nFull-1], [m**4]
        area moment of inertia about y-axis

    """

    def initialize(self):
        self.options.declare("nFull")

    def setup(self):
        nFull = self.options["nFull"]

        self.add_input("d", np.zeros(nFull), units="m")
        self.add_input("t", np.zeros(nFull - 1), units="m")

        self.add_output("Az", np.zeros(nFull - 1), units="m**2")
        self.add_output("Asx", np.zeros(nFull - 1), units="m**2")
        self.add_output("Asy", np.zeros(nFull - 1), units="m**2")
        self.add_output("Jz", np.zeros(nFull - 1), units="m**4")
        self.add_output("Ixx", np.zeros(nFull - 1), units="m**4")
        self.add_output("Iyy", np.zeros(nFull - 1), units="m**4")

        # Derivatives
        self.declare_partials("*", "*", method="fd", form="central", step=1e-6)

    def compute(self, inputs, outputs):
        d, _ = nodal2sectional(inputs["d"])
        tube = Tube(d, inputs["t"])

        outputs["Az"] = tube.Area
        outputs["Asx"] = tube.Asx
        outputs["Asy"] = tube.Asy
        outputs["Jz"] = tube.J0
        outputs["Ixx"] = tube.Jxx
        outputs["Iyy"] = tube.Jyy


class Tube:
    """The Tube Class contains functions to calculate properties of tubular circular cross-sections
    for structural analyses."""

    def __init__(self, D, t, Lgth=np.NaN, Kbuck=1.0):
        self.D = D
        self.t = t
        self.L = Lgth * np.ones(np.size(D))  # this makes sure we exapnd Lght if D,t, arrays
        self.Kbuck = Kbuck * np.ones(np.size(D))  # this makes sure we exapnd Kbuck if D,t, arrays

    @property
    def Area(self):  # Cross sectional area of tube
        return (self.D ** 2 - (self.D - 2 * self.t) ** 2) * np.pi / 4

    @property
    def derivArea(self):
        return {"D": 2 * np.pi / 4 * (self.D ** 2 - (self.D - 2 * self.t)) * (2 * self.D - 1), "t": 0}

    @property
    def Amid(self):  # mid-thickness inscribed area of tube (thin wall torsion calculation)
        return (self.D - self.t) ** 2 * np.pi / 4

    @property
    def Jxx(self):  # 2nd area moment of inertia w.r.t. x-x axis (Jxx=Jyy for tube)
        return (self.D ** 4 - (self.D - 2 * self.t) ** 4) * np.pi / 64

    @property
    def Jyy(self):  # 2nd area moment of inertia w.r.t. x-x axis (Jxx=Jyy for tube)
        return self.Jxx

    @property
    def J0(self):  # polar moment of inertia w.r.t. z-z axis (torsional)
        return 2.0 * self.Jxx

    @property
    def Asy(self):  # Shear Area for tubular cross-section
        Ri = self.D / 2 - self.t
        Ro = self.D / 2
        return self.Area / (1.124235 + 0.055610 * (Ri / Ro) + 1.097134 * (Ri / Ro) ** 2 - 0.630057 * (Ri / Ro) ** 3)

    @property
    def Asx(self):  # Shear Area for tubular cross-section
        return self.Asy

    @property
    def BdgMxx(self):  # Bending modulus for tubular cross-section
        return self.Jxx / (self.D / 2)

    @property
    def BdgMyy(self):  # Bending modulus for tubular cross-section =BdgMxx
        return self.Jyy / (self.D / 2)

    @property
    def TorsConst(self):  # Torsion shear constant for tubular cross-section
        return self.J0 / (self.D / 2)

    @property
    def S(self):  # Bending modulus for tubular cross-section
        return self.BdgMxx

    @property
    def C(self):  # Torsion shear constant for tubular cross-section
        return self.TorsConst

    @property
    def Rgyr(self):  # Radius of Gyration for circular tube
        return np.sqrt(self.Jxx / self.Area)

    @property
    def Klr(self):  # Klr buckling parameter
        return self.Kbuck * self.L / self.Rgyr


class IBeam:
    def __init__(self, L_flange, t_flange, H_web, t_web):
        self.Lf = L_flange
        self.tf = t_flange
        self.Hw = H_web
        self.tw = t_web
        self.H = H_web + 2 * t_flange

    @property
    def AreaFlange(self):  # Cross sectional area of tube
        return self.Lf * self.tf

    @property
    def AreaWeb(self):  # Cross sectional area of tube
        return self.Hw * self.tw

    @property
    def Area(self):  # Cross sectional area of tube
        return self.AreaWeb + 2 * self.AreaFlange

    @property
    def Iyy(self):  # 2nd area moment of inertia w.r.t. y-y axis running parallel to flange through CG
        return (self.Lf * self.H ** 3 - (self.Lf - self.tw) * self.Hw ** 3) / 12.0

    @property
    def Izz(self):  # 2nd area moment of inertia w.r.t. z-z running through center of web
        return (2 * self.tw * self.Lf ** 3 + self.Hw * self.tw ** 3) / 12.0

    @property
    def Jxx(self):  # polar moment of inertia w.r.t. z-z axis (torsional)
        return 2 * self.Lf * self.tf ** 3 + self.H * self.tw ** 3

    @property
    def Asy(self):  # Shear Area for tubular cross-section
        return 1.64 * self.Lf * self.tf

    @property
    def Asz(self):  # Shear Area for tubular cross-section
        return self.tw * self.H

    @property
    def BdgMyy(self):  # Bending modulus for tubular cross-section
        return 2 * self.Iyy / self.H

    @property
    def BdgMzz(self):  # Bending modulus for tubular cross-section =BdgMxx
        return 2 * self.Izz / self.Lf

    @property
    def TorsConst(self):  # Torsion shear constant for tubular cross-section
        return self.Jxx / (1.28 * self.tf)

    @property
    def Syy(self):  # Bending modulus for tubular cross-section
        return self.BdgMyy

    @property
    def Szz(self):  # Bending modulus for tubular cross-section
        return self.BdgMzz

    @property
    def C(self):  # Torsion shear constant for tubular cross-section
        return self.TorsConst

    @property
    def Rgyr(self):  # Radius of Gyration for circular tube
        return np.sqrt(self.Jxx / self.Area)

    @property
    def CG(self):  # Radius of Gyration for circular tube
        return 0.5 * self.Hw + self.tf
