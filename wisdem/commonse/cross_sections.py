from __future__ import print_function

import abc

import numpy as np


class CrossSectionBase(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def Area(self):
        pass

    @property
    @abc.abstractmethod
    def Ixx(self):  # 2nd area moment of inertia w.r.t. x-x axis
        pass

    @property
    @abc.abstractmethod
    def Iyy(self):  # 2nd area moment of inertia w.r.t. z-z axis
        pass

    @property
    @abc.abstractmethod
    def J0(self):  # polar moment of inertia w.r.t. z-z axis (torsional)
        pass

    @property
    @abc.abstractmethod
    def Asx(self):  # Shear Area
        pass

    @property
    @abc.abstractmethod
    def Asy(self):  # Shear Area
        pass

    @property
    @abc.abstractmethod
    def BdgMxx(self):  # Bending modulus
        pass

    @property
    @abc.abstractmethod
    def BdgMyy(self):  # Bending modulus
        pass

    @property
    @abc.abstractmethod
    def TorsConst(self):  # Torsion shear constant
        pass

    @property
    def Sx(self):  # Bending modulus
        return self.BdgMxx

    @property
    def Sy(self):  # Bending modulus
        return self.BdgMyy

    @property
    def C(self):  # Torsion shear constant
        return self.TorsConst


class Tube(CrossSectionBase):
    """The Tube Class contains functions to calculate properties of tubular circular cross-sections
    for structural analyses."""

    def __init__(self, D, t, L=np.NaN, Kbuck=1.0):
        self.D = D
        self.R = 0.5 * D
        self.t = t
        self.L = L * np.ones(np.size(D))
        self.Kbuck = Kbuck * np.ones(np.size(D))

    @property
    def Area(self):  # Cross sectional area of tube
        return (self.D ** 2 - (self.D - 2 * self.t) ** 2) * np.pi / 4.0

    @property
    def derivArea(self):
        return {"D": 2 * np.pi / 4 * (self.D ** 2 - (self.D - 2 * self.t)) * (2 * self.D - 1), "t": 0}

    @property
    def Amid(self):  # mid-thickness inscribed area of tube (thin wall torsion calculation)
        return (self.D - self.t) ** 2 * np.pi / 4.0

    @property
    def Ixx(self):  # 2nd area moment of inertia w.r.t. x-x axis (Iyy=Ixx for tube)
        return (self.D ** 4 - (self.D - 2 * self.t) ** 4) * np.pi / 64.0

    @property
    def Iyy(self):  # 2nd area moment of inertia w.r.t. z-z axis (Iyy=Ixx for tube)
        return self.Ixx

    @property
    def J0(self):  # polar moment of inertia w.r.t. z-z axis (torsional)
        return 2.0 * self.Ixx

    @property
    def Asx(self):  # Shear Area for tubular cross-section
        Ri = self.D / 2 - self.t
        Ro = self.R
        return self.Area / (1.124235 + 0.055610 * (Ri / Ro) + 1.097134 * (Ri / Ro) ** 2 - 0.630057 * (Ri / Ro) ** 3)

    @property
    def Asy(self):  # Shear Area for tubular cross-section
        return self.Asx

    @property
    def BdgMxx(self):  # Bending modulus for tubular cross-section
        return self.Ixx / self.R

    @property
    def BdgMyy(self):  # Bending modulus for tubular cross-section =BdgMxx
        return self.Iyy / self.R

    @property
    def TorsConst(self):  # Torsion shear constant for tubular cross-section
        return self.J0 / self.R

    @property
    def S(self):  # Bending modulus for tubular cross-section
        return self.Sx

    @property
    def Rgyr(self):  # Radius of Gyration for circular tube
        return np.sqrt(self.Ixx / self.Area)

    @property
    def Klr(self):  # Klr buckling parameter
        return self.Kbuck * self.L / self.Rgyr


class IBeam(CrossSectionBase):
    def __init__(self, L_flange, t_flange, H_web, t_web, L=np.nan):
        self.Lf = L_flange
        self.tf = t_flange
        self.Hw = H_web
        self.tw = t_web
        self.H = H_web + 2 * t_flange
        self.L = L * np.ones(np.size(L_flange))

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
    def Ixx(self):  # 2nd area moment of inertia w.r.t. x-x axis running parallel to flange through CG
        return (self.Lf * self.H ** 3 - (self.Lf - self.tw) * self.Hw ** 3) / 12.0

    @property
    def Iyy(self):  # 2nd area moment of inertia w.r.t. z-z running through center of web
        return (2 * self.tw * self.Lf ** 3 + self.Hw * self.tw ** 3) / 12.0

    @property
    def J0(self):  # polar moment of inertia w.r.t. z-z axis (torsional)
        return 2 * self.Lf * self.tf ** 3 + self.H * self.tw ** 3

    @property
    def Asx(self):  # Shear Area for tubular cross-section
        return 1.64 * self.Lf * self.tf

    @property
    def Asy(self):  # Shear Area for tubular cross-section
        return self.tw * self.H

    @property
    def BdgMxx(self):  # Bending modulus for tubular cross-section
        return 2 * self.Ixx / self.H

    @property
    def BdgMyy(self):  # Bending modulus for tubular cross-section =BdgMxx
        return 2 * self.Iyy / self.Lf

    @property
    def TorsConst(self):  # Torsion shear constant for tubular cross-section
        return self.J0 / (1.28 * self.tf)

    @property
    def CG(self):
        return 0.5 * self.Hw + self.tf


class Square(CrossSectionBase):
    def __init__(self, a, t, L=np.nan):
        self.a = a
        self.t = t
        self.L = L * np.ones(np.size(a))

    @property
    def Area(self):  # Cross sectional area of tube
        return self.a ** 2 - (self.a - 2 * self.t) ** 2

    @property
    def Ixx(self):  # 2nd area moment of inertia w.r.t. x-x axis running parallel to flange through CG
        return (self.a ** 4 - (self.a - 2 * self.t) ** 4) / 12.0

    @property
    def Iyy(self):  # 2nd area moment of inertia w.r.t. z-z running through center of web
        return self.Ixx

    @property
    def J0(self):  # polar moment of inertia w.r.t. z-z axis (torsional)
        return self.t * (self.a - 2 * self.t) ** 3

    @property
    def Asx(self):  # Shear Area for tubular cross-section
        return self.Area / (2.39573 - 0.25009 * (self.t / self.a) - 7.89675 * (self.t / self.a) ** 2)

    @property
    def Asy(self):  # Shear Area for tubular cross-section
        return self.Asx

    @property
    def BdgMxx(self):  # Bending modulus for tubular cross-section
        return 2 * self.Ixx / self.a

    @property
    def BdgMyy(self):  # Bending modulus for tubular cross-section =BdgMxx
        return self.BdgMxx

    @property
    def TorsConst(self):  # Torsion shear constant for tubular cross-section
        return 2 * self.t * (self.a - self.t) ** 2

    @property
    def S(self):  # Bending modulus for tubular cross-section
        return self.Sx


class Rectangle(CrossSectionBase):
    def __init__(self, a, b, t, L=np.nan):
        self.a = a
        self.b = b
        self.t = t
        self.L = L * np.ones(np.size(a))

    @property
    def Area(self):  # Cross sectional area of tube
        return self.a * self.b - (self.a - 2 * self.t) * (self.b - 2 * self.t)

    @property
    def Ixx(self):  # 2nd area moment of inertia w.r.t. x-x axis running parallel to flange through CG
        return (self.a * self.b ** 3 - (self.a - 2 * self.t) * (self.b - 2 * self.t) * 3) / 12.0

    @property
    def Iyy(self):  # 2nd area moment of inertia w.r.t. z-z running through center of web
        return (self.b * self.a ** 3 - (self.b - 2 * self.t) * (self.a - 2 * self.t) * 3) / 12.0

    @property
    def J0(self):  # polar moment of inertia w.r.t. z-z axis (torsional)
        return 2 * self.t * (self.a - self.t) ** 2 * (self.b - self.t) ** 2 / (self.a + self.b - 2 * self.t)

    @property
    def Asx(self):  # Shear Area for tubular cross-section
        if self.a > self.b:
            return self.Area / (
                0.93498
                - 1.28084 * (self.t / self.b)
                + 1.36441 * (self.b / self.a)
                + 0.00295 * (self.a / self.b) ** 2
                + 0.25797 * (self.t * self.a / self.b ** 2)
            )
        else:
            return self.Area / (
                1.63544
                - 8.34935 * (self.t / self.a)
                + 0.60125 * (self.b / self.a)
                + 0.41403 * (self.b / self.a) ** 2
                + 4.95373 * (self.t * self.b / self.a ** 2)
            )

    @property
    def Asy(self):  # Shear Area for tubular cross-section
        if self.a > self.b:
            return self.Area / (
                1.63544
                - 8.34935 * (self.t / self.b)
                + 0.60125 * (self.a / self.b)
                + 0.41403 * (self.a / self.b) ** 2
                + 4.95373 * (self.t * self.a / self.b ** 2)
            )
        else:
            return self.Area / (
                0.93498
                - 1.28084 * (self.t / self.a)
                + 1.36441 * (self.a / self.b)
                + 0.00295 * (self.b / self.a) ** 2
                + 0.25797 * (self.t * self.b / self.a ** 2)
            )

    @property
    def BdgMxx(self):  # Bending modulus for tubular cross-section
        return 2 * self.Ixx / self.b

    @property
    def BdgMyy(self):  # Bending modulus for tubular cross-section =BdgMxx
        return 2 * self.Iyy / self.a

    @property
    def TorsConst(self):  # Torsion shear constant for tubular cross-section
        return 2 * self.t * (self.a - self.t) * (self.b - self.t)
