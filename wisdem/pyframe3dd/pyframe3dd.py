#!/usr/bin/env python
# encoding: utf-8
"""
frame.py

Created by Andrew Ning on 2013-11-01.
Copyright (c) NREL. All rights reserved.
"""

from __future__ import print_function

import os
import math
from sys import platform
from ctypes import POINTER, Structure, c_int, pointer, c_double
from collections import namedtuple
from distutils.sysconfig import get_config_var

import numpy as np

libext = get_config_var("EXT_SUFFIX")
if libext is None or libext == "":
    if platform == "linux" or platform == "linux2":
        libext = ".so"
    elif platform == "darwin":
        # libext = '.dyld'
        libext = ".so"
    elif platform == "win32":
        # libext = '.dll'
        libext = ".pyd"
    elif platform == "cygwin":
        libext = ".dll"

libname = "_pyframe3dd" + libext

c_int_p = POINTER(c_int)
c_double_p = POINTER(c_double)


def ip(x):
    return x.ctypes.data_as(c_int_p)


def dp(x):
    return x.ctypes.data_as(c_double_p)


# --------------
# General Inputs
# --------------


class C_Nodes(Structure):
    _fields_ = [
        ("nN", c_int),
        ("N", c_int_p),
        ("x", c_double_p),
        ("y", c_double_p),
        ("z", c_double_p),
        ("r", c_double_p),
    ]


class C_Reactions(Structure):
    _fields_ = [
        ("nK", c_int),
        ("N", c_int_p),
        ("Kx", c_double_p),
        ("Ky", c_double_p),
        ("Kz", c_double_p),
        ("Ktx", c_double_p),
        ("Kty", c_double_p),
        ("Ktz", c_double_p),
        ("rigid", c_double),
    ]


class C_Elements(Structure):
    _fields_ = [
        ("nE", c_int),
        ("EL", c_int_p),
        ("N1", c_int_p),
        ("N2", c_int_p),
        ("Ax", c_double_p),
        ("Asy", c_double_p),
        ("Asz", c_double_p),
        ("Jx", c_double_p),
        ("Iy", c_double_p),
        ("Iz", c_double_p),
        ("E", c_double_p),
        ("G", c_double_p),
        ("roll", c_double_p),
        ("density", c_double_p),
    ]


class C_OtherElementData(Structure):
    _fields_ = [("shear", c_int), ("geom", c_int), ("exagg_static", c_double), ("dx", c_double)]


# --------------
# Load Inputs
# --------------


class C_PointLoads(Structure):
    _fields_ = [
        ("nF", c_int),
        ("N", c_int_p),
        ("Fx", c_double_p),
        ("Fy", c_double_p),
        ("Fz", c_double_p),
        ("Mxx", c_double_p),
        ("Myy", c_double_p),
        ("Mzz", c_double_p),
    ]


class C_UniformLoads(Structure):
    _fields_ = [("nU", c_int), ("EL", c_int_p), ("Ux", c_double_p), ("Uy", c_double_p), ("Uz", c_double_p)]


class C_TrapezoidalLoads(Structure):
    _fields_ = [
        ("nW", c_int),
        ("EL", c_int_p),
        ("xx1", c_double_p),
        ("xx2", c_double_p),
        ("wx1", c_double_p),
        ("wx2", c_double_p),
        ("xy1", c_double_p),
        ("xy2", c_double_p),
        ("wy1", c_double_p),
        ("wy2", c_double_p),
        ("xz1", c_double_p),
        ("xz2", c_double_p),
        ("wz1", c_double_p),
        ("wz2", c_double_p),
    ]


class C_ElementLoads(Structure):
    _fields_ = [
        ("nP", c_int),
        ("EL", c_int_p),
        ("Px", c_double_p),
        ("Py", c_double_p),
        ("Pz", c_double_p),
        ("x", c_double_p),
    ]


class C_TemperatureLoads(Structure):
    _fields_ = [
        ("nT", c_int),
        ("EL", c_int_p),
        ("a", c_double_p),
        ("hy", c_double_p),
        ("hz", c_double_p),
        ("Typ", c_double_p),
        ("Tym", c_double_p),
        ("Tzp", c_double_p),
        ("Tzm", c_double_p),
    ]


class C_PrescribedDisplacements(Structure):
    _fields_ = [
        ("nD", c_int),
        ("N", c_int_p),
        ("Dx", c_double_p),
        ("Dy", c_double_p),
        ("Dz", c_double_p),
        ("Dxx", c_double_p),
        ("Dyy", c_double_p),
        ("Dzz", c_double_p),
    ]


class C_LoadCase(Structure):
    _fields_ = [
        ("gx", c_double),
        ("gy", c_double),
        ("gz", c_double),
        ("pointLoads", C_PointLoads),
        ("uniformLoads", C_UniformLoads),
        ("trapezoidalLoads", C_TrapezoidalLoads),
        ("elementLoads", C_ElementLoads),
        ("temperatureLoads", C_TemperatureLoads),
        ("prescribedDisplacements", C_PrescribedDisplacements),
    ]


# --------------
# Dynamic Inputs
# --------------


class C_DynamicData(Structure):
    _fields_ = [
        ("nM", c_int),
        ("Mmethod", c_int),
        ("lump", c_int),
        ("tol", c_double),
        ("shift", c_double),
        ("exagg_modal", c_double),
    ]


class C_ExtraInertia(Structure):
    _fields_ = [
        ("nI", c_int),
        ("N", c_int_p),
        ("EMs", c_double_p),
        ("EMx", c_double_p),
        ("EMy", c_double_p),
        ("EMz", c_double_p),
        ("EMxy", c_double_p),
        ("EMxz", c_double_p),
        ("EMyz", c_double_p),
        ("rhox", c_double_p),
        ("rhoy", c_double_p),
        ("rhoz", c_double_p),
    ]


class C_ExtraMass(Structure):
    _fields_ = [("nX", c_int), ("EL", c_int_p), ("EMs", c_double_p)]


class C_Condensation(Structure):
    _fields_ = [
        ("Cmethod", c_int),
        ("nC", c_int),
        ("N", c_int_p),
        ("cx", c_double_p),
        ("cy", c_double_p),
        ("cz", c_double_p),
        ("cxx", c_double_p),
        ("cyy", c_double_p),
        ("czz", c_double_p),
        ("m", c_int_p),
    ]


# --------------
# Static Data Outputs
# --------------


class C_Displacements(Structure):
    _fields_ = [
        ("node", c_int_p),
        ("x", c_double_p),
        ("y", c_double_p),
        ("z", c_double_p),
        ("xrot", c_double_p),
        ("yrot", c_double_p),
        ("zrot", c_double_p),
    ]


class C_Forces(Structure):
    _fields_ = [
        ("element", c_int_p),
        ("node", c_int_p),
        ("Nx", c_double_p),
        ("Vy", c_double_p),
        ("Vz", c_double_p),
        ("Txx", c_double_p),
        ("Myy", c_double_p),
        ("Mzz", c_double_p),
    ]


class C_ReactionForces(Structure):
    _fields_ = [
        ("node", c_int_p),
        ("Fx", c_double_p),
        ("Fy", c_double_p),
        ("Fz", c_double_p),
        ("Mxx", c_double_p),
        ("Myy", c_double_p),
        ("Mzz", c_double_p),
    ]


# --------------
# Internal Force Outputs
# --------------


class C_InternalForces(Structure):
    _fields_ = [
        ("x", c_double_p),
        ("Nx", c_double_p),
        ("Vy", c_double_p),
        ("Vz", c_double_p),
        ("Tx", c_double_p),
        ("My", c_double_p),
        ("Mz", c_double_p),
        ("Dx", c_double_p),
        ("Dy", c_double_p),
        ("Dz", c_double_p),
        ("Rx", c_double_p),
    ]


# --------------
# Modal Outputs
# --------------


class C_MassResults(Structure):
    _fields_ = [
        ("total_mass", c_double_p),
        ("struct_mass", c_double_p),
        ("N", c_int_p),
        ("xmass", c_double_p),
        ("ymass", c_double_p),
        ("zmass", c_double_p),
        ("xinrta", c_double_p),
        ("yinrta", c_double_p),
        ("zinrta", c_double_p),
    ]


class C_ModalResults(Structure):
    _fields_ = [
        ("freq", c_double_p),
        ("xmpf", c_double_p),
        ("ympf", c_double_p),
        ("zmpf", c_double_p),
        ("N", c_int_p),
        ("xdsp", c_double_p),
        ("ydsp", c_double_p),
        ("zdsp", c_double_p),
        ("xrot", c_double_p),
        ("yrot", c_double_p),
        ("zrot", c_double_p),
    ]


# inputs

NodeData = namedtuple("NodeData", ["node", "x", "y", "z", "r"])
ReactionData = namedtuple("ReactionData", ["node", "Kx", "Ky", "Kz", "Ktx", "Kty", "Ktz", "rigid"])
ElementData = namedtuple(
    "ElementData", ["element", "N1", "N2", "Ax", "Asy", "Asz", "Jx", "Iy", "Iz", "E", "G", "roll", "density"]
)
Options = namedtuple("Options", ["shear", "geom", "dx"])


# outputs

NodeDisplacements = namedtuple("NodeDisplacements", ["node", "dx", "dy", "dz", "dxrot", "dyrot", "dzrot"])
ElementEndForces = namedtuple("ElementEndForces", ["element", "node", "Nx", "Vy", "Vz", "Txx", "Myy", "Mzz"])
NodeReactions = namedtuple("NodeReactions", ["node", "Fx", "Fy", "Fz", "Mxx", "Myy", "Mzz"])
InternalForces = namedtuple("InternalForces", ["x", "Nx", "Vy", "Vz", "Tx", "My", "Mz", "Dx", "Dy", "Dz", "Rx"])
NodeMasses = namedtuple(
    "NodeMasses", ["total_mass", "struct_mass", "node", "xmass", "ymass", "zmass", "xinrta", "yinrta", "zinrta"]
)
Modes = namedtuple("Modes", ["freq", "xmpf", "ympf", "zmpf", "node", "xdsp", "ydsp", "zdsp", "xrot", "yrot", "zrot"])


class Frame(object):
    def __init__(self, nodes, reactions, elements, options):
        """docstring"""

        self.nodes = nodes
        self.elements = elements
        self.options = options

        # convert to C int size (not longs) and copy to prevent releasing (b/c address space is shared by c)

        # nodes
        self.nnode = np.array(nodes.node).astype(np.int32).flatten()
        self.nx = np.copy(nodes.x).flatten()
        self.ny = np.copy(nodes.y).flatten()
        self.nz = np.copy(nodes.z).flatten()
        self.nr = np.copy(nodes.r).flatten()

        self.set_reactions(reactions)

        # elements
        self.eelement = np.array(elements.element).astype(np.int32).flatten()
        self.eN1 = np.array(elements.N1).astype(np.int32).flatten()
        self.eN2 = np.array(elements.N2).astype(np.int32).flatten()
        self.eAx = np.copy(elements.Ax).flatten()
        self.eAsy = np.copy(elements.Asy).flatten()
        self.eAsz = np.copy(elements.Asz).flatten()
        self.eJx = np.copy(elements.Jx).flatten()
        self.eIy = np.copy(elements.Iy).flatten()
        self.eIz = np.copy(elements.Iz).flatten()
        self.eE = np.copy(elements.E).flatten()
        self.eG = np.copy(elements.G).flatten()
        self.eroll = np.copy(elements.roll).flatten()
        self.edensity = np.copy(elements.density).flatten()

        # Compute length of elements
        self.eL = np.sqrt(
            (self.nx[self.eN2 - 1] - self.nx[self.eN1 - 1]) ** 2.0
            + (self.ny[self.eN2 - 1] - self.ny[self.eN1 - 1]) ** 2.0
            + (self.nz[self.eN2 - 1] - self.nz[self.eN1 - 1]) ** 2.0
        )

        # create c objects
        self.c_nodes = C_Nodes(len(self.nnode), ip(self.nnode), dp(self.nx), dp(self.ny), dp(self.nz), dp(self.nr))

        self.c_elements = C_Elements(
            len(self.eelement),
            ip(self.eelement),
            ip(self.eN1),
            ip(self.eN2),
            dp(self.eAx),
            dp(self.eAsy),
            dp(self.eAsz),
            dp(self.eJx),
            dp(self.eIy),
            dp(self.eIz),
            dp(self.eE),
            dp(self.eG),
            dp(self.eroll),
            dp(self.edensity),
        )

        # options
        exagg_static = 1.0  # not used
        self.c_other = C_OtherElementData(options.shear, options.geom, exagg_static, options.dx)

        # leave off dynamics by default
        self.nM = 0  # number of desired dynamic modes of vibration (below only necessary if nM > 0)
        self.Mmethod = 1  # 1: subspace Jacobi     2: Stodola
        self.lump = 0  # 0: consistent mass ... 1: lumped mass matrix
        self.tol = 1e-9  # mode shape tolerance
        self.shift = 0.0  # shift value ... for unrestrained structures

        # create list for load cases
        self.loadCases = []

        # initialize extra mass data
        i = np.array([], dtype=np.int32)
        d = np.array([])
        self.changeExtraNodeMass(i, d, d, d, d, d, d, d, d, d, d, False)
        self.changeExtraElementMass(i, d, False)
        self.changeCondensationData(0, i, d, d, d, d, d, d, i)

        # load c module
        mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
        try:
            self._pyframe3dd = np.ctypeslib.load_library(libname, mydir)
        except:
            mydir = os.path.abspath(os.path.dirname(mydir))
            self._pyframe3dd = np.ctypeslib.load_library(libname, mydir)

        self._pyframe3dd.run.argtypes = [
            POINTER(C_Nodes),
            POINTER(C_Reactions),
            POINTER(C_Elements),
            POINTER(C_OtherElementData),
            c_int,
            POINTER(C_LoadCase),
            POINTER(C_DynamicData),
            POINTER(C_ExtraInertia),
            POINTER(C_ExtraMass),
            POINTER(C_Condensation),
            POINTER(C_Displacements),
            POINTER(C_Forces),
            POINTER(C_ReactionForces),
            POINTER(POINTER(C_InternalForces)),
            POINTER(C_MassResults),
            POINTER(C_ModalResults),
        ]

        self._pyframe3dd.run.restype = c_int

    def set_reactions(self, reactions):
        # reactions
        self.reactions = reactions
        if len(reactions.node) == 0:
            self.rnode = np.array([]).astype(np.int32)
            self.rKx = self.rKy = self.rKz = self.rKtx = self.rKty = self.rKtz = np.array([]).astype(np.float64)
            rigid = 1
        else:
            self.rnode = np.array(reactions.node).astype(np.int32).flatten()
            # convert rather than copy to allow old syntax of integers
            self.rKx = np.array(reactions.Kx).astype(np.float64).flatten()
            self.rKy = np.array(reactions.Ky).astype(np.float64).flatten()
            self.rKz = np.array(reactions.Kz).astype(np.float64).flatten()
            self.rKtx = np.array(reactions.Ktx).astype(np.float64).flatten()
            self.rKty = np.array(reactions.Kty).astype(np.float64).flatten()
            self.rKtz = np.array(reactions.Ktz).astype(np.float64).flatten()
            rigid = reactions.rigid

        self.c_reactions = C_Reactions(
            len(self.rnode),
            ip(self.rnode),
            dp(self.rKx),
            dp(self.rKy),
            dp(self.rKz),
            dp(self.rKtx),
            dp(self.rKty),
            dp(self.rKtz),
            rigid,
        )

    def addLoadCase(self, loadCase):
        self.loadCases.append(loadCase)

    def clearLoadCases(self):
        self.loadCases = []

    def changeExtraNodeMass(self, node, mass, Ixx, Iyy, Izz, Ixy, Ixz, Iyz, rhox, rhoy, rhoz, addGravityLoad):

        self.ENMnode = np.array(node).astype(np.int32).flatten()
        nnode = len(self.ENMnode)
        self.ENMmass = np.copy(mass).flatten()
        self.ENMIxx = np.copy(Ixx).flatten()
        self.ENMIyy = np.copy(Iyy).flatten()
        self.ENMIzz = np.copy(Izz).flatten()
        self.ENMIxy = np.copy(Ixy).flatten()
        self.ENMIxz = np.copy(Ixz).flatten()
        self.ENMIyz = np.copy(Iyz).flatten()
        self.ENMrhox = np.copy(rhox).flatten()
        self.ENMrhoy = np.copy(rhoy).flatten()
        self.ENMrhoz = np.copy(rhoz).flatten()
        self.addGravityLoadForExtraNodeMass = (
            [addGravityLoad] * nnode if isinstance(addGravityLoad, bool) else list(addGravityLoad)
        )

        self.c_extraInertia = C_ExtraInertia(
            len(self.ENMnode),
            ip(self.ENMnode),
            dp(self.ENMmass),
            dp(self.ENMIxx),
            dp(self.ENMIyy),
            dp(self.ENMIzz),
            dp(self.ENMIxy),
            dp(self.ENMIxz),
            dp(self.ENMIyz),
            dp(self.ENMrhox),
            dp(self.ENMrhoy),
            dp(self.ENMrhoz),
        )

    def changeExtraElementMass(self, element, mass, addGravityLoad):

        self.EEMelement = np.array(element).astype(np.int32)
        nelem = len(self.EEMelement)
        self.EEMmass = np.copy(mass).flatten()
        self.addGravityLoadForExtraElementMass = (
            [addGravityLoad] * nelem if isinstance(addGravityLoad, bool) else list(addGravityLoad)
        )

        self.c_extraMass = C_ExtraMass(len(self.EEMelement), ip(self.EEMelement), dp(self.EEMmass))

    def changeCondensationData(self, Cmethod, N, cx, cy, cz, cxx, cyy, czz, m):
        # I don't think this is actually used in Frame3DD anyway

        self.NC = np.array(N).astype(np.int32).flatten()
        self.cx = np.copy(cx).flatten()
        self.cy = np.copy(cy).flatten()
        self.cz = np.copy(cz).flatten()
        self.cxx = np.copy(cxx).flatten()
        self.cyy = np.copy(cyy).flatten()
        self.czz = np.copy(czz).flatten()
        self.mC = np.array(m).astype(np.int32).flatten()

        self.c_condensation = C_Condensation(
            Cmethod,
            len(N),
            ip(self.NC),
            dp(self.cx),
            dp(self.cy),
            dp(self.cz),
            dp(self.cxx),
            dp(self.cyy),
            dp(self.czz),
            ip(self.mC),
        )

    def enableDynamics(self, nM, Mmethod, lump, tol, shift):

        self.nM = nM
        self.Mmethod = Mmethod
        self.lump = lump
        self.tol = tol
        self.shift = shift

    def __addGravityToExtraMass(self):

        if np.any(self.addGravityLoadForExtraNodeMass):

            # need to save all in memory
            nLC = len(self.loadCases)
            self.PLN = [0] * nLC
            self.PLFx = [0] * nLC
            self.PLFy = [0] * nLC
            self.PLFz = [0] * nLC
            self.PLMx = [0] * nLC
            self.PLMy = [0] * nLC
            self.PLMz = [0] * nLC

            for icase, lc in enumerate(self.loadCases):

                gx = lc.gx
                gy = lc.gy
                gz = lc.gz

                # copy data over for this case
                self.PLN[icase] = np.copy(lc.NF)
                self.PLFx[icase] = np.copy(lc.Fx)
                self.PLFy[icase] = np.copy(lc.Fy)
                self.PLFz[icase] = np.copy(lc.Fz)
                self.PLMx[icase] = np.copy(lc.Mxx)
                self.PLMy[icase] = np.copy(lc.Myy)
                self.PLMz[icase] = np.copy(lc.Mzz)

                for iextra in range(len(self.ENMnode)):
                    if not self.addGravityLoadForExtraNodeMass[iextra]:
                        continue
                    Nm = self.ENMnode[iextra]
                    mass = self.ENMmass[iextra]
                    x = self.ENMrhox[iextra]
                    y = self.ENMrhoy[iextra]
                    z = self.ENMrhoz[iextra]

                    # check if a point load already exists for this node
                    if Nm in self.PLN[icase]:
                        idx = np.where(self.PLN[icase] == Nm)[0]

                        # if so just add it
                        self.PLFx[icase][idx] += mass * gx
                        self.PLFy[icase][idx] += mass * gy
                        self.PLFz[icase][idx] += mass * gz
                        self.PLMx[icase][idx] += mass * (y * gz - z * gy)
                        self.PLMy[icase][idx] += mass * (z * gx - x * gz)
                        self.PLMz[icase][idx] += mass * (x * gy - y * gx)

                    else:
                        # otherwise append to end
                        self.PLN[icase] = np.concatenate([self.PLN[icase], [Nm]])
                        self.PLFx[icase] = np.concatenate([self.PLFx[icase], [mass * gx]])
                        self.PLFy[icase] = np.concatenate([self.PLFy[icase], [mass * gy]])
                        self.PLFz[icase] = np.concatenate([self.PLFz[icase], [mass * gz]])
                        self.PLMx[icase] = np.concatenate([self.PLMx[icase], [mass * (y * gz - z * gy)]])
                        self.PLMy[icase] = np.concatenate([self.PLMy[icase], [mass * (z * gx - x * gz)]])
                        self.PLMz[icase] = np.concatenate([self.PLMz[icase], [mass * (x * gy - y * gx)]])

                lc.pL = C_PointLoads(
                    len(self.PLN[icase]),
                    ip(self.PLN[icase]),
                    dp(self.PLFx[icase]),
                    dp(self.PLFy[icase]),
                    dp(self.PLFz[icase]),
                    dp(self.PLMx[icase]),
                    dp(self.PLMy[icase]),
                    dp(self.PLMz[icase]),
                )

        if np.any(self.addGravityLoadForExtraElementMass):

            L = self.eL

            # add to interior point load

            # save all data in memory
            nLC = len(self.loadCases)
            self.IPLE = np.array([[] * nLC])
            self.IPLPx = np.array([[] * nLC])
            self.IPLPy = np.array([[] * nLC])
            self.IPLPz = np.array([[] * nLC])
            self.IPLxE = np.array([[] * nLC])

            for icase, lc in enumerate(self.loadCases):

                gx = lc.gx
                gy = lc.gy
                gz = lc.gz

                # iterate through additional mass
                for iextra in range(len(self.EEMelement)):
                    if not self.addGravityLoadForExtraElementMass[iextra]:
                        continue

                    element = self.EEMelement[iextra]
                    mass = self.EEMmass[iextra]
                    LE = L[element - 1]

                    # check whether an element load already exists for this element
                    if element in self.IPLE[icase]:
                        idx = np.where(self.IPLE[icase] == element)[0]

                        # if so we just add the weight loads
                        self.IPLPx[icase][idx] += mass * gx
                        self.IPLPy[icase][idx] += mass * gy
                        self.IPLPz[icase][idx] += mass * gz
                        # TODO: assumes xE does not change

                    else:
                        # otherwise append to the end
                        self.IPLE[icase] = np.append(self.IPLE[icase], element)
                        self.IPLPx[icase] = np.append(self.IPLPx[icase], mass * gx)
                        self.IPLPy[icase] = np.append(self.IPLPy[icase], mass * gy)
                        self.IPLPz[icase] = np.append(self.IPLPz[icase], mass * gz)
                        self.IPLxE[icase] = np.append(self.IPLxE[icase], 0.5 * LE)

                # self.IPLE = np.concatenate([lc.ELE, element])
                # self.IPLPx = np.concatenate([lc.Px, mass*gx])
                # self.IPLPy = np.concatenate([lc.Py, mass*gy])
                # self.IPLPz = np.concatenate([lc.Pz, mass*gz])
                # self.IPLxE = np.concatenate([lc.xE, 0.5*LE])

                lc.eL = C_ElementLoads(
                    len(self.IPLE[icase]),
                    ip(self.IPLE[icase]),
                    dp(self.IPLPx[icase]),
                    dp(self.IPLPy[icase]),
                    dp(self.IPLPz[icase]),
                    dp(self.IPLxE[icase]),
                )

    def run(self, nanokay=False):

        nCases = len(self.loadCases)  # number of load cases
        nN = len(self.nodes.node)  # number of nodes
        nE = len(self.elements.element)  # number of elements
        nR = len(self.reactions.node)  # number of reactions
        nM = self.nM  # number of modes

        if nCases == 0:
            print("error: must have at least 1 load case")
            return

        self.__addGravityToExtraMass()

        # initialize output arrays

        dout = NodeDisplacements(
            np.zeros((nCases, nN), dtype=np.int32),
            np.zeros((nCases, nN)),
            np.zeros((nCases, nN)),
            np.zeros((nCases, nN)),
            np.zeros((nCases, nN)),
            np.zeros((nCases, nN)),
            np.zeros((nCases, nN)),
        )
        fout = ElementEndForces(
            np.zeros((nCases, 2 * nE), dtype=np.int32),
            np.zeros((nCases, 2 * nE), dtype=np.int32),
            np.zeros((nCases, 2 * nE)),
            np.zeros((nCases, 2 * nE)),
            np.zeros((nCases, 2 * nE)),
            np.zeros((nCases, 2 * nE)),
            np.zeros((nCases, 2 * nE)),
            np.zeros((nCases, 2 * nE)),
        )
        rout = NodeReactions(
            np.zeros((nCases, nR), dtype=np.int32),
            np.zeros((nCases, nR)),
            np.zeros((nCases, nR)),
            np.zeros((nCases, nR)),
            np.zeros((nCases, nR)),
            np.zeros((nCases, nR)),
            np.zeros((nCases, nR)),
        )

        dx = self.options.dx

        ifout = [0] * nE
        for i in range(nE):
            L = self.eL[i]

            nIF = int(max(math.floor(L / dx), 1)) + 1

            ifout[i] = InternalForces(
                np.zeros((nCases, nIF)),
                np.zeros((nCases, nIF)),
                np.zeros((nCases, nIF)),
                np.zeros((nCases, nIF)),
                np.zeros((nCases, nIF)),
                np.zeros((nCases, nIF)),
                np.zeros((nCases, nIF)),
                np.zeros((nCases, nIF)),
                np.zeros((nCases, nIF)),
                np.zeros((nCases, nIF)),
                np.zeros((nCases, nIF)),
            )

        mout = NodeMasses(
            0.0,
            0.0,
            np.zeros(nN, dtype=np.int32),
            np.zeros(nN),
            np.zeros(nN),
            np.zeros(nN),
            np.zeros(nN),
            np.zeros(nN),
            np.zeros(nN),
        )
        modalout = Modes(
            np.zeros(nM),
            np.zeros(nM),
            np.zeros(nM),
            np.zeros(nM),
            np.zeros((nM, nN), dtype=np.int32),
            np.zeros((nM, nN)),
            np.zeros((nM, nN)),
            np.zeros((nM, nN)),
            np.zeros((nM, nN)),
            np.zeros((nM, nN)),
            np.zeros((nM, nN)),
        )

        # create c structs

        c_loadcases = (C_LoadCase * nCases)()
        c_disp = (C_Displacements * nCases)()
        c_forces = (C_Forces * nCases)()
        c_reactions = (C_ReactionForces * nCases)()
        c_internalForces = (POINTER(C_InternalForces) * nCases)()

        for i in range(nCases):
            lci = self.loadCases[i]
            c_loadcases[i] = C_LoadCase(lci.gx, lci.gy, lci.gz, lci.pL, lci.uL, lci.tL, lci.eL, lci.tempL, lci.pD)
            c_disp[i] = C_Displacements(
                ip(dout.node[i, :]),
                dp(dout.dx[i, :]),
                dp(dout.dy[i, :]),
                dp(dout.dz[i, :]),
                dp(dout.dxrot[i, :]),
                dp(dout.dyrot[i, :]),
                dp(dout.dzrot[i, :]),
            )
            c_forces[i] = C_Forces(
                ip(fout.element[i, :]),
                ip(fout.node[i, :]),
                dp(fout.Nx[i, :]),
                dp(fout.Vy[i, :]),
                dp(fout.Vz[i, :]),
                dp(fout.Txx[i, :]),
                dp(fout.Myy[i, :]),
                dp(fout.Mzz[i, :]),
            )
            c_reactions[i] = C_ReactionForces(
                ip(rout.node[i, :]),
                dp(rout.Fx[i, :]),
                dp(rout.Fy[i, :]),
                dp(rout.Fz[i, :]),
                dp(rout.Mxx[i, :]),
                dp(rout.Myy[i, :]),
                dp(rout.Mzz[i, :]),
            )

            c_internalForces[i] = (C_InternalForces * nE)()
            for j in range(nE):
                (c_internalForces[i])[j] = C_InternalForces(
                    dp(ifout[j].x[i, :]),
                    dp(ifout[j].Nx[i, :]),
                    dp(ifout[j].Vy[i, :]),
                    dp(ifout[j].Vz[i, :]),
                    dp(ifout[j].Tx[i, :]),
                    dp(ifout[j].My[i, :]),
                    dp(ifout[j].Mz[i, :]),
                    dp(ifout[j].Dx[i, :]),
                    dp(ifout[j].Dy[i, :]),
                    dp(ifout[j].Dz[i, :]),
                    dp(ifout[j].Rx[i, :]),
                )

        total_mass = c_double()
        struct_mass = c_double()

        c_massResults = C_MassResults(
            pointer(total_mass),
            pointer(struct_mass),
            ip(mout.node),
            dp(mout.xmass),
            dp(mout.ymass),
            dp(mout.zmass),
            dp(mout.xinrta),
            dp(mout.yinrta),
            dp(mout.zinrta),
        )

        c_modalResults = (C_ModalResults * nM)()

        freq = [0] * nM
        xmpf = [0] * nM
        ympf = [0] * nM
        zmpf = [0] * nM

        for i in range(nM):

            freq[i] = c_double()
            xmpf[i] = c_double()
            ympf[i] = c_double()
            zmpf[i] = c_double()

            c_modalResults[i] = C_ModalResults(
                pointer(freq[i]),
                pointer(xmpf[i]),
                pointer(ympf[i]),
                pointer(zmpf[i]),
                ip(modalout.node[i, :]),
                dp(modalout.xdsp[i, :]),
                dp(modalout.ydsp[i, :]),
                dp(modalout.zdsp[i, :]),
                dp(modalout.xrot[i, :]),
                dp(modalout.yrot[i, :]),
                dp(modalout.zrot[i, :]),
            )

        # set dynamics data
        exagg_modal = 1.0  # not used
        c_dynamicData = C_DynamicData(self.nM, self.Mmethod, self.lump, self.tol, self.shift, exagg_modal)

        exitCode = self._pyframe3dd.run(
            self.c_nodes,
            self.c_reactions,
            self.c_elements,
            self.c_other,
            nCases,
            c_loadcases,
            c_dynamicData,
            self.c_extraInertia,
            self.c_extraMass,
            self.c_condensation,
            c_disp,
            c_forces,
            c_reactions,
            c_internalForces,
            c_massResults,
            c_modalResults,
        )

        nantest = np.isnan(np.c_[fout.Nx, fout.Vy, fout.Vz, fout.Txx, fout.Myy, fout.Mzz])
        if not nanokay and np.any(nantest):
            raise RuntimeError("Frame3DD did not exit gracefully")
        elif exitCode == 182 or exitCode == 183:
            pass
        elif exitCode != 0:
            raise RuntimeError("Frame3DD did not exit gracefully")

        # put mass values back in since tuple is read only
        mout = NodeMasses(
            total_mass.value,
            struct_mass.value,
            mout.node,
            mout.xmass,
            mout.ymass,
            mout.zmass,
            mout.xinrta,
            mout.yinrta,
            mout.zinrta,
        )

        # put modal results back in
        for i in range(nM):
            modalout.freq[i] = freq[i].value
            modalout.xmpf[i] = xmpf[i].value
            modalout.ympf[i] = ympf[i].value
            modalout.zmpf[i] = zmpf[i].value

        return dout, fout, rout, ifout, mout, modalout

    def write(self, fname):
        f = open(fname, "w")
        f.write("pyFrame3dd auto-generated file\n")
        f.write("\n")
        f.write(str(len(self.nnode)) + " # number of nodes\n")
        f.write("#.node  x       y       z       r\n")
        f.write("#        m      m      m      m\n")
        f.write("\n")
        for k in range(len(self.nnode)):
            f.write(
                str(self.nnode[k])
                + "\t"
                + str(self.nx[k])
                + "\t"
                + str(self.ny[k])
                + "\t"
                + str(self.nz[k])
                + "\t"
                + str(self.nr[k])
                + "\n"
            )
        f.write("\n")
        f.write(str(len(self.rnode)) + " # number of nodes with reactions\n")
        f.write("#.n     x  y  z xx yy zz          1=fixed, 0=free\n")
        for k in range(len(self.rnode)):
            f.write(
                str(self.rnode[k])
                + "\t"
                + str(self.rKx[k])
                + "\t"
                + str(self.rKy[k])
                + "\t"
                + str(self.rKz[k])
                + "\t"
                + str(self.rKtx[k])
                + "\t"
                + str(self.rKty[k])
                + "\t"
                + str(self.rKtz[k])
                + "\n"
            )
        f.write("\n")
        f.write(str(len(self.eelement)) + " # number of frame elements\n")
        f.write("#.e n1 n2 Ax    Asy     Asz     Jxx     Iyy     Izz     E       G   roll density\n")
        f.write("#   .  .  m^2   m^2     m^2     m^4     m^4     m^4     Pa      Pa  deg  kg/m^3\n")
        for k in range(len(self.eelement)):
            f.write(
                str(self.eelement[k])
                + "\t"
                + str(self.eN1[k])
                + "\t"
                + str(self.eN2[k])
                + "\t"
                + str(self.eAx[k])
                + "\t"
                + str(self.eAsy[k])
                + "\t"
                + str(self.eAsz[k])
                + "\t"
                + str(self.eJx[k])
                + "\t"
                + str(self.eIy[k])
                + "\t"
                + str(self.eIz[k])
                + "\t"
                + str(self.eE[k])
                + "\t"
                + str(self.eG[k])
                + "\t"
                + str(self.eroll[k])
                + "\t"
                + str(self.edensity[k])
                + "\n"
            )
        f.write("\n")
        f.write("\n")
        ishear = 1 if self.options.shear else 0
        igeom = 1 if self.options.geom else 0
        f.write(str(ishear) + " # 1: include shear deformation\n")
        f.write(str(igeom) + " # 1: include geometric stiffness\n")
        f.write("10.0 # exaggerate static mesh deformations\n")
        f.write("2.5 # zoom scale for 3D plotting\n")
        f.write(str(self.options.dx) + " # x-axis increment for internal forces, m\n")
        f.write("\n")
        f.write(str(len(self.loadCases)) + " # number of static load cases\n")
        for iC in range(len(self.loadCases)):
            mylc = self.loadCases[iC]
            f.write("\n")
            f.write("# Begin Static Load Case " + str(iC + 1) + " of " + str(len(self.loadCases)) + "\n")
            f.write("\n")

            f.write("# gravitational acceleration for self-weight loading (global)\n")
            f.write("#.gX	gY     gZ\n")
            f.write("#.m/s^2   m/s^2   m/s^2\n")
            f.write(str(mylc.gx) + "\t" + str(mylc.gy) + "\t" + str(mylc.gz) + "\n")
            f.write("\n")

            f.write(str(len(mylc.NF)) + "	# number of loaded nodes\n")
            f.write("#.e      Fx       Fy     Fz      Mxx     Myy     Mzz\n")
            f.write("#        N        N      N       N.m     N.m     N.m\n")
            for k in range(len(mylc.NF)):
                f.write(
                    str(mylc.NF[k])
                    + "\t"
                    + str(mylc.Fx[k])
                    + "\t"
                    + str(mylc.Fy[k])
                    + "\t"
                    + str(mylc.Fz[k])
                    + "\t"
                    + str(mylc.Mxx[k])
                    + "\t"
                    + str(mylc.Myy[k])
                    + "\t"
                    + str(mylc.Mzz[k])
                    + "\n"
                )

            f.write(str(len(mylc.ELU)) + " # number of uniform loads\n")
            f.write("#.e    Ux   Uy   Uz\n")
            f.write("#      N/m  N/m  N/m\n")
            for k in range(len(mylc.ELU)):
                f.write(
                    str(mylc.ELU[k]) + "\t" + str(mylc.Ux[k]) + "\t" + str(mylc.Uy[k]) + "\t" + str(mylc.Uz[k]) + "\n"
                )

            f.write(str(len(mylc.ELT)) + " # number of trapezoidal loads\n")
            f.write("#.e     x1       x2        w1      w2\n")
            f.write("#       m        m         N/m     N/m\n")
            for k in range(len(mylc.ELT)):
                f.write(
                    str(mylc.ELT[k])
                    + "\t"
                    + str(mylc.xx1[k])
                    + "\t"
                    + str(mylc.xx2[k])
                    + "\t"
                    + str(mylc.wx1[k])
                    + "\t"
                    + str(mylc.wx2[k])
                    + "\n"
                )
                f.write(
                    "\t"
                    + str(mylc.xy1[k])
                    + "\t"
                    + str(mylc.xy2[k])
                    + "\t"
                    + str(mylc.wy1[k])
                    + "\t"
                    + str(mylc.wy2[k])
                    + "\n"
                )
                f.write(
                    "\t"
                    + str(mylc.xz1[k])
                    + "\t"
                    + str(mylc.xz2[k])
                    + "\t"
                    + str(mylc.wz1[k])
                    + "\t"
                    + str(mylc.wz2[k])
                    + "\n"
                )

            f.write(str(len(mylc.ELE)) + " # number of internal concentrated loads\n")
            f.write("#.e    Px   Py    Pz   x    \n")
            f.write("#      N    N     N    m\n")
            for k in range(len(mylc.ELE)):
                f.write(
                    str(mylc.ELE[k])
                    + "\t"
                    + str(mylc.Px[k])
                    + "\t"
                    + str(mylc.Py[k])
                    + "\t"
                    + str(mylc.Pz[k])
                    + "\t"
                    + str(mylc.xE[k])
                    + "\n"
                )

            f.write(str(len(mylc.ELTemp)) + " # number of temperature loads\n")
            f.write("#.e  alpha   hy   hz   Ty+  Ty-  Tz+  Tz-\n")
            f.write("#    /degC   m    m   degC degC degC degC\n")
            for k in range(len(mylc.ELTemp)):
                f.write(
                    str(mylc.ELTemp[k])
                    + "\t"
                    + str(mylc.a[k])
                    + "\t"
                    + str(mylc.hy[k])
                    + "\t"
                    + str(mylc.hz[k])
                    + "\t"
                    + str(mylc.Typ[k])
                    + "\t"
                    + str(mylc.Tym[k])
                    + str(mylc.Tzp[k])
                    + "\t"
                    + str(mylc.Tzm[k])
                    + "\n"
                )

            f.write("0    # number of nodes with prescribed displacements\n")
            f.write("\n")
            f.write("# End Static Load Case " + str(iC + 1) + " of " + str(len(self.loadCases)) + "\n")
        f.write("\n")
        f.write("\n")
        f.write(str(self.nM) + "    # number of desired dynamic modes of vibration\n")
        f.write(str(self.Mmethod) + "    # 1: subspace Jacobi     2: Stodola\n")
        f.write(str(self.lump) + "    # 0: consistent mass ... 1: lumped mass matrix\n")
        f.write(str(self.tol) + " # mode shape tolerance\n")
        f.write(str(self.shift) + "  # shift value ... for unrestrained structures\n")
        f.write("10.0 # exaggerate modal mesh deformations\n")
        f.write("\n")
        f.write("# nodes and concentrated mass and inertia\n")
        f.write(str(len(self.ENMnode)) + "                               # number of nodes with extra inertia\n")
        f.write("#.n      Mass   Ixx      Iyy      Izz\n")
        f.write("#        kg    kg.m^2   kg.m^2   kg.m^2\n")
        for k in range(len(self.ENMnode)):
            f.write(
                str(self.ENMnode[k])
                + "\t"
                + str(self.ENMmass[k])
                + "\t"
                + str(self.ENMIxx[k])
                + "\t"
                + str(self.ENMIyy[k])
                + "\t"
                + str(self.ENMIzz[k])
                + "\n"
            )
        f.write("\n")
        f.write("0 # frame elements with extra mass\n")
        f.write("\n")
        f.write(str(self.nM) + "				# number of modes to animate, nA\n")
        for k in range(self.nM):
            f.write(" " + str(k + 1))
        f.write("  # list of modes to animate - omit if nA == 0\n")
        f.write("2                               # pan rate during animation\n")
        f.write("\n")
        f.write("# End of input data file\n")
        f.close()

    def draw(self, savefig=False, fig_idx=0):
        # Visualization for debugging
        import matplotlib.pyplot as plt

        nnode = len(self.nx)
        node_array = np.zeros((nnode, 3))
        mynodes = {}
        for k in range(nnode):
            temp = np.r_[self.nx[k], self.ny[k], self.nz[k]]
            mynodes[self.nnode[k]] = temp
            node_array[k, :] = temp
        myelem = []
        for k in range(len(self.eN1)):
            myelem.append((self.eN1[k], self.eN2[k]))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for e in myelem:
            xs = np.array([mynodes[e[0]][0], mynodes[e[1]][0]])
            ys = np.array([mynodes[e[0]][1], mynodes[e[1]][1]])
            zs = np.array([mynodes[e[0]][2], mynodes[e[1]][2]])
            ax.plot(xs, ys, zs, "b-")
        ax.plot(node_array[:, 0], node_array[:, 1], node_array[:, 2], ".k", markersize=10)

        if savefig:
            plt.savefig(f"frame3dd_{fig_idx}.png")
        else:
            plt.show()


class StaticLoadCase(object):
    """docstring"""

    def __init__(self, gx, gy, gz):

        self.gx = gx
        self.gy = gy
        self.gz = gz

        i = np.array([], dtype=np.int32)
        d = np.array([])

        self.changePointLoads(i, d, d, d, d, d, d)

        self.changeUniformLoads(i, d, d, d)

        self.changeTrapezoidalLoads(i, d, d, d, d, d, d, d, d, d, d, d, d)

        self.changeElementLoads(i, d, d, d, d)

        self.changeTemperatureLoads(i, d, d, d, d, d, d, d)

        self.changePrescribedDisplacements(i, d, d, d, d, d, d)

    def changePointLoads(self, N, Fx, Fy, Fz, Mxx, Myy, Mzz):

        # copying to prevent any user error with variables pointing to something else (b/c memory address is shared by C)
        self.NF = np.array(N).astype(np.int32)
        self.Fx = np.copy(Fx)
        self.Fy = np.copy(Fy)
        self.Fz = np.copy(Fz)
        self.Mxx = np.copy(Mxx)
        self.Myy = np.copy(Myy)
        self.Mzz = np.copy(Mzz)

        self.pL = C_PointLoads(
            len(N), ip(self.NF), dp(self.Fx), dp(self.Fy), dp(self.Fz), dp(self.Mxx), dp(self.Myy), dp(self.Mzz)
        )

    def changeUniformLoads(self, EL, Ux, Uy, Uz):

        self.ELU = np.array(EL).astype(np.int32)
        self.Ux = np.copy(Ux)
        self.Uy = np.copy(Uy)
        self.Uz = np.copy(Uz)

        self.uL = C_UniformLoads(len(EL), ip(self.ELU), dp(self.Ux), dp(self.Uy), dp(self.Uz))

    def changeTrapezoidalLoads(self, EL, xx1, xx2, wx1, wx2, xy1, xy2, wy1, wy2, xz1, xz2, wz1, wz2):

        self.ELT = np.array(EL).astype(np.int32)
        self.xx1 = np.copy(xx1)
        self.xx2 = np.copy(xx2)
        self.wx1 = np.copy(wx1)
        self.wx2 = np.copy(wx2)
        self.xy1 = np.copy(xy1)
        self.xy2 = np.copy(xy2)
        self.wy1 = np.copy(wy1)
        self.wy2 = np.copy(wy2)
        self.xz1 = np.copy(xz1)
        self.xz2 = np.copy(xz2)
        self.wz1 = np.copy(wz1)
        self.wz2 = np.copy(wz2)

        self.tL = C_TrapezoidalLoads(
            len(EL),
            ip(self.ELT),
            dp(self.xx1),
            dp(self.xx2),
            dp(self.wx1),
            dp(self.wx2),
            dp(self.xy1),
            dp(self.xy2),
            dp(self.wy1),
            dp(self.wy2),
            dp(self.xz1),
            dp(self.xz2),
            dp(self.wz1),
            dp(self.wz2),
        )

    def changeElementLoads(self, EL, Px, Py, Pz, x):

        self.ELE = np.array(EL).astype(np.int32)
        self.Px = np.copy(Px)
        self.Py = np.copy(Py)
        self.Pz = np.copy(Pz)
        self.xE = np.copy(x)

        self.eL = C_ElementLoads(len(EL), ip(self.ELE), dp(self.Px), dp(self.Py), dp(self.Pz), dp(self.xE))

    def changeTemperatureLoads(self, EL, a, hy, hz, Typ, Tym, Tzp, Tzm):

        self.ELTemp = np.array(EL).astype(np.int32)
        self.a = np.copy(a)
        self.hy = np.copy(hy)
        self.hz = np.copy(hz)
        self.Typ = np.copy(Typ)
        self.Tym = np.copy(Tym)
        self.Tzp = np.copy(Tzp)
        self.Tzm = np.copy(Tzm)

        self.tempL = C_TemperatureLoads(
            len(EL),
            ip(self.ELTemp),
            dp(self.a),
            dp(self.hy),
            dp(self.hz),
            dp(self.Typ),
            dp(self.Tym),
            dp(self.Tzp),
            dp(self.Tzm),
        )

    def changePrescribedDisplacements(self, N, Dx, Dy, Dz, Dxx, Dyy, Dzz):

        self.ND = np.array(N).astype(np.int32)
        self.Dx = np.copy(Dx)
        self.Dy = np.copy(Dy)
        self.Dz = np.copy(Dz)
        self.Dxx = np.copy(Dxx)
        self.Dyy = np.copy(Dyy)
        self.Dzz = np.copy(Dzz)

        self.pD = C_PrescribedDisplacements(
            len(N), ip(self.ND), dp(self.Dx), dp(self.Dy), dp(self.Dz), dp(self.Dxx), dp(self.Dyy), dp(self.Dzz)
        )
