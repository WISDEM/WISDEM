#!/usr/bin/env python
# encoding: utf-8
"""
precomp.py

Created by Andrew Ning on 2012-02-28.
Copyright (c)  NREL. All rights reserved.
"""

from __future__ import print_function

import numpy as np
import math
import copy
import os

# from rotorstruc import SectionStrucInterface
# from wisdem.common import sind, cosd
# from external._precomp import precomp as _precomp
from wisdem.rotorse._precomp import precomp as _precomp


def web_loc(r, chord, le, ib_idx, ob_idx, ib_webc, ob_webc):

    n = len(r)
    loc = np.zeros(n)

    for i in range(n):

        if i < ib_idx or i > ob_idx:
            loc[i] = -1
        else:
            xn = (r[i] - r[ib_idx]) / (r[ob_idx] - r[ib_idx])
            loc[i] = (
                le[i]
                - (le[ib_idx] - ib_webc) * chord[ib_idx] / chord[i] * (1 - xn)
                - (le[ob_idx] - ob_webc) * chord[ob_idx] / chord[i] * xn
            )

    return loc


class PreComp:
    def __init__(
        self,
        r,
        chord,
        theta,
        leLoc,
        precurve,
        presweep,
        profile,
        materials,
        upperCS,
        lowerCS,
        websCS,
        sector_idx_strain_spar_ps,
        sector_idx_strain_spar_ss,
        sector_idx_strain_te_ps,
        sector_idx_strain_te_ss,
    ):
        """Constructor

        Parameters
        ----------
        r : ndarray (m)
            radial positions. r[0] should be the hub location
            while r[-1] should be the blade tip. Any number
            of locations can be specified between these in ascending order.
        chord : ndarray (m)
            array of chord lengths at corresponding radial positions
        theta : ndarray (deg)
            array of twist angles at corresponding radial positions.
            (positive twist decreases angle of attack)
        leLoc : ndarray(float)
            array of leading-edge positions from a reference blade axis (usually blade pitch axis).
            locations are normalized by the local chord length.  e.g. leLoc[i] = 0.2 means leading edge
            is 0.2*chord[i] from reference axis.   positive in -x direction for airfoil-aligned coordinate system
        profile : list(:class:`Profile`)
            airfoil shape at each radial position
        materials : list(:class:`Orthotropic2DMaterial`), optional
            list of all Orthotropic2DMaterial objects used in defining the geometry
        upperCS, lowerCS, websCS : list(:class:`CompositeSection`)
            list of CompositeSection objections defining the properties for upper surface, lower surface,
            and shear webs (if any) for each section

        """

        self.r = np.array(r)
        self.chord = np.array(chord)
        self.theta = np.array(theta)
        self.leLoc = np.array(leLoc)
        self.precurve = np.array(precurve)
        self.presweep = np.array(presweep)

        self.profile = profile
        self.materials = materials
        self.upperCS = upperCS
        self.lowerCS = lowerCS
        self.websCS = websCS

        self.sector_idx_strain_spar_ps = sector_idx_strain_spar_ps
        self.sector_idx_strain_spar_ss = sector_idx_strain_spar_ss
        self.sector_idx_strain_te_ps = sector_idx_strain_te_ps
        self.sector_idx_strain_te_ss = sector_idx_strain_te_ss

        # twist rate
        self.th_prime = _precomp.tw_rate(self.r, self.theta)

    def sectionProperties(self):
        """see meth:`SectionStrucInterface.sectionProperties`"""

        # radial discretization
        nsec = len(self.r)

        # initialize variables
        beam_z = self.r
        beam_EA = np.zeros(nsec)
        beam_EIxx = np.zeros(nsec)
        beam_EIyy = np.zeros(nsec)
        beam_EIxy = np.zeros(nsec)
        beam_GJ = np.zeros(nsec)
        beam_rhoA = np.zeros(nsec)
        beam_A = np.zeros(nsec)
        beam_rhoJ = np.zeros(nsec)
        beam_Tw_iner = np.zeros(nsec)

        beam_flap_iner = np.zeros(nsec)
        beam_edge_iner = np.zeros(nsec)

        beam_x_sc = np.zeros(nsec)
        beam_y_sc = np.zeros(nsec)
        beam_x_tc = np.zeros(nsec)
        beam_y_tc = np.zeros(nsec)
        beam_x_cg = np.zeros(nsec)
        beam_y_cg = np.zeros(nsec)

        # distance to elastic center from point about which structural properties are computed
        # using airfoil coordinate system
        beam_x_ec = np.zeros(nsec)
        beam_y_ec = np.zeros(nsec)

        # distance to elastic center from airfoil nose
        # using profile coordinate system
        self.x_ec_nose = np.zeros(nsec)
        self.y_ec_nose = np.zeros(nsec)

        profile = self.profile
        mat = self.materials
        csU = self.upperCS
        csL = self.lowerCS
        csW = self.websCS

        # arrange materials into array
        n = len(mat)
        E1 = [0] * n
        E2 = [0] * n
        G12 = [0] * n
        nu12 = [0] * n
        rho = [0] * n

        for i in range(n):
            E1[i] = mat[i].E1
            E2[i] = mat[i].E2
            G12[i] = mat[i].G12
            nu12[i] = mat[i].nu12
            rho[i] = mat[i].rho

        for i in range(nsec):
            # print(i)

            xnode, ynode = profile[i]._preCompFormat()
            locU, n_laminaU, n_pliesU, tU, thetaU, mat_idxU = csU[i]._preCompFormat()
            locL, n_laminaL, n_pliesL, tL, thetaL, mat_idxL = csL[i]._preCompFormat()
            locW, n_laminaW, n_pliesW, tW, thetaW, mat_idxW = csW[i]._preCompFormat()

            nwebs = len(locW)

            # address a bug in f2py (need to pass in length 1 arrays even though they are not used)
            if nwebs == 0:
                locW = [0]
                n_laminaW = [0]
                n_pliesW = [0]
                tW = [0]
                thetaW = [0]
                mat_idxW = [0]

            results = _precomp.properties(
                self.chord[i],
                self.theta[i],
                self.th_prime[i],
                self.leLoc[i],
                xnode,
                ynode,
                E1,
                E2,
                G12,
                nu12,
                rho,
                locU,
                n_laminaU,
                n_pliesU,
                tU,
                thetaU,
                mat_idxU,
                locL,
                n_laminaL,
                n_pliesL,
                tL,
                thetaL,
                mat_idxL,
                nwebs,
                locW,
                n_laminaW,
                n_pliesW,
                tW,
                thetaW,
                mat_idxW,
            )

            beam_EIxx[i] = results[1]  # EIedge
            beam_EIyy[i] = results[0]  # EIflat
            beam_GJ[i] = results[2]
            beam_EA[i] = results[3]
            beam_EIxy[i] = results[4]  # EIflapedge
            beam_x_sc[i] = results[10]
            beam_y_sc[i] = results[11]
            beam_x_tc[i] = results[12]
            beam_y_tc[i] = results[13]
            beam_x_ec[i] = results[12] - results[10]
            beam_y_ec[i] = results[13] - results[11]
            beam_rhoA[i] = results[14]
            beam_A[i] = results[15]
            beam_rhoJ[i] = results[16] + results[17]  # perpendicular axis theorem
            beam_Tw_iner[i] = results[18]
            beam_x_cg[i] = results[19]
            beam_y_cg[i] = results[20]

            beam_flap_iner[i] = results[16]
            beam_edge_iner[i] = results[17]

            self.x_ec_nose[i] = results[13] + self.leLoc[i] * self.chord[i]
            self.y_ec_nose[i] = results[12]  # switch b.c of coordinate system used

        return (
            beam_EIxx,
            beam_EIyy,
            beam_GJ,
            beam_EA,
            beam_EIxy,
            beam_x_ec,
            beam_y_ec,
            beam_rhoA,
            beam_A,
            beam_rhoJ,
            beam_Tw_iner,
            beam_flap_iner,
            beam_edge_iner,
            beam_x_tc,
            beam_y_tc,
            beam_x_sc,
            beam_y_sc,
            beam_x_cg,
            beam_y_cg,
        )

    def criticalStrainLocations(self, sector_idx_strain_ss, sector_idx_strain_ps):

        n = len(self.r)

        # find location of max thickness on airfoil
        xun = np.zeros(n)
        xln = np.zeros(n)
        yun = np.zeros(n)
        yln = np.zeros(n)

        # for i, p in enumerate(self.profile):
        #     xun[i], yun[i], yln[i] = p.locationOfMaxThickness()
        # xln = xun

        for i in range(n):
            csU = self.upperCS[i]
            csL = self.lowerCS[i]
            pf = self.profile[i]
            idx_ss = sector_idx_strain_ss[i]
            idx_ps = sector_idx_strain_ps[i]

            if idx_ps == None:
                xln[i] = 0.0
                yln[i] = 0.0
            else:
                xln[i] = 0.5 * (csL.loc[idx_ps] + csL.loc[idx_ps + 1])
                yln[i] = np.interp(xln[i], pf.x, pf.yl)

            if idx_ss == None:
                xun[i] = 0.0
                yun[i] = 0.0
            else:
                xun[i] = 0.5 * (csU.loc[idx_ss] + csU.loc[idx_ss + 1])
                yun[i] = np.interp(xun[i], pf.x, pf.yu)

        # make dimensional and define relative to elastic center
        xu = xun * self.chord - self.x_ec_nose
        xl = xln * self.chord - self.x_ec_nose
        yu = yun * self.chord - self.y_ec_nose
        yl = yln * self.chord - self.y_ec_nose

        # switch to airfoil coordinate system
        xu, yu = yu, xu
        xl, yl = yl, xl

        return xu, xl, yu, yl

    def panelBucklingStrain(self, sector_idx_array):
        """
        see chapter on Structural Component Design Techniques from Alastair Johnson
        section 6.2: Design of composite panels

        assumes: large aspect ratio, simply supported, uniaxial compression, flat rectangular plate

        """
        chord = self.chord
        nsec = len(self.r)

        eps_crit = np.zeros(nsec)

        for i in range(nsec):

            cs = self.upperCS[i]  # TODO: lower surface may be the compression one
            sector_idx = sector_idx_array[i]

            if sector_idx == None:
                eps_crit[i] = 0.0

            else:

                # chord-wise length of sector
                sector_length = chord[i] * (cs.loc[sector_idx + 1] - cs.loc[sector_idx])

                # get matrices
                A, B, D, totalHeight = cs.compositeMatrices(sector_idx)
                E = cs.effectiveEAxial(sector_idx)
                D1 = D[0, 0]
                D2 = D[1, 1]
                D3 = D[0, 1] + 2 * D[2, 2]

                # use empirical formula
                Nxx = 2 * (math.pi / sector_length) ** 2 * (math.sqrt(D1 * D2) + D3)
                # Nxx = 3.6 * (math.pi/sector_length)**2 * D1

                eps_crit[i] = -Nxx / totalHeight / E

        return eps_crit


def skipLines(f, n):
    for i in range(n):
        f.readline()


class CompositeSection:
    """A CompositeSection defines the layup of the entire
    airfoil cross-section

    """

    def __init__(self, loc, n_plies, t, theta, mat_idx, materials):
        """Constructor

        Parameters
        ----------


        """

        self.loc = np.array(loc)  # np.array([0.0, 0.15, 0.50, 1.00])

        # should be list of numpy arrays
        self.n_plies = n_plies  # [ [1, 1, 33],  [1, 1, 17, 38, 0, 37, 16], [1, 1, 17, 0, 16] ]
        self.t = t  # [ [0.000381, 0.00051, 0.00053], [0.000381, 0.00051, 0.00053, 0.00053, 0.003125, 0.00053, 0.00053], [0.000381, 0.00051, 0.00053, 0.003125, 0.00053] ]
        self.theta = theta  # [ [0, 0, 20], [0, 0, 20, 30, 0, 30, 20], [0, 0, 20, 0, 0] ]
        self.mat_idx = mat_idx  # [ [3, 4, 2], [3, 4, 2, 1, 5, 1, 2], [3, 4, 2, 5, 2] ]

        self.materials = materials

    def mycopy(self):
        return CompositeSection(
            copy.deepcopy(self.loc),
            copy.deepcopy(self.n_plies),
            copy.deepcopy(self.t),
            copy.deepcopy(self.theta),
            copy.deepcopy(self.mat_idx),
            self.materials,
        )  # TODO: copy materials (for now it never changes so I'm not looking at it)

    @classmethod
    def initFromPreCompLayupFile(cls, fname, locW, materials, readLocW=False):
        """Construct CompositeSection object from a PreComp input file

        Parameters
        ----------
        fname : str
            name of input file
        webLoc : ndarray
            array of web locations (i.e. [0.15, 0.5] has two webs
            one located at 15% chord from the leading edge and
            the second located at 50% chord)
        materials : list(:class:`Orthotropic2DMaterial`)
            material objects defined in same order as used in the input file
            can use :meth:`Orthotropic2DMaterial.initFromPreCompFile`
        readLocW : optionally read web location from main input file rather than
            have the user provide it

        Returns
        -------
        compSec : CompositeSection
            an initialized CompositeSection object

        """

        f = open(fname)

        skipLines(f, 3)

        # number of sectors
        n_sector = int(f.readline().split()[0])

        skipLines(f, 2)

        # read normalized chord locations
        locU = [float(x) for x in f.readline().split()]

        n_pliesU, tU, thetaU, mat_idxU = CompositeSection.__readSectorsFromFile(f, n_sector)
        upper = cls(locU, n_pliesU, tU, thetaU, mat_idxU, materials)

        skipLines(f, 3)

        # number of sectors
        n_sector = int(f.readline().split()[0])

        skipLines(f, 2)

        locL = [float(x) for x in f.readline().split()]

        n_pliesL, tL, thetaL, mat_idxL = CompositeSection.__readSectorsFromFile(f, n_sector)
        lower = cls(locL, n_pliesL, tL, thetaL, mat_idxL, materials)

        skipLines(f, 4)

        if readLocW:
            locW = CompositeSection.__readWebLocFromFile(fname)
        n_sector = len(locW)

        n_pliesW, tW, thetaW, mat_idxW = CompositeSection.__readSectorsFromFile(f, n_sector)
        webs = cls(locW, n_pliesW, tW, thetaW, mat_idxW, materials)

        f.close()

        return upper, lower, webs

    @staticmethod
    def __readSectorsFromFile(f, n_sector):
        """private method"""

        n_plies = [0] * n_sector
        t = [0] * n_sector
        theta = [0] * n_sector
        mat_idx = [0] * n_sector

        for i in range(n_sector):
            skipLines(f, 2)

            line = f.readline()
            if line == "":
                return []  # no webs
            n_lamina = int(line.split()[1])

            skipLines(f, 4)

            n_plies_S = np.zeros(n_lamina)
            t_S = np.zeros(n_lamina)
            theta_S = np.zeros(n_lamina)
            mat_idx_S = np.zeros(n_lamina)

            for j in range(n_lamina):
                array = f.readline().split()
                n_plies_S[j] = int(array[1])
                t_S[j] = float(array[2])
                theta_S[j] = float(array[3])
                mat_idx_S[j] = int(array[4]) - 1

            n_plies[i] = n_plies_S
            t[i] = t_S
            theta[i] = theta_S
            mat_idx[i] = mat_idx_S

        return n_plies, t, theta, mat_idx

    @staticmethod
    def __readWebLocFromFile(fname):
        # Get web locations from main input file
        f_main = os.path.join(os.path.split(fname)[0], os.path.split(fname)[1].replace("layup", "input"))

        # Error handling for different file extensions
        if not os.path.isfile(f_main):
            extensions = ["dat", "inp", "pci"]
            for ext in extensions:
                f_main = f_main[:-3] + ext
                if os.path.isfile(f_main):
                    break

        fid = open(f_main)

        var = fid.readline().split()[0]
        while var != "Web_num":
            text = fid.readline().split()
            if len(text) > 0:
                var = text[0]
            else:
                var = None

        web_loc = []
        line = fid.readline().split()
        while line:
            web_loc.append(float(line[1]))
            line = fid.readline().split()

        return web_loc

    def compositeMatrices(self, sector):
        """Computes the matrix components defining the constituitive equations
        of the complete laminate stack

        Returns
        -------
        A : ndarray, shape(3, 3)
            upper left portion of constitutive matrix
        B : ndarray, shape(3, 3)
            off-diagonal portion of constitutive matrix
        D : ndarray, shape(3, 3)
            lower right portion of constitutive matrix
        totalHeight : float (m)
            total height of the laminate stack

        Notes
        -----
        | The constitutive equations are arranged in the format
        | [N; M] = [A B; B D] * [epsilon; k]
        | where N = [N_x, N_y, N_xy] are the normal force resultants for the laminate
        | M = [M_x, M_y, M_xy] are the moment resultants
        | epsilon = [epsilon_x, epsilon_y, gamma_xy] are the midplane strains
        | k = [k_x, k_y, k_xy] are the midplane curvates

        See [1]_ for further details, and this :ref:`equation <ABBD>` in the user guide.

        References
        ----------
        .. [1] J. C. Halpin. Primer on Composite Materials Analysis. Technomic, 2nd edition, 1992.

        """

        t = self.t[sector]
        n_plies = self.n_plies[sector]
        mat_idx = self.mat_idx[sector]
        theta = self.theta[sector]

        mat_idx = mat_idx.astype(int)  # convert to integers if actually stored as floats

        n = len(theta)

        # heights (z - absolute, h - relative to mid-plane)
        z = np.zeros(n + 1)
        for i in range(n):
            z[i + 1] = z[i] + t[i] * n_plies[i]

        z_mid = (z[-1] - z[0]) / 2.0
        h = z - z_mid

        # ABD matrices
        A = np.zeros((3, 3))
        B = np.zeros((3, 3))
        D = np.zeros((3, 3))

        for i in range(n):
            Qbar = self.__Qbar(self.materials[mat_idx[i]], theta[i])
            A += Qbar * (h[i + 1] - h[i])
            B += 0.5 * Qbar * (h[i + 1] ** 2 - h[i] ** 2)
            D += 1.0 / 3.0 * Qbar * (h[i + 1] ** 3 - h[i] ** 3)

        totalHeight = z[-1] - z[0]

        return A, B, D, totalHeight

    def effectiveEAxial(self, sector):
        """Estimates the effective axial modulus of elasticity for the laminate

        Returns
        -------
        E : float (N/m^2)
            effective axial modulus of elasticity

        Notes
        -----
        see user guide for a :ref:`derivation <ABBD>`

        """

        A, B, D, totalHeight = self.compositeMatrices(sector)

        # S = [A B; B D]

        S = np.vstack((np.hstack((A, B)), np.hstack((B, D))))

        # E_eff_x = N_x/h/eps_xx and eps_xx = S^{-1}(0,0)*N_x (approximately)
        detS = np.linalg.det(S)
        Eaxial = detS / np.linalg.det(S[1:, 1:]) / totalHeight

        return Eaxial

    def __Qbar(self, material, theta):
        """Computes the lamina stiffness matrix

        Returns
        -------
        Qbar : numpy matrix
            the lamina stifness matrix

        Notes
        -----
        Transforms a specially orthotropic lamina from principal axis to
        an arbitrary axis defined by the ply orientation.
        [sigma_x; sigma_y; tau_xy]^T = Qbar * [epsilon_x; epsilon_y, gamma_xy]^T
        See [1]_ for further details.

        References
        ----------
        .. [1] J. C. Halpin. Primer on Composite Materials Analysis. Technomic, 2nd edition, 1992.


        """

        E11 = material.E1
        E22 = material.E2
        nu12 = material.nu12
        nu21 = nu12 * E22 / E11
        G12 = material.G12
        denom = 1 - nu12 * nu21

        c = math.cos(theta * math.pi / 180.0)
        s = math.sin(theta * math.pi / 180.0)
        c2 = c * c
        s2 = s * s
        cs = c * s

        Q = np.array([[E11 / denom, nu12 * E22 / denom, 0], [nu12 * E22 / denom, E22 / denom, 0], [0, 0, G12]])
        T12 = np.array([[c2, s2, cs], [s2, c2, -cs], [-cs, cs, 0.5 * (c2 - s2)]])
        Tinv = np.array([[c2, s2, -2 * cs], [s2, c2, 2 * cs], [cs, -cs, c2 - s2]])

        return Tinv @ Q @ T12

    def _preCompFormat(self):

        n = len(self.theta)
        n_lamina = np.zeros(n)

        if n == 0:
            return self.loc, n_lamina, self.n_plies, self.t, self.theta, self.mat_idx

        for i in range(n):
            n_lamina[i] = len(self.theta[i])

        mat = np.concatenate(self.mat_idx)
        for i in range(len(mat)):
            mat[i] += 1  # 1-based indexing in Fortran

        return self.loc, n_lamina, np.concatenate(self.n_plies), np.concatenate(self.t), np.concatenate(self.theta), mat


class Orthotropic2DMaterial:
    """Represents a homogeneous orthotropic material in a
    plane stress state.

    """

    def __init__(self, E1, E2, G12, nu12, rho, name=""):
        """a struct-like object.  all inputs are also fields.
        The object also has an identification
        number *.mat_idx so unique materials can be identified.

        Parameters
        ----------
        E1 : float (N/m^2)
            Young's modulus in first principal direction
        E2 : float (N/m^2)
            Young's modulus in second principal direction
        G12 : float (N/m^2)
            shear modulus
        nu12 : float
            Poisson's ratio  (nu12*E22 = nu21*E11)
        rho : float (kg/m^3)
            density
        name : str
            an optional identifier

        """
        self.E1 = E1
        self.E2 = E2
        self.G12 = G12
        self.nu12 = nu12
        self.rho = rho
        self.name = name

    @classmethod
    def listFromPreCompFile(cls, fname):
        """initialize the object by extracting materials from a PreComp materials file

        Parameters
        ----------
        fname : str
            name of input file

        Returns
        -------
        materials : List(:class:`Orthotropic2DMaterial`)
            a list of all materials gathered from the file.

        """

        f = open(fname)

        skipLines(f, 3)

        materials = []
        for line in f:
            array = line.split()
            mat = cls(float(array[1]), float(array[2]), float(array[3]), float(array[4]), float(array[5]), array[6])

            materials.append(mat)
        f.close()

        return materials


class Profile:
    """Defines the shape of an airfoil"""

    def __init__(self, xu, yu, xl, yl):
        """Constructor

        Parameters
        ----------
        xu : ndarray
            x coordinates for upper surface of airfoil
        yu : ndarray
            y coordinates for upper surface of airfoil
        xl : ndarray
            x coordinates for lower surface of airfoil
        yl : ndarray
            y coordinates for lower surface of airfoil

        Notes
        -----
        uses :ref:`airfoil-aligned coordinate system <blade_airfoil_coord>`.
        Nodes should be ordered from the leading edge toward the trailing edge.
        The leading edge can be located at any position and the
        chord may be any size, however the airfoil should be untwisted.
        Normalization to unit chord will happen internally.

        """

        # parse airfoil data
        xu = np.array(xu)
        yu = np.array(yu)
        xl = np.array(xl)
        yl = np.array(yl)

        # ensure leading edge at zero
        xu -= xu[0]
        xl -= xl[0]
        yu -= yu[0]
        yl -= yl[0]

        # ensure unit chord
        c = xu[-1] - xu[0]
        xu /= c
        xl /= c
        yu /= c
        yl /= c

        # interpolate onto common grid
        arc = np.linspace(0, math.pi, 100)
        self.x = 0.5 * (1 - np.cos(arc))  # cosine spacing
        self.yu = np.interp(self.x, xu, yu)
        self.yl = np.interp(self.x, xl, yl)

    @classmethod
    def initWithTEtoTEdata(cls, x, y):
        """Factory constructor for data points ordered from trailing edge to trailing edge.

        Parameters
        ----------
        x, y : ndarray, ndarray
            airfoil coordinates starting at trailing edge and
            ending at trailing edge, traversing airfoil in either direction.

        Returns
        -------
        profile : Profile
            initialized Profile object

        Notes
        -----
        It is not necessary to start and end at the same point
        for an airfoil with trailing edge thickness.
        Although, one point should be right at the nose.
        see also notes for :meth:`__init__`

        """

        # parse airfoil data
        x = np.array(x)
        y = np.array(y)

        # separate into 2 halves
        i = np.argmin(x)

        xu = x[i::-1]
        yu = y[i::-1]
        xl = x[i:]
        yl = y[i:]

        # check if coordinates were input in other direction
        if np.mean(y[0:i]) < np.mean(y[i:]):
            temp = yu
            yu = yl
            yl = temp

            temp = xu
            xu = xl
            xl = temp

        return cls(xu, yu, xl, yl)

    @classmethod
    def initWithLEtoLEdata(cls, x, y):
        """Factory constructor for data points ordered from leading edge to leading edge.

        Parameters
        ----------
        x, y : ndarray, ndarray
            airfoil coordinates starting at leading edge and
            ending at leading edge, traversing airfoil in either direction.

        Returns
        -------
        profile : Profile
            initialized Profile object

        Notes
        -----
        x,y data must start and end at the same point.
        see also notes for :meth:`__init__`

        """

        # parse airfoil data
        x = np.array(x)
        y = np.array(y)

        # separate into 2 halves
        for i in range(len(x)):
            if x[i + 1] <= x[i]:
                iuLast = i
                ilLast = i
                if x[i + 1] == x[i]:  # blunt t.e.
                    ilLast = i + 1  # stop at i+1
                break

        xu = x[: iuLast + 1]
        yu = y[: iuLast + 1]
        xl = x[-1 : ilLast - 1 : -1]
        yl = y[-1 : ilLast - 1 : -1]

        # check if coordinates were input in other direction
        if y[1] < y[0]:
            temp = yu
            yu = yl
            yl = temp

            temp = xu
            xu = xl
            xl = temp

        return cls(xu, yu, xl, yl)

    @staticmethod
    def initFromPreCompFile(precompProfileFile):
        """Construct profile from PreComp formatted file

        Parameters
        ----------
        precompProfileFile : str
            path/name of file

        Returns
        -------
        profile : Profile
            initialized Profile object

        """

        return Profile.initFromFile(precompProfileFile, 4, True)

    @staticmethod
    def initFromFile(filename, numHeaderlines, LEtoLE):
        """Construct profile from a generic form text file (see Notes)

        Parameters
        ----------
        filename : str
            name/path of input file
        numHeaderlines : int
            number of header rows in input file
        LEtoLE : boolean
            True if data is ordered from leading-edge to leading-edge
            False if from trailing-edge to trailing-edge

        Returns
        -------
        profile : Profile
            initialized Profile object

        Notes
        -----
        file should be of the form:

        header row
        header row
        x1 y1
        x2 y2
        x3 y3
        .  .
        .  .
        .  .

        where any number of header rows can be used.

        """

        # open file
        f = open(filename, "r")

        # skip through header
        for i in range(numHeaderlines):
            f.readline()

        # loop through
        x = []
        y = []

        for line in f:
            if not line.strip():
                break  # break if empty line
            data = line.split()
            x.append(float(data[0]))
            y.append(float(data[1]))

        f.close()

        # close nose if LE to LE
        if LEtoLE:
            x.append(x[0])
            y.append(y[0])
            return Profile.initWithLEtoLEdata(x, y)

        else:
            return Profile.initWithTEtoTEdata(x, y)

    def _preCompFormat(self):
        """
        docstring
        """

        # check if they share a common trailing edge point
        te_same = self.yu[-1] == self.yl[-1]

        # count number of points
        nu = len(self.x)
        if te_same:
            nu -= 1
        nl = len(self.x) - 1  # they do share common leading-edge
        n = nu + nl

        # initialize
        x = np.zeros(n)
        y = np.zeros(n)

        # leading edge round to leading edge
        x[0:nu] = self.x[0:nu]
        y[0:nu] = self.yu[0:nu]
        x[nu:] = self.x[:0:-1]
        y[nu:] = self.yl[:0:-1]

        return x, y

    def locationOfMaxThickness(self):
        """Find location of max airfoil thickness

        Returns
        -------
        x : float
            x location of maximum thickness
        yu : float
            upper surface y location of maximum thickness
        yl : float
            lower surface y location of maximum thickness

        Notes
        -----
        uses :ref:`airfoil-aligned coordinate system <blade_airfoil_coord>`

        """

        idx = np.argmax(self.yu - self.yl)
        return (self.x[idx], self.yu[idx], self.yl[idx])

    def blend(self, other, weight):
        """Blend this profile with another one with the specified weighting.

        Parameters
        ----------
        other : Profile
            another Profile to blend with
        weight : float
            blending parameter between 0 and 1.  0 returns self, whereas 1 returns other.

        Returns
        -------
        profile : Profile
            a blended profile

        """

        # blend coordinates
        yu = self.yu + weight * (other.yu - self.yu)
        yl = self.yl + weight * (other.yl - self.yl)

        return Profile(self.x, yu, self.x, yl)

    @property
    def tc(self):
        """thickness to chord ratio of the Profile"""
        return max(self.yu - self.yl)

    def set_tc(self, new_tc):

        factor = new_tc / self.tc

        self.yu *= factor
        self.yl *= factor


class PreCompWriter:
    def __init__(self, dir_out, materials, upper, lower, webs, profile, chord, twist, p_le):
        self.dir_out = dir_out

        self.materials = materials
        self.upper = upper
        self.lower = lower
        self.webs = webs
        self.profile = profile

        self.chord = chord
        self.twist = twist
        self.p_le = p_le

        if not os.path.exists(dir_out):
            os.makedirs(dir_out)

    def execute(self):

        flist_layup = self.writePreCompLayup()
        flist_profile = self.writePreCompProfile()
        self.writePreCompMaterials()
        self.writePreCompInput(flist_layup, flist_profile)

    def writePreCompMaterials(self):

        text = []
        text.append("\n")
        text.append("Mat_Id     E1           E2          G12       Nu12     Density      Mat_Name\n")
        text.append(" (-)      (Pa)         (Pa)        (Pa)       (-)      (Kg/m^3)       (-)\n")

        for i, mat in enumerate(self.materials):
            text.append("%d %e %e %e %f %e %s\n" % (i + 1, mat.E1, mat.E2, mat.G12, mat.nu12, mat.rho, mat.name))

        fout = os.path.join(self.dir_out, "materials.inp")
        f = open(fout, "w")
        for outLine in text:
            f.write(outLine)
        f.close()

    def writePreCompLayup(self):
        f_out = []

        def write_layup_sectors(cs, web):
            text = []
            for i, (n_plies, t, theta, mat_idx) in enumerate(zip(cs.n_plies, cs.t, cs.theta, cs.mat_idx)):
                if web:
                    text.extend(["\n", "web_num    no of laminae (N_weblams)    Name of stack:\n"])
                else:
                    text.extend(
                        [
                            "...........................................................................\n",
                            "Sect_num    no of laminae (N_laminas)          STACK:\n",
                        ]
                    )
                n_lamina = len(n_plies)
                text.append("%d %d\n" % (i + 1, n_lamina))
                text.extend(
                    [
                        "\n",
                        "lamina    num of  thickness   fibers_direction  composite_material ID\n",
                        "number    plies   of ply (m)       (deg)               (-)\n",
                    ]
                )
                if web:
                    text.append("wlam_num N_Plies   w_tply       Tht_Wlam            Wmat_Id\n")
                else:
                    text.append("lam_num  N_plies    Tply         Tht_lam            Mat_id\n")

                for j, (plies_j, t_j, theta_j, mat_idx_j) in enumerate(zip(n_plies, t, theta, mat_idx + 1)):
                    text.append("%d %d %e %.1f %d\n" % (j + 1, plies_j, t_j, theta_j, mat_idx_j))
            return text

        for idx, (lower_i, upper_i, webs_i) in enumerate(zip(self.lower, self.upper, self.webs)):

            text = []
            text.append("Composite laminae lay-up inside the blade section\n")
            text.append("\n")
            text.append("*************************** TOP SURFACE ****************************\n")
            # number of sectors
            n_sector = len(upper_i.loc) - 1
            text.append("%d                N_scts(1):  no of sectors on top surface\n" % n_sector)
            text.extend(["\n", "normalized chord location of  nodes defining airfoil sectors boundaries (xsec_node)\n"])
            locU = upper_i.loc
            text.append(" ".join(["%f" % i for i in locU]) + "\n")
            text.extend(write_layup_sectors(upper_i, False))

            text.extend(["\n", "\n", "*************************** BOTTOM SURFACE ****************************\n"])
            n_sector = len(lower_i.loc) - 1
            text.append("%d                N_scts(1):  no of sectors on top surface\n" % n_sector)
            text.extend(["\n", "normalized chord location of  nodes defining airfoil sectors boundaries (xsec_node)\n"])
            locU = lower_i.loc
            text.append(" ".join(["%f" % i for i in locU]) + "\n")
            text.extend(write_layup_sectors(lower_i, False))

            text.extend(
                [
                    "\n",
                    "\n",
                    "**********************************************************************\n",
                    "Laminae schedule for webs (input required only if webs exist at this section):\n",
                ]
            )
            ########## Webs ##########
            text.extend(write_layup_sectors(webs_i, True))

            fname = os.path.join(self.dir_out, "layup_%00d.inp" % idx)
            f = open(fname, "w")
            for outLine in text:
                f.write(outLine)
            f.close()
            f_out.append(fname)

        return f_out

    def writePreCompProfile(self):
        f_out = []
        for idx, profile_i in enumerate(self.profile):
            # idx = 0
            # profile_i = profile[idx]
            text = []

            text.append(
                "%d                      N_af_nodes :no of airfoil nodes, counted clockwise starting\n"
                % len(profile_i.x)
            )
            text.append("                      with leading edge (see users' manual, fig xx)\n")
            text.append("\n")
            text.append(" Xnode      Ynode   !! chord-normalized coordinated of the airfoil nodes\n")

            x_all = np.concatenate((profile_i.x, np.flip(profile_i.x, 0)))
            y_all = np.concatenate((profile_i.yu, np.flip(profile_i.yl, 0)))

            if max(y_all) > 1.0:
                print(idx)

            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.plot(x_all, y_all)
            # plt.savefig('test.png')

            for x, y in zip(x_all, y_all):
                text.append("%f %f\n" % (x, y))

            fname = os.path.join(self.dir_out, "shape_%d.inp" % idx)
            f = open(fname, "w")
            for outLine in text:
                f.write(outLine)
            f.close()
            f_out.append(fname)

        return f_out

    def writePreCompInput(self, flist_layup, flist_profile):

        for idx in range(0, len(flist_layup)):

            chord = self.chord[idx]
            twist = self.twist[idx]
            p_le = self.p_le[idx]
            webs_i = self.webs[idx]
            layup_i = os.path.split(os.path.abspath(flist_layup[idx]))[1]
            profile_i = os.path.split(os.path.abspath(flist_profile[idx]))[1]

            text = []
            text.append("*****************  main input file for PreComp *****************************\n")
            text.append("Sample Composite Blade Section Properties\n")
            text.append("\n")
            text.append("General information -----------------------------------------------\n")
            text.append("1                Bl_length   : blade length (m)\n")
            text.append("2                N_sections  : no of blade sections (-)\n")
            text.append(
                "%d                N_materials : no of materials listed in the materials table (material.inp)\n"
                % len(self.materials)
            )
            text.append("3                Out_format  : output file   (1: general format, 2: BModes-format, 3: both)\n")
            text.append("f                TabDelim     (true: tab-delimited table; false: space-delimited table)\n")
            text.append("\n")
            text.append("Blade-sections-specific data --------------------------------------\n")
            text.append("Sec span     l.e.     chord   aerodynamic   af_shape    int str layup\n")
            text.append("location   position   length    twist         file          file\n")
            text.append("Span_loc    Le_loc    Chord    Tw_aero   Af_shape_file  Int_str_file\n")
            text.append("  (-)        (-)       (m)    (degrees)       (-)           (-)\n")
            text.append("\n")
            text.append("%.2f %f %e %f %s %s\n" % (0.0, p_le, chord, twist, profile_i, layup_i))
            text.append("%.2f %f %e %f %s %s\n" % (1.0, p_le, chord, twist, profile_i, layup_i))
            text.append("\n")
            text.append("Webs (spars) data  --------------------------------------------------\n")
            text.append("\n")
            text.append(
                "%d                Nweb        : number of webs (-)  ! enter 0 if the blade has no webs\n"
                % len(webs_i.loc)
            )
            text.append(
                "1                Ib_sp_stn   : blade station number where inner-most end of webs is located (-)\n"
            )
            text.append(
                "2                Ob_sp_stn   : blade station number where outer-most end of webs is located (-)\n"
            )
            text.append("\n")
            text.append("Web_num   Inb_end_ch_loc   Oub_end_ch_loc (fraction of chord length)\n")
            for i, loc in enumerate(webs_i.loc):
                text.append("%d %f %f\n" % (i + 1, loc, loc))

            fname = os.path.join(self.dir_out, "input_%d.inp" % idx)
            f = open(fname, "w")
            for outLine in text:
                f.write(outLine)
            f.close()


if __name__ == "__main__":

    import os

    # geometry
    r_str = [
        1.5,
        1.80135,
        1.89975,
        1.99815,
        2.1027,
        2.2011,
        2.2995,
        2.87145,
        3.0006,
        3.099,
        5.60205,
        6.9981,
        8.33265,
        10.49745,
        11.75205,
        13.49865,
        15.84795,
        18.4986,
        19.95,
        21.99795,
        24.05205,
        26.1,
        28.14795,
        32.25,
        33.49845,
        36.35205,
        38.4984,
        40.44795,
        42.50205,
        43.49835,
        44.55,
        46.49955,
        48.65205,
        52.74795,
        56.16735,
        58.89795,
        61.62855,
        63.0,
    ]
    chord_str = [
        3.386,
        3.386,
        3.386,
        3.386,
        3.386,
        3.386,
        3.386,
        3.386,
        3.387,
        3.39,
        3.741,
        4.035,
        4.25,
        4.478,
        4.557,
        4.616,
        4.652,
        4.543,
        4.458,
        4.356,
        4.249,
        4.131,
        4.007,
        3.748,
        3.672,
        3.502,
        3.373,
        3.256,
        3.133,
        3.073,
        3.01,
        2.893,
        2.764,
        2.518,
        2.313,
        2.086,
        1.419,
        1.085,
    ]
    theta_str = [
        13.31,
        13.31,
        13.31,
        13.31,
        13.31,
        13.31,
        13.31,
        13.31,
        13.31,
        13.31,
        13.31,
        13.31,
        13.31,
        13.31,
        13.31,
        12.53,
        11.48,
        10.63,
        10.16,
        9.59,
        9.01,
        8.4,
        7.79,
        6.54,
        6.18,
        5.36,
        4.75,
        4.19,
        3.66,
        3.4,
        3.13,
        2.74,
        2.32,
        1.53,
        0.86,
        0.37,
        0.11,
        0.0,
    ]
    le_str = [
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.498,
        0.497,
        0.465,
        0.447,
        0.43,
        0.411,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
    ]
    web1 = np.array(
        [
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            0.4114,
            0.4102,
            0.4094,
            0.3876,
            0.3755,
            0.3639,
            0.345,
            0.3342,
            0.3313,
            0.3274,
            0.323,
            0.3206,
            0.3172,
            0.3138,
            0.3104,
            0.307,
            0.3003,
            0.2982,
            0.2935,
            0.2899,
            0.2867,
            0.2833,
            0.2817,
            0.2799,
            0.2767,
            0.2731,
            0.2664,
            0.2607,
            0.2562,
            0.1886,
            -1.0,
        ]
    )
    web2 = np.array(
        [
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            0.5886,
            0.5868,
            0.5854,
            0.5508,
            0.5315,
            0.5131,
            0.4831,
            0.4658,
            0.4687,
            0.4726,
            0.477,
            0.4794,
            0.4828,
            0.4862,
            0.4896,
            0.493,
            0.4997,
            0.5018,
            0.5065,
            0.5101,
            0.5133,
            0.5167,
            0.5183,
            0.5201,
            0.5233,
            0.5269,
            0.5336,
            0.5393,
            0.5438,
            0.6114,
            -1.0,
        ]
    )
    web3 = np.array(
        [
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
        ]
    )
    precurve_str = np.zeros_like(r_str)
    presweep_str = np.zeros_like(r_str)

    # -------- materials and composite layup  -----------------
    basepath = os.path.join("5MW_files", "5MW_PreCompFiles")

    materials = Orthotropic2DMaterial.listFromPreCompFile(os.path.join(basepath, "materials.inp"))

    ncomp = len(r_str)
    upper = [0] * ncomp
    lower = [0] * ncomp
    webs = [0] * ncomp
    profile = [0] * ncomp

    # # web 1
    # ib_idx = 7
    # ob_idx = 36
    # ib_webc = 0.4114
    # ob_webc = 0.1886

    # web1 = web_loc(r_str, chord_str, le_str, ib_idx, ob_idx, ib_webc, ob_webc)

    # # web 2
    # ib_idx = 7
    # ob_idx = 36
    # ib_webc = 0.5886
    # ob_webc = 0.6114

    # web2 = web_loc(r_str, chord_str, le_str, ib_idx, ob_idx, ib_webc, ob_webc)

    for i in range(ncomp):

        webLoc = []
        if web1[i] != -1:
            webLoc.append(web1[i])
        if web2[i] != -1:
            webLoc.append(web2[i])
        if web3[i] != -1:
            webLoc.append(web3[i])

        upper[i], lower[i], webs[i] = CompositeSection.initFromPreCompLayupFile(
            os.path.join(basepath, "layup_" + str(i + 1) + ".inp"), webLoc, materials
        )
        profile[i] = Profile.initFromPreCompFile(os.path.join(basepath, "shape_" + str(i + 1) + ".inp"))
    # --------------------------------------

    precomp = PreComp(
        r_str, chord_str, theta_str, le_str, precurve_str, presweep_str, profile, materials, upper, lower, webs
    )

    # evalute section properties
    EA, EIxx, EIyy, EIxy, GJ, rhoA, rhoJ, x_ec_str, y_ec_str = precomp.sectionProperties()

    import matplotlib.pyplot as plt

    r_str = np.array(r_str)
    rstar = (r_str - r_str[0]) / (r_str[-1] - r_str[0])

    plt.figure(1)
    plt.semilogy(rstar, EIxx)
    plt.xlabel("blade fraction")
    plt.ylabel("Edgewise Stiffness ($N m^2$)")

    plt.figure(2)
    plt.semilogy(rstar, EIyy)
    plt.xlabel("blade fraction")
    plt.ylabel("Flapwise Stiffness ($N m^2$)")

    plt.figure(3)
    plt.semilogy(rstar, EA)
    plt.figure(4)
    plt.semilogy(rstar, EIxy)
    plt.figure(5)
    plt.semilogy(rstar, GJ)
    plt.figure(6)
    plt.semilogy(rstar, rhoA)
    plt.figure(7)
    plt.semilogy(rstar, rhoJ)
    plt.figure(8)
    plt.plot(rstar, x_ec_str)
    plt.figure(9)
    plt.plot(rstar, y_ec_str)
    plt.figure(10)
    plt.plot(rstar, precomp.x_ec_nose)
    plt.figure(11)
    plt.plot(rstar, precomp.y_ec_nose)

    plt.show()
