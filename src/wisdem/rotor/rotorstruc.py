#!/usr/bin/env python
# encoding: utf-8
"""
pbeam_wrapper.py

Created by Andrew Ning on 2012-02-28.
Copyright (c)  NREL. All rights reserved.
"""

import numpy as np
import math
from zope.interface import Interface, Attribute

from wisdem.common import DirectionVector, _pBEAM
from wisdem.common.utilities import cosd


# ------------------
#  Interfaces
# ------------------

class SectionStrucInterface(Interface):
    """A class that encapsulates airfoil *structural* properties
    Evaluates mass and stiffness properties of structure at appropriate
    sections"""

    theta = Attribute(':ref:`twist angle <twist_angle>` of each section (deg)')

    def sectionProperties():
        """Get the mass and stiffness properties of the cross-section at specified locations along blade

        Returns
        -------
        r : ndarray (m)
            axial locations along blade where section properties are defined
        EA : ndarray (N)
            axial stiffness
        EIxx : ndarray (N*m^2)
            edgewise stiffness (bending about :ref:`x-direction of airfoil aligned coordinate system <blade_airfoil_coord>`)
        EIyy : ndarray (N*m^2)
            flatwise stiffness (bending about y-direction of airfoil aligned coordinate system)
        EIxy : ndarray (N*m^2)
            coupled flap-edge stiffness
        GJ : ndarray (N*m^2)
            torsional stiffness (about axial z-direction of airfoil aligned coordinate system)
        rhoA : ndarray (kg/m)
            mass per unit length
        rhoJ : ndarray (kg*m)
            polar mass moment of inertia per unit length
        x_ec_str : ndarray (m)
            x-distance to elastic center from point about which above structural properties are computed (airfoil aligned coordinate system)
        y_ec_str : ndarray (m)
            y-distance to elastic center from point about which above structural properties are computed

        Notes
        -----
        All directions are defined relative to the :ref:`airfoil-aligned coordinate system <blade_airfoil_coord>`


        """


    def criticalStrainLocations():
        """Get locations where strain should be evaluating along the blade

        Returns
        -------
        x : ndarray (m)
            x-coordinates
        y : ndarray (m)
            y-coordinates
        z : ndarray (m)
            z-coordinates

        Notes
        -----
        All coordinates use the directions form the airfoil-aligned
        coordinate system and are defined relative to the elastic center.

        """


# ------------------
#  Main Class
# ------------------


class RotorStruc:
    """Structural model of a wind turbine rotor using pBEAM, a beam finite element code."""


    def __init__(self, blade, nblades=3):
        """Constructor

        Parameters
        ----------
        blade : SectionStrucInterface
            any object that implements SectionStrucInterface
        nblades : int
            number of blades

        """

        self.sectionstruc = blade
        self.nblades = nblades

        # extract section properties
        (r, EA, EIxx, EIyy, EIxy, GJ, rhoA, rhoJ, x_ec_str, y_ec_str) = blade.sectionProperties()
        nsec = len(r)

        self.nsec = len(r)
        self.r = r
        self.theta = blade.theta

        # translate to elastic center and rotate to principal axes
        theta = np.zeros(nsec)  # rotation angle from airfoil coordinate system to principal
        EI11 = np.zeros(nsec)
        EI22 = np.zeros(nsec)

        for i in range(nsec):

            # translate to elastic center
            EI = np.array([EIxx[i], EIyy[i], EIxy[i]]) + \
                np.array([-y_ec_str[i]**2, -x_ec_str[i]**2, -x_ec_str[i]*y_ec_str[i]])*EA[i]

            # rotate to principal axes  (sign change due to coordinate system definition)
            if (EI[0] == EI[1]):
                theta[i] = math.pi/4.0
            else:
                theta[i] = 0.5*math.atan(2.0*EI[2] / (EI[0]-EI[1]))
            EI11[i] = EI[0] + EI[2]*math.tan(theta[i])
            EI22[i] = EI[1] - EI[2]*math.tan(theta[i])


        # save for distributed weight
        self.rhoA = rhoA

        # create finite element objects
        p_section = _pBEAM.SectionData(nsec, r, EA, EI11, EI22, GJ, rhoA, rhoJ)
        p_loads = _pBEAM.Loads(nsec)  # no loads
        p_tip = _pBEAM.TipData()  # no tip mass
        k = np.array([float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf')])
        p_base = _pBEAM.BaseData(k, float('inf'))  # rigid base

        # create blade object
        self.blade = _pBEAM.Beam(p_section, p_loads, p_tip, p_base)



        # create utility functions for translation/rotation

        def rotateFromAirfoilXYToPrincipal(x, y):

            r1 = x*np.cos(theta) - y*np.sin(theta)
            r2 = x*np.sin(theta) + y*np.cos(theta)
            # r1 = x*np.cos(theta) + y*np.sin(theta)
            # r2 = -x*np.sin(theta) + y*np.cos(theta)

            return r1, r2

        self.rotateFromAirfoilXYToPrincipal = rotateFromAirfoilXYToPrincipal


        def rotateFromPrincipalToAirfoilXY(r1, r2):

            x = r1*np.cos(theta) + r2*np.sin(theta)
            y = -r1*np.sin(theta) + r2*np.cos(theta)

            # x = r1*np.cos(theta) - r2*np.sin(theta)
            # y = r1*np.sin(theta) + r2*np.cos(theta)

            return x, y

        self.rotateFromPrincipalToAirfoilXY = rotateFromPrincipalToAirfoilXY


        # define method to change loading on blade
        def changeLoads(rp, Px, Py, Pz):

            # interpolate then rotate distributed loads
            Px = np.interp(r, rp, Px)
            Py = np.interp(r, rp, Py)
            Pz = np.interp(r, rp, Pz)


            P1, P2 = self.rotateFromAirfoilXYToPrincipal(Px, Py)
            P3 = Pz

            p_loads = _pBEAM.Loads(nsec, P1, P2, P3)

            # create blade object
            self.blade = _pBEAM.Beam(p_section, p_loads, p_tip, p_base)


        self.changeLoads = changeLoads




    def mass(self):
        """rotor mass (**all** of the rotor blades and hub).

        Returns
        -------
        mass : float (kg)
            total mass of rotor

        """

        mass_one_blade = self.blade.mass()

        return self.nblades*mass_one_blade  # TODO: + hubmass(mass_one_blade)


    def momentsOfInertia(self):
        """Estiamtes mass moments and products of inertia of the rotor

        Returns
        -------
        Ixx : float (kg*m**2)
            mass moment of inertia about x-axis
        Iyy : float (kg*m**2)
            mass moment of inertia about y-axis
        Izz : float (kg*m**2)
            mass moment of inertia about z-axis
        Ixy : float (kg*m**2)
            mass x-y product of inertia
        Ixz : float (kg*m**2)
            mass x-z product of inertia
        Iyz : float (kg*m**2)
            mass y-z product of inertia

        Notes
        -----
        All moments of inertia are defined in the
        :ref:`hub-aligned coordinate system <yaw_rotor_coord>`

        Rotor blades are approximated by lines, but with the correct mass distribution
        (i.e. thickness and chord are neglected and radial dimension is assumed to dominate)

        For rotors with 3 or more blades the values are exact (given above approximation),
        but for 2 blades the values are azimuthal averages.

        """

        n = self.nblades
        Ibeam = self.blade.outOfPlaneMomentOfInertia()

        Ixx = n * Ibeam
        Iyy = n/2 * Ibeam  # azimuthal average for 2 blades, exact for 3+
        Izz = Iyy
        Ixy = 0
        Ixz = 0
        Iyz = 0  # azimuthal average for 2 blades, exact for 3+

        return Ixx, Iyy, Izz, Ixy, Ixz, Iyz



    def naturalFrequencies(self, n):
        """Computes the first n natural frequencies of the rotor blades.

        Parameters
        ----------
        n : int
            number of natural frequencies to return

        Returns
        -------
        freq : ndarray (Hz)
            returns first n natural frequencies in order (lowest to highest),
            if n is greater than the number of total degrees of freedom of the structure
            then as many frequencies are available are returned.

        """

        return self.blade.naturalFrequencies(n)



    def displacements(self, r, Px, Py, Pz):
        """Computes the displacement of the structure due to the applied loading.

        Parameters
        ----------
        r : array_like (m)
            radial locations where loads are defined
        Px : array_like (N/m)
            force per unit length in the x-direction (flatwise)
        Py : array_like (N/m)
            force per unit length in the y-direction (edgewise)
        Pz : array_like (N/m)
            force per unit length in the z-direction (axial)


        Returns
        -------
        dx : ndarry (m)
            deflection in x-direction at radial locations defined in loads
        dy : ndarry (m)
            deflection in y-direction
        dz : ndarry (m)
            deflection in z-direction
        dtheta_x : ndarry (deg)
            rotation in theta_x-direction
        dtheta_y : ndarry (deg)
            rotation in theta_y-direction
        dtheta_z : ndarry (deg)
            rotation in theta_z-direction

        Notes
        -----
        Forces and deflections defined in the
        :ref:`airfoil-aligned coordinate system <blade_airfoil_coord>`

        """

        self.changeLoads(r, Px, Py, Pz)
        dr1, dr2, dz, dtheta_r1, dtheta_r2, dtheta_z = self.blade.displacement()

        dx, dy = self.rotateFromPrincipalToAirfoilXY(dr1, dr2)
        dtheta_x, dtheta_y = self.rotateFromPrincipalToAirfoilXY(dtheta_r1, dtheta_r2)

        return dx, dy, dz, dtheta_x, dtheta_y, dtheta_z


    def tipDeflection(self, r, Px, Py, Pz, pitch, precone):
        """tip deflection of blade in x-direction of azimuth-aligned coordinate system


        """
        dx, dy, dz, dtheta_x, dtheta_y, dtheta_z = self.displacements(r, Px, Py, Pz)

        theta = np.array(self.theta) + pitch

        delta = DirectionVector(dx, dy, dz).airfoilToBlade(theta).bladeToAzimuth(precone)

        return delta.x[-1]





    def axialStrainAlongBlade(self, r, Px, Py, Pz):
        """Computes axial strain at top and bottom surface of each section
        at location of maximum thickness.

        Parameters
        ----------
        r : array_like (m)
            radial locations where loads are defined
        Px : array_like (N/m)
            force per unit length in the x-direction (flatwise)
        Py : array_like (N/m)
            force per unit length in the y-direction (edgewise)
        Pz : array_like (N/m)
            force per unit length in the z-direction (axial)


        Returns
        -------
        r : ndarray (m)
            radial locations where strain was evaluated
        strain : ndarray
            corresponding strain

        Notes
        -----
        Forces input in the :ref:`airfoil-aligned coordinate system <blade_airfoil_coord>`

        """

        self.changeLoads(r, Px, Py, Pz)

        # get strain locations
        xu_e, yu_e, xl_e, yl_e = self.sectionstruc.criticalStrainLocations()

        # rotate to principle axes
        xu, yu = self.rotateFromAirfoilXYToPrincipal(xu_e, yu_e)
        xl, yl = self.rotateFromAirfoilXYToPrincipal(xl_e, yl_e)
        xvec = np.concatenate((xu, xl))
        yvec = np.concatenate((yu, yl))
        zvec = np.concatenate((self.r, self.r))

        strain = self.blade.axialStrain(len(xvec), xvec, yvec, zvec)
        strainU = strain[:self.nsec]
        strainL = strain[self.nsec:]

        return strainU, strainL


    def criticalGlobalBucklingLoads(self, r, Px, Py, Pz):
        """
        Estimates the critical global buckling loads of the blade due to axial loads.

        Parameters
        ----------
        r : array_like (m)
            radial locations where loads are defined
        Px : array_like (N/m)
            force per unit length in the x-direction (flatwise)
        Py : array_like (N/m)
            force per unit length in the y-direction (edgewise)
        Pz : array_like (N/m)
            force per unit length in the z-direction (axial)

        Returns
        -------
        Pcr_x : float (N)
            critical axial buckling load for buckling in the x direction (at blade root).
        Pcr_y : float (N)
            critical axial buckling load for buckling in the y direction (at blade root).

        Notes
        -----
        Critical loads are in addition to any specified applied loads.
        Forces input in the :ref:`airfoil-aligned coordinate system <blade_airfoil_coord>`

        """

        self.changeLoads(r, Px, Py, Pz)
        Pcr_p1, Pcr_p2 = self.blade.criticalBucklingLoads()
        Pcr_x, Pcr_y = self.rotateFromPrincipalToAirfoilXY(Pcr_p1, Pcr_p2)
        return Pcr_x[0], Pcr_y[0]  # others are same loads just with different twists



    def panelBucklingStrain(self, sector_idx_array):
        """
        see chapter on Structural Component Design Techniques from Alastair Johnson
        section 6.2: Design of composite panels

        assumes: large aspect ratio, simply supported, uniaxial compression, flat rectangular plate

        """

        return self.sectionstruc.panelBucklingStrain(sector_idx_array)

        # # pf = self.geometry['profile']
        # cs = self.geometry['compSec']
        # chord = self.geometry['chord']

        # eps_crit = np.zeros(self.nsec)

        # for i in range(self.nsec):

        #     # get sector
        #     sector_idx = sector_idx_array[i]
        #     sector = cs[i].secListUpper[sector_idx]  # TODO: lower surface may be the compression one

        #     # chord-wise length of sector
        #     locations = np.concatenate(([0.0], cs[i].sectorLocU, [1.0]))
        #     sector_length = chord[i] * (locations[sector_idx+1] - locations[sector_idx])

        #     # get matrices
        #     A, B, D, totalHeight = sector.compositeMatrices()
        #     E = sector.effectiveEAxial()
        #     D1 = D[0, 0]
        #     D2 = D[1, 1]
        #     D3 = D[0, 1] + 2*D[2, 2]

        #     # use empirical formula
        #     # Nxx = 2 * (math.pi/sector_length)**2 * (math.sqrt(D1*D2) + D3)
        #     Nxx = 3.6 * (math.pi/sector_length)**2 * D[0, 0]

        #     # a = self.geometry['r'][-1] - self.geometry['r'][i]
        #     # AR = a/chord[i]
        #     # if a == 0:
        #     #     Nxx = 0.0
        #     # else:
        #     #     Nxx = (math.pi/a)**2 * (D1 + D2*AR**4 + 2*D3*AR**2)

        #     eps_crit[i] = - Nxx / totalHeight / E

        # return eps_crit


    def weightLoads(self, tilt, azimuth, precone, pitch, g=9.81):
        """Computes distributed weight loads along the blade

        Parameters
        ----------
        g : float, optional (m/s^2)
            acceleration due to gravity

        Returns
        -------
        r : ndarray (m)
            radial locations of distributed loads
        theta : ndarray (deg)
            blade twist at each radial location
        Px : ndarray (m)
            distributed loads in x-direction
        Py : ndarray (m)
            distributed loads in y-direction
        Pz : ndarray (m)
            distributed loads in z-direction

        Notes
        -----
        Distributed loads are returned in :ref:`airfoil coordinate system <blade_airfoil_coord>`

        """

        weight = DirectionVector(0.0, 0.0, -self.rhoA*g)

        theta = np.array(self.theta) + pitch

        P = weight.yawToHub(tilt).hubToAzimuth(azimuth)\
            .azimuthToBlade(precone).bladeToAirfoil(theta)

        return self.r, P.x, P.y, P.z



    def centrifugalLoads(self, Omega, precone, pitch):

        Omega *= math.pi/30.0  # RPM to rad/s

        load = DirectionVector(0.0, 0.0, self.rhoA*Omega**2*self.r*cosd(precone))

        theta = np.array(self.theta) + pitch

        P = load.azimuthToBlade(precone).bladeToAirfoil(theta)

        return self.r, P.x, P.y, P.z


    def rootStrainDueToGravityLoads(self, tilt, precone):
        """edgewise fully-reversed weight loads"""

        azimuth = 90.0  # fully-reversed
        pitch = 0.0

        r_w, Px_w, Py_w, Pz_w = self.weightLoads(tilt, azimuth, precone, pitch)

        strainU, strainL = self.axialStrainAlongBlade(r_w, Px_w, Py_w, Pz_w)

        return strainU[0]


if __name__ == '__main__':

    from precomp import Orthotropic2DMaterial, CompositeSection, Profile, PreComp

    # geometry
    r_str = [1.5, 1.80135, 1.89975, 1.99815, 2.1027, 2.2011, 2.2995, 2.87145, 3.0006, 3.099, 5.60205, 6.9981, 8.33265, 10.49745, 11.75205, 13.49865, 15.84795, 18.4986, 19.95, 21.99795, 24.05205, 26.1, 28.14795, 32.25, 33.49845, 36.35205, 38.4984, 40.44795, 42.50205, 43.49835, 44.55, 46.49955, 48.65205, 52.74795, 56.16735, 58.89795, 61.62855, 63.]
    chord_str = [3.386, 3.386, 3.386, 3.386, 3.386, 3.386, 3.386, 3.386, 3.387, 3.39, 3.741, 4.035, 4.25, 4.478, 4.557, 4.616, 4.652, 4.543, 4.458, 4.356, 4.249, 4.131, 4.007, 3.748, 3.672, 3.502, 3.373, 3.256, 3.133, 3.073, 3.01, 2.893, 2.764, 2.518, 2.313, 2.086, 1.419, 1.085]
    theta_str = [13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 12.53, 11.48, 10.63, 10.16, 9.59, 9.01, 8.4, 7.79, 6.54, 6.18, 5.36, 4.75, 4.19, 3.66, 3.4, 3.13, 2.74, 2.32, 1.53, 0.86, 0.37, 0.11, 0.0]
    le_str = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.498, 0.497, 0.465, 0.447, 0.43, 0.411, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
    nweb_str = [0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0]

    # -------- materials and composite layup  -----------------
    import os
    basepath = os.path.join('5MW_files', '5MW_PrecompFiles')

    materials = Orthotropic2DMaterial.initFromPreCompFile(os.path.join(basepath, 'materials.inp'))

    ncomp = len(r_str)
    compSec = [0]*ncomp
    profile = [0]*ncomp

    for i in range(ncomp):

        if nweb_str[i] == 3:
            webLoc = [0.3, 0.6]  # the last "shear web" is just a flat trailing edge - negligible
        elif nweb_str[i] == 2:
            webLoc = [0.3, 0.6]
        else:
            webLoc = []

        compSec[i] = CompositeSection.initFromPreCompLayupFile(os.path.join(basepath, 'layup_' + str(i+1) + '.inp'), webLoc, materials)
        profile[i] = Profile.initFromPreCompFile(os.path.join(basepath, 'shape_' + str(i+1) + '.inp'))
    # --------------------------------------

    # create object
    precomp = PreComp(r_str, chord_str, theta_str, profile, compSec, le_str, materials)


    rotor = RotorStruc(precomp, nblades=3)

    print rotor.mass()
    print rotor.momentsOfInertia()
    print rotor.naturalFrequencies(5)


    rloads = np.array([1.5, 2.87, 5.6, 8.33, 11.75, 15.85, 19.95, 24.05, 28.15, 32.25, 36.35, 40.45, 44.55, 48.65, 52.75, 56.17, 58.9, 61.63, 63.0])
    Px = np.array([0.0, 94.8364811803, 129.791836351, 120.512309484, 1122.1746305, 1585.54994028, 1920.49807227, 2314.70230308, 2924.45483362, 3428.12192908, 4047.93300221, 4596.32163203, 4899.05923464, 5354.03619714, 5688.5117827, 5756.48798554, 5578.85761707, 4864.89492594, 0.0])
    Py = np.array([-0, 32.6184808326, 87.1047435066, 120.304760507, -455.51080197, -566.053535742, -562.205058849, -565.017513873, -585.92589722, -587.581591266, -598.24836137, -596.139219268, -585.960839247, -577.819907628, -558.829769746, -521.313645675, -455.931002679, -299.576753695, -0])
    Pz = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    dx, dy, dz, dtheta_x, dtheta_y, dtheta_z = rotor.displacements(rloads, Px, Py, Pz)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(r_str, dx)
    plt.plot(r_str, dy)
    plt.plot(r_str, dz)

    strainU, strainL = rotor.axialStrainAlongBlade(rloads, 2*Px, 2*Py, 2*Pz)
    plt.figure()
    plt.plot(r_str, strainU)
    plt.plot(r_str, strainL)

    Pcr_x, Pcr_y = rotor.criticalGlobalBucklingLoads(rloads, Px, Py, Pz)
    print Pcr_x, Pcr_y

    sector_idx_array = [2]*len(r_str)
    strain_buckling = rotor.panelBucklingStrain(sector_idx_array)
    plt.plot(r_str, strain_buckling)
    plt.ylim([-5e-3, 5e-3])

    rstr, Pxw, Pyw, Pzw = rotor.weightLoads(tilt=0.0, azimuth=0.0, precone=0.0, pitch=0.0)
    plt.figure()
    plt.plot(r_str, Pxw)
    plt.plot(r_str, Pyw)
    plt.plot(r_str, Pzw)
    plt.show()

