#!/usr/bin/env python
# encoding: utf-8
"""
rotorstruc_mdao.py

Created by Andrew Ning on 2013-05-24.
Copyright (c) NREL. All rights reserved.
"""

import numpy as np
import os
# from math import log10

from openmdao.main.datatypes.api import Array, Float, List, Str

from components import RotorStrucBase
from wisdem.rotor import RotorStruc, PreComp, Orthotropic2DMaterial, \
    CompositeSection, Profile
from wisdem.common import _akima




class RotorStrucComp(RotorStrucBase):
    """docstring for RotorStrucComp"""

    # geometry
    r = Array(iotype='in', units='m', desc='radial locations where composite sections are defined (from hub to tip)')
    chord = Array(iotype='in', units='m', desc='chord length at each section')
    theta = Array(iotype='in', units='deg', desc='twist angle at each section (positive decreases angle of attack)')
    le_location = Array(iotype='in', desc='location of pitch axis relative to leading edge in normalized chord units')
    webLoc = List(Array, iotype='in', desc='locations of shear webs')

    # composite section definition
    base_path = Str(iotype='in', desc='path to directory containing files')
    compSec = List(Str, iotype='in', desc='names of composite section layup files')
    profile = List(Str, iotype='in', desc='names of profile shape files')
    materials = Str(iotype='in', desc='name of materials file')

    # panel buckling
    panel_buckling_idx = Array(iotype='in', dtype=np.int, desc='index of sections critical for panel buckling')

    avgOmega = Float(iotype='in', units='rpm', desc='average rotation speed across wind speed distribution')


    # outputs

    f1 = Float(iotype='out', units='Hz', desc='first natural frequency')
    f2 = Float(iotype='out', units='Hz', desc='second natural frequency')

    tip_deflection = Float(iotype='out', units='m', desc='deflection of blade tip in azimuth-aligned x-direction')

    strain_upper = Array(iotype='out', desc='axial strain along upper surface of blade')
    strain_lower = Array(iotype='out', desc='axial strain along lower surface of blade')
    strain_buckling = Array(iotype='out', desc='maximum compressive strain along blade before panel buckling')
    r_strain = Array(iotype='out', units='m', desc='radial locations along blade where strain is evaluted')

    rootFatigue = Float(iotype='out')




    def _combineLoads(self, rotor, azimuth, pitch, r_aero, Px_aero, Py_aero, Pz_aero):

        r_w, Px_w, Py_w, Pz_w = rotor.weightLoads(self.tilt, azimuth, self.precone, pitch)

        Px = Px_w + _akima.interpolate(r_aero, Px_aero, r_w)
        Py = Py_w + _akima.interpolate(r_aero, Py_aero, r_w)
        Pz = Pz_w + _akima.interpolate(r_aero, Pz_aero, r_w)

        return r_w, Px, Py, Pz


    def execute(self):

        # rename
        mat_init = Orthotropic2DMaterial.initFromPreCompFile
        comp_init = CompositeSection.initFromPreCompLayupFile
        pf_init = Profile.initFromPreCompFile

        # init materials
        materials = mat_init(os.path.join(self.base_path, self.materials))

        # initialize section properties and profile
        ncomp = len(self.r)
        compSec = [0]*ncomp
        profile = [0]*ncomp

        for i in range(ncomp):

            compSec[i] = comp_init(os.path.join(self.base_path, self.compSec[i]), self.webLoc[i], materials)
            profile[i] = pf_init(os.path.join(self.base_path, self.profile[i]))


        # initial section analysis
        sectionanalysis = PreComp(self.r, self.chord, self.theta, profile, compSec, self.le_location, materials)

        # initialize rotorstruc
        rotor = RotorStruc(sectionanalysis, self.B)

        # mass, inertia, natural frequencies
        mp = self.mass_properties
        mp.mass = rotor.mass()
        mp.Ixx, mp.Iyy, mp.Izz, mp.Ixy, mp.Ixz, mp.Iyz = rotor.momentsOfInertia()
        self.f1, self.f2 = rotor.naturalFrequencies(2)

        # deflection at rated loads
        lr = self.loads_rated
        r, Px, Py, Pz = self._combineLoads(rotor, lr.azimuth, lr.pitch, lr.r, lr.Px, lr.Py, lr.Pz)

        self.tip_deflection = rotor.tipDeflection(r, Px, Py, Pz, lr.pitch, self.precone)


        # strain at extreme loads
        le = self.loads_extreme
        r, Px, Py, Pz = self._combineLoads(rotor, le.azimuth, le.pitch, le.r, le.Px, le.Py, le.Pz)

        self.strain_upper, self.strain_lower = rotor.axialStrainAlongBlade(r, Px, Py, Pz)


        # buckling strain
        self.strain_buckling = rotor.panelBucklingStrain(self.panel_buckling_idx)

        self.r_strain = rotor.r

        # # ignore this rootStrain stuff for now
        # rootStrain = rotor.rootStrainDueToGravityLoads(self.tilt, self.precone)
        # rootStrain = -rootStrain  # in compression make positive

        # ultimateStrainRoot = 0.01
        # N = self.avgOmega * 60*24*365*20  # assume 20 year life
        # SfN = 1 - 0.1*log10(N)

        # fatigueStrain = ultimateStrainRoot*SfN

        # self.rootFatigue = rootStrain / fatigueStrain





if __name__ == '__main__':

    rotor = RotorStrucComp()


    # geometry
    rotor.r = [1.5, 1.80135, 1.89975, 1.99815, 2.1027, 2.2011, 2.2995, 2.87145, 3.0006, 3.099, 5.60205, 6.9981, 8.33265, 10.49745, 11.75205, 13.49865, 15.84795, 18.4986, 19.95, 21.99795, 24.05205, 26.1, 28.14795, 32.25, 33.49845, 36.35205, 38.4984, 40.44795, 42.50205, 43.49835, 44.55, 46.49955, 48.65205, 52.74795, 56.16735, 58.89795, 61.62855, 63.]
    rotor.chord = [3.386, 3.386, 3.386, 3.386, 3.386, 3.386, 3.386, 3.386, 3.387, 3.39, 3.741, 4.035, 4.25, 4.478, 4.557, 4.616, 4.652, 4.543, 4.458, 4.356, 4.249, 4.131, 4.007, 3.748, 3.672, 3.502, 3.373, 3.256, 3.133, 3.073, 3.01, 2.893, 2.764, 2.518, 2.313, 2.086, 1.419, 1.085]
    rotor.theta = [13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 12.53, 11.48, 10.63, 10.16, 9.59, 9.01, 8.4, 7.79, 6.54, 6.18, 5.36, 4.75, 4.19, 3.66, 3.4, 3.13, 2.74, 2.32, 1.53, 0.86, 0.37, 0.11, 0.0]
    rotor.le_location = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.498, 0.497, 0.465, 0.447, 0.43, 0.411, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
    rotor.B = 3

    # -------- materials and composite layup  -----------------

    rotor.base_path = os.path.join(os.path.expanduser('~'), 'Dropbox', 'NREL', '5MW_files', '5MW_PrecompFiles')

    rotor.materials = 'materials.inp'

    ncomp = len(rotor.r)
    rotor.compSec = ['0']*ncomp
    rotor.profile = ['0']*ncomp
    rotor.webLoc = [np.array([0.0])]*ncomp
    nweb = [0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0]

    for i in range(ncomp):

        if nweb[i] == 3:
            rotor.webLoc[i] = [0.3, 0.6]  # the last "shear web" is just a flat trailing edge - negligible
        elif nweb[i] == 2:
            rotor.webLoc[i] = [0.3, 0.6]
        else:
            rotor.webLoc[i] = []

        rotor.compSec[i] = 'layup_' + str(i+1) + '.inp'
        rotor.profile[i] = 'shape_' + str(i+1) + '.inp'
    # --------------------------------------

    rotor.panel_buckling_idx = [2]*ncomp

    rotor.loads_rated.r = np.array([1.5, 2.87, 5.6, 8.33, 11.75, 15.85, 19.95, 24.05, 28.15, 32.25, 36.35, 40.45, 44.55, 48.65, 52.75, 56.17, 58.9, 61.63, 63.0])
    rotor.loads_rated.Px = np.array([0.0, 125.510260449, 153.614763073, 126.35429001, 1429.20151242, 1803.0623575, 2548.46076556, 3037.87267635, 3406.12415903, 4086.52292262, 4686.31407544, 5195.888152, 5304.08245875, 5601.68486066, 5761.34480779, 5696.54112365, 5476.64616728, 4830.80958981, 0.0])
    rotor.loads_rated.Py = np.array([0.0, 64.4118990209, 124.540838416, 145.058991462, -173.652545714, -310.776311111, -529.545771254, -550.636486066, -536.644723776, -598.027125857, -641.974083013, -690.853541522, -744.975274979, -783.817097287, -801.797269851, -782.866270502, -727.519248389, -538.431415264, 0.0])
    rotor.loads_rated.Pz = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    rotor.loads_rated.pitch = 0.0
    rotor.loads_rated.azimuth = 180.0

    rotor.loads_extreme.r = rotor.loads_rated.r
    rotor.loads_extreme.Px = np.array([0.0, 5082.52942406, 5666.72820844, 4314.30802261, 24318.8847925, 22267.7970366, 21212.2279797, 19011.0378466, 17517.9097294, 16463.9405352, 15639.5560444, 14327.9822593, 12891.5567957, 11573.7297655, 10243.4591476, 9033.28110847, 8035.39324769, 6947.60433306, 0.0])
    rotor.loads_extreme.Py = np.array([0.0, 1199.58592196, 1337.46936127, 1018.26919928, -2118.54580207, -1659.94318636, -1614.67193585, -1540.56672799, -1033.16780249, -946.06399363, -599.227937055, -540.128556585, -434.642110875, -393.340626614, -350.851496816, -311.292985993, -278.176120407, -241.535559089, 0.0])
    rotor.loads_extreme.Pz = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    rotor.loads_extreme.pitch = 0.0
    rotor.loads_extreme.azimuth = 90.0

    rotor.precone = 0.0
    rotor.tilt = 0.0

    rotor.avgOmega = 7.91859872182

    rotor.execute()

    print rotor.mass_properties.mass
    print rotor.f1, rotor.f2
    print rotor.tip_deflection

    import matplotlib.pyplot as plt
    plt.plot(rotor.r, rotor.strain_upper)
    plt.plot(rotor.r, rotor.strain_lower)
    plt.plot(rotor.r, rotor.strain_buckling)
    plt.ylim([-5e-3, 5e-3])
    plt.show()
