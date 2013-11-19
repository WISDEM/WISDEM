#!/usr/bin/env python
# encoding: utf-8
"""
towerstruc_mdao.py

Created by Andrew Ning on 2013-05-29.
Copyright (c) NREL. All rights reserved.
"""

import numpy as np
from openmdao.main.datatypes.api import Array, VarTree, Float

from components import TowerStrucBase
from vartrees import SoilProperties
from wisdem.tower import MonopileStruc, SoilModelCylindricalFoundation



class TowerStruc(TowerStrucBase):

    # geometry
    z = Array(iotype='in', units='m', desc='locations along tower, linear lofting between')
    d = Array(iotype='in', units='m', desc='tower diameter at corresponding locations')
    t = Array(iotype='in', units='m', desc='shell thickness at corresponding locations')
    n = Array(iotype='in', dtype=np.int, desc='number of finite elements between sections.  array length should be one less than ``z``')
    L_reinforced = Float(iotype='in', units='m', desc='reinforcement length for buckling')

    soil = VarTree(SoilProperties(), iotype='in', desc='stiffness properties at base of foundation')

    # material properties
    E = Float(210e9, iotype='in', units='N/m**2', desc='material modulus of elasticity')
    G = Float(80.8e9, iotype='in', units='N/m**2', desc='material shear modulus')
    rho = Float(8500.0, iotype='in', units='kg/m**3', desc='material density')
    sigma_y = Float(450.0e6, iotype='in', units='N/m**2', desc='yield stress')

    # safety factors
    gamma_f = Float(1.35, iotype='in', desc='safety factor on loads')
    gamma_m = Float(1.1, iotype='in', desc='safety factor on materials')
    gamma_n = Float(1.0, iotype='in', desc='safety factor on consequence of failure')

    # outputs
    f1 = Float(iotype='out', units='Hz', desc='first natural frequency')
    f2 = Float(iotype='out', units='Hz', desc='second natural frequency')
    tip_deflection = Float(iotype='out', units='m', desc='deflection of tower top in yaw-aligned +x direction')
    z_stress = Array(iotype='out', units='m', desc='z-locations along tower where stress is evaluted')
    stress = Array(iotype='out', units='N/m**2', desc='von Mises stress along tower on downwind side (yaw-aligned +x).  normalized by yield stress.  includes safety factors.')
    z_buckling = Array(iotype='out', units='m', desc='z-locations along tower where shell buckling is evaluted')
    buckling = Array(iotype='out', desc='a shell buckling constraint.  should be <= 0 for feasibility.  includes safety factors')


    def execute(self):

        mp = self.top_mass_properties
        tip = dict(m=mp.mass,
                   cm=self.top_cm,
                   I=np.array([mp.Ixx, mp.Iyy, mp.Izz, mp.Ixy, mp.Ixz, mp.Iyz]))

        # soil model
        s = self.soil
        idx = np.array([s.rigid_x, s.rigid_theta_x, s.rigid_y, s.rigid_theta_y,
                        s.rigid_z, s.rigid_theta_z], dtype=bool)
        rigid = np.array([0, 1, 2, 3, 4, 5])
        rigid = rigid[idx]
        soil_dict = SoilModelCylindricalFoundation(s.G, s.nu, s.depth, rigid)

        tower = MonopileStruc(self.z, self.d, self.t, self.n, tip, soil_dict,
                              self.E, self.G, self.rho, self.sigma_y)

        self.mass = tower.mass()

        # natural frequncies
        self.f1, self.f2 = tower.naturalFrequencies(2)

        # setup loads
        dl = self.distributed_loads
        tf = self.top_forces
        loads = dl.z, dl.Px, dl.Py, dl.Pz, tf.F, tf.M

        # deflections due to loading from tower top and wind/wave loads
        dx, dy, dz, dthetax, dthetay, dthetaz = \
            tower.displacement(*loads)
        self.tip_deflection = dx[-1]  # in yaw-aligned direction

        # stress
        axial_stress = tower.axialStress(*loads)
        hoop_stress = tower.hoopStress(dl.z, dl.q, self.L_reinforced)
        shear_stress = tower.shearStress(*loads)

        von_mises = tower.vonMisesStress(axial_stress, hoop_stress, shear_stress)
        gamma = self.gamma_f * self.gamma_m * self.gamma_n
        self.z_stress = tower.z
        self.stress = gamma * von_mises[::4] / self.sigma_y  # downwind side (yaw-aligned +x)

        # shell buckling
        gamma_b = self.gamma_m * self.gamma_n
        zb, buckling = tower.shellBuckling(4, axial_stress, hoop_stress, shear_stress,
                                           self.L_reinforced, self.gamma_f, gamma_b)
        self.z_buckling = zb
        self.buckling = buckling[::4]  # yaw-aligned +x side


if __name__ == '__main__':

    t = TowerStruc()

    t.z = [0, 30.0, 73.8, 117.6]         # heights starting from bottom of tower to top (m)
    t.d = [6.0, 6.0, 4.935, 3.87]      # corresponding diameters (m)
    t.t = [0.027, 0.027, 0.023, 0.019]   # corresponding shell thicknesses (m)
    t.t = np.array(t.t)*1.3
    t.n = [5, 5, 5]                  # number of finite elements per section

    t.L_reinforced = 30.0        # reinforcement length of cylindrical sections (m)

    z_surface = 20.0

    t.soil.G = 140e6
    t.soil.nu = 0.4
    t.soil.depth = 10.0
    t.soil.rigid_theta_y = False

    t.distributed_loads.z = np.array([0.0, 4.05517241379, 8.11034482759, 12.1655172414, 16.2206896552, 20.275862069, 24.3310344828, 28.3862068966, 32.4413793103, 36.4965517241, 40.5517241379, 44.6068965517, 48.6620689655, 52.7172413793, 56.7724137931, 60.8275862069, 64.8827586207, 68.9379310345, 72.9931034483, 77.0482758621, 81.1034482759, 85.1586206897, 89.2137931034, 93.2689655172, 97.324137931, 101.379310345, 105.434482759, 109.489655172, 113.544827586, 117.6])
    t.distributed_loads.Px = np.array([102104.885462, 102507.581657, 103721.777656, 105765.915748, 108671.137334, 1214.40838395, 2552.17413924, 3035.82152084, 3337.31126375, 3542.38494025, 3698.58006519, 3821.0768392, 3918.54983011, 3996.47153436, 4058.53909717, 4107.37906758, 4144.93054627, 4172.66909386, 4191.74519722, 4203.07389119, 4207.39498426, 4205.31482325, 4197.33603995, 4183.87922982, 4165.29906834, 4141.89650185, 4113.92811025, 4081.61339417, 4045.14051361, 4004.67085347])
    t.distributed_loads.Py = 0*t.distributed_loads.Px
    t.distributed_loads.Pz = 0*t.distributed_loads.Px
    t.distributed_loads.q = np.array([7754.13965392, 7803.83236038, 7954.32795207, 8209.92750222, 8577.96342287, 286.305426331, 628.803853074, 759.464968238, 850.063655632, 921.42181256, 981.140880371, 1032.94376972, 1078.96026209, 1120.53442822, 1158.57434562, 1193.72518076, 1226.46340514, 1257.1518972, 1286.07403211, 1313.45575043, 1339.48038846, 1364.29896219, 1388.03749155, 1410.80233856, 1432.68417666, 1453.76099505, 1474.10040836, 1493.76145739, 1512.79603073, 1531.25])

    t.top_forces.F = [750000.0, 0.0, 0.0]
    t.top_forces.M = [0.0, 0.0, 0.0]

    t.top_mass_properties.mass = 300000.0
    t.top_mass_properties.Ixx = 2960437.0
    t.top_mass_properties.Iyy = 3253223.0
    t.top_mass_properties.Izz = 3264220.0
    t.top_mass_properties.Ixy = 0.0
    t.top_mass_properties.Ixz = -18400.0
    t.top_mass_properties.Iyz = 0.0


    t.top_cm = np.zeros(3)


    t.execute()

    print t.mass



