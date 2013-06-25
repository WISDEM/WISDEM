#!/usr/bin/env python
# encoding: utf-8
"""
rotor_mdao.py

Created by Andrew Ning on 2013-05-24.
Copyright (c) NREL. All rights reserved.
"""

from openmdao.main.api import Assembly
from openmdao.main.datatypes.api import Slot

from rotoraero_mdao import RotorAeroBase
from rotorstruc_mdao import RotorStrucBase





class RotorAssembly(Assembly):

    raero = Slot(RotorAeroBase)
    rstruc = Slot(RotorStrucBase)


    def configure(self):

        self.add('raero', RotorAeroBase())
        self.add('rstruc', RotorStrucBase())

        self.driver.workflow.add(['raero', 'rstruc'])

        self.connect('raero.loads_rated', 'rstruc.loads_rated')
        self.connect('raero.loads_extreme', 'rstruc.loads_extreme')

        self.create_passthrough('raero.V')
        self.create_passthrough('raero.P')
        self.create_passthrough('raero.AEP')
        self.create_passthrough('rstruc.mass_properties')
        # self.create_passthrough('rstruc.Ixx')
        # self.create_passthrough('rstruc.Iyy')
        # self.create_passthrough('rstruc.Izz')
        # self.create_passthrough('rstruc.Ixy')
        # self.create_passthrough('rstruc.Ixz')
        # self.create_passthrough('rstruc.Iyz')

        # TODO: passthrough everything that isn't already connected


class RotorTWISTER(RotorAssembly):

    def replace(self, name, obj):
        super(RotorTWISTER, self).replace(name, obj)

        if name == 'rstruc':
            self.create_passthrough('rstruc.f1')
            self.create_passthrough('rstruc.tip_deflection')
            self.create_passthrough('rstruc.strain_upper')
            self.create_passthrough('rstruc.strain_lower')
            self.create_passthrough('rstruc.strain_buckling')





if __name__ == '__main__':

    from rotoraero_mdao import RotorAeroComp
    from rotorstruc_mdao import RotorStrucComp
    import numpy as np
    import os

    r = [1.5, 1.80135, 1.89975, 1.99815, 2.1027, 2.2011, 2.2995, 2.87145, 3.0006, 3.099, 5.60205, 6.9981, 8.33265, 10.49745, 11.75205, 13.49865, 15.84795, 18.4986, 19.95, 21.99795, 24.05205, 26.1, 28.14795, 32.25, 33.49845, 36.35205, 38.4984, 40.44795, 42.50205, 43.49835, 44.55, 46.49955, 48.65205, 52.74795, 56.16735, 58.89795, 61.62855, 63.]
    chord = [3.386, 3.386, 3.386, 3.386, 3.386, 3.386, 3.386, 3.386, 3.387, 3.39, 3.741, 4.035, 4.25, 4.478, 4.557, 4.616, 4.652, 4.543, 4.458, 4.356, 4.249, 4.131, 4.007, 3.748, 3.672, 3.502, 3.373, 3.256, 3.133, 3.073, 3.01, 2.893, 2.764, 2.518, 2.313, 2.086, 1.419, 1.085]
    theta = [13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 12.53, 11.48, 10.63, 10.16, 9.59, 9.01, 8.4, 7.79, 6.54, 6.18, 5.36, 4.75, 4.19, 3.66, 3.4, 3.13, 2.74, 2.32, 1.53, 0.86, 0.37, 0.11, 0.0]
    pitch_axis = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.498, 0.497, 0.465, 0.447, 0.43, 0.411, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
    B = 3
    precone = 0.0
    tilt = 0.0

    # grids
    r_grid_aero = [2.87, 5.60, 8.33, 11.75, 15.85, 19.95, 24.05, 28.15, 32.25,
                   36.35, 40.45, 44.55, 48.65, 52.75, 56.17, 58.9, 61.63]
    r_grid_struc = r


    # ----- rotor aerodynamics ------
    raero = RotorAeroComp()

    raero.Rhub = r[0]
    raero.Rtip = r[-1]
    raero.r = r_grid_aero
    raero.chord = np.interp(r_grid_aero, r, chord)
    raero.theta = np.interp(r_grid_aero, r, theta)
    raero.B = B


    # atmosphere
    raero.rho = 1.225
    raero.mu = 1.81206e-5


    raero.af_path = os.path.join(os.pardir, 'rotor', '5MW_files', '5MW_AFFiles')


    # load all airfoils
    airfoil_types = [0]*8
    airfoil_types[0] = 'Cylinder1.dat'
    airfoil_types[1] = 'Cylinder2.dat'
    airfoil_types[2] = 'DU40_A17.dat'
    airfoil_types[3] = 'DU35_A17.dat'
    airfoil_types[4] = 'DU30_A17.dat'
    airfoil_types[5] = 'DU25_A17.dat'
    airfoil_types[6] = 'DU21_A17.dat'
    airfoil_types[7] = 'NACA64_A17.dat'

    # place at appropriate radial stations, and convert format
    af_idx = [0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7]

    raero.af_file = ['0']*len(raero.r)
    for i in range(len(raero.r)):
        raero.af_file[i] = airfoil_types[af_idx[i]]


    raero.aeroanalysis_type = 'ccblade'
    raero.drivetrain_type = 'geared'

    raero.Vin = 3.0
    raero.Vout = 25.0
    raero.ratedPower = 5e6
    raero.varSpeed = True
    raero.varPitch = True
    raero.minOmega = 0.0
    raero.maxOmega = 10.0
    raero.tsr_r2 = 7.55


    raero.Umean = 10.0

    raero.V_extreme = 1.4*50.0
    raero.pitch_extreme_target = 0.0
    raero.azimuth_extreme = 90.0


    # ----- rotor structures ------

    rstruc = RotorStrucComp()


    # geometry
    rstruc.r = r_grid_struc
    rstruc.chord = np.interp(r_grid_struc, r, chord)
    rstruc.theta = np.interp(r_grid_struc, r, theta)
    rstruc.le_location = np.interp(r_grid_struc, r, pitch_axis)
    rstruc.nBlades = B

    # -------- materials and composite layup  -----------------

    rstruc.base_path = os.path.join(os.pardir, 'rotor', '5MW_files', '5MW_PrecompFiles')

    rstruc.materials = 'materials.inp'

    ncomp = len(rstruc.r)
    rstruc.compSec = ['0']*ncomp
    rstruc.profile = ['0']*ncomp
    rstruc.webLoc = [np.array([0.0])]*ncomp
    nweb = [0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0]

    for i in range(ncomp):

        if nweb[i] == 3:
            rstruc.webLoc[i] = [0.3, 0.6]  # the last "shear web" is just a flat trailing edge - negligible
        elif nweb[i] == 2:
            rstruc.webLoc[i] = [0.3, 0.6]
        else:
            rstruc.webLoc[i] = []

        rstruc.compSec[i] = 'layup_' + str(i+1) + '.inp'
        rstruc.profile[i] = 'shape_' + str(i+1) + '.inp'
    # --------------------------------------

    rstruc.panel_buckling_idx = [2]*ncomp


    rstruc.precone = precone
    rstruc.tilt = tilt


    # ----- rotor  ------

    rotor = RotorTWISTER()
    rotor.replace('raero', raero)
    rotor.replace('rstruc', rstruc)
    rotor.run()

    print rotor.mass_properties.mass
    print rotor.AEP

    import matplotlib.pyplot as plt
    plt.plot(rotor.V, rotor.P)
    plt.show()

    print rotor.f1
    print rotor.tip_deflection

    import matplotlib.pyplot as plt
    plt.plot(r_grid_struc, rotor.strain_upper)
    plt.plot(r_grid_struc, rotor.strain_lower)
    plt.plot(r_grid_struc, rotor.strain_buckling)
    plt.ylim([-5e-3, 5e-3])
    plt.show()


