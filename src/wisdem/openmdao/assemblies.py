#!/usr/bin/env python
# encoding: utf-8
"""
assemblies.py

Created by Andrew Ning on 2013-05-16.
Copyright (c) NREL. All rights reserved.
"""

import numpy as np
from openmdao.main.api import Assembly
from openmdao.main.datatypes.api import Slot, Float, Array, VarTree, Int

from components import RotorAeroBase, RotorStrucBase, NacelleBase, DummyNacelle, \
    TowerAeroBase, TowerStrucBase, HubLoadsTransfer, MassTransferToTower
from vartrees import Atmosphere


class TurbineBase(Assembly):

    raero = Slot(RotorAeroBase)
    rstruc = Slot(RotorStrucBase)
    nac = Slot(NacelleBase)
    taero = Slot(TowerAeroBase)
    tstruc = Slot(TowerStrucBase)

    atm = VarTree(Atmosphere(), iotype='in', desc='atmospheric properties')

    B = Int(3, iotype='in', desc='number of blades')
    precone = Float(0.0, iotype='in', units='deg', desc='precone angle')
    tilt = Float(0.0, iotype='in', units='deg', desc='rotor tilt angle')
    yaw = Float(0.0, iotype='in', units='deg', desc='yaw angle')

    mass_rotor = Float(iotype='out', units='kg', desc='mass of rotor')
    mass_nacelle = Float(iotype='out', units='kg', desc='mass of nacelle')
    mass_tower = Float(iotype='out', units='kg', desc='mass of tower')


    # TODO: hub mass

    def configure(self):

        # self.add('site', SiteBase())
        self.add('raero', RotorAeroBase())
        self.add('rstruc', RotorStrucBase())
        self.add('nac', NacelleBase())
        self.add('hltransfer', HubLoadsTransfer())
        self.add('mtransfer', MassTransferToTower())
        self.add('taero', TowerAeroBase())
        self.add('tstruc', TowerStrucBase())

        self.driver.workflow.add(['raero', 'rstruc', 'nac', 'hltransfer', 'mtransfer', 'taero', 'tstruc'])

        self.connect('atm', ['raero.atm', 'taero.atm'])

        self.connect('B', ['raero.B', 'rstruc.B'])
        self.connect('precone', ['raero.precone', 'rstruc.precone'])
        self.connect('tilt', ['raero.tilt', 'rstruc.tilt', 'hltransfer.tilt', 'mtransfer.tilt'])
        self.connect('yaw', ['raero.yaw', 'taero.yaw'])

        # self.connect('site.PDF', 'raero.PDF')
        # self.connect('site.CDF', 'raero.CDF')
        # self.connect('site.shearExp', 'raero.shearExp')
        # self.connect('site.V_extreme', 'raero.V_extreme')

        self.connect('raero.loads_rated', 'rstruc.loads_rated')
        self.connect('raero.loads_extreme', 'rstruc.loads_extreme')
        self.connect('raero.hub_rated', 'hltransfer.forces_hubCS')
        self.connect('raero.rated_conditions', 'nac.rotor_rated_conditions')
        self.connect('raero.rated_conditions.V', 'taero.Uhub')
        self.connect('rstruc.mass_properties', 'mtransfer.rotor_mass_properties')
        self.connect('rstruc.mass_properties.mass', 'mass_rotor')

        self.connect('nac.mass_properties', 'mtransfer.nacelle_mass_properties')
        self.connect('nac.mass_properties.mass', 'mass_nacelle')
        self.connect('nac.cm_location', 'mtransfer.nacelle_cm')
        self.connect('nac.hub_location', ['hltransfer.hub_to_tower_top', 'mtransfer.rotor_cm'])

        self.connect('hltransfer.forces_yawCS', 'tstruc.top_forces')

        self.connect('mtransfer.rna_mass_properties', 'tstruc.top_mass_properties')
        self.connect('mtransfer.rna_cm', 'tstruc.top_cm')

        self.connect('taero.wind_wave_loads', 'tstruc.distributed_loads')

        self.connect('tstruc.mass', 'mass_tower')


        self.create_passthrough('raero.V')
        self.create_passthrough('raero.P')
        self.create_passthrough('raero.AEP')




class TurbineTWISTER(TurbineBase):

    # TODO: safety factors for rotor

    # blade structure
    blade_f1 = Float(iotype='out', units='Hz', desc='first natural frequency of rotor blades')
    blade_f2 = Float(iotype='out', units='Hz', desc='second natural frequency of rotor blades')

    blade_tip_deflection = Float(iotype='out', units='m', desc='deflection of blade tip in azimuth-aligned x-direction')

    r_blade_strain = Array(iotype='out', units='m', desc='radial locations along blade where strain is evaluted')
    blade_strain_upper = Array(iotype='out', desc='axial strain along upper surface of blade')
    blade_strain_lower = Array(iotype='out', desc='axial strain along lower surface of blade')
    blade_strain_buckling = Array(iotype='out', desc='maximum compressive strain along blade before panel buckling')


    # tower structure
    tower_f1 = Float(iotype='out', units='Hz', desc='first natural frequency of tower')
    tower_f2 = Float(iotype='out', units='Hz', desc='second natural frequency of tower')

    tower_tip_deflection = Float(iotype='out', units='m', desc='deflection of tower top in yaw-aligned +x direction')

    z_tower_stress = Array(iotype='out', units='m', desc='z-locations along tower where stress is evaluted')
    tower_stress = Array(iotype='out', units='N/m**2', desc='von Mises stress along tower on downwind side (yaw-aligned +x).  normalized by yield stress.  includes safety factors.')
    z_tower_buckling = Array(iotype='out', units='m', desc='z-locations along tower where shell buckling is evaluted')
    tower_buckling = Array(iotype='out', desc='a shell buckling constraint.  should be <= 0 for feasibility.  includes safety factors')




    def replace(self, name, obj):
        super(TurbineTWISTER, self).replace(name, obj)

        if name == 'rstruc':
            self.connect('rstruc.f1', 'blade_f1')
            self.connect('rstruc.f2', 'blade_f2')
            self.connect('rstruc.tip_deflection', 'blade_tip_deflection')
            self.connect('rstruc.r_strain', 'r_blade_strain')
            self.connect('rstruc.strain_upper', 'blade_strain_upper')
            self.connect('rstruc.strain_lower', 'blade_strain_lower')
            self.connect('rstruc.strain_buckling', 'blade_strain_buckling')

        elif name == 'tstruc':

            self.connect('tstruc.f1', 'tower_f1')
            self.connect('tstruc.f2', 'tower_f2')
            self.connect('tstruc.tip_deflection', 'tower_tip_deflection')
            self.connect('tstruc.z_stress', 'z_tower_stress')
            self.connect('tstruc.stress', 'tower_stress')
            self.connect('tstruc.z_buckling', 'z_tower_buckling')
            self.connect('tstruc.buckling', 'tower_buckling')



if __name__ == '__main__':

    import os

    from rotoraero_mdao import RotorAeroComp
    from rotorstruc_mdao import RotorStrucComp
    from toweraero_mdao import TowerAero
    from towerstruc_mdao import TowerStruc

    r = [1.5, 1.80135, 1.89975, 1.99815, 2.1027, 2.2011, 2.2995, 2.87145, 3.0006, 3.099, 5.60205, 6.9981, 8.33265, 10.49745, 11.75205, 13.49865, 15.84795, 18.4986, 19.95, 21.99795, 24.05205, 26.1, 28.14795, 32.25, 33.49845, 36.35205, 38.4984, 40.44795, 42.50205, 43.49835, 44.55, 46.49955, 48.65205, 52.74795, 56.16735, 58.89795, 61.62855, 63.]
    chord = [3.386, 3.386, 3.386, 3.386, 3.386, 3.386, 3.386, 3.386, 3.387, 3.39, 3.741, 4.035, 4.25, 4.478, 4.557, 4.616, 4.652, 4.543, 4.458, 4.356, 4.249, 4.131, 4.007, 3.748, 3.672, 3.502, 3.373, 3.256, 3.133, 3.073, 3.01, 2.893, 2.764, 2.518, 2.313, 2.086, 1.419, 1.085]
    theta = [13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 13.31, 12.53, 11.48, 10.63, 10.16, 9.59, 9.01, 8.4, 7.79, 6.54, 6.18, 5.36, 4.75, 4.19, 3.66, 3.4, 3.13, 2.74, 2.32, 1.53, 0.86, 0.37, 0.11, 0.0]
    pitch_axis = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.498, 0.497, 0.465, 0.447, 0.43, 0.411, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]

    rho = 1.225
    mu = 1.81206e-5
    shearExp = 0.2

    B = 3
    precone = 0.0
    tilt = 20.0
    yaw = 0.0


    # grids
    r_grid_aero = [2.87, 5.60, 8.33, 11.75, 15.85, 19.95, 24.05, 28.15, 32.25,
                   36.35, 40.45, 44.55, 48.65, 52.75, 56.17, 58.9, 61.63]
    r_grid_struc = r


    # tower
    z = [0, 30.0, 73.8, 117.6]
    d = [6.0, 6.0, 4.935, 3.87]
    t = 1.3*np.array([0.027, 0.027, 0.023, 0.019])   # corresponding shell thicknesses (m)

    # # ----- site
    # site = SiteStandardComp()
    # site.wind_turbine_class = 'I'


    # ----- rotor aerodynamics ------
    raero = RotorAeroComp()

    raero.Rhub = r[0]
    raero.Rtip = r[-1]
    raero.r = r_grid_aero
    raero.chord = np.interp(r_grid_aero, r, chord)
    raero.theta = np.interp(r_grid_aero, r, theta)


    raero.af_path = os.path.join(os.path.expanduser('~'), 'Dropbox', 'NREL', '5MW_files', '5MW_AFFiles')

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
    raero.machine_type = 'VSVP'
    raero.wind_turbine_class = 'I'

    raero.Vin = 3.0
    raero.Vout = 25.0
    raero.ratedPower = 5e6

    raero.minOmega = 0.0
    raero.maxOmega = 10.0
    raero.tsr_r2 = 7.55

    raero.pitch_extreme = 0.0
    raero.azimuth_extreme = 90.0



    # ----- rotor structures ------

    rstruc = RotorStrucComp()


    # geometry
    rstruc.r = r_grid_struc
    rstruc.chord = np.interp(r_grid_struc, r, chord)
    rstruc.theta = np.interp(r_grid_struc, r, theta)
    rstruc.le_location = np.interp(r_grid_struc, r, pitch_axis)

        # -------- materials and composite layup  -----------------

    rstruc.base_path = os.path.join(os.path.expanduser('~'), 'Dropbox', 'NREL', '5MW_files', '5MW_PrecompFiles')

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


    # ----- nacelle ----

    nac = DummyNacelle()


    # -------- tower aero --------
    taero = TowerAero()

    taero.z = z
    taero.d = d
    taero.z_surface = 20.0

    taero.hs = 7.5   # 7.5 is 10 year extreme             # significant wave height (m)
    taero.T = 19.6                # wave period (s)
    taero.Uc = 1.2


    # ----- tower struc ---------
    tstruc = TowerStruc()

    tstruc.z = z         # heights starting from bottom of tower to top (m)
    tstruc.d = d     # corresponding diameters (m)
    tstruc.t = t
    tstruc.n = [5, 5, 5]

    tstruc.soil.G = 140e6
    tstruc.soil.nu = 0.4
    tstruc.soil.depth = 10.0
    tstruc.soil.rigid_x = False

    tstruc.L_reinforced = 30.0        # reinforcement length of cylindrical sections (m)




    # ----- turbine  ------

    turbine = TurbineTWISTER()

    turbine.atm.rho = rho
    turbine.atm.mu = mu
    turbine.atm.shearExp = shearExp

    turbine.B = B
    turbine.precone = precone
    turbine.tilt = tilt
    turbine.yaw = yaw

    # turbine.replace('site', site)
    turbine.replace('raero', raero)
    turbine.replace('rstruc', rstruc)
    turbine.replace('nac', nac)
    turbine.replace('taero', taero)
    turbine.replace('tstruc', tstruc)
    turbine.run()

    # print turbine.mass
    print turbine.AEP
    print turbine.mass_rotor
    print turbine.mass_nacelle
    print turbine.mass_tower


    import matplotlib.pyplot as plt
    plt.plot(turbine.V, turbine.P)

    plt.figure()
    plt.plot(turbine.r_blade_strain, turbine.blade_strain_upper)
    plt.plot(turbine.r_blade_strain, turbine.blade_strain_lower)
    plt.plot(turbine.r_blade_strain, turbine.blade_strain_buckling)
    plt.ylim([-5e-3, 5e-3])

    plt.figure()
    plt.plot(turbine.z_tower_stress, turbine.tower_stress)
    plt.show()

