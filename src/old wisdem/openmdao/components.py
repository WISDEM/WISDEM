#!/usr/bin/env python
# encoding: utf-8
"""
basecomponents.py

Created by Andrew Ning on 2013-05-16.
Copyright (c) NREL. All rights reserved.
"""

import numpy as np
from math import sin, cos
from openmdao.main.api import Component
from openmdao.main.datatypes.api import Array, Float, VarTree, Int

from vartrees import BladeDistributedLoads, Forces, MassProperties, \
    WindWaveDistributedLoads, ConditionsAtRated, Atmosphere
from wisdem.common import DirectionVector


# class SiteBase(Component):
#     """define the wind distribution and extreme loading condition"""

#     # out
#     PDF = Slot(iotype='out', desc='probability distribution function of form P(V)')
#     CDF = Slot(iotype='out', desc='cumulative distribution function of form P(V)')
#     shearExp = Float(iotype='out', desc='shear exponent')

#     V_extreme = Float(iotype='out', units='m/s', desc='extreme wind speed')




class RotorAeroBase(Component):
    """base class for rotor aerodynamics"""

    # ---- in ------

    atm = VarTree(Atmosphere(), iotype='in', desc='atmospheric properties')

    B = Int(3, iotype='in', desc='number of blades')
    precone = Float(iotype='in', units='deg', desc='precone angle')
    tilt = Float(iotype='in', units='deg', desc='rotor tilt angle')
    yaw = Float(iotype='in', units='deg', desc='yaw angle')
    hubHt = Float(iotype='in', units='m', desc='hub height')

    r_structural = Array([0.0], iotype='in', units='m', desc='radial location where structural twist is defined')
    twist_structural = Array([0.0], iotype='in', units='deg', desc='structural twist (for steady aeroelastics)')


    # --- out -----
    V = Array(iotype='out', units='m/s', desc='hub height wind speeds used in power curve')
    P = Array(iotype='out', units='W', desc='corresponding power for wind speed (power curve)')
    AEP = Float(iotype='out', units='kW*h', desc='annual energy production')

    rated_conditions = VarTree(ConditionsAtRated(), iotype='out', desc='operating conditions at rated')

    loads_rated = VarTree(BladeDistributedLoads(), iotype='out', desc='aerodynamic loads at rated in airfoil-aligned c.s.')
    loads_extreme = VarTree(BladeDistributedLoads(), iotype='out', desc='aerodynamic loads at extreme condition in airfoil-aligned c.s.')
    hub_rated = VarTree(Forces(), iotype='out', desc='hub loads at rated in hub-aligned c.s.')





class RotorStrucBase(Component):
    """base class for rotor structural analysis"""

    # ---- in ------
    B = Int(3, iotype='in', desc='number of blades')
    precone = Float(iotype='in', units='deg', desc='precone angle')
    tilt = Float(iotype='in', units='deg', desc='shaft tilt angle')

    # aerodynamic loads
    loads_rated = VarTree(BladeDistributedLoads(), iotype='in', desc='aerodynamic loads at rated in airfoil-aligned c.s.')
    loads_extreme = VarTree(BladeDistributedLoads(), iotype='in', desc='aerodynamic loads at extreme condition in airfoil-aligned c.s.')

    # ---- out ------
    mass_properties = VarTree(MassProperties(), iotype='out', desc='mass properties of rotor about its center of mass (assumed to be hub) in hub-aligned c.s.')



class NacelleBase(Component):
    """base class for nacelle"""

    # ---- in ------
    rotor_rated_conditions = VarTree(ConditionsAtRated(), iotype='in', desc='rotor operating conditions at rated')

    # ---- out ------
    mass_properties = VarTree(MassProperties(), iotype='out', desc='nacelle mass properties about its center of mass in hub-aligned c.s.')
    cm_location = Array(iotype='out', units='m', desc='location of nacelle center of mass relative to tower top in hub-aligned c.s.')
    hub_location = Array(iotype='out', units='m', desc='location of rotor hub relative to tower top in hub-aligned c.s.')



class DummyNacelle(NacelleBase):

    def execute(self):

        mp = self.mass_properties

        mp.mass = 247870.0
        mp.Ixx = 2960437.0
        mp.Iyy = 3253223.0
        mp.Izz = 3264220.0
        mp.Ixy = 0.0
        mp.Ixz = -18400.0
        mp.Iyz = 0.0

        self.cm_location = [-5.0, 0.0, 2.0]
        self.hub_location = [-10.0, 0.0, 2.0]



class TowerAeroBase(Component):
    """base class for tower aerodynamics"""

    atm = VarTree(Atmosphere(), iotype='in', desc='atmospheric properties')

    yaw = Float(iotype='in', units='deg', desc='yaw angle')

    Uhub = Float(iotype='in', desc='magnitude of hub height wind speed')


    # ---- out --------
    wind_wave_loads = VarTree(WindWaveDistributedLoads(), iotype='out', desc='distributed wind/wave loads along tower in yaw-aligned c.s.')



class TowerStrucBase(Component):
    """base class for tower structures"""

    # ---- in --------
    distributed_loads = VarTree(WindWaveDistributedLoads(), iotype='in', desc='applied loading on tower in yaw-aligned c.s.')
    top_forces = VarTree(Forces(), iotype='in', desc='point forces/moments at tower top in yaw-aligned c.s.')
    top_mass_properties = VarTree(MassProperties(), iotype='in', desc='RNA mass properties about tower top in yaw-aligned c.s.')
    top_cm = Array(iotype='in', units='m', desc='location of RNA center of mass relative to tower top in yaw-aligned c.s.')

    # ---- out --------
    mass = Float(iotype='out', units='kg', desc='mass of tower')




class MassTransferToTower(Component):
    """transfer rotor/nacelle mass properties to tower top"""

    # in
    rotor_mass_properties = VarTree(MassProperties(), iotype='in', desc='mass properties of rotor about its center of mass in hub-aligned c.s.')
    nacelle_mass_properties = VarTree(MassProperties(), iotype='in', desc='nacelle mass properties about its center of mass in hub-aligned c.s.')
    rotor_cm = Array(iotype='in', units='m', desc='location of rotor center of mass relative to tower top in hub-aligned c.s.')
    nacelle_cm = Array(iotype='in', units='m', desc='location of nacelle center of mass relative to tower top in hub-aligned c.s.')
    tilt = Float(iotype='in', units='deg', desc='shaft tilt angle')

    # out
    rna_mass_properties = VarTree(MassProperties(), iotype='out', desc='mass properties of RNA about tower top in yaw-aligned c.s.')
    rna_cm = Array(iotype='out', units='m', desc='location of RNA center of mass relative to tower top in yaw-aligned c.s.')


    def execute(self):

        # rename
        mp1 = self.rotor_mass_properties
        mp2 = self.nacelle_mass_properties
        m1 = mp1.mass
        m2 = mp2.mass
        r1 = self.rotor_cm
        r2 = self.nacelle_cm
        I1 = np.array([[mp1.Ixx, mp1.Ixy, mp1.Ixz],
                       [mp1.Ixy, mp1.Iyy, mp1.Iyz],
                       [mp1.Ixz, mp1.Iyz, mp1.Izz]])
        I2 = np.array([[mp2.Ixx, mp2.Ixy, mp2.Ixz],
                       [mp2.Ixy, mp2.Iyy, mp2.Iyz],
                       [mp2.Ixz, mp2.Iyz, mp2.Izz]])

        # mass
        mass = m1 + m2

        # center of mass (hub-aligned)
        cm = (r1*m1 + r2*m2) / mass

        # transfer to yaw-aligned coordinate system
        cm_yaw = DirectionVector(*cm).hubToYaw(self.tilt)
        self.rna_cm = np.array([cm_yaw.x, cm_yaw.y, cm_yaw.z])

        # moments of inertia at new location (hub-aligned)
        I1_translated = I1 + m1*(np.dot(r1, r1)*np.eye(3) - np.outer(r1, r1))
        I2_translated = I2 + m2*(np.dot(r2, r2)*np.eye(3) - np.outer(r2, r2))
        I = I1_translated + I2_translated

        # coordinate transformation
        st = sin(self.tilt)
        ct = cos(self.tilt)
        T = np.array([[ct, 0.0, st],
                      [0.0, 1.0, 0.0],
                      [-st, 0.0, ct]])
        I_yaw = T.dot(I).dot(T.T)


        # fill in vartreee
        mp = self.rna_mass_properties
        mp.mass = mass
        mp.Ixx = I_yaw[0, 0]
        mp.Iyy = I_yaw[1, 1]
        mp.Izz = I_yaw[2, 2]
        mp.Ixy = I_yaw[0, 1]
        mp.Ixz = I_yaw[0, 2]
        mp.Iyz = I_yaw[1, 2]



class HubLoadsTransfer(Component):
    """transfer hub force/moments from rotor to tower top"""

    # in
    forces_hubCS = VarTree(Forces(), iotype='in', desc='hub loads in hub-aligned c.s.')
    hub_to_tower_top = Array(iotype='in', units='m', desc='location of hub relative to tower top in hub-aligned c.s.')
    tilt = Float(iotype='in', units='deg', desc='shaft tilt angle')

    # out
    forces_yawCS = VarTree(Forces(), iotype='out', desc='hub loads in yaw-aligned c.s.')

    def execute(self):

        # rename
        F = self.forces_hubCS.F
        M = self.forces_hubCS.M

        # moment transfer
        Ft = F
        Mt = M + np.cross(self.hub_to_tower_top, F)

        # rotate to hubsystem
        Fyaw = DirectionVector.fromArray(Ft).hubToYaw(self.tilt)
        Myaw = DirectionVector.fromArray(Mt).hubToYaw(self.tilt)

        # save in vartree
        self.forces_yawCS.F = Fyaw.toArray()
        self.forces_yawCS.M = Myaw.toArray()

