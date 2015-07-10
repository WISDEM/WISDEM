#!/usr/bin/env python
# encoding: utf-8
"""
turbine.py

Created by Andrew Ning and Katherine Dykes on 2014-01-13.
Copyright (c) NREL. All rights reserved.
"""

from openmdao.main.api import Assembly, Component
from openmdao.main.datatypes.api import Float, Array, Enum, Bool, Int
from openmdao.lib.drivers.api import FixedPointIterator
import numpy as np

#from rotorse.rotor import RotorSE
#from towerse.tower import TowerSE
#from commonse.rna import RNAMass, RotorLoads
from drivewpact.drive import DriveWPACT
from drivewpact.hub import HubWPACT
from commonse.csystem import DirectionVector
from commonse.utilities import interp_with_deriv, hstack, vstack
from drivese.drive import Drive4pt, Drive3pt
from drivese.drivese_utils import blade_moment_transform, blade_force_transform
from drivese.hub import HubSE, Hub_System_Adder_drive

from SEAMLoads.SEAMLoads import SEAMLoads
from SEAMTower.SEAMTower import SEAMTower
from SEAMAero.SEAM_AEP import SEAMAEP
from SEAMRotor.SEAMRotor import SEAMBladeStructure
# from SEAMGeometry.SEAMGeometry import SEAMGeometry

def connect_io(top, cls):

    cls_name = cls.name
    for name in cls.list_inputs():

        try:
            top.connect(name, cls_name + '.%s' % name)
        except:
            # print 'failed connecting', cls_name, name
            pass

    for name in cls.list_outputs():
        try:
            top.connect(cls_name + '.%s' % name, name)
        except:
            pass


def configure_turbine(assembly, with_new_nacelle=True, flexible_blade=False, with_3pt_drive=False):
    """a stand-alone configure method to allow for flatter assemblies

    Parameters
    ----------
    assembly : Assembly
        an openmdao assembly to be configured
    with_new_nacelle : bool
        False uses the default implementation, True uses an experimental implementation designed
        to smooth out discontinities making in amenable for gradient-based optimization
    flexible_blade : bool
        if True, internally solves the coupled aero/structural deflection using fixed point iteration.
        Note that the coupling is currently only in the flapwise deflection, and is primarily
        only important for highly flexible blades.  If False, the aero loads are passed
        to the structure but there is no further iteration.
    """

    #SEAM variables ----------------------------------
    #d2e = Float(0.73, iotype='in', desc='Dollars to Euro ratio'
    assembly.add('rated_power',Float(3., iotype='in', units='MW', desc='Turbine rated power'))
    assembly.add('hub_height', Float(100., iotype='in', units='m', desc='Hub height'))
    assembly.add('rotor_diameter',  Float(110., iotype='in', units='m', desc='Rotor diameter'))
    assembly.add('site_type',Enum('onshore', values=('onshore', 'offshore'), iotype='in', desc='Site type: onshore or offshore'))

    assembly.add('rho_steel', Float(7.8e3, iotype='in', desc='density of steel'))
    assembly.add('D_bottom', Float(4., iotype='in', desc='Tower bottom diameter'))
    assembly.add('D_top', Float(2., iotype='in', desc='Tower top diameter'))

    assembly.add('Neq', Float(1.e7, iotype='in', desc=''))
    assembly.add('Slim_ext', Float(235., iotype='in', units='MPa', desc=''))
    assembly.add('Slim_fat', Float(14.885, iotype='in', units='MPa', desc=''))
    assembly.add('SF_tower', Float(1.5, iotype='in', desc=''))
    assembly.add('PMtarget', Float(1., iotype='in', desc=''))
    assembly.add('WohlerExpTower', Float(4., iotype='in', desc=''))

    assembly.add('height', Array(iotype='out', desc='Tower discretization'))
    assembly.add('t', Array(iotype='out', desc='Wall thickness'))
    assembly.add('mass', Float(iotype='out', desc='Tower mass'))

    assembly.add('tsr', Float(iotype='in', units='m', desc='Design tip speed ratio'))
    assembly.add('F', Float(iotype='in', desc=''))
    assembly.add('WohlerExpFlap', Float(iotype='in', desc='Wohler Exponent blade flap'))
    assembly.add('WohlerExpTower', Float(iotype='in', desc='Wohler Exponent tower bottom'))
    assembly.add('nSigma4fatFlap', Float(iotype='in', desc=''))
    assembly.add('nSigma4fatTower',  Float(iotype='in', desc=''))
    assembly.add('dLoaddUfactorFlap', Float(iotype='in', desc=''))
    assembly.add('dLoaddUfactorTower', Float(iotype='in', desc=''))
    assembly.add('Neq', Float(iotype='in', desc=''))
    assembly.add('EdgeExtDynFact', Float(iotype='in', desc=''))
    assembly.add('EdgeFatDynFact', Float(iotype='in', desc=''))

    assembly.add('max_tipspeed', Float(iotype='in', desc='Maximum tip speed'))
    assembly.add('n_wsp', Int(iotype='in', desc='Number of wind speed bins'))
    assembly.add('min_wsp', Float(0.0, iotype = 'in', units = 'm/s', desc = 'min wind speed'))
    assembly.add('max_wsp', Float(iotype = 'in', units = 'm/s', desc = 'max wind speed'))

    assembly.add('Iref', Float(iotype='in', desc='Reference turbulence intensity'))
    assembly.add('WeibullInput', Bool(iotype='in', desc='Flag for Weibull input'))
    assembly.add('WeiA_input', Float(iotype = 'in', units='m/s', desc = 'Weibull A'))
    assembly.add('WeiC_input', Float(iotype = 'in', desc='Weibull C'))
    assembly.add('NYears', Float(iotype = 'in', desc='Operating years'))

    assembly.add('overallMaxTower', Float(iotype='out', units='kN*m', desc='Max tower bottom moment'))
    assembly.add('overallMaxFlap', Float(iotype='out', units='kN*m', desc='Max blade root flap moment'))
    assembly.add('FlapLEQ', Float(iotype='out', units='kN*m', desc='Blade root flap lifetime eq. moment'))
    assembly.add('TowerLEQ', Float(iotype='out', units='kN*m', desc='Tower bottom lifetime eq. moment'))

    assembly.add('Nsections', Int(iotype='in', desc='number of sections'))
    assembly.add('Neq', Float(1.e7, iotype='in', desc=''))

    assembly.add('WohlerExpFlap', Float(iotype='in', desc=''))
    assembly.add('PMtarget', Float(iotype='in', desc=''))

    #rotor_diameter = Float(iotype='in', units='m', desc='') #[m]
    assembly.add('MaxChordrR', Float(iotype='in', units='m', desc='')) #[m]

    assembly.add('TIF_FLext', Float(iotype='in', desc='')) # Tech Impr Factor _ flap extreme
    assembly.add('TIF_EDext', Float(iotype='in', desc=''))

    assembly.add('TIF_FLfat', Float(iotype='in', desc=''))

    assembly.add('sc_frac_flap', Float(iotype='in', desc='')) # sparcap fraction of chord flap
    assembly.add('sc_frac_edge', Float(iotype='in', desc='')) # sparcap fraction of thickness edge

    assembly.add('SF_blade', Float(iotype='in', desc='')) #[factor]
    assembly.add('Slim_ext_blade', Float(iotype='in', units='MPa', desc=''))
    assembly.add('Slim_fat_blade', Float(iotype='in', units='MPa', desc=''))

    assembly.add('AddWeightFactorBlade', Float(iotype='in', desc='')) # Additional weight factor for blade shell
    assembly.add('BladeDens', Float(iotype='in', units='kg/m**3', desc='density of blades')) # [kg / m^3]

    assembly.add('BladeWeight', Float(iotype = 'out', units = 'kg', desc = 'BladeMass' ))

    assembly.add('mean_wsp', Float(iotype = 'in', units = 'm/s', desc = 'mean wind speed')  )  # [m/s]
    assembly.add('air_density', Float(iotype = 'in', units = 'kg/m**3', desc = 'density of air')) # [kg / m^3]
    assembly.add('turbulence_int', Float(iotype = 'in', desc = ''))
    assembly.add('max_Cp', Float(iotype = 'in', desc = 'max CP'))
    assembly.add('gearloss_const', Float(iotype = 'in', desc = 'Gear loss constant'))
    assembly.add('gearloss_var', Float(iotype = 'in', desc = 'Gear loss variable'))
    assembly.add('genloss', Float(iotype = 'in', desc = 'Generator loss'))
    assembly.add('convloss', Float(iotype = 'in', desc = 'Converter loss'))

    # Outputs
    assembly.add('rated_wind_speed', Float(units = 'm / s', iotype='out', desc='wind speed for rated power'))
    assembly.add('ideal_power_curve', Array(iotype='out', units='kW', desc='total power before losses and turbulence'))
    assembly.add('power_curve', Array(iotype='out', units='kW', desc='total power including losses and turbulence'))
    assembly.add('wind_curve', Array(iotype='out', units='m/s', desc='wind curve associated with power curve'))

    assembly.add('NYears', Float(iotype = 'in', desc='Operating years'))  # move this to COE calculation

    assembly.add('aep', Float(iotype = 'out', units='mW*h', desc='Annual energy production in mWh'))
    assembly.add('total_aep', Float(iotype = 'out', units='mW*h', desc='AEP for total years of production'))

    # END SEAM Variables ----------------------

    # Add SEAM components and connections
    assembly.add('loads', SEAMLoads())
    assembly.add('tower_design', SEAMTower(21))
    assembly.add('blade_design', SEAMBladeStructure())
    assembly.add('aep_calc', SEAMAEP())
    assembly.driver.workflow.add(['loads', 'tower_design', 'blade_design', 'aep_calc'])

    assembly.connect('loads.overallMaxTower', 'tower_design.overallMaxTower')
    assembly.connect('loads.TowerLEQ', 'tower_design.TowerLEQ')

    assembly.connect('loads.overallMaxFlap', 'blade_design.overallMaxFlap')
    assembly.connect('loads.overallMaxEdge', 'blade_design.overallMaxEdge')
    assembly.connect('loads.FlapLEQ', 'blade_design.FlapLEQ')
    assembly.connect('loads.EdgeLEQ', 'blade_design.EdgeLEQ')

    connect_io(assembly, assembly.aep_calc)
    connect_io(assembly, assembly.loads)
    connect_io(assembly, assembly.tower_design)
    connect_io(assembly, assembly.blade_design)

    # End SEAM add components and connections -------------


    if with_new_nacelle:
        assembly.add('hub',HubSE())
        assembly.add('hubSystem',Hub_System_Adder_drive())
        if with_3pt_drive:
            assembly.add('nacelle', Drive3pt())
        else:
            assembly.add('nacelle', Drive4pt())
    else:
        assembly.add('nacelle', DriveWPACT())
        assembly.add('hub', HubWPACT())

    assembly.driver.workflow.add(['hub', 'nacelle'])
    if with_new_nacelle:
        assembly.driver.workflow.add(['hubSystem'])

    # connections to hub and hub system
    assembly.connect('blade_design.BladeWeight', 'hub.blade_mass')
    assembly.connect('loads.overallMaxFlap', 'hub.rotor_bending_moment')
    assembly.connect('rotor_diameter', ['hub.rotor_diameter'])
    assembly.connect('blade_design.RootChord', 'hub.blade_root_diameter')
    assembly.add('blade_number',Int(3,iotype='in',desc='number of blades - was in RotorSE, now adding here to turbine for SEAM'))
    assembly.connect('blade_number', 'hub.blade_number')
    if with_new_nacelle:
        assembly.connect('rated_power','hub.machine_rating')
        assembly.connect('rotor_diameter', ['hubSystem.rotor_diameter'])
        assembly.connect('nacelle.MB1_location','hubSystem.MB1_location') # TODO: bearing locations
        assembly.connect('nacelle.L_rb','hubSystem.L_rb')
        assembly.add('rotor_tilt',Float(5.0, iotype='in', desc='rotor tilt - was in RotorSE, now adding here to turbine for SEAM'))
        assembly.connect('rotor_tilt','hubSystem.shaft_angle')
        assembly.connect('hub.hub_diameter','hubSystem.hub_diameter')
        assembly.connect('hub.hub_thickness','hubSystem.hub_thickness')
        assembly.connect('hub.hub_mass','hubSystem.hub_mass')
        assembly.connect('hub.spinner_mass','hubSystem.spinner_mass')
        assembly.connect('hub.pitch_system_mass','hubSystem.pitch_system_mass')

    # connections to nacelle #TODO: fatigue option variables
    assembly.connect('rotor_diameter', 'nacelle.rotor_diameter')
    assembly.connect('1.5 * aep_calc.rated_torque', 'nacelle.rotor_torque')
    assembly.connect('loads.max_thrust', 'nacelle.rotor_thrust')
    assembly.connect('aep_calc.rated_speed', 'nacelle.rotor_speed')
    assembly.connect('rated_power', 'nacelle.machine_rating')
    assembly.add('generator_speed',Float(1173.7,iotype='in',units='rpm',desc='speed of generator - should be in nacelle'))
    assembly.connect('generator_speed/aep_calc.rated_speed', 'nacelle.gear_ratio')
    assembly.connect('D_top', 'nacelle.tower_top_diameter')
    assembly.connect('blade_number * blade_design.BladeWeight + hub.hub_system_mass', 'nacelle.rotor_mass') # assuming not already in rotor force / moments
    # variable connections for new nacelle
    if with_new_nacelle:
        assembly.connect('blade_number','nacelle.blade_number')
        assembly.connect('rotor_tilt','nacelle.shaft_angle')
        assembly.connect('333.3 * rated_power / 1000.0','nacelle.shrink_disc_mass')
        assembly.connect('blade_design.RootChord','nacelle.blade_root_diameter')

        #moments - ignoring for now (nacelle will use internal defaults)
        #assembly.connect('rotor.Mxyz_0','moments.b1')
        #assembly.connect('rotor.Mxyz_120','moments.b2')
        #assembly.connect('rotor.Mxyz_240','moments.b3')
        #assembly.connect('rotor.Pitch','moments.pitch_angle')
        #assembly.connect('rotor.TotalCone','moments.cone_angle')
        assembly.connect('1.5 * aep_calc.rated_torque','nacelle.rotor_bending_moment_x') #accounted for in ratedConditions.Q
        #assembly.connect('moments.My','nacelle.rotor_bending_moment_y')
        #assembly.connect('moments.Mz','nacelle.rotor_bending_moment_z')

        #forces - ignoring for now (nacelle will use internal defaults)
        #assembly.connect('rotor.Fxyz_0','forces.b1')
        #assembly.connect('rotor.Fxyz_120','forces.b2')
        #assembly.connect('rotor.Fxyz_240','forces.b3')
        #assembly.connect('rotor.Pitch','forces.pitch_angle')
        #assembly.connect('rotor.TotalCone','forces.cone_angle')
        assembly.connect('loads.max_thrust','nacelle.rotor_force_x')
        #assembly.connect('forces.Fy','nacelle.rotor_force_y')
        #assembly.connect('forces.Fz','nacelle.rotor_force_z')


class Turbine_SE_SEAM(Assembly):

    def configure(self):
        configure_turbine(self)


if __name__ == '__main__':

    turbine = Turbine_SE_SEAM()

    #=========== SEAM inputs

    turbine.AddWeightFactorBlade = 1.2
    turbine.BladeDens = 2100.0
    turbine.D_bottom = 8.3
    turbine.D_top = 5.5
    turbine.EdgeExtDynFact = 2.5
    turbine.EdgeFatDynFact = 0.75
    turbine.F = 0.777
    turbine.Iref = 0.16
    turbine.MaxChordrR = 0.2
    turbine.NYears = 20.0
    turbine.Neq = 10000000.0
    turbine.Nsections = 21
    turbine.PMtarget = 1.0
    turbine.SF_blade = 1.1
    turbine.SF_tower = 1.5
    turbine.Slim_ext = 235.0
    turbine.Slim_fat = 14.885
    turbine.Slim_ext_blade = 200.0
    turbine.Slim_fat_blade = 27.0
    turbine.TIF_EDext = 1.0
    turbine.TIF_FLext = 1.0
    turbine.TIF_FLfat = 1.0
    turbine.WeiA_input = 11.0
    turbine.WeiC_input = 2.0
    turbine.WeibullInput = True
    turbine.WohlerExpFlap = 10.0
    turbine.WohlerExpTower = 4.0
    turbine.d2e = 0.73
    turbine.dLoaddUfactorFlap = 0.9
    turbine.dLoaddUfactorTower = 0.8
    turbine.hub_height = 120.0
    turbine.max_tipspeed = 90.0
    turbine.n_wsp = 26
    turbine.min_wsp = 0.0
    turbine.max_wsp = 25.0
    turbine.nSigma4fatFlap = 1.2
    turbine.nSigma4fatTower = 0.8
    turbine.rated_power = 10.0
    turbine.rho_steel = 7800.0
    turbine.rotor_diameter = 178.0
    turbine.sc_frac_edge = 0.8
    turbine.sc_frac_flap = 0.3
    turbine.site_type = 'onshore'
    turbine.tsr = 8.0
    turbine.air_density = 1.225
    turbine.turbulence_int = 0.1
    turbine.max_Cp = 0.49
    turbine.gearloss_const = 0.01    # Fraction
    turbine.gearloss_var = 0.014     # Fraction
    turbine.genloss = 0.03          # Fraction
    turbine.convloss = 0.03         # Fraction

    #==============

    # === nacelle ======
    turbine.blade_number = 3 # turbine level that must be added for SEAM
    turbine.rotor_tilt = 5.0 # turbine level that must be added for SEAM
    turbine.generator_speed = 1173.7

    turbine.nacelle.L_ms = 1.0  # (Float, m): main shaft length downwind of main bearing in low-speed shaft
    turbine.nacelle.L_mb = 2.5  # (Float, m): main shaft length in low-speed shaft

    turbine.nacelle.h0_front = 1.7  # (Float, m): height of Ibeam in bedplate front
    turbine.nacelle.h0_rear = 1.35  # (Float, m): height of Ibeam in bedplate rear

    turbine.nacelle.drivetrain_design = 'geared'
    turbine.nacelle.crane = True  # (Bool): flag for presence of crane
    turbine.nacelle.bevel = 0  # (Int): Flag for the presence of a bevel stage - 1 if present, 0 if not
    turbine.nacelle.gear_configuration = 'eep'  # (Str): tring that represents the configuration of the gearbox (stage number and types)

    turbine.nacelle.Np = [3, 3, 1]  # (Array): number of planets in each stage
    turbine.nacelle.ratio_type = 'optimal'  # (Str): optimal or empirical stage ratios
    turbine.nacelle.shaft_type = 'normal'  # (Str): normal or short shaft length
    #turbine.nacelle.shaft_angle = 5.0  # (Float, deg): Angle of the LSS inclindation with respect to the horizontal
    turbine.nacelle.shaft_ratio = 0.10  # (Float): Ratio of inner diameter to outer diameter.  Leave zero for solid LSS
    turbine.nacelle.carrier_mass = 8000.0 # estimated for 5 MW
    turbine.nacelle.mb1Type = 'CARB'  # (Str): Main bearing type: CARB, TRB or SRB
    turbine.nacelle.mb2Type = 'SRB'  # (Str): Second bearing type: CARB, TRB or SRB
    turbine.nacelle.yaw_motors_number = 8.0  # (Float): number of yaw motors
    turbine.nacelle.uptower_transformer = True
    turbine.nacelle.flange_length = 0.5 #m
    turbine.nacelle.gearbox_cm = 0.1
    turbine.nacelle.hss_length = 1.5
    turbine.nacelle.overhang = 5.0 #TODO - should come from turbine configuration level

    turbine.nacelle.check_fatigue = 0 #0 if no fatigue check, 1 if parameterized fatigue check, 2 if known loads inputs

    # =================

    # === run ===
    turbine.run()
    print 'mass rotor blades (kg) =', turbine.blade_number * turbine.blade_design.BladeWeight
    print 'mass hub system (kg) =', turbine.hubSystem.hub_system_mass
    print 'mass nacelle (kg) =', turbine.nacelle.nacelle_mass
    print 'mass tower (kg) =', turbine.tower_design.mass
    # =================
