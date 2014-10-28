#!/usr/bin/env python
# encoding: utf-8
"""
turbine.py

Created by Andrew Ning and Katherine Dykes on 2014-01-13.
Copyright (c) NREL. All rights reserved.
"""

from openmdao.main.api import Assembly, Component
from openmdao.main.datatypes.api import Float, Array, Enum, Bool
from openmdao.lib.drivers.api import FixedPointIterator
import numpy as np

from rotorse.rotor import RotorSE
#from towerse.tower import TowerSE
from commonse.rna import RNAMass, RotorLoads
from jacketse.jacket import JacketSE
from jacketse.jacket import JcktGeoInputs,SoilGeoInputs,WaterInputs,WindInputs,RNAprops,TPlumpMass,Frame3DDaux,\
                    MatInputs,LegGeoInputs,XBrcGeoInputs,MudBrcGeoInputs,HBrcGeoInputs,TPGeoInputs,PileGeoInputs,\
                    TwrGeoInputs
from drivewpact.drive import DriveWPACT
from drivewpact.hub import HubWPACT
from commonse.csystem import DirectionVector
from commonse.utilities import interp_with_deriv, hstack, vstack
from drivese.drive_smooth import NacelleTS
from drivese.drive import Drive4pt, Drive3pt
from drivese.hub import HubSE


class MaxTipDeflection(Component):

    Rtip = Float(iotype='in', units='m')
    precurveTip = Float(iotype='in', units='m')  # TODO: add gradients for precurveTip and presweepTip
    presweepTip = Float(iotype='in', units='m')
    precone = Float(iotype='in', units='deg')
    tilt = Float(iotype='in', units='deg')
    hub_tt = Array(iotype='in', units='m', desc='location of hub relative to tower-top in yaw-aligned c.s.')
    tower_z = Array(iotype='in', units='m')
    tower_d = Array(np.array([3.87, 3.87]),iotype='in', units='m') #TODO remove default
    towerHt = Float(iotype='in', units='m')

    max_tip_deflection = Float(iotype='out', units='m', desc='clearance between undeflected blade and tower')
    ground_clearance = Float(iotype='out', units='m', desc='distance between blade tip and ground')

    def execute(self):

        # coordinates of blade tip in yaw c.s.
        blade_yaw = DirectionVector(self.precurveTip, self.presweepTip, self.Rtip)\
            .bladeToAzimuth(self.precone).azimuthToHub(180.0).hubToYaw(self.tilt)

        # find corresponding radius of tower
        ztower = (self.towerHt + self.hub_tt[2] + blade_yaw.z)/self.towerHt  # nondimensional location
        # rtower = np.interp(ztower, self.tower_z, self.tower_d) / 2.0
        dtower, ddtower_dztower, ddtower_dtowerz, ddtower_dtowerd = interp_with_deriv(ztower, self.tower_z, self.tower_d)
        rtower = dtower / 2.0
        self.drtower_dztower = ddtower_dztower / 2.0
        self.drtower_dtowerz = ddtower_dtowerz / 2.0
        self.drtower_dtowerd = ddtower_dtowerd / 2.0

        # max deflection before strike
        if self.precone >= 0:  # upwind
            self.max_tip_deflection = -self.hub_tt[0] - blade_yaw.x - rtower
        else:
            self.max_tip_deflection = -self.hub_tt[0] + blade_yaw.x - rtower
            # TODO: need to redo gradients for this case.

        # ground clearance
        self.ground_clearance = self.towerHt + self.hub_tt[2] + blade_yaw.z

        # save for derivs
        self.blade_yaw = blade_yaw


    def list_deriv_vars(self):

        inputs = ('Rtip', 'precurveTip', 'presweepTip', 'precone', 'tilt', 'hub_tt',
            'tower_z', 'tower_d', 'towerHt')
        outputs = ('max_tip_deflection', 'ground_clearance')

        return inputs, outputs

    def provideJ(self):

        dbyx = self.blade_yaw.dx
        # dbyy = self.blade_yaw.dy
        dbyz = self.blade_yaw.dz

        # Rtip
        drtower_dRtip = self.drtower_dztower * dbyz['dz']/self.towerHt
        if self.precone >= 0:
            dtd_dRtip = -dbyx['dz'] - drtower_dRtip
        else:
            dtd_dRtip = dbyx['dz'] - drtower_dRtip
        dgc_dRtip = dbyz['dz']

        # precurveTip
        drtower_dprecurveTip = self.drtower_dztower * dbyz['dx']/self.towerHt
        if self.precone >= 0:
            dtd_dprecurveTip = -dbyx['dx'] - drtower_dprecurveTip
        else:
            dtd_dprecurveTip = dbyx['dx'] - drtower_dprecurveTip
        dgc_dprecurveTip = dbyz['dx']

        # presweep
        drtower_dpresweepTip = self.drtower_dztower * dbyz['dy']/self.towerHt
        if self.precone >= 0:
            dtd_dpresweepTip = -dbyx['dy'] - drtower_dpresweepTip
        else:
            dtd_dpresweepTip = dbyx['dy'] - drtower_dpresweepTip
        dgc_dpresweepTip = dbyz['dy']


        # precone
        drtower_dprecone = self.drtower_dztower * dbyz['dprecone']/self.towerHt
        if self.precone >= 0:
            dtd_dprecone = -dbyx['dprecone'] - drtower_dprecone
        else:
            dtd_dprecone = dbyx['dprecone'] - drtower_dprecone
        dgc_dprecone = dbyz['dprecone']

        # tilt
        drtower_dtilt = self.drtower_dztower * dbyz['dtilt']/self.towerHt
        if self.precone >= 0:
            dtd_dtilt = -dbyx['dtilt'] - drtower_dtilt
        else:
            dtd_dtilt = dbyx['dtilt'] - drtower_dtilt
        dgc_dtilt = dbyz['dtilt']

        # hubtt
        drtower_dhubtt = self.drtower_dztower * np.array([0.0, 0.0, 1.0/self.towerHt])
        dtd_dhubtt = np.array([-1.0, 0.0, 0.0]) - drtower_dhubtt
        dgc_dhubtt = np.array([0.0, 0.0, 1.0])

        # tower_z
        dtd_dtowerz = -self.drtower_dtowerz
        dgc_dtowerz = np.zeros_like(self.tower_z)

        # tower_d
        dtd_dtowerd = -self.drtower_dtowerd
        dgc_dtowerd = np.zeros_like(self.tower_d)

        # towerHt
        drtower_dtowerHt = self.drtower_dztower * -(self.hub_tt[2] + self.blade_yaw.z)/self.towerHt**2
        dtd_dtowerHt = -drtower_dtowerHt
        dgc_dtowerHt = 1.0

        dtd = hstack([dtd_dRtip, dtd_dprecurveTip, dtd_dpresweepTip, dtd_dprecone, dtd_dtilt,
            dtd_dhubtt, dtd_dtowerz, dtd_dtowerd, dtd_dtowerHt])
        dgc = np.concatenate([[dgc_dRtip], [dgc_dprecurveTip], [dgc_dpresweepTip], [dgc_dprecone],
            [dgc_dtilt], dgc_dhubtt, dgc_dtowerz, dgc_dtowerd, [dgc_dtowerHt]])

        J = vstack([dtd, dgc])

        return J




def configure_turbine_with_jacket(assembly, with_new_nacelle=True, flexible_blade=False, with_3pt_drive=False):
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

    # --- general turbine configuration inputs---
    assembly.add('rho', Float(1.225, iotype='in', units='kg/m**3', desc='density of air', deriv_ignore=True))
    assembly.add('mu', Float(1.81206e-5, iotype='in', units='kg/m/s', desc='dynamic viscosity of air', deriv_ignore=True))
    assembly.add('shear_exponent', Float(0.2, iotype='in', desc='shear exponent', deriv_ignore=True))
    assembly.add('hub_height', Float(90.0, iotype='in', units='m', desc='hub height'))
    assembly.add('turbine_class', Enum('I', ('I', 'II', 'III'), iotype='in', desc='IEC turbine class'))
    assembly.add('turbulence_class', Enum('B', ('A', 'B', 'C'), iotype='in', desc='IEC turbulence class class'))
    assembly.add('g', Float(9.81, iotype='in', units='m/s**2', desc='acceleration of gravity', deriv_ignore=True))
    assembly.add('cdf_reference_height_wind_speed', Float(90.0, iotype='in', desc='reference hub height for IEC wind speed (used in CDF calculation)'))
    assembly.add('downwind', Bool(False, iotype='in', desc='flag if rotor is downwind'))
    assembly.add('tower_dt', Float(iotype='in', units='m', desc='tower top diameter')) # update for jacket
    assembly.add('generator_speed', Float(iotype='in', units='rpm', desc='generator speed'))
    assembly.add('machine_rating', Float(5000.0, units='kW', iotype='in', desc='machine rated power'))
    assembly.add('rna_weightM', Bool(True, iotype='in', desc='flag to consider or not the RNA weight effect on Moment'))

    assembly.add('rotor', RotorSE())
    if with_new_nacelle:
        assembly.add('hub',HubSE())
        if with_3pt_drive:
            assembly.add('nacelle', Drive3pt())
        else:
            assembly.add('nacelle', Drive4pt())
    else:
        assembly.add('nacelle', DriveWPACT())
        assembly.add('hub', HubWPACT())
    assembly.add('rna', RNAMass())
    assembly.add('rotorloads1', RotorLoads())
    assembly.add('rotorloads2', RotorLoads())
    assembly.add('jacket', JacketSE())
    assembly.add('maxdeflection', MaxTipDeflection())

    if flexible_blade:
        assembly.add('fpi', FixedPointIterator())

        assembly.fpi.workflow.add(['rotor'])
        assembly.fpi.add_parameter('rotor.delta_precurve_sub', low=-1.e99, high=1.e99)
        assembly.fpi.add_parameter('rotor.delta_bladeLength', low=-1.e99, high=1.e99)
        assembly.fpi.add_constraint('rotor.delta_precurve_sub = rotor.delta_precurve_sub_out')
        assembly.fpi.add_constraint('rotor.delta_bladeLength = rotor.delta_bladeLength_out')
        assembly.fpi.max_iteration = 20
        assembly.fpi.tolerance = 1e-8

        assembly.driver.workflow.add(['fpi'])

    else:
        assembly.driver.workflow.add(['rotor'])

    assembly.driver.workflow.add(['hub', 'nacelle', 'jacket', 'maxdeflection', 'rna', 'rotorloads1', 'rotorloads2'])

    # TODO: rotor drivetrain design should be connected to nacelle drivetrain design

    # connections to rotor
    assembly.connect('machine_rating','rotor.control.ratedPower')
    assembly.connect('rho', 'rotor.rho')
    assembly.connect('mu', 'rotor.mu')
    assembly.connect('shear_exponent', 'rotor.shearExp')
    assembly.connect('hub_height', 'rotor.hubHt')
    assembly.connect('turbine_class', 'rotor.turbine_class')
    assembly.connect('turbulence_class', 'rotor.turbulence_class')
    assembly.connect('g', 'rotor.g')
    assembly.connect('cdf_reference_height_wind_speed', 'rotor.cdf_reference_height_wind_speed')


    # connections to hub
    assembly.connect('rotor.mass_one_blade', 'hub.blade_mass')
    assembly.connect('rotor.root_bending_moment', 'hub.rotor_bending_moment')
    assembly.connect('rotor.diameter', 'hub.rotor_diameter')
    assembly.connect('rotor.hub_diameter', 'hub.blade_root_diameter')
    assembly.connect('rotor.nBlades', 'hub.blade_number')
    if with_new_nacelle:
        assembly.connect('nacelle.MB1_location','hub.MB1_location')
        assembly.connect('rotor.tilt','hub.gamma')
        assembly.connect('nacelle.L_rb','hub.L_rb')


    # connections to nacelle #TODO: fatigue option variables
    assembly.connect('rotor.diameter', 'nacelle.rotor_diameter')
    if not with_new_nacelle:
        assembly.connect('rotor.mass_all_blades + hub.hub_system_mass', 'nacelle.rotor_mass') #DODO: circular dependency if using DriveSE (nacelle csm --> hub, hub mass --> nacelle)
    if with_new_nacelle:
        assembly.connect('rotor.nBlades','nacelle.blade_number')
        assembly.connect('rotor.tilt','nacelle.shaft_angle')
        assembly.connect('333.3 * machine_rating / 1000.0','nacelle.shrink_disc_mass')
    assembly.connect('1.5 * rotor.ratedConditions.Q', 'nacelle.rotor_torque')
    assembly.connect('rotor.ratedConditions.T', 'nacelle.rotor_thrust')
    assembly.connect('rotor.ratedConditions.Omega', 'nacelle.rotor_speed')
    assembly.connect('machine_rating', 'nacelle.machine_rating')
    assembly.connect('rotor.root_bending_moment', 'nacelle.rotor_bending_moment')
    assembly.connect('generator_speed/rotor.ratedConditions.Omega', 'nacelle.gear_ratio')
    '''if  with_new_nacelle:
        assembly.connect('rotor.g', 'nacelle.g')''' # Only drive smooth taking g from rotor; TODO: update when drive_smooth is updated
    assembly.connect('tower_dt', 'nacelle.tower_top_diameter')  # OpenMDAO circular dependency issue # update for jacket input


    # connections to rna
    assembly.connect('rotor.mass_all_blades', 'rna.blades_mass')
    assembly.connect('rotor.I_all_blades', 'rna.blades_I')
    assembly.connect('hub.hub_system_mass', 'rna.hub_mass')
    assembly.connect('hub.hub_system_cm', 'rna.hub_cm')
    assembly.connect('hub.hub_system_I', 'rna.hub_I')
    assembly.connect('nacelle.nacelle_mass', 'rna.nac_mass')
    assembly.connect('nacelle.nacelle_cm', 'rna.nac_cm')
    assembly.connect('nacelle.nacelle_I', 'rna.nac_I')

    # connections to rotorloads1
    assembly.connect('downwind', 'rotorloads1.downwind')
    assembly.connect('rna_weightM', 'rotorloads1.rna_weightM')
    assembly.connect('1.8 * rotor.ratedConditions.T', 'rotorloads1.F[0]')
    assembly.connect('rotor.ratedConditions.Q', 'rotorloads1.M[0]')
    assembly.connect('hub.hub_system_cm', 'rotorloads1.r_hub')
    assembly.connect('rna.rna_cm', 'rotorloads1.rna_cm')
    assembly.connect('rotor.tilt', 'rotorloads1.tilt')
    assembly.connect('g', 'rotorloads1.g')
    assembly.connect('rna.rna_mass', 'rotorloads1.m_RNA')

    # connections to rotorloads2
    assembly.connect('downwind', 'rotorloads2.downwind')
    assembly.connect('rna_weightM', 'rotorloads2.rna_weightM')
    assembly.connect('rotor.T_extreme', 'rotorloads2.F[0]')
    assembly.connect('rotor.Q_extreme', 'rotorloads2.M[0]')
    assembly.connect('hub.hub_system_cm', 'rotorloads2.r_hub')
    assembly.connect('rna.rna_cm', 'rotorloads2.rna_cm')
    assembly.connect('rotor.tilt', 'rotorloads2.tilt')
    assembly.connect('g', 'rotorloads2.g')
    assembly.connect('rna.rna_mass', 'rotorloads2.m_RNA')

    # connections to jacket
    assembly.connect('rho', 'jacket.Windinputs.rho') # jacket input
    assembly.connect('mu', 'jacket.Windinputs.mu') # jacket input
    assembly.connect('-g', 'jacket.FrameAuxIns.gvector[2]') # jacket input
    assembly.connect('hub_height', 'jacket.Windinputs.HH') # jacket input
    assembly.connect('tower_dt', 'jacket.Twrinputs.Dt') # jacket input
    assembly.connect('rotor.yaw', 'jacket.RNAinputs.yawangle') # jacket input
    #assembly.connect('hub_height - nacelle.nacelle_cm[2]', 'jacket.Twrinputs.Htwr') # jacket input; TODO: probably irrelevant for this purpose, tower length is now determined in jacket
    assembly.connect('rna.rna_mass', 'jacket.RNAinputs.mass') # jacket input
    assembly.connect('rna.rna_cm', 'jacket.RNAinputs.CMoff') # jacket input
    assembly.connect('rna.rna_I_TT', 'jacket.RNAinputs.I') # jacket input
    # Rated rotor loads (Option 1)
    assembly.connect('rotor.ratedConditions.V', 'jacket.Windinputs.U50HH') # jacket input
    assembly.connect('rotorloads1.top_F', 'jacket.RNA_F[0:3]') # jacket input
    assembly.connect('rotorloads1.top_M', 'jacket.RNA_F[3:6]') # jacket input
    # Survival rotor loads (Option 2)
    #assembly.connect('rotor.V_extreme', 'tower.Windinputs.U50HH') # jacket input
    #assembly.connect('rotorloads2.top_F', 'jacket.RNA_F') # jacket input
    #assembly.connect('rotorloads2.top_M', 'jacket.RNA_M') # jacket input

    # connections to maxdeflection
    assembly.connect('rotor.Rtip', 'maxdeflection.Rtip')
    assembly.connect('rotor.precurveTip', 'maxdeflection.precurveTip')
    assembly.connect('rotor.presweepTip', 'maxdeflection.presweepTip')
    assembly.connect('rotor.precone', 'maxdeflection.precone')
    assembly.connect('rotor.tilt', 'maxdeflection.tilt')
    assembly.connect('hub.hub_system_cm', 'maxdeflection.hub_tt')
    assembly.connect('jacket.Twrouts.nodes[2,:]', 'maxdeflection.tower_z') # TODO: jacket input; it doesnt like the [2,:] syntax  ---THIS is the z at CMzoff, not necessarily the top flange
    #assembly.connect('jacket.legouts.LegObj.D', 'maxdeflection.tower_d') # TODO: jacket input - doesnt recognize logobj
    assembly.connect('jacket.Twrouts.Htwr', 'maxdeflection.towerHt') # TODO: jacket input



class TurbineSE_jacket(Assembly):

    def configure(self):
        configure_turbine_with_jacket(self)


if __name__ == '__main__':

    # === setup ===
    import os
    from rotorse.precomp import Profile, Orthotropic2DMaterial, CompositeSection  # TODO: can just pass file names and do this initialization inside of rotor
    from towerse.tower import TowerWithpBEAM
    from commonse.environment import PowerWind, TowerSoil
    from commonse.utilities import print_vars

    turbine = TurbineSE()
    # print_vars(turbine, list_type='inputs', prefix='turbine')

    rotor = turbine.rotor
    nacelle = turbine.nacelle
    jacket = turbine.jacket

    # =================

    # === Turbine Configuration ===

    # --- atmosphere ---
    turbine.rho = 1.225  # (Float, kg/m**3): density of air
    turbine.mu = 1.81206e-5  # (Float, kg/m/s): dynamic viscosity of air
    turbine.shear_exponent = 0.2  # (Float): shear exponent
    turbine.hub_height = 90.0  # (Float, m): hub height
    turbine.turbine_class = 'I'  # (Enum): IEC turbine class
    turbine.turbulence_class = 'B'  # (Enum): IEC turbulence class class
    turbine.cdf_reference_height_wind_speed = 90.0  # (Float): reference hub height for IEC wind speed (used in CDF calculation)
    turbine.g = 9.81  # (Float, m/s**2): acceleration of gravity
    turbine.downwind = False  # (Bool): flag if rotor is downwind
    turbine.generator_speed = 1173.7  # (Float, rpm)  # generator speed
    turbine.tower_dt = 3.87
    # ----------------------

    # ============================


    # === rotor ===
    # --- blade grid ---
    rotor.initial_aero_grid = np.array([0.02222276, 0.06666667, 0.11111057, 0.16666667, 0.23333333, 0.3, 0.36666667,
        0.43333333, 0.5, 0.56666667, 0.63333333, 0.7, 0.76666667, 0.83333333, 0.88888943, 0.93333333,
        0.97777724])  # (Array): initial aerodynamic grid on unit radius
    rotor.initial_str_grid = np.array([0.0, 0.00492790457512, 0.00652942887106, 0.00813095316699, 0.00983257273154,
        0.0114340970275, 0.0130356213234, 0.02222276, 0.024446481932, 0.026048006228, 0.06666667, 0.089508406455,
        0.11111057, 0.146462614229, 0.16666667, 0.195309105255, 0.23333333, 0.276686558545, 0.3, 0.333640766319,
        0.36666667, 0.400404310407, 0.43333333, 0.5, 0.520818918408, 0.56666667, 0.602196371696, 0.63333333,
        0.667358391486, 0.683573824984, 0.7, 0.73242031601, 0.76666667, 0.83333333, 0.88888943, 0.93333333, 0.97777724,
        1.0])  # (Array): initial structural grid on unit radius
    rotor.idx_cylinder_aero = 3  # (Int): first idx in r_aero_unit of non-cylindrical section, constant twist inboard of here
    rotor.idx_cylinder_str = 14  # (Int): first idx in r_str_unit of non-cylindrical section
    rotor.hubFraction = 0.025  # (Float): hub location as fraction of radius
    # ------------------

    # --- blade geometry ---
    rotor.r_aero = np.array([0.02222276, 0.06666667, 0.11111057, 0.2, 0.23333333, 0.3, 0.36666667, 0.43333333,
        0.5, 0.56666667, 0.63333333, 0.64, 0.7, 0.83333333, 0.88888943, 0.93333333,
        0.97777724])  # (Array): new aerodynamic grid on unit radius
    rotor.r_max_chord = 0.23577  # (Float): location of max chord on unit radius
    rotor.chord_sub = [3.2612, 4.5709, 3.3178, 1.4621]  # (Array, m): chord at control points. defined at hub, then at linearly spaced locations from r_max_chord to tip
    rotor.theta_sub = [13.2783, 7.46036, 2.89317, -0.0878099]  # (Array, deg): twist at control points.  defined at linearly spaced locations from r[idx_cylinder] to tip
    rotor.precurve_sub = [0.0, 0.0, 0.0]  # (Array, m): precurve at control points.  defined at same locations at chord, starting at 2nd control point (root must be zero precurve)
    rotor.delta_precurve_sub = [0.0, 0.0, 0.0]  # (Array, m): adjustment to precurve to account for curvature from loading
    rotor.sparT = [0.05, 0.047754, 0.045376, 0.031085, 0.0061398]  # (Array, m): spar cap thickness parameters
    rotor.teT = [0.1, 0.09569, 0.06569, 0.02569, 0.00569]  # (Array, m): trailing-edge thickness parameters
    rotor.bladeLength = 61.5  # (Float, m): blade length (if not precurved or swept) otherwise length of blade before curvature
    rotor.delta_bladeLength = 0.0  # (Float, m): adjustment to blade length to account for curvature from loading
    rotor.precone = 2.5  # (Float, deg): precone angle
    rotor.tilt = 5.0  # (Float, deg): shaft tilt
    rotor.yaw = 0.0  # (Float, deg): yaw error
    rotor.nBlades = 3  # (Int): number of blades
    # ------------------

    # --- airfoil files ---
    import rotorse
    basepath = os.path.join(os.path.dirname(rotorse.__file__), '5MW_AFFiles')
    # basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), '5MW_AFFiles')

    # load all airfoils
    airfoil_types = [0]*8
    airfoil_types[0] = os.path.join(basepath, 'Cylinder1.dat')
    airfoil_types[1] = os.path.join(basepath, 'Cylinder2.dat')
    airfoil_types[2] = os.path.join(basepath, 'DU40_A17.dat')
    airfoil_types[3] = os.path.join(basepath, 'DU35_A17.dat')
    airfoil_types[4] = os.path.join(basepath, 'DU30_A17.dat')
    airfoil_types[5] = os.path.join(basepath, 'DU25_A17.dat')
    airfoil_types[6] = os.path.join(basepath, 'DU21_A17.dat')
    airfoil_types[7] = os.path.join(basepath, 'NACA64_A17.dat')

    # place at appropriate radial stations
    af_idx = [0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7]

    n = len(af_idx)
    af = [0]*n
    for i in range(n):
        af[i] = airfoil_types[af_idx[i]]
    rotor.airfoil_files = af  # (List): names of airfoil file
    # ----------------------

    # --- control ---
    rotor.control.Vin = 3.0  # (Float, m/s): cut-in wind speed
    rotor.control.Vout = 25.0  # (Float, m/s): cut-out wind speed
    #rotor.control.ratedPower = 5e6  # (Float, W): rated power
    turbine.machine_rating = 5e3 # (Float, kW): rated power
    rotor.control.minOmega = 0.0  # (Float, rpm): minimum allowed rotor rotation speed
    rotor.control.maxOmega = 12.0  # (Float, rpm): maximum allowed rotor rotation speed
    rotor.control.tsr = 7.55  # (Float): tip-speed ratio in Region 2 (should be optimized externally)
    rotor.control.pitch = 0.0  # (Float, deg): pitch angle in region 2 (and region 3 for fixed pitch machines)
    rotor.pitch_extreme = 0.0  # (Float, deg): worst-case pitch at survival wind condition
    rotor.azimuth_extreme = 0.0  # (Float, deg): worst-case azimuth at survival wind condition
    rotor.VfactorPC = 0.7  # (Float): fraction of rated speed at which the deflection is assumed to representative throughout the power curve calculation
    # ----------------------

    # --- aero and structural analysis options ---
    rotor.nSector = 4  # (Int): number of sectors to divide rotor face into in computing thrust and power
    rotor.npts_coarse_power_curve = 20  # (Int): number of points to evaluate aero analysis at
    rotor.npts_spline_power_curve = 200  # (Int): number of points to use in fitting spline to power curve
    rotor.AEP_loss_factor = 1.0  # (Float): availability and other losses (soiling, array, etc.)
    rotor.drivetrainType = 'geared'  # (Enum)
    rotor.nF = 5  # (Int): number of natural frequencies to compute
    rotor.dynamic_amplication_tip_deflection = 1.35  # (Float): a dynamic amplification factor to adjust the static deflection calculation
    # ----------------------

    # --- materials and composite layup  ---
    basepath = os.path.join(os.path.dirname(rotorse.__file__), '5MW_PreCompFiles')
    # basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), '5MW_PrecompFiles')

    materials = Orthotropic2DMaterial.listFromPreCompFile(os.path.join(basepath, 'materials.inp'))

    ncomp = len(rotor.initial_str_grid)
    upper = [0]*ncomp
    lower = [0]*ncomp
    webs = [0]*ncomp
    profile = [0]*ncomp

    rotor.leLoc = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.498, 0.497, 0.465, 0.447, 0.43, 0.411,
        0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4,
        0.4, 0.4, 0.4, 0.4])    # (Array): array of leading-edge positions from a reference blade axis (usually blade pitch axis). locations are normalized by the local chord length. e.g. leLoc[i] = 0.2 means leading edge is 0.2*chord[i] from reference axis.  positive in -x direction for airfoil-aligned coordinate system
    rotor.sector_idx_strain_spar = [2]*ncomp  # (Array): index of sector for spar (PreComp definition of sector)
    rotor.sector_idx_strain_te = [3]*ncomp  # (Array): index of sector for trailing-edge (PreComp definition of sector)
    web1 = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.4114, 0.4102, 0.4094, 0.3876, 0.3755, 0.3639, 0.345, 0.3342, 0.3313, 0.3274, 0.323, 0.3206, 0.3172, 0.3138, 0.3104, 0.307, 0.3003, 0.2982, 0.2935, 0.2899, 0.2867, 0.2833, 0.2817, 0.2799, 0.2767, 0.2731, 0.2664, 0.2607, 0.2562, 0.1886, -1.0])
    web2 = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.5886, 0.5868, 0.5854, 0.5508, 0.5315, 0.5131, 0.4831, 0.4658, 0.4687, 0.4726, 0.477, 0.4794, 0.4828, 0.4862, 0.4896, 0.493, 0.4997, 0.5018, 0.5065, 0.5101, 0.5133, 0.5167, 0.5183, 0.5201, 0.5233, 0.5269, 0.5336, 0.5393, 0.5438, 0.6114, -1.0])
    web3 = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
    rotor.chord_str_ref = np.array([3.2612, 3.3100915356, 3.32587052924, 3.34159388653, 3.35823798667, 3.37384375335,
        3.38939112914, 3.4774055542, 3.49839685, 3.51343645709, 3.87017220335, 4.04645623801, 4.19408216643,
         4.47641008477, 4.55844487985, 4.57383098262, 4.57285771934, 4.51914315648, 4.47677655262, 4.40075650022,
         4.31069949379, 4.20483735936, 4.08985563932, 3.82931757126, 3.74220276467, 3.54415796922, 3.38732428502,
         3.24931446473, 3.23421422609, 3.22701537997, 3.21972125648, 3.08979310611, 2.95152261813, 2.330753331,
         2.05553464181, 1.82577817774, 1.5860853279, 1.4621])  # (Array, m): chord distribution for reference section, thickness of structural layup scaled with reference thickness (fixed t/c for this case)

    for i in range(ncomp):

        webLoc = []
        if web1[i] != -1:
            webLoc.append(web1[i])
        if web2[i] != -1:
            webLoc.append(web2[i])
        if web3[i] != -1:
            webLoc.append(web3[i])

        upper[i], lower[i], webs[i] = CompositeSection.initFromPreCompLayupFile(os.path.join(basepath, 'layup_' + str(i+1) + '.inp'), webLoc, materials)
        profile[i] = Profile.initFromPreCompFile(os.path.join(basepath, 'shape_' + str(i+1) + '.inp'))

    rotor.materials = materials  # (List): list of all Orthotropic2DMaterial objects used in defining the geometry
    rotor.upperCS = upper  # (List): list of CompositeSection objections defining the properties for upper surface
    rotor.lowerCS = lower  # (List): list of CompositeSection objections defining the properties for lower surface
    rotor.websCS = webs  # (List): list of CompositeSection objections defining the properties for shear webs
    rotor.profile = profile  # (List): airfoil shape at each radial position
    # --------------------------------------


    # --- fatigue ---
    rotor.rstar_damage = np.array([0.000, 0.022, 0.067, 0.111, 0.167, 0.233, 0.300, 0.367, 0.433, 0.500,
        0.567, 0.633, 0.700, 0.767, 0.833, 0.889, 0.933, 0.978])  # (Array): nondimensional radial locations of damage equivalent moments
    rotor.Mxb_damage = 1e3*np.array([2.3743E+003, 2.0834E+003, 1.8108E+003, 1.5705E+003, 1.3104E+003,
        1.0488E+003, 8.2367E+002, 6.3407E+002, 4.7727E+002, 3.4804E+002, 2.4458E+002, 1.6339E+002,
        1.0252E+002, 5.7842E+001, 2.7349E+001, 1.1262E+001, 3.8549E+000, 4.4738E-001])  # (Array, N*m): damage equivalent moments about blade c.s. x-direction
    rotor.Myb_damage = 1e3*np.array([2.7732E+003, 2.8155E+003, 2.6004E+003, 2.3933E+003, 2.1371E+003,
        1.8459E+003, 1.5582E+003, 1.2896E+003, 1.0427E+003, 8.2015E+002, 6.2449E+002, 4.5229E+002,
        3.0658E+002, 1.8746E+002, 9.6475E+001, 4.2677E+001, 1.5409E+001, 1.8426E+000])  # (Array, N*m): damage equivalent moments about blade c.s. y-direction
    rotor.strain_ult_spar = 1.0e-2  # (Float): ultimate strain in spar cap
    rotor.strain_ult_te = 2500*1e-6 * 2   # (Float): uptimate strain in trailing-edge panels, note that I am putting a factor of two for the damage part only.
    rotor.eta_damage = 1.35*1.3*1.0  # (Float): safety factor for fatigue
    rotor.m_damage = 10.0  # (Float): slope of S-N curve for fatigue analysis
    rotor.N_damage = 365*24*3600*20.0  # (Float): number of cycles used in fatigue analysis  TODO: make function of rotation speed
    # ----------------
    # =================

    # === nacelle ======
    nacelle.L_ms = 1.0  # (Float, m): main shaft length downwind of main bearing in low-speed shaft
    nacelle.L_mb = 2.5  # (Float, m): main shaft length in low-speed shaft

    nacelle.h0_front = 1.7  # (Float, m): height of Ibeam in bedplate front
    nacelle.h0_rear = 1.35  # (Float, m): height of Ibeam in bedplate rear

    # TODO: sync with rotor drivetrainType variable
    nacelle.drivetrain_design = 'geared'
    nacelle.crane = True  # (Bool): flag for presence of crane
    nacelle.bevel = 0  # (Int): Flag for the presence of a bevel stage - 1 if present, 0 if not
    nacelle.gear_configuration = 'eep'  # (Str): tring that represents the configuration of the gearbox (stage number and types)

    nacelle.Np = [3, 3, 1]  # (Array): number of planets in each stage
    nacelle.ratio_type = 'optimal'  # (Str): optimal or empirical stage ratios
    nacelle.shaft_type = 'normal'  # (Str): normal or short shaft length
    #nacelle.shaft_angle = 5.0  # (Float, deg): Angle of the LSS inclindation with respect to the horizontal
    nacelle.shaft_ratio = 0.10  # (Float): Ratio of inner diameter to outer diameter.  Leave zero for solid LSS
    #nacelle.shrink_disc_mass = 1000.0  # (Float, kg): Mass of the shrink disc
    nacelle.mb1Type = 'CARB'  # (Str): Main bearing type: CARB, TRB or SRB
    nacelle.mb2Type = 'SRB'  # (Str): Second bearing type: CARB, TRB or SRB
    nacelle.yaw_motors_number = 8.0  # (Float): number of yaw motors
    nacelle.uptower_transformer = True
    nacelle.flange_length = 0.5 #m
    nacelle.gearbox_cm = 0.1
    nacelle.hss_length = 1.5
    nacelle.overhang = 5.0 #TODO - should come from turbine configuration level

    nacelle.check_fatigue = 0 #0 if no fatigue check, 1 if parameterized fatigue check, 2 if known loads inputs

    # TODO: should come from rotor (these are FAST outputs)
    nacelle.DrivetrainEfficiency = 0.95
    nacelle.rotor_bending_moment_x = 330770.0# Nm
    nacelle.rotor_bending_moment_y = -16665000.0 # Nm
    nacelle.rotor_bending_moment_z = 2896300.0 # Nm
    nacelle.rotor_force_x = 599610.0 # N
    nacelle.rotor_force_y = 186780.0 # N
    nacelle.rotor_force_z = -842710.0 # N

    #nacelle.h0_rear = 1.35 # only used in drive smooth
    #nacelle.h0_front = 1.7

    # =================

    # === jacket ===

    #--- Set Jacket Input Parameters ---#
    Jcktins=JcktGeoInputs()
    Jcktins.nlegs =4
    Jcktins.nbays =5
    Jcktins.batter=12.
    Jcktins.dck_botz =16.
    Jcktins.weld2D   =0.5
    Jcktins.VPFlag = True    #vertical pile T/F;  to enable piles in frame3DD set pileinputs.ndiv>0
    Jcktins.clamped= False    #whether or not the bottom of the structure is rigidly connected. Use False when equivalent spring constants are being used.
    Jcktins.AFflag = False  #whether or not to use apparent fixity piles
    Jcktins.PreBuildTPLvl = 2  #if >0, the TP is prebuilt according to rules per PreBuildTP

    #Soil inputs
    Soilinputs=SoilGeoInputs()
    Soilinputs.zbots   =-np.array([3.,5.,7.,15.,30.,50.])
    Soilinputs.gammas  =np.array([10000.,10000.,10000.,10000.,10000.,10000.])
    Soilinputs.cus     =np.array([60000.,60000.,60000.,60000.,60000.,60000.])
    Soilinputs.phis    =np.array([26.,26.,26.,26.,26.,26])#np.array([36.,33.,26.,37.,35.,37.5])#np.array([36.,33.,26.,37.,35.,37.5])
    Soilinputs.delta   =25.
    Soilinputs.sndflg   =True
    Soilinputs.PenderSwtch   =False #True
    Soilinputs.SoilSF   =1.

    #Water and wind inputs
    Waterinputs=WaterInputs()
    Waterinputs.wdepth   =30.
    Waterinputs.wlevel   =30. #Distance from bottom of structure to surface  THIS, I believe is no longer needed as piles may be negative in z, to check and remove in case
    Waterinputs.T=12.  #Wave Period
    Waterinputs.HW=10. #Wave Height
    '''Windinputs=WindInputs()
    Windinputs.HH=100. #CHECK HOW THIS COMPLIES....
    Windinputs.U50HH=30. #assumed gust speed'''

    #RNA loads              Fx-z,         Mxx-zz
    #RNA_F=np.array([1000.e3,0.,0.,0.,0.,0.])

    #Pile data
    Pilematin=MatInputs()
    Pilematin.matname=np.array(['steel'])
    Pilematin.E=np.array([ 25.e9])
    Dpile=2.5#0.75 # 2.0
    tpile=0.01
    Lp=20. #45

    Pileinputs=PileGeoInputs()
    Pileinputs.Pilematins=Pilematin
    Pileinputs.ndiv=0 #3
    Pileinputs.Dpile=Dpile
    Pileinputs.tpile=tpile
    Pileinputs.Lp=Lp #[m] Embedment length

    #Legs data
    legmatin=MatInputs()
    legmatin.matname=(['steel','steel','steel','steel'])
    legmatin.E=np.array([2.0e11])
    Dleg=np.array([2.0,1.8,1.8,1.8,1.8,1.8])
    tleg=1.55*np.array([0.0254]).repeat(Dleg.size)
    leginputs=LegGeoInputs()
    leginputs.legZbot   = 1.0
    leginputs.ndiv=1
    leginputs.legmatins=legmatin
    leginputs.Dleg=Dleg
    leginputs.tleg=tleg

    legbot_stmphin =1.5  #Distance from bottom of leg to second joint along z; must be>0

    #Xbrc data
    Xbrcmatin=MatInputs()
    Xbrcmatin.matname=np.array(['steel']).repeat(Jcktins.nbays)
    Xbrcmatin.E=np.array([ 2.2e11, 2.0e11,2.0e11,2.0e11,2.0e11])
    Dbrc=np.array([1.,1.,1.0,1.0,1.0])
    tbrc=np.array([1.,1.,1.0,1.0,1.0])*0.0254

    Xbrcinputs=XBrcGeoInputs()
    Xbrcinputs.Dbrc=Dbrc
    Xbrcinputs.tbrc=tbrc
    Xbrcinputs.ndiv=2#2
    Xbrcinputs.Xbrcmatins=Xbrcmatin
    Xbrcinputs.precalc=True   #This can be set to true if we want Xbraces to be precalculated in D and t, in which case the above set Dbrc and tbrc would be overwritten

    #Mbrc data
    Mbrcmatin=MatInputs()
    Mbrcmatin.matname=np.array(['steel'])
    Mbrcmatin.E=np.array([ 2.5e11])
    Dbrc_mud=1.5

    Mbrcinputs=MudBrcGeoInputs()
    Mbrcinputs.Dbrc_mud=Dbrc_mud
    Mbrcinputs.ndiv=2
    Mbrcinputs.Mbrcmatins=Mbrcmatin
    Mbrcinputs.precalc=True   #This can be set to true if we want Mudbrace to be precalculated in D and t, in which case the above set Dbrc_mud and tbrc_mud would be overwritten
    #Hbrc data
    Hbrcmatin=MatInputs()
    Hbrcmatin.matname=np.array(['steel'])
    Hbrcmatin.E=np.array([ 2.5e11])
    Dbrc_hbrc=1.1

    Hbrcinputs=HBrcGeoInputs()
    Hbrcinputs.Dbrch=Dbrc_hbrc
    Hbrcinputs.ndiv=0#2
    Hbrcinputs.Hbrcmatins=Hbrcmatin
    Hbrcinputs.precalc=True   #This can be set to true if we want Hbrace to be set=Xbrace top D and t, in which case the above set Dbrch and tbrch would be overwritten

    #TP data
    TPlumpinputs=TPlumpMass()
    TPlumpinputs.mass=300.e3 #[kg]

    TPstmpsmatin=MatInputs()
    TPbrcmatin=MatInputs()
    TPstemmatin=MatInputs()
    TPbrcmatin.matname=np.array(['steel'])
    TPbrcmatin.E=np.array([ 2.5e11])
    TPstemmatin.matname=np.array(['steel']).repeat(2)
    TPstemmatin.E=np.array([ 2.1e11]).repeat(2)

    TPinputs=TPGeoInputs()
    TPinputs.TPbrcmatins=TPbrcmatin
    TPinputs.TPstemmatins=TPstemmatin
    TPinputs.TPstmpmatins=TPstmpsmatin
    TPinputs.Dstrut=1.6
    TPinputs.Dgir=Dbrc_hbrc
    TPinputs.Dbrc=1.1
    TPinputs.hstump=0.0#1.0
    TPinputs.stumpndiv=1#2
    TPinputs.brcndiv=1#2
    TPinputs.girndiv=1#2
    TPinputs.strutndiv=1#2
    TPinputs.stemndiv=1#2
    TPinputs.nstems=3
    TPinputs.Dstem=np.array([6.]).repeat(TPinputs.nstems)
    TPinputs.tstem=np.array([0.1,0.11,0.11])
    TPinputs.hstem=np.array([4.,3.,1.])

    #Tower data
    Twrmatin=MatInputs()
    Twrmatin.matname=np.array(['steel'])
    Twrmatin.E=np.array([ 2.77e11])
    Db=5.6
    tb=0.05
    Dt=Db*0.55

    '''Twrinputs=TwrGeoInputs()
    Twrinputs.Twrmatins=Twrmatin
    #Twrinputs.Htwr=70.  #Trumped by HH
    Twrinputs.Htwr2frac=0.2   #fraction of tower height with constant x-section
    Twrinputs.ndiv=np.array([6,6])  #ndiv for uniform and tapered section
    Twrinputs.Db=Db
    Twrinputs.DTRb=Db/tb
    #Twrinputs.Dt=Dt'''

    TwrRigidTop=True #False       #False=Account for RNA via math rather than a physical rigidmember

    #RNA data
    '''RNAins=RNAprops()
    RNAins.mass=3*350.e3
    RNAins.I[0]=86.579E+6
    RNAins.I[1]=53.530E+6
    RNAins.I[2]=58.112E+6
    RNAins.CMoff[2]=2.34
    RNAins.yawangle=45.  #angle with respect to global X, CCW looking from above, wind from left
    RNAins.rna_weightM=True'''

    #Frame3DD parameters
    FrameAuxIns=Frame3DDaux()
    FrameAuxIns.sh_fg=1               #shear flag-->Timoshenko
    FrameAuxIns.deltaz=5.
    FrameAuxIns.geo_fg=0
    FrameAuxIns.nModes = 6             # number of desired dynamic modes of vibration
    FrameAuxIns.Mmethod = 1            # 1: subspace Jacobi     2: Stodola
    FrameAuxIns.lump = 0               # 0: consistent mass ... 1: lumped mass matrix
    FrameAuxIns.tol = 1e-9             # mode shape tolerance
    FrameAuxIns.shift = 0.0            # shift value ... for unrestrained structures
    FrameAuxIns.gvector=np.array([0.,0.,-9.8065])    #GRAVITY

    #Pass all inputs to jacket assembly
    jacket.JcktGeoIn=Jcktins
    jacket.Soilinputs=Soilinputs
    jacket.Waterinputs=Waterinputs
    #jacket.Windinputs=Windinputs
    #jacket.RNA_F=RNA_F
    jacket.Pileinputs=Pileinputs
    jacket.leginputs=leginputs
    jacket.legbot_stmphin =legbot_stmphin
    jacket.Xbrcinputs=Xbrcinputs
    jacket.Mbrcinputs=Mbrcinputs
    jacket.Hbrcinputs=Hbrcinputs
    jacket.TPlumpinputs=TPlumpinputs
    jacket.TPinputs=TPinputs
    #jacket.RNAinputs=RNAins
    #jacket.Twrinputs=Twrinputs
    jacket.Twrinputs.Twrmatins=Twrmatin
    jacket.Twrinputs.Htwr2frac=0.2   #fraction of tower height with constant x-section
    jacket.Twrinputs.ndiv=np.array([6,6])  #ndiv for uniform and tapered section
    jacket.Twrinputs.Db=Db
    jacket.Twrinputs.DTRb=Db/tb
    jacket.TwrRigidTop=TwrRigidTop
    jacket.FrameAuxIns=FrameAuxIns

    # =================

    # === run ===
    turbine.run()

    print [c.name for c in turbine.driver.workflow]
    print 'mass rotor blades (kg) =', turbine.rotor.mass_all_blades
    print 'mass hub system (kg) =', turbine.hub.hub_system_mass
    print 'mass nacelle (kg) =', turbine.nacelle.nacelle_mass
    print 'mass tower (kg) =', jacket.Tower.Twrouts.mass
    print 'maximum tip deflection (m) =', turbine.maxdeflection.max_tip_deflection
    print 'ground clearance (m) =', turbine.maxdeflection.ground_clearance


    # Jacket specific outputs
    print
    print('First two Freqs.= {:5.4f} and {:5.4f} Hz'.format(*jacket.Frameouts.Freqs))
    print
    #print component masses
    print('jacket+TP(structural+lumped) mass (no tower, no piles) [kg] = {:6.0f}'.format(jacket.Frameouts.mass[0]+jacket.TP.TPlumpinputs.mass-jacket.Tower.Twrouts.mass))
    print('tower mass [kg] = {:6.0f}'.format(jacket.Tower.Twrouts.mass))
    print('TP mass structural + lumped mass [kg] = {:6.0f}'.format(jacket.TP.TPouts.mass+jacket.TP.TPlumpinputs.mass))
    print('piles (all) mass (for assigned (not optimum) Lp [kg] = {:6.0f}'.format(jacket.Mpiles))
    print('frame3dd model mass (structural + TP lumped) [kg] = {:6.0f}'.format(jacket.Frameouts.mass[0]+jacket.TP.TPlumpinputs.mass))
    print

    #print tower top displacement
    print('Tower Top Displacement in Global Coordinate System [m] ={:5.4f}'.format(*jacket.Frameouts.top_deflection))
    print

    #print max API code checks
    print('MAX member compression-bending utilization at joints = {:5.4f}'.format(np.max(jacket.jacket_utilization.cb_util)))
    print('MAX member tension utilization at joints = {:5.4f}'.format(np.max(jacket.jacket_utilization.t_util)))
    print('MAX X-joint  utilization at joints = {:5.4f}'.format(np.max(jacket.jacket_utilization.XjntUtil)))
    print('MAX K-joint  utilization at joints = {:5.4f}'.format(np.max(jacket.jacket_utilization.KjntUtil)))

    # =================
