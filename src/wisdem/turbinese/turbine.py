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
from towerse.tower import TowerSE
from commonse.rna import RNAMass, RotorLoads
from drivewpact.drive import DriveWPACT
from drivewpact.hub import HubWPACT
from commonse.csystem import DirectionVector
from commonse.utilities import interp_with_deriv, hstack, vstack
from drivese.drive_smooth import NacelleTS
from drivese.drive import Drive4pt, Drive3pt
from drivese.drivese_utils import blade_moment_transform
from drivese.hub import HubSE, Hub_System_Adder_drive


class MaxTipDeflection(Component):

    Rtip = Float(iotype='in', units='m')
    precurveTip = Float(iotype='in', units='m')  # TODO: add gradients for precurveTip and presweepTip
    presweepTip = Float(iotype='in', units='m')
    precone = Float(iotype='in', units='deg')
    tilt = Float(iotype='in', units='deg')
    hub_tt = Array(iotype='in', units='m', desc='location of hub relative to tower-top in yaw-aligned c.s.')
    tower_z = Array(iotype='in', units='m')
    tower_d = Array(iotype='in', units='m')
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

    # --- general turbine configuration inputs---
    assembly.add('rho', Float(1.225, iotype='in', units='kg/m**3', desc='density of air', deriv_ignore=True))
    assembly.add('mu', Float(1.81206e-5, iotype='in', units='kg/m/s', desc='dynamic viscosity of air', deriv_ignore=True))
    assembly.add('shear_exponent', Float(0.2, iotype='in', desc='shear exponent', deriv_ignore=True))
    assembly.add('hub_height', Float(90.0, iotype='in', units='m', desc='hub height'))
    assembly.add('turbine_class', Enum('I', ('I', 'II', 'III', 'IV'), iotype='in', desc='IEC turbine class'))
    assembly.add('turbulence_class', Enum('B', ('A', 'B', 'C'), iotype='in', desc='IEC turbulence class class'))
    assembly.add('g', Float(9.81, iotype='in', units='m/s**2', desc='acceleration of gravity', deriv_ignore=True))
    assembly.add('cdf_reference_height_wind_speed', Float(90.0, iotype='in', desc='reference hub height for IEC wind speed (used in CDF calculation)'))
    assembly.add('downwind', Bool(False, iotype='in', desc='flag if rotor is downwind'))
    assembly.add('tower_d', Array([0.0], iotype='in', units='m', desc='diameters along tower'))
    assembly.add('generator_speed', Float(iotype='in', units='rpm', desc='generator speed'))
    assembly.add('machine_rating', Float(5000.0, units='kW', iotype='in', desc='machine rated power'))
    assembly.add('rna_weightM', Bool(True, iotype='in', desc='flag to consider or not the RNA weight effect on Moment'))

    assembly.add('rotor', RotorSE())
    if with_new_nacelle:
        assembly.add('hub',HubSE())
        assembly.add('hubSystem',Hub_System_Adder_drive())
        assembly.add('moments',blade_moment_transform())
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
    assembly.add('tower', TowerSE())
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

    assembly.driver.workflow.add(['hub', 'nacelle', 'tower', 'maxdeflection', 'rna', 'rotorloads1', 'rotorloads2'])

    if with_new_nacelle:
        assembly.driver.workflow.add(['hubSystem','moments'])

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


    # connections to hub and hub system
    assembly.connect('rotor.mass_one_blade', 'hub.blade_mass')
    assembly.connect('rotor.root_bending_moment', 'hub.rotor_bending_moment')
    assembly.connect('rotor.diameter', ['hub.rotor_diameter','hubSystem.rotor_diameter'])
    assembly.connect('rotor.hub_diameter', 'hub.blade_root_diameter')
    assembly.connect('rotor.nBlades', 'hub.blade_number')
    if with_new_nacelle:
        # TODO: circular dependency
        assembly.connect('nacelle.MB1_location','hubSystem.MB1_location')
        assembly.connect('nacelle.L_rb','hubSystem.L_rb')
        assembly.connect('rotor.tilt','hubSystem.shaft_angle')
        assembly.connect('hub.hub_diameter','hubSystem.hub_diameter')
        assembly.connect('hub.hub_thickness','hubSystem.hub_thickness')
        assembly.connect('hub.hub_mass','hubSystem.hub_mass')
        assembly.connect('hub.spinner_mass','hubSystem.spinner_mass')
        assembly.connect('hub.pitch_system_mass','hubSystem.pitch_system_mass')

    # connections to nacelle #TODO: fatigue option variables
    assembly.connect('rotor.diameter', 'nacelle.rotor_diameter')
    assembly.connect('1.5 * rotor.ratedConditions.Q', 'nacelle.rotor_torque')
    assembly.connect('rotor.ratedConditions.T', 'nacelle.rotor_thrust')
    assembly.connect('rotor.ratedConditions.Omega', 'nacelle.rotor_speed')
    assembly.connect('machine_rating', 'nacelle.machine_rating')
    assembly.connect('rotor.root_bending_moment', 'nacelle.rotor_bending_moment')
    assembly.connect('generator_speed/rotor.ratedConditions.Omega', 'nacelle.gear_ratio')
    assembly.connect('tower_d[-1]', 'nacelle.tower_top_diameter')  # OpenMDAO circular dependency issue
    assembly.connect('rotor.mass_all_blades + hub.hub_system_mass', 'nacelle.rotor_mass') # assuming not already in rotor force / moments
    # variable connections for new nacelle
    if with_new_nacelle:
        assembly.connect('rotor.nBlades','nacelle.blade_number')
        assembly.connect('rotor.tilt','nacelle.shaft_angle')
        assembly.connect('333.3 * machine_rating / 1000.0','nacelle.shrink_disc_mass')
        assembly.connect('rotor.hub_diameter','nacelle.blade_root_diameter')
        # assembly.connect('rotor.ratedConditions.Q','nacelle.rotor_bending_moment_x')
        assembly.connect('rotor.Mxyz_0','moments.b1')
        assembly.connect('rotor.Mxyz_120','moments.b2')
        assembly.connect('rotor.Mxyz_240','moments.b3')
        assembly.connect('rotor.Pitch','moments.pitch_angle')
        assembly.connect('rotor.TotalCone','moments.cone_angle')
        assembly.connect('moments.Mx','nacelle.rotor_bending_moment_x')
        assembly.connect('moments.My','nacelle.rotor_bending_moment_y')
        assembly.connect('moments.Mz','nacelle.rotor_bending_moment_z')
        assembly.connect('rotor.ratedConditions.T','nacelle.rotor_force_x')
        #TODO find/add force calculations from rotor
        #assembly.connect('...','nacelle.rotor_force_y')
        #assembly.connect('...','nacelle.rotor_force_z')

    '''if  with_new_nacelle:
        assembly.connect('rotor.g', 'nacelle.g') # Only drive smooth taking g from rotor; TODO: update when drive_smooth is updated'''


    # connections to rna
    assembly.connect('rotor.mass_all_blades', 'rna.blades_mass')
    assembly.connect('rotor.I_all_blades', 'rna.blades_I')
    assembly.connect('hubSystem.hub_system_mass', 'rna.hub_mass')
    assembly.connect('hubSystem.hub_system_cm', 'rna.hub_cm')
    assembly.connect('hubSystem.hub_system_I', 'rna.hub_I')
    assembly.connect('nacelle.nacelle_mass', 'rna.nac_mass')
    assembly.connect('nacelle.nacelle_cm', 'rna.nac_cm')
    assembly.connect('nacelle.nacelle_I', 'rna.nac_I')

    # connections to rotorloads1
    assembly.connect('downwind', 'rotorloads1.downwind')
    assembly.connect('rna_weightM', 'rotorloads1.rna_weightM')
    assembly.connect('1.8 * rotor.ratedConditions.T', 'rotorloads1.F[0]')
    assembly.connect('rotor.ratedConditions.Q', 'rotorloads1.M[0]')
    assembly.connect('hubSystem.hub_system_cm', 'rotorloads1.r_hub')
    assembly.connect('rna.rna_cm', 'rotorloads1.rna_cm')
    assembly.connect('rotor.tilt', 'rotorloads1.tilt')
    assembly.connect('g', 'rotorloads1.g')
    assembly.connect('rna.rna_mass', 'rotorloads1.m_RNA')

    # connections to rotorloads2
    assembly.connect('downwind', 'rotorloads2.downwind')
    assembly.connect('rna_weightM', 'rotorloads2.rna_weightM')
    assembly.connect('rotor.T_extreme', 'rotorloads2.F[0]')
    assembly.connect('rotor.Q_extreme', 'rotorloads2.M[0]')
    assembly.connect('hubSystem.hub_system_cm', 'rotorloads2.r_hub')
    assembly.connect('rna.rna_cm', 'rotorloads2.rna_cm')
    assembly.connect('rotor.tilt', 'rotorloads2.tilt')
    assembly.connect('g', 'rotorloads2.g')
    assembly.connect('rna.rna_mass', 'rotorloads2.m_RNA')

    # connections to tower
    assembly.connect('rho', 'tower.wind_rho')
    assembly.connect('mu', 'tower.wind_mu')
    assembly.connect('g', 'tower.g')
    assembly.connect('hub_height', 'tower.wind_zref')
    assembly.connect('tower_d', 'tower.d')
    assembly.connect('rotor.ratedConditions.V', 'tower.wind_Uref1')
    assembly.connect('rotor.V_extreme', 'tower.wind_Uref2')
    assembly.connect('rotor.yaw', 'tower.yaw')
    assembly.connect('hub_height - nacelle.nacelle_cm[2]', 'tower.towerHeight')
    assembly.connect('rna.rna_mass', 'tower.top_m')
    assembly.connect('rna.rna_cm', 'tower.top_cm')
    assembly.connect('rna.rna_I_TT', 'tower.top_I')
    assembly.connect('rotorloads1.top_F', 'tower.top1_F')
    assembly.connect('rotorloads1.top_M', 'tower.top1_M')
    assembly.connect('rotorloads2.top_F', 'tower.top2_F')
    assembly.connect('rotorloads2.top_M', 'tower.top2_M')

    # connections to maxdeflection
    assembly.connect('rotor.Rtip', 'maxdeflection.Rtip')
    assembly.connect('rotor.precurveTip', 'maxdeflection.precurveTip')
    assembly.connect('rotor.presweepTip', 'maxdeflection.presweepTip')
    assembly.connect('rotor.precone', 'maxdeflection.precone')
    assembly.connect('rotor.tilt', 'maxdeflection.tilt')
    assembly.connect('hubSystem.hub_system_cm', 'maxdeflection.hub_tt')
    assembly.connect('tower.z', 'maxdeflection.tower_z')
    assembly.connect('tower_d', 'maxdeflection.tower_d')
    assembly.connect('tower.towerHeight', 'maxdeflection.towerHt')



class TurbineSE(Assembly):

    def configure(self):
        configure_turbine(self)


if __name__ == '__main__':

    turbine = TurbineSE()
    turbine.sea_depth = 0.0 # 0.0 for land-based turbine
    wind_class = 'I'

    # from wisdem.reference_turbines.nrel5mw.nrel5mw import configure_nrel5mw_turbine
    # configure_nrel5mw_turbine(turbine,wind_class,turbine.sea_depth)

    #copied from reference_turbine due to import error TODO delete...
    import os
    from rotorse.precomp import Profile, Orthotropic2DMaterial, CompositeSection  # TODO: can just pass file names and do this initialization inside of rotor
    from towerse.tower import TowerWithpBEAM
    from commonse.environment import PowerWind, TowerSoil, LinearWaves
    from commonse.utilities import cosd, sind
    from rotorse.rotoraero import RS2RPM
        # === Turbine ===
    turbine.rho = 1.225  # (Float, kg/m**3): density of air
    turbine.mu = 1.81206e-5  # (Float, kg/m/s): dynamic viscosity of air
    turbine.shear_exponent = 0.2  # (Float): shear exponent
    turbine.hub_height = 90.0  # (Float, m): hub height
    turbine.turbine_class = 'I'  # (Enum): IEC turbine class
    turbine.turbulence_class = 'B'  # (Enum): IEC turbulence class class
    turbine.cdf_reference_height_wind_speed = 90.0  # (Float): reference hub height for IEC wind speed (used in CDF calculation)
    turbine.g = 9.81  # (Float, m/s**2): acceleration of gravity
    # ======================

    # === rotor ===
    # --- blade grid ---
    turbine.rotor.initial_aero_grid = np.array([0.02222276, 0.06666667, 0.11111057, 0.16666667, 0.23333333, 0.3, 0.36666667,
        0.43333333, 0.5, 0.56666667, 0.63333333, 0.7, 0.76666667, 0.83333333, 0.88888943, 0.93333333,
        0.97777724])  # (Array): initial aerodynamic grid on unit radius
    turbine.rotor.initial_str_grid = np.array([0.0, 0.00492790457512, 0.00652942887106, 0.00813095316699, 0.00983257273154,
        0.0114340970275, 0.0130356213234, 0.02222276, 0.024446481932, 0.026048006228, 0.06666667, 0.089508406455,
        0.11111057, 0.146462614229, 0.16666667, 0.195309105255, 0.23333333, 0.276686558545, 0.3, 0.333640766319,
        0.36666667, 0.400404310407, 0.43333333, 0.5, 0.520818918408, 0.56666667, 0.602196371696, 0.63333333,
        0.667358391486, 0.683573824984, 0.7, 0.73242031601, 0.76666667, 0.83333333, 0.88888943, 0.93333333, 0.97777724,
        1.0])  # (Array): initial structural grid on unit radius
    turbine.rotor.idx_cylinder_aero = 3  # (Int): first idx in r_aero_unit of non-cylindrical section, constant twist inboard of here
    turbine.rotor.idx_cylinder_str = 14  # (Int): first idx in r_str_unit of non-cylindrical section
    turbine.rotor.hubFraction = 0.025  # (Float): hub location as fraction of radius
    # ------------------

    # --- blade geometry ---
    turbine.rotor.r_aero = np.array([0.02222276, 0.06666667, 0.11111057, 0.2, 0.23333333, 0.3, 0.36666667, 0.43333333,
        0.5, 0.56666667, 0.63333333, 0.64, 0.7, 0.83333333, 0.88888943, 0.93333333,
        0.97777724])  # (Array): new aerodynamic grid on unit radius
    turbine.rotor.r_max_chord = 0.23577  # (Float): location of max chord on unit radius
    turbine.rotor.chord_sub = [3.2612, 4.5709, 3.3178, 1.4621]  # (Array, m): chord at control points. defined at hub, then at linearly spaced locations from r_max_chord to tip
    turbine.rotor.theta_sub = [13.2783, 7.46036, 2.89317, -0.0878099]  # (Array, deg): twist at control points.  defined at linearly spaced locations from r[idx_cylinder] to tip
    turbine.rotor.precurve_sub = [0.0, 0.0, 0.0]  # (Array, m): precurve at control points.  defined at same locations at chord, starting at 2nd control point (root must be zero precurve)
    turbine.rotor.delta_precurve_sub = [0.0, 0.0, 0.0]  # (Array, m): adjustment to precurve to account for curvature from loading
    turbine.rotor.sparT = [0.05, 0.047754, 0.045376, 0.031085, 0.0061398]  # (Array, m): spar cap thickness parameters
    turbine.rotor.teT = [0.1, 0.09569, 0.06569, 0.02569, 0.00569]  # (Array, m): trailing-edge thickness parameters
    turbine.rotor.bladeLength = 61.5  # (Float, m): blade length (if not precurved or swept) otherwise length of blade before curvature
    turbine.rotor.delta_bladeLength = 0.0  # (Float, m): adjustment to blade length to account for curvature from loading
    turbine.rotor.precone = 2.5  # (Float, deg): precone angle
    turbine.rotor.tilt = 5.0  # (Float, deg): shaft tilt
    turbine.rotor.yaw = 0.0  # (Float, deg): yaw error
    turbine.rotor.nBlades = 3  # (Int): number of blades
    # ------------------

    # --- airfoil files ---
    import rotorse
    #basepath = os.path.join('5MW_files', '5MW_AFFiles')
    basepath = os.path.join('..','reference_turbines','nrel5mw','airfoils')

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
    turbine.rotor.airfoil_files = af  # (List): names of airfoil file
    # ----------------------

    # --- control ---
    turbine.rotor.control.Vin = 3.0  # (Float, m/s): cut-in wind speed
    turbine.rotor.control.Vout = 25.0  # (Float, m/s): cut-out wind speed
    turbine.rotor.control.ratedPower = 5e6  # (Float, W): rated power
    turbine.rotor.control.minOmega = 0.0  # (Float, rpm): minimum allowed rotor rotation speed
    turbine.rotor.control.maxOmega = 12.0  # (Float, rpm): maximum allowed rotor rotation speed
    turbine.rotor.control.tsr = 7.55  # (Float): tip-speed ratio in Region 2 (should be optimized externally)
    turbine.rotor.control.pitch = 0.0  # (Float, deg): pitch angle in region 2 (and region 3 for fixed pitch machines)
    turbine.rotor.pitch_extreme = 0.0  # (Float, deg): worst-case pitch at survival wind condition
    turbine.rotor.azimuth_extreme = 0.0  # (Float, deg): worst-case azimuth at survival wind condition
    turbine.rotor.VfactorPC = 0.7  # (Float): fraction of rated speed at which the deflection is assumed to representative throughout the power curve calculation
    # ----------------------

    # --- aero and structural analysis options ---
    turbine.rotor.nSector = 4  # (Int): number of sectors to divide rotor face into in computing thrust and power
    turbine.rotor.npts_coarse_power_curve = 20  # (Int): number of points to evaluate aero analysis at
    turbine.rotor.npts_spline_power_curve = 200  # (Int): number of points to use in fitting spline to power curve
    turbine.rotor.AEP_loss_factor = 1.0  # (Float): availability and other losses (soiling, array, etc.)
    turbine.rotor.drivetrainType = 'geared'  # (Enum)
    turbine.rotor.nF = 5  # (Int): number of natural frequencies to compute
    turbine.rotor.dynamic_amplication_tip_deflection = 1.35  # (Float): a dynamic amplification factor to adjust the static deflection calculation
    # ----------------------


    # --- materials and composite layup  ---
    #basepath = os.path.join('5MW_files', '5MW_PrecompFiles')
    basepath = os.path.join('..', 'reference_turbines','nrel5mw','blade')

    materials = Orthotropic2DMaterial.listFromPreCompFile(os.path.join(basepath, 'materials.inp'))

    ncomp = len(turbine.rotor.initial_str_grid)
    upper = [0]*ncomp
    lower = [0]*ncomp
    webs = [0]*ncomp
    profile = [0]*ncomp

    turbine.rotor.leLoc = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.498, 0.497, 0.465, 0.447, 0.43, 0.411,
        0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4,
        0.4, 0.4, 0.4, 0.4])    # (Array): array of leading-edge positions from a reference blade axis (usually blade pitch axis). locations are normalized by the local chord length. e.g. leLoc[i] = 0.2 means leading edge is 0.2*chord[i] from reference axis.  positive in -x direction for airfoil-aligned coordinate system
    turbine.rotor.sector_idx_strain_spar = [2]*ncomp  # (Array): index of sector for spar (PreComp definition of sector)
    turbine.rotor.sector_idx_strain_te = [3]*ncomp  # (Array): index of sector for trailing-edge (PreComp definition of sector)
    web1 = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.4114, 0.4102, 0.4094, 0.3876, 0.3755, 0.3639, 0.345, 0.3342, 0.3313, 0.3274, 0.323, 0.3206, 0.3172, 0.3138, 0.3104, 0.307, 0.3003, 0.2982, 0.2935, 0.2899, 0.2867, 0.2833, 0.2817, 0.2799, 0.2767, 0.2731, 0.2664, 0.2607, 0.2562, 0.1886, -1.0])
    web2 = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.5886, 0.5868, 0.5854, 0.5508, 0.5315, 0.5131, 0.4831, 0.4658, 0.4687, 0.4726, 0.477, 0.4794, 0.4828, 0.4862, 0.4896, 0.493, 0.4997, 0.5018, 0.5065, 0.5101, 0.5133, 0.5167, 0.5183, 0.5201, 0.5233, 0.5269, 0.5336, 0.5393, 0.5438, 0.6114, -1.0])
    web3 = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
    turbine.rotor.chord_str_ref = np.array([3.2612, 3.3100915356, 3.32587052924, 3.34159388653, 3.35823798667, 3.37384375335,
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

    turbine.rotor.materials = materials  # (List): list of all Orthotropic2DMaterial objects used in defining the geometry
    turbine.rotor.upperCS = upper  # (List): list of CompositeSection objections defining the properties for upper surface
    turbine.rotor.lowerCS = lower  # (List): list of CompositeSection objections defining the properties for lower surface
    turbine.rotor.websCS = webs  # (List): list of CompositeSection objections defining the properties for shear webs
    turbine.rotor.profile = profile  # (List): airfoil shape at each radial position
    # --------------------------------------

    strain_ult_spar = 1.0e-2
    strain_ult_te = 2500*1e-6

    # --- fatigue ---
    turbine.rotor.rstar_damage = np.array([0.000, 0.022, 0.067, 0.111, 0.167, 0.233, 0.300, 0.367, 0.433, 0.500,
        0.567, 0.633, 0.700, 0.767, 0.833, 0.889, 0.933, 0.978])  # (Array): nondimensional radial locations of damage equivalent moments
    turbine.rotor.Mxb_damage = 1e3*np.array([2.3743E+003, 2.0834E+003, 1.8108E+003, 1.5705E+003, 1.3104E+003,
        1.0488E+003, 8.2367E+002, 6.3407E+002, 4.7727E+002, 3.4804E+002, 2.4458E+002, 1.6339E+002,
        1.0252E+002, 5.7842E+001, 2.7349E+001, 1.1262E+001, 3.8549E+000, 4.4738E-001])  # (Array, N*m): damage equivalent moments about blade c.s. x-direction
    turbine.rotor.Myb_damage = 1e3*np.array([2.7732E+003, 2.8155E+003, 2.6004E+003, 2.3933E+003, 2.1371E+003,
        1.8459E+003, 1.5582E+003, 1.2896E+003, 1.0427E+003, 8.2015E+002, 6.2449E+002, 4.5229E+002,
        3.0658E+002, 1.8746E+002, 9.6475E+001, 4.2677E+001, 1.5409E+001, 1.8426E+000])  # (Array, N*m): damage equivalent moments about blade c.s. y-direction
    turbine.rotor.strain_ult_spar = 1.0e-2  # (Float): ultimate strain in spar cap
    turbine.rotor.strain_ult_te = 2500*1e-6 * 2   # (Float): uptimate strain in trailing-edge panels, note that I am putting a factor of two for the damage part only.
    turbine.rotor.eta_damage = 1.35*1.3*1.0  # (Float): safety factor for fatigue
    turbine.rotor.m_damage = 10.0  # (Float): slope of S-N curve for fatigue analysis
    turbine.rotor.N_damage = 365*24*3600*20.0  # (Float): number of cycles used in fatigue analysis  TODO: make function of rotation speed
    # ----------------
    # =================


    # === nacelle ======
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

    # TODO: should come from rotor (these are FAST outputs)
    turbine.nacelle.DrivetrainEfficiency = 0.95
    turbine.nacelle.rotor_bending_moment_x = 330770.0# Nm
    turbine.nacelle.rotor_bending_moment_y = -16665000.0 # Nm
    turbine.nacelle.rotor_bending_moment_z = 2896300.0 # Nm
    turbine.nacelle.rotor_force_x = 599610.0 # N
    turbine.nacelle.rotor_force_y = 186780.0 # N
    turbine.nacelle.rotor_force_z = -842710.0 # N'''

    #turbine.nacelle.h0_rear = 1.35 # only used in drive smooth
    #turbine.nacelle.h0_front = 1.7

    # =================

    # === tower ===

    turbine.tower.replace('wind1', PowerWind())
    turbine.tower.replace('wind2', PowerWind())
    # turbine.tower.replace('wave1', LinearWaves())  # no waves (onshore)
    turbine.tower.replace('soil', TowerSoil())
    turbine.tower.replace('tower1', TowerWithpBEAM())
    turbine.tower.replace('tower2', TowerWithpBEAM())

    # --- geometry ---
    turbine.tower.z = [0.0, 0.5, 1.0]  # (Array): locations along unit tower, linear lofting between
    turbine.tower.t = [0.027*1.3, 0.023*1.3, 0.019*1.3]  # (Array, m): shell thickness at corresponding locations
    turbine.tower.n = [10, 10]  # (Array): number of finite elements between sections.  array length should be ``len(z)-1``
    turbine.tower.L_reinforced=np.array([30., 30., 30.])#,30.,30.]) #[m] buckling length
    turbine.tower.downwind = False  # (Bool): flag if rotor is downwind
    # ---------------

    # --- wind ---
    turbine.tower.wind_z0 = 0.0  # (Float, m): bottom of wind profile (height of ground/sea)
    turbine.tower.wind1.shearExp = 0.2
    turbine.tower.wind2.shearExp = 0.2
    # ---------------

    # --- soil ---
    turbine.tower.soil.rigid = 6*[True]
    # ---------------

    # --- safety factors ---
    turbine.tower.gamma_f = 1.35  # (Float): safety factor on loads
    turbine.tower.gamma_m = 1.3  # (Float): safety factor on materials
    turbine.tower.gamma_n = 1.0  # (Float): safety factor on consequence of failure
    turbine.tower.gamma_b = 1.1  # (Float): buckling safety factor
    # ---------------

    # --- fatigue ---
    turbine.tower.z_DEL = 1.0/87.6*np.array([0.000, 1.327, 3.982, 6.636, 9.291, 11.945, 14.600, 17.255, 19.909,
        22.564, 25.218, 27.873, 30.527, 33.182, 35.836, 38.491, 41.145, 43.800, 46.455, 49.109, 51.764,
        54.418, 57.073, 59.727, 62.382, 65.036, 67.691, 70.345, 73.000, 75.655, 78.309, 80.964, 83.618,
        86.273, 87.600])  # (Array): locations along unit tower for M_DEL
    turbine.tower.M_DEL = 1e3*np.array([8.2940E+003, 8.1518E+003, 7.8831E+003, 7.6099E+003, 7.3359E+003,
        7.0577E+003, 6.7821E+003, 6.5119E+003, 6.2391E+003, 5.9707E+003, 5.7070E+003, 5.4500E+003,
        5.2015E+003, 4.9588E+003, 4.7202E+003, 4.4884E+003, 4.2577E+003, 4.0246E+003, 3.7942E+003,
        3.5664E+003, 3.3406E+003, 3.1184E+003, 2.8977E+003, 2.6811E+003, 2.4719E+003, 2.2663E+003,
        2.0673E+003, 1.8769E+003, 1.7017E+003, 1.5479E+003, 1.4207E+003, 1.3304E+003, 1.2780E+003,
        1.2673E+003, 1.2761E+003])  # (Array, N*m): damage equivalent moments along tower
    turbine.tower.gamma_fatigue = 1.35*1.3*1.0  # (Float): total safety factor for fatigue
    turbine.tower.life = 20.0  # (Float): fatigue life of tower
    turbine.tower.m_SN = 4  # (Int): slope of S/N curve
    turbine.tower.DC = 80.0  # (Float): standard value of stress
    # ---------------

    # --- constraints ---
    turbine.tower.min_d_to_t = 120.0  # (Float): minimum allowable diameter to thickness ratio
    turbine.tower.min_taper = 0.4  # (Float): minimum allowable taper ratio from tower top to tower bottom
    # ---------------

    # --- material properties
    turbine.tower.sigma_y = 450000000.0  # (Float, N/m**2): yield stress
    turbine.tower.rho = 8500.0  # (Float, kg/m**3): material density
    turbine.tower.E = 2.1e+11  # (Float, N/m**2): material modulus of elasticity
    turbine.tower.G = 80800000000.0  # (Float, N/m**2): material shear modulus
    # ----------------
    
    if turbine.sea_depth <> 0.0:
          configure_offshore(turbine.tower,sea_depth)

    if wind_class == 'I':
        turbine.rotor.turbine_class = 'I'

    elif wind_class == 'III':
        turbine.rotor.turbine_class = 'III'

        # for fatigue based analysis of class III wind turbine
        turbine.tower.M_DEL = 1.028713178 * 1e3*np.array([7.8792E+003, 7.7507E+003, 7.4918E+003, 7.2389E+003, 6.9815E+003, 6.7262E+003, 6.4730E+003, 6.2174E+003, 5.9615E+003, 5.7073E+003, 5.4591E+003, 5.2141E+003, 4.9741E+003, 4.7399E+003, 4.5117E+003, 4.2840E+003, 4.0606E+003, 3.8360E+003, 3.6118E+003, 3.3911E+003, 3.1723E+003, 2.9568E+003, 2.7391E+003, 2.5294E+003, 2.3229E+003, 2.1246E+003, 1.9321E+003, 1.7475E+003, 1.5790E+003, 1.4286E+003, 1.3101E+003, 1.2257E+003, 1.1787E+003, 1.1727E+003, 1.1821E+003])

        turbine.rotor.Mxb_damage = 1e3*np.array([2.3617E+003, 2.0751E+003, 1.8051E+003, 1.5631E+003, 1.2994E+003, 1.0388E+003, 8.1384E+002, 6.2492E+002, 4.6916E+002, 3.4078E+002, 2.3916E+002, 1.5916E+002, 9.9752E+001, 5.6139E+001, 2.6492E+001, 1.0886E+001, 3.7210E+000, 4.3206E-001])
        turbine.rotor.Myb_damage = 1e3*np.array([2.5492E+003, 2.6261E+003, 2.4265E+003, 2.2308E+003, 1.9882E+003, 1.7184E+003, 1.4438E+003, 1.1925E+003, 9.6251E+002, 7.5564E+002, 5.7332E+002, 4.1435E+002, 2.8036E+002, 1.7106E+002, 8.7732E+001, 3.8678E+001, 1.3942E+001, 1.6600E+000])

    elif wind_class == 'Offshore':
        turbine.rotor.turbine_class = 'I'
    
    # #------------------End of turbine copy

    # TODO: these should be specified at the turbine level and connected to other system inputs
    turbine.tower_d = [6.0, 4.935, 3.87]  # (Array, m): diameters along tower
    turbine.generator_speed = 1173.7  # (Float, rpm)  # generator speed
    # extra variable constant for now
    #lcoe_se.nacelle.bedplate.rotor_bending_moment_y = -2.3250E+06 # shouldnt be needed anymore

    # === run ===
    turbine.run()
    print 'mass rotor blades (kg) =', turbine.rotor.mass_all_blades
    print 'mass hub system (kg) =', turbine.hubSystem.hub_system_mass
    print 'mass nacelle (kg) =', turbine.nacelle.nacelle_mass
    print 'mass tower (kg) =', turbine.tower.mass
    print 'maximum tip deflection (m) =', turbine.maxdeflection.max_tip_deflection
    print 'ground clearance (m) =', turbine.maxdeflection.ground_clearance
    # print
    # print '"Torque":',turbine.nacelle.rotor_torque
    # print 'Mx:',turbine.nacelle.rotor_bending_moment_x
    # print 'My:',turbine.nacelle.rotor_bending_moment_y
    # print 'Mz:',turbine.nacelle.rotor_bending_moment_z
    # =================