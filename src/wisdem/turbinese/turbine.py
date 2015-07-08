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
from drivese.drive import Drive4pt, Drive3pt
from drivese.drivese_utils import blade_moment_transform, blade_force_transform
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
        assembly.add('forces',blade_force_transform())
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
        assembly.driver.workflow.add(['hubSystem','moments','forces'])

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
    assembly.connect('rotor.diameter', ['hub.rotor_diameter'])
    assembly.connect('rotor.hub_diameter', 'hub.blade_root_diameter')
    assembly.connect('rotor.nBlades', 'hub.blade_number')
    if with_new_nacelle:
        assembly.connect('rotor.diameter', ['hubSystem.rotor_diameter'])
        assembly.connect('nacelle.MB1_location','hubSystem.MB1_location') # TODO: bearing locations
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

        #moments
        # assembly.connect('rotor.Q_extreme','nacelle.rotor_bending_moment_x')
        assembly.connect('rotor.Mxyz_0','moments.b1')
        assembly.connect('rotor.Mxyz_120','moments.b2')
        assembly.connect('rotor.Mxyz_240','moments.b3')
        assembly.connect('rotor.Pitch','moments.pitch_angle')
        assembly.connect('rotor.TotalCone','moments.cone_angle')
        assembly.connect('moments.Mx','nacelle.rotor_bending_moment_x') #accounted for in ratedConditions.Q
        assembly.connect('moments.My','nacelle.rotor_bending_moment_y')
        assembly.connect('moments.Mz','nacelle.rotor_bending_moment_z')

        #forces
        # assembly.connect('rotor.T_extreme','nacelle.rotor_force_x')
        assembly.connect('rotor.Fxyz_0','forces.b1')
        assembly.connect('rotor.Fxyz_120','forces.b2')
        assembly.connect('rotor.Fxyz_240','forces.b3')
        assembly.connect('rotor.Pitch','forces.pitch_angle')
        assembly.connect('rotor.TotalCone','forces.cone_angle')
        assembly.connect('forces.Fx','nacelle.rotor_force_x')
        assembly.connect('forces.Fy','nacelle.rotor_force_y')
        assembly.connect('forces.Fz','nacelle.rotor_force_z')

    '''if  with_new_nacelle:
        assembly.connect('rotor.g', 'nacelle.g') # Only drive smooth taking g from rotor; TODO: update when drive_smooth is updated'''


    # connections to rna
    assembly.connect('rotor.mass_all_blades', 'rna.blades_mass')
    assembly.connect('rotor.I_all_blades', 'rna.blades_I')
    if with_new_nacelle:
        assembly.connect('hubSystem.hub_system_mass', 'rna.hub_mass')
        assembly.connect('hubSystem.hub_system_cm', 'rna.hub_cm')
        assembly.connect('hubSystem.hub_system_I', 'rna.hub_I')
    else:
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
    if with_new_nacelle:
        assembly.connect('hubSystem.hub_system_cm', 'rotorloads1.r_hub')
    else:
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
    if with_new_nacelle:
        assembly.connect('hubSystem.hub_system_cm', 'rotorloads2.r_hub')
    else:
        assembly.connect('hub.hub_system_cm', 'rotorloads2.r_hub')
    assembly.connect('rna.rna_cm', 'rotorloads2.rna_cm')
    assembly.connect('rotor.tilt', 'rotorloads2.tilt')
    assembly.connect('g', 'rotorloads2.g')
    assembly.connect('rna.rna_mass', 'rotorloads2.m_RNA')

    # connections to tower
    assembly.connect('rho', 'tower.wind_rho')
    assembly.connect('mu', 'tower.wind_mu')
    assembly.connect('g', 'tower.g')
    assembly.connect('hub_height', 'tower.wind_zref')
    assembly.connect('tower_d', 'tower.d_param')
    assembly.connect('rotor.ratedConditions.V', 'tower.wind_Uref1')
    assembly.connect('rotor.V_extreme', 'tower.wind_Uref2')
    assembly.connect('rotor.yaw', 'tower.yaw')
    assembly.connect('hub_height - nacelle.nacelle_cm[2]', 'tower.z_full[-1]')
    assembly.connect('rna.rna_mass', 'tower.m[0]')
    assembly.connect('rna.rna_cm[0]', 'tower.mrhox[0]')
    assembly.connect('rna.rna_cm[1]', 'tower.mrhoy[0]')
    assembly.connect('rna.rna_cm[2]', 'tower.mrhoz[0]')
    assembly.connect('rna.rna_I_TT[0]', 'tower.mIxx[0]')
    assembly.connect('rna.rna_I_TT[1]', 'tower.mIyy[0]')
    assembly.connect('rna.rna_I_TT[2]', 'tower.mIzz[0]')
    assembly.connect('rna.rna_I_TT[3]', 'tower.mIxy[0]')
    assembly.connect('rna.rna_I_TT[4]', 'tower.mIxz[0]')
    assembly.connect('rna.rna_I_TT[5]', 'tower.mIyz[0]')
    assembly.connect('rotorloads1.top_F[0]', 'tower.Fx1[0]')
    assembly.connect('rotorloads1.top_F[1]', 'tower.Fy1[0]')
    assembly.connect('rotorloads1.top_F[2]', 'tower.Fz1[0]')
    assembly.connect('rotorloads1.top_M[0]', 'tower.Mxx1[0]')
    assembly.connect('rotorloads1.top_M[1]', 'tower.Myy1[0]')
    assembly.connect('rotorloads1.top_M[2]', 'tower.Mzz1[0]')
    assembly.connect('rotorloads2.top_F[0]', 'tower.Fx2[0]')
    assembly.connect('rotorloads2.top_F[1]', 'tower.Fy2[0]')
    assembly.connect('rotorloads2.top_F[2]', 'tower.Fz2[0]')
    assembly.connect('rotorloads2.top_M[0]', 'tower.Mxx2[0]')
    assembly.connect('rotorloads2.top_M[1]', 'tower.Myy2[0]')
    assembly.connect('rotorloads2.top_M[2]', 'tower.Mzz2[0]')

    # connections to maxdeflection
    assembly.connect('rotor.Rtip', 'maxdeflection.Rtip')
    assembly.connect('rotor.precurveTip', 'maxdeflection.precurveTip')
    assembly.connect('rotor.presweepTip', 'maxdeflection.presweepTip')
    assembly.connect('rotor.precone', 'maxdeflection.precone')
    assembly.connect('rotor.tilt', 'maxdeflection.tilt')
    if with_new_nacelle:
        assembly.connect('hubSystem.hub_system_cm', 'maxdeflection.hub_tt')
    else:
        assembly.connect('hub.hub_system_cm', 'maxdeflection.hub_tt')
    assembly.connect('tower.z_param', 'maxdeflection.tower_z')
    assembly.connect('tower.d_param', 'maxdeflection.tower_d')
    assembly.connect('hub_height - nacelle.nacelle_cm[2]', 'maxdeflection.towerHt')



class TurbineSE(Assembly):

    def configure(self):
        configure_turbine(self)


if __name__ == '__main__':

    turbine = TurbineSE()
    turbine.sea_depth = 0.0 # 0.0 for land-based turbine
    wind_class = 'I'

    from wisdem.reference_turbines.nrel5mw.nrel5mw import configure_nrel5mw_turbine
    configure_nrel5mw_turbine(turbine,wind_class,turbine.sea_depth)

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
    # print 'Fx:',turbine.nacelle.rotor_force_x
    # print 'Fy:',turbine.nacelle.rotor_force_y
    # print 'Fz:',turbine.nacelle.rotor_force_z
    # =================