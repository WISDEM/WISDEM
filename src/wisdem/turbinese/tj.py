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
#from jacketse.jacket import TowerSE
from commonse.rna import RNAMass, RotorLoads
from jacketse.jacket import JacketSE
from jacketse.jacket import JcktGeoInputs,SoilGeoInputs,WaterInputs,WindInputs,RNAprops,TPlumpMass,Frame3DDaux,\
                    MatInputs,LegGeoInputs,XBrcGeoInputs,MudBrcGeoInputs,HBrcGeoInputs,TPGeoInputs,PileGeoInputs,\
                    TwrGeoInputs, LegGeoOutputs, TwrGeoOutputs
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
    hub_tt = Array(iotype='in', units='m', desc='location of hub relative to jacket-top in yaw-aligned c.s.')
    jacket_z = Array(iotype='in', units='m')
    jacket_d = Array(iotype='in', units='m')
    jacketHt = Float(iotype='in', units='m')

    max_tip_deflection = Float(iotype='out', units='m', desc='clearance between undeflected blade and jacket')
    ground_clearance = Float(iotype='out', units='m', desc='distance between blade tip and ground')

    def execute(self):

        # coordinates of blade tip in yaw c.s.
        blade_yaw = DirectionVector(self.precurveTip, self.presweepTip, self.Rtip)\
            .bladeToAzimuth(self.precone).azimuthToHub(180.0).hubToYaw(self.tilt)

        # find corresponding radius of jacket
        zjacket = (self.jacketHt + self.hub_tt[2] + blade_yaw.z)/self.jacketHt  # nondimensional location
        # rjacket = np.interp(zjacket, self.jacket_z, self.jacket_d) / 2.0
        djacket, ddjacket_dzjacket, ddjacket_djacketz, ddjacket_djacketd = interp_with_deriv(zjacket, self.jacket_z, self.jacket_d)
        rjacket = djacket / 2.0
        self.drjacket_dzjacket = ddjacket_dzjacket / 2.0
        self.drjacket_djacketz = ddjacket_djacketz / 2.0
        self.drjacket_djacketd = ddjacket_djacketd / 2.0

        # max deflection before strike
        if self.precone >= 0:  # upwind
            self.max_tip_deflection = -self.hub_tt[0] - blade_yaw.x - rjacket
        else:
            self.max_tip_deflection = -self.hub_tt[0] + blade_yaw.x - rjacket
            # TODO: need to redo gradients for this case.

        # ground clearance
        self.ground_clearance = self.jacketHt + self.hub_tt[2] + blade_yaw.z

        # save for derivs
        self.blade_yaw = blade_yaw


    def list_deriv_vars(self):

        inputs = ('Rtip', 'precurveTip', 'presweepTip', 'precone', 'tilt', 'hub_tt',
            'jacket_z', 'jacket_d', 'jacketHt')
        outputs = ('max_tip_deflection', 'ground_clearance')

        return inputs, outputs

    def provideJ(self):

        dbyx = self.blade_yaw.dx
        # dbyy = self.blade_yaw.dy
        dbyz = self.blade_yaw.dz

        # Rtip
        drjacket_dRtip = self.drjacket_dzjacket * dbyz['dz']/self.jacketHt
        if self.precone >= 0:
            dtd_dRtip = -dbyx['dz'] - drjacket_dRtip
        else:
            dtd_dRtip = dbyx['dz'] - drjacket_dRtip
        dgc_dRtip = dbyz['dz']

        # precurveTip
        drjacket_dprecurveTip = self.drjacket_dzjacket * dbyz['dx']/self.jacketHt
        if self.precone >= 0:
            dtd_dprecurveTip = -dbyx['dx'] - drjacket_dprecurveTip
        else:
            dtd_dprecurveTip = dbyx['dx'] - drjacket_dprecurveTip
        dgc_dprecurveTip = dbyz['dx']

        # presweep
        drjacket_dpresweepTip = self.drjacket_dzjacket * dbyz['dy']/self.jacketHt
        if self.precone >= 0:
            dtd_dpresweepTip = -dbyx['dy'] - drjacket_dpresweepTip
        else:
            dtd_dpresweepTip = dbyx['dy'] - drjacket_dpresweepTip
        dgc_dpresweepTip = dbyz['dy']


        # precone
        drjacket_dprecone = self.drjacket_dzjacket * dbyz['dprecone']/self.jacketHt
        if self.precone >= 0:
            dtd_dprecone = -dbyx['dprecone'] - drjacket_dprecone
        else:
            dtd_dprecone = dbyx['dprecone'] - drjacket_dprecone
        dgc_dprecone = dbyz['dprecone']

        # tilt
        drjacket_dtilt = self.drjacket_dzjacket * dbyz['dtilt']/self.jacketHt
        if self.precone >= 0:
            dtd_dtilt = -dbyx['dtilt'] - drjacket_dtilt
        else:
            dtd_dtilt = dbyx['dtilt'] - drjacket_dtilt
        dgc_dtilt = dbyz['dtilt']

        # hubtt
        drjacket_dhubtt = self.drjacket_dzjacket * np.array([0.0, 0.0, 1.0/self.jacketHt])
        dtd_dhubtt = np.array([-1.0, 0.0, 0.0]) - drjacket_dhubtt
        dgc_dhubtt = np.array([0.0, 0.0, 1.0])

        # jacket_z
        dtd_djacketz = -self.drjacket_djacketz
        dgc_djacketz = np.zeros_like(self.jacket_z)

        # jacket_d
        dtd_djacketd = -self.drjacket_djacketd
        dgc_djacketd = np.zeros_like(self.jacket_d)

        # jacketHt
        drjacket_djacketHt = self.drjacket_dzjacket * -(self.hub_tt[2] + self.blade_yaw.z)/self.jacketHt**2
        dtd_djacketHt = -drjacket_djacketHt
        dgc_djacketHt = 1.0

        dtd = hstack([dtd_dRtip, dtd_dprecurveTip, dtd_dpresweepTip, dtd_dprecone, dtd_dtilt,
            dtd_dhubtt, dtd_djacketz, dtd_djacketd, dtd_djacketHt])
        dgc = np.concatenate([[dgc_dRtip], [dgc_dprecurveTip], [dgc_dpresweepTip], [dgc_dprecone],
            [dgc_dtilt], dgc_dhubtt, dgc_djacketz, dgc_djacketd, [dgc_djacketHt]])

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
    assembly.add('tower_dt', Float(iotype='in', units='m', desc='tower top diameter')) # update for jacket
    assembly.add('jacket_d', Array([0.0], iotype='in', units='m', desc='diameters along jacket'))
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
        assembly.connect('machine_rating','hub.machine_rating')
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
    assembly.connect('jacket_d[-1]', 'nacelle.tower_top_diameter')  # OpenMDAO circular dependency issue
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
    #assembly.connect('jacket.z_param', 'maxdeflection.jacket_z')
    #assembly.connect('jacket.d_param', 'maxdeflection.jacket_d')

    # assembly.connect('hub_height - nacelle.nacelle_cm[2]', 'maxdeflection.towerHt')



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
    print 'mass jacket (kg) =', turbine.jacket.mass
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
