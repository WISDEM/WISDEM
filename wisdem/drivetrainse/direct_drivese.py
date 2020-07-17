
import numpy as np
import openmdao.api as om
from layout import Geometry
from generator import Generator
import drive_structure as ds
from rna import RNAMass
import drive_components as dc

#--------------------------------------------

        
class DriveSE(om.Group):
    ''' Class Drive4pt defines an OpenMDAO group that represents a wind turbine drivetrain with a 4-point suspension
      (two main bearings). This Group can serve as the root of an OpenMDAO Problem.
    '''

    def initialize(self):
        self.options.declare('topLevelFlag', default=True)
        
    def setup(self):

        # Independent variables that are unique to DriveSE
        ivc = om.IndepVarComp()
        # ivc.add_output('gear_ratio', 0.0)
        # ivc.add_output('shaft_angle', 0.0, units='rad')
        # ivc.add_output('shaft_ratio', 0.0)
        # ivc.add_output('shrink_disc_mass', 0.0, units='kg')
        # ivc.add_output('carrier_mass', 0.0, units='kg')
        # ivc.add_output('flange_length', 0.0, units='m')
        # ivc.add_output('overhang', 0.0, units='m')
        # ivc.add_output('distance_hub2mb', 0.0, units='m')
        # ivc.add_output('gearbox_input_xcm', 0.0, units='m')
        # ivc.add_output('hss_input_length', 0.0, units='m')
        # ivc.add_discrete_output('planet_numbers', np.array([0, 0, 0]))
        # ivc.add_discrete_output('drivetrain_design', 'geared')
        # ivc.add_discrete_output('gear_configuration', 'eep')
        ivc.add_discrete_output('mb1Type', 'CARB')
        ivc.add_discrete_output('mb2Type', 'SRB')
        ivc.add_discrete_output('IEC_Class', 'B')
        ivc.add_discrete_output('shaft_factor', 'normal')
        ivc.add_discrete_output('uptower_transformer', True)
        ivc.add_discrete_output('crane', True)
        ivc.add_discrete_output('rna_weightM', True)
        ivc.add_discrete_output('downwind', True)
        self.add_subsystem('ivc', ivc, promotes=['*'])

        # Independent variables that may be duplicated at higher levels of aggregation
        if self.options['topLevelFlag']:
            sivc = IndepVarComp()
            sivc.add_discrete_output('number_of_blades', 0)
            sivc.add_output('tower_top_diameter',     0.0, units='m')
            sivc.add_output('rotor_diameter',         0.0, units='m')
            sivc.add_output('rotor_rpm',              0.0, units='rpm')
            sivc.add_output('rotor_torque',           0.0, units='N*m')
            sivc.add_output('Fxyz',                   np.zeros(3), units='N')
            sivc.add_output('Mxyz',                   np.zeros(3), units='N*m')
            sivc.add_output('blade_mass',             0.0, units='kg')
            sivc.add_output('blade_root_diameter',    0.0, units='m')
            sivc.add_output('blade_length',           0.0, units='m')
            sivc.add_output('blades_I',               np.zeros(6), units='kg*m**2')
            sivc.add_output('gearbox_efficiency',     0.0)
            sivc.add_output('generator_efficiency',   0.0)
            sivc.add_output('tile',                   0.0, units='deg')
            sivc.add_output('machine_rating',         0.0, units='kW')
            self.add_subsystem('sivc', sivc, promotes=['*'])

        # select components
        #self.add_subsystem('hub', HubSE(mass_only=True, topLevelFlag=False, debug=debug), promotes=['*'])
        self.add_subsystem('layout', Geometry(n_points=n_points), promotes=['*'])
        #self.add_subsystem('generator', Generator(), promotes=['*'])
        self.add_subsystem('lss', ds.Hub_Rotor_Shaft_Frame(), promotes=['*'])
        self.add_subsystem('mainBearing', dc.MainBearing(bearing_position='main'), promotes=['lss_design_torque','rotor_diameter'])
        self.add_subsystem('secondBearing', dc.MainBearing(bearing_position='second'), promotes=['lss_design_torque','rotor_diameter'])

        self.add_subsystem('highSpeedSide', dc.HighSpeedSide(), promotes=['*']) # TODO- Include in generatorSE
        self.add_subsystem('electronics', dc.Electronics(), promotes=['*'])
        self.add_subsystem('yaw', dc.YawSystem(), promotes=['*'])
        self.add_subsystem('misc', dc.MiscNacelleComponents(), promotes=['*'])
        #self.add_subsystem('nacelleSystem', dc.NacelleSystemAdder(), promotes=['*'])
        self.add_subsystem('rna', RNAMass(), promotes=['*'])
        self.add_subsystem('nose', ds.Nose_Stator_Bedplate_Frame(), promotes=['*'])
        #self.add_subsystem('loads', RotorLoads(), promotes=['*']) Get this from Frame3DD reaction forces, although careful about mass/force inclusion
            
