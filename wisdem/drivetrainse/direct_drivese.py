
import numpy as np
import openmdao.api as om
from hub import Hub_System
from layout import Layout
from generator import Generator
from gearbox import Gearbox
import drive_structure as ds
import drive_components as dc

        
class DrivetrainSE(om.Group):
    ''' Class Drive4pt defines an OpenMDAO group that represents a wind turbine drivetrain with a 4-point suspension
      (two main bearings). This Group can serve as the root of an OpenMDAO Problem.
    '''
    def initialize(self):
        self.options.declare('n_points')
        self.options.declare('n_dlcs')
        self.options.declare('model_generator')
        self.options.declare('topLevelFlag', default=True)
    
    def setup(self):
        n_points = self.options['n_points']
        n_dlcs   = self.options['n_dlcs']

        # Independent variables that are unique to DriveSE
        ivc = om.IndepVarComp()
        ivc.add_output('gear_ratio', 1.0)
        ivc.add_output('L_bedplate', 0.0, units='m')
        ivc.add_output('H_bedplate', 0.0, units='m')
        ivc.add_output('L_12', 0.0, units='m')
        ivc.add_output('L_h1', 0.0, units='m')
        ivc.add_output('L_2n', 0.0, units='m')
        ivc.add_output('L_grs', 0.0, units='m')
        ivc.add_output('L_gsn', 0.0, units='m')
        ivc.add_output('access_diameter',0.0, units='m')

        ivc.add_output('lss_diameter', np.zeros(5), units='m')
        ivc.add_output('nose_diameter', np.zeros(5), units='m')
        ivc.add_output('lss_wall_thickness', np.zeros(5), units='m')
        ivc.add_output('nose_wall_thickness', np.zeros(5), units='m')
        ivc.add_output('bedplate_wall_thickness', np.zeros(n_points), units='m')
        #ivc.add_output('tilt', 0.0, units='deg')
        # ivc.add_output('shaft_ratio', 0.0)
        # ivc.add_output('shrink_disc_mass', 0.0, units='kg')
        # ivc.add_output('carrier_mass', 0.0, units='kg')
        # ivc.add_output('flange_length', 0.0, units='m')
        # ivc.add_output('hss_input_length', 0.0, units='m')
        # ivc.add_discrete_output('planet_numbers', np.array([0, 0, 0]))
        # ivc.add_discrete_output('drivetrain_design', 'geared')
        # ivc.add_discrete_output('gear_configuration', 'eep')
        ivc.add_discrete_output('mb1Type', 'CARB')
        ivc.add_discrete_output('mb2Type', 'SRB')
        #ivc.add_discrete_output('IEC_Class', 'B')
        #ivc.add_discrete_output('shaft_factor', 'normal')
        ivc.add_discrete_output('uptower', True)
        #ivc.add_discrete_output('crane', True)
        ivc.add_discrete_output('upwind', True)
        ivc.add_discrete_output('direct_drive', True)
        self.add_subsystem('ivc', ivc, promotes=['*'])

        # Independent variables that may be duplicated at higher levels of aggregation
        if self.options['topLevelFlag']:
            sivc = om.IndepVarComp()
            sivc.add_discrete_output('n_blades', 3)
            sivc.add_output('tilt', 0.0, units='deg')
            sivc.add_output('E', 0.0, units='Pa')
            sivc.add_output('G', 0.0, units='Pa')
            sivc.add_output('v', 0.0)
            sivc.add_output('sigma_y', 0.0, units='Pa')
            sivc.add_output('Xy', 0.0, units='Pa')
            sivc.add_output('rho', 0.0, units='kg/m**3')
            sivc.add_output('gamma_f', 0.0)
            sivc.add_output('gamma_m', 0.0)
            sivc.add_output('gamma_n', 0.0)
            sivc.add_output('D_top',     0.0, units='m')
            sivc.add_output('rotor_diameter',         0.0, units='m')
            sivc.add_output('rotor_rpm',              0.0, units='rpm')
            sivc.add_output('rotor_torque',           0.0, units='N*m')
            #sivc.add_output('Fxyz',                   np.zeros(3), units='N')
            #sivc.add_output('Mxyz',                   np.zeros(3), units='N*m')
            #sivc.add_output('blades_I',               np.zeros(6), units='kg*m**2')
            #sivc.add_output('blade_mass',             0.0, units='kg')
            #sivc.add_output('blade_root_diameter',    0.0, units='m')
            #sivc.add_output('blade_length',           0.0, units='m')
            #sivc.add_output('gearbox_efficiency',     0.0)
            #sivc.add_output('generator_efficiency',   0.0)
            #sivc.add_output('tile',                   0.0, units='deg')
            sivc.add_output('machine_rating',         0.0, units='kW')
            self.add_subsystem('sivc', sivc, promotes=['*'])

        # select components
        self.add_subsystem('hub', Hub_System(), promotes=['*'])
        self.add_subsystem('layout', Layout(n_points=n_points), promotes=['*'])
        self.add_subsystem('lss', ds.Hub_Rotor_Shaft_Frame(n_points=n_points, n_dlcs=n_dlcs), promotes=['*'])
        self.add_subsystem('bear1', dc.MainBearing())
        self.add_subsystem('bear2', dc.MainBearing())
        self.add_subsystem('gear', Gearbox(), promotes=['*']) 

        self.add_subsystem('hss', dc.HighSpeedSide(), promotes=['*']) # TODO- Include in generatorSE?
        if self.options['model_generator']:
            self.add_subsystem('generator', Generator(topLevelFlag=False, design='pmsg_outer'), promotes=['generator_mass','generator_I','E','G','v','machine_rating'])
        else:
            self.add_subsystem('gensimp', dc.GeneratorSimple(), promotes=['*'])
        self.add_subsystem('elec', dc.Electronics(), promotes=['*'])
        self.add_subsystem('yaw', dc.YawSystem(), promotes=['*'])
        self.add_subsystem('misc', dc.MiscNacelleComponents(), promotes=['*'])
        self.add_subsystem('nac', dc.NacelleSystemAdder(), promotes=['*'])
        self.add_subsystem('nose', ds.Nose_Stator_Bedplate_Frame(n_points=n_points, n_dlcs=n_dlcs), promotes=['*'])
        self.add_subsystem('rna', dc.RNA_Adder(), promotes=['*'])
        #self.add_subsystem('loads', RotorLoads(), promotes=['*']) Get this from Frame3DD reaction forces, although careful about mass/force inclusion

        self.connect('D_shaft','D_shaft_end', src_indices=[-1])
        self.connect('D_shaft','bear1.D_shaft', src_indices=[0])
        self.connect('D_shaft','bear2.D_shaft', src_indices=[-1])
        self.connect('D_bearing1','bear1.D_bearing')
        self.connect('D_bearing2','bear2.D_bearing')
        self.connect('mb1Type', 'bear1.bearing_type')
        self.connect('mb2Type', 'bear2.bearing_type')
        self.connect('bear1.mb_mass','mb1_mass')
        self.connect('bear1.mb_I','mb1_I')
        self.connect('s_mb1','mb1_cm')
        self.connect('bear2.mb_mass','mb2_mass')
        self.connect('bear2.mb_I','mb2_I')
        self.connect('s_mb2','mb2_cm')
        self.connect('D_bedplate','D_bedplate_base', src_indices=[0])
        if self.options['model_generator']:
            self.connect('lss_diameter','generator.D_shaft', src_indices=[0])
            self.connect('nose_diameter','generator.D_nose', src_indices=[-1])
            self.connect('generator.R_out','R_generator')
            self.connect('rotor_torque','generator.T_rated')
            self.connect('rotor_rpm','generator.n_nom')
        
if __name__ == '__main__':
    prob = om.Problem()
    prob.model = DrivetrainSE(topLevelFlag=True, n_points=10, n_dlcs=1, model_generator=True)
    prob.setup()

    prob['upwind'] = True
    prob['direct_drive'] = True
    prob['n_blades'] = 3
    prob['rotor_rpm'] = 10.0

    prob['L_12'] = 2.0
    prob['L_h1'] = 1.0
    prob['L_2n'] = 1.5
    prob['L_grs'] = 1.1
    prob['L_gsn'] = 1.1
    prob['L_bedplate'] = 5.0
    prob['H_bedplate'] = 4.875
    prob['tilt'] = 4.0
    prob['access_diameter'] = 0.9

    npts = 10
    myones = np.ones(5)
    prob['lss_diameter'] = 3.3*myones
    prob['nose_diameter'] = 2.2*myones
    prob['lss_wall_thickness'] = 0.45*myones
    prob['nose_wall_thickness'] = 0.1*myones
    prob['bedplate_wall_thickness'] = 0.06*np.ones(npts)
    prob['D_top'] = 6.5

    prob['m_stator'] = 100e3
    prob['cm_stator'] = -0.3
    prob['I_stator'] = np.array([1e6, 5e5, 5e5])

    prob['m_rotor'] = 100e3
    prob['cm_rotor'] = -0.3
    prob['I_rotor'] = np.array([1e6, 5e5, 5e5])

    #prob['F_mb1'] = np.array([2409.750e3, -1716.429e3, 74.3529e3]).reshape((3,1))
    #prob['F_mb2'] = np.array([2409.750e3, -1716.429e3, 74.3529e3]).reshape((3,1))
    #prob['M_mb1'] = np.array([-1.83291e7, 6171.7324e3, 5785.82946e3]).reshape((3,1))
    #prob['M_mb2'] = np.array([-1.83291e7, 6171.7324e3, 5785.82946e3]).reshape((3,1))

    prob['F_hub'] = np.array([2409.750e3, 0.0, 74.3529e2]).reshape((3,1))
    prob['M_hub'] = np.array([-1.83291e4, 6171.7324e2, 5785.82946e2]).reshape((3,1))

    prob['E'] = 210e9
    prob['G'] = 80.8e9
    prob['v'] = 0.3
    prob['rho'] = 7850.
    prob['sigma_y'] = 250e6
    prob['gamma_f'] = 1.35
    prob['gamma_m'] = 1.3
    prob['gamma_n'] = 1.0
    
    prob['pitch_system.blade_mass']       = 17000.
    prob['pitch_system.BRFM']             = 1.e+6
    prob['pitch_system.scaling_factor']   = 0.54
    prob['pitch_system.rho']              = 7850.
    prob['pitch_system.Xy']               = 371.e+6

    prob['hub_shell.blade_root_diameter'] = 4.
    prob['flange_t2shell_t']              = 4.
    prob['flange_OD2hub_D']               = 0.5
    prob['flange_ID2flange_OD']           = 0.8
    prob['hub_shell.rho']                 = 7200.
    prob['in2out_circ']                   = 1.2 
    prob['hub_shell.max_torque']          = 30.e+6
    prob['hub_shell.Xy']                  = 200.e+6
    prob['stress_concentration']          = 2.5
    prob['hub_shell.gamma']               = 2.0
    prob['hub_shell.metal_cost']          = 3.00

    prob['n_front_brackets']              = 3
    prob['n_rear_brackets']               = 3
    prob['spinner.blade_root_diameter']   = 4.
    prob['clearance_hub_spinner']         = 0.5
    prob['spin_hole_incr']                = 1.2
    prob['spinner.gust_ws']               = 70
    prob['spinner.gamma']                 = 1.5
    prob['spinner.composite_Xt']          = 60.e6
    prob['spinner.composite_SF']          = 1.5
    prob['spinner.composite_rho']         = 1600.
    prob['spinner.Xy']                    = 225.e+6
    prob['spinner.metal_SF']              = 1.5
    prob['spinner.metal_rho']             = 7850.
    prob['spinner.composite_cost']        = 7.00
    prob['spinner.metal_cost']            = 3.00
    
    prob.run_model()
    
