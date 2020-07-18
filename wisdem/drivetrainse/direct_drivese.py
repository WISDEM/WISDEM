
import numpy as np
import openmdao.api as om
from layout import Layout
#from generator import Generator
import drive_structure as ds
from rna import RNAMass
import drive_components as dc

        
class DrivetrainSE(om.Group):
    ''' Class Drive4pt defines an OpenMDAO group that represents a wind turbine drivetrain with a 4-point suspension
      (two main bearings). This Group can serve as the root of an OpenMDAO Problem.
    '''
    def initialize(self):
        self.options.declare('n_points')
        self.options.declare('n_dlcs')
        self.options.declare('topLevelFlag', default=True)
    
    def setup(self):
        n_points = self.options['n_points']
        n_dlcs   = self.options['n_dlcs']

        # Independent variables that are unique to DriveSE
        ivc = om.IndepVarComp()
        ivc.add_output('gear_ratio', 1.0)
        ivc.add_output('L_bedplate', 0.0, units='m')
        ivc.add_output('H_bedplate', 0.0, units='m')
        #ivc.add_output('tilt', 0.0, units='deg')
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
        #ivc.add_discrete_output('IEC_Class', 'B')
        #ivc.add_discrete_output('shaft_factor', 'normal')
        ivc.add_discrete_output('uptower_electronics', True)
        #ivc.add_discrete_output('crane', True)
        #ivc.add_discrete_output('rna_weightM', True)
        ivc.add_discrete_output('upwind', True)
        ivc.add_discrete_output('direct_drive', True)
        self.add_subsystem('ivc', ivc, promotes=['*'])

        # Independent variables that may be duplicated at higher levels of aggregation
        if self.options['topLevelFlag']:
            sivc = om.IndepVarComp()
            sivc.add_discrete_output('number_of_blades', 0)
            sivc.add_output('tilt', 0.0, units='deg')
            sivc.add_output('E', 0.0, units='Pa')
            sivc.add_output('G', 0.0, units='Pa')
            sivc.add_output('sigma_y', 0.0, units='Pa')
            sivc.add_output('rho', 0.0, units='kg/m**3')
            sivc.add_output('gamma_f', 0.0)
            sivc.add_output('gamma_m', 0.0)
            sivc.add_output('gamma_n', 0.0)
            sivc.add_output('D_top',     0.0, units='m')
            #sivc.add_output('rotor_diameter',         0.0, units='m')
            #sivc.add_output('rotor_rpm',              0.0, units='rpm')
            #sivc.add_output('rotor_torque',           0.0, units='N*m')
            #sivc.add_output('Fxyz',                   np.zeros(3), units='N')
            #sivc.add_output('Mxyz',                   np.zeros(3), units='N*m')
            #sivc.add_output('blade_mass',             0.0, units='kg')
            #sivc.add_output('blade_root_diameter',    0.0, units='m')
            #sivc.add_output('blade_length',           0.0, units='m')
            #sivc.add_output('blades_I',               np.zeros(6), units='kg*m**2')
            #sivc.add_output('gearbox_efficiency',     0.0)
            #sivc.add_output('generator_efficiency',   0.0)
            #sivc.add_output('tile',                   0.0, units='deg')
            #sivc.add_output('machine_rating',         0.0, units='kW')
            self.add_subsystem('sivc', sivc, promotes=['*'])

        # select components
        #self.add_subsystem('hub', HubSE(mass_only=True, topLevelFlag=False, debug=debug), promotes=['*'])
        self.add_subsystem('layout', Layout(n_points=n_points), promotes=['*'])
        #self.add_subsystem('generator', Generator(), promotes=['*'])
        self.add_subsystem('lss', ds.Hub_Rotor_Shaft_Frame(n_points=n_points, n_dlcs=n_dlcs), promotes=['*'])
        self.add_subsystem('bear1', dc.MainBearing())
        self.add_subsystem('bear2', dc.MainBearing())
        self.add_subsystem('gear', dc.Gearbox(), promotes=['*']) 

        self.add_subsystem('hss', dc.HighSpeedSide(), promotes=['*']) # TODO- Include in generatorSE?
        self.add_subsystem('elec', dc.Electronics(), promotes=['*'])
        self.add_subsystem('yaw', dc.YawSystem(), promotes=['*'])
        self.add_subsystem('misc', dc.MiscNacelleComponents(), promotes=['*'])
        #self.add_subsystem('nac', dc.NacelleSystemAdder(), promotes=['*'])
        #self.add_subsystem('rna', RNAMass(), promotes=['*'])
        self.add_subsystem('nose', ds.Nose_Stator_Bedplate_Frame(n_points=n_points, n_dlcs=n_dlcs), promotes=['*'])
        #self.add_subsystem('loads', RotorLoads(), promotes=['*']) Get this from Frame3DD reaction forces, although careful about mass/force inclusion

        self.connect('D_shaft','D_shaft_end', src_indices=[-1])
        self.connect('D_shaft','bear1.D_shaft', src_indices=[0])
        self.connect('D_shaft','bear2.D_shaft', src_indices=[-1])
        self.connect('D_bearing1','bear1.D_bearing')
        self.connect('D_bearing2','bear2.D_bearing')
        self.connect('mb1Type', 'bear1.bearing_type')
        self.connect('mb2Type', 'bear2.bearing_type')
        self.connect('D_bedplate','D_bedplate_base', src_indices=[0])
        
if __name__ == '__main__':
    prob = om.Problem()
    prob.model = DrivetrainSE(topLevelFlag=True, n_points=10, n_dlcs=1)
    prob.setup()

    prob['upwind'] = True
    prob['direct_drive'] = True

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
    prob['shaft_diameter'] = 3.3*myones
    prob['nose_diameter'] = 2.2*myones
    prob['shaft_wall_thickness'] = 0.45*myones
    prob['nose_wall_thickness'] = 0.1*myones
    prob['bedplate_wall_thickness'] = 0.06*np.ones(npts)
    prob['D_top'] = 6.5

    prob['m_other'] = 200e3

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

    prob['m_hub'] = 100e3
    prob['cm_hub'] = 2.0
    prob['I_hub'] = np.array([2409.750e3, -1716.429e3, 74.3529e3])
    prob['F_hub'] = np.array([2409.750e3, 0.0, 74.3529e2]).reshape((3,1))
    prob['M_hub'] = np.array([-1.83291e4, 6171.7324e2, 5785.82946e2]).reshape((3,1))

    prob['E'] = 210e9
    prob['G'] = 80.8e9
    prob['rho'] = 7850.
    prob['sigma_y'] = 250e6
    prob['gamma_f'] = 1.35
    prob['gamma_m'] = 1.3
    prob['gamma_n'] = 1.0
    
    prob.run_model()
    
