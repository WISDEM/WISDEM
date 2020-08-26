'''
Components for the new spherical hub model - July 2019
  Sph_Hub, PitchSystem, Sph_Spinner, Hub_System_Adder, Hub_Mass_Adder, Hub_CM_Adder are now imported 
  from sph_hubse_components.py (instead of hubse_components.py)

'''

import numpy as np

from openmdao.api import ExplicitComponent, Group, Problem, IndepVarComp
from wisdem.drivetrainse.hubse_components import Hub, PitchSystem, Spinner, Hub_System_Adder, Hub_Mass_Adder, Hub_CM_Adder

#-------------------------------------------------------------------------

class Hub_System_Adder_OM(ExplicitComponent):
    ''' 
    Compute hub mass, cm, and I

    NOTE: THIS COMPONENT DOES NOT PLAY WELL WITH DRIVESE BECAUSE IT CREATES CIRCULAR REFERENCES!
    (requires main bearing location, but outputs from here are required to determine MB1_Location in drivese_omdao.py)
    '''
    def initialize(self):
        self.options.declare('debug', default=False)
        
    def setup(self):
        # variables
        self.add_input('rotor_diameter',    val=0.0,                     units='m',   desc='rotor diameter')
        self.add_input('distance_hub2mb',   val=0.0,                     units='m',   desc='distance between hub center and upwind main bearing')
        self.add_input('shaft_angle',       val=0.0,                     units='rad', desc='shaft angle')
        self.add_input('MB1_location',      val=np.zeros(3), shape=(3,), units='m',   desc='center of mass of main bearing in [x,y,z] for an arbitrary coordinate system')
        self.add_input('hub_mass',          val=0.0,                     units='kg',  desc='mass of Hub')
        self.add_input('hub_diameter',      val=0.03,                    units='m',   desc='hub diameter')
        self.add_input('hub_thickness',     val=0.0,                     units='m',   desc='hub thickness')
        self.add_input('pitch_system_mass', val=0.0,                     units='kg',  desc='mass of Pitch System')
        self.add_input('spinner_mass',      val=0.0,                     units='kg',  desc='mass of spinner')
        self.add_input('blade_mass',        val=0.0,                     units='kg',  desc='mass of one blade')
        self.add_discrete_input('number_of_blades', val=0, desc='number of blades on the rotor')
        
        # outputs
        self.add_output('hub_system_cm',   val=np.zeros(3),             units='m',  desc='center of mass of the hub relative to tower to in yaw-aligned c.s.')
        self.add_output('hub_system_I',    val=np.zeros(6),             units='kg*m**2', desc='mass moments of Inertia of hub [Ixx, Iyy, Izz, Ixy, Ixz, Iyz] around its center of mass in yaw-aligned c.s.')
        self.add_output('hub_I',           val=np.zeros(3),             units='kg*m**2', desc='Hub inertia about rotor axis (does not include pitch and spinner masses)')
        self.add_output('hub_system_mass', val=0.0,                     units='kg', desc='mass of hub system')
        self.add_output('rotor_mass',      val=0.0,                     units='kg', desc='mass of rotor')

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
    
        hub_adder = Hub_System_Adder(discrete_inputs['number_of_blades'], debug=self.options['debug'])

        (outputs['rotor_mass'], outputs['hub_system_mass'], outputs['hub_system_cm'], outputs['hub_system_I'], outputs['hub_I']) \
            = hub_adder.compute(inputs['rotor_diameter'], inputs['blade_mass'], inputs['distance_hub2mb'], inputs['shaft_angle'], inputs['MB1_location'], \
                                inputs['hub_mass'], inputs['hub_diameter'], inputs['hub_thickness'], inputs['pitch_system_mass'], inputs['spinner_mass'])        


#-------------------------------------------------------------------------

class Hub_Mass_Adder_OM(ExplicitComponent):
    ''' 
    Compute hub mass and I
    Excluding cm here, because it has a dependency on main bearing location, which can only be calculated once the full rotor mass is set
    '''
    def initialize(self):
        self.options.declare('debug', default=False)
        
    def setup(self):
        # variables
        self.add_input('hub_mass',          val=0.0,  units='kg', desc='mass of Hub')
        self.add_input('hub_diameter',      val=0.03, units='m',  desc='hub diameter')
        self.add_input('hub_thickness',     val=0.0,  units='m',  desc='hub thickness')
        self.add_input('pitch_system_mass', val=0.0,  units='kg', desc='mass of Pitch System')
        self.add_input('spinner_mass',      val=0.0,  units='kg', desc='mass of spinner')
        self.add_input('blade_mass',        val=0.0,  units='kg', desc='mass of one blade')
        self.add_discrete_input('number_of_blades', val=0, desc='number of blades on the rotor')

        # outputs
        self.add_output('hub_system_I',    val=np.zeros(6),             units='kg*m**2', desc='mass moments of Inertia of hub [Ixx, Iyy, Izz, Ixy, Ixz, Iyz] around its center of mass in yaw-aligned c.s.')
        self.add_output('hub_system_mass', val=0.0,                     units='kg', desc='mass of hub system')
        self.add_output('rotor_mass',      val=0.0,                     units='kg', desc='mass of rotor')
        self.add_output('hub_I',           val=np.zeros(3),             units='kg*m**2', desc='Hub inertia about rotor axis (does not include pitch and spinner masses)')


    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        hub_adder = Hub_Mass_Adder(discrete_inputs['number_of_blades'], debug=self.options['debug'])
        
        (outputs['rotor_mass'], outputs['hub_system_mass'], outputs['hub_system_I'], outputs['hub_I'])\
            = hub_adder.compute(inputs['blade_mass'], inputs['hub_mass'], inputs['hub_diameter'], 
                                inputs['hub_thickness'], inputs['pitch_system_mass'], inputs['spinner_mass'])        
        

#-------------------------------------------------------------------------

class Hub_CM_Adder_OM(ExplicitComponent):
    ''' 
    Compute hub cm
    Separating cm here, because it has a dependency on main bearing location, which can only be calculated once the full rotor mass is set
    '''

    def setup(self):
        # variables
        self.add_input('rotor_diameter',  val=0.0,         units='m',   desc='rotor diameter')
        self.add_input('distance_hub2mb', val=0.0,         units='m',   desc='distance between hub center and upwind main bearing')
        self.add_input('shaft_angle',     val=0.0,         units='rad', desc='shaft angle')
        self.add_input('MB1_location',    val=np.zeros(3), units='m',   desc='center of mass of main bearing in [x,y,z] for an arbitrary coordinate system')

        # outputs
        self.add_output('hub_system_cm', val=np.zeros(3), units='m',  desc='center of mass of the hub relative to tower to in yaw-aligned c.s.')

    def compute(self, inputs, outputs):
    
        hub_adder = Hub_CM_Adder()

        outputs['hub_system_cm'] = hub_adder.compute(inputs['rotor_diameter'], inputs['distance_hub2mb'],
                                                     inputs['shaft_angle'], inputs['MB1_location'])
        
    
# -------------------------------------------------

class Hub_OM(ExplicitComponent):
    ''' Hub class    
          The Hub class is used to represent the hub component of a wind turbine. 
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.            
    '''
    def initialize(self):
        self.options.declare('debug', default=False)
        
    def setup(self):
        # variables
        self.add_input('blade_root_diameter', val=0.0, units='m',   desc='blade root diameter')
        self.add_input('machine_rating',      val=0.0, units='kW',  desc='machine rating of turbine')
        self.add_input('rotor_rpm',           val=9.0, units='rpm', desc='RPM at rated power')
        self.add_input('blade_mass',          val=0.0, units='kg',  desc='mass of one blade')
        self.add_input('rotor_diameter',      val=0.0, units='m',   desc='diameter of rotor')
        self.add_input('blade_length',        val=0.0, units='m',   desc='length of blade')
        self.add_discrete_input('number_of_blades', val=0, desc='number of blades on the rotor')

        # outputs
        self.add_output('hub_diameter',  val=0.0, units='m',  desc='hub diameter')
        self.add_output('hub_thickness', val=0.0, units='m',  desc='hub thickness')
        self.add_output('hub_mass',      val=0.0, units='kg', desc='overall component mass')
        self.add_output('hub_cm',        val=0.0, units='m',  desc='hub center of mass')
        self.add_output('hub_cost',      val=0.0, units='USD',  desc='hub cost')

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
    
        myhub = Hub(discrete_inputs['number_of_blades'], debug=self.options['debug'])

        (outputs['hub_mass'], outputs['hub_diameter'], outputs['hub_cm'], outputs['hub_cost'], outputs['hub_thickness']) \
            = myhub.compute(inputs['blade_root_diameter'], inputs['rotor_rpm'],
                            inputs['blade_mass'], inputs['rotor_diameter'], inputs['blade_length']) 


class PitchSystem_OM(ExplicitComponent):
    '''
     PitchSystem class
      The PitchSystem class is used to represent the pitch system of a wind turbine.
      It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
      It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''
    def initialize(self):
        self.options.declare('debug', default=False)
        
    def setup(self):
        # variables
        self.add_input('blade_mass',             val=0.0, units='kg',  desc='mass of one blade')
        self.add_input('rotor_bending_moment_y', val=0.0, units='N*m', desc='flapwise bending moment at blade root')
        self.add_discrete_input('number_of_blades', val=0, desc='number of blades on the rotor')

        # outputs
        self.add_output('pitch_system_mass',     val=0.0, units='kg',  desc='overall component mass')

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
    
        mypitch = PitchSystem(discrete_inputs['number_of_blades'], debug=self.options['debug'])
        
        outputs['pitch_system_mass'] = mypitch.compute(inputs['blade_mass'], inputs['rotor_bending_moment_y'])


#-------------------------------------------------------------------------

class Spinner_OM(ExplicitComponent):
    '''
       Spinner class
          The SpinnerClass is used to represent the spinner of a wind turbine.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    def initialize(self):
        self.options.declare('debug', default=False)
        
    def setup(self):
        # variables
        #self.add_input('rotor_diameter', val=0.0, units='m', desc='rotor diameter')
        self.add_input('blade_root_diameter', val=0.0, units='m',  desc='blade root diameter')
        self.add_discrete_input('number_of_blades', val=0, desc='number of blades on the rotor')

        # outputs
        self.add_output('spinner_mass',       val=0.0, units='kg', desc='overall component mass')

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        myspinner = Spinner(discrete_inputs['number_of_blades'], debug=self.options['debug'])

        (outputs['spinner_mass'], _, _)  = myspinner.compute(inputs['blade_root_diameter']) 
        

#%%-----------------------

class HubSE(Group):
    def initialize(self):
        self.options.declare('debug', default=False)
        self.options.declare('mass_only', default=False)
        self.options.declare('topLevelFlag', default=True)
        
    def setup(self):
        debug = self.options['debug']

        '''
           HubSE class
              The HubSE class is used to represent the hub system of a wind turbine. 
              HubSE integrates the hub, pitch system and spinner / nose cone components for the hub system.
        '''
        self.add_subsystem('hub',         Hub_OM(        debug=debug), promotes=['*'])
        self.add_subsystem('pitchSystem', PitchSystem_OM(debug=debug), promotes=['*'])
        self.add_subsystem('spinner',     Spinner_OM(    debug=debug), promotes=['*'])
        if self.options['mass_only']:
            self.add_subsystem('adder',   Hub_Mass_Adder_OM(debug=debug), promotes=['*'])
        else:
            self.add_subsystem('adder',   Hub_System_Adder_OM(debug=debug), promotes=['*'])
        
        if self.options['topLevelFlag']:
            sharedIndeps = IndepVarComp()
            sharedIndeps.add_output('rotor_diameter', 0.0, units='m')
            sharedIndeps.add_discrete_output('number_of_blades', 0)
            sharedIndeps.add_output('blade_root_diameter', 0.0, units='m')
            sharedIndeps.add_output('blade_mass', 0.0, units='kg')
            sharedIndeps.add_output('machine_rating', 0.0, units='kW')
            sharedIndeps.add_output('shaft_angle', 0.0, units='rad')
            sharedIndeps.add_output('rotor_rpm', 0.0, units='rpm')
            self.add_subsystem('sharedIndeps', sharedIndeps, promotes=['*'])

        
#%%----------------------------
        
        #-------------------------------------------------------------------------
        # Examples based on reference turbines including the NREL 5 MW, WindPACT 1.5 MW and the GRC 750 kW system.

def example_5MW_4pt(debug=False):
    
    # simple test of module
    
    prob=Problem()
    prob.model = HubSE(debug=debug)
    '''
    recorder = HDF5Recorder('dump5MW4pt.h5')
    recorder.options['record_params'] = True
    recorder.options['record_metadata'] = True
    prob.driver.add_recorder(recorder)
    '''
    
    prob.setup()
    blade_number = 3
    prob['number_of_blades'] = blade_number
    prob['rotor_diameter'] = 126.0  # m
    prob['blade_root_diameter'] = 3.542
    prob['machine_rating'] = 5000.0
    prob['blade_mass'] = 17740.0  # kg
    prob['shaft_angle'] = np.radians(5)
    prob['rotor_rpm'] = 12.1
    #prob['distance_hub2mb'] = 0.0
    #prob['MB1_location'] = [0.0, 0.0, 0.0]

    AirDensity = 1.225  # kg/(m^3)
    Solidity = 0.0517
    RatedWindSpeed = 11.05  # m/s
    prob['rotor_bending_moment_y'] = (3.06 * np.pi / 8) * AirDensity \
         * (RatedWindSpeed ** 2) * (Solidity * (prob['rotor_diameter'] ** 3)) / blade_number

    prob.run_driver()

    print("NREL 5 MW spherical-hub turbine test")
    prob.model.list_inputs(units=True)#values = False, hierarchical=False)
    prob.model.list_outputs(units=True)#values = False, hierarchical=False)    

#%%----------------------------
        
def example_BAR_4pt():
    
    # created 2019 05 03 by GNS - copied and modified example_5MW_4pt()
    
    blade_number = 3

    prob=Problem()
    prob.model = HubSE(debug=debug)
    prob.setup()

    blade_number = 3
    prob['number_of_blades'] = blade_number
    prob['rotor_diameter'] = 200.0  # m
    prob['blade_root_diameter'] = 4.5
    prob['machine_rating'] = 5000.0
    prob['blade_mass'] = 61400.0  # kg
    prob['shaft_angle'] = np.radians(5)

    AirDensity = 1.225  # kg/(m^3)
    Solidity = 0.0517
    RatedWindSpeed = 11.05  # m/s
    prob['rotor_bending_moment_y'] = (3.06 * np.pi / 8) * AirDensity \
         * (RatedWindSpeed ** 2) * (Solidity * (prob['rotor_diameter'] ** 3)) / blade_number

    prob.run_driver()

    print("NREL BAR turbine test")
    prob.model.list_inputs(units=True)#values = False, hierarchical=False)
    prob.model.list_outputs(units=True)#values = False, hierarchical=False)    

'''
# TODO: update other examples for the hub system
def example_1p5MW_4pt():

    # WindPACT 1.5 MW turbine
    prob = Problem()
    prob.model = HubSE()
    prob.setup()

    prob['blade_mass'] = 4470.0
    prob['rotor_diameter'] = 70.0
    prob['blade_number'] = 3
    prob['hub_diameter'] = 3.0
    prob['machine_rating'] = 1500.0
    prob['blade_root_diameter'] = 2.0  # TODO - find actual number

    AirDensity = 1.225
    Solidity = 0.065
    RatedWindSpeed = 12.12
    prob['rotor_bending_moment'] = (3.06 * pi / 8) * AirDensity * (
        RatedWindSpeed ** 2) * (Solidity * (prob['rotor_diameter'] ** 3)) / prob['blade_number']

    prob.run_driver()

    print("WindPACT 1.5 MW turbine test")
    print("Hub Objects")
    print('  hub         {0:8.1f} kg'.format(prob['hub_mass']))  # 31644.47
    print('  pitch mech  {0:8.1f} kg'.format(prob['pitch_system_mass']))  # 17003.98
    print('  nose cone   {0:8.1f} kg'.format(prob['spinner_mass']))  # 1810.50
    # 50458.95
    print('Hub system total {0:8.1f} kg'.format(prob['hub_system_mass']))
    print('    cm {0:6.2f} {1:6.2f} {2:6.2f}'.format(prob['hub_system_cm'][0], prob['hub_system_cm'][1], prob['hub_system_cm'][2]))
    print('    I {0:6.1f} {1:6.1f} {2:6.1f}'.format(prob['hub_system_I'][0], prob['hub_system_I'][1], prob['hub_system_I'][2]))
    print()

def example_750kW_4pt():

    # GRC 750 kW turbine
    prob = Problem()
    prob.model = HubSE()
    prob.setup()

    prob['blade_mass'] = 3400.0
    prob['rotor_diameter'] = 48.2
    prob['blade_number'] = 3
    prob['hub_diameter'] = 3.0
    prob['machine_rating'] = 750.0
    prob['blade_root_diameter'] = 1.0  # TODO - find actual number

    AirDensity = 1.225
    Solidity = 0.07  # uknown value
    RatedWindSpeed = 16.0
    prob['rotor_bending_moment'] = (3.06 * pi / 8) * AirDensity * (
        RatedWindSpeed ** 2) * (Solidity * (prob['rotor_diameter'] ** 3)) / prob['blade_number']

    prob.run_driver()

    print("windpact 750 kW turbine test")
    print("Hub Objects")
    print('  hub         {0:8.1f} kg'.format(prob['hub_mass']))  # 31644.47
    print('  pitch mech  {0:8.1f} kg'.format(prob['pitch_system_mass']))  # 17003.98
    print('  nose cone   {0:8.1f} kg'.format(prob['spinner_mass']))  # 1810.50
    # 50458.95
    print('Hub system total {0:8.1f} kg'.format(prob['hub_system_mass']))
    print('    cm {0:6.2f} {1:6.2f} {2:6.2f}'.format(prob['hub_system_cm'][0], prob['hub_system_cm'][1], prob['hub_system_cm'][2]))
    print('    I {0:6.1f} {1:6.1f} {2:6.1f}'.format(prob['hub_system_I'][0], prob['hub_system_I'][1], prob['hub_system_I'][2]))
    print()

'''

# Main code to run hub system examples
if __name__ == "__main__":

    debug = True
    example_5MW_4pt(debug=debug)
