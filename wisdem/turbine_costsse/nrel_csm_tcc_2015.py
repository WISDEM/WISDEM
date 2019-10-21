"""
tcc_csm_component.py

Created by NWTC Systems Engineering Sub-Task on 2012-08-01.
Copyright (c) NREL. All rights reserved.
"""

from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent, Problem, Group, IndepVarComp

from wisdem.turbine_costsse.turbine_costsse_2015 import Turbine_CostsSE_2015

# --------------------------------------------------------------------
class BladeMass(ExplicitComponent):
    
    def setup(self):
        
        # Variables
        self.add_input('rotor_diameter', 0.0, units='m', desc= 'rotor diameter of the machine')
        self.add_discrete_input('turbine_class', 1, desc='turbine class')
        self.add_discrete_input('blade_has_carbon', False, desc= 'does the blade have carbon?') #default to doesn't have carbon
        self.add_input('blade_mass_coeff', 0.5, desc= 'A in the blade mass equation: A*(rotor_diameter/B)^exp') #default from ppt
        self.add_input('blade_user_exp', 2.5, desc='optional user-entered exp for the blade mass equation')
        
        # Outputs
        self.add_output('blade_mass', 0.0, units='kg', desc= 'component mass [kg]')
  
    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        
        rotor_diameter = inputs['rotor_diameter']
        turbine_class = discrete_inputs['turbine_class']
        blade_has_carbon = discrete_inputs['blade_has_carbon']
        blade_mass_coeff = inputs['blade_mass_coeff']
        blade_user_exp = inputs['blade_user_exp']
    
        # select the exp for the blade mass equation
        exp = 0.0
        if turbine_class == 1:
            if blade_has_carbon:
              exp = 2.47
            else:
              exp = 2.54
        elif turbine_class > 1:
            if blade_has_carbon:
              exp = 2.44
            else:
              exp = 2.50
        else:
            exp = blade_user_exp
        
        # calculate the blade mass
        outputs['blade_mass'] = blade_mass_coeff * (rotor_diameter / 2)**exp

  # --------------------------------------------------------------------
class HubMass(ExplicitComponent):

    def setup(self):
        
        # Variables
        self.add_input('blade_mass', 0.0, units='kg', desc= 'component mass [kg]')
        self.add_input('hub_mass_coeff', 2.3, desc= 'A in the hub mass equation: A*blade_mass + B') #default from ppt
        self.add_input('hub_mass_intercept', 1320., desc= 'B in the hub mass equation: A*blade_mass + B') #default from ppt
        
        # Outputs
        self.add_output('hub_mass', 0.0, units='kg', desc='component mass [kg]')
  
    def compute(self, inputs, outputs):
      
        blade_mass = inputs['blade_mass']
        hub_mass_coeff = inputs['hub_mass_coeff']
        hub_mass_intercept = inputs['hub_mass_intercept']
        
        # calculate the hub mass
        outputs['hub_mass'] = hub_mass_coeff * blade_mass + hub_mass_intercept

# --------------------------------------------------------------------
class PitchSystemMass(ExplicitComponent):
    
    def setup(self):
        
        self.add_input('blade_mass', 0.0, units='kg', desc= 'component mass [kg]')
        self.add_discrete_input('blade_number', 3, desc='number of rotor blades')
        self.add_input('pitch_bearing_mass_coeff', 0.1295, desc='A in the pitch bearing mass equation: A*blade_mass*blade_number + B') #default from old CSM
        self.add_input('pitch_bearing_mass_intercept', 491.31, desc='B in the pitch bearing mass equation: A*blade_mass*blade_number + B') #default from old CSM
        self.add_input('bearing_housing_percent', .3280, desc='bearing housing percentage (in decimal form: ex 10% is 0.10)') #default from old CSM
        self.add_input('mass_sys_offset', 555.0, desc='mass system offset') #default from old CSM
        
        # Outputs
        self.add_output('pitch_system_mass', 0.0, units='kg', desc='component mass [kg]')
    
    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        
        blade_mass = inputs['blade_mass']
        blade_number = discrete_inputs['blade_number']
        pitch_bearing_mass_coeff = inputs['pitch_bearing_mass_coeff']
        pitch_bearing_mass_intercept = inputs['pitch_bearing_mass_intercept']
        bearing_housing_percent = inputs['bearing_housing_percent']
        mass_sys_offset = inputs['mass_sys_offset']
        
        # calculate the hub mass
        pitchBearingMass = pitch_bearing_mass_coeff * blade_mass * blade_number + pitch_bearing_mass_intercept
        outputs['pitch_system_mass'] = pitchBearingMass * (1 + bearing_housing_percent) + mass_sys_offset

# --------------------------------------------------------------------
class SpinnerMass(ExplicitComponent):

    def setup(self):
        
    
        # Variables
        self.add_input('rotor_diameter', 0.0, units='m', desc= 'rotor diameter of the machine')
        self.add_input('spinner_mass_coeff', 15.5, desc= 'A in the spinner mass equation: A*rotor_diameter + B')
        self.add_input('spinner_mass_intercept', -980.0, desc= 'B in the spinner mass equation: A*rotor_diameter + B')
        
        # Outputs
        self.add_output('spinner_mass', 0.0, units='kg', desc='component mass [kg]')
    
    def compute(self, inputs, outputs):
        
        rotor_diameter = inputs['rotor_diameter']
        spinner_mass_coeff = inputs['spinner_mass_coeff']
        spinner_mass_intercept = inputs['spinner_mass_intercept']
        
        # calculate the spinner mass
        outputs['spinner_mass'] = spinner_mass_coeff * rotor_diameter + spinner_mass_intercept

# --------------------------------------------------------------------
class LowSpeedShaftMass(ExplicitComponent):

    def setup(self):
        
        
        # Variables
        self.add_input('blade_mass', 0.0, units='kg', desc='mass for a single wind turbine blade')
        self.add_input('machine_rating', 0.0, units='kW', desc='machine rating')
        self.add_input('lss_mass_coeff', 13., desc='A in the lss mass equation: A*(blade_mass*rated_power)^exp + B')
        self.add_input('lss_mass_exp', 0.65, desc='exp in the lss mass equation: A*(blade_mass*rated_power)^exp + B')
        self.add_input('lss_mass_intercept', 775., desc='B in the lss mass equation: A*(blade_mass*rated_power)^exp + B')
        
        # Outputs
        self.add_output('lss_mass', 0.0, units='kg', desc='component mass [kg]')
    
    def compute(self, inputs, outputs):
        
        blade_mass = inputs['blade_mass']
        machine_rating = inputs['machine_rating']
        lss_mass_coeff = inputs['lss_mass_coeff']
        lss_mass_exp = inputs['lss_mass_exp']
        lss_mass_intercept = inputs['lss_mass_intercept']
    
        # calculate the lss mass
        outputs['lss_mass'] = lss_mass_coeff * (blade_mass * machine_rating/1000.)**lss_mass_exp + lss_mass_intercept

# --------------------------------------------------------------------
class BearingMass(ExplicitComponent):

    def setup(self):


        # Variables
        self.add_input('rotor_diameter', 0.0, units='m', desc= 'rotor diameter of the machine')
        self.add_input('bearing_mass_coeff', 0.0001, desc= 'A in the bearing mass equation: A*rotor_diameter^exp') #default from ppt
        self.add_input('bearing_mass_exp', 3.5, desc= 'exp in the bearing mass equation: A*rotor_diameter^exp') #default from ppt
        
        # Outputs
        self.add_output('main_bearing_mass', 0.0, units='kg', desc='component mass [kg]')
    
    def compute(self, inputs, outputs):
        
        rotor_diameter = inputs['rotor_diameter']
        bearing_mass_coeff = inputs['bearing_mass_coeff']
        bearing_mass_exp = inputs['bearing_mass_exp']
        
        # calculates the mass of a SINGLE bearing
        outputs['main_bearing_mass'] = bearing_mass_coeff * rotor_diameter ** bearing_mass_exp

# --------------------------------------------------------------------
class RotorTorque(ExplicitComponent):

    def setup(self):
  
        # Variables
        self.add_input('rotor_diameter', 0.0, units='m', desc= 'rotor diameter of the machine')
        self.add_input('machine_rating', 0.0, units='kW', desc='machine rating')
        self.add_input('max_tip_speed', 0.0, units='m/s', desc='Maximum allowable blade tip speed')
        self.add_input('max_efficiency', 0.0, desc='Maximum possible drivetrain efficiency')

        self.add_output('rotor_torque', 0.0, units='N*m', desc = 'torque from rotor at rated power') #JMF do we want this default?

    def compute(self, inputs, outputs):
        # Rotor force calculations for nacelle inputs
        maxTipSpd = inputs['max_tip_speed']
        maxEfficiency = inputs['max_efficiency']
        
        ratedHubPower_W  = inputs['machine_rating']*1000. / maxEfficiency 
        rotorSpeed       = maxTipSpd / (0.5*inputs['rotor_diameter'])
        outputs['rotor_torque'] = ratedHubPower_W / rotorSpeed
# --------------------------------------------------------------------
class GearboxMass(ExplicitComponent):

    def setup(self):
  
  
        # Variables
        self.add_input('rotor_torque', 0.0, units='N*m', desc = 'torque from rotor at rated power') #JMF do we want this default?
        self.add_input('gearbox_mass_coeff', 113., desc= 'A in the gearbox mass equation: A*rotor_torque^exp')
        self.add_input('gearbox_mass_exp', 0.71, desc= 'exp in the gearbox mass equation: A*rotor_torque^exp')
        
        # Outputs
        self.add_output('gearbox_mass', 0.0, units='kg', desc='component mass [kg]')
    
    def compute(self, inputs, outputs):
        
        rotor_torque = inputs['rotor_torque']
        gearbox_mass_coeff = inputs['gearbox_mass_coeff']
        gearbox_mass_exp = inputs['gearbox_mass_exp']
        
        # calculate the gearbox mass
        outputs['gearbox_mass'] = gearbox_mass_coeff * (rotor_torque/1000.0)**gearbox_mass_exp

# --------------------------------------------------------------------
class HighSpeedSideMass(ExplicitComponent):

    def setup(self):
      

        # Variables
        self.add_input('machine_rating', 0.0, units='kW', desc='machine rating')
        self.add_input('hss_mass_coeff', 0.19894, desc= 'NREL CSM hss equation; removing intercept since it is negligible')
        
        # Outputs
        self.add_output('hss_mass', 0.0, units='kg', desc='component mass [kg]')
    
    def compute(self, inputs, outputs):
        
        machine_rating = inputs['machine_rating']
        hss_mass_coeff = inputs['hss_mass_coeff']
        
        # TODO: this is in DriveSE; replace this with code in DriveSE and have DriveSE use this code??
        outputs['hss_mass'] = hss_mass_coeff * machine_rating

# --------------------------------------------------------------------
class GeneratorMass(ExplicitComponent):

    def setup(self):
      
        
  
        # Variables
        self.add_input('machine_rating', 0.0, units='kW', desc='machine rating')
        self.add_input('generator_mass_coeff', 2300., desc= 'A in the generator mass equation: A*rated_power + B') #default from ppt
        self.add_input('generator_mass_intercept', 3400., desc= 'B in the generator mass equation: A*rated_power + B') #default from ppt
        
        # Outputs
        self.add_output('generator_mass', 0.0, units='kg', desc='component mass [kg]')
    
    def compute(self, inputs, outputs):

        machine_rating = inputs['machine_rating']
        generator_mass_coeff = inputs['generator_mass_coeff']
        generator_mass_intercept = inputs['generator_mass_intercept']
    
        # calculate the generator mass
        outputs['generator_mass'] = generator_mass_coeff * machine_rating/1000. + generator_mass_intercept

# --------------------------------------------------------------------
class BedplateMass(ExplicitComponent):

    def setup(self):


        # Variables
        self.add_input('rotor_diameter', 0.0, units='m', desc= 'rotor diameter of the machine')
        self.add_input('bedplate_mass_exp', 2.2, desc= 'exp in the bedplate mass equation: rotor_diameter^exp')
        
        # Outputs
        self.add_output('bedplate_mass', 0.0, units='kg', desc='component mass [kg]')
    
    def compute(self, inputs, outputs):
        
        rotor_diameter = inputs['rotor_diameter']
        bedplate_mass_exp = inputs['bedplate_mass_exp']
        
        # calculate the bedplate mass
        outputs['bedplate_mass'] = rotor_diameter**bedplate_mass_exp

# --------------------------------------------------------------------
class YawSystemMass(ExplicitComponent):
  
    def setup(self):


        # Variables
        self.add_input('rotor_diameter', 0.0, units='m', desc= 'rotor diameter of the machine')
        self.add_input('yaw_mass_coeff', 0.0009, desc= 'A in the yaw mass equation: A*rotor_diameter^exp') #NREL CSM
        self.add_input('yaw_mass_exp', 3.314, desc= 'exp in the yaw mass equation: A*rotor_diameter^exp') #NREL CSM
        
        # Outputs
        self.add_output('yaw_mass', 0.0, units='kg', desc='component mass [kg]')
    
    def compute(self, inputs, outputs):
      
        rotor_diameter = inputs['rotor_diameter']
        yaw_mass_coeff = inputs['yaw_mass_coeff']
        yaw_mass_exp = inputs['yaw_mass_exp']
    
        # calculate yaw system mass #TODO - 50% adder for non-bearing mass
        outputs['yaw_mass'] = 1.5 * (yaw_mass_coeff * rotor_diameter ** yaw_mass_exp) #JMF do we really want to expose all these?

#TODO: no variable speed mass; ignore for now

# --------------------------------------------------------------------
class HydraulicCoolingMass(ExplicitComponent):
    
    def setup(self):
    
        
        # Variables
        self.add_input('machine_rating', 0.0, units='kW', desc='machine rating')
        self.add_input('hvac_mass_coeff', 0.08, desc= 'hvac linear coeff') #NREL CSM
        
        # Outputs
        self.add_output('hvac_mass', 0.0, units='kg', desc='component mass [kg]')
    
    def compute(self, inputs, outputs):
        
        machine_rating = inputs['machine_rating']
        hvac_mass_coeff = inputs['hvac_mass_coeff']
        
        # calculate hvac system mass
        outputs['hvac_mass'] = hvac_mass_coeff * machine_rating

# --------------------------------------------------------------------
class NacelleCoverMass(ExplicitComponent):

    def setup(self):
    
    
        # Variables
        self.add_input('machine_rating', 0.0, units='kW', desc='machine rating')
        self.add_input('cover_mass_coeff', 1.2817, desc= 'A in the spinner mass equation: A*rotor_diameter + B')
        self.add_input('cover_mass_intercept', 428.19, desc= 'B in the spinner mass equation: A*rotor_diameter + B')
        
        # Outputs
        self.add_output('cover_mass', 0.0, units='kg', desc='component mass [kg]')
    
    def compute(self, inputs, outputs):
        
        machine_rating = inputs['machine_rating']
        cover_mass_coeff = inputs['cover_mass_coeff']
        cover_mass_intercept = inputs['cover_mass_intercept']
        
        # calculate nacelle cover mass
        outputs['cover_mass'] = cover_mass_coeff * machine_rating + cover_mass_intercept

# TODO: ignoring controls and electronics mass for now

# --------------------------------------------------------------------
class OtherMainframeMass(ExplicitComponent):
    # nacelle platforms, service crane, base hardware
    
    def setup(self):
    
        
        # Variables
        self.add_input('bedplate_mass', 0.0, units='kg', desc='component mass [kg]')
        self.add_input('platforms_mass_coeff', 0.125, desc='nacelle platforms mass coeff as a function of bedplate mass [kg/kg]') #default from old CSM
        self.add_discrete_input('crane', False, desc='flag for presence of onboard crane')
        self.add_input('crane_weight', 3000., desc='weight of onboard crane')
        #TODO: there is no base hardware mass model in the old model. Cost is not dependent on mass.
        
        # Outputs
        self.add_output('other_mass', 0.0, units='kg', desc='component mass [kg]')
    
    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        
        bedplate_mass = inputs['bedplate_mass']
        platforms_mass_coeff = inputs['platforms_mass_coeff']
        crane = discrete_inputs['crane']
        crane_weight = inputs['crane_weight']
        
        # calculate nacelle cover mass           
        platforms_mass = platforms_mass_coeff * bedplate_mass

        # --- crane ---        
        if (crane):
            crane_mass =  crane_weight
        else:
            crane_mass = 0.  
        
        outputs['other_mass'] = platforms_mass + crane_mass

# --------------------------------------------------------------------
class TransformerMass(ExplicitComponent):

    def setup(self):
    
    
        # Variables
        self.add_input('machine_rating', 0.0, units='kW', desc='machine rating')
        self.add_input('transformer_mass_coeff', 1915., desc= 'A in the transformer mass equation: A*rated_power + B') #default from ppt
        self.add_input('transformer_mass_intercept', 1910., desc= 'B in the transformer mass equation: A*rated_power + B') #default from ppt
        
        # Outputs
        self.add_output('transformer_mass', 0.0, units='kg', desc='component mass [kg]')
    
    def compute(self, inputs, outputs):
        
        machine_rating = inputs['machine_rating']
        transformer_mass_coeff = inputs['transformer_mass_coeff']
        transformer_mass_intercept = inputs['transformer_mass_intercept']
        
        # calculate the transformer mass
        outputs['transformer_mass'] = transformer_mass_coeff * machine_rating/1000. + transformer_mass_intercept

# --------------------------------------------------------------------
class TowerMass(ExplicitComponent):
  
    def setup(self):


        # Variables
        self.add_input('hub_height', 0.0, desc= 'hub height of wind turbine above ground / sea level')
        self.add_input('tower_mass_coeff', 19.828, desc= 'A in the tower mass equation: A*hub_height^B') #default from ppt
        self.add_input('tower_mass_exp', 2.0282, desc= 'B in the tower mass equation: A*hub_height^B') #default from ppt
        
        # Outputs
        self.add_output('tower_mass', 0.0, units='kg', desc='component mass [kg]')
    
    def compute(self, inputs, outputs):
        
        hub_height = inputs['hub_height']
        tower_mass_coeff = inputs['tower_mass_coeff']
        tower_mass_exp = inputs['tower_mass_exp']
        
        # calculate the tower mass
        outputs['tower_mass'] = tower_mass_coeff * hub_height ** tower_mass_exp
 

# Turbine mass adder
class turbine_mass_adder(ExplicitComponent):
    
    def setup(self):
        
    
        # Inputs
        # rotor
        self.add_input('blade_mass', 0.0, units='kg', desc= 'component mass [kg]')
        self.add_input('hub_mass', 0.0, units='kg', desc='component mass [kg]')
        self.add_input('pitch_system_mass', 0.0, units='kg', desc='component mass [kg]')
        self.add_input('spinner_mass', 0.0, units='kg', desc='component mass [kg]')
        # nacelle
        self.add_input('lss_mass', 0.0, units='kg', desc='component mass [kg]')
        self.add_input('main_bearing_mass', 0.0, units='kg', desc='component mass [kg]')
        self.add_input('gearbox_mass', 0.0, units='kg', desc='component mass [kg]')
        self.add_input('hss_mass', 0.0, units='kg', desc='component mass [kg]')
        self.add_input('generator_mass', 0.0, units='kg', desc='component mass [kg]')
        self.add_input('bedplate_mass', 0.0, units='kg', desc='component mass [kg]')
        self.add_input('yaw_mass', 0.0, units='kg', desc='component mass [kg]')
        self.add_input('hvac_mass', 0.0, units='kg', desc='component mass [kg]')
        self.add_input('cover_mass', 0.0, units='kg', desc='component mass [kg]')
        self.add_input('other_mass', 0.0, units='kg', desc='component mass [kg]')
        self.add_input('transformer_mass', 0.0, units='kg', desc='component mass [kg]')
        # tower
        self.add_input('tower_mass', 0.0, units='kg', desc='component mass [kg]')
    
        # Parameters
        self.add_discrete_input('blade_number', 3, desc = 'number of rotor blades')
        self.add_discrete_input('bearing_number', 2, desc = 'number of main bearings')
    
        # Outputs
        self.add_output('hub_system_mass', 0.0, units='kg', desc='hub system mass')
        self.add_output('rotor_mass', 0.0, units='kg', desc='hub system mass')
        self.add_output('nacelle_mass', 0.0, units='kg', desc='nacelle mass')
        self.add_output('turbine_mass', 0.0, units='kg', desc='turbine mass')
    
    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        blade_mass = inputs['blade_mass']
        hub_mass = inputs['hub_mass']
        pitch_system_mass = inputs['pitch_system_mass']
        spinner_mass = inputs['spinner_mass']
        lss_mass = inputs['lss_mass']
        main_bearing_mass = inputs['main_bearing_mass']
        gearbox_mass = inputs['gearbox_mass']
        hss_mass = inputs['hss_mass']
        generator_mass = inputs['generator_mass']
        bedplate_mass = inputs['bedplate_mass']
        yaw_mass = inputs['yaw_mass']
        hvac_mass = inputs['hvac_mass']
        cover_mass = inputs['cover_mass']
        other_mass = inputs['other_mass']
        transformer_mass = inputs['transformer_mass']
        tower_mass = inputs['tower_mass']
        blade_number = discrete_inputs['blade_number']
        bearing_number = discrete_inputs['bearing_number']
        
        
        outputs['hub_system_mass'] = hub_mass + pitch_system_mass + spinner_mass
        outputs['rotor_mass'] = blade_mass * blade_number + outputs['hub_system_mass']
        outputs['nacelle_mass'] = lss_mass + bearing_number * main_bearing_mass + \
                            gearbox_mass + hss_mass + generator_mass + \
                            bedplate_mass + yaw_mass + hvac_mass + \
                            cover_mass + other_mass + transformer_mass
        outputs['turbine_mass'] = outputs['rotor_mass'] + outputs['nacelle_mass'] + tower_mass

# --------------------------------------------------------------------

class nrel_csm_mass_2015(Group):
    
    def setup(self):

        sharedIndeps = IndepVarComp()
        sharedIndeps.add_output('machine_rating',     units='kW', val=0.0)
        sharedIndeps.add_output('rotor_diameter',         units='m', val=0.0)
        sharedIndeps.add_discrete_output('blade_number',  val=0)
        sharedIndeps.add_discrete_output('crane', val=False)
        self.add_subsystem('sharedIndeps', sharedIndeps, promotes=['*'])

        self.add_subsystem('blade',BladeMass(), promotes=['*'])
        self.add_subsystem('hub',HubMass(), promotes=['*'])
        self.add_subsystem('pitch',PitchSystemMass(), promotes=['*'])
        self.add_subsystem('spinner',SpinnerMass(), promotes=['*'])
        self.add_subsystem('lss',LowSpeedShaftMass(), promotes=['*'])
        self.add_subsystem('bearing',BearingMass(), promotes=['*'])
        self.add_subsystem('torque',RotorTorque(), promotes=['*'])
        self.add_subsystem('gearbox',GearboxMass(), promotes=['*'])
        self.add_subsystem('hss',HighSpeedSideMass(), promotes=['*'])
        self.add_subsystem('generator',GeneratorMass(), promotes=['*'])
        self.add_subsystem('bedplate',BedplateMass(), promotes=['*'])
        self.add_subsystem('yaw',YawSystemMass(), promotes=['*'])
        self.add_subsystem('hvac',HydraulicCoolingMass(), promotes=['*'])
        self.add_subsystem('cover',NacelleCoverMass(), promotes=['*'])
        self.add_subsystem('other',OtherMainframeMass(), promotes=['*'])
        self.add_subsystem('transformer',TransformerMass(), promotes=['*'])
        self.add_subsystem('tower',TowerMass(), promotes=['*'])
        self.add_subsystem('turbine',turbine_mass_adder(), promotes=['*'])
       

class nrel_csm_2015(Group):
    
    def setup(self):
        self.add_subsystem('nrel_csm_mass', nrel_csm_mass_2015(), promotes=['*'])
        self.add_subsystem('turbine_costs', Turbine_CostsSE_2015(verbosity=False, topLevelFlag=False), promotes=['*'])

#-----------------------------------------------------------------

def mass_example():

    # simple test of module
    trb = nrel_csm_mass_2015()
    prob = Problem(trb)
    prob.setup()
    
    prob['rotor_diameter'] = 126.0
    prob['turbine_class'] = 1
    prob['blade_has_carbon'] = False
    prob['blade_number'] = 3    
    prob['machine_rating'] = 5000.0
    prob['hub_height'] = 90.0
    prob['bearing_number'] = 2
    prob['crane'] = True
    prob['max_tip_speed'] = 80.0
    prob['max_efficiency'] = 0.90

    prob.run_model()
   
    print("The MASS results for the NREL 5 MW Reference Turbine in an offshore 20 m water depth location are:")
    #print "Overall turbine mass is {0:.2f} kg".format(trb.turbine.params['turbine_mass'])
    for io in trb._outputs:
        print(io, str(trb._outputs[io]))

def cost_example():

    # simple test of module
    trb = nrel_csm_2015()
    prob = Problem(trb)
    prob.setup()

    # simple test of module
    prob['rotor_diameter'] = 126.0
    prob['turbine_class'] = 1
    prob['blade_has_carbon'] = False
    prob['blade_number'] = 3    
    prob['machine_rating'] = 5000.0
    prob['hub_height'] = 90.0
    prob['bearing_number'] = 2
    prob['crane'] = True
    prob['max_tip_speed'] = 80.0
    prob['max_efficiency'] = 0.90

    prob.run_model()

    print("The COST results for the NREL 5 MW Reference Turbine:")
    for io in trb._outputs:
        print(io, str(trb._outputs[io]))

if __name__ == "__main__":
    mass_example()
    cost_example()
