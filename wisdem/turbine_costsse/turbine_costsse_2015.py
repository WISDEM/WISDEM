"""
turbine_costsse_2015.py

Created by Janine Freeman 2015 based on turbine_costsse.py 2012.
Copyright (c) NREL. All rights reserved.
"""

from openmdao.api import ExplicitComponent, Problem, Group, IndepVarComp

import numpy as np

###### Rotor
#-------------------------------------------------------------------------------
class BladeCost2015(ExplicitComponent):

    def setup(self):


        # Inputs
        self.add_input('blade_mass',            0.0,  units='kg',     desc='component mass')
        self.add_input('blade_mass_cost_coeff', 14.6, units='USD/kg', desc='blade mass-cost coeff')
        self.add_input('blade_cost_external',   0.0,  units='USD',    desc='Blade cost computed by RotorSE')
        
        # Outputs
        self.add_output('blade_cost',           0.0,  units='USD',    desc='Overall wind turbine component capital costs excluding transportation costs')

    def compute(self, inputs, outputs):

        blade_mass = inputs['blade_mass']
        blade_mass_cost_coeff = inputs['blade_mass_cost_coeff']

        # calculate component cost
        if inputs['blade_cost_external'] < 1.:
            outputs['blade_cost'] = blade_mass_cost_coeff * blade_mass
        else:
            outputs['blade_cost'] = inputs['blade_cost_external']
        

# -----------------------------------------------------------------------------------------------
class HubCost2015(ExplicitComponent):

    def setup(self):


        # variables
        self.add_input('hub_mass', 0.0, desc='component mass', units='kg')
        self.add_input('hub_mass_cost_coeff', 3.9, desc='hub mass-cost coeff', units='USD/kg')
    
        # Outputs
        self.add_output('hub_cost', 0.0, units='USD', desc='Overall wind turbine component capial costs excluding transportation costs')

    def compute(self, inputs, outputs):

        hub_mass_cost_coeff = inputs['hub_mass_cost_coeff']
        hub_mass = inputs['hub_mass']

        # calculate component cost
        HubCost2015 = hub_mass_cost_coeff * hub_mass
        outputs['hub_cost'] = HubCost2015
        

#-------------------------------------------------------------------------------
class PitchSystemCost2015(ExplicitComponent):

    def setup(self):


        # variables
        self.add_input('pitch_system_mass', 0.0, desc='component mass', units='kg')
        self.add_input('pitch_system_mass_cost_coeff', 22.1, desc='pitch system mass-cost coeff', units='USD/kg')
    
        # Outputs
        self.add_output('pitch_system_cost', 0.0, units='USD', desc='Overall wind turbine component capial costs excluding transportation costs')

    def compute(self, inputs, outputs):
        
        pitch_system_mass = inputs['pitch_system_mass']
        pitch_system_mass_cost_coeff = inputs['pitch_system_mass_cost_coeff']
        
        #calculate system costs
        PitchSystemCost2015 = pitch_system_mass_cost_coeff * pitch_system_mass
        outputs['pitch_system_cost'] = PitchSystemCost2015
        
#-------------------------------------------------------------------------------
class SpinnerCost2015(ExplicitComponent):

    def setup(self):


        # variables
        self.add_input('spinner_mass', 0.0, desc='component mass', units='kg')
        self.add_input('spinner_mass_cost_coeff', 11.1, desc='spinner/nose cone mass-cost coeff', units='USD/kg')
    
        # Outputs
        self.add_output('spinner_cost', 0.0, units='USD', desc='Overall wind turbine component capial costs excluding transportation costs')

    def compute(self, inputs, outputs):

        spinner_mass_cost_coeff = inputs['spinner_mass_cost_coeff']
        spinner_mass = inputs['spinner_mass']

        #calculate system costs
        SpinnerCost2015 = spinner_mass_cost_coeff * spinner_mass
        outputs['spinner_cost'] = SpinnerCost2015

#-------------------------------------------------------------------------------
class HubSystemCostAdder2015(ExplicitComponent):

    def setup(self):


        # Inputs
        self.add_input('hub_cost',          0.0, units='USD', desc='Hub component cost')
        self.add_input('hub_mass',          0.0, units='kg',  desc='Hub component mass')
        self.add_input('pitch_system_cost', 0.0, units='USD', desc='Pitch system cost')
        self.add_input('pitch_system_mass', 0.0, units='kg',  desc='Pitch system mass')
        self.add_input('spinner_cost',      0.0, units='USD', desc='Spinner component cost')
        self.add_input('spinner_mass',      0.0, units='kg', desc='Spinner component mass')
        self.add_input('hub_assemblyCostMultiplier',    0.0, desc='Rotor assembly cost multiplier')
        self.add_input('hub_overheadCostMultiplier',    0.0, desc='Rotor overhead cost multiplier')
        self.add_input('hub_profitMultiplier',          0.0, desc='Rotor profit multiplier')
        self.add_input('hub_transportMultiplier',       0.0, desc='Rotor transport multiplier')
    
        # Outputs
        self.add_output('hub_system_mass_tcc',  0.0, units='kg',  desc='Mass of the hub system, including hub, spinner, and pitch system for the blades')
        self.add_output('hub_system_cost',  0.0, units='USD', desc='Overall wind sub-assembly capial costs including transportation costs')

    def compute(self, inputs, outputs):

        hub_cost            = inputs['hub_cost']
        pitch_system_cost   = inputs['pitch_system_cost']
        spinner_cost        = inputs['spinner_cost']
        
        hub_mass            = inputs['hub_mass']
        pitch_system_mass   = inputs['pitch_system_mass']
        spinner_mass        = inputs['spinner_mass']
        
        hub_assemblyCostMultiplier  = inputs['hub_assemblyCostMultiplier']
        hub_overheadCostMultiplier  = inputs['hub_overheadCostMultiplier']
        hub_profitMultiplier        = inputs['hub_profitMultiplier']
        hub_transportMultiplier     = inputs['hub_transportMultiplier']

        # Updated calculations below to account for assembly, transport, overhead and profit
        outputs['hub_system_mass_tcc'] = hub_mass + pitch_system_mass + spinner_mass
        partsCost = hub_cost + pitch_system_cost + spinner_cost
        outputs['hub_system_cost'] = (1 + hub_transportMultiplier + hub_profitMultiplier) * ((1 + hub_overheadCostMultiplier + hub_assemblyCostMultiplier) * partsCost)

#-------------------------------------------------------------------------------
class RotorCostAdder2015(ExplicitComponent):
    """
    RotorCostAdder adds up individual rotor system and component costs to get overall rotor cost.
    """

    def setup(self):
        

        # Inputs
        self.add_input('blade_cost',        0.0, units='USD',   desc='Individual blade cost')
        self.add_input('blade_mass',        0.0, units='kg',    desc='Individual blade mass')
        self.add_input('hub_system_cost',   0.0, units='USD',   desc='Cost for hub system')
        self.add_input('hub_system_mass_tcc',   0.0, units='kg',    desc='Mass for hub system')
        self.add_discrete_input('blade_number',      3,                  desc='Number of rotor blades')
    
        # Outputs
        self.add_output('rotor_cost',       0.0, units='USD',   desc='Overall wind sub-assembly capial costs including transportation costs')
        self.add_output('rotor_mass_tcc',   0.0, units='kg',    desc='Rotor mass, including blades, pitch system, hub, and spinner')
        
    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        blade_cost      = inputs['blade_cost']
        blade_mass      = inputs['blade_mass']
        blade_number    = discrete_inputs['blade_number']
        hub_system_cost = inputs['hub_system_cost']
        hub_system_mass = inputs['hub_system_mass_tcc']

        outputs['rotor_cost']      = blade_cost * blade_number + hub_system_cost
        outputs['rotor_mass_tcc']  = blade_mass * blade_number + hub_system_mass

#-------------------------------------------------------------------------------


###### Nacelle
# -------------------------------------------------
class LowSpeedShaftCost2015(ExplicitComponent):

    def setup(self):


        # variables
        self.add_input('lss_mass', 0.0, desc='component mass', units='kg') #mass input
        self.add_input('lss_mass_cost_coeff', 11.9, desc='low speed shaft mass-cost coeff', units='USD/kg')
    
        # Outputs
        self.add_output('lss_cost', 0.0, units='USD', desc='Overall wind turbine component capial costs excluding transportation costs') #initialize cost output

    def compute(self, inputs, outputs):

        lss_mass_cost_coeff = inputs['lss_mass_cost_coeff']
        lss_mass = inputs['lss_mass']

        # calculate component cost
        outputs['lss_cost'] = lss_mass_cost_coeff * lss_mass

#-------------------------------------------------------------------------------
class BearingsCost2015(ExplicitComponent):

    def setup(self):


        # variables
        self.add_input('main_bearing_mass', 0.0, desc='component mass', units='kg') #mass input
        self.add_discrete_input('main_bearing_number', 2, desc='number of main bearings') #number of main bearings- defaults to 2
        self.add_input('bearings_mass_cost_coeff', 4.5, desc='main bearings mass-cost coeff', units='USD/kg')
    
        # Outputs
        self.add_output('main_bearing_cost', 0.0, units='USD', desc='Overall wind turbine component capial costs excluding transportation costs')

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        main_bearing_mass = inputs['main_bearing_mass']
        main_bearing_number = discrete_inputs['main_bearing_number']
        bearings_mass_cost_coeff = inputs['bearings_mass_cost_coeff']

        #calculate component cost 
        outputs['main_bearing_cost'] = bearings_mass_cost_coeff * main_bearing_mass * main_bearing_number

#-------------------------------------------------------------------------------
class GearboxCost2015(ExplicitComponent):

    def setup(self):


        # variables
        self.add_input('gearbox_mass', 0.0, units='kg', desc='component mass')
        self.add_input('gearbox_mass_cost_coeff', 12.9, desc='gearbox mass-cost coeff', units='USD/kg')
    
        # Outputs
        self.add_output('gearbox_cost', 0.0, units='USD', desc='Overall wind turbine component capial costs excluding transportation costs')

    def compute(self, inputs, outputs):

        gearbox_mass = inputs['gearbox_mass']
        gearbox_mass_cost_coeff = inputs['gearbox_mass_cost_coeff']

        outputs['gearbox_cost'] = gearbox_mass_cost_coeff * gearbox_mass

#-------------------------------------------------------------------------------
class HighSpeedSideCost2015(ExplicitComponent):

    def setup(self):
        

        # variables
        self.add_input('hss_mass', 0.0, desc='component mass', units='kg')
        self.add_input('hss_mass_cost_coeff', 6.8, desc='high speed side mass-cost coeff', units='USD/kg')
    
        # Outputs
        self.add_output('hss_cost', 0.0, units='USD', desc='Overall wind turbine component capial costs excluding transportation costs')

    def compute(self, inputs, outputs):

        hss_mass = inputs['hss_mass']
        hss_mass_cost_coeff = inputs['hss_mass_cost_coeff']
        
        outputs['hss_cost'] = hss_mass_cost_coeff * hss_mass

#-------------------------------------------------------------------------------
class GeneratorCost2015(ExplicitComponent):

    def setup(self):


        # variables
        self.add_input('generator_mass', 0.0, desc='component mass', units='kg')
        self.add_input('generator_mass_cost_coeff', 12.4, desc='generator mass cost coeff', units='USD/kg')
    
        # Outputs
        self.add_output('generator_cost', 0.0, units='USD', desc='Overall wind turbine component capial costs excluding transportation costs')

    def compute(self, inputs, outputs):

        generator_mass = inputs['generator_mass']
        generator_mass_cost_coeff = inputs['generator_mass_cost_coeff']
        
        outputs['generator_cost'] = generator_mass_cost_coeff * generator_mass

#-------------------------------------------------------------------------------
class BedplateCost2015(ExplicitComponent):

    def setup(self):
        
        
        # variables
        self.add_input('bedplate_mass', 0.0, desc='component mass', units='kg')
        self.add_input('bedplate_mass_cost_coeff', 2.9, desc='bedplate mass-cost coeff', units='USD/kg')
    
        # Outputs
        self.add_output('bedplate_cost', 0.0, units='USD', desc='Overall wind turbine component capial costs excluding transportation costs')

    def compute(self, inputs, outputs):

        bedplate_mass = inputs['bedplate_mass']
        bedplate_mass_cost_coeff = inputs['bedplate_mass_cost_coeff']

        outputs['bedplate_cost'] = bedplate_mass_cost_coeff * bedplate_mass

#---------------------------------------------------------------------------------
class YawSystemCost2015(ExplicitComponent):

    def setup(self):


        # variables
        self.add_input('yaw_mass', 0.0, desc='component mass', units='kg')
        self.add_input('yaw_mass_cost_coeff', 8.3, desc='yaw system mass cost coeff', units='USD/kg')
    
        # Outputs
        self.add_output('yaw_system_cost', 0.0, units='USD', desc='Overall wind turbine component capial costs excluding transportation costs')

    def compute(self, inputs, outputs):

        yaw_mass = inputs['yaw_mass']
        yaw_mass_cost_coeff = inputs['yaw_mass_cost_coeff']
        
        outputs['yaw_system_cost'] = yaw_mass_cost_coeff * yaw_mass

#---------------------------------------------------------------------------------
class VariableSpeedElecCost2015(ExplicitComponent):

    def setup(self):


        # variables
        self.add_input('vs_electronics_mass', 0.0, desc='component mass', units='kg')
        self.add_input('vs_electronics_mass_cost_coeff', 18.8, desc='variable speed electronics mass cost coeff', units='USD/kg')
    
        # Outputs
        self.add_output('vs_cost', 0.0, units='USD', desc='Overall wind turbine component capial costs excluding transportation costs')

    def compute(self, inputs, outputs):

        vs_electronics_mass = inputs['vs_electronics_mass']
        vs_electronics_mass_cost_coeff = inputs['vs_electronics_mass_cost_coeff']

        outputs['vs_cost'] = vs_electronics_mass_cost_coeff * vs_electronics_mass

#---------------------------------------------------------------------------------
class HydraulicCoolingCost2015(ExplicitComponent):

    def setup(self):


        # variables
        self.add_input('hvac_mass', 0.0, desc='component mass', units='kg')
        self.add_input('hvac_mass_cost_coeff', 124.0, desc='hydraulic and cooling system mass cost coeff', units='USD/kg')
    
        # Outputs
        self.add_output('hvac_cost', 0.0, units='USD', desc='Overall wind turbine component capial costs excluding transportation costs')

    def compute(self, inputs, outputs):

        hvac_mass = inputs['hvac_mass']
        hvac_mass_cost_coeff = inputs['hvac_mass_cost_coeff']

        # calculate cost
        outputs['hvac_cost'] = hvac_mass_cost_coeff * hvac_mass

#---------------------------------------------------------------------------------
class NacelleCoverCost2015(ExplicitComponent):

    def setup(self):


        # variables
        self.add_input('cover_mass', 0.0, desc='component mass', units='kg')
        self.add_input('cover_mass_cost_coeff', 5.7, desc='nacelle cover mass cost coeff', units='USD/kg')
    
        # Outputs
        self.add_output('cover_cost', 0.0, units='USD', desc='Overall wind turbine component capial costs excluding transportation costs')

    def compute(self, inputs, outputs):

        cover_mass = inputs['cover_mass']
        cover_mass_cost_coeff = inputs['cover_mass_cost_coeff']

        outputs['cover_cost'] = cover_mass_cost_coeff * cover_mass

#---------------------------------------------------------------------------------
class ElecConnecCost2015(ExplicitComponent):

    def setup(self):


        # variables
        self.add_input('machine_rating', 0.0, desc='machine rating', units='kW')
        self.add_input('elec_connec_machine_rating_cost_coeff', 41.85, units='USD/kW', desc='electrical connections cost coefficient per kW')
    
        # Outputs
        self.add_output('elec_cost', 0.0, units='USD', desc='Overall wind turbine component capial costs excluding transportation costs')

    def compute(self, inputs, outputs):

        machine_rating = inputs['machine_rating']
        elec_connec_machine_rating_cost_coeff = inputs['elec_connec_machine_rating_cost_coeff']

        outputs['elec_cost'] = elec_connec_machine_rating_cost_coeff * machine_rating


#---------------------------------------------------------------------------------
class ControlsCost2015(ExplicitComponent):

    def setup(self):


        # variables
        self.add_input('machine_rating', 0.0, desc='machine rating', units='kW')
        self.add_input('controls_machine_rating_cost_coeff', 21.15, units='USD/kW', desc='controls cost coefficient per kW')
    
        # Outputs
        self.add_output('controls_cost', 0.0, units='USD', desc='Overall wind turbine component capial costs excluding transportation costs')

    def compute(self, inputs, outputs):

        machine_rating = inputs['machine_rating']
        coeff          = inputs['controls_machine_rating_cost_coeff']

        outputs['controls_cost'] = machine_rating * coeff

#---------------------------------------------------------------------------------
class OtherMainframeCost2015(ExplicitComponent):

    def setup(self):  


        # variables
        self.add_input('platforms_mass', 0.0, desc='component mass', units='kg')
        self.add_input('platforms_mass_cost_coeff', 17.1, desc='nacelle platforms mass cost coeff', units='USD/kg')
        self.add_discrete_input('crane', False, desc='flag for presence of onboard crane')
        self.add_input('crane_cost', 12000.0, desc='crane cost if present', units='USD')
        # self.add_input('bedplate_cost', 0.0, desc='component cost', units='USD')
        # self.add_input('base_hardware_cost_coeff', 0.7, desc='base hardware cost coeff based on bedplate cost')
    
        # Outputs
        self.add_output('other_cost', 0.0, units='USD', desc='Overall wind turbine component capial costs excluding transportation costs')

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        platforms_mass = inputs['platforms_mass']
        platforms_mass_cost_coeff = inputs['platforms_mass_cost_coeff']
        crane = discrete_inputs['crane']
        crane_cost = inputs['crane_cost']
        # bedplate_cost = inputs['bedplate_cost']
        # base_hardware_cost_coeff = inputs['base_hardware_cost_coeff']

        # nacelle platform cost

        # crane cost
        if (crane):
            craneCost  = crane_cost
            craneMass  = 3e3
            NacellePlatformsCost = platforms_mass_cost_coeff * (platforms_mass - craneMass)
        else:
            craneCost  = 0.0
            NacellePlatformsCost = platforms_mass_cost_coeff * platforms_mass

        # base hardware cost
        #BaseHardwareCost = bedplate_cost * base_hardware_cost_coeff
    
        #aggregate all three mainframe costs
        outputs['other_cost'] = NacellePlatformsCost + craneCost #+ BaseHardwareCost

#-------------------------------------------------------------------------------
class TransformerCost2015(ExplicitComponent):

    def setup(self):


        # variables
        self.add_input('transformer_mass', 0.0, desc='component mass', units='kg')
        self.add_input('transformer_mass_cost_coeff', 18.8, desc='transformer mass cost coeff', units='USD/kg') #mass-cost coeff with default from ppt
    
        # Outputs
        self.add_output('transformer_cost', 0.0, units='USD', desc='Overall wind turbine component capial costs excluding transportation costs')

    def compute(self, inputs, outputs):

        transformer_mass = inputs['transformer_mass']
        transformer_mass_cost_coeff = inputs['transformer_mass_cost_coeff']
        
        outputs['transformer_cost'] = transformer_mass_cost_coeff * transformer_mass

#-------------------------------------------------------------------------------
class NacelleSystemCostAdder2015(ExplicitComponent):

    def setup(self):
        

        # variables
        self.add_input('lss_cost',          0.0, units='USD', desc='Component cost')
        self.add_input('lss_mass',          0.0, units='kg',  desc='Component mass')
        self.add_input('main_bearing_cost', 0.0, units='USD', desc='Component cost')
        self.add_input('main_bearing_mass', 0.0, units='kg',  desc='Component mass')
        self.add_input('gearbox_cost',      0.0, units='USD', desc='Component cost')
        self.add_input('gearbox_mass',      0.0, units='kg',  desc='Component mass')
        self.add_input('hss_cost',          0.0, units='USD', desc='Component cost')
        self.add_input('hss_mass',          0.0, units='kg',  desc='Component mass')
        self.add_input('generator_cost',    0.0, units='USD', desc='Component cost')
        self.add_input('generator_mass',    0.0, units='kg',  desc='Component mass')
        self.add_input('bedplate_cost',     0.0, units='USD', desc='Component cost')
        self.add_input('bedplate_mass',     0.0, units='kg',  desc='Component mass')
        self.add_input('yaw_system_cost',   0.0, units='USD', desc='Component cost')
        self.add_input('yaw_mass',          0.0, units='kg',  desc='Component mass')
        self.add_input('vs_cost',           0.0, units='USD', desc='Component cost')
        self.add_input('vs_mass',           0.0, units='kg',  desc='Component mass')
        self.add_input('hvac_cost',         0.0, units='USD', desc='Component cost')
        self.add_input('hvac_mass',         0.0, units='kg',  desc='Component mass')
        self.add_input('cover_cost',        0.0, units='USD', desc='Component cost')
        self.add_input('cover_mass',        0.0, units='kg',  desc='Component mass')
        self.add_input('elec_cost',         0.0, units='USD', desc='Component cost')
        self.add_input('controls_cost',     0.0, units='USD', desc='Component cost')
        self.add_input('other_cost',        0.0, units='USD', desc='Component cost')
        self.add_input('transformer_cost',  0.0, units='USD', desc='Component cost')
        self.add_input('transformer_mass',  0.0, units='kg',  desc='Component mass')
        self.add_discrete_input('main_bearing_number', 2, desc ='number of bearings')
        
        #multipliers
        self.add_input('nacelle_assemblyCostMultiplier', 0.0, desc='nacelle assembly cost multiplier')
        self.add_input('nacelle_overheadCostMultiplier', 0.0, desc='nacelle overhead cost multiplier')
        self.add_input('nacelle_profitMultiplier',       0.0, desc='nacelle profit multiplier')
        self.add_input('nacelle_transportMultiplier',    0.0, desc='nacelle transport multiplier')
    
        # returns
        self.add_output('nacelle_cost', 0.0, units='USD', desc='component cost')
        self.add_output('nacelle_mass_tcc', 0.0, units='kg',  desc='Nacelle mass, with all nacelle components, without the rotor')

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        lss_cost            = inputs['lss_cost']
        main_bearing_cost   = inputs['main_bearing_cost']
        gearbox_cost        = inputs['gearbox_cost']
        hss_cost            = inputs['hss_cost']
        generator_cost      = inputs['generator_cost']
        bedplate_cost       = inputs['bedplate_cost']
        yaw_system_cost     = inputs['yaw_system_cost']
        vs_cost             = inputs['vs_cost']
        hvac_cost           = inputs['hvac_cost']
        cover_cost          = inputs['cover_cost']
        elec_cost           = inputs['elec_cost']
        controls_cost       = inputs['controls_cost']
        other_cost          = inputs['other_cost']
        transformer_cost    = inputs['transformer_cost']
        
        lss_mass            = inputs['lss_mass']
        main_bearing_mass   = inputs['main_bearing_mass']
        gearbox_mass        = inputs['gearbox_mass']
        hss_mass            = inputs['hss_mass']
        generator_mass      = inputs['generator_mass']
        bedplate_mass       = inputs['bedplate_mass']
        yaw_mass            = inputs['yaw_mass']
        vs_mass             = inputs['vs_mass']
        hvac_mass           = inputs['hvac_mass']
        cover_mass          = inputs['cover_mass']
        transformer_mass    = inputs['transformer_mass']
        
        main_bearing_number = discrete_inputs['main_bearing_number']

        nacelle_assemblyCostMultiplier  = inputs['nacelle_assemblyCostMultiplier']
        nacelle_overheadCostMultiplier  = inputs['nacelle_overheadCostMultiplier']
        nacelle_profitMultiplier        = inputs['nacelle_profitMultiplier']
        nacelle_transportMultiplier     = inputs['nacelle_transportMultiplier']        

        #apply multipliers for assembly, transport, overhead, and profits
        outputs['nacelle_mass_tcc'] = lss_mass + main_bearing_number * main_bearing_mass + gearbox_mass + hss_mass + generator_mass + bedplate_mass + yaw_mass + vs_mass + hvac_mass + cover_mass + transformer_mass
        partsCost = lss_cost + main_bearing_number * main_bearing_cost + gearbox_cost + hss_cost + generator_cost + bedplate_cost + yaw_system_cost + vs_cost + hvac_cost + cover_cost + elec_cost + controls_cost + other_cost + transformer_cost
        outputs['nacelle_cost'] = (1 + nacelle_transportMultiplier + nacelle_profitMultiplier) * ((1 + nacelle_overheadCostMultiplier + nacelle_assemblyCostMultiplier) * partsCost)

###### Tower
#-------------------------------------------------------------------------------
class TowerCost2015(ExplicitComponent):

    def setup(self):
      

        # variables
        self.add_input('tower_mass',            0.0, units='kg',     desc='tower mass')
        self.add_input('tower_mass_cost_coeff', 2.9, units='USD/kg', desc='tower mass-cost coeff') #mass-cost coeff with default from ppt
        self.add_input('tower_cost_external',   0.0, units='USD',    desc='Tower cost computed by TowerSE')
        
        # Outputs
        self.add_output('tower_parts_cost',     0.0, units='USD', desc='Overall wind turbine component capial costs excluding transportation costs')

    def compute(self, inputs, outputs):

        tower_mass = inputs['tower_mass']
        tower_mass_cost_coeff = inputs['tower_mass_cost_coeff']
        
        # calculate component cost
        if inputs['tower_cost_external'] < 1.:
            outputs['tower_parts_cost'] = tower_mass_cost_coeff * tower_mass
        else:
            outputs['tower_parts_cost'] = inputs['tower_cost_external']
        
        
#-------------------------------------------------------------------------------
class TowerCostAdder2015(ExplicitComponent):

    def setup(self):


        # variables
        self.add_input('tower_parts_cost', 0.0, units='USD', desc='component cost')
      
        # multipliers
        self.add_input('tower_assemblyCostMultiplier', 0.0, desc='tower assembly cost multiplier')
        self.add_input('tower_overheadCostMultiplier', 0.0, desc='tower overhead cost multiplier')
        self.add_input('tower_profitMultiplier', 0.0, desc='tower profit cost multiplier')
        self.add_input('tower_transportMultiplier', 0.0, desc='tower transport cost multiplier')
        
        # returns
        self.add_output('tower_cost', 0.0, units='USD', desc='component cost') 

    def compute(self, inputs, outputs):

        tower_parts_cost = inputs['tower_parts_cost']

        tower_assemblyCostMultiplier = inputs['tower_assemblyCostMultiplier']
        tower_overheadCostMultiplier = inputs['tower_overheadCostMultiplier']
        tower_profitMultiplier = inputs['tower_profitMultiplier']
        tower_transportMultiplier = inputs['tower_transportMultiplier']

        partsCost = tower_parts_cost
        outputs['tower_cost'] = (1 + tower_transportMultiplier + tower_profitMultiplier) * ((1 + tower_overheadCostMultiplier + tower_assemblyCostMultiplier) * partsCost)

#-------------------------------------------------------------------------------
class TurbineCostAdder2015(ExplicitComponent):

    def setup(self):


        # Variables
        self.add_input('rotor_cost',        0.0, units='USD',   desc='Rotor cost')
        self.add_input('rotor_mass_tcc',    0.0, units='kg',    desc='Rotor mass')
        self.add_input('nacelle_cost',      0.0, units='USD',   desc='Nacelle cost')
        self.add_input('nacelle_mass_tcc',      0.0, units='kg',    desc='Nacelle mass')
        self.add_input('tower_cost',        0.0, units='USD',   desc='Tower cost')
        self.add_input('tower_mass',        0.0, units='kg',    desc='Tower mass')
        self.add_input('machine_rating',    0.0, units='kW',    desc='Machine rating')
    
        # parameters
        self.add_input('turbine_assemblyCostMultiplier',    0.0, desc='Turbine multiplier for assembly cost in manufacturing')
        self.add_input('turbine_overheadCostMultiplier',    0.0, desc='Turbine multiplier for overhead')
        self.add_input('turbine_profitMultiplier',          0.0, desc='Turbine multiplier for profit markup')
        self.add_input('turbine_transportMultiplier',       0.0, desc='Turbine multiplier for transport costs')
    
        # Outputs
        self.add_output('turbine_mass_tcc',     0.0, units='kg',    desc='Turbine total mass, without foundation')
        self.add_output('turbine_cost',     0.0, units='USD',   desc='Overall wind turbine capital costs including transportation costs')
        self.add_output('turbine_cost_kW',  0.0, units='USD/kW',desc='Overall wind turbine capial costs including transportation costs')
        
    def compute(self, inputs, outputs):

        rotor_cost      = inputs['rotor_cost']
        nacelle_cost    = inputs['nacelle_cost']
        tower_cost      = inputs['tower_cost']
        
        rotor_mass_tcc  = inputs['rotor_mass_tcc']
        nacelle_mass_tcc= inputs['nacelle_mass_tcc']
        tower_mass      = inputs['tower_mass']
        
        turbine_assemblyCostMultiplier = inputs['turbine_assemblyCostMultiplier']
        turbine_overheadCostMultiplier = inputs['turbine_overheadCostMultiplier']
        turbine_profitMultiplier = inputs['turbine_profitMultiplier']
        turbine_transportMultiplier = inputs['turbine_transportMultiplier']

        partsCost = rotor_cost + nacelle_cost + tower_cost
        
        
        outputs['turbine_mass_tcc']    =  rotor_mass_tcc + nacelle_mass_tcc + tower_mass
        outputs['turbine_cost']    = (1 + turbine_transportMultiplier + turbine_profitMultiplier) * ((1 + turbine_overheadCostMultiplier + turbine_assemblyCostMultiplier) * partsCost)
        outputs['turbine_cost_kW'] = outputs['turbine_cost'] / inputs['machine_rating']

class Outputs2Screen(ExplicitComponent):
    def initialize(self):
        self.options.declare('verbosity', default=False)
        
    def setup(self):
        
        self.add_input('blade_cost',       0.0,  units='USD', desc='Overall wind turbine component capital costs excluding transportation costs')
        self.add_input('blade_mass',       0.0,  units='kg',  desc='Blade mass')
        self.add_input('hub_cost',         0.0,  units='USD', desc='Overall wind turbine component capital costs excluding transportation costs')
        self.add_input('hub_mass',         0.0,  units='kg',  desc='Hub mass')
        self.add_input('pitch_system_cost',       0.0,  units='USD', desc='Overall wind turbine component capital costs excluding transportation costs')
        self.add_input('pitch_system_mass',0.0,  units='kg',  desc='Pitch system mass')
        self.add_input('spinner_cost',     0.0,  units='USD', desc='Overall wind turbine component capital costs excluding transportation costs')
        self.add_input('spinner_mass',     0.0,  units='kg',  desc='Spinner mass')
        self.add_input('lss_cost',         0.0,  units='USD', desc='Overall wind turbine component capital costs excluding transportation costs')
        self.add_input('lss_mass',         0.0,  units='kg',  desc='LSS mass')
        self.add_input('main_bearing_cost',0.0,  units='USD', desc='Overall wind turbine component capital costs excluding transportation costs')
        self.add_input('main_bearing_mass',0.0,  units='kg',  desc='Main bearing mass')
        self.add_input('gearbox_cost',     0.0,  units='USD', desc='Overall wind turbine component capital costs excluding transportation costs')
        self.add_input('gearbox_mass',     0.0,  units='kg',  desc='LSS mass')
        self.add_input('hss_cost',         0.0,  units='USD', desc='Overall wind turbine component capital costs excluding transportation costs')
        self.add_input('hss_mass',         0.0,  units='kg',  desc='HSS mass')
        self.add_input('generator_cost',   0.0,  units='USD', desc='Overall wind turbine component capital costs excluding transportation costs')
        self.add_input('generator_mass',   0.0,  units='kg',  desc='Generator mass')
        self.add_input('bedplate_cost',    0.0,  units='USD', desc='Overall wind turbine component capital costs excluding transportation costs')
        self.add_input('bedplate_mass',    0.0,  units='kg',  desc='Bedplate mass')
        self.add_input('yaw_system_cost',  0.0,  units='USD', desc='Overall wind turbine component capital costs excluding transportation costs')
        self.add_input('yaw_mass',         0.0,  units='kg',  desc='Yaw system mass')
        self.add_input('hvac_cost',        0.0,  units='USD', desc='Overall wind turbine component capital costs excluding transportation costs')
        self.add_input('hvac_mass',        0.0,  units='kg',  desc='HVAC mass')
        self.add_input('cover_cost',       0.0,  units='USD', desc='Overall wind turbine component capital costs excluding transportation costs')
        self.add_input('cover_mass',       0.0,  units='kg',  desc='Cover mass')
        self.add_input('elec_cost',        0.0,  units='USD', desc='Overall wind turbine component capital costs excluding transportation costs')
        self.add_input('controls_cost',    0.0,  units='USD', desc='Overall wind turbine component capital costs excluding transportation costs')
        self.add_input('other_cost',       0.0,  units='USD', desc='Overall wind turbine component capital costs excluding transportation costs')
        self.add_input('transformer_cost', 0.0,  units='USD', desc='Overall wind turbine component capital costs excluding transportation costs')
        self.add_input('transformer_mass', 0.0,  units='kg',  desc='Transformer mass')
                                           
        self.add_input('rotor_cost',       0.0,  units='USD', desc='Overall wind turbine rotor capital costs')
        self.add_input('rotor_mass_tcc',   0.0,  units='kg',  desc='Rotor mass')
        self.add_input('nacelle_cost',     0.0,  units='USD', desc='Overall wind turbine nacelle capital costs')
        self.add_input('nacelle_mass_tcc', 0.0,  units='kg',  desc='Nacelle mass')
        self.add_input('tower_cost',       0.0,  units='USD', desc='Overall wind turbine tower capital costs')
        self.add_input('tower_mass',       0.0,  units='kg',  desc='Tower mass')
        self.add_input('turbine_cost',     0.0,  units='USD', desc='Overall wind turbine capital costs including transportation costs')
        self.add_input('turbine_cost_kW',  0.0,  units='USD/kW', desc='Overall wind turbine capital costs including transportation costs per kW')
        self.add_input('turbine_mass_tcc', 0.0,  units='kg',  desc='Turbine mass')
        
        
    def compute(self, inputs, outputs):        
        
        if self.options['verbosity'] == True:
        
            
            print('################################################')
            print('Computation of costs of the main turbine components from TurbineCostSE')
            print('Blade cost              %.3f k USD       mass %.3f kg' % (inputs['blade_cost'] * 1.e-003,        inputs['blade_mass']))
            print('Pitch system cost       %.3f k USD       mass %.3f kg' % (inputs['pitch_system_cost'] * 1.e-003, inputs['pitch_system_mass']))
            print('Hub cost                %.3f k USD       mass %.3f kg' % (inputs['hub_cost'] * 1.e-003,          inputs['hub_mass']))
            print('Spinner cost            %.3f k USD       mass %.3f kg' % (inputs['spinner_cost'] * 1.e-003,      inputs['spinner_mass']))
            print('------------------------------------------------')
            print('Rotor cost              %.3f k USD       mass %.3f kg' % (inputs['rotor_cost'] * 1.e-003,        inputs['rotor_mass_tcc']))
            print('')
            print('LSS cost                %.3f k USD       mass %.3f kg' % (inputs['lss_cost'] * 1.e-003,          inputs['lss_mass']))
            print('Main bearing cost       %.3f k USD       mass %.3f kg' % (inputs['main_bearing_cost'] * 1.e-003, inputs['main_bearing_mass']))
            print('Gearbox cost            %.3f k USD       mass %.3f kg' % (inputs['gearbox_cost'] * 1.e-003,      inputs['gearbox_mass']))
            print('HSS cost                %.3f k USD       mass %.3f kg' % (inputs['hss_cost'] * 1.e-003,          inputs['hss_mass']))
            print('Generator cost          %.3f k USD       mass %.3f kg' % (inputs['generator_cost'] * 1.e-003,    inputs['generator_mass']))
            print('Bedplate cost           %.3f k USD       mass %.3f kg' % (inputs['bedplate_cost'] * 1.e-003,     inputs['bedplate_mass']))
            print('Yaw system cost         %.3f k USD       mass %.3f kg' % (inputs['yaw_system_cost'] * 1.e-003,   inputs['yaw_mass']))
            print('HVAC cost               %.3f k USD       mass %.3f kg' % (inputs['hvac_cost'] * 1.e-003,         inputs['hvac_mass']))
            print('Nacelle cover cost      %.3f k USD       mass %.3f kg' % (inputs['cover_cost'] * 1.e-003,        inputs['cover_mass']))
            print('Electr connection cost  %.3f k USD'                    % (inputs['elec_cost'] * 1.e-003))
            print('Controls cost           %.3f k USD'                    % (inputs['controls_cost'] * 1.e-003))
            print('Other main frame cost   %.3f k USD'                    % (inputs['other_cost'] * 1.e-003))
            print('Transformer cost        %.3f k USD       mass %.3f kg' % (inputs['transformer_cost'] * 1.e-003,  inputs['transformer_mass']))
            print('------------------------------------------------')
            print('Nacelle cost            %.3f k USD       mass %.3f kg' % (inputs['nacelle_cost'] * 1.e-003,      inputs['nacelle_mass_tcc']))
            print('')
            print('Tower cost              %.3f k USD       mass %.3f kg' % (inputs['tower_cost'] * 1.e-003,        inputs['tower_mass']))
            print('------------------------------------------------')
            print('------------------------------------------------')
            print('Turbine cost            %.3f k USD       mass %.3f kg' % (inputs['turbine_cost'] * 1.e-003,      inputs['turbine_mass_tcc']))
            print('Turbine cost per kW     %.3f k USD/kW'                 % inputs['turbine_cost_kW'])
            print('################################################')
                
    

#-------------------------------------------------------------------------------
class Turbine_CostsSE_2015(Group):
    def initialize(self):
        self.options.declare('verbosity', default=False)
        self.options.declare('topLevelFlag', default=True)

    def setup(self):
        self.verbosity = self.options['verbosity']

        if self.options['topLevelFlag']:
            sharedIndeps = IndepVarComp()
            sharedIndeps.add_output('machine_rating',     units='kW', val=0.0)
            sharedIndeps.add_output('blade_mass',         units='kg', val=0.0)
            sharedIndeps.add_output('hub_mass',           units='kg', val=0.0)
            sharedIndeps.add_output('pitch_system_mass',  units='kg', val=0.0)
            sharedIndeps.add_output('spinner_mass',       units='kg', val=0.0)
            sharedIndeps.add_output('lss_mass',           units='kg', val=0.0)
            sharedIndeps.add_output('bearings_mass',      units='kg', val=0.0)
            sharedIndeps.add_output('gearbox_mass',       units='kg', val=0.0)
            sharedIndeps.add_output('main_bearing_mass',  units='kg', val=0.0)
            sharedIndeps.add_discrete_output('main_bearing_number',  val=0)
            sharedIndeps.add_output('hss_mass',           units='kg', val=0.0)
            sharedIndeps.add_output('generator_mass',     units='kg', val=0.0)
            sharedIndeps.add_output('bedplate_mass',      units='kg', val=0.0)
            sharedIndeps.add_output('yaw_mass',           units='kg', val=0.0)
            sharedIndeps.add_output('vs_electronics_mass',units='kg', val=0.0)
            sharedIndeps.add_output('hvac_mass',          units='kg', val=0.0)
            sharedIndeps.add_output('cover_mass',         units='kg', val=0.0)
            sharedIndeps.add_output('platforms_mass',     units='kg', val=0.0)
            sharedIndeps.add_output('transformer_mass',   units='kg', val=0.0)
            sharedIndeps.add_output('tower_mass',         units='kg', val=0.0)
            sharedIndeps.add_discrete_output('crane', val=False)
            self.add_subsystem('sharedIndeps', sharedIndeps, promotes=['*'])
        
        indeps = IndepVarComp()
        indeps.add_output('blade_mass_cost_coeff',         units='USD/kg', val=14.6)
        indeps.add_output('hub_mass_cost_coeff',           units='USD/kg', val=3.9)
        indeps.add_output('pitch_system_mass_cost_coeff',  units='USD/kg', val=22.1)
        indeps.add_output('spinner_mass_cost_coeff',       units='USD/kg', val=11.1)
        indeps.add_output('lss_mass_cost_coeff',           units='USD/kg', val=11.9)
        indeps.add_output('bearings_mass_cost_coeff',      units='USD/kg', val=4.5)
        indeps.add_output('gearbox_mass_cost_coeff',       units='USD/kg', val=12.9)
        indeps.add_output('hss_mass_cost_coeff',           units='USD/kg', val=6.8)
        indeps.add_output('generator_mass_cost_coeff',     units='USD/kg', val=12.4)
        indeps.add_output('bedplate_mass_cost_coeff',      units='USD/kg', val=2.9)
        indeps.add_output('yaw_mass_cost_coeff',           units='USD/kg', val=8.3)
        indeps.add_output('vs_electronics_mass_cost_coeff',units='USD/kg', val=18.8)
        indeps.add_output('hvac_mass_cost_coeff',          units='USD/kg', val=124.0)
        indeps.add_output('cover_mass_cost_coeff',         units='USD/kg', val=5.7)
        indeps.add_output('elec_connec_machine_rating_cost_coeff',units='USD/kW', val=41.85)
        indeps.add_output('platforms_mass_cost_coeff',     units='USD/kg', val=17.1)
        indeps.add_output('base_hardware_cost_coeff',      units='USD/kg', val=0.7)
        indeps.add_output('transformer_mass_cost_coeff',   units='USD/kg', val=18.8)
        indeps.add_output('tower_mass_cost_coeff',         units='USD/kg', val=2.9)
        indeps.add_output('controls_machine_rating_cost_coeff', units='USD/kW', val=21.15)
        indeps.add_output('crane_cost',                    units='USD', val=12e3)
        
        indeps.add_output('hub_assemblyCostMultiplier',    val=0.0)
        indeps.add_output('hub_overheadCostMultiplier',    val=0.0)
        indeps.add_output('nacelle_assemblyCostMultiplier',val=0.0)
        indeps.add_output('nacelle_overheadCostMultiplier',val=0.0)
        indeps.add_output('tower_assemblyCostMultiplier',  val=0.0)
        indeps.add_output('tower_overheadCostMultiplier',  val=0.0)
        indeps.add_output('turbine_assemblyCostMultiplier',val=0.0)
        indeps.add_output('turbine_overheadCostMultiplier',val=0.0)
        indeps.add_output('hub_profitMultiplier',          val=0.0)
        indeps.add_output('nacelle_profitMultiplier',      val=0.0)
        indeps.add_output('tower_profitMultiplier',        val=0.0)
        indeps.add_output('turbine_profitMultiplier',      val=0.0)
        indeps.add_output('hub_transportMultiplier',       val=0.0)
        indeps.add_output('nacelle_transportMultiplier',   val=0.0)
        indeps.add_output('tower_transportMultiplier',     val=0.0)
        indeps.add_output('turbine_transportMultiplier',   val=0.0)
        self.add_subsystem('indeps', indeps, promotes=['*'])
        
        self.add_subsystem('blade_c'       , BladeCost2015(),         promotes=['*'])
        self.add_subsystem('hub_c'         , HubCost2015(),           promotes=['*'])
        self.add_subsystem('pitch_c'       , PitchSystemCost2015(),   promotes=['*'])
        self.add_subsystem('spinner_c'     , SpinnerCost2015(),       promotes=['*'])
        self.add_subsystem('hub_adder'     , HubSystemCostAdder2015(),promotes=['*'])
        self.add_subsystem('rotor_adder'   , RotorCostAdder2015(),    promotes=['*'])
        self.add_subsystem('lss_c'         , LowSpeedShaftCost2015(), promotes=['*'])
        self.add_subsystem('bearing_c'     , BearingsCost2015(),      promotes=['*'])
        self.add_subsystem('gearbox_c'     , GearboxCost2015(),       promotes=['*'])
        self.add_subsystem('hss_c'         , HighSpeedSideCost2015(), promotes=['*'])
        self.add_subsystem('generator_c'   , GeneratorCost2015(),     promotes=['*'])
        self.add_subsystem('bedplate_c'    , BedplateCost2015(),      promotes=['*'])
        self.add_subsystem('yaw_c'         , YawSystemCost2015(),     promotes=['*'])
        self.add_subsystem('hvac_c'        , HydraulicCoolingCost2015(), promotes=['*'])
        self.add_subsystem('controls_c'    , ControlsCost2015(),      promotes=['*'])
        self.add_subsystem('vs_c'          , VariableSpeedElecCost2015(), promotes=['*'])
        self.add_subsystem('elec_c'        , ElecConnecCost2015(),    promotes=['*'])
        self.add_subsystem('cover_c'       , NacelleCoverCost2015(),  promotes=['*'])
        self.add_subsystem('other_c'       , OtherMainframeCost2015(),promotes=['*'])
        self.add_subsystem('transformer_c' , TransformerCost2015(),   promotes=['*'])
        self.add_subsystem('nacelle_adder' , NacelleSystemCostAdder2015(), promotes=['*'])
        self.add_subsystem('tower_c'       , TowerCost2015(),         promotes=['*'])
        self.add_subsystem('tower_adder'   , TowerCostAdder2015(),    promotes=['*'])
        self.add_subsystem('turbine_c'     , TurbineCostAdder2015(),  promotes=['*'])
        self.add_subsystem('outputs'       , Outputs2Screen(verbosity=self.verbosity), promotes=['*'])

#-------------------------------------------------------------------------------
def example():

    # simple test of module
    turbine = Turbine_CostsSE_2015(verbosity=True)
    prob = Problem(turbine)
    prob.setup()

    prob['blade_mass']          = 17650.67  # inline with the windpact estimates
    prob['hub_mass']            = 31644.5
    prob['pitch_system_mass']   = 17004.0
    prob['spinner_mass']        = 1810.5
    prob['lss_mass']            = 31257.3
    #bearingsMass'] = 9731.41
    prob['main_bearing_mass']   = 9731.41 / 2
    prob['gearbox_mass']        = 30237.60
    prob['hss_mass']            = 1492.45
    prob['generator_mass']      = 16699.85
    prob['bedplate_mass']       = 93090.6
    prob['yaw_mass']            = 11878.24
    prob['tower_mass']          = 434559.0
    prob['vs_electronics_mass'] = 1000.
    prob['hvac_mass']           = 1000.
    prob['cover_mass']          = 1000.
    prob['platforms_mass']      = 1000.
    prob['transformer_mass']    = 1000.

    # other inputs
    prob['machine_rating']      = 5000.0
    prob['blade_number']        = 3
    prob['crane']               = True
    prob['main_bearing_number'] = 2

    prob.run_model()

    #print('The results for the NREL 5 MW Reference Turbine in an offshore 20 m water depth location are')
    for io in turbine._outputs:
        print(io, str(turbine._outputs[io]))


if __name__ == "__main__":

    example()
