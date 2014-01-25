"""
LCOE_csm_ssembly.py

Created by NWTC Systems Engineering Sub-Task on 2012-08-01.
Copyright (c) NREL. All rights reserved.
"""

from openmdao.main.api import Component, Assembly, VariableTree
from openmdao.main.datatypes.api import Int, Bool, Float, Array, VarTree

from fusedwind.plant_cost.fused_fin_asym import ExtendedFinancialAnalysis
from fusedwind.vartrees.varTrees import Turbine

# NREL cost and scaling model sub-assemblies
from Turbine_CostsSE.Turbine_CostsSE.Turbine_CostsSE import Turbine_CostsSE
#from TurbineSE.turbine import TurbineSE
from TurbineSE.turbineSEdummy import TurbineSE # Temporary fix till TurbineSE works on windows
from Plant_CostsSE.Plant_BOS.NREL_CSM_BOS.nrel_csm_bos import bos_csm_assembly
from Plant_CostsSE.Plant_OM.NREL_CSM_OM.nrel_csm_om  import om_csm_assembly
from Plant_FinanceSE.NREL_CSM_FIN.nrel_csm_fin import fin_csm_assembly
from Plant_AEPSE.Basic_AEP.basic_aep import aep_weibull_assembly


class lcoe_se_csm_assembly(ExtendedFinancialAnalysis):

    # variables
    blade_number = Int(iotype='in', desc='number of rotor blades')
    #machine_rating = Float(5000.0, units = 'kW', iotype='in', desc= 'rated machine power in kW')
    #rotor_diameter = Float(126.0, units = 'm', iotype='in', desc= 'rotor diameter of the machine')
    #max_tip_speed = Float(80.0, units = 'm/s', iotype='in', desc= 'maximum allowable tip speed for the rotor')
    #hub_height = Float(90.0, units = 'm', iotype='in', desc='hub height of wind turbine above ground / sea level') #TODO: tie-in with turbine model
    sea_depth = Float(0.0, units = 'm', iotype='in', desc = 'sea depth for offshore wind project')

    # parameters
    drivetrain_design = Int(1, iotype='in', desc= 'drivetrain design type 1 = 3-stage geared, 2 = single-stage geared, 3 = multi-generator, 4 = direct drive')
    turbine_number = Int(100, iotype='in', desc = 'total number of wind turbines at the plant')
    year = Int(2009, iotype='in', desc = 'year of project start')
    month = Int(12, iotype='in', desc = 'month of project start')
                
    def configure(self):
        
        super(lcoe_se_csm_assembly,self).configure()
        
        self.replace('tcc_a', Turbine_CostsSE())
        self.replace('bos_a', bos_csm_assembly())
        self.replace('opex_a', om_csm_assembly())
        self.replace('aep_a', aep_weibull_assembly())
        self.replace('fin_a', fin_csm_assembly())
        
        self.add('trb', TurbineSE())
        
        self.driver.workflow.add('trb')

        # connect i/o to component and assembly inputs
        # turbine configuration
        # rotor
        self.connect('blade_number', ['trb.blade_number', 'tcc_a.blade_number'])
        self.connect('trb.rotor_diameter', 'bos_a.rotor_diameter')
        # drivetrain
        self.connect('trb.machine_rating', ['bos_a.machine_rating', 'opex_a.machine_rating'])
        self.connect('drivetrain_design', ['trb.drivetrain_design', 'tcc_a.drivetrain_design'])
        # tower
        self.connect('trb.hub_height', ['bos_a.hub_height'])
        # plant configuration
        # climate
        self.connect('sea_depth', ['bos_a.sea_depth', 'opex_a.sea_depth', 'fin_a.sea_depth'])
        # plant operation       
        self.connect('turbine_number', ['aep_a.turbine_number', 'bos_a.turbine_number', 'opex_a.turbine_number', 'fin_a.turbine_number']) 
        # financial
        self.connect('year', ['tcc_a.year', 'bos_a.year', 'opex_a.year'])
        self.connect('month', ['tcc_a.month', 'bos_a.month', 'opex_a.month'])

        # inter-model connections
        self.connect('trb.wind_curve', 'aep_a.wind_curve')
        self.connect('trb.power_curve', 'aep_a.power_curve')
        self.connect('trb.blade_mass', 'tcc_a.blade_mass')
        self.connect('trb.hub_mass', 'tcc_a.hub_mass')
        self.connect('trb.pitch_system_mass', 'tcc_a.pitch_system_mass')
        self.connect('trb.spinner_mass', 'tcc_a.spinner_mass')
        self.connect('trb.low_speed_shaft_mass', 'tcc_a.low_speed_shaft_mass')
        self.connect('trb.main_bearing_mass', 'tcc_a.main_bearing_mass')
        self.connect('trb.second_bearing_mass', 'tcc_a.second_bearing_mass')
        self.connect('trb.gearbox_mass', 'tcc_a.gearbox_mass')
        self.connect('trb.high_speed_side_mass', 'tcc_a.high_speed_side_mass')
        self.connect('trb.generator_mass', 'tcc_a.generator_mass')
        self.connect('trb.bedplate_mass', 'tcc_a.bedplate_mass')
        self.connect('trb.yaw_system_mass', 'tcc_a.yaw_system_mass')
        self.connect('aep_a.net_aep', ['opex_a.net_aep'])

        # create passthroughs for key input variables of interest
        # turbine
        self.create_passthrough('tcc_a.advanced_blade')
        self.connect('trb.crane', 'tcc_a.crane') 
        self.create_passthrough('tcc_a.offshore') # todo connections
        # plant
        self.create_passthrough('aep_a.A')
        self.create_passthrough('aep_a.k')
        self.create_passthrough('aep_a.other_losses')
        self.create_passthrough('aep_a.array_losses')
        self.create_passthrough('aep_a.availability')
        self.create_passthrough('fin_a.fixed_charge_rate')
 
        # create passthroughs for key output variables of interest
        # turb_a
        self.create_passthrough('trb.power_curve')
        self.create_passthrough('trb.turbine_mass')
        # tcc_a
        #self.create_passthrough('tcc_a.turbineVT')


def example():

    lcoe = lcoe_se_csm_assembly()

    lcoe.trb.crane = True
    lcoe.trb.gear_configuration = 'eep'
    lcoe.trb.gear_ratio = 97.0
    lcoe.drivetrain_design = 1
    lcoe.offshore = False
    lcoe.A = 8.35
    lcoe.k = 2.15

    lcoe.execute()

    print "COE: {0}".format(lcoe.coe)
    print "\n"
    print "AEP per turbine: {0}".format(lcoe.net_aep / lcoe.turbine_number)
    print "Turbine _cost: {0}".format(lcoe.turbine_cost)
    print "BOS costs per turbine: {0}".format(lcoe.bos_costs / lcoe.turbine_number)
    print "OnM costs per turbine: {0}".format(lcoe.avg_annual_opex / lcoe.turbine_number)

    #print "Turbine variable tree:"
    #lcoe.turbineVT.printVT()
    #print


if __name__=="__main__":

    example()