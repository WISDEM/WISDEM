"""
LCOE_csm_ssembly.py

Created by NWTC Systems Engineering Sub-Task on 2012-08-01.
Copyright (c) NREL. All rights reserved.
"""

from openmdao.main.api import Component, Assembly, VariableTree
from openmdao.main.datatypes.api import Int, Bool, Float, Array, VarTree, Enum

from fusedwind.plant_cost.fused_finance import ExtendedFinancialAnalysis, configure_extended_financial_analysis
from fusedwind.plant_cost.fused_bos_costs import BOSVarTree
from fusedwind.plant_cost.fused_opex import OPEXVarTree
from fusedwind.interface import implement_base

# NREL cost and scaling model sub-assemblies
from nrel_csm_tcc.nrel_csm_tcc import tcc_csm_assembly
from nrel_csm_bos.nrel_csm_bos import bos_csm_assembly
from nrel_csm_opex.nrel_csm_opex  import opex_csm_assembly
from nrel_csm_fin.nrel_csm_fin import fin_csm_assembly
from nrel_csm_aep.nrel_csm_aep import aep_csm_assembly

@implement_base(ExtendedFinancialAnalysis)
class lcoe_csm_assembly(Assembly):

    # Variables
    turbine_number = Int(iotype = 'in', desc = 'number of turbines at plant')
    machine_rating = Float(5000.0, units = 'kW', iotype='in', desc= 'rated machine power in kW')
    rotor_diameter = Float(126.0, units = 'm', iotype='in', desc= 'rotor diameter of the machine')
    max_tip_speed = Float(80.0, units = 'm/s', iotype='in', desc= 'maximum allowable tip speed for the rotor')
    hub_height = Float(90.0, units = 'm', iotype='in', desc='hub height of wind turbine above ground / sea level')
    sea_depth = Float(20.0, units = 'm', iotype='in', desc = 'sea depth for offshore wind project')

    # Parameters
    drivetrain_design = Enum('geared', ('geared', 'single_stage', 'multi_drive', 'pm_direct_drive'), iotype='in')
    altitude = Float(0.0, units = 'm', iotype='in', desc= 'altitude of wind plant')
    turbine_number = Int(100, iotype='in', desc = 'total number of wind turbines at the plant')
    year = Int(2009, iotype='in', desc = 'year of project start')
    month = Int(12, iotype='in', desc = 'month of project start')

    #Outputs
    turbine_cost = Float(iotype='out', desc = 'A Wind Turbine Capital _cost')
    bos_costs = Float(iotype='out', desc='A Wind Plant Balance of Station _cost Model')
    avg_annual_opex = Float(iotype='out', desc='A Wind Plant Operations Expenditures Model')
    net_aep = Float(iotype='out', desc='A Wind Plant Annual Energy Production Model', units='kW*h')
    coe = Float(iotype='out', desc='Levelized cost of energy for the wind plant')
    opex_breakdown = VarTree(OPEXVarTree(),iotype='out')
    bos_breakdown = VarTree(BOSVarTree(), iotype='out', desc='BOS cost breakdown')
                
    def configure(self):
        
        configure_extended_financial_analysis(self)
        
        self.replace('tcc_a', tcc_csm_assembly())
        self.replace('bos_a', bos_csm_assembly())
        self.replace('opex_a', opex_csm_assembly())
        self.replace('aep_a', aep_csm_assembly())
        self.replace('fin_a', fin_csm_assembly())

        # connect i/o to component and assembly inputs
        # turbine configuration
        # rotor
        self.connect('rotor_diameter', ['aep_a.rotor_diameter', 'tcc_a.rotor_diameter', 'bos_a.rotor_diameter'])
        self.connect('max_tip_speed', ['aep_a.max_tip_speed'])
        # drivetrain
        self.connect('machine_rating', ['aep_a.machine_rating', 'tcc_a.machine_rating', 'bos_a.machine_rating', 'opex_a.machine_rating'])
        self.connect('drivetrain_design', ['aep_a.drivetrain_design', 'tcc_a.drivetrain_design'])
        # tower
        self.connect('hub_height', ['aep_a.hub_height', 'tcc_a.hub_height', 'bos_a.hub_height'])
        # plant configuration
        # climate
        self.connect('altitude', ['aep_a.altitude'])
        self.connect('sea_depth', ['bos_a.sea_depth', 'opex_a.sea_depth', 'fin_a.sea_depth'])
        # plant operation       
        self.connect('turbine_number', ['aep_a.turbine_number', 'bos_a.turbine_number', 'opex_a.turbine_number']) 
        # financial
        self.connect('year', ['tcc_a.year', 'bos_a.year', 'opex_a.year'])
        self.connect('month', ['tcc_a.month', 'bos_a.month', 'opex_a.month'])
        self.connect('aep_a.net_aep', ['opex_a.net_aep'])

        # create passthroughs for key input variables of interest
        # turbine
        self.create_passthrough('tcc_a.blade_number')
        self.create_passthrough('tcc_a.advanced_blade')
        self.create_passthrough('aep_a.max_power_coefficient')
        self.create_passthrough('aep_a.opt_tsr')
        self.create_passthrough('aep_a.cut_in_wind_speed')
        self.create_passthrough('aep_a.cut_out_wind_speed')
        self.create_passthrough('tcc_a.crane')
        self.create_passthrough('tcc_a.advanced_bedplate')
        # plant
        self.create_passthrough('aep_a.shear_exponent')
        self.create_passthrough('aep_a.weibull_k')
        self.create_passthrough('aep_a.air_density')
        self.create_passthrough('aep_a.soiling_losses')
        self.create_passthrough('aep_a.array_losses')
        self.create_passthrough('aep_a.availability')
        self.create_passthrough('fin_a.fixed_charge_rate')
        self.create_passthrough('fin_a.construction_time')
        self.create_passthrough('fin_a.project_lifetime')
 
        # create passthroughs for key output variables of interest
        # aep_a
        self.create_passthrough('aep_a.rated_rotor_speed')
        self.create_passthrough('aep_a.rated_wind_speed')
        self.create_passthrough('aep_a.power_curve')
        # tcc_a
        self.create_passthrough('tcc_a.turbine_mass')
        # fin_a
        self.create_passthrough('fin_a.lcoe')


def example():

    lcoe = lcoe_csm_assembly()

    lcoe.run()
    
    print "LCOE: {0}".format(lcoe.lcoe)
    print "COE: {0}".format(lcoe.coe)
    print "\n"
    print "AEP per turbine: {0}".format(lcoe.net_aep / lcoe.turbine_number)
    print "Turbine _cost: {0}".format(lcoe.turbine_cost)
    print "BOS costs per turbine: {0}".format(lcoe.bos_costs / lcoe.turbine_number)
    print "OnM costs per turbine: {0}".format(lcoe.avg_annual_opex / lcoe.turbine_number)


if __name__=="__main__":

    example()