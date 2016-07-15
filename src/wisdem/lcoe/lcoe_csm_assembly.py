"""
LCOE_csm_ssembly.py
Created by NWTC Systems Engineering Sub-Task on 2012-08-01.
Copyright (c) NREL. All rights reserved.
"""
from openmdao.main.api import Component, Assembly, VariableTree
from openmdao.main.datatypes.api import Int, Bool, Float, Array, VarTree, Enum
import numpy as np

from fusedwind.plant_cost.fused_finance import ExtendedFinancialAnalysis, configure_extended_financial_analysis
from fusedwind.plant_cost.fused_bos_costs import BOSVarTree
from fusedwind.plant_cost.fused_opex import OPEXVarTree
from fusedwind.interface import implement_base

# NREL cost and scaling model sub-assemblies
from turbine_costsse.nrel_csm_tcc import tcc_csm_assembly
from plant_costsse.nrel_csm_bos.nrel_csm_bos import bos_csm_assembly
from plant_costsse.nrel_csm_opex.nrel_csm_opex  import opex_csm_assembly
from plant_financese.nrel_csm_fin.nrel_csm_fin import fin_csm_assembly
from plant_energyse.nrel_csm_aep.nrel_csm_aep import aep_csm_assembly

@implement_base(ExtendedFinancialAnalysis)
class lcoe_csm_assembly(Assembly):

    # Variables
    machine_rating = Float(units = 'kW', iotype='in', desc= 'rated machine power in kW', group='Global')
    rotor_diameter = Float(units = 'm', iotype='in', desc= 'rotor diameter of the machine', group='Global')
    max_tip_speed = Float(units = 'm/s', iotype='in', desc= 'maximum allowable tip speed for the rotor', group='Global')
    hub_height = Float(units = 'm', iotype='in', desc='hub height of wind turbine above ground / sea level', group='Global')
    sea_depth = Float(units = 'm', iotype='in', desc = 'sea depth for offshore wind project', group='Global')

    # Parameters
    drivetrain_design = Enum('geared', ('geared', 'single_stage', 'multi_drive', 'pm_direct_drive'), iotype='in', group='Global')
    altitude = Float(0.0, units = 'm', iotype='in', desc= 'altitude of wind plant', group='Plant_AEP')
    turbine_number = Int(100, iotype='in', desc = 'total number of wind turbines at the plant', group='Global')
    year = Int(2009, iotype='in', desc = 'year of project start', group='Plant_Finance')
    month = Int(12, iotype='in', desc = 'month of project start', group='Plant_Finance')
    # Extra AEP parameters
    max_power_coefficient = Float(0.488, iotype='in', desc= 'maximum power coefficient of rotor for operation in region 2', group='Aero')
    opt_tsr = Float(7.525, iotype='in', desc= 'optimum tip speed ratio for operation in region 2', group='Aero')
    cut_in_wind_speed = Float(3.0, units = 'm/s', iotype='in', desc= 'cut in wind speed for the wind turbine', group='Aero')
    cut_out_wind_speed = Float(25.0, units = 'm/s', iotype='in', desc= 'cut out wind speed for the wind turbine', group='Aero')
    shear_exponent = Float(0.1, iotype='in', desc= 'shear exponent for wind plant', group='Plant_AEP')
    wind_speed_50m = Float(8.35, units = 'm/s', iotype='in', desc='mean annual wind speed at 50 m height', group='Plant_AEP')
    weibull_k= Float(2.1, iotype='in', desc = 'weibull shape factor for annual wind speed distribution', group='Plant_AEP')
    soiling_losses = Float(0.0, iotype='in', desc = 'energy losses due to blade soiling for the wind plant - average across turbines', group='Plant_AEP')
    array_losses = Float(0.06, iotype='in', desc = 'energy losses due to turbine interactions - across entire plant', group='Plant_AEP')
    availability = Float(0.94287630736, iotype='in', desc = 'average annual availbility of wind turbines at plant', group='Plant_AEP')
    thrust_coefficient = Float(0.50, iotype='in', desc='thrust coefficient at rated power', group='Plant_AEP')
    max_efficiency = Float(0.90, iotype='in', desc = 'maximum efficiency of rotor and drivetrain - at rated power', group='Plant_AEP') # TODO: should come from drivetrain
    # Extra TCC parameters
    blade_number = Int(3, iotype='in', desc = 'number of rotor blades', group='Turbine_Cost')
    offshore = Bool(True, iotype='in', desc = 'boolean for offshore', group='Global')
    advanced_blade = Bool(False, iotype='in', desc = 'boolean for use of advanced blade curve', group='Turbine_Cost')
    crane = Bool(True, iotype='in', desc = 'boolean for presence of a service crane up tower', group='Turbine_Cost')
    advanced_bedplate = Int(0, iotype='in', desc= 'indicator for drivetrain bedplate design 0 - conventional', group='Turbine_Cost')
    advanced_tower = Bool(False, iotype='in', desc = 'advanced tower configuration', group='Turbine_Cost')
    # Extra Finance parameters
    fixed_charge_rate = Float(0.12, iotype = 'in', desc = 'fixed charge rate for coe calculation', group='Plant_Finance')
    construction_finance_rate = Float(0.00, iotype='in', desc = 'construction financing rate applied to overnight capital costs', group='Plant_Finance')
    tax_rate = Float(0.4, iotype = 'in', desc = 'tax rate applied to operations', group='Plant_Finance')
    discount_rate = Float(0.07, iotype = 'in', desc = 'applicable project discount rate', group='Plant_Finance')
    construction_time = Float(1.0, iotype = 'in', desc = 'number of years to complete project construction', group='Plant_Finance')
    project_lifetime = Float(20.0, iotype = 'in', desc = 'project lifetime for LCOE calculation', group='Plant_Finance')

    #Outputs
    turbine_cost = Float(iotype='out', desc = 'A Wind Turbine Capital _cost')
    bos_costs = Float(iotype='out', desc='A Wind Plant Balance of Station _cost Model')
    avg_annual_opex = Float(iotype='out', desc='A Wind Plant Operations Expenditures Model')
    net_aep = Float(iotype='out', desc='A Wind Plant Annual Energy Production Model', units='kW*h')
    coe = Float(iotype='out', desc='Levelized cost of energy for the wind plant')
    opex_breakdown = VarTree(OPEXVarTree(),iotype='out')
    bos_breakdown = VarTree(BOSVarTree(), iotype='out', desc='BOS cost breakdown')
    #AEP outputs
    rated_wind_speed = Float(11.506, units = 'm / s', iotype='out', desc='wind speed for rated power')
    rated_rotor_speed = Float(12.126, units = 'rpm', iotype='out', desc = 'rotor speed at rated power')
    rotor_thrust = Float(iotype='out', units='N', desc='maximum thrust from rotor')
    rotor_torque = Float(iotype='out', units='N * m', desc = 'torque from rotor at rated power')
    power_curve = Array(np.array([[4.0,80.0],[25.0, 5000.0]]), iotype='out', desc = 'power curve for a particular rotor')
    #max_efficiency = Float(0.902, iotype='out', desc = 'maximum efficiency of rotor and drivetrain - at rated power')
    gross_aep = Float(0.0, iotype='out', desc='Gross Annual Energy Production before availability and loss impacts', unit='kWh')
    capacity_factor = Float(iotype='out',desc='plant capacity factor')
    #TCC outputs
    turbine_mass = Float(0.0, units='kg', iotype='out', desc='turbine mass')
    #Finance outputs
    lcoe = Float(iotype='out', desc='_cost of energy - unlevelized')

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
        self.connect('max_power_coefficient', 'aep_a.max_power_coefficient')
        self.connect('opt_tsr','aep_a.opt_tsr')
        self.connect('cut_in_wind_speed','aep_a.cut_in_wind_speed')
        self.connect('cut_out_wind_speed','aep_a.cut_out_wind_speed')
        self.connect('altitude','aep_a.altitude')
        self.connect('shear_exponent','aep_a.shear_exponent')
        self.connect('wind_speed_50m','aep_a.wind_speed_50m')
        self.connect('weibull_k','aep_a.weibull_k')
        self.connect('soiling_losses','aep_a.soiling_losses')
        self.connect('array_losses','aep_a.array_losses')
        self.connect('availability','aep_a.availability')
        self.connect('thrust_coefficient','aep_a.thrust_coefficient')
        self.connect('max_efficiency', 'aep_a.max_efficiency')
        self.connect('blade_number','tcc_a.blade_number')
        self.connect('advanced_blade','tcc_a.advanced_blade')
        # drivetrain
        self.connect('machine_rating', ['aep_a.machine_rating', 'tcc_a.machine_rating', 'bos_a.machine_rating', 'opex_a.machine_rating'])
        self.connect('drivetrain_design', ['aep_a.drivetrain_design', 'tcc_a.drivetrain_design'])
        self.connect('crane','tcc_a.crane')
        self.connect('advanced_bedplate','tcc_a.advanced_bedplate')
        # tower
        self.connect('hub_height', ['aep_a.hub_height', 'tcc_a.hub_height', 'bos_a.hub_height'])
        self.connect('advanced_tower','tcc_a.advanced_tower')
        # plant configuration
        # climate
        self.connect('sea_depth', ['bos_a.sea_depth', 'opex_a.sea_depth', 'fin_a.sea_depth'])
        self.connect('offshore','tcc_a.offshore')
        # plant operation
        self.connect('turbine_number', ['aep_a.turbine_number', 'bos_a.turbine_number', 'opex_a.turbine_number'])
        # financial
        self.connect('year', ['tcc_a.year', 'bos_a.year', 'opex_a.year'])
        self.connect('month', ['tcc_a.month', 'bos_a.month', 'opex_a.month'])
        self.connect('fixed_charge_rate','fin_a.fixed_charge_rate')
        self.connect('construction_finance_rate','fin_a.construction_finance_rate')
        self.connect('tax_rate','fin_a.tax_rate')
        self.connect('discount_rate','fin_a.discount_rate')
        self.connect('construction_time','fin_a.construction_time')
        self.connect('project_lifetime','fin_a.project_lifetime')

        # connections
        self.connect('aep_a.rotor_thrust','tcc_a.rotor_thrust')
        self.connect('aep_a.rotor_torque','tcc_a.rotor_torque')
        self.connect('aep_a.net_aep', ['opex_a.net_aep'])
        self.connect('tcc_a.turbine_cost','bos_a.turbine_cost')

        # create passthroughs for key output variables of interest
        # aep_a
        self.connect('aep_a.rated_rotor_speed','rated_rotor_speed')
        self.connect('aep_a.rated_wind_speed','rated_wind_speed')
        self.connect('aep_a.rotor_thrust','rotor_thrust')
        self.connect('aep_a.rotor_torque','rotor_torque')
        self.connect('aep_a.power_curve','power_curve')
        #self.connect('aep_a.max_efficiency','max_efficiency')
        self.connect('aep_a.gross_aep','gross_aep')
        self.connect('aep_a.capacity_factor','capacity_factor')
        # tcc_a
        self.connect('tcc_a.turbine_mass','turbine_mass')
        # fin_a
        self.connect('fin_a.lcoe','lcoe')

    # Pie Chart for CAPEX
    def plot(self, fig):
        from plot_capex import plot_capex
        fig = plot_capex(self)
        return fig

    # Waterfall Plot
    def lcoe_plot(self, fig):

        from plot_lcoe_csm import plot_lcoe
        from bokeh.io import show, output_file
        fig = plot_lcoe(self)
        output_file('bokeh plot', title='Bokeh Plot')

        #show(fig)
        return fig


def example():

    lcoe = lcoe_csm_assembly()

    lcoe.machine_rating = 5000.0 # Float(units = 'kW', iotype='in', desc= 'rated machine power in kW')
    lcoe.rotor_diameter = 126.0 # Float(units = 'm', iotype='in', desc= 'rotor diameter of the machine')
    lcoe.max_tip_speed = 80.0 # Float(units = 'm/s', iotype='in', desc= 'maximum allowable tip speed for the rotor')
    lcoe.hub_height = 90.0 # Float(units = 'm', iotype='in', desc='hub height of wind turbine above ground / sea level')
    lcoe.sea_depth = 20.0 # Float(units = 'm', iotype='in', desc = 'sea depth for offshore wind project')

    # Parameters
    lcoe.drivetrain_design = 'geared' # Enum('geared', ('geared', 'single_stage', 'multi_drive', 'pm_direct_drive'), iotype='in')
    lcoe.altitude = 0.0 # Float(0.0, units = 'm', iotype='in', desc= 'altitude of wind plant')
    lcoe.turbine_number = 100 # Int(100, iotype='in', desc = 'total number of wind turbines at the plant')
    lcoe.year = 2009 # Int(2009, iotype='in', desc = 'year of project start')
    lcoe.month = 12 # Int(12, iotype='in', desc = 'month of project start')

    # Extra AEP inputs
    lcoe.max_power_coefficient = 0.488 #Float(0.488, iotype='in', desc= 'maximum power coefficient of rotor for operation in region 2')
    lcoe.opt_tsr = 7.525 #Float(7.525, iotype='in', desc= 'optimum tip speed ratio for operation in region 2')
    lcoe.cut_in_wind_speed = 3.0 #Float(3.0, units = 'm/s', iotype='in', desc= 'cut in wind speed for the wind turbine')
    lcoe.cut_out_wind_speed = 25.0 #Float(25.0, units = 'm/s', iotype='in', desc= 'cut out wind speed for the wind turbine')
    lcoe.hub_height = 90.0 #Float(90.0, units = 'm', iotype='in', desc= 'hub height of wind turbine above ground / sea level')
    lcoe.altitude = 0.0 #Float(0.0, units = 'm', iotype='in', desc= 'altitude of wind plant')
    #lcoe.air_density = Float(0.0, units = 'kg / (m * m * m)', iotype='in', desc= 'air density at wind plant site')  # default air density value is 0.0 - forces aero csm to calculate air density in model
    lcoe.drivetrain_design = 'geared' #Enum('geared', ('geared', 'single_stage', 'multi_drive', 'pm_direct_drive'), iotype='in')
    lcoe.shear_exponent = 0.1 #Float(0.1, iotype='in', desc= 'shear exponent for wind plant') #TODO - could use wind model here
    lcoe.wind_speed_50m = 8.02 #Float(8.35, units = 'm/s', iotype='in', desc='mean annual wind speed at 50 m height')
    lcoe.weibull_k= 2.15 #Float(2.1, iotype='in', desc = 'weibull shape factor for annual wind speed distribution')
    lcoe.soiling_losses = 0.0 #Float(0.0, iotype='in', desc = 'energy losses due to blade soiling for the wind plant - average across turbines')
    lcoe.array_losses = 0.10 #Float(0.06, iotype='in', desc = 'energy losses due to turbine interactions - across entire plant')
    lcoe.availability = 0.941 #Float(0.94287630736, iotype='in', desc = 'average annual availbility of wind turbines at plant')
    lcoe.turbine_number = 100 #Int(100, iotype='in', desc = 'total number of wind turbines at the plant')
    lcoe.thrust_coefficient = 0.50 #Float(0.50, iotype='in', desc='thrust coefficient at rated power')
    lcoe.max_efficiency = 0.902

    # Extra TCC inputs
    lcoe.blade_number = 3 #Int(3, iotype='in', desc = 'number of rotor blades')
    lcoe.offshore = True #Bool(True, iotype='in', desc = 'boolean for offshore')
    lcoe.advanced_blade = True #Bool(False, iotype='in', desc = 'boolean for use of advanced blade curve')
    lcoe.crane = True #Bool(True, iotype='in', desc = 'boolean for presence of a service crane up tower')
    lcoe.advanced_bedplate = 0 #Int(0, iotype='in', desc= 'indicator for drivetrain bedplate design 0 - conventional')
    lcoe.advanced_tower = False #Bool(False, iotype='in', desc = 'advanced tower configuration')

    # Extra Finance inputs
    lcoe.fixed_charge_rate = 0.12 #Float(0.12, iotype = 'in', desc = 'fixed charge rate for coe calculation')
    lcoe.construction_finance_rate = 0.00 #Float(0.00, iotype='in', desc = 'construction financing rate applied to overnight capital costs')
    lcoe.tax_rate = 0.4 #Float(0.4, iotype = 'in', desc = 'tax rate applied to operations')
    lcoe.discount_rate = 0.07 #Float(0.07, iotype = 'in', desc = 'applicable project discount rate')
    lcoe.construction_time = 1.0 #Float(1.0, iotype = 'in', desc = 'number of years to complete project construction')
    lcoe.project_lifetime = 20.0 #Float(20.0, iotype = 'in', desc = 'project lifetime for LCOE calculation')

    lcoe.run()
    
    print "Cost of Energy results for a 500 MW offshore wind farm using the NREL 5 MW reference turbine"
    print "LCOE: ${0:.4f} USD/kWh".format(lcoe.lcoe)
    print "COE: ${0:.4f} USD/kWh".format(lcoe.coe)
    print
    print "AEP per turbine: {0:1f} kWh/turbine".format(lcoe.net_aep / lcoe.turbine_number)
    print "Turbine Cost: ${0:2f} USD".format(lcoe.turbine_cost)
    print "BOS costs per turbine: ${0:2f} USD/turbine".format(lcoe.bos_costs / lcoe.turbine_number)
    print "OPEX per turbine: ${0:2f} USD/turbine".format(lcoe.avg_annual_opex / lcoe.turbine_number)
    from bokeh.plotting import figure
    from bokeh.io import output_file, show
    fig = figure()
    lcoe.plot(fig)

if __name__=="__main__":

    example()
