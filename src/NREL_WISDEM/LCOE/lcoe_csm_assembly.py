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
from Turbine_CostsSE.NREL_CSM_TCC.nrel_csm_tcc import tcc_csm_assembly
from Plant_CostsSE.Plant_BOS.NREL_CSM_BOS.nrel_csm_bos import bos_csm_assembly
from Plant_CostsSE.Plant_OM.NREL_CSM_OM.nrel_csm_om  import om_csm_assembly
from Plant_FinanceSE.NREL_CSM_FIN.nrel_csm_fin import fin_csm_assembly
from Plant_AEPSE.NREL_CSM_AEP.nrel_csm_aep import aep_csm_assembly

from NREL_CSM.csmDriveEfficiency import DrivetrainEfficiencyModel, csmDriveEfficiency

class lcoe_csm_assembly(ExtendedFinancialAnalysis):

    # variables
    machine_rating = Float(5000.0, units = 'kW', iotype='in', desc= 'rated machine power in kW')
    rotor_diameter = Float(126.0, units = 'm', iotype='in', desc= 'rotor diameter of the machine')
    max_tip_speed = Float(80.0, units = 'm/s', iotype='in', desc= 'maximum allowable tip speed for the rotor')
    hub_height = Float(90.0, units = 'm', iotype='in', desc='hub height of wind turbine above ground / sea level')
    sea_depth = Float(20.0, units = 'm', iotype='in', desc = 'sea depth for offshore wind project')

    # parameters
    drivetrain_design = Int(1, iotype='in', desc= 'drivetrain design type 1 = 3-stage geared, 2 = single-stage geared, 3 = multi-generator, 4 = direct drive')
    altitude = Float(0.0, units = 'm', iotype='in', desc= 'altitude of wind plant')
    turbine_number = Int(100, iotype='in', desc = 'total number of wind turbines at the plant')
    year = Int(2009, units = 'yr', iotype='in', desc = 'year of project start')
    month = Int(12, units = 'mon', iotype='in', desc = 'month of project start')
                
    def configure(self):
        
        super(lcoe_csm_assembly,self).configure()
        
        self.replace('tcc_a', tcc_csm_assembly())
        self.replace('bos_a', bos_csm_assembly())
        self.replace('opex_a', om_csm_assembly())
        self.replace('aep_a', aep_csm_assembly())
        self.replace('fin_a', fin_csm_assembly())

        # connect i/o to component and assembly inputs
        # turbine configuration
        # rotor
        self.connect('rotor_diameter', ['aep_a.rotor_diameter', 'tcc_a.rotor_diameter', 'bos_a.rotor_diameter'])
        self.connect('max_tip_speed', ['aep_a.max_tip_speed', 'tcc_a.max_tip_speed'])
        self.connect('aep_a.rated_wind_speed', 'tcc_a.rated_wind_speed')
        self.connect('aep_a.max_efficiency', 'tcc_a.max_efficiency')
        # drivetrain
        self.connect('machine_rating', ['aep_a.machine_rating', 'tcc_a.machine_rating', 'bos_a.machine_rating', 'opex_a.machine_rating', 'fin_a.machine_rating'])
        self.connect('drivetrain_design', ['aep_a.drivetrain_design', 'tcc_a.drivetrain_design'])
        # tower
        self.connect('hub_height', ['aep_a.hub_height', 'tcc_a.hub_height', 'bos_a.hub_height'])   
        # plant configuration
        # climate
        self.connect('altitude', ['aep_a.altitude', 'tcc_a.altitude'])
        self.connect('sea_depth', ['tcc_a.sea_depth', 'bos_a.sea_depth', 'opex_a.sea_depth', 'fin_a.sea_depth'])
        # plant operation       
        self.connect('turbine_number', ['aep_a.turbine_number', 'bos_a.turbine_number', 'opex_a.turbine_number', 'fin_a.turbine_number']) 
        # financial
        self.connect('year', ['tcc_a.year', 'bos_a.year', 'opex_a.year'])
        self.connect('month', ['tcc_a.month', 'bos_a.month', 'opex_a.month'])
        self.connect('aep_a.net_aep', ['opex_a.net_aep'])
        self.connect('opex_a.OPEX_breakdown.preventative_opex', 'fin_a.preventative_opex')
        self.connect('opex_a.OPEX_breakdown.corrective_opex', 'fin_a.corrective_opex')
        self.connect('opex_a.OPEX_breakdown.lease_opex', 'fin_a.lease_opex')

        # create passthroughs for key input variables of interest
        # turbine
        self.create_passthrough('tcc_a.blade_number')
        self.create_passthrough('tcc_a.advanced_blade')
        self.create_passthrough('tcc_a.thrust_coefficient')
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
        self.create_passthrough('aep_a.aep_per_turbine')
        # tcc_a
        self.create_passthrough('tcc_a.turbine_mass')
        self.create_passthrough('tcc_a.turbineVT')
        # fin_a
        self.create_passthrough('fin_a.lcoe')



def example():

    lcoe = lcoe_csm_assembly()
    
    lcoe.advanced_blade = True
    lcoe.aep_a.drive.drivetrain = csmDriveEfficiency(1)
    lcoe.machine_rating = 5000.001
    lcoe.sea_depth = 20.0001
    
    lcoe.execute()
    
    print "LCOE: {0}".format(lcoe.lcoe)
    print "COE: {0}".format(lcoe.coe)
    print "\n"
    print "AEP per turbine: {0}".format(lcoe.net_aep / lcoe.turbine_number)
    print "Turbine _cost: {0}".format(lcoe.turbine_cost)
    print "BOS costs per turbine: {0}".format(lcoe.bos_costs / lcoe.turbine_number)
    print "OnM costs per turbine: {0}".format(lcoe.avg_annual_opex / lcoe.turbine_number)

    '''fname = 'CSM.txt'
    f = file(fname,'w')

    f.write("File Name: | {0}\n".format(fname))
    f.write("Turbine Conditions:\n")
    f.write("Rated Power: | {0}\n".format(lcoe.machine_rating))
    f.write("Rotor Diameter: | {0}\n".format(lcoe.rotor_diameter))
    f.write("Rotor maximum tip speed: | {0}\n".format(lcoe.max_tip_speed))
    f.write("_cost and mass outputs:\n")
    f.write("LCOE: |{0}\n".format(lcoe.lcoe))
    f.write("COE: |{0}\n".format(lcoe.coe))
    f.write("AEP : |{0}\n".format(lcoe.aep_a))
    f.write("Turbine _cost: |{0}\n".format(lcoe.turbine_cost))
    f.write("BOS costs : |{0}\n".format(lcoe.bos_a_costs))
    f.write("OnM costs : |{0}\n".format(lcoe.opex_a))
    f.close()'''


if __name__=="__main__":

    example()