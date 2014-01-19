"""
lcoe_csm-bos-ecn_ssembly.py

Created by NWTC Systems Engineering Sub-Task on 2012-08-01.
Copyright (c) NREL. All rights reserved.
"""

from openmdao.main.api import Component, Assembly, set_as_top, VariableTree
from openmdao.main.datatypes.api import Int, Bool, Float, Array, Str, Enum, VarTree

from fusedwind.plant_cost.fused_fin_asym import ExtendedFinancialAnalysis
from fusedwind.vartrees.varTrees import Turbine

from Turbine_CostsSE.NREL_CSM_TCC.nrel_csm_tcc import tcc_csm_assembly
from Plant_CostsSE.Plant_BOS.NREL_Offshore_BOS.nrel_bos_offshore import bos_nrel_offshore_assembly
from Plant_CostsSE.Plant_OM.ECN_Offshore_OM.ecn_offshore_om  import om_ecn_assembly
from Plant_FinanceSE.Basic_Finance.coe_fin import fin_cst_assembly
from Plant_AEPSE.NREL_CSM_AEP.nrel_csm_aep import aep_csm_assembly
from Plant_AEPSE.NREL_CSM_AEP.csmDriveEfficiency import DrivetrainEfficiencyModel, csmDriveEfficiency

class lcoe_csm_bos_ecn_assembly(ExtendedFinancialAnalysis):

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
    year = Int(2009, iotype='in', desc = 'year of project start')
    month = Int(12, iotype='in', desc = 'month of project start')
    project_lifetime = Float(20.0, iotype = 'in', desc = 'project lifetime for LCOE calculation')    

    def __init__(self, ssfile_1, ssfile_2):

        self.ssfile_1 = ssfile_1
        self.ssfile_2 = ssfile_2
    
        super(lcoe_csm_bos_ecn_assembly, self).__init__()


    def configure(self):
        """ Creates a new LCOE Assembly object """

        super(lcoe_csm_bos_ecn_assembly, self).configure()
        
        self.replace('tcc_a', tcc_csm_assembly())
        self.replace('bos_a', bos_nrel_offshore_assembly(self.ssfile_1))
        self.replace('opex_a', om_ecn_assembly(self.ssfile_2))
        self.replace('aep_a', aep_csm_assembly())
        self.replace('fin_a', fin_cst_assembly())

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
        # dimensions
        self.connect('tcc_a.turbine_cost', 'opex_a.turbine_cost')
        self.connect('tcc_a.turbineVT.rotor.blades.length','bos_a.blade_length')
        self.connect('tcc_a.turbineVT.rotor.blades.width', 'bos_a.blade_width')
        self.connect('tcc_a.turbineVT.rotor.hubsystem.hub.diameter', 'bos_a.hub_diameter')
        self.connect('tcc_a.turbineVT.nacelle.length','bos_a.nacelle_length')
        self.connect('tcc_a.turbineVT.nacelle.height','bos_a.nacelle_height')
        self.connect('tcc_a.turbineVT.nacelle.width','bos_a.nacelle_width')
        self.connect('tcc_a.turbineVT.tower.height', 'bos_a.tower_length')
        self.connect('tcc_a.turbineVT.tower.maxDiameter', 'bos_a.max_tower_diameter')
        self.connect('tcc_a.turbineVT.RNAmass', 'bos_a.RNA_mass')
        # plant configuration
        # climate
        self.connect('altitude', ['aep_a.altitude'])
        self.connect('sea_depth', ['bos_a.sea_depth'])
        # plant operation       
        self.connect('turbine_number', ['aep_a.turbine_number', 'bos_a.turbine_number', 'opex_a.turbine_number', 'fin_a.turbine_number']) 
        # financial
        self.connect('year', ['tcc_a.year'])
        self.connect('month', ['tcc_a.month'])
        self.connect('opex_a.availability', 'aep_a.availability')
        self.connect('project_lifetime', ['opex_a.project_lifetime'])

        # create passthroughs for key input variables of interest
        # turbine
        self.create_passthrough('tcc_a.blade_number')
        self.create_passthrough('tcc_a.advanced_blade')
        self.create_passthrough('tcc_a.crane')
        self.create_passthrough('tcc_a.advanced_bedplate')
        # plant
        self.create_passthrough('aep_a.shear_exponent')
        self.create_passthrough('aep_a.weibull_k')
        self.create_passthrough('aep_a.air_density')
        self.create_passthrough('aep_a.soiling_losses')
        self.create_passthrough('aep_a.array_losses')
        self.create_passthrough('opex_a.availability')
        self.create_passthrough('bos_a.distance_from_shore')
        self.create_passthrough('bos_a.soil_type')
        self.create_passthrough('fin_a.fixed_charge_rate')
 
        # create passthroughs for key output variables of interest
        # aep_a
        self.create_passthrough('aep_a.power_curve')
        # tcc_a
        self.create_passthrough('tcc_a.turbine_mass')
        self.create_passthrough('tcc_a.turbineVT')


def example(ssfile_1, ssfile_2):

    lcoe = lcoe_csm_bos_ecn_assembly(ssfile_1, ssfile_2)
    lcoe.advanced_blade = True
    lcoe.aep_a.drive.drivetrain = csmDriveEfficiency(1)
    lcoe.execute()

    print "COE: {0}".format(lcoe.coe)
    print
    print "AEP per turbine: {0}".format(lcoe.net_aep / lcoe.turbine_number)
    print "Turbine _cost: {0}".format(lcoe.turbine_cost)
    print "BOS costs per turbine: {0}".format(lcoe.bos_costs / lcoe.turbine_number)
    print "OnM costs per turbine: {0}".format(lcoe.avg_annual_opex / lcoe.turbine_number)
    print
    print "Turbine output variable tree:"
    lcoe.turbineVT.printVT()
    print


if __name__=="__main__":

    ssfile_1 = 'C:/Models/BOS/Offshore BOS Model.xlsx'

    ssfile_2 = 'C:/Models/ECN Model/ECN O&M Model.xls' 
    
    example(ssfile_1, ssfile_2)