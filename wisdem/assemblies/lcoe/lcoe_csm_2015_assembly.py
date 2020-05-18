"""
LCOE_csm_ssembly.py

Created by NWTC Systems Engineering Sub-Task on 2012-08-01.
Copyright (c) NREL. All rights reserved.
"""

import numpy as np

from openmdao.main.api import Assembly, Component
from openmdao.main.datatypes.api import Int, Float, Enum, VarTree, Bool, Str, Array


from fusedwind.plant_cost.fused_finance import configure_base_financial_analysis, configure_extended_financial_analysis, ExtendedFinancialAnalysis
from fusedwind.plant_cost.fused_opex import OPEXVarTree
from fusedwind.plant_cost.fused_bos_costs import BOSVarTree
from fusedwind.interface import implement_base

#from wisdem.turbinese.turbine import configure_turbine
from turbine_costsse.turbine_costsse_2015 import Turbine_CostsSE_2015
from turbine_costsse.nrel_csm_tcc_2015 import nrel_csm_tcc_2015
from plant_costsse.nrel_csm_bos.nrel_csm_bos import bos_csm_assembly
from plant_costsse.nrel_csm_opex.nrel_csm_opex import opex_csm_assembly
from plant_costsse.nrel_land_bosse.nrel_land_bosse import NREL_Land_BOSSE
from plant_costsse.ecn_offshore_opex.ecn_offshore_opex  import opex_ecn_assembly
from plant_financese.nrel_csm_fin.nrel_csm_fin import fin_csm_assembly
from plant_energyse.nrel_csm_aep.nrel_csm_aep import aep_csm_assembly

# Current configuration assembly options for LCOE SE
# Turbine Costs
def configure_lcoe_with_turb_costs(assembly):
    """
    tcc_a inputs:
        advanced_blade = Bool
        offshore = Bool
        assemblyCostMultiplier = Float
        overheadCostMultiplier = Float
        profitMultiplier = Float
        transportMultiplier = Float
    """

    assembly.replace('tcc_a', nrel_csm_tcc_2015())

    # Turbine Cost and Mass Inputs
    # parameters / high level inputs
    assembly.add('machine_rating', Float(iotype='in', units='kW', desc='machine rating'))
    assembly.add('blade_number', Int(iotype='in', desc='number of rotor blades'))
    assembly.add('offshore', Bool(iotype='in', desc='flag for offshore project'))
    assembly.add('crane', Bool(iotype='in', desc='flag for presence of onboard crane'))
    assembly.add('bearing_number', Int(2, iotype='in', desc='number of main bearings []') )#number of main bearings- defaults to 2
    assembly.add('rotor_diameter', Float(units = 'm', iotype='in', desc= 'rotor diameter of the machine'))
    assembly.add('turbine_class', Enum('I', ('I', 'II/III', 'User Exponent'), iotype = 'in', desc='turbine class'))
    assembly.add('blade_has_carbon', Bool(False, iotype='in', desc= 'does the blade have carbon?')) #default to doesn't have carbon
    #assembly.add('rotor_torque', Float(iotype='in', units='N * m', desc = 'torque from rotor at rated power')) #JMF do we want this default?
    assembly.add('hub_height', Float(units = 'm', iotype='in', desc= 'hub height of wind turbine above ground / sea level'))

    assembly.connect('machine_rating','tcc_a.machine_rating')
    assembly.connect('blade_number',['tcc_a.blade_number'])
    assembly.connect('offshore',['tcc_a.offshore'])
    assembly.connect('crane',['tcc_a.crane'])
    assembly.connect('bearing_number',['tcc_a.bearing_number'])
    assembly.connect('rotor_diameter','tcc_a.rotor_diameter')
    assembly.connect('turbine_class','tcc_a.turbine_class')
    assembly.connect('blade_has_carbon','tcc_a.blade_has_carbon')
    #assembly.connect('rotor_torque','tcc_a.rotor_torque') #TODO - fix
    assembly.connect('hub_height','tcc_a.hub_height')


# Balance of Station Costs
def configure_lcoe_with_csm_bos(assembly):
    """
    bos inputs:
        bos_multiplier = Float
    """

    #assembly.replace('bos_a', bos_csm_assembly())

    assembly.add('bos_multiplier', Float(1.0, iotype='in'))

    # connections to bos
    assembly.connect('machine_rating', 'bos_a.machine_rating')
    assembly.connect('rotor_diameter', 'bos_a.rotor_diameter')
    assembly.connect('hub_height', 'bos_a.hub_height')
    assembly.connect('turbine_number', 'bos_a.turbine_number')
    #assembly.connect('rotor.mass_all_blades + hub.hub_system_mass + nacelle.nacelle_mass', 'bos_a.RNA_mass')

    assembly.connect('sea_depth', 'bos_a.sea_depth')
    assembly.connect('year', 'bos_a.year')
    assembly.connect('month', 'bos_a.month')
    assembly.connect('bos_multiplier','bos_a.multiplier')

def configure_lcoe_with_landbos(assembly):
    """
    if with_landbos additional inputs:
        voltage
        distInter
        terrain
        layout
        soil
    """

    #assembly.replace('bos_a', NREL_Land_BOSSE())

    assembly.add('voltage', Float(iotype='in', units='kV', desc='interconnect voltage'))
    assembly.add('distInter', Float(iotype='in', units='mi', desc='distance to interconnect'))
    assembly.add('terrain', Enum('FLAT_TO_ROLLING', ('FLAT_TO_ROLLING', 'RIDGE_TOP', 'MOUNTAINOUS'),
        iotype='in', desc='terrain options'))
    assembly.add('layout', Enum('SIMPLE', ('SIMPLE', 'COMPLEX'), iotype='in',
        desc='layout options'))
    assembly.add('soil', Enum('STANDARD', ('STANDARD', 'BOUYANT'), iotype='in',
        desc='soil options'))
    assembly.add('transportDist',Float(0.0, iotype='in', units='mi', desc='transportation distance'))
    # TODO: add rest of land-bos connections

    # connections to bos
    assembly.connect('machine_rating', 'bos_a.machine_rating')
    assembly.connect('rotor.diameter', 'bos_a.rotor_diameter')
    assembly.connect('rotor.hubHt', 'bos_a.hub_height')
    assembly.connect('turbine_number', 'bos_a.turbine_number')
    assembly.connect('rotor.mass_all_blades + hub.hub_system_mass + nacelle.nacelle_mass', 'bos_a.RNA_mass')

    assembly.connect('voltage', 'bos_a.voltage')
    assembly.connect('distInter', 'bos_a.distInter')
    assembly.connect('terrain', 'bos_a.terrain')
    assembly.connect('layout', 'bos_a.layout')
    assembly.connect('soil', 'bos_a.soil')
    assembly.connect('transportDist','bos_a.transportDist')

# Operational Expenditures
def configure_lcoe_with_csm_opex(assembly):
    """
    opex inputs:
       availability = Float()
    """

    #assembly.replace('opex_a', opex_csm_assembly())

    # connections to opex
    assembly.connect('machine_rating', 'opex_a.machine_rating')
    assembly.connect('sea_depth', 'opex_a.sea_depth')
    assembly.connect('year', 'opex_a.year')
    assembly.connect('month', 'opex_a.month')
    assembly.connect('turbine_number', 'opex_a.turbine_number')
    assembly.connect('aep_a.net_aep', 'opex_a.net_aep')


def configure_lcoe_with_ecn_opex(assembly,ecn_file):

    #assembly.replace('opex_a', opex_ecn_assembly(ecn_file))

    assembly.connect('machine_rating', 'opex_a.machine_rating')
    assembly.connect('turbine_number', 'opex_a.turbine_number')
    assembly.connect('tcc_a.turbine_cost','opex_a.turbine_cost')
    assembly.connect('project_lifetime','opex_a.project_lifetime')

# Energy production
def configure_lcoe_with_csm_aep(assembly):
    """
    aep inputs
        power_curve    = Array([], iotype='in', desc='wind turbine power curve')
        array_losses = Float
        other_losses = Float
        A = Float
        k = Float
    """

    # Variables
    #machine_rating = Float(units = 'kW', iotype='in', desc= 'rated machine power in kW')
    assembly.add('max_tip_speed', Float(units = 'm/s', iotype='in', desc= 'maximum allowable tip speed for the rotor'))
    #rotor_diameter = Float(units = 'm', iotype='in', desc= 'rotor diameter of the machine') 
    assembly.add('max_power_coefficient', Float(iotype='in', desc= 'maximum power coefficient of rotor for operation in region 2'))
    assembly.add('opt_tsr', Float(iotype='in', desc= 'optimum tip speed ratio for operation in region 2'))
    assembly.add('cut_in_wind_speed', Float(units = 'm/s', iotype='in', desc= 'cut in wind speed for the wind turbine'))
    assembly.add('cut_out_wind_speed', Float(units = 'm/s', iotype='in', desc= 'cut out wind speed for the wind turbine'))
    assembly.add('hub_height', Float(units = 'm', iotype='in', desc= 'hub height of wind turbine above ground / sea level'))
    assembly.add('altitude', Float(units = 'm', iotype='in', desc= 'altitude of wind plant'))
    assembly.add('air_density', Float(0.0, units = 'kg / (m * m * m)', iotype='in', desc= 'air density at wind plant site')) # default air density value is 0.0 - forces aero csm to calculate air density in model
    assembly.add('drivetrain_design', Enum('geared', ('geared', 'single_stage', 'multi_drive', 'pm_direct_drive'), iotype='in'))
    assembly.add('shear_exponent', Float(iotype='in', desc= 'shear exponent for wind plant')) #TODO - could use wind model here
    assembly.add('wind_speed_50m', Float(iotype='in', units = 'm/s', desc='mean annual wind speed at 50 m height'))
    assembly.add('weibull_k', Float(iotype='in', desc = 'weibull shape factor for annual wind speed distribution'))
    assembly.add('soiling_losses', Float(iotype='in', desc = 'energy losses due to blade soiling for the wind plant - average across turbines'))
    assembly.add('array_losses', Float(iotype='in', desc = 'energy losses due to turbine interactions - across entire plant'))
    #assembly.add('availability', Float(iotype='in', desc = 'average annual availbility of wind turbines at plant'))
    #turbine_number = Int(iotype='in', desc = 'total number of wind turbines at the plant')
    assembly.add('thrust_coefficient', Float(iotype='in', desc='thrust coefficient at rated power'))
    assembly.add('max_efficiency', Float(iotype='in', desc = 'maximum efficiency of rotor and drivetrain - at rated power')) # TODO: should come from drivetrain

    assembly.add('capacity_factor', Float(iotype='out'))

    assembly.connect('rotor_diameter','aep_a.rotor_diameter')
    assembly.connect('machine_rating','aep_a.machine_rating')
    assembly.connect('turbine_number','aep_a.turbine_number')
    
    assembly.connect('max_tip_speed','aep_a.max_tip_speed')
    assembly.connect('max_power_coefficient','aep_a.max_power_coefficient')
    assembly.connect('opt_tsr','aep_a.opt_tsr')
    assembly.connect('cut_in_wind_speed','aep_a.cut_in_wind_speed')
    assembly.connect('cut_out_wind_speed','aep_a.cut_out_wind_speed')
    assembly.connect('hub_height','aep_a.hub_height')
    assembly.connect('altitude','aep_a.altitude')
    assembly.connect('air_density','aep_a.air_density')
    assembly.connect('drivetrain_design','aep_a.drivetrain_design')
    assembly.connect('shear_exponent','aep_a.shear_exponent')
    assembly.connect('wind_speed_50m','aep_a.wind_speed_50m')
    assembly.connect('weibull_k','aep_a.weibull_k')
    assembly.connect('soiling_losses','aep_a.soiling_losses')
    assembly.connect('array_losses','aep_a.array_losses')
    #assembly.connect('availability','aep_a.availability')
    assembly.connect('thrust_coefficient','aep_a.thrust_coefficient')
    assembly.connect('max_efficiency','aep_a.max_efficiency')
    assembly.connect('aep_a.capacity_factor','capacity_factor')

# Finance
def configure_lcoe_with_csm_fin(assembly):
    """
    fin inputs:
        fixed_charge_rate = Float
        construction_finance_rate = Float
        tax_rate = Float
        discount_rate = Float
        construction_time = Float
    """

    #assembly.replace('fin_a', fin_csm_assembly())

    assembly.add('fixed_charge_rate', Float(0.12, iotype = 'in', desc = 'fixed charge rate for coe calculation'))
    assembly.add('construction_finance_rate', Float(0.00, iotype='in', desc = 'construction financing rate applied to overnight capital costs'))
    assembly.add('tax_rate', Float(0.4, iotype = 'in', desc = 'tax rate applied to operations'))
    assembly.add('discount_rate', Float(0.07, iotype = 'in', desc = 'applicable project discount rate'))
    assembly.add('construction_time', Float(1.0, iotype = 'in', desc = 'number of years to complete project construction'))

    assembly.add('lcoe',Float(iotype='out'))

    # connections to fin
    assembly.connect('sea_depth', 'fin_a.sea_depth')
    assembly.connect('project_lifetime','fin_a.project_lifetime')
    assembly.connect('fixed_charge_rate','fin_a.fixed_charge_rate')
    assembly.connect('construction_finance_rate','fin_a.construction_finance_rate')
    assembly.connect('tax_rate','fin_a.tax_rate')
    assembly.connect('discount_rate','fin_a.discount_rate')
    assembly.connect('construction_time','fin_a.construction_time')
    assembly.connect('fin_a.lcoe','lcoe')

# =============================================================================
# Overall LCOE Assembly
@implement_base(ExtendedFinancialAnalysis)
class lcoe_se_assembly(Assembly):

    # Base I/O
    # Inputs
    turbine_number = Int(iotype = 'in', desc = 'number of turbines at plant')

    #Outputs
    turbine_cost = Float(iotype='out', desc = 'A Wind Turbine Capital _cost')
    bos_costs = Float(iotype='out', desc='A Wind Plant Balance of Station _cost Model')
    avg_annual_opex = Float(iotype='out', desc='A Wind Plant Operations Expenditures Model')
    net_aep = Float(iotype='out', desc='A Wind Plant Annual Energy Production Model', units='kW*h')
    coe = Float(iotype='out', desc='Levelized cost of energy for the wind plant')
    opex_breakdown = VarTree(OPEXVarTree(),iotype='out')
    bos_breakdown = VarTree(BOSVarTree(), iotype='out', desc='BOS cost breakdown')

    # Configuration options
    with_landbose = Bool(False, iotype='in', desc='configure with CSM BOS if false, else configure with new LandBOS model')
    with_ecn_opex = Bool(False, iotype='in', desc='configure with CSM OPEX if flase, else configure with ECN OPEX model')
    ecn_file = Str(iotype='in', desc='location of ecn excel file if used')

    # Other I/O needed at lcoe system level
    sea_depth = Float(0.0, units='m', iotype='in', desc='sea depth for offshore wind project')
    year = Int(2009, iotype='in', desc='year of project start')
    month = Int(12, iotype='in', desc='month of project start')
    project_lifetime = Float(20.0, iotype='in', desc = 'project lifetime for wind plant')

    def __init__(self, with_landbos=False, with_ecn_opex=False, ecn_file=None):

        self.with_landbos = with_landbos
        self.with_ecn_opex = with_ecn_opex
        if ecn_file == None:
            self.ecn_file=''
        else:
            self.ecn_file = ecn_file
        
        super(lcoe_se_assembly,self).__init__()

    def configure(self):
        """
        tcc_a inputs:
            advanced_blade = Bool
            offshore = Bool
            assemblyCostMultiplier = Float
            overheadCostMultiplier = Float
            profitMultiplier = Float
            transportMultiplier = Float
        aep inputs:
            array_losses = Float
            other_losses = Float
        fin inputs:
            fixed_charge_rate = Float
            construction_finance_rate = Float
            tax_rate = Float
            discount_rate = Float
            construction_time = Float
        bos inputs:
            bos_multiplier = Float
        inputs:
            sea_depth
            year
            month
            project lifetime
        if csm opex additional inputs:
            availability = Float()
        if openwind opex additional inputs:
            power_curve 
            rpm 
            ct 
        if with_landbos additional inputs:
            voltage
            distInter
            terrain
            layout
            soil
        """
    
        # configure base assembly
        configure_extended_financial_analysis(self)

        # putting replace statements here for now; TODO - openmdao bug
        # replace BOS with either CSM or landbos
        if self.with_landbos:
            self.replace('bos_a', NREL_Land_BOSSE())
        else:
            self.replace('bos_a', bos_csm_assembly())
        #self.replace('tcc_a', Turbine_CostsSE_2015())
        if self.with_ecn_opex:  
            self.replace('opex_a', opex_ecn_assembly(ecn_file))
        else:
            self.replace('opex_a', opex_csm_assembly())
        self.replace('aep_a', aep_csm_assembly()) # TODO include AEP assembly from CSM and use to bridge rotor torque
        self.replace('fin_a', fin_csm_assembly())
    
        # replace TCC with turbine_costs
        configure_lcoe_with_turb_costs(self)
    
        # replace BOS with either CSM or landbos
        if self.with_landbos:
            configure_lcoe_with_landbos(self)
        else:
            configure_lcoe_with_csm_bos(self)

        # replace AEP with weibull AEP (TODO: option for basic aep)
        configure_lcoe_with_csm_aep(self)
        self.connect('aep_a.rotor_torque','tcc_a.rotor_torque')
        
        # replace OPEX with CSM or ECN opex and add AEP
        if self.with_ecn_opex:  
            configure_lcoe_with_ecn_opex(self,ecn_file)     
            self.connect('opex_a.availability','aep_a.availability') # connecting here due to aep / opex reversal depending on model 
        else:
            configure_lcoe_with_csm_opex(self)
            self.add('availability',Float(0.94, iotype='in', desc='average annual availbility of wind turbines at plant'))
            self.connect('availability','aep_a.availability') # connecting here due to aep / opex reversal depending on model
    
        # replace Finance with CSM Finance
        configure_lcoe_with_csm_fin(self)


def create_example_se_assembly(with_landbos=False,with_ecn_opex=False, ecn_file=None,with_openwind=False,ow_file=None,ow_wkbook=None):
    """
    Inputs:
        wind_class : str ('I', 'III', 'Offshore' - selected wind class for project)
        sea_depth : float (sea depth if an offshore wind plant)
    """

    # === Create LCOE SE assembly ========
    lcoe_se = lcoe_se_assembly(with_landbos,with_ecn_opex,ecn_file)

    # === Set assembly variables and objects ===
    lcoe_se.sea_depth = 0.0 # 0.0 for land-based turbine
    lcoe_se.turbine_number = 100
    lcoe_se.year = 2009
    lcoe_se.month = 12

    tcc_a = lcoe_se.tcc_a
    # bos_a = lcoe_se.bos_a
    # opex_a = lcoe_se.opex_a
    aep_a = lcoe_se.aep_a
    fin_a = lcoe_se.fin_a

    # tcc ====
    lcoe_se.rotor_diameter = 126.0
    lcoe_se.turbine_class = 'I'
    lcoe_se.blade_has_carbon = False
    lcoe_se.blade_number = 3    
    lcoe_se.machine_rating = 5000.0
    lcoe_se.hub_height = 90.0
    lcoe_se.bearing_number = 2
    lcoe_se.crane = True

    # Rotor force calculations for nacelle inputs
    #maxTipSpd = 80.0
    #maxEfficiency = 0.90

    #ratedHubPower  = lcoe_se.machine_rating*1000. / maxEfficiency 
    #rotorSpeed     = (maxTipSpd/(0.5*lcoe_se.rotor_diameter)) * (60.0 / (2*np.pi))
    #lcoe_se.rotor_torque = ratedHubPower/(rotorSpeed*(np.pi/30))

    # for new landBOS
    # === new landBOS ===
    if with_landbos:
        lcoe_se.voltage = 137
        lcoe_se.distInter = 5
        lcoe_se.terrain = 'FLAT_TO_ROLLING'
        lcoe_se.layout = 'SIMPLE'
        lcoe_se.soil = 'STANDARD'

    # aep ==== # based on COE review for land-based machines
    if not with_openwind:
        lcoe_se.machine_rating = 5000.0 # Float(units = 'kW', iotype='in', desc= 'rated machine power in kW')
        lcoe_se.rotor_diameter = 126.0 # Float(units = 'm', iotype='in', desc= 'rotor diameter of the machine')
        lcoe_se.max_tip_speed = 80.0 # Float(units = 'm/s', iotype='in', desc= 'maximum allowable tip speed for the rotor')
        lcoe_se.drivetrain_design = 'geared' # Enum('geared', ('geared', 'single_stage', 'multi_drive', 'pm_direct_drive'), iotype='in')
        lcoe_se.altitude = 0.0 # Float(0.0, units = 'm', iotype='in', desc= 'altitude of wind plant')
        lcoe_se.turbine_number = 100 # Int(100, iotype='in', desc = 'total number of wind turbines at the plant')
        lcoe_se.hub_height = 90.0 # Float(units = 'm', iotype='in', desc='hub height of wind turbine above ground / sea level')s
        lcoe_se.max_power_coefficient = 0.488 #Float(0.488, iotype='in', desc= 'maximum power coefficient of rotor for operation in region 2')
        lcoe_se.opt_tsr = 7.525 #Float(7.525, iotype='in', desc= 'optimum tip speed ratio for operation in region 2')
        lcoe_se.cut_in_wind_speed = 3.0 #Float(3.0, units = 'm/s', iotype='in', desc= 'cut in wind speed for the wind turbine')
        lcoe_se.cut_out_wind_speed = 25.0 #Float(25.0, units = 'm/s', iotype='in', desc= 'cut out wind speed for the wind turbine')
        lcoe_se.hub_height = 90.0 #Float(90.0, units = 'm', iotype='in', desc= 'hub height of wind turbine above ground / sea level')
        #lcoe_se.air_density = Float(0.0, units = 'kg / (m * m * m)', iotype='in', desc= 'air density at wind plant site')  # default air density value is 0.0 - forces aero csm to calculate air density in model
        lcoe_se.shear_exponent = 0.1 #Float(0.1, iotype='in', desc= 'shear exponent for wind plant') #TODO - could use wind model here
        lcoe_se.wind_speed_50m = 8.02 #Float(8.35, units = 'm/s', iotype='in', desc='mean annual wind speed at 50 m height')
        lcoe_se.weibull_k= 2.15 #Float(2.1, iotype='in', desc = 'weibull shape factor for annual wind speed distribution')
        lcoe_se.soiling_losses = 0.0 #Float(0.0, iotype='in', desc = 'energy losses due to blade soiling for the wind plant - average across turbines')
        lcoe_se.array_losses = 0.10 #Float(0.06, iotype='in', desc = 'energy losses due to turbine interactions - across entire plant')
        lcoe_se.thrust_coefficient = 0.50 #Float(0.50, iotype='in', desc='thrust coefficient at rated power')
        lcoe_se.max_efficiency = 0.902
    lcoe_se.other_losses = 0.101
    if not with_ecn_opex:
        lcoe_se.availability = 0.941 #Float(0.94287630736, iotype='in', desc = 'average annual availbility of wind turbines at plant')


    # fin ===
    lcoe_se.fixed_charge_rate = 0.095
    lcoe_se.construction_finance_rate = 0.0
    lcoe_se.tax_rate = 0.4
    lcoe_se.discount_rate = 0.07
    lcoe_se.construction_time = 1.0
    lcoe_se.project_lifetime = 20.0

    # Set plant level inputs ===
    shearExp = 0.2 #TODO : should be an input to lcoe
    #rotor.cdf_reference_height_wind_speed = 90.0
    if not with_openwind:
        lcoe_se.array_losses = 0.1
    lcoe_se.other_losses = 0.0
    if not with_ecn_opex:
        lcoe_se.availability = 0.98
    lcoe_se.multiplier = 2.23

    '''if wind_class == 'Offshore':
        # rotor.cdf_reference_mean_wind_speed = 8.4 # TODO - aep from its own module
        # rotor.cdf_reference_height_wind_speed = 50.0
        # rotor.weibull_shape = 2.1
        shearExp = 0.14 # TODO : should be an input to lcoe
        lcoe_se.array_losses = 0.15
        if not with_ecn_opex:
            lcoe_se.availability = 0.96
        lcoe_se.offshore = True
        lcoe_se.multiplier = 2.33
        lcoe_se.fixed_charge_rate = 0.118
    # ===='''

    # === Run default assembly and print results
    lcoe_se.run()
    # ====

    # === Print ===


    print("Key Plant Outputs for wind plant with NREL 5 MW Turbine")
    #print "LCOE: ${0:.4f} USD/kWh".format(lcoe_se.lcoe) # not in base output set (add to assembly output if desired)
    print("COE: ${0:.4f} USD/kWh".format(lcoe_se.coe))
    print()
    print("AEP per turbine: {0:.1f} kWh/turbine".format(lcoe_se.net_aep / lcoe_se.turbine_number))
    print("Turbine Cost: ${0:.2f} USD".format(lcoe_se.turbine_cost))
    print("BOS costs per turbine: ${0:.2f} USD/turbine".format(lcoe_se.bos_costs / lcoe_se.turbine_number))
    print("OPEX per turbine: ${0:.2f} USD/turbine".format(lcoe_se.avg_annual_opex / lcoe_se.turbine_number))    
    print()
    # ====

if __name__ == '__main__':

    # NREL 5 MW in land-based wind plant with high winds (as class I)
    with_landbos = False
    with_ecn_opex = False
    ecn_file = ''
    create_example_se_assembly(with_landbos,with_ecn_opex,ecn_file) 

    '''
    #with_3pt_drive = True
    #create_example_se_assembly(wind_class,sea_depth,with_new_nacelle,with_landbos,flexible_blade,with_3pt_drive,with_ecn_opex,ecn_file)

    #TODO: not working with new updated to DriveSE
    #with_new_nacelle = False
    #create_example_se_assembly(wind_class,sea_depth,with_new_nacelle,with_landbos,flexible_blade,with_3pt_drive,with_ecn_opex,ecn_file) 

    #with_landbos = True
    #create_example_se_assembly(wind_class,sea_depth,with_new_nacelle,with_landbos,flexible_blade,with_3pt_drive,with_ecn_opex,ecn_file) 

    #flexible_blade = True
    #create_example_se_assembly(wind_class,sea_depth,with_new_nacelle,with_landbos,flexible_blade,with_3pt_drive,with_ecn_opex,ecn_file) 

    # NREL 5 MW in land-based wind plant with low winds (as class III)
    #wind_class = 'III'
    #with_new_nacelle = True
    #create_example_se_assembly(wind_class,sea_depth,with_new_nacelle,with_landbos,flexible_blade,with_3pt_drive,with_ecn_opex,ecn_file) 

    # NREL 5 MW in offshore plant with high winds and 20 m sea depth (as class I)
    #wind_class = 'Offshore'
    #sea_depth = 20.0
    #create_example_se_assembly(wind_class,sea_depth,with_new_nacelle,with_landbos,flexible_blade,with_3pt_drive,with_ecn_opex,ecn_file) 
    
    # NREL 5 MW in offshore plant with high winds, 20 m sea depth and ECN opex model
    #wind_class = 'Offshore'
    #sea_depth = 20.0
    #with_ecn_opex = True
    #ecn_file = 'C:/Models/ECN Model/ECN O&M Model.xls' # replace with your file path
    create_example_se_assembly(wind_class,sea_depth,with_new_nacelle,with_landbos,flexible_blade,with_3pt_drive,with_ecn_opex,ecn_file) 
   
    '''