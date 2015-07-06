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

from wisdem.turbinese.turbine import configure_turbine
from turbine_costsse.turbine_costsse.turbine_costsse import Turbine_CostsSE
from plant_costsse.nrel_csm_bos.nrel_csm_bos import bos_csm_assembly
from plant_costsse.nrel_csm_opex.nrel_csm_opex import opex_csm_assembly
from plant_costsse.ecn_offshore_opex.ecn_offshore_opex  import opex_ecn_assembly
from plant_financese.nrel_csm_fin.nrel_csm_fin import fin_csm_assembly
from fusedwind.plant_flow.basic_aep import aep_assembly, aep_weibull_assembly

import platform
if platform.system() == 'Windows':from landbos import LandBOS

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

    #assembly.replace('tcc_a', Turbine_CostsSE())

    assembly.add('advanced_blade', Bool(True, iotype='in', desc='advanced (True) or traditional (False) blade design'))
    assembly.add('offshore', Bool(iotype='in', desc='flag for offshore site'))
    assembly.add('assemblyCostMultiplier',Float(0.0, iotype='in', desc='multiplier for assembly cost in manufacturing'))
    assembly.add('overheadCostMultiplier', Float(0.0, iotype='in', desc='multiplier for overhead'))
    assembly.add('profitMultiplier', Float(0.0, iotype='in', desc='multiplier for profit markup'))
    assembly.add('transportMultiplier', Float(0.0, iotype='in', desc='multiplier for transport costs'))

    # connections to turbine costs
    assembly.connect('rotor.mass_one_blade', 'tcc_a.blade_mass')
    assembly.connect('hub.hub_mass', 'tcc_a.hub_mass')
    assembly.connect('hub.pitch_system_mass', 'tcc_a.pitch_system_mass')
    assembly.connect('hub.spinner_mass', 'tcc_a.spinner_mass')
    assembly.connect('nacelle.low_speed_shaft_mass', 'tcc_a.low_speed_shaft_mass')
    assembly.connect('nacelle.main_bearing_mass', 'tcc_a.main_bearing_mass')
    assembly.connect('nacelle.second_bearing_mass', 'tcc_a.second_bearing_mass')
    assembly.connect('nacelle.gearbox_mass', 'tcc_a.gearbox_mass')
    assembly.connect('nacelle.high_speed_side_mass', 'tcc_a.high_speed_side_mass')
    assembly.connect('nacelle.generator_mass', 'tcc_a.generator_mass')
    assembly.connect('nacelle.bedplate_mass', 'tcc_a.bedplate_mass')
    assembly.connect('nacelle.yaw_system_mass', 'tcc_a.yaw_system_mass')
    assembly.connect('tower.mass', 'tcc_a.tower_mass')
    assembly.connect('rotor.control.ratedPower', 'tcc_a.machine_rating')
    assembly.connect('rotor.nBlades', 'tcc_a.blade_number')
    assembly.connect('nacelle.crane', 'tcc_a.crane')
    assembly.connect('year', 'tcc_a.year')
    assembly.connect('month', 'tcc_a.month')
    assembly.connect('nacelle.drivetrain_design', 'tcc_a.drivetrain_design')
    assembly.connect('advanced_blade','tcc_a.advanced_blade')
    assembly.connect('offshore','tcc_a.offshore')
    assembly.connect('assemblyCostMultiplier','tcc_a.assemblyCostMultiplier')
    assembly.connect('overheadCostMultiplier','tcc_a.overheadCostMultiplier')
    assembly.connect('profitMultiplier','tcc_a.profitMultiplier')
    assembly.connect('transportMultiplier','tcc_a.transportMultiplier')

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
    assembly.connect('rotor.diameter', 'bos_a.rotor_diameter')
    assembly.connect('rotor.hubHt', 'bos_a.hub_height')
    assembly.connect('turbine_number', 'bos_a.turbine_number')
    assembly.connect('rotor.mass_all_blades + hub.hub_system_mass + nacelle.nacelle_mass', 'bos_a.RNA_mass')

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

    #assembly.replace('bos_a', LandBOS())

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

# Energy Production
def configure_lcoe_with_basic_aep(assembly):
    """
    aep inputs:
        array_losses = Float
        other_losses = Float
        availability = Float
    """

    #assembly.replace('aep_a', aep_assembly())

    assembly.add('array_losses',Float(0.059, iotype='in', desc='energy losses due to turbine interactions - across entire plant'))
    assembly.add('other_losses',Float(0.0, iotype='in', desc='energy losses due to blade soiling, electrical, etc'))

    # connections to aep
    assembly.connect('rotor.AEP', 'aep_a.AEP_one_turbine')
    assembly.connect('turbine_number', 'aep_a.turbine_number')
    assembly.connect('machine_rating','aep_a.machine_rating')
    assembly.connect('array_losses','aep_a.array_losses')
    assembly.connect('other_losses','aep_a.other_losses')

def configure_lcoe_with_weibull_aep(assembly):
    """
    aep inputs
        power_curve    = Array([], iotype='in', desc='wind turbine power curve')
        array_losses = Float
        other_losses = Float
        A = Float
        k = Float
    """

    assembly.add('array_losses',Float(0.059, iotype='in', desc='energy losses due to turbine interactions - across entire plant'))
    assembly.add('other_losses',Float(0.0, iotype='in', desc='energy losses due to blade soiling, electrical, etc'))
    assembly.add('A',Float(8.2,iotype='in', desc='scale factor'))
    assembly.add('k', Float(2.0,iotype='in', desc='shape or form factor'))

    #assembly.replace('aep_a', aep_weibull_assembly())
    
    assembly.connect('turbine_number', 'aep_a.turbine_number')
    assembly.connect('machine_rating','aep_a.machine_rating')
    assembly.connect('array_losses','aep_a.array_losses')
    assembly.connect('other_losses','aep_a.other_losses')
    assembly.connect('A','aep_a.A')
    assembly.connect('k','aep_a.k')
    assembly.connect('rotor.V','aep_a.wind_curve')
    assembly.connect('rotor.P','aep_a.power_curve')


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

    # connections to fin
    assembly.connect('sea_depth', 'fin_a.sea_depth')
    assembly.connect('project_lifetime','fin_a.project_lifetime')
    assembly.connect('fixed_charge_rate','fin_a.fixed_charge_rate')
    assembly.connect('construction_finance_rate','fin_a.construction_finance_rate')
    assembly.connect('tax_rate','fin_a.tax_rate')
    assembly.connect('discount_rate','fin_a.discount_rate')
    assembly.connect('construction_time','fin_a.construction_time')


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
    with_new_nacelle = Bool(False, iotype='in', desc='configure with DriveWPACT if false, else configure with DriveSE')
    with_landbose = Bool(False, iotype='in', desc='configure with CSM BOS if false, else configure with new LandBOS model')
    flexible_blade = Bool(False, iotype='in', desc='configure rotor with flexible blade if True')
    with_3pt_drive = Bool(False, iotype='in', desc='only used if configuring DriveSE - selects 3 pt or 4 pt design option') # TODO: change nacelle selection to enumerated rather than nested boolean
    with_ecn_opex = Bool(False, iotype='in', desc='configure with CSM OPEX if flase, else configure with ECN OPEX model')
    ecn_file = Str(iotype='in', desc='location of ecn excel file if used')

    # Other I/O needed at lcoe system level
    sea_depth = Float(0.0, units='m', iotype='in', desc='sea depth for offshore wind project')
    year = Int(2009, iotype='in', desc='year of project start')
    month = Int(12, iotype='in', desc='month of project start')
    project_lifetime = Float(20.0, iotype='in', desc = 'project lifetime for wind plant')

    def __init__(self, with_new_nacelle=False, with_landbos=False, flexible_blade=False, with_3pt_drive=False, with_ecn_opex=False, ecn_file=None):
        
        self.with_new_nacelle = with_new_nacelle
        self.with_landbos = with_landbos
        self.flexible_blade = flexible_blade
        self.with_3pt_drive = with_3pt_drive
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
            self.replace('bos_a', LandBOS())
        else:
            self.replace('bos_a', bos_csm_assembly())
        self.replace('tcc_a', Turbine_CostsSE())
        if self.with_ecn_opex:  
            self.replace('opex_a', opex_ecn_assembly(ecn_file))
        else:
            self.replace('opex_a', opex_csm_assembly())
        self.replace('aep_a', aep_weibull_assembly())
        self.replace('fin_a', fin_csm_assembly())
    
        # add TurbineSE assembly
        configure_turbine(self, self.with_new_nacelle, self.flexible_blade, self.with_3pt_drive)
    
        # replace TCC with turbine_costs
        configure_lcoe_with_turb_costs(self)
    
        # replace BOS with either CSM or landbos
        if self.with_landbos:
            configure_lcoe_with_landbos(self)
        else:
            configure_lcoe_with_csm_bos(self)

        # replace AEP with weibull AEP (TODO: option for basic aep)
        configure_lcoe_with_weibull_aep(self)
        
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


def create_example_se_assembly(wind_class='I',sea_depth=0.0,with_new_nacelle=False,with_landbos=False,flexible_blade=False,with_3pt_drive=False, with_ecn_opex=False, ecn_file=None,with_openwind=False,ow_file=None,ow_wkbook=None):
    """
    Inputs:
        wind_class : str ('I', 'III', 'Offshore' - selected wind class for project)
        sea_depth : float (sea depth if an offshore wind plant)
    """

    # === Create LCOE SE assembly ========
    lcoe_se = lcoe_se_assembly(with_new_nacelle,with_landbos,flexible_blade,with_3pt_drive,with_ecn_opex,ecn_file)

    # === Set assembly variables and objects ===
    lcoe_se.sea_depth = sea_depth # 0.0 for land-based turbine
    lcoe_se.turbine_number = 100
    lcoe_se.year = 2009
    lcoe_se.month = 12

    rotor = lcoe_se.rotor
    nacelle = lcoe_se.nacelle
    tower = lcoe_se.tower
    tcc_a = lcoe_se.tcc_a
    # bos_a = lcoe_se.bos_a
    # opex_a = lcoe_se.opex_a
    aep_a = lcoe_se.aep_a
    fin_a = lcoe_se.fin_a

    # Turbine ===========
    from wisdem.reference_turbines.nrel5mw.nrel5mw import configure_nrel5mw_turbine
    configure_nrel5mw_turbine(lcoe_se,wind_class,lcoe_se.sea_depth)

    # TODO: these should be specified at the turbine level and connected to other system inputs
    lcoe_se.tower_d = [6.0, 4.935, 3.87]  # (Array, m): diameters along tower
    lcoe_se.generator_speed = 1173.7  # (Float, rpm)  # generator speed
    # extra variable constant for now
    #lcoe_se.nacelle.bedplate.rotor_bending_moment_y = -2.3250E+06 # shouldnt be needed anymore

    # tcc ====
    lcoe_se.advanced_blade = True
    lcoe_se.offshore = False
    lcoe_se.assemblyCostMultiplier = 0.30
    lcoe_se.profitMultiplier = 0.20
    lcoe_se.overheadCostMultiplier = 0.0
    lcoe_se.transportMultiplier = 0.0

    # for new landBOS
    ''' # === new landBOS ===
    lcoe_se.voltage = 137
    lcoe_se.distInter = 5
    lcoe_se.terrain = 'FLAT_TO_ROLLING'
    lcoe_se.layout = 'SIMPLE'
    lcoe_se.soil = 'STANDARD' '''

    # aep ==== # based on COE review for land-based machines
    if not with_openwind:
        lcoe_se.array_losses = 0.059
        lcoe_se.A = 8.9 # weibull of 7.25 at 50 m with shear exp of 0.143
        lcoe_se.k = 2.0
    lcoe_se.other_losses = 0.101
    if not with_ecn_opex:
        lcoe_se.availability = 0.94

    # fin ===
    lcoe_se.fixed_charge_rate = 0.095
    lcoe_se.construction_finance_rate = 0.0
    lcoe_se.tax_rate = 0.4
    lcoe_se.discount_rate = 0.07
    lcoe_se.construction_time = 1.0
    lcoe_se.project_lifetime = 20.0

    # Set plant level inputs ===
    shearExp = 0.2 #TODO : should be an input to lcoe
    rotor.cdf_reference_height_wind_speed = 90.0
    if not with_openwind:
        lcoe_se.array_losses = 0.1
    lcoe_se.other_losses = 0.0
    if not with_ecn_opex:
        lcoe_se.availability = 0.98
    rotor.turbulence_class = 'B'
    lcoe_se.multiplier = 2.23

    if wind_class == 'Offshore':
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

    rotor.shearExp = shearExp
    tower.wind1.shearExp = shearExp
    tower.wind2.shearExp = shearExp

    # ====

    # === Run default assembly and print results
    lcoe_se.run()
    # ====

    # === Print ===

    print "Key Turbine Outputs for NREL 5 MW Reference Turbine"
    print 'mass rotor blades:{0:.2f} (kg) '.format(lcoe_se.rotor.mass_all_blades)
    print 'mass hub system: {0:.2f} (kg) '.format(lcoe_se.hub.hub_system_mass)
    print 'mass nacelle: {0:.2f} (kg) '.format(lcoe_se.nacelle.nacelle_mass)
    print 'mass tower: {0:.2f} (kg) '.format(lcoe_se.tower.mass)
    print 'maximum tip deflection: {0:.2f} (m) '.format(lcoe_se.maxdeflection.max_tip_deflection)
    print 'ground clearance: {0:.2f} (m) '.format(lcoe_se.maxdeflection.ground_clearance)
    print
    print "Key Plant Outputs for wind plant with NREL 5 MW Turbine"
    #print "LCOE: ${0:.4f} USD/kWh".format(lcoe_se.lcoe) # not in base output set (add to assembly output if desired)
    print "COE: ${0:.4f} USD/kWh".format(lcoe_se.coe)
    print
    print "AEP per turbine: {0:.1f} kWh/turbine".format(lcoe_se.net_aep / lcoe_se.turbine_number)
    print "Turbine Cost: ${0:.2f} USD".format(lcoe_se.turbine_cost)
    print "BOS costs per turbine: ${0:.2f} USD/turbine".format(lcoe_se.bos_costs / lcoe_se.turbine_number)
    print "OPEX per turbine: ${0:.2f} USD/turbine".format(lcoe_se.avg_annual_opex / lcoe_se.turbine_number)    

    # ====

if __name__ == '__main__':

    # NREL 5 MW in land-based wind plant with high winds (as class I)
    wind_class = 'I'
    sea_depth = 0.0
    with_new_nacelle = True
    with_landbos = False
    flexible_blade = False
    with_3pt_drive = False
    with_ecn_opex = False
    ecn_file = ''
    create_example_se_assembly(wind_class,sea_depth,with_new_nacelle,with_landbos,flexible_blade,with_3pt_drive,with_ecn_opex,ecn_file) 

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
    #ecn_file = 'C:/Models/ECN Model/ECN O&M Model.xls'
    #create_example_se_assembly(wind_class,sea_depth,with_new_nacelle,with_landbos,flexible_blade,with_3pt_drive,with_ecn_opex,ecn_file) 
   
