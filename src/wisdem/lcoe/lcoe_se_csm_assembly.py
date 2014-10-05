"""
LCOE_csm_ssembly.py

Created by NWTC Systems Engineering Sub-Task on 2012-08-01.
Copyright (c) NREL. All rights reserved.
"""

import numpy as np

from openmdao.main.api import Assembly
from openmdao.main.datatypes.api import Int, Float, Enum, VarTree, Bool

from turbinese.turbine import configure_turbine
from fusedwind.plant_cost.fused_finance import configure_extended_financial_analysis, ExtendedFinancialAnalysis
from fusedwind.plant_cost.fused_opex import OPEXVarTree
from fusedwind.plant_cost.fused_bos_costs import BOSVarTree
from fusedwind.interface import implement_base
from turbine_costsse.turbine_costsse.turbine_costsse import Turbine_CostsSE
from plant_costsse.nrel_csm_bos.nrel_csm_bos import bos_csm_assembly
from plant_costsse.nrel_csm_opex.nrel_csm_opex import opex_csm_assembly
from plant_financese.nrel_csm_fin.nrel_csm_fin import fin_csm_assembly
from plant_energyse.basic_aep.basic_aep import aep_assembly
from landbos import LandBOS

# Current configuration assembly options for LCOE SE
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

    assembly.replace('tcc_a', Turbine_CostsSE())

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

def configure_lcoe_with_csm_bos(assembly):

    assembly.replace('bos_a', bos_csm_assembly())

    # connections to bos
    assembly.connect('machine_rating', 'bos_a.machine_rating')
    assembly.connect('rotor.diameter', 'bos_a.rotor_diameter')
    assembly.connect('rotor.hubHt', 'bos_a.hub_height')
    assembly.connect('turbine_number', 'bos_a.turbine_number')
    assembly.connect('rotor.mass_all_blades + hub.hub_system_mass + nacelle.nacelle_mass', 'bos_a.RNA_mass')

    assembly.connect('sea_depth', 'bos_a.sea_depth')
    assembly.connect('year', 'bos_a.year')
    assembly.connect('month', 'bos_a.month')

def configure_lcoe_with_landbos(assembly):

    assembly.replace('bos_a', LandBOS())

    assembly.add('voltage', Float(iotype='in', units='kV', desc='interconnect voltage'))
    assembly.add('distInter', Float(iotype='in', units='mi', desc='distance to interconnect'))
    assembly.add('terrain', Enum('FLAT_TO_ROLLING', ('FLAT_TO_ROLLING', 'RIDGE_TOP', 'MOUNTAINOUS'),
        iotype='in', desc='terrain options'))
    assembly.add('layout', Enum('SIMPLE', ('SIMPLE', 'COMPLEX'), iotype='in',
        desc='layout options'))
    assembly.add('soil', Enum('STANDARD', ('STANDARD', 'BOUYANT'), iotype='in',
        desc='soil options'))

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

def configure_lcoe_with_csm_opex(assembly):

    assembly.replace('opex_a', opex_csm_assembly())

    # connections to opex
    assembly.connect('machine_rating', 'opex_a.machine_rating')
    assembly.connect('sea_depth', 'opex_a.sea_depth')
    assembly.connect('year', 'opex_a.year')
    assembly.connect('month', 'opex_a.month')
    assembly.connect('turbine_number', 'opex_a.turbine_number')
    assembly.connect('aep_a.net_aep', 'opex_a.net_aep')

def configure_lcoe_with_basic_aep(assembly):
    """
    aep inputs:
        array_losses = Float
        other_losses = Float
        availability = Float
    """

    assembly.replace('aep_a', aep_assembly())

    # connections to aep
    assembly.connect('rotor.AEP', 'aep_a.AEP_one_turbine')
    assembly.connect('turbine_number', 'aep_a.turbine_number')

def configure_lcoe_with_csm_fin(assembly):
    """
    fin inputs:
        fixed_charge_rate = Float
        construction_finance_rate = Float
        tax_rate = Float
        discount_rate = Float
        construction_time = Float
        project_lifetime = Float
    """

    assembly.replace('fin_a', fin_csm_assembly())

    # connections to fin
    assembly.connect('sea_depth', 'fin_a.sea_depth')

# ====================================================================
# Overall assembly configuration
def configure_lcoe_se(assembly, with_new_nacelle=False, with_landbos=False, flexible_blade=False):
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
        availability = Float
    fin inputs:
        fixed_charge_rate = Float
        construction_finance_rate = Float
        tax_rate = Float
        discount_rate = Float
        construction_time = Float
        project_lifetime = Float
    inputs:
        sea_depth
        year
        month
    if with_landbos additional inputs:
        voltage
        distInter
        terrain
        layout
        soil
    """

    # configure base assembly
    configure_extended_financial_analysis(assembly)

    # add inputs #TODO: awkward adding class inputs in configuration step
    assembly.add('sea_depth', Float(0.0, units='m', iotype='in', desc='sea depth for offshore wind project'))
    assembly.add('year', Int(2009, iotype='in', desc='year of project start'))
    assembly.add('month', Int(12, iotype='in', desc='month of project start'))

    # add TurbineSE assembly
    configure_turbine(assembly, with_new_nacelle=with_new_nacelle, flexible_blade=flexible_blade)

    # replace TCC with turbine_costs
    configure_lcoe_with_turb_costs(assembly)

    # replace BOS with either CSM or landbos
    if with_landbos:
    	  configure_lcoe_with_landbos(assembly)
    else:
    	  configure_lcoe_with_csm_bos(assembly)
    
    # replace OPEX with CSM opex
    configure_lcoe_with_csm_opex(assembly)
    
    # replace AEP with CSM AEP
    configure_lcoe_with_basic_aep(assembly)

    # replace Finance with CSM Finance
    configure_lcoe_with_csm_fin(assembly)


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
    with_landbose = Bool(False, iotype='in', desc='configure with CSM BOS if flase, else configure with new LandBOS model')
    flexible_blade = Bool(False, iotype='in', desc='configure rotor with flexible blade if True')

    # Other I/O
    eta_strain = Float(1.35*1.3*1.0, iotype='in')
    eta_dfl = Float(1.35*1.1*1.0, iotype='in')
    strain_ult_spar = Float(1.0e-2, iotype='in')
    strain_ult_te = Float(2500*1e-6, iotype='in')
    # idx_strain = Array([0, 12, 14, 18, 22, 28, 34], iotype='in', dtype=np.int)
    # idx_buckling = Array([10, 12, 14, 20, 23, 27, 31, 33], iotype='in', dtype=np.int)
    freq_margin = Float(1.1, iotype='in')
    min_ground_clearance = Float(20.0, iotype='in')
    # idx_tower_stress = Array([0, 4, 8, 10, 12, 14], iotype='in', dtype=np.int)
    # idx_tower_fatigue = Array([0, 3, 6, 9, 12, 15, 18, 20], iotype='in', dtype=np.int)


    def __init__(self, with_new_nacelle=False, with_landbos=False, flexible_blade=False):
        
        self.with_new_nacelle = with_new_nacelle
        self.with_landbos = with_landbos
        self.flexible_blade = flexible_blade
        
        super(lcoe_se_assembly,self).__init__()

    def configure(self):

        configure_lcoe_se(self, self.with_new_nacelle, self.with_landbos, self.flexible_blade)

def example():

    # Configuration choices
    with_new_nacelle = False
    with_landbos = False
    flexible_blade = False

    lcoe_se = lcoe_se_assembly(with_new_nacelle,with_landbos,flexible_blade)

    rotor = lcoe_se.rotor
    nacelle = lcoe_se.nacelle
    tower = lcoe_se.tower
    tcc_a = lcoe_se.tcc_a
    # bos_a = lcoe_se.bos_a
    # opex_a = lcoe_se.opex_a
    aep_a = lcoe_se.aep_a
    fin_a = lcoe_se.fin_a

    # ===== Turbine ===========
    from wisdem.reference_turbines.nrel5mw.nrel5mw import configure_nrel5mw_turbine
    configure_nrel5mw_turbine(rotor,nacelle,tower)

    lcoe_se.tower_d = [6.0, 4.935, 3.87]  # (Array, m): diameters along tower
    lcoe_se.generator_speed = 1173.7  # (Float, rpm)  # generator speed
    # extra variable constant for now
    lcoe_se.nacelle.bedplate.rotor_bending_moment_y = -2.3250E+06

    # ===== tcc ====
    tcc_a.advanced_blade = True
    tcc_a.offshore = False
    tcc_a.assemblyCostMultiplier = 0.30
    tcc_a.profitMultiplier = 0.20
    tcc_a.overheadCostMultiplier = 0.0
    tcc_a.transportMultiplier = 0.0

    # ==== aep ====
    aep_a.array_losses = 0.059
    aep_a.other_losses = 0.0
    aep_a.availability = 0.94

    # === fin ===
    fin_a.fixed_charge_rate = 0.095
    fin_a.construction_finance_rate = 0.0
    fin_a.tax_rate = 0.4
    fin_a.discount_rate = 0.07
    fin_a.construction_time = 1.0
    fin_a.project_lifetime = 20.0

    # === assembly variables ===
    lcoe_se.sea_depth = 0.0
    lcoe_se.turbine_number = 100
    lcoe_se.year = 2009
    lcoe_se.month = 12

    # additional 5 MW reference variables #TODO: where should these go?
    lcoe_se.strain_ult_spar = rotor.strain_ult_spar
    lcoe_se.strain_ult_te = rotor.strain_ult_te
    lcoe_se.eta_strain = 1.35*1.3*1.0
    lcoe_se.eta_dfl = 1.35*1.1*1.0
    lcoe_se.freq_margin = 1.1
    lcoe_se.min_ground_clearance = 20.0

    '''# for new landBOS
    lcoe_se.voltage = 137
    lcoe_se.distInter = 5
    lcoe_se.terrain = 'FLAT_TO_ROLLING'
    lcoe_se.layout = 'SIMPLE'
    lcoe_se.soil = 'STANDARD' '''


    # set up reference wind plant site conditions
    shearExp = 0.2
    rotor.cdf_reference_height_wind_speed = 90.0
    aep_a.array_losses = 0.15
    aep_a.other_losses = 0.0
    aep_a.availability = 0.98
    rotor.turbulence_class = 'B'
    lcoe_se.bos_a.multiplier = 2.23

    rotor.shearExp = shearExp
    tower.wind1.shearExp = shearExp
    tower.wind2.shearExp = shearExp

    wind_class = 'I'

    if wind_class == 'I':
        rotor.turbine_class = 'I'

    elif wind_class == 'III':
        rotor.turbine_class = 'III'

        # for fatigue based analysis of class III wind turbine
        #tower.M_DEL = 1.028713178 * 1e3*np.array([7.8792E+003, 7.7507E+003, 7.4918E+003, 7.2389E+003, 6.9815E+003, 6.7262E+003, 6.4730E+003, 6.2174E+003, 5.9615E+003, 5.7073E+003, 5.4591E+003, 5.2141E+003, 4.9741E+003, 4.7399E+003, 4.5117E+003, 4.2840E+003, 4.0606E+003, 3.8360E+003, 3.6118E+003, 3.3911E+003, 3.1723E+003, 2.9568E+003, 2.7391E+003, 2.5294E+003, 2.3229E+003, 2.1246E+003, 1.9321E+003, 1.7475E+003, 1.5790E+003, 1.4286E+003, 1.3101E+003, 1.2257E+003, 1.1787E+003, 1.1727E+003, 1.1821E+003])

        #rotor.Mxb_damage = 1e3*np.array([2.3617E+003, 2.0751E+003, 1.8051E+003, 1.5631E+003, 1.2994E+003, 1.0388E+003, 8.1384E+002, 6.2492E+002, 4.6916E+002, 3.4078E+002, 2.3916E+002, 1.5916E+002, 9.9752E+001, 5.6139E+001, 2.6492E+001, 1.0886E+001, 3.7210E+000, 4.3206E-001])
        #rotor.Myb_damage = 1e3*np.array([2.5492E+003, 2.6261E+003, 2.4265E+003, 2.2308E+003, 1.9882E+003, 1.7184E+003, 1.4438E+003, 1.1925E+003, 9.6251E+002, 7.5564E+002, 5.7332E+002, 4.1435E+002, 2.8036E+002, 1.7106E+002, 8.7732E+001, 3.8678E+001, 1.3942E+001, 1.6600E+000])

    # set up for an offshore analysis - change site conditions
    if wind_class == 'offshore':

        rotor.turbine_class = 'I'
        # rotor.turbulence_class = 'B'
        # rotor.cdf_reference_mean_wind_speed = 8.4
        # rotor.cdf_reference_height_wind_speed = 50.0
        # rotor.weibull_shape = 2.1
        shearExp = 0.14
        # aep_a.array_losses = 0.15
        # aep_a.other_losses = 0.0
        aep_a.availability = 0.96
        rotor.shearExp = shearExp
        tower.wind1.shearExp = shearExp
        tower.wind2.shearExp = shearExp
        tcc_a.offshore = True
        lcoe_se.bos_a.multiplier = 2.33
        lcoe_se.fin_a.fixed_charge_rate = 0.118

        depth = 20.0

        lcoe_se.sea_depth = depth
        lcoe_se.offshore = True
        tower.replace('wave1', LinearWaves())
        tower.replace('wave2', LinearWaves())

        tower.wave1.Uc = 0.0
        tower.wave1.hs = 8.0 * 1.86
        tower.wave1.T = 10.0
        tower.wave1.z_surface = 0.0
        tower.wave1.z_floor = -depth
        tower.wave1.g = 9.81
        tower.wave1.betaWave = 0.0

        tower.wave2.Uc = 0.0
        tower.wave2.hs = 8.0 * 1.86
        tower.wave2.T = 10.0
        tower.wave2.z_surface = 0.0
        tower.wave2.z_floor = -depth
        tower.wave2.g = 9.81
        tower.wave2.betaWave = 0.0

        tower.monopileHeight = depth
        tower.n_monopile = 5
        tower.d_monopile = 6.0
        tower.t_monopile = 6.0/80.0


    # Run default assembly and print results
    lcoe_se.run()

    print "Key Turbine Outputs for NREL 5 MW Reference Turbine"
    print 'mass rotor blades (kg) =', lcoe_se.rotor.mass_all_blades
    print 'mass hub system (kg) =', lcoe_se.hub.hub_system_mass
    print 'mass nacelle (kg) =', lcoe_se.nacelle.nacelle_mass
    print 'mass tower (kg) =', lcoe_se.tower.mass
    print 'maximum tip deflection (m) =', lcoe_se.maxdeflection.max_tip_deflection
    print 'ground clearance (m) =', lcoe_se.maxdeflection.ground_clearance
    print
    print "Key Plant Outputs for land-based wind plant with NREL 5 MW Turbine"
    #print "LCOE: ${0:.4f} USD/kWh".format(lcoe_se.lcoe) # not in base output set (add to assembly output if desired)
    print "COE: ${0:.4f} USD/kWh".format(lcoe_se.coe)
    print
    print "AEP per turbine: {0:1f} kWh/turbine".format(lcoe_se.net_aep / lcoe_se.turbine_number)
    print "Turbine Cost: ${0:2f} USD".format(lcoe_se.turbine_cost)
    print "BOS costs per turbine: ${0:2f} USD/turbine".format(lcoe_se.bos_costs / lcoe_se.turbine_number)
    print "OPEX per turbine: ${0:2f} USD/turbine".format(lcoe_se.avg_annual_opex / lcoe_se.turbine_number)    

if __name__ == '__main__':

    example()
