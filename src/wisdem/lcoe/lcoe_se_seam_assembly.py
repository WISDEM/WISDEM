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

from wisdem.turbinese.turbine_se_seam import configure_turbine
from turbine_costsse.turbine_costsse import Turbine_CostsSE
from plant_costsse.nrel_csm_bos.nrel_csm_bos import bos_csm_assembly
from plant_costsse.nrel_csm_opex.nrel_csm_opex import opex_csm_assembly
from plant_costsse.nrel_land_bosse.nrel_land_bosse import NREL_Land_BOSSE
from plant_costsse.ecn_offshore_opex.ecn_offshore_opex  import opex_ecn_assembly
from plant_financese.nrel_csm_fin.nrel_csm_fin import fin_csm_assembly
from fusedwind.plant_flow.basic_aep import aep_assembly, aep_weibull_assembly

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

    assembly.add('advanced_blade', Bool(True, iotype='in', desc='advanced (True) or traditional (False) blade design', group='Turbine_Cost'))
    assembly.add('offshore', Bool(iotype='in', desc='flag for offshore site', group='Global'))
    assembly.add('assemblyCostMultiplier',Float(0.0, iotype='in', desc='multiplier for assembly cost in manufacturing', group='Turbine_Cost'))
    assembly.add('overheadCostMultiplier', Float(0.0, iotype='in', desc='multiplier for overhead', group='Turbine_Cost'))
    assembly.add('profitMultiplier', Float(0.0, iotype='in', desc='multiplier for profit markup', group='Turbine_Cost'))
    assembly.add('transportMultiplier', Float(0.0, iotype='in', desc='multiplier for transport costs', group='Turbine_Cost'))

    # connections to turbine costs
    assembly.connect('blade_design.blade_mass', 'tcc_a.blade_mass')
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
    assembly.connect('tower_design.tower_mass', 'tcc_a.tower_mass')
    assembly.connect('rated_power', 'tcc_a.machine_rating')
    assembly.connect('blade_number', 'tcc_a.blade_number')
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

    assembly.add('bos_multiplier', Float(1.0, iotype='in', group='Turbine_Cost'))

    # connections to bos
    assembly.connect('rated_power', 'bos_a.machine_rating')
    assembly.connect('rotor_diameter', 'bos_a.rotor_diameter')
    assembly.connect('hub_height', 'bos_a.hub_height')
    assembly.connect('turbine_number', 'bos_a.turbine_number')
    assembly.connect('blade_number * blade_design.blade_mass + hub.hub_system_mass + nacelle.nacelle_mass', 'bos_a.RNA_mass')

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
    assembly.connect('rated_power', 'bos_a.machine_rating')
    assembly.connect('rotor_diameter', 'bos_a.rotor_diameter')
    assembly.connect('hub_height', 'bos_a.hub_height')
    assembly.connect('turbine_number', 'bos_a.turbine_number')
    assembly.connect('blade_number * blade_design.blade_mass + hub.hub_system_mass + nacelle.nacelle_mass', 'bos_a.RNA_mass')

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
    assembly.connect('rated_power', 'opex_a.machine_rating')
    assembly.connect('sea_depth', 'opex_a.sea_depth')
    assembly.connect('year', 'opex_a.year')
    assembly.connect('month', 'opex_a.month')
    assembly.connect('turbine_number', 'opex_a.turbine_number')
    assembly.connect('aep_a.net_aep', 'opex_a.net_aep')


def configure_lcoe_with_ecn_opex(assembly,ecn_file):

    #assembly.replace('opex_a', opex_ecn_assembly(ecn_file))

    assembly.connect('rated_power', 'opex_a.machine_rating')
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

    assembly.add('array_losses',Float(0.059, iotype='in', desc='energy losses due to turbine interactions - across entire plant', group='Plant_AEP'))
    assembly.add('other_losses',Float(0.0, iotype='in', desc='energy losses due to blade soiling, electrical, etc', group='Plant_AEP'))

    # connections to aep
    assembly.connect('aep_calc.aep', 'aep_a.AEP_one_turbine')
    assembly.connect('turbine_number', 'aep_a.turbine_number')
    assembly.connect('rated_power','aep_a.machine_rating')
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

    assembly.add('array_losses',Float(0.059, iotype='in', desc='energy losses due to turbine interactions - across entire plant', group='Plant_AEP'))
    assembly.add('other_losses',Float(0.0, iotype='in', desc='energy losses due to blade soiling, electrical, etc', group='Plant_AEP'))
    assembly.add('weibull_C',Float(8.2,iotype='in', desc='Weibull scale factor', group='Plant_AEP'))
    assembly.add('weibull_k', Float(2.0,iotype='in', desc='Weibull shape or form factor', group='Plant_AEP'))

    #assembly.replace('aep_a', aep_weibull_assembly())

    assembly.connect('turbine_number', 'aep_a.turbine_number')
    assembly.connect('rated_power','aep_a.machine_rating')
    assembly.connect('array_losses','aep_a.array_losses')
    assembly.connect('other_losses','aep_a.other_losses')
    assembly.connect('weibull_C','aep_a.A')
    assembly.connect('weibull_k','aep_a.k')
    assembly.connect('aep_calc.wind_curve','aep_a.wind_curve')
    assembly.connect('aep_calc.power_curve','aep_a.power_curve')


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

    assembly.add('fixed_charge_rate', Float(0.12, iotype = 'in', desc = 'fixed charge rate for coe calculation', group='Plant_Finance'))
    assembly.add('construction_finance_rate', Float(0.00, iotype='in', desc = 'construction financing rate applied to overnight capital costs', group='Plant_Finance'))
    assembly.add('tax_rate', Float(0.4, iotype = 'in', desc = 'tax rate applied to operations', group='Plant_Finance'))
    assembly.add('discount_rate', Float(0.07, iotype = 'in', desc = 'applicable project discount rate', group='Plant_Finance'))
    assembly.add('construction_time', Float(1.0, iotype = 'in', desc = 'number of years to complete project construction', group='Plant_Finance'))

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
class lcoe_se_seam_assembly(Assembly):

    # Base I/O
    # Inputs
    turbine_number = Int(iotype = 'in', desc = 'number of turbines at plant', group='Global')

    #Outputs
    turbine_cost = Float(iotype='out', desc = 'A Wind Turbine Capital _cost')
    bos_costs = Float(iotype='out', desc='A Wind Plant Balance of Station _cost Model')
    avg_annual_opex = Float(iotype='out', desc='A Wind Plant Operations Expenditures Model')
    net_aep = Float(iotype='out', desc='A Wind Plant Annual Energy Production Model', units='kW*h')
    coe = Float(iotype='out', desc='Levelized cost of energy for the wind plant')
    opex_breakdown = VarTree(OPEXVarTree(),iotype='out')
    bos_breakdown = VarTree(BOSVarTree(), iotype='out', desc='BOS cost breakdown')

    # Configuration options
    with_new_nacelle = Bool(False, iotype='in', desc='configure with DriveWPACT if false, else configure with DriveSE', group='Drivetrain')
    with_landbose = Bool(False, iotype='in', desc='configure with CSM BOS if false, else configure with new LandBOS model', group='Plant_cost')
    # flexible_blade = Bool(False, iotype='in', desc='configure rotor with flexible blade if True')
    with_3pt_drive = Bool(False, iotype='in', desc='only used if configuring DriveSE - selects 3 pt or 4 pt design option', group='Drivetrain') # TODO: change nacelle selection to enumerated rather than nested boolean
    with_ecn_opex = Bool(False, iotype='in', desc='configure with CSM OPEX if flase, else configure with ECN OPEX model', group='Other')
    ecn_file = Str(iotype='in', desc='location of ecn excel file if used', group='Other')

    # Other I/O needed at lcoe system level
    sea_depth = Float(0.0, units='m', iotype='in', desc='sea depth for offshore wind project', group='Global')
    year = Int(2009, iotype='in', desc = 'year of project start', group='Plant_Finance')
    month = Int(12, iotype='in', desc = 'month of project start', group='Plant_Finance')
    project_lifetime = Float(20.0, iotype='in', desc = 'project lifetime for wind plant', group='Plant_Finance')

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

        super(lcoe_se_seam_assembly,self).__init__()

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
            self.add('availability',Float(0.94, iotype='in', desc='average annual availbility of wind turbines at plant', group='Plant_AEP'))
            self.connect('availability','aep_a.availability') # connecting here due to aep / opex reversal depending on model

        # replace Finance with CSM Finance
        configure_lcoe_with_csm_fin(self)


    def plot(self, fig):

        from plot_capex import plot_capex
        fig = plot_capex(self)
        return fig

def create_example_se_assembly(wind_class='I',sea_depth=0.0,with_new_nacelle=False,with_landbos=False,flexible_blade=False,with_3pt_drive=False, with_ecn_opex=False, ecn_file=None,with_openwind=False,ow_file=None,ow_wkbook=None):
    """
    Inputs:
        wind_class : str ('I', 'III', 'Offshore' - selected wind class for project)
        sea_depth : float (sea depth if an offshore wind plant)
    """

    # === Create LCOE SE assembly ========
    from openmdao.main.api import set_as_top
    lcoe_se = set_as_top(lcoe_se_seam_assembly(with_new_nacelle,with_landbos,flexible_blade,with_3pt_drive,with_ecn_opex,ecn_file))

    # === Set assembly variables and objects ===
    lcoe_se.sea_depth = sea_depth # 0.0 for land-based turbine
    lcoe_se.turbine_number = 100
    lcoe_se.year = 2009
    lcoe_se.month = 12

    # bos_a = lcoe_se.bos_a
    # opex_a = lcoe_se.opex_a
    aep_a = lcoe_se.aep_a
    fin_a = lcoe_se.fin_a

    # Turbine ===========
    #=========== SEAM inputs

    # DTU 10 MW Turbine
    '''lcoe_se.site_type = 'onshore'
    lcoe_se.rotor_diameter = 178.0
    lcoe_se.rated_power = 10.0
    lcoe_se.hub_height = 120.0
    lcoe_se.max_tipspeed = 90.0'''

    # NREL 5 MW Turbine
    # lcoe_se.site_type = 'onshore'
    lcoe_se.rotor_diameter = 126.0
    lcoe_se.rated_power = 5000.0
    lcoe_se.hub_height = 90.0
    lcoe_se.max_tipspeed = 80.0

    lcoe_se.BladeCostPerMass = 15.0
    lcoe_se.HubCostPerMass = 3.5
    lcoe_se.SpinnerCostPerMass = 4.5
    lcoe_se.hub_cost_per_mass = 3.5
    lcoe_se.spinner_cost_per_mass = 4.5
    lcoe_se.tower_cost_per_mass = 4.0

    lcoe_se.AddWeightFactorBlade = 1.2
    lcoe_se.blade_material_density = 2100.0
    lcoe_se.tower_bottom_diameter = 6.
    lcoe_se.tower_top_diameter = 3.78
    lcoe_se.blade_edge_dynload_factor_ext = 2.5
    lcoe_se.blade_edge_dynload_factor_fat = 0.75
    lcoe_se.F = 0.777
    lcoe_se.MaxChordrR = 0.2
    lcoe_se.project_lifetime = 20.0
    lcoe_se.lifetime_cycles = 10000000.0
    lcoe_se.blade_sections = 21
    lcoe_se.PMtarget_blades = 1.0
    lcoe_se.PMtarget_tower = 1.0
    lcoe_se.safety_factor_blade = 1.1
    lcoe_se.safety_factor_tower = 1.5
    lcoe_se.stress_limit_extreme_tower = 235.0
    lcoe_se.stress_limit_fatigue_tower = 14.885
    lcoe_se.stress_limit_extreme_blade = 200.0
    lcoe_se.stress_limit_fatigue_blade = 27.0
    lcoe_se.tif_blade_root_flap_ext = 1.0
    lcoe_se.tif_blade_root_flap_fat = 1.0
    lcoe_se.tif_blade_root_edge_ext = 1.0
    lcoe_se.weibull_C = 11.0
    lcoe_se.weibull_k = 2.0
    lcoe_se.wohler_exponent_blade_flap = 10.0
    lcoe_se.wohler_exponent_tower = 4.0
    lcoe_se.dLoad_dU_factor_flap = 0.9
    lcoe_se.dLoad_dU_factor_tower = 0.8
    lcoe_se.n_wsp = 26
    lcoe_se.min_wsp = 0.0
    lcoe_se.max_wsp = 25.0
    lcoe_se.nSigma4fatFlap = 1.2
    lcoe_se.nSigma4fatTower = 0.8
    lcoe_se.rho_steel = 7800.0
    lcoe_se.sc_frac_edge = 0.8
    lcoe_se.sc_frac_flap = 0.3
    lcoe_se.tsr = 8.0
    lcoe_se.air_density = 1.225
    lcoe_se.turbulence_int = 0.16
    lcoe_se.max_Cp = 0.49
    lcoe_se.gearloss_const = 0.01    # Fraction
    lcoe_se.gearloss_var = 0.014     # Fraction
    lcoe_se.genloss = 0.03          # Fraction
    lcoe_se.convloss = 0.03         # Fraction

    #==============

    # === nacelle ======
    lcoe_se.blade_number = 3 # turbine level that must be added for SEAM
    lcoe_se.rotor_tilt = 5.0 # turbine level that must be added for SEAM
    lcoe_se.generator_speed = 1173.7

    lcoe_se.nacelle.L_ms = 1.0  # (Float, m): main shaft length downwind of main bearing in low-speed shaft
    lcoe_se.nacelle.L_mb = 2.5  # (Float, m): main shaft length in low-speed shaft

    lcoe_se.nacelle.h0_front = 1.7  # (Float, m): height of Ibeam in bedplate front
    lcoe_se.nacelle.h0_rear = 1.35  # (Float, m): height of Ibeam in bedplate rear

    lcoe_se.nacelle.drivetrain_design = 'geared'
    lcoe_se.nacelle.crane = True  # (Bool): flag for presence of crane
    lcoe_se.nacelle.bevel = 0  # (Int): Flag for the presence of a bevel stage - 1 if present, 0 if not
    lcoe_se.nacelle.gear_configuration = 'eep'  # (Str): tring that represents the configuration of the gearbox (stage number and types)

    lcoe_se.nacelle.Np = [3, 3, 1]  # (Array): number of planets in each stage
    lcoe_se.nacelle.ratio_type = 'optimal'  # (Str): optimal or empirical stage ratios
    lcoe_se.nacelle.shaft_type = 'normal'  # (Str): normal or short shaft length
    #lcoe_se.nacelle.shaft_angle = 5.0  # (Float, deg): Angle of the LSS inclindation with respect to the horizontal
    lcoe_se.nacelle.shaft_ratio = 0.10  # (Float): Ratio of inner diameter to outer diameter.  Leave zero for solid LSS
    lcoe_se.nacelle.carrier_mass = 8000.0 # estimated for 5 MW
    lcoe_se.nacelle.mb1Type = 'CARB'  # (Str): Main bearing type: CARB, TRB or SRB
    lcoe_se.nacelle.mb2Type = 'SRB'  # (Str): Second bearing type: CARB, TRB or SRB
    lcoe_se.nacelle.yaw_motors_number = 8.0  # (Float): number of yaw motors
    lcoe_se.nacelle.uptower_transformer = True
    lcoe_se.nacelle.flange_length = 0.5 #m
    lcoe_se.nacelle.gearbox_cm = 0.1
    lcoe_se.nacelle.hss_length = 1.5
    lcoe_se.nacelle.overhang = 5.0 #TODO - should come from turbine configuration level

    lcoe_se.nacelle.check_fatigue = 0 #0 if no fatigue check, 1 if parameterized fatigue check, 2 if known loads inputs

    # =================

    # tcc ====
    lcoe_se.advanced_blade = True
    lcoe_se.offshore = False
    lcoe_se.assemblyCostMultiplier = 0.30
    lcoe_se.profitMultiplier = 0.20
    lcoe_se.overheadCostMultiplier = 0.0
    lcoe_se.transportMultiplier = 0.0

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
    if not with_openwind:
        lcoe_se.array_losses = 0.1
    lcoe_se.other_losses = 0.0
    if not with_ecn_opex:
        lcoe_se.availability = 0.98
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

    # ====

    # === Run default assembly and print results
    # lcoe_se.run()
    return lcoe_se
    # ====

    # === Print ===

#   print "Key Turbine Outputs for Reference Turbine"
#   print 'mass rotor blades:{0:.2f} (kg) '.format(lcoe_se.blade_number * lcoe_se.blade_design.BladeWeight)
#   print 'mass hub system: {0:.2f} (kg) '.format(lcoe_se.hubSystem.hub_system_mass)
#   print 'mass nacelle: {0:.2f} (kg) '.format(lcoe_se.nacelle.nacelle_mass)
#   print 'mass tower: {0:.2f} (kg) '.format(lcoe_se.tower_design.mass)
#   print
#   print "Key Plant Outputs for wind plant with Reference Turbine"
#   #print "LCOE: ${0:.4f} USD/kWh".format(lcoe_se.lcoe) # not in base output set (add to assembly output if desired)
#   print "COE: ${0:.4f} USD/kWh".format(lcoe_se.coe)
#   print
#   print "AEP per turbine: {0:.1f} kWh/turbine".format(lcoe_se.net_aep / lcoe_se.turbine_number)
#   print "Turbine Cost: ${0:.2f} USD".format(lcoe_se.turbine_cost)
#   print "BOS costs per turbine: ${0:.2f} USD/turbine".format(lcoe_se.bos_costs / lcoe_se.turbine_number)
#   print "OPEX per turbine: ${0:.2f} USD/turbine".format(lcoe_se.avg_annual_opex / lcoe_se.turbine_number)

    # ====

if __name__ == '__main__':

    # NREL 5 MW in land-based wind plant with high winds (as class I)
    wind_class = 'I'
    sea_depth = 0.0
    with_new_nacelle = True
    with_landbos = False
    with_3pt_drive = False
    with_ecn_opex = False
    flexible_blade = False
    ecn_file = ''
    # create_example_se_assembly(wind_class,sea_depth,with_new_nacelle,with_landbos,flexible_blade,with_3pt_drive,with_ecn_opex,ecn_file)
    lcoe_se = create_example_se_assembly('I', 0., True, False, False,False,False, '')
    lcoe_se.run()
    print 'cost:', lcoe_se.turbine_cost
    #with_3pt_drive = True
    #create_example_se_assembly(wind_class,sea_depth,with_new_nacelle,with_landbos,flexible_blade,with_3pt_drive,with_ecn_opex,ecn_file)

    #with_new_nacelle = False
    #create_example_se_assembly(wind_class,sea_depth,with_new_nacelle,with_landbos,flexible_blade,with_3pt_drive,with_ecn_opex,ecn_file)

    #with_landbos = True
    #create_example_se_assembly(wind_class,sea_depth,with_new_nacelle,with_landbos,flexible_blade,with_3pt_drive,with_ecn_opex,ecn_file)

    # NREL 5 MW in offshore plant with high winds and 20 m sea depth (as class I)
    #wind_class = 'Offshore'
    #sea_depth = 20.0
    #create_example_se_assembly(wind_class,sea_depth,with_new_nacelle,with_landbos,flexible_blade,with_3pt_drive,with_ecn_opex,ecn_file)
