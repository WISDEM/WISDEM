"""
LCOE_csm_ssembly.py

Created by NWTC Systems Engineering Sub-Task on 2012-08-01.
Copyright (c) NREL. All rights reserved.
"""


from openmdao.main.api import Assembly
from openmdao.main.datatypes.api import Int, Float, Enum

from turbinese.turbine import configure_turbine
from fusedwind.plant_cost.fused_finance import configure_extended_financial_analysis
from turbine_costsse.turbine_costsse.turbine_costsse import Turbine_CostsSE
from plant_costsse.nrel_csm_bos.nrel_csm_bos import bos_csm_assembly
from plant_costsse.nrel_csm_opex.nrel_csm_opex import opex_csm_assembly
from plant_financese.nrel_csm_fin.nrel_csm_fin import fin_csm_assembly
from plant_energyse.basic_aep.basic_aep import aep_assembly
from landbos import LandBOS


def configure_lcoe(assembly, with_new_nacelle=True, with_new_BOS=True, flexible_blade=False):
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
        turbine_number
        year
        month

        if with_new_BOS:
        voltage
        distInter
        terrain
        layout
        soil

    """


    configure_turbine(assembly, with_new_nacelle=with_new_nacelle, flexible_blade=flexible_blade)
    configure_extended_financial_analysis(assembly)

    assembly.replace('tcc_a', Turbine_CostsSE())
    if with_new_BOS:
        assembly.replace('bos_a', LandBOS())
    else:
        assembly.replace('bos_a', bos_csm_assembly())
    assembly.replace('opex_a', om_csm_assembly())
    assembly.replace('aep_a', aep_assembly())
    assembly.replace('fin_a', fin_csm_assembly())

    # inputs
    assembly.add('sea_depth', Float(0.0, units='m', iotype='in', desc='sea depth for offshore wind project'))
    assembly.add('turbine_number', Int(100, iotype='in', desc='total number of wind turbines at the plant'))
    assembly.add('year', Int(2009, iotype='in', desc='year of project start'))
    assembly.add('month', Int(12, iotype='in', desc='month of project start'))

    if with_new_BOS:
        assembly.add('voltage', Float(iotype='in', units='kV', desc='interconnect voltage'))
        assembly.add('distInter', Float(iotype='in', units='mi', desc='distance to interconnect'))
        assembly.add('terrain', Enum('FLAT_TO_ROLLING', ('FLAT_TO_ROLLING', 'RIDGE_TOP', 'MOUNTAINOUS'),
            iotype='in', desc='terrain options'))
        assembly.add('layout', Enum('SIMPLE', ('SIMPLE', 'COMPLEX'), iotype='in',
            desc='layout options'))
        assembly.add('soil', Enum('STANDARD', ('STANDARD', 'BOUYANT'), iotype='in',
            desc='soil options'))

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

    # connections to bos
    assembly.connect('rotor.control.ratedPower', 'bos_a.machine_rating')
    assembly.connect('rotor.diameter', 'bos_a.rotor_diameter')
    assembly.connect('rotor.hubHt', 'bos_a.hub_height')
    assembly.connect('turbine_number', 'bos_a.turbine_number')
    assembly.connect('rotor.mass_all_blades + hub.hub_system_mass + nacelle.nacelle_mass', 'bos_a.RNA_mass')

    if with_new_BOS:
        assembly.connect('voltage', 'bos_a.voltage')
        assembly.connect('distInter', 'bos_a.distInter')
        assembly.connect('terrain', 'bos_a.terrain')
        assembly.connect('layout', 'bos_a.layout')
        assembly.connect('soil', 'bos_a.soil')

    else:
        assembly.connect('sea_depth', 'bos_a.sea_depth')
        assembly.connect('year', 'bos_a.year')
        assembly.connect('month', 'bos_a.month')

    # connections to opex
    assembly.connect('rotor.control.ratedPower', 'opex_a.machine_rating')
    assembly.connect('sea_depth', 'opex_a.sea_depth')
    assembly.connect('year', 'opex_a.year')
    assembly.connect('month', 'opex_a.month')
    assembly.connect('turbine_number', 'opex_a.turbine_number')
    assembly.connect('aep_a.net_aep', 'opex_a.net_aep')

    # connections to aep
    assembly.connect('rotor.AEP', 'aep_a.AEP_one_turbine')
    assembly.connect('turbine_number', 'aep_a.turbine_number')

    # connections to fin
    assembly.connect('sea_depth', 'fin_a.sea_depth')
    assembly.connect('turbine_number',  'fin_a.turbine_number')






class lcoe_se_csm_assembly(Assembly):

    def configure(self):
        configure_lcoe(self)
