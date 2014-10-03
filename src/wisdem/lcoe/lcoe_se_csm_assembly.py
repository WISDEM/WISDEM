"""
LCOE_csm_ssembly.py

Created by NWTC Systems Engineering Sub-Task on 2012-08-01.
Copyright (c) NREL. All rights reserved.
"""

import numpy as np
import os

from openmdao.main.api import Assembly
from openmdao.main.datatypes.api import Int, Float, Enum, VarTree

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
    assembly.replace('opex_a', opex_csm_assembly())
    assembly.replace('aep_a', aep_assembly())
    assembly.replace('fin_a', fin_csm_assembly())

    # inputs
    assembly.add('sea_depth', Float(0.0, units='m', iotype='in', desc='sea depth for offshore wind project'))
    #assembly.add('turbine_number', Int(100, iotype='in', desc='total number of wind turbines at the plant')) # in base class
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
    #assembly.connect('turbine_number',  'fin_a.turbine_number') # in base configure

# basic assembly
@implement_base(ExtendedFinancialAnalysis)
class lcoe_se_csm_assembly(Assembly):

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

    def configure(self):
        configure_lcoe(self)


# test assembly
@implement_base(ExtendedFinancialAnalysis)
class BaseLCOE(Assembly):

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


    def configure(self):
        with_new_nacelle = True
        with_new_BOS = False
        flexible_blade = False

        configure_lcoe(self, with_new_nacelle=with_new_nacelle, with_new_BOS=with_new_BOS, flexible_blade=flexible_blade)

def example():

    from rotorse.precomp import Profile, Orthotropic2DMaterial, CompositeSection  # TODO: can just pass file names and do this initialization inside of rotor
    from towerse.tower import TowerWithpBEAM
    from commonse.environment import PowerWind, TowerSoil, LinearWaves
    from commonse.utilities import cosd, sind
    from rotorse.rotoraero import RS2RPM

    tipspeed = BaseLCOE()

    rotor = tipspeed.rotor
    nacelle = tipspeed.nacelle
    tower = tipspeed.tower
    tcc_a = tipspeed.tcc_a
    # bos_a = tipspeed.bos_a
    # opex_a = tipspeed.opex_a
    aep_a = tipspeed.aep_a
    fin_a = tipspeed.fin_a

    # --- blade grid ---
    rotor.initial_aero_grid = np.array([0.02222276, 0.06666667, 0.11111057, 0.16666667, 0.23333333, 0.3, 0.36666667, 0.43333333,
        0.5, 0.56666667, 0.63333333, 0.7, 0.76666667, 0.83333333, 0.88888943, 0.93333333, 0.97777724])
    rotor.initial_str_grid = np.array([0.0, 0.00492790457512, 0.00652942887106, 0.00813095316699, 0.00983257273154, 0.0114340970275,
        0.0130356213234, 0.02222276, 0.024446481932, 0.026048006228, 0.06666667, 0.089508406455, 0.11111057,
        0.146462614229, 0.16666667, 0.195309105255, 0.23333333, 0.276686558545, 0.3, 0.333640766319, 0.36666667,
        0.400404310407, 0.43333333, 0.5, 0.520818918408, 0.56666667, 0.602196371696, 0.63333333, 0.667358391486,
        0.683573824984, 0.7, 0.73242031601, 0.76666667, 0.83333333, 0.88888943, 0.93333333, 0.97777724, 1.0])
    rotor.idx_cylinder_aero = 3
    rotor.idx_cylinder_str = 14
    rotor.hubFraction = 0.025

    # --- blade geometry ---
    rotor.r_aero = np.array([0.02222276, 0.06666667, 0.11111057, 0.2, 0.23333333, 0.3, 0.36666667, 0.43333333, 0.5,
        0.56666667, 0.63333333, 0.64, 0.7, 0.83333333, 0.88888943, 0.93333333, 0.97777724])
    rotor.r_max_chord = 0.23577
    rotor.chord_sub = [3.2612, 4.5709, 3.3178, 1.4621]
    rotor.theta_sub = [13.2783, 7.46036, 2.89317, -0.0878099]
    rotor.precurve_sub = [0.0, 0.0, 0.0]
    rotor.delta_precurve_sub = [0.0, 0.0, 0.0]
    rotor.sparT = [1.0, 0.047754, 0.045376, 0.031085, 0.0061398]
    rotor.teT = [1.0, 0.09569, 0.06569, 0.02569, 0.00569]
    rotor.bladeLength = 61.5
    rotor.delta_bladeLength = 0.0
    rotor.precone = 2.5
    rotor.tilt = 5.0
    rotor.yaw = 0.0
    rotor.nBlades = 3

    # --- airfoil files ---
    #basepath = os.path.join('5MW_files', '5MW_AFFiles')
    basepath = os.path.join('..','reference_turbines','nrel5mw','airfoils')

    # load all airfoils
    airfoil_types = [0]*8
    airfoil_types[0] = os.path.join(basepath, 'Cylinder1.dat')
    airfoil_types[1] = os.path.join(basepath, 'Cylinder2.dat')
    airfoil_types[2] = os.path.join(basepath, 'DU40_A17.dat')
    airfoil_types[3] = os.path.join(basepath, 'DU35_A17.dat')
    airfoil_types[4] = os.path.join(basepath, 'DU30_A17.dat')
    airfoil_types[5] = os.path.join(basepath, 'DU25_A17.dat')
    airfoil_types[6] = os.path.join(basepath, 'DU21_A17.dat')
    airfoil_types[7] = os.path.join(basepath, 'NACA64_A17.dat')

    # place at appropriate radial stations
    af_idx = [0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7]

    n = len(af_idx)
    af = [0]*n
    for i in range(n):
        af[i] = airfoil_types[af_idx[i]]
    rotor.airfoil_files = af


    # --- atmosphere ---
    rotor.rho = 1.225
    rotor.mu = 1.81206e-5
    rotor.shearExp = 0.2
    rotor.hubHt = 90.0
    rotor.turbine_class = 'I'
    rotor.turbulence_class = 'B'
    rotor.g = 9.81


    # --- control ---
    rotor.control.Vin = 3.0
    rotor.control.Vout = 25.0
    rotor.control.ratedPower = 5e6
    rotor.control.minOmega = 0.0
    rotor.control.maxOmega = 12.0
    rotor.control.tsr = 7.55
    rotor.control.pitch = 0.0
    rotor.max_tip_speed = 80.0
    rotor.pitch_extreme = 0.0
    rotor.azimuth_extreme = 0.0

    # rotor.yawW = 130.0
    # rotor.worst_case_pitch_yaw_error = 90.0
    rotor.VfactorPC = 0.7


    # --- aero and structural analysis options ---
    rotor.nSector = 4
    rotor.npts_coarse_power_curve = 20
    rotor.npts_spline_power_curve = 200
    rotor.AEP_loss_factor = 1.0
    rotor.drivetrainType = 'geared'
    rotor.nF = 5
    rotor.dynamic_amplication_tip_deflection = 1.35


    # --- materials and composite layup  ---
    #basepath = os.path.join('5MW_files', '5MW_PrecompFiles')
    basepath = os.path.join('..', 'reference_turbines','nrel5mw','blade')

    materials = Orthotropic2DMaterial.listFromPreCompFile(os.path.join(basepath, 'materials.inp'))

    ncomp = len(rotor.initial_str_grid)
    upper = [0]*ncomp
    lower = [0]*ncomp
    webs = [0]*ncomp
    profile = [0]*ncomp

    rotor.leLoc = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.498, 0.497, 0.465, 0.447, 0.43, 0.411, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
    rotor.sector_idx_strain_spar = [2]*ncomp
    rotor.sector_idx_strain_te = [3]*ncomp
    web1 = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.4114, 0.4102, 0.4094, 0.3876, 0.3755, 0.3639, 0.345, 0.3342, 0.3313, 0.3274, 0.323, 0.3206, 0.3172, 0.3138, 0.3104, 0.307, 0.3003, 0.2982, 0.2935, 0.2899, 0.2867, 0.2833, 0.2817, 0.2799, 0.2767, 0.2731, 0.2664, 0.2607, 0.2562, 0.1886, -1.0])
    web2 = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.5886, 0.5868, 0.5854, 0.5508, 0.5315, 0.5131, 0.4831, 0.4658, 0.4687, 0.4726, 0.477, 0.4794, 0.4828, 0.4862, 0.4896, 0.493, 0.4997, 0.5018, 0.5065, 0.5101, 0.5133, 0.5167, 0.5183, 0.5201, 0.5233, 0.5269, 0.5336, 0.5393, 0.5438, 0.6114, -1.0])
    web3 = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
    rotor.chord_str_ref = np.array([3.2612, 3.3100915356, 3.32587052924, 3.34159388653, 3.35823798667, 3.37384375335, 3.38939112914, 3.4774055542, 3.49839685, 3.51343645709, 3.87017220335, 4.04645623801, 4.19408216643, 4.47641008477, 4.55844487985, 4.57383098262, 4.57285771934, 4.51914315648, 4.47677655262, 4.40075650022, 4.31069949379, 4.20483735936, 4.08985563932, 3.82931757126, 3.74220276467, 3.54415796922, 3.38732428502, 3.24931446473, 3.23421422609, 3.22701537997, 3.21972125648, 3.08979310611, 2.95152261813, 2.330753331, 2.05553464181, 1.82577817774, 1.5860853279, 1.4621])

    for i in range(ncomp):

        webLoc = []
        if web1[i] != -1:
            webLoc.append(web1[i])
        if web2[i] != -1:
            webLoc.append(web2[i])
        if web3[i] != -1:
            webLoc.append(web3[i])

        upper[i], lower[i], webs[i] = CompositeSection.initFromPreCompLayupFile(os.path.join(basepath, 'layup_' + str(i+1) + '.inp'), webLoc, materials)
        profile[i] = Profile.initFromPreCompFile(os.path.join(basepath, 'shape_' + str(i+1) + '.inp'))

    rotor.materials = materials
    rotor.upperCS = upper
    rotor.lowerCS = lower
    rotor.websCS = webs
    rotor.profile = profile
    # --------------------------------------

    strain_ult_spar = 1.0e-2
    strain_ult_te = 2500*1e-6

    # --- fatigue ---
    rotor.rstar_damage = np.array([0.000, 0.022, 0.067, 0.111, 0.167, 0.233, 0.300, 0.367, 0.433, 0.500, 0.567, 0.633, 0.700, 0.767, 0.833, 0.889, 0.933, 0.978])
    rotor.Mxb_damage = 1e3*np.array([2.3743E+003, 2.0834E+003, 1.8108E+003, 1.5705E+003, 1.3104E+003, 1.0488E+003, 8.2367E+002, 6.3407E+002, 4.7727E+002, 3.4804E+002, 2.4458E+002, 1.6339E+002, 1.0252E+002, 5.7842E+001, 2.7349E+001, 1.1262E+001, 3.8549E+000, 4.4738E-001])
    rotor.Myb_damage = 1e3*np.array([2.7732E+003, 2.8155E+003, 2.6004E+003, 2.3933E+003, 2.1371E+003, 1.8459E+003, 1.5582E+003, 1.2896E+003, 1.0427E+003, 8.2015E+002, 6.2449E+002, 4.5229E+002, 3.0658E+002, 1.8746E+002, 9.6475E+001, 4.2677E+001, 1.5409E+001, 1.8426E+000])
    rotor.strain_ult_spar = strain_ult_spar
    rotor.strain_ult_te = strain_ult_te * 2  # note that I am putting a factor of two for the damage part only.  I think this strain value is to restrictive otherwise.
    rotor.eta_damage = 1.35*1.3*1.0
    rotor.m_damage = 10.0
    rotor.N_damage = 365*24*3600*20.0


    # ---- nacelle --------

    nacelle.L_ms = 1.0
    nacelle.L_mb = 2.5
    # nacelle.tf_rear = 0.01905 * 2
    # nacelle.tw_rear = 0.0127 * 2
    # nacelle.h0_rear = 0.6096 * 2
    # nacelle.tf_front = 0.01905 * 2.3
    # nacelle.tw_front = 0.0127 * 2.3
    # nacelle.h0_front = 0.6096 * 2.3

    # nacelle.gear_ratio = 87.965
    tipspeed.generator_speed = 1173.7
    nacelle.drivetrain_design = 'geared'
    nacelle.crane = True
    nacelle.gear_configuration = 'eep'

    nacelle.Np = [3, 3, 1]
    nacelle.ratio_type = 'optimal'  # optimal or empirical
    nacelle.shaft_type = 'normal'  # normal or short
    nacelle.shaft_angle = 5.0
    nacelle.shaft_ratio = 0.10
    nacelle.shrink_disc_mass = 1000.0
    nacelle.mb1Type = 'CARB'
    nacelle.mb2Type = 'SRB'
    nacelle.g = 9.81
    nacelle.yaw_motors_number = 8.0

    # ---- tower ------
    tower.replace('wind1', PowerWind())
    tower.replace('wind2', PowerWind())
    # tower.replace('wave1', LinearWaves())  # no waves
    tower.replace('soil', TowerSoil())
    tower.replace('tower1', TowerWithpBEAM())
    tower.replace('tower2', TowerWithpBEAM())

    # geometry
    tower.z = np.array([0.0, 0.5, 1.0])
    tipspeed.tower_d = [6.0, 4.935, 3.87]
    tower.t = 1.3*np.array([0.027, 0.023, 0.019])
    tower.n = [10, 10]
    tower.L_reinforced = 30.0
    # tower.n_reinforced = 3

    tower.gamma_f = 1.35
    tower.gamma_m = 1.1
    tower.gamma_n = 1.0
    # tower.gamma_b = 1.3  # made it higher b.c. we don't have GL buckling yet
    tower.gamma_b = 1.1
    tower.sigma_y = 345e6

    # damage
    tower.z_DEL = 1.0/87.6*np.array([0.000, 1.327, 3.982, 6.636, 9.291, 11.945, 14.600, 17.255, 19.909, 22.564, 25.218, 27.873, 30.527, 33.182, 35.836, 38.491, 41.145, 43.800, 46.455, 49.109, 51.764, 54.418, 57.073, 59.727, 62.382, 65.036, 67.691, 70.345, 73.000, 75.655, 78.309, 80.964, 83.618, 86.273, 87.600])
    tower.M_DEL = 1e3*np.array([8.2940E+003, 8.1518E+003, 7.8831E+003, 7.6099E+003, 7.3359E+003, 7.0577E+003, 6.7821E+003, 6.5119E+003, 6.2391E+003, 5.9707E+003, 5.7070E+003, 5.4500E+003, 5.2015E+003, 4.9588E+003, 4.7202E+003, 4.4884E+003, 4.2577E+003, 4.0246E+003, 3.7942E+003, 3.5664E+003, 3.3406E+003, 3.1184E+003, 2.8977E+003, 2.6811E+003, 2.4719E+003, 2.2663E+003, 2.0673E+003, 1.8769E+003, 1.7017E+003, 1.5479E+003, 1.4207E+003, 1.3304E+003, 1.2780E+003, 1.2673E+003, 1.2761E+003])
    tower.gamma_fatigue = 1.35*1.3*1.0
    tower.life = 20.0
    tower.m_SN = 4

    # wind
    tower.wind_z0 = 0.0
    tower.wind1.shearExp = rotor.shearExp
    tower.wind2.shearExp = rotor.shearExp

    # soil
    tower.soil.rigid = 6*[True]

    # constraints
    tower.min_d_to_t = 120.0
    tower.min_taper = 0.4



    # ----- tcc ----
    tcc_a.advanced_blade = True
    tcc_a.offshore = False
    tcc_a.assemblyCostMultiplier = 0.30
    tcc_a.profitMultiplier = 0.20
    tcc_a.overheadCostMultiplier = 0.0
    tcc_a.transportMultiplier = 0.0

    # ---- aep ----
    aep_a.array_losses = 0.059
    aep_a.other_losses = 0.0
    aep_a.availability = 0.94


    # --- fin ---
    fin_a.fixed_charge_rate = 0.095
    fin_a.construction_finance_rate = 0.0
    fin_a.tax_rate = 0.4
    fin_a.discount_rate = 0.07
    fin_a.construction_time = 1.0
    fin_a.project_lifetime = 20.0

    # --- assembly variables ---
    tipspeed.sea_depth = 0.0
    tipspeed.turbine_number = 100
    tipspeed.year = 2010
    tipspeed.month = 12
    tipspeed.strain_ult_spar = strain_ult_spar
    tipspeed.strain_ult_te = strain_ult_te
    tipspeed.eta_strain = 1.35*1.3*1.0
    tipspeed.eta_dfl = 1.35*1.1*1.0
    tipspeed.freq_margin = 1.1
    tipspeed.min_ground_clearance = 20.0

    # BOS
    tipspeed.voltage = 137
    tipspeed.distInter = 5
    tipspeed.terrain = 'FLAT_TO_ROLLING'
    tipspeed.layout = 'SIMPLE'
    tipspeed.soil = 'STANDARD'

    # extra variable constant for now
    tipspeed.nacelle.bedplate.rotor_bending_moment_y = -2.3250E+06
    nacelle.h0_rear = 1.35
    nacelle.h0_front = 1.7

    # common site conditions
    shearExp = 0.2
    rotor.cdf_reference_height_wind_speed = 90.0
    aep_a.array_losses = 0.15
    aep_a.other_losses = 0.0
    aep_a.availability = 0.98
    rotor.turbulence_class = 'B'
    tipspeed.bos_a.multiplier = 2.23

    rotor.shearExp = shearExp
    tower.wind1.shearExp = shearExp
    tower.wind2.shearExp = shearExp

    # optimization options
    wind_class = 'I'  # if offshore remember to change BOS
    rotor.control.ratedPower = 5e6
    downwind = False
    optimize_precurve = False
    # dir = 'classIII_7MW_upwind_fill/'

    optimize_hubheight = True
    include_rotor_fatigue = False
    include_tower_fatigue = False
    use_analytic = True

    if wind_class == 'I' and rotor.control.ratedPower == 5e6:
        # Dvec = np.arange(105.0, 146.0, 10.0)
        Dvec = np.arange(110.0, 141.0, 10.0)
    elif wind_class == 'III' and rotor.control.ratedPower == 5e6:
        # Dvec = np.arange(125.0, 166.0, 10.0)
        Dvec = np.arange(130.0, 161.0, 10.0)
    elif wind_class == 'III' and rotor.control.ratedPower == 7e6:
        # Dvec = np.arange(135.0, 176.0, 10.0)
        Dvec = np.arange(140.0, 171.0, 10.0)


    if optimize_precurve:
        # include_rotor_fatigue = True
        # tipspeed.driver.add_constraint('rotor.curvatureOutput > -0.1')
        Dvec = [145.0]


    if wind_class == 'I':
        rotor.turbine_class = 'I'

    elif wind_class == 'III':
        rotor.turbine_class = 'III'

        tower.M_DEL = 1.028713178 * 1e3*np.array([7.8792E+003, 7.7507E+003, 7.4918E+003, 7.2389E+003, 6.9815E+003, 6.7262E+003, 6.4730E+003, 6.2174E+003, 5.9615E+003, 5.7073E+003, 5.4591E+003, 5.2141E+003, 4.9741E+003, 4.7399E+003, 4.5117E+003, 4.2840E+003, 4.0606E+003, 3.8360E+003, 3.6118E+003, 3.3911E+003, 3.1723E+003, 2.9568E+003, 2.7391E+003, 2.5294E+003, 2.3229E+003, 2.1246E+003, 1.9321E+003, 1.7475E+003, 1.5790E+003, 1.4286E+003, 1.3101E+003, 1.2257E+003, 1.1787E+003, 1.1727E+003, 1.1821E+003])

        rotor.Mxb_damage = 1e3*np.array([2.3617E+003, 2.0751E+003, 1.8051E+003, 1.5631E+003, 1.2994E+003, 1.0388E+003, 8.1384E+002, 6.2492E+002, 4.6916E+002, 3.4078E+002, 2.3916E+002, 1.5916E+002, 9.9752E+001, 5.6139E+001, 2.6492E+001, 1.0886E+001, 3.7210E+000, 4.3206E-001])
        rotor.Myb_damage = 1e3*np.array([2.5492E+003, 2.6261E+003, 2.4265E+003, 2.2308E+003, 1.9882E+003, 1.7184E+003, 1.4438E+003, 1.1925E+003, 9.6251E+002, 7.5564E+002, 5.7332E+002, 4.1435E+002, 2.8036E+002, 1.7106E+002, 8.7732E+001, 3.8678E+001, 1.3942E+001, 1.6600E+000])


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
        tipspeed.bos_a.multiplier = 2.33
        tipspeed.fin_a.fixed_charge_rate = 0.118


        depth = 20.0

        tipspeed.sea_depth = depth
        tipspeed.offshore = True
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

    tipspeed.run()
    print 'mass rotor blades (kg) =', tipspeed.rotor.mass_all_blades
    print 'mass hub system (kg) =', tipspeed.hub.hub_system_mass
    print 'mass nacelle (kg) =', tipspeed.nacelle.nacelle_mass
    print 'mass tower (kg) =', tipspeed.tower.mass
    print 'maximum tip deflection (m) =', tipspeed.maxdeflection.max_tip_deflection
    print 'ground clearance (m) =', tipspeed.maxdeflection.ground_clearance

if __name__ == '__main__':

    example()
