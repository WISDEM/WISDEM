from openmdao.api import Group, Component, IndepVarComp, DirectSolver, ScipyGMRES, Newton, NLGaussSeidel, Brent, RunOnce
from floatingse.floating import FloatingSE
from offshorebos.wind_obos_component import WindOBOS
from plant_financese.plant_finance import PlantFinance
from rotorse.rotor import RotorSE
from commonse.rna import RNA
from commonse.turbine_constraints import TurbineConstraints
from turbine_costsse.turbine_costsse_2015 import Turbine_CostsSE_2015

import numpy as np

        
class FloatingTurbine(Group):

    def __init__(self, RefBlade, nSection):
        super(FloatingTurbine, self).__init__()

        self.add('hub_height', IndepVarComp('hub_height', 0.0), promotes=['*'])

        # TODO: 
        #Weibull/Rayleigh CDF
        
        # Rotor
        self.add('rotor', RotorSE(RefBlade), promotes=['*'])

        # RNA
        self.add('rna', RNA(1), promotes=['downwind'])
        
        # Tower and substructure
        myfloat = FloatingSE(nSection)
        self.add('sm', myfloat, promotes=['*'])

        # Turbine constraints
        self.add('tcons', TurbineConstraints(myfloat.nFull), promotes=['*'])#blade_number','Rtip','precurveTip','presweepTip','precone','tilt','tip_deflection','downwind'])
        
        # Turbine costs
        self.add('tcost', Turbine_CostsSE_2015(), promotes=['*'])
        
        # Balance of station
        self.add('wobos', WindOBOS(), promotes=['*'])

        # LCOE Calculation
        self.add('lcoe', PlantFinance(), promotes=['*'])

        # Define all input variables from all models
        self.add('offshore',             IndepVarComp('offshore', True, pass_by_obj=True), promotes=['*'])
        self.add('crane',                IndepVarComp('crane', False, pass_by_obj=True), promotes=['*'])

        # Turbine Costs
        # REMOVE ONCE DRIVESE AND GENERATORSE ARE CONNECTED
        self.add('bearing_number',           IndepVarComp('bearing_number', 0, pass_by_obj=True), promotes=['*'])
        self.add('bedplate_mass',            IndepVarComp('bedplate_mass', 0.0), promotes=['*'])
        self.add('crane_cost',               IndepVarComp('crane_cost', 0.0), promotes=['*'])
        self.add('gearbox_mass',             IndepVarComp('gearbox_mass', 0.0), promotes=['*'])
        self.add('generator_mass',           IndepVarComp('generator_mass', 0.0), promotes=['*'])
        self.add('hss_mass',     IndepVarComp('hss_mass', 0.0), promotes=['*'])
        self.add('hvac_mass',   IndepVarComp('hvac_mass', 0.0), promotes=['*'])
        self.add('lss_mass',     IndepVarComp('lss_mass', 0.0), promotes=['*'])
        self.add('main_bearing_mass',        IndepVarComp('main_bearing_mass', 0.0), promotes=['*'])
        self.add('cover_mass',       IndepVarComp('cover_mass', 0.0), promotes=['*'])
        self.add('platforms_mass',   IndepVarComp('platforms_mass', 0.0), promotes=['*'])
        self.add('pitch_system_mass',        IndepVarComp('pitch_system_mass', 0.0), promotes=['*'])
        self.add('spinner_mass',             IndepVarComp('spinner_mass', 0.0), promotes=['*'])
        self.add('transformer_mass',         IndepVarComp('transformer_mass', 0.0), promotes=['*'])
        self.add('vs_electronics_mass', IndepVarComp('vs_electronics_mass', 0.0), promotes=['*'])
        self.add('yaw_mass',          IndepVarComp('yaw_mass', 0.0), promotes=['*'])
        
        # Tower
        #self.add('stress_standard_value',          IndepVarComp('stress_standard_value', 0.0), promotes=['*'])
        #self.add('fatigue_parameters',             IndepVarComp('fatigue_parameters', np.zeros(nDEL)), promotes=['*'])
        #self.add('fatigue_z',                      IndepVarComp('fatigue_z', np.zeros(nDEL)), promotes=['*'])
        #self.add('frame3dd_matrix_method',         IndepVarComp('frame3dd_matrix_method', 0, pass_by_obj=True), promotes=['*'])
        #self.add('compute_stiffnes',               IndepVarComp('compute_stiffnes', False, pass_by_obj=True), promotes=['*'])
        self.add('project_lifetime',               IndepVarComp('project_lifetime', 0.0), promotes=['*'])
        #self.add('lumped_mass_matrix',             IndepVarComp('lumped_mass_matrix', 0, pass_by_obj=True), promotes=['*'])
        #self.add('slope_SN',                       IndepVarComp('slope_SN', 0, pass_by_obj=True), promotes=['*'])
        #self.add('number_of_modes',                IndepVarComp('number_of_modes', 0, pass_by_obj=True), promotes=['*'])
        #self.add('compute_shear',                  IndepVarComp('compute_shear', True, pass_by_obj=True), promotes=['*'])
        #self.add('shift_value',                    IndepVarComp('shift_value', 0.0), promotes=['*'])
        #self.add('frame3dd_convergence_tolerance', IndepVarComp('frame3dd_convergence_tolerance', 1e-9), promotes=['*'])

        # TODO: Multiple load cases
        # Environment
        self.add('air_density',                IndepVarComp('air_density', 0.0), promotes=['*'])
        self.add('air_viscosity',              IndepVarComp('air_viscosity', 0.0), promotes=['*'])
        self.add('wind_reference_speed',       IndepVarComp('wind_reference_speed', 0.0), promotes=['*'])
        self.add('wind_reference_height',      IndepVarComp('wind_reference_height', 0.0), promotes=['*'])
        self.add('shearExp',                      IndepVarComp('shearExp', 0.0), promotes=['*'])
        self.add('wind_bottom_height',         IndepVarComp('wind_bottom_height', 0.0), promotes=['*'])
        self.add('wind_beta',                  IndepVarComp('wind_beta', 0.0), promotes=['*'])
        self.add('cd_usr',                     IndepVarComp('cd_usr', np.inf), promotes=['*'])

        # Environment
        self.add('water_depth',                IndepVarComp('water_depth', 0.0), promotes=['*'])
        self.add('water_density',              IndepVarComp('water_density', 0.0), promotes=['*'])
        self.add('water_viscosity',            IndepVarComp('water_viscosity', 0.0), promotes=['*'])
        self.add('wave_height',                IndepVarComp('wave_height', 0.0), promotes=['*'])
        self.add('wave_period',                IndepVarComp('wave_period', 0.0), promotes=['*'])
        self.add('mean_current_speed',          IndepVarComp('mean_current_speed', 0.0), promotes=['*'])
        #self.add('wave_beta',                  IndepVarComp('wave_beta', 0.0), promotes=['*'])
        #self.add('wave_velocity_z0',           IndepVarComp('wave_velocity_z0', 0.0), promotes=['*'])
        #self.add('wave_acceleration_z0',       IndepVarComp('wave_acceleration_z0', 0.0), promotes=['*'])
        
        # Design standards
        self.add('gamma_freq',      IndepVarComp('gamma_freq', 0.0), promotes=['*'])
        self.add('gamma_f',         IndepVarComp('gamma_f', 0.0), promotes=['*'])
        self.add('gamma_m',         IndepVarComp('gamma_m', 0.0), promotes=['*'])
        self.add('gamma_b',         IndepVarComp('gamma_b', 0.0), promotes=['*'])
        self.add('gamma_fatigue',   IndepVarComp('gamma_fatigue', 0.0), promotes=['*'])
        self.add('gamma_n',         IndepVarComp('gamma_n', 0.0), promotes=['*'])

        # RNA
        self.add('dummy_mass',                 IndepVarComp('dummy_mass', 0.0), promotes=['*'])
        self.add('hub_mass',                   IndepVarComp('hub_mass', 0.0), promotes=['*'])
        self.add('nac_mass',                   IndepVarComp('nac_mass', 0.0), promotes=['*'])
        self.add('hub_cm',                     IndepVarComp('hub_cm', np.zeros((3,))), promotes=['*'])
        self.add('nac_cm',                     IndepVarComp('nac_cm', np.zeros((3,))), promotes=['*'])
        self.add('hub_I',                      IndepVarComp('hub_I', np.zeros(6)), promotes=['*'])
        self.add('nac_I',                      IndepVarComp('nac_I', np.zeros(6)), promotes=['*'])
        self.add('downwind',                   IndepVarComp('downwind', False, pass_by_obj=True), promotes=['*'])
        self.add('rna_weightM',                IndepVarComp('rna_weightM', True, pass_by_obj=True), promotes=['*'])
        
        # Column
        self.add('morison_mass_coefficient',   IndepVarComp('morison_mass_coefficient', 0.0), promotes=['*'])
        self.add('material_density',           IndepVarComp('material_density', 0.0), promotes=['*'])
        self.add('E',                          IndepVarComp('E', 0.0), promotes=['*'])
        self.add('nu',                         IndepVarComp('nu', 0.0), promotes=['*'])
        self.add('yield_stress',               IndepVarComp('yield_stress', 0.0), promotes=['*'])

        # Pontoons
        self.add('G',                          IndepVarComp('G', 0.0), promotes=['*'])
        
        # LCOE
        self.add('number_of_turbines', IndepVarComp('number_of_turbines', 0, pass_by_obj=True), promotes=['*'])
        self.add('annual_opex',        IndepVarComp('annual_opex', 0.0), promotes=['*']) # TODO: Replace with output connection
        self.add('fixed_charge_rate',  IndepVarComp('fixed_charge_rate', 0.0), promotes=['*'])
        self.add('discount_rate',      IndepVarComp('discount_rate', 0.0), promotes=['*'])

        # Connect all input variables from all models
        self.connect('water_depth', ['waterD', 'sea_depth'])
        self.connect('hub_height', 'hubH')
        self.connect('tower_outer_diameter', 'towerD', src_indices=[0])
        
        self.connect('wind_beta', 'beta')
        self.connect('mean_current_speed', 'Uc')
        
        self.connect('project_lifetime', ['struc.lifetime','projLife'])
        #self.connect('slope_SN', 'm_SN')
        #self.connect('compute_shear', 'shear')
        #self.connect('compute_stiffnes', 'geom')
        #self.connect('frame3dd_matrix_method', 'Mmethod')
        #self.connect('shift_value', 'shift')
        # TODO:
        #self.connect('number_of_modes', ['nM', 'nF'])
        #self.connect('frame3dd_convergence_tolerance', 'tol')
        #self.connect('lumped_mass_matrix', 'lump')
        #self.connect('stress_standard_value', 'DC')
        
        self.connect('rna.loads.top_F', 'rna_force')
        self.connect('rna.loads.top_M', 'rna_moment')
        self.connect('rna.rna_I_TT', 'rna_I')
        self.connect('rna.rna_mass', ['rnaM', 'rna_mass'])
        self.connect('rna.rna_cm', 'rna_cg')
        self.connect('mass_all_blades', 'rna.blades_mass')
        self.connect('I_all_blades', 'rna.blades_I')
        self.connect('hub_mass', 'rna.hub_mass')
        self.connect('nac_mass', 'rna.nac_mass')
        self.connect('hub_cm', ['rna.hub_cm','hub_tt'])
        self.connect('nac_cm', 'rna.nac_cm')
        self.connect('hub_I', 'rna.hub_I')
        self.connect('nac_I', 'rna.nac_I')
        self.connect('tilt', 'rna.tilt')
        self.connect('Fxyz_total','rna.loads.F')
        self.connect('Mxyz_total','rna.loads.M')
        self.connect('rna_weightM','rna.rna_weightM')
        
        self.connect('air_density', ['base.windLoads.rho','analysis.rho'])
        self.connect('air_viscosity', ['base.windLoads.mu','analysis.mu'])
        self.connect('water_density', 'water_density')
        self.connect('water_viscosity', 'base.waveLoads.mu')
        self.connect('wave_height', 'Hs')
        self.connect('wave_period', 'T')
        self.connect('wind_reference_speed', 'Uref')
        self.connect('wind_reference_height', ['zref','wind.zref'])
        self.connect('wind_bottom_height', ['z0','wind.z0'])
        self.connect('shearExp', 'wind.shearExp')
        self.connect('morison_mass_coefficient', 'cm')
        
        self.connect('ballast_cost_rate', 'ballCR')
        #self.connect('drag_embedment_extra_length', 'deaFixLeng')
        self.connect('mooring_cost_rate', 'moorCR')
        self.connect('mooring_cost', 'moorCost')
        self.connect('mooring_diameter', 'moorDia')
        self.connect('number_of_mooring_lines', 'moorLines')
        self.connect('outfitting_cost_rate', 'sSteelCR')
        self.connect('tapered_col_cost_rate', ['spStifColCR', 'spTapColCR', 'ssStifColCR'])
        self.connect('pontoon_cost_rate', 'ssTrussCR')

        self.connect('nBlades','blade_number')
        self.connect('mass_one_blade', 'blade_mass')
        self.connect('control_maxOmega', 'rotor_omega')
        self.connect('structural_frequencies', 'tower_freq')
        self.connect('tow.z_full', 'tower_z')
        self.connect('tow.d_full', 'tower_d')
        
        self.connect('turbine_cost_kW', 'turbCapEx')
        self.connect('machine_rating', 'turbR')
        self.connect('diameter', 'rotorD')
        self.connect('bladeLength', 'bladeL')
        self.connect('hub_diameter', 'hubD')
        
        # Link outputs from one model to inputs to another
        self.connect('tower_mass', 'towerM')
        self.connect('dummy_mass', 'aux.stack_mass_in')

        self.connect('total_cost', 'subTotCost')
        self.connect('total_mass', 'subTotM')
        self.connect('total_bos_cost', 'bos_costs')

        self.connect('number_of_turbines', ['nTurb', 'turbine_number'])
        self.connect('annual_opex', 'avg_annual_opex')
        self.connect('AEP', 'net_aep')
        self.connect('totInstTime', 'construction_time')
        
         # Use complex number finite differences
        typeStr = 'fd'
        formStr = 'central'
        stepVal = 1e-5
        stepStr = 'relative'
        
        self.deriv_options['type'] = typeStr
        self.deriv_options['form'] = formStr
        self.deriv_options['step_size'] = stepVal
        self.deriv_options['step_calc'] = stepStr

