from fusedwind.fused_openmdao import FUSED_Component, FUSED_Group, FUSED_add, FUSED_connect, FUSED_print, \
                                     FUSED_Problem, FUSED_setup, FUSED_run, FUSED_VarComp
from openmdao.api import Group, Component, IndepVarComp, Problem, ScipyOptimizer, DumpRecorder
from offshorebos.wind_obos_component import WindOBOS
from rotorse.rotor import RotorSE
from towerse.tower import TowerSE
from commonse import NFREQ
from commonse.rna import RNA, RNAMass
from commonse.environment import PowerWind, LogWind
from commonse.turbine_constraints import TurbineConstraints
from turbine_costsse.turbine_costsse_2015 import Turbine_CostsSE_2015
from plant_financese.plant_finance import PlantFinance
from drivese.drivese_omdao import Drive3pt, Drive4pt

import numpy as np
from .monopile_assembly import wind, nLC, nDEL, NSECTION

    
class MonopileTurbine(Group):

    def __init__(self, RefBlade):
        super(MonopileTurbine, self).__init__()
        nFull   = 5*(NSECTION) + 1

        self.add('hub_height', IndepVarComp('hub_height', 0.0), promotes=['*'])
        self.add('foundation_height', IndepVarComp('foundation_height', 0.0), promotes=['*'])

        
        # Rotor
        self.add('rotor', RotorSE(RefBlade), promotes=['*'])#hubFraction','nBlades','turbine_class','sparT_in','teT_in','bladeLength','precone','tilt','yaw',
        '''
                                                       'r_max_chord','chord_in','theta_in','precurve_in','presweep_in','precurve_tip','presweep_tip',
                                                       'turbulence_class','gust_stddev','VfactorPC','shape_parameter','Rtip','precurveTip','presweepTip',
                                                       'control_Vin','control_Vout','machine_rating','control_minOmega','control_maxOmega',
                                                       'control_tsr','control_pitch','pitch_extreme','azimuth_extreme','drivetrainType',
                                                       'rstar_damage','Mxb_damage','Myb_damage','strain_ult_spar','strain_ult_te','m_damage',
                                                       'nSector','tiploss','hubloss','wakerotation','usecd','AEP_loss_factor','tip_deflection',
                                                       'dynamic_amplication_tip_deflection'])
        '''

        # RNA
        self.add('rna', RNA(nLC), promotes=['*'])
        
        # Tower and substructure
        self.add('tow',TowerSE(nLC, NSECTION+1, nFull, nDEL, wind='PowerWind'), promotes=['material_density','E','G','tower_section_height',
                                                                                          'tower_outer_diameter','tower_wall_thickness',
                                                                                          'tower_outfitting_factor','tower_buckling_length',
                                                                                          'max_taper','min_d_to_t','rna_mass','rna_cg','rna_I',
                                                                                          'tower_mass','tower_I_base','hub_height',
                                                                                          'foundation_height','monopile','soil_G','soil_nu',
                                                                                          'suctionpile_depth','gamma_f','gamma_m','gamma_b','gamma_n','gamma_fatigue'
        ])
                 
        # Turbine constraints
        self.add('tcons', TurbineConstraints(nFull), promotes=['*'])#blade_number','Rtip','precurveTip','presweepTip','precone','tilt','tip_deflection','downwind'])
        
        # Turbine costs
        self.add('tcost', Turbine_CostsSE_2015(), promotes=['*'])
        
        # Balance of station
        self.add('wobos', WindOBOS(), promotes=['*'])#tax_rate'])

        # LCOE Calculation
        self.add('lcoe', PlantFinance(), promotes=['*'])

        # Define all input variables from all models
        self.add('offshore',             IndepVarComp('offshore', True, pass_by_obj=True), promotes=['*'])
        self.add('crane',                IndepVarComp('crane', False, pass_by_obj=True), promotes=['*'])

        # Turbine Costs
        # REMOVE ONCE DRIVESE AND GENERATORSE ARE CONNECTED
        self.add('bearing_number',           IndepVarComp('bearing_number', 0, pass_by_obj=True), promotes=['*'])
        self.add('hub_mass',            IndepVarComp('hub_mass', 0.0), promotes=['*'])
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
        
        # Tower and Frame3DD options
        self.add('stress_standard_value',          IndepVarComp('stress_standard_value', 0.0), promotes=['*'])
        self.add('frame3dd_matrix_method',         IndepVarComp('frame3dd_matrix_method', 0, pass_by_obj=True), promotes=['*'])
        self.add('compute_stiffness',              IndepVarComp('compute_stiffness', False, pass_by_obj=True), promotes=['*'])
        self.add('project_lifetime',               IndepVarComp('project_lifetime', 0.0), promotes=['*'])
        self.add('lumped_mass_matrix',             IndepVarComp('lumped_mass_matrix', 0, pass_by_obj=True), promotes=['*'])
        self.add('slope_SN',                       IndepVarComp('slope_SN', 0, pass_by_obj=True), promotes=['*'])
        self.add('number_of_modes',                IndepVarComp('number_of_modes', NFREQ, pass_by_obj=True), promotes=['*'])
        self.add('compute_shear',                  IndepVarComp('compute_shear', True, pass_by_obj=True), promotes=['*'])
        self.add('shift_value',                    IndepVarComp('shift_value', 0.0), promotes=['*'])
        self.add('frame3dd_convergence_tolerance', IndepVarComp('frame3dd_convergence_tolerance', 1e-7), promotes=['*'])
        self.add('max_taper_ratio',                IndepVarComp('max_taper_ratio', 0.0), promotes=['*'])
        self.add('min_diameter_thickness_ratio',   IndepVarComp('min_diameter_thickness_ratio', 0.0), promotes=['*'])
        
        # Environment
        self.add('air_density',                IndepVarComp('air_density', 0.0), promotes=['*'])
        self.add('air_viscosity',              IndepVarComp('air_viscosity', 0.0), promotes=['*'])
        self.add('wind_reference_speed',       IndepVarComp('wind_reference_speed', 0.0), promotes=['*'])
        self.add('wind_reference_height',      IndepVarComp('wind_reference_height', 0.0), promotes=['*'])
        self.add('shearExp',                   IndepVarComp('shearExp', 0.0), promotes=['*'])
        self.add('wind_bottom_height',         IndepVarComp('wind_bottom_height', 0.0), promotes=['*'])
        self.add('wind_beta',                  IndepVarComp('wind_beta', 0.0), promotes=['*'])
        self.add('cd_usr',                     IndepVarComp('cd_usr', np.inf), promotes=['*'])

        # Environment
        self.add('water_depth',                IndepVarComp('water_depth', 0.0), promotes=['*'])
        self.add('water_density',              IndepVarComp('water_density', 0.0), promotes=['*'])
        self.add('water_viscosity',            IndepVarComp('water_viscosity', 0.0), promotes=['*'])
        self.add('wave_height',                IndepVarComp('wave_height', 0.0), promotes=['*'])
        self.add('wave_period',                IndepVarComp('wave_period', 0.0), promotes=['*'])
        self.add('mean_current_speed',         IndepVarComp('mean_current_speed', 0.0), promotes=['*'])
        self.add('wave_beta',                  IndepVarComp('wave_beta', 0.0), promotes=['*'])
        
        # Design standards
        self.add('gamma_freq',      IndepVarComp('gamma_freq', 0.0), promotes=['*'])
        self.add('gamma_f',         IndepVarComp('gamma_f', 0.0), promotes=['*'])
        self.add('gamma_m',         IndepVarComp('gamma_m', 0.0), promotes=['*'])
        self.add('gamma_b',         IndepVarComp('gamma_b', 0.0), promotes=['*'])
        self.add('gamma_fatigue',   IndepVarComp('gamma_fatigue', 0.0), promotes=['*'])
        self.add('gamma_n',         IndepVarComp('gamma_n', 0.0), promotes=['*'])

        # RNA
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
        self.add('yield_stress',               IndepVarComp('yield_stress', 0.0), promotes=['*'])

        # Pontoons
        self.add('G',                          IndepVarComp('G', 0.0), promotes=['*'])
        
        # LCOE
        self.add('number_of_turbines', IndepVarComp('number_of_turbines', 0, pass_by_obj=True), promotes=['*'])
        self.add('annual_opex',        IndepVarComp('annual_opex', 0.0), promotes=['*']) # TODO: Replace with output connection
        self.add('fixed_charge_rate',  IndepVarComp('fixed_charge_rate', 0.0), promotes=['*'])
        self.add('discount_rate',      IndepVarComp('discount_rate', 0.0), promotes=['*'])

        
        # Connect all input variables from all models
        self.connect('water_depth', ['tow.z_floor','waterD', 'sea_depth','mpileL'])
        self.connect('hub_height', ['hub_height', 'hubH'])
        self.connect('tower_outer_diameter', 'towerD', src_indices=[NSECTION])
        self.connect('tower_outer_diameter', 'mpileD', src_indices=[0])
        self.connect('suctionpile_depth', 'mpEmbedL')
        
        self.connect('wind_beta', 'tow.windLoads.beta')
        self.connect('wave_beta', 'tow.waveLoads.beta')
        self.connect('cd_usr', 'tow.cd_usr')
        self.connect('yaw', 'tow.distLoads.yaw')
        self.connect('mean_current_speed', 'tow.wave.Uc')
        
        self.connect('project_lifetime', ['struc.lifetime','tow.life', 'projLife'])
        self.connect('number_of_modes', 'tow.nM')
        self.connect('frame3dd_convergence_tolerance', 'tow.tol')
        self.connect('lumped_mass_matrix', 'tow.lump')
        self.connect('stress_standard_value', 'tow.DC')
        self.connect('shift_value', 'tow.shift')
        self.connect('compute_shear', 'tow.shear')
        self.connect('slope_SN', 'tow.m_SN')
        self.connect('compute_stiffness', 'tow.geom')
        self.connect('frame3dd_matrix_method', 'tow.Mmethod')
        
        self.connect('loads.top_F', 'tow.pre.rna_F')
        self.connect('loads.top_M', 'tow.pre.rna_M')
        self.connect('rna_I_TT', 'rna_I')
        self.connect('rna_mass', 'rnaM')
        self.connect('rna_cm', 'rna_cg')
        self.connect('mass_all_blades', 'blades_mass')
        self.connect('I_all_blades', 'blades_I')
        self.connect('hub_cm', 'hub_tt')
        self.connect('Fxyz_total','loads.F')
        self.connect('Mxyz_total','loads.M')
        
        self.connect('air_density', ['tow.windLoads.rho','analysis.rho'])
        self.connect('air_viscosity', ['tow.windLoads.mu','analysis.mu'])
        self.connect('water_density',['tow.wave.rho','tow.waveLoads.rho'])
        self.connect('water_viscosity', 'tow.waveLoads.mu')
        self.connect('wave_height', 'tow.wave.hmax')
        self.connect('wave_period', 'tow.wave.T')
        self.connect('wind_reference_speed', 'tow.wind.Uref')
        self.connect('wind_reference_height', ['tow.wind.zref','wind.zref'])
        self.connect('wind_bottom_height', ['tow.z0','wind.z0'])
        self.connect('shearExp', ['tow.wind.shearExp', 'wind.shearExp'])
        self.connect('morison_mass_coefficient', 'tow.cm')

        self.connect('yield_stress', 'tow.sigma_y')
        self.connect('max_taper_ratio', 'max_taper')
        self.connect('min_diameter_thickness_ratio', 'min_d_to_t')

        self.connect('nBlades','blade_number')
        self.connect('mass_one_blade', 'blade_mass')
        self.connect('control_maxOmega', 'rotor_omega')
        self.connect('tow.post.structural_frequencies', 'tower_freq')
        self.connect('tow.z_full', 'tower_z')
        self.connect('tow.d_full', 'tower_d')
        
        self.connect('turbine_cost_kW', 'turbCapEx')
        self.connect('machine_rating', 'turbR')
        self.connect('diameter', 'rotorD')
        self.connect('bladeLength', 'bladeL')
        self.connect('hub_diameter', 'hubD')
        
        # Link outputs from one model to inputs to another
        self.connect('tower_mass', 'towerM')

        self.connect('tower_cost', 'subTotCost')
        self.connect('tower_mass', 'subTotM')
        self.connect('total_bos_cost', 'bos_costs')

        self.connect('number_of_turbines', ['nTurb', 'turbine_number'])
        self.connect('annual_opex', 'avg_annual_opex')
        self.connect('AEP', 'net_aep')
        self.connect('totInstTime', 'construction_time')
        
         # Use complex number finite differences
        self.deriv_options['type'] = 'fd'
        self.deriv_options['form'] = 'central'
        self.deriv_options['step_size'] = 1e-5
        self.deriv_options['step_calc'] = 'relative'

