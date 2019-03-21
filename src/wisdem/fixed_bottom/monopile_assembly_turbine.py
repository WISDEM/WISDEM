from fusedwind.fused_openmdao import FUSED_Component, FUSED_Group, FUSED_add, FUSED_connect, FUSED_print, \
                                     FUSED_Problem, FUSED_setup, FUSED_run, FUSED_VarComp
from openmdao.api import Group, Component, IndepVarComp, Problem, ScipyOptimizer, DumpRecorder
from offshorebos.wind_obos_component import WindOBOS
import offshorebos.wind_obos as wind_obos
from rotorse.rotor import RotorSE, DTU10MW, NREL5MW, TURBULENCE_CLASS
from towerse.tower import TowerSE
from commonse import NFREQ
from commonse.rna import RNA
from commonse.environment import PowerWind, LogWind
from commonse.turbine_constraints import TurbineConstraints
from turbine_costsse.turbine_costsse_2015 import Turbine_CostsSE_2015
from plant_financese.plant_finance import PlantFinance
from drivese.drivese_omdao import Drive3pt, Drive4pt

import numpy as np
from wisdem.fixed_bottom.monopile_assembly import wind, nLC, nDEL, NSECTION
# Helpful for finding warnings in numpy or scipy functions
#np.seterr(all='raise')
    
class MonopileTurbine(Group):

    def __init__(self, RefBlade):
        super(MonopileTurbine, self).__init__()
        nFull   = 5*(NSECTION) + 1

        self.add('hub_height', IndepVarComp('hub_height', 0.0), promotes=['*'])
        self.add('foundation_height', IndepVarComp('foundation_height', 0.0), promotes=['*'])

        
        # Rotor
        self.add('rotor', RotorSE(RefBlade, regulation_reg_II5=False, regulation_reg_III=True, Analysis_Level=-1), promotes=['*'])

        # RNA
        self.add('drive', Drive4pt('CARB', 'SRB', 'B', 'eep', 'normal', 'geared', True, 0, True, 3),
                 promotes=['bedplate_mass','gearbox_mass','generator_mass','hss_mass','hvac_mass','lss_mass','cover_mass',
                           'pitch_system_mass','platforms_mass','spinner_mass','transformer_mass','vs_electronics_mass','yaw_mass'])
        self.add('rna', RNA(nLC), promotes=['*'])
        
        # Tower and substructure
        self.add('tow',TowerSE(nLC, NSECTION+1, nFull, nDEL, wind='PowerWind'), promotes=['material_density','E','G','tower_section_height',
                                                                                          'tower_outer_diameter','tower_wall_thickness',
                                                                                          'tower_outfitting_factor','tower_buckling_length','downwind',
                                                                                          'max_taper','min_d_to_t','rna_mass','rna_cg','rna_I','hub_cm',
                                                                                          'tower_mass','tower_I_base','hub_height','tip_position',
                                                                                          'foundation_height','monopile','soil_G','soil_nu',
                                                                                          'suctionpile_depth','tip_deflection_margin',
                                                                                          'gamma_f','gamma_m','gamma_b','gamma_n','gamma_fatigue',
                                                                                          'labor_cost_rate','material_cost_rate','painting_cost_rate'])
                 
        # Turbine constraints
        self.add('tcons', TurbineConstraints(nFull), promotes=['*'])
        
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
        #self.add('hub_mass',            IndepVarComp('hub_mass', 0.0), promotes=['*'])
        #self.add('bedplate_mass',            IndepVarComp('bedplate_mass', 0.0), promotes=['*'])
        self.add('crane_cost',               IndepVarComp('crane_cost', 0.0), promotes=['*'])
        #self.add('gearbox_mass',             IndepVarComp('gearbox_mass', 0.0), promotes=['*'])
        #self.add('generator_mass',           IndepVarComp('generator_mass', 0.0), promotes=['*'])
        #self.add('hss_mass',     IndepVarComp('hss_mass', 0.0), promotes=['*'])
        #self.add('hvac_mass',   IndepVarComp('hvac_mass', 0.0), promotes=['*'])
        #self.add('lss_mass',     IndepVarComp('lss_mass', 0.0), promotes=['*'])
        #self.add('main_bearing_mass',        IndepVarComp('main_bearing_mass', 0.0), promotes=['*'])
        #self.add('cover_mass',       IndepVarComp('cover_mass', 0.0), promotes=['*'])
        #self.add('platforms_mass',   IndepVarComp('platforms_mass', 0.0), promotes=['*'])
        #self.add('pitch_system_mass',        IndepVarComp('pitch_system_mass', 0.0), promotes=['*'])
        #self.add('spinner_mass',             IndepVarComp('spinner_mass', 0.0), promotes=['*'])
        #self.add('transformer_mass',         IndepVarComp('transformer_mass', 0.0), promotes=['*'])
        #self.add('vs_electronics_mass', IndepVarComp('vs_electronics_mass', 0.0), promotes=['*'])
        #self.add('yaw_mass',          IndepVarComp('yaw_mass', 0.0), promotes=['*'])
        
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
        #self.add('air_density',                IndepVarComp('air_density', 0.0), promotes=['*'])
        #self.add('air_viscosity',              IndepVarComp('air_viscosity', 0.0), promotes=['*'])
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
        #self.add('nac_mass',                   IndepVarComp('nac_mass', 0.0), promotes=['*'])
        #self.add('hub_cm',                     IndepVarComp('hub_cm', np.zeros((3,))), promotes=['*'])
        #self.add('nac_cm',                     IndepVarComp('nac_cm', np.zeros((3,))), promotes=['*'])
        #self.add('hub_I',                      IndepVarComp('hub_I', np.zeros(6)), promotes=['*'])
        #self.add('nac_I',                      IndepVarComp('nac_I', np.zeros(6)), promotes=['*'])
        self.add('rna_weightM',                IndepVarComp('rna_weightM', True, pass_by_obj=True), promotes=['*'])
        
        # Column
        self.add('morison_mass_coefficient',   IndepVarComp('morison_mass_coefficient', 0.0), promotes=['*'])
        self.add('material_density',           IndepVarComp('material_density', 0.0), promotes=['*'])
        self.add('E',                          IndepVarComp('E', 0.0), promotes=['*'])
        self.add('yield_stress',               IndepVarComp('yield_stress', 0.0), promotes=['*'])

        # Pontoons
        self.add('G',                          IndepVarComp('G', 0.0), promotes=['*'])
        
        # LCOE
        self.add('labor_cost_rate',      IndepVarComp('labor_cost_rate', 0.0), promotes=['*'])
        self.add('material_cost_rate',      IndepVarComp('material_cost_rate', 0.0), promotes=['*'])
        self.add('painting_cost_rate',      IndepVarComp('painting_cost_rate', 0.0), promotes=['*'])
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

        self.connect('diameter','drive.rotor_diameter')
        self.connect('rated_Q','drive.rotor_torque')
        self.connect('rated_Omega','drive.rotor_rpm')
        self.connect('Mxyz_total','drive.rotor_bending_moment_x', src_indices=[0])
        self.connect('Mxyz_total','drive.rotor_bending_moment_y', src_indices=[1])
        self.connect('Mxyz_total','drive.rotor_bending_moment_z', src_indices=[2])
        self.connect('Fxyz_total','drive.rotor_thrust', src_indices=[0])
        self.connect('Fxyz_total','drive.rotor_force_y', src_indices=[1])
        self.connect('Fxyz_total','drive.rotor_force_z', src_indices=[2])
        self.connect('mass_one_blade','drive.blade_mass')
        self.connect('chord', 'drive.blade_root_diameter', src_indices=[0])
        self.connect('drivetrainEff', 'drive.drivetrain_efficiency', src_indices=[0])

        self.connect('mass_all_blades', 'blades_mass')
        self.connect('drive.hub_system_mass', 'hub_mass')
        self.connect('drive.nacelle_mass', 'nac_mass')
        self.connect('I_all_blades', 'blades_I')
        self.connect('drive.hub_system_I', 'hub_I')
        self.connect('drive.nacelle_I', 'nac_I')
        self.connect('drive.hub_system_cm', 'hub_cm')
        self.connect('drive.nacelle_cm', 'nac_cm')
        self.connect('drive.mb1_mass', 'main_bearing_mass')

        #self.connect('drive.hub_system_cm', 'hub_tt')
        self.connect('Fxyz_total','loads.F')
        self.connect('Mxyz_total','loads.M')
        
        self.connect('loads.top_F', 'tow.pre.rna_F')
        self.connect('loads.top_M', 'tow.pre.rna_M')
        self.connect('rna_I_TT', 'rna_I')
        self.connect('rna_mass', 'rnaM')
        self.connect('rna_cm', 'rna_cg')
        
        self.connect('rho', 'tow.windLoads.rho') #,'powercurve.rho'])
        self.connect('mu', 'tow.windLoads.mu')#,'powercurve.mu'])
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
        #self.connect('tow.z_full', 'tower_z')
        #self.connect('tow.d_full', 'tower_d')
        
        self.connect('turbine_cost_kW', 'turbCapEx')
        self.connect('machine_rating', ['turbR','drive.machine_rating'])
        self.connect('diameter', 'rotorD')
        self.connect('bladeLength', 'bladeL')
        self.connect('hub_diameter', 'hubD')
        
        # Link outputs from one model to inputs to another
        self.connect('tower_mass', 'towerM')

        self.connect('material_cost_rate', 'sSteelCR')
        self.connect('tower_cost', 'subTotCost')
        self.connect('tower_mass', 'subTotM')
        self.connect('total_bos_cost', 'bos_costs')

        self.connect('number_of_turbines', ['nTurb', 'turbine_number'])
        self.connect('annual_opex', 'avg_annual_opex')
        self.connect('AEP', 'net_aep')
        self.connect('totInstTime', 'construction_time')

        # TODO Compare Rotor hub_diameter to Drive hub_diameter
        # TODO: Match Rotor Drivetrain Type to DriveSE Drivetrain options
        # TODO: Match Rotor CSMDrivetrain and efficiency to DriveSE
        
         # Use complex number finite differences
        self.deriv_options['type'] = 'fd'
        self.deriv_options['form'] = 'central'
        self.deriv_options['step_size'] = 1e-5
        self.deriv_options['step_calc'] = 'relative'


if __name__ == '__main__':
    optFlag = False #True
    
    # Number of sections to be used in the design
    nsection = NSECTION

    # Reference rotor design
    RefBlade = NREL5MW() #DTU10MW()
    
    # Initialize OpenMDAO problem and FloatingSE Group
    prob = Problem(root=MonopileTurbine(RefBlade))
    
    if optFlag:
        prob.driver  = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-6
        prob.driver.options['maxiter'] = 100
        # ----------------------

        # --- Objective ---
        prob.driver.add_objective('tow.tower.mass', scaler=1e-6)
        # ----------------------

        # --- Design Variables ---
        prob.driver.add_desvar('tower_section_height', lower=5.0, upper=80.0)
        prob.driver.add_desvar('tower_outer_diameter', lower=3.87, upper=30.0)
        prob.driver.add_desvar('tower_wall_thickness', lower=4e-3, upper=2e-1)
        # ----------------------

        # Recorder
        recorder = DumpRecorder('optimization.dat')
        recorder.options['record_params'] = True
        recorder.options['record_metadata'] = False
        recorder.options['record_derivs'] = False
        prob.driver.add_recorder(recorder)
        # ----------------------

        # --- Constraints ---
        prob.driver.add_constraint('tow.height_constraint', lower=-1e-2, upper=1.e-2)
        prob.driver.add_constraint('tow.post.stress', upper=1.0)
        prob.driver.add_constraint('tow.post.global_buckling', upper=1.0)
        prob.driver.add_constraint('tow.post.shell_buckling', upper=1.0)
        prob.driver.add_constraint('tow.weldability', upper=0.0)
        prob.driver.add_constraint('tow.manufacturability', lower=0.0)
        prob.driver.add_constraint('tcons.frequency1P_margin_low', upper=1.0)
        prob.driver.add_constraint('tcons.frequency1P_margin_high', lower=1.0)
        prob.driver.add_constraint('tcons.frequency3P_margin_low', upper=1.0)
        prob.driver.add_constraint('tcons.frequency3P_margin_high', lower=1.0)
        prob.driver.add_constraint('tcons.tip_deflection_ratio', upper=1.0)
        prob.driver.add_constraint('tcons.ground_clearance', lower=20.0)
        
    prob.setup()



    # Environmental parameters
    prob['water_depth']                    = 20.0
    prob['rho']                    = 1.198
    prob['mu']                  = 1.81e-5
    prob['water_density']                  = 1025.0
    prob['water_viscosity']                = 8.9e-4
    prob['wave_height']                    = 5.0
    prob['wave_period']                    = 9.8
    prob['mean_current_speed']             = 0.0
    prob['wind_reference_speed']           = 11.0
    prob['wind_reference_height']          = 90.0
    prob['shearExp']                       = 0.11
    prob['morison_mass_coefficient']       = 2.0
    prob['wind_bottom_height']             = 0.0
    prob['yaw']                            = 0.0
    prob['wind_beta']                      = 0.0
    prob['wave_beta']                      = 0.0
    prob['cd_usr']                         = np.inf

    # Encironmental constaints
    #prob['wave_period_range_low']         = 2.0
    #prob['wave_period_range_high']        = 20.0

    # Steel properties
    prob['material_density']               = 7850.0
    prob['E']                              = 200e9
    prob['G']                              = 79.3e9
    prob['yield_stress']                   = 3.45e8

    # Design constraints
    prob['max_taper_ratio']                = 0.4
    prob['min_diameter_thickness_ratio']   = 120.0

    # Safety factors
    #prob['safety_factor_frequency']        = 1.1
    #prob['safety_factor_stress']           = 1.35
    #prob['safety_factor_materials']        = 1.3
    #prob['safety_factor_buckling']         = 1.1
    #prob['safety_factor_fatigue']          = 1.35*1.3*1.0
    #prob['safety_factor_consequence']      = 1.0
    prob['gamma_fatigue'] = 1.755 # (Float): safety factor for fatigue
    prob['gamma_f'] = 1.35 # (Float): safety factor for loads/stresses
    prob['gamma_m'] = 1.3 # (Float): safety factor for materials
    prob['gamma_freq'] = 1.1 # (Float): safety factor for resonant frequencies
    prob['gamma_n'] = 1.0
    prob['gamma_b'] = 1.1
    
    # Tower
    prob['hub_height']                     = RefBlade.hub_height
    prob['foundation_height']              = -prob['water_depth']
    prob['tower_outer_diameter']           = np.linspace(10.0, 3.87, nsection+1)
    prob['tower_section_height']           = (prob['hub_height'] - prob['foundation_height']) / nsection * np.ones(nsection)
    prob['tower_wall_thickness']           = np.linspace(0.027, 0.019, nsection)
    prob['tower_buckling_length']          = 30.0
    prob['tower_outfitting_factor']        = 1.07
    prob['stress_standard_value']          = 80.0
    prob['frame3dd_convergence_tolerance'] = 1e-7
    prob['shift_value']                    = 0.0
    prob['compute_shear']                  = True
    prob['number_of_modes']                = 2
    prob['lumped_mass_matrix']             = 0
    prob['compute_stiffness']              = False
    prob['frame3dd_matrix_method']         = 1
    prob['tow.tower_force_discretization'] = 5.0

    # Plant size
    prob['project_lifetime']               = 20.0
    prob['number_of_turbines']             = 20
    prob['annual_opex']                    = 7e5
    prob['fixed_charge_rate']              = 0.12
    prob['discount_rate']                  = 0.07

    
    # For RotorSE
    prob['hubFraction']                        = RefBlade.hubFraction
    prob['bladeLength']                        = RefBlade.bladeLength
    prob['precone']                            = RefBlade.precone
    prob['tilt']                               = RefBlade.tilt
    prob['yaw']                                = 0.0 
    prob['nBlades']                            = RefBlade.nBlades
    prob['r_max_chord']                        =  RefBlade.r_max_chord 
    prob['chord_in']                           = RefBlade.chord
    prob['theta_in']                           = RefBlade.theta
    prob['precurve_in']                        = RefBlade.precurve
    prob['presweep_in']                        = RefBlade.presweep
   
    prob['sparT_in']                           = RefBlade.spar_thickness
    prob['teT_in']                             = RefBlade.te_thickness
    prob['turbine_class']                      = RefBlade.turbine_class
    prob['turbulence_class']                   = TURBULENCE_CLASS['B'] 
    prob['control_Vin']                        = RefBlade.control_Vin
    prob['control_Vout']                       = RefBlade.control_Vout
    prob['control_minOmega']                   = RefBlade.control_minOmega
    prob['control_maxOmega']                   = RefBlade.control_maxOmega
    prob['control_tsr']                        = RefBlade.control_tsr
    prob['control_pitch']                      = RefBlade.control_pitch
    prob['control_maxTS']                      = RefBlade.control_maxTS
    prob['machine_rating']                     = RefBlade.rating
    prob['pitch_extreme']                      = 0.0 
    prob['azimuth_extreme']                    = 0.0 
    prob['VfactorPC']                          = 0.7 
    prob['yaw']                                = 0.0
    prob['gust_stddev']                        = 3
    prob['strain_ult_spar']                    = 1e-2
    prob['strain_ult_te']                      = 2*2500*1e-6
    prob['m_damage']                           = 10.0
    prob['nSector']                            = 4
    prob['tiploss']                            = True
    prob['hubloss']                            = True
    prob['wakerotation']                       = True 
    prob['usecd']                              = True
    prob['AEP_loss_factor']                    = 1.0
    prob['dynamic_amplication_tip_deflection'] = 1.35
    prob['shape_parameter']                    = 0.0
    prob['rstar_damage'] = np.linspace(0,1,39)
    prob['Mxb_damage'] = 1e-10*np.ones(39)
    prob['Myb_damage'] = 1e-10*np.ones(39)
    
    # For RNA
    #prob['hub_mass']    = 105520.0
    #prob['nac_mass']    = 446036.25
    #prob['hub_cm']      = np.array([-7.1, 0.0, 2.75])
    #prob['nac_cm']      = np.array([2.69, 0.0, 2.40])
    #prob['hub_I']       = prob['hub_mass']*2.152**2. * np.r_[(2./3.), (5./12.), (5./12.), np.zeros(3)]
    #prob['nac_I']       = prob['nac_mass']*(1./12.) * np.r_[(10**2+10**2), (10**2+15**2), (15**2+10**2), np.zeros(3)]
    prob['rna_weightM'] = True

    # For turbine costs
    prob['offshore']             = True
    prob['crane']                = False
    prob['bearing_number']       = 2
    prob['controls_cost_base']   = np.array([35000.0,55900.0])
    prob['controls_esc']         = 1.5
    prob['crane_cost']           = 0.0
    prob['elec_connec_cost_esc'] = 1.5
    #prob['main_bearing_mass']    = 9731.41 / 2
    prob['labor_cost_rate']      = 3.0
    prob['material_cost_rate']   = 2.0
    prob['painting_cost_rate']   = 28.8

    # Offshore BOS
    # Turbine / Plant parameters
    #prob['ballast_cost_rate']            = 100.0
    #prob['tapered_col_cost_rate']        = 4720.0
    #prob['outfitting_cost_rate']         = 6980.0
    prob['nacelleL']                     = -np.inf
    prob['nacelleW']                     = -np.inf
    prob['distShore']                    = 30.0
    prob['distPort']                     = 30.0
    prob['distPtoA']                     = 30.0
    prob['distAtoS']                     = 30.0
    prob['substructure']                 = wind_obos.Substructure.MONOPILE
    prob['anchor']                       = wind_obos.Anchor.SUCTIONPILE
    prob['turbInstallMethod']            = wind_obos.TurbineInstall.INDIVIDUAL
    prob['towerInstallMethod']           = wind_obos.TowerInstall.ONEPIECE
    prob['installStrategy']              = wind_obos.InstallStrategy.PRIMARYVESSEL
    prob['cableOptimizer']               = False
    prob['buryDepth']                    = 2.0
    prob['arrayY']                       = 9.0
    prob['arrayX']                       = 9.0
    prob['substructCont']                = 0.30
    prob['turbCont']                     = 0.30
    prob['elecCont']                     = 0.30
    prob['interConVolt']                 = 345.0
    prob['distInterCon']                 = 3.0
    prob['scrapVal']                     = 0.0
    #General']                           = , 
    prob['inspectClear']                 = 2.0
    prob['plantComm']                    = 0.01
    prob['procurement_contingency']      = 0.05
    prob['install_contingency']          = 0.30
    prob['construction_insurance']       = 0.01
    prob['capital_cost_year_0']          = 0.20
    prob['capital_cost_year_1']          = 0.60
    prob['capital_cost_year_2']          = 0.10
    prob['capital_cost_year_3']          = 0.10
    prob['capital_cost_year_4']          = 0.0
    prob['capital_cost_year_5']          = 0.0
    prob['tax_rate']                     = 0.40
    prob['interest_during_construction'] = 0.08
    #Substructure & Foundation']         = , 
    prob['mpileCR']                      = 2250.0
    prob['mtransCR']                     = 3230.0
    #prob['mpileD']                       = 0.0
    #prob['mpileL']                       = 0.0
    #prob['mpEmbedL']                     = 30.0
    prob['jlatticeCR']                   = 4680.0
    prob['jtransCR']                     = 4500.0
    prob['jpileCR']                      = 2250.0
    prob['jlatticeA']                    = 26.0
    prob['jpileL']                       = 47.50
    prob['jpileD']                       = 1.60
    prob['ssHeaveCR']                    = 6250.0
    prob['scourMat']                     = 250000.0
    prob['number_install_seasons']       = 1.0
    prob['ssTrussCR']                    = 3120.0
    # Mooring
    prob['moorLines']                    = 0
    prob['moorDia']                      = 0.0
    prob['moorCR']                       = 100.0
    #Electrical Infrastructure']         = , 
    prob['pwrFac']                       = 0.95
    prob['buryFac']                      = 0.10
    prob['catLengFac']                   = 0.04
    prob['exCabFac']                     = 0.10
    prob['subsTopFab']                   = 14500.0
    prob['subsTopDes']                   = 4500000.0
    prob['topAssemblyFac']               = 0.075
    prob['subsJackCR']                   = 6250.0
    prob['subsPileCR']                   = 2250.0
    prob['dynCabFac']                    = 2.0
    prob['shuntCR']                      = 35000.0
    prob['highVoltSG']                   = 950000.0
    prob['medVoltSG']                    = 500000.0
    prob['backUpGen']                    = 1000000.0
    prob['workSpace']                    = 2000000.0
    prob['otherAncillary']               = 3000000.0
    prob['mptCR']                        = 12500.0
    prob['arrVoltage']                   = 33.0
    prob['cab1CR']                       = 185.889
    prob['cab2CR']                       = 202.788
    prob['cab1CurrRating']               = 300.0
    prob['cab2CurrRating']               = 340.0
    prob['arrCab1Mass']                  = 20.384
    prob['arrCab2Mass']                  = 21.854
    prob['cab1TurbInterCR']              = 8410.0
    prob['cab2TurbInterCR']              = 8615.0
    prob['cab2SubsInterCR']              = 19815.0
    prob['expVoltage']                   = 220.0
    prob['expCurrRating']                = 530.0
    prob['expCabMass']                   = 71.90
    prob['expCabCR']                     = 495.411
    prob['expSubsInterCR']               = 57500.0
    # Vector inputs
    #prob['arrayCables']                 = [33, 66]
    #prob['exportCables']                = [132, 220]
    #Assembly & Installation',
    prob['moorTimeFac']                  = 0.005
    prob['moorLoadout']                  = 5.0
    prob['moorSurvey']                   = 4.0
    prob['prepAA']                       = 168.0
    prob['prepSpar']                     = 18.0
    prob['upendSpar']                    = 36.0
    prob['prepSemi']                     = 12.0
    prob['turbFasten']                   = 8.0
    prob['boltTower']                    = 7.0
    prob['boltNacelle1']                 = 7.0
    prob['boltNacelle2']                 = 7.0
    prob['boltNacelle3']                 = 7.0
    prob['boltBlade1']                   = 3.50
    prob['boltBlade2']                   = 3.50
    prob['boltRotor']                    = 7.0
    prob['vesselPosTurb']                = 2.0
    prob['vesselPosJack']                = 8.0
    prob['vesselPosMono']                = 3.0
    prob['subsVessPos']                  = 6.0
    prob['monoFasten']                   = 12.0
    prob['jackFasten']                   = 20.0
    prob['prepGripperMono']              = 1.50
    prob['prepGripperJack']              = 8.0
    prob['placePiles']                   = 12.0
    prob['prepHamMono']                  = 2.0
    prob['prepHamJack']                  = 2.0
    prob['removeHamMono']                = 2.0
    prob['removeHamJack']                = 4.0
    prob['placeTemplate']                = 4.0
    prob['placeJack']                    = 12.0
    prob['levJack']                      = 24.0
    prob['hamRate']                      = 20.0
    prob['placeMP']                      = 3.0
    prob['instScour']                    = 6.0
    prob['placeTP']                      = 3.0
    prob['groutTP']                      = 8.0
    prob['tpCover']                      = 1.50
    prob['prepTow']                      = 12.0
    prob['spMoorCon']                    = 20.0
    prob['ssMoorCon']                    = 22.0
    prob['spMoorCheck']                  = 16.0
    prob['ssMoorCheck']                  = 12.0
    prob['ssBall']                       = 6.0
    prob['surfLayRate']                  = 375.0
    prob['cabPullIn']                    = 5.50
    prob['cabTerm']                      = 5.50
    prob['cabLoadout']                   = 14.0
    prob['buryRate']                     = 125.0
    prob['subsPullIn']                   = 48.0
    prob['shorePullIn']                  = 96.0
    prob['landConstruct']                = 7.0
    prob['expCabLoad']                   = 24.0
    prob['subsLoad']                     = 60.0
    prob['placeTop']                     = 24.0
    prob['pileSpreadDR']                 = 2500.0
    prob['pileSpreadMob']                = 750000.0
    prob['groutSpreadDR']                = 3000.0
    prob['groutSpreadMob']               = 1000000.0
    prob['seaSpreadDR']                  = 165000.0
    prob['seaSpreadMob']                 = 4500000.0
    prob['compRacks']                    = 1000000.0
    prob['cabSurveyCR']                  = 240.0
    prob['cabDrillDist']                 = 500.0
    prob['cabDrillCR']                   = 3200.0
    prob['mpvRentalDR']                  = 72000.0
    prob['diveTeamDR']                   = 3200.0
    prob['winchDR']                      = 1000.0
    prob['civilWork']                    = 40000.0
    prob['elecWork']                     = 25000.0
    #Port & Staging']                    = , 
    prob['nCrane600']                    = 0.0
    prob['nCrane1000']                   = 0.0
    prob['crane600DR']                   = 5000.0
    prob['crane1000DR']                  = 8000.0
    prob['craneMobDemob']                = 150000.0
    prob['entranceExitRate']             = 0.525
    prob['dockRate']                     = 3000.0
    prob['wharfRate']                    = 2.75
    prob['laydownCR']                    = 0.25
    #Engineering & Management']          = , 
    prob['estEnMFac']                    = 0.04
    #Development']                       = , 
    prob['preFEEDStudy']                 = 5000000.0
    prob['feedStudy']                    = 10000000.0
    prob['stateLease']                   = 250000.0
    prob['outConShelfLease']             = 1000000.0
    prob['saPlan']                       = 500000.0
    prob['conOpPlan']                    = 1000000.0
    prob['nepaEisMet']                   = 2000000.0
    prob['physResStudyMet']              = 1500000.0
    prob['bioResStudyMet']               = 1500000.0
    prob['socEconStudyMet']              = 500000.0
    prob['navStudyMet']                  = 500000.0
    prob['nepaEisProj']                  = 5000000.0
    prob['physResStudyProj']             = 500000.0
    prob['bioResStudyProj']              = 500000.0
    prob['socEconStudyProj']             = 200000.0
    prob['navStudyProj']                 = 250000.0
    prob['coastZoneManAct']              = 100000.0
    prob['rivsnHarbsAct']                = 100000.0
    prob['cleanWatAct402']               = 100000.0
    prob['cleanWatAct404']               = 100000.0
    prob['faaPlan']                      = 10000.0
    prob['endSpecAct']                   = 500000.0
    prob['marMamProtAct']                = 500000.0
    prob['migBirdAct']                   = 500000.0
    prob['natHisPresAct']                = 250000.0
    prob['addLocPerm']                   = 200000.0
    prob['metTowCR']                     = 11518.0
    prob['decomDiscRate']                = 0.03

    prob['drivetrainType'] = RefBlade.drivetrain #DRIVETRAIN_TYPE['GEARED']  # (Enum)
    prob['drive.gear_ratio']=96.76  # 97:1 as listed in the 5 MW reference document
    prob['drive.shaft_angle']=5.0*np.pi / 180.0  # rad
    prob['drive.shaft_ratio']=0.10
    prob['drive.planet_numbers']=[3, 3, 1]
    prob['drive.shrink_disc_mass']=333.3 * prob['machine_rating'] / 1000.0  # estimated
    prob['drive.carrier_mass']=8000.0  # estimated
    prob['drive.flange_length']=0.5
    prob['drive.overhang']=5.0
    prob['drive.distance_hub2mb']=1.912  # length from hub center to main bearing, leave zero if unknown
    prob['drive.gearbox_input_xcm'] = 0.1
    prob['drive.hss_input_length'] = 1.5

    # Tower inputs
    #prob['tower_top_diameter']=3.78  # m
    #print(prob.root.unknowns.dump())
    prob.run()
