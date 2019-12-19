
from __future__ import print_function
import numpy as np
from pprint import pprint
from openmdao.api import IndepVarComp, ExplicitComponent, Group, Problem, ScipyOptimizeDriver, SqliteRecorder, NonlinearRunOnce, DirectSolver
try:
    from openmdao.api import pyOptSparseDriver
except:
    pass
from wisdem.rotorse.rotor import RotorSE, Init_RotorSE_wRefBlade
from wisdem.rotorse.rotor_geometry_yaml import ReferenceBlade
from wisdem.towerse.tower import TowerSE
from wisdem.commonse import NFREQ
from wisdem.commonse.environment import PowerWind, LogWind
from wisdem.commonse.turbine_constraints import TurbineConstraints
from wisdem.turbine_costsse.turbine_costsse_2015 import Turbine_CostsSE_2015
from wisdem.plant_financese.plant_finance import PlantFinance
from wisdem.drivetrainse.drivese_omdao import DriveSE

from wisdem.commonse.mpi_tools import MPI

# np.seterr(all ='raise')
        
# Group to link the openmdao components
class MonopileTurbine(Group):

    def initialize(self):
        self.options.declare('RefBlade')
        self.options.declare('FASTpref', default={})
        self.options.declare('Nsection_Tow', default = 6)
        self.options.declare('VerbosityCosts', default = True)
        self.options.declare('user_update_routine',     default=None)
        
        
    def setup(self):
        
        RefBlade            = self.options['RefBlade']
        Nsection_Tow        = self.options['Nsection_Tow']        
        user_update_routine = self.options['user_update_routine']
        if 'Analysis_Level' in self.options['FASTpref']:
            Analysis_Level  = self.options['FASTpref']['Analysis_Level']
        else:
            Analysis_Level  = 0
        
        # Define all input variables from all models
        myIndeps = IndepVarComp()
        myIndeps.add_discrete_output('crane',    False)

        # Turbine Costs
        myIndeps.add_discrete_output('bearing_number', 0)
        
        # Tower and Frame3DD options
        myIndeps.add_output('project_lifetime',             0.0, units='yr')
        myIndeps.add_output('max_taper_ratio',              0.0)
        myIndeps.add_output('min_diameter_thickness_ratio', 0.0)

        # Environment
        myIndeps.add_output('wind_bottom_height',   0.0, units='m')
        myIndeps.add_output('wind_beta',            0.0, units='deg')
        myIndeps.add_output('cd_usr', -1.)

        # Environment Offshore
        myIndeps.add_output('offshore',           True)
        myIndeps.add_output('water_depth',        0.0)
        myIndeps.add_output('wave_height',        0.0)
        myIndeps.add_output('wave_period',        0.0)
        myIndeps.add_output('mean_current_speed', 0.0)

        # Design standards
        myIndeps.add_output('gamma_b', 0.0)
        myIndeps.add_output('gamma_n', 0.0)

        # RNA
        myIndeps.add_discrete_output('rna_weightM', True)
        
        # Column
        myIndeps.add_output('morison_mass_coefficient', 2.0)
        myIndeps.add_output('material_density',         0.0, units='kg/m**3')
        myIndeps.add_output('E',                        0.0, units='N/m**2')
        myIndeps.add_output('yield_stress',             0.0, units='N/m**2')

        # Pontoons
        myIndeps.add_output('G', 0.0, units='N/m**2')
        
        # LCOE
        myIndeps.add_output('labor_cost_rate',      0.0, units='USD/min')
        myIndeps.add_output('material_cost_rate',   0.0, units='USD/kg')
        myIndeps.add_output('painting_cost_rate',   0.0, units='USD/m**2')
        myIndeps.add_discrete_output('number_of_turbines', 0)
        myIndeps.add_output('annual_opex',          0.0, units='USD/kW/yr') # TODO: Replace with output connection
        myIndeps.add_output('bos_costs',            0.0, units='USD/kW') # TODO: Replace with output connection
        myIndeps.add_output('fixed_charge_rate',    0.0)
        myIndeps.add_output('wake_loss_factor',     0.0)
        
        self.add_subsystem('myIndeps', myIndeps, promotes=['*'])

        
        # Add components
        self.add_subsystem('rotorse', RotorSE(RefBlade=RefBlade,
                                              npts_coarse_power_curve=20,
                                              npts_spline_power_curve=200,
                                              regulation_reg_II5=True,
                                              regulation_reg_III=True,
                                              Analysis_Level=Analysis_Level,
                                              FASTpref=self.options['FASTpref'],
                                              topLevelFlag=True,
                                              user_update_routine=user_update_routine), promotes=['*'])
        
        self.add_subsystem('drive', DriveSE(debug=False,
                                            number_of_main_bearings=1,
                                            topLevelFlag=False),
                           promotes=['machine_rating', 'overhang',
                                     'hub_mass','bedplate_mass','gearbox_mass','generator_mass','hss_mass','hvac_mass','lss_mass','cover_mass',
                                     'pitch_system_mass','platforms_mass','spinner_mass','transformer_mass','vs_electronics_mass','yaw_mass'])
        
        # Tower and substructure
        self.add_subsystem('tow',TowerSE(nLC=1,
                                         nPoints=Nsection_Tow+1,
                                         nFull=5*Nsection_Tow+1,
                                         wind='PowerWind',
                                         topLevelFlag=False,
                                         monopile=True),
                           promotes=['water_density','water_viscosity','wave_beta',
                                     'significant_wave_height','significant_wave_period',
                                     'material_density','E','G','tower_section_height',
                                     'tower_wall_thickness', 'tower_outer_diameter',
                                     'tower_outfitting_factor','tower_buckling_length',
                                     'max_taper','min_d_to_t','rna_mass','rna_cg','rna_I',
                                     'tower_mass','tower_I_base','hub_height',
                                     'foundation_height','monopile','soil_G','soil_nu',
                                     'suctionpile_depth','gamma_f','gamma_m','gamma_b','gamma_n','gamma_fatigue',
                                     'labor_cost_rate','material_cost_rate','painting_cost_rate','z_full','d_full','t_full',
                                     'DC','shear','geom','tower_force_discretization','nM','Mmethod','lump','tol','shift'])

        # Turbine constraints
        self.add_subsystem('tcons', TurbineConstraints(nFull=5*Nsection_Tow+1), promotes=['*'])
        
        # Turbine costs
        self.add_subsystem('tcost', Turbine_CostsSE_2015(verbosity=self.options['VerbosityCosts'], topLevelFlag=False), promotes=['*'])

        # LCOE Calculation
        self.add_subsystem('plantfinancese', PlantFinance(verbosity=self.options['VerbosityCosts']), promotes=['machine_rating','lcoe'])
        
    
        # Set up connections

        # Connections to DriveSE
        self.connect('diameter',        'drive.rotor_diameter')     
        self.connect('rated_Q',         'drive.rotor_torque')
        self.connect('rated_Omega',     'drive.rotor_rpm')
        self.connect('Fxyz_total',      'drive.Fxyz')
        self.connect('Mxyz_total',      'drive.Mxyz')
        self.connect('I_all_blades',        'drive.blades_I')
        self.connect('mass_one_blade',  'drive.blade_mass')
        self.connect('chord',           'drive.blade_root_diameter', src_indices=[0])
        self.connect('Rtip',            'drive.blade_length', src_indices=[0])
        self.connect('drivetrainEff',   'drive.drivetrain_efficiency', src_indices=[0])
        self.connect('tower_outer_diameter', 'drive.tower_top_diameter', src_indices=[-1])

        self.connect('material_density', 'tow.tower.rho')

        # Connections to TowerSE
        self.connect('drive.top_F',         'tow.pre.rna_F')
        self.connect('drive.top_M',         'tow.pre.rna_M')
        self.connect('drive.rna_I_TT',            ['rna_I','tow.pre.mI'])
        self.connect('drive.rna_cm',              ['rna_cg','tow.pre.mrho'])
        self.connect('drive.rna_mass',            ['rna_mass','tow.pre.mass'])
        self.connect('rs.gust.V_gust',          'tow.wind.Uref')
        self.connect('wind_reference_height',   ['tow.wind.zref','wind.zref'])
        # self.connect('wind_bottom_height',      ['tow.wind.z0','tow.wave.z_surface', 'wind.z0'])  # offshore
        self.connect('wind_bottom_height',      ['tow.wind.z0', 'wind.z0'])
        self.connect('shearExp',                ['tow.wind.shearExp'])
        # self.connect('morison_mass_coefficient','tow.cm')                                         # offshore
        self.connect('yield_stress',            'tow.sigma_y')
        self.connect('max_taper_ratio',         'max_taper')
        self.connect('min_diameter_thickness_ratio', 'min_d_to_t')

        self.connect('rho',         'tow.windLoads.rho')
        self.connect('mu',          'tow.windLoads.mu')
        self.connect('wind_beta',   'tow.windLoads.beta')
        
        # Connections to TurbineConstraints
        self.connect('nBlades',                 ['blade_number', 'drive.number_of_blades'])
        self.connect('control_maxOmega',        'rotor_omega')
        self.connect('tow.post.structural_frequencies', 'tower_freq')        
                
        # Connections to TurbineCostSE
        self.connect('mass_one_blade',              'blade_mass')
        self.connect('drive.mainBearing.mb_mass',   'main_bearing_mass')
        self.connect('total_blade_cost',            'blade_cost_external')
        
        # Connections to PlantFinanceSE
        self.connect('AEP',                 'plantfinancese.turbine_aep')
        self.connect('turbine_cost_kW',     'plantfinancese.tcc_per_kW')
        self.connect('number_of_turbines',  'plantfinancese.turbine_number')
        self.connect('bos_costs',           'plantfinancese.bos_per_kW')
        self.connect('annual_opex',         'plantfinancese.opex_per_kW')
    

def Init_MonopileTurbine(prob, blade, Nsection_Tow, Analysis_Level = 0, fst_vt = {}):

    prob = Init_RotorSE_wRefBlade(prob, blade, Analysis_Level = Analysis_Level, fst_vt = fst_vt)

    # Environmental parameters for the tower
    # prob['wind_reference_speed']           = 11.0
    prob['wind_reference_height']          = prob['hub_height']
    prob['shearExp']                       = 0.11
    prob['rho']                            = 1.225
    prob['mu']                             = 1.7934e-5
    prob['water_density']                  = 1025.0
    prob['water_viscosity']                = 1.3351e-3
    prob['wind_beta'] = prob['wave_beta'] = 0.0
    prob['significant_wave_height']        = 5.0
    prob['significant_wave_period']        = 10.0
    prob['monopile']                       = True

    # Steel properties for the tower
    prob['material_density']               = 7850.0
    prob['E']                              = 210e9
    prob['G']                              = 79.3e9
    prob['yield_stress']                   = 345e6
    prob['soil_G']                         = 140e6
    prob['soil_nu']                        = 0.4

    # Design constraints
    prob['max_taper_ratio']                = 0.4
    prob['min_diameter_thickness_ratio']   = 120.0

    # Safety factors
    prob['gamma_fatigue']   = 1.755 # (Float): safety factor for fatigue
    prob['gamma_f']         = 1.35  # (Float): safety factor for loads/stresses
    prob['gamma_m']         = 1.3   # (Float): safety factor for materials
    prob['gamma_freq']      = 1.1   # (Float): safety factor for resonant frequencies
    prob['gamma_n']         = 1.0
    prob['gamma_b']         = 1.1
    
    # Tower
    # prob['foundation_height']              = -prob['water_depth']
    # prob['tower_outer_diameter']           = np.linspace(10.0, 3.87, Nsection_Tow+1)
    prob['tower_outer_diameter']           = np.linspace(6.0, 3.87, Nsection_Tow+1)
    prob['tower_section_height']           = (prob['hub_height'] - prob['foundation_height']) / Nsection_Tow * np.ones(Nsection_Tow)
    prob['tower_wall_thickness']           = np.linspace(0.027, 0.019, Nsection_Tow)
    prob['tower_buckling_length']          = 30.0
    prob['tower_outfitting_factor']        = 1.07

    prob['DC']      = 80.0
    prob['shear']   = True
    prob['geom']    = False
    prob['tower_force_discretization'] = 5.0
    prob['nM']      = 2
    prob['Mmethod'] = 1
    prob['lump']    = 0
    prob['tol']     = 1e-9
    prob['shift']   = 0.0
    
    # Plant size
    prob['project_lifetime'] = prob['lifetime'] = 20.0    
    prob['number_of_turbines']             = 200. * 1.e+006 / prob['machine_rating']
    prob['annual_opex']                    = 43.56 # $/kW/yr
    prob['bos_costs']                      = 517.0 # $/kW
    
    # For RNA
    prob['rna_weightM'] = True

    # For turbine costs
    prob['offshore']             = True
    prob['crane']                = False
    prob['bearing_number']       = 2
    prob['crane_cost']           = 0.0
    prob['labor_cost_rate']      = 3.0
    prob['material_cost_rate']   = 2.0
    prob['painting_cost_rate']   = 28.8
    
    # Gearbox
    prob['drive.gear_ratio']        = 96.76  # 97:1 as listed in the 5 MW reference document
    prob['drive.shaft_angle']       = prob['tilt']*np.pi / 180.0  # rad
    prob['drive.shaft_ratio']       = 0.10
    prob['drive.planet_numbers']    = [3, 3, 1]
    prob['drive.shrink_disc_mass']  = 333.3 * prob['machine_rating'] / 1e6  # estimated
    prob['drive.carrier_mass']      = 8000.0  # estimated
    prob['drive.flange_length']     = 0.5
    prob['overhang']                = 5.0
    prob['drive.distance_hub2mb']   = 1.912  # length from hub center to main bearing, leave zero if unknown
    prob['drive.gearbox_input_xcm'] = 0.1
    prob['drive.hss_input_length']  = 1.5
    prob['drive.yaw_motors_number'] = 1
    
    return prob
    

if __name__ == "__main__":
    
    optFlag = True



    # Reference rotor design
    fname_schema          = "../../rotorse/turbine_inputs/IEAontology_schema.yaml"
    fname_input           = "../../rotorse/turbine_inputs/nrel5mw_mod_update.yaml"
    Analysis_Level        = 0 # 0: Run CCBlade; 1: Update FAST model at each iteration but do not run; 2: Run FAST w/ ElastoDyn; 3: (Not implemented) Run FAST w/ BeamDyn
    # Initialize blade design
    refBlade = ReferenceBlade()
    refBlade.verbose      = True
    refBlade.NINPUT       = 8
    refBlade.NPTS         = 50
    refBlade.spar_var     = ['Spar_Cap_SS', 'Spar_Cap_PS'] # SS, then PS
    refBlade.te_var       = 'TE_reinforcement'
    refBlade.validate     = False
    refBlade.fname_schema = fname_schema
    blade = refBlade.initialize(fname_input)
    # Initialize tower design
    Nsection_Tow = 6

    
    # Initialize OpenMDAO problem and FloatingSE Group
    if MPI:
        num_par_fd = MPI.COMM_WORLD.Get_size()
        prob = Problem(model=Group(num_par_fd=num_par_fd))
        prob.model.approx_totals(method='fd')
        prob.model.add_subsystem('comp', LandBasedTurbine(RefBlade=blade, Nsection_Tow = Nsection_Tow, VerbosityCosts = True), promotes=['*'])
    else:
        prob = Problem()
        prob.model=LandBasedTurbine(RefBlade=blade, Nsection_Tow = Nsection_Tow, VerbosityCosts = True)
    
    if optFlag:
        # --- Solver ---
        prob.driver  = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol']       = 1.e-6
        prob.driver.options['maxiter']   = 100

        #prob.driver = pyOptSparseDriver()
        #prob.driver.options['optimizer'] = 'CONMIN'
        # prob.driver.options['optimizer'] = 'SNOPT'
        # prob.driver.options['gradient method'] = "pyopt_fd"
        # ----------------------

        # --- Objective ---
        prob.model.add_objective('lcoe')
        # ----------------------

        # --- Design Variables ---
        indices_no_root         = range(2,refBlade.NINPUT)
        indices_no_root_no_tip  = range(2,refBlade.NINPUT-1)
        prob.model.add_design_var('chord_in',    indices = indices_no_root_no_tip, lower=0.5,      upper=7.0)
        prob.model.add_design_var('theta_in',    indices = indices_no_root,        lower=-5.0,     upper=20.0)
        prob.model.add_design_var('sparT_in',    indices = indices_no_root_no_tip, lower=0.001,    upper=0.200)
        prob.model.add_design_var('control_tsr',                                   lower=6.000,    upper=11.00)
        prob.model.add_design_var('tower_section_height', lower=5.0,  upper=80.0)
        prob.model.add_design_var('tower_outer_diameter', lower=3.87, upper=30.0)
        prob.model.add_design_var('tower_wall_thickness', lower=4e-3, upper=2e-1)
        # ----------------------
        
        # --- Constraints ---
        # Rotor
        prob.model.add_constraint('tip_deflection_ratio',                      upper=1.0)  
        # Tower
        prob.model.add_constraint('tow.height_constraint',         lower=-1e-2,upper=1.e-2)
        prob.model.add_constraint('tow.post.stress',                           upper=1.0)
        prob.model.add_constraint('tow.post.global_buckling',                  upper=1.0)
        prob.model.add_constraint('tow.post.shell_buckling',                   upper=1.0)
        prob.model.add_constraint('tow.weldability',                           upper=0.0)
        prob.model.add_constraint('tow.manufacturability',         lower=0.0)
        # prob.model.add_constraint('frequency1P_margin_low',              upper=1.0)
        # prob.model.add_constraint('frequency1P_margin_high', lower=1.0)
        # prob.model.add_constraint('frequencyNP_margin_low',              upper=1.0)
        # prob.model.add_constraint('frequencyNP_margin_high', lower=1.0)
        prob.model.add_constraint('frequencyNP_margin', upper=0.)
        prob.model.add_constraint('frequency1P_margin', upper=0.)
        prob.model.add_constraint('ground_clearance',        lower=20.0)
        # ----------------------
        
        # --- Recorder ---
        prob.driver.add_recorder(SqliteRecorder('log_opt.sql'))
        prob.driver.recording_options['includes'] = ['AEP','rc.total_blade_cost','lcoe','tip_deflection_ratio']
        prob.driver.recording_options['record_objectives']  = True
        prob.driver.recording_options['record_constraints'] = True
        prob.driver.recording_options['record_desvars']     = True
        # ----------------------


    prob.setup(check=True)
    
    prob = Init_LandBasedAssembly(prob, blade, Nsection_Tow)
    prob.model.nonlinear_solver = NonlinearRunOnce()
    prob.model.linear_solver = DirectSolver()

    if not MPI:
        prob.model.approx_totals()

    # prob.run_model()
    # prob.model.list_inputs(units=True)
    # prob.model.list_outputs(units=True)


    prob.run_driver()
    # prob.check_partials(compact_print=True, method='fd', step=1e-6, form='central')
    





    
