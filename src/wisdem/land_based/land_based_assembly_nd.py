

from __future__ import print_function
import numpy as np
import os, time, shutil
import sqlitedict
from pprint import pprint
import matplotlib.pyplot as plt
from openmdao.api import IndepVarComp, Component, Group, Problem, ScipyOptimizer, SqliteRecorder, pyOptSparseDriver
from rotorse.rotor_geometry_yaml import ReferenceBlade
from rotorse.rotor_geometry import RotorGeometry, TURBULENCE_CLASS, TURBINE_CLASS, DRIVETRAIN_TYPE
from rotorse.rotor import RotorSE
from rotorse.bladecost_openmdao import blade_cost_model_mdao

# from offshorebos.wind_obos_component import WindOBOS
# import offshorebos.wind_obos as wind_obos
from towerse.tower import TowerSE
from commonse import NFREQ
from commonse.rna import RNA
from commonse.environment import PowerWind, LogWind
from commonse.turbine_constraints import TurbineConstraints
from turbine_costsse.turbine_costsse_2015 import Turbine_CostsSE_2015
from plant_financese.plant_finance import PlantFinance
from drivese.drivese_omdao import Drive3pt, Drive4pt
# from wisdem.fixed_bottom.monopile_assembly import wind, nLC, nDEL, NSECTION
wind = 'PowerWind'
nLC = 1
nDEL = 0
NSECTION = 6




# clear = lambda: os.system('cls')
# clear()



# Class to print outputs on screen
class Outputs_2_Screen(Component):
    def __init__(self, NPTS):
        super(Outputs_2_Screen, self).__init__()
        
        self.add_param('chord',                val=np.zeros(NPTS))
        self.add_param('theta',                val=np.zeros(NPTS))
        self.add_param('bladeLength',          val=0.0)
        self.add_param('total_blade_cost',     val=0.0)
        self.add_param('mass_one_blade',       val=0.0)
        self.add_param('tower_mass',           val=0.0)
        self.add_param('tower_cost',           val=0.0)
        self.add_param('control_tsr',          val=0.0)
        self.add_param('AEP',                  val=0.0)
        self.add_param('lcoe',                 val=0.0)
        self.add_param('rated_T',              val=0.0)
        self.add_param('root_bending_moment',  val=0.0)
        self.add_param('tip_deflection',       val=0.0)
        self.add_param('tip_deflection_ratio', val=0.0)
        self.add_param('buckling_const',       val=np.zeros(NPTS))
        
    def solve_nonlinear(self, params, unknowns, resids):
        print('########################################')
        print('Optimization variables')
        print('Max chord:   {:8.3f} m'.format(max(params['chord'])))
        print('TSR:         {:8.3f} -'.format(params['control_tsr']))
        print('')
        print('Constraints')
        print('Max TD:      {:8.3f} m'.format(params['tip_deflection']))
        print('TD ratio:    {:8.4f} -'.format(params['tip_deflection_ratio']))
        print('Buckling:    {:8.2f} -'.format(max(params['buckling_const'])*1e+3))
        print('')
        print('Objectives')
        print('AEP:         {:8.5f} GWh'.format(params['AEP']*1e-6))
        print('Blade mass:  {:8.3f} kg'.format(params['mass_one_blade']))
        print('Blade cost:  {:8.3f} $'.format(params['total_blade_cost']))
        print('Tower mass:  {:8.3f} kg'.format(params['tower_mass']))
        print('Tower cost:  {:8.3f} $'.format(params['tower_cost']))
        print('LCoE:        {:8.3f} $/MWh'.format(params['lcoe']*1.e3))
        print('########################################')

class Convergence_Trends_Opt(Component):
    def __init__(self, folder_output, optimization_log, options={}):
        super(Convergence_Trends_Opt, self).__init__()
        
        
        self.optimization_log = folder_output + optimization_log
        self.folder_output    = folder_output
        
        if options == {}:
            self.options = {}
            self.options['verbosity']        = False
            self.options['tex_table']        = False
            self.options['generate_plots']   = True
            self.options['show_plots']       = False
            self.options['show_warnings']    = False
        else:
            self.options = options
        
    def solve_nonlinear(self, params, unknowns, resids):
        

        rec_data_raw = sqlitedict.SqliteDict(self.optimization_log, 'iterations')
        rec_data   = {}
        iterations = []
        for i, it in enumerate(rec_data_raw.keys()):
            iterations.append(i)
            it_data = rec_data_raw[it]
            parameters = it_data['Unknowns']
            for j, param in enumerate(parameters.keys()):
                if i == 0:
                    rec_data[param] = []
                rec_data[param].append(parameters[param])

        # Plots
        if self.options['generate_plots'] == True:
            for param in rec_data.keys():
                fig, ax = plt.subplots(1,1,figsize=(5.3, 4))
                ax.plot(iterations, rec_data[param])
                ax.set(xlabel='Number of Iterations' , ylabel=param)
                fig_name = 'Convergence_trend_' + param + '.png'
                fig.savefig(self.folder_output + fig_name)
                plt.close(fig)
            
        return iterations, rec_data      
      

# Class to check the buckling strain along the blade
class Buckling(Component):
    def __init__(self, NPTS):
        super(Buckling, self).__init__()

        self.add_param('eps_crit_spar', val=np.zeros(NPTS))
        self.add_param('strainU_spar', val=np.zeros(NPTS))
        self.add_output('buckling_const', val=np.zeros(NPTS))

    def solve_nonlinear(self, params, unknowns, resids):
        unknowns['buckling_const'] = np.asarray([crit-strain for crit, strain in zip(params['eps_crit_spar'], params['strainU_spar'])])

# Class to run an opt loop with nondimensional optimization variables
class DimensionalizeOptVar(Component):
    def __init__(self, blade, twist_bounds):
        super(DimensionalizeOptVar, self).__init__()
                
        self.lb_theta_deg = twist_bounds['lb_theta_deg']
        self.range_theta  = twist_bounds['range_theta']
        
        
        init_opt_theta    = (blade['ctrl_pts']['theta_in'] - self.lb_theta_deg) / self.range_theta
        
        self.add_param('chord_init', val=np.zeros(len(blade['ctrl_pts']['chord_in'])))
        self.add_param('theta_init', val=np.zeros(len(blade['ctrl_pts']['theta_in'])))
        self.add_param('sparT_init', val=np.zeros(len(blade['ctrl_pts']['sparT_in'])))
        
        self.add_param('opt_chord',  val=np.ones(len(blade['ctrl_pts']['chord_in'])))
        self.add_param('opt_theta',  val=np.ones(len(blade['ctrl_pts']['theta_in'])) * init_opt_theta)
        self.add_param('opt_sparT',  val=np.ones(len(blade['ctrl_pts']['sparT_in'])))
        
        self.add_output('dim_chord', val=np.zeros(len(blade['ctrl_pts']['chord_in'])))
        self.add_output('dim_theta', val=np.zeros(len(blade['ctrl_pts']['theta_in'])))
        self.add_output('dim_sparT', val=np.zeros(len(blade['ctrl_pts']['sparT_in'])))
        
    def solve_nonlinear(self, params, unknowns, resids):
        
        # print(params['opt_chord'])
        # print(params['opt_theta'])
        # print(params['opt_sparT'])
        
        unknowns['dim_chord'] = params['chord_init'] * params['opt_chord']
        unknowns['dim_theta'] = self.lb_theta_deg + params['opt_theta'] * self.range_theta
        unknowns['dim_sparT'] = params['sparT_init'] * params['opt_sparT']

# Class to define a constraint so that the blade cannot operate in stall conditions
class NoStallConstraint(Component):
    def __init__(self, blade, verbosity = False):
        super(NoStallConstraint, self).__init__()
        
        self.blade = blade
        NPTS = len(blade['pf']['s'])
        self.verbosity = verbosity
        
        self.add_param('stall_angle_along_span', val=np.zeros(NPTS), units = 'deg', desc = 'Stall angle along blade span')
        self.add_param('aoa_along_span',         val=np.zeros(NPTS), units = 'deg', desc = 'Angle of attack along blade span')
        self.add_param('stall_margin',           val=0.0,            units = 'deg', desc = 'Minimum margin from the stall angle')
        self.add_param('min_s',                  val=0.0,            desc = 'Minimum nondimensional coordinate along blade span where to define the constraint (blade root typically stalls)')
        
        self.add_output('no_stall_constraint',   val=np.zeros(NPTS), desc = 'Constraint, ratio between angle of attack plus a margin and stall angle')

    def solve_nonlinear(self, params, unknowns, resids):
        
        i_min = np.argmin(abs(params['min_s'] - self.blade['pf']['s']))
        
        for i in range(len(self.blade['pf']['s'])):
            params['stall_angle_along_span'][i] = self.blade['airfoils'][i].unsteady['alpha1']
            if params['stall_angle_along_span'][i] == 0:
                params['stall_angle_along_span'][i] = 1e-6 # To avoid nan
        
        for i in range(i_min, len(self.blade['pf']['s'])):
            unknowns['no_stall_constraint'][i] = (params['aoa_along_span'][i] + params['stall_margin']) / params['stall_angle_along_span'][i]
        
            if self.verbosity == True:
                if unknowns['no_stall_constraint'][i] > 1:
                    print('Blade is stalling at span location %.2f %%' % (self.blade['pf']['s'][i]*100.))
        
        
# Group to link the openmdao components
class Onshore_Assembly(Group):

    def __init__(self, blade, folder_output, FASTpref = {}, options = {}):
        super(Onshore_Assembly, self).__init__()
        
        twist_bounds = {}
        twist_bounds['lb_theta_deg'] = np.ones(len(blade['ctrl_pts']['theta_in'])) * -5 # In deg
        twist_bounds['ub_theta_deg'] = np.ones(len(blade['ctrl_pts']['theta_in'])) * 20 # In deg
        twist_bounds['range_theta']  = twist_bounds['ub_theta_deg'] - twist_bounds['lb_theta_deg']
        
        
        init_opt_theta    = (blade['ctrl_pts']['theta_in'] - twist_bounds['lb_theta_deg']) / twist_bounds['range_theta']
        
        # print(init_opt_theta)
        # exit()
        
        # self.add('hub_height', IndepVarComp('hub_height', 0.0), promotes=['*'])
        # self.add('foundation_height', IndepVarComp('foundation_height', 0.0), promotes=['*'])
        self.add('opt_chord', IndepVarComp('opt_chord', np.ones(len(blade['ctrl_pts']['chord_in']))), promotes=['*'])
        self.add('opt_theta', IndepVarComp('opt_theta', np.ones(len(blade['ctrl_pts']['theta_in'])) * init_opt_theta), promotes=['*'])
        self.add('opt_sparT', IndepVarComp('opt_sparT', np.ones(len(blade['ctrl_pts']['sparT_in']))), promotes=['*'])
        
        # Add components
        self.add('opt_var', DimensionalizeOptVar(blade, twist_bounds), promotes =['opt_chord','opt_theta','opt_sparT'])
        
        self.add('rotorse', RotorSE(blade, npts_coarse_power_curve=100, npts_spline_power_curve=200, regulation_reg_II5=True, regulation_reg_III=False, Analysis_Level=-1, FASTpref=FASTpref, flag_nd_opt = True), promotes=['*'])
        self.add('bcm', blade_cost_model_mdao(len(blade['pf']['s']), name=blade['config']['name'], options = options), promotes=['total_blade_cost'])
        self.add('buckling', Buckling(len(blade['pf']['s'])), promotes=['buckling_const'])
        self.add('nostallconstraint', NoStallConstraint(blade, verbosity = True), promotes=['no_stall_constraint'])
        
        
        # RNA
        self.add('drive', Drive4pt('CARB', 'SRB', 'B', 'eep', 'normal', 'geared', True, 0, True, 3),
                 promotes=['hub_mass','bedplate_mass','gearbox_mass','generator_mass','hss_mass','hvac_mass','lss_mass','cover_mass',
                           'pitch_system_mass','platforms_mass','spinner_mass','transformer_mass','vs_electronics_mass','yaw_mass'])
        self.add('rna', RNA(nLC))
        
        # Tower and substructure
        self.add('tow',TowerSE(nLC, NSECTION+1, 5*(NSECTION) + 1, nDEL, wind='PowerWind'), promotes=['material_density','E','G','tower_section_height',
                                                                                          'tower_outer_diameter','tower_wall_thickness',
                                                                                          'tower_outfitting_factor','tower_buckling_length',
                                                                                          'max_taper','min_d_to_t','rna_mass','rna_cg','rna_I',
                                                                                          'tower_mass','tower_I_base','hub_height',
                                                                                          'foundation_height','monopile','soil_G','soil_nu',
                                                                                          'suctionpile_depth','gamma_f','gamma_m','gamma_b','gamma_n','gamma_fatigue',
                                                                                          'labor_cost_rate','material_cost_rate','painting_cost_rate','z_full','d_full','t_full'])
        # Turbine constraints
        self.add('tcons', TurbineConstraints(5*(NSECTION) + 1), promotes=['*'])
        
        # Turbine costs
        self.add('tcost', Turbine_CostsSE_2015(verbosity = False), promotes=['*'])

        # LCOE Calculation
        self.add('plantfinancese', PlantFinance(verbosity = False))
        
        # Post-processing
        self.add('outputs_2_screen', Outputs_2_Screen(len(blade['pf']['s'])), promotes=['*'])
        self.add('convergence_trends_opt', Convergence_Trends_Opt(folder_output, 'log_opt_' + blade['config']['name']))
        
        
        # Define all input variables from all models
        self.add('offshore',             IndepVarComp('offshore', True, pass_by_obj=True), promotes=['*'])
        self.add('crane',                IndepVarComp('crane', False, pass_by_obj=True), promotes=['*'])

        # Turbine Costs
        # REMOVE ONCE DRIVESE AND GENERATORSE ARE CONNECTED
        self.add('bearing_number',           IndepVarComp('bearing_number', 0, pass_by_obj=True), promotes=['*'])

        
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

        # # Environment
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
        self.add('labor_cost_rate',    IndepVarComp('labor_cost_rate', 0.0), promotes=['*'])
        self.add('material_cost_rate', IndepVarComp('material_cost_rate', 0.0), promotes=['*'])
        self.add('painting_cost_rate', IndepVarComp('painting_cost_rate', 0.0), promotes=['*'])
        self.add('number_of_turbines', IndepVarComp('number_of_turbines', 0, pass_by_obj=True), promotes=['*'])
        self.add('annual_opex',        IndepVarComp('annual_opex', 0.0), promotes=['*']) # TODO: Replace with output connection
        self.add('fixed_charge_rate',  IndepVarComp('fixed_charge_rate', 0.0), promotes=['*'])
        self.add('discount_rate',      IndepVarComp('discount_rate', 0.0), promotes=['*'])
        
        
        
        
        # Set up connections
        
        # from DimensionalizeOptVar to RotorSE
        self.connect('opt_var.dim_chord',   'chord_in')
        self.connect('opt_var.dim_theta',   'theta_in')
        self.connect('opt_var.dim_sparT',   'sparT_in')
        
        
        # from RotorSE to BCM
        self.connect('materials',           'bcm.materials')
        self.connect('upperCS',             'bcm.upperCS')
        self.connect('lowerCS',             'bcm.lowerCS')
        self.connect('websCS',              'bcm.websCS')
        self.connect('profile',             'bcm.profile')
        self.connect('bladeLength',         'bcm.bladeLength')
        self.connect('Rtip',                'bcm.Rtip')
        self.connect('Rhub',                'bcm.Rhub')
        self.connect('r_pts',               'bcm.r_pts')
        self.connect('chord',               'bcm.chord')
        self.connect('le_location',         'bcm.le_location')
        # from RotorSE to Buckling
        self.connect('eps_crit_spar',       'buckling.eps_crit_spar')
        self.connect('strainU_spar',        'buckling.strainU_spar')
        # from RotorSE-RegulatedPowerCurve to NoStallConstraint
        self.connect('powercurve.aoa_cutin','nostallconstraint.aoa_along_span')

        # Connections to DriveSE
        self.connect('diameter',        'drive.rotor_diameter')        
        self.connect('rated_Q',         'drive.rotor_torque')
        self.connect('rated_Omega',     'drive.rotor_rpm')
        self.connect('machine_rating',  'drive.machine_rating')
        self.connect('Mxyz_total',      'drive.rotor_bending_moment_x', src_indices=[0])
        self.connect('Mxyz_total',      'drive.rotor_bending_moment_y', src_indices=[1])
        self.connect('Mxyz_total',      'drive.rotor_bending_moment_z', src_indices=[2])
        self.connect('Fxyz_total',      'drive.rotor_thrust', src_indices=[0])
        self.connect('Fxyz_total',      'drive.rotor_force_y', src_indices=[1])
        self.connect('Fxyz_total',      'drive.rotor_force_z', src_indices=[2])
        self.connect('mass_one_blade',  'drive.blade_mass')
        self.connect('chord',           'drive.blade_root_diameter', src_indices=[0])
        self.connect('drivetrainEff',   'drive.drivetrain_efficiency', src_indices=[0])
        
        
        # Connections to RNA (CommonSE)
        self.connect('mass_all_blades',     'rna.blades_mass')
        self.connect('drive.hub_system_mass', 'rna.hub_mass')
        self.connect('drive.nacelle_mass',  'rna.nac_mass')
        self.connect('I_all_blades',        'rna.blades_I')
        self.connect('drive.hub_system_I',  'rna.hub_I')
        self.connect('drive.nacelle_I',     'rna.nac_I')
        self.connect('drive.hub_system_cm', 'rna.hub_cm')
        self.connect('drive.nacelle_cm',    'rna.nac_cm')
        self.connect('Fxyz_total',          'rna.loads.F')
        self.connect('Mxyz_total',          'rna.loads.M')
        

        
        # Connections to TowerSE
        self.connect('rna.loads.top_F', 'tow.pre.rna_F')
        self.connect('rna.loads.top_M', 'tow.pre.rna_M')
        self.connect('rna.rna_I_TT',    'rna_I')
        self.connect('rna.rna_cm',      'rna_cg')
        self.connect('rho',             'tow.windLoads.rho') #,'powercurve.rho'])
        self.connect('mu',              'tow.windLoads.mu')#,'powercurve.mu'])
        self.connect('water_density',   ['tow.wave.rho','tow.waveLoads.rho'])
        self.connect('water_viscosity', 'tow.waveLoads.mu')
        self.connect('wave_height',     'tow.wave.hmax')
        self.connect('wave_period',     'tow.wave.T')
        self.connect('wind_reference_speed',    'tow.wind.Uref')
        self.connect('wind_reference_height',   ['tow.wind.zref','wind.zref'])
        self.connect('wind_bottom_height',      ['tow.z0','wind.z0'])
        self.connect('shearExp',                ['tow.wind.shearExp', 'wind.shearExp'])
        self.connect('morison_mass_coefficient', 'tow.cm')
        self.connect('yield_stress',            'tow.sigma_y')
        self.connect('max_taper_ratio',         'max_taper')
        self.connect('min_diameter_thickness_ratio', 'min_d_to_t')
        self.connect('water_depth', ['tow.z_floor', 'plantfinancese.sea_depth'])

        
        
        
        
        
        
        # Connections to TurbineConstraints
        self.connect('nBlades',                 'blade_number')
        self.connect('control_maxOmega',        'rotor_omega')
        self.connect('drive.hub_system_cm',     'hub_center')
        self.connect('tow.post.structural_frequencies', 'tower_freq')        
                
        # Connections to TurbineCostSE
        self.connect('mass_one_blade',      'blade_mass')
        self.connect('drive.mb1_mass',      'main_bearing_mass')
        self.connect('total_blade_cost',    'blade_cost_external')
        
        # Connections to PlantFinanceSE
        self.connect('AEP',                 'plantfinancese.turbine_aep')
        self.connect('turbine_cost',        'plantfinancese.turbine_cost')
        self.connect('number_of_turbines',  'plantfinancese.turbine_number')
        self.connect('annual_opex',         'plantfinancese.turbine_avg_annual_opex')
        self.connect('fixed_charge_rate',   'plantfinancese.fixed_charge_rate')
        self.connect('discount_rate',       'plantfinancese.discount_rate')
        
        # Connections to Outputs_2_Screen
        self.connect('plantfinancese.lcoe', 'lcoe')
        
        
        
        
        # Use complex number finite differences
        self.deriv_options['type'] = 'fd'
        # self.deriv_options['form'] = 'central'
        # self.deriv_options['step_size'] = 1e-5
        self.deriv_options['step_calc'] = 'relative'
        
        
        
        

def Init_Assembly(prob, blade, fst_vt={}):

    # Rotor inputs
    # === blade grid ===
    prob['hubFraction']    = blade['config']['hubD']/2./blade['pf']['r'][-1] # (Float): hub location as fraction of radius
    prob['bladeLength']    = blade['ctrl_pts']['bladeLength'] # (Float, m): blade length (if not precurved or swept) otherwise length of blade before curvature
    # prob['delta_bladeLength'] = 0.0  # (Float, m): adjustment to blade length to account for curvature from loading
    prob['precone']        = blade['config']['cone_angle'] # (Float, deg): precone angle
    prob['tilt']           = blade['config']['tilt_angle'] # (Float, deg): shaft tilt
    prob['yaw']            = 0.0  # (Float, deg): yaw error
    prob['nBlades']        = blade['config']['number_of_blades']# (Int): number of blades
    # ------------------
    
    # === blade geometry ===
    prob['r_max_chord']    = blade['ctrl_pts']['r_max_chord']  #(Float): location of max chord on unit radius
    prob['opt_var.chord_init'] = np.array(blade['ctrl_pts']['chord_in']) # (Array, m): chord at control points. defined at hub, then at linearly spaced locations from r_max_chord to tip
    prob['opt_var.theta_init'] = np.array(blade['ctrl_pts']['theta_in']) # (Array, deg): twist at control points.  defined at linearly spaced locations from r[idx_cylinder] to tip
    prob['opt_var.sparT_init'] = np.array(blade['ctrl_pts']['sparT_in']) # (Array, m): spar cap thickness parameters
    
    
    prob['precurve_in']    = np.array(blade['ctrl_pts']['precurve_in']) # (Array, m): precurve at control points.  defined at same locations at chord, starting at 2nd control point (root must be zero precurve)
    prob['precurve_tip']   = -blade['pf']['precurve'][-1]
    prob['presweep_tip']   = blade['pf']['presweep'][-1]
    prob['presweep_in']    = np.array(blade['ctrl_pts']['presweep_in']) # (Array, m): precurve at control points.  defined at same locations at chord, starting at 2nd control point (root must be zero precurve)
    
    prob['teT_in']         = np.array(blade['ctrl_pts']['teT_in']) # (Array, m): trailing-edge thickness parameters
    # ------------------

    # === atmosphere ===
    prob['rho']              = 1.225  # (Float, kg/m**3): density of air
    prob['mu']               = 1.81206e-5  # (Float, kg/m/s): dynamic viscosity of air
    # prob['wind.shearExp']    = 0.00  # (Float): shear exponent
    prob['shape_parameter']  = 2.0
    prob['hub_height']       = blade['config']['hub_height']  # (Float, m): hub height
    prob['turbine_class']    = TURBINE_CLASS[blade['config']['turbine_class'].upper()] #TURBINE_CLASS['I']  # (Enum): IEC turbine class
    prob['turbulence_class'] = TURBULENCE_CLASS[blade['config']['turbulence_class'].upper()]  # (Enum): IEC turbulence class class
    # prob['wind.zref']        = blade['config']['hub_height']
    prob['gust_stddev']      = 3
    # ----------------------

    # === control ===
    prob['control_Vin']      = blade['config']['Vin'] # (Float, m/s): cut-in wind speed
    prob['control_Vout']     = blade['config']['Vout']# (Float, m/s): cut-out wind speed
    prob['control_minOmega'] = blade['config']['minOmega'] # (Float, rpm): minimum allowed prob rotation speed
    prob['control_maxOmega'] = blade['config']['maxOmega'] # (Float, rpm): maximum allowed prob rotation speed
    prob['control_tsr']      = blade['config']['tsr'] # (Float): tip-speed ratio in Region 2 (should be optimized externally)
    prob['control_pitch']    = blade['config']['pitch']  # (Float, deg): pitch angle in region 2 (and region 3 for fixed pitch machines)
    prob['control_maxTS']    = blade['config']['maxTS']
    prob['machine_rating']   = blade['config']['rating'] # (Float, W): rated power
    prob['pitch_extreme']    = 0.0  # (Float, deg): worst-case pitch at survival wind condition
    prob['azimuth_extreme']  = 0.0  # (Float, deg): worst-case azimuth at survival wind condition
    prob['VfactorPC']        = 0.7  # (Float): fraction of rated speed at which the deflection is assumed to representative throughout the power curve calculation
    
    # prob['dynamic_amplication'] = 5
    
    # ----------------------

    # === aero and structural analysis options ===
    prob['nSector']          = 4  # (Int): number of azimuth values for the rotor to estimate thrust and power
    prob['nostallconstraint.min_s']        = 0.15  # The stall constraint is only computed from this value (nondimensional coordinate along blade span) to blade tip
    prob['nostallconstraint.stall_margin'] = 2.0   # Values in deg of stall margin
    
    prob['AEP_loss_factor']  = 1.0  # (Float): availability and other losses (soiling, array, etc.)
    prob['drivetrainType']   = DRIVETRAIN_TYPE[blade['config']['drivetrain'].upper()] #DRIVETRAIN_TYPE['GEARED']  # (Enum)
    prob['dynamic_amplication'] = 1.  # (Float): a dynamic amplification factor to adjust the static structural loads
    prob['tiploss']          = True
    prob['hubloss']          = True
    
    
    
    # Environmental parameters
    prob['wind_reference_height']          = blade['config']['hub_height']
    prob['shearExp']                       = 0.

    # Steel properties
    prob['material_density']               = 7850.0
    prob['E']                              = 200e9
    prob['G']                              = 79.3e9
    prob['yield_stress']                   = 3.45e8

    # Design constraints
    prob['max_taper_ratio']                = 0.4
    prob['min_diameter_thickness_ratio']   = 120.0


    prob['gamma_fatigue']   = 1.755   # (Float): safety factor for fatigue
    prob['gamma_f']         = 1.35          # (Float): safety factor for loads/stresses
    prob['gamma_m']         = 1.3           # (Float): safety factor for materials
    prob['gamma_freq']      = 1.1        # (Float): safety factor for resonant frequencies
    prob['gamma_n']         = 1.0
    prob['gamma_b']         = 1.1
    
    # Tower
    prob['hub_height']                     = blade['config']['hub_height']
    prob['foundation_height']              = -prob['water_depth']
    prob['tower_outer_diameter']           = np.linspace(10.00, 6.00, NSECTION+1)
    prob['tower_section_height']           = (prob['hub_height'] - prob['foundation_height']) / NSECTION * np.ones(NSECTION)
    prob['tower_wall_thickness']           = np.linspace(0.030, 0.022, NSECTION)
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

    
    
    
    prob['rna_weightM'] = True

    # For turbine costs
    prob['offshore']             = False
    prob['crane']                = False
    prob['bearing_number']       = 2
    

    prob['drivetrainType']          = DRIVETRAIN_TYPE[blade['config']['drivetrain'].upper()]
    prob['drive.gear_ratio']        = 96.76  # 97:1 as listed in the 5 MW reference document
    prob['drive.shaft_angle']       = 6.0*np.pi / 180.0  # rad
    prob['drive.shaft_ratio']       = 0.10
    prob['drive.planet_numbers']    = [3, 3, 1]
    prob['drive.shrink_disc_mass']  = 333.3 * prob['machine_rating'] / 1.e006  # estimated

    
    prob['drive.carrier_mass']      = 8000.0  # estimated
    prob['drive.flange_length']     = 0.5
    prob['drive.overhang']          = 7.0
    prob['drive.distance_hub2mb']   = 1.912  # length from hub center to main bearing, leave zero if unknown
    prob['drive.gearbox_input_xcm'] = 0.1
    prob['drive.hss_input_length']  = 1.5
    
    
    
    
    # Plant finance
    prob['plantfinancese.project_lifetime']  = 20.0
    prob['number_of_turbines']               = 20
    prob['annual_opex']                      = 70.  * prob['machine_rating'] * 1.e-003 # 70 $/kW/yr
    prob['fixed_charge_rate']                = 0.079 # 7.9 %
    prob['discount_rate']                    = 0.07
    prob['plantfinancese.turbine_bos_costs'] = 250. * prob['machine_rating'] * 1.e-003 # 250 $/kW
    prob['plantfinancese.wake_loss_factor']  = 0.15
    
    
    
    
    
    
    
    
    
    
    # ----------------------


    return prob    
    
    


if __name__ == "__main__":
    
    # Input wind turbine files
    WT_input      = 'BAR004e.yaml'
    WT_output     = 'BAR004e_outputs.yaml'
    folder_output = './../reference_turbines/bar/outputs/it_assembly_1/'
    schema        = 'IEAontology_schema.yaml'
    folder_input  = './../reference_turbines/bar/'
    
    fname_schema  = folder_input + schema
    fname_input   = folder_input + WT_input
    fname_output  = folder_input + WT_output
    
    
    
    if os.path.isdir(folder_output):
        shutil.rmtree(folder_output)
    os.mkdir(folder_output)
    
    
    # Input options
    refBlade              = ReferenceBlade()
    refBlade.verbose      = True
    refBlade.NINPUT       = 8
    refBlade.NPITS        = 100
    refBlade.spar_var     = ['Spar_Cap_SS', 'Spar_Cap_PS']
    refBlade.te_var       = 'TE_reinforcement'
    refBlade.validate     = True
    refBlade.fname_schema = fname_schema    
    generate_plots        = True
    show_plots            = False
    flag_write_out        = True
    
    # Problem initialization
    blade           = refBlade.initialize(fname_input)
    
    refBlade.plot_design(blade, folder_output, show_plots = True)  
    # refBlade.smooth_outer_shape(blade, folder_output, show_plots = True)
    # exit()
    
    
    
    
    # Initialize OpenMDAO problem and FloatingSE Group
    prob_ref = Problem(root=Onshore_Assembly(blade, folder_output))
    
    prob_ref.setup(out_stream = open(os.devnull, 'w'))
    
    
    prob_ref = Init_Assembly(prob_ref, blade, fst_vt={})
    
    prob_ref.run()
    
    if generate_plots == True:
        # Pitch
        fp, axp  = plt.subplots(1,1,figsize=(5.3, 4))
        axp.plot(prob_ref['powercurve.V'], prob_ref['powercurve.pitch'])
        plt.xlabel('Wind Speed [m/s]', fontsize=14, fontweight='bold')
        plt.ylabel('Pitch Angle [deg]', fontsize=14, fontweight='bold')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(color=[0.8,0.8,0.8], linestyle='--')
        plt.subplots_adjust(bottom = 0.15, left = 0.15)
        fig_name = 'pitch.png'
        fp.savefig(folder_output + fig_name)
        
        # Power
        fpw, axpw  = plt.subplots(1,1,figsize=(5.3, 4))
        axpw.plot(prob_ref['powercurve.V'], prob_ref['powercurve.P'] * 1.00e-006)
        plt.xlabel('Wind Speed [m/s]', fontsize=14, fontweight='bold')
        plt.ylabel('Electrical Power [MW]', fontsize=14, fontweight='bold')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(color=[0.8,0.8,0.8], linestyle='--')
        plt.subplots_adjust(bottom = 0.15, left = 0.15)
        fig_name = 'power.png'
        fpw.savefig(folder_output + fig_name)
        
        # Omega
        fo, axo  = plt.subplots(1,1,figsize=(5.3, 4))
        axo.plot(prob_ref['powercurve.V'], prob_ref['powercurve.Omega'])
        plt.xlabel('Wind Speed [m/s]', fontsize=14, fontweight='bold')
        plt.ylabel('Rotor Speed [rpm]', fontsize=14, fontweight='bold')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(color=[0.8,0.8,0.8], linestyle='--')
        plt.subplots_adjust(bottom = 0.15, left = 0.15)
        fig_name = 'omega.png'
        fo.savefig(folder_output + fig_name)
        
        # Omega
        fts, axts  = plt.subplots(1,1,figsize=(5.3, 4))
        axts.plot(prob_ref['powercurve.V'], prob_ref['powercurve.Omega'] * np.pi / 30. * prob_ref['r_pts'][-1])
        plt.xlabel('Wind Speed [m/s]', fontsize=14, fontweight='bold')
        plt.ylabel('Blade Tip Speed [m/s]', fontsize=14, fontweight='bold')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(color=[0.8,0.8,0.8], linestyle='--')
        plt.subplots_adjust(bottom = 0.15, left = 0.15)
        fig_name = 'tip_speed.png'
        fts.savefig(folder_output + fig_name)
        
        # Thrust
        ft, axt  = plt.subplots(1,1,figsize=(5.3, 4))
        axt.plot(prob_ref['powercurve.V'], prob_ref['powercurve.T'] * 1.00e-006)
        plt.xlabel('Wind Speed [m/s]', fontsize=14, fontweight='bold')
        plt.ylabel('Rotor Thrust [MN]', fontsize=14, fontweight='bold')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(color=[0.8,0.8,0.8], linestyle='--')
        plt.subplots_adjust(bottom = 0.15, left = 0.15)
        fig_name = 'thrust.png'
        ft.savefig(folder_output + fig_name)
        
        # Torque
        fq, axq  = plt.subplots(1,1,figsize=(5.3, 4))
        axq.plot(prob_ref['powercurve.V'], prob_ref['powercurve.Q'] * 1.00e-006)
        plt.xlabel('Wind Speed [m/s]', fontsize=14, fontweight='bold')
        plt.ylabel('Rotor Torque [MNm]', fontsize=14, fontweight='bold')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(color=[0.8,0.8,0.8], linestyle='--')
        plt.subplots_adjust(bottom = 0.15, left = 0.15)
        fig_name = 'torque.png'
        fq.savefig(folder_output + fig_name)
        
        if show_plots == True:
            plt.show()


    # Optimization
    prob            = Problem(root=Onshore_Assembly(blade, folder_output))
    # prob.driver = pyOptSparseDriver()
    # prob.driver.options['optimizer']                       = 'SNOPT'
    # prob.driver.opt_settings['Major optimality tolerance'] = 1.e-3
    # prob.driver.opt_settings['Major iterations limit']     = 10
    # prob.root.deriv_options['check_form'] = 'central'
    # # prob.driver.opt_settings['Difference interval']        = 1.e-4
    # prob.root.deriv_options['type']    = 'fd'
    # prob.root.deriv_options['step_calc'] = 'relative'
    
    prob.driver    = ScipyOptimizer()
    prob.driver.options['optimizer']   = 'SLSQP'
    prob.driver.options['maxiter']     = 10
    prob.driver.options['tol'] = 1.0e-6
    prob.root.deriv_options['step_size'] = 1e-2
    
    
    
    # Define opt objective
    # merit_figure_ref = prob_ref['AEP']
    # prob.driver.add_objective('AEP', scaler=-1./merit_figure_ref)

    merit_figure_ref = prob_ref['plantfinancese.lcoe']
    prob.driver.add_objective('plantfinancese.lcoe', scaler=1./merit_figure_ref)
    
    # merit_figure_ref = prob_ref['mass_one_blade']
    # prob.driver.add_objective('mass_one_blade', scaler=1./merit_figure_ref)
    
    
    # Optimization variables
    indices_no_root         = range(2,len(prob_ref['chord_in']))
    indices_no_root_no_tip  = range(2,len(prob_ref['chord_in'])-1)

    
    # Chord
    lb_chord_meter = 1.0 # In meters
    ub_chord_meter = 7.0 # In meters
    lb_chord_prop  = 0.8 # Proportional to existing chord
    ub_chord_prop  = 1.7 # Proportional to existing chord
    
    lb_opt_chord   = np.maximum(lb_chord_prop * np.ones(len(indices_no_root_no_tip)), lb_chord_meter  * np.ones(len(indices_no_root_no_tip)) / prob_ref['chord_in'][indices_no_root_no_tip])
    ub_opt_chord   = np.minimum(ub_chord_prop * np.ones(len(indices_no_root_no_tip)), ub_chord_meter  * np.ones(len(indices_no_root_no_tip)) / prob_ref['chord_in'][indices_no_root_no_tip])
    if min(lb_opt_chord) < 1.e-6 or min(ub_opt_chord) < 1.e-6:
        exit('Negative or zero chord values are not allowed. Check the optimization bounds')
    elif max(lb_opt_chord) > 1. or min(ub_opt_chord) < 1.:
        print('Initial chord design violates the bounds')
    
    # Twist
    lb_opt_theta   = np.zeros(len(indices_no_root))
    ub_opt_theta   = np.ones(len(indices_no_root))   
    
    # Spar caps thickness
    lb_sparT_meter = 0.001 # In meters
    ub_sparT_meter = 0.250 # In meters
    lb_sparT_prop  = 0.100 # Proportional to existing spar thickness
    ub_sparT_prop  = 3.000 # Proportional to existing spar thickness
    
    lb_opt_sparT   = np.maximum(lb_sparT_prop * np.ones(len(indices_no_root_no_tip)), lb_sparT_meter  * np.ones(len(indices_no_root_no_tip)) / prob_ref['sparT_in'][indices_no_root_no_tip])
    ub_opt_sparT   = np.minimum(ub_sparT_prop * np.ones(len(indices_no_root_no_tip)), ub_sparT_meter  * np.ones(len(indices_no_root_no_tip)) / prob_ref['sparT_in'][indices_no_root_no_tip])
    if min(lb_opt_sparT) < 1.e-6 or min(ub_opt_sparT) < 1.e-6:
        exit('Negative or zero spar caps thickness values are not allowed. Check the optimization bounds')
    elif max(lb_opt_sparT) > 1. or min(ub_opt_sparT) < 1.:
        print('Initial spar caps thickness design violates the bounds')
    
    
    prob.driver.add_desvar('opt_chord', indices = indices_no_root_no_tip, lower = lb_opt_chord, upper = ub_opt_chord)
    prob.driver.add_desvar('opt_theta', indices = indices_no_root,        lower = lb_opt_theta, upper = ub_opt_theta)
    prob.driver.add_desvar('opt_sparT', indices = indices_no_root_no_tip, lower = lb_opt_sparT, upper = ub_opt_sparT)
    
    # prob.driver.add_desvar('control_tsr', lower=  8.00, upper=11.0)
    # prob.driver.add_desvar('tower_section_height', lower=5.0, upper=80.0)
    # prob.driver.add_desvar('tower_outer_diameter', lower=4.00, upper=20.0)
    # prob.driver.add_desvar('tower_wall_thickness', lower=4e-3, upper=2e-1)
    
    # Define constraints
    # RBM0 = prob_ref['root_bending_moment']
    # AEP0 = prob_ref['AEP']
    # BC0 = prob_ref['total_blade_cost']
    # prob.driver.add_constraint('total_blade_cost',     upper=BC0)
    # prob.driver.add_constraint('root_bending_moment',  upper=RBM0)
    # prob.driver.add_constraint('AEP',                  lower=AEP0)
    # prob.driver.add_constraint('buckling_const',       upper=np.zeros(len(blade['pf']['s'])))
    
    # --- Constraints ---
    # prob.driver.add_constraint('tow.height_constraint',     lower=-1e-2, upper=1.e-2)
    # prob.driver.add_constraint('tow.post.stress',           upper=1.0)
    # prob.driver.add_constraint('tow.post.global_buckling',  upper=1.0)
    # prob.driver.add_constraint('tow.post.shell_buckling',   upper=1.0)
    # prob.driver.add_constraint('tow.weldability',           upper=0.0)
    # prob.driver.add_constraint('tow.manufacturability',     lower=0.0)
    # prob.driver.add_constraint('frequency1P_margin_low',    upper=1.0)
    # prob.driver.add_constraint('frequency1P_margin_high',   lower=1.0)
    # prob.driver.add_constraint('frequency3P_margin_low',    upper=1.0)
    # prob.driver.add_constraint('frequency3P_margin_high',   lower=1.0)
    prob.driver.add_constraint('tip_deflection_ratio',  upper=1.0)  
    prob.driver.add_constraint('no_stall_constraint',   upper=1.0)  
    
    # Optimization recorder
    filename_opt_log = folder_output + 'log_opt_' + blade['config']['name']

    
    rec = SqliteRecorder(filename_opt_log)
    rec.options['record_metadata'] = False
    rec.options['record_unknowns'] = True
    rec.options['record_params']   = False
    rec.options['includes']        = ['chord_in','theta_in','sparT_in','AEP','control_tsr','total_blade_cost','bcm.total_blade_mass','plantfinancese.lcoe','z_full','d_full','t_full','tow.height_constraint','tow.post.stress','tow.post.global_buckling','tow.post.shell_buckling','tow.weldability', 'tow.manufacturability', 'frequency1P_margin_low', 'frequency1P_margin_high', 'frequency3P_margin_low', 'frequency3P_margin_high', 'tip_deflection_ratio']
    prob.driver.add_recorder(rec)
    
    prob.setup(out_stream = open(os.devnull, 'w'))
    prob = Init_Assembly(prob, blade, fst_vt={})
    prob.run()
    
    
    
    # Outputs plotting
    print('AEP:         \t\t\t %f\t%f GWh \t Difference: %f %%' % (prob_ref['AEP']*1e-6, prob['AEP']*1e-6, (prob['AEP']-prob_ref['AEP'])/prob_ref['AEP']*100.))
    print('LCoE:        \t\t\t %f\t%f USD/MWh \t Difference: %f %%' % (prob_ref['plantfinancese.lcoe']*1.e003, prob['plantfinancese.lcoe']*1.e003, (prob['plantfinancese.lcoe']-prob_ref['plantfinancese.lcoe'])/prob_ref['plantfinancese.lcoe']*100.))
    print('Blade cost:  \t\t\t %f\t%f USD \t Difference: %f %%' % (prob_ref['total_blade_cost'], prob['total_blade_cost'], (prob['total_blade_cost']-prob_ref['total_blade_cost'])/prob_ref['total_blade_cost']*100.))
    print('Blade mass:  \t\t\t %f\t%f kg  \t Difference: %f %%' % (prob_ref['bcm.total_blade_mass'], prob['bcm.total_blade_mass'], (prob['bcm.total_blade_mass']-prob_ref['bcm.total_blade_mass'])/prob_ref['bcm.total_blade_mass']*100.))
    print('Tower cost:  \t\t\t %f\t%f USD \t Difference: %f %%' % (prob_ref['tower_cost'], prob['tower_cost'], (prob['tower_cost']-prob_ref['tower_cost'])/prob_ref['tower_cost']*100.))
    print('Tower mass:  \t\t\t %f\t%f kg  \t Difference: %f %%' % (prob_ref['tower_mass'], prob['tower_mass'], (prob['tower_mass']-prob_ref['tower_mass'])/prob_ref['tower_mass']*100.))
    
    ## save output yaml
    if flag_write_out:
        t3 = time.time()
        refBlade.write_ontology(fname_output, prob['blade_out'], refBlade.wt_ref)
        if refBlade.verbose:
            print('Complete: Write Output: \t%f s'%(time.time()-t3))
    
    
    r       = prob['r_pts']
    rc_ref  = prob['r_in']
    rc      = prob['r_in']
    
    
    shutil.copyfile(fname_input,  folder_output + WT_input)
    shutil.copyfile(fname_output, folder_output + WT_output)

    if generate_plots == True:
        # Angle of attack and stall angle
        faoa, axaoa = plt.subplots(1,1,figsize=(5.3, 4))
        axaoa.plot(r, prob_ref['nostallconstraint.aoa_along_span'], label='Initial aoa')
        axaoa.plot(r, prob_ref['nostallconstraint.stall_angle_along_span'], '.', label='Initial stall')
        axaoa.plot(r, prob['nostallconstraint.aoa_along_span'], label='Optimized aoa')
        axaoa.plot(r, prob['nostallconstraint.stall_angle_along_span'], '.', label='Optimized stall')
        axaoa.legend(fontsize=12)
        plt.xlabel('Blade Span [m]', fontsize=14, fontweight='bold')
        plt.ylabel('Angle [deg]', fontsize=14, fontweight='bold')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(color=[0.8,0.8,0.8], linestyle='--')
        plt.subplots_adjust(bottom = 0.15, left = 0.15)
        fig_name = 'aoa.png'
        faoa.savefig(folder_output + fig_name)
        
        # Induction
        fi, axi = plt.subplots(1,1,figsize=(5.3, 4))
        axi.plot(r, prob_ref['powercurve.ax_induct_cutin'], label='Initial')
        axi.plot(r, prob['powercurve.ax_induct_cutin'], label='Optimized')
        axi.legend(fontsize=12)
        plt.xlabel('Blade Span [m]', fontsize=14, fontweight='bold')
        plt.ylabel('Axial Induction [-]', fontsize=14, fontweight='bold')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(color=[0.8,0.8,0.8], linestyle='--')
        plt.subplots_adjust(bottom = 0.15, left = 0.15)
        fig_name = 'induction.png'
        fi.savefig(folder_output + fig_name)
        
        # Chord
        fc, axc = plt.subplots(1,1,figsize=(5.3, 4))
        axc.plot(r, prob_ref['chord'], label='Initial')
        axc.plot(rc_ref, prob_ref['chord_in'], '.')
        axc.plot(r, prob['chord'], label='Optimized')
        axc.plot(rc, prob['chord_in'], '.')
        axc.legend(fontsize=12)
        plt.xlabel('Blade Span [m]', fontsize=14, fontweight='bold')
        plt.ylabel('Chord [m]', fontsize=14, fontweight='bold')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(color=[0.8,0.8,0.8], linestyle='--')
        plt.subplots_adjust(bottom = 0.15, left = 0.15)
        fig_name = 'chord.png'
        fc.savefig(folder_output + fig_name)
        
        
        
        # Theta
        ft, axt = plt.subplots(1,1,figsize=(5.3, 4))
        axt.plot(r, prob_ref['theta'], label='Initial')
        axt.plot(rc_ref, prob_ref['theta_in'], '.')
        axt.plot(r, prob['theta'], label='Optimized')
        axt.plot(rc, prob['theta_in'], '.')
        axt.legend(fontsize=12)
        plt.xlabel('Blade Span [m]', fontsize=14, fontweight='bold')
        plt.ylabel('Twist [deg]', fontsize=14, fontweight='bold')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(color=[0.8,0.8,0.8], linestyle='--')
        plt.subplots_adjust(bottom = 0.15, left = 0.15)
        fig_name = 'theta.png'
        ft.savefig(folder_output + fig_name)

        # Spar caps
        fs, axs = plt.subplots(1,1,figsize=(5.3, 4))
        axs.plot(rc_ref, prob_ref['sparT_in']*1e+003, label='Initial')
        axs.plot(rc, prob['sparT_in']*1e+003, label='Optimized')
        axs.legend(fontsize=12)
        plt.xlabel('Blade Span [m]', fontsize=14, fontweight='bold')
        plt.ylabel('Spar Caps Thickness [m]', fontsize=14, fontweight='bold')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(color=[0.8,0.8,0.8], linestyle='--')
        plt.subplots_adjust(bottom = 0.15, left = 0.15)
        fig_name = 'spars.png'
        fs.savefig(folder_output + fig_name)
        
        
        # Tower diameter
        ftd, axtd = plt.subplots(1,1,figsize=(5.3, 4))
        axtd.plot(prob_ref['z_full'], prob_ref['d_full'], label='Initial')
        axtd.plot(prob['z_full'], prob['d_full'], label='Optimized')
        axtd.legend(fontsize=12)
        plt.xlabel('Tower Height [m]', fontsize=14, fontweight='bold')
        plt.ylabel('Tower Diameter [m]', fontsize=14, fontweight='bold')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(color=[0.8,0.8,0.8], linestyle='--')
        plt.subplots_adjust(bottom = 0.15, left = 0.15)
        fig_name = 'tower_d.png'
        ftd.savefig(folder_output + fig_name)

        
        # Tower thickness
        ftt, axtt = plt.subplots(1,1,figsize=(5.3, 4))
        axtt.plot(prob_ref['z_full'], np.hstack((prob_ref['t_full'],prob_ref['t_full'][-1])), label='Initial')
        axtt.plot(prob['z_full'], np.hstack((prob['t_full'],prob['t_full'][-1])), label='Optimized')
        axtt.legend(fontsize=12)
        plt.xlabel('Tower Height [m]', fontsize=14, fontweight='bold')
        plt.ylabel('Tower Wall Thickness [m]', fontsize=14, fontweight='bold')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(color=[0.8,0.8,0.8], linestyle='--')
        plt.subplots_adjust(bottom = 0.15, left = 0.15)
        fig_name = 'tower_t.png'
        ftt.savefig(folder_output + fig_name)
        
        print(prob['tower_outer_diameter'])
        print(prob['tower_section_height'])
        print(prob['tower_wall_thickness'])
        
        if show_plots == True:
            plt.show()


    
