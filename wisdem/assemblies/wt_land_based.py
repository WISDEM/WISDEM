import numpy as np
import os
import matplotlib.pyplot as plt
from openmdao.api import ExplicitComponent, Group, IndepVarComp, Problem, SqliteRecorder, ScipyOptimizeDriver, CaseReader
from wisdem.assemblies.load_IEA_yaml import WindTurbineOntologyPython, WindTurbineOntologyOpenMDAO, yaml2openmdao
from wisdem.rotorse.rotor_geometry import TurbineClass
from wisdem.rotorse.wt_rotor import WT_Rotor
from wisdem.drivetrainse.drivese_omdao import DriveSE
from wisdem.turbine_costsse.turbine_costsse_2015 import Turbine_CostsSE_2015
from wisdem.plant_financese.plant_finance import PlantFinance
from wisdem.commonse.turbine_constraints  import TurbineConstraints

class Opt_Data(object):
    # Pure python class to set the optimization parameters:

    def __init__(self):
        
        self.opt_options = {}

        # Save data
        self.folder_output    = 'it_0/'
        self.optimization_log = 'log_opt.sql'
        self.costs_verbosity  = False

        # Blade aerodynamic optimization parameters
        self.n_opt_twist   = 8
        self.n_opt_chord   = 8
        self.n_opt_spar_ss = 8
        self.n_opt_spar_ps = 8

    def initialize(self):

        self.opt_options['folder_output']    = self.folder_output
        self.opt_options['optimization_log'] = self.folder_output + self.optimization_log
        self.opt_options['costs_verbosity']  = self.costs_verbosity

        self.opt_options['blade_aero'] = {}
        self.opt_options['blade_aero']['n_opt_twist'] = self.n_opt_twist
        self.opt_options['blade_aero']['n_opt_chord'] = self.n_opt_chord

        self.opt_options['blade_struct'] = {}
        self.opt_options['blade_struct']['te_ss_var']   = 'TE_reinforcement'
        self.opt_options['blade_struct']['te_ps_var']   = 'TE_reinforcement'
        self.opt_options['blade_struct']['spar_ss_var'] = 'Spar_Cap_SS'
        self.opt_options['blade_struct']['spar_ps_var'] = 'Spar_Cap_PS'
        self.opt_options['blade_struct']['n_opt_spar_ss'] = self.n_opt_spar_ss
        self.opt_options['blade_struct']['n_opt_spar_ps'] = self.n_opt_spar_ps



        return self.opt_options

class WT_RNTA(Group):
    # Openmdao group to run the analysis of the wind turbine
    
    def initialize(self):
        self.options.declare('wt_init_options')
        self.options.declare('opt_options')
        
    def setup(self):
        wt_init_options = self.options['wt_init_options']
        opt_options     = self.options['opt_options']

        # Analysis components
        self.add_subsystem('wt_init',   WindTurbineOntologyOpenMDAO(wt_init_options = wt_init_options), promotes=['*'])
        self.add_subsystem('wt_class',  TurbineClass())
        self.add_subsystem('rotorse',   WT_Rotor(wt_init_options = wt_init_options, opt_options = opt_options))
        self.add_subsystem('drivese',   DriveSE(debug=False,
                                            number_of_main_bearings=1,
                                            topLevelFlag=False))
        # self.add_subsystem('towerse',   TowerSE())
        self.add_subsystem('tcons',     TurbineConstraints(wt_init_options = wt_init_options))
        self.add_subsystem('tcc',       Turbine_CostsSE_2015(verbosity=opt_options['costs_verbosity'], topLevelFlag=False))
        # Post-processing
        self.add_subsystem('outputs_2_screen',  Outputs_2_Screen())
        self.add_subsystem('conv_plots',        Convergence_Trends_Opt(opt_options = opt_options))

        # Connections to wind turbine class
        self.connect('configuration.ws_class' , 'wt_class.turbine_class')
        # Connections to rotorse and its subcomponents
        # Connections to blade parametrization
        self.connect('blade.outer_shape_bem.s',    ['rotorse.pa.s', 'rotorse.ps.s','rotorse.rs.constr.s'])
        self.connect('blade.outer_shape_bem.twist', 'rotorse.pa.twist_original')
        self.connect('blade.outer_shape_bem.chord', 'rotorse.pa.chord_original')
        self.connect('blade.internal_structure_2d_fem.layer_name',      'rotorse.ps.layer_name')
        self.connect('blade.internal_structure_2d_fem.layer_thickness', 'rotorse.ps.layer_thickness_original')
        # Connections to rotor aeropower
        self.connect('wt_class.V_mean',         'rotorse.ra.cdf.xbar')
        self.connect('control.V_in' ,           'rotorse.ra.control_Vin')
        self.connect('control.V_out' ,          'rotorse.ra.control_Vout')
        self.connect('control.rated_power' ,    'rotorse.ra.control_ratedPower')
        self.connect('control.min_Omega' ,      'rotorse.ra.control_minOmega')
        self.connect('control.max_Omega' ,      'rotorse.ra.control_maxOmega')
        self.connect('control.max_TS' ,         'rotorse.ra.control_maxTS')
        self.connect('control.rated_TSR' ,      'rotorse.ra.control_tsr')
        self.connect('control.rated_pitch' ,        'rotorse.ra.control_pitch')
        self.connect('configuration.gearbox_type' , 'rotorse.ra.drivetrainType')
        self.connect('assembly.r_blade',            'rotorse.ra.r')
        self.connect('assembly.rotor_radius',       'rotorse.ra.Rtip')
        self.connect('hub.radius',                  'rotorse.ra.Rhub')
        self.connect('assembly.hub_height',         'rotorse.ra.hub_height')
        self.connect('hub.cone',                    'rotorse.ra.precone')
        self.connect('nacelle.uptilt',              'rotorse.ra.tilt')
        self.connect('airfoils.aoa',                    'rotorse.ra.airfoils_aoa')
        self.connect('airfoils.Re',                     'rotorse.ra.airfoils_Re')
        self.connect('blade.interp_airfoils.cl_interp', 'rotorse.ra.airfoils_cl')
        self.connect('blade.interp_airfoils.cd_interp', 'rotorse.ra.airfoils_cd')
        self.connect('blade.interp_airfoils.cm_interp', 'rotorse.ra.airfoils_cm')
        self.connect('configuration.n_blades',          'rotorse.ra.nBlades')
        self.connect('blade.outer_shape_bem.s',         'rotorse.ra.stall_check.s')
        self.connect('env.rho_air',                     'rotorse.ra.rho')
        self.connect('env.mu_air',                      'rotorse.ra.mu')
        self.connect('env.weibull_k',                   'rotorse.ra.cdf.k')
        # Connections to rotor structure
        self.connect('assembly.r_blade',                                'rotorse.rs.r')
        self.connect('blade.outer_shape_bem.pitch_axis',                'rotorse.rs.precomp.pitch_axis')
        self.connect('blade.interp_airfoils.coord_xy_interp',           'rotorse.rs.precomp.coord_xy_interp')
        self.connect('blade.internal_structure_2d_fem.layer_start_nd',  'rotorse.rs.precomp.layer_start_nd')
        self.connect('blade.internal_structure_2d_fem.layer_end_nd',    'rotorse.rs.precomp.layer_end_nd')
        self.connect('blade.internal_structure_2d_fem.layer_name',      'rotorse.rs.precomp.layer_name')
        self.connect('blade.internal_structure_2d_fem.layer_web',       'rotorse.rs.precomp.layer_web')
        self.connect('blade.internal_structure_2d_fem.layer_mat',       'rotorse.rs.precomp.layer_mat')
        self.connect('blade.internal_structure_2d_fem.definition_layer','rotorse.rs.precomp.definition_layer')
        self.connect('blade.internal_structure_2d_fem.web_start_nd',    'rotorse.rs.precomp.web_start_nd')
        self.connect('blade.internal_structure_2d_fem.web_end_nd',      'rotorse.rs.precomp.web_end_nd')
        self.connect('blade.internal_structure_2d_fem.web_name',        'rotorse.rs.precomp.web_name')
        self.connect('materials.name',  'rotorse.rs.precomp.mat_name')
        self.connect('materials.orth',  'rotorse.rs.precomp.orth')
        self.connect('materials.E',     'rotorse.rs.precomp.E')
        self.connect('materials.G',     'rotorse.rs.precomp.G')
        self.connect('materials.nu',    'rotorse.rs.precomp.nu')
        self.connect('materials.rho',   'rotorse.rs.precomp.rho')
        self.connect('assembly.rotor_radius',           'rotorse.rs.Rtip')
        self.connect('hub.radius',                      'rotorse.rs.Rhub')
        self.connect('assembly.hub_height',             'rotorse.rs.hub_height')
        self.connect('hub.cone',                        'rotorse.rs.precone')
        self.connect('nacelle.uptilt',                  'rotorse.rs.tilt')
        self.connect('airfoils.aoa',                    'rotorse.rs.airfoils_aoa')
        self.connect('airfoils.Re',                     'rotorse.rs.airfoils_Re')
        self.connect('blade.interp_airfoils.cl_interp', 'rotorse.rs.airfoils_cl')
        self.connect('blade.interp_airfoils.cd_interp', 'rotorse.rs.airfoils_cd')
        self.connect('blade.interp_airfoils.cm_interp', 'rotorse.rs.airfoils_cm')
        self.connect('configuration.n_blades',          'rotorse.rs.nBlades')
        self.connect('env.rho_air',                     'rotorse.rs.rho')
        self.connect('env.mu_air',                      'rotorse.rs.mu')
        # Connections to rotorse-rs-gustetm
        self.connect('wt_class.V_mean',                 'rotorse.rs.gust.V_mean')
        self.connect('configuration.turb_class',        'rotorse.rs.gust.turbulence_class')
        # self.connect('wt_class.V_extreme1',             'rotorse.rs.aero_storm_1yr.V_load')
        # self.connect('wt_class.V_extreme50',            'rotorse.rs.aero_storm_50yr.V_load')
        
        # Connections to rotorse-rc
        # self.connect('blade.length',                                    'rotorse.rc.blade_length')
        # self.connect('blade.outer_shape_bem.s',                         'rotorse.rc.s')
        # self.connect('blade.outer_shape_bem.pitch_axis',                'rotorse.rc.pitch_axis')
        # self.connect('blade.interp_airfoils.coord_xy_interp',           'rotorse.rc.coord_xy_interp')
        # self.connect('blade.internal_structure_2d_fem.layer_start_nd',  'rotorse.rc.layer_start_nd')
        # self.connect('blade.internal_structure_2d_fem.layer_end_nd',    'rotorse.rc.layer_end_nd')
        # self.connect('blade.internal_structure_2d_fem.layer_name',      'rotorse.rc.layer_name')
        # self.connect('blade.internal_structure_2d_fem.layer_web',       'rotorse.rc.layer_web')
        # self.connect('blade.internal_structure_2d_fem.layer_mat',       'rotorse.rc.layer_mat')
        # self.connect('blade.internal_structure_2d_fem.web_start_nd',    'rotorse.rc.web_start_nd')
        # self.connect('blade.internal_structure_2d_fem.web_end_nd',      'rotorse.rc.web_end_nd')
        # self.connect('blade.internal_structure_2d_fem.web_name',        'rotorse.rc.web_name')
        # self.connect('materials.name',          'rotorse.rc.mat_name')
        # self.connect('materials.rho',           'rotorse.rc.rho')
        self.connect('materials.component_id',  'rotorse.rs.precomp.component_id')
        self.connect('materials.unit_cost',     'rotorse.rs.precomp.unit_cost')
        self.connect('materials.waste',         'rotorse.rs.precomp.waste')
        self.connect('materials.rho_fiber',     'rotorse.rs.precomp.rho_fiber')
        self.connect('materials.rho_area_dry',  'rotorse.rs.precomp.rho_area_dry')
        self.connect('materials.ply_t',         'rotorse.rs.precomp.ply_t')
        self.connect('materials.fvf',           'rotorse.rs.precomp.fvf')
        self.connect('materials.fwf',           'rotorse.rs.precomp.fwf')
        self.connect('materials.roll_mass',     'rotorse.rs.precomp.roll_mass')

        # Connections to DriveSE
        self.connect('assembly.rotor_diameter',    'drivese.rotor_diameter')     
        self.connect('control.rated_power',        'drivese.machine_rating')    
        self.connect('nacelle.overhang',           'drivese.overhang') 
        self.connect('nacelle.uptilt',             'drivese.shaft_angle')
        self.connect('configuration.n_blades',     'drivese.number_of_blades') 
        self.connect('rotorse.ra.powercurve.rated_Q',         'drivese.rotor_torque')
        self.connect('rotorse.ra.powercurve.rated_Omega',     'drivese.rotor_rpm')
        # self.connect('rotorse.rs.Fxyz_total',      'drivese.Fxyz')
        # self.connect('rotorse.rs.Mxyz_total',      'drivese.Mxyz')
        # self.connect('rotorse.rs.I_all_blades',    'drivese.blades_I')
        self.connect('rotorse.rs.blade_mass',      'drivese.blade_mass')
        self.connect('rotorse.pa.chord_param',     'drivese.blade_root_diameter', src_indices=[0])
        self.connect('blade.length',               'drivese.blade_length')
        self.connect('nacelle.gear_ratio',         'drivese.gear_ratio')
        self.connect('nacelle.shaft_ratio',        'drivese.shaft_ratio')
        self.connect('nacelle.planet_numbers',     'drivese.planet_numbers')
        self.connect('nacelle.shrink_disc_mass',   'drivese.shrink_disc_mass')
        self.connect('nacelle.carrier_mass',       'drivese.carrier_mass')
        self.connect('nacelle.flange_length',      'drivese.flange_length')
        self.connect('nacelle.gearbox_input_xcm',  'drivese.gearbox_input_xcm')
        self.connect('nacelle.hss_input_length',   'drivese.hss_input_length')
        self.connect('nacelle.distance_hub2mb',    'drivese.distance_hub2mb')
        self.connect('nacelle.yaw_motors_number',  'drivese.yaw_motors_number')
        self.connect('nacelle.drivetrain_eff',     'drivese.drivetrain_efficiency')
        self.connect('tower.diameter',             'drivese.tower_top_diameter', src_indices=[-1])
        # Connections to turbine constraints
        self.connect('rotorse.rs.tip_pos.tip_deflection',   'tcons.tip_deflection')
        self.connect('assembly.rotor_radius',               'tcons.Rtip')
        self.connect('blade.outer_shape_bem.ref_axis',  'tcons.ref_axis_blade')
        self.connect('hub.cone',                        'tcons.precone')
        self.connect('nacelle.uptilt',                  'tcons.tilt')
        self.connect('nacelle.overhang',                'tcons.overhang')
        self.connect('tower.ref_axis',                  'tcons.ref_axis_tower')
        self.connect('tower.diameter',                  'tcons.d_full')

        # Connections to turbine capital cost
        self.connect('control.rated_power',         'tcc.machine_rating')
        self.connect('rotorse.rs.blade_mass',       'tcc.blade_mass')
        self.connect('drivese.hub_mass',            'tcc.hub_mass')
        self.connect('drivese.pitch_system_mass',   'tcc.pitch_system_mass')
        self.connect('drivese.spinner_mass',        'tcc.spinner_mass')
        self.connect('drivese.lss_mass',            'tcc.lss_mass')
        self.connect('drivese.mainBearing.mb_mass', 'tcc.main_bearing_mass')
        self.connect('drivese.hss_mass',            'tcc.hss_mass')
        self.connect('drivese.generator_mass',      'tcc.generator_mass')
        self.connect('drivese.bedplate_mass',       'tcc.bedplate_mass')
        self.connect('drivese.yaw_mass',            'tcc.yaw_mass')
        self.connect('drivese.vs_electronics_mass', 'tcc.vs_electronics_mass')
        self.connect('drivese.hvac_mass',           'tcc.hvac_mass')
        self.connect('drivese.cover_mass',          'tcc.cover_mass')
        self.connect('drivese.platforms_mass',      'tcc.platforms_mass')
        self.connect('drivese.transformer_mass',    'tcc.transformer_mass')
        # self.connect('towerse.tower_mass',          'tcc.tower_mass')
        # Connections to outputs
        self.connect('rotorse.ra.AEP',          'outputs_2_screen.AEP')
        self.connect('rotorse.rs.blade_mass',   'outputs_2_screen.blade_mass')
        # self.connect('financese.lcoe',          'outputs_2_screen.lcoe')

class WindPark(Group):
    # Openmdao group to run the cost analysis of a wind park
    
    def initialize(self):
        self.options.declare('wt_init_options')
        self.options.declare('opt_options')
        
    def setup(self):
        wt_init_options = self.options['wt_init_options']
        opt_options     = self.options['opt_options']

        self.add_subsystem('wt',        WT_RNTA(wt_init_options = wt_init_options, opt_options = opt_options), promotes=['*'])
        self.add_subsystem('financese', PlantFinance(verbosity=opt_options['costs_verbosity']))
        
        # Input to plantfinancese from wt group
        self.connect('rotorse.ra.AEP',          'financese.turbine_aep')
        self.connect('tcc.turbine_cost_kW',     'financese.tcc_per_kW')
        # Inputs to plantfinancese from input yaml
        self.connect('control.rated_power',     'financese.machine_rating')
        self.connect('costs.turbine_number',    'financese.turbine_number')
        self.connect('costs.bos_per_kW',        'financese.bos_per_kW')
        self.connect('costs.opex_per_kW',       'financese.opex_per_kW')
        self.connect('costs.wake_loss_factor',  'financese.wake_loss_factor')
        self.connect('costs.fixed_charge_rate', 'financese.fixed_charge_rate')

class Convergence_Trends_Opt(ExplicitComponent):
    def initialize(self):
        
        self.options.declare('opt_options')
        
    def compute(self, inputs, outputs):
        
        folder_output       = self.options['opt_options']['folder_output']
        optimization_log    = self.options['opt_options']['optimization_log']

        if os.path.exists(optimization_log):
        
            cr = CaseReader(optimization_log)
            cases = cr.list_cases()
            rec_data = {}
            iterations = []
            for i, casei in enumerate(cases):
                iterations.append(i)
                it_data = cr.get_case(casei)
                
                # parameters = it_data.get_responses()
                for parameters in [it_data.get_responses(), it_data.get_design_vars()]:
                    for j, param in enumerate(parameters.keys()):
                        if i == 0:
                            rec_data[param] = []
                        rec_data[param].append(parameters[param])

            for param in rec_data.keys():
                fig, ax = plt.subplots(1,1,figsize=(5.3, 4))
                ax.plot(iterations, rec_data[param])
                ax.set(xlabel='Number of Iterations' , ylabel=param)
                fig_name = 'Convergence_trend_' + param + '.png'
                fig.savefig(folder_output + fig_name)
                plt.close(fig)

class Outputs_2_Screen(ExplicitComponent):
    # Class to print outputs on screen
    def setup(self):
        
        self.add_input('AEP', val=0.0, units = 'GW * h')
        self.add_input('blade_mass', val=0.0, units = 'kg')
        self.add_input('lcoe', val=0.0, units = 'USD/kW/h')
    def compute(self, inputs, outputs):
        print('########################################')
        print('Objectives')
        print('AEP:         {:8.10f} GWh'.format(inputs['AEP'][0]))
        print('Blade Mass:  {:8.10f} kg'.format(inputs['blade_mass'][0]))
        print('LCOE:        {:8.10f} $/kWh'.format(inputs['lcoe'][0]))
        print('########################################')

if __name__ == "__main__":

    ## File management
    fname_input    = "wisdem/wisdem/assemblies/reference_turbines/nrel5mw/nrel5mw_mod_update.yaml"
    fname_output   = "wisdem/wisdem/assemblies/reference_turbines/nrel5mw/nrel5mw_mod_update_output.yaml"
    folder_output  = 'it_1/'
    opt_flag_twist = False
    opt_flag_chord = False
    opt_flag_spar_ss = True
    opt_flag_spar_ps = True
    merit_figure     = 'Blade Mass' # 'AEP' - 'LCOE'
    # Load yaml data into a pure python data structure
    wt_initial               = WindTurbineOntologyPython()
    wt_initial.validate      = False
    wt_initial.fname_schema  = "wisdem/wisdem/assemblies/reference_turbines/IEAontology_schema.yaml"
    wt_init_options, wt_init = wt_initial.initialize(fname_input)
    
    # Optimization options
    optimization_data       = Opt_Data()
    optimization_data.folder_output = folder_output
    if opt_flag_twist == True:
        optimization_data.n_opt_twist = 8
    else:
        optimization_data.n_opt_twist = wt_initial.n_span
    if opt_flag_chord == True:
        optimization_data.n_opt_chord = 8
    else:
        optimization_data.n_opt_chord = wt_initial.n_span
    if opt_flag_spar_ss == True:
        optimization_data.n_opt_spar_ss = 8
    else:
        optimization_data.n_opt_spar_ss = wt_initial.n_span
    if opt_flag_spar_ps == True:
        optimization_data.n_opt_spar_ps = 8
    else:
        optimization_data.n_opt_spar_ps = wt_initial.n_span

    if opt_flag_twist or opt_flag_chord or opt_flag_spar_ss or opt_flag_spar_ps:
        opt_flag = True

    opt_options = optimization_data.initialize()
    if not os.path.isdir(folder_output):
        os.mkdir(folder_output)

    # Initialize openmdao problem
    wt_opt          = Problem()
    wt_opt.model    = WindPark(wt_init_options = wt_init_options, opt_options = opt_options)
    wt_opt.model.approx_totals(method='fd')
    
    if opt_flag == True:
        # Set optimization solver and options
        wt_opt.driver  = ScipyOptimizeDriver()
        wt_opt.driver.options['optimizer'] = 'SLSQP'
        wt_opt.driver.options['tol']       = 1.e-6
        wt_opt.driver.options['maxiter']   = 15

        # Set merit figure
        if merit_figure == 'AEP':
            wt_opt.model.add_objective('rotorse.ra.AEP', scaler = -1.e-6)
        elif merit_figure == 'Blade Mass':
            wt_opt.model.add_objective('rotorse.rs.blade_mass', scaler = 1.e-4)
        elif merit_figure == 'LCOE':
            wt_opt.model.add_objective('financese.lcoe', scaler = 1.e+2)
        else:
            exit('The merit figure ' + merit_figure + ' is not supported.')
        
        # Set optimization variables
        if opt_flag_twist == True:
            indices        = range(2,opt_options['blade_aero']['n_opt_twist'])
            wt_opt.model.add_design_var('rotorse.opt_var.twist_opt_gain', indices = indices, lower=0., upper=1.)
        if opt_flag_chord == True:
            indices  = range(2,opt_options['blade_aero']['n_opt_chord'] - 1)
            wt_opt.model.add_design_var('rotorse.opt_var.chord_opt_gain', indices = indices, lower=0.5, upper=1.5)
        if opt_flag_spar_ss == True:
            indices  = range(2,opt_options['blade_struct']['n_opt_spar_ss'] - 1)
            wt_opt.model.add_design_var('rotorse.opt_var.spar_ss_opt_gain', indices = indices, lower=0.5, upper=1.5)
        if opt_flag_spar_ps == True:
            indices  = range(2,opt_options['blade_struct']['n_opt_spar_ps'] - 1)
            wt_opt.model.add_design_var('rotorse.opt_var.spar_ps_opt_gain', indices = indices, lower=0.5, upper=1.5)

        # Set non-linear constraints
        wt_opt.model.add_constraint('rotorse.rs.pbeam.strainU_spar', upper= 1.) 
        wt_opt.model.add_constraint('rotorse.rs.pbeam.strainL_spar', upper= 1.) 
        wt_opt.model.add_constraint('tcons.tip_deflection_ratio',    upper= 1.0) 
        
        # Set recorder
        wt_opt.driver.add_recorder(SqliteRecorder(opt_options['optimization_log']))
        wt_opt.driver.recording_options['includes'] = ['rotorse.ra.AEP, rotorse.rs.blade_mass, financese.lcoe']
        wt_opt.driver.recording_options['record_objectives']  = True
        wt_opt.driver.recording_options['record_constraints'] = True
        wt_opt.driver.recording_options['record_desvars']     = True
    
    # Setup openmdao problem
    wt_opt.setup()
    
    # Load initial wind turbine data from wt_initial to the openmdao problem
    wt_opt = yaml2openmdao(wt_opt, wt_init_options, wt_init)
    wt_opt['rotorse.pa.s_opt_twist']   = np.linspace(0., 1., optimization_data.n_opt_twist)
    wt_opt['rotorse.pa.s_opt_chord']   = np.linspace(0., 1., optimization_data.n_opt_chord)
    wt_opt['rotorse.ps.s_opt_spar_ss'] = np.linspace(0., 1., optimization_data.n_opt_spar_ss)
    wt_opt['rotorse.ps.s_opt_spar_ps'] = np.linspace(0., 1., optimization_data.n_opt_spar_ps)
    wt_opt['rotorse.rs.constr.min_strainU_spar'] = -0.003
    wt_opt['rotorse.rs.constr.max_strainU_spar'] =  0.003
    wt_opt['rotorse.rs.constr.min_strainL_spar'] = -0.003
    wt_opt['rotorse.rs.constr.max_strainL_spar'] =  0.003

    # Build and run openmdao problem
    wt_opt.run_driver()

    # Save data coming from openmdao to an output yaml file
    wt_initial.write_ontology(wt_opt, fname_output)

    # Printing and plotting results
    print('AEP in GWh = ' + str(wt_opt['rotorse.ra.AEP']*1.e-6))
    print('Nat frequencies blades in Hz = ' + str(wt_opt['rotorse.rs.curvefem.freq']))
    print('Tip tower clearance in m     = ' + str(wt_opt['tcons.blade_tip_tower_clearance']))
    print('Tip deflection constraint    = ' + str(wt_opt['tcons.tip_deflection_ratio']))

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(wt_opt['assembly.r_blade'], wt_opt['rotorse.rs.pbeam.strainU_spar'], label='spar ss')
    plt.plot(wt_opt['assembly.r_blade'], wt_opt['rotorse.rs.pbeam.strainL_spar'], label='spar ps')
    plt.plot(wt_opt['assembly.r_blade'], wt_opt['rotorse.rs.pbeam.strainU_te'], label='te ss')
    plt.plot(wt_opt['assembly.r_blade'], wt_opt['rotorse.rs.pbeam.strainL_te'], label='te ps')
    plt.ylim([-5e-3, 5e-3])
    plt.xlabel('r [m]')
    plt.ylabel('strain [-]')
    plt.legend()
    plt.show()