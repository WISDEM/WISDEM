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


class Opt_Data(object):
    # Pure python class to set the optimization parameters:

    def __init__(self):
        
        self.opt_options = {}

        # Save data
        self.folder_output    = 'it_0/'
        self.optimization_log = 'log_opt.sql'

        # Blade aerodynamic optimization parameters
        self.n_opt_twist = 8
        self.n_opt_chord = 8

    def initialize(self):

        self.opt_options['folder_output']    = self.folder_output
        self.opt_options['optimization_log'] = self.folder_output + self.optimization_log
        
        self.opt_options['blade_aero'] = {}
        self.opt_options['blade_aero']['n_opt_twist'] = self.n_opt_twist
        self.opt_options['blade_aero']['n_opt_chord'] = self.n_opt_chord

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
        self.add_subsystem('tcc',       Turbine_CostsSE_2015(verbosity=True, topLevelFlag=False))
        # Post-processing
        self.add_subsystem('outputs_2_screen',  Outputs_2_Screen())
        self.add_subsystem('conv_plots',        Convergence_Trends_Opt(opt_options = opt_options))

        # Connections to wind turbine class
        self.connect('configuration.ws_class' , 'wt_class.turbine_class')
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
        # Connections to blade parametrization
        self.connect('blade.outer_shape_bem.s',     'rotorse.param.s')
        self.connect('blade.outer_shape_bem.twist', 'rotorse.param.twist_original')
        self.connect('blade.outer_shape_bem.chord', 'rotorse.param.chord_original')
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
        self.connect('rotorse.rs.mass.mass_one_blade',  'drivese.blade_mass')
        self.connect('rotorse.param.chord_param',  'drivese.blade_root_diameter', src_indices=[0])
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

        # Connections to turbine capital cost
        self.connect('control.rated_power',         'tcc.machine_rating')
        self.connect('rotorse.rs.mass.mass_one_blade','tcc.blade_mass')
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
        self.connect('rotorse.ra.AEP', 'outputs_2_screen.AEP')

class WindPark(Group):
    # Openmdao group to run the cost analysis of a wind park
    
    def initialize(self):
        self.options.declare('wt_init_options')
        self.options.declare('opt_options')
        
    def setup(self):
        wt_init_options = self.options['wt_init_options']
        opt_options     = self.options['opt_options']

        self.add_subsystem('wt',        WT_RNTA(wt_init_options = wt_init_options, opt_options = opt_options), promotes=['*'])
        self.add_subsystem('financese', PlantFinance(verbosity=True))
        
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

# Class to print outputs on screen
class Outputs_2_Screen(ExplicitComponent):
    def setup(self):
        
        self.add_input('AEP', val=0.0, units = 'GW * h')
        
    def compute(self, inputs, outputs):
        print('########################################')
        print('Objectives')
        print('AEP:         {:8.10f} GWh'.format(inputs['AEP'][0]))
        print('########################################')

if __name__ == "__main__":

    ## File management
    fname_input    = "reference_turbines/nrel5mw/nrel5mw_mod_update.yaml"
    fname_output   = "reference_turbines/nrel5mw/nrel5mw_mod_update_output.yaml"
    folder_output  = 'it_1/'
    opt_flag       = False
    # Load yaml data into a pure python data structure
    wt_initial               = WindTurbineOntologyPython()
    wt_initial.validate      = False
    wt_initial.fname_schema  = "reference_turbines/IEAontology_schema.yaml"
    wt_init_options, wt_init = wt_initial.initialize(fname_input)
    
    # Optimization options
    optimization_data       = Opt_Data()
    optimization_data.folder_output = folder_output
    if opt_flag == True:
        optimization_data.n_opt_twist = 8
        optimization_data.n_opt_chord = wt_initial.n_span
    else:
        optimization_data.n_opt_twist = wt_initial.n_span
        optimization_data.n_opt_chord = wt_initial.n_span
    opt_options             = optimization_data.initialize()
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
        wt_opt.model.add_objective('rotorse.ra.AEP', scaler = -1.e-6)
        
        # Set optimization variables
        indices_no_root         = range(2,opt_options['blade_aero']['n_opt_twist'])
        wt_opt.model.add_design_var('rotorse.opt_var.twist_opt_gain', indices = indices_no_root, lower=0., upper=1.)    
        wt_opt.model.add_design_var('rotorse.opt_var.chord_opt_gain', indices = indices_no_root, lower=0.5, upper=1.5)    
        
        # Set recorder
        wt_opt.driver.add_recorder(SqliteRecorder(opt_options['optimization_log']))
        wt_opt.driver.recording_options['includes'] = ['rotorse.ra.AEP']
        wt_opt.driver.recording_options['record_objectives']  = True
        wt_opt.driver.recording_options['record_constraints'] = True
        wt_opt.driver.recording_options['record_desvars']     = True
    
    # Setup openmdao problem
    wt_opt.setup()
    
    # Load initial wind turbine data from wt_initial to the openmdao problem
    wt_opt = yaml2openmdao(wt_opt, wt_init_options, wt_init)
    wt_opt['rotorse.param.s_opt_twist'] = np.linspace(0., 1., optimization_data.n_opt_twist)
    wt_opt['rotorse.param.s_opt_chord'] = np.linspace(0., 1., optimization_data.n_opt_chord)

    # Build and run openmdao problem
    wt_opt.run_driver()

    # Save data coming from openmdao to an output yaml file
    wt_initial.write_ontology(wt_opt, fname_output)

    # Printing and plotting results
    print('AEP = ' + str(wt_opt['rotorse.ra.AEP']*1.e-6) + ' GWh')