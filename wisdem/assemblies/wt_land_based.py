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
from wisdem.aeroelasticse.openmdao_openfast import FASTLoadCases
from wisdem.assemblies.parametrize_wt import WT_Parametrize
from wisdem.rotorse.dac import RunXFOIL
from wisdem.rotorse.rotor_aeropower import RotorAeroPower
from wisdem.rotorse.rotor_elasticity import RotorElasticity
from wisdem.rotorse.rotor_loads_defl_strains import RotorLoadsDeflStrains


class Opt_Data(object):
    # Pure python class to set the optimization parameters:

    def __init__(self):
        
        self.opt_options = {}

        # Save data
        self.folder_output    = 'it_0/'
        self.optimization_log = 'log_opt.sql'
        self.costs_verbosity  = False
        self.openfast         = False

        # Blade aerodynamic optimization parameters
        self.n_opt_twist   = 8
        self.n_opt_chord   = 8
        self.n_opt_spar_ss = 8
        self.n_opt_spar_ps = 8

    def initialize(self):

        self.opt_options['folder_output']    = self.folder_output
        self.opt_options['optimization_log'] = self.folder_output + self.optimization_log
        self.opt_options['costs_verbosity']  = self.costs_verbosity
        self.opt_options['openfast']         = self.openfast

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
        self.add_subsystem('param',     WT_Parametrize(wt_init_options = wt_init_options, opt_options = opt_options))
        self.add_subsystem('elastic',   RotorElasticity(wt_init_options = wt_init_options, opt_options = opt_options))
        self.add_subsystem('xf',        RunXFOIL(wt_init_options = wt_init_options)) # Recompute polars with xfoil (for flaps)
        self.add_subsystem('ra',        RotorAeroPower(wt_init_options = wt_init_options)) # Aero analysis
        
        if opt_options['openfast'] == True:
            self.add_subsystem('aeroelastic',  FASTLoadCases(wt_init_options = wt_init_options))
        
        self.add_subsystem('rlds',      RotorLoadsDeflStrains(wt_init_options = wt_init_options, opt_options = opt_options))
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
        
        # Connections from input yaml to parametrization
        self.connect('blade.outer_shape_bem.s',        ['param.pa.s', 'param.ps.s', 'xf.s'])
        self.connect('blade.outer_shape_bem.twist', 'param.pa.twist_original')
        self.connect('blade.outer_shape_bem.chord', 'param.pa.chord_original')
        self.connect('blade.internal_structure_2d_fem.layer_name',      'param.ps.layer_name')
        self.connect('blade.internal_structure_2d_fem.layer_thickness', 'param.ps.layer_thickness_original')

        # Connections from blade aero parametrization to other modules
        self.connect('param.pa.twist_param',           ['ra.theta','elastic.theta','rlds.theta'])
        self.connect('param.pa.twist_param',            'rlds.tip_pos.theta_tip',   src_indices=[-1])
        self.connect('param.pa.chord_param',           ['xf.chord', 'elastic.chord', 'ra.chord','rlds.chord'])


        # Connections from blade struct parametrization to rotor elasticity
        self.connect('param.ps.layer_thickness_param', 'elastic.precomp.layer_thickness')

        # Connections to rotor elastic and frequency analysis
        self.connect('assembly.r_blade',                                'elastic.r')
        self.connect('blade.outer_shape_bem.pitch_axis',                'elastic.precomp.pitch_axis')
        self.connect('blade.interp_airfoils.coord_xy_interp',           'elastic.precomp.coord_xy_interp')
        self.connect('blade.internal_structure_2d_fem.layer_start_nd',  'elastic.precomp.layer_start_nd')
        self.connect('blade.internal_structure_2d_fem.layer_end_nd',    'elastic.precomp.layer_end_nd')
        self.connect('blade.internal_structure_2d_fem.layer_name',      'elastic.precomp.layer_name')
        self.connect('blade.internal_structure_2d_fem.layer_web',       'elastic.precomp.layer_web')
        self.connect('blade.internal_structure_2d_fem.layer_mat',       'elastic.precomp.layer_mat')
        self.connect('blade.internal_structure_2d_fem.definition_layer','elastic.precomp.definition_layer')
        self.connect('blade.internal_structure_2d_fem.web_start_nd',    'elastic.precomp.web_start_nd')
        self.connect('blade.internal_structure_2d_fem.web_end_nd',      'elastic.precomp.web_end_nd')
        self.connect('blade.internal_structure_2d_fem.web_name',        'elastic.precomp.web_name')
        self.connect('materials.name',  'elastic.precomp.mat_name')
        self.connect('materials.orth',  'elastic.precomp.orth')
        self.connect('materials.E',     'elastic.precomp.E')
        self.connect('materials.G',     'elastic.precomp.G')
        self.connect('materials.nu',    'elastic.precomp.nu')
        self.connect('materials.rho',   'elastic.precomp.rho')
        self.connect('materials.component_id',  'elastic.precomp.component_id')
        self.connect('materials.unit_cost',     'elastic.precomp.unit_cost')
        self.connect('materials.waste',         'elastic.precomp.waste')
        self.connect('materials.rho_fiber',     'elastic.precomp.rho_fiber')
        self.connect('materials.rho_area_dry',  'elastic.precomp.rho_area_dry')
        self.connect('materials.ply_t',         'elastic.precomp.ply_t')
        self.connect('materials.fvf',           'elastic.precomp.fvf')
        self.connect('materials.fwf',           'elastic.precomp.fwf')
        self.connect('materials.roll_mass',     'elastic.precomp.roll_mass')

        # Connections from blade struct parametrization to rotor load anlysis
        self.connect('param.ps.s_opt_spar_ss',   'rlds.constr.s_opt_spar_ss')
        self.connect('param.ps.s_opt_spar_ps',   'rlds.constr.s_opt_spar_ps')

        # Connection from ra to rs for the rated conditions
        # self.connect('ra.powercurve.rated_V',        'rlds.aero_rated.V_load')
        self.connect('ra.powercurve.rated_V',        'rlds.gust.V_hub')
        self.connect('rlds.gust.V_gust',              ['rlds.aero_gust.V_load'])
        self.connect('ra.powercurve.rated_Omega',   ['rlds.Omega_load', 'rlds.aeroloads_Omega', 'elastic.curvefem.Omega', 'rlds.constr.rated_Omega'])
        self.connect('ra.powercurve.rated_pitch',   ['rlds.pitch_load', 'rlds.aeroloads_pitch'])
        

        
        # Connections to run xfoil for te flaps
        self.connect('blade.interp_airfoils.coord_xy_interp', 'xf.coord_xy_interp')
        self.connect('airfoils.aoa',                          'xf.aoa')
        self.connect('assembly.r_blade',                      'xf.r')
        self.connect('blade.dac_te_flaps.span_start',         'xf.span_start')
        self.connect('blade.dac_te_flaps.span_end',           'xf.span_end')
        self.connect('blade.dac_te_flaps.chord_start',        'xf.chord_start')
        self.connect('blade.dac_te_flaps.delta_max_pos',      'xf.delta_max_pos')
        self.connect('blade.dac_te_flaps.delta_max_neg',      'xf.delta_max_neg')
        self.connect('env.speed_sound_air',                   'xf.speed_sound_air')
        self.connect('env.rho_air',                           'xf.rho_air')
        self.connect('env.mu_air',                            'xf.mu_air')
        self.connect('control.rated_TSR',                     'xf.rated_TSR')
        self.connect('control.max_TS',                        'xf.max_TS')
        self.connect('blade.interp_airfoils.cl_interp',       'xf.cl_interp')
        self.connect('blade.interp_airfoils.cd_interp',       'xf.cd_interp')
        self.connect('blade.interp_airfoils.cm_interp',       'xf.cm_interp')

        # Connections to rotor aeropower
        self.connect('wt_class.V_mean',         'ra.cdf.xbar')
        self.connect('control.V_in' ,           'ra.control_Vin')
        self.connect('control.V_out' ,          'ra.control_Vout')
        self.connect('control.rated_power' ,    'ra.control_ratedPower')
        self.connect('control.min_Omega' ,      'ra.control_minOmega')
        self.connect('control.max_Omega' ,      'ra.control_maxOmega')
        self.connect('control.max_TS' ,         'ra.control_maxTS')
        self.connect('control.rated_TSR' ,      'ra.control_tsr')
        self.connect('control.rated_pitch' ,        'ra.control_pitch')
        self.connect('configuration.gearbox_type' , 'ra.drivetrainType')
        self.connect('assembly.r_blade',            'ra.r')
        self.connect('assembly.rotor_radius',       'ra.Rtip')
        self.connect('hub.radius',                  'ra.Rhub')
        self.connect('assembly.hub_height',         'ra.hub_height')
        self.connect('hub.cone',                    'ra.precone')
        self.connect('nacelle.uptilt',              'ra.tilt')
        self.connect('airfoils.aoa',                    'ra.airfoils_aoa')
        self.connect('airfoils.Re',                     'ra.airfoils_Re')
        self.connect('blade.interp_airfoils.cl_interp', 'ra.airfoils_cl')
        self.connect('blade.interp_airfoils.cd_interp', 'ra.airfoils_cd')
        self.connect('blade.interp_airfoils.cm_interp', 'ra.airfoils_cm')
        self.connect('configuration.n_blades',          'ra.nBlades')
        self.connect('blade.outer_shape_bem.s',         'ra.stall_check.s')
        self.connect('env.rho_air',                     'ra.rho')
        self.connect('env.mu_air',                      'ra.mu')
        self.connect('env.weibull_k',                   'ra.cdf.k')
        
        

        # Connections to rotor load analysis
        self.connect('elastic.EA',   'rlds.EA')
        self.connect('elastic.EIxx', 'rlds.EIxx')
        self.connect('elastic.EIyy', 'rlds.EIyy')
        self.connect('elastic.GJ',   'rlds.GJ')
        self.connect('elastic.rhoA', 'rlds.rhoA')
        self.connect('elastic.rhoJ', 'rlds.rhoJ')
        self.connect('elastic.precomp.x_ec', 'rlds.x_ec')
        self.connect('elastic.precomp.y_ec', 'rlds.y_ec')
        self.connect('elastic.precomp.xu_strain_spar', 'rlds.xu_strain_spar')
        self.connect('elastic.precomp.xl_strain_spar', 'rlds.xl_strain_spar')
        self.connect('elastic.precomp.yu_strain_spar', 'rlds.yu_strain_spar')
        self.connect('elastic.precomp.yl_strain_spar', 'rlds.yl_strain_spar')
        self.connect('elastic.precomp.xu_strain_te',   'rlds.xu_strain_te')
        self.connect('elastic.precomp.xl_strain_te',   'rlds.xl_strain_te')
        self.connect('elastic.precomp.yu_strain_te',   'rlds.yu_strain_te')
        self.connect('elastic.precomp.yl_strain_te',   'rlds.yl_strain_te')
        self.connect('blade.outer_shape_bem.s','rlds.constr.s')

        # Frequencies from curvefem to constraint
        self.connect('elastic.curvefem.freq',   'rlds.constr.freq') 

        self.connect('assembly.r_blade',                'rlds.r')
        self.connect('assembly.rotor_radius',           'rlds.Rtip')
        self.connect('hub.radius',                      'rlds.Rhub')
        self.connect('assembly.hub_height',             'rlds.hub_height')
        self.connect('hub.cone',                        'rlds.precone')
        self.connect('nacelle.uptilt',                  'rlds.tilt')
        self.connect('airfoils.aoa',                    'rlds.airfoils_aoa')
        self.connect('airfoils.Re',                     'rlds.airfoils_Re')
        self.connect('blade.interp_airfoils.cl_interp', 'rlds.airfoils_cl')
        self.connect('blade.interp_airfoils.cd_interp', 'rlds.airfoils_cd')
        self.connect('blade.interp_airfoils.cm_interp', 'rlds.airfoils_cm')
        self.connect('configuration.n_blades',          'rlds.nBlades')
        self.connect('env.rho_air',                     'rlds.rho')
        self.connect('env.mu_air',                      'rlds.mu')
        # Connections to rotorse-rs-gustetm
        self.connect('wt_class.V_mean',                 'rlds.gust.V_mean')
        self.connect('configuration.turb_class',        'rlds.gust.turbulence_class')
        # self.connect('wt_class.V_extreme1',             'rlds.aero_storm_1yr.V_load')
        # self.connect('wt_class.V_extreme50',            'rlds.aero_storm_50yr.V_load')
        
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

        # Connections to DriveSE
        self.connect('assembly.rotor_diameter',    'drivese.rotor_diameter')     
        self.connect('control.rated_power',        'drivese.machine_rating')    
        self.connect('nacelle.overhang',           'drivese.overhang') 
        self.connect('nacelle.uptilt',             'drivese.shaft_angle')
        self.connect('configuration.n_blades',     'drivese.number_of_blades') 
        self.connect('ra.powercurve.rated_Q',         'drivese.rotor_torque')
        self.connect('ra.powercurve.rated_Omega',     'drivese.rotor_rpm')
        # self.connect('rlds.Fxyz_total',      'drivese.Fxyz')
        # self.connect('rlds.Mxyz_total',      'drivese.Mxyz')
        # self.connect('rlds.I_all_blades',    'drivese.blades_I')
        self.connect('rlds.blade_mass',            'drivese.blade_mass')
        self.connect('param.pa.chord_param',       'drivese.blade_root_diameter', src_indices=[0])
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
        
        # Connections to aeroelasticse
        # promotes=['fst_vt_in'])
        if opt_options['openfast'] == True:
            self.connect('blade.outer_shape_bem.ref_axis',  'aeroelastic.ref_axis_blade')
            self.connect('assembly.r_blade',                'aeroelastic.r')
            self.connect('blade.outer_shape_bem.pitch_axis','aeroelastic.le_location')
            self.connect('param.pa.chord_param',            'aeroelastic.chord')
            self.connect('param.pa.twist_param',            'aeroelastic.theta')
            self.connect('blade.interp_airfoils.coord_xy_interp', 'aeroelastic.coord_xy_interp')
            self.connect('env.rho_air',                     'aeroelastic.rho')
            self.connect('env.mu_air',                      'aeroelastic.mu')                    
            self.connect('env.shear_exp',                   'aeroelastic.shearExp')                    
            self.connect('assembly.rotor_radius',           'aeroelastic.Rtip')
            self.connect('hub.radius',                      'aeroelastic.Rhub')
            self.connect('assembly.hub_height',             'aeroelastic.hub_height')
            # self.connect('hub.cone',                        'aeroelastic.precone')
            # self.connect('nacelle.uptilt',                  'aeroelastic.tilt')
            self.connect('airfoils.aoa',                    'aeroelastic.airfoils_aoa')
            self.connect('airfoils.Re',                     'aeroelastic.airfoils_Re')
            self.connect('blade.interp_airfoils.cl_interp', 'aeroelastic.airfoils_cl')
            self.connect('blade.interp_airfoils.cd_interp', 'aeroelastic.airfoils_cd')
            self.connect('blade.interp_airfoils.cm_interp', 'aeroelastic.airfoils_cm')
            self.connect('blade.interp_airfoils.r_thick_interp', 'aeroelastic.rthick')
            self.connect('elastic.rhoA',                'aeroelastic.beam:rhoA')
            self.connect('elastic.EIxx',                'aeroelastic.beam:EIxx')
            self.connect('elastic.EIyy',                'aeroelastic.beam:EIyy')
            self.connect('elastic.Tw_iner',             'aeroelastic.beam:Tw_iner')
            self.connect('elastic.curvefem.modes_coef', 'aeroelastic.modes_coef_curvefem')
            self.connect('ra.powercurve.V',      'aeroelastic.U_init')
            self.connect('ra.powercurve.Omega',  'aeroelastic.Omega_init')
            self.connect('ra.powercurve.pitch',  'aeroelastic.pitch_init')
            self.connect('ra.powercurve.V_R25',  'aeroelastic.V_R25')
            self.connect('ra.powercurve.rated_V','aeroelastic.Vrated')
            self.connect('rlds.gust.V_gust',       'aeroelastic.Vgust')
            self.connect('wt_class.V_mean',              'aeroelastic.V_mean_iec')
            self.connect('control.rated_power',          'aeroelastic.control_ratedPower')
            self.connect('control.max_TS',               'aeroelastic.control_maxTS')
            self.connect('control.max_Omega',            'aeroelastic.control_maxOmega')
            self.connect('configuration.turb_class',     'aeroelastic.turbulence_class')
            self.connect('configuration.ws_class' ,      'aeroelastic.turbine_class')
            self.connect('ra.aeroperf_tables.pitch_vector', 'aeroelastic.pitch_vector')
            self.connect('ra.aeroperf_tables.tsr_vector', 'aeroelastic.tsr_vector')
            self.connect('ra.aeroperf_tables.U_vector', 'aeroelastic.U_vector')
            self.connect('ra.aeroperf_tables.Cp', 'aeroelastic.Cp_aero_table')
            self.connect('ra.aeroperf_tables.Ct', 'aeroelastic.Ct_aero_table')
            self.connect('ra.aeroperf_tables.Cq', 'aeroelastic.Cq_aero_table')

            # Temporary
            self.connect('xf.Re_loc',           'aeroelastic.airfoils_Re_loc')
            self.connect('xf.Ma_loc',           'aeroelastic.airfoils_Ma_loc')
        
        # Connections to turbine constraints
        self.connect('rlds.tip_pos.tip_deflection',     'tcons.tip_deflection')
        self.connect('assembly.rotor_radius',           'tcons.Rtip')
        self.connect('blade.outer_shape_bem.ref_axis',  'tcons.ref_axis_blade')
        self.connect('hub.cone',                        'tcons.precone')
        self.connect('nacelle.uptilt',                  'tcons.tilt')
        self.connect('nacelle.overhang',                'tcons.overhang')
        self.connect('tower.ref_axis',                  'tcons.ref_axis_tower')
        self.connect('tower.diameter',                  'tcons.d_full')

        # Connections to turbine capital cost
        self.connect('control.rated_power',         'tcc.machine_rating')
        self.connect('rlds.blade_mass',             'tcc.blade_mass')
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
        self.connect('ra.AEP',          'outputs_2_screen.AEP')
        self.connect('rlds.blade_mass',   'outputs_2_screen.blade_mass')
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
        
        # Inputs to plantfinancese from wt group
        self.connect('ra.AEP',          'financese.turbine_aep')
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
    fname_input    = "wisdem/assemblies/reference_turbines/nrel5mw/nrel5mw_mod_update.yaml"
    fname_output   = "wisdem/assemblies/reference_turbines/nrel5mw/nrel5mw_mod_update_output.yaml"
    # fname_input    = "wisdem/wisdem/assemblies/reference_turbines/bar/BAR2010n.yaml"
    # fname_output   = "wisdem/wisdem/assemblies/reference_turbines/bar/BAR2011n.yaml"
    folder_output  = 'it_1/'
    opt_flag_twist = False
    opt_flag_chord = False
    opt_flag_spar_ss = False
    opt_flag_spar_ps = False
    merit_figure     = 'Blade Mass' # 'AEP' - 'LCOE'
    # Optimization options
    optimization_data       = Opt_Data()
    optimization_data.folder_output = folder_output
    optimization_data.openfast = True
    
    
    
    # Load yaml data into a pure python data structure
    wt_initial                  = WindTurbineOntologyPython()
    wt_initial.validate         = False
    wt_initial.fname_schema     = "wisdem/wisdem/assemblies/reference_turbines/IEAontology_schema.yaml"
    if optimization_data.openfast == True:
        wt_initial.xfoil_path       = '/Users/pbortolo/work/1_wisdem/Xfoil/bin/xfoil'
        wt_initial.Analysis_Level   = 1
        wt_initial.FAST_ver         = 'OpenFAST'
        wt_initial.dev_branch       = True
        wt_initial.FAST_exe         = '/Users/pbortolo/work/2_openfast/openfast/build/glue-codes/openfast/openfast'
        wt_initial.FAST_directory   = '/Users/pbortolo/work/2_openfast/BAR/OpenFAST_Models/RotorSE_FAST_BAR_2010n_noRe_0_70_to_0_95'
        wt_initial.FAST_InputFile   = 'RotorSE_FAST_BAR_2010n_noRe.fst'
        wt_initial.path2dll         = '/Users/pbortolo/work/2_openfast/ROSCO_w_flaps/build/libdiscon.dylib'
        wt_initial.Turbsim_exe      = "/Users/pbortolo/work/2_openfast/TurbSim/bin/TurbSim_glin64"
        wt_initial.FAST_namingOut   = 'WISDEM_NREL5MW'
        wt_initial.FAST_runDirectory= 'temp/' + wt_initial.FAST_namingOut
        wt_initial.cores            = 1
        wt_initial.debug_level      = 2
        wt_initial.n_pitch          = 20
        wt_initial.n_tsr            = 20
    wt_init_options, wt_init    = wt_initial.initialize(fname_input)
    
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
    else:
        opt_flag = False

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
            wt_opt.model.add_objective('ra.AEP', scaler = -1.e-6)
        elif merit_figure == 'Blade Mass':
            wt_opt.model.add_objective('rlds.blade_mass', scaler = 1.e-4)
        elif merit_figure == 'LCOE':
            wt_opt.model.add_objective('financese.lcoe', scaler = 1.e+2)
        else:
            exit('The merit figure ' + merit_figure + ' is not supported.')
        
        # Set optimization variables
        if opt_flag_twist == True:
            indices        = range(2,opt_options['blade_aero']['n_opt_twist'])
            wt_opt.model.add_design_var('param.opt_var.twist_opt_gain', indices = indices, lower=0., upper=1.)
        if opt_flag_chord == True:
            indices  = range(2,opt_options['blade_aero']['n_opt_chord'] - 1)
            wt_opt.model.add_design_var('param.opt_var.chord_opt_gain', indices = indices, lower=0.5, upper=1.5)
        if opt_flag_spar_ss == True:
            indices  = range(2,opt_options['blade_struct']['n_opt_spar_ss'] - 1)
            wt_opt.model.add_design_var('param.opt_var.spar_ss_opt_gain', indices = indices, lower=0.5, upper=1.5)
        if opt_flag_spar_ps == True:
            indices  = range(2,opt_options['blade_struct']['n_opt_spar_ps'] - 1)
            wt_opt.model.add_design_var('param.opt_var.spar_ps_opt_gain', indices = indices, lower=0.5, upper=1.5)

        # Set non-linear constraints
        wt_opt.model.add_constraint('rlds.pbeam.strainU_spar', upper= 1.) 
        wt_opt.model.add_constraint('rlds.pbeam.strainL_spar', upper= 1.) 
        wt_opt.model.add_constraint('tcons.tip_deflection_ratio',    upper= 1.0) 
        
        # Set recorder
        wt_opt.driver.add_recorder(SqliteRecorder(opt_options['optimization_log']))
        wt_opt.driver.recording_options['includes'] = ['ra.AEP, rlds.blade_mass, financese.lcoe']
        wt_opt.driver.recording_options['record_objectives']  = True
        wt_opt.driver.recording_options['record_constraints'] = True
        wt_opt.driver.recording_options['record_desvars']     = True
    
    # Setup openmdao problem
    wt_opt.setup()
    
    # Load initial wind turbine data from wt_initial to the openmdao problem
    wt_opt = yaml2openmdao(wt_opt, wt_init_options, wt_init)
    wt_opt['param.pa.s_opt_twist']   = np.linspace(0., 1., optimization_data.n_opt_twist)
    wt_opt['param.pa.s_opt_chord']   = np.linspace(0., 1., optimization_data.n_opt_chord)
    wt_opt['param.ps.s_opt_spar_ss'] = np.linspace(0., 1., optimization_data.n_opt_spar_ss)
    wt_opt['param.ps.s_opt_spar_ps'] = np.linspace(0., 1., optimization_data.n_opt_spar_ps)
    wt_opt['rlds.constr.min_strainU_spar'] = -0.003
    wt_opt['rlds.constr.max_strainU_spar'] =  0.003
    wt_opt['rlds.constr.min_strainL_spar'] = -0.003
    wt_opt['rlds.constr.max_strainL_spar'] =  0.003

    # Build and run openmdao problem
    wt_opt.run_driver()

    # Save data coming from openmdao to an output yaml file
    wt_initial.write_ontology(wt_opt, fname_output)

    # Printing and plotting results
    print('AEP in GWh = ' + str(wt_opt['ra.AEP']*1.e-6))
    print('Nat frequencies blades in Hz = ' + str(wt_opt['elastic.curvefem.freq']))
    print('Tip tower clearance in m     = ' + str(wt_opt['tcons.blade_tip_tower_clearance']))
    print('Tip deflection constraint    = ' + str(wt_opt['tcons.tip_deflection_ratio']))

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(wt_opt['assembly.r_blade'], wt_opt['rlds.pbeam.strainU_spar'], label='spar ss')
    plt.plot(wt_opt['assembly.r_blade'], wt_opt['rlds.pbeam.strainL_spar'], label='spar ps')
    plt.plot(wt_opt['assembly.r_blade'], wt_opt['rlds.pbeam.strainU_te'], label='te ss')
    plt.plot(wt_opt['assembly.r_blade'], wt_opt['rlds.pbeam.strainL_te'], label='te ps')
    plt.ylim([-5e-3, 5e-3])
    plt.xlabel('r [m]')
    plt.ylabel('strain [-]')
    plt.legend()
    plt.show()