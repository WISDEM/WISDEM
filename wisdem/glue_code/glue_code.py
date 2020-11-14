import numpy as np
import openmdao.api as om
from wisdem.glue_code.gc_WT_DataStruc import WindTurbineOntologyOpenMDAO
from wisdem.ccblade.ccblade_component import CCBladeTwist
from wisdem.commonse.turbine_class import TurbineClass
from wisdem.drivetrainse.drivetrain import DrivetrainSE
from wisdem.towerse.tower import TowerSE
from wisdem.nrelcsm.nrel_csm_cost_2015 import Turbine_CostsSE_2015
from wisdem.orbit.api.wisdem.fixed import Orbit
from wisdem.landbosse.landbosse_omdao.landbosse import LandBOSSE
from wisdem.plant_financese.plant_finance import PlantFinance
from wisdem.commonse.turbine_constraints  import TurbineConstraints
from wisdem.rotorse.servose import ServoSE, NoStallConstraint
from wisdem.rotorse.rotor_elasticity import RotorElasticity
from wisdem.rotorse.rotor_loads_defl_strains import RotorLoadsDeflStrains
from wisdem.glue_code.gc_RunTools import Outputs_2_Screen, Convergence_Trends_Opt

class WT_RNTA(om.Group):
    # Openmdao group to run the analysis of the wind turbine
    
    def initialize(self):
        self.options.declare('modeling_options')
        self.options.declare('opt_options')
        
    def setup(self):
        modeling_options = self.options['modeling_options']
        opt_options      = self.options['opt_options']

        if modeling_options['flags']['blade'] and modeling_options['flags']['nacelle']:
            self.linear_solver = lbgs = om.LinearBlockGS()
            self.nonlinear_solver = nlbgs = om.NonlinearBlockGS()
            nlbgs.options['maxiter'] = 2
            nlbgs.options['atol'] = nlbgs.options['atol'] = 1e-2
        
        # Analysis components
        self.add_subsystem('wt_init',   WindTurbineOntologyOpenMDAO(modeling_options = modeling_options, opt_options = opt_options), promotes=['*'])
        if modeling_options['flags']['blade']:
            self.add_subsystem('ccblade',   CCBladeTwist(modeling_options = modeling_options, opt_options = opt_options)) # Run standalong CCBlade and possibly determine optimal twist from user-defined margin to stall
            self.add_subsystem('wt_class',  TurbineClass())
            self.add_subsystem('elastic',   RotorElasticity(modeling_options = modeling_options, opt_options = opt_options))
            self.add_subsystem('sse',       ServoSE(modeling_options = modeling_options)) # Aero analysis
            self.add_subsystem('stall_check', NoStallConstraint(modeling_options = modeling_options))
            self.add_subsystem('rlds',      RotorLoadsDeflStrains(modeling_options = modeling_options, opt_options = opt_options, freq_run=False))
        if modeling_options['flags']['nacelle']:
            self.add_subsystem('drivese',   DrivetrainSE(modeling_options=modeling_options, n_dlcs=1))
        if modeling_options['flags']['tower']:
            self.add_subsystem('towerse',   TowerSE(modeling_options=modeling_options))
        if modeling_options['flags']['blade'] and modeling_options['flags']['tower']:
            self.add_subsystem('tcons',     TurbineConstraints(modeling_options = modeling_options))
        self.add_subsystem('tcc',       Turbine_CostsSE_2015(verbosity=modeling_options['general']['verbosity']))

        if modeling_options['flags']['blade']:
            n_span  = modeling_options['blade']['n_span']
            
            # Conncetions to ccblade
            self.connect('blade.pa.chord_param',            'ccblade.chord')
            self.connect('blade.pa.twist_param',            'ccblade.twist')
            self.connect('blade.opt_var.s_opt_chord',       'ccblade.s_opt_chord')
            self.connect('blade.opt_var.s_opt_twist',       'ccblade.s_opt_twist')
            self.connect('assembly.r_blade',                'ccblade.r')
            self.connect('assembly.rotor_radius',           'ccblade.Rtip')
            self.connect('hub.radius',                      'ccblade.Rhub')
            self.connect('blade.interp_airfoils.r_thick_interp', 'ccblade.rthick')
            self.connect('airfoils.aoa',                    'ccblade.airfoils_aoa')
            self.connect('airfoils.Re',                     'ccblade.airfoils_Re')
            self.connect('blade.interp_airfoils.cl_interp', 'ccblade.airfoils_cl')
            self.connect('blade.interp_airfoils.cd_interp', 'ccblade.airfoils_cd')
            self.connect('blade.interp_airfoils.cm_interp', 'ccblade.airfoils_cm')
            self.connect('assembly.hub_height',             'ccblade.hub_height')
            self.connect('hub.cone',                        'ccblade.precone')
            self.connect('nacelle.uptilt',                  'ccblade.tilt')
            self.connect('blade.outer_shape_bem.ref_axis',  'ccblade.precurve', src_indices=[(i, 0) for i in np.arange(n_span)])
            self.connect('blade.outer_shape_bem.ref_axis',  'ccblade.precurveTip', src_indices=[(-1, 0)])
            self.connect('blade.outer_shape_bem.ref_axis',  'ccblade.presweep', src_indices=[(i, 1) for i in np.arange(n_span)])
            self.connect('blade.outer_shape_bem.ref_axis',  'ccblade.presweepTip', src_indices=[(-1, 1)])
            self.connect('configuration.n_blades',          'ccblade.nBlades')
            if modeling_options['flags']['control']:
                self.connect('control.rated_pitch' ,            'ccblade.pitch')
            self.connect('pc.tsr_opt',                      'ccblade.tsr')
            self.connect('env.rho_air',                     'ccblade.rho')
            self.connect('env.mu_air',                      'ccblade.mu')
            self.connect('env.shear_exp',                   'ccblade.shearExp') 

            # Connections to wind turbine class
            self.connect('configuration.ws_class' , 'wt_class.turbine_class')

            # Connections from blade aero parametrization to other modules
            self.connect('blade.pa.twist_param',           ['elastic.theta','rlds.theta'])
            #self.connect('blade.pa.twist_param',            'rlds.tip_pos.theta_tip',   src_indices=[-1])
            self.connect('blade.pa.chord_param',            'elastic.chord')
            self.connect('blade.pa.chord_param',           ['rlds.chord'])
            if modeling_options['flags']['blade']:
                self.connect('blade.pa.twist_param',           'sse.theta')
                self.connect('blade.pa.chord_param',           'sse.chord')


            # Connections from blade struct parametrization to rotor elasticity
            self.connect('blade.ps.layer_thickness_param', 'elastic.precomp.layer_thickness')

            # Connections to rotor elastic and frequency analysis
            self.connect('nacelle.uptilt',                                  'elastic.precomp.uptilt')
            self.connect('configuration.n_blades',                          'elastic.precomp.n_blades')
            self.connect('assembly.r_blade',                                'elastic.r')
            self.connect('blade.outer_shape_bem.pitch_axis',                'elastic.precomp.pitch_axis')
            self.connect('blade.interp_airfoils.coord_xy_interp',           'elastic.precomp.coord_xy_interp')
            self.connect('blade.internal_structure_2d_fem.layer_start_nd',  'elastic.precomp.layer_start_nd')
            self.connect('blade.internal_structure_2d_fem.layer_end_nd',    'elastic.precomp.layer_end_nd')
            self.connect('blade.internal_structure_2d_fem.layer_web',       'elastic.precomp.layer_web')
            self.connect('blade.internal_structure_2d_fem.definition_layer','elastic.precomp.definition_layer')
            self.connect('blade.internal_structure_2d_fem.web_start_nd',    'elastic.precomp.web_start_nd')
            self.connect('blade.internal_structure_2d_fem.web_end_nd',      'elastic.precomp.web_end_nd')
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

            # Conncetions to rail transport module
            if opt_options['constraints']['blade']['rail_transport']['flag']:
                self.connect('blade.outer_shape_bem.pitch_axis',        'elastic.rail.pitch_axis')
                self.connect('blade.outer_shape_bem.ref_axis',          'elastic.rail.blade_ref_axis')
                self.connect('blade.interp_airfoils.coord_xy_dim',      'elastic.rail.coord_xy_dim')
                self.connect('blade.interp_airfoils.coord_xy_interp',   'elastic.rail.coord_xy_interp')

            # Connections from blade struct parametrization to rotor load anlysis
            self.connect('blade.ps.s_opt_spar_cap_ss',   'rlds.constr.s_opt_spar_cap_ss')
            self.connect('blade.ps.s_opt_spar_cap_ps',   'rlds.constr.s_opt_spar_cap_ps')

            # Connection from ra to rs for the rated conditions
            # self.connect('sse.powercurve.rated_V',        'rlds.aero_rated.V_load')
            self.connect('sse.powercurve.rated_V',        'sse.gust.V_hub')
            self.connect('sse.gust.V_gust',              ['rlds.aero_gust.V_load', 'rlds.aero_hub_loads.V_load'])
            self.connect('env.shear_exp',                ['sse.powercurve.shearExp', 'rlds.aero_gust.shearExp']) 
            self.connect('sse.powercurve.rated_Omega',   ['rlds.Omega_load', 'rlds.tot_loads_gust.aeroloads_Omega', 'rlds.constr.rated_Omega'])
            self.connect('sse.powercurve.rated_pitch',   ['rlds.pitch_load', 'rlds.tot_loads_gust.aeroloads_pitch'])

            # Connections to ServoSE
            self.connect('control.V_in' ,                  'sse.v_min')
            self.connect('control.V_out' ,                 'sse.v_max')
            self.connect('control.rated_power' ,           'sse.rated_power')
            self.connect('control.minOmega' ,              'sse.omega_min')
            self.connect('control.maxOmega' ,              'sse.omega_max')
            self.connect('control.max_TS' ,                'sse.control_maxTS')
            self.connect('pc.tsr_opt' ,                    'sse.tsr_operational')
            self.connect('control.rated_pitch' ,           'sse.control_pitch')
            self.connect('configuration.gearbox_type' ,    'sse.drivetrainType')
            self.connect('nacelle.gearbox_efficiency',     'sse.powercurve.gearbox_efficiency')
            if modeling_options['flags']['nacelle']:
                self.connect('drivese.lss_rpm',              'sse.powercurve.lss_rpm')
                self.connect('drivese.generator_efficiency', 'sse.powercurve.generator_efficiency')
            self.connect('assembly.r_blade',               'sse.r')
            # self.connect('blade.pa.chord_param',           'sse.chord')
            # self.connect('blade.pa.twist_param',           'sse.theta')
            self.connect('hub.radius',                     'sse.Rhub')
            self.connect('assembly.rotor_radius',          'sse.Rtip')
            self.connect('assembly.hub_height',            'sse.hub_height')
            self.connect('hub.cone',                       'sse.precone')
            self.connect('nacelle.uptilt',                 'sse.tilt')
            self.connect('blade.outer_shape_bem.ref_axis', 'sse.precurve', src_indices=[(i, 0) for i in np.arange(n_span)])
            self.connect('blade.outer_shape_bem.ref_axis', 'sse.precurveTip', src_indices=[(-1, 0)])
            self.connect('blade.outer_shape_bem.ref_axis', 'sse.presweep', src_indices=[(i, 1) for i in np.arange(n_span)])
            self.connect('blade.outer_shape_bem.ref_axis', 'sse.presweepTip', src_indices=[(-1, 1)])
            self.connect('airfoils.aoa',                   'sse.airfoils_aoa')
            self.connect('airfoils.Re',                    'sse.airfoils_Re')
            self.connect('blade.interp_airfoils.cl_interp','sse.airfoils_cl')
            self.connect('blade.interp_airfoils.cd_interp','sse.airfoils_cd')
            self.connect('blade.interp_airfoils.cm_interp','sse.airfoils_cm')
            self.connect('configuration.n_blades',         'sse.nBlades')
            self.connect('env.rho_air',                    'sse.rho')
            self.connect('env.mu_air',                     'sse.mu')
            self.connect('wt_class.V_mean',                'sse.cdf.xbar')
            self.connect('env.weibull_k',                  'sse.cdf.k')
            # Connections to rotorse-rs-gustetm
            self.connect('wt_class.V_mean',                 'sse.gust.V_mean')
            self.connect('configuration.turb_class',        'sse.gust.turbulence_class')

            # Connections to the stall check
            self.connect('blade.outer_shape_bem.s',        'stall_check.s')
            self.connect('airfoils.aoa',                   'stall_check.airfoils_aoa')
            self.connect('blade.interp_airfoils.cl_interp','stall_check.airfoils_cl')
            self.connect('blade.interp_airfoils.cd_interp','stall_check.airfoils_cd')
            self.connect('blade.interp_airfoils.cm_interp','stall_check.airfoils_cm')
            if modeling_options['flags']['blade']:
                self.connect('sse.powercurve.aoa_regII',   'stall_check.aoa_along_span')
            else:
                self.connect('ccblade.alpha',  'stall_check.aoa_along_span')

            # Connections to rotor load analysis
            self.connect('blade.interp_airfoils.cl_interp','rlds.airfoils_cl')
            self.connect('blade.interp_airfoils.cd_interp','rlds.airfoils_cd')
            self.connect('blade.interp_airfoils.cm_interp','rlds.airfoils_cm')
            self.connect('airfoils.aoa',              'rlds.airfoils_aoa')
            self.connect('airfoils.Re',               'rlds.airfoils_Re')
            self.connect('assembly.rotor_radius',     'rlds.Rtip')
            self.connect('hub.radius',                'rlds.Rhub')
            self.connect('env.rho_air',               'rlds.rho')
            self.connect('env.mu_air',                'rlds.mu')
            self.connect('env.shear_exp',             'rlds.aero_hub_loads.shearExp')
            self.connect('assembly.hub_height',       'rlds.hub_height')
            self.connect('configuration.n_blades',    'rlds.nBlades')
            self.connect('assembly.r_blade',          'rlds.r')
            self.connect('hub.cone',                  'rlds.precone')
            self.connect('nacelle.uptilt',            'rlds.tilt')

            self.connect('elastic.A',    'rlds.A')
            self.connect('elastic.EA',   'rlds.EA')
            self.connect('elastic.EIxx', 'rlds.EIxx')
            self.connect('elastic.EIyy', 'rlds.EIyy')
            self.connect('elastic.EIxy', 'rlds.EIxy')
            self.connect('elastic.GJ',   'rlds.GJ')
            self.connect('elastic.rhoA', 'rlds.rhoA')
            self.connect('elastic.rhoJ', 'rlds.rhoJ')
            self.connect('elastic.x_ec', 'rlds.x_ec')
            self.connect('elastic.y_ec', 'rlds.y_ec')
            self.connect('elastic.precomp.xu_strain_spar', 'rlds.xu_strain_spar')
            self.connect('elastic.precomp.xl_strain_spar', 'rlds.xl_strain_spar')
            self.connect('elastic.precomp.yu_strain_spar', 'rlds.yu_strain_spar')
            self.connect('elastic.precomp.yl_strain_spar', 'rlds.yl_strain_spar')
            self.connect('elastic.precomp.xu_strain_te',   'rlds.xu_strain_te')
            self.connect('elastic.precomp.xl_strain_te',   'rlds.xl_strain_te')
            self.connect('elastic.precomp.yu_strain_te',   'rlds.yu_strain_te')
            self.connect('elastic.precomp.yl_strain_te',   'rlds.yl_strain_te')
            self.connect('blade.outer_shape_bem.s','rlds.constr.s')

            # Connections to rotorse-rc
            # self.connect('blade.length',                                    'rotorse.rc.blade_length')
            # self.connect('blade.outer_shape_bem.s',                         'rotorse.rc.s')
            # self.connect('blade.outer_shape_bem.pitch_axis',                'rotorse.rc.pitch_axis')
            # self.connect('blade.interp_airfoils.coord_xy_interp',           'rotorse.rc.coord_xy_interp')
            # self.connect('blade.internal_structure_2d_fem.layer_start_nd',  'rotorse.rc.layer_start_nd')
            # self.connect('blade.internal_structure_2d_fem.layer_end_nd',    'rotorse.rc.layer_end_nd')
            # self.connect('blade.internal_structure_2d_fem.layer_web',       'rotorse.rc.layer_web')
            # self.connect('blade.internal_structure_2d_fem.web_start_nd',    'rotorse.rc.web_start_nd')
            # self.connect('blade.internal_structure_2d_fem.web_end_nd',      'rotorse.rc.web_end_nd')
            # self.connect('materials.name',          'rotorse.rc.mat_name')
            # self.connect('materials.rho',           'rotorse.rc.rho')

        # Connections to DriveSE
        if modeling_options['flags']['nacelle']:
            self.connect('hub.diameter'                    , 'drivese.hub_diameter')
            self.connect('hub.hub_in2out_circ'             , 'drivese.hub_in2out_circ')
            self.connect('hub.flange_t2shell_t'            , 'drivese.flange_t2shell_t')
            self.connect('hub.flange_OD2hub_D'             , 'drivese.flange_OD2hub_D')
            self.connect('hub.flange_ID2flange_OD'         , 'drivese.flange_ID2flange_OD')
            self.connect('hub.hub_stress_concentration'    , 'drivese.hub_stress_concentration')
            self.connect('hub.n_front_brackets'            , 'drivese.n_front_brackets')
            self.connect('hub.n_rear_brackets'             , 'drivese.n_rear_brackets')
            self.connect('hub.clearance_hub_spinner'       , 'drivese.clearance_hub_spinner')
            self.connect('hub.spin_hole_incr'              , 'drivese.spin_hole_incr')
            self.connect('hub.pitch_system_scaling_factor' , 'drivese.pitch_system_scaling_factor')
            self.connect('hub.spinner_gust_ws'             , 'drivese.spinner_gust_ws')

            self.connect('configuration.n_blades',          'drivese.n_blades')
            
            self.connect('assembly.rotor_diameter',    'drivese.rotor_diameter')
            self.connect('configuration.upwind',       'drivese.upwind')
            self.connect('control.minOmega' ,          'drivese.minimum_rpm')
            self.connect('sse.powercurve.rated_Omega', 'drivese.rated_rpm')
            self.connect('sse.powercurve.rated_Q',     'drivese.rated_torque')
            self.connect('control.rated_power',        'drivese.machine_rating')    
            self.connect('tower.diameter',             'drivese.D_top', src_indices=[-1])
            
            self.connect('rlds.aero_hub_loads.Fxyz_hub_aero', 'drivese.F_hub')
            self.connect('rlds.aero_hub_loads.Mxyz_hub_aero', 'drivese.M_hub')
            self.connect('rlds.frame.root_M',                 'drivese.pitch_system.BRFM', src_indices=[1])
                
            self.connect('blade.pa.chord_param',              'drivese.blade_root_diameter', src_indices=[0])
            self.connect('elastic.precomp.blade_mass',        'drivese.blade_mass')
            self.connect('elastic.precomp.mass_all_blades',   'drivese.blades_mass')
            self.connect('elastic.precomp.I_all_blades',      'drivese.blades_I')

            self.connect('nacelle.distance_hub2mb',           'drivese.L_h1')
            self.connect('nacelle.distance_mb2mb',            'drivese.L_12')
            self.connect('nacelle.L_generator',               'drivese.L_generator')
            self.connect('nacelle.overhang',                  'drivese.overhang')
            self.connect('nacelle.distance_tt_hub',           'drivese.drive_height')
            self.connect('nacelle.uptilt',                    'drivese.tilt')
            self.connect('nacelle.gear_ratio',                'drivese.gear_ratio')
            self.connect('nacelle.mb1Type',                   'drivese.bear1.bearing_type')
            self.connect('nacelle.mb2Type',                   'drivese.bear2.bearing_type')
            self.connect('nacelle.lss_diameter',              'drivese.lss_diameter')
            self.connect('nacelle.lss_wall_thickness',        'drivese.lss_wall_thickness')
            if modeling_options['drivetrainse']['direct']:
                self.connect('nacelle.nose_diameter',              'drivese.bear1.D_shaft', src_indices=[0])
                self.connect('nacelle.nose_diameter',              'drivese.bear2.D_shaft', src_indices=[-1])
            else:
                self.connect('nacelle.lss_diameter',              'drivese.bear1.D_shaft', src_indices=[0])
                self.connect('nacelle.lss_diameter',              'drivese.bear2.D_shaft', src_indices=[-1])
            self.connect('nacelle.uptower',                   'drivese.uptower')
            self.connect('nacelle.brake_mass_user',           'drivese.brake_mass_user')
            self.connect('nacelle.hvac_mass_coeff',           'drivese.hvac_mass_coeff')
            self.connect('nacelle.converter_mass_user',       'drivese.converter_mass_user')
            self.connect('nacelle.transformer_mass_user',     'drivese.transformer_mass_user')

            if modeling_options['drivetrainse']['direct']:
                self.connect('nacelle.access_diameter',           'drivese.access_diameter') # only used in direct
                self.connect('nacelle.nose_diameter',             'drivese.nose_diameter') # only used in direct
                self.connect('nacelle.nose_wall_thickness',       'drivese.nose_wall_thickness') # only used in direct
                self.connect('nacelle.bedplate_wall_thickness',   'drivese.bedplate_wall_thickness') # only used in direct
            else:
                self.connect('nacelle.hss_length',                'drivese.L_hss') # only used in geared
                self.connect('nacelle.hss_diameter',              'drivese.hss_diameter') # only used in geared
                self.connect('nacelle.hss_wall_thickness',        'drivese.hss_wall_thickness') # only used in geared
                self.connect('nacelle.hss_material',              'drivese.hss_material')
                self.connect('nacelle.planet_numbers',            'drivese.planet_numbers') # only used in geared
                self.connect('nacelle.gear_configuration',        'drivese.gear_configuration') # only used in geared
                self.connect('nacelle.bedplate_flange_width',     'drivese.bedplate_flange_width') # only used in geared
                self.connect('nacelle.bedplate_flange_thickness', 'drivese.bedplate_flange_thickness') # only used in geared
                self.connect('nacelle.bedplate_web_thickness',    'drivese.bedplate_web_thickness') # only used in geared
                
            self.connect('hub.hub_material',                  'drivese.hub_material')
            self.connect('hub.spinner_material',              'drivese.spinner_material')
            self.connect('nacelle.lss_material',              'drivese.lss_material')
            self.connect('nacelle.bedplate_material',         'drivese.bedplate_material')
            self.connect('materials.name',                    'drivese.material_names')
            self.connect('materials.E',                       'drivese.E_mat')
            self.connect('materials.G',                       'drivese.G_mat')
            self.connect('materials.rho',                     'drivese.rho_mat')
            self.connect('materials.sigma_y',                 'drivese.sigma_y_mat')
            self.connect('materials.Xt',                      'drivese.Xt_mat')
            self.connect('materials.unit_cost',               'drivese.unit_cost_mat')

            if modeling_options['flags']['generator']:

                self.connect('generator.B_r'          , 'drivese.generator.B_r')
                self.connect('generator.P_Fe0e'       , 'drivese.generator.P_Fe0e')
                self.connect('generator.P_Fe0h'       , 'drivese.generator.P_Fe0h')
                self.connect('generator.S_N'          , 'drivese.generator.S_N')
                self.connect('generator.alpha_p'      , 'drivese.generator.alpha_p')
                self.connect('generator.b_r_tau_r'    , 'drivese.generator.b_r_tau_r')
                self.connect('generator.b_ro'         , 'drivese.generator.b_ro')
                self.connect('generator.b_s_tau_s'    , 'drivese.generator.b_s_tau_s')
                self.connect('generator.b_so'         , 'drivese.generator.b_so')
                self.connect('generator.cofi'         , 'drivese.generator.cofi')
                self.connect('generator.freq'         , 'drivese.generator.freq')
                self.connect('generator.h_i'          , 'drivese.generator.h_i')
                self.connect('generator.h_sy0'        , 'drivese.generator.h_sy0')
                self.connect('generator.h_w'          , 'drivese.generator.h_w')
                self.connect('generator.k_fes'        , 'drivese.generator.k_fes')
                self.connect('generator.k_fillr'      , 'drivese.generator.k_fillr')
                self.connect('generator.k_fills'      , 'drivese.generator.k_fills')
                self.connect('generator.k_s'          , 'drivese.generator.k_s')
                self.connect('generator.m'            , 'drivese.generator.m')
                self.connect('generator.mu_0'         , 'drivese.generator.mu_0')
                self.connect('generator.mu_r'         , 'drivese.generator.mu_r')
                self.connect('generator.p'            , 'drivese.generator.p')
                self.connect('generator.phi'          , 'drivese.generator.phi')
                self.connect('generator.q1'           , 'drivese.generator.q1')
                self.connect('generator.q2'           , 'drivese.generator.q2')
                self.connect('generator.ratio_mw2pp'  , 'drivese.generator.ratio_mw2pp')
                self.connect('generator.resist_Cu'    , 'drivese.generator.resist_Cu')
                self.connect('generator.sigma'        , 'drivese.generator.sigma')
                self.connect('generator.y_tau_p'      , 'drivese.generator.y_tau_p')
                self.connect('generator.y_tau_pr'     , 'drivese.generator.y_tau_pr')

                self.connect('generator.I_0'          , 'drivese.generator.I_0')
                self.connect('generator.d_r'          , 'drivese.generator.d_r')
                self.connect('generator.h_m'          , 'drivese.generator.h_m')
                self.connect('generator.h_0'          , 'drivese.generator.h_0')
                self.connect('generator.h_s'          , 'drivese.generator.h_s')
                self.connect('generator.len_s'        , 'drivese.generator.len_s')
                self.connect('generator.n_r'          , 'drivese.generator.n_r')
                self.connect('generator.rad_ag'       , 'drivese.generator.rad_ag')
                self.connect('generator.t_wr'         , 'drivese.generator.t_wr')

                self.connect('generator.n_s'          , 'drivese.generator.n_s')
                self.connect('generator.b_st'         , 'drivese.generator.b_st')
                self.connect('generator.d_s'          , 'drivese.generator.d_s')
                self.connect('generator.t_ws'         , 'drivese.generator.t_ws')

                self.connect('generator.rho_Copper'   , 'drivese.generator.rho_Copper')
                self.connect('generator.rho_Fe'       , 'drivese.generator.rho_Fe')
                self.connect('generator.rho_Fes'      , 'drivese.generator.rho_Fes')
                self.connect('generator.rho_PM'       , 'drivese.generator.rho_PM')

                self.connect('generator.C_Cu'         , 'drivese.generator.C_Cu')
                self.connect('generator.C_Fe'         , 'drivese.generator.C_Fe')
                self.connect('generator.C_Fes'        , 'drivese.generator.C_Fes')
                self.connect('generator.C_PM'         , 'drivese.generator.C_PM')

                if modeling_options['GeneratorSE']['type'] in ['pmsg_outer']:
                    self.connect('generator.N_c'          , 'drivese.generator.N_c')
                    self.connect('generator.b'            , 'drivese.generator.b')
                    self.connect('generator.c'            , 'drivese.generator.c')
                    self.connect('generator.E_p'          , 'drivese.generator.E_p')
                    self.connect('generator.h_yr'         , 'drivese.generator.h_yr')
                    self.connect('generator.h_ys'         , 'drivese.generator.h_ys')
                    self.connect('generator.h_sr'         , 'drivese.generator.h_sr')
                    self.connect('generator.h_ss'         , 'drivese.generator.h_ss')
                    self.connect('generator.t_r'          , 'drivese.generator.t_r')
                    self.connect('generator.t_s'          , 'drivese.generator.t_s')

                    self.connect('generator.u_allow_pcent', 'drivese.generator.u_allow_pcent')
                    self.connect('generator.y_allow_pcent', 'drivese.generator.y_allow_pcent')
                    self.connect('generator.z_allow_deg'  , 'drivese.generator.z_allow_deg')
                    self.connect('generator.B_tmax'       , 'drivese.generator.B_tmax')
                    self.connect('sse.powercurve.rated_mech', 'drivese.generator.P_mech')

                if modeling_options['GeneratorSE']['type'] in ['eesg','pmsg_arms','pmsg_disc']:
                    self.connect('generator.tau_p'        , 'drivese.generator.tau_p')
                    self.connect('generator.h_ys'         , 'drivese.generator.h_ys')
                    self.connect('generator.h_yr'         , 'drivese.generator.h_yr')
                    self.connect('generator.b_arm'        , 'drivese.generator.b_arm')

                elif modeling_options['GeneratorSE']['type'] in ['scig','dfig']:
                    self.connect('generator.B_symax'      , 'drivese.generator.B_symax')
                    self.connect('generator.S_Nmax'      , 'drivese.generator.S_Nmax')

                if modeling_options['drivetrainse']['direct']:
                    self.connect('nacelle.nose_diameter',             'drivese.generator.D_nose', src_indices=[-1])
                    self.connect('nacelle.lss_diameter',              'drivese.generator.D_shaft', src_indices=[0])
                else:
                    self.connect('nacelle.hss_diameter',              'drivese.generator.D_shaft', src_indices=[-1])

            else:
                self.connect('generator.generator_mass_user', 'drivese.generator_mass_user')
                self.connect('generator.generator_efficiency_user', 'drivese.generator_efficiency_user')


        # Connections to TowerSE
        if modeling_options['flags']['tower']:
            if modeling_options['flags']['nacelle']:
                self.connect('drivese.base_F',                'towerse.pre.rna_F')
                self.connect('drivese.base_M',                'towerse.pre.rna_M')
                self.connect('drivese.rna_I_TT',             'towerse.rna_I')
                self.connect('drivese.rna_cm',               'towerse.rna_cg')
                self.connect('drivese.rna_mass',             'towerse.rna_mass')
            if modeling_options['flags']['blade']:
                self.connect('sse.gust.V_gust',               'towerse.wind.Uref')
            self.connect('assembly.hub_height',           'towerse.wind_reference_height')  # TODO- environment
            self.connect('foundation.height',             'towerse.wind_z0') # TODO- environment
            self.connect('env.rho_air',                   'towerse.rho_air')
            self.connect('env.mu_air',                    'towerse.mu_air')                    
            self.connect('env.shear_exp',                 'towerse.shearExp')                    
            self.connect('assembly.hub_height',           'towerse.hub_height')
            self.connect('foundation.height',             'towerse.foundation_height')
            self.connect('tower.diameter',                'towerse.tower_outer_diameter_in')
            self.connect('tower.height',                  'towerse.tower_height')
            self.connect('tower.s',                       'towerse.tower_s')
            self.connect('tower.layer_thickness',         'towerse.tower_layer_thickness')
            self.connect('tower.outfitting_factor',       'towerse.tower_outfitting_factor')
            self.connect('tower.layer_mat',               'towerse.tower_layer_materials')
            self.connect('materials.name',                'towerse.material_names')
            self.connect('materials.E',                   'towerse.E_mat')
            self.connect('materials.G',                   'towerse.G_mat')
            self.connect('materials.rho',                 'towerse.rho_mat')
            self.connect('materials.sigma_y',             'towerse.sigma_y_mat')
            self.connect('materials.unit_cost',           'towerse.unit_cost_mat')
            self.connect('costs.labor_rate',              'towerse.labor_cost_rate')
            self.connect('costs.painting_rate',           'towerse.painting_cost_rate')
            if modeling_options['flags']['monopile']:
                self.connect('env.rho_water',                    'towerse.rho_water')
                self.connect('env.mu_water',                     'towerse.mu_water')                    
                self.connect('env.G_soil',                       'towerse.G_soil')                    
                self.connect('env.nu_soil',                      'towerse.nu_soil')
                self.connect('env.hsig_wave',                    'towerse.hsig_wave')
                self.connect('env.Tsig_wave',                    'towerse.Tsig_wave')
                self.connect('monopile.diameter',                'towerse.monopile_outer_diameter_in')
                self.connect('monopile.height',                  'towerse.monopile_height')
                self.connect('monopile.s',                       'towerse.monopile_s')
                self.connect('monopile.layer_thickness',         'towerse.monopile_layer_thickness')
                self.connect('monopile.layer_mat',               'towerse.monopile_layer_materials')
                self.connect('monopile.outfitting_factor',       'towerse.monopile_outfitting_factor')
                self.connect('monopile.transition_piece_height', 'towerse.transition_piece_height')
                self.connect('monopile.transition_piece_mass',   'towerse.transition_piece_mass')
                self.connect('monopile.gravity_foundation_mass', 'towerse.gravity_foundation_mass')
                self.connect('monopile.suctionpile_depth',       'towerse.suctionpile_depth')
                self.connect('monopile.suctionpile_depth_diam_ratio', 'towerse.suctionpile_depth_diam_ratio')

        # Connections to turbine constraints
        if modeling_options['flags']['blade'] and modeling_options['flags']['tower']:
            self.connect('configuration.rotor_orientation', 'tcons.rotor_orientation')
            self.connect('rlds.tip_pos.tip_deflection',     'tcons.tip_deflection')
            self.connect('assembly.rotor_radius',           'tcons.Rtip')
            self.connect('blade.outer_shape_bem.ref_axis',  'tcons.ref_axis_blade')
            self.connect('hub.cone',                        'tcons.precone')
            self.connect('nacelle.uptilt',                  'tcons.tilt')
            self.connect('nacelle.overhang',                'tcons.overhang')
            self.connect('tower.ref_axis',                  'tcons.ref_axis_tower')
            self.connect('tower.diameter',                  'tcons.d_full')

        # Connections to turbine capital cost
        self.connect('configuration.n_blades',      'tcc.blade_number')
        if modeling_options['flags']['control']:
            self.connect('control.rated_power',         'tcc.machine_rating')
        if modeling_options['flags']['blade']:
            self.connect('elastic.precomp.blade_mass',      'tcc.blade_mass')
            self.connect('elastic.precomp.total_blade_cost','tcc.blade_cost_external')

        if modeling_options['flags']['nacelle']:
            self.connect('drivese.hub_mass',            'tcc.hub_mass')
            self.connect('drivese.pitch_mass',          'tcc.pitch_system_mass')
            self.connect('drivese.spinner_mass',        'tcc.spinner_mass')
            self.connect('drivese.lss_mass',            'tcc.lss_mass')
            self.connect('drivese.mean_bearing_mass',   'tcc.main_bearing_mass')
            self.connect('drivese.gearbox_mass',        'tcc.gearbox_mass')
            self.connect('drivese.hss_mass',            'tcc.hss_mass')
            self.connect('drivese.brake_mass',          'tcc.brake_mass')
            self.connect('drivese.generator_mass',      'tcc.generator_mass')
            self.connect('drivese.total_bedplate_mass', 'tcc.bedplate_mass')
            self.connect('drivese.yaw_mass',            'tcc.yaw_mass')
            self.connect('drivese.converter_mass',      'tcc.converter_mass')
            self.connect('drivese.transformer_mass',    'tcc.transformer_mass')
            self.connect('drivese.hvac_mass',           'tcc.hvac_mass')
            self.connect('drivese.cover_mass',          'tcc.cover_mass')
            self.connect('drivese.platforms_mass',      'tcc.platforms_mass')
            
            if modeling_options['flags']['generator']:
                self.connect('drivese.generator_cost',  'tcc.generator_cost_external')

        if modeling_options['flags']['tower']:
            self.connect('towerse.structural_mass',          'tcc.tower_mass')
            self.connect('towerse.structural_cost',          'tcc.tower_cost_external')
            
        self.connect('costs.blade_mass_cost_coeff'                 , 'tcc.blade_mass_cost_coeff')
        self.connect('costs.hub_mass_cost_coeff'                   , 'tcc.hub_mass_cost_coeff')
        self.connect('costs.pitch_system_mass_cost_coeff'          , 'tcc.pitch_system_mass_cost_coeff')
        self.connect('costs.spinner_mass_cost_coeff'               , 'tcc.spinner_mass_cost_coeff')
        self.connect('costs.lss_mass_cost_coeff'                   , 'tcc.lss_mass_cost_coeff')
        self.connect('costs.bearing_mass_cost_coeff'               , 'tcc.bearing_mass_cost_coeff')
        self.connect('costs.gearbox_mass_cost_coeff'               , 'tcc.gearbox_mass_cost_coeff')
        self.connect('costs.hss_mass_cost_coeff'                   , 'tcc.hss_mass_cost_coeff')
        self.connect('costs.generator_mass_cost_coeff'             , 'tcc.generator_mass_cost_coeff')
        self.connect('costs.bedplate_mass_cost_coeff'              , 'tcc.bedplate_mass_cost_coeff')
        self.connect('costs.yaw_mass_cost_coeff'                   , 'tcc.yaw_mass_cost_coeff')
        self.connect('costs.converter_mass_cost_coeff'             , 'tcc.converter_mass_cost_coeff')
        self.connect('costs.transformer_mass_cost_coeff'           , 'tcc.transformer_mass_cost_coeff')
        self.connect('costs.hvac_mass_cost_coeff'                  , 'tcc.hvac_mass_cost_coeff')
        self.connect('costs.cover_mass_cost_coeff'                 , 'tcc.cover_mass_cost_coeff')
        self.connect('costs.elec_connec_machine_rating_cost_coeff' , 'tcc.elec_connec_machine_rating_cost_coeff')
        self.connect('costs.platforms_mass_cost_coeff'             , 'tcc.platforms_mass_cost_coeff')
        self.connect('costs.tower_mass_cost_coeff'                 , 'tcc.tower_mass_cost_coeff')
        self.connect('costs.controls_machine_rating_cost_coeff'    , 'tcc.controls_machine_rating_cost_coeff')
        self.connect('costs.crane_cost'                            , 'tcc.crane_cost')

class WindPark(om.Group):
    # Openmdao group to run the cost analysis of a wind park
    
    def initialize(self):
        self.options.declare('modeling_options')
        self.options.declare('opt_options')
        
    def setup(self):
        modeling_options = self.options['modeling_options']
        opt_options     = self.options['opt_options']

        self.add_subsystem('wt',        WT_RNTA(modeling_options = modeling_options, opt_options = opt_options), promotes=['*'])
        if modeling_options['flags']['bos']:
            if modeling_options['offshore']:
                self.add_subsystem('orbit',     Orbit())
            else:
                self.add_subsystem('landbosse', LandBOSSE())

        if modeling_options['flags']['blade']:
            self.add_subsystem('financese', PlantFinance(verbosity=modeling_options['general']['verbosity']))
            self.add_subsystem('outputs_2_screen',  Outputs_2_Screen(modeling_options = modeling_options, opt_options = opt_options))
        
        if opt_options['opt_flag'] and opt_options['recorder']['flag']:
            self.add_subsystem('conv_plots',    Convergence_Trends_Opt(opt_options = opt_options))

        # BOS inputs
        if modeling_options['flags']['bos']:
            if modeling_options['offshore']:
                # Inputs into ORBIT
                self.connect('control.rated_power',                   'orbit.turbine_rating')
                self.connect('env.water_depth',                       'orbit.site_depth')
                self.connect('costs.turbine_number',                  'orbit.number_of_turbines')
                self.connect('configuration.n_blades',                'orbit.number_of_blades')
                self.connect('assembly.hub_height',                   'orbit.hub_height')
                self.connect('assembly.rotor_diameter',               'orbit.turbine_rotor_diameter')     
                self.connect('towerse.tower_mass',                    'orbit.tower_mass')
                self.connect('towerse.monopile_mass',                 'orbit.monopile_mass')
                self.connect('towerse.monopile_length',               'orbit.monopile_length')
                self.connect('monopile.transition_piece_mass',        'orbit.transition_piece_mass')
                self.connect('elastic.precomp.blade_mass',            'orbit.blade_mass')
                self.connect('tcc.turbine_cost_kW',                   'orbit.turbine_capex')
                self.connect('drivese.nacelle_mass',                  'orbit.nacelle_mass')
                self.connect('monopile.diameter',                     'orbit.monopile_diameter', src_indices=[0])
                self.connect('wt_class.V_mean',                       'orbit.site_mean_windspeed')
                self.connect('sse.powercurve.rated_V',                'orbit.turbine_rated_windspeed')
                self.connect('bos.plant_turbine_spacing',             'orbit.plant_turbine_spacing')
                self.connect('bos.plant_row_spacing',                 'orbit.plant_row_spacing')
                self.connect('bos.commissioning_pct',                 'orbit.commissioning_pct')
                self.connect('bos.decommissioning_pct',               'orbit.decommissioning_pct')
                self.connect('bos.distance_to_substation',            'orbit.plant_substation_distance')
                self.connect('bos.distance_to_interconnection',       'orbit.interconnection_distance')
                self.connect('bos.site_distance',                     'orbit.site_distance')
                self.connect('bos.distance_to_landfall',              'orbit.site_distance_to_landfall')
                self.connect('bos.port_cost_per_month',               'orbit.port_cost_per_month')
                self.connect('bos.site_auction_price',                'orbit.site_auction_price')
                self.connect('bos.site_assessment_plan_cost',         'orbit.site_assessment_plan_cost')
                self.connect('bos.site_assessment_cost',              'orbit.site_assessment_cost')
                self.connect('bos.construction_operations_plan_cost', 'orbit.construction_operations_plan_cost')
                self.connect('bos.boem_review_cost',                  'orbit.boem_review_cost')
                self.connect('bos.design_install_plan_cost',          'orbit.design_install_plan_cost')
            else:
                # Inputs into LandBOSSE
                self.connect('assembly.hub_height',             'landbosse.hub_height_meters')
                self.connect('costs.turbine_number',            'landbosse.num_turbines')
                self.connect('control.rated_power',             'landbosse.turbine_rating_MW')
                self.connect('env.shear_exp',                   'landbosse.wind_shear_exponent')
                self.connect('assembly.rotor_diameter',         'landbosse.rotor_diameter_m')
                self.connect('configuration.n_blades',          'landbosse.number_of_blades')
                if modeling_options['flags']['blade']:
                    self.connect('sse.powercurve.rated_T',          'landbosse.rated_thrust_N')
                self.connect('towerse.tower_mass',              'landbosse.tower_mass')
                self.connect('drivese.nacelle_mass',            'landbosse.nacelle_mass')
                self.connect('elastic.precomp.blade_mass',      'landbosse.blade_mass')
                self.connect('drivese.hub_system_mass',         'landbosse.hub_mass')
                self.connect('foundation.height',               'landbosse.foundation_height')
                self.connect('bos.plant_turbine_spacing',       'landbosse.turbine_spacing_rotor_diameters')
                self.connect('bos.plant_row_spacing',           'landbosse.row_spacing_rotor_diameters')
                self.connect('bos.commissioning_pct',           'landbosse.commissioning_pct')
                self.connect('bos.decommissioning_pct',         'landbosse.decommissioning_pct')
                self.connect('bos.distance_to_substation',      'landbosse.trench_len_to_substation_km')
                self.connect('bos.distance_to_interconnection', 'landbosse.distance_to_interconnect_mi')
                self.connect('bos.interconnect_voltage',        'landbosse.interconnect_voltage_kV')
            
        # Inputs to plantfinancese from wt group
        if modeling_options['flags']['blade']:
            self.connect('sse.AEP',             'financese.turbine_aep')
            self.connect('tcc.turbine_cost_kW',     'financese.tcc_per_kW')
            
            if modeling_options['flags']['bos']:
                if 'offshore' in modeling_options and modeling_options['offshore']:
                    self.connect('orbit.total_capex_kW',    'financese.bos_per_kW')
                else:
                    self.connect('landbosse.bos_capex_kW',  'financese.bos_per_kW')
            else:
                self.connect('costs.bos_per_kW',  'financese.bos_per_kW')

            # Inputs to plantfinancese from input yaml
            if modeling_options['flags']['control']:
                self.connect('control.rated_power',     'financese.machine_rating')
            self.connect('costs.turbine_number',    'financese.turbine_number')
            self.connect('costs.opex_per_kW',       'financese.opex_per_kW')
            self.connect('costs.offset_tcc_per_kW', 'financese.offset_tcc_per_kW')
            self.connect('costs.wake_loss_factor',  'financese.wake_loss_factor')
            self.connect('costs.fixed_charge_rate', 'financese.fixed_charge_rate')

        # Connections to outputs to screen
        if modeling_options['flags']['blade']:
            self.connect('sse.AEP',                 'outputs_2_screen.aep')
            self.connect('financese.lcoe',          'outputs_2_screen.lcoe')
            self.connect('elastic.precomp.blade_mass',  'outputs_2_screen.blade_mass')
            self.connect('rlds.tip_pos.tip_deflection', 'outputs_2_screen.tip_deflection')
