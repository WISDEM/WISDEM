import numpy as np
from openmdao.api import ExplicitComponent, Group, Problem
from wisdem.assemblies.load_IEA_yaml import WindTurbineOntologyOpenMDAO
from wisdem.rotorse.rotor_geometry import TurbineClass
from wisdem.drivetrainse.drivese_omdao import DriveSE
from wisdem.turbine_costsse.turbine_costsse_2015 import Turbine_CostsSE_2015
from wisdem.plant_financese.plant_finance import PlantFinance
from wisdem.commonse.turbine_constraints  import TurbineConstraints
from wisdem.aeroelasticse.openmdao_openfast import FASTLoadCases
from wisdem.assemblies.parametrize_wt import WT_Parametrize
from wisdem.rotorse.dac import RunXFOIL
from wisdem.servose.servose import ServoSE
from wisdem.rotorse.rotor_elasticity import RotorElasticity
from wisdem.rotorse.rotor_loads_defl_strains import RotorLoadsDeflStrains
from wisdem.assemblies.run_tools import Outputs_2_Screen, Convergence_Trends_Opt

class WT_RNTA(Group):
    # Openmdao group to run the analysis of the wind turbine
    
    def initialize(self):
        self.options.declare('analysis_options')
        self.options.declare('opt_options')
        
    def setup(self):
        analysis_options = self.options['analysis_options']
        opt_options     = self.options['opt_options']

        # Analysis components
        self.add_subsystem('wt_init',   WindTurbineOntologyOpenMDAO(analysis_options = analysis_options), promotes=['*'])
        self.add_subsystem('wt_class',  TurbineClass())
        self.add_subsystem('param',     WT_Parametrize(analysis_options = analysis_options, opt_options = opt_options))
        self.add_subsystem('elastic',   RotorElasticity(analysis_options = analysis_options, opt_options = opt_options))
        self.add_subsystem('xf',        RunXFOIL(analysis_options = analysis_options)) # Recompute polars with xfoil (for flaps)
        self.add_subsystem('sse',       ServoSE(analysis_options = analysis_options)) # Aero analysis
        
        if analysis_options['openfast']['run_openfast'] == True:
            self.add_subsystem('aeroelastic',  FASTLoadCases(analysis_options = analysis_options))
        
        self.add_subsystem('rlds',      RotorLoadsDeflStrains(analysis_options = analysis_options, opt_options = opt_options))
        self.add_subsystem('drivese',   DriveSE(debug=False,
                                            number_of_main_bearings=1,
                                            topLevelFlag=False))
        # self.add_subsystem('towerse',   TowerSE())

        self.add_subsystem('tcons',     TurbineConstraints(analysis_options = analysis_options))
        self.add_subsystem('tcc',       Turbine_CostsSE_2015(verbosity=analysis_options['general']['verbosity'], topLevelFlag=False))

        # Connections to wind turbine class
        self.connect('configuration.ws_class' , 'wt_class.turbine_class')
        
        # Connections from input yaml to parametrization
        self.connect('blade.outer_shape_bem.s',        ['param.pa.s', 'param.ps.s', 'xf.s'])
        self.connect('blade.outer_shape_bem.twist', 'param.pa.twist_original')
        self.connect('blade.outer_shape_bem.chord', 'param.pa.chord_original')
        self.connect('blade.internal_structure_2d_fem.layer_name',      'param.ps.layer_name')
        self.connect('blade.internal_structure_2d_fem.layer_thickness', 'param.ps.layer_thickness_original')

        # Connections from blade aero parametrization to other modules
        self.connect('param.pa.twist_param',           ['sse.theta','elastic.theta','rlds.theta'])
        self.connect('param.pa.twist_param',            'rlds.tip_pos.theta_tip',   src_indices=[-1])
        self.connect('param.pa.chord_param',           ['xf.chord', 'elastic.chord', 'sse.chord','rlds.chord'])


        # Connections from blade struct parametrization to rotor elasticity
        self.connect('param.ps.layer_thickness_param', 'elastic.precomp.layer_thickness')

        # Connections to rotor elastic and frequency analysis
        self.connect('nacelle.uptilt',                                  'elastic.precomp.uptilt')
        self.connect('configuration.n_blades',                          'elastic.precomp.n_blades')
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
        self.connect('param.ps.s_opt_spar_cap_ss',   'rlds.constr.s_opt_spar_cap_ss')
        self.connect('param.ps.s_opt_spar_cap_ps',   'rlds.constr.s_opt_spar_cap_ps')

        # Connection from ra to rs for the rated conditions
        # self.connect('sse.powercurve.rated_V',        'rlds.aero_rated.V_load')
        self.connect('sse.powercurve.rated_V',        'rlds.gust.V_hub')
        self.connect('rlds.gust.V_gust',              ['rlds.aero_gust.V_load', 'rlds.aero_hub_loads.V_load'])
        self.connect('sse.powercurve.rated_Omega',   ['rlds.Omega_load', 'rlds.aeroloads_Omega', 'elastic.curvefem.Omega', 'rlds.constr.rated_Omega'])
        self.connect('sse.powercurve.rated_pitch',   ['rlds.pitch_load', 'rlds.aeroloads_pitch'])
        

        
        # Connections to run xfoil for te flaps
        self.connect('blade.interp_airfoils.coord_xy_interp', 'xf.coord_xy_interp')
        self.connect('airfoils.aoa',                          'xf.aoa')
        self.connect('assembly.r_blade',                      'xf.r')
        self.connect('param.opt_var.te_flap_end',             'xf.span_end')
        self.connect('param.opt_var.te_flap_ext',             'xf.span_ext')
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

        # Connections to ServoSE
        self.connect('wt_class.V_mean',         'sse.cdf.xbar')
        self.connect('control.V_in' ,           'sse.v_min')
        self.connect('control.V_out' ,          'sse.v_max')
        self.connect('control.rated_power' ,    'sse.rated_power')
        self.connect('control.minOmega' ,       'sse.omega_min')
        self.connect('control.maxOmega' ,       'sse.omega_max')
        self.connect('control.max_TS' ,         'sse.control_maxTS')
        self.connect('control.rated_TSR' ,      'sse.tsr_operational')
        self.connect('control.rated_pitch' ,    'sse.control_pitch')

        self.connect('configuration.gearbox_type' , 'sse.drivetrainType')
        self.connect('assembly.r_blade',            'sse.r')
        self.connect('assembly.rotor_radius',       'sse.Rtip')
        self.connect('hub.radius',                  'sse.Rhub')
        self.connect('assembly.hub_height',         'sse.hub_height')
        self.connect('hub.cone',                    'sse.precone')
        self.connect('nacelle.uptilt',              'sse.tilt')
        self.connect('airfoils.aoa',                    'sse.airfoils_aoa')

        self.connect('airfoils.Re',                     'sse.airfoils_Re')
        self.connect('blade.interp_airfoils.cl_interp', 'sse.airfoils_cl')
        self.connect('blade.interp_airfoils.cd_interp', 'sse.airfoils_cd')
        self.connect('blade.interp_airfoils.cm_interp', 'sse.airfoils_cm')
        self.connect('configuration.n_blades',          'sse.nBlades')
        self.connect('blade.outer_shape_bem.s',         'sse.stall_check.s')
        self.connect('env.rho_air',                     'sse.rho')
        self.connect('env.mu_air',                      'sse.mu')
        self.connect('env.weibull_k',                   'sse.cdf.k')
        
        if analysis_options['openfast']['run_openfast']:
            self.connect('nacelle.gear_ratio',              'sse.tune_rosco.gear_ratio')
            self.connect('assembly.rotor_radius',           'sse.tune_rosco.R')
            self.connect('elastic.precomp.I_all_blades',    'sse.tune_rosco.rotor_inertia', src_indices=[0])
            self.connect('nacelle.drivetrain_eff',  'sse.tune_rosco.gen_eff')
            self.connect('elastic.curvefem.freq',   'sse.tune_rosco.flap_freq', src_indices=[0])
            self.connect('elastic.curvefem.freq',   'sse.tune_rosco.edge_freq', src_indices=[1])
            self.connect('control.max_pitch',       'sse.tune_rosco.max_pitch') 
            self.connect('control.min_pitch',       'sse.tune_rosco.min_pitch')
            self.connect('control.max_pitch_rate' , 'sse.tune_rosco.max_pitch_rate')
            self.connect('control.max_torque_rate' , 'sse.tune_rosco.max_torque_rate')
            self.connect('control.vs_minspd',       'sse.tune_rosco.vs_minspd') 
            self.connect('control.ss_vsgain',       'sse.tune_rosco.ss_vsgain') 
            self.connect('control.ss_pcgain',       'sse.tune_rosco.ss_pcgain') 
            self.connect('control.ps_percent',      'sse.tune_rosco.ps_percent') 
            self.connect('control.PC_omega',        'sse.tune_rosco.PC_omega')
            self.connect('control.PC_zeta',         'sse.tune_rosco.PC_zeta')
            self.connect('control.VS_omega',        'sse.tune_rosco.VS_omega')
            self.connect('control.VS_zeta',         'sse.tune_rosco.VS_zeta')
            self.connect('control.Kp_flap',         'sse.tune_rosco.Kp_flap')
            self.connect('control.Ki_flap',         'sse.tune_rosco.Ki_flap')
        

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
        self.connect('env.shear_exp',                   'rlds.aero_hub_loads.shearExp')
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
        self.connect('sse.powercurve.rated_Q',         'drivese.rotor_torque')
        self.connect('sse.powercurve.rated_Omega',     'drivese.rotor_rpm')
        self.connect('rlds.aero_hub_loads.Fxyz_hub_aero', 'drivese.Fxyz')
        self.connect('rlds.aero_hub_loads.Mxyz_hub_aero', 'drivese.Mxyz')
        self.connect('elastic.precomp.I_all_blades',   'drivese.blades_I')
        self.connect('elastic.precomp.blade_mass', 'drivese.blade_mass')
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
        if analysis_options['openfast']['run_openfast'] == True:
            self.connect('blade.outer_shape_bem.ref_axis',  'aeroelastic.ref_axis_blade')
            self.connect('configuration.rotor_orientation', 'aeroelastic.rotor_orientation')
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
            self.connect('hub.cone',                        'aeroelastic.cone')
            self.connect('nacelle.uptilt',                  'aeroelastic.tilt')
            self.connect('nacelle.overhang',                'aeroelastic.overhang')
            self.connect('assembly.hub_height',             'aeroelastic.hub_height')
            self.connect('tower.height',                    'aeroelastic.tower_height')
            self.connect('foundation.height',               'aeroelastic.tower_base_height')
            self.connect('airfoils.aoa',                    'aeroelastic.airfoils_aoa')
            self.connect('airfoils.Re',                     'aeroelastic.airfoils_Re')
            self.connect('xf.cl_interp_flaps',              'aeroelastic.airfoils_cl')
            self.connect('xf.cd_interp_flaps',              'aeroelastic.airfoils_cd')
            self.connect('xf.cm_interp_flaps',              'aeroelastic.airfoils_cm')
            self.connect('blade.interp_airfoils.r_thick_interp', 'aeroelastic.rthick')
            self.connect('elastic.rhoA',                'aeroelastic.beam:rhoA')
            self.connect('elastic.EIxx',                'aeroelastic.beam:EIxx')
            self.connect('elastic.EIyy',                'aeroelastic.beam:EIyy')
            self.connect('elastic.Tw_iner',             'aeroelastic.beam:Tw_iner')
            self.connect('elastic.curvefem.modes_coef', 'aeroelastic.modes_coef_curvefem')
            self.connect('sse.powercurve.V',      'aeroelastic.U_init')
            self.connect('sse.powercurve.Omega',  'aeroelastic.Omega_init')
            self.connect('sse.powercurve.pitch',  'aeroelastic.pitch_init')
            self.connect('sse.powercurve.V_R25',  'aeroelastic.V_R25')
            self.connect('sse.powercurve.rated_V','aeroelastic.Vrated')
            self.connect('rlds.gust.V_gust',       'aeroelastic.Vgust')
            self.connect('wt_class.V_mean',              'aeroelastic.V_mean_iec')
            self.connect('control.rated_power',          'aeroelastic.control_ratedPower')
            self.connect('control.max_TS',               'aeroelastic.control_maxTS')
            self.connect('control.maxOmega',            'aeroelastic.control_maxOmega')
            self.connect('configuration.turb_class',     'aeroelastic.turbulence_class')
            self.connect('configuration.ws_class' ,      'aeroelastic.turbine_class')
            self.connect('sse.aeroperf_tables.pitch_vector', 'aeroelastic.pitch_vector')
            self.connect('sse.aeroperf_tables.tsr_vector', 'aeroelastic.tsr_vector')
            self.connect('sse.aeroperf_tables.U_vector', 'aeroelastic.U_vector')
            self.connect('sse.aeroperf_tables.Cp', 'aeroelastic.Cp_aero_table')
            self.connect('sse.aeroperf_tables.Ct', 'aeroelastic.Ct_aero_table')
            self.connect('sse.aeroperf_tables.Cq', 'aeroelastic.Cq_aero_table')

            # Temporary
            self.connect('xf.Re_loc',           'aeroelastic.airfoils_Re_loc')
            self.connect('xf.Ma_loc',           'aeroelastic.airfoils_Ma_loc')
            self.connect('xf.flap_angles',      'aeroelastic.airfoils_Ctrl')
        
        # Connections to turbine constraints
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
        self.connect('control.rated_power',         'tcc.machine_rating')
        self.connect('elastic.precomp.blade_mass',  'tcc.blade_mass')
        self.connect('elastic.precomp.total_blade_cost',  'tcc.blade_cost_external')
        self.connect('drivese.hub_mass',            'tcc.hub_mass')
        self.connect('drivese.pitch_system_mass',   'tcc.pitch_system_mass')
        self.connect('drivese.spinner_mass',        'tcc.spinner_mass')
        self.connect('drivese.lss_mass',            'tcc.lss_mass')
        self.connect('drivese.mainBearing.mb_mass', 'tcc.main_bearing_mass')
        self.connect('drivese.gearbox_mass',        'tcc.gearbox_mass')
        self.connect('drivese.hss_mass',            'tcc.hss_mass')
        self.connect('drivese.generator_mass',      'tcc.generator_mass')
        self.connect('drivese.bedplate_mass',       'tcc.bedplate_mass')
        self.connect('drivese.yaw_mass',            'tcc.yaw_mass')
        self.connect('drivese.vs_electronics_mass', 'tcc.vs_electronics_mass')
        self.connect('drivese.hvac_mass',           'tcc.hvac_mass')
        self.connect('drivese.cover_mass',          'tcc.cover_mass')
        self.connect('drivese.platforms_mass',      'tcc.platforms_mass')
        self.connect('drivese.transformer_mass',    'tcc.transformer_mass')
        # Temporary
        self.connect('tower.mass',                  'tcc.tower_mass')

class WindPark(Group):
    # Openmdao group to run the cost analysis of a wind park
    
    def initialize(self):
        self.options.declare('analysis_options')
        self.options.declare('opt_options')
        
    def setup(self):
        analysis_options = self.options['analysis_options']
        opt_options     = self.options['opt_options']

        self.add_subsystem('wt',        WT_RNTA(analysis_options = analysis_options, opt_options = opt_options), promotes=['*'])
        self.add_subsystem('financese', PlantFinance(verbosity=analysis_options['general']['verbosity']))
        # Post-processing
        self.add_subsystem('outputs_2_screen',  Outputs_2_Screen())
        self.add_subsystem('conv_plots',        Convergence_Trends_Opt(opt_options = opt_options))

        # Inputs to plantfinancese from wt group
        self.connect('sse.AEP',                 'financese.turbine_aep')
        self.connect('tcc.turbine_cost_kW',     'financese.tcc_per_kW')
        # Inputs to plantfinancese from input yaml
        self.connect('control.rated_power',     'financese.machine_rating')
        self.connect('costs.turbine_number',    'financese.turbine_number')
        self.connect('costs.bos_per_kW',        'financese.bos_per_kW')
        self.connect('costs.opex_per_kW',       'financese.opex_per_kW')
        self.connect('costs.wake_loss_factor',  'financese.wake_loss_factor')
        self.connect('costs.fixed_charge_rate', 'financese.fixed_charge_rate')

        # Connections to outputs to screen
        self.connect('sse.AEP',                    'outputs_2_screen.aep')
        self.connect('elastic.precomp.blade_mass', 'outputs_2_screen.blade_mass')
        self.connect('financese.lcoe',             'outputs_2_screen.lcoe')


