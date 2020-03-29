import numpy as np
from openmdao.api import ExplicitComponent, Group, Problem, IndepVarComp
from wisdem.assemblies.load_IEA_yaml import WindTurbineOntologyOpenMDAO
from wisdem.rotorse.rotor_geometry import TurbineClass
from wisdem.drivetrainse.drivese_omdao import DriveSE
from wisdem.towerse.tower import TowerSE
from wisdem.turbine_costsse.turbine_costsse_2015 import Turbine_CostsSE_2015
from wisdem.plant_financese.plant_finance import PlantFinance
from wisdem.commonse.turbine_constraints  import TurbineConstraints
from wisdem.aeroelasticse.openmdao_openfast import FASTLoadCases
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

        opt_var_flap = IndepVarComp()
        opt_var_flap.add_output('te_flap_end', val = np.ones(analysis_options['blade']['n_te_flaps']))
        opt_var_flap.add_output('te_flap_ext', val = np.ones(analysis_options['blade']['n_te_flaps']))
        self.add_subsystem('opt_var_flap', opt_var_flap)


        # Analysis components
        self.add_subsystem('wt_init',   WindTurbineOntologyOpenMDAO(analysis_options = analysis_options, opt_options = opt_options), promotes=['*'])
        self.add_subsystem('wt_class',  TurbineClass())
        self.add_subsystem('elastic',   RotorElasticity(analysis_options = analysis_options, opt_options = opt_options))
        self.add_subsystem('xf',        RunXFOIL(analysis_options = analysis_options)) # Recompute polars with xfoil (for flaps)
        self.add_subsystem('sse',       ServoSE(analysis_options = analysis_options)) # Aero analysis
        
        if analysis_options['openfast']['run_openfast'] == True:
            self.add_subsystem('aeroelastic',  FASTLoadCases(analysis_options = analysis_options))
        
        self.add_subsystem('rlds',      RotorLoadsDeflStrains(analysis_options = analysis_options, opt_options = opt_options))
        self.add_subsystem('drivese',   DriveSE(debug=False,
                                            number_of_main_bearings=1,
                                            topLevelFlag=False))
        self.add_subsystem('towerse',   TowerSE(analysis_options=analysis_options, topLevelFlag=False))

        self.add_subsystem('tcons',     TurbineConstraints(analysis_options = analysis_options))
        self.add_subsystem('tcc',       Turbine_CostsSE_2015(verbosity=analysis_options['general']['verbosity'], topLevelFlag=False))

        # Connections to wind turbine class
        self.connect('configuration.ws_class' , 'wt_class.turbine_class')
        
        # Connections from input yaml to parametrization
        self.connect('blade.re_interp_bem.s',        ['param.pa.s', 'xf.s'])
        self.connect('blade.outer_shape_bem.s',        'param.ps.s')        # keep the s coordinate for the structural components at the ref definition, e.g. linspace
        self.connect('blade.re_interp_bem.twist', 'param.pa.twist_original')
        self.connect('blade.re_interp_bem.chord', 'param.pa.chord_original')
        self.connect('blade.internal_structure_2d_fem.layer_name',      'param.ps.layer_name')
        self.connect('blade.internal_structure_2d_fem.layer_thickness', 'param.ps.layer_thickness_original')

        # Connections from blade aero parametrization to other modules
        # self.connect('param.pa.twist_param',           ['sse.theta','elastic.theta','rlds.theta'])
        # self.connect('param.pa.twist_param',            'rlds.tip_pos.theta_tip',   src_indices=[-1])
        # self.connect('param.pa.chord_param',           ['xf.chord', 'elastic.chord', 'sse.chord','rlds.chord'])

        self.connect('blade.outer_shape_bem.twist',     ['elastic.theta','rlds.theta'])
        self.connect('param.pa.twist_param',            ['sse.theta'])
        self.connect('param.pa.twist_param',            'rlds.tip_pos.theta_tip',   src_indices=[-1])
        self.connect('blade.outer_shape_bem.chord',     ['elastic.chord', 'rlds.chord'])
        self.connect('param.pa.chord_param',            ['xf.chord', 'sse.chord'])


        # Connections from blade struct parametrization to rotor elasticity
        self.connect('blade.ps.layer_thickness_param', 'elastic.precomp.layer_thickness')

        # Connections to rotor elastic and frequency analysis  (in non-dynamic _ref grid system)
        self.connect('nacelle.uptilt',                                  'elastic.precomp.uptilt')
        self.connect('configuration.n_blades',                          'elastic.precomp.n_blades')
        # self.connect('assembly.r_blade',                                'elastic.r')
        self.connect('assembly.r_blade_ref',                                'elastic.r')
        self.connect('blade.outer_shape_bem.pitch_axis',                'elastic.precomp.pitch_axis')
        self.connect('blade.interp_airfoils_struct.coord_xy_interp',    'elastic.precomp.coord_xy_interp')
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
            self.connect('blade.pa.twist_param',                    'elastic.rail.theta')
            self.connect('blade.pa.chord_param',                    'elastic.rail.chord')
            self.connect('blade.outer_shape_bem.pitch_axis',        'elastic.rail.pitch_axis')
            self.connect('blade.outer_shape_bem.ref_axis',          'elastic.rail.blade_ref_axis')
            self.connect('blade.interp_airfoils_aero.coord_xy_dim',      'elastic.rail.coord_xy_dim')
            self.connect('blade.interp_airfoils_aero.coord_xy_interp',   'elastic.rail.coord_xy_interp')

        # Connections from blade struct parametrization to rotor load anlysis
        self.connect('blade.ps.s_opt_spar_cap_ss',   'rlds.constr.s_opt_spar_cap_ss')
        self.connect('blade.ps.s_opt_spar_cap_ps',   'rlds.constr.s_opt_spar_cap_ps')

        # Connection from ra to rs for the rated conditions
        # self.connect('sse.powercurve.rated_V',        'rlds.aero_rated.V_load')
        self.connect('sse.powercurve.rated_V',        'rlds.gust.V_hub')
        self.connect('rlds.gust.V_gust',              ['rlds.aero_gust.V_load', 'rlds.aero_hub_loads.V_load'])
        self.connect('sse.powercurve.rated_Omega',   ['rlds.Omega_load', 'rlds.aeroloads_Omega', 'rlds.constr.rated_Omega'])
        self.connect('sse.powercurve.rated_pitch',   ['rlds.pitch_load', 'rlds.aeroloads_pitch'])
        
        # Connections to Update blade grid (s-coordinate)
        self.connect('opt_var_flap.te_flap_end',             'blade.re_interp_bem.span_end')
        self.connect('opt_var_flap.te_flap_ext',             'blade.re_interp_bem.span_ext')
        
        # Connections to run xfoil for te flaps  (in dynamic grid system)
        self.connect('blade.interp_airfoils_aero.coord_xy_interp', 'xf.coord_xy_interp')
        self.connect('airfoils.aoa',                          'xf.aoa')
        self.connect('assembly.r_blade',                      'xf.r')
        self.connect('blade.opt_var.te_flap_end',             'xf.span_end')
        self.connect('blade.opt_var.te_flap_ext',             'xf.span_ext')
        self.connect('blade.dac_te_flaps.chord_start',        'xf.chord_start')
        self.connect('blade.dac_te_flaps.delta_max_pos',      'xf.delta_max_pos')
        self.connect('blade.dac_te_flaps.delta_max_neg',      'xf.delta_max_neg')
        self.connect('env.speed_sound_air',                   'xf.speed_sound_air')
        self.connect('env.rho_air',                           'xf.rho_air')
        self.connect('env.mu_air',                            'xf.mu_air')
        self.connect('pc.tsr_opt',                            'xf.rated_TSR')
        self.connect('control.max_TS',                        'xf.max_TS')
        self.connect('blade.interp_airfoils_aero.cl_interp',       'xf.cl_interp')
        self.connect('blade.interp_airfoils_aero.cd_interp',       'xf.cd_interp')
        self.connect('blade.interp_airfoils_aero.cm_interp',       'xf.cm_interp')

        # Connections to ServoSE
        self.connect('wt_class.V_mean',             'sse.cdf.xbar')
        self.connect('control.V_in' ,               'sse.v_min')
        self.connect('control.V_out' ,              'sse.v_max')
        self.connect('control.rated_power' ,        'sse.rated_power')
        self.connect('control.minOmega' ,           'sse.omega_min')
        self.connect('control.maxOmega' ,           'sse.omega_max')
        self.connect('control.max_TS' ,             'sse.control_maxTS')
        self.connect('pc.tsr_opt' ,                 'sse.tsr_operational')
        self.connect('control.rated_pitch' ,        'sse.control_pitch')

        self.connect('configuration.gearbox_type' , 'sse.drivetrainType')
        self.connect('assembly.r_blade',            'sse.r')
        self.connect('assembly.rotor_radius',       'sse.Rtip')
        self.connect('hub.radius',                  'sse.Rhub')
        self.connect('assembly.hub_height',         'sse.hub_height')
        self.connect('hub.cone',                    'sse.precone')
        self.connect('nacelle.uptilt',              'sse.tilt')
        self.connect('airfoils.aoa',                'sse.airfoils_aoa')
            
        self.connect('xf.flap_angles',              'sse.airfoils_Ctrl')
        self.connect('airfoils.Re',                 'sse.airfoils_Re')
        self.connect('xf.cl_interp_flaps',          'sse.airfoils_cl')
        self.connect('xf.cd_interp_flaps',          'sse.airfoils_cd')
        self.connect('xf.cm_interp_flaps',          'sse.airfoils_cm')
        self.connect('configuration.n_blades',      'sse.nBlades')
        self.connect('blade.re_interp_bem.s',       'sse.stall_check.s')
        self.connect('env.rho_air',                 'sse.rho')
        self.connect('env.mu_air',                  'sse.mu')
        self.connect('env.weibull_k',               'sse.cdf.k')
        
        # Connnections to curvefem_rated
        self.connect('assembly.r_blade',            'sse.curvefem_rated.r')
        self.connect('elastic.EA',                  'sse.curvefem_rated.EA')
        self.connect('elastic.EIxx',                'sse.curvefem_rated.EIxx')
        self.connect('elastic.EIyy',                'sse.curvefem_rated.EIyy')
        self.connect('elastic.GJ',                  'sse.curvefem_rated.GJ')
        self.connect('elastic.rhoA',                'sse.curvefem_rated.rhoA')
        self.connect('elastic.rhoJ',                'sse.curvefem_rated.rhoJ')
        self.connect('elastic.Tw_iner',             'sse.curvefem_rated.Tw_iner')
        # self.connect('elastic.precurve',            'sse.curvefem_rated.precurve')
        # self.connect('elastic.presweep',            'sse.curvefem_rated.presweep')

        if analysis_options['openfast']['run_openfast']:
            self.connect('xf.flap_angles',              'sse.airfoils_Ctrl')
            self.connect('nacelle.gear_ratio',              'sse.tune_rosco.gear_ratio')
            self.connect('assembly.rotor_radius',           'sse.tune_rosco.R')
            self.connect('elastic.precomp.I_all_blades',    'sse.tune_rosco.rotor_inertia', src_indices=[0])
            self.connect('nacelle.drivetrain_eff',          'sse.tune_rosco.gen_eff')
            self.connect('elastic.curvefem.freq',           'sse.tune_rosco.flap_freq', src_indices=[0])
            self.connect('elastic.curvefem.freq',           'sse.tune_rosco.edge_freq', src_indices=[1])
            self.connect('control.max_pitch',               'sse.tune_rosco.max_pitch') 
            self.connect('control.min_pitch',               'sse.tune_rosco.min_pitch')
            self.connect('control.max_pitch_rate' ,         'sse.tune_rosco.max_pitch_rate')
            self.connect('control.max_torque_rate' ,        'sse.tune_rosco.max_torque_rate')
            self.connect('control.vs_minspd',               'sse.tune_rosco.vs_minspd') 
            self.connect('control.ss_vsgain',               'sse.tune_rosco.ss_vsgain') 
            self.connect('control.ss_pcgain',               'sse.tune_rosco.ss_pcgain') 
            self.connect('control.ps_percent',              'sse.tune_rosco.ps_percent') 
            self.connect('control.PC_omega',                'sse.tune_rosco.PC_omega')
            self.connect('control.PC_zeta',                 'sse.tune_rosco.PC_zeta')
            self.connect('control.VS_omega',                'sse.tune_rosco.VS_omega')
            self.connect('control.VS_zeta',                 'sse.tune_rosco.VS_zeta')
            self.connect('blade.dac_te_flaps.delta_max_pos','sse.tune_rosco.delta_max_pos')
            if analysis_options['servose']['Flp_Mode'] > 0:
                self.connect('control.Flp_omega',               'sse.tune_rosco.Flp_omega')
                self.connect('control.Flp_zeta',                'sse.tune_rosco.Flp_zeta')
        

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
        self.connect('sse.curvefem_rated.freq',   'rlds.constr.freq') 

        self.connect('assembly.r_blade_ref',                'rlds.r')
        self.connect('assembly.rotor_radius',           'rlds.Rtip')
        self.connect('hub.radius',                      'rlds.Rhub')
        self.connect('assembly.hub_height',             'rlds.hub_height')
        self.connect('hub.cone',                        'rlds.precone')
        self.connect('nacelle.uptilt',                  'rlds.tilt')
        self.connect('airfoils.aoa',                    'rlds.airfoils_aoa')
        self.connect('airfoils.Re',                     'rlds.airfoils_Re')
        self.connect('blade.interp_airfoils_aero.cl_interp', 'rlds.airfoils_cl')
        self.connect('blade.interp_airfoils_aero.cd_interp', 'rlds.airfoils_cd')
        self.connect('blade.interp_airfoils_aero.cm_interp', 'rlds.airfoils_cm')
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
        # self.connect('blade.internal_structure_2d_fem.layer_web',       'rotorse.rc.layer_web')
        # self.connect('blade.internal_structure_2d_fem.web_start_nd',    'rotorse.rc.web_start_nd')
        # self.connect('blade.internal_structure_2d_fem.web_end_nd',      'rotorse.rc.web_end_nd')
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
        self.connect('blade.pa.chord_param',       'drivese.blade_root_diameter', src_indices=[0])
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

        # Connections to TowerSE
        self.connect('drivese.top_F',                 'towerse.pre.rna_F')
        self.connect('drivese.top_M',                 'towerse.pre.rna_M')
        self.connect('drivese.rna_I_TT',             ['towerse.rna_I','towerse.pre.mI'])
        self.connect('drivese.rna_cm',               ['towerse.rna_cg','towerse.pre.mrho'])
        self.connect('drivese.rna_mass',             ['towerse.rna_mass','towerse.pre.mass'])
        self.connect('rlds.gust.V_gust',              'towerse.wind.Uref')
        self.connect('assembly.hub_height',           'towerse.wind.zref')  # TODO- environment
        self.connect('foundation.height',             'towerse.wind.z0') # TODO- environment
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
        if analysis_options['tower']['monopile']:
            self.connect('env.rho_water',                 'towerse.rho_water')
            self.connect('env.mu_water',                  'towerse.mu_water')                    
            self.connect('env.G_soil',                    'towerse.G_soil')                    
            self.connect('env.nu_soil',                   'towerse.nu_soil')                    
            self.connect('monopile.diameter',                'towerse.monopile_outer_diameter_in')
            self.connect('monopile.height',                  'towerse.monopile_height')
            self.connect('monopile.s',                       'towerse.monopile_s')
            self.connect('monopile.layer_thickness',         'towerse.monopile_layer_thickness')
            self.connect('monopile.layer_mat',               'towerse.monopile_layer_materials')
            self.connect('monopile.outfitting_factor',       'towerse.monopile_outfitting_factor')
            self.connect('monopile.transition_piece_height', 'towerse.transition_piece_height')
            self.connect('monopile.transition_piece_mass',   'towerse.transition_piece_maxx')
            self.connect('monopile.gravity_foundation_mass', 'towerse.gravity_foundation_mass')
            self.connect('monopile.suctionpile_depth',       'towerse.suctionpile_depth')
            self.connect('monopile.suctionpile_depth_diam_ratio', 'towerse.suctionpile_depth_diam_ratio')

        #self.connect('yield_stress',            'tow.sigma_y') # TODO- materials
        #self.connect('max_taper_ratio',         'max_taper') # TODO- 
        #self.connect('min_diameter_thickness_ratio', 'min_d_to_t')
          
        # Connections to aeroelasticse
        if analysis_options['openfast']['run_openfast'] == True:
            self.connect('blade.re_interp_bem.ref_axis',  'aeroelastic.ref_axis_blade')
            # self.connect('blade.outer_shape_bem.ref_axis',  'aeroelastic.ref_axis_blade')
            self.connect('configuration.rotor_orientation', 'aeroelastic.rotor_orientation')
            self.connect('assembly.r_blade',                'aeroelastic.r')
            self.connect('blade.outer_shape_bem.pitch_axis','aeroelastic.le_location')
            self.connect('blade.pa.chord_param',            'aeroelastic.chord')
            self.connect('blade.pa.twist_param',            'aeroelastic.theta')
            self.connect('blade.interp_airfoils_struct.coord_xy_interp', 'aeroelastic.coord_xy_interp')
            self.connect('env.rho_air',                     'aeroelastic.rho')
            self.connect('env.mu_air',                      'aeroelastic.mu')                    
            self.connect('env.shear_exp',                   'aeroelastic.shearExp')                    
            self.connect('assembly.rotor_radius',           'aeroelastic.Rtip')
            self.connect('hub.radius',                      'aeroelastic.Rhub')
            self.connect('hub.cone',                        'aeroelastic.cone')
            self.connect('hub.system_mass',                 'aeroelastic.hub_system_mass')
            self.connect('hub.system_I',                    'aeroelastic.hub_system_I')
            # self.connect('hub.system_cm',                    'aeroelastic.hub_system_cm')
            self.connect('nacelle.above_yaw_mass',          'aeroelastic.above_yaw_mass')
            self.connect('nacelle.yaw_mass',                'aeroelastic.yaw_mass')
            self.connect('nacelle.nacelle_I',               'aeroelastic.nacelle_I')
            self.connect('nacelle.nacelle_cm',              'aeroelastic.nacelle_cm')

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
            self.connect('blade.interp_airfoils_aero.r_thick_interp', 'aeroelastic.rthick')
            self.connect('elastic.rhoA',                'aeroelastic.beam:rhoA')
            self.connect('elastic.EIxx',                'aeroelastic.beam:EIxx')
            self.connect('elastic.EIyy',                'aeroelastic.beam:EIyy')
            self.connect('elastic.Tw_iner',             'aeroelastic.beam:Tw_iner')
            self.connect('sse.curvefem_rated.modes_coef', 'aeroelastic.modes_coef_curvefem')
            self.connect('sse.powercurve.V',            'aeroelastic.U_init')
            self.connect('sse.powercurve.Omega',        'aeroelastic.Omega_init')
            self.connect('sse.powercurve.pitch',        'aeroelastic.pitch_init')
            self.connect('sse.powercurve.V_R25',        'aeroelastic.V_R25')
            self.connect('sse.powercurve.rated_V',      'aeroelastic.Vrated')
            self.connect('rlds.gust.V_gust',            'aeroelastic.Vgust')
            self.connect('wt_class.V_mean',             'aeroelastic.V_mean_iec')
            self.connect('control.rated_power',         'aeroelastic.control_ratedPower')
            self.connect('control.max_TS',              'aeroelastic.control_maxTS')
            self.connect('control.maxOmega',            'aeroelastic.control_maxOmega')
            self.connect('configuration.turb_class',        'aeroelastic.turbulence_class')
            self.connect('configuration.ws_class' ,         'aeroelastic.turbine_class')
            self.connect('sse.aeroperf_tables.pitch_vector','aeroelastic.pitch_vector')
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
        self.connect('blade.re_interp_bem.ref_axis',  'tcons.ref_axis_blade')
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
        self.connect('towerse.tower_mass',                  'tcc.tower_mass')

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
        if opt_options['opt_flag']:
            self.add_subsystem('conv_plots',        Convergence_Trends_Opt(opt_options = opt_options))

        # Inputs to plantfinancese from wt group
        self.connect('sse.AEP',                 'financese.turbine_aep')
        self.connect('tcc.turbine_cost_kW',     'financese.tcc_per_kW')
        # Inputs to plantfinancese from input yaml
        self.connect('control.rated_power',     'financese.machine_rating')
        self.connect('costs.turbine_number',    'financese.turbine_number')
        self.connect('costs.bos_per_kW',        'financese.bos_per_kW')
        self.connect('costs.opex_per_kW',       'financese.opex_per_kW')
        self.connect('costs.offset_tcc_per_kW', 'financese.offset_tcc_per_kW')
        self.connect('costs.wake_loss_factor',  'financese.wake_loss_factor')
        self.connect('costs.fixed_charge_rate', 'financese.fixed_charge_rate')

        # Connections to outputs to screen
        self.connect('sse.AEP',                    'outputs_2_screen.aep')
        self.connect('elastic.precomp.blade_mass', 'outputs_2_screen.blade_mass')
        self.connect('financese.lcoe',             'outputs_2_screen.lcoe')
        self.connect('aeroelastic.My_std',         'outputs_2_screen.My_std')
        self.connect('aeroelastic.flp1_std',       'outputs_2_screen.flp1_std')


