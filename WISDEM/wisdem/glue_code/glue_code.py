import numpy as np
from openmdao.api import ExplicitComponent, Group, Problem
from wisdem.glue_code.gc_WT_DataStruc import WindTurbineOntologyOpenMDAO
from wisdem.ccblade.ccblade_component import CCBladeTwist
from wisdem.commonse.turbine_class import TurbineClass
from wisdem.drivetrainse.drivese_omdao import DriveSE
from wisdem.towerse.tower import TowerSE
from wisdem.turbine_costsse.turbine_costsse_2015 import Turbine_CostsSE_2015
from wisdem.orbit.api.wisdem.fixed import Orbit
from wisdem.landbosse.landbosse_omdao.landbosse import LandBOSSE
from wisdem.plant_financese.plant_finance import PlantFinance
from wisdem.commonse.turbine_constraints  import TurbineConstraints
from wisdem.servose.servose import ServoSE, NoStallConstraint
from wisdem.rotorse.rotor_elasticity import RotorElasticity
from wisdem.rotorse.rotor_loads_defl_strains import RotorLoadsDeflStrains, RunFrame3DD
from wisdem.glue_code.gc_RunTools import Outputs_2_Screen, Convergence_Trends_Opt

class WT_RNTA(Group):
    # Openmdao group to run the analysis of the wind turbine
    
    def initialize(self):
        self.options.declare('modeling_options')
        self.options.declare('opt_options')
        
    def setup(self):
        modeling_options = self.options['modeling_options']
        opt_options      = self.options['opt_options']
        n_span           = modeling_options['blade']['n_span']

        # Analysis components
        self.add_subsystem('wt_init',   WindTurbineOntologyOpenMDAO(modeling_options = modeling_options, opt_options = opt_options), promotes=['*'])
        self.add_subsystem('ccblade',   CCBladeTwist(modeling_options = modeling_options, opt_options = opt_options)) # Run standalong CCBlade and possibly determine optimal twist from user-defined margin to stall
        self.add_subsystem('wt_class',  TurbineClass())
        self.add_subsystem('elastic',   RotorElasticity(modeling_options = modeling_options, opt_options = opt_options))
        if modeling_options['Analysis_Flags']['ServoSE']:
            self.add_subsystem('sse',       ServoSE(modeling_options = modeling_options)) # Aero analysis
        self.add_subsystem('stall_check', NoStallConstraint(modeling_options = modeling_options))
        self.add_subsystem('rlds',      RotorLoadsDeflStrains(modeling_options = modeling_options, opt_options = opt_options, freq_run=False))
        if modeling_options['Analysis_Flags']['DriveSE']:
            self.add_subsystem('drivese',   DriveSE(debug=False,
                                                number_of_main_bearings=1,
                                                topLevelFlag=False))
        if modeling_options['flags']['tower']:
            self.add_subsystem('towerse',   TowerSE(modeling_options=modeling_options))
            self.add_subsystem('tcons',     TurbineConstraints(modeling_options = modeling_options))
        self.add_subsystem('tcc',       Turbine_CostsSE_2015(verbosity=modeling_options['general']['verbosity']))

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
        if modeling_options['Analysis_Flags']['ServoSE']:
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
        if modeling_options['Analysis_Flags']['ServoSE']:
            self.connect('sse.powercurve.rated_V',        'sse.gust.V_hub')
            self.connect('sse.gust.V_gust',              ['rlds.aero_gust.V_load', 'rlds.aero_hub_loads.V_load'])
            self.connect('sse.powercurve.rated_Omega',   ['rlds.Omega_load', 'rlds.tot_loads_gust.aeroloads_Omega', 'rlds.constr.rated_Omega'])
            self.connect('sse.powercurve.rated_pitch',   ['rlds.pitch_load', 'rlds.tot_loads_gust.aeroloads_pitch'])

        # Connections to ServoSE
        if modeling_options['Analysis_Flags']['ServoSE']:
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
            self.connect('nacelle.generator_efficiency',   'sse.powercurve.generator_efficiency')
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
        if modeling_options['Analysis_Flags']['ServoSE']:
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
        if modeling_options['Analysis_Flags']['DriveSE']:
            self.connect('assembly.rotor_diameter',    'drivese.rotor_diameter')     
            self.connect('control.rated_power',        'drivese.machine_rating')    
            self.connect('nacelle.overhang',           'drivese.overhang') 
            self.connect('nacelle.uptilt',             'drivese.shaft_angle')
            self.connect('configuration.n_blades',     'drivese.number_of_blades') 
            if modeling_options['Analysis_Flags']['ServoSE']:
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
            self.connect('nacelle.gearbox_efficiency', 'drivese.gearbox_efficiency')
            self.connect('nacelle.generator_efficiency','drivese.generator_efficiency')
            if modeling_options['flags']['tower']:
                self.connect('tower.diameter',             'drivese.tower_top_diameter', src_indices=[-1])

        # Connections to TowerSE
        if modeling_options['Analysis_Flags']['DriveSE'] and modeling_options['flags']['tower']:
            self.connect('drivese.top_F',                 'towerse.pre.rna_F')
            self.connect('drivese.top_M',                 'towerse.pre.rna_M')
            self.connect('drivese.rna_I_TT',             'towerse.rna_I')
            self.connect('drivese.rna_cm',               'towerse.rna_cg')
            self.connect('drivese.rna_mass',             'towerse.rna_mass')
            if modeling_options['Analysis_Flags']['ServoSE']:
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
            if modeling_options['flags']['monopile']:
                self.connect('env.rho_water',                    'towerse.rho_water')
                self.connect('env.mu_water',                     'towerse.mu_water')                    
                self.connect('env.G_soil',                       'towerse.G_soil')                    
                self.connect('env.nu_soil',                      'towerse.nu_soil')                    
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

        #self.connect('yield_stress',            'tow.sigma_y') # TODO- materials
        #self.connect('max_taper_ratio',         'max_taper') # TODO- 
        #self.connect('min_diameter_thickness_ratio', 'min_d_to_t')
        # Connections to turbine constraints
        if modeling_options['flags']['tower']:
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
        self.connect('elastic.precomp.blade_mass',  'tcc.blade_mass')
        self.connect('elastic.precomp.total_blade_cost',  'tcc.blade_cost_external')
        if modeling_options['Analysis_Flags']['DriveSE']:
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
            self.connect('drivese.converter_mass',      'tcc.converter_mass')
            self.connect('drivese.hvac_mass',           'tcc.hvac_mass')
            self.connect('drivese.cover_mass',          'tcc.cover_mass')
            self.connect('drivese.platforms_mass',      'tcc.platforms_mass')
            self.connect('drivese.transformer_mass',    'tcc.transformer_mass')

        if modeling_options['flags']['tower']:
            self.connect('towerse.tower_mass',          'tcc.tower_mass')

class WindPark(Group):
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
        self.add_subsystem('financese', PlantFinance(verbosity=modeling_options['general']['verbosity']))
            
        # Post-processing
        self.add_subsystem('outputs_2_screen',  Outputs_2_Screen(modeling_options = modeling_options, opt_options = opt_options))
        if opt_options['opt_flag']:
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
                if modeling_options['Analysis_Flags']['ServoSE']:
                    self.connect('sse.powercurve.rated_T',          'landbosse.rated_thrust_N')
                self.connect('towerse.tower_mass',              'landbosse.tower_mass')
                self.connect('drivese.nacelle_mass',            'landbosse.nacelle_mass')
                self.connect('elastic.precomp.blade_mass',      'landbosse.blade_mass')
                self.connect('hub.system_mass',                 'landbosse.hub_mass')
                self.connect('foundation.height',               'landbosse.foundation_height')
                self.connect('bos.plant_turbine_spacing',       'landbosse.turbine_spacing_rotor_diameters')
                self.connect('bos.plant_row_spacing',           'landbosse.row_spacing_rotor_diameters')
                self.connect('bos.commissioning_pct',           'landbosse.commissioning_pct')
                self.connect('bos.decommissioning_pct',         'landbosse.decommissioning_pct')
                self.connect('bos.distance_to_substation',      'landbosse.trench_len_to_substation_km')
                self.connect('bos.distance_to_interconnection', 'landbosse.distance_to_interconnect_mi')
                self.connect('bos.interconnect_voltage',        'landbosse.interconnect_voltage_kV')
            
        # Inputs to plantfinancese from wt group
        if modeling_options['Analysis_Flags']['ServoSE']:
            self.connect('sse.AEP',             'financese.turbine_aep')

        self.connect('tcc.turbine_cost_kW',     'financese.tcc_per_kW')
        if modeling_options['flags']['bos']:
            if 'offshore' in modeling_options and modeling_options['offshore']:
                self.connect('orbit.total_capex_kW',    'financese.bos_per_kW')
            else:
                self.connect('landbosse.bos_capex_kW',  'financese.bos_per_kW')
        # Inputs to plantfinancese from input yaml
        if modeling_options['flags']['control']:
            self.connect('control.rated_power',     'financese.machine_rating')
        self.connect('costs.turbine_number',    'financese.turbine_number')
        self.connect('costs.opex_per_kW',       'financese.opex_per_kW')
        self.connect('costs.offset_tcc_per_kW', 'financese.offset_tcc_per_kW')
        self.connect('costs.wake_loss_factor',  'financese.wake_loss_factor')
        self.connect('costs.fixed_charge_rate', 'financese.fixed_charge_rate')

        # Connections to outputs to screen
        if modeling_options['Analysis_Flags']['ServoSE']:
            self.connect('sse.AEP',                 'outputs_2_screen.aep')
            self.connect('financese.lcoe',          'outputs_2_screen.lcoe')
        self.connect('elastic.precomp.blade_mass',  'outputs_2_screen.blade_mass')
        self.connect('rlds.tip_pos.tip_deflection', 'outputs_2_screen.tip_deflection')
