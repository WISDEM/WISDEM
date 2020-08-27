import numpy as np
import openmdao.api as om
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
from weis.aeroelasticse.openmdao_openfast import FASTLoadCases, ModesElastoDyn
from weis.control.dac import RunXFOIL
from wisdem.servose.servose import ServoSE, NoStallConstraint
from weis.control.tune_rosco import ServoSE_ROSCO
from wisdem.rotorse.rotor_elasticity import RotorElasticity
from weis.aeroelasticse.rotor_loads_defl_strainsWEIS import RotorLoadsDeflStrainsWEIS
from wisdem.glue_code.gc_RunTools import Convergence_Trends_Opt
from weis.glue_code.gc_RunTools import Outputs_2_Screen


class WT_RNTA(om.Group):
    # Openmdao group to run the analysis of the wind turbine
    
    def initialize(self):
        self.options.declare('modeling_options')
        self.options.declare('opt_options')
        
    def setup(self):
        modeling_options = self.options['modeling_options']
        opt_options      = self.options['opt_options']
        
        dac_ivc = om.IndepVarComp()
        n_te_flaps = modeling_options['blade']['n_te_flaps']
        dac_ivc.add_output('te_flap_ext',   val = np.ones(n_te_flaps))
        dac_ivc.add_output('te_flap_start', val=np.zeros(n_te_flaps),               desc='1D array of the start positions along blade span of the trailing edge flap(s). Only values between 0 and 1 are meaningful.')
        dac_ivc.add_output('te_flap_end',   val=np.zeros(n_te_flaps),               desc='1D array of the end positions along blade span of the trailing edge flap(s). Only values between 0 and 1 are meaningful.')
        dac_ivc.add_output('chord_start',   val=np.zeros(n_te_flaps),               desc='1D array of the positions along chord where the trailing edge flap(s) start. Only values between 0 and 1 are meaningful.')
        dac_ivc.add_output('delta_max_pos', val=np.zeros(n_te_flaps), units='rad',  desc='1D array of the max angle of the trailing edge flaps.')
        dac_ivc.add_output('delta_max_neg', val=np.zeros(n_te_flaps), units='rad',  desc='1D array of the min angle of the trailing edge flaps.')
        self.add_subsystem('dac_ivc',dac_ivc)

        tune_rosco_ivc = om.IndepVarComp()
        tune_rosco_ivc.add_output('PC_omega',         val=0.0, units='rad/s',     desc='Pitch controller natural frequency')
        tune_rosco_ivc.add_output('PC_zeta',          val=0.0,                    desc='Pitch controller damping ratio')
        tune_rosco_ivc.add_output('VS_omega',         val=0.0, units='rad/s',     desc='Generator torque controller natural frequency')
        tune_rosco_ivc.add_output('VS_zeta',          val=0.0,                    desc='Generator torque controller damping ratio')
        tune_rosco_ivc.add_output('Flp_omega',        val=0.0, units='rad/s',     desc='Flap controller natural frequency')
        tune_rosco_ivc.add_output('Flp_zeta',         val=0.0,                    desc='Flap controller damping ratio')
        # optional inputs - not connected right now!!
        tune_rosco_ivc.add_output('max_pitch',        val=0.0, units='rad',       desc='Maximum pitch angle , {default = 90 degrees}')
        tune_rosco_ivc.add_output('min_pitch',        val=0.0, units='rad',       desc='Minimum pitch angle [rad], {default = 0 degrees}')
        tune_rosco_ivc.add_output('vs_minspd',        val=0.0, units='rad/s',     desc='Minimum rotor speed [rad/s], {default = 0 rad/s}')
        tune_rosco_ivc.add_output('ss_cornerfreq',    val=0.0, units='rad/s',     desc='First order low-pass filter cornering frequency for setpoint smoother [rad/s]')
        tune_rosco_ivc.add_output('ss_vsgain',        val=0.0,                    desc='Torque controller setpoint smoother gain bias percentage [%, <= 1 ], {default = 100%}')
        tune_rosco_ivc.add_output('ss_pcgain',        val=0.0,                    desc='Pitch controller setpoint smoother gain bias percentage  [%, <= 1 ], {default = 0.1%}')
        tune_rosco_ivc.add_output('ps_percent',       val=0.0,                    desc='Percent peak shaving  [%, <= 1 ], {default = 80%}')
        tune_rosco_ivc.add_output('sd_maxpit',        val=0.0, units='rad',       desc='Maximum blade pitch angle to initiate shutdown [rad], {default = bld pitch at v_max}')
        tune_rosco_ivc.add_output('sd_cornerfreq',    val=0.0, units='rad/s',     desc='Cutoff Frequency for first order low-pass filter for blade pitch angle [rad/s], {default = 0.41888 ~ time constant of 15s}')
        tune_rosco_ivc.add_output('Kp_flap',          val=0.0, units='s',         desc='Proportional term of the PI controller for the trailing-edge flaps')
        tune_rosco_ivc.add_output('Ki_flap',          val=0.0,                    desc='Integral term of the PI controller for the trailing-edge flaps')
        self.add_subsystem('tune_rosco_ivc',tune_rosco_ivc)

        # Analysis components
        self.add_subsystem('wt_init',   WindTurbineOntologyOpenMDAO(modeling_options = modeling_options, opt_options = opt_options), promotes=['*'])
        self.add_subsystem('ccblade',   CCBladeTwist(modeling_options = modeling_options, opt_options = opt_options)) # Run standalone CCBlade and possibly determine optimal twist from user-defined margin to stall
        self.add_subsystem('wt_class',  TurbineClass())
        self.add_subsystem('elastic',   RotorElasticity(modeling_options = modeling_options, opt_options = opt_options))
        self.add_subsystem('xf',        RunXFOIL(modeling_options = modeling_options, opt_options = opt_options)) # Recompute polars with xfoil (for flaps)
        
        if modeling_options['Analysis_Flags']['ServoSE']:
            self.add_subsystem('sse',       ServoSE(modeling_options = modeling_options)) # Aero analysis
            
        self.add_subsystem('stall_check', NoStallConstraint(modeling_options = modeling_options))
    
        if modeling_options['Analysis_Flags']['OpenFAST']:
            self.add_subsystem('modes_elastodyn',   ModesElastoDyn(modeling_options = modeling_options))
            self.add_subsystem('freq_rotor',        RotorLoadsDeflStrainsWEIS(modeling_options = modeling_options, opt_options = opt_options, freq_run=True))
            #if modeling_options['tower']['run_towerse']:
            self.add_subsystem('freq_tower',        TowerSE(modeling_options=modeling_options))
            self.add_subsystem('sse_tune',          ServoSE_ROSCO(modeling_options = modeling_options)) # Aero analysis
            self.add_subsystem('aeroelastic',       FASTLoadCases(modeling_options = modeling_options, opt_options = opt_options))

        self.add_subsystem('rlds',      RotorLoadsDeflStrainsWEIS(modeling_options = modeling_options, opt_options = opt_options, freq_run=False))
        
        if modeling_options['Analysis_Flags']['DriveSE']:
            self.add_subsystem('drivese',   DriveSE(debug=False,
                                                number_of_main_bearings=1,
                                                topLevelFlag=False))
                                                
        if modeling_options['Analysis_Flags']['TowerSE']:
            self.add_subsystem('towerse',   TowerSE(modeling_options=modeling_options))
            self.add_subsystem('tcons',     TurbineConstraints(modeling_options = modeling_options))
            
        self.add_subsystem('tcc',       Turbine_CostsSE_2015(verbosity=modeling_options['general']['verbosity']))

        # Connections to blade 
        self.connect('dac_ivc.te_flap_end',             'blade.outer_shape_bem.span_end')
        self.connect('dac_ivc.te_flap_ext',             'blade.outer_shape_bem.span_ext')

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
        self.connect('blade.outer_shape_bem.ref_axis',  'ccblade.precurve', src_indices=om.slicer[:, 0])
        self.connect('blade.outer_shape_bem.ref_axis',  'ccblade.precurveTip', src_indices=[(-1, 0)])
        self.connect('blade.outer_shape_bem.ref_axis',  'ccblade.presweep', src_indices=om.slicer[:, 1])
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
        self.connect('blade.pa.chord_param',           ['xf.chord', 'elastic.chord'])
        if not modeling_options['Analysis_Flags']['OpenFAST'] or modeling_options['openfast']['analysis_settings']['Analysis_Level'] == 1:
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
            if not modeling_options['Analysis_Flags']['OpenFAST']:
                self.connect('sse.gust.V_gust',              ['rlds.aero_gust.V_load', 'rlds.aero_hub_loads.V_load'])
                self.connect('sse.powercurve.rated_Omega',   ['rlds.Omega_load', 'rlds.tot_loads_gust.aeroloads_Omega', 'rlds.constr.rated_Omega'])
                self.connect('sse.powercurve.rated_pitch',   ['rlds.pitch_load', 'rlds.tot_loads_gust.aeroloads_pitch'])
        
        # Connections to run xfoil for te flaps
        self.connect('blade.outer_shape_bem.s',               'xf.s')
        self.connect('blade.interp_airfoils.coord_xy_interp', 'xf.coord_xy_interp')
        self.connect('airfoils.aoa',                          'xf.aoa')
        self.connect('assembly.r_blade',                      'xf.r')
        self.connect('dac_ivc.te_flap_end',                   'xf.span_end')
        self.connect('dac_ivc.te_flap_ext',                   'xf.span_ext')
        self.connect('dac_ivc.chord_start',                   'xf.chord_start')
        self.connect('dac_ivc.delta_max_pos',                 'xf.delta_max_pos')
        self.connect('dac_ivc.delta_max_neg',                 'xf.delta_max_neg')
        self.connect('env.speed_sound_air',                   'xf.speed_sound_air')
        self.connect('env.rho_air',                           'xf.rho_air')
        self.connect('env.mu_air',                            'xf.mu_air')
        self.connect('pc.tsr_opt',                            'xf.rated_TSR')
        if modeling_options['flags']['control']:
            self.connect('control.max_TS',                        'xf.max_TS')
        self.connect('blade.interp_airfoils.cl_interp',       'xf.cl_interp')
        self.connect('blade.interp_airfoils.cd_interp',       'xf.cd_interp')
        self.connect('blade.interp_airfoils.cm_interp',       'xf.cm_interp')

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
            self.connect('blade.outer_shape_bem.ref_axis', 'sse.precurve', src_indices=om.slicer[:, 0])
            self.connect('blade.outer_shape_bem.ref_axis', 'sse.precurveTip', src_indices=[(-1, 0)])
            self.connect('blade.outer_shape_bem.ref_axis', 'sse.presweep', src_indices=om.slicer[:, 1])
            self.connect('blade.outer_shape_bem.ref_axis', 'sse.presweepTip', src_indices=[(-1, 1)])
            self.connect('airfoils.aoa',                   'sse.airfoils_aoa')
            self.connect('airfoils.Re',                    'sse.airfoils_Re')
            self.connect('xf.cl_interp_flaps',             'sse.airfoils_cl')
            self.connect('xf.cd_interp_flaps',             'sse.airfoils_cd')
            self.connect('xf.cm_interp_flaps',             'sse.airfoils_cm')
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
        self.connect('xf.cl_interp_flaps',             'stall_check.airfoils_cl')
        self.connect('xf.cd_interp_flaps',             'stall_check.airfoils_cd')
        self.connect('xf.cm_interp_flaps',             'stall_check.airfoils_cm')
        if modeling_options['Analysis_Flags']['ServoSE']:
            self.connect('sse.powercurve.aoa_regII',   'stall_check.aoa_along_span')
        else:
            self.connect('ccblade.alpha',  'stall_check.aoa_along_span')

        if modeling_options['Analysis_Flags']['OpenFAST'] and modeling_options['Analysis_Flags']['ServoSE']:
            self.connect('sse.powercurve.rated_V',         ['sse_tune.tune_rosco.v_rated'])
            self.connect('sse.gust.V_gust',                ['freq_rotor.aero_gust.V_load', 'freq_rotor.aero_hub_loads.V_load'])
            self.connect('sse.powercurve.rated_Omega',     ['freq_rotor.Omega_load', 'freq_rotor.tot_loads_gust.aeroloads_Omega', 'freq_rotor.constr.rated_Omega', 'sse_tune.tune_rosco.rated_rotor_speed'])
            self.connect('sse.powercurve.rated_pitch',     ['freq_rotor.pitch_load', 'freq_rotor.tot_loads_gust.aeroloads_pitch'])
            self.connect('sse.powercurve.rated_Q',          'sse_tune.tune_rosco.rated_torque')

            self.connect('blade.ps.s_opt_spar_cap_ss',      'freq_rotor.constr.s_opt_spar_cap_ss')
            self.connect('blade.ps.s_opt_spar_cap_ps',      'freq_rotor.constr.s_opt_spar_cap_ps')

            # Stiffen up the terms modeled by frame3dd and not by ElastoDyn, namely EA, GJ, and EIxy
            self.connect('elastic.EA',                      'modes_elastodyn.EA')
            self.connect('elastic.GJ',                      'modes_elastodyn.GJ')
            self.connect('elastic.EIxy',                    'modes_elastodyn.EIxy')
            self.connect('materials.G',                     'modes_elastodyn.G')

            self.connect('modes_elastodyn.EA_stiff',        'freq_rotor.EA')
            self.connect('modes_elastodyn.GJ_stiff',        'freq_rotor.GJ')
            self.connect('modes_elastodyn.EIxy_zero',       'freq_rotor.EIxy')
            self.connect('elastic.A',                       'freq_rotor.A')

            self.connect('elastic.EIxx',                    'freq_rotor.EIxx')
            self.connect('elastic.EIyy',                    'freq_rotor.EIyy')

            self.connect('elastic.rhoA',                    'freq_rotor.rhoA')
            self.connect('elastic.rhoJ',                    'freq_rotor.rhoJ')
            self.connect('elastic.x_ec',                    'freq_rotor.x_ec')
            self.connect('elastic.y_ec',                    'freq_rotor.y_ec')
            self.connect('elastic.precomp.xu_strain_spar',  'freq_rotor.xu_strain_spar')
            self.connect('elastic.precomp.xl_strain_spar',  'freq_rotor.xl_strain_spar')
            self.connect('elastic.precomp.yu_strain_spar',  'freq_rotor.yu_strain_spar')
            self.connect('elastic.precomp.yl_strain_spar',  'freq_rotor.yl_strain_spar')
            self.connect('elastic.precomp.xu_strain_te',    'freq_rotor.xu_strain_te')
            self.connect('elastic.precomp.xl_strain_te',    'freq_rotor.xl_strain_te')
            self.connect('elastic.precomp.yu_strain_te',    'freq_rotor.yu_strain_te')
            self.connect('elastic.precomp.yl_strain_te',    'freq_rotor.yl_strain_te')
            self.connect('blade.outer_shape_bem.s',         'freq_rotor.constr.s')

            self.connect('drivese.top_F',                   'freq_tower.pre.rna_F')
            self.connect('drivese.top_M',                   'freq_tower.pre.rna_M')
            self.connect('drivese.rna_I_TT',                'freq_tower.rna_I')
            self.connect('drivese.rna_cm',                  'freq_tower.rna_cg')
            self.connect('drivese.rna_mass',                'freq_tower.rna_mass')
            self.connect('sse.gust.V_gust',                 'freq_tower.wind.Uref')
            self.connect('assembly.hub_height',             'freq_tower.wind_reference_height')  # TODO- environment
            self.connect('foundation.height',               'freq_tower.wind_z0') # TODO- environment
            self.connect('env.rho_air',                     'freq_tower.rho_air')
            self.connect('env.mu_air',                      'freq_tower.mu_air')                    
            self.connect('env.shear_exp',                   'freq_tower.shearExp')                    
            self.connect('assembly.hub_height',             'freq_tower.hub_height')
            self.connect('foundation.height',               'freq_tower.foundation_height')
            self.connect('tower.diameter',                  'freq_tower.tower_outer_diameter_in')
            self.connect('tower.height',                    'freq_tower.tower_height')
            self.connect('tower.s',                         'freq_tower.tower_s')
            self.connect('tower.layer_thickness',           'freq_tower.tower_layer_thickness')
            self.connect('tower.outfitting_factor',         'freq_tower.tower_outfitting_factor')
            self.connect('tower.layer_mat',                 'freq_tower.tower_layer_materials')
            self.connect('materials.name',                  'freq_tower.material_names')
            self.connect('materials.E',                     'freq_tower.E_mat')
            self.connect('modes_elastodyn.G_stiff',         'freq_tower.G_mat')
            self.connect('materials.rho',                   'freq_tower.rho_mat')
            self.connect('materials.sigma_y',               'freq_tower.sigma_y_mat')
            self.connect('materials.unit_cost',             'freq_tower.unit_cost_mat')
            
            if modeling_options['flags']['monopile']:
                self.connect('env.rho_water',                    'freq_tower.rho_water')
                self.connect('env.mu_water',                     'freq_tower.mu_water')                    
                self.connect('env.hsig_wave',                    'freq_tower.hsig_wave')                    
                self.connect('env.Tsig_wave',                    'freq_tower.Tsig_wave')                    
                self.connect('env.G_soil',                       'freq_tower.G_soil')                   
                self.connect('env.nu_soil',                      'freq_tower.nu_soil')                    
                self.connect('monopile.diameter',                'freq_tower.monopile_outer_diameter_in')
                self.connect('monopile.height',                  'freq_tower.monopile_height')
                self.connect('monopile.s',                       'freq_tower.monopile_s')
                self.connect('monopile.layer_thickness',         'freq_tower.monopile_layer_thickness')
                self.connect('monopile.layer_mat',               'freq_tower.monopile_layer_materials')
                self.connect('monopile.outfitting_factor',       'freq_tower.monopile_outfitting_factor')
                self.connect('monopile.transition_piece_height', 'freq_tower.transition_piece_height')
                self.connect('monopile.transition_piece_mass',   'freq_tower.transition_piece_mass')
                self.connect('monopile.gravity_foundation_mass', 'freq_tower.gravity_foundation_mass')
                self.connect('monopile.suctionpile_depth',       'freq_tower.suctionpile_depth')
                self.connect('monopile.suctionpile_depth_diam_ratio', 'freq_tower.suctionpile_depth_diam_ratio')

            self.connect('assembly.r_blade',               ['freq_rotor.r',            'sse_tune.r'])
            self.connect('assembly.rotor_radius',          ['freq_rotor.Rtip',         'sse_tune.Rtip'])
            self.connect('hub.radius',                     ['freq_rotor.Rhub',         'sse_tune.Rhub'])
            self.connect('assembly.hub_height',            ['freq_rotor.hub_height',   'sse_tune.hub_height'])
            self.connect('hub.cone',                       ['freq_rotor.precone',      'sse_tune.precone'])
            self.connect('nacelle.uptilt',                 ['freq_rotor.tilt',         'sse_tune.tilt'])
            self.connect('airfoils.aoa',                   ['freq_rotor.airfoils_aoa', 'sse_tune.airfoils_aoa'])
            self.connect('airfoils.Re',                    ['freq_rotor.airfoils_Re',  'sse_tune.airfoils_Re'])
            self.connect('xf.cl_interp_flaps',             ['freq_rotor.airfoils_cl',  'sse_tune.airfoils_cl'])
            self.connect('xf.cd_interp_flaps',             ['freq_rotor.airfoils_cd',  'sse_tune.airfoils_cd'])
            self.connect('xf.cm_interp_flaps',             ['freq_rotor.airfoils_cm',  'sse_tune.airfoils_cm'])
            self.connect('configuration.n_blades',         ['freq_rotor.nBlades',      'sse_tune.nBlades'])
            self.connect('env.rho_air',                    ['freq_rotor.rho',          'sse_tune.rho'])
            self.connect('env.mu_air',                     ['freq_rotor.mu',           'sse_tune.mu'])
            self.connect('blade.pa.chord_param',           ['freq_rotor.chord',        'sse_tune.chord'])
            self.connect('blade.pa.twist_param',           ['freq_rotor.theta',        'sse_tune.theta'])
            self.connect('env.shear_exp',                   'freq_rotor.aero_hub_loads.shearExp')

            self.connect('control.V_in' ,                   'sse_tune.v_min')
            self.connect('control.V_out' ,                  'sse_tune.v_max')
            self.connect('blade.outer_shape_bem.ref_axis',  'sse_tune.precurve', src_indices=om.slicer[:, 0])
            self.connect('blade.outer_shape_bem.ref_axis',  'sse_tune.precurveTip', src_indices=[(-1, 0)])
            self.connect('blade.outer_shape_bem.ref_axis',  'sse_tune.presweep', src_indices=om.slicer[:, 1])
            self.connect('blade.outer_shape_bem.ref_axis',  'sse_tune.presweepTip', src_indices=[(-1, 1)])
            self.connect('xf.flap_angles',                  'sse_tune.airfoils_Ctrl')
            self.connect('control.minOmega',                'sse_tune.omega_min')
            self.connect('pc.tsr_opt',                      'sse_tune.tsr_operational')
            self.connect('control.rated_power',             'sse_tune.rated_power')

            self.connect('nacelle.gear_ratio',              'sse_tune.tune_rosco.gear_ratio')
            self.connect('assembly.rotor_radius',           'sse_tune.tune_rosco.R')
            self.connect('elastic.precomp.I_all_blades',    'sse_tune.tune_rosco.rotor_inertia', src_indices=[0])
            self.connect('freq_rotor.frame.flap_mode_freqs','sse_tune.tune_rosco.flap_freq', src_indices=[0])
            self.connect('freq_rotor.frame.edge_mode_freqs','sse_tune.tune_rosco.edge_freq', src_indices=[0])
            self.connect('nacelle.generator_efficiency',    'sse_tune.tune_rosco.generator_efficiency')
            self.connect('nacelle.gearbox_efficiency',      'sse_tune.tune_rosco.gearbox_efficiency')
            self.connect('tune_rosco_ivc.max_pitch',        'sse_tune.tune_rosco.max_pitch') 
            self.connect('tune_rosco_ivc.min_pitch',        'sse_tune.tune_rosco.min_pitch')
            self.connect('control.max_pitch_rate' ,         'sse_tune.tune_rosco.max_pitch_rate')
            self.connect('control.max_torque_rate' ,        'sse_tune.tune_rosco.max_torque_rate')
            self.connect('tune_rosco_ivc.vs_minspd',        'sse_tune.tune_rosco.vs_minspd') 
            self.connect('tune_rosco_ivc.ss_vsgain',        'sse_tune.tune_rosco.ss_vsgain') 
            self.connect('tune_rosco_ivc.ss_pcgain',        'sse_tune.tune_rosco.ss_pcgain') 
            self.connect('tune_rosco_ivc.ps_percent',       'sse_tune.tune_rosco.ps_percent') 
            self.connect('tune_rosco_ivc.PC_omega',         'sse_tune.tune_rosco.PC_omega')
            self.connect('tune_rosco_ivc.PC_zeta',          'sse_tune.tune_rosco.PC_zeta')
            self.connect('tune_rosco_ivc.VS_omega',         'sse_tune.tune_rosco.VS_omega')
            self.connect('tune_rosco_ivc.VS_zeta',          'sse_tune.tune_rosco.VS_zeta')
            self.connect('dac_ivc.delta_max_pos',           'sse_tune.tune_rosco.delta_max_pos')
            if modeling_options['servose']['Flp_Mode'] > 0:
                self.connect('tune_rosco_ivc.Flp_omega',    'sse_tune.tune_rosco.Flp_omega')
                self.connect('tune_rosco_ivc.Flp_zeta',     'sse_tune.tune_rosco.Flp_zeta')
                
        elif modeling_options['Analysis_Flags']['OpenFAST']==True and modeling_options['Analysis_Flags']['ServoSE']==False:
            exit("ERROR: WISDEM does not support openfast without the tuning of ROSCO")
        else:
            pass

        # Connections to rotor load analysis
        if modeling_options['Analysis_Flags']['OpenFAST'] and modeling_options['openfast']['analysis_settings']['Analysis_Level'] == 2:
            self.connect('aeroelastic.loads_Px',      'rlds.tot_loads_gust.aeroloads_Px')
            self.connect('aeroelastic.loads_Py',      'rlds.tot_loads_gust.aeroloads_Py')
            self.connect('aeroelastic.loads_Pz',      'rlds.tot_loads_gust.aeroloads_Pz')
            self.connect('aeroelastic.loads_Omega',   'rlds.tot_loads_gust.aeroloads_Omega')
            self.connect('aeroelastic.loads_pitch',   'rlds.tot_loads_gust.aeroloads_pitch')
            self.connect('aeroelastic.loads_azimuth', 'rlds.tot_loads_gust.aeroloads_azimuth')
        else:
            self.connect('xf.cl_interp_flaps',        'rlds.airfoils_cl')
            self.connect('xf.cd_interp_flaps',        'rlds.airfoils_cd')
            self.connect('xf.cm_interp_flaps',        'rlds.airfoils_cm')
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

        
        if modeling_options['Analysis_Flags']['OpenFAST'] and modeling_options['openfast']['dlc_settings']['run_blade_fatigue']:
            self.connect('elastic.precomp.x_tc',                            'aeroelastic.x_tc')
            self.connect('elastic.precomp.y_tc',                            'aeroelastic.y_tc')
            self.connect('materials.E',                                     'aeroelastic.E')
            self.connect('materials.Xt',                                    'aeroelastic.Xt')
            self.connect('materials.Xc',                                    'aeroelastic.Xc')
            self.connect('blade.outer_shape_bem.pitch_axis',                'aeroelastic.pitch_axis')
            self.connect('elastic.sc_ss_mats',                              'aeroelastic.sc_ss_mats')
            self.connect('elastic.sc_ps_mats',                              'aeroelastic.sc_ps_mats')
            self.connect('elastic.te_ss_mats',                              'aeroelastic.te_ss_mats')
            self.connect('elastic.te_ps_mats',                              'aeroelastic.te_ps_mats')
            # self.connect('blade.interp_airfoils.r_thick_interp',            'aeroelastic.rthick')
            # self.connect('blade.internal_structure_2d_fem.layer_name',      'aeroelastic.layer_name')
            # self.connect('blade.internal_structure_2d_fem.layer_mat',       'aeroelastic.layer_mat')
            self.connect('blade.internal_structure_2d_fem.definition_layer','aeroelastic.definition_layer')
            # self.connect('gamma_m',     'rlds.gamma_m')
            # self.connect('gamma_f',     'rlds.gamma_f') # TODO

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
            if modeling_options['Analysis_Flags']['OpenFAST'] and modeling_options['openfast']['analysis_settings']['Analysis_Level'] == 2:
                self.connect('aeroelastic.Fxyz', 'drivese.Fxyz')
                self.connect('aeroelastic.Mxyz', 'drivese.Mxyz')
            else:
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
            if modeling_options['Analysis_Flags']['TowerSE']:
                self.connect('tower.diameter',             'drivese.tower_top_diameter', src_indices=[-1])

        # Connections to TowerSE
        if modeling_options['Analysis_Flags']['DriveSE'] and modeling_options['Analysis_Flags']['TowerSE']:
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
          
        # Connections to aeroelasticse
        if modeling_options['Analysis_Flags']['OpenFAST']:
            self.connect('blade.outer_shape_bem.ref_axis',  'aeroelastic.ref_axis_blade')
            self.connect('configuration.rotor_orientation', 'aeroelastic.rotor_orientation')
            self.connect('assembly.r_blade',                'aeroelastic.r')
            self.connect('blade.outer_shape_bem.pitch_axis','aeroelastic.le_location')
            self.connect('blade.pa.chord_param',            'aeroelastic.chord')
            self.connect('blade.pa.twist_param',            'aeroelastic.theta')
            self.connect('blade.interp_airfoils.coord_xy_interp', 'aeroelastic.coord_xy_interp')
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
            self.connect('nacelle.gear_ratio',              'aeroelastic.gearbox_ratio')
            self.connect('nacelle.gearbox_efficiency',      'aeroelastic.gearbox_efficiency')
            self.connect('nacelle.generator_efficiency',    'aeroelastic.generator_efficiency')

            #if modeling_options['Analysis_Flags']['TowerSE']:
            self.connect('freq_tower.post.mass_den',           'aeroelastic.mass_den')
            self.connect('freq_tower.post.foreaft_stff',       'aeroelastic.foreaft_stff')
            self.connect('freq_tower.post.sideside_stff',      'aeroelastic.sideside_stff')
            self.connect('freq_tower.post.sec_loc',            'aeroelastic.sec_loc')
            self.connect('freq_tower.post.fore_aft_modes',     'aeroelastic.fore_aft_modes')
            self.connect('freq_tower.post.side_side_modes',    'aeroelastic.side_side_modes')
            self.connect('freq_tower.tower_section_height',    'aeroelastic.tower_section_height')
            self.connect('freq_tower.tower_outer_diameter',    'aeroelastic.tower_outer_diameter')

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
            self.connect('elastic.rhoA',                    'aeroelastic.beam:rhoA')
            self.connect('elastic.EIxx',                    'aeroelastic.beam:EIxx')
            self.connect('elastic.EIyy',                    'aeroelastic.beam:EIyy')
            self.connect('elastic.Tw_iner',                 'aeroelastic.beam:Tw_iner')
            self.connect('freq_rotor.frame.flap_mode_shapes', 'aeroelastic.flap_mode_shapes')
            self.connect('freq_rotor.frame.edge_mode_shapes', 'aeroelastic.edge_mode_shapes')
            self.connect('sse.powercurve.V',                'aeroelastic.U_init')
            self.connect('sse.powercurve.Omega',            'aeroelastic.Omega_init')
            self.connect('sse.powercurve.pitch',            'aeroelastic.pitch_init')
            self.connect('sse.powercurve.V_R25',            'aeroelastic.V_R25')
            self.connect('sse.powercurve.rated_V',          'aeroelastic.Vrated')
            self.connect('sse.gust.V_gust',                 'aeroelastic.Vgust')
            self.connect('wt_class.V_extreme1',             'aeroelastic.V_extreme1')
            self.connect('wt_class.V_extreme50',            'aeroelastic.V_extreme50')
            self.connect('wt_class.V_mean',                 'aeroelastic.V_mean_iec')
            self.connect('control.V_out',                   'aeroelastic.V_cutout')
            self.connect('control.rated_power',             'aeroelastic.control_ratedPower')
            self.connect('control.max_TS',                  'aeroelastic.control_maxTS')
            self.connect('control.maxOmega',                'aeroelastic.control_maxOmega')
            self.connect('configuration.turb_class',        'aeroelastic.turbulence_class')
            self.connect('configuration.ws_class' ,         'aeroelastic.turbine_class')
            self.connect('sse_tune.aeroperf_tables.pitch_vector','aeroelastic.pitch_vector')
            self.connect('sse_tune.aeroperf_tables.tsr_vector', 'aeroelastic.tsr_vector')
            self.connect('sse_tune.aeroperf_tables.U_vector', 'aeroelastic.U_vector')
            self.connect('sse_tune.aeroperf_tables.Cp',     'aeroelastic.Cp_aero_table')
            self.connect('sse_tune.aeroperf_tables.Ct',     'aeroelastic.Ct_aero_table')
            self.connect('sse_tune.aeroperf_tables.Cq',     'aeroelastic.Cq_aero_table')

            # Temporary
            self.connect('xf.Re_loc',                       'aeroelastic.airfoils_Re_loc')
            self.connect('xf.Ma_loc',                       'aeroelastic.airfoils_Ma_loc')
            self.connect('xf.flap_angles',                  'aeroelastic.airfoils_Ctrl')
        
        # Connections to turbine constraints
        if modeling_options['Analysis_Flags']['TowerSE']:
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

        if modeling_options['Analysis_Flags']['TowerSE']:
            self.connect('towerse.tower_mass',          'tcc.tower_mass')

class WindPark(om.Group):
    # Openmdao group to run the cost analysis of a wind park
    
    def initialize(self):
        self.options.declare('modeling_options')
        self.options.declare('opt_options')
        
    def setup(self):
        modeling_options = self.options['modeling_options']
        opt_options     = self.options['opt_options']

        self.add_subsystem('wt',        WT_RNTA(modeling_options = modeling_options, opt_options = opt_options), promotes=['*'])
        
        if modeling_options['Analysis_Flags']['BOS']:
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
        if modeling_options['Analysis_Flags']['BOS']:
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
                
                if modeling_options['Analysis_Flags']['OpenFAST'] and modeling_options['openfast']['analysis_settings']['Analysis_Level'] == 2:
                    self.connect('aeroelastic.rated_T',          'landbosse.rated_thrust_N')
                elif modeling_options['Analysis_Flags']['ServoSE']:
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
        if modeling_options['Analysis_Flags']['OpenFAST'] and modeling_options['openfast']['dlc_settings']['run_power_curve'] and modeling_options['openfast']['analysis_settings']['Analysis_Level'] == 2:
            self.connect('aeroelastic.AEP',     'financese.turbine_aep')
        elif modeling_options['Analysis_Flags']['ServoSE']:
            self.connect('sse.AEP',             'financese.turbine_aep')

        self.connect('tcc.turbine_cost_kW',     'financese.tcc_per_kW')
        if modeling_options['Analysis_Flags']['BOS']:
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
            if modeling_options['Analysis_Flags']['OpenFAST'] and modeling_options['openfast']['dlc_settings']['run_power_curve'] and modeling_options['openfast']['analysis_settings']['Analysis_Level'] == 2:
                self.connect('aeroelastic.AEP',     'outputs_2_screen.aep')
            else:
                self.connect('sse.AEP',             'outputs_2_screen.aep')
            self.connect('financese.lcoe',          'outputs_2_screen.lcoe')
            
        self.connect('elastic.precomp.blade_mass',  'outputs_2_screen.blade_mass')
        self.connect('rlds.tip_pos.tip_deflection', 'outputs_2_screen.tip_deflection')
        
        if modeling_options['Analysis_Flags']['OpenFAST'] and modeling_options['openfast']['analysis_settings']['Analysis_Level'] == 2:
            self.connect('aeroelastic.My_std',      'outputs_2_screen.My_std')
            self.connect('aeroelastic.flp1_std',    'outputs_2_screen.flp1_std')
            self.connect('tune_rosco_ivc.PC_omega',        'outputs_2_screen.PC_omega')
            self.connect('tune_rosco_ivc.PC_zeta',         'outputs_2_screen.PC_zeta')
            self.connect('tune_rosco_ivc.VS_omega',        'outputs_2_screen.VS_omega')
            self.connect('tune_rosco_ivc.VS_zeta',         'outputs_2_screen.VS_zeta')
            self.connect('tune_rosco_ivc.Flp_omega',       'outputs_2_screen.Flp_omega')
            self.connect('tune_rosco_ivc.Flp_zeta',        'outputs_2_screen.Flp_zeta')
