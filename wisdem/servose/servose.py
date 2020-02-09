'''
Controller tuning script.

Nikhar J. Abbas
January 2020
'''

from __future__ import print_function
import numpy as np
import datetime

from ROSCO_toolbox import controller as ROSCO_controller
from ROSCO_toolbox import turbine as ROSCO_turbine
from ROSCO_toolbox import sim as ROSCO_sim
from ROSCO_toolbox import utilities as ROSCO_utilities

import numpy as np
import os
from openmdao.api import IndepVarComp, ExplicitComponent, Group, Problem
from scipy import interpolate, gradient
from scipy.optimize import minimize_scalar, minimize
from scipy.interpolate import PchipInterpolator

from wisdem.ccblade import CCAirfoil, CCBlade
from wisdem.commonse.distribution import RayleighCDF, WeibullWithMeanCDF
from wisdem.commonse.utilities import vstack, trapz_deriv, linspace_with_deriv, smooth_min, smooth_abs
from wisdem.commonse.environment import PowerWind
from wisdem.rotorse.rotor_fast import eval_unsteady

class ServoSE(Group):
    def initialize(self):
        self.options.declare('analysis_options')
        self.options.declare('opt_options')

    def setup(self):
        analysis_options = self.options['analysis_options']

        self.add_subsystem('powercurve',        RegulatedPowerCurve(analysis_options   = analysis_options), promotes = ['v_min', 'v_max','rated_power','omega_min','omega_max', 'control_maxTS','tsr_operational','control_pitch','drivetrainType','drivetrainEff','r','chord', 'theta','Rhub', 'Rtip', 'hub_height','precone', 'tilt','yaw','precurve','precurveTip','presweep','presweepTip', 'airfoils_aoa','airfoils_Re','airfoils_cl','airfoils_cd','airfoils_cm', 'nBlades', 'rho', 'mu'])
        self.add_subsystem('stall_check',       NoStallConstraint(analysis_options   = analysis_options), promotes = ['airfoils_aoa','airfoils_cl','airfoils_cd','airfoils_cm'])
        self.add_subsystem('cdf',               WeibullWithMeanCDF(nspline=analysis_options['servose']['n_pc_spline']))
        self.add_subsystem('aep',               AEP(), promotes=['AEP'])
        if analysis_options['openfast']['run_openfast'] == True:
            self.add_subsystem('aeroperf_tables',   Cp_Ct_Cq_Tables(analysis_options   = analysis_options), promotes = ['v_min', 'v_max','r','chord', 'theta','Rhub', 'Rtip', 'hub_height','precone', 'tilt','yaw','precurve','precurveTip','presweep','presweepTip', 'airfoils_aoa','airfoils_Re','airfoils_cl','airfoils_cd','airfoils_cm', 'nBlades', 'rho', 'mu'])
            self.add_subsystem('tune_rosco',        TuneROSCO(analysis_options = analysis_options), promotes = ['v_min', 'v_max', 'rho', 'omega_min', 'tsr_operational', 'rated_power'])
        # Connections to the stall check
        self.connect('powercurve.aoa_cutin','stall_check.aoa_along_span')

        # Connections to the Weibull CDF
        self.connect('powercurve.V_spline', 'cdf.x')
        
        # Connections to the aep computation component
        self.connect('cdf.F',               'aep.CDF_V')
        self.connect('powercurve.P_spline', 'aep.P')   

        if analysis_options['openfast']['run_openfast'] == True:
            # Connect ROSCO Power curve
            self.connect('powercurve.rated_V',      'tune_rosco.v_rated')
            self.connect('powercurve.rated_Omega',  'tune_rosco.rated_rotor_speed')
            self.connect('powercurve.rated_Q',      'tune_rosco.rated_torque')

            # Connect ROSCO for Rotor Performance tables
            self.connect('aeroperf_tables.Cp',              'tune_rosco.Cp_table')
            self.connect('aeroperf_tables.Ct',              'tune_rosco.Ct_table')
            self.connect('aeroperf_tables.Cq',              'tune_rosco.Cq_table')
            self.connect('aeroperf_tables.pitch_vector',    'tune_rosco.pitch_vector')
            self.connect('aeroperf_tables.tsr_vector',      'tune_rosco.tsr_vector')
            self.connect('aeroperf_tables.U_vector',        'tune_rosco.U_vector')

class TuneROSCO(ExplicitComponent):
    def initialize(self):
        self.options.declare('analysis_options')

    def setup(self):
        self.analysis_options = self.options['analysis_options']
        servose_init_options = self.analysis_options['servose']

        # Input parameters
        self.controller_params = {}
        # Controller Flags
        self.controller_params['LoggingLevel'] = self.analysis_options['servose']['LoggingLevel']
        self.controller_params['F_LPFType'] = self.analysis_options['servose']['F_LPFType']
        self.controller_params['F_NotchType'] = self.analysis_options['servose']['F_NotchType']
        self.controller_params['IPC_ControlMode'] = self.analysis_options['servose']['IPC_ControlMode']
        self.controller_params['VS_ControlMode'] = self.analysis_options['servose']['VS_ControlMode']
        self.controller_params['PC_ControlMode'] = self.analysis_options['servose']['PC_ControlMode']
        self.controller_params['Y_ControlMode'] = self.analysis_options['servose']['Y_ControlMode']
        self.controller_params['SS_Mode'] = self.analysis_options['servose']['SS_Mode']
        self.controller_params['WE_Mode'] = self.analysis_options['servose']['WE_Mode']
        self.controller_params['PS_Mode'] = self.analysis_options['servose']['PS_Mode']
        self.controller_params['SD_Mode'] = self.analysis_options['servose']['SD_Mode']
        self.controller_params['Fl_Mode'] = self.analysis_options['servose']['Fl_Mode']
        self.controller_params['Flp_Mode'] = self.analysis_options['servose']['Flp_Mode']

        # Necessary parameters
        # Turbine parameters
        self.add_input('rotor_inertia',     val=0.0,        units='kg*m**2',        desc='Rotor inertia')
        self.add_input('rho',               val=0.0,        units='kg/m**3',        desc='Air Density')
        self.add_input('R',                 val=0.0,        units='m',              desc='Rotor Radius')              
        self.add_input('gear_ratio',        val=0.0,                                desc='Gearbox Ratio')        
        self.add_input('rated_rotor_speed', val=0.0,        units='rad/s',          desc='Rated rotor speed')                    
        self.add_input('rated_power',       val=0.0,        units='W',              desc='Rated power')            
        self.add_input('rated_torque',     val=0.0,                units='N*m', desc='rotor aerodynamic torque at rated')        
        self.add_input('v_rated',           val=0.0,        units='m/s',            desc='Rated wind speed')
        self.add_input('v_min',             val=0.0,        units='m/s',            desc='Minimum wind speed (cut-in)')
        self.add_input('v_max',             val=0.0,        units='m/s',            desc='Maximum wind speed (cut-out)')
        self.add_input('max_pitch_rate',    val=0.0,        units='rad/s',          desc='Maximum allowed blade pitch rate')
        self.add_input('max_torque_rate',   val=0.0,        units='N*m/s',          desc='Maximum allowed generator torque rate')
        self.add_input('tsr_operational',   val=0.0,                                desc='Operational tip-speed ratio')
        self.add_input('omega_min',         val=0.0,        units='rad/s',          desc='Minimum rotor speed')
        self.add_input('flap_freq',         val=0.0,        units='Hz',             desc='Blade flapwise first natural frequency') 
        self.add_input('edge_freq',         val=0.0,        units='Hz',             desc='Blade edgewise first natural frequency')
        self.add_input('gen_eff',           val=0.0,                                desc='Drivetrain efficiency')
        # 
        self.add_input('max_pitch',         val=0.0,        units='rad',            desc='')
        self.add_input('min_pitch',         val=0.0,        units='rad',            desc='')
        self.add_input('vs_minspd',         val=0.0,        units='rad/s',          desc='') 
        self.add_input('ss_vsgain',         val=0.0,                                desc='')
        self.add_input('ss_pcgain',         val=0.0,                                desc='')
        self.add_input('ps_percent',        val=0.0,                                desc='')
        # Rotor Power
        self.n_pitch    = n_pitch   = servose_init_options['n_pitch_perf_surfaces']
        self.n_tsr      = n_tsr     = servose_init_options['n_tsr_perf_surfaces']
        self.n_U        = n_U       = servose_init_options['n_U_perf_surfaces']
        self.add_input('Cp_table',          val=np.zeros((n_tsr, n_pitch, n_U)),                desc='table of aero power coefficient')
        self.add_input('Ct_table',          val=np.zeros((n_tsr, n_pitch, n_U)),                desc='table of aero thrust coefficient')
        self.add_input('Cq_table',          val=np.zeros((n_tsr, n_pitch, n_U)),                desc='table of aero torque coefficient')
        self.add_input('pitch_vector',      val=np.zeros(n_pitch),              units='rad',    desc='Pitch vector used')
        self.add_input('tsr_vector',        val=np.zeros(n_tsr),                                desc='TSR vector used')
        self.add_input('U_vector',          val=np.zeros(n_U),                  units='m/s',    desc='Wind speed vector used')

        # Controller Parameters
        self.add_input('PC_zeta',           val=0.0,                                            desc='Pitch controller damping ratio')
        self.add_input('PC_omega',          val=0.0,        units='rad/s',                      desc='Pitch controller natural frequency')
        self.add_input('VS_zeta',           val=0.0,                                            desc='Generator torque controller damping ratio')
        self.add_input('VS_omega',          val=0.0,        units='rad/s',                      desc='Generator torque controller natural frequency')
        self.add_input('Kp_flap',           val=0.0,        units='s',                          desc='Flap actuation gain') 
        self.add_input('Ki_flap',           val=0.0,                                            desc='Flap actuation gain') 

    def compute(self,inputs,outputs):
        '''
        Call ROSCO toolbox to define controller
        '''

        # Add control tuning parameters to dictionary
        self.analysis_options['servose']['omega_pc']  = inputs['PC_omega']
        self.analysis_options['servose']['zeta_pc']   = inputs['PC_zeta']
        self.analysis_options['servose']['omega_vs']  = inputs['VS_omega']
        self.analysis_options['servose']['zeta_vs']   = inputs['VS_zeta']
        #
        self.analysis_options['servose']['max_pitch']   = inputs['max_pitch'][0]
        self.analysis_options['servose']['min_pitch']   = inputs['min_pitch'][0]
        self.analysis_options['servose']['vs_minspd']   = inputs['vs_minspd'][0]
        self.analysis_options['servose']['ss_vsgain']   = inputs['ss_vsgain'][0]
        self.analysis_options['servose']['ss_pcgain']   = inputs['ss_pcgain'][0]
        self.analysis_options['servose']['ps_percent']  = inputs['ps_percent'][0]
        #
        self.analysis_options['servose']['ss_cornerfreq']   = None
        self.analysis_options['servose']['sd_maxpit']       = None
        self.analysis_options['servose']['sd_cornerfreq']   = None

        # Define necessary turbine parameters
        WISDEM_turbine = type('', (), {})()
        WISDEM_turbine.v_min = inputs['v_min'][0]
        WISDEM_turbine.J = inputs['rotor_inertia'][0]
        WISDEM_turbine.rho = inputs['rho'][0]
        WISDEM_turbine.rotor_radius = inputs['R'][0]
        WISDEM_turbine.Ng = inputs['gear_ratio'][0]
        WISDEM_turbine.gen_eff = inputs['gen_eff'][0]
        WISDEM_turbine.rated_rotor_speed = inputs['rated_rotor_speed'][0]
        WISDEM_turbine.rated_power = inputs['rated_power'][0]
        WISDEM_turbine.rated_torque = inputs['rated_torque'][0] / WISDEM_turbine.Ng * WISDEM_turbine.gen_eff
        WISDEM_turbine.v_rated = inputs['v_rated'][0]
        WISDEM_turbine.v_min = inputs['v_min'][0]
        WISDEM_turbine.v_max = inputs['v_max'][0]
        WISDEM_turbine.max_pitch_rate = inputs['max_pitch_rate'][0]
        WISDEM_turbine.TSR_operational = inputs['tsr_operational'][0]
        WISDEM_turbine.max_torque_rate = inputs['max_torque_rate'][0]

        # Load Cp tables
        self.Cp_table = inputs['Cp_table']
        self.Ct_table = inputs['Ct_table']
        self.Cq_table = inputs['Cq_table']
        self.pitch_vector = WISDEM_turbine.pitch_initial_rad = inputs['pitch_vector']
        self.tsr_vector = WISDEM_turbine.TSR_initial = inputs['tsr_vector']
        self.Cp_table = WISDEM_turbine.Cp_table = self.Cp_table.reshape(len(self.pitch_vector),len(self.tsr_vector))
        self.Ct_table = WISDEM_turbine.Ct_table = self.Ct_table.reshape(len(self.pitch_vector),len(self.tsr_vector))
        self.Cq_table = WISDEM_turbine.Cq_table = self.Cq_table.reshape(len(self.pitch_vector),len(self.tsr_vector))

        RotorPerformance = ROSCO_turbine.RotorPerformance
        WISDEM_turbine.Cp = RotorPerformance(self.Cp_table,self.pitch_vector,self.tsr_vector)
        WISDEM_turbine.Ct = RotorPerformance(self.Ct_table,self.pitch_vector,self.tsr_vector)
        WISDEM_turbine.Cq = RotorPerformance(self.Cq_table,self.pitch_vector,self.tsr_vector)

        # initialize and tune controller
        self.analysis_options['servose']['Flp_Mode'] = 0 # Don't do generic tuning for flaps right now
        controller = ROSCO_controller.Controller(self.analysis_options['servose'])
        controller.tune_controller(WISDEM_turbine)
        if controller.Flp_Mode == 0:
            controller.Kp_flap = np.array([0.0]) # inputs['Kp_flap'][0]
            controller.Ki_flap = np.array([0.0]) # inputs['Ki_flap'][0]

        # DISCON Parameters
        #   - controller
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['LoggingLevel'] = controller.LoggingLevel
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['F_LPFType'] = controller.F_LPFType
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['F_NotchType'] = controller.F_NotchType
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['IPC_ControlMode'] = controller.IPC_ControlMode
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['VS_ControlMode'] = controller.VS_ControlMode
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['PC_ControlMode'] = controller.PC_ControlMode
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['Y_ControlMode'] = controller.Y_ControlMode
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['SS_Mode'] = controller.SS_Mode
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['WE_Mode'] = controller.WE_Mode
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['PS_Mode'] = controller.PS_Mode
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['SD_Mode'] = controller.SD_Mode
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['Fl_Mode'] = controller.Fl_Mode
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['Flp_Mode'] = controller.Flp_Mode
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['F_LPFDamping'] = controller.F_LPFDamping
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['F_SSCornerFreq'] = controller.ss_cornerfreq
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['PC_GS_angles'] = controller.pitch_op_pc
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['PC_GS_KP'] = controller.pc_gain_schedule.Kp
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['PC_GS_KI'] = controller.pc_gain_schedule.Ki
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['PC_MaxRat'] = controller.max_pitch
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['PC_MinRat'] = controller.min_pitch
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['VS_MinOMSpd'] = controller.vs_minspd
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['VS_Rgn2K'] = controller.vs_rgn2K
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['VS_RefSpd'] = controller.vs_refspd
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['VS_KP'] = controller.vs_gain_schedule.Kp
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['VS_KI'] = controller.vs_gain_schedule.Ki
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['SS_VSGain'] = controller.ss_vsgain
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['SS_PCGain'] = controller.ss_pcgain
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['WE_FOPoles_N'] = len(controller.v)
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['WE_FOPoles_v'] = controller.v
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['WE_FOPoles'] = controller.A
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['ps_wind_speeds'] = controller.v
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['PS_BldPitchMin'] = controller.ps_min_bld_pitch
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['SD_MaxPit'] = controller.sd_maxpit
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['SD_CornerFreq'] = controller.sd_cornerfreq
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['Fl_Kp'] = controller.Kp_float
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['Flp_Kp'] = controller.Kp_flap
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['Flp_Ki'] = controller.Ki_flap
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['Flp_Angle'] = 0.
        # - turbine
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['WE_BladeRadius'] = WISDEM_turbine.rotor_radius
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['v_rated'] = inputs['v_rated'][0]
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['F_FlpCornerFreq']  = [inputs['flap_freq'][0] * 2 * np.pi, 0.7]
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['F_LPFCornerFreq']  = inputs['edge_freq'][0] * 2 * np.pi / 4.
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['twr_freq'] = 0.0 # inputs(['twr_freq']) # zero for now, fix when floating introduced to WISDEM
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['ptfm_freq'] = 0.0 # inputs(['ptfm_freq']) # zero for now, fix when floating introduced to WISDEM
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['PC_MaxRat'] = WISDEM_turbine.max_pitch_rate
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['PC_MinRat'] = -WISDEM_turbine.max_pitch_rate
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['VS_MaxRat'] = WISDEM_turbine.max_torque_rate
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['rated_rotor_speed'] = WISDEM_turbine.rated_rotor_speed
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['VS_RtPwr'] = WISDEM_turbine.rated_power
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['VS_RtTq'] = WISDEM_turbine.rated_torque
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['VS_MaxTq'] = WISDEM_turbine.rated_torque * 1.1
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['VS_TSRopt'] = WISDEM_turbine.TSR_operational
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['WE_RhoAir'] = WISDEM_turbine.rho
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['WE_GearboxRatio'] = WISDEM_turbine.Ng
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['WE_Jtot'] = WISDEM_turbine.J
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['Cp_pitch_initial_rad'] = self.pitch_vector
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['Cp_TSR_initial'] = self.tsr_vector
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['Cp_table'] = WISDEM_turbine.Cp_table
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['Ct_table'] = WISDEM_turbine.Ct_table
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['Cq_table'] = WISDEM_turbine.Cq_table
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['Cp'] = WISDEM_turbine.Cp
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['Ct'] = WISDEM_turbine.Ct
        self.analysis_options['openfast']['fst_vt']['DISCON_in']['Cq'] = WISDEM_turbine.Cq

class RegulatedPowerCurve(ExplicitComponent): # Implicit COMPONENT

    def initialize(self):
        # self.options.declare('naero')
        # self.options.declare('n_pc')
        # self.options.declare('n_pc_spline')
        # self.options.declare('regulation_reg_II5',default=True)
        # self.options.declare('regulation_reg_III',default=False)
        # self.options.declare('lock_pitchII',default=False)

        # self.options.declare('n_aoa_grid')
        # self.options.declare('n_Re_grid')
        self.options.declare('analysis_options')
    
    def setup(self):
        analysis_options = self.options['analysis_options']
        self.n_span        = n_span    = analysis_options['blade']['n_span']
        # self.n_af          = n_af      = af_init_options['n_af'] # Number of airfoils
        self.n_aoa         = n_aoa     = analysis_options['airfoils']['n_aoa']# Number of angle of attacks
        self.n_Re          = n_Re      = analysis_options['airfoils']['n_Re'] # Number of Reynolds, so far hard set at 1
        self.n_tab         = n_tab     = analysis_options['airfoils']['n_tab']# Number of tabulated data. For distributed aerodynamic control this could be > 1
        # self.n_xy          = n_xy      = af_init_options['n_xy'] # Number of coordinate points to describe the airfoil geometry
        self.regulation_reg_III = analysis_options['servose']['regulation_reg_III']
        # naero       = self.naero = self.options['naero']
        self.n_pc          = analysis_options['servose']['n_pc']
        self.n_pc_spline   = analysis_options['servose']['n_pc_spline']
        # n_aoa_grid  = self.options['n_aoa_grid']
        # n_Re_grid   = self.options['n_Re_grid']

        # parameters
        self.add_input('v_min',        val=0.0, units='m/s',  desc='cut-in wind speed')
        self.add_input('v_max',       val=0.0, units='m/s',  desc='cut-out wind speed')
        self.add_input('rated_power', val=0.0, units='W',    desc='electrical rated power')
        self.add_input('omega_min',   val=0.0, units='rpm',  desc='minimum allowed rotor rotation speed')
        self.add_input('omega_max',   val=0.0, units='rpm',  desc='maximum allowed rotor rotation speed')
        self.add_input('control_maxTS',      val=0.0, units='m/s',  desc='maximum allowed blade tip speed')
        self.add_input('tsr_operational',        val=0.0,               desc='tip-speed ratio in Region 2 (should be optimized externally)')
        self.add_input('control_pitch',      val=0.0, units='deg',  desc='pitch angle in region 2 (and region 3 for fixed pitch machines)')
        self.add_discrete_input('drivetrainType',     val='GEARED')
        self.add_input('drivetrainEff',     val=0.0,               desc='overwrite drivetrain model with a given efficiency, used for FAST analysis')
        
        self.add_input('r',         val=np.zeros(n_span), units='m',   desc='radial locations where blade is defined (should be increasing and not go all the way to hub or tip)')
        self.add_input('chord',     val=np.zeros(n_span), units='m',   desc='chord length at each section')
        self.add_input('theta',     val=np.zeros(n_span), units='deg', desc='twist angle at each section (positive decreases angle of attack)')
        self.add_input('Rhub',      val=0.0,             units='m',   desc='hub radius')
        self.add_input('Rtip',      val=0.0,             units='m',   desc='tip radius')
        self.add_input('hub_height',val=0.0,             units='m',   desc='hub height')
        self.add_input('precone',   val=0.0,             units='deg', desc='precone angle', )
        self.add_input('tilt',      val=0.0,             units='deg', desc='shaft tilt', )
        self.add_input('yaw',       val=0.0,             units='deg', desc='yaw error', )
        self.add_input('precurve',      val=np.zeros(n_span),    units='m', desc='precurve at each section')
        self.add_input('precurveTip',   val=0.0,                units='m', desc='precurve at tip')
        self.add_input('presweep',      val=np.zeros(n_span),    units='m', desc='presweep at each section')
        self.add_input('presweepTip',   val=0.0,                units='m', desc='presweep at tip')
        
        # self.add_discrete_input('airfoils',  val=[0]*naero,                      desc='CCAirfoil instances')
        self.add_input('airfoils_cl', val=np.zeros((n_span, n_aoa, n_Re, n_tab)), desc='lift coefficients, spanwise')
        self.add_input('airfoils_cd', val=np.zeros((n_span, n_aoa, n_Re, n_tab)), desc='drag coefficients, spanwise')
        self.add_input('airfoils_cm', val=np.zeros((n_span, n_aoa, n_Re, n_tab)), desc='moment coefficients, spanwise')
        self.add_input('airfoils_aoa', val=np.zeros((n_aoa)), units='deg', desc='angle of attack grid for polars')
        self.add_input('airfoils_Re', val=np.zeros((n_Re)), desc='Reynolds numbers of polars')
        self.add_discrete_input('nBlades',         val=0,                              desc='number of blades')
        self.add_input('rho',       val=1.225,        units='kg/m**3',    desc='density of air')
        self.add_input('mu',        val=1.81e-5,      units='kg/(m*s)',   desc='dynamic viscosity of air')
        self.add_input('shearExp',  val=0.0,                            desc='shear exponent')
        self.add_discrete_input('nSector',   val=4,                         desc='number of sectors to divide rotor face into in computing thrust and power')
        self.add_discrete_input('tiploss',   val=True,                      desc='include Prandtl tip loss model')
        self.add_discrete_input('hubloss',   val=True,                      desc='include Prandtl hub loss model')
        self.add_discrete_input('wakerotation', val=True,                   desc='include effect of wake rotation (i.e., tangential induction factor is nonzero)')
        self.add_discrete_input('usecd',     val=True,                      desc='use drag coefficient in computing induction factors')

        # outputs
        self.add_output('V',        val=np.zeros(self.n_pc), units='m/s',        desc='wind vector')
        self.add_output('Omega',    val=np.zeros(self.n_pc), units='rpm',        desc='rotor rotational speed')
        self.add_output('pitch',    val=np.zeros(self.n_pc), units='deg',        desc='rotor pitch schedule')
        self.add_output('P',        val=np.zeros(self.n_pc), units='W',          desc='rotor electrical power')
        self.add_output('T',        val=np.zeros(self.n_pc), units='N',          desc='rotor aerodynamic thrust')
        self.add_output('Q',        val=np.zeros(self.n_pc), units='N*m',        desc='rotor aerodynamic torque')
        self.add_output('M',        val=np.zeros(self.n_pc), units='N*m',        desc='blade root moment')
        self.add_output('Cp',       val=np.zeros(self.n_pc),                     desc='rotor electrical power coefficient')
        self.add_output('Cp_aero',  val=np.zeros(self.n_pc),                     desc='rotor aerodynamic power coefficient')
        self.add_output('Ct_aero',  val=np.zeros(self.n_pc),                     desc='rotor aerodynamic thrust coefficient')
        self.add_output('Cq_aero',  val=np.zeros(self.n_pc),                     desc='rotor aerodynamic torque coefficient')
        self.add_output('Cm_aero',  val=np.zeros(self.n_pc),                     desc='rotor aerodynamic moment coefficient')
        self.add_output('V_spline', val=np.zeros(self.n_pc_spline), units='m/s', desc='wind vector')
        self.add_output('P_spline', val=np.zeros(self.n_pc_spline), units='W',   desc='rotor electrical power')
        self.add_output('V_R25',       val=0.0,                units='m/s', desc='region 2.5 transition wind speed')
        self.add_output('rated_V',     val=0.0,                units='m/s', desc='rated wind speed')
        self.add_output('rated_Omega', val=0.0,                units='rpm', desc='rotor rotation speed at rated')
        self.add_output('rated_pitch', val=0.0,                units='deg', desc='pitch setting at rated')
        self.add_output('rated_T',     val=0.0,                units='N',   desc='rotor aerodynamic thrust at rated')
        self.add_output('rated_Q',     val=0.0,                units='N*m', desc='rotor aerodynamic torque at rated')
        self.add_output('ax_induct_cutin',   val=np.zeros(n_span),           desc='rotor axial induction at cut-in wind speed along blade span')
        self.add_output('tang_induct_cutin', val=np.zeros(n_span),           desc='rotor tangential induction at cut-in wind speed along blade span')
        self.add_output('aoa_cutin',val=np.zeros(n_span),       units='deg', desc='angle of attack distribution along blade span at cut-in wind speed')
        self.add_output('cl_cutin', val=np.zeros(n_span),                    desc='lift coefficient distribution along blade span at cut-in wind speed')
        self.add_output('cd_cutin', val=np.zeros(n_span),                    desc='drag coefficient distribution along blade span at cut-in wind speed')

        # self.declare_partials('*', '*', method='fd', form='central', step=1e-6)
        
    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        # Create Airfoil class instances
        af = [None]*self.n_span
        for i in range(self.n_span):
            af[i] = CCAirfoil(inputs['airfoils_aoa'], inputs['airfoils_Re'], inputs['airfoils_cl'][i,:,:,0], inputs['airfoils_cd'][i,:,:,0], inputs['airfoils_cm'][i,:,:,0])
        

        self.ccblade = CCBlade(inputs['r'], inputs['chord'], inputs['theta'], af, inputs['Rhub'], inputs['Rtip'], discrete_inputs['nBlades'], inputs['rho'], inputs['mu'], inputs['precone'], inputs['tilt'], inputs['yaw'], inputs['shearExp'], inputs['hub_height'], discrete_inputs['nSector'], inputs['precurve'], inputs['precurveTip'],inputs['presweep'], inputs['presweepTip'], discrete_inputs['tiploss'], discrete_inputs['hubloss'],discrete_inputs['wakerotation'], discrete_inputs['usecd'])

        Uhub     = np.linspace(inputs['v_min'],inputs['v_max'], self.n_pc).flatten()
        
        P_aero   = np.zeros_like(Uhub)
        Cp_aero  = np.zeros_like(Uhub)
        Ct_aero  = np.zeros_like(Uhub)
        Cq_aero  = np.zeros_like(Uhub)
        Cm_aero  = np.zeros_like(Uhub)
        P        = np.zeros_like(Uhub)
        Cp       = np.zeros_like(Uhub)
        T        = np.zeros_like(Uhub)
        Q        = np.zeros_like(Uhub)
        M        = np.zeros_like(Uhub)
        Omega    = np.zeros_like(Uhub)
        pitch    = np.zeros_like(Uhub) + inputs['control_pitch']

        Omega_max = min([inputs['control_maxTS'] / inputs['Rtip'], inputs['omega_max']*np.pi/30.])
        
        # Region II
        for i in range(len(Uhub)):
            Omega[i] = Uhub[i] * inputs['tsr_operational'] / inputs['Rtip']
        
        # self.ccblade.induction = True
        P_aero, T, Q, M, Cp_aero, Ct_aero, Cq_aero, Cm_aero = self.ccblade.evaluate(Uhub, Omega * 30. / np.pi, pitch, coefficients=True)
        
        # print(Cp_aero)
        # exit()
        
        # print(self.ccblade.a)
        # import matplotlib.pyplot as plt

        # # Induction
        # ft, axt = plt.subplots(1,1,figsize=(5.3, 4))
        # axt.plot(inputs['r'], self.ccblade.a)
        # # axt.legend(fontsize=12)
        # plt.xlabel('Blade Span [m]', fontsize=14, fontweight='bold')
        # plt.ylabel('Induction [-]', fontsize=14, fontweight='bold')
        # plt.xticks(fontsize=12)
        # plt.yticks(fontsize=12)
        # plt.grid(color=[0.8,0.8,0.8], linestyle='--')
        # plt.subplots_adjust(bottom = 0.15, left = 0.15)
        # plt.show()
        
        # exit()
        
        P, eff  = CSMDrivetrain(P_aero, inputs['rated_power'], discrete_inputs['drivetrainType'], inputs['drivetrainEff'])
        Cp      = Cp_aero*eff
        
        # search for Region 2.5 bounds
        for i in range(len(Uhub)):
        
            if Omega[i] > Omega_max and P[i] < inputs['rated_power']:
                Omega[i]        = Omega_max
                Uhub[i]         = Omega[i] * inputs['Rtip'] / inputs['tsr_operational']
                P_aero[i], T[i], Q[i], M[i], Cp_aero[i], Ct_aero[i], Cq_aero[i], Cm_aero[i] = self.ccblade.evaluate([Uhub[i]], [Omega[i] * 30. / np.pi], [pitch[i]], coefficients=True)
                P[i], eff       = CSMDrivetrain(P_aero[i], inputs['rated_power'], discrete_inputs['drivetrainType'], inputs['drivetrainEff'])
                Cp[i]           = Cp_aero[i]*eff
                regionIIhalf    = True
                i_IIhalf_start  = i

                outputs['V_R25'] = Uhub[i]
                break


            if P[i] > inputs['rated_power']:
                
                regionIIhalf = False
                break

        
        def maxPregionIIhalf(pitch, Uhub, Omega):
            Uhub_i  = Uhub
            Omega_i = Omega
            pitch   = pitch
                        
            P, _, _, _ = self.ccblade.evaluate([Uhub_i], [Omega_i * 30. / np.pi], [pitch], coefficients=False)
            return -P
        
        # Solve for regoin 2.5 pitch
        options             = {}
        options['disp']     = False
        options['xatol']    = 1.e-2
        if regionIIhalf == True:
            for i in range(i_IIhalf_start + 1, len(Uhub)):   
                Omega[i]    = Omega_max
                pitch0      = pitch[i-1]
                
                bnds        = [pitch0 - 10., pitch0 + 10.]
                pitch_regionIIhalf = minimize_scalar(lambda x: maxPregionIIhalf(x, Uhub[i], Omega[i]), bounds=bnds, method='bounded', options=options)['x']
                pitch[i]    = pitch_regionIIhalf
                
                P_aero[i], T[i], Q[i], M[i], Cp_aero[i], Ct_aero[i], Cq_aero[i], Cm_aero[i] = self.ccblade.evaluate([Uhub[i]], [Omega[i] * 30. / np.pi], [pitch[i]], coefficients=True)
                
                P[i], eff  = CSMDrivetrain(P_aero[i], inputs['rated_power'], discrete_inputs['drivetrainType'], inputs['drivetrainEff'])
                Cp[i]      = Cp_aero[i]*eff

                if P[i] > inputs['rated_power']:    
                    break    
                        
        options             = {}
        options['disp']     = False
        def constantPregionIII(pitch, Uhub, Omega):
            Uhub_i  = Uhub
            Omega_i = Omega
            pitch   = pitch           
            P_aero, _, _, _ = self.ccblade.evaluate([Uhub_i], [Omega_i * 30. / np.pi], [pitch], coefficients=False)
            P, eff          = CSMDrivetrain(P_aero, inputs['rated_power'], discrete_inputs['drivetrainType'], inputs['drivetrainEff'])
            return abs(P - inputs['rated_power'])
            

        
        if regionIIhalf == True:
            # Rated conditions
            
            def min_Uhub_rated_II12(min_inputs):
                return min_inputs[1]
                
            def get_Uhub_rated_II12(min_inputs):

                Uhub_i  = min_inputs[1]
                Omega_i = Omega_max
                pitch   = min_inputs[0]           
                P_aero_i, _, _, _ = self.ccblade.evaluate([Uhub_i], [Omega_i * 30. / np.pi], [pitch], coefficients=False)
                P_i,eff          = CSMDrivetrain(P_aero_i.flatten(), inputs['rated_power'], discrete_inputs['drivetrainType'], inputs['drivetrainEff'])
                return abs(P_i - inputs['rated_power'])

            x0              = [pitch[i] + 2. , Uhub[i]]
            bnds            = [(pitch0, pitch0 + 10.),(Uhub[i-1],Uhub[i+1])]
            const           = {}
            const['type']   = 'eq'
            const['fun']    = get_Uhub_rated_II12
            params_rated    = minimize(min_Uhub_rated_II12, x0, method='SLSQP', bounds=bnds, constraints=const)
            U_rated         = params_rated.x[1]
            
            if not np.isnan(U_rated):
                Uhub[i]         = U_rated
                pitch[i]        = params_rated.x[0]
            else:
                print('Regulation trajectory is struggling to find a solution for rated wind speed. Check rotor_aeropower.py')
                U_rated         = Uhub[i]
            
            Omega[i]        = Omega_max
            P_aero[i], T[i], Q[i], M[i], Cp_aero[i], Ct_aero[i], Cq_aero[i], Cm_aero[i] = self.ccblade.evaluate([Uhub[i]], [Omega[i] * 30. / np.pi], [pitch0], coefficients=True)
            P_i, eff        = CSMDrivetrain(P_aero[i], inputs['rated_power'], discrete_inputs['drivetrainType'], inputs['drivetrainEff'])
            Cp[i]           = Cp_aero[i]*eff
            P[i]            = inputs['rated_power']
            
            
        else:
            # Rated conditions
            def get_Uhub_rated_noII12(pitch, Uhub):
                Uhub_i  = Uhub
                Omega_i = min([Uhub_i * inputs['tsr_operational'] / inputs['Rtip'], Omega_max])
                pitch_i = pitch           
                P_aero_i, _, _, _ = self.ccblade.evaluate([Uhub_i], [Omega_i * 30. / np.pi], [pitch_i], coefficients=False)
                P_i, eff          = CSMDrivetrain(P_aero_i, inputs['rated_power'], discrete_inputs['drivetrainType'], inputs['drivetrainEff'])
                return abs(P_i - inputs['rated_power'])
            
            bnds     = [Uhub[i-1], Uhub[i+1]]
            U_rated  = minimize_scalar(lambda x: get_Uhub_rated_noII12(pitch[i], x), bounds=bnds, method='bounded', options=options)['x']
            
            if not np.isnan(U_rated):
                Uhub[i]         = U_rated
            else:
                print('Regulation trajectory is struggling to find a solution for rated wind speed. Check rotor_aeropower.py. For now, U rated is assumed equal to ' + str(Uhub[i]) + ' m/s')
                U_rated         = Uhub[i]
            
            
            Omega[i] = min([Uhub[i] * inputs['tsr_operational'] / inputs['Rtip'], Omega_max])
            pitch0   = pitch[i]
            
            P_aero[i], T[i], Q[i], M[i], Cp_aero[i], Ct_aero[i], Cq_aero[i], Cm_aero[i] = self.ccblade.evaluate([Uhub[i]], [Omega[i] * 30. / np.pi], [pitch0], coefficients=True)
            P[i], eff    = CSMDrivetrain(P_aero[i], inputs['rated_power'], discrete_inputs['drivetrainType'], inputs['drivetrainEff'])
            Cp[i]        = Cp_aero[i]*eff
        
        
        for j in range(i + 1,len(Uhub)):
            Omega[j] = Omega[i]
            if self.regulation_reg_III:
                
                pitch0   = pitch[j-1]
                bnds     = [pitch0, pitch0 + 15.]
                pitch_regionIII = minimize_scalar(lambda x: constantPregionIII(x, Uhub[j], Omega[j]), bounds=bnds, method='bounded', options=options)['x']
                pitch[j]        = pitch_regionIII
                P_aero[j], T[j], Q[j], M[j], Cp_aero[j], Ct_aero[j], Cq_aero[j], Cm_aero[j] = self.ccblade.evaluate([Uhub[j]], [Omega[j] * 30. / np.pi], [pitch[j]], coefficients=True)
                P[j], eff       = CSMDrivetrain(P_aero[j], inputs['rated_power'], discrete_inputs['drivetrainType'], inputs['drivetrainEff'])
                Cp[j]           = Cp_aero[j]*eff


                if abs(P[j] - inputs['rated_power']) > 1e+4:
                    print('The pitch in region III is not being determined correctly at wind speed ' + str(Uhub[j]) + ' m/s')
                    P[j]        = inputs['rated_power']
                    T[j]        = T[j-1]
                    Q[j]        = P[j] / Omega[j]
                    M[j]        = M[j-1]
                    pitch[j]    = pitch[j-1]
                    Cp[j]       = P[j] / (0.5 * inputs['rho'] * np.pi * inputs['Rtip']**2 * Uhub[i]**3)
                    Ct_aero[j]  = Ct_aero[j-1]
                    Cq_aero[j]  = Cq_aero[j-1]
                    Cm_aero[j]  = Cm_aero[j-1]

                P[j] = inputs['rated_power']
                
            else:
                P[j]        = inputs['rated_power']
                T[j]        = 0
                Q[j]        = Q[i]
                M[j]        = 0
                pitch[j]    = 0
                Cp[j]       = P[j] / (0.5 * inputs['rho'] * np.pi * inputs['Rtip']**2 * Uhub[i]**3)
                Ct_aero[j]  = 0
                Cq_aero[j]  = 0
                Cm_aero[j]  = 0

        
        outputs['T']       = T
        outputs['Q']       = Q
        outputs['Omega']   = Omega * 30. / np.pi


        outputs['P']       = P  
        outputs['Cp']      = Cp  
        outputs['Cp_aero'] = Cp_aero
        outputs['Ct_aero'] = Ct_aero
        outputs['Cq_aero'] = Cq_aero
        outputs['Cm_aero'] = Cm_aero
        outputs['V']       = Uhub
        outputs['M']       = M
        outputs['pitch']   = pitch
                
        self.ccblade.induction_inflow = True
        a_regII, ap_regII, alpha_regII, cl_regII, cd_regII = self.ccblade.distributedAeroLoads(Uhub[0], Omega[0] * 30. / np.pi, pitch[0], 0.0)
        
        # Fit spline to powercurve for higher grid density
        spline   = PchipInterpolator(Uhub, P)
        V_spline = np.linspace(inputs['v_min'],inputs['v_max'], self.n_pc_spline)
        P_spline = spline(V_spline)
        
        # outputs
        idx_rated = list(Uhub).index(U_rated)
        outputs['rated_V']     = U_rated.flatten()
        outputs['rated_Omega'] = Omega[idx_rated] * 30. / np.pi
        outputs['rated_pitch'] = pitch[idx_rated]
        outputs['rated_T']     = T[idx_rated]
        outputs['rated_Q']     = Q[idx_rated]
        outputs['V_spline']    = V_spline.flatten()
        outputs['P_spline']    = P_spline.flatten()
        outputs['ax_induct_cutin']   = a_regII
        outputs['tang_induct_cutin'] = ap_regII
        outputs['aoa_cutin']         = alpha_regII
        outputs['cl_cutin']         = cl_regII
        outputs['cd_cutin']         = cd_regII

class Cp_Ct_Cq_Tables(ExplicitComponent):
    def initialize(self):
        self.options.declare('analysis_options')
        # self.options.declare('naero')
        # self.options.declare('n_pitch', default=20)
        # self.options.declare('n_tsr', default=20)
        # self.options.declare('n_U', default=1)
        # self.options.declare('n_aoa_grid')
        # self.options.declare('n_Re_grid')

    def setup(self):
        analysis_options = self.options['analysis_options']
        blade_init_options = analysis_options['blade']
        servose_init_options = analysis_options['servose']
        airfoils = analysis_options['airfoils']
        self.n_span        = n_span    = blade_init_options['n_span']
        self.n_aoa         = n_aoa     = airfoils['n_aoa']# Number of angle of attacks
        self.n_Re          = n_Re      = airfoils['n_Re'] # Number of Reynolds, so far hard set at 1
        self.n_tab         = n_tab     = airfoils['n_tab']# Number of tabulated data. For distributed aerodynamic control this could be > 1
        self.n_pitch       = n_pitch   = servose_init_options['n_pitch_perf_surfaces']
        self.n_tsr         = n_tsr     = servose_init_options['n_tsr_perf_surfaces']
        self.n_U           = n_U       = servose_init_options['n_U_perf_surfaces']
        self.min_TSR       = servose_init_options['min_tsr_perf_surfaces']
        self.max_TSR       = servose_init_options['max_tsr_perf_surfaces']
        self.min_pitch     = servose_init_options['min_pitch_perf_surfaces']
        self.max_pitch     = servose_init_options['max_pitch_perf_surfaces']
        
        # parameters        
        self.add_input('v_min',   val=0.0,             units='m/s',       desc='cut-in wind speed')
        self.add_input('v_max',  val=0.0,             units='m/s',       desc='cut-out wind speed')
        self.add_input('r',             val=np.zeros(n_span), units='m',         desc='radial locations where blade is defined (should be increasing and not go all the way to hub or tip)')
        self.add_input('chord',         val=np.zeros(n_span), units='m',         desc='chord length at each section')
        self.add_input('theta',         val=np.zeros(n_span), units='deg',       desc='twist angle at each section (positive decreases angle of attack)')
        self.add_input('Rhub',          val=0.0,             units='m',         desc='hub radius')
        self.add_input('Rtip',          val=0.0,             units='m',         desc='tip radius')
        self.add_input('hub_height',    val=0.0,             units='m',         desc='hub height')
        self.add_input('precone',       val=0.0,             units='deg',       desc='precone angle')
        self.add_input('tilt',          val=0.0,             units='deg',       desc='shaft tilt')
        self.add_input('yaw',           val=0.0,                units='deg',       desc='yaw error')
        self.add_input('precurve',      val=np.zeros(n_span),   units='m',         desc='precurve at each section')
        self.add_input('precurveTip',   val=0.0,                units='m',         desc='precurve at tip')
        self.add_input('presweep',      val=np.zeros(n_span),   units='m',         desc='presweep at each section')
        self.add_input('presweepTip',   val=0.0,                units='m',         desc='presweep at tip')
        self.add_input('rho',           val=1.225,              units='kg/m**3',    desc='density of air')
        self.add_input('mu',            val=1.81e-5,            units='kg/(m*s)',   desc='dynamic viscosity of air')
        self.add_input('shearExp',      val=0.0,                                desc='shear exponent')
        # self.add_discrete_input('airfoils',      val=[0]*naero,                 desc='CCAirfoil instances')
        self.add_input('airfoils_cl', val=np.zeros((n_span, n_aoa, n_Re, n_tab)), desc='lift coefficients, spanwise')
        self.add_input('airfoils_cd', val=np.zeros((n_span, n_aoa, n_Re, n_tab)), desc='drag coefficients, spanwise')
        self.add_input('airfoils_cm', val=np.zeros((n_span, n_aoa, n_Re, n_tab)), desc='moment coefficients, spanwise')
        self.add_input('airfoils_aoa', val=np.zeros((n_aoa)), units='deg', desc='angle of attack grid for polars')
        self.add_input('airfoils_Re', val=np.zeros((n_Re)), desc='Reynolds numbers of polars')
        self.add_discrete_input('nBlades',       val=0,                         desc='number of blades')
        self.add_discrete_input('nSector',       val=4,                         desc='number of sectors to divide rotor face into in computing thrust and power')
        self.add_discrete_input('tiploss',       val=True,                      desc='include Prandtl tip loss model')
        self.add_discrete_input('hubloss',       val=True,                      desc='include Prandtl hub loss model')
        self.add_discrete_input('wakerotation',  val=True,                      desc='include effect of wake rotation (i.e., tangential induction factor is nonzero)')
        self.add_discrete_input('usecd',         val=True,                      desc='use drag coefficient in computing induction factors')
        self.add_input('pitch_vector_in',  val=np.zeros(n_pitch), units='deg',  desc='pitch vector specified by the user')
        self.add_input('tsr_vector_in',    val=np.zeros(n_tsr),                 desc='tsr vector specified by the user')
        self.add_input('U_vector_in',      val=np.zeros(n_U),     units='m/s',  desc='wind vector specified by the user')

        # outputs
        self.add_output('Cp',   val=np.zeros((n_tsr, n_pitch, n_U)), desc='table of aero power coefficient')
        self.add_output('Ct',   val=np.zeros((n_tsr, n_pitch, n_U)), desc='table of aero thrust coefficient')
        self.add_output('Cq',   val=np.zeros((n_tsr, n_pitch, n_U)), desc='table of aero torque coefficient')
        self.add_output('pitch_vector',    val=np.zeros(n_pitch), units='deg',  desc='pitch vector used')
        self.add_output('tsr_vector',      val=np.zeros(n_tsr),                 desc='tsr vector used')
        self.add_output('U_vector',        val=np.zeros(n_U),     units='m/s',  desc='wind vector used')
        
    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        # Create Airfoil class instances
        af = [None]*self.n_span
        for i in range(self.n_span):
            af[i] = CCAirfoil(inputs['airfoils_aoa'], inputs['airfoils_Re'], inputs['airfoils_cl'][i,:,:,0], inputs['airfoils_cd'][i,:,:,0], inputs['airfoils_cm'][i,:,:,0])

        n_pitch    = self.n_pitch
        n_tsr      = self.n_tsr
        n_U        = self.n_U
        min_TSR    = self.min_TSR
        max_TSR    = self.max_TSR
        min_pitch  = self.min_pitch
        max_pitch  = self.max_pitch
        U_vector   = inputs['U_vector_in']
        V_in       = inputs['v_min']
        V_out      = inputs['v_max']
        
        tsr_vector = inputs['tsr_vector_in']
        pitch_vector = inputs['pitch_vector_in']
        
        self.ccblade = CCBlade(inputs['r'], inputs['chord'], inputs['theta'], af, inputs['Rhub'], inputs['Rtip'], discrete_inputs['nBlades'], inputs['rho'], inputs['mu'], inputs['precone'], inputs['tilt'], inputs['yaw'], inputs['shearExp'], inputs['hub_height'], discrete_inputs['nSector'], inputs['precurve'], inputs['precurveTip'],inputs['presweep'], inputs['presweepTip'], discrete_inputs['tiploss'], discrete_inputs['hubloss'],discrete_inputs['wakerotation'], discrete_inputs['usecd'])
        
        if max(U_vector) == 0.:
            U_vector    = np.linspace(V_in[0],V_out[0], n_U)
        if max(tsr_vector) == 0.:
            tsr_vector = np.linspace(min_TSR, max_TSR, n_tsr)
        if max(pitch_vector) == 0.:
            pitch_vector = np.linspace(min_pitch, max_pitch, n_pitch)

        outputs['pitch_vector'] = pitch_vector
        outputs['tsr_vector']   = tsr_vector        
        outputs['U_vector']     = U_vector
                
        R = inputs['Rtip']
        k=0
        for i in range(n_U):
            for j in range(n_tsr):
                k +=1
                # if k/2. == int(k/2.) :
                print('Cp-Ct-Cq surfaces completed at ' + str(int(k/(n_U*n_tsr)*100.)) + ' %')
                U     =  U_vector[i] * np.ones(n_pitch)
                Omega = tsr_vector[j] *  U_vector[i] / R * 30. / np.pi * np.ones(n_pitch)
                _, _, _, _, outputs['Cp'][j,:,i], outputs['Ct'][j,:,i], outputs['Cq'][j,:,i], _ = self.ccblade.evaluate(U, Omega, pitch_vector, coefficients=True)

# Class to define a constraint so that the blade cannot operate in stall conditions
class NoStallConstraint(ExplicitComponent):
    def initialize(self):
        
        self.options.declare('analysis_options')
    
    def setup(self):
        
        analysis_options = self.options['analysis_options']
        self.n_span        = n_span    = analysis_options['blade']['n_span']
        self.n_aoa         = n_aoa     = analysis_options['airfoils']['n_aoa']# Number of angle of attacks
        self.n_Re          = n_Re      = analysis_options['airfoils']['n_Re'] # Number of Reynolds, so far hard set at 1
        self.n_tab         = n_tab     = analysis_options['airfoils']['n_tab']# Number of tabulated data. For distributed aerodynamic control this could be > 1
        
        self.add_input('s',                     val=np.zeros(n_span),                 desc='1D array of the non-dimensional spanwise grid defined along blade axis (0-blade root, 1-blade tip)')
        self.add_input('stall_angle_along_span',val=np.zeros(n_span), units = 'deg', desc = 'Stall angle along blade span')
        self.add_input('aoa_along_span',        val=np.zeros(n_span), units = 'deg', desc = 'Angle of attack along blade span')
        self.add_input('stall_margin',          val=3.0,            units = 'deg', desc = 'Minimum margin from the stall angle')
        self.add_input('min_s',                 val=0.25,            desc = 'Minimum nondimensional coordinate along blade span where to define the constraint (blade root typically stalls)')
        self.add_input('airfoils_cl',           val=np.zeros((n_span, n_aoa, n_Re, n_tab)), desc='lift coefficients, spanwise')
        self.add_input('airfoils_cd',           val=np.zeros((n_span, n_aoa, n_Re, n_tab)), desc='drag coefficients, spanwise')
        self.add_input('airfoils_cm',           val=np.zeros((n_span, n_aoa, n_Re, n_tab)), desc='moment coefficients, spanwise')
        self.add_input('airfoils_aoa',          val=np.zeros((n_aoa)), units='deg', desc='angle of attack grid for polars')
        
        self.add_output('no_stall_constraint',  val=np.zeros(n_span), desc = 'Constraint, ratio between angle of attack plus a margin and stall angle')

    def compute(self, inputs, outputs):
        
        verbosity = True
        
        i_min = np.argmin(abs(inputs['min_s'] - inputs['s']))
        
        for i in range(self.n_span):
            unsteady = eval_unsteady(inputs['airfoils_aoa'], inputs['airfoils_cl'][i,:,0,0], inputs['airfoils_cd'][i,:,0,0], inputs['airfoils_cm'][i,:,0,0])
            inputs['stall_angle_along_span'][i] = unsteady['alpha1']
            if inputs['stall_angle_along_span'][i] == 0:
                inputs['stall_angle_along_span'][i] = 1e-6 # To avoid nan
        
        for i in range(i_min, self.n_span):
            outputs['no_stall_constraint'][i] = (inputs['aoa_along_span'][i] + inputs['stall_margin']) / inputs['stall_angle_along_span'][i]
        
            if verbosity == True:
                if outputs['no_stall_constraint'][i] > 1:
                    print('Blade is stalling at span location %.2f %%' % (inputs['s'][i]*100.))


class AEP(ExplicitComponent):
    # def initialize(self):
    #     self.options.declare('n_pc_spline', default = 200)
    
    def setup(self):
        n_pc_spline = 200
        """integrate to find annual energy production"""

        # inputs
        self.add_input('CDF_V', val=np.zeros(n_pc_spline), units='m/s', desc='cumulative distribution function evaluated at each wind speed')
        self.add_input('P', val=np.zeros(n_pc_spline), units='W', desc='power curve (power)')
        self.add_input('lossFactor', val=1.0, desc='multiplicative factor for availability and other losses (soiling, array, etc.)')

        # outputs
        self.add_output('AEP', val=0.0, units='kW*h', desc='annual energy production')

        #self.declare_partials('*', '*', method='fd', form='central', step=1e-6)

    def compute(self, inputs, outputs):

        lossFactor = inputs['lossFactor']
        P = inputs['P']
        CDF_V = inputs['CDF_V']
        
        factor = lossFactor/1e3*365.0*24.0
        outputs['AEP'] = factor*np.trapz(P, CDF_V)  # in kWh
        '''
        dAEP_dP, dAEP_dCDF = trapz_deriv(P, CDF_V)
        dAEP_dP *= factor
        dAEP_dCDF *= factor

        dAEP_dlossFactor = np.array([outputs['AEP']/lossFactor])
        self.J = {}
        self.J['AEP', 'CDF_V'] = np.reshape(dAEP_dCDF, (1, len(dAEP_dCDF)))
        self.J['AEP', 'P'] = np.reshape(dAEP_dP, (1, len(dAEP_dP)))
        self.J['AEP', 'lossFactor'] = dAEP_dlossFactor

    def compute_partials(self, inputs, J):
        J = self.J
        '''

def CSMDrivetrain(aeroPower, ratedPower, drivetrainType, drivetrainEff):

    if drivetrainEff == 0.0:
        drivetrainType = drivetrainType.upper()
        if drivetrainType == 'GEARED':
            constant = 0.01289
            linear = 0.08510
            quadratic = 0.0

        elif drivetrainType == 'SINGLE_STAGE':
            constant = 0.01331
            linear = 0.03655
            quadratic = 0.06107

        elif drivetrainType == 'MULTI_DRIVE':
            constant = 0.01547
            linear = 0.04463
            quadratic = 0.05790

        elif drivetrainType == 'PM_DIRECT_DRIVE':
            constant = 0.01007
            linear = 0.02000
            quadratic = 0.06899
        elif drivetrainType == 'CONSTANT_EFF':
            constant = 0.00
            linear = 0.07
            quadratic = 0.0
        
        Pbar0 = aeroPower / ratedPower

        # handle negative power case (with absolute value)
        Pbar1, dPbar1_dPbar0 = smooth_abs(Pbar0, dx=0.01)

        # truncate idealized power curve for purposes of efficiency calculation
        Pbar, dPbar_dPbar1, _ = smooth_min(Pbar1, 1.0, pct_offset=0.01)

        # compute efficiency
        eff = 1.0 - (constant/Pbar + linear + quadratic*Pbar)
    else:
        eff = drivetrainEff
        
    return aeroPower * eff, eff



# if __name__ = '__main__':
    