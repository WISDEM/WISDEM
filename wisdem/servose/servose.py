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
from ROSCO_toolbox import utilities as ROSCO_utilities

import numpy as np
import os
from openmdao.api import IndepVarComp, ExplicitComponent, Group, Problem
from scipy import interpolate, gradient
from scipy.optimize import minimize_scalar, minimize, brentq
from scipy.interpolate import PchipInterpolator
from wisdem.ccblade import CCAirfoil, CCBlade
from wisdem.commonse.distribution import RayleighCDF, WeibullWithMeanCDF
from wisdem.commonse.utilities import vstack, trapz_deriv, linspace_with_deriv, smooth_min, smooth_abs
from wisdem.commonse.environment import PowerWind
from wisdem.aeroelasticse.openmdao_openfast import eval_unsteady

class ServoSE(Group):
    def initialize(self):
        self.options.declare('modeling_options')
        self.options.declare('opt_options')

    def setup(self):
        modeling_options = self.options['modeling_options']

        self.add_subsystem('powercurve',        RegulatedPowerCurve(modeling_options   = modeling_options), promotes = ['v_min', 'v_max','rated_power','omega_min','omega_max', 'control_maxTS','tsr_operational','control_pitch','drivetrainType','r','chord', 'theta','Rhub', 'Rtip', 'hub_height','precone', 'tilt','yaw','precurve','precurveTip','presweep','presweepTip', 'airfoils_aoa','airfoils_Re','airfoils_cl','airfoils_cd','airfoils_cm', 'nBlades', 'rho', 'mu'])
        self.add_subsystem('gust',              GustETM())
        self.add_subsystem('cdf',               WeibullWithMeanCDF(nspline=modeling_options['servose']['n_pc_spline']))
        self.add_subsystem('aep',               AEP(nspline=modeling_options['servose']['n_pc_spline']), promotes=['AEP'])


        # Connections to the Weibull CDF
        self.connect('powercurve.V_spline', 'cdf.x')
        
        # Connections to the aep computation component
        self.connect('cdf.F',               'aep.CDF_V')
        self.connect('powercurve.P_spline', 'aep.P')   

class ServoSE_ROSCO(Group):
    def initialize(self):
        self.options.declare('modeling_options')
        self.options.declare('opt_options')

    def setup(self):
        modeling_options = self.options['modeling_options']

        self.add_subsystem('aeroperf_tables',   Cp_Ct_Cq_Tables(modeling_options   = modeling_options), promotes = ['v_min', 'v_max','r','chord', 'theta','Rhub', 'Rtip', 'hub_height','precone', 'tilt','yaw','precurve','precurveTip','presweep','presweepTip', 'airfoils_aoa','airfoils_Re','airfoils_cl','airfoils_cd','airfoils_cm', 'nBlades', 'rho', 'mu'])
        self.add_subsystem('tune_rosco',        TuneROSCO(modeling_options = modeling_options), promotes = ['v_min', 'v_max', 'rho', 'omega_min', 'tsr_operational', 'rated_power', 'r','chord', 'theta','Rhub', 'Rtip', 'hub_height','precone', 'tilt','yaw','precurve','precurveTip','presweep','presweepTip', 'airfoils_Ctrl', 'airfoils_aoa','airfoils_Re','airfoils_cl','airfoils_cd','airfoils_cm', 'nBlades', 'rho', 'mu'])

        # Connect ROSCO for Rotor Performance tables
        self.connect('aeroperf_tables.Cp',              'tune_rosco.Cp_table')
        self.connect('aeroperf_tables.Ct',              'tune_rosco.Ct_table')
        self.connect('aeroperf_tables.Cq',              'tune_rosco.Cq_table')
        self.connect('aeroperf_tables.pitch_vector',    'tune_rosco.pitch_vector')
        self.connect('aeroperf_tables.tsr_vector',      'tune_rosco.tsr_vector')
        self.connect('aeroperf_tables.U_vector',        'tune_rosco.U_vector')

class GustETM(ExplicitComponent):
    # OpenMDAO component that generates an "equivalent gust" wind speed by summing an user-defined wind speed at hub height with 3 times sigma. sigma is the turbulent wind speed standard deviation for the extreme turbulence model, see IEC-61400-1 Eq. 19 paragraph 6.3.2.3
    
    def setup(self):
        # Inputs
        self.add_input('V_mean', val=0.0, units='m/s', desc='IEC average wind speed for turbine class')
        self.add_input('V_hub', val=0.0, units='m/s', desc='hub height wind speed')
        self.add_discrete_input('turbulence_class', val='A', desc='IEC turbulence class')
        self.add_input('std', val=3.0, desc='number of standard deviations for strength of gust')

        # Output
        self.add_output('V_gust', val=0.0, units='m/s', desc='gust wind speed')

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        V_mean = inputs['V_mean']
        V_hub  = inputs['V_hub']
        std    = inputs['std']
        turbulence_class = discrete_inputs['turbulence_class']

        if turbulence_class.upper() == 'A':
            Iref = 0.16
        elif turbulence_class.upper() == 'B':
            Iref = 0.14
        elif turbulence_class.upper() == 'C':
            Iref = 0.12
        else:
            raise ValueError('Unknown Turbulence Class: '+str(turbulence_class)+' . Permitted values are A / B / C')

        c = 2.0
        sigma = c * Iref * (0.072*(V_mean/c + 3)*(V_hub/c - 4) + 10)
        V_gust = V_hub + std*sigma
        outputs['V_gust'] = V_gust

class TuneROSCO(ExplicitComponent):
    def initialize(self):
        self.options.declare('modeling_options')

    def setup(self):
        self.modeling_options = self.options['modeling_options']
        servose_init_options = self.modeling_options['servose']

        # Input parameters
        self.controller_params = {}
        # Controller Flags
        self.controller_params['LoggingLevel'] = self.modeling_options['servose']['LoggingLevel']
        self.controller_params['F_LPFType'] = self.modeling_options['servose']['F_LPFType']
        self.controller_params['F_NotchType'] = self.modeling_options['servose']['F_NotchType']
        self.controller_params['IPC_ControlMode'] = self.modeling_options['servose']['IPC_ControlMode']
        self.controller_params['VS_ControlMode'] = self.modeling_options['servose']['VS_ControlMode']
        self.controller_params['PC_ControlMode'] = self.modeling_options['servose']['PC_ControlMode']
        self.controller_params['Y_ControlMode'] = self.modeling_options['servose']['Y_ControlMode']
        self.controller_params['SS_Mode'] = self.modeling_options['servose']['SS_Mode']
        self.controller_params['WE_Mode'] = self.modeling_options['servose']['WE_Mode']
        self.controller_params['PS_Mode'] = self.modeling_options['servose']['PS_Mode']
        self.controller_params['SD_Mode'] = self.modeling_options['servose']['SD_Mode']
        self.controller_params['Fl_Mode'] = self.modeling_options['servose']['Fl_Mode']
        self.controller_params['Flp_Mode'] = self.modeling_options['servose']['Flp_Mode']

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
        self.add_input('gearbox_efficiency',val=0.0,                                desc='Gearbox efficiency')
        self.add_input('generator_efficiency', val=0.0,                             desc='Generator efficiency')
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

        # For cc-blade & flaps tuning
        self.n_span     = n_span       = self.modeling_options['blade']['n_span']
        # self.n_af       = n_af         = af_init_options['n_af'] # Number of airfoils
        self.n_aoa      = n_aoa        = self.modeling_options['airfoils']['n_aoa']# Number of angle of attacks
        self.n_Re       = n_Re         = self.modeling_options['airfoils']['n_Re'] # Number of Reynolds, so far hard set at 1
        self.n_tab      = n_tab        = self.modeling_options['airfoils']['n_tab']# Number of tabulated data. For distributed aerodynamic control this could be > 1
        self.n_te_flaps = n_te_flaps   = self.modeling_options['blade']['n_te_flaps']
        self.add_input('r',             val=np.zeros(n_span),               units='m',          desc='radial locations where blade is defined (should be increasing and not go all the way to hub or tip)')
        self.add_input('chord',         val=np.zeros(n_span),               units='m',          desc='chord length at each section')
        self.add_input('theta',         val=np.zeros(n_span),               units='deg',        desc='twist angle at each section (positive decreases angle of attack)')
        self.add_input('Rhub',          val=0.0,                            units='m',          desc='hub radius')
        self.add_input('Rtip',          val=0.0,                            units='m',          desc='tip radius')
        self.add_input('hub_height',    val=0.0,                            units='m',          desc='hub height')
        self.add_input('precone',       val=0.0,                            units='deg',        desc='precone angle', )
        self.add_input('tilt',          val=0.0,                            units='deg',        desc='shaft tilt', )
        self.add_input('yaw',           val=0.0,                            units='deg',        desc='yaw error', )
        self.add_input('precurve',      val=np.zeros(n_span),               units='m',          desc='precurve at each section')
        self.add_input('precurveTip',   val=0.0,                            units='m',          desc='precurve at tip')
        self.add_input('presweep',      val=np.zeros(n_span),               units='m',          desc='presweep at each section')
        self.add_input('presweepTip',   val=0.0,                            units='m',          desc='presweep at tip')
        self.add_input('airfoils_cl',   val=np.zeros((n_span, n_aoa, n_Re, n_tab)),             desc='lift coefficients, spanwise')
        self.add_input('airfoils_cd',   val=np.zeros((n_span, n_aoa, n_Re, n_tab)),             desc='drag coefficients, spanwise')
        self.add_input('airfoils_cm',   val=np.zeros((n_span, n_aoa, n_Re, n_tab)),             desc='moment coefficients, spanwise')
        self.add_input('airfoils_aoa',  val=np.zeros((n_aoa)),              units='deg',        desc='angle of attack grid for polars')
        self.add_input('airfoils_Re',   val=np.zeros((n_Re)),                                   desc='Reynolds numbers of polars')
        self.add_input('airfoils_Ctrl', val=np.zeros((n_span, n_Re, n_tab)), units='deg',       desc='Airfoil control paremeter (i.e. flap angle)')
        self.add_discrete_input('nBlades',         val=0,                                       desc='number of blades')
        self.add_input('mu',            val=1.81e-5,                        units='kg/(m*s)',   desc='dynamic viscosity of air')
        self.add_input('shearExp',      val=0.0,                                                desc='shear exponent')
        self.add_input('delta_max_pos', val=np.zeros(n_te_flaps),           units='rad',        desc='1D array of the max angle of the trailing edge flaps.')
        self.add_discrete_input('nSector',      val=4,                                          desc='number of sectors to divide rotor face into in computing thrust and power')
        self.add_discrete_input('tiploss',      val=True,                                       desc='include Prandtl tip loss model')
        self.add_discrete_input('hubloss',      val=True,                                       desc='include Prandtl hub loss model')
        self.add_discrete_input('wakerotation', val=True,                                       desc='include effect of wake rotation (i.e., tangential induction factor is nonzero)')
        self.add_discrete_input('usecd',        val=True,                                       desc='use drag coefficient in computing induction factors')

        # Controller Tuning Parameters
        self.add_input('PC_zeta',           val=0.0,                                            desc='Pitch controller damping ratio')
        self.add_input('PC_omega',          val=0.0,        units='rad/s',                      desc='Pitch controller natural frequency')
        self.add_input('VS_zeta',           val=0.0,                                            desc='Generator torque controller damping ratio')
        self.add_input('VS_omega',          val=0.0,        units='rad/s',                      desc='Generator torque controller natural frequency')
        if self.modeling_options['servose']['Flp_Mode'] > 0:
            self.add_input('Flp_omega',        val=0.0, units='rad/s',                         desc='Flap controller natural frequency')
            self.add_input('Flp_zeta',         val=0.0,                                        desc='Flap controller damping ratio')

        # Outputs for constraints and optimizations
        self.add_output('Flp_Kp',           val=0.0,            units='rad',        desc='Flap control proportional gain')
        self.add_output('Flp_Ki',           val=0.0,            units='rad',        desc='Flap control integral gain')

    def compute(self,inputs,outputs, discrete_inputs, discrete_outputs):
        '''
        Call ROSCO toolbox to define controller
        '''

        # Add control tuning parameters to dictionary
        self.modeling_options['servose']['omega_pc']    = inputs['PC_omega']
        self.modeling_options['servose']['zeta_pc']     = inputs['PC_zeta']
        self.modeling_options['servose']['omega_vs']    = inputs['VS_omega']
        self.modeling_options['servose']['zeta_vs']     = inputs['VS_zeta']
        if self.modeling_options['servose']['Flp_Mode'] > 0:
            self.modeling_options['servose']['omega_flp'] = inputs['Flp_omega']
            self.modeling_options['servose']['zeta_flp']  = inputs['Flp_zeta']
        else:
            self.modeling_options['servose']['omega_flp'] = 0.0
            self.modeling_options['servose']['zeta_flp']  = 0.0
        #
        self.modeling_options['servose']['max_pitch']   = inputs['max_pitch'][0]
        self.modeling_options['servose']['min_pitch']   = inputs['min_pitch'][0]
        self.modeling_options['servose']['vs_minspd']   = inputs['vs_minspd'][0]
        self.modeling_options['servose']['ss_vsgain']   = inputs['ss_vsgain'][0]
        self.modeling_options['servose']['ss_pcgain']   = inputs['ss_pcgain'][0]
        self.modeling_options['servose']['ps_percent']  = inputs['ps_percent'][0]
        if self.modeling_options['servose']['Flp_Mode'] > 0:
            self.modeling_options['servose']['flp_maxpit']  = inputs['delta_max_pos'][0]
        else:
            self.modeling_options['servose']['flp_maxpit']  = None
        #
        self.modeling_options['servose']['ss_cornerfreq']   = None
        self.modeling_options['servose']['sd_maxpit']       = None
        self.modeling_options['servose']['sd_cornerfreq']   = None

        # Define necessary turbine parameters
        WISDEM_turbine = type('', (), {})()
        WISDEM_turbine.v_min        = inputs['v_min'][0]
        WISDEM_turbine.J            = inputs['rotor_inertia'][0]
        WISDEM_turbine.rho          = inputs['rho'][0]
        WISDEM_turbine.rotor_radius = inputs['R'][0]
        WISDEM_turbine.Ng           = inputs['gear_ratio'][0]
        WISDEM_turbine.GenEff       = inputs['generator_efficiency'][0] * 100.
        WISDEM_turbine.GBoxEff      = inputs['gearbox_efficiency'][0] * 100.
        WISDEM_turbine.rated_rotor_speed   = inputs['rated_rotor_speed'][0]
        WISDEM_turbine.rated_power  = inputs['rated_power'][0]
        WISDEM_turbine.rated_torque = inputs['rated_torque'][0] / WISDEM_turbine.Ng * inputs['gearbox_efficiency'][0]
        WISDEM_turbine.v_rated      = inputs['v_rated'][0]
        WISDEM_turbine.v_min        = inputs['v_min'][0]
        WISDEM_turbine.v_max        = inputs['v_max'][0]
        WISDEM_turbine.max_pitch_rate   = inputs['max_pitch_rate'][0]
        WISDEM_turbine.TSR_operational  = inputs['tsr_operational'][0]
        WISDEM_turbine.max_torque_rate  = inputs['max_torque_rate'][0]

        # Load Cp tables
        self.Cp_table       = inputs['Cp_table']
        self.Ct_table       = inputs['Ct_table']
        self.Cq_table       = inputs['Cq_table']
        self.pitch_vector   = WISDEM_turbine.pitch_initial_rad = inputs['pitch_vector']
        self.tsr_vector     = WISDEM_turbine.TSR_initial = inputs['tsr_vector']
        self.Cp_table       = WISDEM_turbine.Cp_table = self.Cp_table.reshape(len(self.pitch_vector),len(self.tsr_vector))
        self.Ct_table       = WISDEM_turbine.Ct_table = self.Ct_table.reshape(len(self.pitch_vector),len(self.tsr_vector))
        self.Cq_table       = WISDEM_turbine.Cq_table = self.Cq_table.reshape(len(self.pitch_vector),len(self.tsr_vector))

        RotorPerformance = ROSCO_turbine.RotorPerformance
        WISDEM_turbine.Cp   = RotorPerformance(self.Cp_table,self.pitch_vector,self.tsr_vector)
        WISDEM_turbine.Ct   = RotorPerformance(self.Ct_table,self.pitch_vector,self.tsr_vector)
        WISDEM_turbine.Cq   = RotorPerformance(self.Cq_table,self.pitch_vector,self.tsr_vector)

        # Load blade info to pass to flap controller tuning process
        if self.modeling_options['servose']['Flp_Mode'] >= 1:
            # Create airfoils
            af = [None]*self.n_span
            for i in range(self.n_span):
                if self.n_tab > 1:
                    ref_tab = int(np.floor(self.n_tab/2))
                    af[i] = CCAirfoil(inputs['airfoils_aoa'], inputs['airfoils_Re'], inputs['airfoils_cl'][i,:,:,ref_tab], inputs['airfoils_cd'][i,:,:,ref_tab], inputs['airfoils_cm'][i,:,:,ref_tab])
                else:
                    af[i] = CCAirfoil(inputs['airfoils_aoa'], inputs['airfoils_Re'], inputs['airfoils_cl'][i,:,:,0], inputs['airfoils_cd'][i,:,:,0], inputs['airfoils_cm'][i,:,:,0])
            
            # Initialize CCBlade as cc_rotor object 
            WISDEM_turbine.cc_rotor = CCBlade(inputs['r'], inputs['chord'], inputs['theta'], af, inputs['Rhub'], inputs['Rtip'], discrete_inputs['nBlades'], inputs['rho'], inputs['mu'], inputs['precone'], inputs['tilt'], inputs['yaw'], inputs['shearExp'], inputs['hub_height'], discrete_inputs['nSector'], inputs['precurve'], inputs['precurveTip'],inputs['presweep'], inputs['presweepTip'], discrete_inputs['tiploss'], discrete_inputs['hubloss'],discrete_inputs['wakerotation'], discrete_inputs['usecd'])
        
            # Load aerodynamic performance data for blades
            WISDEM_turbine.af_data = [{} for i in range(self.n_span)]
            for i in range(self.n_span):
                # Check number of flap positions for each airfoil section
                if self.n_tab > 1:
                    if inputs['airfoils_Ctrl'][i,0,0] == inputs['airfoils_Ctrl'][i,0,1]:
                        n_tabs = 1  # If all Ctrl angles of the flaps are identical then no flaps
                    else:
                        n_tabs = self.n_tab
                else:
                    n_tabs = 1
                # Save data for each flap position
                for j in range(n_tabs):
                    WISDEM_turbine.af_data[i][j] = {}
                    WISDEM_turbine.af_data[i][j]['NumTabs'] = n_tabs
                    WISDEM_turbine.af_data[i][j]['Ctrl']    = inputs['airfoils_Ctrl'][i,0,j]
                    WISDEM_turbine.af_data[i][j]['Alpha']   = np.array(inputs['airfoils_aoa']).flatten().tolist()
                    WISDEM_turbine.af_data[i][j]['Cl']      = np.array(inputs['airfoils_cl'][i,:,0,j]).flatten().tolist()
                    WISDEM_turbine.af_data[i][j]['Cd']      = np.array(inputs['airfoils_cd'][i,:,0,j]).flatten().tolist()
                    WISDEM_turbine.af_data[i][j]['Cm']      = np.array(inputs['airfoils_cm'][i,:,0,j]).flatten().tolist()
   
            # Save some more airfoil info
            WISDEM_turbine.span     = inputs['r'] 
            WISDEM_turbine.chord    = inputs['chord']
            WISDEM_turbine.twist    = inputs['theta']
            WISDEM_turbine.bld_flapwise_freq = inputs['flap_freq'][0] * 2*np.pi
            WISDEM_turbine.bld_flapwise_damp = self.modeling_options['openfast']['fst_vt']['ElastoDynBlade']['BldFlDmp1']/100 * 0.7

        # Tune Controller!
        controller = ROSCO_controller.Controller(self.modeling_options['servose'])
        controller.tune_controller(WISDEM_turbine)

        # DISCON Parameters
        #   - controller
        
        if 'DISCON_in' not in self.modeling_options['openfast']['fst_vt'].keys():
            self.modeling_options['openfast']['fst_vt']['DISCON_in']  = {}
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['LoggingLevel'] = controller.LoggingLevel
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['F_LPFType'] = controller.F_LPFType
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['F_NotchType'] = controller.F_NotchType
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['IPC_ControlMode'] = controller.IPC_ControlMode
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['VS_ControlMode'] = controller.VS_ControlMode
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['PC_ControlMode'] = controller.PC_ControlMode
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['Y_ControlMode'] = controller.Y_ControlMode
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['SS_Mode'] = controller.SS_Mode
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['WE_Mode'] = controller.WE_Mode
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['PS_Mode'] = controller.PS_Mode
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['SD_Mode'] = controller.SD_Mode
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['Fl_Mode'] = controller.Fl_Mode
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['Flp_Mode'] = controller.Flp_Mode
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['F_LPFDamping'] = controller.F_LPFDamping
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['F_SSCornerFreq'] = controller.ss_cornerfreq
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['PC_GS_angles'] = controller.pitch_op_pc
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['PC_GS_KP'] = controller.pc_gain_schedule.Kp
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['PC_GS_KI'] = controller.pc_gain_schedule.Ki
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['PC_MaxPit'] = controller.max_pitch
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['PC_MinPit'] = controller.min_pitch
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['VS_MinOMSpd'] = controller.vs_minspd
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['VS_Rgn2K'] = controller.vs_rgn2K
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['VS_RefSpd'] = controller.vs_refspd
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['VS_KP'] = controller.vs_gain_schedule.Kp
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['VS_KI'] = controller.vs_gain_schedule.Ki
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['SS_VSGain'] = controller.ss_vsgain
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['SS_PCGain'] = controller.ss_pcgain
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['WE_FOPoles_N'] = len(controller.v)
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['WE_FOPoles_v'] = controller.v
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['WE_FOPoles'] = controller.A
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['ps_wind_speeds'] = controller.v
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['PS_BldPitchMin'] = controller.ps_min_bld_pitch
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['SD_MaxPit'] = controller.sd_maxpit + 0.1
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['SD_CornerFreq'] = controller.sd_cornerfreq
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['Fl_Kp'] = controller.Kp_float
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['Flp_Kp'] = controller.Kp_flap
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['Flp_Ki'] = controller.Ki_flap
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['Flp_MaxPit'] = controller.flp_maxpit
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['Flp_Angle'] = 0.
        
        # - turbine
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['WE_BladeRadius'] = WISDEM_turbine.rotor_radius
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['v_rated'] = inputs['v_rated'][0]
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['F_FlpCornerFreq']  = [inputs['flap_freq'][0] * 2 * np.pi / 3., 0.7]
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['F_LPFCornerFreq']  = inputs['edge_freq'][0] * 2 * np.pi / 4.
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['F_NotchCornerFreq'] = 0.0    # inputs(['twr_freq']) # zero for now, fix when floating introduced to WISDEM
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['F_FlCornerFreq'] = [0.0, 0.0] # inputs(['ptfm_freq']) # zero for now, fix when floating introduced to WISDEM
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['PC_MaxRat'] = WISDEM_turbine.max_pitch_rate
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['PC_MinRat'] = -WISDEM_turbine.max_pitch_rate
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['VS_MaxRat'] = WISDEM_turbine.max_torque_rate
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['PC_RefSpd'] = WISDEM_turbine.rated_rotor_speed * WISDEM_turbine.Ng
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['VS_RtPwr'] = WISDEM_turbine.rated_power
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['VS_RtTq'] = WISDEM_turbine.rated_torque
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['VS_MaxTq'] = WISDEM_turbine.rated_torque * 1.1
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['VS_TSRopt'] = WISDEM_turbine.TSR_operational
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['WE_RhoAir'] = WISDEM_turbine.rho
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['WE_GearboxRatio'] = WISDEM_turbine.Ng
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['WE_Jtot'] = WISDEM_turbine.J
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['Cp_pitch_initial_rad'] = self.pitch_vector
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['Cp_TSR_initial'] = self.tsr_vector
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['Cp_table'] = WISDEM_turbine.Cp_table
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['Ct_table'] = WISDEM_turbine.Ct_table
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['Cq_table'] = WISDEM_turbine.Cq_table
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['Cp'] = WISDEM_turbine.Cp
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['Ct'] = WISDEM_turbine.Ct
        self.modeling_options['openfast']['fst_vt']['DISCON_in']['Cq'] = WISDEM_turbine.Cq

        # Outputs 
        outputs['Flp_Kp'] = controller.Kp_flap[-1]
        outputs['Flp_Ki'] = controller.Ki_flap[-1]

class RegulatedPowerCurve(Group):
    
    def initialize(self):
        self.options.declare('modeling_options')

    def setup(self):
        modeling_options = self.options['modeling_options']
        
        self.add_subsystem('compute_power_curve', ComputePowerCurve(modeling_options=modeling_options), promotes=['*'])
        
        self.add_subsystem('compute_splines', ComputeSplines(modeling_options=modeling_options), promotes=['*'])

class ComputePowerCurve(ExplicitComponent):
    """
    Iteratively call CCBlade to compute the power curve.
    """
    
    def initialize(self):
        self.options.declare('modeling_options')

    def setup(self):
        modeling_options = self.options['modeling_options']
        self.n_span        = n_span    = modeling_options['blade']['n_span']
        # self.n_af          = n_af      = af_init_options['n_af'] # Number of airfoils
        self.n_aoa         = n_aoa     = modeling_options['airfoils']['n_aoa']# Number of angle of attacks
        self.n_Re          = n_Re      = modeling_options['airfoils']['n_Re'] # Number of Reynolds, so far hard set at 1
        self.n_tab         = n_tab     = modeling_options['airfoils']['n_tab']# Number of tabulated data. For distributed aerodynamic control this could be > 1
        # self.n_xy          = n_xy      = af_init_options['n_xy'] # Number of coordinate points to describe the airfoil geometry
        self.regulation_reg_III = modeling_options['servose']['regulation_reg_III']
        # n_span       = self.n_span = self.options['n_span']
        self.n_pc          = modeling_options['servose']['n_pc']
        self.n_pc_spline   = modeling_options['servose']['n_pc_spline']
        # n_aoa  = self.options['n_aoa']
        # n_re   = self.options['n_re']
        
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
        self.add_input('gearbox_efficiency',     val=0.0,               desc='Gearbox efficiency')
        self.add_input('generator_efficiency',   val=0.0,               desc='Generator efficiency')
        
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
        
        # self.add_discrete_input('airfoils',  val=[0]*n_span,                      desc='CCAirfoil instances')
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
        
        self.add_output('V_R25',       val=0.0,                units='m/s', desc='region 2.5 transition wind speed')
        self.add_output('rated_V',     val=0.0,                units='m/s', desc='rated wind speed')
        self.add_output('rated_Omega', val=0.0,                units='rpm', desc='rotor rotation speed at rated')
        self.add_output('rated_pitch', val=0.0,                units='deg', desc='pitch setting at rated')
        self.add_output('rated_T',     val=0.0,                units='N',   desc='rotor aerodynamic thrust at rated')
        self.add_output('rated_Q',     val=0.0,                units='N*m', desc='rotor aerodynamic torque at rated')
        self.add_output('ax_induct_regII',   val=np.zeros(n_span),           desc='rotor axial induction at cut-in wind speed along blade span')
        self.add_output('tang_induct_regII', val=np.zeros(n_span),           desc='rotor tangential induction at cut-in wind speed along blade span')
        self.add_output('aoa_regII',val=np.zeros(n_span),       units='deg', desc='angle of attack distribution along blade span at cut-in wind speed')
        self.add_output('Cp_regII', val=0.0,                                 desc='power coefficient at cut-in wind speed')
        self.add_output('cl_regII', val=np.zeros(n_span),                    desc='lift coefficient distribution along blade span at cut-in wind speed')
        self.add_output('cd_regII', val=np.zeros(n_span),                    desc='drag coefficient distribution along blade span at cut-in wind speed')

        # self.declare_partials('*', '*', method='fd', form='central', step=1e-6)

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        
        # Create Airfoil class instances
        af = [None]*self.n_span
        for i in range(self.n_span):
            if self.n_tab > 1:
                ref_tab = int(np.floor(self.n_tab/2))
                af[i] = CCAirfoil(inputs['airfoils_aoa'], inputs['airfoils_Re'], inputs['airfoils_cl'][i,:,:,ref_tab], inputs['airfoils_cd'][i,:,:,ref_tab], inputs['airfoils_cm'][i,:,:,ref_tab])
            else:
                af[i] = CCAirfoil(inputs['airfoils_aoa'], inputs['airfoils_Re'], inputs['airfoils_cl'][i,:,:,0], inputs['airfoils_cd'][i,:,:,0], inputs['airfoils_cm'][i,:,:,0])
                
        self.ccblade = CCBlade(inputs['r'], inputs['chord'], inputs['theta'], af, inputs['Rhub'], inputs['Rtip'], discrete_inputs['nBlades'], inputs['rho'], inputs['mu'], inputs['precone'], inputs['tilt'], inputs['yaw'], inputs['shearExp'], inputs['hub_height'], discrete_inputs['nSector'], inputs['precurve'], inputs['precurveTip'],inputs['presweep'], inputs['presweepTip'], discrete_inputs['tiploss'], discrete_inputs['hubloss'],discrete_inputs['wakerotation'], discrete_inputs['usecd'])
        
        # JPJ: what is this grid for? Seems to be a special distribution of velocities
        # for the hub
        grid0 = np.cumsum(np.abs(np.diff(np.cos(np.linspace(-np.pi/4.,np.pi/2.,self.n_pc + 1)))))
        grid1 = (grid0 - grid0[0])/(grid0[-1]-grid0[0])
        Uhub  = grid1 * (inputs['v_max'] - inputs['v_min']) + inputs['v_min']
        
        P_aero   = np.zeros( Uhub.shape )
        Cp_aero  = np.zeros( Uhub.shape )
        Ct_aero  = np.zeros( Uhub.shape )
        Cq_aero  = np.zeros( Uhub.shape )
        Cm_aero  = np.zeros( Uhub.shape )
        P        = np.zeros( Uhub.shape )
        Cp       = np.zeros( Uhub.shape )
        T        = np.zeros( Uhub.shape )
        Q        = np.zeros( Uhub.shape )
        M        = np.zeros( Uhub.shape )
        pitch    = np.zeros( Uhub.shape ) + inputs['control_pitch']

        # Unpack variables
        P_rated   = inputs['rated_power']
        R_tip     = inputs['Rtip']
        tsr       = inputs['tsr_operational']
        driveType = discrete_inputs['drivetrainType']
        driveEta  = inputs['gearbox_efficiency'] * inputs['generator_efficiency']
        
        # Set rotor speed based on TSR
        Omega_tsr = Uhub * tsr / R_tip

        # Determine maximum rotor speed (rad/s)- either by TS or by control input        
        Omega_max = min([inputs['control_maxTS'] / R_tip, inputs['omega_max']*np.pi/30.])

        # Apply maximum and minimum rotor speed limits
        Omega     = np.maximum(np.minimum(Omega_tsr, Omega_max), inputs['omega_min']*np.pi/30.)
        Omega_rpm = Omega * 30. / np.pi

        # Set baseline power production
        myout, derivs = self.ccblade.evaluate(Uhub, Omega_rpm, pitch, coefficients=True)
        P_aero, T, Q, M, Cp_aero, Ct_aero, Cq_aero, Cm_aero = [myout[key] for key in ['P','T','Q','M','CP','CT','CQ','CM']]
        P, eff  = compute_P_and_eff(P_aero, P_rated, driveType, driveEta)
        Cp      = Cp_aero*eff

        
        # Find Region 3 index
        region_bool = np.nonzero(P >= P_rated)[0]
        if len(region_bool)==0:
            i_3     = self.n_pc
            region3 = False
        else:
            i_3     = region_bool[0] + 1
            region3 = True

        # Guess at Region 2.5, but we will do a more rigorous search below
        if Omega_max < Omega_tsr[-1]:
            U_2p5 = np.interp(Omega[-1], Omega_tsr, Uhub)
            outputs['V_R25'] = U_2p5
        else:
            U_2p5 = Uhub[-1]
        i_2p5   = np.nonzero(U_2p5 <= Uhub)[0][0]

        # Find rated index and guess at rated speed
        if P_aero[-1] > P_rated:
            U_rated = np.interp(P_rated, P_aero*eff, Uhub)
        else:
            U_rated = Uhub[-1]
        i_rated = np.nonzero(U_rated <= Uhub)[0][0]

        # Function to be used inside of power maximization until Region 3
        def maximizePower(pitch, Uhub, Omega_rpm):
            myout, _ = self.ccblade.evaluate([Uhub], [Omega_rpm], [pitch], coefficients=False)
            return -myout['P']

        # Maximize power until Region 3
        region2p5 = False
        for i in range(i_3):
            # No need to optimize if already doing well
            if Omega[i] == Omega_tsr[i]: continue

            # Find pitch value that gives highest power rating
            pitch0   = pitch[i] if i==0 else pitch[i-1]
            bnds     = [pitch0-10., pitch0+10.]
            pitch[i] = minimize_scalar(lambda x: maximizePower(x, Uhub[i], Omega_rpm[i]),
                                       bounds=bnds, method='bounded', options={'disp':False, 'xatol':1e-2, 'maxiter':40})['x']

            # Find associated power
            myout, _ = self.ccblade.evaluate([Uhub[i]], [Omega_rpm[i]], [pitch[i]], coefficients=True)
            P_aero[i], T[i], Q[i], M[i], Cp_aero[i], Ct_aero[i], Cq_aero[i], Cm_aero[i] = [myout[key] for key in ['P','T','Q','M','CP','CT','CQ','CM']]
            P[i], eff  = compute_P_and_eff(P_aero[i], P_rated, driveType, driveEta)
            Cp[i]      = Cp_aero[i]*eff

            # Note if we find Region 2.5
            if ( (not region2p5) and (Omega[i] == Omega_max) and (P[i] < P_rated) ):
                region2p5 = True
                i_2p5     = i

            # Stop if we find Region 3 early
            if P[i] > P_rated:
                i_3     = i+1
                i_rated = i
                break

        # Solve for rated velocity
        # JPJ: why rename i_rated to i here? It removes clarity in the following 50 lines that we're looking at the rated properties
        i = i_rated
        if i < self.n_pc-1:
            def const_Urated(x):
                pitch    = x[0]           
                Uhub_i   = x[1]
                Omega_i  = min([Uhub_i * tsr / R_tip, Omega_max])
                myout, _ = self.ccblade.evaluate([Uhub_i], [Omega_i*30./np.pi], [pitch], coefficients=False)
                P_aero_i = myout['P']
                P_i,eff  = compute_P_and_eff(P_aero_i.flatten(), P_rated, driveType, driveEta)
                return (P_i - P_rated)

            if region2p5:
                # Have to search over both pitch and speed
                x0            = [0.0, Uhub[i]]
                bnds          = [ np.sort([pitch[i-1], pitch[i+1]]), [Uhub[i-1], Uhub[i+1]] ]
                const         = {}
                const['type'] = 'eq'
                const['fun']  = const_Urated
                params_rated  = minimize(lambda x: x[1], x0, method='slsqp', bounds=bnds, constraints=const, tol=1e-3)

                if params_rated.success and not np.isnan(params_rated.x[1]):
                    U_rated  = params_rated.x[1]
                    pitch[i] = params_rated.x[0]
                else:
                    U_rated = U_rated # Use guessed value earlier
                    pitch[i] = 0.0
            else:
                # Just search over speed
                pitch[i] = 0.0
                try:
                    U_rated = brentq(lambda x: const_Urated([0.0, x]), Uhub[i-1], Uhub[i+1],
                                     xtol = 1e-4, rtol = 1e-5, maxiter=40, disp=False)
                except ValueError:
                    U_rated = minimize_scalar(lambda x: np.abs(const_Urated([0.0, x])), bounds=[Uhub[i-1], Uhub[i+1]],
                                              method='bounded', options={'disp':False, 'xatol':1e-3, 'maxiter':40})['x']

            Omega_rated  = min([U_rated * tsr / R_tip, Omega_max])
            Omega[i:]    = np.minimum(Omega[i:], Omega_rated) # Stay at this speed if hit rated too early
            Omega_rpm    = Omega * 30. / np.pi
            myout, _     = self.ccblade.evaluate([U_rated], [Omega_rpm[i]], [pitch[i]], coefficients=True)
            P_aero[i], T[i], Q[i], M[i], Cp_aero[i], Ct_aero[i], Cq_aero[i], Cm_aero[i] = [myout[key] for key in ['P','T','Q','M','CP','CT','CQ','CM']]
            P[i], eff    = compute_P_and_eff(P_aero[i], P_rated, driveType, driveEta)
            Cp[i]        = Cp_aero[i]*eff
            P[i]         = P_rated
            
        # Store rated speed in array
        Uhub[i_rated] = U_rated

        # Store outputs
        outputs['rated_V']     = np.float64(U_rated)
        outputs['rated_Omega'] = Omega_rpm[i]
        outputs['rated_pitch'] = pitch[i]
        outputs['rated_T']     = T[i]
        outputs['rated_Q']     = Q[i]

        # JPJ: this part can be converted into a BalanceComp with a solver.
        # This will be less expensive and allow us to get derivatives through the process.
        if region3:
            # Function to be used to stay at rated power in Region 3
            def rated_power_dist(pitch, Uhub, Omega_rpm):
                myout, _ = self.ccblade.evaluate([Uhub], [Omega_rpm], [pitch], coefficients=False)
                P_aero = myout['P']
                P, eff          = compute_P_and_eff(P_aero, P_rated, driveType, driveEta)
                return (P - P_rated)

            # Solve for Region 3 pitch
            options = {'disp':False}
            if self.regulation_reg_III:
                for i in range(i_3, self.n_pc):
                    pitch0   = pitch[i-1]
                    try:
                        pitch[i] = brentq(lambda x: rated_power_dist(x, Uhub[i], Omega_rpm[i]), pitch0, pitch0+10.,
                                          xtol = 1e-4, rtol = 1e-5, maxiter=40, disp=False)
                    except ValueError:
                        pitch[i] = minimize_scalar(lambda x: np.abs(rated_power_dist(x, Uhub[i], Omega_rpm[i])), bounds=[pitch0-5., pitch0+15.],
                                                   method='bounded', options={'disp':False, 'xatol':1e-3, 'maxiter':40})['x']

                    myout, _   = self.ccblade.evaluate([Uhub[i]], [Omega_rpm[i]], [pitch[i]], coefficients=True)
                    P_aero[i], T[i], Q[i], M[i], Cp_aero[i], Ct_aero[i], Cq_aero[i], Cm_aero[i] = [myout[key] for key in ['P','T','Q','M','CP','CT','CQ','CM']]
                    P[i], eff  = compute_P_and_eff(P_aero[i], P_rated, driveType, driveEta)
                    Cp[i]      = Cp_aero[i]*eff
                    #P[i]       = P_rated

            else:
                P[i_3:]       = P_rated
                T[i_3:]       = 0
                Q[i_3:]       = P[i_3:] / Omega[i_3:]
                M[i_3:]       = 0
                pitch[i_3:]   = 0
                Cp[i_3:]      = P[i_3:] / (0.5 * inputs['rho'] * np.pi * R_tip**2 * Uhub[i_3:]**3)
                Ct_aero[i_3:] = 0
                Cq_aero[i_3:] = 0
                Cm_aero[i_3:] = 0

                    
        outputs['T']       = T
        outputs['Q']       = Q
        outputs['Omega']   = Omega_rpm

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
        tsr_vec = Omega_rpm / 30. * np.pi *  R_tip / Uhub
        id_regII = np.argmin(abs(tsr_vec - inputs['tsr_operational']))
        loads, derivs = self.ccblade.distributedAeroLoads(Uhub[id_regII], Omega_rpm[id_regII], pitch[id_regII], 0.0)
        
        # outputs
        outputs['ax_induct_regII']   = loads['a']
        outputs['tang_induct_regII'] = loads['ap']
        outputs['aoa_regII']         = loads['alpha']
        outputs['cl_regII']          = loads['Cl']
        outputs['cd_regII']          = loads['Cd']
        outputs['Cp_regII']          = Cp_aero[id_regII]
        
        
class ComputeSplines(ExplicitComponent):
    """
    Compute splined quantities for V, P, and Omega.
    """
    
    def initialize(self):
        self.options.declare('modeling_options')

    def setup(self):
        modeling_options = self.options['modeling_options']
        self.n_pc          = modeling_options['servose']['n_pc']
        self.n_pc_spline   = modeling_options['servose']['n_pc_spline']
        
        self.add_input('v_min',        val=0.0, units='m/s',  desc='cut-in wind speed')
        self.add_input('v_max',       val=0.0, units='m/s',  desc='cut-out wind speed')
        self.add_input('V',        val=np.zeros(self.n_pc), units='m/s',        desc='wind vector')
        self.add_input('Omega',    val=np.zeros(self.n_pc), units='rpm',        desc='rotor rotational speed')
        self.add_input('P',        val=np.zeros(self.n_pc), units='W',          desc='rotor electrical power')

        self.add_output('V_spline', val=np.zeros(self.n_pc_spline), units='m/s', desc='wind vector')
        self.add_output('P_spline', val=np.zeros(self.n_pc_spline), units='W',   desc='rotor electrical power')
        self.add_output('Omega_spline', val=np.zeros(self.n_pc_spline), units='rpm',   desc='omega')
        
        self.declare_partials(of='V_spline', wrt='v_min')
        self.declare_partials(of='V_spline', wrt='v_max')
        
        self.declare_partials(of='P_spline', wrt='v_min', method='fd')
        self.declare_partials(of='P_spline', wrt='v_max', method='fd')
        self.declare_partials(of='P_spline', wrt='V', method='fd')
        self.declare_partials(of='P_spline', wrt='P', method='fd')
        
        self.declare_partials(of='Omega_spline', wrt='v_min', method='fd')
        self.declare_partials(of='Omega_spline', wrt='v_max', method='fd')
        self.declare_partials(of='Omega_spline', wrt='V', method='fd')
        self.declare_partials(of='Omega_spline', wrt='Omega', method='fd')
    
    def compute(self, inputs, outputs):
        # Fit spline to powercurve for higher grid density
        V_spline = np.linspace(inputs['v_min'], inputs['v_max'], self.n_pc_spline)
        spline   = PchipInterpolator(inputs['V'], inputs['P'])
        P_spline = spline(V_spline)
        spline   = PchipInterpolator(inputs['V'], inputs['Omega'])
        Omega_spline = spline(V_spline)
        
        # outputs
        outputs['V_spline']          = V_spline.flatten()
        outputs['P_spline']          = P_spline.flatten()
        outputs['Omega_spline']      = Omega_spline.flatten()
        
    def compute_partials(self, inputs, partials):
        linspace_with_deriv
        V_spline, dy_dstart, dy_dstop = linspace_with_deriv(inputs['v_min'], inputs['v_max'], self.n_pc_spline)
        partials['V_spline', 'v_min'] = dy_dstart
        partials['V_spline', 'v_max'] = dy_dstop
        
class Cp_Ct_Cq_Tables(ExplicitComponent):
    def initialize(self):
        self.options.declare('modeling_options')
        # self.options.declare('n_span')
        # self.options.declare('n_pitch', default=20)
        # self.options.declare('n_tsr', default=20)
        # self.options.declare('n_U', default=1)
        # self.options.declare('n_aoa')
        # self.options.declare('n_re')

    def setup(self):
        modeling_options = self.options['modeling_options']
        blade_init_options = modeling_options['blade']
        servose_init_options = modeling_options['servose']
        airfoils = modeling_options['airfoils']
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
        # self.add_discrete_input('airfoils',      val=[0]*n_span,                 desc='CCAirfoil instances')
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
            if self.n_tab > 1:
                ref_tab = int(np.floor(self.n_tab/2))
                af[i] = CCAirfoil(inputs['airfoils_aoa'], inputs['airfoils_Re'], inputs['airfoils_cl'][i,:,:,ref_tab], inputs['airfoils_cd'][i,:,:,ref_tab], inputs['airfoils_cm'][i,:,:,ref_tab])
            else:
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
                myout, _  = self.ccblade.evaluate(U, Omega, pitch_vector, coefficients=True)
                outputs['Cp'][j,:,i], outputs['Ct'][j,:,i], outputs['Cq'][j,:,i] = [myout[key] for key in ['CP','CT','CQ']]

# Class to define a constraint so that the blade cannot operate in stall conditions
class NoStallConstraint(ExplicitComponent):
    def initialize(self):
        
        self.options.declare('modeling_options')
    
    def setup(self):
        
        modeling_options = self.options['modeling_options']
        self.n_span        = n_span    = modeling_options['blade']['n_span']
        self.n_aoa         = n_aoa     = modeling_options['airfoils']['n_aoa']# Number of angle of attacks
        self.n_Re          = n_Re      = modeling_options['airfoils']['n_Re'] # Number of Reynolds, so far hard set at 1
        self.n_tab         = n_tab     = modeling_options['airfoils']['n_tab']# Number of tabulated data. For distributed aerodynamic control this could be > 1
        
        self.add_input('s',                     val=np.zeros(n_span),                 desc='1D array of the non-dimensional spanwise grid defined along blade axis (0-blade root, 1-blade tip)')
        self.add_input('aoa_along_span',        val=np.zeros(n_span), units = 'deg', desc = 'Angle of attack along blade span')
        self.add_input('stall_margin',          val=3.0,            units = 'deg', desc = 'Minimum margin from the stall angle')
        self.add_input('min_s',                 val=0.25,            desc = 'Minimum nondimensional coordinate along blade span where to define the constraint (blade root typically stalls)')
        self.add_input('airfoils_cl',           val=np.zeros((n_span, n_aoa, n_Re, n_tab)), desc='lift coefficients, spanwise')
        self.add_input('airfoils_cd',           val=np.zeros((n_span, n_aoa, n_Re, n_tab)), desc='drag coefficients, spanwise')
        self.add_input('airfoils_cm',           val=np.zeros((n_span, n_aoa, n_Re, n_tab)), desc='moment coefficients, spanwise')
        self.add_input('airfoils_aoa',          val=np.zeros((n_aoa)), units='deg', desc='angle of attack grid for polars')
        
        self.add_output('no_stall_constraint',   val=np.zeros(n_span), desc = 'Constraint, ratio between angle of attack plus a margin and stall angle')
        self.add_output('stall_angle_along_span',val=np.zeros(n_span), units = 'deg', desc = 'Stall angle along blade span')

    def compute(self, inputs, outputs):
        
        verbosity = True
        
        i_min = np.argmin(abs(inputs['min_s'] - inputs['s']))
        
        for i in range(self.n_span):
            unsteady = eval_unsteady(inputs['airfoils_aoa'], inputs['airfoils_cl'][i,:,0,0], inputs['airfoils_cd'][i,:,0,0], inputs['airfoils_cm'][i,:,0,0])
            outputs['stall_angle_along_span'][i] = unsteady['alpha1']
            if outputs['stall_angle_along_span'][i] == 0:
                outputs['stall_angle_along_span'][i] = 1e-6 # To avoid nan
        
        for i in range(i_min, self.n_span):
            outputs['no_stall_constraint'][i] = (inputs['aoa_along_span'][i] + inputs['stall_margin']) / outputs['stall_angle_along_span'][i]
        
            if verbosity == True:
                if outputs['no_stall_constraint'][i] > 1:
                    print('Blade is stalling at span location %.2f %%' % (inputs['s'][i]*100.))


class AEP(ExplicitComponent):
    def initialize(self):

        self.options.declare('nspline')
    
    def setup(self):
        n_pc_spline      = self.options['nspline']
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

def compute_P_and_eff(aeroPower, ratedPower, drivetrainType, drivetrainEff):

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
        elif drivetrainType == 'DIRECT DRIVE':
            constant = 0.00
            linear = 0.07
            quadratic = 0.0
        else:
            exit('The drivetrain model is not supported! Please check servose.py')
        
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
    
