'''
Controller tuning script.

Nikhar J. Abbas
January 2020
'''

from ROSCO_toolbox import controller as ROSCO_controller
from ROSCO_toolbox import turbine as ROSCO_turbine
from ROSCO_toolbox import utilities as ROSCO_utilities
import numpy as np
from openmdao.api import ExplicitComponent, Group
from wisdem.ccblade import CCAirfoil, CCBlade

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

        # self.add_output('PC_GS_angles', val=np.zeros(n_pitch+1), units='rad', desc='Gain-schedule table: pitch angles')
        # self.add_output('PC_GS_KP',     val=np.zeros(n_pitch+1),              desc='Gain-schedule table: pitch controller kp gains')
        # self.add_output('PC_GS_KI',     val=np.zeros(n_pitch+1),              desc='Gain-schedule table: pitch controller ki gains')
        # self.add_output('VS_Rgn2K',     val=0.0, units='N*m/(rad/s)**2',      desc='Generator torque constant in Region 2 (HSS side), [N-m/(rad/s)^2]')

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
        outputs['Flp_Kp']   = controller.Kp_flap[-1]
        outputs['Flp_Ki']   = controller.Ki_flap[-1]


        # outputs['PC_GS_angles'] = controller.pitch_op_pc
        # outputs['PC_GS_KP']     = controller.pc_gain_schedule.Kp
        # outputs['PC_GS_KI']     = controller.pc_gain_schedule.Ki
        # outputs['VS_Rgn2K']     = controller.vs_rgn2K
 
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