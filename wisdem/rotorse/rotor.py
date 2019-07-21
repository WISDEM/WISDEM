#!/usr/bin/env python
# encoding: utf-8
"""
rotor.py

Created by Andrew Ning on 2012-02-28.
Copyright (c)  NREL. All rights reserved.
"""

# from __future__ import print_function
import numpy as np
import os, time
from openmdao.api import IndepVarComp, Group, Problem, ExecComp
from rotor_aeropower import RotorAeroPower
from rotor_structure import RotorStructure
from rotor_geometry import RotorGeometry, NREL5MW, DTU10MW, TUM3_35MW, NINPUT
from rotorse import RPM2RS, RS2RPM, TURBULENCE_CLASS, DRIVETRAIN_TYPE

try:
    from AeroelasticSE.FAST_reader import InputReader_Common, InputReader_OpenFAST, InputReader_FAST7
    from AeroelasticSE.FAST_writer import InputWriter_Common, InputWriter_OpenFAST, InputWriter_FAST7
    from AeroelasticSE.FAST_wrapper import FastWrapper
    from AeroelasticSE.runFAST_pywrapper import runFAST_pywrapper, runFAST_pywrapper_batch
    from AeroelasticSE.CaseGen_IEC import CaseGen_IEC
    from AeroelasticSE.CaseLibrary import RotorSE_rated, RotorSE_DLC_1_4_Rated, RotorSE_DLC_7_1_Steady, RotorSE_DLC_1_1_Turb
except:
    pass

class RotorSE(Group):
    def initialize(self):
        self.options.declare('RefBlade')
        self.options.declare('npts_coarse_power_curve', default=20)
        self.options.declare('npts_spline_power_curve', default=200)
        self.options.declare('regulation_reg_II5',default=True)
        self.options.declare('regulation_reg_III',default=True)
        self.options.declare('Analysis_Level',default=0)
        self.options.declare('FASTpref',default={})
        self.options.declare('topLevelFlag',default=False)
    
    def setup(self):
        RefBlade                = self.options['RefBlade']
        npts_coarse_power_curve = self.options['npts_coarse_power_curve']
        npts_spline_power_curve = self.options['npts_spline_power_curve']
        regulation_reg_II5      = self.options['regulation_reg_II5']
        regulation_reg_III      = self.options['regulation_reg_III']
        Analysis_Level          = self.options['Analysis_Level']
        FASTpref                = self.options['FASTpref']
        topLevelFlag            = self.options['topLevelFlag']

        rotorIndeps = IndepVarComp()
        rotorIndeps.add_discrete_output('tiploss', True)
        rotorIndeps.add_discrete_output('hubloss', True)
        rotorIndeps.add_discrete_output('wakerotation', True)
        rotorIndeps.add_discrete_output('usecd', True)
        rotorIndeps.add_discrete_output('nSector', val=4, desc='number of sectors to divide rotor face into in computing thrust and power')
        self.add_subsystem('rotorIndeps', rotorIndeps, promotes=['*'])

        if topLevelFlag:
            sharedIndeps = IndepVarComp()
            sharedIndeps.add_output('lifetime', val=20.0, units='year', desc='project lifetime for fatigue analysis')
            sharedIndeps.add_output('hub_height', val=0.0, units='m')
            sharedIndeps.add_output('rho', val=1.225, units='kg/m**3')
            sharedIndeps.add_output('mu', val=1.81e-5, units='kg/(m*s)')
            sharedIndeps.add_output('shearExp', val=0.2)
            self.add_subsystem('sharedIndeps', sharedIndeps, promotes=['*'])
                
        # --- Rotor Aero & Power ---
        #self.add_subsystem('rg', RotorGeometry(RefBlade=RefBlade, topLevelFlag=True), promotes=['*'])
        self.add_subsystem('ra', RotorAeroPower(RefBlade=RefBlade,
                                                npts_coarse_power_curve=npts_coarse_power_curve,
                                                npts_spline_power_curve=npts_spline_power_curve,
                                                regulation_reg_II5=regulation_reg_II5,
                                                regulation_reg_III=regulation_reg_III,
                                                topLevelFlag=False), promotes=['*'])
        self.add_subsystem('rs', RotorStructure(RefBlade=RefBlade,
                                                npts_coarse_power_curve=npts_coarse_power_curve,
                                                Analysis_Level=Analysis_Level,
                                                FASTpref=FASTpref,
                                                topLevelFlag=False), 
                           promotes=['fst_vt_in','VfactorPC','turbulence_class','gust_stddev','pitch_extreme',
                                     'azimuth_extreme','rstar_damage','Mxb_damage','Myb_damage',
                                     'strain_ult_spar','strain_ult_te','m_damage',
                                     'gamma_fatigue','gamma_freq','gamma_f','gamma_m',
                                     'dynamic_amplification_tip_deflection',
                                     'pitch_load89','azimuth_load0','azimuth_load120','azimuth_load240',
                                     'nSector','rho','mu','shearExp','tiploss','hubloss','wakerotation','usecd',
                                     'bladeLength','hubFraction','r_max_chord','chord_in','theta_in',
                                     'precurve_in','presweep_in','precurveTip','presweepTip','precone',
                                     'tilt','yaw','nBlades','downwind','sparT_in','teT_in','turbine_class',
                                     'control_tsr','control_pitch','lifetime','hubHt',
                                     'mass_one_blade','mass_all_blades','I_all_blades',
                                     'freq','freq_curvefem','modes_coef_curvefem','tip_deflection', 
                                     'tip_position','ground_clearance','strainU_spar','strainL_spar',
                                     'strainU_te','strainL_te','eps_crit_spar','eps_crit_te',
                                     'root_bending_moment','Mxyz','damageU_spar','damageL_spar','damageU_te',
                                     'damageL_te','delta_bladeLength_out','delta_precurve_sub_out',
                                     'Fxyz_1','Fxyz_2','Fxyz_3','Fxyz_4','Fxyz_5','Fxyz_6',
                                     'Mxyz_1','Mxyz_2','Mxyz_3','Mxyz_4','Mxyz_5','Mxyz_6',
                                     'Fxyz_total','Mxyz_total','TotalCone','Pitch'])
        
        self.add_subsystem('obj_cmp', ExecComp('obj = -AEP',
                                               AEP={'units':'kW*h','value':1000000.0},
                                               obj={'units':'kW*h'}), promotes=['*'])

        self.connect('hub_height','hubHt')
        # Connections between rotor_aero and rotor_structure
        self.connect('powercurve.rated_V', ['rs.gust.V_hub', 'rs.setuppc.Vrated'])
        self.connect('powercurve.rated_Omega', ['rs.Omega', 'rs.aero_rated.Omega_load',
                                                'rs.curvefem.Omega','rs.aero_0.Omega_load',
                                                'rs.aero_120.Omega_load','rs.aero_240.Omega_load'])
        self.connect('powercurve.rated_pitch', 'rs.aero_rated.pitch_load')


        
if __name__ == '__main__':
    myref = NREL5MW() 
    # myref = DTU10MW()
    # myref = TUM3_35MW()

    Analysis_Level = 0 # 0: Run CCBlade; 1: Update FAST model at each iteration but do not run; 2: Run FAST w/ ElastoDyn; 3: (Not implemented) Run FAST w/ BeamDyn

    # Set FAST Inputs
    if Analysis_Level >= 1:
        # File management
        FASTpref                        = {}
        FASTpref['Analysis_Level']      = Analysis_Level
        FASTpref['FAST_ver']            = 'OpenFAST'
        FASTpref['dev_branch']          = True
        FASTpref['FAST_exe']            = '/mnt/c/Material/Programs/openfast/build/glue-codes/openfast/openfast'
        FASTpref['FAST_directory']      = '/mnt/c/Material/Programs/xloads_tc/templates/openfast/5MW_Land_DLL_WTurb-NoAero'   # Path to fst directory files
        FASTpref['Turbsim_exe']         = '/mnt/c/Material/Programs/TurbSim/TurbSim_glin64'
        FASTpref['FAST_namingOut']      = 'RotorSE_FAST_'+myref.name
        FASTpref['FAST_runDirectory']   = 'temp/' + FASTpref['FAST_namingOut']
        FASTpref['FAST_InputFile']      = '5MW_Land_DLL_WTurb.fst' # FAST input file (ext=.fst)

        # Run Settings
        FASTpref['cores']               = 1
        FASTpref['debug_level']         = 2 # verbosity: set to 0 for quiet, 1 & 2 for increasing levels of output

        # DLCs
        FASTpref['DLC_powercurve']      = None      # AEP               ### Not in place yet
        # FASTpref['DLC_gust']            = None      # Max deflection
        # FASTpref['DLC_extrm']           = None      # Max strain
        FASTpref['DLC_gust']            = RotorSE_DLC_1_4_Rated       # Max deflection    ### Not in place yet
        FASTpref['DLC_extrm']           = RotorSE_DLC_7_1_Steady      # Max strain        ### Not in place yet
        # FASTpref['DLC_turbulent']       = RotorSE_DLC_1_1_Turb      # Alternate turbulent case, replacing rated and extreme DLCs for calculating max deflection and strain
        FASTpref['DLC_turbulent']       = None

        # Initialize, read initial FAST files to avoid doing it iteratively
        fast = InputReader_OpenFAST(FAST_ver=FASTpref['FAST_ver'], dev_branch=FASTpref['dev_branch'])
        fast.FAST_InputFile = FASTpref['FAST_InputFile']
        fast.FAST_directory = FASTpref['FAST_directory']
        fast.execute()
    else:
        FASTpref = {}


    rotor = Problem()
    npts_coarse_power_curve = 20 # (Int): number of points to evaluate aero analysis at
    npts_spline_power_curve = 200  # (Int): number of points to use in fitting spline to power curve
    regulation_reg_II5 = False # calculate Region 2.5 pitch schedule, False will not maximize power in region 2.5
    regulation_reg_III = True # calculate Region 3 pitch schedule, False will return erroneous Thrust, Torque, and Moment for above rated

    rotor.model = RotorSE(RefBlade=myref,
                          npts_coarse_power_curve=npts_coarse_power_curve,
                          npts_spline_power_curve=npts_spline_power_curve,
                          regulation_reg_II5=regulation_reg_II5,
                          regulation_reg_III=regulation_reg_III, 
                          Analysis_Level=Analysis_Level,
                          FASTpref=FASTpref,
                          topLevelFlag=True)
    rotor.setup()

    # === FAST model ===
    if Analysis_Level >= 1:
        rotor['fst_vt_in'] = fast.fst_vt
    if Analysis_Level > 1:
        rotor['drivetrainEff'] = fast.fst_vt['ServoDyn']['GenEff']/100.

    # === blade grid ===
    rotor['hubFraction'] = myref.hubFraction #0.025  # (Float): hub location as fraction of radius
    rotor['bladeLength'] = myref.bladeLength #61.5  # (Float, m): blade length (if not precurved or swept) otherwise length of blade before curvature
    # rotor['delta_bladeLength'] = 0.0  # (Float, m): adjustment to blade length to account for curvature from loading
    rotor['precone'] = myref.precone #2.5  # (Float, deg): precone angle
    rotor['tilt'] = myref.tilt #5.0  # (Float, deg): shaft tilt
    rotor['yaw'] = 0.0  # (Float, deg): yaw error
    rotor['nBlades'] = myref.nBlades #3  # (Int): number of blades
    # ------------------
    
    # === blade geometry ===
    rotor['r_max_chord'] =  myref.r_max_chord  # 0.23577 #(Float): location of max chord on unit radius
    rotor['chord_in'] = myref.chord # np.array([3.2612, 4.3254, 4.5709, 3.7355, 2.69923333, 1.4621])  # (Array, m): chord at control points. defined at hub, then at linearly spaced locations from r_max_chord to tip
    rotor['theta_in'] = myref.theta # np.array([0.0, 13.2783, 12.30514836,  6.95106536,  2.72696309, -0.0878099]) # (Array, deg): twist at control points.  defined at linearly spaced locations from r[idx_cylinder] to tip
    rotor['precurve_in'] = myref.precurve #np.array([0.0, 0.0, 0.0])  # (Array, m): precurve at control points.  defined at same locations at chord, starting at 2nd control point (root must be zero precurve)
    rotor['precurveTip'] = myref.precurveT #np.array([0.0, 0.0, 0.0])  # (Array, m): precurve at control points.  defined at same locations at chord, starting at 2nd control point (root must be zero precurve)
    rotor['presweep_in'] = myref.presweep #np.array([0.0, 0.0, 0.0])  # (Array, m): precurve at control points.  defined at same locations at chord, starting at 2nd control point (root must be zero precurve)
    # rotor['delta_precurve_in'] = np.array([0.0, 0.0, 0.0])  # (Array, m): adjustment to precurve to account for curvature from loading
    rotor['sparT_in'] = myref.spar_thickness # np.array([0.0, 0.05, 0.047754, 0.045376, 0.031085, 0.0061398])  # (Array, m): spar cap thickness parameters
    rotor['teT_in'] = myref.te_thickness # np.array([0.0, 0.1, 0.09569, 0.06569, 0.02569, 0.00569])  # (Array, m): trailing-edge thickness parameters
    # ------------------

    # === atmosphere ===
    rotor['rho'] = 1.225  # (Float, kg/m**3): density of air
    rotor['mu'] = 1.81206e-5  # (Float, kg/m/s): dynamic viscosity of air
    rotor['shearExp'] = 0.25  # (Float): shear exponent
    rotor['hub_height'] = myref.hubHt #90.0  # (Float, m): hub height
    rotor['turbine_class'] = myref.turbine_class #TURBINE_CLASS['I']  # (Enum): IEC turbine class
    rotor['turbulence_class'] = TURBULENCE_CLASS['B']  # (Enum): IEC turbulence class class
    rotor['cdf_wind_speed_reference_height'] = myref.hubHt #90.0  # (Float): reference hub height for IEC wind speed (used in CDF calculation)
    rotor['gust_stddev'] = 3
    # ----------------------

    # === control ===
    rotor['control_Vin'] = myref.control_Vin #3.0  # (Float, m/s): cut-in wind speed
    rotor['control_Vout'] = myref.control_Vout #25.0  # (Float, m/s): cut-out wind speed
    rotor['control_minOmega'] = myref.control_minOmega #0.0  # (Float, rpm): minimum allowed rotor rotation speed
    rotor['control_maxOmega'] = myref.control_maxOmega #12.0  # (Float, rpm): maximum allowed rotor rotation speed
    rotor['control_tsr'] = myref.control_tsr #7.55  # (Float): tip-speed ratio in Region 2 (should be optimized externally)
    rotor['control_pitch'] = myref.control_pitch #0.0  # (Float, deg): pitch angle in region 2 (and region 3 for fixed pitch machines)
    rotor['control_maxTS'] = myref.control_maxTS
    rotor['machine_rating'] = myref.rating #5e6  # (Float, W): rated power
    rotor['pitch_extreme'] = 0.0  # (Float, deg): worst-case pitch at survival wind condition
    rotor['azimuth_extreme'] = 0.0  # (Float, deg): worst-case azimuth at survival wind condition
    rotor['VfactorPC'] = 0.7  # (Float): fraction of rated speed at which the deflection is assumed to representative throughout the power curve calculation
    # ----------------------

    # === aero and structural analysis options ===
    rotor['nSector'] = 4  # (Int): number of sectors to divide rotor face into in computing thrust and power
    rotor['AEP_loss_factor'] = 1.0  # (Float): availability and other losses (soiling, array, etc.)
    rotor['drivetrainType'] = myref.drivetrain #DRIVETRAIN_TYPE['GEARED']  # (Enum)
    if Analysis_Level > 1:
        rotor['dynamic_amplification_tip_deflection'] = 1.
    else:
        rotor['dynamic_amplification_tip_deflection'] = 1.35  # (Float): a dynamic amplification factor to adjust the static deflection calculation
    # ----------------------

    # === fatigue ===
    r_aero = np.array([0.02222276, 0.06666667, 0.11111057, 0.2, 0.23333333, 0.3, 0.36666667, 0.43333333,
	               0.5, 0.56666667, 0.63333333, 0.64, 0.7, 0.83333333, 0.88888943, 0.93333333,
	               0.97777724])  # (Array): new aerodynamic grid on unit radius
    rstar_damage = np.array([0.000, 0.022, 0.067, 0.111, 0.167, 0.233, 0.300, 0.367, 0.433, 0.500,
        0.567, 0.633, 0.700, 0.767, 0.833, 0.889, 0.933, 0.978])  # (Array): nondimensional radial locations of damage equivalent moments
    Mxb_damage = 1e3*np.array([2.3743E+003, 2.0834E+003, 1.8108E+003, 1.5705E+003, 1.3104E+003,
        1.0488E+003, 8.2367E+002, 6.3407E+002, 4.7727E+002, 3.4804E+002, 2.4458E+002, 1.6339E+002,
        1.0252E+002, 5.7842E+001, 2.7349E+001, 1.1262E+001, 3.8549E+000, 4.4738E-001])  # (Array, N*m): damage equivalent moments about blade c.s. x-direction
    Myb_damage = 1e3*np.array([2.7732E+003, 2.8155E+003, 2.6004E+003, 2.3933E+003, 2.1371E+003,
        1.8459E+003, 1.5582E+003, 1.2896E+003, 1.0427E+003, 8.2015E+002, 6.2449E+002, 4.5229E+002,
        3.0658E+002, 1.8746E+002, 9.6475E+001, 4.2677E+001, 1.5409E+001, 1.8426E+000])  # (Array, N*m): damage equivalent moments about blade c.s. y-direction
    xp = np.r_[0.0, r_aero]
    xx = np.r_[0.0, myref.r]
    rotor['rstar_damage'] = np.interp(xx, xp, rstar_damage)
    rotor['Mxb_damage'] = np.interp(xx, xp, Mxb_damage)
    rotor['Myb_damage'] = np.interp(xx, xp, Myb_damage)
    rotor['strain_ult_spar'] = 1.0e-2  # (Float): ultimate strain in spar cap
    rotor['strain_ult_te'] = 2500*1e-6 * 2   # (Float): uptimate strain in trailing-edge panels, note that I am putting a factor of two for the damage part only.
    rotor['gamma_fatigue'] = 1.755 # (Float): safety factor for fatigue
    rotor['gamma_f'] = 1.35 # (Float): safety factor for loads/stresses
    rotor['gamma_m'] = 1.1 # (Float): safety factor for materials
    rotor['gamma_freq'] = 1.1 # (Float): safety factor for resonant frequencies
    rotor['m_damage'] = 10.0  # (Float): slope of S-N curve for fatigue analysis
    rotor['lifetime'] = 20.0  # (Float): number of cycles used in fatigue analysis  TODO: make function of rotation speed
    # ----------------

    # from myutilities import plt

    # === run and outputs ===
    tt = time.time()
    rotor.run_driver()

    print('Run Time = ', time.time()-tt)
    print('AEP =', rotor['AEP'])
    print('diameter =', rotor['diameter'])
    print('ratedConditions.V =', rotor['rated_V'])
    print('ratedConditions.Omega =', rotor['rated_Omega'])
    print('ratedConditions.pitch =', rotor['rated_pitch'])
    print('ratedConditions.T =', rotor['rated_T'])
    print('ratedConditions.Q =', rotor['rated_Q'])
    print('mass_one_blade =', rotor['mass_one_blade'])
    print('mass_all_blades =', rotor['mass_all_blades'])
    print('I_all_blades =', rotor['I_all_blades'])
    print('freq =', rotor['freq'])
    print('tip_deflection =', rotor['tip_deflection'])
    print('root_bending_moment =', rotor['root_bending_moment'])
    #for io in rotor.model.unknowns:
    #    print(io + ' ' + str(rotor.model.unknowns[io]))
    '''
    print('Pn_margin', rotor[ 'Pn_margin'])
    print('P1_margin', rotor[ 'P1_margin'])
    print('Pn_margin_cfem', rotor[ 'Pn_margin_cfem'])
    print('P1_margin_cfem', rotor[ 'P1_margin_cfem'])
    print('rotor_strain_sparU', rotor[ 'rotor_strain_sparU'])
    print('rotor_strain_sparL', rotor[ 'rotor_strain_sparL'])
    print('rotor_strain_teU', rotor[ 'rotor_strain_teU'])
    print('rotor_strain_teL', rotor[ 'rotor_strain_teL'])
    print('eps_crit_spar', rotor['eps_crit_spar'])
    print('strain_ult_spar', rotor['strain_ult_spar'])
    print('eps_crit_te', rotor['eps_crit_te'])
    print('strain_ult_te', rotor['strain_ult_te'])
    print('rotor_buckling_sparU', rotor[ 'rotor_buckling_sparU'])
    print('rotor_buckling_sparL', rotor[ 'rotor_buckling_sparL'])
    print('rotor_buckling_teU', rotor[ 'rotor_buckling_teU'])
    print('rotor_buckling_teL', rotor[ 'rotor_buckling_teL'])
    print('rotor_damage_sparU', rotor[ 'rotor_damage_sparU'])
    print('rotor_damage_sparL', rotor[ 'rotor_damage_sparL'])
    print('rotor_damage_teU', rotor[ 'rotor_damage_teU'])
    print('rotor_damage_teL', rotor[ 'rotor_damage_teL'])
    '''

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(rotor['V'], rotor['P']/1e6)
    plt.xlabel('wind speed (m/s)')
    plt.xlabel('power (W)')

    plt.figure()

    plt.plot(rotor['r_pts'], rotor['strainU_spar'], label='suction')
    plt.plot(rotor['r_pts'], rotor['strainL_spar'], label='pressure')
    plt.plot(rotor['r_pts'], rotor['eps_crit_spar'], label='critical')
    plt.ylim([-5e-3, 5e-3])
    plt.xlabel('r')
    plt.ylabel('strain')
    plt.legend()
    # plt.savefig('/Users/sning/Desktop/strain_spar.pdf')
    # plt.savefig('/Users/sning/Desktop/strain_spar.png')

    plt.figure()

    plt.plot(rotor['r_pts'], rotor['strainU_te'], label='suction')
    plt.plot(rotor['r_pts'], rotor['strainL_te'], label='pressure')
    plt.plot(rotor['r_pts'], rotor['eps_crit_te'], label='critical')
    plt.ylim([-5e-3, 5e-3])
    plt.xlabel('r')
    plt.ylabel('strain')
    plt.legend()
    # plt.savefig('/Users/sning/Desktop/strain_te.pdf')
    # plt.savefig('/Users/sning/Desktop/strain_te.png')

    plt.show()
    # ----------------
