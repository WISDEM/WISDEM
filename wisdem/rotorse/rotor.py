#!/usr/bin/env python
# encoding: utf-8
"""
rotor.py

Created by Andrew Ning on 2012-02-28.
Copyright (c)  NREL. All rights reserved.
"""

from __future__ import print_function
import numpy as np
import os, time, shutil, copy
from openmdao.api import IndepVarComp, ExplicitComponent, Group, Problem, ExecComp
from wisdem.rotorse.rotor_aeropower import RotorAeroPower
from wisdem.rotorse.rotor_structure import RotorStructure
from wisdem.rotorse.rotor_geometry import RotorGeometry
from wisdem.rotorse.rotor_geometry_yaml import ReferenceBlade
from wisdem.rotorse import RPM2RS, RS2RPM
#from wisdem.rotorse.rotor_fast import FASTLoadCases


class RotorSE(Group):
    def initialize(self):
        self.options.declare('RefBlade')
        self.options.declare('npts_coarse_power_curve', default=20)
        self.options.declare('npts_spline_power_curve', default=200)
        self.options.declare('regulation_reg_II5',default=True)
        self.options.declare('regulation_reg_III',default=True)
        self.options.declare('Analysis_Level',default=0)
        self.options.declare('FASTpref',default={})
        self.options.declare('flag_nd_opt',default=False)
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
                                                topLevelFlag=False,
                                                Analysis_Level=Analysis_Level), 
                           promotes=['fst_vt_in','VfactorPC','turbulence_class','gust_stddev','pitch_extreme',
                                     'azimuth_extreme','rstar_damage','Mxb_damage','Myb_damage',
                                     'strain_ult_spar','strain_ult_te','m_damage',
                                     'gamma_fatigue','gamma_freq','gamma_f','gamma_m','dynamic_amplification',
                                     'azimuth_load180','azimuth_load0','azimuth_load120','azimuth_load240',
                                     'nSector','rho','mu','shearExp','tiploss','hubloss','wakerotation','usecd',
                                     'bladeLength','hubFraction','r_max_chord','chord_in','theta_in',
                                     'precurve_in','presweep_in','precurveTip','presweepTip','precone',
                                     'tilt','yaw','nBlades','downwind','sparT_in','teT_in','airfoil_position','turbine_class',
                                     'control_tsr','control_pitch','lifetime','hub_height',
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

        # Connections between rotor_aero and rotor_structure
        self.connect('powercurve.rated_V', ['rs.gust.V_hub', 'rs.setuppc.Vrated'])
        self.connect('powercurve.rated_Omega', ['rs.Omega', 'rs.aero_rated.Omega_load',
                                                'rs.curvefem.Omega','rs.aero_rated_0.Omega_load',
                                                'rs.aero_rated_120.Omega_load','rs.aero_rated_240.Omega_load'])
        self.connect('powercurve.rated_pitch', 'rs.aero_rated.pitch_load')



def Init_RotorSE_wRefBlade(rotor, blade, fst_vt={}):

    Analysis_Level = rotor.model.options['Analysis_Level']

    # === FAST model ===
    if Analysis_Level >= 1:
        rotor['fst_vt_in'] = fst_vt
    if Analysis_Level > 1:
        rotor['drivetrainEff'] = fst_vt['ServoDyn']['GenEff']/100.

    # === blade grid ===
    rotor['hubFraction'] = blade['config']['hubD']/2./blade['ctrl_pts']['bladeLength'] #0.025  # (Float): hub location as fraction of radius
    rotor['bladeLength'] = blade['ctrl_pts']['bladeLength'] #61.5  # (Float, m): blade length (if not precurved or swept) otherwise length of blade before curvature
    # rotor['delta_bladeLength'] = 0.0  # (Float, m): adjustment to blade length to account for curvature from loading
    rotor['precone'] = blade['config']['cone_angle'] #2.5  # (Float, deg): precone angle
    rotor['tilt'] = blade['config']['tilt_angle'] #5.0  # (Float, deg): shaft tilt
    rotor['yaw'] = 0.0  # (Float, deg): yaw error
    rotor['nBlades'] = blade['config']['number_of_blades'] #3  # (Int): number of blades
    # ------------------
    
    # === blade geometry ===
    rotor['r_max_chord']      = blade['ctrl_pts']['r_max_chord']  # 0.23577 #(Float): location of max chord on unit radius
    rotor['chord_in']         = np.array(blade['ctrl_pts']['chord_in']) # np.array([3.2612, 4.3254, 4.5709, 3.7355, 2.69923333, 1.4621])  # (Array, m): chord at control points. defined at hub, then at linearly spaced locations from r_max_chord to tip
    rotor['theta_in']         = np.array(blade['ctrl_pts']['theta_in']) # np.array([0.0, 13.2783, 12.30514836,  6.95106536,  2.72696309, -0.0878099]) # (Array, deg): twist at control points.  defined at linearly spaced locations from r[idx_cylinder] to tip
    rotor['precurve_in']      = np.array(blade['ctrl_pts']['precurve_in']) #np.array([0.0, 0.0, 0.0])  # (Array, m): precurve at control points.  defined at same locations at chord, starting at 2nd control point (root must be zero precurve)
    rotor['presweep_in']      = np.array(blade['ctrl_pts']['presweep_in']) #np.array([0.0, 0.0, 0.0])  # (Array, m): precurve at control points.  defined at same locations at chord, starting at 2nd control point (root must be zero precurve)
    rotor['sparT_in']         = np.array(blade['ctrl_pts']['sparT_in']) # np.array([0.0, 0.05, 0.047754, 0.045376, 0.031085, 0.0061398])  # (Array, m): spar cap thickness parameters
    rotor['teT_in']           = np.array(blade['ctrl_pts']['teT_in']) # np.array([0.0, 0.1, 0.09569, 0.06569, 0.02569, 0.00569])  # (Array, m): trailing-edge thickness parameters
    # rotor['thickness_in']     = np.array(blade['ctrl_pts']['thickness_in'])
    rotor['airfoil_position'] = np.array(blade['outer_shape_bem']['airfoil_position']['grid'])
    # ------------------

    # === atmosphere ===
    rotor['rho']              = 1.225  # (Float, kg/m**3): density of air
    rotor['mu']               = 1.81206e-5  # (Float, kg/m/s): dynamic viscosity of air
    rotor['shearExp']         = 0.25  # (Float): shear exponent
    rotor['shape_parameter']  = 2.0
    rotor['hub_height']       = blade['config']['hub_height']  # (Float, m): hub height
    rotor['turbine_class']    = blade['config']['turbine_class'].upper() #TURBINE_CLASS['I']  # (Enum): IEC turbine class
    rotor['turbulence_class'] = blade['config']['turbulence_class'].upper()  # (Enum): IEC turbulence class class
    rotor['wind_reference_height']        = blade['config']['hub_height']
    rotor['gust_stddev']      = 3
    # ----------------------

    # === control ===
    rotor['control_Vin']      = blade['config']['Vin'] #3.0  # (Float, m/s): cut-in wind speed
    rotor['control_Vout']     = blade['config']['Vout'] #25.0  # (Float, m/s): cut-out wind speed
    rotor['control_minOmega'] = blade['config']['minOmega'] #0.0  # (Float, rpm): minimum allowed rotor rotation speed
    rotor['control_maxOmega'] = blade['config']['maxOmega'] #12.0  # (Float, rpm): maximum allowed rotor rotation speed
    rotor['control_tsr']      = blade['config']['tsr'] #7.55  # (Float): tip-speed ratio in Region 2 (should be optimized externally)
    rotor['control_pitch']    = blade['config']['pitch'] #0.0  # (Float, deg): pitch angle in region 2 (and region 3 for fixed pitch machines)
    rotor['control_maxTS']    = blade['config']['maxTS']
    rotor['machine_rating']   = blade['config']['rating'] #5e6  # (Float, W): rated power
    rotor['pitch_extreme']    = 0.0  # (Float, deg): worst-case pitch at survival wind condition
    rotor['azimuth_extreme']  = 0.0  # (Float, deg): worst-case azimuth at survival wind condition
    rotor['VfactorPC']        = 0.7  # (Float): fraction of rated speed at which the deflection is assumed to representative throughout the power curve calculation
    
    
    
    # ----------------------

    # === aero and structural analysis options ===
    rotor['nSector']          = 4  # (Int): number of sectors to divide rotor face into in computing thrust and power
    rotor['AEP_loss_factor']  = 1.0  # (Float): availability and other losses (soiling, array, etc.)
    rotor['drivetrainType']   = blade['config']['drivetrain'].upper() #DRIVETRAIN_TYPE['GEARED']  # (Enum)
    rotor['dynamic_amplification'] = 1.  # (Float): a dynamic amplification factor to adjust the static structural loads
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
    xx = np.r_[0.0, blade['pf']['s']]
    rotor['rstar_damage']    = np.interp(xx, xp, rstar_damage)
    rotor['Mxb_damage']      = np.interp(xx, xp, Mxb_damage)
    rotor['Myb_damage']      = np.interp(xx, xp, Myb_damage)
    rotor['strain_ult_spar'] = 1.0e-2  # (Float): ultimate strain in spar cap
    rotor['strain_ult_te']   = 2500*1e-6   # (Float): uptimate strain in trailing-edge panels, note that I am putting a factor of two for the damage part only.
    rotor['gamma_fatigue']   = 1.755 # (Float): safety factor for fatigue
    rotor['gamma_f']         = 1.35 # (Float): safety factor for loads/stresses
    rotor['gamma_m']         = 1.1 # (Float): safety factor for materials
    rotor['gamma_freq']      = 1.1 # (Float): safety factor for resonant frequencies
    rotor['m_damage']        = 10.0  # (Float): slope of S-N curve for fatigue analysis
    rotor['lifetime']  = 20.0  # (Float): number of cycles used in fatigue analysis  TODO: make function of rotation speed
    # ----------------
    return rotor
        
if __name__ == '__main__':

    # Turbine Ontology input
    fname_schema  = "turbine_inputs/IEAontology_schema.yaml"
    fname_input   = "turbine_inputs/nrel5mw_mod_update.yaml"
    output_folder = "test/"
    # fname_output  = output_folder + 'test_out.yaml'
    
    Analysis_Level = 0 # 0: Run CCBlade; 1: Update FAST model at each iteration but do not run; 2: Run FAST w/ ElastoDyn; 3: (Not implemented) Run FAST w/ BeamDyn

    # Initialize blade design
    refBlade = ReferenceBlade()
    refBlade.verbose      = True
    refBlade.NINPUT       = 8
    refBlade.NPTS         = 50
    refBlade.spar_var     = ['Spar_Cap_SS', 'Spar_Cap_PS'] # SS, then PS
    refBlade.te_var       = 'TE_reinforcement'
    refBlade.validate     = False
    refBlade.fname_schema = fname_schema
    blade = refBlade.initialize(fname_input)

    # Set FAST Inputs
    if Analysis_Level >= 1:
        # File management
        FASTpref                        = {}
        FASTpref['Analysis_Level']      = Analysis_Level
        FASTpref['FAST_ver']            = 'OpenFAST'
        FASTpref['dev_branch']          = True
        FASTpref['FAST_exe']            = '/mnt/c/Material/Programs/openfast/build/glue-codes/openfast/openfast'
        FASTpref['FAST_directory']      = '/mnt/c/Material/Projects/RefTurbines/BAR/RotorSE_FAST_BAR_005a'   # Path to fst directory files
        FASTpref['FAST_InputFile']      = 'RotorSE_FAST_BAR_005a.fst' # FAST input file (ext=.fst)
        # FASTpref['FAST_directory']      = 'C:/Users/egaertne/WT_Codes/models/openfast-dev/r-test/glue-codes/openfast/5MW_Land_DLL_WTurb'   # Path to fst directory files
        # FASTpref['FAST_InputFile']      = '5MW_Land_DLL_WTurb.fst' # FAST input file (ext=.fst)
        FASTpref['Turbsim_exe']         = "/mnt/c/Material/Programs/TurbSim/TurbSim_glin64"
        # FASTpref['FAST_exe']            = '/mnt/c/linux/WT_Codes/openfast_dev/build/glue-codes/openfast/openfast'
        # FASTpref['FAST_directory']      = '/mnt/c/linux/IS/xloads_tc/templates/openfast/5MW_Land_DLL_WTurb-NoAero'   # Path to fst directory files
        # FASTpref['FAST_InputFile']      = '5MW_Land_DLL_WTurb.fst' # FAST input file (ext=.fst)
        # FASTpref['Turbsim_exe']         = '/mnt/c/linux/WT_Codes/TurbSim_v2.00.07a-bjj/TurbSim_glin64'
        FASTpref['FAST_namingOut']      = 'RotorSE_FAST_'+ blade['config']['name']
        FASTpref['FAST_runDirectory']   = 'temp/' + FASTpref['FAST_namingOut']
        
        # Run Settings
        FASTpref['cores']               = 1
        FASTpref['debug_level']         = 2 # verbosity: set to 0 for quiet, 1 & 2 for increasing levels of output

        # DLCs
        FASTpref['DLC_gust']            = None      # Max deflection
        # FASTpref['DLC_gust']            = RotorSE_DLC_1_4_Rated       # Max deflection    ### Not in place yet
        FASTpref['DLC_extrm']           = None      # Max strain
        # FASTpref['DLC_extrm']           = RotorSE_DLC_7_1_Steady      # Max strain        ### Not in place yet
        FASTpref['DLC_turbulent']       = RotorSE_DLC_1_1_Turb
        # FASTpref['DLC_turbulent']       = RotorSE_DLC_1_1_Turb      # Alternate turbulent case, replacing rated and extreme DLCs for calculating max deflection and strain
        FASTpref['DLC_powercurve']      = power_curve      # AEP
        # FASTpref['DLC_powercurve']      = None      # AEP

        # Initialize, read initial FAST files to avoid doing it iteratively
        fast = InputReader_OpenFAST(FAST_ver=FASTpref['FAST_ver'], dev_branch=FASTpref['dev_branch'])
        fast.FAST_InputFile = FASTpref['FAST_InputFile']
        fast.FAST_directory = FASTpref['FAST_directory']
        fast.execute()
        fst_vt = fast.fst_vt
    else:
        FASTpref = {}
        fst_vt = {}

    rotor = Problem()
    npts_coarse_power_curve = 20 # (Int): number of points to evaluate aero analysis at
    npts_spline_power_curve = 200  # (Int): number of points to use in fitting spline to power curve
    regulation_reg_II5 = True # calculate Region 2.5 pitch schedule, False will not maximize power in region 2.5
    regulation_reg_III = False # calculate Region 3 pitch schedule, False will return erroneous Thrust, Torque, and Moment for above rated

    rotor.model = RotorSE(RefBlade=blade,
                          npts_coarse_power_curve=npts_coarse_power_curve,
                          npts_spline_power_curve=npts_spline_power_curve,
                          regulation_reg_II5=regulation_reg_II5,
                          regulation_reg_III=regulation_reg_III, 
                          Analysis_Level=Analysis_Level,
                          FASTpref=FASTpref,
                          topLevelFlag=True)
    rotor.setup()
    rotor = Init_RotorSE_wRefBlade(rotor, blade, fst_vt=fst_vt)
    
    # rotor['chord_in'] = np.array([3.542, 3.54451799, 2.42342374, 2.44521374, 4.69032208, 6.3306303, 4.41245811, 1.419])
    # rotor['theta_in'] = np.array([13.30800018, 13.30800018, 0.92624531, 10.41054813, 11.48955724, -0.60858835, -1.41595352, 4.89747605])
    # rotor['sparT_in'] = np.array([0.00047, 0.00059925, 0.07363709, 0.13907431, 0.19551095, 0.03357394, 0.12021584, 0.00047])
    # rotor['r_in']     = np.array([0., 0.02565783, 0.23892874, 0.39114299, 0.54335725, 0.6955715, 0.84778575, 1.])

    # === run and outputs ===
    tt = time.time()
    rotor.run_driver()

    # refBlade.write_ontology(fname_output, rotor['blade_out'], refBlade.wt_ref)

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
    print('moments at the hub =', rotor['Mxyz_total'])
    
    
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

    plt.figure()

    plt.plot(rotor['r_pts'], rotor['strainU_te'], label='suction')
    plt.plot(rotor['r_pts'], rotor['strainL_te'], label='pressure')
    plt.plot(rotor['r_pts'], rotor['eps_crit_te'], label='critical')
    plt.ylim([-5e-3, 5e-3])
    plt.xlabel('r')
    plt.ylabel('strain')
    plt.legend()

    plt.figure()
    plt.plot(rotor['r_pts'], rotor['rthick'], label='airfoil relative thickness')
    plt.xlabel('r')
    plt.ylabel('rthick')
    plt.legend()
    
    plt.show()
    
    print(rotor['cpctcq_tables.pitch_vector'])
    exit()
    n_pitch = len(rotor['cpctcq_tables.pitch_vector'])
    n_tsr   = len(rotor['cpctcq_tables.tsr_vector'])
    n_U     = len(rotor['cpctcq_tables.U_vector'])
    
    # file = open(output_folder + 'Cp_Ct_Cq.txt','w')
    # file.write('# Pitch angle vector - x axis (matrix columns) (deg)\n')
    # for i in range(n_pitch):
        # file.write('%.2f   ' % rotor['ra.cpctcq_tables.pitch_vector'][i])
    # file.write('\n# TSR vector - y axis (matrix rows) (-)\n')
    # for i in range(n_tsr):
        # file.write('%.2f   ' % rotor['ra.cpctcq_tables.tsr_vector'][i])
    # file.write('\n# Wind speed vector - z axis (m/s)\n')
    # for i in range(n_U):
        # file.write('%.2f   ' % rotor['ra.cpctcq_tables.U_vector'][i])
    # file.write('\n')
    
    # file.write('\n# Power coefficient\n\n')
    
    
    
    # for i in range(n_U):
        # for j in range(n_tsr):
            # for k in range(n_pitch):
                # file.write('%.5f   ' % rotor['ra.cpctcq_tables.Cp_aero_table'][j,k,i])
            # file.write('\n')
        # file.write('\n')
    
    # file.write('\n#  Thrust coefficient\n\n')
    # for i in range(n_U):
        # for j in range(n_tsr):
            # for k in range(n_pitch):
                # file.write('%.5f   ' % rotor['ra.cpctcq_tables.Ct_aero_table'][j,k,i])
            # file.write('\n')
        # file.write('\n')
    
    # file.write('\n# Torque coefficient\n\n')
    # for i in range(n_U):
        # for j in range(n_tsr):
            # for k in range(n_pitch):
                # file.write('%.5f   ' % rotor['ra.cpctcq_tables.Cq_aero_table'][j,k,i])
            # file.write('\n')
        # file.write('\n')
        
    # file.close()
    
    print(rotor['ra.cpctcq_tables.pitch_vector'])
    exit()
    for i in range(n_U):
        fig0, ax0 = plt.subplots()
        CS0 = ax0.contour(rotor['ra.cpctcq_tables.pitch_vector'], rotor['ra.cpctcq_tables.tsr_vector'], rotor['ra.cpctcq_tables.Cp_aero_table'][:, :, i], levels=[0.0, 0.3, 0.40, 0.42, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50 ])
        ax0.clabel(CS0, inline=1, fontsize=12)
        plt.title('Power Coefficient', fontsize=14, fontweight='bold')
        plt.xlabel('Pitch Angle [deg]', fontsize=14, fontweight='bold')
        plt.ylabel('TSR [-]', fontsize=14, fontweight='bold')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(color=[0.8,0.8,0.8], linestyle='--')
        plt.subplots_adjust(bottom = 0.15, left = 0.15)

        fig0, ax0 = plt.subplots()
        CS0 = ax0.contour(rotor['ra.cpctcq_tables.pitch_vector'], rotor['ra.cpctcq_tables.tsr_vector'], rotor['ra.cpctcq_tables.Ct_aero_table'][:, :, i])
        ax0.clabel(CS0, inline=1, fontsize=12)
        plt.title('Thrust Coefficient', fontsize=14, fontweight='bold')
        plt.xlabel('Pitch Angle [deg]', fontsize=14, fontweight='bold')
        plt.ylabel('TSR [-]', fontsize=14, fontweight='bold')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(color=[0.8,0.8,0.8], linestyle='--')
        plt.subplots_adjust(bottom = 0.15, left = 0.15)

        
        fig0, ax0 = plt.subplots()
        CS0 = ax0.contour(rotor['ra.cpctcq_tables.pitch_vector'], rotor['ra.cpctcq_tables.tsr_vector'], rotor['ra.cpctcq_tables.Cq_aero_table'][:, :, i])
        ax0.clabel(CS0, inline=1, fontsize=12)
        plt.title('Torque Coefficient', fontsize=14, fontweight='bold')
        plt.xlabel('Pitch Angle [deg]', fontsize=14, fontweight='bold')
        plt.ylabel('TSR [-]', fontsize=14, fontweight='bold')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(color=[0.8,0.8,0.8], linestyle='--')
        plt.subplots_adjust(bottom = 0.15, left = 0.15)
        
        plt.show()
        
    
    
    
    
    
    # ----------------
