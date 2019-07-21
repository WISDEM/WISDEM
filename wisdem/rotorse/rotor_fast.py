from __future__ import print_function

import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import PchipInterpolator
import os, copy
from openmdao.api import IndepVarComp, Component, Group, Problem
from ccblade.ccblade_component import CCBladePower, CCBladeLoads, CCBladeGeometry
from commonse import gravity, NFREQ
from commonse.csystem import DirectionVector
from commonse.utilities import trapz_deriv, interp_with_deriv
from precomp import _precomp
from akima import Akima, akima_interp_with_derivs
from rotor_geometry import RotorGeometry, NREL5MW, DTU10MW, TUM3_35MW, NINPUT, TURBULENCE_CLASS, TURBINE_CLASS
import _pBEAM
# import ccblade._bem as _bem  # TODO: move to rotoraero
import _bem  # TODO: move to rotoraero

from rotorse import RPM2RS, RS2RPM

try:
    from AeroelasticSE.FAST_reader import InputReader_Common, InputReader_OpenFAST, InputReader_FAST7
    from AeroelasticSE.FAST_writer import InputWriter_Common, InputWriter_OpenFAST, InputWriter_FAST7
    from AeroelasticSE.FAST_wrapper import FastWrapper
    from AeroelasticSE.runFAST_pywrapper import runFAST_pywrapper, runFAST_pywrapper_batch
    # from AeroelasticSE.CaseGen_IEC import CaseGen_IEC
    from AeroelasticSE.CaseLibrary import RotorSE_rated, RotorSE_DLC_1_4_Rated, RotorSE_DLC_7_1_Steady, RotorSE_DLC_1_1_Turb
    from AeroelasticSE.FAST_post import return_timeseries
except:
    pass

class FASTLoadCases(Component):
    def __init__(self, NPTS, npts_coarse_power_curve, FASTpref):
        super(FASTLoadCases, self).__init__()
        self.add_param('fst_vt_in', val={})

        # ElastoDyn Inputs
        # Assuming the blade modal damping to be unchanged. Cannot directly solve from the Rayleigh Damping without making assumptions. J.Jonkman recommends 2-3% https://wind.nrel.gov/forum/wind/viewtopic.php?t=522
        self.add_param('r', val=np.zeros(NPTS), units='m', desc='radial positions. r[0] should be the hub location \
            while r[-1] should be the blade tip. Any number \
            of locations can be specified between these in ascending order.')
        self.add_param('le_location', val=np.zeros(NPTS), desc='Leading-edge positions from a reference blade axis (usually blade pitch axis). Locations are normalized by the local chord length. Positive in -x direction for airfoil-aligned coordinate system')
        self.add_param('beam:Tw_iner', val=np.zeros(NPTS), units='m', desc='y-distance to elastic center from point about which above structural properties are computed')
        self.add_param('beam:rhoA', val=np.zeros(NPTS), units='kg/m', desc='mass per unit length')
        self.add_param('beam:EIyy', val=np.zeros(NPTS), units='N*m**2', desc='flatwise stiffness (bending about y-direction of airfoil aligned coordinate system)')
        self.add_param('beam:EIxx', val=np.zeros(NPTS), units='N*m**2', desc='edgewise stiffness (bending about :ref:`x-direction of airfoil aligned coordinate system <blade_airfoil_coord>`)')
        self.add_param('modes_coef_curvefem', val=np.zeros((3, 5)), desc='mode shapes as 6th order polynomials, in the format accepted by ElastoDyn, [[c_x2, c_],..]')

        # AeroDyn Inputs
        self.add_param('z_az', val=np.zeros(NPTS), units='m', desc='dimensional aerodynamic grid')
        self.add_param('chord', val=np.zeros(NPTS), units='m', desc='chord at airfoil locations')
        self.add_param('theta', val=np.zeros(NPTS), units='deg', desc='twist at airfoil locations')
        self.add_param('precurve', val=np.zeros(NPTS), units='m', desc='precurve at airfoil locations')
        self.add_param('presweep', val=np.zeros(NPTS), units='m', desc='presweep at structural locations')
        self.add_param('Rhub', val=0.0, units='m', desc='dimensional radius of hub')
        self.add_param('Rtip', val=0.0, units='m', desc='dimensional radius of tip')
        self.add_param('airfoils', val=[0]*NPTS, desc='CCAirfoil instances', pass_by_obj=True)

        # Turbine level inputs
        self.add_param('hubHt', val=0.0, units='m', desc='hub height')
        self.add_param('turbulence_class', val=TURBULENCE_CLASS['A'], desc='IEC turbulence class', pass_by_obj=True)
        self.add_param('turbine_class', val=TURBINE_CLASS['I'], desc='IEC turbulence class', pass_by_obj=True)
        
        # Initial conditions
        self.add_param('U_init', val=np.zeros(npts_coarse_power_curve), units='m/s', desc='wind speeds')
        self.add_param('Omega_init', val=np.zeros(npts_coarse_power_curve), units='rpm', desc='rotation speeds to run')
        self.add_param('pitch_init', val=np.zeros(npts_coarse_power_curve), units='deg', desc='pitch angles to run')

        # Environmental conditions 
        self.add_param('Vrated', val=11.0, units='m/s', desc='rated wind speed')
        self.add_param('Vgust', val=11.0, units='m/s', desc='gust wind speed')
        self.add_param('Vextreme', val=11.0, units='m/s', desc='IEC extreme wind speed at hub height')
        self.add_param('rho',       val=0.0,        units='kg/m**3',    desc='density of air')
        self.add_param('mu',        val=0.0,        units='kg/(m*s)',   desc='dynamic viscosity of air')
        self.add_param('shearExp',  val=0.0,                            desc='shear exponent')

        # FAST run preferences
        self.Analysis_Level      = FASTpref['Analysis_Level']
        self.FAST_ver            = FASTpref['FAST_ver']
        self.FAST_exe            = os.path.abspath(FASTpref['FAST_exe'])
        self.FAST_directory      = os.path.abspath(FASTpref['FAST_directory'])
        self.Turbsim_exe         = os.path.abspath(FASTpref['Turbsim_exe'])
        self.debug_level         = FASTpref['debug_level']
        self.FAST_runDirectory   = FASTpref['FAST_runDirectory']
        self.FAST_InputFile      = FASTpref['FAST_InputFile']
        self.FAST_namingOut      = FASTpref['FAST_namingOut']
        self.dev_branch          = FASTpref['dev_branch']
        self.cores               = FASTpref['cores']
        self.case                = {}
        self.channels            = {}

        # DLC Flags
        self.DLC_powercurve      = FASTpref['DLC_powercurve']
        self.DLC_gust            = FASTpref['DLC_gust']
        self.DLC_extrm           = FASTpref['DLC_extrm']
        self.DLC_turbulent       = FASTpref['DLC_turbulent']

        
        self.add_output('dx_defl', val=0., desc='deflection of blade section in airfoil x-direction under max deflection loading')
        self.add_output('dy_defl', val=0., desc='deflection of blade section in airfoil y-direction under max deflection loading')
        self.add_output('dz_defl', val=0., desc='deflection of blade section in airfoil z-direction under max deflection loading')
    
        self.add_output('root_bending_moment', val=0.0, units='N*m', desc='total magnitude of bending moment at root of blade 1')
        self.add_output('Mxyz', val=np.array([0.0, 0.0, 0.0]), units='N*m', desc='individual moments [x,y,z] at the blade root in blade c.s.')
        
        self.add_output('loads_r', val=np.zeros(NPTS), units='m', desc='radial positions along blade going toward tip')
        self.add_output('loads_Px', val=np.zeros(NPTS), units='N/m', desc='distributed loads in blade-aligned x-direction')
        self.add_output('loads_Py', val=np.zeros(NPTS), units='N/m', desc='distributed loads in blade-aligned y-direction')
        self.add_output('loads_Pz', val=np.zeros(NPTS), units='N/m', desc='distributed loads in blade-aligned z-direction')
        self.add_output('loads_Omega', val=0.0, units='rpm', desc='rotor rotation speed')
        self.add_output('loads_pitch', val=0.0, units='deg', desc='pitch angle')
        self.add_output('loads_azimuth', val=0.0, units='deg', desc='azimuthal angle')
        self.add_output('model_updated', val=False, desc='boolean, Analysis Level 0: fast model written, but not run')

    def solve_nonlinear(self, params, unknowns, resids):

        fst_vt, R_out = self.update_FAST_model(params)

        if self.Analysis_Level == 2:
            # Run FAST with ElastoDyn
            list_cases, list_casenames, required_channels, case_keys = self.DLC_creation(params, fst_vt)
            FAST_Output = self.run_FAST(fst_vt, list_cases, list_casenames, required_channels)
            self.post_process(FAST_Output, case_keys, R_out, params, unknowns)

        elif self.Analysis_Level == 1:
            # Write FAST files, do not run
            self.write_FAST(fst_vt, unknowns)


    def update_FAST_model(self, params):

        # Create instance of FAST reference model 

        fst_vt = copy.deepcopy(params['fst_vt_in'])

        fst_vt['Fst']['OutFileFmt'] = 2

        # Update ElastoDyn
        fst_vt['ElastoDyn']['TipRad'] = params['Rtip']
        fst_vt['ElastoDyn']['HubRad'] = params['Rhub']
        tower2hub = fst_vt['InflowWind']['RefHt'] - fst_vt['ElastoDyn']['TowerHt']
        fst_vt['ElastoDyn']['TowerHt'] = params['hubHt']

        # Update Inflowwind
        fst_vt['InflowWind']['RefHt'] = params['hubHt']
        fst_vt['InflowWind']['PLexp'] = params['shearExp']

        # Update ElastoDyn Blade Input File
        fst_vt['ElastoDynBlade']['NBlInpSt']   = len(params['r'])
        fst_vt['ElastoDynBlade']['BlFract']    = (params['r']-params['Rhub'])/(params['Rtip']-params['Rhub'])
        fst_vt['ElastoDynBlade']['BlFract'][0] = 0.
        fst_vt['ElastoDynBlade']['BlFract'][-1]= 1.
        fst_vt['ElastoDynBlade']['PitchAxis']  = params['le_location']
        fst_vt['ElastoDynBlade']['StrcTwst']   = params['beam:Tw_iner']
        fst_vt['ElastoDynBlade']['BMassDen']   = params['beam:rhoA']
        fst_vt['ElastoDynBlade']['FlpStff']    = params['beam:EIyy']
        fst_vt['ElastoDynBlade']['EdgStff']    = params['beam:EIxx']
        for i in range(5):
            fst_vt['ElastoDynBlade']['BldFl1Sh'][i] = params['modes_coef_curvefem'][0,i]
            fst_vt['ElastoDynBlade']['BldFl2Sh'][i] = params['modes_coef_curvefem'][1,i]
            fst_vt['ElastoDynBlade']['BldEdgSh'][i] = params['modes_coef_curvefem'][2,i]
        
        # Update AeroDyn15
        fst_vt['AeroDyn15']['AirDens'] = params['rho']
        fst_vt['AeroDyn15']['KinVisc'] = params['mu']        

        # Update AeroDyn15 Blade Input File
        r = (params['r']-params['Rhub'])
        r[0]  = 0.
        r[-1] = params['Rtip']-params['Rhub']
        fst_vt['AeroDynBlade']['NumBlNds'] = len(r)
        fst_vt['AeroDynBlade']['BlSpn']    = r
        fst_vt['AeroDynBlade']['BlCrvAC']  = params['precurve']
        fst_vt['AeroDynBlade']['BlSwpAC']  = params['presweep']
        fst_vt['AeroDynBlade']['BlCrvAng'] = np.degrees(np.arcsin(np.gradient(params['precurve'])/np.gradient(r)))
        fst_vt['AeroDynBlade']['BlTwist']  = params['theta']
        fst_vt['AeroDynBlade']['BlChord']  = params['chord']
        fst_vt['AeroDynBlade']['BlAFID']   = np.asarray(range(1,len(params['airfoils'])+1))

        # Update AeroDyn15 Airfoile Input Files
        airfoils = params['airfoils']
        fst_vt['AeroDyn15']['af_data'] = [{}]*len(airfoils)
        for i in range(len(airfoils)):
            af = airfoils[i]
            fst_vt['AeroDyn15']['af_data'][i]['InterpOrd'] = "DEFAULT"
            fst_vt['AeroDyn15']['af_data'][i]['NonDimArea']= 1
            fst_vt['AeroDyn15']['af_data'][i]['NumCoords'] = 0          # TODO: link the airfoil profiles to this component and write the coordinate files (no need as of yet)
            fst_vt['AeroDyn15']['af_data'][i]['NumTabs']   = 1
            fst_vt['AeroDyn15']['af_data'][i]['Re']        = 0.75       # TODO: functionality for multiple Re tables
            fst_vt['AeroDyn15']['af_data'][i]['Ctrl']      = 0
            fst_vt['AeroDyn15']['af_data'][i]['InclUAdata']= "True"
            fst_vt['AeroDyn15']['af_data'][i]['alpha0']    = af.unsteady['alpha0']
            fst_vt['AeroDyn15']['af_data'][i]['alpha1']    = af.unsteady['alpha1']
            fst_vt['AeroDyn15']['af_data'][i]['alpha2']    = af.unsteady['alpha2']
            fst_vt['AeroDyn15']['af_data'][i]['eta_e']     = af.unsteady['eta_e']
            fst_vt['AeroDyn15']['af_data'][i]['C_nalpha']  = af.unsteady['C_nalpha']
            fst_vt['AeroDyn15']['af_data'][i]['T_f0']      = af.unsteady['T_f0']
            fst_vt['AeroDyn15']['af_data'][i]['T_V0']      = af.unsteady['T_V0']
            fst_vt['AeroDyn15']['af_data'][i]['T_p']       = af.unsteady['T_p']
            fst_vt['AeroDyn15']['af_data'][i]['T_VL']      = af.unsteady['T_VL']
            fst_vt['AeroDyn15']['af_data'][i]['b1']        = af.unsteady['b1']
            fst_vt['AeroDyn15']['af_data'][i]['b2']        = af.unsteady['b2']
            fst_vt['AeroDyn15']['af_data'][i]['b5']        = af.unsteady['b5']
            fst_vt['AeroDyn15']['af_data'][i]['A1']        = af.unsteady['A1']
            fst_vt['AeroDyn15']['af_data'][i]['A2']        = af.unsteady['A2']
            fst_vt['AeroDyn15']['af_data'][i]['A5']        = af.unsteady['A5']
            fst_vt['AeroDyn15']['af_data'][i]['S1']        = af.unsteady['S1']
            fst_vt['AeroDyn15']['af_data'][i]['S2']        = af.unsteady['S2']
            fst_vt['AeroDyn15']['af_data'][i]['S3']        = af.unsteady['S3']
            fst_vt['AeroDyn15']['af_data'][i]['S4']        = af.unsteady['S4']
            fst_vt['AeroDyn15']['af_data'][i]['Cn1']       = af.unsteady['Cn1']
            fst_vt['AeroDyn15']['af_data'][i]['Cn2']       = af.unsteady['Cn2']
            fst_vt['AeroDyn15']['af_data'][i]['St_sh']     = af.unsteady['St_sh']
            fst_vt['AeroDyn15']['af_data'][i]['Cd0']       = af.unsteady['Cd0']
            fst_vt['AeroDyn15']['af_data'][i]['Cm0']       = af.unsteady['Cm0']
            fst_vt['AeroDyn15']['af_data'][i]['k0']        = af.unsteady['k0']
            fst_vt['AeroDyn15']['af_data'][i]['k1']        = af.unsteady['k1']
            fst_vt['AeroDyn15']['af_data'][i]['k2']        = af.unsteady['k2']
            fst_vt['AeroDyn15']['af_data'][i]['k3']        = af.unsteady['k3']
            fst_vt['AeroDyn15']['af_data'][i]['k1_hat']    = af.unsteady['k1_hat']
            fst_vt['AeroDyn15']['af_data'][i]['x_cp_bar']  = af.unsteady['x_cp_bar']
            fst_vt['AeroDyn15']['af_data'][i]['UACutout']  = af.unsteady['UACutout']
            fst_vt['AeroDyn15']['af_data'][i]['filtCutOff']= af.unsteady['filtCutOff']
            fst_vt['AeroDyn15']['af_data'][i]['NumAlf']    = len(af.unsteady['Alpha'])
            fst_vt['AeroDyn15']['af_data'][i]['Alpha']     = af.unsteady['Alpha']
            fst_vt['AeroDyn15']['af_data'][i]['Cl']        = af.unsteady['Cl']
            fst_vt['AeroDyn15']['af_data'][i]['Cd']        = af.unsteady['Cd']
            fst_vt['AeroDyn15']['af_data'][i]['Cm']        = af.unsteady['Cm']
            fst_vt['AeroDyn15']['af_data'][i]['Cpmin']     = np.zeros_like(af.unsteady['Cm'])

        # AeroDyn spanwise output positions
        r = r/r[-1]
        r_out_target = [0.0, 0.1, 0.20, 0.40, 0.6, 0.75, 0.85, 0.925, 1.0]
        idx_out = [np.argmin(abs(r-ri)) for ri in r_out_target]
        R_out = [fst_vt['AeroDynBlade']['BlSpn'][i] for i in idx_out]
        
        fst_vt['AeroDyn15']['BlOutNd'] = [str(idx+1) for idx in idx_out]
        fst_vt['AeroDyn15']['NBlOuts'] = len(idx_out)

        return fst_vt, R_out

    def DLC_creation(self, params, fst_vt):
        # Case Generations

        # TMax = 99999. # Overwrite runtime if TMax is less than predefined DLC length (primarily for debugging purposes)
        TMax = 5.

        list_cases        = []
        list_casenames    = []
        required_channels = []
        case_keys         = []

        turbulence_class = TURBULENCE_CLASS[params['turbulence_class']]
        turbine_class    = TURBINE_CLASS[params['turbine_class']]

        if self.DLC_gust != None:
            list_cases_gust, list_casenames_gust, requited_channels_gust = self.DLC_gust(fst_vt, self.FAST_runDirectory, self.FAST_namingOut, TMax, turbine_class, turbulence_class, params['Vrated'], U_init=params['U_init'], Omega_init=params['Omega_init'], pitch_init=params['pitch_init'])
            list_cases        += list_cases_gust
            list_casenames    += list_casenames_gust
            required_channels += requited_channels_gust
            case_keys         += [2]*len(list_cases_gust)

        
        if self.DLC_extrm != None:
            list_cases_rated, list_casenames_rated, requited_channels_rated = self.DLC_extrm(fst_vt, self.FAST_runDirectory, self.FAST_namingOut, TMax, turbine_class, turbulence_class, params['Vextreme'])
            list_cases        += list_cases_rated
            list_casenames    += list_casenames_rated
            required_channels += requited_channels_rated
            case_keys         += [3]*len(list_cases_rated)

        if self.DLC_turbulent != None:
            list_cases_turb, list_casenames_turb, requited_channels_turb = self.DLC_turbulent(fst_vt, self.FAST_runDirectory, self.FAST_namingOut, TMax, turbine_class, turbulence_class, params['Vrated'], U_init=params['U_init'], Omega_init=params['Omega_init'], pitch_init=params['pitch_init'], Turbsim_exe=self.Turbsim_exe)
            list_cases        += list_cases_turb
            list_casenames    += list_casenames_turb
            required_channels += requited_channels_turb
            case_keys         += [4]*len(list_cases_turb)


        required_channels = sorted(list(set(required_channels)))
        channels_out = {}
        for var in required_channels:
            channels_out[var] = True

        return list_cases, list_casenames, channels_out, case_keys


    def run_FAST(self, fst_vt, case_list, case_name_list, channels):

        # FAST wrapper setup
        fastBatch = runFAST_pywrapper_batch(FAST_ver=self.FAST_ver)

        fastBatch.FAST_exe          = self.FAST_exe
        fastBatch.FAST_runDirectory = self.FAST_runDirectory
        fastBatch.FAST_InputFile    = self.FAST_InputFile
        fastBatch.FAST_directory    = self.FAST_directory
        fastBatch.debug_level       = self.debug_level
        fastBatch.dev_branch        = self.dev_branch
        fastBatch.fst_vt            = fst_vt
        fastBatch.post              = return_timeseries

        fastBatch.case_list         = case_list
        fastBatch.case_name_list    = case_name_list
        fastBatch.channels          = channels

        # Run FAST
        if self.cores == 1:
            FAST_Output = fastBatch.run_serial()
        else:
            FAST_Output = fastBatch.run_multi(self.cores)

        return FAST_Output

    def write_FAST(self, fst_vt, unknowns):
        writer                   = InputWriter_OpenFAST(FAST_ver=self.FAST_ver)
        writer.fst_vt            = fst_vt
        writer.FAST_runDirectory = self.FAST_runDirectory
        writer.FAST_namingOut    = self.FAST_namingOut
        writer.dev_branch        = self.dev_branch
        writer.execute()

        unknowns['model_updated'] = True
        if self.debug_level > 0:
            print('RAN UPDATE: ', self.FAST_runDirectory, self.FAST_namingOut)

    def post_process(self, FAST_Output, case_keys, R_out, params, unknowns):

        def post_gust(data, case_type):

            if case_type == 2:
                t_s = min(max(data['Time'][0], 30.), data['Time'][-2])
                t_e = min(data['Time'][-1], 90.)
                idx_s = list(data['Time']).index(t_s)
                idx_e = list(data['Time']).index(t_e)
            else:
                idx_s = 0
                idx_e = -1

            # Tip Deflections
            # return tip x,y,z for max out of plane deflection
            blade_max_tip = np.argmax([max(data['TipDxc1'][idx_s:idx_e]), max(data['TipDxc2'][idx_s:idx_e]), max(data['TipDxc3'][idx_s:idx_e])])
            if blade_max_tip == 0:
                tip_var = ["TipDxc1", "TipDyc1", "TipDzc1"]
            elif blade_max_tip == 1:
                tip_var = ["TipDxc2", "TipDyc2", "TipDzc2"]
            elif blade_max_tip == 2:
                tip_var = ["TipDxc3", "TipDyc3", "TipDzc3"]
            idx_max_tip = np.argmax(data[tip_var[0]][idx_s:idx_e])
            unknowns['dx_defl'] = data[tip_var[0]][idx_s+idx_max_tip]
            unknowns['dy_defl'] = data[tip_var[1]][idx_s+idx_max_tip]
            unknowns['dz_defl'] = data[tip_var[2]][idx_s+idx_max_tip]

            # Root bending moments
            # return root bending moment for blade with the highest blade bending moment magnitude
            root_bending_moment_1 = np.sqrt(data['RootMxc1'][idx_s:idx_e]**2. + data['RootMyc1'][idx_s:idx_e]**2. + data['RootMzc1'][idx_s:idx_e]**2.)
            root_bending_moment_2 = np.sqrt(data['RootMxc2'][idx_s:idx_e]**2. + data['RootMyc2'][idx_s:idx_e]**2. + data['RootMzc2'][idx_s:idx_e]**2.)
            root_bending_moment_3 = np.sqrt(data['RootMxc3'][idx_s:idx_e]**2. + data['RootMyc3'][idx_s:idx_e]**2. + data['RootMzc3'][idx_s:idx_e]**2.)
            root_bending_moment_max       = [max(root_bending_moment_1), max(root_bending_moment_2), max(root_bending_moment_3)]
            root_bending_moment_idxmax    = [np.argmax(root_bending_moment_1), np.argmax(root_bending_moment_2), np.argmax(root_bending_moment_3)]
            blade_root_bending_moment_max = np.argmax(root_bending_moment_max)

            unknowns['root_bending_moment'] = root_bending_moment_max[blade_root_bending_moment_max]*1.e3
            idx = root_bending_moment_idxmax[blade_root_bending_moment_max]
            if blade_root_bending_moment_max == 0:
                unknowns['Mxyz'] = np.array([data['RootMxc1'][idx_s+idx]*1.e3, data['RootMyc1'][idx_s+idx]*1.e3, data['RootMzc1'][idx_s+idx]*1.e3])
            elif blade_root_bending_moment_max == 1:
                unknowns['Mxyz'] = np.array([data['RootMxc2'][idx_s+idx]*1.e3, data['RootMyc2'][idx_s+idx]*1.e3, data['RootMzc2'][idx_s+idx]*1.e3])
            elif blade_root_bending_moment_max == 2:
                unknowns['Mxyz'] = np.array([data['RootMxc3'][idx_s+idx]*1.e3, data['RootMyc3'][idx_s+idx]*1.e3, data['RootMzc3'][idx_s+idx]*1.e3])

        def post_extreme(data, case_type):

            if case_type == 3:
                t_s = min(max(data['Time'][0], 30.), data['Time'][-2])
                t_e = min(data['Time'][-1], 90.)
                idx_s = list(data['Time']).index(t_s)
                idx_e = list(data['Time']).index(t_e)
            else:
                idx_s = 0
                idx_e = -1

            Time = data['Time'][idx_s:idx_e]
            var_Fx = ["B1N1Fx", "B1N2Fx", "B1N3Fx", "B1N4Fx", "B1N5Fx", "B1N6Fx", "B1N7Fx", "B1N8Fx", "B1N9Fx"]
            var_Fy = ["B1N1Fy", "B1N2Fy", "B1N3Fy", "B1N4Fy", "B1N5Fy", "B1N6Fy", "B1N7Fy", "B1N8Fy", "B1N9Fy"]
            for i, (varFxi, varFyi) in enumerate(zip(var_Fx, var_Fy)):
                if i == 0:
                    Fx = np.array(data[varFxi][idx_s:idx_e])
                    Fy = np.array(data[varFyi][idx_s:idx_e])
                else:
                    Fx = np.column_stack((Fx, np.array(data[varFxi][idx_s:idx_e])))
                    Fy = np.column_stack((Fy, np.array(data[varFyi][idx_s:idx_e])))

            Fx_sum = np.zeros_like(Time)
            Fy_sum = np.zeros_like(Time)
            for i in range(len(Time)):
                Fx_sum[i] = np.trapz(Fx[i,:], R_out)
                Fy_sum[i] = np.trapz(Fy[i,:], R_out)
            idx_max_strain = np.argmax(np.sqrt(Fx_sum**2.+Fy_sum**2.))

            Fx = [data[Fxi][idx_max_strain] for Fxi in var_Fx]
            Fy = [data[Fyi][idx_max_strain] for Fyi in var_Fy]
            spline_Fx = PchipInterpolator(R_out, Fx)
            spline_Fy = PchipInterpolator(R_out, Fy)

            r = params['r']-params['Rhub']
            Fx_out = spline_Fx(r)
            Fy_out = spline_Fy(r)
            Fz_out = np.zeros_like(Fx_out)

            unknowns['loads_Px'] = Fx_out
            unknowns['loads_Py'] = Fy_out*-1.
            unknowns['loads_Pz'] = Fz_out

            unknowns['loads_Omega'] = data['RotSpeed'][idx_max_strain]
            unknowns['loads_pitch'] = data['BldPitch1'][idx_max_strain]
            unknowns['loads_azimuth'] = data['Azimuth'][idx_max_strain]

        ############

        Gust_Outputs = False
        Extreme_Outputs = False
        #
        for casei in case_keys:
            if Gust_Outputs and Extreme_Outputs:
                break

            if casei in[1]:
                # power curve
                pass

            if casei in [2]:
                # gust: return tip deflections and bending moments
                idx_gust = case_keys.index(casei)
                data = FAST_Output[idx_gust]
                post_gust(data, casei)
                Gust_Outputs = True

            if casei in [3]:
                # extreme wind speed: return aeroloads for strains
                idx_extreme = case_keys.index(casei)
                data = FAST_Output[idx_extreme]
                post_extreme(data, casei)
                Extreme_Outputs = True


            if casei in [4]:
                # turbulent wind with multiplt seeds
                idx_turb = [i for i, casej in enumerate(case_keys) if casej==4]
                data_concat = {}
                for i, fast_out_idx in enumerate(idx_turb):
                    datai = FAST_Output[idx_turb[fast_out_idx]]

                    for var in datai.keys():
                        if i == 0:
                            data_concat[var] = []
                        data_concat[var].extend(list(datai[var]))

                for var in data_concat.keys():
                    data_concat[var] = np.array(data_concat[var])

                post_gust(data_concat, casei)
                post_extreme(data_concat, casei)
                Gust_Outputs = True
                Extreme_Outputs = True





                

