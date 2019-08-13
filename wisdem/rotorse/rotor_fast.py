from __future__ import print_function

import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import PchipInterpolator
import os, copy, warnings, shutil
from openmdao.api import IndepVarComp, ExplicitComponent, Group, Problem
from openmdao.core.mpi_wrap import MPI

from AeroelasticSE.FAST_reader import InputReader_Common, InputReader_OpenFAST, InputReader_FAST7
from AeroelasticSE.FAST_writer import InputWriter_Common, InputWriter_OpenFAST, InputWriter_FAST7
from AeroelasticSE.FAST_wrapper import FastWrapper
from AeroelasticSE.runFAST_pywrapper import runFAST_pywrapper, runFAST_pywrapper_batch
from AeroelasticSE.CaseLibrary import RotorSE_rated, RotorSE_DLC_1_4_Rated, RotorSE_DLC_7_1_Steady, RotorSE_DLC_1_1_Turb, power_curve
from AeroelasticSE.FAST_post import return_timeseries


if MPI:
    from openmdao.api import PetscImpl as impl
    from mpi4py import MPI
    from petsc4py import PETSc
else:
    from openmdao.api import BasicImpl as impl


class FASTLoadCases(ExplicitComponent):
    def initialize(self):
        self.options.declare('NPTS')
        self.options.declare('npts_coarse_power_curve', default=20)
        self.options.declare('npts_spline_power_curve', default=200)
        self.options.declare('FASTpref',default={})

    def setup(self, NPTS, npts_coarse_power_curve, npts_spline_power_curve, FASTpref):
        NPTS                    = self.options['NPTS']
        npts_coarse_power_curve = self.options['npts_coarse_power_curve']
        npts_spline_power_curve = self.options['npts_spline_power_curve']
        FASTpref                = self.options['FASTpref']

        self.add_discrete_input('fst_vt_in', val={})

        # ElastoDyn Inputs
        # Assuming the blade modal damping to be unchanged. Cannot directly solve from the Rayleigh Damping without making assumptions. J.Jonkman recommends 2-3% https://wind.nrel.gov/forum/wind/viewtopic.php?t=522
        self.add_input('r', val=np.zeros(NPTS), units='m', desc='radial positions. r[0] should be the hub location \
            while r[-1] should be the blade tip. Any number \
            of locations can be specified between these in ascending order.')
        self.add_input('le_location', val=np.zeros(NPTS), desc='Leading-edge positions from a reference blade axis (usually blade pitch axis). Locations are normalized by the local chord length. Positive in -x direction for airfoil-aligned coordinate system')
        self.add_input('beam:Tw_iner', val=np.zeros(NPTS), units='m', desc='y-distance to elastic center from point about which above structural properties are computed')
        self.add_input('beam:rhoA', val=np.zeros(NPTS), units='kg/m', desc='mass per unit length')
        self.add_input('beam:EIyy', val=np.zeros(NPTS), units='N*m**2', desc='flatwise stiffness (bending about y-direction of airfoil aligned coordinate system)')
        self.add_input('beam:EIxx', val=np.zeros(NPTS), units='N*m**2', desc='edgewise stiffness (bending about :ref:`x-direction of airfoil aligned coordinate system <blade_airfoil_coord>`)')
        self.add_input('modes_coef_curvefem', val=np.zeros((3, 5)), desc='mode shapes as 6th order polynomials, in the format accepted by ElastoDyn, [[c_x2, c_],..]')

        # AeroDyn Inputs
        self.add_input('z_az', val=np.zeros(NPTS), units='m', desc='dimensional aerodynamic grid')
        self.add_input('chord', val=np.zeros(NPTS), units='m', desc='chord at airfoil locations')
        self.add_input('theta', val=np.zeros(NPTS), units='deg', desc='twist at airfoil locations')
        self.add_input('precurve', val=np.zeros(NPTS), units='m', desc='precurve at airfoil locations')
        self.add_input('presweep', val=np.zeros(NPTS), units='m', desc='presweep at structural locations')
        self.add_input('Rhub', val=0.0, units='m', desc='dimensional radius of hub')
        self.add_input('Rtip', val=0.0, units='m', desc='dimensional radius of tip')
        self.add_discrete_input('airfoils', val=[0]*NPTS, desc='CCAirfoil instances')

        # Turbine level inputs
        self.add_input('hubHt', val=0.0, units='m', desc='hub height')
        self.add_discrete_input('turbulence_class', val=TURBULENCE_CLASS['A'], desc='IEC turbulence class')
        self.add_discrete_input('turbine_class', val=TURBINE_CLASS['I'], desc='IEC turbulence class')
        self.add_input('control_ratedPower', val=0., desc='machine power rating')
        self.add_input('control_maxOmega',   val=0.0, units='rpm',  desc='maximum allowed rotor rotation speed')
        self.add_input('control_maxTS',      val=0.0, units='m/s',  desc='maximum allowed blade tip speed')

        # Initial conditions
        self.add_input('U_init', val=np.zeros(npts_coarse_power_curve), units='m/s', desc='wind speeds')
        self.add_input('Omega_init', val=np.zeros(npts_coarse_power_curve), units='rpm', desc='rotation speeds to run')
        self.add_input('pitch_init', val=np.zeros(npts_coarse_power_curve), units='deg', desc='pitch angles to run')
        self.add_input('V_out', val=np.zeros(npts_spline_power_curve), units='m/s', desc='wind speeds to output powercurve')
        self.add_input('V',        val=np.zeros(npts_coarse_power_curve), units='m/s',  desc='wind vector')


        # Environmental conditions 
        self.add_input('Vrated', val=11.0, units='m/s', desc='rated wind speed')
        self.add_input('V_R25', val=0.0, units='m/s', desc='region 2.5 transition wind speed')
        self.add_input('Vgust', val=11.0, units='m/s', desc='gust wind speed')
        self.add_input('Vextreme', val=11.0, units='m/s', desc='IEC extreme wind speed at hub height')
        self.add_input('V_mean_iec', val=11.0, units='m/s', desc='IEC mean wind for turbulence class')
        self.add_input('rho',       val=0.0,        units='kg/m**3',    desc='density of air')
        self.add_input('mu',        val=0.0,        units='kg/(m*s)',   desc='dynamic viscosity of air')
        self.add_input('shearExp',  val=0.0,                            desc='shear exponent')

        # FAST run preferences
        self.FASTpref            = FASTpref 
        self.Analysis_Level      = FASTpref['Analysis_Level']
        self.FAST_ver            = FASTpref['FAST_ver']
        self.FAST_exe            = os.path.abspath(FASTpref['FAST_exe'])
        self.FAST_directory      = os.path.abspath(FASTpref['FAST_directory'])
        self.Turbsim_exe         = os.path.abspath(FASTpref['Turbsim_exe'])
        self.debug_level         = FASTpref['debug_level']
        self.FAST_InputFile      = FASTpref['FAST_InputFile']
        if MPI:
            self.FAST_runDirectory = os.path.join(FASTpref['FAST_runDirectory'],'rank_%000d'%int(impl.world_comm().rank))
            self.FAST_namingOut  = FASTpref['FAST_namingOut']+'_%000d'%int(impl.world_comm().rank)
            # try:
            #     if not os.path.exists(directory):
            #         os.makedirs(self.FAST_runDirectory)
            # except:
            #     pass
        else:
            self.FAST_runDirectory = FASTpref['FAST_runDirectory']
            self.FAST_namingOut  = FASTpref['FAST_namingOut']
        self.dev_branch          = FASTpref['dev_branch']
        self.cores               = FASTpref['cores']
        self.case                = {}
        self.channels            = {}

        # DLC Flags
        self.DLC_powercurve      = FASTpref['DLC_powercurve']
        self.DLC_gust            = FASTpref['DLC_gust']
        self.DLC_extrm           = FASTpref['DLC_extrm']
        self.DLC_turbulent       = FASTpref['DLC_turbulent']

        self.clean_FAST_directory = False
        if 'clean_FAST_directory' in FASTpref.keys():
            self.clean_FAST_directory = FASTpref['clean_FAST_directory']

        self.mpi_run             = False
        if 'mpi_run' in FASTpref.keys():
            self.mpi_run         = FASTpref['mpi_run']
            if self.mpi_run:
                self.mpi_comm_map_down   = FASTpref['mpi_comm_map_down']
        
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
        self.add_discrete_output('model_updated', val=False, desc='boolean, Analysis Level 0: fast model written, but not run')
        self.add_discrete_output('FASTpref_updated', val={}, desc='updated fast preference dictionary')

        self.add_output('P_out', val=np.zeros(npts_spline_power_curve), units='W', desc='electrical power from rotor')
        self.add_output('P',        val=np.zeros(npts_coarse_power_curve), units='W',    desc='rotor electrical power')
        self.add_output('Cp',       val=np.zeros(npts_coarse_power_curve),               desc='rotor electrical power coefficient')
        self.add_output('rated_V',     val=0.0, units='m/s', desc='rated wind speed')
        self.add_output('rated_Omega', val=0.0, units='rpm', desc='rotor rotation speed at rated')
        self.add_output('rated_pitch', val=0.0, units='deg', desc='pitch setting at rated')
        self.add_output('rated_T',     val=0.0, units='N', desc='rotor aerodynamic thrust at rated')
        self.add_output('rated_Q',     val=0.0, units='N*m', desc='rotor aerodynamic torque at rated')

        self.add_discrete_output('fst_vt_out', val={})

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        #print(impl.world_comm().rank, 'Rotor_fast','start')

        fst_vt, R_out = self.update_FAST_model(inputs, discrete_inputs)

        # if MPI:
            # rank = int(PETSc.COMM_WORLD.getRank())
            # self.FAST_namingOut = self.FAST_namingOut + '_%00d'%rank

        if self.Analysis_Level == 2:
            # Run FAST with ElastoDyn
            list_cases, list_casenames, required_channels, case_keys = self.DLC_creation(params, fst_vt)
            FAST_Output = self.run_FAST(fst_vt, list_cases, list_casenames, required_channels)
            self.post_process(FAST_Output, case_keys, R_out, params, outputs)

        elif self.Analysis_Level == 1:
            # Write FAST files, do not run
            self.write_FAST(fst_vt, outputs)

        outputs['fst_vt_out'] = fst_vt

        # delete run directory. not recommended for most cases, use for large parallelization problems where disk storage will otherwise fill up
        if self.clean_FAST_directory:
            try:
                shutil.rmtree(self.FAST_runDirectory)
            except:
                print('Failed to delete directory: %s'%self.FAST_runDirectory)

        #print(impl.world_comm().rank, 'Rotor_fast','end')


    def update_FAST_model(self, inputs, discrete_inputs):

        # Create instance of FAST reference model 

        fst_vt = copy.deepcopy(inputs['fst_vt_in'])

        fst_vt['Fst']['OutFileFmt'] = 2

        # Update ElastoDyn
        fst_vt['ElastoDyn']['TipRad'] = inputs['Rtip']
        fst_vt['ElastoDyn']['HubRad'] = inputs['Rhub']
        tower2hub = fst_vt['InflowWind']['RefHt'] - fst_vt['ElastoDyn']['TowerHt']
        fst_vt['ElastoDyn']['TowerHt'] = inputs['hubHt']

        # Update Inflowwind
        fst_vt['InflowWind']['RefHt'] = inputs['hubHt']
        fst_vt['InflowWind']['PLexp'] = inputs['shearExp']

        # Update ElastoDyn Blade Input File
        fst_vt['ElastoDynBlade']['NBlInpSt']   = len(inputs['r'])
        fst_vt['ElastoDynBlade']['BlFract']    = (inputs['r']-inputs['Rhub'])/(inputs['Rtip']-inputs['Rhub'])
        fst_vt['ElastoDynBlade']['BlFract'][0] = 0.
        fst_vt['ElastoDynBlade']['BlFract'][-1]= 1.
        fst_vt['ElastoDynBlade']['PitchAxis']  = inputs['le_location']
        # fst_vt['ElastoDynBlade']['StrcTwst']   = inputs['beam:Tw_iner']
        fst_vt['ElastoDynBlade']['StrcTwst']   = inputs['theta'] # to do: structural twist is not nessessarily (nor likely to be) the same as aero twist
        fst_vt['ElastoDynBlade']['BMassDen']   = inputs['beam:rhoA']
        fst_vt['ElastoDynBlade']['FlpStff']    = inputs['beam:EIyy']
        fst_vt['ElastoDynBlade']['EdgStff']    = inputs['beam:EIxx']
        for i in range(5):
            fst_vt['ElastoDynBlade']['BldFl1Sh'][i] = inputs['modes_coef_curvefem'][0,i]
            fst_vt['ElastoDynBlade']['BldFl2Sh'][i] = inputs['modes_coef_curvefem'][1,i]
            fst_vt['ElastoDynBlade']['BldEdgSh'][i] = inputs['modes_coef_curvefem'][2,i]
        
        # Update AeroDyn15
        fst_vt['AeroDyn15']['AirDens'] = inputs['rho']
        fst_vt['AeroDyn15']['KinVisc'] = inputs['mu']        

        # Update AeroDyn15 Blade Input File
        r = (inputs['r']-inputs['Rhub'])
        r[0]  = 0.
        r[-1] = inputs['Rtip']-inputs['Rhub']
        fst_vt['AeroDynBlade']['NumBlNds'] = len(r)
        fst_vt['AeroDynBlade']['BlSpn']    = r
        fst_vt['AeroDynBlade']['BlCrvAC']  = inputs['precurve']
        fst_vt['AeroDynBlade']['BlSwpAC']  = inputs['presweep']
        fst_vt['AeroDynBlade']['BlCrvAng'] = np.degrees(np.arcsin(np.gradient(inputs['precurve'])/np.gradient(r)))
        fst_vt['AeroDynBlade']['BlTwist']  = inputs['theta']
        fst_vt['AeroDynBlade']['BlChord']  = inputs['chord']
        fst_vt['AeroDynBlade']['BlAFID']   = np.asarray(range(1,len(inputs['airfoils'])+1))

        # Update AeroDyn15 Airfoile Input Files
        airfoils = inputs['airfoils']
        fst_vt['AeroDyn15']['NumAFfiles'] = len(airfoils)
        # fst_vt['AeroDyn15']['af_data'] = [{}]*len(airfoils)
        fst_vt['AeroDyn15']['af_data'] = []
        for i in range(len(airfoils)):
            af = airfoils[i]
            fst_vt['AeroDyn15']['af_data'].append({})
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
            fst_vt['AeroDyn15']['af_data'][i]['Alpha']     = np.array(af.unsteady['Alpha'])
            fst_vt['AeroDyn15']['af_data'][i]['Cl']        = np.array(af.unsteady['Cl'])
            fst_vt['AeroDyn15']['af_data'][i]['Cd']        = np.array(af.unsteady['Cd'])
            fst_vt['AeroDyn15']['af_data'][i]['Cm']        = np.array(af.unsteady['Cm'])
            fst_vt['AeroDyn15']['af_data'][i]['Cpmin']     = np.zeros_like(af.unsteady['Cm'])

        # AeroDyn spanwise output positions
        r = r/r[-1]
        r_out_target = [0.0, 0.1, 0.20, 0.40, 0.6, 0.75, 0.85, 0.925, 1.0]
        idx_out = [np.argmin(abs(r-ri)) for ri in r_out_target]
        R_out = [fst_vt['AeroDynBlade']['BlSpn'][i] for i in idx_out]
        
        fst_vt['AeroDyn15']['BlOutNd'] = [str(idx+1) for idx in idx_out]
        fst_vt['AeroDyn15']['NBlOuts'] = len(idx_out)

        return fst_vt, R_out

    def DLC_creation(self, inputs, discrete_inputs, fst_vt):
        # Case Generations

        TMax = 99999. # Overwrite runtime if TMax is less than predefined DLC length (primarily for debugging purposes)
        # TMax = 5.

        list_cases        = []
        list_casenames    = []
        required_channels = []
        case_keys         = []

        turbulence_class = TURBULENCE_CLASS[inputs['turbulence_class']]
        turbine_class    = TURBINE_CLASS[inputs['turbine_class']]

        if self.DLC_powercurve != None:
            self.U_init     = copy.deepcopy(inputs['U_init'])
            self.Omega_init = copy.deepcopy(inputs['Omega_init'])
            self.pitch_init = copy.deepcopy(inputs['pitch_init'])
            # self.max_omega  = min([inputs['control_maxTS'] / inputs['Rtip'], inputs['control_maxOmega']*np.pi/30.])*30/np.pi
            # print('U_init    ', self.U_init    )
            # print('Omega_init', self.Omega_init)
            # print('pitch_init', self.pitch_init)
            # for i, (Ui, Omegai, pitchi) in enumerate(zip(self.U_init, self.Omega_init, self.pitch_init)):
            #     if pitchi > 0. and Omegai < self.max_omega*0.99:
            #         self.pitch_init[i] = 0.
            # print('U_init    ', self.U_init    )
            # print('Omega_init', self.Omega_init)
            # print('pitch_init', self.pitch_init)

            list_cases_PwrCrv, list_casenames_PwrCrv, requited_channels_PwrCrv = self.DLC_powercurve(fst_vt, self.FAST_runDirectory, self.FAST_namingOut, TMax, turbine_class, turbulence_class, inputs['Vrated'], U_init=self.U_init, Omega_init=self.Omega_init, pitch_init=self.pitch_init, V_R25=inputs['V_R25'])
            list_cases        += list_cases_PwrCrv
            list_casenames    += list_casenames_PwrCrv
            required_channels += requited_channels_PwrCrv
            case_keys         += [1]*len(list_cases_PwrCrv)    

        if self.DLC_gust != None:
            list_cases_gust, list_casenames_gust, requited_channels_gust = self.DLC_gust(fst_vt, self.FAST_runDirectory, self.FAST_namingOut, TMax, turbine_class, turbulence_class, inputs['V_mean_iec'], U_init=inputs['U_init'], Omega_init=inputs['Omega_init'], pitch_init=inputs['pitch_init'])
            list_cases        += list_cases_gust
            list_casenames    += list_casenames_gust
            required_channels += requited_channels_gust
            case_keys         += [2]*len(list_cases_gust)

        if self.DLC_extrm != None:
            list_cases_rated, list_casenames_rated, requited_channels_rated = self.DLC_extrm(fst_vt, self.FAST_runDirectory, self.FAST_namingOut, TMax, turbine_class, turbulence_class, inputs['Vextreme'])
            list_cases        += list_cases_rated
            list_casenames    += list_casenames_rated
            required_channels += requited_channels_rated
            case_keys         += [3]*len(list_cases_rated)

        if self.DLC_turbulent != None:
            if self.mpi_run:
                list_cases_turb, list_casenames_turb, requited_channels_turb = self.DLC_turbulent(fst_vt, self.FAST_runDirectory, self.FAST_namingOut, TMax, turbine_class, turbulence_class, inputs['Vrated'], U_init=inputs['U_init'], Omega_init=inputs['Omega_init'], pitch_init=inputs['pitch_init'], Turbsim_exe=self.Turbsim_exe, debug_level=self.debug_level, cores=self.cores, mpi_run=self.mpi_run, mpi_comm_map_down=self.mpi_comm_map_down)
            else:
                list_cases_turb, list_casenames_turb, requited_channels_turb = self.DLC_turbulent(fst_vt, self.FAST_runDirectory, self.FAST_namingOut, TMax, turbine_class, turbulence_class, inputs['Vrated'], U_init=inputs['U_init'], Omega_init=inputs['Omega_init'], pitch_init=inputs['pitch_init'], Turbsim_exe=self.Turbsim_exe, debug_level=self.debug_level, cores=self.cores)
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
        if self.mpi_run:
            FAST_Output = fastBatch.run_mpi(self.mpi_comm_map_down)
        else:
            if self.cores == 1:
                FAST_Output = fastBatch.run_serial()
            else:
                FAST_Output = fastBatch.run_multi(self.cores)

        self.fst_vt = fst_vt

        return FAST_Output

    def write_FAST(self, fst_vt, outputs):
        writer                   = InputWriter_OpenFAST(FAST_ver=self.FAST_ver)
        writer.fst_vt            = fst_vt
        writer.FAST_runDirectory = self.FAST_runDirectory
        writer.FAST_namingOut    = self.FAST_namingOut
        writer.dev_branch        = self.dev_branch
        writer.execute()

        outputs['FASTpref_updated'] = copy.deepcopy(self.FASTpref)
        outputs['FASTpref_updated']['FAST_runDirectory'] = self.FAST_runDirectory
        outputs['FASTpref_updated']['FAST_directory']    = self.FAST_runDirectory
        outputs['FASTpref_updated']['FAST_InputFile']    = os.path.split(writer.FAST_InputFileOut)[-1]

        outputs['model_updated'] = True
        if self.debug_level > 0:
            print('RAN UPDATE: ', self.FAST_runDirectory, self.FAST_namingOut)

    def post_process(self, FAST_Output, case_keys, R_out, inputs, discrete_inputs, outputs):

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
            outputs['dx_defl'] = data[tip_var[0]][idx_s+idx_max_tip]
            outputs['dy_defl'] = data[tip_var[1]][idx_s+idx_max_tip]
            outputs['dz_defl'] = data[tip_var[2]][idx_s+idx_max_tip]

            # Root bending moments
            # return root bending moment for blade with the highest blade bending moment magnitude
            root_bending_moment_1 = np.sqrt(data['RootMxc1'][idx_s:idx_e]**2. + data['RootMyc1'][idx_s:idx_e]**2. + data['RootMzc1'][idx_s:idx_e]**2.)
            root_bending_moment_2 = np.sqrt(data['RootMxc2'][idx_s:idx_e]**2. + data['RootMyc2'][idx_s:idx_e]**2. + data['RootMzc2'][idx_s:idx_e]**2.)
            root_bending_moment_3 = np.sqrt(data['RootMxc3'][idx_s:idx_e]**2. + data['RootMyc3'][idx_s:idx_e]**2. + data['RootMzc3'][idx_s:idx_e]**2.)
            root_bending_moment_max       = [max(root_bending_moment_1), max(root_bending_moment_2), max(root_bending_moment_3)]
            root_bending_moment_idxmax    = [np.argmax(root_bending_moment_1), np.argmax(root_bending_moment_2), np.argmax(root_bending_moment_3)]
            blade_root_bending_moment_max = np.argmax(root_bending_moment_max)

            outputs['root_bending_moment'] = root_bending_moment_max[blade_root_bending_moment_max]*1.e3
            idx = root_bending_moment_idxmax[blade_root_bending_moment_max]
            if blade_root_bending_moment_max == 0:
                outputs['Mxyz'] = np.array([data['RootMxc1'][idx_s+idx]*1.e3, data['RootMyc1'][idx_s+idx]*1.e3, data['RootMzc1'][idx_s+idx]*1.e3])
            elif blade_root_bending_moment_max == 1:
                outputs['Mxyz'] = np.array([data['RootMxc2'][idx_s+idx]*1.e3, data['RootMyc2'][idx_s+idx]*1.e3, data['RootMzc2'][idx_s+idx]*1.e3])
            elif blade_root_bending_moment_max == 2:
                outputs['Mxyz'] = np.array([data['RootMxc3'][idx_s+idx]*1.e3, data['RootMyc3'][idx_s+idx]*1.e3, data['RootMzc3'][idx_s+idx]*1.e3])

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

            r = inputs['r']-inputs['Rhub']
            Fx_out = spline_Fx(r)
            Fy_out = spline_Fy(r)
            Fz_out = np.zeros_like(Fx_out)

            outputs['loads_Px'] = Fx_out
            outputs['loads_Py'] = Fy_out*-1.
            outputs['loads_Pz'] = Fz_out

            outputs['loads_Omega'] = data['RotSpeed'][idx_max_strain]
            outputs['loads_pitch'] = data['BldPitch1'][idx_max_strain]
            outputs['loads_azimuth'] = data['Azimuth'][idx_max_strain]

        # def post_AEP_fit(data):
        #     def my_cubic(f, x):
        #         return np.array([f[3]+ f[2]*xi + f[1]*xi**2. + f[0]*xi**3. for xi in x])

        #     U = np.array([np.mean(datai['Wind1VelX']) for datai in data])
        #     P = np.array([np.mean(datai['GenPwr']) for datai in data])*1000.
        #     P_coef = np.polyfit(U, P, 3)

        #     P_out = my_cubic(P_coef, inputs['V_out'])
        #     np.place(P_out, P_out>inputs['control_ratedPower'], inputs['control_ratedPower'])
        #     outputs['P_out'] = P_out

        #     # import matplotlib.pyplot as plt
        #     # plt.plot(U, P, 'o')
        #     # plt.plot(inputs['V_out'], outputs['P_out'])            
        #     # plt.show()

        def post_AEP(data):
            U = list(sorted([4., 6., 8., 9., 10., 10.5, 11., 11.5, 11.75, 12., 12.5, 13., 14., 19., 25., inputs['Vrated']]))
            if inputs['V_R25'] != 0.:
                U.append(inputs['V_R25'])
                U = list(sorted(U))
            U = np.array(U)

            U_below = [Vi for Vi in U if Vi <= inputs['Vrated']]
            # P_below = np.array([np.mean(datai['GenPwr'])*1000. for datai in data])
            P_below = np.array([np.mean(datai['GenPwr'])*1000. for datai, Vi in zip(data, U) if Vi <= inputs['Vrated']])
            np.place(P_below, P_below>inputs['control_ratedPower'], inputs['control_ratedPower'])

            U_rated = [Vi for Vi in U if Vi > inputs['Vrated']]
            P_rated = [inputs['control_ratedPower']]*len(U_rated)

            if len(U_below) < len(U):
                P_fast = np.array(P_below.tolist() + P_rated)
            else:
                P_fast = P_below

            data_rated = data[-1]

            # U_fit = np.array([4.,8.,9.,10.])

            ## Find rated 
            # def my_cubic(f, x):
                # return np.array([f[3]+ f[2]*xi + f[1]*xi**2. + f[0]*xi**3. for xi in x])

            # idx_fit = [U.tolist().index(Ui) for Ui in U_fit]
            # P_fit = np.array([np.mean(data[i]['GenPwr']) for i in idx_fit])
            # P_coef = np.polyfit(U_fit, P_fit, 3)

            # P_find_rated = my_cubic(P_coef, inputs['V_out'])
            # np.place(P_find_rated, P_find_rated>inputs['control_ratedPower'], inputs['control_ratedPower'])
            # idx_rated = min([i for i, Pi in enumerate(P_find_rated) if Pi*1000 >= inputs['control_ratedPower']])
            # outputs['rated_V'] = inputs['V_out'][idx_rated]

            # if outputs['rated_V'] not in U:
            #     ## Run Rated
            #     TMax = 99999.
            #     # TMax = 10.
            #     turbulence_class = TURBULENCE_CLASS[inputs['turbulence_class']]
            #     turbine_class    = TURBINE_CLASS[inputs['turbine_class']]
            #     list_cases_rated, list_casenames_rated, requited_channels_rated = RotorSE_rated(self.fst_vt, self.FAST_runDirectory, self.FAST_namingOut, TMax, turbine_class, turbulence_class, outputs['rated_V'], U_init=self.U_init, Omega_init=self.Omega_init, pitch_init=np.zeros_like(self.Omega_init))
            #     requited_channels_rated = sorted(list(set(requited_channels_rated)))
            #     channels_out = {}
            #     for var in requited_channels_rated:
            #         channels_out[var] = True
            #     data_rated = self.run_FAST(self.fst_vt, list_cases_rated, list_casenames_rated, channels_out)[0]

            #     ## Sort in Rated Power
            #     U_wR = []
            #     data_wR = []
            #     U_added = False
            #     for i in range(len(U)):
            #         if outputs['rated_V']<U[i] and U_added == False:
            #             U_wR.append(outputs['rated_V'])
            #             data_wR.append(data_rated)
            #             U_added = True
            #         U_wR.append(U[i])
            #         data_wR.append(data[i])
            # else:
            #     U_wR = U

            # P_fast = np.array([np.mean(datai['GenPwr']) for datai in data_wR])*1000.
            # for i, (Pi, Vi) in enumerate(zip(P_fast, U_wR)):
            #     if Vi > outputs['rated_V']:
            #         if np.abs((Pi-inputs['control_ratedPower'])/inputs['control_ratedPower']) > 0.2:
            #             P_fast[i] = inputs['control_ratedPower']
            #             above_rate_power_warning = "FAST instability expected at U=%f m/s, abs(outputted power) > +/-20%% of rated power.  Replaceing %f with %f"%(Vi, Pi, inputs['control_ratedPower'])
            #             warnings.warn(above_rate_power_warning)

            # P_spline = PchipInterpolator(U_wR, P_fast)

            P_spline = PchipInterpolator(U, P_fast)

            P_out = P_spline(inputs['V_out'])
            # np.place(P_out, P_out>inputs['control_ratedPower'], inputs['control_ratedPower'])
            outputs['P_out'] = P_out

            P = P_spline(inputs['V'])
            # np.place(P, P>inputs['control_ratedPower'], inputs['control_ratedPower'])
            outputs['P'] = P


            outputs['Cp']          = np.mean(data_rated["RtAeroCp"])
            outputs['rated_V']     = np.mean(data_rated["Wind1VelX"])
            outputs['rated_Omega'] = np.mean(data_rated["RotSpeed"])
            outputs['rated_pitch'] = np.mean(data_rated["BldPitch1"])
            outputs['rated_T']     = np.mean(data_rated["RotThrust"])*1000
            outputs['rated_Q']     = np.mean(data_rated["RotTorq"])*1000

            # import matplotlib.pyplot as plt
            # plt.plot(U, P, 'o')
            # plt.plot(inputs['V_out'], outputs['P_out'])            
            # plt.show()

        ############

        Gust_Outputs = False
        Extreme_Outputs = False
        AEP_Outputs = False
        #
        for casei in case_keys:
            if Gust_Outputs and Extreme_Outputs:
                break

            if casei == 1:
                # power curve
                if AEP_Outputs:
                    pass
                else:
                    idx_AEP = [i for i, casej in enumerate(case_keys) if casej==1]
                    data = [datai for i, datai in enumerate(FAST_Output) if i in idx_AEP]
                    post_AEP(data)
                    AEP_Outputs = True

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





                

