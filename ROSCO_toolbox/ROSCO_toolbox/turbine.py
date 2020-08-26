# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import os
import numpy as np
import datetime
from wisdem.ccblade import CCAirfoil, CCBlade
from scipy import interpolate
from numpy import gradient
import pickle
import matplotlib.pyplot as plt
import pandas as pd

from ROSCO_toolbox import utilities as ROSCO_utilities

# Some useful constants
now = datetime.datetime.now()
pi = np.pi
deg2rad = np.deg2rad(1)
rad2deg = np.rad2deg(1)
rpm2RadSec = 2.0*(np.pi)/60.0
RadSec2rpm = 60/(2.0 * np.pi)

class Turbine():
    """
    Class Turbine defines a turbine in terms of what is needed to 
        design the controller and to run the 'tiny' simulation.

    Primary functions (at a high level):
        - Reads an OpenFAST input deck 
        - Runs cc-blade rotor performance analysis
        - Writes a text file containing Cp, Ct, and Cq tables

    Methods:
    -------
    __str__
    load
    save
    load_from_fast
    load_from_ccblade
    load_from_txt
    generate_rotperf_fast
    write_rotor_performance

    Parameters:
    -----------
    turbine_params : dict
                    Dictionary containing known turbine paramaters that are not directly available from OpenFAST input files
    """

    def __init__(self, turbine_params):
        """
        Load turbine parameters from input dictionary
        """
        print('-----------------------------------------------------------------------------')
        print('Loading wind turbine data for NREL\'s ROSCO tuning and simulation processeses')
        print('-----------------------------------------------------------------------------')

        # ------ Turbine Parameters------
        self.rotor_inertia = turbine_params['rotor_inertia']         
        self.rated_rotor_speed = turbine_params['rated_rotor_speed']     
        self.v_min = turbine_params['v_min']                 
        self.v_rated = turbine_params['v_rated']               
        self.v_max = turbine_params['v_max']
        self.max_pitch_rate = turbine_params['max_pitch_rate'] 
        self.min_pitch_rate = -1 * self.max_pitch_rate
        self.max_torque_rate = turbine_params['max_torque_rate']             
        self.rated_power = turbine_params['rated_power']           
        self.bld_edgewise_freq = turbine_params['bld_edgewise_freq']     

        if turbine_params['twr_freq']:
            self.twr_freq = turbine_params['twr_freq']
        else:
            self.twr_freq = 0.0

        if turbine_params['ptfm_freq']:
            self.ptfm_freq = turbine_params['ptfm_freq']
        else:
            self.ptfm_freq = 0.0

        if turbine_params['TSR_operational']:
            self.TSR_operational = turbine_params['TSR_operational']
        else:
            self.TSR_operational = None

        if turbine_params['bld_flapwise_freq']:
            self.bld_flapwise_freq = turbine_params['bld_flapwise_freq']
        else:
            self.bld_flapwise_freq = 0.0


    # Allow print out of class
    def __str__(self): 
        '''
        Print some data about the turbine.
        '''
        print('-------------- Turbine Info --------------')
        print('Turbine Name: {}'.format(self.TurbineName))
        print('Rated Power: {} [W]'.format(self.rated_power))
        print('Total Inertia: {:.1f} [kg m^2]'.format(self.J))
        print('Rotor Radius: {:.1f} [m]'.format(self.rotor_radius))
        print('Rated Rotor Speed: {:.1f} [rad/s]'.format(self.rated_rotor_speed))
        print('Max Cp: {:.2f}'.format(self.Cp.max))
        print('------------------------------------------')
        return ' '

    # Save function
    def save(self,filename):
        '''
        Save turbine to pickle

        Parameters:
        ----------
        filename : str
                   Name of file to save pickle 
        # '''
        pickle.dump(self, open( filename, "wb" ) )

    # Load function
    @staticmethod
    def load(filename):
        '''
        Load turbine from pickle - outdated, but might be okay!!

        Parameters:
        ----------
        filename : str
                   Name of pickle file
        '''
        turbine = pickle.load(open(filename,'rb'))
        return turbine
    
    # Load data from fast input deck
    def load_from_fast(self, FAST_InputFile,FAST_directory, FAST_ver='OpenFAST',dev_branch=True,rot_source=None, txt_filename=None):
        """
        Load the parameter files directly from a FAST input deck

        Parameters:
        -----------
            Fast_InputFile: str
                            Primary fast model input file (*.fst)
            FAST_directory: str
                            Directory for primary fast model input file
            FAST_ver: string, optional
                      fast version, usually OpenFAST
            dev_branch: bool, optional
                        dev_branch input to InputReader_OpenFAST, probably True
            rot_source: str, optional
                        desired source for rotor to get Cp, Ct, Cq tables. Default is to run cc-blade. 
                            options: cc-blade - run cc-blade
                                     txt - from *.txt file
            txt_filename: str, optional
                          filename for *.txt, only used if rot_source='txt'
        """
        from wisdem.aeroelasticse.FAST_reader import InputReader_OpenFAST

        print('Loading FAST model: %s ' % FAST_InputFile)
        self.TurbineName = FAST_InputFile.strip('.fst')
        fast = self.fast = InputReader_OpenFAST(FAST_ver=FAST_ver,dev_branch=dev_branch)
        fast.FAST_InputFile = FAST_InputFile
        fast.FAST_directory = FAST_directory
        fast.execute()

        if txt_filename:
            self.rotor_performance_filename = txt_filename
        else:
            self.rotor_performance_filename = 'Cp_Ct_Cq.txt'


        # Grab general turbine parameters
        self.TipRad             = fast.fst_vt['ElastoDyn']['TipRad']
        self.Rhub               = fast.fst_vt['ElastoDyn']['HubRad']
        self.hubHt              = fast.fst_vt['ElastoDyn']['TowerHt'] + fast.fst_vt['ElastoDyn']['Twr2Shft']
        self.NumBl              = fast.fst_vt['ElastoDyn']['NumBl']
        self.TowerHt            = fast.fst_vt['ElastoDyn']['TowerHt']
        self.shearExp           = 0.2  #HARD CODED FOR NOW
        self.rho                = fast.fst_vt['AeroDyn15']['AirDens']
        self.mu                 = fast.fst_vt['AeroDyn15']['KinVisc']
        self.Ng                 = fast.fst_vt['ElastoDyn']['GBRatio']
        self.GenEff             = fast.fst_vt['ServoDyn']['GenEff']
        self.GBoxEff            = fast.fst_vt['ElastoDyn']['GBoxEff']
        self.DTTorSpr           = fast.fst_vt['ElastoDyn']['DTTorSpr']
        self.generator_inertia  = fast.fst_vt['ElastoDyn']['GenIner']
        self.tilt               = fast.fst_vt['ElastoDyn']['ShftTilt'] 
        try:
            self.precone = fast.fst_vt['ElastoDyn']['PreCone1'] # May need to change to PreCone(1) depending on OpenFAST files
        except:
            self.precone = fast.fst_vt['ElastoDyn']['PreCone(1)']
        self.yaw = 0.0
        self.J = self.rotor_inertia + self.generator_inertia * self.Ng**2
        self.rated_torque = self.rated_power/(self.GenEff/100*self.rated_rotor_speed*self.Ng)
        self.max_torque = self.rated_torque * 1.1
        self.rotor_radius = self.TipRad
        # self.omega_dt = np.sqrt(self.DTTorSpr/self.J)

        # Load Cp, Ct, Cq tables
        if rot_source == 'cc-blade': # Use cc-blade
            self.load_from_ccblade()
        elif rot_source == 'txt':    # Use specified text file
            file_processing = ROSCO_utilities.FileProcessing()
            self.pitch_initial_rad, self.TSR_initial, self.Cp_table, self.Ct_table, self.Cq_table = file_processing.load_from_txt(
                txt_filename)
        else:   # Use text file from DISCON.in
            if os.path.exists(os.path.join(FAST_directory, fast.fst_vt['ServoDyn']['DLL_InFile'])):
                try:
                    self.pitch_initial_rad = fast.fst_vt['DISCON_in']['Cp_pitch_initial_rad']
                    self.TSR_initial = fast.fst_vt['DISCON_in']['Cp_TSR_initial']
                    self.Cp_table = fast.fst_vt['DISCON_in']['Cp_table']
                    self.Ct_table = fast.fst_vt['DISCON_in']['Ct_table']
                    self.Cq_table = fast.fst_vt['DISCON_in']['Cq_table']
                except:   # Load from cc-blade
                    print('No rotor performance data source available, running CC-Blade.')
                    self.load_from_ccblade()

        # Parse rotor performance data
        self.Cp = RotorPerformance(self.Cp_table,self.pitch_initial_rad,self.TSR_initial)
        self.Ct = RotorPerformance(self.Ct_table,self.pitch_initial_rad,self.TSR_initial)
        self.Cq = RotorPerformance(self.Cq_table,self.pitch_initial_rad,self.TSR_initial)

        # Define operational TSR
        if not self.TSR_operational:
            self.TSR_operational = self.Cp.TSR_opt

        # Pull out some floating-related data
        wave_tp = fast.fst_vt['HydroDyn']['WaveTp'] 
        try:
            self.wave_peak_period = 1/wave_tp       # Will work if HydroDyn exists and a peak period is defined...
        except:
            self.wave_peak_period = 0.0             # Set as 0.0 when HydroDyn doesn't exist (fixed bottom)

    # Load rotor performance data from CCBlade 
    def load_from_ccblade(self):
        '''
        Loads rotor performance information by running cc-blade aerodynamic analysis. Designed to work with Aerodyn15 blade input files. 

        Parameters:
        -----------
            fast: dict
                  Dictionary containing fast model details - defined using from InputReader_OpenFAST (distributed as a part of AeroelasticSE)

        '''
        print('Loading rotor performance data from CC-Blade.')
        
        # Create CC-Blade Rotor
        r0 = np.array(self.fast.fst_vt['AeroDynBlade']['BlSpn']) 
        chord0 = np.array(self.fast.fst_vt['AeroDynBlade']['BlChord'])
        theta0 = np.array(self.fast.fst_vt['AeroDynBlade']['BlTwist'])
        # -- Adjust for Aerodyn15
        r = r0 + self.Rhub
        chord_intfun = interpolate.interp1d(r0,chord0, bounds_error=None, fill_value='extrapolate', kind='zero')
        chord = chord_intfun(r)
        theta_intfun = interpolate.interp1d(r0,theta0, bounds_error=None, fill_value='extrapolate', kind='zero')
        theta = theta_intfun(r)
        af_idx = np.array(self.fast.fst_vt['AeroDynBlade']['BlAFID']).astype(int) - 1 #Reset to 0 index
        AFNames = self.fast.fst_vt['AeroDyn15']['AFNames']   

        # Use airfoil data from FAST file read, assumes AeroDyn 15, assumes 1 Re num per airfoil
        af_dict = {}
        try:
            for i, _ in enumerate(self.fast.fst_vt['AeroDyn15']['af_data']):
                Re    = [self.fast.fst_vt['AeroDyn15']['af_data'][i][0]['Re']]
                Alpha = self.fast.fst_vt['AeroDyn15']['af_data'][i][0]['Alpha']
                Cl    = self.fast.fst_vt['AeroDyn15']['af_data'][i][0]['Cl']
                Cd    = self.fast.fst_vt['AeroDyn15']['af_data'][i][0]['Cd']
                Cm    = self.fast.fst_vt['AeroDyn15']['af_data'][i][0]['Cm']
                af_dict[i] = CCAirfoil(Alpha, Re, Cl, Cd, Cm)
        except: # Read airfoil tables without tab cabalities (will remove once wisdem master branch cleans up)
            for i, _ in enumerate(self.fast.fst_vt['AeroDyn15']['af_data']):
                Re    = [self.fast.fst_vt['AeroDyn15']['af_data'][i]['Re']]
                Alpha = self.fast.fst_vt['AeroDyn15']['af_data'][i]['Alpha']
                Cl    = self.fast.fst_vt['AeroDyn15']['af_data'][i]['Cl']
                Cd    = self.fast.fst_vt['AeroDyn15']['af_data'][i]['Cd']
                Cm    = self.fast.fst_vt['AeroDyn15']['af_data'][i]['Cm']
                af_dict[i] = CCAirfoil(Alpha, Re, Cl, Cd, Cm)

        # define airfoils for CCBlade
        af = [0]*len(r)
        for i in range(len(r)):
            af[i] = af_dict[af_idx[i]]

        # Now save the CC-Blade rotor
        nSector = 8  # azimuthal discretizations
        self.cc_rotor = CCBlade(r, chord, theta, af, self.Rhub, self.rotor_radius, self.NumBl, rho=self.rho, mu=self.mu,
                        precone=-self.precone, tilt=-self.tilt, yaw=self.yaw, shearExp=self.shearExp, hubHt=self.hubHt, nSector=nSector)
        print('CCBlade initiated successfully.')
        
        # Generate the look-up tables, mesh the grid and flatten the arrays for cc_rotor aerodynamic analysis
        TSR_initial = np.arange(3, 15,0.5)
        pitch_initial = np.arange(-1,25,0.5)
        pitch_initial_rad = pitch_initial * deg2rad
        ws_array = np.ones_like(TSR_initial) * self.v_rated # evaluate at rated wind speed
        omega_array = (TSR_initial * ws_array / self.rotor_radius) * RadSec2rpm
        ws_mesh, pitch_mesh = np.meshgrid(ws_array, pitch_initial)
        ws_flat = ws_mesh.flatten()
        pitch_flat = pitch_mesh.flatten()
        omega_mesh, _ = np.meshgrid(omega_array, pitch_initial)
        omega_flat = omega_mesh.flatten()
        # tsr_flat = (omega_flat * rpm2RadSec * self.rotor_radius)  / ws_flat


        # Get values from cc-blade
        print('Running CCBlade aerodynamic analysis, this may take a minute...')
        try:
            _, _, _, _, CP, CT, CQ, _ = self.cc_rotor.evaluate(ws_flat, omega_flat, pitch_flat, coefficients=True)
        except ValueError: # On IEAontology4all
            outputs, derivs = self.cc_rotor.evaluate(ws_flat, omega_flat, pitch_flat, coefficients=True)
            CP = outputs['CP']
            CT = outputs['CT']
            CQ = outputs['CQ']
        print('CCBlade aerodynamic analysis run successfully.')

        # Reshape Cp, Ct and Cq
        Cp = np.transpose(np.reshape(CP, (len(pitch_initial), len(TSR_initial))))
        Ct = np.transpose(np.reshape(CT, (len(pitch_initial), len(TSR_initial))))
        Cq = np.transpose(np.reshape(CQ, (len(pitch_initial), len(TSR_initial))))

        # Store necessary metrics for analysis
        self.pitch_initial_rad = pitch_initial_rad
        self.TSR_initial = TSR_initial
        self.Cp_table = Cp
        self.Ct_table = Ct 
        self.Cq_table = Cq
        
        # Save some blade parameters
        self.span = r
        self.chord = chord
        self.twist = theta
    
    def generate_rotperf_fast(self, openfast_path, FAST_runDirectory=None, run_BeamDyn=False,
                              debug_level=1, run_type='multi'):
        '''
        Use openfast to generate Cp surface data. Will be slow, especially if using BeamDyn,
        but may be necessary if cc-blade is not sufficient.

        Parameters:
        -----------
        openfast_path: str
            path to openfast
        FAST_runDirectory: str
            directory to run openfast simulations in
        run_BeamDyn: bool
            Flag to run beamdyn - does not exist yet
        debug_level: float
            0 - no outputs, 1 - simple outputs, 2 - all outputs
        run_type: str
            'serial' - run in serial, 'multi' - run using python multiprocessing tools, 
            'mpi' - run using mpi tools
        '''

        # Load additional WISDEM tools
        from wisdem.aeroelasticse import runFAST_pywrapper, CaseGen_General
        from wisdem.aeroelasticse.Util import FileTools
        # Load pCrunch tools
        from pCrunch import pdTools, Processing


        # setup values for surface
        v0 = self.v_rated + 2
        TSR_initial = np.arange(3, 15,1)
        pitch_initial = np.arange(-1,25,1)
        rotspeed_initial = TSR_initial*v0/self.rotor_radius * RadSec2rpm # rpms

        # Specify Case Inputs
        case_inputs = {}

        # ------- Setup OpenFAST inputs --------
        case_inputs[('Fst','TMax')] = {'vals': [330], 'group': 0}
        case_inputs[('Fst','Compinflow')] = {'vals': [1], 'group': 0}
        case_inputs[('Fst','CompAero')] = {'vals': [2], 'group': 0}
        case_inputs[('Fst','CompServo')] = {'vals': [1], 'group': 0}
        case_inputs[('Fst','CompHydro')] = {'vals': [0], 'group': 0}
        if run_BeamDyn:
            case_inputs[('Fst','CompElast')] = {'vals': [2], 'group': 0}
        else:
            case_inputs[('Fst','CompElast')] = {'vals': [1], 'group': 0}
        case_inputs[('Fst', 'OutFileFmt')] = {'vals': [2], 'group': 0}

        # AeroDyn15
        case_inputs[('AeroDyn15', 'WakeMod')] = {'vals': [1], 'group': 0}
        case_inputs[('AeroDyn15', 'AfAeroMod')] = {'vals': [1], 'group': 0}
        case_inputs[('AeroDyn15', 'TwrPotent')] = {'vals': [0], 'group': 0}
        
        # ElastoDyn
        case_inputs[('ElastoDyn', 'FlapDOF1')] = {'vals': ['True'], 'group': 0}
        case_inputs[('ElastoDyn', 'FlapDOF2')] = {'vals': ['True'], 'group': 0}
        case_inputs[('ElastoDyn', 'EdgeDOF')] = {'vals': ['True'], 'group': 0}
        case_inputs[('ElastoDyn', 'TeetDOF')] = {'vals': ['False'], 'group': 0}
        case_inputs[('ElastoDyn', 'DrTrDOF')] = {'vals': ['False'], 'group': 0}
        case_inputs[('ElastoDyn', 'GenDOF')] = {'vals': ['False'], 'group': 0}
        case_inputs[('ElastoDyn', 'YawDOF')] = {'vals': ['False'], 'group': 0}
        case_inputs[('ElastoDyn', 'TwFADOF1')] = {'vals': ['False'], 'group': 0}
        case_inputs[('ElastoDyn', 'TwFADOF2')] = {'vals': ['False'], 'group': 0}
        case_inputs[('ElastoDyn', 'TwSSDOF1')] = {'vals': ['False'], 'group': 0}
        case_inputs[('ElastoDyn', 'TwSSDOF2')] = {'vals': ['False'], 'group': 0}
        case_inputs[('ElastoDyn', 'PtfmSgDOF')] = {'vals': ['False'], 'group': 0}
        case_inputs[('ElastoDyn', 'PtfmSwDOF')] = {'vals': ['False'], 'group': 0}
        case_inputs[('ElastoDyn', 'PtfmHvDOF')] = {'vals': ['False'], 'group': 0}
        case_inputs[('ElastoDyn', 'PtfmRDOF')] = {'vals': ['False'], 'group': 0}
        case_inputs[('ElastoDyn', 'PtfmPDOF')] = {'vals': ['False'], 'group': 0}
        case_inputs[('ElastoDyn', 'PtfmYDOF')] = {'vals': ['False'], 'group': 0}

        # BeamDyn
        # NEEDED

        # InflowWind
        case_inputs[('InflowWind', 'WindType')] = {'vals': [1], 'group': 0}
        case_inputs[('InflowWind', 'HWindSpeed')] = {'vals': [v0], 'group': 0}
        case_inputs[('InflowWind', 'PLexp')] = {'vals': [0], 'group': 0}

        # ServoDyn
        case_inputs[('ServoDyn', 'PCMode')] = {'vals': [0], 'group': 0}
        case_inputs[('ServoDyn', 'VSContrl')] = {'vals': [0], 'group': 0} 
        case_inputs[('ServoDyn', 'HSSBrMode')] = {'vals': [0], 'group': 0}
        case_inputs[('ServoDyn', 'YCMode')] = {'vals': [0], 'group': 0}

        # ------- Setup sweep values inputs --------
        case_inputs[('ElastoDyn', 'BlPitch1')] = {'vals': list(pitch_initial), 'group': 1}
        case_inputs[('ElastoDyn', 'BlPitch2')] = {'vals': list(pitch_initial), 'group': 1}
        case_inputs[('ElastoDyn', 'BlPitch3')] = {'vals': list(pitch_initial), 'group': 1}
        case_inputs[('ElastoDyn', 'RotSpeed')] = {'vals': list(rotspeed_initial), 'group': 2}


        # FAST details
        fastBatch = runFAST_pywrapper.runFAST_pywrapper_batch(FAST_ver='OpenFAST', dev_branch=True)
        fastBatch.FAST_exe = openfast_path  # Path to executable
        fastBatch.FAST_InputFile = self.fast.FAST_InputFile
        fastBatch.FAST_directory = self.fast.FAST_directory
        if not FAST_runDirectory:
            FAST_runDirectory = os.path.join(os.getcwd(), 'RotPerf_OpenFAST')
        fastBatch.FAST_runDirectory = FAST_runDirectory
        fastBatch.debug_level = debug_level

        # Generate cases
        case_name_base = self.TurbineName + '_rotperf'
        case_list, case_name_list = CaseGen_General.CaseGen_General(
            case_inputs, dir_matrix=fastBatch.FAST_runDirectory, namebase=case_name_base)
        fastBatch.case_list = case_list
        fastBatch.case_name_list = case_name_list

        # Make sure proper outputs exist
        var_out = [
            # ElastoDyn (this is probably overkill on the outputs)
            "BldPitch1", "BldPitch2", "BldPitch3", "Azimuth", "RotSpeed", "GenSpeed", "NacYaw",
            "OoPDefl1", "IPDefl1", "TwstDefl1", "OoPDefl2", "IPDefl2", "TwstDefl2", "OoPDefl3",
            "IPDefl3", "TwstDefl3", "RootFxc1",
            "RootFyc1", "RootFzc1", "RootMxc1", "RootMyc1", "RootMzc1", "RootFxc2", "RootFyc2",
            "RootFzc2", "RootMxc2", "RootMyc2", "RootMzc2", "RootFxc3", "RootFyc3", "RootFzc3",
            "RootMxc3", "RootMyc3", "RootMzc3", "Spn1MLxb1", "Spn1MLyb1", "Spn1MLzb1", "Spn1MLxb2",
            "Spn1MLyb2", "Spn1MLzb2", "Spn1MLxb3", "Spn1MLyb3", "Spn1MLzb3", "RotThrust", "LSSGagFya",
            "LSSGagFza", "RotTorq", "LSSGagMya", "LSSGagMza", 
            # ServoDyn
            "GenPwr", "GenTq",
            # AeroDyn15
            "RtArea", "RtVAvgxh", "B1N3Clrnc", "B2N3Clrnc", "B3N3Clrnc",
            "RtAeroCp", 'RtAeroCq', 'RtAeroCt', 'RtTSR', # NECESSARY
            # InflowWind
            "Wind1VelX", 
        ]
        channels = {}
        for var in var_out:
            channels[var] = True
        fastBatch.channels = channels

        # Run OpenFAST
        if run_type.lower() == 'multi':
            fastBatch.run_multi()
        elif run_type.lower()=='mpi':
            fastBatch.run_mpi()
        elif run_type.lower()=='serial':
            fastBatch.run_serial()

        # ========== Post Processing ==========
        # Save statistics
        fp = Processing.FAST_Processing()

        # Find all outfiles
        fname_case_matrix = os.path.join(FAST_runDirectory, 'case_matrix.yaml')
        case_matrix = FileTools.load_yaml(fname_case_matrix, package=1)
        cm = pd.DataFrame(case_matrix)
        # Parse case matrix and find outfiles names
        outfiles = []
        case_names = cm['Case_Name']
        outfiles = []
        for name in case_names:
            outfiles.append(os.path.join(FAST_runDirectory, name + '.outb'))



        # Set some processing parameters
        fp.OpenFAST_outfile_list = outfiles
        fp.namebase = case_name_base
        fp.t0 = 270 
        fp.parallel_analysis = True
        fp.results_dir = os.path.join(FAST_runDirectory,'stats')
        fp.verbose = True
        # Save for debug!
        fp.save_LoadRanking = False
        fp.save_SummaryStats = False

        print('Processing openfast data on {} cores.'.format(fp.parallel_cores))

        # Load and save statistics and load rankings
        stats, load_rankings = fp.batch_processing()

        # Get means of last 30 seconds of 300 second simulation
        CP = stats[0]['RtAeroCp']['mean']
        CT = stats[0]['RtAeroCt']['mean']
        CQ = stats[0]['RtAeroCq']['mean']

        # Reshape Cp, Ct and Cq
        Cp = np.transpose(np.reshape(CP, (len(pitch_initial), len(TSR_initial))))
        Ct = np.transpose(np.reshape(CT, (len(pitch_initial), len(TSR_initial))))
        Cq = np.transpose(np.reshape(CQ, (len(pitch_initial), len(TSR_initial))))

        # Store necessary metrics for analysis
        self.pitch_initial_rad = pitch_initial * deg2rad
        self.TSR_initial = TSR_initial
        self.Cp_table = Cp
        self.Ct_table = Ct
        self.Cq_table = Cq

    
    def load_blade_info(self):
        '''
        Loads wind turbine blade data from an OpenFAST model. 
        Should be used if blade information is needed (i.e.) for flap controller tuning, but a rotor performance file is defined and and cc-blade is not run already. 
        
        Parameters:
        -----------
            self - note: needs to contain fast input file info provided by load_from_fast.
        '''
        from wisdem.aeroelasticse.FAST_reader import InputReader_OpenFAST

        # Load Fast input deck
        # self.TurbineName = FAST_InputFile.strip('.fst')
        # fast = InputReader_OpenFAST(FAST_ver=FAST_ver,dev_branch=dev_branch)
        # self.fast.FAST_InputFile = FAST_InputFile
        # self.fast.FAST_directory = FAST_directory
        # self.fast.execute()

        # Make sure cc_rotor exists for DAC analysis
        try:
            if self.cc_rotor:
                self.af_data = self.fast.fst_vt['AeroDyn15']['af_data']
                self.bld_flapwise_damp = self.fast.fst_vt['ElastoDynBlade']['BldFlDmp1']/100 * 0.7
        except AttributeError:
            # Create CC-Blade Rotor
            r0 = np.array(self.fast.fst_vt['AeroDynBlade']['BlSpn']) 
            chord0 = np.array(self.fast.fst_vt['AeroDynBlade']['BlChord'])
            theta0 = np.array(self.fast.fst_vt['AeroDynBlade']['BlTwist'])
            # -- Adjust for Aerodyn15
            r = r0 + self.Rhub
            chord_intfun = interpolate.interp1d(r0,chord0, bounds_error=None, fill_value='extrapolate', kind='zero')
            chord = chord_intfun(r)
            theta_intfun = interpolate.interp1d(r0,theta0, bounds_error=None, fill_value='extrapolate', kind='zero')
            theta = theta_intfun(r)
            af_idx = np.array(self.fast.fst_vt['AeroDynBlade']['BlAFID']).astype(int) - 1 #Reset to 0 index
            AFNames = self.fast.fst_vt['AeroDyn15']['AFNames']   

            # Use airfoil data from FAST file read, assumes AeroDyn 15, assumes 1 Re num per airfoil
            af_dict = {}
            for i, section in enumerate(self.fast.fst_vt['AeroDyn15']['af_data']):
                Re = [section[0]['Re']]
                Alpha = section[0]['Alpha']
                if section[0]['NumTabs'] > 1:  # sections with flaps
                    ref_tab = int(np.floor(section[0]['NumTabs']/2))
                    Cl = section[ref_tab]['Cl']
                    Cd = section[ref_tab]['Cd']
                    Cm = section[ref_tab]['Cm']
                    af_dict[i] = CCAirfoil(Alpha, Re, Cl, Cd, Cm)
                else:                           # sections without flaps
                    Cl = section[0]['Cl']
                    Cd = section[0]['Cd']
                    Cm = section[0]['Cm']
                    af_dict[i] = CCAirfoil(Alpha, Re, Cl, Cd, Cm)





            # define airfoils for CCBlade
            af = [0]*len(r)
            for i in range(len(r)):
                af[i] = af_dict[af_idx[i]]

            # Now save the CC-Blade rotor
            nSector = 8  # azimuthal discretizations
            self.cc_rotor = CCBlade(r, chord, theta, af, self.Rhub, self.rotor_radius, self.NumBl, rho=self.rho, mu=self.mu,
                            precone=-self.precone, tilt=-self.tilt, yaw=self.yaw, shearExp=self.shearExp, hubHt=self.hubHt, nSector=nSector)

            # Save some blade  data 
            self.af_data = self.fast.fst_vt['AeroDyn15']['af_data']
            self.span = r 
            self.chord = chord
            self.twist = theta
            self.bld_flapwise_damp = self.fast.fst_vt['ElastoDynBlade']['BldFlDmp1']/100 * 0.7

class RotorPerformance():
    '''
    Class RotorPerformance used to find details from rotor performance 
        tables generated by CC-blade or similer BEM-solvers. 

    Methods:
    --------
    interp_surface
    interp_gradient
    plot_performance

    Parameters:
    -----------
    performance_table : array_like (-)
                        An [n x m] array containing a table of rotor performance data (Cp, Ct, Cq).
    pitch_initial_rad : array_like (rad)
                        An [m x 1] or [1 x m] array containing blade pitch angles corresponding to performance_table. 
    TSR_initial : array_like (rad)
                    An [n x 1] or [1 x n] array containing tip-speed ratios corresponding to  performance_table 
    '''
    def __init__(self,performance_table, pitch_initial_rad, TSR_initial):

        # Store performance data tables
        self.performance_table = performance_table          # Table containing rotor performance data, i.e. Cp, Ct, Cq
        self.pitch_initial_rad = pitch_initial_rad          # Pitch angles corresponding to x-axis of performance_table (rad)
        self.TSR_initial = TSR_initial                      # Tip-Speed-Ratios corresponding to y-axis of performance_table (rad)

        # Calculate Gradients
        self.gradient_TSR, self.gradient_pitch = gradient(performance_table)             # gradient_TSR along y-axis, gradient_pitch along x-axis (rows, columns)

        # "Optimal" below rated TSR and blade pitch (for Cp) - note this may be limited by resolution of Cp-surface
        self.max = np.amax(performance_table)
        self.max_ind = np.where(performance_table == np.amax(performance_table))
        self.pitch_opt = pitch_initial_rad[self.max_ind[1]]
        # --- Find TSR ---
        # Make finer mesh for Tip speed ratios at "optimal" blade pitch angle, do a simple lookup. 
        #       -- nja: this seems to work a little better than interpolating
        performance_beta_max = np.ndarray.flatten(performance_table[:,self.max_ind[1]]) # performance metric at maximizing pitch angle
        TSR_ind = np.arange(0,len(TSR_initial))
        TSR_fine_ind = np.linspace(TSR_initial[0],TSR_initial[-1],int(TSR_initial[-1] - TSR_initial[0])*100)
        f_TSR = interpolate.interp1d(TSR_initial,TSR_initial,bounds_error='False',kind='quadratic')    # interpolate function for Cp(tsr) values
        TSR_fine = f_TSR(TSR_fine_ind)
        f_performance = interpolate.interp1d(TSR_initial,performance_beta_max,bounds_error='False',kind='quadratic')    # interpolate function for Cp(tsr) values
        performance_fine = f_performance(TSR_fine_ind)
        performance_max_ind = np.where(performance_fine == np.max(performance_fine))
        self.TSR_opt = float(TSR_fine[performance_max_ind[0]])

    def interp_surface(self,pitch,TSR):
        '''
        2d interpolation to find point on rotor performance surface
        
        Parameters:
        -----------
        pitch : float (rad)
                Pitch angle to look up
        TSR : float (rad)
              Tip-speed ratio to look up
        '''
        
        # Form the interpolant functions which can look up any arbitrary location on rotor performance surface
        interp_fun = interpolate.interp2d(self.pitch_initial_rad, self.TSR_initial, self.performance_table, kind='linear')
        return interp_fun(pitch,TSR)

    def interp_gradient(self,pitch,TSR):
        '''
        2d interpolation to find gradient at a specified point on rotor performance surface
        
        Parameters:
        -----------
        pitch : float (rad)
                Pitch angle to look up
        TSR : float (rad)
              Tip-speed ratio to look up

        Returns:
        --------
        interp_gradient : array_like
                          [1 x 2] array coresponding to gradient in pitch and TSR directions, respectively
        '''
        # Form the interpolant functions to find gradient at any arbitrary location on rotor performance surface
        dCP_beta_interp = interpolate.interp2d(self.pitch_initial_rad, self.TSR_initial, self.gradient_pitch, kind='linear')
        dCP_TSR_interp = interpolate.interp2d(self.pitch_initial_rad, self.TSR_initial, self.gradient_TSR, kind='linear')

        # grad.shape output as (2,) numpy array, equivalent to (pitch-direction,TSR-direction)
        grad = np.array([dCP_beta_interp(pitch,TSR), dCP_TSR_interp(pitch,TSR)])
        return np.ndarray.flatten(grad)
    
    def plot_performance(self,performance_table, pitch_initial_rad, TSR_initial):
        '''
        Plot rotor performance data surface. 
        
        Parameters:
        -----------
        performance_table : array_like (-)
                            An [n x m] array containing a table of rotor performance data (Cp, Ct, Cq) to plot.
        pitch_initial_rad : array_like (rad)
                            An [m x 1] or [1 x m] array containing blade pitch angles corresponding to performance_table (x-axis). 
        TSR_initial : array_like (rad)
                      An [n x 1] or [1 x n] array containing tip-speed ratios corresponding to  performance_table (y-axis)
        '''

        P = plt.contour(pitch_initial_rad * rad2deg, TSR_initial, performance_table, levels=[0.0, 0.3, 0.40, 0.42, 0.44, 0.45, 0.46, 0.47, 0.48,0.481,0.482,0.483,0.484,0.485,0.486,0.487,0.488,0.489, 0.49, 0.491,0.492,0.493,0.494,0.495,0.496,0.497,0.498,0.499, 0.50 ])
        plt.clabel(P, inline=1, fontsize=12)
        plt.title('Power Coefficient', fontsize=14, fontweight='bold')
        plt.xlabel('Pitch Angle [deg]', fontsize=14, fontweight='bold')
        plt.ylabel('TSR [-]', fontsize=14, fontweight='bold')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(color=[0.8,0.8,0.8], linestyle='--')
        plt.subplots_adjust(bottom = 0.15, left = 0.15)
