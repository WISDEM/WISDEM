'''
Run ROSCO and test against baseline results:
    - set up case(s) using CaseLibrary from aeroelasticse
    - run fast simulations
    - evaluate & compare results with pCrunch
    - report results to user


'''

import numpy as np
# from wisdem.aeroelasticse.CaseLibrary import ROSCO_Test
import os

from wisdem.aeroelasticse.FAST_reader import InputReader_Common, InputReader_OpenFAST, InputReader_FAST7
from wisdem.aeroelasticse.FAST_writer import InputWriter_Common, InputWriter_OpenFAST, InputWriter_FAST7
from wisdem.aeroelasticse.runFAST_pywrapper import runFAST_pywrapper_batch
from wisdem.aeroelasticse.CaseGen_General import CaseGen_General
from wisdem.aeroelasticse.CaseGen_IEC import CaseGen_IEC
from pCrunch import pdTools
from pCrunch import Processing, Analysis
from wisdem.aeroelasticse.Util import FileTools
import pandas as pd

# Moved ROSCO_Test into toolbox so that it doesn't rely on my fork of aeroelasticse
def ROSCO_Test(fst_vt, runDir, namebase, TMax, turbine_class, turbulence_class, Turbsim_exe='', debug_level=0, cores=0, mpi_run=False, mpi_comm_map_down=[]):

    iec = CaseGen_IEC()

    # I'd like to start them all at the same place.  The controller should be able to handle startup transients.
    iec.init_cond[("ElastoDyn","RotSpeed")] = {'U':  [2,30]}
    iec.init_cond[("ElastoDyn","RotSpeed")]['val'] = np.ones([2]) * fst_vt['ElastoDyn']['RotSpeed'] *.75
    iec.init_cond[("ElastoDyn","BlPitch1")] = {'U':  [2,30]}
    iec.init_cond[("ElastoDyn","BlPitch1")]['val'] = np.ones([2]) * 0
    iec.init_cond[("ElastoDyn","BlPitch2")] = iec.init_cond[("ElastoDyn","BlPitch1")]
    iec.init_cond[("ElastoDyn","BlPitch3")] = iec.init_cond[("ElastoDyn","BlPitch1")]
    iec.Turbine_Class = turbine_class
    iec.Turbulence_Class = turbulence_class
    iec.D = fst_vt['ElastoDyn']['TipRad']*2.
    iec.z_hub = fst_vt['InflowWind']['RefHt']

    iec.dlc_inputs = {}
    iec.dlc_inputs['DLC']   = [1.3,1.4]#,6.1,6.3]
    iec.dlc_inputs['U']     = [[4,6,8,10,12,14,16,18,20,22,24],[8.88,12.88]]
    # iec.dlc_inputs['Seeds'] = [[1]]
    iec.dlc_inputs['Seeds'] = [[1],[]] # nothing special about these seeds, randomly generated (???)
    iec.dlc_inputs['Yaw']   = [[], []]
    iec.transient_dir_change        = '-'  # '+','-','both': sign for transient events in EDC, EWS
    iec.transient_shear_orientation = 'v'  # 'v','h','both': vertical or horizontal shear for EWS

    iec.wind_dir        = os.path.join(runDir,'wind')
    iec.case_name_base  = namebase
    iec.Turbsim_exe     = Turbsim_exe
    iec.debug_level     = debug_level
    iec.cores           = cores
    iec.run_dir         = runDir
    iec.overwrite       = False
    # iec.overwrite       = False
    if cores > 1:
        iec.parallel_windfile_gen = True
    else:
        iec.parallel_windfile_gen = False

    # mpi_run = False
    if mpi_run:
        iec.mpi_run           = mpi_run
        iec.comm_map_down = mpi_comm_map_down

    case_inputs = {}
    case_inputs[("Fst","TMax")]              = {'vals':[TMax], 'group':0}
    case_inputs[("Fst","OutFileFmt")]        = {'vals':[2], 'group':0}

    case_inputs[('ServoDyn','GenTiStr')]     = {'vals': ['False'], 'group': 0}
    case_inputs[('ServoDyn','GenTiStp')]     = {'vals': ['True'], 'group': 0}
    case_inputs[('ServoDyn','SpdGenOn')]     = {'vals': [0.], 'group': 0}

    case_inputs[("AeroDyn15","WakeMod")]     = {'vals':[1], 'group':0}
    case_inputs[("AeroDyn15","AFAeroMod")]   = {'vals':[2], 'group':0}
    case_inputs[("AeroDyn15","TwrPotent")]   = {'vals':[0], 'group':0}
    case_inputs[("AeroDyn15","TwrShadow")]   = {'vals':['False'], 'group':0}
    case_inputs[("AeroDyn15","TwrAero")]     = {'vals':['False'], 'group':0}
    case_inputs[("AeroDyn15","SkewMod")]     = {'vals':[1], 'group':0}
    case_inputs[("AeroDyn15","TipLoss")]     = {'vals':['True'], 'group':0}
    case_inputs[("AeroDyn15","HubLoss")]     = {'vals':['True'], 'group':0}
    case_inputs[("AeroDyn15","TanInd")]      = {'vals':['True'], 'group':0}
    case_inputs[("AeroDyn15","AIDrag")]      = {'vals':['True'], 'group':0}
    case_inputs[("AeroDyn15","TIDrag")]      = {'vals':['True'], 'group':0}
    case_inputs[("AeroDyn15","IndToler")]    = {'vals':[1.e-5], 'group':0}
    case_inputs[("AeroDyn15","MaxIter")]     = {'vals':[5000], 'group':0}
    case_inputs[("AeroDyn15","UseBlCm")]     = {'vals':['True'], 'group':0}
    
    case_list, case_name_list = iec.execute(case_inputs=case_inputs)

    channels  = ["TipDxc1", "TipDyc1", "TipDzc1", "TipDxc2", "TipDyc2", "TipDzc2", "TipDxc3", "TipDyc3", "TipDzc3"]
    channels += ["RootMxc1", "RootMyc1", "RootMzc1", "RootMxc2", "RootMyc2", "RootMzc2", "RootMxc3", "RootMyc3", "RootMzc3"]

    return case_list, case_name_list, channels


if __name__ == '__main__':

    ### Parameter Set up
    # Run Directory (for all turbines)
    baseRunDir          = '/Users/dzalkind/Tools/SaveData/ROSCO/WSE_Params'

    # Compare results to this Directory:
    compRunDir          = '/Users/dzalkind/Tools/SaveData/ROSCO/PitchSat'
    
    # Executable paths
    Turbsim_exe         = '/Users/dzalkind/Tools/openfast/build/modules/turbsim/turbsim'  # move up
    Openfast_exe        = '/Users/dzalkind/Tools/openfast/install/bin/openfast'

    # Runtime options
    cores               = 8 # how many cores? if > 1, will run in parallel but might be harder to debug
    overwrite           = False  # do you want to overwrite fast sims?
    reCrunch            = False  # do you want to re-run pCrunch?

    # Turbine Setup
    testTurbines = ['IEA-15MW']   # current options are: IEA-15MW, NREL-5MW

    # Loop through test turbines
    for turbine in testTurbines:

        # Get FAST input (15MW only for now)
        if turbine == 'IEA-15MW':
            dev_branch = True
            FAST_ver = 'OpenFAST'

            fastRead = InputReader_OpenFAST(FAST_ver=FAST_ver, dev_branch=dev_branch)
            fastRead.FAST_InputFile = 'IEA-15-240-RWT-UMaineSemi.fst'   # FAST input file (ext=.fst)
            fastRead.FAST_directory = '/Users/dzalkind/Tools/ROSCO_toolbox/Test_Cases/IEA-15-240-RWT-UMaineSemi'   # Path to fst directory files
            
        elif turbine == 'NREL-5MW':
            dev_branch = True
            FAST_ver = 'OpenFAST'

            fastRead = InputReader_OpenFAST(FAST_ver=FAST_ver, dev_branch=dev_branch)
            fastRead.FAST_InputFile = '5MW_Land_DLL_WTurb.fst'   # FAST input file (ext=.fst)
            fastRead.FAST_directory = '/Users/dzalkind/Tools/ROSCO_toolbox/Test_Cases/5MW_Land_DLL_WTurb'   # Path to fst directory files

        # Read FAST inputs for generating cases
        fastRead.execute()


        runDir = os.path.join(baseRunDir,turbine)
        namebase = 'ROTest'
        TMax     = 700
        turbine_class = 'I'
        turbulence_class = 'A'


        # Generate Cases
        case_list, case_name_list, channels = ROSCO_Test(fastRead.fst_vt, runDir, namebase, TMax, turbine_class, turbulence_class, \
            Turbsim_exe=Turbsim_exe, debug_level=2, cores=cores, mpi_run=False, mpi_comm_map_down=[])


        # Set up FAST Sims
        fastBatch = runFAST_pywrapper_batch(FAST_ver='OpenFAST', dev_branch=True)
        fastBatch.FAST_exe = Openfast_exe   # Path to executable
        fastBatch.FAST_runDirectory = os.path.join(baseRunDir,turbine)
        fastBatch.FAST_InputFile = fastRead.FAST_InputFile  # FAST input file (ext=.fst)
        fastBatch.FAST_directory = fastRead.FAST_directory   # Path to fst directory files
        fastBatch.debug_level       = 2

        fastBatch.case_list = case_list
        fastBatch.case_name_list = case_name_list
        fastBatch.debug_level = 2
        fastBatch.dev_branch = True

        # Check if simulation has been run
        outFileNames = [os.path.join(fastBatch.FAST_runDirectory,case_name + '.outb') for case_name in case_name_list]
        outFileThere = [os.path.exists(outFileName) for outFileName in outFileNames]
        # print('here')
        
        # Run simulations if they're not all there or if you want to overwrite
        if not all(outFileThere) or overwrite:
            if cores > 1:
                fastBatch.run_multi(cores)
            else:
                fastBatch.run_serial()
        
        

        

    