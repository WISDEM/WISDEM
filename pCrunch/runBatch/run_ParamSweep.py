
# %%
"""
A python script to run a parameter sweep
"""
# Python tools
import numpy as np
import yaml
import os
import time 
# WISDEM tools
from wisdem.aeroelasticse import runFAST_pywrapper, CaseGen_General
from wisdem.aeroelasticse.Util import FileTools
# ROSCO tools
from ROSCO_toolbox import controller as ROSCO_controller
from ROSCO_toolbox import turbine as ROSCO_turbine
from ROSCO_toolbox import utilities as ROSCO_utilities
import pCrunch.CaseGen_Control as CaseGen_Control

# FLAGS
eagle = False
multi = False

# Controller tuning yaml
parameter_filename = '../../Turbine_Tuning/NREL5MW/NREL5MW.yaml'

# Controller parameters to study
input_params = ['omega_pc', 'zeta_pc']
DISCON_params = ['PC_GS_KP', 'PC_GS_KI']
values = [np.around(np.arange(0.3, 0.81, 0.1),    decimals=3),  # use np.around to avoid precision issues
               np.around(np.arange(0.7, 0.81, 0.1),    decimals=3)]
group       = 1

# Some path specifics/
FAST_InputFile = '5MW_Land.fst'    # FAST input file (ext=.fst)
FAST_directory = '/Users/nabbas/Documents/TurbineModels/NREL_5MW/5MW_Land'
FAST_runDirectory = 'temp'
wind_dir = '/Users/nabbas/Documents/WindEnergyToolbox/Turbine_Tuning/Parameter_Analysis/wind'
case_name_base = 'testing'
Turbsim_exe = 'turbsim_dev'
FAST_exe = 'openfast_dev'
debug_level = 2

# Wind Speeds
WindType = [3]
Uref = [11.0]

# Time
TMax = 1

# Turbine Definition
D = 126.
z_hub = 87

# Multiprocessing/Eagle related
if eagle:
    cores = 36
else:
    cores = 4

# Initialize CaseGen
cgc = CaseGen_Control.CaseGen_Control(parameter_filename)
cgc.AnalysisTime = TMax
cgc.case_name_base = case_name_base
cgc.D = D
cgc.z_hub = z_hub
cgc.debug_level = debug_level

# Generate wind speeds
cgc.seed = 136134
cgc.wind_dir = wind_dir
cgc.Turbsim_exe = Turbsim_exe
# wind_file, wind_file_type = cgc.gen_turbwind(Uref)
wind_file = 'unused'
wind_file_type = 1

# Generate control case inputs
# NOTE: Usually, group=- is easiest. Then some baseline characteristics in group 1, etc...
case_inputs, tuning_inputs = cgc.gen_control_cases(input_params, DISCON_params, values, group)

# Add time specification if group 0
if group == 0:
    ci_key = list(case_inputs.keys())[0]
    TMax_list = [TMax]*len(case_inputs[ci_key]['vals'])
    case_inputs[("Fst", "TMax")] = {'vals': TMax_list, 'group': 0}
else:
    case_inputs[("Fst", "TMax")] = {'vals': [TMax], 'group': 0}

# Wind
case_inputs[("InflowWind", "WindType")] = {'vals': [wind_file_type], 'group': 0}
case_inputs[("InflowWind", "Filename")] = {'vals': [wind_file], 'group': 0}

# Controller modes
for key in cgc.controller_params.keys():
    if 'mode' in key.lower():
        case_inputs[('DISCON_in', key)] = {'vals': [cgc.controller_params[key]],
                                           'group': 0}
# FAST details
fastBatch = runFAST_pywrapper.runFAST_pywrapper_batch(FAST_ver='OpenFAST', dev_branch=True)
fastBatch.FAST_exe = FAST_exe  # Path to executable
fastBatch.FAST_InputFile = FAST_InputFile
fastBatch.FAST_directory = FAST_directory
fastBatch.FAST_runDirectory = FAST_runDirectory
fastBatch.debug_level = 2

# Generate cases
case_list, case_name_list = CaseGen_General.CaseGen_General(
    case_inputs, dir_matrix=fastBatch.FAST_runDirectory, namebase=case_name_base)

# Append case matrix with controller tuning parameters
for file in os.listdir(fastBatch.FAST_runDirectory):
    if file.endswith(".yaml"):
        yfile = file
        yamldata = FileTools.load_yaml(os.path.join(fastBatch.FAST_runDirectory, yfile), package=1)

CaseGen_Control.append_case_matrix_yaml(
    fastBatch.FAST_runDirectory, yfile, tuning_inputs, 'tuning_inputs')

# Make sure flags are on
var_out = [
    # ElastoDyn
    "BldPitch1", "BldPitch2", "BldPitch3", "Azimuth", "RotSpeed", "GenSpeed", "NacYaw",
    "OoPDefl1", "IPDefl1", "TwstDefl1", "OoPDefl2", "IPDefl2", "TwstDefl2", "OoPDefl3",
    "IPDefl3", "TwstDefl3", "TwrClrnc1", "TwrClrnc2", "TwrClrnc3", "NcIMUTAxs", "NcIMUTAys",
    "NcIMUTAzs", "TTDspFA", "TTDspSS", "TTDspTwst", "PtfmSurge", "PtfmSway", "PtfmHeave",
    "PtfmRoll", "PtfmPitch", "PtfmYaw", "PtfmTAxt", "PtfmTAyt", "PtfmTAzt", "RootFxc1",
    "RootFyc1", "RootFzc1", "RootMxc1", "RootMyc1", "RootMzc1", "RootFxc2", "RootFyc2",
    "RootFzc2", "RootMxc2", "RootMyc2", "RootMzc2", "RootFxc3", "RootFyc3", "RootFzc3",
    "RootMxc3", "RootMyc3", "RootMzc3", "Spn1MLxb1", "Spn1MLyb1", "Spn1MLzb1", "Spn1MLxb2",
    "Spn1MLyb2", "Spn1MLzb2", "Spn1MLxb3", "Spn1MLyb3", "Spn1MLzb3", "RotThrust", "LSSGagFya",
    "LSSGagFza", "RotTorq", "LSSGagMya", "LSSGagMza", "YawBrFxp", "YawBrFyp", "YawBrFzp",
    "YawBrMxp", "YawBrMyp", "YawBrMzp", "TwrBsFxt", "TwrBsFyt", "TwrBsFzt", "TwrBsMxt",
    "TwrBsMyt", "TwrBsMzt", "TwHt1MLxt", "TwHt1MLyt", "TwHt1MLzt",
    # ServoDyn
    "GenPwr", "GenTq",
    # AeroDyn15
    "RtArea", "RtVAvgxh", "B1N3Clrnc", "B2N3Clrnc", "B3N3Clrnc",
    "RtAeroCp", 'RtAeroCq', 'RtAeroCt', 'RtTSR',
    # InflowWind
    "Wind1VelX", "Wind1VelY", "Wind1VelZ",
    # FLAPS
    # "BLFLAP1", "BLFLAP2", "BLFLAP3", "RtVAvgxh", "OoPDefl1")
]
channels = {}
for var in var_out:
    channels[var] = True
fastBatch.channels = channels

fastBatch.case_list = case_list
fastBatch.case_name_list = case_name_list

# %%
# if multi:
#     fastBatch.run_multi(cores)
#     # fastBatch.run_mpi()
# else:
#     fastBatch.run_serial()
