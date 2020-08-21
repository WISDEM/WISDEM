'''
Run DLC test suites

Kind of like run DMC, but less hip-hop, so less fun. 
'''
import os
from wisdem.aeroelasticse.CaseGen_IEC import CaseGen_IEC
from wisdem.aeroelasticse.runFAST_pywrapper import runFAST_pywrapper_batch
from wisdem.aeroelasticse.Util import FileTools
# FLAGS
eagle = True
multi = True
floating = False
rosco = True
save_stats = True

# Debug
debug_level = 2

# Input filepaths, etc...
if eagle:
    FAST_exe = 'openfast'
    Turbsim_exe = 'turbsim'
    cores = 36
    if floating:
        FAST_directory = '/projects/ssc/nabbas/TurbineModels/5MW_OC3Spar_DLL_WTurb_WavesIrr'
        FAST_InputFile = '5MW_OC3Spar_DLL_WTurb_WavesIrr.fst'
        dll_filename = ['/home/nabbas/ROSCO_toolbox/ROSCO/build/libdiscon.so',
                        '/projects/ssc/nabbas/TurbineModels/5MW_Baseline/ServoData/DISCON_OC3/build/DISCON_OC3Hywind.dll']
    else:
        FAST_directory = '/projects/ssc/nabbas/TurbineModels/5MW_Land_DLL_WTurb'  
        FAST_InputFile = '5MW_Land_DLL_WTurb.fst'  
        dll_filename = ['/home/nabbas/ROSCO_toolbox/ROSCO/build/libdiscon.so',
                        '/projects/ssc/nabbas/TurbineModels/5MW_Baseline/ServoData/DISCON/build/DISCON.dll']

    PerfFile_path = '/projects/ssc/nabbas/TurbineModels/5MW_Baseline/Cp_Ct_Cq.OpenFAST5MW.txt'

else: # Just for local testing... 
    FAST_exe = '/Users/nabbas/openfast/install/bin/openfast_single'
    Turbsim_exe = '/Users/nabbas/openfast/install/bin/turbsim_single'
    cores = 4
    FAST_directory = '/Users/nabbas/Documents/WindEnergyToolbox/ROSCO_toolbox/Test_Cases/5MW_Land_DLL_WTurb'
    FAST_InputFile = '5MW_Land_DLL_WTurb.fst'
    dll_filename = ['/Users/nabbas/Documents/TurbineModels/TurbineControllers/FortranControllers/ROSCO/build/libdiscon.dylib',
                    '/Users/nabbas/Documents/TurbineModels/NREL_5MW/5MW_Baseline/ServoData/DISCON/build/DISCON.dll']
    PerfFile_path = ['../5MW_Baseline/Cp_Ct_Cq.OpenFAST5MW.txt']

# Output filepaths        
if eagle: 
    if floating:
        if rosco: 
            case_name_base = '5MW_OC3Spar_rosco'
            run_dir = '/projects/ssc/nabbas/DLC_Analysis/5MW_OC3Spar/5MW_OC3Spar_rosco/'
        else:
            case_name_base = '5MW_OC3Spar_legacy'
            run_dir = '/projects/ssc/nabbas/DLC_Analysis/5MW_OC3Spar/5MW_OC3Spar_legacy/'
    else:
        if rosco:
            case_name_base = '5MW_Land_rosco'
            run_dir = '/projects/ssc/nabbas/DLC_Analysis/5MW_Land/5MW_Land_rosco/'
        else:
            case_name_base = '5MW_Land_legacy'
            run_dir = '/projects/ssc/nabbas/DLC_Analysis/5MW_Land/5MW_Land_legacy/'
    wind_dir = '/projects/ssc/nabbas/DLC_Analysis/wind/NREL5MW/'
else:
    case_name_base = '5MW_Land_legacy'
    run_dir = '../BatchOutputs/5MW_Land/5MW_Land_legacy/'
    wind_dir = '../BatchOutputs/wind/NREL5MW'

# DLC inputs
DLCs = [1.1, 1.3]
windspeeds = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
seeds = [54321, 12345]

# Analysis time
if floating:
    TMax = 900
else: 
    TMax = 660

# Turbine definition
Turbine_Class = 'I'  # I, II, III, IV
Turbulence_Class = 'A'
D = 126.
z_hub = 90

# ================== THE ACTION ==================
# Initialize iec
iec = CaseGen_IEC()
# Turbine Data
iec.init_cond = {}  # can leave as {} if data not available
iec.init_cond[("ElastoDyn", "RotSpeed")] = {'U': [3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.,
                                                  14., 15., 16., 17., 18., 19., 20., 21., 22., 23.,
                                                  24., 25]}
iec.init_cond[("ElastoDyn", "RotSpeed")]['val'] = [6.972, 7.183, 7.506, 7.942, 8.469, 9.156, 10.296,
                                                   11.431, 11.89, 12.1, 12.1, 12.1, 12.1, 12.1, 12.1,
                                                   12.1, 12.1, 12.1, 12.1, 12.1, 12.1, 12.1, 12.1]
iec.init_cond[("ElastoDyn", "BlPitch1")] = {'U': [3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.,
                                                  14., 15., 16., 17., 18., 19., 20., 21., 22., 23.,
                                                  24., 25]}
iec.init_cond[("ElastoDyn", "BlPitch1")]['val'] = [0., 0., 0., 0., 0., 0., 0., 0., 0., 3.823, 6.602,
                                                   8.668, 10.450, 12.055, 13.536, 14.920, 16.226,
                                                   17.473, 18.699, 19.941, 21.177, 22.347, 23.469]
iec.init_cond[("ElastoDyn", "BlPitch2")] = iec.init_cond[("ElastoDyn", "BlPitch1")]
iec.init_cond[("ElastoDyn", "BlPitch3")] = iec.init_cond[("ElastoDyn", "BlPitch1")]

if floating:
    iec.init_cond[("ElastoDyn", "PtfmSurge")] = {'U': [4., 5., 6., 7., 8., 9., 10., 10.5, 11., 12.,
                                                       14., 19., 24.]}
    iec.init_cond[("ElastoDyn", "PtfmSurge")]['val'] = [3.8758245863838807, 5.57895688031965, 7.619719770801395, 9.974666446553552, 12.675469235464321, 16.173740623041965,
                                                        20.069526574594757, 22.141906121375552, 23.835466098954708, 22.976075549477354, 17.742743260748373, 14.464576583154068, 14.430969814391759]
    iec.init_cond[("ElastoDyn", "PtfmHeave")] = {'U': [4., 5., 6., 7., 8., 9., 10., 10.5, 11., 12.,
                                                       14., 19., 24.]}
    iec.init_cond[("ElastoDyn", "PtfmHeave")]['val'] = [0.030777174904620515, 0.008329930604820483, -0.022973502300090893, -0.06506947653943342, -0.12101317451310406, -0.20589689839069836, -
                                                        0.3169518280533253, -0.3831692055885472, -0.4409624802614755, -0.41411738171337675, -0.2375323506471747, -0.1156867221814119, -0.07029955933167854]
    iec.init_cond[("ElastoDyn", "PtfmPitch")] = {'U': [4., 5., 6., 7., 8., 9., 10., 10.5, 11., 12.,
                                                       14., 19., 24.]}
    iec.init_cond[("ElastoDyn", "PtfmPitch")]['val'] = [0.7519976895165884, 1.104483050851386, 1.5180416334025146, 1.9864587671004394, 2.5152769741130134, 3.1937704945765795,
                                                        3.951314212429935, 4.357929703098016, 4.693765745171944, 4.568760630312074, 3.495057478277534, 2.779958240049992, 2.69008798174216]
    # metocean_U_init  = [4.00, 6.00, 8.00, 10.00, 12.00, 14.00, 16.00, 18.00, 20.00, 22.00, 24.00]
    # metocean_Hs_init = [1.908567568, 1.960162595, 2.062722244, 2.224539415, 2.489931091, 2.802984019, 3.182301485, 3.652236101, 4.182596165, 4.695439504, 5.422289377]
    # metocean_Tp_init = [12.23645701, 12.14497777, 11.90254947, 11.5196666, 11.05403739, 10.65483551, 10.27562225, 10.13693777, 10.27842325, 10.11660396, 10.96177917]

iec.Turbine_Class = Turbine_Class
iec.Turbulence_Class = Turbulence_Class
iec.D = D
iec.z_hub = z_hub


iec.dlc_inputs = {}
iec.dlc_inputs['DLC'] = DLCs
iec.dlc_inputs['U'] = [windspeeds, windspeeds]
iec.dlc_inputs['Seeds'] = [seeds, seeds]
iec.dlc_inputs['Yaw'] = [[], []]

# '+','-','both': sign for transient events in EDC, EWS
iec.transient_dir_change = 'both'
# 'v','h','both': vertical or horizontal shear for EWS
iec.transient_shear_orientation = 'both'

# Analysis time
iec.AnalysisTime = TMax
iec.TF = TMax

# Some specifics for the cases
case_inputs = {}

case_inputs[('Fst', 'OutFileFmt')] = {'vals': [2], 'group': 0}
case_inputs[("Fst", "TMax")] = {'vals': [TMax], 'group': 0}
case_inputs[('AeroDyn15', 'TwrAero')] = {'vals': ['True'], 'group': 0}

case_inputs[('DISCON_in', 'PerfFileName')] = {'vals': [PerfFile_path], 'group': 0}

if floating:
    case_inputs[('DISCON_in', 'PS_Mode')] = {'vals': [1], 'group': 0}
    case_inputs[('DISCON_in', 'Fl_Mode')] = {'vals': [1], 'group': 0}
else:
    case_inputs[('DISCON_in', 'PS_Mode')] = {'vals': [0], 'group': 0}
    case_inputs[('DISCON_in', 'Fl_Mode')] = {'vals': [0], 'group': 0}

case_inputs[('ServoDyn', 'DLL_FileName')] = {'vals': dll_filename, 'group': 2}

# Naming, file management, etc
iec.case_name_base = case_name_base
iec.run_dir = run_dir

iec.wind_dir = wind_dir
iec.debug_level = debug_level
if multi:
    iec.parallel_windfile_gen = True
else:
    iec.parallel_windfile_gen = False

# Run FAST cases
fastBatch = runFAST_pywrapper_batch(FAST_ver='OpenFAST', dev_branch=True)
fastBatch.FAST_runDirectory = iec.run_dir

fastBatch.FAST_directory = FAST_directory
fastBatch.FAST_InputFile = FAST_InputFile
fastBatch.FAST_exe = FAST_exe
iec.Turbsim_exe = Turbsim_exe
iec.cores = cores


# Make sure output flags are on
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
    "LSShftFys", "LSShftFzs", "RotTorq", "LSSTipMys", "LSSTipMzs",
    # ServoDyn
    "GenPwr", "GenTq",
    # AeroDyn15
    "RtArea", "RtVAvgxh", "B1N3Clrnc", "B2N3Clrnc", "B3N3Clrnc",
    "RtAeroCp", 'RtAeroCq', 'RtAeroCt', 'RtTSR',
    # InflowWind
    "Wind1VelX", "Wind1VelY", "Wind1VelZ"
]
channels = {}
for var in var_out:
    channels[var] = True
fastBatch.channels = channels

# Execute
case_list, case_name_list = iec.execute(case_inputs=case_inputs)
fastBatch.case_list = case_list
fastBatch.case_name_list = case_name_list
fastBatch.debug_level = debug_level
if multi:
    fastBatch.run_multi()
else:
    fastBatch.run_serial()


# Save statistics
if save_stats:
    from pCrunch import pdTools, Processing
    fp = Processing.FAST_Processing()

    # Find all outfiles
    outfiles = []
    for file in os.listdir(run_dir):
        if file.endswith('.outb'):
            print(file)
            outfiles.append(os.path.join(run_dir, file))
        elif file.endswith('.out'):
            outfiles.append(os.path.join(run_dir, file))

    outfiles = outfiles

    # Set some processing parameters
    fp.OpenFAST_outfile_list = outfiles
    fp.namebase = case_name_base
    fp.t0 = 30
    fp.parallel_analysis = True
    fp.results_dir = os.path.join(run_dir,'stats')
    fp.verbose = True

    fp.save_LoadRanking = True
    fp.save_SummaryStats = True

    # Load and save statistics and load rankings
    stats, load_rankings = fp.batch_processing()




