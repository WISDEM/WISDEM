from weis.aeroelasticse.runFAST_pywrapper   import runFAST_pywrapper, runFAST_pywrapper_batch
from weis.aeroelasticse.CaseGen_IEC         import CaseGen_IEC
from wisdem.commonse.mpi_tools              import MPI
import sys, os, platform
import numpy as np

# Turbine inputs
iec = CaseGen_IEC()
iec.Turbine_Class       = 'I'   # Wind class I, II, III, IV
iec.Turbulence_Class    = 'B'   # Turbulence class 'A', 'B', or 'C'
iec.D                   = 240.  # Rotor diameter to size the wind grid
iec.z_hub               = 150.  # Hub height to size the wind grid
cut_in                  = 3.    # Cut in wind speed
cut_out                 = 25.   # Cut out wind speed
n_ws                    = 3    # Number of wind speed bins
TMax                    = 1.    # Length of wind grids and OpenFAST simulations, suggested 720 s
Vrated                  = 10.59 # Rated wind speed
Ttrans                  = max([0., TMax - 60.])  # Start of the transient for DLC with a transient, e.g. DLC 1.4
TStart                  = max([0., TMax - 600.]) # Start of the recording of the channels of OpenFAST

# Initial conditions to start the OpenFAST runs
u_ref     = np.arange(3.,26.) # Wind speed
pitch_ref = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5058525323662666, 5.253759185225932, 7.50413344606208, 9.310153958810268, 10.8972969450052, 12.412247669440042, 13.883219268525659, 15.252012626933068, 16.53735488246438, 17.76456777500061, 18.953261878035104, 20.11055307762722, 21.238680277668898, 22.30705111326602, 23.455462501156205] # Pitch values in deg
omega_ref = [2.019140272160114, 2.8047214918577925, 3.594541645994511, 4.359025795823625, 5.1123509774611025, 5.855691196288371, 6.589281196735111, 7.312788026081227, 7.514186181824161, 7.54665511646938, 7.573823812448151, 7.600476033113538, 7.630243938880304, 7.638301051122195, 7.622050377183605, 7.612285710588359, 7.60743945212863, 7.605865650155881, 7.605792924227456, 7.6062185247519825, 7.607153933765292, 7.613179734210654, 7.606737845170748] # Rotor speeds in rpm
iec.init_cond = {}
iec.init_cond[("ElastoDyn","RotSpeed")]        = {'U':u_ref}
iec.init_cond[("ElastoDyn","RotSpeed")]['val'] = omega_ref
iec.init_cond[("ElastoDyn","BlPitch1")]        = {'U':u_ref}
iec.init_cond[("ElastoDyn","BlPitch1")]['val'] = pitch_ref
iec.init_cond[("ElastoDyn","BlPitch2")]        = iec.init_cond[("ElastoDyn","BlPitch1")]
iec.init_cond[("ElastoDyn","BlPitch3")]        = iec.init_cond[("ElastoDyn","BlPitch1")]
iec.init_cond[("HydroDyn","WaveHs")]           = {'U':[3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 25, 40, 50]}
iec.init_cond[("HydroDyn","WaveHs")]['val']    = [1.101917033, 1.101917033, 1.179052649, 1.315715154, 1.536867124, 1.835816514, 2.187994638, 2.598127096, 3.061304068, 3.617035443, 4.027470219, 4.51580671, 4.51580671, 6.98, 10.7]
iec.init_cond[("HydroDyn","WaveTp")]           = {'U':[3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 25, 40, 50]}
iec.init_cond[("HydroDyn","WaveTp")]['val']    = [8.515382435, 8.515382435, 8.310063688, 8.006300889, 7.6514231, 7.440581338, 7.460834063, 7.643300307, 8.046899942, 8.521314105, 8.987021024, 9.451641026, 9.451641026, 11.7, 14.2]
iec.init_cond[("HydroDyn","PtfmSurge")]        = {'U':[3., 15., 25.]}
iec.init_cond[("HydroDyn","PtfmSurge")]['val'] = [4., 15., 10.]
iec.init_cond[("HydroDyn","PtfmPitch")]        = {'U':[3., 15., 25.]}
iec.init_cond[("HydroDyn","PtfmPitch")]['val'] = [-1., 3., 1.3]
iec.init_cond[("HydroDyn","PtfmHeave")]        = {'U':[3., 25.]}
iec.init_cond[("HydroDyn","PtfmHeave")]['val'] = [0.5,0.5]

# DLC inputs
wind_speeds = range(int(cut_in), int(cut_out), int(n_ws))
iec.dlc_inputs = {}
iec.dlc_inputs['DLC']   = [1.1, 1.3, 1.4, 1.5, 5.1, 6.1, 6.3]
iec.dlc_inputs['U']     = [wind_speeds, wind_speeds,[Vrated - 2., Vrated, Vrated + 2.],wind_speeds, [Vrated - 2., Vrated, Vrated + 2., cut_out], [], []]
iec.dlc_inputs['Seeds'] = [[1],[1],[],[],[1],[1],[1]]
# iec.dlc_inputs['Seeds'] = [range(1,7), range(1,7),[],[], range(1,7), range(1,7), range(1,7)]
iec.dlc_inputs['Yaw']   = [[], [], [], [], [], [], []]
iec.PC_MaxRat           = 2.

iec.TStart              = Ttrans
iec.TMax                = TMax    # wind file length
iec.transient_dir_change        = 'both'  # '+','-','both': sign for transient events in EDC, EWS
iec.transient_shear_orientation = 'both'  # 'v','h','both': vertical or horizontal shear for EWS

# Management of parallelization
if MPI:
    from wisdem.commonse.mpi_tools import map_comm_heirarchical, subprocessor_loop, subprocessor_stop
    n_OF_runs = 0
    for i in range(len(iec.dlc_inputs['DLC'])):
        # Number of wind speeds
        if iec.dlc_inputs['DLC'][i] == 1.4: # assuming 1.4 is run at [V_rated-2, V_rated, V_rated] and +/- direction change
            if iec.dlc_inputs['U'][i] == []:
                n_U = 6
            else:
                n_U = len(iec.dlc_inputs['U'][i]) * 2
        elif iec.dlc_inputs['DLC'][i] == 5.1: # assuming 5.1 is run at [V_rated-2, V_rated, V_rated]
            if iec.dlc_inputs['U'][i] == []:
                n_U = 3
            else:
                n_U = len(iec.dlc_inputs['U'][i])
        elif iec.dlc_inputs['DLC'][i] in [6.1, 6.3]: # assuming V_50 for [-8, 8] deg yaw error
            if iec.dlc_inputs['U'][i] == []:
                n_U = 2
            else:
                n_U = len(iec.dlc_inputs['U'][i])
        else:
            n_U = len(iec.dlc_inputs['U'][i])
        # Number of seeds
        if iec.dlc_inputs['DLC'][i] == 1.4: # not turbulent
            n_Seeds = 1
        else:
            n_Seeds = len(iec.dlc_inputs['Seeds'][i])
        n_OF_runs += n_U*n_Seeds
        available_cores = MPI.COMM_WORLD.Get_size()
        n_parallel_OFruns = np.min([available_cores - 1, n_OF_runs])
        comm_map_down, comm_map_up, color_map = map_comm_heirarchical(1, n_parallel_OFruns)
        sys.stdout.flush()

# Naming, file management, etc
iec.wind_dir        = 'outputs/wind'
iec.case_name_base  = 'iea15mw'
run_dir1            = os.path.dirname( os.path.dirname( os.path.dirname( os.path.realpath(__file__) ) ) ) + os.sep
iec.Turbsim_exe     = os.path.join(run_dir1, 'local/bin/turbsim')
if MPI:
    iec.cores = available_cores
else:
    iec.cores = 1

iec.debug_level = 2
if MPI:
    iec.parallel_windfile_gen = True
    iec.mpi_run               = True
    iec.comm_map_down         = comm_map_down
else:
    iec.parallel_windfile_gen = False
    iec.mpi_run               = False
iec.run_dir = 'outputs/iea15mw'

# Run case generator / wind file writing
case_inputs = {}
case_inputs[("Fst","TMax")]              = {'vals':[TMax], 'group':0}
case_inputs[("Fst","TStart")]            = {'vals':[TStart], 'group':0}
case_inputs[("Fst","DT")]                = {'vals':[0.005], 'group':0}
case_inputs[("Fst","DT_Out")]            = {'vals':[0.01], 'group':0}  #0.005  
case_inputs[("Fst","OutFileFmt")]        = {'vals':[2], 'group':0}
case_inputs[("Fst","CompHydro")]         = {'vals':[1], 'group':0}
case_inputs[("Fst","CompSub")]           = {'vals':[0], 'group':0}
case_inputs[("InflowWind","WindType")]   = {'vals':[1], 'group':0}
case_inputs[("ElastoDyn","TwFADOF1")]    = {'vals':["True"], 'group':0}
case_inputs[("ElastoDyn","TwFADOF2")]    = {'vals':["True"], 'group':0}
case_inputs[("ElastoDyn","TwSSDOF1")]    = {'vals':["True"], 'group':0}
case_inputs[("ElastoDyn","TwSSDOF2")]    = {'vals':["True"], 'group':0}
case_inputs[("ElastoDyn","FlapDOF1")]    = {'vals':["True"], 'group':0}
case_inputs[("ElastoDyn","FlapDOF2")]    = {'vals':["True"], 'group':0}
case_inputs[("ElastoDyn","EdgeDOF")]     = {'vals':["True"], 'group':0}
case_inputs[("ElastoDyn","DrTrDOF")]     = {'vals':["False"], 'group':0}
case_inputs[("ElastoDyn","GenDOF")]      = {'vals':["True"], 'group':0}
case_inputs[("ElastoDyn","YawDOF")]      = {'vals':["False"], 'group':0}
case_inputs[("ElastoDyn","PtfmSgDOF")]   = {'vals':["False"], 'group':0}
case_inputs[("ElastoDyn","PtfmSwDOF")]   = {'vals':["False"], 'group':0}
case_inputs[("ElastoDyn","PtfmHvDOF")]   = {'vals':["False"], 'group':0}
case_inputs[("ElastoDyn","PtfmRDOF")]    = {'vals':["False"], 'group':0}
case_inputs[("ElastoDyn","PtfmPDOF")]    = {'vals':["False"], 'group':0}
case_inputs[("ElastoDyn","PtfmYDOF")]    = {'vals':["False"], 'group':0}
case_inputs[("ServoDyn","PCMode")]       = {'vals':[5], 'group':0}
case_inputs[("ServoDyn","VSContrl")]     = {'vals':[5], 'group':0}
if platform.system() == 'Windows':
    path2dll = os.path.join(run_dir1, 'local/lib/libdiscon.dll')
elif platform.system() == 'Darwin':
    path2dll = os.path.join(run_dir1, 'local/lib/libdiscon.dylib')
else:
    path2dll = os.path.join(run_dir1, 'local/lib/libdiscon.so')

case_inputs[("ServoDyn","DLL_FileName")] = {'vals':[path2dll], 'group':0}
case_inputs[("AeroDyn15","TwrAero")]     = {'vals':["True"], 'group':0}
case_inputs[("AeroDyn15","TwrPotent")]   = {'vals':[1], 'group':0}
case_inputs[("AeroDyn15","TwrShadow")]   = {'vals':["True"], 'group':0}
case_inputs[("Fst","CompHydro")]         = {'vals':[1], 'group':0}
case_inputs[("HydroDyn","WaveMod")]      = {'vals':[2], 'group':0}
case_inputs[("HydroDyn","WvDiffQTF")]    = {'vals':["False"], 'group':0}
channels = {}
for var in ["TipDxc1", "TipDyc1", "TipDzc1", "TipDxb1", "TipDyb1", "TipDxc2", "TipDyc2", "TipDzc2", "TipDxb2", "TipDyb2", "TipDxc3", "TipDyc3", "TipDzc3", "TipDxb3", "TipDyb3", "RootMxc1", "RootMyc1", "RootMzc1", "RootMxb1", "RootMyb1", "RootMxc2", "RootMyc2", "RootMzc2", "RootMxb2", "RootMyb2", "RootMxc3", "RootMyc3", "RootMzc3", "RootMxb3", "RootMyb3", "TwrBsMxt", "TwrBsMyt", "TwrBsMzt", "GenPwr", "GenTq", "RotThrust", "RtAeroCp", "RtAeroCt", "RotSpeed", "BldPitch1", "TTDspSS", "TTDspFA", "NacYaw", "Wind1VelX", "Wind1VelY", "Wind1VelZ", "LSSTipMxa","LSSTipMya","LSSTipMza","LSSTipMxs","LSSTipMys","LSSTipMzs","LSShftFys","LSShftFzs", "TipRDxr", "TipRDyr", "TipRDzr"]:
    channels[var] = True


# Parallel file generation with MPI
if MPI:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
else:
    rank = 0
if rank == 0:
    case_list, case_name_list, dlc_list = iec.execute(case_inputs=case_inputs)

    #for var in var_out+[var_x]:

    # Run FAST cases
    fastBatch                   = runFAST_pywrapper_batch(FAST_ver='OpenFAST',dev_branch = True)
    fastBatch.FAST_exe          = os.path.join(run_dir1, 'local/bin/openfast')   # Path to executable
    
    # Monopile
    fastBatch.FAST_InputFile    = 'IEA-15-240-RWT-Monopile.fst'   # FAST input file (ext=.fst)
    run_dir2                    = os.path.dirname( os.path.dirname( os.path.realpath(__file__) ) ) + os.sep
    fastBatch.FAST_directory    = os.path.join(run_dir2, 'IEA-15-240-RWT/IEA-15-240-RWT-Monopile')   # Path to fst directory files
    fastBatch.channels          = channels
    fastBatch.FAST_runDirectory = iec.run_dir
    fastBatch.case_list         = case_list
    fastBatch.case_name_list    = case_name_list
    fastBatch.debug_level       = 2

    if MPI:
        fastBatch.run_mpi(comm_map_down)
    else:
        fastBatch.run_serial()

    # U-Maine semi-sub
    fastBatch.FAST_InputFile    = 'IEA-15-240-RWT-UMaineSemi.fst'   # FAST input file (ext=.fst)
    run_dir2                    = os.path.dirname( os.path.dirname( os.path.realpath(__file__) ) ) + os.sep
    fastBatch.FAST_directory    = os.path.join(run_dir2, 'IEA-15-240-RWT/IEA-15-240-RWT-UMaineSemi')   # Path to fst directory files

    if MPI:
        fastBatch.run_mpi(comm_map_down)
    else:
        fastBatch.run_serial()

if MPI:
    sys.stdout.flush()
    if rank in comm_map_up.keys():
        subprocessor_loop(comm_map_up)
    sys.stdout.flush()

# Close signal to subprocessors
if rank == 0 and MPI:
    subprocessor_stop(comm_map_down)
sys.stdout.flush()
    
    
    
    
    
