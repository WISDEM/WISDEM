import os, sys
import numpy as np
import multiprocessing as mp
from weis.aeroelasticse.runFAST_pywrapper import runFAST_pywrapper_batch
from weis.aeroelasticse.Turbsim_mdao.pyturbsim_wrapper import pyTurbsim_wrapper
from weis.aeroelasticse.CaseGen_General import CaseGen_General, save_case_matrix_direct

from openmdao.core.mpi_wrap import MPI
if MPI:
    from openmdao.api import PetscImpl as impl
    from mpi4py import MPI
    from petsc4py import PETSc
else:
    from openmdao.api import BasicImpl as impl

class runTS_pywrapper_batch(object):

    def __init__(self, filedict, case_list=None, case_name_list = None, overwrite=True):
        self.case_list          = case_list ### user needs to set this if they don't provide it.
        self.case_name_list = case_name_list ### ditto
        self.filedict = filedict

        self.overwrite = overwrite

    def common_init(self):
        pass
#        if not os.path.exists(self.FAST_runDirectory):
#            os.makedirs(self.FAST_runDirectory)

    def run_serial(self):
        # Run batch serially
        self.common_init()

        res = []
        for case_idx in range(len(self.case_list)):
            case = self.case_list[case_idx]
            case_name = self.case_name_list[case_idx]
            dat = [case,self.filedict, case_idx, case_name, self.overwrite]
            r = tseval(dat)
            res.append(r)
        for r in res:
            idx = r[0]
            fname = r[1]
            self.case_list[idx]['tswind_file'] = fname

    def run_multi(self, cores=None):
        # Run cases in parallel, threaded with multiprocessing module

        self.common_init()

        if not cores:
            cores = mp.cpu_count()
        pool = mp.Pool(cores)

        caseNfile = []
        for case_idx in range(len(self.case_list)):
            case = self.case_list[case_idx]
            case_name = self.case_name_list[case_idx]
            caseNfile.append([case,self.filedict, case_idx, case_name, self.overwrite])
        res = pool.map(tseval, caseNfile)
        # map returns a list of all the return vals from each call.
        # now add them to the cases:
        for r in res:
            idx = r[0]
            fname = r[1]
            self.case_list[idx]['tswind_file'] = fname

    def run_mpi(self, mpi_comm_map_down):

        self.common_init()

        comm = MPI.COMM_WORLD
        # size = comm.Get_size()
        rank = comm.Get_rank()
        sub_ranks = mpi_comm_map_down[rank]
        size = len(sub_ranks)

        caseNfile = []
        for case_idx in range(len(self.case_list)):
            case = self.case_list[case_idx]
            case_name = self.case_name_list[case_idx]
            caseNfile.append([case,self.filedict, case_idx, case_name, self.overwrite])

        N_cases = len(caseNfile)
        N_loops = int(np.ceil(float(N_cases)/float(size)))

        U_out = []
        WindFile_out = []
        WindFile_type_out = []
        for i in range(N_loops):
            idx_s    = i*size
            idx_e    = min((i+1)*size, N_cases)

            for j, var_vals in enumerate(caseNfile[idx_s:idx_e]):
                data   = [tseval, var_vals]
                rank_j = sub_ranks[j]
                comm.send(data, dest=rank_j, tag=0)

            for j, var_vals in enumerate(caseNfile[idx_s:idx_e]):
                rank_j = sub_ranks[j]
                data_out = comm.recv(source=rank_j, tag=1)

                idx   = data_out[0]
                fname = data_out[1]
                self.case_list[idx]['tswind_file'] = fname

    # def run_mpi(self, comm=None):
    #     from mpi4py import MPI

    #     self.common_init()

    #     # file management
    #     if not os.path.exists(self.FAST_runDirectory):
    #         os.makedirs(self.FAST_runDirectory)

    #     # mpi comm management
    #     if not comm:
    #         comm = MPI.COMM_WORLD
    #     size = comm.Get_size()
    #     rank = comm.Get_rank()

    #     N_cases = len(self.case_list)
    #     N_loops = int(np.ceil(float(N_cases)/float(size)))

    #     if rank == 0:
    #         caseNfile = []
    #         for case_idx in range(len(self.case_list)):
    #             case = self.case_list[case_idx]
    #             case_name = self.case_name_list[case_idx]
    #             caseNfile.append([case,self.filedict, case_idx, case_name])
    #     else:
    #         caseNfile = []

    #     output = []
    #     for i in range(N_loops):
    #         # if # of cases left to run is less than comm size, split comm
    #         n_resid = N_cases - i*size
    #         if n_resid < size: 
    #             split_comm = True
    #             color = np.zeros(size)
    #             for i in range(n_resid):
    #                 color[i] = 1
    #             color = [int(j) for j in color]
    #             comm_i  = MPI.COMM_WORLD.Split(color_i, 1)
    #         else:
    #             split_comm = False
    #             comm_i = comm

    #         # position in case list
    #         idx_s  = i*size
    #         idx_e  = min((i+1)*size, N_cases)

    #         # scatter out cases
    #         if split_comm:
    #             if color[rank] == 1:
    #                 case_data_i = comm_i.scatter(caseNfile[idx_s:idx_e], root=0)    
    #         else:
    #             case_data_i = comm_i.scatter(caseNfile[idx_s:idx_e], root=0)
            
    #         # eval
    #         out = tseval(case_data_i)

    #         # gather results
    #         if split_comm:
    #             if color[rank] == 1:
    #                 output_i = comm_i.gather(out, root=0)
    #         else:
    #             output_i = comm_i.gather(out, root=0)

    #         if rank == 0:
    #             output.extend(output_i)

    #     if rank == 0:
    #         output_fname = []
    #         for outi in output:
    #             output_fname.append(outi[1])

    #     return output
        
        



def tseval(dat):
    # Batch FAST pyWrapper call, as a function outside the runFAST_pywrapper_batch class for pickle-ablility
    case = dat[0]
    filedict = dat[1]
    case_idx = dat[2]
    case_name = dat[3]
    # print("running ts", case, filedict)
    pyturb = pyTurbsim_wrapper(filedict, case, case_name) # initialize runner with case variable inputs
    try:
        overwrite = dat[4]
        pyturb.overwrite = overwrite
    except:
        pass

#        pyturb.ny = 20 # example of changing an attribute
    pyturb.execute() # run
    #case['tswind_file'] = pyturb.tswind_file  ### need to really return it!
    #case['tswind_dir'] = pyturb.tswind_dir
    #batch_wrapper.set_case_outputs(case_idx, pyturb.tswind_dir, pyturb.tswind_file) ### future reference: does not work!
    return [case_idx, pyturb.tswind_file]

######################### example drivers.... #####

def example_runTSFAST_Batch():
    fastBatch = runFAST_pywrapper_batch(FAST_ver='OpenFAST')


    mac = False
    if mac:
        fastBatch.FAST_InputFile = '5MW_Land_DLL_WTurb-fast.fst'   # FAST input file (ext=.fst)
        fastBatch.FAST_exe = '/Users/pgraf/opt/openfast/openfast/install/bin/openfast'   # Path to executable
        fastBatch.FAST_directory = '/Users/pgraf/work/wese/templates/openfast/5MW_Land_DLL_WTurb-ModifiedForPyturbsim'
    else:
        fastBatch.FAST_InputFile = '5MW_Land_DLL_WTurb-fast.fst'   # FAST input file (ext=.fst)
        fastBatch.FAST_exe = '/home/pgraf/opt/openfast/openfast/install/bin/openfast'   # Path to executable
        fastBatch.FAST_directory = '/home/pgraf/projects/wese/newaero/templates/openfast/5MW_Land_DLL_WTurb-ModifiedForPyturbsim'

    fastBatch.FAST_runDirectory = 'temp/OpenFAST'
    fastBatch.debug_level = 2

    ## Define case list explicitly
    # case_list = [{}, {}]
    # case_list[0]['Fst', 'TMax'] = 4.
    # case_list[1]['Fst', 'TMax'] = 5.
    # case_name_list = ['test01', 'test02']


    tmaxs = [60.0]
    vhubs = [12.0]
    rhos = [0.0]
    seeds = [1,2]

    case_inputs = {}
    case_inputs[("WrFMTFF")] = {'vals':[False], 'group':0}
    case_inputs[("WrBLFF")] = {'vals':[True], 'group':0}
    case_inputs[("WrADFF")] = {'vals':[True], 'group':0}
    case_inputs[("WrADTWR")] = {'vals':[False], 'group':0}   #WrADTWR
    case_inputs[("TMax")] = {'vals':[t + 30 for t in tmaxs], 'group':0}
    #case_inputs[("TMax")] = {'vals':[t for t in tmaxs], 'group':0}
    case_inputs[("Vhub")] = {'vals':vhubs, 'group':1}
    case_inputs[("Rho")] = {'vals':rhos, 'group':1}
    case_inputs[("RandSeed1")] = {'vals':seeds, 'group':2}
    case_inputs[("RandSeed")] = {'vals':seeds, 'group':2}
    ts_case_list, ts_case_name_list = CaseGen_General(case_inputs, dir_matrix='', namebase='pyTurbsim_testing')

    if mac:
        ts_filedict = {
            'ts_dir':"/Users/pgraf/work/wese/templates/turbsim/pyturbsim/",
            'ts_file':"evans_faster.inp",
            'run_dir':"test_ts_run_dir"}
    else:
        ts_filedict = {
            'ts_dir':"/home/pgraf/projects/wese/newaero/templates/turbsim/pyturbsim/",
            'ts_file':"evans_faster.inp",
            'run_dir':"test_ts_run_dir"}
        

    tsBatch = runTS_pywrapper_batch(ts_filedict, ts_case_list, ts_case_name_list)
    #tsBatch.run_serial()
    tsBatch.run_multi(4)
    ### At this point turbsim is done running and the .bts file names have been added to each case in tsBatch.case_list

    ## Generate case list using General Case Generator
    ## Specify several variables that change independently or collectly
    case_inputs = {}
    case_inputs[("Fst","TMax")] = {'vals':tmaxs, 'group':0}
#    case_inputs[("AeroDyn15","TwrPotent")] = {'vals':[0], 'group':0}
#    case_inputs[("AeroDyn15","TwrShadow")] = {'vals':['False'], 'group':0}
#    case_inputs[("AeroDyn15","TwrAero")] = {'vals':['True'], 'group':0}
    case_inputs[("InflowWind","WindType")] = {'vals':[3], 'group':0}   # 1 = steady, 3 = turbsim binary
    case_inputs[("Fst","OutFileFmt")] = {'vals':[1], 'group':0}
    case_inputs[("InflowWind","HWindSpeed")] = {'vals':vhubs, 'group':1}
    case_inputs[("InflowWind","Rho")] = {'vals':rhos, 'group':1}
    case_inputs[("InflowWind","RandSeed1")] = {'vals':seeds, 'group':2}

    # case_inputs[("ElastoDyn","RotSpeed")] = {'vals':[9.156, 10.296, 11.431, 11.89, 12.1], 'group':1}
    # case_inputs[("ElastoDyn","BlPitch1")] = {'vals':[0., 0., 0., 0., 3.823], 'group':1}
    # case_inputs[("ElastoDyn","BlPitch2")] = case_inputs[("ElastoDyn","BlPitch1")]
    # case_inputs[("ElastoDyn","BlPitch3")] = case_inputs[("ElastoDyn","BlPitch1")]
    # case_inputs[("ElastoDyn","GenDOF")] = {'vals':['True','False'], 'group':2}
    case_list, case_name_list = CaseGen_General(case_inputs, dir_matrix=fastBatch.FAST_runDirectory, namebase='testing')
    # manually adding the wind file names from turb sim run.  This seems a little sketchy
    ### Note to Evan: I feel I should be able to use ONE call to CaseGen_General, instead of one for turbsim, and one for FAStTunr1
    ## thoughts?  solution?
    #####
    print("ADDING WIND FILE NAMES")
    for i in range(len(case_list)):
        case_list[i][("InflowWind","Filename")] = tsBatch.case_list[i]['tswind_file']
        case_list[i][("InflowWind","FilenameRoot")] = tsBatch.case_list[i]['tswind_file'].replace(".wnd", "")
#        case_list[i][("InflowWind","InflowFile")] = tsBatch.case_name_list[i]
        # print(case_list[i])

    fastBatch.case_list = case_list
    fastBatch.case_name_list = case_name_list

    #fastBatch.run_serial()
    fastBatch.run_multi(4)
    ## at this point FAST has run, and the output file names have been added to the case list.
    print("ADDED FAST OUTPUT FILE NAMES")
    for i in range(len(fastBatch.case_list)):
        print(fastBatch.case_list[i])
    save_case_matrix_direct(fastBatch.case_list, dir_matrix=os.path.join(os.getcwd(),fastBatch.FAST_runDirectory))

if __name__=="__main__":


    example_runTSFAST_Batch()