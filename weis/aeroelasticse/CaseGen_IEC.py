import numpy as np
import os, sys, copy, itertools
import multiprocessing as mp

from weis.aeroelasticse.CaseGen_General import CaseGen_General, save_case_matrix, save_case_matrix_yaml
from weis.aeroelasticse.pyIECWind import pyIECWind_extreme, pyIECWind_turb

try:
    from mpi4py import MPI
except:
    pass
    
# Generate wind files
def gen_windfile(data):
    # function for calling wind file execution
    iecwind = data[0]
    IEC_WindType = data[1]
    change_vars = data[2]
    var_vals = data[3]

    if 'Seeds' in change_vars:
        iecwind.seed = var_vals[change_vars.index('Seeds')]
    U = var_vals[change_vars.index('U')]

    wind_file, wind_file_type = iecwind.execute(IEC_WindType, U)
    if type(wind_file) is str:
        U_out = [U]
        WindFile_out = [wind_file]
        WindFile_type_out = [wind_file_type]
    elif type(wind_file) is list:
        U_out = [U]*len(wind_file)
        WindFile_out = wind_file
        WindFile_type_out = wind_file_type
    return [U_out, WindFile_out, WindFile_type_out]

class CaseGen_IEC():

    def __init__(self):
        
        self.init_cond = {} # Dictionary of steady state operating conditions as a function of wind speed, used for setting inital conditions

        self.Turbine_Class               = 'I' # I, II, III, IV
        self.Turbulence_Class            = 'A'
        self.D                           = 126.
        self.z_hub                       = 90.

        # DLC inputs
        self.dlc_inputs = {}
        self.transient_dir_change        = 'both'  # '+','-','both': sign for transient events in EDC, EWS
        self.transient_shear_orientation = 'both'  # 'v','h','both': vertical or horizontal shear for EWS
        self.PC_MaxRat                   = 5.      # (deg), maximum blade pitch rate, used for emergency stop, DLC 5.1
        self.TMax                        = 0.
        self.TStart                      = 30.

        self.debug_level                 = 2
        self.parallel_windfile_gen       = False
        self.cores                       = 0
        self.overwrite                   = False

        self.mpi_run                     = False
        self.comm_map_down               = []


    def execute(self, case_inputs={}):

        case_list_all = {}
        dlc_all = []

        for i, dlc in enumerate(self.dlc_inputs['DLC']):
            case_inputs_i = copy.deepcopy(case_inputs)

            # DLC specific variable changes
            if dlc in [1.1, 1.2, 5.1]:
                IEC_WindType = 'NTM'
                alpha = 0.2
                iecwind = pyIECWind_turb()
                TMax = 630.

            elif dlc in [1.3, 6.1, 6.3]:
                if self.Turbine_Class == 'I':
                    x = 1
                elif self.Turbine_Class == 'II':
                    x = 2
                elif self.Turbine_Class == 'III':
                    x = 3
                else:
                    exit('Class of the WT is needed for the ETM wind, but it is currently not set to neither 1,2 or 3.')
                IEC_WindType = '%uETM'%x
                alpha = 0.11
                iecwind = pyIECWind_turb()
                TMax = 630.

            elif dlc == 1.4:
                IEC_WindType = 'ECD'
                alpha = 0.2
                iecwind = pyIECWind_extreme()
                TMax = 90.

            elif dlc == 1.5:
                IEC_WindType = 'EWS'
                alpha = 0.2
                iecwind = pyIECWind_extreme()
                TMax = 90.

            # Windfile generation setup
            if self.TMax == 0.:
                iecwind.AnalysisTime = TMax
                iecwind.TF           = TMax
            else:
                iecwind.AnalysisTime = self.TMax
                iecwind.TF           = self.TMax

            iecwind.TStart           = self.TStart
            iecwind.Turbine_Class    = self.Turbine_Class
            iecwind.Turbulence_Class = self.Turbulence_Class
            iecwind.IEC_WindType     = IEC_WindType
            iecwind.dir_change       = self.transient_dir_change
            iecwind.shear_orient     = self.transient_shear_orientation
            iecwind.z_hub            = self.z_hub
            iecwind.D                = self.D
            iecwind.PLExp            = alpha
            
            iecwind.outdir           = self.wind_dir
            iecwind.case_name        = self.case_name_base
            iecwind.Turbsim_exe      = self.Turbsim_exe
            iecwind.debug_level      = self.debug_level
            iecwind.overwrite        = self.overwrite

            # Set DLC specific settings
            iecwind_ex = pyIECWind_extreme()
            iecwind_ex.Turbine_Class    = self.Turbine_Class
            iecwind_ex.Turbulence_Class = self.Turbulence_Class
            iecwind_ex.z_hub            = self.z_hub
            iecwind_ex.setup()
            _, V_e50, V_e1, V_50, V_1   = iecwind_ex.EWM(0.)


            if dlc == 5.1:
                case_inputs_i[("ServoDyn","TPitManS1")]  = {'vals':[self.TStart], 'group':0}
                case_inputs_i[("ServoDyn","TPitManS2")]  = {'vals':[self.TStart], 'group':0}
                case_inputs_i[("ServoDyn","TPitManS3")]  = {'vals':[self.TStart], 'group':0}
                case_inputs_i[("ServoDyn","PitManRat1")] = {'vals':[self.PC_MaxRat], 'group':0}
                case_inputs_i[("ServoDyn","PitManRat2")] = {'vals':[self.PC_MaxRat], 'group':0}
                case_inputs_i[("ServoDyn","PitManRat3")] = {'vals':[self.PC_MaxRat], 'group':0}
                case_inputs_i[("ServoDyn","BlPitchF1")]  = {'vals':[90.], 'group':0}
                case_inputs_i[("ServoDyn","BlPitchF2")]  = {'vals':[90.], 'group':0}
                case_inputs_i[("ServoDyn","BlPitchF3")]  = {'vals':[90.], 'group':0}
                case_inputs_i[("ServoDyn","GenTiStp")]   = {'vals':["True"], 'group':0}
                case_inputs_i[("ServoDyn","TimGenOf")]   = {'vals':[self.TStart], 'group':0}
            else:
                case_inputs_i[("ServoDyn","TPitManS1")]  = {'vals':[9999.9], 'group':0}
                case_inputs_i[("ServoDyn","TPitManS2")]  = {'vals':[9999.9], 'group':0}
                case_inputs_i[("ServoDyn","TPitManS3")]  = {'vals':[9999.9], 'group':0}
                case_inputs_i[("ServoDyn","PitManRat1")] = {'vals':[2.], 'group':0}
                case_inputs_i[("ServoDyn","PitManRat2")] = {'vals':[2.], 'group':0}
                case_inputs_i[("ServoDyn","PitManRat3")] = {'vals':[2.], 'group':0}
                case_inputs_i[("ServoDyn","BlPitchF1")]  = {'vals':[0.], 'group':0}
                case_inputs_i[("ServoDyn","BlPitchF2")]  = {'vals':[0.], 'group':0}
                case_inputs_i[("ServoDyn","BlPitchF3")]  = {'vals':[0.], 'group':0}
                case_inputs_i[("ServoDyn","GenTiStp")]   = {'vals':["False"], 'group':0}
                case_inputs_i[("ServoDyn","TimGenOf")]   = {'vals':[9999.9], 'group':0}

            if dlc == 6.1:
                self.dlc_inputs['U'][i] = [V_50]
                self.dlc_inputs['Yaw'][i] = [-8.,8.]
                case_inputs_i[("ElastoDyn","GenDOF")]   = {'vals':["False"], 'group':0}
                case_inputs_i[("ElastoDyn","YawDOF")]   = {'vals':["False"], 'group':0}
                case_inputs_i[("ElastoDyn","RotSpeed")] = {'vals':[0.], 'group':0}
                case_inputs_i[("ElastoDyn","BlPitch1")] = {'vals':[90.], 'group':0}
                case_inputs_i[("ElastoDyn","BlPitch2")] = {'vals':[90.], 'group':0}
                case_inputs_i[("ElastoDyn","BlPitch3")] = {'vals':[90.], 'group':0}
                case_inputs_i[("ServoDyn","PCMode")]    = {'vals':[0], 'group':0}
                case_inputs_i[("AeroDyn15","AFAeroMod")]= {'vals':[1], 'group':0}
            elif dlc == 6.3:
                self.dlc_inputs['U'][i] = [V_1]
                self.dlc_inputs['Yaw'][i] = [-20.,20.]
                case_inputs_i[("ElastoDyn","GenDOF")]   = {'vals':["False"], 'group':0}
                case_inputs_i[("ElastoDyn","YawDOF")]   = {'vals':["False"], 'group':0}
                case_inputs_i[("ElastoDyn","RotSpeed")] = {'vals':[0.], 'group':0}
                case_inputs_i[("ElastoDyn","BlPitch1")] = {'vals':[90.], 'group':0}
                case_inputs_i[("ElastoDyn","BlPitch2")] = {'vals':[90.], 'group':0}
                case_inputs_i[("ElastoDyn","BlPitch3")] = {'vals':[90.], 'group':0}
                case_inputs_i[("ServoDyn","PCMode")]    = {'vals':[0], 'group':0}
                case_inputs_i[("AeroDyn15","AFAeroMod")]= {'vals':[1], 'group':0}
            else:
                self.dlc_inputs['Yaw'][i] = [0.]
                case_inputs_i[("ServoDyn","PCMode")]    = {'vals':[5], 'group':0}
                case_inputs_i[("AeroDyn15","AFAeroMod")]= {'vals':[2], 'group':0}
                case_inputs_i[("ElastoDyn","GenDOF")]   = {'vals':["True"], 'group':0}
                case_inputs_i[("ElastoDyn","YawDOF")]   = {'vals':["True"], 'group':0}


            # Matrix combining N dlc variables that affect wind file generation
            # Done so a single loop can be used for generating wind files in parallel instead of using nested loops
            var_list = ['U', 'Seeds']
            group_len = []
            change_vars = []
            change_vals = []
            for var in var_list:
                if len(self.dlc_inputs[var][i]) > 0:
                    group_len.append(len(self.dlc_inputs[var][i]))
                    change_vars.append(var)
                    change_vals.append(self.dlc_inputs[var][i])
            group_idx = [range(n) for n in group_len]
            matrix_idx = list(itertools.product(*group_idx))
            matrix_group_idx = [np.where([group_i == group_j for group_j in range(0,len(group_len))])[0].tolist() for group_i in range(0,len(group_len))]
            matrix_out = []
            for idx, row in enumerate(matrix_idx):
                row_out = [None]*len(change_vars)
                for j, val in enumerate(row):
                    for g in matrix_group_idx[j]:
                        row_out[g] = change_vals[g][val]
                matrix_out.append(row_out)
            matrix_out = np.asarray(matrix_out)
            
            if self.parallel_windfile_gen and not self.mpi_run:
                # Parallel wind file generation (threaded with multiprocessing)
                if self.cores != 0:
                    p = mp.Pool(self.cores)
                else:
                    p = mp.Pool()
                data_out = p.map(gen_windfile, [(iecwind, IEC_WindType, change_vars, var_vals) for var_vals in matrix_out])
                U_out = []
                WindFile_out = []
                WindFile_type_out = []
                for case in data_out:
                    U_out.extend(case[0])
                    WindFile_out.extend(case[1])
                    WindFile_type_out.extend(case[2])

            elif self.parallel_windfile_gen and self.mpi_run:
                # Parallel wind file generation with MPI
                comm = MPI.COMM_WORLD
                # size = comm.Get_size()
                rank = comm.Get_rank()
                sub_ranks = self.comm_map_down[rank]
                size = len(sub_ranks)

                N_cases = len(matrix_out)
                N_loops = int(np.ceil(float(N_cases)/float(size)))

                U_out = []
                WindFile_out = []
                WindFile_type_out = []
                for j in range(N_loops):
                    idx_s    = j*size
                    idx_e    = min((j+1)*size, N_cases)

                    for k, var_vals in enumerate(matrix_out[idx_s:idx_e]):
                        data   = [gen_windfile, [iecwind, IEC_WindType, change_vars, var_vals]]
                        rank_k = sub_ranks[k]
                        comm.send(data, dest=rank_k, tag=0)

                    for k, var_vals in enumerate(matrix_out[idx_s:idx_e]):
                        rank_k = sub_ranks[k]
                        data_out = comm.recv(source=rank_k, tag=1)
                        U_out.extend(data_out[0])
                        WindFile_out.extend(data_out[1])
                        WindFile_type_out.extend(data_out[2])

            else:
                # Serial
                U_out = []
                WindFile_out = []
                WindFile_type_out = []
                for var_vals in matrix_out:
                    [U_out_i, WindFile_out_i, WindFile_type_out_i] = gen_windfile([iecwind, IEC_WindType, change_vars, var_vals])
                    U_out.extend(U_out_i)
                    WindFile_out.extend(WindFile_out_i)
                    WindFile_type_out.extend(WindFile_type_out_i)
            
            # Set FAST variables from DLC setup
            if ("Fst","TMax") not in case_inputs_i:
                case_inputs_i[("Fst","TMax")] = {'vals':[TMax], 'group':0}
            case_inputs_i[("InflowWind","WindType")] = {'vals':WindFile_type_out, 'group':1}
            case_inputs_i[("InflowWind","Filename")] = {'vals':WindFile_out, 'group':1}

            if len(self.dlc_inputs['Yaw'][i]) > 0:
                    case_inputs_i[("ElastoDyn","NacYaw")] = {'vals':self.dlc_inputs['Yaw'][i], 'group':2}

            # Set FAST variables from inital conditions
            if self.init_cond:
                for var in self.init_cond.keys():
                    if var not in case_inputs_i.keys():
                        inital_cond_i = [np.interp(U, self.init_cond[var]['U'], self.init_cond[var]['val']) for U in U_out]
                        case_inputs_i[var] = {'vals':inital_cond_i, 'group':1}
            
            # Append current DLC to full list of cases
            case_list, case_name = CaseGen_General(case_inputs_i, self.run_dir, self.case_name_base, save_matrix=False)
            case_list_all = self.join_case_dicts(case_list_all, case_list)
            case_names_all = [self.case_name_base +'_'+ ('%d'%i).zfill(len('%d'%(len(case_list_all)-1))) for i in range(len(case_list_all))]
            dlc_all.extend([dlc]*len(case_list))

        # Save case matrix file
        self.save_joined_case_matrix(case_list_all, dlc_all, case_names_all)

        return case_list_all, case_names_all, dlc_all


    def join_case_dicts(self, caselist, caselist_add):
        if caselist:
            keys1 = caselist[0].keys()
            keys2 = caselist_add[0].keys()
            n1 = len(caselist)
            n2 = len(caselist_add)

            common = list(set(keys1) & set(keys2))
            missing1 = list(set(keys1).difference(keys2))
            missing2 = list(set(keys2).difference(keys1))

            # caselist_out = copy.copy(case_list)
            for i in range(n1):
                for var in missing2:
                    caselist[i][var] = np.nan
            for i in range(n2):
                for var in missing1:
                    caselist_add[i][var] = np.nan

            return caselist + caselist_add
        else:
            return caselist_add

    def save_joined_case_matrix(self, caselist, dlc_list, case_names_all):

        change_vars = sorted(caselist[0].keys())

        matrix_out = []
        for case in caselist:
            row_out = [None]*len(change_vars)
            for i, var in enumerate(change_vars):
                # row_out[i] = str(case[var])
                row_out[i] = case[var]
            matrix_out.append(row_out)

        change_vars = [('IEC', 'DLC')] + change_vars
        [matrix_row.insert(0, dlc) for dlc, matrix_row in zip(dlc_list,matrix_out)]

        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)

        save_case_matrix(matrix_out, change_vars, self.run_dir)
        save_case_matrix_yaml(matrix_out, change_vars, self.run_dir, case_names_all)



if __name__=="__main__":
    
    iec = CaseGen_IEC()

    # Turbine Data
    iec.init_cond = {} # can leave as {} if data not available
    iec.init_cond[("ElastoDyn","RotSpeed")] = {'U':[3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25]}
    iec.init_cond[("ElastoDyn","RotSpeed")]['val'] = [6.972, 7.183, 7.506, 7.942, 8.469, 9.156, 10.296, 11.431, 11.89, 12.1, 12.1, 12.1, 12.1, 12.1, 12.1, 12.1, 12.1, 12.1, 12.1, 12.1, 12.1, 12.1, 12.1]
    iec.init_cond[("ElastoDyn","BlPitch1")] = {'U':[3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25]}
    iec.init_cond[("ElastoDyn","BlPitch1")]['val'] = [0., 0., 0., 0., 0., 0., 0., 0., 0., 3.823, 6.602, 8.668, 10.450, 12.055, 13.536, 14.920, 16.226, 17.473, 18.699, 19.941, 21.177, 22.347, 23.469]
    iec.init_cond[("ElastoDyn","BlPitch2")] = iec.init_cond[("ElastoDyn","BlPitch1")]
    iec.init_cond[("ElastoDyn","BlPitch3")] = iec.init_cond[("ElastoDyn","BlPitch1")]

    iec.Turbine_Class = 'I' # I, II, III, IV
    iec.Turbulence_Class = 'A'
    iec.D = 126.
    iec.z_hub = 90.

    # DLC inputs
    iec.dlc_inputs = {}
    iec.dlc_inputs['DLC']   = [1.1, 1.5]
    iec.dlc_inputs['U']     = [[8, 9, 10], [8]]
    iec.dlc_inputs['Seeds'] = [[5, 6, 7], []]
    iec.dlc_inputs['Yaw']   = [[], []]

    iec.transient_dir_change        = 'both'  # '+','-','both': sign for transient events in EDC, EWS
    iec.transient_shear_orientation = 'both'  # 'v','h','both': vertical or horizontal shear for EWS

    # Naming, file management, etc
    iec.wind_dir = 'temp/wind'
    iec.case_name_base = 'testing'
    iec.Turbsim_exe = 'C:/Users/egaertne/WT_Codes/Turbsim_v2.00.07/bin/TurbSim_x64.exe'
    iec.debug_level = 1
    iec.run_dir = 'temp'

    iec.parallel_windfile_gen = True
    iec.cores = 4

    # Run
    iec.execute()

    
