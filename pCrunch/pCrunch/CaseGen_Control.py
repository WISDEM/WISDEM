import numpy as np
import os, sys, copy, itertools
import yaml
import itertools
# WISDEM 
from wisdem.aeroelasticse.CaseGen_General import CaseGen_General, save_case_matrix, save_case_matrix_yaml
from wisdem.aeroelasticse.pyIECWind import pyIECWind_extreme, pyIECWind_turb
from wisdem.aeroelasticse.Util import FileTools

# ROSCO 
from ROSCO_toolbox import controller as ROSCO_controller
from ROSCO_toolbox import turbine as ROSCO_turbine
from ROSCO_toolbox import utilities as ROSCO_Utilities

class CaseGen_Control():
    '''
    Python class to ease controller parameter studies. A lot of this is very similar to CaseGen_IEC, 
    but a little bit of control flavor is added!.

    Parameters:
    parameter_filename: str
        controller tuning input yaml file
    '''

    def __init__(self, parameter_filename):

        # Load controller parameter file
        inps = yaml.safe_load(open(parameter_filename))
        self.path_params = inps['path_params']
        self.turbine_params = inps['turbine_params']
        self.controller_params = inps['controller_params']

        # Turbulent Wind Defaults (NTM)
        self.seed = np.random.uniform(1, 1e8)
        self.Turbulence_Class = 'B'  # IEC Turbulence Class
        self.IEC_WindType = 'NTM'
        self.z_hub = 90.  # wind turbine hub height (m)
        self.D = 126.  # rotor diameter (m)
        self.PLExp = 0.2
        self.AnalysisTime = 720.
        self.debug_level = 0
        self.overwrite = True

        # Cases
        case_inputs={}

    def gen_turbwind(self, Uref):
        '''
        Generate turbulent wind files. 

        Parameters
        ----------
        Uref: list
            list of wind speeds 
        '''
        # Load pyIECWind_Turb
        turbwind = pyIECWind_turb()
        turbwind.seed                = self.seed
        turbwind.Turbulence_Class    = self.Turbulence_Class
        turbwind.z_hub               = self.z_hub
        turbwind.D                   = self.D
        turbwind.PLExp               = self.PLExp
        turbwind.AnalysisTime        = self.AnalysisTime
        turbwind.debug_level         = self.debug_level
        turbwind.overwrite           = self.overwrite
        # User defined
        turbwind.outdir              = self.wind_dir
        turbwind.case_name           = self.case_name_base
        turbwind.Turbsim_exe         = self.Turbsim_exe
        turbwind.debug_level         = self.debug_level

        for U in Uref:
            wind_file, wind_file_type  = turbwind.execute(self.IEC_WindType, U)

        return wind_file, wind_file_type



    def gen_control_cases(self, input_params, DISCON_params, values, group):
        '''
        Generate control case input dictionary for controller parameters

        Parameters
        ----------
        input_params: list
            List of parameters that exist in the controller tuning yaml file that will be modified
        DISCON_params: list
            List of parameters in DISCON.IN that will be changed by the paremeters in input_params
        values: list
            List with nested lists of floats that correspond to the input_params. 
                - i.e. [[0.1],[0.5]]
        group: int
            Group number for case_inputs

        Returns
        -------
        case_inputs: dict
            Dictionary of case inputs for WISDEM's CaseGen_General
        tuning_inputs: dict
            Dictionary containing yaml input tuning parameters
        '''
        # Load turbine
        turbine = ROSCO_turbine.Turbine(self.turbine_params)
        turbine.load_from_fast(self.path_params['FAST_InputFile'],
                               self.path_params['FAST_directory'], dev_branch=True)
        # if not os.path.exists(self.path_params['rotor_performance_filename']):
        #     R.write_rotor_performance(turbine,txt_filename=path_params['rotor_performance_filename'])

        # Check for flaps
        if 'zeta_flp' in input_params or 'omega_flp' in input_params:
            self.controller_params['Flp_Mode'] = 2
            turbine.load_blade_info()

        # Create tuning value matrix
        tuning_matrix = list(itertools.product(*values))

        # Generate cases 
        case_inputs = {}
        tuning_inputs = {key: [] for key in input_params}
        DISCON_vals_list = {key: [] for key in DISCON_params}
        for tuning_params in tuning_matrix:
            for inp_ind, inp_p in enumerate(input_params):
                # Modify controller parameters
                self.controller_params[inp_p] = tuning_params[inp_ind]
                # tuning_inputs[inp_p].append(tuning_params[inp_ind])

            # Initialize and tune controller
            controller = ROSCO_controller.Controller(self.controller_params)
            controller.tune_controller(turbine)

            # Get DISCON input parameters
            data_processing = ROSCO_Utilities.DataProcessing()
            DISCON_inputs = data_processing.DISCON_dict(turbine, controller)

            # Check for invalid controller inputs
            if (any(item in ['VS_KP', 'VS_KI'] for item in DISCON_params) and
                any(DISCON_inputs[DISCON_p][0] > 0 for DISCON_p in DISCON_params)):
                    print('WARNING: Control gains cannot be greater than zero and are for {} = {}'
                        .format([param for param in input_params],  tuning_params))
            elif (any(item in ['Flp_Kp', 'Flp_Ki'] for item in DISCON_params) and
                any(DISCON_inputs[DISCON_p][0] < 0 for DISCON_p in DISCON_params)):
                    print('WARNING: Control gains cannot be less than zero and are for {} = {}'
                        .format([param for param in input_params],  tuning_params))
            # Save DISCON and tuning inputs
            else:
                for DISCON_p in DISCON_params:
                    DISCON_vals = DISCON_inputs[DISCON_p]
                    DISCON_vals_list[DISCON_p].append(DISCON_vals)
                for inp_ind, inp_p in enumerate(input_params):
                    tuning_inputs[inp_p].append(tuning_params[inp_ind]) 
        # Write case_inputs
        for DISCON_p in DISCON_params:
            case_inputs[('DISCON_in', DISCON_p)] = {'vals': DISCON_vals_list[DISCON_p], 
                                                    'group': group}

        return case_inputs, tuning_inputs

def append_case_matrix_yaml(file_dir, file_name, append_dict, name_base, package=1):
    '''
    Append the case_matrix_yaml with some more dictionary entries
        - probably controller tuning inputs

    Parameters
    ----------
    file_dir: str
        yaml file directory
    file_name: str
        yaml file name
    append_dict: dict
        dictionary to append to yaml file
    name_base: str
        base name of parameter to be appended { X in yaml_data[(X,'param)] }
    '''
    yaml_data = FileTools.load_yaml(os.path.join(file_dir, file_name), package=package)

    for key in append_dict.keys():
        rep_list = []
        for val in append_dict[key]:
            if type(val) in [np.float32, np.float64, np.single, np.double, np.longdouble]:
                val = float(val)
            elif type(val) in [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.intc, np.uintc, np.uint]:
                val = int(val)
            elif type(val) in [np.array, np.ndarray]:
                val = val.tolist()
            elif type(val) in [np.str_]:
                val = str(val)
            rep_list.append(val)
    
        yaml_data[(name_base, key)] = rep_list

    FileTools.save_yaml(file_dir, file_name, yaml_data)
