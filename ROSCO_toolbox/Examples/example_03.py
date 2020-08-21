'''
----------- Example_03 --------------
Run CCblade, save a rotor performance text file
-------------------------------------

In this example:
- Read .yaml input file
- Load an openfast turbine model
- Run ccblade to get rotor performance properties
- Write a text file with rotor performance properties
'''
# Python modules
import yaml 
# ROSCO toolbox modules 
from ROSCO_toolbox import turbine as ROSCO_turbine
from ROSCO_toolbox import utilities as ROSCO_utilities
# Initialize parameter dictionaries
turbine_params = {}
control_params = {}

# Load yaml file
parameter_filename = '../Tune_Cases/NREL5MW.yaml'
inps = yaml.safe_load(open(parameter_filename))
path_params         = inps['path_params']
turbine_params      = inps['turbine_params']
controller_params   = inps['controller_params']

# Load turbine data from openfast model
turbine = ROSCO_turbine.Turbine(turbine_params)
turbine.load_from_fast(path_params['FAST_InputFile'],path_params['FAST_directory'],dev_branch=True,rot_source=None,txt_filename=None)

# Write rotor performance text file
txt_filename = 'Cp_Ct_Cq.Ex03.txt'
file_processing = ROSCO_utilities.FileProcessing()
file_processing.write_rotor_performance(turbine,txt_filename=txt_filename)
