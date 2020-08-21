# Controller Tuning Script for NREL-5MW Wind Turbine
#  -- Made to run the tools distributed as a part of the ROSCO_Toolbox

#-------------------------------- LOAD INPUT PARAMETERS ---------------------------------#
# Change this for your turbine
parameter_filename = 'NREL5MW.yaml'                         # Name of .yaml input file for the specific turbine




#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#--------------------- NOTHING SHOULD NEED TO CHANGE AFTER THIS -----------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#


#------------------------------------- INITIALIZATION ----------------------------------#
# Import python modules
import matplotlib.pyplot as plt 
import yaml 
import os
# Import ROSCO_toolbox modules 
from ROSCO_toolbox import controller as ROSCO_controller
from ROSCO_toolbox import turbine as ROSCO_turbine
from ROSCO_toolbox import utilities as ROSCO_utilities
# Initialize parameter dictionaries
turbine_params = {}
control_params = {}

# Load input file contents, put them in some dictionaries to keep things cleaner
inps = yaml.safe_load(open(parameter_filename))
path_params = inps['path_params']
turbine_params = inps['turbine_params']
controller_params = inps['controller_params']

#---------------------------------- DO THE FUN STUFF ------------------------------------#
# Initialiize turbine and controller
turbine         = ROSCO_turbine.Turbine(turbine_params)
file_processing = ROSCO_utilities.FileProcessing()

# Load Turbine, write rotor performance file if it doesn't exist
if os.path.exists(path_params['rotor_performance_filename']):
    turbine.load_from_fast(path_params['FAST_InputFile'],path_params['FAST_directory'],dev_branch=True,rot_source='txt',txt_filename=path_params['rotor_performance_filename'])
else:
    turbine.load_from_fast(path_params['FAST_InputFile'],path_params['FAST_directory'],dev_branch=True,rot_source=None, txt_filename=path_params['rotor_performance_filename'])
    file_processing.write_rotor_performance(turbine,txt_filename=path_params['rotor_performance_filename'])
    
# Flap tuning if necessary
if controller_params['Flp_Mode']:
    turbine.load_blade_info()

# Instantiate controller tuning and tune controller
controller      = ROSCO_controller.Controller(controller_params)
controller.tune_controller(turbine)

# Write parameter input file
param_file = 'DISCON.IN'   
file_processing.write_DISCON(turbine,controller,param_file=param_file, txt_filename=path_params['rotor_performance_filename'])

# Plot rotor performance 
turbine.Cp.plot_performance(turbine.Cp_table, turbine.pitch_initial_rad, turbine.TSR_initial)
plt.show()
