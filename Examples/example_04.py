'''
----------- Example_04 --------------
Load a turbine model and tune the controller
-------------------------------------

In this example:
  - Read a .yaml file
  - Load a turbine model from OpenFAST
  - Tune a controller
  - Write a controller input file
  - Plot gain schedule
'''
# Python modules
import matplotlib.pyplot as plt 
import yaml 
# ROSCO toolbox modules 
from ROSCO_toolbox import controller as ROSCO_controller
from ROSCO_toolbox import turbine as ROSCO_turbine
from ROSCO_toolbox import sim as ROSCO_sim
from ROSCO_toolbox import utilities as ROSCO_utilities

# Load yaml file 
parameter_filename = 'NREL5MW_example.yaml'
inps = yaml.safe_load(open(parameter_filename))
path_params         = inps['path_params']
turbine_params      = inps['turbine_params']
controller_params   = inps['controller_params']

# Instantiate turbine, controller, and file processing classes
turbine         = ROSCO_turbine.Turbine(turbine_params)
controller      = ROSCO_controller.Controller(controller_params)
file_processing = ROSCO_utilities.FileProcessing()

# Load turbine data from OpenFAST and rotor performance text file
turbine.load_from_fast(path_params['FAST_InputFile'],path_params['FAST_directory'],dev_branch=True,rot_source='txt',txt_filename=path_params['rotor_performance_filename'])

# Tune controller 
controller.tune_controller(turbine)

# Write parameter input file
param_file = 'DISCON.IN'   
file_processing.write_DISCON(turbine,controller,param_file=param_file, txt_filename=path_params['rotor_performance_filename'])

# Plot gain schedule
plt.figure(0)
plt.plot(controller.v[len(controller.vs_gain_schedule.Kp):], controller.pc_gain_schedule.Kp)
plt.xlabel('Wind Speed')
plt.ylabel('Proportional Gain')

plt.figure(1)
plt.plot(controller.v[len(controller.vs_gain_schedule.Ki):], controller.pc_gain_schedule.Ki)
plt.xlabel('Wind Speed')
plt.ylabel('Integral Gain')

plt.show()