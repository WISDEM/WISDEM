'''
----------- Example_01 --------------
Load and save a turbine model
-------------------------------------
In this example:
- Read .yaml input file
- Load an openfast turbine model
- Read text file with rotor performance properties
- Print some basic turbine properties
- Save the turbine as a picklle

Note: Uses the NREL 5MW included in the Test Cases and is a part of the OpenFAST distribution
'''

# Python Modules
import yaml
# ROSCO Modules
from ROSCO_toolbox import turbine as ROSCO_turbine

# Load yaml file
parameter_filename = 'NREL5MW_example.yaml'
inps = yaml.safe_load(open(parameter_filename))
path_params         = inps['path_params']
turbine_params      = inps['turbine_params']

# Load turbine data from openfast model
turbine = ROSCO_turbine.Turbine(turbine_params)
turbine.load_from_fast(path_params['FAST_InputFile'],path_params['FAST_directory'],dev_branch=True,rot_source='txt',txt_filename=path_params['rotor_performance_filename'])

# Print some basic turbine info
print(turbine)

# Save the turbine model
turbine.save('NREL5MW_saved.p')