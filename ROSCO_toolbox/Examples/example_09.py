'''
----------- Example_09 --------------
Run TurbSim to create wind field binary
-------------------------------------

In this example:
  - Leverage the run_openfast functionality to compile a turbsim binary
'''

# Python Modules
import numpy as np
import matplotlib.pyplot as plt 
# ROSCO toolbox modules 
from ROSCO_toolbox import utilities as ROSCO_utilities

# Instantiate fast_IO
fast_io = ROSCO_utilities.FAST_IO()

# Define openfast output filenames
wind_directory = '../Test_Cases/Wind/'
turbsim_infile = '90m_12mps_twr.inp'

fast_io.run_openfast(wind_directory, fastcall='turbsim', fastfile=turbsim_infile, chdir=False)

