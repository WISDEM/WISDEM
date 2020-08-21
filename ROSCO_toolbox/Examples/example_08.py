'''
----------- Example_08 --------------
Plot some OpenFAST output data
-------------------------------------

In this example:
  - Load openfast output data
  - Trim the time series
  - Plot some available channels

Note: need to run openfast model in '../Test_Cases/5MW_Land_DLL_WTurb/' to plot
'''

# Python Modules
import numpy as np
import matplotlib.pyplot as plt 
# ROSCO toolbox modules 
from ROSCO_toolbox import utilities as ROSCO_utilities

# Instantiate fast_IO
fast_io = ROSCO_utilities.FAST_IO()

# Define openfast output filenames
# filenames = ["../Test_Cases/5MW_Land/5MW_Land.outb"]

# ---- Note: Could plot multiple cases, textfiles, and binaries...
filenames = ["../Test_Cases/5MW_Land_DLL_WTurb/5MW_Land_DLL_WTurb.out",
            "../Test_Cases/5MW_Land_DLL_WTurb/5MW_Land_DLL_WTurb.outb"]

# Load output info and data
allinfo, alldata = fast_io.load_output(filenames)

# Trim time series
for i,(info,data) in enumerate(zip(allinfo,alldata)):
    alldata[i] = fast_io.trim_output(info, data, tmin=0, tmax=50)

#  Define Plot cases 
#  --- Comment,uncomment, create, and change these as desired...
cases = {}
cases['Baseline'] = ['Wind1VelX', 'BldPitch1', 'GenTq', 'RotSpeed']
cases['Rotor'] = ['BldPitch1', 'GenTq', 'GenPwr']
cases['Rotor Performance'] = ['RtVAvgxh', 'RtTSR', 'RtAeroCp']

# Plot, woohoo!
fast_io.plot_fast_out(cases, allinfo, alldata)
