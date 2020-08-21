'''
----------- Example_02 --------------
Load a turbine model from saved pickle, make a quick cp plot 
-------------------------------------

In this example:
  - Load a turbine from a saved pickle
  - Plot Cp Surface
'''

# Python modules
import matplotlib.pyplot as plt
# ROSCO toolbox modules 
from ROSCO_toolbox import turbine as ROSCO_turbine


# Initialize a turbine class -- Don't need to instantiate!
turbine = ROSCO_turbine.Turbine

# Load quick from python pickle
turbine = turbine.load('NREL5MW_saved.p')

# plot rotor performance 
print('Plotting Cp data')
turbine.Cp.plot_performance(turbine.Cp_table, turbine.pitch_initial_rad, turbine.TSR_initial)
plt.show()