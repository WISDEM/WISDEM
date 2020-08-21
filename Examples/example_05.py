'''
----------- Example_05 --------------
Run and plot a simple simple step wind simulation
-------------------------------------

In this example:
  - Load turbine from saved pickle
  - Tune a controller
  - Run and plot a simple step wind simulation

Notes - You will need to have a compiled controller in ROSCO, and 
        properly point to it in the `lib_name` variable.
      - The complex nature of the wind speed estimators implemented in ROSCO
        make using them for simulations is known to cause problems for 
        the simple simulator. We suggesting using WE_Mode = 0 in DISCON.IN.
'''
# Python modules
import matplotlib.pyplot as plt 
import numpy as np
import yaml 
# ROSCO toolbox modules 
from ROSCO_toolbox import controller as ROSCO_controller
from ROSCO_toolbox import turbine as ROSCO_turbine
from ROSCO_toolbox import sim as ROSCO_sim
from ROSCO_toolbox import utilities as ROSCO_utilities
from ROSCO_toolbox import control_interface as ROSCO_ci

# Specify controller dynamic library path and name
lib_name = '../ROSCO/build/libdiscon.dylib'
param_filename = 'DISCON.IN'

# Load turbine model from saved pickle
turbine = ROSCO_turbine.Turbine
turbine = turbine.load('NREL5MW_saved.p')

# Load controller library
controller_int = ROSCO_ci.ControllerInterface(lib_name,param_filename=param_filename)

# Load the simulator
sim = ROSCO_sim.Sim(turbine,controller_int)

# Define a wind speed history
dt = 0.1
tlen = 1000      # length of time to simulate (s)
ws0 = 7         # initial wind speed (m/s)
t= np.arange(0,tlen,dt) 
ws = np.ones_like(t) * ws0
# add steps at every 100s
for i in range(len(t)):
    ws[i] = ws[i] + t[i]//100

# Run simulator and plot results
sim.sim_ws_series(t,ws,rotor_rpm_init=4)
plt.show()

