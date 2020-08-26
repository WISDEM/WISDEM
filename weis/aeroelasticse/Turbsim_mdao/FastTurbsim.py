"""
Demonstration of setting up an OpenMDAO 1.x problem using the FST8Workflow component
(in FST8_aeroelasticsolver), which executes the FST8 reader, writer, and wrapper and assigns
all variables in the FAST outlist to OpenMDAO 'Unknowns'. It also
implements an "input config" function which allows the user to put all variables that they
wish to explicitly define into a dictionary. The input config function assigns these
variables to the correct locations in the variable tree.
"""
# Hacky way of doing relative imports
import numpy as np
import os, sys
sys.path.insert(0, os.path.abspath(".."))
import matplotlib.pyplot as plt
from openmdao.api import Group, Problem, Component, IndepVarComp, ParallelGroup
from openmdao.api import SqliteRecorder
from weis.aeroelasticse.FAST_mdao.FST8_aeroelasticsolver import FST8Workflow
from weis.aeroelasticse.Turbsim_mdao.turbsim_openmdao import turbsimGroup

# Initial OpenMDAO problem setup
top = Problem()
root = top.root = Group()

# Setup input config--file/directory locations, executable, types
caseid = "omdaoCase1.fst"
config = {}
config['fst_masterfile'] = 'Test01.fst' 
config['fst_masterdir']= '../FAST_mdao/wrapper_examples/FST8inputfiles'
config['fst_runfile'] = caseid
config['fst_rundir'] = './rundir/'
config['fst_exe'] = 'openfast'
#config['fst_exe'] = '../../../../../FAST_v8/bin/FAST_glin64'
#config['libmap'] = '../../../../../FAST_v8/bin/libmap-1.20.10.dylib'
config['ad_file_type'] = 1 

# Additional parameters
TMAX = 90
config['TMax'] = TMAX

# Add Turbsim then FAST
root.add('fast_component', FST8Workflow(config, caseid))
root.add('turbsim_component', turbsimGroup())

# # Set up recorder
recorder = SqliteRecorder('omdaoCase1.sqlite')
top.driver.add_recorder(recorder)

# Perform setup and run OpenMDAO problem
top.setup()

top.root.turbsim_component.wrapper.turbsim_exe = '/Users/jquick/TurbSim/bin/TurbSim_glin64'
#top.root.fast_component.writer.fst_vt.steady_wind_params.HWindSpeed = 15.12345
top.root.fast_component.writer.fst_vt.turbsim_wind_params.Filename = './turbsim_default.bts' # one directory below true location....
top.root.fast_component.writer.fst_vt.inflow_wind.WindType = 3
top.root.fast_component.writer.fst_vt.fst_sim_ctrl.TMax = TMAX
#top.root.turbsim_component.run_dir = './rundir/'
#for i in range(2): # Can't set fst rundir
#top.root.turbsim_component.run_dir = './rundir%i/'%i
top.root.turbsim_component.run_dir = './rundir'
#top.root.fast_component.writer.fst_file = './rundir%i/'%i
top.root.turbsim_component.writer.turbsim_vt.runtime_options.RandSeed1 = 10000 + 11
top.root.turbsim_component.execute()
top.run()
plt.plot(top['fast_component.RootMxc1'])
np.save('fout', top['fast_component.RootMxc1'])
top.cleanup()   #Good practice, especially when using recorder
plt.savefig('seeds.pdf')
