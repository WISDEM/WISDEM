''' 
----------------- Run_TestCases -----------------
- Run the simulation cases in the Test_Cases folder
- Plot some basic results to make sure they worked

Notes:
    By default, the provided test case models point to a compiled controller at:
    "../../ROSCO/build/libdiscon.dylib". You will need have this compiled, or 
    change the path to the controller in the ServoDyn input files.
-------------------------------------------------
'''
# Import and define modules and classes
import os
import matplotlib.pyplot as plt
from ROSCO_toolbox import utilities as ROSCO_Utilities
fast_io = ROSCO_Utilities.FAST_IO()

# Define call for OpenFAST and turbsim
openfast_call    = 'openfast_dev'
turbsim_call = 'turbsim_dev'

# Define folder names in Test_Cases to run
test_cases = ['5MW_Land_DLL_WTurb',
                '5MW_OC3SPAR_DLL_WTurb_WavesIrr']

# Names of wind turbulent wind binaries
wind_binaries = ['Wind/90m_12mps_twr.bts']

# Define data to plot 
plot_categories = {}
plot_categories['LandBased'] = ['Wind1VelX', 'BldPitch1', 'GenTq', 'RotSpeed']
plot_categories['Floating'] = ['Wind1VelX', 'BldPitch1', 'GenTq', 'RotSpeed', 'PtfmPitch']


# --------------- THE ACTION ---------------
# - shouldn't need to change this section

# Compile turbsim binaries if needed
for turbfile in wind_binaries:
    if os.path.exists(turbfile):
        print('Turbsim binary file {} exists, moving on...'.format(turbfile))
    else:
        # Run turbsim
        turbinput = turbfile.split('.')[:-1][0] + '.inp'
        fast_io.run_openfast(os.getcwd(), fastcall=turbsim_call, fastfile=turbinput)


# Run test cases
for case in test_cases:
    # Run
    OpenFAST_dir = case
    fast_io.run_openfast(OpenFAST_dir,fastcall=openfast_call, chdir=True)

# Plot the results
for (cid, case), key in zip(enumerate(test_cases),plot_categories.keys()):
    # Load outdata
    filename = os.path.join(case,'{}.out'.format(case))
    print(filename)
    allinfo, alldata = fast_io.load_output(filename)

    # Plot some results
    plot_case = {}
    plot_case[key] = plot_categories[key]
    fast_io.plot_fast_out(plot_case, allinfo,
                          alldata, showplot=False)

plt.show()
