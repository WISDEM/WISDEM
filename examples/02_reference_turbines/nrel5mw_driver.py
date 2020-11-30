#!/usr/bin/env python3
import os
from wisdem import run_wisdem


## File management
mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
fname_wt_input = mydir + os.sep + "nrel5mw.yaml"
fname_modeling_options = mydir + os.sep + "modeling_options.yaml"
fname_analysis_options = mydir + os.sep + "analysis_options.yaml"

wt_opt, analysis_options, opt_options = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options)
# end

print(wt_opt["wt.tcc.blade_mass"])
print(wt_opt["wt.drivese.blades_I"])
print(wt_opt["wt.drivese.pitch_system.BRFM"])
print(wt_opt["wt.drivese.F_hub"])
print(wt_opt["wt.drivese.M_hub"])
