#!/usr/bin/env python3
import os

from wisdem import run_wisdem

## File management
mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
fname_wt_input = mydir + os.sep + "nrel5mw_jacket.yaml"
fname_modeling_options = mydir + os.sep + "modeling_options_jacket.yaml"
fname_analysis_options = mydir + os.sep + "analysis_options_jacket.yaml"

wt_opt, analysis_options, opt_options = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options)

# print results from the analysis or optimization
print()
print("mass (kg) =", wt_opt["fixedse.jacket_mass"])
print("stress1 =", wt_opt["fixedse.constr_stress"])
print("GL buckling =", wt_opt["fixedse.constr_global_buckling"])
print("Shell buckling =", wt_opt["fixedse.constr_shell_buckling"])
