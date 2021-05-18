import os
import sys
import time

from wisdem import run_wisdem

"""
Example showing how WISDEM values can be changed programmatically in Python.

This uses the `overridden_values` dict given to `run_wisdem`.
Specifically, you can supply a dictionary of values to overwrite after
setup is called.
"""


mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
fname_wt_input = os.path.join(os.path.dirname(mydir), "02_reference_turbines", "IEA-15-240-RWT.yaml")
fname_modeling_options = mydir + os.sep + "modeling_options.yaml"
fname_analysis_options = mydir + os.sep + "analysis_options.yaml"

wt_opt, modeling_options, analysis_options = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options)
print(f"Tip deflection: {wt_opt['rotorse.rs.tip_pos.tip_deflection'][0]} meters")


# Construct a dict with values to overwrite
overridden_values = {}
overridden_values["rotorse.wt_class.V_mean_overwrite"] = 11.5

# Run the modified simulation with the overwritten values
wt_opt, modeling_options, opt_options = run_wisdem(
    fname_wt_input,
    fname_modeling_options,
    fname_analysis_options,
    overridden_values=overridden_values,
)
print(f"Tip deflection: {wt_opt['rotorse.rs.tip_pos.tip_deflection'][0]} meters")
