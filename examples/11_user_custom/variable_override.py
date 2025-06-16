from os import path as osp

from wisdem import run_wisdem

"""
Example showing how WISDEM values can be changed programmatically in Python.

This uses the `overridden_values` dict given to `run_wisdem`.
Specifically, you can supply a dictionary of values to overwrite after
setup is called.
"""

mydir = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), "02_reference_turbines")
fname_wt_input = osp.join(mydir, "IEA-15-240-RWT.yaml")
fname_modeling_options = osp.join(mydir, "modeling_options_iea15.yaml")
fname_analysis_options = osp.join(mydir, "analysis_options.yaml")

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
