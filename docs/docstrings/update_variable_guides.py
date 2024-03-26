#!/usr/bin/env python
import numpy as np
import os.path as osp

from wisdem import run_wisdem
from wisdem.commonse import fileIO
from get_docstrings import get_all_docstrings

# Get all docstrings from WISDEM files
parsed_dict = get_all_docstrings()

# Run comprehensive cases that should grab almost all inputs and outputs (might be missing some generators and BOS options)
examp_dir = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__)))), "examples")
mydir = osp.join(examp_dir, "02_reference_turbines")
iea15_mono_geom = osp.join(mydir, "IEA-15-240-RWT.yaml")
iea3p4_geom = osp.join(mydir, "IEA-3p4-130-RWT.yaml")
rwt_modeling = osp.join(mydir, "modeling_options.yaml")
rwt_analysis = osp.join(mydir, "analysis_options.yaml")

mydir = osp.join(examp_dir, "09_floating")
iea15_float_geom = osp.join(mydir, "IEA-15-240-RWT_VolturnUS-S.yaml")
rwt_float_modeling = osp.join(mydir, "modeling_options.yaml")
rwt_float_analysis = osp.join(mydir, "analysis_options.yaml")

mydir = osp.join(examp_dir, "17_jacket")
nrel_jack_geom = osp.join(mydir, "nrel5mw_jacket.yaml")
rwt_jack_modeling = osp.join(mydir, "modeling_options_jacket.yaml")
rwt_jack_analysis = osp.join(mydir, "analysis_options_jacket.yaml")

iea15_mono_prob, model_dict, anal_dict              = run_wisdem(iea15_mono_geom, rwt_modeling, rwt_analysis)
iea3p4_prob, _, _                                   = run_wisdem(iea3p4_geom    , rwt_modeling, rwt_analysis)
iea15_float_prob, model_float_dict, anal_float_dict = run_wisdem(iea15_float_geom, rwt_float_modeling, rwt_float_analysis)
nrel_jack_prob, model_jack_dict, anal_jack_dict     = run_wisdem(iea15_float_geom, rwt_float_modeling, rwt_float_analysis)

# Extract inputs and outputs from the models
all_inputs = []
all_outputs = []
for p in [iea15_mono_prob, iea3p4_prob, iea15_float_prob, nrel_jack_prob]:
    input_k, output_k, _ = fileIO.get_variable_list(p)
    if len(all_inputs) == 0:
        all_inputs = input_k
        all_outputs = output_k
    else:
        all_inputs.extend(input_k)
        all_outputs.extend(output_k)

# Use Pandas for some data cleansing and writing to csv
def write_guide(in_dict, fname):
    mydf = fileIO.variable_dict2df(in_dict)
    mydf.rename(columns={"variables":"Variable",
                         "units":"Units",
                         "description":"Description"}, inplace=True)
    mydf = mydf[["Variable", "Units", "Description"]]
    mydf.set_index("Variable", inplace=True)
    mydf = mydf[~mydf.index.duplicated(keep='first')]
    mydf.reset_index(inplace=True)

    # Fold in docstrings
    mynames = mydf["Variable"].to_list()
    mydesc  = mydf["Description"].to_list()
    for k in range(len(mynames)):
        ivar = mynames[k]
        if ivar in parsed_dict:
            idesc = parsed_dict[ivar]
            mydesc[k] += "" if idesc is None else idesc
    mydf["Description"] = mydesc
    mydf['Units'] = mydf['Units'].replace(np.nan, '-')
    mydf['Description'] = mydf['Description'].replace(np.nan,'None')

    # Write everything out
    mydf.to_csv(fname, index=False)
    mydf.to_json(fname.replace('csv','json'))#, index=False)

    return mydf

inputs_df = write_guide(all_inputs, "input_variable_guide.csv")
outputs_df = write_guide(all_outputs, "output_variable_guide.csv")
    
