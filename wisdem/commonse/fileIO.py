import os
import pickle

import numpy as np
import pandas as pd
import scipy.io as sio

def get_variable_list(prob):

    # Get all OpenMDAO inputs and outputs into a dictionary
    input_dict = prob.model.list_inputs(prom_name=True, units=True, desc=True, is_indep_var=True, out_stream=None)
    for k in range(len(input_dict)):
        input_dict[k][1]["type"] = "input"

    inter_dict = prob.model.list_inputs(prom_name=True, units=True, desc=True, is_indep_var=False, out_stream=None)
    for k in range(len(inter_dict)):
        inter_dict[k][1]["type"] = "intermediate"

    #var_dict = prob.model.list_inputs(prom_name=True, units=True, desc=True, out_stream=None)
    #for k in range(len(var_dict)):
    #    var_dict[k][1]["type"] = "output"

    out_dict = prob.model.list_outputs(prom_name=True, units=True, desc=True, out_stream=None)
    for k in range(len(out_dict)):
        out_dict[k][1]["type"] = "output"
        
    var_dict = input_dict.copy()
    var_dict.extend(inter_dict)
    var_dict.extend(out_dict)
    return input_dict, out_dict, var_dict


def variable_dict2df(var_dict):
    data = {}
    data["variables"] = []
    data["type"] = []
    data["units"] = []
    data["values"] = []
    data["description"] = []
    for k in range(len(var_dict)):
        unit_str = var_dict[k][1]["units"]
        if unit_str is None:
            unit_str = ""

        iname = var_dict[k][1]["prom_name"]
        itype = var_dict[k][1]["type"]
        if iname in data["variables"]:
            iprev = data["variables"].index( iname )
            if itype == "output":
                data["type"][iprev] = itype
            continue

        data["variables"].append(iname)
        data["type"].append(itype)
        data["units"].append(unit_str)
        data["values"].append(var_dict[k][1]["val"])
        data["description"].append(var_dict[k][1]["desc"])
        
    return pd.DataFrame(data)


def save_data(fname, prob, npz_file=True, mat_file=True, xls_file=True):
    # Get the variables
    _, _, var_dict = get_variable_list(prob)
    
    # Remove file extension
    froot = os.path.splitext(fname)[0]

    # Pickle the full archive so that we can load it back in if we need
    with open(froot + ".pkl", "wb") as f:
        pickle.dump(var_dict, f)

    # Reduce to variables we can save for matlab or python
    if npz_file or mat_file:
        array_dict = {}
        for k in range(len(var_dict)):
            unit_str = var_dict[k][1]["units"]
            if unit_str is None or unit_str in ["Unavailable","n/a"]:
                unit_str = ""
            elif len(unit_str) > 0:
                unit_str = "_" + unit_str

            iname = var_dict[k][1]["prom_name"] + unit_str
            value = var_dict[k][1]["val"]

            if iname in array_dict:
                continue

            if type(value) in [type(np.array([])), type(0.0), type(0), np.float64, np.int64]:
                array_dict[iname] = value
            elif type(value) == type(True):
                array_dict[iname] = np.bool_(value)
            elif type(value) == type(""):
                array_dict[iname] = np.str_(value)
            elif type(value) == type([]):
                temp_val = np.empty(len(value), dtype=object)
                temp_val[:] = value[:]
                array_dict[iname] = temp_val
            # else:
            #    print(var_dict[k])

    # Save to numpy compatible
    if npz_file:
        kwargs = {key: array_dict[key] for key in array_dict.keys()}
        np.savez_compressed(froot + ".npz", **kwargs)

    # Save to matlab compatible
    if mat_file:
        sio.savemat(froot + ".mat", array_dict, long_field_names=True)

    if xls_file:
        df = variable_dict2df(var_dict)
        df.to_excel(froot + ".xlsx", index=False)
        df.to_csv(froot + ".csv", index=False)


def load_data(fname, prob):
    # Remove file extension
    froot = os.path.splitext(fname)[0]

    # Load in the pickled data
    with open(froot + ".pkl", "rb") as f:
        var_dict = pickle.load(f)

    # Store into Problem object
    for k in range(len(var_dict)):
        iname = var_dict[k][0]
        iname2 = var_dict[k][1]["prom_name"]
        value = var_dict[k][1]["val"]
        try:
            prob.set_val(iname, value)
        except Exception:
            pass
        try:
            prob.set_val(iname2, value)
        except Exception:
            pass

    return prob
