import os
import numpy as np
import yaml
import ruamel_yaml as ry

"""
Common utilites for handling the text I/O for using AeroelasticSE
"""

def convert_str(val):
    # string parsing tool

    def try_type(val, data_type):
        try:
            data_type(val)
            return True
        except ValueError:
            return False

    if try_type(val, int):
        return int(val)
    elif try_type(val, float):
        return float(val)
    elif val=='True':
        return True
    elif val=='False':
        return False
    else:
        return val


def str_repeats(string):
    # find repeated sequences in a string
    for x in range(1, len(string)):
        substring = string[:x]
        if substring * (len(string)//len(substring))+(substring[:len(string)%len(substring)]) == string:
            return substring


def load_case_matrix(fname):
    # load the AeroelasticSE case_matrix

    fid = open(fname)
    text = fid.readlines()
    fid.close()

    module    = text[0].strip().split()
    var_names = text[1].strip().split()
    if len(module) == len(var_names)+1:
        var_names = ["ID"] + var_names

    out = {}
    for var in var_names:
        out[var] = []

    for ln in text[2:]:
        val_list = ln.strip().split()
        for var, val in zip(var_names, val_list):
            out[var].append(convert_str(val))

    return out

def load_yaml(fname_input, package=0):
    # Import a .yaml file

    if package == 0:
        with open(fname_input) as f:
            data = yaml.load(f)
        return data

    elif package == 1:
        with open(fname_input, 'r') as myfile:
            text_input = myfile.read()
        myfile.close()
        yaml = ry.YAML()
        return dict(yaml.load(text_input))


def save_yaml(outdir, fname, data_out):
    fname = os.path.join(outdir, fname)

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    f = open(fname, "w")
    yaml=ry.YAML()
    yaml.default_flow_style = None
    yaml.width = float("inf")
    yaml.indent(mapping=4, sequence=6, offset=3)
    yaml.dump(data_out, f)


def select_cases(cases, var_sel, val_sel):
    # Find a variable value from the AeroelasticSE case_matrix

    n_var = len(var_sel)
    n_cases = len(cases[var_sel[0]])

    truth = [True]*n_cases
    for vari, vali in zip(var_sel, val_sel):
        test = [valj == vali for valj in cases[vari]]
        truth = [truthi and testi for truthi, testi in zip(truth, test)]

    case_idx = [i for i, x in enumerate(truth) if x]
    return case_idx