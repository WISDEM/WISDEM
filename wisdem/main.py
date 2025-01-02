import os
import sys

# Numpy deprecation warnings
import warnings

import numpy as np

from wisdem.inputs import load_yaml
from wisdem.inputs.gui import run as guirun
from wisdem.glue_code.runWISDEM import run_wisdem

warnings.filterwarnings("ignore", category=np.exceptions.VisibleDeprecationWarning)
warnings.simplefilter("ignore", RuntimeWarning, lineno=175)
warnings.simplefilter("ignore", RuntimeWarning, lineno=177)

def read_master_file(fyaml):
    if os.path.exists(fyaml):
        print("...Reading master input file,", fyaml)
    else:
        raise FileNotFoundError("The master input file, " + fyaml + ", cannot be found.")

    input_yaml = load_yaml(fyaml)

    check_list = ["geometry_file", "modeling_file", "analysis_file"]
    for f in check_list:
        if not os.path.exists(input_yaml[f]):
            raise FileNotFoundError("The " + f + " entry, " + input_yaml[f] + ", cannot be found.")

    return input_yaml


def wisdem_cmd():
    usg_msg = "WISDEM command line launcher\n    Arguments: \n    wisdem : Starts GUI\n    wisdem input.yaml : Runs master yaml file that specifies geometry, modeling, and analysis files\n    wisdem geom.yaml modeling.yaml analysis.yaml : Runs specific geometry, modeling, and analysis files\n"

    # Look for help message
    help_flag = False
    for k in range(len(sys.argv)):
        if sys.argv[k] in ["-h", "--help"]:
            help_flag = True

    if help_flag:
        print(usg_msg)

    elif len(sys.argv) == 1:
        # Launch GUI
        guirun()

    elif len(sys.argv) == 2:
        # Grab master input file
        fyaml = sys.argv[1]
        if os.path.exists(fyaml):
            print("...Reading master input file,", fyaml)
        else:
            raise FileNotFoundError("The master input file, " + fyaml + ", cannot be found.")
        yaml_dict = load_yaml(fyaml)

        check_list = ["geometry_file", "modeling_file", "analysis_file"]
        for f in check_list:
            if not os.path.exists(yaml_dict[f]):
                raise FileNotFoundError("The " + f + " entry, " + yaml_dict[f] + ", cannot be found.")

        # Run WISDEM (also saves output)
        wt_opt, modeling_options, opt_options = run_wisdem(
            yaml_dict["geometry_file"], yaml_dict["modeling_file"], yaml_dict["analysis_file"]
        )

    elif len(sys.argv) == 4:
        check_list = ["geometry", "modeling", "analysis"]
        for k, f in enumerate(sys.argv[1:]):
            if not os.path.exists(f):
                raise FileNotFoundError("The " + check_list[k] + " file, " + f + ", cannot be found.")

        # Run WISDEM (also saves output)
        wt_opt, modeling_options, opt_options = run_wisdem(sys.argv[1], sys.argv[2], sys.argv[3])

    else:
        # As if asked for help
        print("Unrecognized set of inputs.  Usage:")
        print(usg_msg)

    sys.exit(0)


if __name__ == "__main__":
    wisdem_cmd()
