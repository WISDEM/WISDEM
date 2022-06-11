__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


import os

import yaml
from yaml import Dumper

from wisdem.orbit.core import loader


def load_config(filepath):
    """
    Load an ORBIT config at `filepath`.

    Parameters
    ----------
    filepath : str
        Path to yaml config file.
    """

    with open(filepath, "r") as f:
        data = yaml.load(f, Loader=loader)

    return data


def save_config(config, filepath, overwrite=False):
    """
    Save an ORBIT `config` to `filepath`.

    Parameters
    ----------
    config : dict
        ORBIT configuration.
    filepath : str
        Location to save config.
    overwrite : bool (optional)
        Overwrite file if it already exists. Default: False.
    """

    dirs = os.path.split(filepath)[0]
    if dirs and not os.path.isdir(dirs):
        os.makedirs(dirs)

    if overwrite is False:
        if os.path.exists(filepath):
            raise FileExistsError(f"File already exists at '{filepath}'.")

    with open(filepath, "w") as f:
        yaml.dump(config, f, Dumper=Dumper, default_flow_style=False)
