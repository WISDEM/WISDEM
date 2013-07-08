#!/usr/bin/env python
# encoding: utf-8
"""
utilities.py

Created by Andrew Ning on 2013-05-31.
Copyright (c) NREL. All rights reserved.
"""

import os
import shutil
import numpy as np
import atexit
from math import pi

RPM2RS = pi/30.0
RS2RPM = 30.0/pi


def exe_path(defaultPath, exeName, searchPath):
    """find path to an executable

    Parameters
    ----------
    defaultPath : str
        path where executable should be located by default (may be None if no default exists)
    exeName : str
        name of the executable without extension (not case sensitive)
    searchPath : str
        path to look for executable in

    Returns
    -------
    exe_path : str
        full path to the executable

    """

    found = False

    if defaultPath is not None and os.path.exists(defaultPath):
        foundPath = defaultPath
        found = True

    else:
        names = [exeName]  # seems to be case-insensitive
        extensions = ['', '.exe']
        for filename in [(os.path.join(searchPath, name + ext)) for name in names for ext in extensions]:
            if os.path.exists(filename):
                foundPath = filename
                found = True
                break

    if not found:
        raise Exception('Did not find ' + exeName + ' executable')

    return foundPath



def mktmpdir(dirname, DEBUG, tmp_files=None):
    """create a working directory at location dirname"""

    # create working directory
    try:
        os.mkdir(dirname)

    except OSError as e:
        if e.errno != 17:  # silently ignore case where directory exists
            print 'OS error({0}): {1}'.format(e.errno, e.strerror)

    # schedule deletion of working directory
    @atexit.register
    def cleanup():
        if not DEBUG:
            shutil.rmtree(dirname)

            if tmp_files is not None:
                for f in tmp_files:
                    os.remove(f)



# def rmdir(dirname):
#     """remove working directory dirname"""

#     shutil.rmtree(dirname)


def cosd(value):
    """cosine of value where value is given in degrees"""

    return np.cos(np.radians(value))


def sind(value):
    """sine of value where value is given in degrees"""

    return np.sin(np.radians(value))


