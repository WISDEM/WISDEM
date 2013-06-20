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


def exe_path(defaultPath, exeName, searchPath):
    """if defaultPath exists and is not None then it is used,
    otherwise an executable named 'name' (case-insensitive) is looked
    for in the same path as the searchPath"""

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


def mkdir(dirname):

    # create working directory
    try:
        os.mkdir(dirname)

    except OSError as e:
        if e.errno != 17:  # silently ignore case where directory exists
            print 'OS error({0}): {1}'.format(e.errno, e.strerror)


def rmdir(dirname):

    shutil.rmtree(dirname)


def cosd(value):

    return np.cos(np.radians(value))


def sind(value):

    return np.sin(np.radians(value))


