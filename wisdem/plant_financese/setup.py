#!/usr/bin/env python

import os
import sys
from numpy.distutils.misc_util import Configuration


def configuration(parent_package='',top_path=None):
    
    config = Configuration('', parent_package, top_path)
    config.add_data_files('LICENSE','README')

    return config
   
