#!/usr/bin/env python

import os
import sys
import platform
from numpy.distutils.misc_util import Configuration


def configuration(parent_package='',top_path=None):
    
    config = Configuration('', parent_package, top_path)
    #config.add_data_files('LICENSE','README')
    config.add_library('_pyframe3dd', sources=[os.path.join('src','*.c')])
    config.add_extension('_pyframe3dd', sources=['src/_pyframe3dd.so'], 
                         libraries=['_pyframe3dd'])
    return config
   
