#!/usr/bin/env python

import os
import sys
import platform
from numpy.distutils.misc_util import Configuration

if platform.system() == 'Windows':
    arglist = ['-std=gnu++11','-fPIC']
else:
    arglist = ['-std=c++11','-fPIC']

def configuration(parent_package='',top_path=None):
    
    config = Configuration('', parent_package, top_path)
    #config.add_data_files('LICENSE','README')
    config.add_library('_pBEAM', sources=[os.path.join('src','*.cpp')])
    config.add_extension('_pBEAM', sources=['_pBEAM.so'], extra_compile_args=arglist,
                         include_dirs=['include'], libraries=['_pBEAM'])
    return config
   
