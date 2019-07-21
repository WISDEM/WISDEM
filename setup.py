#!/usr/bin/env python

import os
import sys
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration

if len(sys.argv) == 1:
    print("\nTo install, run: python setup.py install --user\n\n"
          "To build, run: python setup.py build_ext --inplace\n\n"
          "For help on C-compiler options run: python setup.py build --help-compiler\n\n"
          "For help on Fortran-compiler options run: python setup.py build --help-fcompiler\n\n"
          "To specify a Fortran compiler to use run: python setup.py install --user --fcompiler=<fcompiler name>\n\n"
          "For further help run: python setup.py build --help"
      )
    sys.exit(-1)

def configuration(parent_package='',top_path=None):
    config = Configuration(None,parent_package,top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)
    config.add_subpackage('wisdem')
    return config

setup(
    name             = 'wisdem',
    version          = '2.0.0',
    author           = 'NREL WISDEM Team',
    author_email     = 'systems.engineering@nrel.gov',
    maintainer       = 'NREL WISDEM Team',
    maintainer_email = 'systems.engineering@nrel.gov',
    description      = 'Wind-Plant Integrated System Design & Engineering Model',
    long_description = 'WISDEM is a Python package for conducting multidisciplinary analysis and optimization of wind turbines and plants',
    keywords         = ['wind','systems engineering','mdao'],
    license          = 'Apache License, Version 2.0',
    platforms        = ['Windows','Linux','Solaris','Mac OS-X','Unix'],
    classifiers      = ['Development Status :: 4 - Beta',
                        'Environment :: Console',
                        'Intended Audience :: Science/Research',
                        'Intended Audience :: Developers',
                        'Intended Audience :: Education',
                        'License :: Apache',
                        'Operating System :: Microsoft :: Windows',
                        'Operating System :: POSIX :: Linux',
                        'Operating System :: Unix',
                        'Operating System :: MacOS',
                        'Programming Language :: Python',
                        'Topic :: Scientific/Engineering',
                        'Topic :: Software Development',
                        'Topic :: Education'],
    configuration = configuration,
)
    
