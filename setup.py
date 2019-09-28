#!/usr/bin/env python

import os
import sys
import platform
import glob
from setuptools import setup, find_packages
from numpy.distutils.core import setup, Extension

os.environ['NPY_DISTUTILS_APPEND_FLAGS'] = '1'

# CXXFLAGS for pBEAM
if platform.system() == 'Windows':
    pbeamArgs = ['-std=gnu++11','-fPIC']
else:
    pbeamArgs = ['-std=c++11','-fPIC']

# CFLAGS for pyMAP
if platform.system() == 'Windows': # For Anaconda
    pymapArgs = ['-O1', '-m64', '-fPIC', '-std=c99','-DCMINPACK_NO_DLL']
elif sys.platform == 'cygwin':
    pymapArgs = ['-O1', '-m64', '-fPIC', '-std=c99']
elif platform.system() == 'Darwin':
    pymapArgs = ['-O1', '-m64', '-fno-omit-frame-pointer', '-fPIC']#, '-std=c99']
else:
    #pymapArgs = ['-O1', '-m64', '-fPIC', '-std=c99', '-D WITH_LAPACK']
    pymapArgs = ['-O1', '-m64', '-fPIC', '-std=c99']

# All the extensions
bemExt     = Extension('wisdem.ccblade._bem',
                       sources=[os.path.join('wisdem','ccblade','bem.f90')],
                       extra_compile_args=['-O2','-fPIC'])
pyframeExt = Extension('wisdem.pyframe3dd._pyframe3dd',
                       sources=glob.glob(os.path.join('wisdem','pyframe3dd','src','*.c')))
precompExt = Extension('wisdem.rotorse._precomp',
                       sources=[os.path.join('wisdem','rotorse','PreCompPy.f90')],
                       extra_compile_args=['-O2','-fPIC'])
pbeamExt   = Extension('wisdem.pBeam._pBEAM',
                       sources=glob.glob(os.path.join('wisdem','pBeam','src','*.cpp')),
                       extra_compile_args=pbeamArgs,
                       include_dirs=[os.path.join('wisdem','include')])
pymapExt   = Extension('wisdem.pymap._libmap', sources=glob.glob(os.path.join('wisdem','pymap','**','*.c'), recursive=True)+
                       glob.glob(os.path.join('wisdem','pymap','**','*.cc'), recursive=True),
                       extra_compile_args=pymapArgs,
                       include_dirs=[os.path.join('wisdem','include','lapack')])

# Top-level setup
setup(
    name             = 'WISDEM',
    version          = '2.0.1',
    description      = 'Wind-Plant Integrated System Design & Engineering Model',
    long_description =  'WISDEM is a Python package for conducting multidisciplinary analysis and optimization of wind turbines and plants',
    url              = 'https://github.com/WISDEM/WISDEM',
    author           = 'NREL WISDEM Team',
    author_email     = 'systems.engineering@nrel.gov',
    install_requires = ['openmdao>= 2.0','numpy','scipy','pandas','simpy'],
    package_data     =  {'wisdem': []},
    #package_dir      = {'': 'wisdem'},
    packages         = find_packages(exclude=['docs', 'tests', 'ext']),
    license          = 'Apache License, Version 2.0',
    python_requires  = '>= 3.4',
    ext_modules      = [bemExt, pyframeExt, precompExt, pbeamExt, pymapExt],
    zip_safe         = False
)
