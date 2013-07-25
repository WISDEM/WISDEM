#!/usr/bin/env python
# encoding: utf-8

# from setuptools import setup
from numpy.distutils.core import setup, Extension
import numpy
import sys


# copy specific files into distribution
if 'sdist' in sys.argv:
    import os
    from shutil import copytree, copyfile

    # copy airfoilprep documentation
    copytree('airfoilpreppy-setup/docs/_build/html', 'docs/airfoilpreppy-docs')

    # copy test files
    os.mkdir('test')
    testdir = os.path.join(os.getcwd(), 'src', 'wisdem', 'rotor', 'test')
    copyfile(os.path.join(testdir, 'test_ccblade.py'), 'test/test_ccblade.py')
    copytree(os.path.join(testdir, os.pardir, '5MW_files', '5MW_AFFiles'), 'test/5MW_AFFiles')




setup(
    name='CCBlade',
    version='0.3.0',
    description='Blade element momentum aerodynamics for wind turbines',
    author='S. Andrew Ning',
    author_email='andrew.ning@nrel.gov',
    package_dir={'': 'src/wisdem/rotor', 'common': 'src/wisdem/common'},
    py_modules=['ccblade_sa', 'airfoilprep', 'common.csystem', 'common.utilities'],
    # install_requires=['numpy', 'scipy'],
    # test_suite='test.test_ccblade.py',
    license='Apache License, Version 2.0',
    ext_modules=[Extension('_bemroutines', ['src/wisdem/rotor/bemroutines.f90'], extra_compile_args=['-O2']),
                 Extension('_akima', ['src/wisdem/common/akima.c'], include_dirs=[numpy.get_include()])]
)


