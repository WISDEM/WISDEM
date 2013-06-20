#!/usr/bin/env python
# encoding: utf-8

from setuptools import setup  # , find_packages


setup(
    name='AirfoilPrep.py',
    version='0.1.0',
    description='Airfoil preprocessing for wind turbine applications',
    author='S. Andrew Ning',
    author_email='andrew.ning@nrel.gov',
    package_dir={'': 'src/twister/rotoraero'},
    py_modules=['airfoilprep'],
    license='Apache License, Version 2.0',
    # install_requires=['numpy'],
)
