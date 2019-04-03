#!/usr/bin/env python
# encoding: utf-8

from setuptools import setup, find_packages


setup(
    name='WISDEM',
    version='0.2.0',
    description='Wind-Plant Integrated System Design & Engineering Model',
    author='NREL WISDEM Team',
    author_email='systems.engineering@nrel.gov',
    install_requires=['openmdao>=1.6','akima','airfoilprep','ccblade','pbeam','pyframe3dd','pymap','commonse',
                      'offshorebos','plant_financese','nrel_csm','rotorse','towerse','drivese',
                      'floatingse','pyoptsparse','turbine_costsse'],
    package_data= {'WISDEM': []},
    package_dir={'': 'src'},
    packages=['wisdem'],
    license='Apache License, Version 2.0',
    dependency_links=[
        'https://github.com/WISDEM/akima/tarball/master#egg=akima',
        'https://github.com/WISDEM/AirfoilPreppy/tarball/master#egg=airfoilprep',
        'https://github.com/WISDEM/CCBlade/tarball/master#egg=ccblade',
        'https://github.com/WISDEM/pBeam/tarball/master#egg=pbeam',
        'https://github.com/WISDEM/pyFrame3DD/tarball/master#egg=pyframe3dd',
        'https://github.com/WISDEM/pyMAP/tarball/master#egg=pymap',
        'https://github.com/WISDEM/CommonSE/tarball/master#egg=commonse',
        'https://github.com/WISDEM/OffshoreBOS/tarball/master#egg=offshorebos',
        'https://github.com/WISDEM/Plant_FinanceSE/tarball/master#egg=plant_financese',
        'https://github.com/WISDEM/Turbine_CostsSE/tarball/master#egg=turbine_costsse',
        'https://github.com/WISDEM/NREL_CSM/tarball/master#egg=nrel_csm',
        'https://github.com/WISDEM/TowerSE/tarball/master#egg=towerse',
        'https://github.com/WISDEM/RotorSE/tarball/master#egg=rotorse',
        'https://github.com/WISDEM/DriveSE/tarball/master#egg=drivese',
        'https://github.com/WISDEM/FloatingSE/tarball/master#egg=floatingse',
        'https://github.com/OpenMDAO/pyoptsparse/tarball/master#egg=pyoptsparse',
        'https://github.com/OpenMDAO/OpenMDAO1/tarball/master#egg=openmdao',
    ],
    zip_safe=False
)
