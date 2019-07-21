#!/usr/bin/env python

import os,sys

def configuration(parent_package='', top_path=None):
    
    from numpy.distutils.misc_util import Configuration
    
    config = Configuration('wisdem', parent_package, top_path)
    
    config.add_subpackage('airfoilprep')
    config.add_subpackage('assemblies')
    config.add_subpackage('ccblade')
    config.add_subpackage('commonse')
    #config.add_subpackage('drivese')
    #config.add_subpackage('generatorse')
    config.add_subpackage('floatingse')
    #config.add_subpackage('landbosse')
    config.add_subpackage('nrelcsm')
    #config.add_subpackage('offshorebosse')
    config.add_subpackage('pBeam')
    config.add_subpackage('plant_financese')
    config.add_subpackage('pyframe3dd')
    #config.add_subpackage('pymap')
    #config.add_subpackage('rotorse')
    config.add_subpackage('towerse')
    config.add_subpackage('turbine_costsse')
    config.add_data_files('LICENSE')

return config
