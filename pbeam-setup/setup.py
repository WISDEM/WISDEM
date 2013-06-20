#!/usr/bin/env python
# encoding: utf-8

# import setuptools
from numpy.distutils.core import setup, Extension
from os.path import join


path = join('src', 'twister', 'rotorstruc', 'pBEAM')
src = ['Poly.cpp', 'myMath.cpp', 'BeamFEA.cpp', 'Beam.cpp', 'pyBEAM.cpp']

for i in range(4):
    src[i] = join(path, 'pBEAM', src[i])
src[4] = join(path, 'pyBEAM', src[4])


f = open('MANIFEST.in', 'a')
f.write('recursive-include ' + path + ' * \n')
f.write('recursive-exclude ' + join(path, 'pBEAM.xcodeproj') + ' * \n')
f.write('exclude ' + join(path, 'pBEAM', 'main.cpp') + '\n')
f.close()

setup(
    name='pBEAM',
    version='0.1.0',
    description='Polynomial Beam Element Analysis Module. Finite element analysis for beam-like structures.',
    author='S. Andrew Ning',
    author_email='andrew.ning@nrel.gov',
    # install_requires=['numpy', 'scipy'],
    license='Apache License, Version 2.0',
    ext_modules=[Extension('_pBEAM', sources=src, extra_compile_args=['-O2'],
                           include_dirs=[join(path, 'pBEAM')],
                           libraries=['boost_python-mt', 'boost_system-mt', 'lapack'])]

)