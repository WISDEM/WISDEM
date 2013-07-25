# CCBlade

A blade element momentum method for analyzing wind turbine aerodynamic performance that is robust (guaranteed convergence), fast (superlinear convergence rate), and smooth (continuously differentiable).

Author: S. Andrew Ning

## Prerequisites

C compiler, Fortran compiler, NumPy, SciPy

## Installation

Install CCBlade with the following command.

    $ python setup.py install

Note that the installation also includes AirfoilPrep.py.  Though not strictly necessary to use with CCBlade, it is convenient when working with AeroDyn input files or doing any aerodynamic preprocessing of airfoil data.

## Run Unit Tests

To check if installation was successful, run the unit tests

    $ python test/test_ccblade.py

## Detailed Documentation

Open `docs/index.html` in your browser.  The HTML version is hyperlinked with the source code, but an alternative PDF version is also available at `docs/UserGuide.pdf`.