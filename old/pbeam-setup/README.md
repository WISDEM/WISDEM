# pBEAM: polynomial beam element analysis module

A finite element method for beam-like structures.

Author: S. Andrew Ning

## Prerequisites

C++ compiler, [Boost C++ Libraries](http://www.boost.org), LAPACK, NumPy, SciPy

## Installation

Install pBEAM with the following command.

    $ python setup.py install


To check if installation was successful run Python from the command line

    $ python

and import the module.  If no errors are issued, then the installation was successful.

    >>> import _pBEAM

## Unit Tests

pBEAM has a large range of unit tests, but they are only accessible through C++.  They are intended to test the integrity of the underying code for development purposes, rather than the python interface.  However, if you want to run the tests then change directory to `src/twister/rotorstruc/pBEAM` and run


    $ make test CXX=g++

where the name of your C++ compiler should be inserted in the place of g++.  The script will build the test executable and run all tests.  The phrase "No errors detected" signifies that all the tests passed.  You can remove the remove the test executable and all object files by running

    $ make clean


## Detailed Documentation

Open `docs/index.html` in your browser.
