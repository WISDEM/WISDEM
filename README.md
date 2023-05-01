# WISDEM&reg;

[![Actions Status](https://github.com/WISDEM/WISDEM/workflows/CI_WISDEM/badge.svg?branch=develop)](https://github.com/WISDEM/WISDEM/actions)
[![Coverage Status](https://coveralls.io/repos/github/WISDEM/WISDEM/badge.svg?branch=develop)](https://coveralls.io/github/WISDEM/WISDEM?branch=develop)
[![Documentation Status](https://readthedocs.org/projects/wisdem/badge/?version=master)](https://wisdem.readthedocs.io/en/master/?badge=master)


The Wind-Plant Integrated System Design and Engineering Model (WISDEM&reg;) is a set of models for assessing overall wind plant cost of energy (COE). The models use wind turbine and plant cost and energy production as well as financial models to estimate COE and other wind plant system attributes. WISDEM&reg; is accessed through Python, is built using [OpenMDAO](https://openmdao.org/), and uses several sub-models that are also implemented within OpenMDAO. These sub-models can be used independently but they are required to use the overall WISDEM&reg; turbine design capability. Please install all of the pre-requisites prior to installing WISDEM&reg; by following the directions below. For additional information about the NWTC effort in systems engineering that supports WISDEM&reg; development, please visit the official [NREL systems engineering for wind energy website](https://www.nrel.gov/wind/systems-engineering.html).

Author: [NREL WISDEM Team](mailto:systems.engineering@nrel.gov)

## Documentation

See local documentation in the `docs`-directory or access the online version at <https://wisdem.readthedocs.io/en/master/>

## Packages

WISDEM&reg; is a family of modules.  The core modules are:

* _CommonSE_ includes several libraries shared among modules
* _FloatingSE_ works with the floating platforms
* _DrivetrainSE_ sizes the drivetrain and generator systems (formerly DriveSE and GeneratorSE)
* _TowerSE_ is a tool for tower (and monopile) design
* _RotorSE_ is a tool for rotor design
* _NREL CSM_ is the regression-based turbine mass, cost, and performance model
* _ORBIT_ is the process-based balance of systems cost model for offshore plants
* _LandBOSSE_ is the process-based balance of systems cost model for land-based plants
* _Plant_FinanceSE_ runs the financial analysis of a wind plant

The core modules draw upon some utility packages, which are typically compiled code with python wrappers:

* _Airfoil Preppy_ is a tool to handle airfoil polar data
* _CCBlade_ is the BEM module of WISDEM
* _pyFrame3DD_ brings libraries to handle various coordinate transformations
* _MoorPy_ is a quasi-static mooring line model
* [_pyOptSparse_](https://github.com/mdolab/pyoptsparse) provides some additional optimization algorithms to OpenMDAO


## Installation

Installation with [Anaconda](https://www.anaconda.com) is the recommended approach because of the ability to create self-contained environments suitable for testing and analysis.  WISDEM&reg; requires [Anaconda 64-bit](https://www.anaconda.com/distribution/).

The installation instructions below use the environment name, "wisdem-env," but any name is acceptable.  For those working behind company firewalls, you may have to change the conda authentication with `conda config --set ssl_verify no`.  Proxy servers can also be set with `conda config --set proxy_servers.http http://id:pw@address:port` and `conda config --set proxy_servers.https https://id:pw@address:port`. To setup an environment based on a different Github branch of WISDEM, simply substitute the branch name for `master` in the setup line.

1.  Setup and activate the Anaconda environment from a prompt (Anaconda3 Power Shell on Windows or Terminal.app on Mac)

        conda config --add channels conda-forge
        conda env create --name wisdem-env -f https://raw.githubusercontent.com/WISDEM/WISDEM/master/environment.yml python=3.10
        conda activate wisdem-env

2.  In order to directly use the examples in the repository and peek at the code when necessary, we recommend all users install WISDEM in *developer / editable* mode using the instructions here.  If you really just want to use WISDEM as a library and lean on the documentation, you can always do `conda install wisdem` and be done.  Note the differences between Windows and Mac/Linux build systems. For Linux, we recommend using the native compilers (for example, gcc and gfortran in the default GNU suite).

        conda install -y petsc4py mpi4py                 # (Mac / Linux only)
        conda install -y gfortran                        # (Mac only without Homebrew or Macports compilers)
        conda install -y m2w64-toolchain libpython       # (Windows only)
        git clone https://github.com/WISDEM/WISDEM.git
        cd WISDEM
        python setup.py develop				 # Currently more reliable than: pip install -e


**NOTE:** To use WISDEM again after installation is complete, you will always need to activate the conda environment first with `conda activate wisdem-env`


## Run Unit Tests

Each package has its own set of unit tests.  These can be run in batch with the `test_all.py` script located in the top level `test`-directory.

## Feedback

For software issues please use <https://github.com/WISDEM/WISDEM/issues>.  For functionality and theory related questions and comments please use the NWTC forum for [Systems Engineering Software Questions](https://wind.nrel.gov/forum/wind/viewtopic.php?f=34&t=1002).
