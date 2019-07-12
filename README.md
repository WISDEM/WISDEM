# WISDEM&reg;

The Wind-Plant Integrated System Design and Engineering Model (WISDEM&reg;) is a set of models for assessing overall wind plant cost of energy (COE).  The models use wind turbine and plant cost and energy production as well as financial models to estimate coe and other wind plant system attributes.  It is built in OpenMDAO and uses several sub-models that are also designed as OpenMDAO plugin-ins.  These sub-models can be used independently but they are required to use the overall WISDEM&reg; capability.  Please install all of the pre-requisites prior to installing WISDEM&reg;.  For additional information about the NWTC effort in systems engineering that supports WISDEM&reg; development, please visit the official [NREL systems engineering for wind energy website](https://www.nrel.gov/wind/systems-engineering.html).

Author: [NREL WISDEM Team](mailto:systems.engineering@nrel.gov) 

## Version

This software is a beta version 0.2.0.

## Documentation

See local documentation in the `docs`-directory or access the online version at <http://wisdem.github.io/WISDEM/>

## Packages

WISDEM&reg; is a family of modules.  The core modules are:

* _CommonSE_ includes several libraries shared among modules
* _DriveSE_ sizes the drive-train system
* _FloatingSE_ works with the floating platforms
* _OffshoreBOS_ sizes the balance of systems for offshore plants
* _Plant_FinanceSE_ runs the financial analysis of a wind plant
* _RotorSE_ is a tool for rotor design
* _TowerSE_ is a tool for tower (and monopile) design
* _Turbine_CostsSE_ is a turbine cost model
* _NREL CSM_ is the old cost-and-scaling model
* _WISDEM_ provides the interface between models

The core modules draw upon some utility packages, which are typically compiled code with python wrappers:

* _Akima_ is a spline package
* _Airfoil Preppy_ is a tool to handle airfoil polar data
* _CCBlade_ is the BEM module of WISDEM
* _pBEAM_ provides a basic beam model
* _pyFrame3DD_ brings libraries to handle various coordinate transformations
* _pyMAP_ provides a python interface to MAP++, a quasi-static mooring line model
* _pyoptsparse_ provides some additional optimization algorithms to OpenMDAO


## Installation (Anaconda)

Installation with [Anaconda](https://www.anaconda.com) is the recommended approach because of the ability to create self-contained environments suitable for testing and analysis.  WISDEM&reg; requires [Anaconda 64-bit](https://www.anaconda.com/distribution/).

The installation instructions below use the environment name, "wisdem-env," but any name is acceptable.

1.  Setup and activate the Anaconda environment from a prompt

        conda config --add channels conda-forge
        conda create -y --name wisdem-env python=3.7

    on Linux or MacOS

        source activate wisdem-env

    on Windows

        conda activate wisdem-env


2.  Install Anaconda package dependencies

        conda install -y numpy scipy matplotlib conda-build six numpydoc networkx pyparsing packaging snowballstemmer pandas openpyxl xlrd jinja2 git mpi4py imagesize idna docutils chardet babel alabaster sphinx sphinxcontrib ipython cython swig make sphinxcontrib-bibtex jsonschema ruamel.yaml compilers
	
	pip install simpy

    if sphinxcontrib-bibtex does not install correctly, it is not critical and you can press on.
    
    on Windows add on:
    
        conda install -y mingw m2w64-toolchain libpython

    on MacOS add on:
    
        xcode-select --install


3.  Next install [OpenMDAO](http://openmdao.org), the glue code and optimization library for WISDEM&reg;

        pip install OpenMDAO==1.7.4

4.  Now create a directory to store all of the WISDEM&reg; files.  This directory may be placed anywhere in the user's filesystem.

        mkdir wisdem
        cd  wisdem

5.  Git clone all of the core WISDEM&reg; modules

        git clone https://github.com/WISDEM/akima.git
        git clone https://github.com/WISDEM/AirfoilPreppy.git
        git clone https://github.com/WISDEM/CCBlade.git
        git clone https://github.com/WISDEM/CommonSE.git
        git clone https://github.com/WISDEM/DriveSE.git
        git clone https://github.com/WISDEM/FloatingSE.git
        git clone https://github.com/WISDEM/OffshoreBOS.git
        git clone https://github.com/WISDEM/Plant_FinanceSE.git
        git clone https://github.com/WISDEM/RotorSE.git
        git clone https://github.com/WISDEM/TowerSE.git
        git clone https://github.com/WISDEM/Turbine_CostsSE.git
        git clone https://github.com/WISDEM/NREL_CSM.git
        git clone https://github.com/WISDEM/WISDEM.git
        git clone https://github.com/WISDEM/pBeam.git
        git clone https://github.com/WISDEM/pyFrame3DD.git
        git clone https://github.com/WISDEM/pyMAP.git
        git clone https://github.com/mdolab/pyoptsparse.git

6.  Now install all of the packages.  The instructions here assume that the user will be interacting with the source code and incorporating code updates frequently, so the python packages are set-up for development (`python setup.py develop`), instead of hard installs (`python setup.py install`).

        cd akima
        python setup.py install 

        cd ../AirfoilPreppy
        python setup.py develop 

        cd ../CCBlade
        python setup.py develop 

        cd ../pBeam
        python setup.py develop 

        cd ../pyFrame3DD
        python setup.py develop 

        cd ../pyMAP
        python setup.py develop 

        cd ../CommonSE
        python setup.py develop 

        cd ../OffshoreBOS
        python setup.py develop 

        cd ../Plant_FinanceSE
        python setup.py develop 

        cd ../Turbine_CostsSE
        python setup.py develop 

        cd ../NREL_CSM
        python setup.py develop 

        cd ../TowerSE
        python setup.py develop 

        cd ../RotorSE
        python setup.py develop 

        cd ../WISDEM
        python setup.py develop 

        cd ../DriveSE
        python setup.py develop 

        cd ../FloatingSE
        python setup.py develop 

        cd ../pyoptsparse
	
    on Windows edit pyoptsparse\pyoptsparse\setup.py and comment out the line:
    
        #    config.add_subpackage('pyFSQP')

    for all OSes continue:
	
        python setup.py install
	cd ..



## Installation (Linux or MacOS with package management)

WISDEM&reg; can also be used with native or add-on package managers, instead of relying on the Anaconda system.  WISDEM&reg; installations have succeeded on Linux ([Ubuntu](https://www.ubuntu.com/), [Fedora](https://getfedora.org), etc), MacOS with [MacPorts](https://www.macports.org) or [Homebrew](https://brew.sh), and Windows with [Cygwin](http://cygwin.com).

1. Obtain the local version of Python3 and the packages listed in Anaconda Step 2 above.  Each package manager will have a different name for these packages.  If your package manager does not have a particular Python package, you can obtain it via `pip install` from the Python Package Index, [PyPI](https://pypi.org/).

2. Continue with Anaconda Steps 3-6


## Run Unit Tests

Each package has its own set of unit tests, some of which are more comprehensive than others.

## Feedback

For software issues please use <https://github.com/WISDEM/WISDEM/issues>.  For functionality and theory related questions and comments please use the NWTC forum for [Systems Engineering Software Questions](https://wind.nrel.gov/forum/wind/viewtopic.php?f=34&t=1002).
