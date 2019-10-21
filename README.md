# WISDEM&reg;

The Wind-Plant Integrated System Design and Engineering Model (WISDEM&reg;) is a set of models for assessing overall wind plant cost of energy (COE).  The models use wind turbine and plant cost and energy production as well as financial models to estimate coe and other wind plant system attributes.  It is built in OpenMDAO and uses several sub-models that are also designed as OpenMDAO plugin-ins.  These sub-models can be used independently but they are required to use the overall WISDEM&reg; capability.  Please install all of the pre-requisites prior to installing WISDEM&reg;.  For additional information about the NWTC effort in systems engineering that supports WISDEM&reg; development, please visit the official [NREL systems engineering for wind energy website](https://www.nrel.gov/wind/systems-engineering.html).

Author: [NREL WISDEM Team](mailto:systems.engineering@nrel.gov) 

## Version

This software is a version 2.0.1.

## Documentation

See local documentation in the `docs`-directory or access the online version at <http://wisdem.github.io/WISDEM/>

## Packages

WISDEM&reg; is a family of modules.  The core modules are:

* _AeroelasticSE_ provides multi-fidelity capability for rotor analysis by calling [OpenFAST]<https://github.com/OpenFAST/openfast>
* _CommonSE_ includes several libraries shared among modules
* _DrivetrainSE_ sizes the drivetrain and generator systems (formerly DriveSE and GeneratorSE)
* _FloatingSE_ works with the floating platforms
* _OffshoreBOS_ sizes the balance of systems for offshore plants
* _Plant_FinanceSE_ runs the financial analysis of a wind plant
* _RotorSE_ is a tool for rotor design
* _TowerSE_ is a tool for tower (and monopile) design
* _Turbine_CostsSE_ is a turbine cost model
* _NREL CSM_ is the old cost-and-scaling model
* _WISDEM_ provides the interface between models

The core modules draw upon some utility packages, which are typically compiled code with python wrappers:

* _Airfoil Preppy_ is a tool to handle airfoil polar data
* _CCBlade_ is the BEM module of WISDEM
* _pBEAM_ provides a basic beam model
* _pyFrame3DD_ brings libraries to handle various coordinate transformations
* _pyMAP_ provides a python interface to MAP++, a quasi-static mooring line model
* _pyoptsparse_ provides some additional optimization algorithms to OpenMDAO


## Installation

Installation with [Anaconda](https://www.anaconda.com) is the recommended approach because of the ability to create self-contained environments suitable for testing and analysis.  WISDEM&reg; requires [Anaconda 64-bit](https://www.anaconda.com/distribution/).

The installation instructions below use the environment name, "wisdem-env," but any name is acceptable.

1.  Setup and activate the Anaconda environment from a prompt (Anaconda3 Power Shell on Windows or Terminal.app on Mac)

        conda config --add channels conda-forge
        conda create -y --name wisdem-env python=3.7
        conda activate wisdem-env
    
    Note that older versions of anaconda on MacOS and Linux may require `source activate wisdem-env`

2.  FOR USERS (NOT DEVELOPERS): Install WISDEM and its dependencies

        conda install -y wisdem

3.  To open up the WISDEM tutorials, navigate to a directory where you want to place WISDEM and all of its files.

        conda install -y git jupyter
        git clone https://github.com/WISDEM/WISDEM.git
	cd WISDEM/tutorial-notebooks
	jupyter notebook
	
2.  FOR DEVELOPERS (NOT USERS): Use conda to install the build dependencies, but then install WISDEM from source

        conda install -y wisdem git jupyter
        conda remove --force wisdem
        git clone https://github.com/WISDEM/WISDEM.git
        cd WISDEM
        python setup.py develop	


4. OPTIONAL: Install pyOptSparse, an package that provides a handful of additional optimization solvers and has OpenMDAO support:

        git clone https://github.com/evan-gaertner/pyoptsparse.git
        cd pyoptsparse
        python setup.py install
        cd ..


## Run Unit Tests

Each package has its own set of unit tests, some of which are more comprehensive than others.

## Feedback

For software issues please use <https://github.com/WISDEM/WISDEM/issues>.  For functionality and theory related questions and comments please use the NWTC forum for [Systems Engineering Software Questions](https://wind.nrel.gov/forum/wind/viewtopic.php?f=34&t=1002).
