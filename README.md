# NREL's Reference OpenSource Controller (ROSCO) toolbox for wind turbine applications
NREL's Reference OpenSource Controller (ROSCO) toolbox for wind turbine applications is a toolbox designed to ease controller implementation for the wind turbine researcher. Some primary capabilities include:
* Generic tuning of NREL's ROSCO controller
* Simple 1-DOF turbine simulations for quick controller capability verifications
* Parsing of OpenFAST input and output files

Block diagrams of these capabilities can be seen in [architecture.png](architecture.png).

## Introduction
The NREL Reference OpenSource Controller (ROSCO) provides an open, modular and fully adaptable baseline wind turbine controller to the scientific community. The ROSCO toolbox leverages this architecture and implementation to provide a generic tuning process for the controller. Because of the open character and modular set-up, scientists are able to collaborate and contribute in making continuous improvements to the code for the controller and the toolbox. The ROSCO toolbox is a mostly-python code base with a number of functionalities.

* [ROSCO](https://github.com/NREL/ROSCO) - the fortran source code for the ROSCO controller. 
* [Examples](https://github.com/NREL/ROSCO_toolbox/tree/master/examples) - short working examples of the capabilities of the ROSCO toolbox. 
* [Tune_Cases](https://github.com/NREL/ROSCO_toolbox/tree/master/Tune_Cases) - example generic tuning scripts for a number of open-source reference turbines.
* [Test_Cases](https://github.com/NREL/ROSCO_toolbox/tree/master/Test_Cases) - numerous NREL 5MW bases cases to run for controller updates and comparisons. A "test-suite", if you will...
* [Matlab_Toolbox](https://github.com/NREL/ROSCO_toolbox/tree/master/Matlab_Toolbox) - MATLAB scripts to parse and plot simulation output data.

## Using the ROSCO Toolbox
There is a short (but _hopefully_ sweet) installation and run process for basic controller tuning...

### Installing the complete ROSCO Toolbox
If you would like to use the ROSCO toolbox, you will need to install the ROSCO toolbox. For the tuning (and a few other) capabilities, you will additionally need to install WISDEM. 

If you do not have WISDEM or the ROSCO toolbox installed and would like to install WISDEM, the ROSCO toolbox, and compile the controller on your unix machine please do the following: open your terminal, navigate to the folder of your choosing, and copy and paste the below text into the command line. This code block is broken up in a piece-wise description in the following sections.
```
#  -- Install WISDEM --
conda config --add channels conda-forge
conda create -y --name wisdem-env python=3.7
conda activate wisdem-env
conda install -y wisdem
#  -- Clone and install the ROSCO toolbox --
git clone https://github.com/NREL/ROSCO_toolbox.git
cd ROSCO_toolbox
git submodule init
git submodule update
python setup.py install
# -- Compile the ROSCO controller --
cd ROSCO
mkdir build
cd build
cmake ..
make
```
#### WISDEM Dependencies
The ROSCO toolbox uses two NREL tools that are distributed as a part of the WISDEM packages. AerolasticSE and CCBlade are currently used, with future dependencies or support possible. As such, it is necessary to install WISDEM. This can be done fairly easily by following the [WISDEM installation instructions](https://github.com/wisdem/wisdem). A brief overview of the _user_ steps is provided here, for more detail, especially if you would like to contribute to the development of WISDEM, see the [WISDEM github page](https://github.com/wisdem/wisdem).

We recommend creating and activating the same Anaconda environment for both WISDEM and ROSCO:

1. Setup and active this environment form the command prompt (Anaconda3 Power Shell on Windows or Terminal.app on Mac)
```
conda config --add channels conda-forge
conda create -y --name wisdem-env python=3.7
conda activate wisdem-env
```
2. Install WISDEM and it's dependencies
``` 
conda install -y wisdem
```

If you wish to use the generic controller tuning capabilities for distributed aerodynamic control that are available in the ROSCO toolbox, you will need to install the `IEAontology4all` branch of WISDEM. For this, please follow the "for developers" instructions on downloading and compiling WISDEM, and be sure to `git checkout IEAontology4all` before `python setup.py develop`.

#### Installing ROSCO
You should first be sure that you are still in the `wisdem-env` environment that you installed wisdem in.

If you would like to take a deeper dive into the source-code, see example tuning cases, or contribute to the toolbox, you should:

1. clone the git repository and initiate the ROSCO submodule:
``` 
git clone https://github.com/NREL/ROSCO_toolbox.git
cd ROSCO_toolbox
git submodule init
git submodule update
```
2. Install ROSCO from the cloned home directory
```
python setup.py develop
```
or
```
pip install -e ROSCO_toolbox
```

#### Compiling ROSCO
The controller itself is installed as a submodule in the ROSCO toolbox. For further information on compiling and running ROSCO itself, especially if you are on a Windows machine, we point you to the [ROSCO github page](https://github.com/NREL/ROSCO_toolbox.git). For Unix systems, (or Unix shells on Windows), cmake makes it easy to compile. In order to compile the controller, you should run the following commands from the ROSCO_toolbox folder.
```
cd ROSCO
mkdir build
cd build
cmake ..
make
```
These commands will compile a binary titled `libdiscon.*` in the build folder, which is the binary necessary run the controller. This should only need to be compiled once. The extension should be `.dll`, `.so`, or `.dylib`, depending on the user operating system. 

### Running ROSCO with Generic Tuning
The [Tune_Cases](Tune_Cases) folder hosts examples on what needs to happen to write the input file to the ROSCO controller. See below on some details for compiling ROSCO:

#### ROSCO Toolbox Generic Tuning
IF you would like to run the generic tuning process for ROSCO, examples are shown in the [Tune_Cases](Tune_Cases) folder. When you run your own version of [tune_ROSCO.py](Tune_Cases/tune_ROSCO.py), you will have two files that are necessary to run the controller. 
1. `DISCON.IN` (or similar) - the input file to `libdiscon.*`. When running the controller in OpenFAST, `DISCON.IN` must be appropriately pointed to by the `DLL_FileName` parameter in ServoDyn. 
2. `Cp_Cq_Ct.txt` (or similar) - This file contains rotor performance tables that are necessary to run the wind speed estimators in ROSCO. This can live wherever you desire, just be sure to point to it properly with the `PerfFileName` parameter in `DISCON.IN`.

### Updating ROSCO Toolbox
Simple git commands should update the toolbox and controller as development continues:
```
git pull
git submodule update 
```
and then recompile and reinstall as necessary...

## Referencing
If the ROSCO Toolbox played a role in your research, please cite it. This software can be
cited as:

   ROSCO. Version 1.0.0 (2020). Available at https://github.com/nrel/rosco_toolbox.

For LaTeX users:

```
@misc{ROSCO_toolbox_2019,
    author = {NREL},
    title = {{ROSCO Toolbox. Version 0.1.0}},
    year = {2019},
    publisher = {GitHub},
    journal = {GitHub repository},
    url = {https://github.com/NREL/rosco_toolbox}
    }
```
If the ROSCO generic tuning theory and implementation played a roll in your research, please cite the following paper
```
@inproceedings{Abbas_WindTech2019,
	doi = {10.1088/1742-6596/1452/1/012002},
	url = {https://doi.org/10.1088%2F1742-6596%2F1452%2F1%2F012002},
	year = 2020,
	month = {jan},
	publisher = {{IOP} Publishing},
	volume = {1452},
	pages = {012002},
	author = {Nikhar J. Abbas and Alan Wright and Lucy Pao},
	title = {An Update to the National Renewable Energy Laboratory Baseline Wind Turbine Controller},
	journal = {Journal of Physics: Conference Series}
}
```
Additionally, if you have extensively used the [ROSCO](https://github.com/NREL/ROSCO) controller or [WISDEM](https://github.com/wisdem/wisdem), please cite them accordingly. 


## Additional Contributors and Acknowledgments
Primary contributions to the ROSCO Toolbox have been provided by researchers the National Renewable Energy Laboratory (Nikhar J. Abbas, Alan Wright, and Paul Fleming) and the University of Colorado Boulder (Lucy Pao). Much of the intellect behind these contributions has been inspired or derived from an extensive amount of work in the literature. The bulk of this has been cited through the primary publications about this work. 

There have been a number of contributors to the logic of the ROSCO controller itself. Please see the [ROSCO github page](https://github.com/NREL/ROSCO) for more information on who these contributors have been. 