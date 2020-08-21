# NREL's Reference OpenSource Controller (ROSCO) for wind turbine applications
NREL's Reference OpenSource Controller (ROSCO) for wind turbine applications uses the Bladed-style DISCON interface used by OpenFAST, Bladed (versions 4.5 or earlier), HAWC2, and more.

## Introduction
The NREL Reference OpenSource Controller (ROSCO) provides an open, modular and fully adaptable baseline wind turbine controller to the scientific community. Because of the open character and modular set-up, scientists are able to collaborate and contribute in making continuous improvements to the code. New control implementations can be added to the existing baseline controller, and in this way, convenient assessments of the proposed algorithms is possible. ROSCO is being developed in Fortran and uses the Bladed-style DISCON controller interface. The compiled controller is configured by a single control settings parameter file, and can work with any wind turbine model and simulation software using the DISCON interface. Baseline parameter files are supplied for the NREL 5-MW and DTU 10-MW reference wind turbines.

## Compiling ROSCO
Compiling ROSCO is made simple using [cmake](https://cmake.org/) on Unix based machines. Currently (November 2019), we recommend that you compile the code using a unix machine (or subsystem on Windows). Some compiled binaries are also provided via github.

### Required Software to build ROSCO
* Fortran compiler (GNU compiler version above 4.6.0 or Intel compiler version above 11)
* C/C++ compiler
* GNU Make (version 3.81 or later)
* CMake (version 2.8.12 or later) - for unix

### Steps to compile
First, clone the git repository:
``` 
git clone https://github.com/NREL/ROSCO.git
```
Second, you will need to compile the controller. From the ROSCO home directory:
```
mkdir build
cd build
cmake ..
make
```
A dynamic link library will be compiled into the directory with the title `libdiscon.*`, where the file extension is `.so`, `.dll`, or `.dylib`, depending on the user's operating system.  

## Running ROSCO
A few files are needed to run ROSCO. Of course, the compiled binary, named `libdiscon.*` by default, is needed. This should be appropriately pointed to by your ServoDyn input file (for OpenFAST). In addition to the binary, a controller input file title DISCON.IN is necessary. Two example input files are provided for the NREL 5MW and DTU 10MW wind turbine controllers in the [parameter_files](parameter_files) folder. Note that DISCON.IN (or similar) is pointed to by the `DLL_InFile` parameter in ServoDyn. For generic controller tuning methods, and an automated writing of this DISCON.IN file, we point you to the complete [ROSCO_toolbox](https://github.com/nrel/rosco_toolbox). 

In addition to DISCON.IN, if you wish to use either of the wind speed estimators offered by ROSCO (`WE_mode > 0`), you will need a rotor performance input file. This is, again, provided for the NREL 5MW and DTU 10MW wind turbines, and can be easily made for other turbines using the [ROSCO_toolbox](https://github.com/nrel/rosco_toolbox). The input `PerfFileName` in DISCON.IN points to this file. 

## Using ROSCO for Bladed
If you want to use the controller with DNV GL Bladed v4.5 or earlier (which still has support for the DISCON external controller interface), do the following:
1. Be sure to use and place the 32-bit DLL in the same folder as where you put your project .$PJ-file
2. Copy in that same folder the DISCON.IN controller configuration file
3. Set-up the 32-bit DLL as an external controller (Control -> Discrete External Controller -> Define...)
3. Open the DISCON.IN file with a text editor and copy its entire contents in the "External controller data:" section (Control -> Discrete External Controller -> Define...)
4. Run a "Power Production" simulation

## Referencing
If ROSCO played a role in your research, please cite it. This software can be
cited as:

   ROSCO. Version 1.0.0 (2020). Available at https://github.com/nrel/rosco.

For LaTeX users:

```
    @misc{ROSCO_2020,
    author = {NREL},
    title = {{ROSCO. Version 1.0.0}},
    year = {2020},
    publisher = {GitHub},
    journal = {GitHub repository},
    url = {https://github.com/NREL/rosco}
    }
```
If you have been using the entirety of the [ROSCO toolbox](https://github.com/nrel/rosco_toolbox), please see the ROSCO toolbox README for information on how to cite it.


## Acknowledgments 
The initial release of this controller was the Delft Research Controller. This work should be cited as
* Mulders, S.P. and van Wingerden, J.W. "Delft Research Controller: an open-source and community-driven wind turbine baseline controller." Journal of Physics: Conference Series. Vol. 1037. No. 3. IOP Publishing, 2018. [Link to the paper](https://iopscience.iop.org/article/10.1088/1742-6596/1037/3/032009/meta)
The Delft Research Controller was the initial version of this work. It has since been grown significantly and become NREL's ROSCO. 

Primary contributions to ROSCO have been provided by researchers the National Renewable Energy Laboratory (Nikhar J. Abbas, Alan Wright, and Paul Fleming), Delft University of Technology (Sebastiaan Mulders and Jan-Willem van Wingerden), and the University of Colorado Boulder (Lucy Pao). Much of the intellect behind these contributions has been inspired or derived from an extensive amount of work in the literature. The bulk of this has been cited through the primary publications about this work. 

There are also some specific acknowledgements we would like to communicate:
* The setpoint smoothing regime implemented through the ROSCO controller was contributed by sowento GmbH. 