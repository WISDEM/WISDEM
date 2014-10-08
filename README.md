The Wind-Plant Integrated System Design and Engineering Model (WISDEM) is a set of models for assessing overall wind plant cost of energy (coe).  The models use wind turbine and plant cost and energy production as well as financial models to estimate coe and other wind plant system attributes.  It is built in OpenMDAO and uses several sub-models that are also designed as OpenMDAO plugin-ins.  These sub-models can be used independently but they are required to use the overall WISDEM capability.  Please install all of the pre-requisites prior to installing WISDEM.  For additional information about the NWTC effort in systems engineering that supports WISDEM development, please visit the official [NREL systems engineering for wind energy website](http://www.nrel.gov/wind/systems_engineering/).

Authors: [NREL WISDEM Team](mailto:nrel.wisdem+wisdem@gmail.com)
K. Dykes, S. A. Ning, P. Graf, G. Scott, Y. Guo, R. King, T. Parsons, R. Damiani, P. Fleming

## Version

This software is a beta version 0.1.0.

## Detailed Documentation

For detailed documentation see <http://wisdem.github.io/WISDEM/>

## Prerequisites

General: C compiler, Fortran compiler, NumPy, SciPy, Swig, pyWin32, MatlPlotLib, Lxml, OpenMDAO

## Dependencies installed with WISDEM using by setup_all.py (see installation below):

Wind Plant Framework: [FUSED-Wind](http://fusedwind.org) (Framework for Unified Systems Engineering and Design of Wind Plants)

Sub-Models: CommonSE, AeroelasticSE, RotorSE, DriveSE, DriveWPACT, TowerSE, JacketSE, Turbine_CostsSE, Plant_CostsSE, Plant_EnergySE, Plant_FinanceSE, pBEAM, CCBlade, Akima

Supporting python packages: Pandas, Algopy, Zope.interface, Sphinx, Xlrd, PyOpt, py2exe, Pyzmq, Sphinxcontrib-bibtex, Sphinxcontrib-zopeext, Numpydoc, Ipython

## Installation (Windows)

First, clone the [repository](https://github.com/WISDEM/WISDEM)
or download the releases and uncompress/unpack (WISDEM.py-|release|.tar.gz or WISDEM.py-|release|.zip) from the website link at the bottom the [WISDEM site](http://nwtc.nrel.gov/WISDEM).

These instructions assume you are using MinGW and have already installed gcc and g++.
Also you should already have successfully installed Python (for a [single user only](http://bugs.python.org/issue5459#msg101098)), NumPy, and setuptools.
The example directories may need to be modified depending on where you installed things.  See this [Windows guideline set](https://nwtc.nrel.gov/system/files/Windows%20OpenMDAO%20Install%20Tips.pdf) for additional support on installing python.

1) Edit (or create) a distutils config 'distutils.cfg' file in your Python Lib directory or in your openmdao Lib directory if working from an activated openmdao environment.

    C:\Python27\Lib\distutils\distutils.cfg

or

    "Path to openmdao"\Lib\distutils.cfg

and put the following in it:

    [build]
    compiler=mingw32


2) Download [Boost](http://www.boost.org) (v 1.55 as of this writing) and setup bjam

At the command prompt:

    > cd boost_1_55_0\tools\build\v2\engine
    > build.bat mingw

This should create a folder called: bin.ntx86.  For convenience in the next step you can add this folder to your PATH so bjam is accessible.  Otherwise, use the whole path when calling bjam.

    C:\boost_1_55_0\tools\build\v2\engine\bin.ntx86

3) Download [Boost](http://www.boost.org) (v 1.55 as of this writing) and setup bjam:

In the boost root directory (must be in the root directory) type the following at the command prompt:

    > bjam toolset=gcc --with-python link=shared

the libraries should be built in stage/lib and will be needed in steps 5 and 6.

4) Install LAPACK and BLAS.  I just used [prebuilt libraries](http://icl.cs.utk.edu/lapack-for-windows/lapack/#libraries).  Make sure to grab all three libraries - BLAS, LAPACK and LAPACKE and make sure they are the 32-bit versions.
Remember the location for steps 5 and 6.

5) Make sure the following are on your system PATH.  The dynamic libraries are needed in order to actually run pBEAM.

    C:\Python27  (for Python)
    C:\Python27\Scripts  (for easy_install)
    C:\MinGW\bin  (for g++, gcc, etc.)
    C:\lapack  (LAPACK dynamic libraries)
    C:\boost_1_55_0\stage\lib  (Boost Python dynamic libraries)

For the remainder of the setup, use the below directions for *nix systems.  If you have issues with installation of pBEAM and RotorSE, then do this additional step:

6) Modify the 'setup.py' and script in the pBEAM and RotorSE main directories.  Unlike GCC on *nix systems, Windows does not have typical locations to store headers and libraries (e.g., /usr/local/include) and so you will need manually specify them.  Add the header locations for Boost in the include_dirs.  Add the library locations for Boost and LAPACK.  You may also need to rename the boost_python library.  Use the example below, modifying as needed based on where you installed things.  Note that setup.py expects unix style slashes (forward), and that you do not need to include 'lib' at the front of the library names (i.e., 'lapack' corresponds to 'liblapack.dll' or 'liblapack.a').  Note: make sure your boost version matches the boost version installed (i.e. mgw48, mgw46, etc).

    include_dirs=[join(path, 'pBEAM'), 'C:/boost_1_55_0'],
    library_dirs=['C:/boost_1_55_0/stage/lib', 'C:/lapack'],
    libraries=['boost_python-mgw48-mt-1_55', 'lapack']


## Installation (OS X, Linux)

If you want to install WISDEM with all its underlying dependencies, then use the following command from within an [activated OpenMDAO](http://openmdao.org/docs/getting-started/install.html) environment:

    $ python setup_all.py

If all dependencies were already installed separately and you just want to install the WISDEM plugin, then install WISDEM with the following command from within an activated OpenMDAO environment:

    $ plugin install

## Run Unit Tests

To check if installation was successful try to import the module

	$ python
	> import wisdem.lcoe.lcoe_csm_assembly
	> import wisdem.lcoe.lcoe_se_assembly
	> import wisdem.turbinese.turbine

You may also run unit tests.

	$ python src/test/test_turbine_gradients.py

For software issues please use <https://github.com/WISDEM/WISDEM/issues>.  For functionality and theory related questions and comments please use the NWTC forum for [Systems Engineering Software Questions](https://wind.nrel.gov/forum/wind/viewtopic.php?f=34&t=1002).