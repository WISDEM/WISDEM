The Wind-Plant Integrated System Design and Engineering Model (WISDEM) is a set of models for assessing overall wind plant cost of energy (coe).  The models use wind turbine and plant cost and energy production as well as financial models to estimate coe and other wind plant system attributes.  It is built in OpenMDAO and uses several sub-models that are also designed as OpenMDAO plugin-ins.  These sub-models can be used independently but they are required to use the overall WISDEM capability.  Please install all of the pre-requisites prior to installing WISDEM.  For additional information about the NWTC effort in systems engineering that supports WISDEM development, please visit the official [NREL systems engineering for wind energy website](http://www.nrel.gov/wind/systems_engineering/).

You can also watch the [overview video of WISDEM capabilities and development](https://nwtc.nrel.gov/system/files/SE%20Webinar%20Oct%208%202014.wmv).

The corresponding slides are available [here](https://nwtc.nrel.gov/system/files/SE%20Webinar%202014-10-08.pdf).

Authors: [NREL WISDEM Team](mailto:nrel.wisdem+wisdem@gmail.com)
K. Dykes, S. A. Ning, P. Graf, G. Scott, Y. Guo, R. King, T. Parsons, R. Damiani, P. Fleming

## Version

This software is a beta version 0.1.1.

## Detailed Documentation

For detailed documentation see <http://wisdem.github.io/WISDEM/>

## Dependencies

* **General Prerequisites**
    * A C compiler and a Fortran compiler
    * [LaPack](http://www.netlib.org/lapack/) and [BLAS](http://www.netlib.org/blas/)
    * [Swig](http://www.swig.org/)
    * [Lxml](http://lxml.de/)
    * [boost-python](http://www.boost.org/doc/libs/1_55_0/libs/python/doc/)
    * [Python 2.7.x](www.python.org) - Python 3 will not work with OpenMDAO and WISDEM
    * [NumPy](http://www.numpy.org/)
    * [SciPy](http://scipy.org/)
    * [MatPlotLib](http://matplotlib.org/)
    * [Pandas](http://pandas.pydata.org/)
    * [git](http://git-scm.com/)
    * [pyWin32](http://docs.activestate.com/activepython/2.6/pywin32/PyWin32.HTML) (For Windows installations only)
* **Supporting python packages**
    * [OpenMDAO](http://openmdao.org/)
    * [Algopy](https://pythonhosted.org/algopy/)
    * [zope.interface](http://docs.zope.org/zope.interface/)
    * [sphinx](http://sphinx-doc.org/)
    * [Xlrd](pypi.python.org/pypi/xlrd)
    * [PyOpt](http://www.pyopt.org/)
    * [py2exe](http://www.py2exe.org/) (For Windows installations only)
    * [Pyzmq](http://zeromq.github.io/pyzmq/)
    * [sphinxcontrib-bibtex](https://sphinxcontrib-bibtex.readthedocs.org/en/latest/)
    * [sphinxcontrib-zopeext](http://pythonhosted.org//sphinxcontrib-zopeext/)
    * [Numpydoc](https://pypi.python.org/pypi/numpydoc)
    * [Ipython](ipython.org)
    * python-dateutil
* **Dependencies installed with WISDEM** by setup_all.py (see installation below):
    * The [FUSED-Wind](http://fusedwind.org) Wind Plant Framework (Framework for Unified Systems Engineering and Design of Wind Plants)
    * [WISDEM Sub-Models](https://github.com/WISDEM/)
        * CommonSE
        * AeroelasticSE
        * RotorSE
        * DriveSE
        * DriveWPACT
        * TowerSE
        * JacketSE
        * Turbine\_CostsSE
        * Plant\_CostsSE
        * Plant\_EnergySE
        * Plant\_FinanceSE
        * pBEAM
        * CCBlade
        * Akima

## Download

WISDEM can be obtained by either cloning the git [repository](https://github.com/WISDEM/WISDEM) (this requires the `git` tool), or by downloading the releases (WISDEM.py-|release|.tar.gz or WISDEM.py-|release|.zip) from the website link at the bottom of the [WISDEM site](http://nwtc.nrel.gov/WISDEM).


## Installation (Windows)

First, clone the 
or 
These instructions assume you are using MinGW and have already installed gcc and g++.
Also you should already have successfully installed Python (for a [single user only](http://bugs.python.org/issue5459#msg101098)), NumPy, and setuptools.
The example directories may need to be modified depending on where you installed things.  See this [Windows guideline set](https://nwtc.nrel.gov/system/files/Windows%20OpenMDAO%20Install%20Tips_04062015.pdf) for additional support on installing python.

1.  Edit (or create) a distutils config 'distutils.cfg' file in your Python Lib directory or in your openmdao Lib directory if working from an activated openmdao environment.

        C:\Python27\Lib\distutils\distutils.cfg

    or

        "Path to openmdao"\Lib\distutils.cfg

    and put the following in it:

        [build]
        compiler=mingw32


2. Download [Boost](http://www.boost.org) (v 1.55 as of this writing) and setup bjam

   At the command prompt:

        > cd boost_1_55_0\tools\build\v2\engine
        > build.bat mingw

    This should create a folder called: bin.ntx86.  For convenience in the next step you can add this folder to your PATH so bjam is accessible.  Otherwise, use the whole path when calling bjam.

        C:\boost_1_55_0\tools\build\v2\engine\bin.ntx86

3.  Download [Boost](http://www.boost.org) (v 1.55 as of this writing) and setup bjam:

    In the boost root directory (must be in the root directory) type the following at the command prompt:

        > bjam toolset=gcc --with-python link=shared

    the libraries should be built in stage/lib and will be needed in steps 5 and 6.

4. Install LAPACK and BLAS.  I just used [prebuilt libraries](http://icl.cs.utk.edu/lapack-for-windows/lapack/#libraries).  Make sure to grab all three libraries - BLAS, LAPACK and LAPACKE and make sure they are the 32-bit versions.
   Remember the location for steps 5 and 6.

5.  Make sure the following are on your system PATH.  The dynamic libraries are needed in order to actually run pBEAM.

        C:\Python27  (for Python)
        C:\Python27\Scripts  (for easy_install)
        C:\MinGW\bin  (for g++, gcc, etc.)
        C:\lapack  (LAPACK dynamic libraries)
        C:\boost_1_55_0\stage\lib  (Boost Python dynamic libraries)

    For the remainder of the setup, use the below directions for *nix systems.  If you have issues with installation of pBEAM and RotorSE, then do this additional step:

6. Modify the 'setup.py' and script in the pBEAM and RotorSE main directories.  Unlike GCC on *nix systems, Windows does not have typical locations to store headers and libraries (e.g., /usr/local/include) and so you will need manually specify them.  Add the header locations for Boost in the include_dirs.  Add the library locations for Boost and LAPACK.  You may also need to rename the boost_python library.  Use the example below, modifying as needed based on where you installed things.  Note that setup.py expects unix style slashes (forward), and that you do not need to include 'lib' at the front of the library names (i.e., 'lapack' corresponds to 'liblapack.dll' or 'liblapack.a').  Note: make sure your boost version matches the boost version installed (i.e. mgw48, mgw46, etc).

        include_dirs=[join(path, 'pBEAM'), 'C:/boost_1_55_0'],
        library_dirs=['C:/boost_1_55_0/stage/lib', 'C:/lapack'],
        libraries=['boost_python-mgw48-mt-1_55', 'lapack']

7. Then you are now finally ready to install! Activate your OpenMDAO environment and then navigate to your WISDEM directory (which you have cloned or downloaded and unzipped) and run from the WISDEM root directory:

        python setup_all.py 

    It should return a message that all models were properly installed.  If there are errors (likely with compiled code such as pBEAM) check the MinGW error messages.  You may have to rebuild the python library for mingw for some combinations of python and mingw.  To do this follow these steps:

    Download gendef for your version of mingw from the mingw installer.  In the mysys shell, navigate to the location of your python dll (for python 2.7 it will be python27.dll and likely can be found in the python27 folder).  Then run:
    	
	gendef python27.dll
    	dlltool -D python27.dll -d python27.def -l libpython27.a
    
    Copy libpython27.a to your ./python27/libs directory.



## Installation (Unix/Linux)

1. First make sure you have the *general prerequisites* installed on your system. In particular, you should install compilers, LaPack, BLAS and Boost-Python libraries. For example on a debian-based environment (e.g. Ubuntu) these packages can be installed by executing the following (as root):

    $ apt-get install gfortran g++ liblapack3 liblapack-dev libblas3 libblas-dev libboost-all-dev swig python-lxml python-matplotlib python-scipy python-numpy 

2. Install OpenMDAO using the most recent 'go-openmdao' script.

    a. Download the script for version 10.3.2 from <http://openmdao.org/releases/0.10.3.2/go-openmdao-0.10.3.2.py> and place it in a directory in which you want OpenMDAO to run from.

    b. Execute the script, e.g.:

        $ ./go-openmdao-0.xx.x.py

    This will create a new directory, `./openmdao-0.xx.x/`, and install several python tools into it, then it will install openmdao into that directory.

    c. Now activate the OpenMDAO virtual environment by:

        $ cd openmdao-0.xx.x
        $ . bin/activate
        
    You are now operating in an [OpenMDAO Virtual environment](http://openmdao.org/docs/getting-started/install.html) (note the `(openmdao-0.xx.x)` at the beginning of your prompt).  All Python software installed from within this environment will be local to the environment; that is it will not be accessible without being in this virtual environment (see the documentation on [Python Virtual Environments](http://docs.python-guide.org/en/latest/dev/virtualenvs/) for more information).
    
3. You are now ready to install WISDEM into this *activated OpenMDAO virtual environment*.

    a. If you haven't already downloaded WISDEM, `cd` to a directory where you want WISDEM installed then clone the [repository](https://github.com/WISDEM/WISDEM) into this directory:
    
        $ git clone http://github.com/WISDEM/WISDEM

    This will create a `./WISDEM/` directory.

    b. Now install WISDEM,

        $ cd WISDEM
        $ python setup_all.py

If you complete all of these steps without error, then you have successfully installed WISDEM!  Consider running the tests to confirm that things are working as expected.


## Installation (OS X)

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
