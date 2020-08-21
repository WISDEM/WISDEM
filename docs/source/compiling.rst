.. _compiling:

Compiling ROSCO
=================
The standard ROSCO controller is based in Fortran and must be compiled. This code can be found at: https://github.com/NREL/ROSCO. Of course, the advanced user can compile the downloaded code using their own desired methods (e.g. Visual Studio). Otherwise, a few of the more common compiling methods are detailed on this page. Additionally, the most recent tagged version releases are `available for download <https://github.com/NREL/ROSCO/tags>`_. 

If one wishes to download the code via the command line, we provide two supported options. For non-developers (those not interested in modifying the source code), the controller can be downloaded via Anaconda. For developers, CMake can be used to compile the Fortran code. Using CMake is generally a simple and straightforward process for Mac/Linux users, however, the process of compiling using CMake and MinGW is possible on Windows, but can produce complications if not done carefully. 

Anaconda for non-developers:
------------------------------------
For users familiar with Anaconda_, ROSCO is available through the conda-forge channel. In order to download the compiled, most recent version release, from an anaconda powershell (Windows) or terminal (Mac/Linux) window, one can create a new anaconda virtual environment: 
::

    conda config --add channels conda-forge
    conda create -y --name ROSCO-env
    conda activate ROSCO-env

navigate to your desired folder to save the compiled binary using:
::

    cd <my_desired_folder>
    
and download the controller:
::

    conda install -y ROSCO

This will download a compiled ROSCO binary file into the default filepath for any dynamic libraries downloaded via anaconda while in the ROSCO-env. This can be copied to your desired folder using:
::

    cp $CONDA_PREFIX/lib/libdiscon.* .

CMake for developers:
-------------------------------
CMake_ provides a straightforward option for many users, particularly those on a Mac or Linux. ROSCO can be compiled by first cloning the source code from git using:
::

    git clone https://github.com/NREL/ROSCO.git

And then compiling using CMake:
::

    cd ROSCO
    mkdir build
    cd build
    cmake ..
    make

This will generate a file titled :code:`libdiscon.*` in the current directory. 

.. _Anaconda: https://www.anaconda.com/
.. _CMake: https://cmake.org/