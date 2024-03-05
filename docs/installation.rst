WISDEM Installation
-------------------

Installation with `Anaconda <https://www.anaconda.com>`_ is the recommended approach because of the ability to create self-contained environments suitable for testing and analysis.  WISDEM requires `Anaconda 64-bit <https://www.anaconda.com/distribution/>`_.  However, the `conda` command has begun to show its age and we now recommend the one-for-one replacement with `mamba` via the `Miniforge distribution <https://github.com/conda-forge/miniforge/releases>`_, which is much more lightweight and more easily solves for the WISDEM package dependencies.

Install WISDEM as a "Library"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use WISDEM's modules as a library for incorporation into other scripts or tools, WISDEM is available via `mamba install wisdem` or `pip install wisdem`, assuming that you have already setup your python environment.  Note that on Windows platforms, we suggest using `conda/mamba` exclusively.


Configure Anaconda Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The installation instructions below use the environment name, "wisdem-env," but any name is acceptable.

Setup and activate the Anaconda environment from a prompt (Anaconda3 Power Shell on Windows or Terminal.app on Mac):

.. code-block:: bash

    mamba config --add channels conda-forge
    mamba env create --name wisdem-env -f https://raw.githubusercontent.com/WISDEM/WISDEM/master/environment.yml python=3.10
    mamba activate wisdem-env

Note that any future occasion on which you wish to use WISDEM, you will only have to start with ``mamba activate wisdem-env``.  For those working behind company firewalls, you may have to change the conda authentication with ``conda config --set ssl_verify no``.  Proxy servers can also be set with ``conda config --set proxy_servers.http http://id:pw@address:port`` and ``conda config --set proxy_servers.https https://id:pw@address:port``.  To setup an environment based on a different Github branch of WISDEM, simply substitute the branch name for `master` in the line above.

Install WISDEM for Direct Use
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to directly use the examples in the repository and peek at the code when necessary, we recommend all users install WISDEM in *developer/editable* mode (`pip` has been unreliable for this type of install, so we recommend a slightly outdated approach).  This is done by first installing WISDEM dependencies and then installing WISDEM from the Github source code.  Note the differences between Windows and Mac build systems.  For Linux, we recommend using the native compilers (for example, gcc and gfortran in the default GNU suite).

For Linux and Mac systems:

.. code-block:: bash

    mamba install petsc4py mpi4py

For Mac systems that *are not* using Homebrew or Macports compilers:

.. code-block:: bash

    mamba install gfortran

For Windows systems:

.. code-block:: bash

    mamba install m2w64-toolchain libpython

Finally, for all systems:

.. code-block:: bash

    git clone https://github.com/WISDEM/WISDEM.git
    cd WISDEM
    python setup.py develop

Run Unit Tests
^^^^^^^^^^^^^^

Each package has its own set of unit tests.  These can be run in batch with the `test_all.py` script located in the top level `test`-directory:

.. code-block:: bash

    cd test
    python test_all.py
