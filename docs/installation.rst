WISDEM Installation
-------------------

The recommended method for installing WISDEM is with `Anaconda <https://www.anaconda.com>`_.  This streamlines the installation of dependencies and creates self-contained environments suitable for testing and analysis.  WISDEM requires `Anaconda 64-bit <https://www.anaconda.com/distribution/>`_.

Configure Anaconda Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Setup and activate the Anaconda environment from a prompt (Anaconda3 Power Shell on Windows or Terminal.app on Mac)

The installation instructions below use the environment name, "wisdem-env," but any name is acceptable.

.. code-block:: bash

    conda config --add channels conda-forge
    conda create -y --name wisdem-env python=3.7
    conda activate wisdem-env

Note that older versions of anaconda on MacOS and Linux may instead require `source activate wisdem-env`

.. _install_as_user:
Install WISDEM as a User
^^^^^^^^^^^^^^^^^^^^^^^^

From an active Anaconda environment, install WISDEM and its dependencies with:

.. code-block:: bash

    conda install -y wisdem

Running WISDEM Tutorials
^^^^^^^^^^^^^^^^^^^^^^^^

To open up the WISDEM tutorials, you will have to clone the WISDEM source code.  First navigate to a directory where you want to place WISDEM and all of its files and then run with:

.. code-block:: bash

    conda install -y git jupyter
    git clone https://github.com/WISDEM/WISDEM.git
    cd WISDEM/tutorial-notebooks
    jupyter notebook

.. _install_as_developer:
Install WISDEM as a Developer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you wish to edit source files and/or contribute to WISDEM, you will need to install as a developer.  WISDEM is first installed with Anaconda to install all dependencies, but then reinstall WISDEM from source.  Note the differences between Windows and Mac/Linux build systems.

.. code-block:: bash

    conda install -y wisdem git jupyter
    conda remove --force wisdem

For Mac / Linux systems:

.. code-block:: bash

    conda install compilers

For Windows systems:

.. code-block:: bash

    conda install m2w64-toolchain libpython

Finally, for all systems:

.. code-block:: bash

    git clone https://github.com/WISDEM/WISDEM.git
    cd WISDEM
    python setup.py develop

Install pyOptSparse (`Optional`)
""""""""""""""""""""""""""""""""

`pyOptSparse <https://github.com/mdolab/pyoptsparse>`_ is a package that provides additional optimization solvers with OpenMDAO support:

.. code-block:: bash

    git clone https://github.com/evan-gaertner/pyoptsparse.git
    cd pyoptsparse
    python setup.py install
    cd ..