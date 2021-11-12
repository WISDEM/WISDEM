WISDEM Installation
-------------------

Installation with `Anaconda <https://www.anaconda.com>`_ is the recommended approach because of the ability to create self-contained environments suitable for testing and analysis.  WISDEM requires `Anaconda 64-bit <https://www.anaconda.com/distribution/>`_.

Configure Anaconda Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The installation instructions below use the environment name, "wisdem-env," but any name is acceptable.

Setup and activate the Anaconda environment from a prompt (Anaconda3 Power Shell on Windows or Terminal.app on Mac):

.. code-block:: bash

    conda config --add channels conda-forge
    conda create -y --name wisdem-env python=3.7
    conda activate wisdem-env

Note that any future occasion on which you wish to use WISDEM, you will only have to start with ``conda activate wisdem-env``.

Install WISDEM
^^^^^^^^^^^^^^

In order to directly use the examples in the repository and peek at the code when necessary, we recommend all users install WISDEM in *developer* mode.  This is done by first installing WISDEM as a conda package to easily satisfy all dependencies, but then removing the WISDEM conda package and reinstalling from the Github source code.  Note the differences between Windows and Mac build systems.  For Linux, we recommend using the native compilers (for example, gcc and gfortran in the default GNU suite).

.. code-block:: bash

    conda install -y wisdem git
    conda remove --force wisdem
    pip install simpy marmot-agents nlopt

For Mac systems:

.. code-block:: bash

    conda install compilers

For Windows systems:

.. code-block:: bash

    conda install m2w64-toolchain libpython

Finally, for all systems:

.. code-block:: bash

    git clone https://github.com/WISDEM/WISDEM.git
    cd WISDEM
    git checkout develop
    pip install -e .

Install pyOptSparse (`Optional`)
""""""""""""""""""""""""""""""""

`pyOptSparse <https://github.com/mdolab/pyoptsparse>`_ is a package that provides additional optimization solvers with OpenMDAO support:

.. code-block:: bash

    git clone https://github.com/evan-gaertner/pyoptsparse.git
    pip install -e pyoptsparse

Run Unit Tests
^^^^^^^^^^^^^^

Each package has its own set of unit tests.  These can be run in batch with the `test_all.py` script located in the top level `test`-directory:

.. code-block:: bash

    cd test
    python test_all.py
