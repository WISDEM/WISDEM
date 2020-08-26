WEIS Installation
-----------------

The recommended method for installing WEIS is with `Anaconda <https://www.anaconda.com>`_.
This streamlines the installation of dependencies and creates self-contained environments suitable for testing and analysis.
WEIS requires `Anaconda 64-bit <https://www.anaconda.com/distribution/>`_.

Configure Anaconda Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Setup and activate the Anaconda environment from a prompt (Anaconda3 Power Shell on Windows or Terminal.app on Mac)

The installation instructions below use the environment name, "weis-env," but any name is acceptable.

.. code-block:: bash

    conda config --add channels conda-forge
    conda create -y --name weis-env python=3.7
    conda activate weis-env

Note that older versions of anaconda on MacOS and Linux may instead require `source activate weis-env`

.. _install:
Install WEIS
^^^^^^^^^^^^

For all systems:

.. code-block:: bash

    git clone https://github.com/WEIS/WEIS.git
    cd WEIS
    python setup.py develop


TODO : add to this as needed