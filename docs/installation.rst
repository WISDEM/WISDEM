WISDEM Installation
===================

Installation with `Anaconda <https://www.anaconda.com>`_ is the recommended approach because of the ability to create self-contained environments suitable for testing and analysis.  WISDEM requires `Anaconda 64-bit <https://www.anaconda.com/distribution/>`_.  However, the `conda` command has begun to show its age and we now recommend the one-for-one replacement with the `Miniforge3 distribution <https://github.com/conda-forge/miniforge?tab=readme-ov-file#miniforge3>`_, which is much more lightweight and more easily solves for the WISDEM package dependencies.


Install WISDEM as a "Library"
-----------------------------

1. Create a conda environment with your preferred name (``wisdem-env`` in the following example) and favorite, approved Python version:

    .. code-block:: bash

        conda create -n wisdem-env python=3.13 -y

2. Activate the environment:

    .. code-block:: bash

        conda activate wisdem-env

3. Install WISDEM via a ``conda`` or ``pip``. We highly recommend via conda.

    .. code-block:: bash

        conda install wisdem

    or

    .. code-block:: bash

        pip install wisdem


To use WISDEM's modules as a library for incorporation into other scripts or tools, WISDEM is available via ``conda install wisdem`` or ``pip install wisdem``, assuming that you have already setup your python environment.  Note that on Windows platforms, we suggest using ``conda`` exclusively.

Install For Direct Use or Development
-------------------------------------

These instructions are for interaction with WISDEM directly, the use of its examples, and the direct inspection of its source code.

The installation instructions below use the environment name, "wisdem-env," but any name is acceptable. Below are a series of considerations:

* For those working behind company firewalls, you may have to change the conda authentication with ``conda config --set ssl_verify no``.
* Proxy servers can also be set with ``conda config --set proxy_servers.http http://id:pw@address:port`` and ``conda config --set proxy_servers.https https://id:pw@address:port``.
* To setup an environment based on a different Github branch of WISDEM, simply substitute the branch name for ``master`` in the setup line.

.. important::
    For Windows users, we recommend installing ``git`` and the ``m2w64`` packages in separate environments as some of the
    libraries appear to conflict such that WISDEM cannot be successfully built from source.  The ``git`` package is best
    installed in the ``base`` environment.

Direct use
^^^^^^^^^^

We still highly recommend users use ``conda install wisdem`` into an environment, but if there is a reason that is not
desired, please use the following instructions.

Setup and activate the Anaconda environment from a prompt (Anaconda3 Power Shell on Windows or Terminal.app on Mac)

.. important::
    In the ``environment.yaml`` please uncomment out the OS-dependent dependencies at the top

1. Install ``git`` if you don't already have it:

    .. code-block:: bash

        conda install git


2. Clone the repository and enter it:

    .. code-block:: bash

        git clone https://github.com/WISDEM/WISDEM.git
        cd WISDEM


3. Checkout the desired branch, if necessary:

    .. code-block:: bash

        git checkout <branch>


4. Create and activate your ``wisdem-env`` environment, substituting "wisdem-env" with a different desired name.

    .. code-block:: bash

        conda env create --name wisdem-env -f environment.yml
        conda activate wisdem-env


5. Install WISDEM

    .. code-block:: bash

        pip install --no-deps . -v


Development
^^^^^^^^^^^

In order to directly use the examples in the repository and peek at the code when necessary, we recommend all users install WISDEM in *developer / editable* mode using the instructions here.  If you really just want to use WISDEM as a library and lean on the documentation, you can always do ``conda install wisdem`` and be done.  Note the differences between Windows and Mac/Linux build systems.

.. important::
    In the ``environment_dev.yaml`` please uncomment out the OS-dependent dependencies at the top

    For Linux, we recommend using the native compilers (for example, gcc and gfortran in the default GNU suite).

Please follow steps 1-3 in the Direct Use section above, replacing steps 4 & 5 (and the additional 6) with the following
to ensure the development dependencies for building, testing, and documentation are also installed:

4. Create and activate your ``wisdem-env`` environment, substituting "wisdem-env" with a different desired name.

    .. code-block:: bash

        conda env create --name wisdem-env -f environment_dev.yml
        conda activate wisdem-env


5. Install WISDEM. Please note the ``-e`` (editable) flag used to ensure your code changes are registered dynamically every
   time you save modifications.

    .. code-block:: bash

        pip install --no-deps -e . -v

6. Register the pre-commit hooks for automatic code linting and formatting.

    .. code-block:: bash

        pre-commit install


Run Unit and Integration Tests
------------------------------

Each package has its own set of unit tests, and the project runs the examples as integration tests.

These can be run in batch with the following command

.. code-block:: bash

    pytest

Users can add either the ``--unit`` or ``--integration`` flags if they would like to skip running
the examples or just run the examples. Otherwise, all tests will be run.

.. note::
    Legacy users can continue to run ``python test/test_all.py`` to run the scipts, though it is recommend to adopt the
    simpler ``pytest`` call. In a future version, ``test_all.py`` will be removed.
