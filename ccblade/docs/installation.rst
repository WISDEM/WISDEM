Installation
------------

.. admonition:: Prerequisites
   :class: warning

   C compiler, Fortran compiler, NumPy, SciPy

Download either CCBlade.py-|release|.tar.gz or CCBlade.py-|release|.zip and uncompress/unpack it.

Install CCBlade with the following command.

.. code-block:: bash

   $ python setup.py install

To check if installation was successful run the unit tests for the NREL 5-MW model

.. code-block:: bash

   $ python test/test_ccblade.py

An "OK" signifies that all the tests passed.

.. only:: latex

    To access an HTML version of this documentation that contains further details and links to the source code, open docs/index.html.



.. currentmodule:: airfoilprep

.. note::

    The CCBlade installation also installs the module :ref:`AirfoilPrep.py <airfoilprep:airfoilpreppy-label>`.  Although it is not necessary to use AirfoilPrep.py with CCBlade, its inclusion is convenient when working with AeroDyn input files or doing any aerodynamic preprocessing of airfoil data.

