Installation
------------

.. admonition:: Prerequisites
   :class: warning

   Fortran compiler, NumPy, SciPy, zope.interface

Download either CCBlade.py-|release|.tar.gz or CCBlade.py-|release|.zip and uncompress/unpack it.

Install CCBlade with the following command.

.. code-block:: bash

   $ python setup.py install

To check if installation was successful run the unit tests for the NREL 5-MW model

.. code-block:: bash

   $ python test/test_ccblade.py

Additional tests for the gradients are available at:

.. code-block:: bash

   $ python test/test_gradients.py

An "OK" signifies that all the tests passed.

.. only:: latex

    To access an HTML version of this documentation that contains further details and links to the source code, open docs/index.html.

A current copy of the documentation for this code is also available online at http://nrel-wisdem.github.io/CCBlade


.. note::

    The CCBlade installation also installs the module `AirfoilPrep.py <https://github.com/NREL-WISDEM/AirfoilPreppy>`_.  Although it is not strictly necessary to use AirfoilPrep.py with CCBlade, its inclusion is convenient when working with AeroDyn input files or doing any aerodynamic preprocessing of airfoil data.  If you wish to do more with AirfoilPrep.py please see its documentation `here <http://nrel-wisdem.github.io/AirfoilPreppy>`_.

