Installation
------------

.. admonition:: Prerequisites
   :class: warning

   C++ compiler, `Boost C++ Libraries <http://www.boost.org>`_: specifically boost_python-mt, boost_system-mt (and boost_unit_test_framework-mt if you want to run the unit tests), LAPACK, NumPy, and SciPy


Download either pBEAM.py-|release|.tar.gz or pBEAM.py-|release|.zip, and uncompress/unpack it.

Install pBEAM with the following command.

.. code-block:: bash

   $ python setup.py install

To verify that the installation was successful, run Python from the command line,

.. code-block:: bash

   $ python

and import the module.  If no errors are issued, the installation was successful.

>>> import _pBEAM

pBEAM has a large range of unit tests, but they are only accessible through C++.  These tests verify the integrity of the underlying C++ code for development purposes.  If you want to run the tests, change the working directory to src/twister/rotorstruc/pBEAM and run

.. code-block:: bash

   $ make test CXX=g++

where the name of your C++ compiler should be inserted in the place of g++.  The script will build the test executable and run all tests.  The phrase "No errors detected" signifies that all the tests passed.

.. only:: latex

    To access an HTML version of this documentation, which contains further details and links to the source code, open docs/index.html.
