Installation
------------

.. admonition:: Prerequisites
   :class: warning

   C compiler, Fortran compiler, NumPy, SciPy

Clone the repository from `github <https://github.com/WISDEM/JacketSE>`_ or download the releases and uncompress/unpack
(JacketSE.py-|release|.tar.gz or JacketSE.py-|release|.zip)

Install pre-requisite pyFrame3DD following instructions in the `README.md <https://github.com/WISDEM/pyFrame3DD>`_.

Make sure towerSupplement.py is in the path and adjust the import statement in Utilization.py as needed. 
If TowerSE is installed (`TowerSE README.md <https://github.com/WISDEM/towerSE>`_), there is no need for modifying the source codes.

Install JacketSE with the following command.

.. code-block:: bash

   $ python setup.py install

To check if installation was successful try to import the module

.. code-block:: bash

    $ python

.. code-block:: python

    > import jacketse.jacket

or run the unit tests 

.. code-block:: bash

   $ python src/jacketse/test/test_jacketSE.py

An "OK" signifies that all the tests passed.

.. only:: latex

    An HTML version of this documentation that contains further details and links to the source code is available `here <http://wisdem.github.io/JacketSE>`_
