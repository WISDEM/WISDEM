Installation
------------

.. admonition:: Prerequisites
   :class: warning

   C compiler, Fortran compiler, NumPy, SciPy

Clone the repository at `<https://github.com/WISDEM/RotorSE>`_ or download the releases and uncompress/unpack
(RotorSE.py-|release|.tar.gz or RotorSE.py-|release|.zip)

Install RotorSE with the following command.

.. code-block:: bash

   $ python setup.py install

To check if installation was successful try to import the module

.. code-block:: bash

    $ python

.. code-block:: python

    > import rotorse.rotor

or run the unit tests for the gradient checks

.. code-block:: bash

   $ python src/towerse/test/test_rotor_gradients.py
   $ python src/towerse/test/test_rotoraero_gradients.py

An "OK" signifies that all the tests passed.

.. only:: latex

    An HTML version of this documentation that contains further details and links to the source code is available at `<http://wisdem.github.io/RotorSE>`_
