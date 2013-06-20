Installation
------------

.. admonition:: Prerequisites
   :class: warning

   NumPy


Download either AirfoilPrep.py-|release|.tar.gz or AirfoilPrep.py-|release|.zip and uncompress/unpack it.

If you are only going to use AirfoilPrep.py from the :ref:`command-line <command-line-usage-label>` for simple preprocessing, no installation is necessary.  The ``airfoilprep.py`` file in the ``src`` directory can be copied to any location on your computer and used directly.  For convenience you may want to add the directory it is contained in to the system path.  If you will use AirfoilPrep.py from within Python for more advanced preprocessing or for integration with other codes, AirfoilPrep.py should be installed using:

.. code-block:: bash

   $ python setup.py install

To verify that the installation was successful and to run all the unit tests:

.. code-block:: bash

   $ python test/test_airfoilprep.py

An "OK" signifies that all the tests passed.

.. only:: html

    See :ref:`module documentation <interfaces-label>` for more details on usage from with Python.

.. only:: latex

    See :ref:`module documentation <interfaces-label>` for more details on usage within Python.  To access an HTML version of this documentation with improved formatting and links to the source code, open ``docs/index.html``.