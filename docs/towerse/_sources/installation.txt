Installation
------------

.. admonition:: Prerequisites
   :class: warning

	General: C compiler, Fortran compiler, NumPy, SciPy, Swig, pyWin32, MatlPlotLib, Lxml, OpenMDAO

	Wind Plant Framework: FUSED-Wind (Framework for Unified Systems Engineering and Design of Wind Plants)

	Sub-Models: CommonSE, pBEAM

	Supporting python packages: Pandas, Algopy, Zope.interface, Sphinx, Xlrd, PyOpt, py2exe, Pyzmq, Sphinxcontrib-bibtex, Sphinxcontrib-zopeext, Numpydoc, Ipython

Clone the repository at `<https://github.com/WISDEM/TowerSE>`_
or download the releases and uncompress/unpack (TowerSE.py-|release|.tar.gz or TowerSE.py-|release|.zip) from the website link at the bottom the `TowerSE site<http://nwtc.nrel.gov/TowerSE>`_.

Install TowerSE with the following command:

.. code-block:: bash

   $ python setup.py install

or from within an activated OpenMDAO environment:

.. code-block:: bash

   $ plugin install

To check if installation was successful try to import the module from within an activated OpenMDAO environment:

.. code-block:: bash

    $ python

.. code-block:: python

    > import towerse.tower

or run the unit tests for the gradient checks

.. code-block:: bash

   $ python src/towerse/test/test_tower_gradients.py

An "OK" signifies that all the tests passed.

.. only:: latex

    An HTML version of this documentation that contains further details and links to the source code is available at `<http://wisdem.github.io/TowerSE>`_
