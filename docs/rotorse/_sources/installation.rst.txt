Installation
------------

.. admonition:: Prerequisites
   :class: warning

	General: C compiler, Fortran compiler, NumPy, SciPy, MatlPlotLib, OpenMDAO

	Wind Plant Framework: FUSED-Wind (Framework for Unified Systems Engineering and Design of Wind Plants)

	Sub-Models: CommonSE, CCBlade, Akima, pBeam

	Supporting python packages: Algopy, Sphinx, Xlrd, PyOpt, py2exe, Pyzmq, Sphinxcontrib-bibtex, Numpydoc, Ipython

Clone the repository at `<https://github.com/WISDEM/RotorSE>`_
or download the releases and uncompress/unpack (RotorSE.py-|release|.tar.gz or RotorSE.py-|release|.zip) from the website link at the bottom the `RotorSE site<http://nwtc.nrel.gov/RotorSE>`_.

Install RotorSE with the following command (for windows see the additional instructions in the README.md file).  For full instructions for Windows, see the README.md file.

.. code-block:: bash

   $ python setup.py install

To check if installation was successful try to import the module from within an activated OpenMDAO environment:

.. code-block:: bash

    $ python

.. code-block:: python

    > import rotorse.rotor

or run the unit tests for the gradient checks:

.. code-block:: bash

   $ python src/towerse/test/test_rotor_gradients.py
   $ python src/towerse/test/test_rotoraero_gradients.py

An "OK" signifies that all the tests passed.

.. only:: latex

    An HTML version of this documentation that contains further details and links to the source code is available at `<http://wisdem.github.io/RotorSE>`_
