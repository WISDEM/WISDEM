Installation
------------

.. admonition:: prerequisites
   :class: warning

	General: NumPy, SciPy, Swig, pyWin32, MatlPlotLib, Lxml, OpenMDAO

	Wind Plant Framework: FUSED-Wind (Framework for Unified Systems Engineering and Design of Wind Plants)

	Supporting python packages: Pandas, Algopy, Zope.interface, Sphinx, Xlrd, PyOpt, py2exe, Pyzmq, Sphinxcontrib-bibtex, Sphinxcontrib-zopeext, Numpydoc, Ipython

Clone the repository at `<https://github.com/WISDEM/GeneratorSE>`_
or download the releases and uncompress/unpack (GeneratorSE.py-|release|.tar.gz or GeneratorSE.py-|release|.zip) from the website link at the bottom the `GeneratorSE site <http://nwtc.nrel.gov/GeneratorSE>`_.

To install GeneratorSE, first activate the OpenMDAO environment and then install with the following command.

.. code-block:: bash

   $ plugin install

To check if installation was successful try to import the module from within an activated OpenMDAO environment:

.. code-block:: bash

    $ python

.. code-block:: python

    > import generatorsE.PMSG_arms
    > import generatorsE.PMSG_disc
    > import generatorse.EESG
    > import generatorse.DFIG
    > import generatorse.SCIG

or run the unit tests which include functional and gradient tests.  Analytic gradients are provided for variables only so warnings will appear for missing gradients on model input parameters; these can be ignored.

.. code-block:: bash

   $ python src/test/test_GeneratorSE.py

An "OK" signifies that all the tests passed.

For software issues please use `<https://github.com/WISDEM/GeneratorSE/issues>`_.  For functionality and theory related questions and comments please use the NWTC forum for `Systems Engineering Software Questions <https://wind.nrel.gov/forum/wind/viewtopic.php?f=34&t=1002>`_.

.. only:: latex

    An HTML version of this documentation that contains further details and links to the source code is available at `<http://wisdem.github.io/GeneratorSE>`_