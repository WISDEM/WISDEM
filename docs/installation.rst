Installation
------------

.. admonition:: prerequisites
   :class: warning

	General: C compiler, Fortran compiler, NumPy, SciPy, Swig, pyWin32, MatlPlotLib, Lxml, OpenMDAO

	Wind Plant Framework: FUSED-Wind (Framework for Unified Systems Engineering and Design of Wind Plants)

	Sub-Models: CommonSE, AeroelasticSE, RotorSE, DriveSE, DriveWPACT, TowerSE, JacketSE, Turbine_CostsSE, Plant_CostsSE, Plant_EnergySE, Plant_FinanceSEE, pBEAM, CCBlade, Akima

	Supporting python packages: Pandas, Algopy, Zope.interface, Sphinx, Xlrd, PyOpt, py2exe, Pyzmq, Sphinxcontrib-bibtex, Sphinxcontrib-zopeext, Numpydoc, Ipython

Clone the repository at `<https://github.com/WISDEM/WISDEM>`_
or download the releases and uncompress/unpack (WISDEM.py-|release|.tar.gz or WISDEM.py-|release|.zip) from the website link at the bottom the `WISDEM site<http://nwtc.nrel.gov/WISDEM>`_.

To install WISDEM, first activate the OpenMDAO environment and then install with the following command.  For full instructions for Windows, see the README.md file.

.. code-block:: bash

   $ python setup_all.py

To check if installation was successful try to import the module from within an activated OpenMDAO environment:

.. code-block:: bash

    $ python

.. code-block:: python

	> import wisdem.lcoe.lcoe_csm_assembly
	> import wisdem.lcoe.lcoe_se_csm_assembly
	> import wisdem.turbinese.turbine

or run the unit tests which include functional and gradient tests.  Analytic gradients are provided for variables only so warnings will appear for missing gradients on model input parameters; these can be ignored.

.. code-block:: bash

   $ python src/test/test_WISDEM.py
   $ python src/test/test_turbine_gradients.py

An "OK" signifies that all the tests passed.

For software issues please use `<https://github.com/WISDEM/WISDEM/issues>`_.  For functionality and theory related questions and comments please use the NWTC forum for `Systems Engineering Software Questions <https://wind.nrel.gov/forum/wind/viewtopic.php?f=34&t=1002>`_.

.. only:: latex

    An HTML version of this documentation that contains further details and links to the source code is available at `<http://wisdem.github.io/WISDEM>`_

