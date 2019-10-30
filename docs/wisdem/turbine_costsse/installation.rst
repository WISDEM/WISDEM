Installation
------------

.. admonition:: prerequisites
   :class: warning

	General: NumPy, SciPy, Swig, pyWin32, MatlPlotLib, Lxml, OpenMDAO

	Wind Plant Framework: FUSED-Wind (Framework for Unified Systems Engineering and Design of Wind Plants)

	Sub-Models: CommonSE, AeroelasticSE, RotorSE, DriveSE, DriveWPACT, TowerSE, JacketSE, Turbine_CostsSE, Plant_CostsSE, Plant_EnergySE, Plant_FinanceSEE

	Supporting python packages: Pandas, Algopy, Zope.interface, Sphinx, Xlrd, PyOpt, py2exe, Pyzmq, Sphinxcontrib-bibtex, Sphinxcontrib-zopeext, Numpydoc, Ipython

Clone the repository at `<https://github.com/WISDEM/Turbine_CostsSE>`_
or download the releases and uncompress/unpack (Turbine_CostsSE.py-|release|.tar.gz or Turbine_CostsSE.py-|release|.zip) from the website link at the bottom the `Turbine_CostsSE site<http://nwtc.nrel.gov/Turbine_CostsSE>`_.

To install Turbine_CostsSE, first activate the OpenMDAO environment and then install with the following command.

.. code-block:: bash

   $ plugin install

To check if installation was successful try to import the module from within an activated OpenMDaO environment:

.. code-block:: bash

    $ python

.. code-block:: python

    > import turbine_costsse.turbine_costsse
    > import turbine_costsse.nrel_csm_tcc
    > import turbine_costsse.turbine_costsse_2015
    > import turbine_costsse.nrel_csm_tcc_2015

or run the unit tests which include functional and gradient tests.  Analytic gradients are provided for variables only so warnings will appear for missing gradients on model input parameters; these can be ignored.

.. code-block:: bash

   $ python src/test/test_Turbine_CostsSE.py
   $ python src/test/test_Turbine_CostsSE_gradients.py
   $ python src/test/test_turbine_costsse_2015.py

An "OK" signifies that all the tests passed.

For software issues please use `<https://github.com/WISDEM/Turbine_CostsSE/issues>`_.  For functionality and theory related questions and comments please use the NWTC forum for `Systems Engineering Software Questions <https://wind.nrel.gov/forum/wind/viewtopic.php?f=34&t=1002>`_.

.. only:: latex

    An HTML version of this documentation that contains further details and links to the source code is available at `<http://wisdem.github.io/Turbine_CostsSE>`_

