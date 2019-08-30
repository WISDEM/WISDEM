.. _package-label:

Package Documentation
=====================

*FloatingSE* is a module within the larger `WISDEM <http://www.github.com/WISDEM>`_ project, developed
primarily by engineers at the `National Renewable Energy Laboratory (NREL) <http://www.nrel.gov>`_. *FloatingSE* is open-source and the project repository is housed
on GitHub at `<https://github.com/WISDEM/FloatingSE>`_

Installation
------------
.. admonition:: Prerequisites
   :class: warning

	General: NumPy, SciPy, OpenMDAO

	WISDEM-Models: `CommonSE <http://www.github.com/WISDEM/CommonSE>`_, `TowerSE <http://www.github.com/WISDEM/TowerSE>`_, `pyMAP <http://www.github.com/WISDEM/pyMAP>`_, `pyFrame3DD <http://www.github.com/WISDEM/pyFrame3DD>`_, `Akima <http://www.github.com/WISDEM/akima>`_

	Supporting python packages: Sphinx, Sphinxcontrib-bibtex, Numpydoc

	Optional visualization packages: Mayavi, VTK

Clone the repository at `<https://github.com/WISDEM/FloatingSE>`_.

Install *FloatingSE* with the following command for a permanent installation:

.. code-block:: bash

   $ python setup.py install

For a development installation use:

.. code-block:: bash

   $ python setup.py develop --user

To check if installation was successful try to import the module:

.. code-block:: bash

    $ python

.. code-block:: python

    > import floatingse

or run the unit tests:

.. code-block:: bash

   $ python test/package.py

An "OK" signifies that all the tests passed.
    
Package Files
-------------

The files that comprise the *FloatingSE* package are found in the
:file:`src/floatingse` directory in accordance with standard Python package
conventions. In addition to these files, there are also affiliated unit
tests that probe the behavior of individual functions. These are located
in the :file:`test` sub-directory. A summary of all package files is
included in :numref:`tbl_package`.

.. _tbl_package:

.. table::
   File contents of the :file:`src/floatingse` Python package within *FloatingSE*.

   +---------------------------------------+-----------------------------+----------------------------------------------------------------------------------------------------------------------+
   | **File Name**                         | **Unit Test File**          | **Description**                                                                                                      |
   +=======================================+=============================+======================================================================================================================+
   | :file:`floating.py`                   | :file:`floating_PyU.py`     | Top level *FloatingSE* OpenMDAO Group                                                                                |
   +---------------------------------------+-----------------------------+----------------------------------------------------------------------------------------------------------------------+
   | :file:`column.py`                     | :file:`column_PyU.py`       | Components calculating mass, buoyancy, and static stability of vertical frustum columns                              |
   +---------------------------------------+-----------------------------+----------------------------------------------------------------------------------------------------------------------+
   | :file:`loading.py`                    | :file:`loading_PyU.py`      | Components for `Frame3DD <http://frame3dd.sourceforge.net>`_ analysis of structure, mass summation, and displacement |
   +---------------------------------------+-----------------------------+----------------------------------------------------------------------------------------------------------------------+
   | :file:`map_mooring.py`                | :file:`map_mooring_PyU.py`  | Mooring analysis using `pyMAP <http://www.github.com/WISDEM/pyMAP>`_ module                                          |
   +---------------------------------------+-----------------------------+----------------------------------------------------------------------------------------------------------------------+
   | :file:`substructure.py`               | :file:`substructure_PyU.py` | Final buoyancyand stability checks of the substructure                                                               |
   +---------------------------------------+-----------------------------+----------------------------------------------------------------------------------------------------------------------+
   |                                       | :file:`package_PyU.py`      | Convenience aggregator of all unit test files                                                                        |
   +---------------------------------------+-----------------------------+----------------------------------------------------------------------------------------------------------------------+
   | :file:`instance/floating_instance.py` |                             | Parent class controlling optimization drivers, constraints, and visualization                                        |
   +---------------------------------------+-----------------------------+----------------------------------------------------------------------------------------------------------------------+
   | :file:`instance/spar_instance.py`     |                             | Spar example implementing design parameters and visualization                                                        |
   +---------------------------------------+-----------------------------+----------------------------------------------------------------------------------------------------------------------+
   | :file:`instance/semi_instance.py`     |                             | Semisubmersible example implementing design parameters and visualization                                             |
   +---------------------------------------+-----------------------------+----------------------------------------------------------------------------------------------------------------------+
   | :file:`instance/tlp_instance.py`      |                             | Tension leg platform example implementing design parameters and visualization                                        |
   +---------------------------------------+-----------------------------+----------------------------------------------------------------------------------------------------------------------+

Documentation
-------------

The class structure for all the modules is listed below.

Referenced *FloatingSE* High-Level Group
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. module:: floatingse.floating
.. class:: FloatingSE

	    
Referenced *FloatingSE* Vertical, Submerged Column of Frustums
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. module:: floatingse.column
.. class:: BulkheadMass
.. class:: BuoyancyTankProperties
.. class:: StiffenerMass
.. class:: ColumnGeometry
.. class:: ColumnProperties
.. class:: ColumnBuckling
.. class:: Column

	    
Referenced *FloatingSE* Structural Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. module:: floatingse.loading
.. class:: FloatingFrame
.. class:: TrussIntegerToBoolean
.. class:: Loading

	    
Referenced *FloatingSE* Mooring Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. module:: floatingse.map_mooring
.. class:: MapMooring

	    
Referenced *FloatingSE* Stability Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. module:: floatingse.substructure
.. class:: SubstructureGeometry
.. class:: Substructure

