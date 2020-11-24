.. _intro-label:

.. warning::
   In our use of FloatingSE, we have not been able to obtain conceptual platform geoemtry results that we trust are valid and worth pursuing further.  This may be due to a fundamental error in the formulation and implementation of FloatingSE.  It may also be due to the inherent limitations in using steady-state or quasi-static analysis methods to tackle a problem that is driven by its dynamic nature and dynamic loads.  NREL currently advises against reliance on FloatingSE.  Instead, we are developing a multifidelity floating turbine and platform design capability in the `Wind Energy with Integrated Servo-control (WEIS) project <https://github.com/WISDEM/WEIS>`_.

Introduction
============

The complexity of the physics, economics, logistics, and operation of
a floating offshore wind turbine makes it well suited to
systems-focused solutions. *FloatingSE* is the floating substructure
cost and sizing module for the WISDEM.

This document serves as both a User Manual and Theory Guide for
*FloatingSE*. *FloatingSE* can be executed as a stand-alone module or
coupled to the rest of the turbine through the WISDEM glue code. An
overview of the package contents is below and substructure geometry parameterization in Section
:ref:`geometry-label`. With this foundation, the underlying theory of
*FloatingSE’s* methodology is explained in Section
:ref:`theory-label`. This helps to understand the analysis execution
flow described in Section :ref:`execution-label` and the setup of
design variables and constraints for optimization problems in Section
:ref:`optimization-label`.


Package Files
-------------

The files that comprise the *FloatingSE* package are listed in :numref:`tbl_package`. Each file has a corresponding unit test file in the :file:`test_floatingse` directory.

.. _tbl_package:

.. table::
   File contents of the *FloatingSE* module.

   +---------------------------+----------------------------------------------------------------------------------------------------------------------+
   | **File Name**             | **Description**                                                                                                      |
   +===========================+======================================================================================================================+
   | :file:`floating.py`       | Top level *FloatingSE* OpenMDAO Group                                                                                |
   +---------------------------+-----------------------------+----------------------------------------------------------------------------------------+
   | :file:`column.py`         | Components calculating mass, buoyancy, and static stability of vertical frustum columns                              |
   +---------------------------+----------------------------------------------------------------------------------------------------------------------+
   | :file:`loading.py`        | Components for `Frame3DD <http://frame3dd.sourceforge.net>`_ analysis of structure, mass summation, and displacement |
   +---------------------------+----------------------------------------------------------------------------------------------------------------------+
   | :file:`map_mooring.py`    | Mooring analysis using `pyMAP <http://www.github.com/WISDEM/pyMAP>`_ module                                          |
   +---------+-----------------+----------------------------------------------------------------------------------------------------------------------+
   | :file:`substructure.py`   | Final buoyancyand stability checks of the substructure                                                               |
   +---------------------------+----------------------------------------------------------------------------------------------------------------------+
   | :file:`visualize.py`      | Standalone script that uses MayaVI to view substructure geometry                                                     |
   +---------------------------+----------------------------------------------------------------------------------------------------------------------+
