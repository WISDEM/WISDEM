.. _intro-label:

Introduction
============

The complexity of the physics, economics, logistics, and operation of a
floating offshore wind turbine makes it well suited to systems-focused
solutions. *FloatingSE* is the floating substructure cost and sizing
module for the `Wind-Plant Integrated System Design and Engineering
Model (WISDEM) <http://www.github.com/WISDEM>`_ tool, developed by the `National
Renewable Energy Laboratory (NREL) <http://www.nrel.gov>`_. `WISDEM <http://www.github.com/WISDEM>`_ is a set of integrated modules that creates
a virtual, vertically integrated wind plant. The models use engineering
principles for conceptual design and preliminary analysis, and link to
financial modules for LCOE estimation.

The `WISDEM <http://www.github.com/WISDEM>`_ modules, including *FloatingSE*, are built around the
`OpenMDAO <http://openmdao.org/>`_ library :cite:`openmdao`, an open-source high-performance computing platform for
systems analysis and multidisciplinary optimization, written in Python .
Due to the structure of `OpenMDAO <http://openmdao.org/>`_, and modular design of `WISDEM <http://www.github.com/WISDEM>`_,
individual modules can be exercised individually or in unison for
turbine or plant level studies. This feature also applies to
*FloatingSE*, in that module-specific optimizations of a floating
substructure and its anchoring and mooring systems can be executed while
treating the rest of the turbine, plant, and operational strategy as
static. Alternatively, *FloatingSE* can be linked to other `WISDEM <http://www.github.com/WISDEM>`_
modules to execute turbine-level and/or plant-level tradeoff and
optimization studies.

This document serves as both a User Manual and Theory Guide for
*FloatingSE*. An overview of the package contents is in Section
:ref:`package-label` and substructure geometry parameterization in Section
:ref:`geometry-label`. With this foundation, the underlying theory of
*FloatingSE’s* methodology is explained in Section :ref:`theory-label`. This
helps to understand the analysis execution flow described in Section
:ref:`execution-label` and the setup of design variables and constraints for
optimization problems in Section :ref:`optimization-label`. Finally, some discussion of
how *FloatingSE* can be linked to other `WISDEM <http://www.github.com/WISDEM>`_ modules via a high-level
OpenMDAO Group is described in Section :ref:`other-label`.


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

   +---------------------------------------+-----------------------------+----------------------------------------------------------------------------------------+
   | **File Name**                         | **Description**                                                                                                      |
   +=======================================+=============================+========================================================================================+
   | :file:`floating.py`                   | Top level *FloatingSE* OpenMDAO Group                                                                                |
   +---------------------------------------+-----------------------------+----------------------------------------------------------------------------------------+
   | :file:`column.py`                     | Components calculating mass, buoyancy, and static stability of vertical frustum columns                              |
   +---------------------------------------+-----------------------------+----------------------------------------------------------------------------------------+
   | :file:`loading.py`                    | Components for `Frame3DD <http://frame3dd.sourceforge.net>`_ analysis of structure, mass summation, and displacement |
   +---------------------------------------+-----------------------------+----------------------------------------------------------------------------------------+
   | :file:`map_mooring.py`                | Mooring analysis using `pyMAP <http://www.github.com/WISDEM/pyMAP>`_ module                                          |
   +---------+-----------------------------+----------------------------------------------------------------------------------------------------------------------+
   | :file:`substructure.py`               | Final buoyancyand stability checks of the substructure                                                               |
   +---------------------------------------+-----------------------------+----------------------------------------------------------------------------------------+
   | :file:`instance/floating_instance.py` | Parent class controlling optimization drivers, constraints, and visualization                                        |
   +---------+-----------------------------+----------------------------------------------------------------------------------------+
   | :file:`instance/spar_instance.py`     | Spar example implementing design parameters and visualization                                                        |
   +---------------------------------------+-----------------------------+----------------------------------------------------------------------------------------+
   | :file:`instance/semi_instance.py`     | Semisubmersible example implementing design parameters and visualization                                             |
   +---------------------------------------+-----------------------------+----------------------------------------------------------------------------------------+
   | :file:`instance/tlp_instance.py`      | Tension leg platform example implementing design parameters and visualization                                        |
   +---------------------------------------+-----------------------------+----------------------------------------------------------------------------------------+
