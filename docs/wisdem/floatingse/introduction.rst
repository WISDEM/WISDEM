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
`OpenMDAO 1.x <http://openmdao.org/>`_ library :cite:`openmdao`, an open-source high-performance computing platform for
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
