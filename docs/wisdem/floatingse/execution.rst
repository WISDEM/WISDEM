.. _execution-label:

Execution
=========

Executing *FloatingSE* requires additional inputs beyond those of the
geometry definition described above in Section
:ref:`geometry-label`. Other user inputs for the metocean and loading
environment, and the operational constraints, are required to evaluate
the total mass, cost, and code compliance. These variables are also
included in the `WindIO <https://windio.readthedocs.io/en/latest/>`_
effort or found in the `floating-specific examples
<https://github.com/WISDEM/WISDEM/tree/master/examples/09_floating>`_
for standalone execution.


Simulation Flow
---------------

Once the input variables are completely specified, *FloatingSE* executes
the analysis of the substructure. Conceptually, the simulation is
organized by the flowchart in :numref:`fig_floatingse`.

.. _fig_floatingse:
.. figure::  /images/floatingse/floatingse.*
    :width: 90%
    :align: center

    Conceptual diagram of *FloatingSE* execution.


From a more
technical perspective, *FloatingSE* is an OpenMDAO Group, so the
analysis sequence is broken down by the sub-groups and sub-components in
the order that they are listed in Table [tbl:exec]. In an OpenMDAO
group, sub-groups and components are given prefixes to aid in referring
to specific variables. The prefixes used in *FloatingSE* are also listed
in :numref:`tbl_exec`.


.. _tbl_exec:
.. table::
   Components and sub-groups within *FloatingSE*.

   +------+--------------+--------------------------+------------------------------------------------------------------------------------------------------------------------+
   |      | **Prefix**   | **Name**                 | **Description**                                                                                                        |
   +======+==============+==========================+========================================================================================================================+
   | 1)   | ``tow``      | *TowerLeanSE*            | Discretization of tower geometry (but no analysis)                                                                     |
   +------+--------------+--------------------------+------------------------------------------------------------------------------------------------------------------------+
   | 2)   | ``main``     | *Column*                 | Discretization and API Bulletin 2U compliance of main.vertical column                                                  |
   +------+--------------+--------------------------+------------------------------------------------------------------------------------------------------------------------+
   | 3)   | ``off``      | *Column*                 | Discretization and API Bulletin 2U compliance of offset columns                                                        |
   +------+--------------+--------------------------+------------------------------------------------------------------------------------------------------------------------+
   | 4)   | ``sg``       | *SubstructureGeometry*   | Geometrical constraints on substructure                                                                                |
   +------+--------------+--------------------------+------------------------------------------------------------------------------------------------------------------------+
   | 5)   | ``mm``       | *MapMooring*             | Mooring system analysis via `pyMAP <http://www.github.com/WISDEM/pyMAP>`_                                              |
   +------+--------------+--------------------------+------------------------------------------------------------------------------------------------------------------------+
   | 6)   | ``load``     | *FloatingLoading*        | Structural analysis of complete floating turbine load path via `pyFrame3DD <http://www.github.com/WISDEM/pyFrame3DD>`_ |
   +------+--------------+--------------------------+------------------------------------------------------------------------------------------------------------------------+
   | 7)   | ``subs``     | *Substructure*           | Static stability and final mass and cost summation for generic substructure                                            |
   +------+--------------+--------------------------+------------------------------------------------------------------------------------------------------------------------+


Outputs are accumulated in each sub-group or component, and they either
become inputs to other components, become constraints for optimization
problems, become design variables for optimization problems, or can
simply be ignored. Currently, a single execution of FloatingSE takes
only a handful of seconds on a modern laptop computer.

Examples
--------

As mentioned previously `floating-specific examples
<https://github.com/WISDEM/WISDEM/tree/master/examples/09_floating>`_
examples are provided. These files are encoded with default starting
configurations (from :cite:`OC3` and :cite:`OC4`, respectively), with
some modifications. There is an additional spar example that also has
a ready configurations for optimization with design variables,
constraints, and solvers options.  A visualization of the geometries
described by these examples is shown in :numref:`fig_initial-spar` and
:numref:`fig_initial-semi`.


.. _fig_initial-spar:
.. figure::  /images/floatingse/spar-initial.*
    :width: 75%
    :align: center

    Spar example in *FloatingSE* taken from OC3 :cite:`OC3` project.


.. _fig_initial-semi:
.. figure::  /images/floatingse/semi-initial.*
    :width: 75%
    :align: center

    Semi example in *FloatingSE* taken from OC4 :cite:`OC4` project.


.. bibliography:: ../../references.bib
   :filter: docname in docnames
