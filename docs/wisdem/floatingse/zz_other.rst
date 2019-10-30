.. _other-label:

WISDEM Floating Turbine Analysis
================================

Direct `WISDEM <http://www.github.com/WISDEM>`_ dependencies to `CommonSE <http://www.github.com/WISDEM/CommonSE>`_ and `TowerSE <http://www.github.com/WISDEM/TowerSE>`_, as well as some
of the supporting utilities housed under the `WISDEM <http://www.github.com/WISDEM>`_ umbrella, were
mentioned in Section :ref:`package-label`. There are other inputs into
*FloatingSE* that are outputs of other `WISDEM <http://www.github.com/WISDEM>`_ modules. For example
force, moment, and mass properties of the RNA. An OpenMDAO Group that
links these other modules together to form a virtual floating turbine
does not explicitly fit within the conceptual boundaries of the
:file:`src/floatingse` package. However, two files within the *WISDEM*
module (meant for high-level coupling of multiple `WISDEM <http://www.github.com/WISDEM>`_ components)
:file:`src/floating`\ -directory do assemble the virtual floating turbine,

* :file:`floating_turbine_assembly.py`: OpenMDAO Group that connects multiple `WISDEM <http://www.github.com/WISDEM>`_ modules for a complete floating offshore turbine simulation and optimization.
* :file:`floating_turbine_instance.py`: Implements the above assembly and extends.

The `WISDEM <http://www.github.com/WISDEM>`_ modules that exchange inputs and outputs within this
high-level assembly to crteate a virtual floating wind turbine are
listed in :numref:`tbl_new-wisdem`. In addition to *FloatingSE*,
two other new `WISDEM <http://www.github.com/WISDEM>`_ modules are also required to fully represent a
floating offshore wind plant (beyond just a single turbine),
*OffshoreBOS\_SE* and *Offshore\_OandM\_SE*. With these two additions,
`WISDEM <http://www.github.com/WISDEM>`_ can be diagrammed as shown in :numref:`fig_new-wisdem`.
While the core development of *OffshoreBOS\_SE* is effectively complete,
*Offshore\_OandM\_SE* has not yet been implemented. Note that as of this
writing, *DriveSE* is not yet connected to the others, but doing so is
part of the near-term development plan.


.. _tbl_new-wisdem:
.. table::
   WISDEM modules that comprise a virtual floating offshore wind plant.

   +------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   |  **Module**                                                            | **Description**                                                                                                                                                                                 |
   +========================================================================+=================================================================================================================================================================================================+
   | `RotorSE <http://www.github.com/WISDEM/RotorSE>`_                      | Analysis of aerodynamic and structural loading of rotor blades, determination of turbine power curve, and calculation of annual energy production (AEP)                                         |
   +------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | `CommonSE <http://www.github.com/WISDEM/CommonSE>`_                    | Wind and wave velocity profiles, drag calculations, probabilities distributions, frustum and tubular mass properties, math utilities, structural code compliance checks, and RNA aggregator     |
   +------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | `OffshoreBOS\_SE <http://www.github.com/WISDEM/OffshoreBOSSE>`_        | Capital costs for items aside from the turbine. Assembly, installation, commissioning, decommissioning, substructure, and financing costs for all components. See :cite:`obos`                  |
   +------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | `FloatingSE <http://www.github.com/WISDEM/FloatingSE>`_                | Floating substructure, including mooring and anchors, and static stability calculations                                                                                                         |
   +------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | `TurbineCostsSE (2015) <http://www.github.com/WISDEM/TurbineCostsSE>`_ | Capital costs for all turbine components above the water line                                                                                                                                   |
   +------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | `PlantFinanceSE <http://www.github.com/WISDEM/PlantFinanceSE>`_        | Roll-up of capital costs, balance of station costs, operational costs, financing rates, and net energy production into the levelized cost of energy (LCOE)                                      |
   +------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | `DriveSE <http://www.github.com/WISDEM/DriveSE>`_                      | Analysis of drive shaft, bearings, and gearbox. See :cite:`DriveSE`. NOT YET IMPLEMENTED                                                                                                        |
   +------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | *Offshore\_OandM\_SE*                                                  | Operational and maintenance costs. NOT YET DEVELOPED                                                                                                                                            |
   +------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | `WISDEM (module) <http://www.github.com/WISDEM/WISDEM>`_               | Top level groups and assemblies for full-turbine or plant analysis                                                                                                                              |
   +------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

   
.. _fig_new-wisdem:
.. figure::  figs/new_wisdem.png
    :width: 90%
    :align: center

    Conceptual diagram of WISDEM following the addition of *FloatingSE* and other modules (green boxes) to support offshore floating wind turbines.

    
With a floating offshore turbine constructed, system-wide optimization
and sensitivity studies can be conducted. An obvious objective function
for these optimizations would be the levelized cost of energy (LCOE) as
output from the *PlantFinanceSE* module. This optimization would require
additional constraints pertinent to the other modules to produce
relevant results. These other constraints are more suitably discussed
within the documentation of their home modules. Depending on the nature
of the analysis, the user may wish to include other design variables in
the optimization that are inputs to one of these other modules. As with
the constraints, the documentation of these design variables is best
found in their home modules.


.. _tbl_constraints-turb:
.. table::
   Additional constraints used in full floating offshore turbine optimization.

   +-------------+-----------------------------------+-------------+------------------------------------------------------------------------------+
   | **Lower**   | **Name**                          | **Upper**   | **Description**                                                              |
   +=============+===================================+=============+==============================================================================+
   |             | **Rotor**                         |             |                                                                              |
   +-------------+-----------------------------------+-------------+------------------------------------------------------------------------------+
   |             | rotor.P1\_margin                  | 1.00        | Blade frequency keep away from 1P rotor frequency                            |
   +-------------+-----------------------------------+-------------+------------------------------------------------------------------------------+
   |             | rotor.Pn\_margin                  | 1.00        | Blade frequency keep away from 3P rotor frequency                            |
   +-------------+-----------------------------------+-------------+------------------------------------------------------------------------------+
   |             | rotor.rotor\_buckling\_sparL      | 1.00        | Rotor blade upper spar cap structural buckling unity constraint              |
   +-------------+-----------------------------------+-------------+------------------------------------------------------------------------------+
   |             | rotor.rotor\_buckling\_sparU      | 1.00        | Rotor blade lower spar cap structural buckling unity constraint              |
   +-------------+-----------------------------------+-------------+------------------------------------------------------------------------------+
   |             | rotor.rotor\_buckling\_teL        | 1.00        | Rotor blade upper trailing edge panel structural buckling unity constraint   |
   +-------------+-----------------------------------+-------------+------------------------------------------------------------------------------+
   |             | rotor.rotor\_buckling\_teU        | 1.00        | Rotor blade lower trailing edge panel structural buckling unity constraint   |
   +-------------+-----------------------------------+-------------+------------------------------------------------------------------------------+
   |             | rotor.rotor\_damage\_sparL        | 0.00        | Rotor blade upper spar cap structural damage constraint                      |
   +-------------+-----------------------------------+-------------+------------------------------------------------------------------------------+
   |             | rotor.rotor\_damage\_sparU        | 0.00        | Rotor blade lower spar cap structural damage constraint                      |
   +-------------+-----------------------------------+-------------+------------------------------------------------------------------------------+
   |             | rotor.rotor\_damage\_teL          | 0.00        | Rotor blade upper trailing edge panel structural damage constraint           |
   +-------------+-----------------------------------+-------------+------------------------------------------------------------------------------+
   |             | rotor.rotor\_damage\_teU          | 0.00        | Rotor blade lower trailing edge panel structural damage constraint           |
   +-------------+-----------------------------------+-------------+------------------------------------------------------------------------------+
   |             | rotor.rotor\_strain\_sparL        | 1.00        | Rotor blade upper spar cap structural strain unity constraint                |
   +-------------+-----------------------------------+-------------+------------------------------------------------------------------------------+
   | -1.00       | rotor.rotor\_strain\_sparU        |             | Rotor blade lower spar cap structural strain unity constraint                |
   +-------------+-----------------------------------+-------------+------------------------------------------------------------------------------+
   |             | rotor.rotor\_strain\_teL          | 1.00        | Rotor blade upper trailing edge panel structural strain unity constraint     |
   +-------------+-----------------------------------+-------------+------------------------------------------------------------------------------+
   | -1.00       | rotor.rotor\_strain\_teU          |             | Rotor blade lower trailing edge panel structural strain unity constraint     |
   +-------------+-----------------------------------+-------------+------------------------------------------------------------------------------+
   |             | **Geometry**                      |             |                                                                              |
   +-------------+-----------------------------------+-------------+------------------------------------------------------------------------------+
   | 20.00       | tcons.ground\_clearance           |             | Minimum ground clearance of rotor blades                                     |
   +-------------+-----------------------------------+-------------+------------------------------------------------------------------------------+
   |             | tcons.tip\_deflection\_ratio      | 1.00        | Tip deflection limit to prevent tower strike as unity                        |
   +-------------+-----------------------------------+-------------+------------------------------------------------------------------------------+
   |             | **Stability**                     |             |                                                                              |
   +-------------+-----------------------------------+-------------+------------------------------------------------------------------------------+
   | 1.00        | tcons.frequency1P\_margin\_high   |             | Eigenfrequencies of entire structure must be below 1P frequency              |
   +-------------+-----------------------------------+-------------+------------------------------------------------------------------------------+
   |             | tcons.frequency1P\_margin\_low    | 1.00        | Eigenfrequencies of entire structure must be above 1P frequency              |
   +-------------+-----------------------------------+-------------+------------------------------------------------------------------------------+
   | 1.00        | tcons.frequency3P\_margin\_high   |             | Eigenfrequencies of entire structure must be below 3P frequency              |
   +-------------+-----------------------------------+-------------+------------------------------------------------------------------------------+
   |             | tcons.frequency3P\_margin\_low    | 1.00        | Eigenfrequencies of entire structure must be above 3P frequency              |
   +-------------+-----------------------------------+-------------+------------------------------------------------------------------------------+


References
==========


.. only:: html

    :bib:`Bibliography`

.. bibliography:: references.bib
   :cited:
   :style: unsrt
