.. _optimization-label:

Optimization
============

Executing *FloatingSE* by hand is sufficient to explore some simple
one-off or comparison analyses between a few runs. OpenMDAO provides
extensive optimization capability, which can give yield richer and more
insightful analyses.

Design Variables
----------------

In WISDEM, via OpenMDAO, any input parameter can be designated a design
variable. The design variables used in this study focused on the
geometric specification of the floating substructure and mooring
subsystem. Slightly different design variables and bounds were used for
spar, semisubmersible, and TLP optimizations. The complete listing of
the design variables for each optimization configuration is shown in
:numref:`tbl_designvar`. Note that the integer design variables were only
used in the global optimization with the genetic algorithm, not the
local search with the simplex algorithm.


.. _tbl_designvar:
.. table::
   Standard design variables, their size, and units used for optimization in *FloatingSE*. Note that :math:`n_s` denotes the number of sections in the column discretization.

   +------------------------------------------------+-----------------------------------------+-------------+-------------------------------+--------------+
   | **Variable**                                   | **Name**                                | **Units**   | **Type**                      | **Bounds**   |
   +================================================+=========================================+=============+===============================+==============+
   | Main col section height                        | ``main_section_height``                 |             | Float array (:math:`n_s`)     | 0.1–50       |
   +------------------------------------------------+-----------------------------------------+-------------+-------------------------------+--------------+
   | Main col outer diameter                        | ``main_outer_diameter``                 |             | Float array (:math:`n_s+1`)   | 2.1–40       |
   +------------------------------------------------+-----------------------------------------+-------------+-------------------------------+--------------+
   | Main col wall thickness                        | ``main_wall_thickness``                 |             | Float array (:math:`n_s+1`)   | 0.001–0.5    |
   +------------------------------------------------+-----------------------------------------+-------------+-------------------------------+--------------+
   | Main col freeboard                             | ``main_freeboard``                      |             | Float scalar                  | 0–50         |
   +------------------------------------------------+-----------------------------------------+-------------+-------------------------------+--------------+
   | Main col stiffener web height                  | ``main_stiffener_web_height``           |             | Float array (:math:`n_s`)     | 0.01–1       |
   +------------------------------------------------+-----------------------------------------+-------------+-------------------------------+--------------+
   | Main col stiffener web thickness               | ``main_stiffener_web_thickness``        |             | Float array (:math:`n_s`)     | 0.001–0.5    |
   +------------------------------------------------+-----------------------------------------+-------------+-------------------------------+--------------+
   | Main col stiffener flange width                | ``main_stiffener_flange_width``         |             | Float array (:math:`n_s`)     | 0.01–5       |
   +------------------------------------------------+-----------------------------------------+-------------+-------------------------------+--------------+
   | Main col stiffener flange thickness            | ``main_stiffener_flange_thickness``     |             | Float array (:math:`n_s`)     | 0.001–0.5    |
   +------------------------------------------------+-----------------------------------------+-------------+-------------------------------+--------------+
   | Main col stiffener spacing                     | ``main_stiffener_spacing``              |             | Float array (:math:`n_s`)     | 0.1–100      |
   +------------------------------------------------+-----------------------------------------+-------------+-------------------------------+--------------+
   | Main col permanent ballast height              | ``main_permanent_ballast_height``       |             | Float scalar                  | 0.1–50       |
   +------------------------------------------------+-----------------------------------------+-------------+-------------------------------+--------------+
   | Main col buoyancy tank diameter                | ``main_buoyancy_tank_diameter``         |             | Float scalar                  | 0–50         |
   +------------------------------------------------+-----------------------------------------+-------------+-------------------------------+--------------+
   | Main col buoyancy tank height                  | ``main_buoyancy_tank_height``           |             | Float scalar                  | 0–20         |
   +------------------------------------------------+-----------------------------------------+-------------+-------------------------------+--------------+
   | Main col buoyancy tank location (fraction)     | ``main_buoyancy_tank_location``         |             | Float scalar                  | 0–1          |
   +------------------------------------------------+-----------------------------------------+-------------+-------------------------------+--------------+
   | Number of offset cols                          | ``number_of_offset_columns``            |             | Integer scalar                | 3-5          |
   +------------------------------------------------+-----------------------------------------+-------------+-------------------------------+--------------+
   | Offset col section height                      | ``offset_section_height``               |             | Float array (:math:`n_s`)     | 0.1–50       |
   +------------------------------------------------+-----------------------------------------+-------------+-------------------------------+--------------+
   | Offset col outer diameter                      | ``offset_outer_diameter``               |             | Float array (:math:`n_s+1`)   | 1.1–40       |
   +------------------------------------------------+-----------------------------------------+-------------+-------------------------------+--------------+
   | Offset col wall thickness                      | ``offset_wall_thickness``               |             | Float array (:math:`n_s+1`)   | 0.001–0.5    |
   +------------------------------------------------+-----------------------------------------+-------------+-------------------------------+--------------+
   | Offset col freeboard                           | ``offset_freeboard``                    |             | Float scalar                  | 2–15         |
   +------------------------------------------------+-----------------------------------------+-------------+-------------------------------+--------------+
   | Offset col stiffener web height                | ``offset_stiffener_web_height``         |             | Float array (:math:`n_s`)     | 0.01–1       |
   +------------------------------------------------+-----------------------------------------+-------------+-------------------------------+--------------+
   | Offset col stiffener web thickness             | ``offset_stiffener_web_thickness``      |             | Float array (:math:`n_s`)     | 0.001–0.5    |
   +------------------------------------------------+-----------------------------------------+-------------+-------------------------------+--------------+
   | Offset col stiffener flange width              | ``offset_stiffener_flange_width``       |             | Float array (:math:`n_s`)     | 0.01–5       |
   +------------------------------------------------+-----------------------------------------+-------------+-------------------------------+--------------+
   | Offset col stiffener flange thickness          | ``offset_stiffener_flange_thickness``   |             | Float array (:math:`n_s`)     | 0.001–0.5    |
   +------------------------------------------------+-----------------------------------------+-------------+-------------------------------+--------------+
   | Offset col stiffener spacing                   | ``offset_stiffener_spacing``            |             | Float array (:math:`n_s`)     | 0.01–100     |
   +------------------------------------------------+-----------------------------------------+-------------+-------------------------------+--------------+
   | Offset col permanent ballast height            | ``offset_permanent_ballast_height``     |             | Float scalar                  | 0.1–50       |
   +------------------------------------------------+-----------------------------------------+-------------+-------------------------------+--------------+
   | Offset col buoyancy tank diameter              | ``offset_buoyancy_tank_diameter``       |             | Float scalar                  | 0–50         |
   +------------------------------------------------+-----------------------------------------+-------------+-------------------------------+--------------+
   | Offset col buoyancy tank height                | ``offset_buoyancy_tank_height``         |             | Float scalar                  | 0–20         |
   +------------------------------------------------+-----------------------------------------+-------------+-------------------------------+--------------+
   | Offset col buoyancy tank location (fraction)   | ``main_buoyancy_tank_location``         |             | Float scalar                  | 0–1          |
   +------------------------------------------------+-----------------------------------------+-------------+-------------------------------+--------------+
   | Radius to offset col                           | ``radius_to_offset_column``             |             | Float scalar                  | 5–100        |
   +------------------------------------------------+-----------------------------------------+-------------+-------------------------------+--------------+
   | Pontoon outer diameter                         | ``pontoon_outer_diameter``              |             | Float scalar                  | 0.1–10       |
   +------------------------------------------------+-----------------------------------------+-------------+-------------------------------+--------------+
   | Pontoon wall thickness                         | ``pontoon_wall_thickness``              |             | Float scalar                  | 0.01–1       |
   +------------------------------------------------+-----------------------------------------+-------------+-------------------------------+--------------+
   | Lower main-offset pontoons                     | ``lower_attachment_pontoons_int``       |             | Integer scalar                | 0–1          |
   +------------------------------------------------+-----------------------------------------+-------------+-------------------------------+--------------+
   | Upper main-offset pontoons                     | ``upper_attachment_pontoons_int``       |             | Integer scalar                | 0–1          |
   +------------------------------------------------+-----------------------------------------+-------------+-------------------------------+--------------+
   | Cross main-offset pontoons                     | ``cross_attachment_pontoons_int``       |             | Integer scalar                | 0–1          |
   +------------------------------------------------+-----------------------------------------+-------------+-------------------------------+--------------+
   | Lower offset ring pontoons                     | ``lower_ring_pontoons_int``             |             | Integer scalar                | 0–1          |
   +------------------------------------------------+-----------------------------------------+-------------+-------------------------------+--------------+
   | Upper offset ring pontoons                     | ``upper_ring_pontoons_int``             |             | Integer scalar                | 0–1          |
   +------------------------------------------------+-----------------------------------------+-------------+-------------------------------+--------------+
   | Outer V-pontoons                               | ``outer_cross_pontoons_int``            |             | Integer scalar                | 0–1          |
   +------------------------------------------------+-----------------------------------------+-------------+-------------------------------+--------------+
   | Main col pontoon attach lower (fraction)       | ``main_pontoon_attach_lower``           |             | Float scalar                  | 0–0.5        |
   +------------------------------------------------+-----------------------------------------+-------------+-------------------------------+--------------+
   | Main col pontoon attach upper (fraction)       | ``main_pontoon_attach_upper``           |             | Float scalar                  | 0.5–1        |
   +------------------------------------------------+-----------------------------------------+-------------+-------------------------------+--------------+
   | Fairlead (fraction)                            | ``fairlead_location``                   |             | Float scalar                  | 0–1          |
   +------------------------------------------------+-----------------------------------------+-------------+-------------------------------+--------------+
   | Fairlead offset from col                       | ``fairlead_offset_from_shell``          |             | Float scalar                  | 5–30         |
   +------------------------------------------------+-----------------------------------------+-------------+-------------------------------+--------------+
   | Fairlead pontoon diameter                      | ``fairlead_support_outer_diameter``     |             | Float scalar                  | 0.1–10       |
   +------------------------------------------------+-----------------------------------------+-------------+-------------------------------+--------------+
   | Fairlead pontoon wall thickness                | ``fairlead_support_outer_thickness``    |             | Float scalar                  | 0.001–1      |
   +------------------------------------------------+-----------------------------------------+-------------+-------------------------------+--------------+
   | Number of mooring connections                  | ``number_of_mooring_connections``       |             | Integer scalar                | 3–5          |
   +------------------------------------------------+-----------------------------------------+-------------+-------------------------------+--------------+
   | Mooring lines per connection                   | ``mooring_lines_per_connection``        |             | Integer scalar                | 1–3          |
   +------------------------------------------------+-----------------------------------------+-------------+-------------------------------+--------------+
   | Mooring diameter                               | ``mooring_diameter``                    |             | Float scalar                  | 0.05–2       |
   +------------------------------------------------+-----------------------------------------+-------------+-------------------------------+--------------+
   | Mooring line length                            | ``mooring_line_length``                 |             | Float scalar                  | 0–3000       |
   +------------------------------------------------+-----------------------------------------+-------------+-------------------------------+--------------+
   | Anchor distance                                | ``anchor_radius``                       |             | Float scalar                  | 0–5000       |
   +------------------------------------------------+-----------------------------------------+-------------+-------------------------------+--------------+



Constraints
-----------

Due to the many design variables, permutations of settings, and applied
physics, there are many constraints that must be applied for an
optimization to close. The constraints capture both physical
limitations, such as column buckling, but also inject industry
standards, guidelines, and lessons learned from engineering experience
into the optimization. As described in the Introduction, this is a
critically important element in building a MDAO framework for conceptual
design that yields feasible results worth interrogating further with
higher-fidelity tools. The constraints used in the substructure design
optimization and sensitivity studies are listed in :numref:`tbl_constraints`. Where appropriate, some of the constraint values
differ from one type of substructure to another. Some additional
explanation is provided for a handful of constraints in the subsections
below.


.. _tbl_constraints:
.. table::
   Optimization constraints used in *FloatingSE*.

   +-------------+-------------------------------------------+-------------+-------------------------------------------------------+
   | **Lower**   | **Variable**                              | **Upper**   | **Comments**                                          |
   +=============+===========================================+=============+=======================================================+
   |             | **Tower / Main / Offset Columns**         |             |                                                       |
   +-------------+-------------------------------------------+-------------+-------------------------------------------------------+
   |             | Eurocode global buckling                  | 1.0         |                                                       |
   +-------------+-------------------------------------------+-------------+-------------------------------------------------------+
   |             | Eurocode shell buckling                   | 1.0         |                                                       |
   +-------------+-------------------------------------------+-------------+-------------------------------------------------------+
   |             | Eurocode stress limit                     | 1.0         |                                                       |
   +-------------+-------------------------------------------+-------------+-------------------------------------------------------+
   |             | Manufacturability                         | 0.5         | Taper ratio limit                                     |
   +-------------+-------------------------------------------+-------------+-------------------------------------------------------+
   | 120.0       | Weld-ability                              |             | Diameter:thickness ratio limit                        |
   +-------------+-------------------------------------------+-------------+-------------------------------------------------------+
   |             | **Main / Offset Columns**                 |             |                                                       |
   +-------------+-------------------------------------------+-------------+-------------------------------------------------------+
   |             | Draft ratio                               | 1.0         | Ratio of draft to max value                           |
   +-------------+-------------------------------------------+-------------+-------------------------------------------------------+
   |             | API 2U general buckling- axial loads      | 1.0         |                                                       |
   +-------------+-------------------------------------------+-------------+-------------------------------------------------------+
   |             | API 2U local buckling- axial loads        | 1.0         |                                                       |
   +-------------+-------------------------------------------+-------------+-------------------------------------------------------+
   |             | API 2U general buckling- external loads   | 1.0         |                                                       |
   +-------------+-------------------------------------------+-------------+-------------------------------------------------------+
   |             | API 2U local buckling- external loads     | 1.0         |                                                       |
   +-------------+-------------------------------------------+-------------+-------------------------------------------------------+
   |             | Wave height:freeboard ratio               | 1.0         | Maximum wave height relative to freeboard             |
   +-------------+-------------------------------------------+-------------+-------------------------------------------------------+
   | 1.0         | Stiffener flange compactness              |             |                                                       |
   +-------------+-------------------------------------------+-------------+-------------------------------------------------------+
   | 1.0         | Stiffener web compactness                 |             |                                                       |
   +-------------+-------------------------------------------+-------------+-------------------------------------------------------+
   |             | Stiffener flange spacing ratio            | 1.0         | Stiffener spacing relative to flange width            |
   +-------------+-------------------------------------------+-------------+-------------------------------------------------------+
   |             | Stiffener radius ratio                    | 0.50        | Stiffener height relative to diameter                 |
   +-------------+-------------------------------------------+-------------+-------------------------------------------------------+
   |             | **Offset Columns**                        |             | *Semi only*                                           |
   +-------------+-------------------------------------------+-------------+-------------------------------------------------------+
   | 0.0         | Heel freeboard margin                     |             | Height required to stay above waterline at max heel   |
   +-------------+-------------------------------------------+-------------+-------------------------------------------------------+
   | 0.0         | Heel draft margin                         |             | Draft required to stay submerged at max heel          |
   +-------------+-------------------------------------------+-------------+-------------------------------------------------------+
   |             | **Pontoons**                              |             | *Semi only*                                           |
   +-------------+-------------------------------------------+-------------+-------------------------------------------------------+
   |             | Eurocode stress limit                     | 1.0         |                                                       |
   +-------------+-------------------------------------------+-------------+-------------------------------------------------------+
   |             | **Tower**                                 |             |                                                       |
   +-------------+-------------------------------------------+-------------+-------------------------------------------------------+
   | -0.01       | Hub height error                          | 0.01        |                                                       |
   +-------------+-------------------------------------------+-------------+-------------------------------------------------------+
   |             | **Mooring**                               |             |                                                       |
   +-------------+-------------------------------------------+-------------+-------------------------------------------------------+
   | 0.0         | Axial stress limit                        | 1.0         |                                                       |
   +-------------+-------------------------------------------+-------------+-------------------------------------------------------+
   |             | Line length limit                         | 1.0         | Loss of tension or catenary hang                      |
   +-------------+-------------------------------------------+-------------+-------------------------------------------------------+
   |             | Heel moment ratio                         | 1.0         | Ratio of overturning moment to restoring moment       |
   +-------------+-------------------------------------------+-------------+-------------------------------------------------------+
   |             | Surge force ratio                         | 1.0         | Ratio of surge force to restoring force               |
   +-------------+-------------------------------------------+-------------+-------------------------------------------------------+
   |             | **Geometry**                              |             |                                                       |
   +-------------+-------------------------------------------+-------------+-------------------------------------------------------+
   | 1.0         | Main-offset spacing                       |             | Minimum spacing between main and offset columns       |
   +-------------+-------------------------------------------+-------------+-------------------------------------------------------+
   | 0.0         | Nacelle transition buffer                 |             | Tower diameter limit at nacelle junction              |
   +-------------+-------------------------------------------+-------------+-------------------------------------------------------+
   | -1.0        | Tower transition buffer                   | 1.0         | Diameter consistency at freeboard point               |
   +-------------+-------------------------------------------+-------------+-------------------------------------------------------+
   |             | **Stability**                             |             |                                                       |
   +-------------+-------------------------------------------+-------------+-------------------------------------------------------+
   | 0.10        | Metacentric height                        |             | *Not applied to TLPs*                                 |
   +-------------+-------------------------------------------+-------------+-------------------------------------------------------+
   | 1.0         | Wave-Eigenmode boundary (upper)           |             | Natural frequencies below wave frequency range        |
   +-------------+-------------------------------------------+-------------+-------------------------------------------------------+
   |             | Wave-Eigenmode boundary (lower)           | 1.0         | Natural frequencies above wave frequency range        |
   +-------------+-------------------------------------------+-------------+-------------------------------------------------------+
   | 0.0         | Water ballast height limit                | 1.0         |                                                       |
   +-------------+-------------------------------------------+-------------+-------------------------------------------------------+
   | 0.0         | Water ballast mass                        |             | Neutral buoyancy                                      |
   +-------------+-------------------------------------------+-------------+-------------------------------------------------------+



Geometry Constraints
~~~~~~~~~~~~~~~~~~~~

Words :numref:`tbl_geomconvar`


.. _tbl_geomconvar:
.. table::
   Constraint variables for the geometry in *FloatingSE*.

   +-----------------+----------------+------------------------------------------------+
   | **Variable**    | **Type**       | **Description**                                |
   +-----------------+----------------+------------------------------------------------+
   | ``max_draft``   | Float scalar   | Maximum allowable draft for the substructure   |
   +-----------------+----------------+------------------------------------------------+

Manufacturing Constraints
~~~~~~~~~~~~~~~~~~~~~~~~~

Manufacturing steel frustum shells requires rolling steel plates into
shape and welding along a seam to close the section. To accommodate
traditional rolling and welding practices, both the diameter taper over
the course of a section and the wall thickness ratio relative to the
diameter are capped. Similarly, to facilitate welding the
semisubmersible pontoons to the columns, constraints regarding the radio
of diameters between the two are enforced. These limits are determined
by user parameters in :numref:`tbl_manconvar` and constraints,


.. _tbl_manconvar:
.. table::
   Constraint variables for the manufacturability in *FloatingSE*.

   +------------------------------------+----------------+------------------------------------------+
   | **Variable**                       | **Type**       | **Description**                          |
   +------------------------------------+----------------+------------------------------------------+
   | ``min_taper_ratio``                | Float scalar   | For manufacturability of rolling steel   |
   +------------------------------------+----------------+------------------------------------------+
   | ``min_diameter_thickness_ratio``   | Float scalar   | For weld-ability                         |
   +------------------------------------+----------------+------------------------------------------+
   | ``connection_ratio_max``           | Float scalar   | For welding pontoons to columns          |
   +------------------------------------+----------------+------------------------------------------+


Stress Limits and Code Compliance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The stress and buckling code compliance constraints are formulated as
utilization ratios (ratio of actual to maximum values), with a safety
factor, which must be less than one. The safety factor parameters are
listed in :numref:`tbl_safetyvar`.


.. _tbl_safetyvar:
.. table::
   Variables specifying the factors of safety within *FloatingSE*.

   +---------------------+----------------+-------------------------------------------+
   | **Variable**        | **Type**       | **Description**                           |
   +---------------------+----------------+-------------------------------------------+
   | ``gamma_f``         | Float scalar   | Safety factor on                          |
   +---------------------+----------------+-------------------------------------------+
   | ``gamma_b``         | Float scalar   | Safety factor on buckling                 |
   +---------------------+----------------+-------------------------------------------+
   | ``gamma_m``         | Float scalar   | Safety factor on materials                |
   +---------------------+----------------+-------------------------------------------+
   | ``gamma_n``         | Float scalar   | Safety factor on consequence of failure   |
   +---------------------+----------------+-------------------------------------------+
   | ``gamma_fatigue``   | Float scalar   | Not currently used                        |
   +---------------------+----------------+-------------------------------------------+



Stability
~~~~~~~~~

As described above, surge and pitch stability are enforced through
similar approaches. The total force and moment acting on the turbine are
compared to the restoring forces and moments applied by the mooring
system, buoyancy, or other sources at the maximum allowable point of
displacement. These constraints are formulated as ratios with the user
specifying the maximum allowable limits via the variables in :numref:`tbl_moorcon`.

.. _tbl_moorcon:
.. table::
   Constraint variables for the mooring system in *FloatingSE*.

   +-------------------------+----------------+---------------+-----------------------------------------------------+
   | **Variable**            | **Type**       | **Units**     | **Description**                                     |
   +-------------------------+----------------+---------------+-----------------------------------------------------+
   | ``max_offset``          | Float scalar   | :math:`m`     | Max surge/sway offset                               |
   +-------------------------+----------------+---------------+-----------------------------------------------------+
   | ``operational_heel``    | Float scalar   | :math:`deg`   | Max heel (pitching) angle in operating conditions   |
   +-------------------------+----------------+---------------+-----------------------------------------------------+
   | ``max_survival_heel``   | Float scalar   | :math:`deg`   | Max heel (pitching) angle in parked conditions      |
   +-------------------------+----------------+---------------+-----------------------------------------------------+

Objectives
----------

Different anaylses will emphasize different metrics, requiring different
objective functions. Under the default assumption that the user wishes
to minimize cost and adhere to stability constraints, the objective
function would be total substructure cost (variable name,
``total_cost``) or mass (variable name, ``total_mass``).

Example
-------

.. _fig_exopt-spar:
.. figure::  /images/floatingse/spar-cost1.*
    :width: 30%
    :align: center

    Example of optimized spar.


.. _fig_exopt-semi:
.. figure::  /images/floatingse/semi-mass2.*
    :width: 40%
    :align: center

    Example of optimized semi.


.. _fig_exopt-tlp:
.. figure::  /images/floatingse/tlp-cost2.*
    :width: 30%
    :align: center

    Example of optimized TLP.
