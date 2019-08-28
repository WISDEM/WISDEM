.. _optimization-label:

Optimization
============

Executing *FloatingSE* by hand is sufficient to explore some simple
one-off or comparison analyses between a few runs. OpenMDAO provides
extensive optimization capability, which can give yield richer and more
insightful analyses.

Methodology
-----------

For a full substructure optimization, we begin with the formulation of a
constrained, nonlinear single-objective optimization problem with
mixed-integer design variables,

.. math::

   \begin{array}{ll}
     \min & f\left({\mathbf{x}}\right)\\
     \text{subject to} & {\mathbf{g}}\left({\mathbf{x}}\right) \leq 0,\\
     \text{and}& {\mathbf{x}} \in {\mathbf{X}} \\
     \end{array}

where,

-  :math:`{\mathbf{x}}` is a vector of :math:`n` *design variables*, the
   variables that are adjusted in order to find the optimal solution
   (see Table [tbl:designvar]);

-  :math:`f({\mathbf{x}})` is the nonlinear *objective function*, the
   metric to be minimized by the optimization algorithm;

-  :math:`{\mathbf{g}} ({\mathbf{x}})` is the vector of *inequality
   constraints*, the set of conditions that the solution must satisfy
   (see Table [tbl:constraints]). There are no equality constraints;

-  :math:`{\mathbf{X}}` is the design variable *bounds*, the bracket of
   allowable design variable values.

Note that this problem statement imposes no requirements on the types of
variables in :math:`{\mathbf{x}}`. A mixed-integer solution is desired,
where some design variables are continuous (:math:`x \in {\mathbb{R}}`)
and others are discrete variables that can only take integer values
(:math:`x \in
{\mathbb{Z}}`). An example of an integer design variable in this
application is the number of offset columns or the number of mooring
line connections.

Gradient-Based versus Derivative-Free Algorithms
------------------------------------------------

Derivative-free optimization algorithms are preferable for substructure
optimization problems for a few reasons, despite their known performance
drawbacks in terms of wall-clock time. First, to do a complete
configuration optimization of the substructure, a mixed-integer capable
algorithm is required. No gradient-based optimization algorithm is
capable of handling these types of variables directly (unless a rounding
approximation is used). A genetic algorithm, properly designed, can
support mixed-integer variables for a global design space optimization.

Another reason for the selection of derivative-free algorithms is that
the *FloatingSE* uses a number of third-party, black box tools or
algorithms that do not come with analytical gradients. This includes
`Frame3DD <http://frame3dd.sourceforge.net>`_, `MAP++ <https://nwtc.nrel.gov/MAP>`_, and some of the API 2U procedures that rely on roots of
nonlinear equations. Thus, gradient-based optimization algorithms would
be forced to use finite difference approximations around these tools at
the very least. However, derivatives approximated with finite
differences are expensive to compute accurately. If computed
inaccurately, for the sake of reducing computational time, finite
difference derivatives can easily lead an optimization algorithm astray,
especially in highly nonlinear or tightly constrained regions of the
design space. This is another reason for the use of derivative-free
algorithms, even when conducting local neighborhood design space
optimization and/or sensitivity studies.

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
into the optimization. As described in Section :ref:`intro-label`, this is a
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
.. figure::  figs/spar-cost1.png
    :width: 30%
    :align: center

    Example of optimized spar.

       
.. _fig_exopt-semi:
.. figure::  figs/semi-mass2.png
    :width: 40%
    :align: center

    Example of optimized semi.


.. _fig_exopt-tlp:
.. figure::  figs/tlp-cost2.png
    :width: 30%
    :align: center

    Example of optimized TLP.
