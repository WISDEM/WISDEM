.. _geometry-label:

Geometry
========

This section describes the variables and methods used to parameterize
the substructure geometry in *FloatingSE*. Typically, substructure
designs have fallen into three classical regimes, which are shown in
:numref:`fig_archetype`, each of which attains static stability through
different physical mechanisms. A spar derives its stability from a deep
drafted ballast. A semisubmersible derives its stability from
distributed waterplane area, achieved with offset columns spread evenly
around a main column or central point. A tension leg platform (TLP) uses
taut mooring lines for its stability.


.. _fig_archetype:
.. figure::  figs/archetypes.png
    :width: 50%
    :align: center

    Three classical designs for floating turbine substructures.

Similar to :cite:`karimi2017`, care was taken to parameterize the substructure in a
general manner, so as to be able to use the same set of design variables
to describe spars, semisubmersibles, TLPs, and hybrids of those
archetypes. The intent is that this modular approach to substructure
definition will enable rapid analysis of the majority of designs
currently proposed by the floating wind development community, whether
classical or novel in nature. Furthermore, generalizing the substructure
definition also empowers the optimization algorithm to search a broad
tradespace more efficiently by moving fluidly from one region to
another.

With that intent in mind, the general configuration of a spar-type
substructure is shown in :numref:`fig_diagram`, with nomenclature
borrowed from the field of naval architecture. A semisubmersible
configuration would have a similar diagram, but with multiple offset
columns connected with pontoon elements. A TLP might look similar to a
spar or semisubmersible, with taut mooring lines instead of the catenary
ones shown.


.. _fig_diagram:
.. figure::  figs/diagram.png
    :width: 90%
    :align: center

    Geometry parameterization with common wind turbine and naval architecture conventions.
   

   
Tapered Cylinders (Vertical Frustums)
-------------------------------------

A number of typical floating substructure designs, such as the spar or
semisubmersible, contain vertically oriented columns. In *FloatingSE*,
these columns are assumed to have a circular cross-section making them,
formally, vertical frustums. These frustums are assumed to be
ring-stiffened to support the buckling loads inherent in a submerged
support structure. The number of columns, their geometry, and the ring
stiffeners are parameterized in the *FloatingSE* module according to the
diagrams in :numref:`fig_diagram`, :numref:`fig_column`, :numref:`fig_stiffenerCut`, and :numref:`fig_stiffenerZoom`. The main column is
assumed to be centered at :math:`(x=0, y=0)`, directly underneath the
turbine tower (note that off-centered turbines are not yet supported).
Other columns are referred to as *offset* columns, and are assumed to be
evenly spread around the main column. The material of the vertical
columns is currently assumed to be ASTM 992 steel. Future developments
will include the option to select one of multiple material options for
each section in each cylinder.

.. _fig_column:
.. figure::  figs/colGeom.png
    :width: 30%
    :align: center

    Vertical frustum geometry parameterization.

       
.. _fig_stiffenerCut:
.. figure::  figs/stiffenerCut.png
    :width: 30%
    :align: center

    Vertical frustum cross-section with stiffeners


.. _fig_stiffenerZoom:
.. figure::  figs/stiffenerZoom.png
    :width: 30%
    :align: center

    Vertical frustum stiffener geometry parameterization.
   
The variables that set the geometry of the main and offset columns are
listed in :numref:`tbl_mainvar`. Two additional variables are also
included that describe the placement of the offset columns within the
substructure configuration.

.. _tbl_mainvar:
.. table::
   Variables specifying the main column geometry within *FloatingSE*

   +--------------------------------+-------------------------------+-------------+----------------------------------------------------------------+
   | **Variable**                   | **Type**                      | **Units**   | **Description**                                                |
   +================================+===============================+=============+================================================================+
   | ``main_section_height``        | Float array (:math:`n_s`)     | :math:`m`   | Height of each section                                         |
   +--------------------------------+-------------------------------+-------------+----------------------------------------------------------------+
   | ``main_outer_diameter``        | Float array (:math:`n_s+1`)   | :math:`m`   | Diameter at each section node (linear lofting between)         |
   +--------------------------------+-------------------------------+-------------+----------------------------------------------------------------+
   | ``main_wall_thickness``        | Float array (:math:`n_s+1`)   | :math:`m`   | Wall thickness at each section node (linear lofting between)   |
   +--------------------------------+-------------------------------+-------------+----------------------------------------------------------------+
   | ``main_freeboard``             | Float scalar                  | :math:`m`   | Design height above waterline                                  |
   +--------------------------------+-------------------------------+-------------+----------------------------------------------------------------+
   | ``offset_section_height``      | Float array (:math:`n_s`)     | :math:`m`   | Height of each section                                         |
   +--------------------------------+-------------------------------+-------------+----------------------------------------------------------------+
   | ``offset_outer_diameter``      | Float array (:math:`n_s+1`)   | :math:`m`   | Diameter at each section node (linear lofting between)         |
   +--------------------------------+-------------------------------+-------------+----------------------------------------------------------------+
   | ``offset_wall_thickness``      | Float array (:math:`n_s+1`)   | :math:`m`   | Wall thickness at each section node (linear lofting between)   |
   +--------------------------------+-------------------------------+-------------+----------------------------------------------------------------+
   | ``offset_freeboard``           | Float scalar                  | :math:`m`   | Design height above waterline                                  |
   +--------------------------------+-------------------------------+-------------+----------------------------------------------------------------+
   | ``number_of_offset_columns``   | Integer scalar                |             | Number of offset columns in substructure (for spar set to 0)   |
   +--------------------------------+-------------------------------+-------------+----------------------------------------------------------------+
   | ``radius_to_offset_column``    | Float scalar                  | :math:`m`   | Centerline of main.column to centerline of offset column       |
   +--------------------------------+-------------------------------+-------------+----------------------------------------------------------------+


Discretization
~~~~~~~~~~~~~~

To allow for varying geometry parameters along the length of
substructure columns, the larger components are divided into sections.
The user may specify the number of overall sections, :math:`n_s` and the
geometry of each section. Some of the geometry parameters are tied to
the nodes that bracket each section, such as column diameter and wall
thickness, with linear variation between each node. Other parameters are
considered constant within each section, such as the spacing between
ring stiffeners. The number of sections should resemble the physical
number of cans or sections used in the manufacturing of the real
article.

Stiffeners
~~~~~~~~~~

The ring stiffener geometry is depicted in :numref:`fig_stiffenerCut`, and :numref:`fig_stiffenerZoom` with
geometry variables listed in :numref:`tbl_stiffvar`.

.. _tbl_stiffvar:

.. table::
   Variables specifying the stiffener geometry within *FloatingSE*.

   +---------------------------------------+-----------------------------+-------------+-----------------------------------------------+
   | **Variable**                          | **Type**                    | **Units**   | **Description**                               |
   +=======================================+=============================+=============+===============================================+
   | ``main_stiffener_web_height``         | Float array (:math:`n_s`)   | :math:`m`   | Stiffener web height for each section         |
   +---------------------------------------+-----------------------------+-------------+-----------------------------------------------+
   | ``main_stiffener_web_thickness``      | Float array (:math:`n_s`)   | :math:`m`   | Stiffener web thickness for each section      |
   +---------------------------------------+-----------------------------+-------------+-----------------------------------------------+
   | ``main_stiffener_flange_width``       | Float array (:math:`n_s`)   | :math:`m`   | Stiffener flange width for each section       |
   +---------------------------------------+-----------------------------+-------------+-----------------------------------------------+
   | ``main_stiffener_flange_thickness``   | Float array (:math:`n_s`)   | :math:`m`   | Stiffener flange thickness for each section   |
   +---------------------------------------+-----------------------------+-------------+-----------------------------------------------+
   | ``main_stiffener_spacing``            | Float array (:math:`n_s`)   | :math:`m`   | Stiffener spacing for each section            |
   +---------------------------------------+-----------------------------+-------------+-----------------------------------------------+


Material Properties
~~~~~~~~~~~~~~~~~~~

The material of the vertical columns is currently assumed to uniformly
be ASTM 992 steel. Future developments will include the option to select
one of multiple material options for each section in each cylinder.
Currently, to globally change to a different material, use the variables
listed in :numref:`tbl_materialvar`.

.. _tbl_materialvar:

.. table::
   Variables specifying the material properties within *FloatingSE*.

   +------------------------+----------------+------------------+-----------------------------------+
   | **Variable**           | **Type**       | **Units**        | **Description**                   |
   +========================+================+==================+===================================+
   | ``material_density``   | Float scalar   | :math:`kg/m^3`   | Mass density (assumed steel)      |
   +------------------------+----------------+------------------+-----------------------------------+
   | ``E``                  | Float scalar   | :math:`N/m^2`    | Young’s modulus (of elasticity)   |
   +------------------------+----------------+------------------+-----------------------------------+
   | ``G``                  | Float scalar   | :math:`N/m^2`    | Shear modulus                     |
   +------------------------+----------------+------------------+-----------------------------------+
   | ``yield_stress``       | Float scalar   | :math:`N/m^2`    | Elastic yield stress              |
   +------------------------+----------------+------------------+-----------------------------------+
   | ``nu``                 | Float scalar   |                  | Poisson’s ratio (:math:`\nu`)     |
   +------------------------+----------------+------------------+-----------------------------------+


Ballast
~~~~~~~

Stability of substructure columns with long drafts can be enhanced by
placing heavy ballast, such as magnetite iron ore, at their bottom
sections. The user can specify the density of the permanent ballast
added and the height of the ballast extent within the column. The
variables that govern the implementation of the permanent ballast and
bulkhead nodes are listed in :numref:`tbl_ballastvar`. Variable ballast,
as opposed to permanent ballast, is water that is added or removed above
the permanent ballast to achieve neutral buoyancy as the operating
conditions of the turbine change. A discussion of variable water balance
in the model is found in Section :ref:`static-label`.

.. _tbl_ballastvar:

.. table::
   Variables specifying the ballast and bulkheads within *FloatingSE*.

   +---------------------------------------+--------------------------------+------------------+-------------------------------------------------------+
   | **Variable**                          | **Type**                       | **Units**        | **Description**                                       |
   +=======================================+================================+==================+=======================================================+
   | ``permanent_ballast_density``         | Float scalar                   | :math:`kg/m^3`   | Mass density of ballast material                      |
   +---------------------------------------+--------------------------------+------------------+-------------------------------------------------------+
   | ``main_permanent_ballast_height``     | Float scalar                   | :math:`m`        | Height above keel for permanent ballast               |
   +---------------------------------------+--------------------------------+------------------+-------------------------------------------------------+
   | ``main_bulkhead_thickness``           | Float vector (:math:`n_s+1`)   | :math:`m`        | Internal bulkhead thicknesses at section interfaces   |
   +---------------------------------------+--------------------------------+------------------+-------------------------------------------------------+
   | ``offset_permanent_ballast_height``   | Float scalar                   | :math:`m`        | Height above keel for permanent ballast               |
   +---------------------------------------+--------------------------------+------------------+-------------------------------------------------------+
   | ``offset_bulkhead_thickness``         | Float vector (:math:`n_s+1`)   | :math:`m`        | Internal bulkhead thicknesses at section interfaces   |
   +---------------------------------------+--------------------------------+------------------+-------------------------------------------------------+
   

Buoyancy Tanks (and Heave Plates)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Buoyancy tanks are modeled as a collar around the column and are not
subject the same taper or connectivity constraints as the frustum
sections. They therefore offer added buoyancy without incurring as much
structural mass or cost. Moreover, they can also serve to augment the
heave added mass like a plate. The variables that govern their
configuration are listed in :numref:`tbl_buoytankvar`. In addition to their
diameter and height, the user can adjust the location of the buoyancy
tank from the column base to the top. Buoyancy tanks can be added to
either the main and/or offset columns.

.. _tbl_buoytankvar:

.. table::
   Variables specifying the buoyancy tank geometry within *FloatingSE*.

   +-------------------------------------+----------------+-------------+-----------------------------------------------------------------------------+
   | **Variable**                        | **Type**       | **Units**   | **Description**                                                             |
   +=====================================+================+=============+=============================================================================+
   | ``main_buoyancy_tank_diameter``     | Float scalar   | :math:`m`   | Diameter of buoyancy tank / heave plate on main.column                      |
   +-------------------------------------+----------------+-------------+-----------------------------------------------------------------------------+
   | ``main_buoyancy_tank_height``       | Float scalar   | :math:`m`   | Height of buoyancy tank / heave plate on main.column                        |
   +-------------------------------------+----------------+-------------+-----------------------------------------------------------------------------+
   | ``main_buoyancy_tank_location``     | Float scalar   |             | Location of buoyancy tank along main.column (0 for bottom, 1 for top)       |
   +-------------------------------------+----------------+-------------+-----------------------------------------------------------------------------+
   | ``offset_buoyancy_tank_diameter``   | Float scalar   | :math:`m`   | Diameter of buoyancy tank / heave plate on offset column                    |
   +-------------------------------------+----------------+-------------+-----------------------------------------------------------------------------+
   | ``offset_buoyancy_tank_height``     | Float scalar   | :math:`m`   | Height of buoyancy tank / heave plate on offset column                      |
   +-------------------------------------+----------------+-------------+-----------------------------------------------------------------------------+
   | ``offset_buoyancy_tank_location``   | Float scalar   |             | Location of buoyancy tank along offliary column (0 for bottom, 1 for top)   |
   +-------------------------------------+----------------+-------------+-----------------------------------------------------------------------------+



Pontoons and Support Structure
------------------------------

Many substructure designs include the use of pontoons that form a truss
to connect the different components, usually columns, together. In this
model, all of the pontoons are assumed to have the identical thin-walled
tube cross section and made of the same material as the rest of the
substructure. The truss configuration and the parameterization of the
pontoon elements is based on the members shown in :numref:`fig_pontoon`
with lettered labels. The members are broken out into the upper and
lower rings connecting the offset columns (:math:`B` and :math:`D`,
respectively), the upper and lower main-to-offset connections (:math:`A`
and :math:`C`, respectively), the lower-base to upper-offset cross
members (:math:`E`), and the V-shaped cross members between offset
columns (:math:`F`). The variables that drive this parameterization are
listed in :numref:`tbl_trussvar`.

.. _fig_pontoon:
.. figure::  figs/semi.png
    :width: 50%
    :align: center

    Parameterization of truss elements in substructure.


.. _tbl_trussvar:

.. table::
   Variables specifying the pontoon and truss geometry within *FloatingSE*.

   +---------------------------------+----------------+----------------------------+-------------+------------------------------------------------------+
   | **Variable**                    | **Type**       |   :numref:`fig_pontoon`    | **Units**   | **Description**                                      |
   +=================================+================+============================+=============+======================================================+
   | ``pontoon_outer_diameter``      | Float scalar   |                            | :math:`m`   | Diameter of all pontoon/truss elements               |
   +---------------------------------+----------------+----------------------------+-------------+------------------------------------------------------+
   | ``pontoon_wall_thickness``      | Float scalar   |                            | :math:`m`   | Thickness of all pontoon/truss elements              |
   +---------------------------------+----------------+----------------------------+-------------+------------------------------------------------------+
   | ``main_pontoon_attach_lower``   | Float scalar   |                            | :math:`m`   | Lower z-coordinate on main.where truss attaches      |
   +---------------------------------+----------------+----------------------------+-------------+------------------------------------------------------+
   | ``main_pontoon_attach_upper``   | Float scalar   |                            | :math:`m`   | Upper z-coordinate on main.where truss attaches      |
   +---------------------------------+----------------+----------------------------+-------------+------------------------------------------------------+
   | ``upper_attachment_pontoons``   | Integer scalar | A                          |             | Upper main.to-offset connecting pontoons             |
   +---------------------------------+----------------+----------------------------+-------------+------------------------------------------------------+
   | ``lower_attachment_pontoons``   | Integer scalar | C                          |             | Lower main.to-offset connecting pontoons             |
   +---------------------------------+----------------+----------------------------+-------------+------------------------------------------------------+
   | ``cross_attachment_pontoons``   | Integer scalar | E                          |             | Lower-Upper main.to-offset connecting cross braces   |
   +---------------------------------+----------------+----------------------------+-------------+------------------------------------------------------+
   | ``upper_ring_pontoons``         | Integer scalar | B                          |             | Upper ring of pontoons connecting offset columns     |
   +---------------------------------+----------------+----------------------------+-------------+------------------------------------------------------+
   | ``lower_ring_pontoons``         | Integer scalar | D                          |             | Lower ring of pontoons connecting offset columns     |
   +---------------------------------+----------------+----------------------------+-------------+------------------------------------------------------+
   | ``outer_cross_pontoons``        | Integer scalar | F                          |             | Auxiliary ring connecting V-cross braces             |
   +---------------------------------+----------------+----------------------------+-------------+------------------------------------------------------+
   


Mooring Lines
-------------

The mooring system is described by the number of lines, their geometry,
and their interface to the substructure. The mooring diameter is set by
the user and determines the breaking load and stiffness of the chain,
via correlation, described in Section :ref:`theory-label`. The mooring lines
attach to the substructure at the *fairlead* distance below the water
plane, as shown in :numref:`fig_diagram`. The lines can attach directly
to a substructure column or at a some offset from the outer shell. Note
that bridle connections are not yet implemented in the model. The
mooring lines attach to the sea floor at a variable distance, the anchor
radius, from the substructure centerline, also set by the user.

By default, the mooring system is assumed to use a steel chain with drag
embedment anchors. Other mooring available for selection are nylon,
polyester, steel wire rope (IWRC) and fiber-core wire rope. The only
alternative anchor type is currently suction pile anchors, but there are
plans to include gravity anchors as well. The standard configuration for
TLPs is the use of taut nylon mooring lines with suction-pile anchors.
The variables that control the mooring system properties are listed in
:numref:`tbl_moorvar`.

.. _tbl_moorvar:
.. table::
   Variables specifying the mooring system within *FloatingSE*.

   +-------------------------------------+------------------+-------------+-------------------------------------------------------------------+
   | **Variable**                        | **Type**         | **Units**   | **Description**                                                   |
   +=====================================+==================+=============+===================================================================+
   | ``number_of_mooring_connections``   | Integer scalar   |             | Number of evenly spaced mooring connection points                 |
   +-------------------------------------+------------------+-------------+-------------------------------------------------------------------+
   | ``mooring_lines_per_connection``    | Integer scalar   |             | Number of mooring lines at each connection point                  |
   +-------------------------------------+------------------+-------------+-------------------------------------------------------------------+
   | ``mooring_diameter``                | Float scalar     | :math:`m`   | Diameter of mooring line/chain                                    |
   +-------------------------------------+------------------+-------------+-------------------------------------------------------------------+
   | ``mooring_line_length``             | Float scalar     | :math:`m`   | Total unstretched line length of mooring line                     |
   +-------------------------------------+------------------+-------------+-------------------------------------------------------------------+
   | ``fairlead_location``               | Float scalar     |             | Fractional length from column bottom to mooring line attachment   |
   +-------------------------------------+------------------+-------------+-------------------------------------------------------------------+
   | ``fairlead_offset_from_shell``      | Float scalar     | :math:`m`   | Offset from shell surface for mooring attachment                  |
   +-------------------------------------+------------------+-------------+-------------------------------------------------------------------+
   | ``anchor_radius``                   | Float scalar     | :math:`m`   | Distance from centerline to sea floor landing                     |
   +-------------------------------------+------------------+-------------+-------------------------------------------------------------------+
   | ``mooring_type``                    | Enumerated       |             | Options are CHAIN, NYLON, POLYESTER, FIBER, or IWRC               |
   +-------------------------------------+------------------+-------------+-------------------------------------------------------------------+
   | ``anchor_type``                     | Enumerated       |             | Options are SUCTIONPILE or DRAGEMBEDMENT                          |
   +-------------------------------------+------------------+-------------+-------------------------------------------------------------------+



Mass and Cost Scaling
---------------------

The mass of all components in the modeled substructure is captured
through calculation of each components’ volume and multiplying by its
material density. This applies to the frustum shells, the ring
stiffeners, the permanent ballast, the pontoons, and the mooring lines.
However, the model also acknowledges that the modeled substructure is
merely an approximation of an actual substructure and various secondary
elements are not captured. These include ladders, walkways, handles,
finishing, paint, wiring, etc. To account for these features en masse,
multipliers of component masses are offered as parameters for the user
as well. Capital cost for all substructure components except the mooring
system is assumed to be a linear scaling of the components masses. For
the mooring system, cost is dependent on the tension carrying capacity
of the line, which itself is an empirical function of the diameter.
Default values for all mass and cost scaling factors are found in :numref:`tbl_factors`. Cost factors are especially difficult to estimate given
the proprietary nature of commercial cost data, so cost rates and
estimates should be considered notional.


.. _tbl_factors:

.. table::
   Variables specifying the mass and cost scaling within *FloatingSE*.

   +--------------------------------+----------------+------------------+---------------+----------------------------------------------------------+
   | **Variable**                   | **Type**       | **Units**        | **Default**   | **Description**                                          |
   +================================+================+==================+===============+==========================================================+
   | ``bulkhead_mass_factor``       | Float scalar   |                  | 1.0           | Scaling for unaccounted bulkhead mass                    |
   +--------------------------------+----------------+------------------+---------------+----------------------------------------------------------+
   | ``ring_mass_factor``           | Float scalar   |                  | 1.0           | Scaling for unaccounted stiffener mass                   |
   +--------------------------------+----------------+------------------+---------------+----------------------------------------------------------+
   | ``shell_mass_factor``          | Float scalar   |                  | 1.0           | Scaling for unaccounted shell mass                       |
   +--------------------------------+----------------+------------------+---------------+----------------------------------------------------------+
   | ``column_mass_factor``         | Float scalar   |                  | 1.05          | Scaling for unaccounted column mass                      |
   +--------------------------------+----------------+------------------+---------------+----------------------------------------------------------+
   | ``outfitting_mass_fraction``   | Float scalar   |                  | 0.06          | Fraction of additional outfitting mass for each column   |
   +--------------------------------+----------------+------------------+---------------+----------------------------------------------------------+
   | ``ballast_cost_rate``          | Float scalar   | :math:`USD/kg`   | 100           | Cost factor for ballast mass                             |
   +--------------------------------+----------------+------------------+---------------+----------------------------------------------------------+
   | ``tapered_col_cost_rate``      | Float scalar   | :math:`USD/kg`   | 4,720         | Cost factor for column mass                              |
   +--------------------------------+----------------+------------------+---------------+----------------------------------------------------------+
   | ``outfitting_cost_rate``       | Float scalar   | :math:`USD/kg`   | 6,980         | Cost factor for outfitting mass                          |
   +--------------------------------+----------------+------------------+---------------+----------------------------------------------------------+
   | ``mooring_cost_rate``          | Float scalar   | :math:`USD/kg`   | depends       | Mooring cost factor (depends on diam and material)       |
   +--------------------------------+----------------+------------------+---------------+----------------------------------------------------------+
   | ``pontoon_cost_rate``          | Float scalar   | :math:`USD/kg`   | 6.5           | Cost factor for pontoons                                 |
   +--------------------------------+----------------+------------------+---------------+----------------------------------------------------------+


