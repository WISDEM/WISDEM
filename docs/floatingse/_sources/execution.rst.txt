.. _execution-label:

Execution
=========

Executing *FloatingSE* requires additional inputs beyond those of the
geometry definition described above in Section :ref:`geometry-label`. Other user
inputs for the metocean and loading environment, and the operational
constraints, are required to evaluate the total mass, cost, and code
compliance. These variables are described below, followed by the
sequence of *FloatingSE* calculations.

User Inputs
-----------

The remaining input variables beyond those listed in Section :ref:`geometry-label`
describe the metocean environment, the turbine geometry and loading, as
well as loading constraints. The loading constraints are more relevant
in the context of optimization, so they are described later in Section
:ref:`optimization-label`. The remaining variables are explained below.

Metocean Environment
~~~~~~~~~~~~~~~~~~~~

The metocean condition is specified by the water and wind environment.
Geographical dependence is chiefly captured by specifying the water
depth. The wave loading is parameterized by setting the significant wave
height, periodicity, and mean current speed (if any). Physical
properties of the water, specifically the density and viscosity, are
also captured. The user may also set the added mass constant used in
Equation [eqn:morison] (Morison’s equation). All of these variables
cumulatively are in :numref:`tbl_metocean`.

.. _tbl_metocean:
.. table::
   Variables specifying the wave environment within *FloatingSE*.

   +-------------------------+----------------+------------------+------------------------------+
   | **Variable**            | **Type**       | **Units**        | **Description**              |
   +=========================+================+==================+==============================+
   | ``water_depth``         | Float scalar   | :math:`m`        | Distance to sea floor        |
   +-------------------------+----------------+------------------+------------------------------+
   | ``Hs``                  | Float scalar   | :math:`m`        | Significant wave height      |
   +-------------------------+----------------+------------------+------------------------------+
   | ``T``                   | Float scalar   | :math:`s`        | Wave period                  |
   +-------------------------+----------------+------------------+------------------------------+
   | ``cm``                  | Float scalar   |                  | Added mass coefficient       |
   +-------------------------+----------------+------------------+------------------------------+
   | ``Uc``                  | Float scalar   | :math:`m/s`      | Mean current speed           |
   +-------------------------+----------------+------------------+------------------------------+
   | ``z0``                  | Float scalar   | :math:`m`        | z-coordinate of water line   |
   +-------------------------+----------------+------------------+------------------------------+
   | ``water_density``       | Float scalar   | :math:`kg/m^3`   | Density of water             |
   +-------------------------+----------------+------------------+------------------------------+
   | ``main.waveLoads.mu``   | Float scalar   | :math:`kg/m/s`   | Viscosity of water           |
   +-------------------------+----------------+------------------+------------------------------+



Not that the some of these variables, especially the one setting water
viscosity, are awkwardly named. This is due to the need for OpenMDAO to
have only one formal independent variable in any high-level group. Since
*FloatingSE* is intended to be married with other WISDEM modules in
simulation of a full turbine, the design load cases must be specified at
this higher-level group. Thus, *FloatingSE* cannot declare independent
variables relevant to the load cases on its own, and must therefore use
the variable names as written in the module code.

The wind profile is specified by the user with a reference height and
velocity. From there, wind speeds at other heights are determined using
a shear exponent, for power-law scaling, althrough logarithmic scaling
is available as well. The physical properties of the air at the turbine
location must also be set. As with the water-relevant variables, the
comment of awkwardly labeled variables applies. Cumulatively, all of the
wind-related variables are in :numref:`tbl_windvar`.

.. _tbl_windvar:
.. table::
   Variables specifying the wind environment within *FloatingSE*.

   +--------------------------+----------------+------------------+------------------------------------+
   | **Variable**             | **Type**       | **Units**        | **Description**                    |
   +==========================+================+==================+====================================+
   | ``Uref``                 | Float scalar   | :math:`m/s`      | Wind reference speed               |
   +--------------------------+----------------+------------------+------------------------------------+
   | ``zref``                 | Float scalar   | :math:`m`        | Wind reference height              |
   +--------------------------+----------------+------------------+------------------------------------+
   | ``shearExp``             | Float scalar   |                  | Shear exponent in wind power law   |
   +--------------------------+----------------+------------------+------------------------------------+
   | ``beta``                 | Float scalar   | :math:`deg`      | Wind beta angle                    |
   +--------------------------+----------------+------------------+------------------------------------+
   | ``main.windLoads.rho``   | Float scalar   | :math:`kg/m^3`   | Density of air                     |
   +--------------------------+----------------+------------------+------------------------------------+
   | ``main.windLoads.mu``    | Float scalar   | :math:`kg/m/s`   | Viscosity of air                   |
   +--------------------------+----------------+------------------+------------------------------------+
   


As mentioned in Section :ref:`theory-label`, only a single load case is
currently supported. Future development will allow for optimization
around multiple load cases, each with their own metocean environment.

Turbine Description
~~~~~~~~~~~~~~~~~~~

To successfully analyze the entire load path through the substructure,
the user, or other WISDEM modules, must input the geometry and loading
of the wind turbine above the substructure. The next component after the
substructure in the load path is the tower. As a long, slender column,
the tower geometry parameterization is similar to that of the
substructure columns and has similar variable names, listed in :numref:`tbl_towervar`.

.. _tbl_towervar:
.. table::
   Variables specifying the tower geometry within *FloatingSE*.

   +-------------------------------+-------------------------------+-------------+----------------------------------------------------------------+
   | **Variable**                  | **Type**                      | **Units**   | **Description**                                                |
   +===============================+===============================+=============+================================================================+
   | ``hub_height``                | Float scalar                  | :math:`m`   | Length from tower main.to top (not including freeboard)        |
   +-------------------------------+-------------------------------+-------------+----------------------------------------------------------------+
   | ``tower_section_height``      | Float array (:math:`n_s`)     | :math:`m`   | Length of each tower section                                   |
   +-------------------------------+-------------------------------+-------------+----------------------------------------------------------------+
   | ``tower_outer_diameter``      | Float array (:math:`n_s+1`)   | :math:`m`   | Diameter at each tower section node (linear lofting between)   |
   +-------------------------------+-------------------------------+-------------+----------------------------------------------------------------+
   | ``tower_wall_thickness``      | Float array (:math:`n_s+1`)   | :math:`m`   | Diameter at each tower section node (linear lofting between)   |
   +-------------------------------+-------------------------------+-------------+----------------------------------------------------------------+
   | ``tower_buckling_length``     | Float scalar                  | :math:`m`   | Tower buckling reinforcement spacing                           |
   +-------------------------------+-------------------------------+-------------+----------------------------------------------------------------+
   | ``tower_outfitting_factor``   | Float scalar                  |             | Scaling for unaccounted tower mass in outfitting               |
   +-------------------------------+-------------------------------+-------------+----------------------------------------------------------------+



At the top of the load path, above the tower, is the rotor nacelle
assembly (RNA). The RNA includes the blades, hub, shaft(s), gearbox,
generator, nacelle housing, etc. All of these components are addressed
separately across multiple WISDEM modules, but *FloatingSE* only
requires aggregate summations of the mass properties, forces, and
moments. These cumulative variables are in :numref:`tbl_rnavar`.

.. _tbl_rnavar:
.. table::
   Variables specifying the RNA geometry and loads within *FloatingSE*.

   +------------------+-------------------+------------------+-------------------------------------------------------+
   | **Variable**     | **Type**          | **Units**        | **Description**                                       |
   +==================+===================+==================+=======================================================+
   | ``rna_mass``     | Float scalar      | :math:`kg`       | Mass                                                  |
   +------------------+-------------------+------------------+-------------------------------------------------------+
   | ``rna_I``        | Float array (6)   | :math:`kg/m^2`   | Moment of intertia (xx,yy,zz,xy,xz,yz)                |
   +------------------+-------------------+------------------+-------------------------------------------------------+
   | ``rna_cg``       | Float array (3)   | :math:`m`        | Offset of RNA center of mass from tower top (x,y,z)   |
   +------------------+-------------------+------------------+-------------------------------------------------------+
   | ``rna_force``    | Float array (3)   | :math:`N`        | Net force acting on RNA (x,y,z)                       |
   +------------------+-------------------+------------------+-------------------------------------------------------+
   | ``rna_moment``   | Float array (3)   | :math:`N*m`      | Net moment acting on RNA (x,y,z)                      |
   +------------------+-------------------+------------------+-------------------------------------------------------+
   | ``yaw``          | Float scalar      |                  | Turbine yaw angle                                     |
   +------------------+-------------------+------------------+-------------------------------------------------------+
   


Simulation Flow
---------------

Once the input variables are completely specified, *FloatingSE* executes
the analysis of the substructure. Conceptually, the simulation is
organized by the flowchart in :numref:`fig_floatingse`. From a more
technical perspective, *FloatingSE* is an OpenMDAO Group, so the
analysis sequence is broken down by the sub-groups and sub-components in
the order that they are listed in Table [tbl:exec]. In an OpenMDAO
group, sub-groups and components are given prefixes to aid in referring
to specific variables. The prefixes used in *FloatingSE* are also listed
in :numref:`tbl_exec`. These prefixes also appear in some of the variable
names listed above (e.g., ``main.waveLoads.mu``) and will appear in the
discussion of optimization constraints in Section :ref:`optimization-label`.


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



.. _fig_floatingse:
.. figure::  figs/floatingse.png
    :width: 90%
    :align: center

    Conceptual diagram of *FloatingSE* execution.

Outputs are accumulated in each sub-group or component, and they either
become inputs to other components, become constraints for optimization
problems, become design variables for optimization problems, or can
simply be ignored. Currently, a single execution of FloatingSE takes
only a handful of seconds on a modern laptop computer.

Examples
--------

As mentioned in Section :ref:`package-label`, two files are meant as analysis
starting points, :file:`sparInstance.py` and :file:`semiInstance.py`. These
files are encoded with default starting configurations (from :cite:`OC3` and :cite:`OC4`,
respectively). They also have ready configurations for optimization with
design variables, constraints, and solvers options. However, due to the
flexible and object-oriented approach to programming these capabilities,
some complexity was introduced into the code and it is difficult to use
as simple examples. For demonstration purposes, the spar and
semisubmersible examples from OC3 and OC4 are also provided at the
bottom of the :file:`floating.py` file, and are copied below. A
visualization of the geometries described by these examples is shown in
:numref:`fig_initial-spar` and :numref:`fig_initial-semi`.


.. _fig_initial-spar:
.. figure::  figs/spar-initial.png
    :width: 75%
    :align: center

    Spar example in *FloatingSE* taken from OC3 :cite:`OC3` project.

       
.. _fig_initial-semi:
.. figure::  figs/semi-initial.png
    :width: 75%
    :align: center

    Semi example in *FloatingSE* taken from OC4 :cite:`OC4` project.
    
