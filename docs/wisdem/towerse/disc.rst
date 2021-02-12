
Discretization
===============

As a user, the key concept to understand in order to take full advantage of TowerSE is its discretization.  This is presented in some examples and bullet points below, for the standard land-based tower and the extension of that to include a monopile.

Land-Based Tower
-----------------

The tower assumed to be divided into sections ("cans"), where the outer diameter varies linearly from the base to the top of each section and the wall thickness is constant.  This is consistent with rolling a steel plate of constant thickness that is cut into a trapezoid and rolled into a tapered cylinder, or frustum.  Sections are created by the grid point nodes in the :code:`reference_axis` section of the yaml ontology or in the non-dimensional :code:`tower_s` coordinates that vary from 0 to 1 applied to :code:`tower_height` when using the TowerSE Python code directly.  This means that the outer diameter is specified at each node, but the wall thickness is specified at each section, so internally the wall thickness vector is always one element less than the diameter vector.  When using Python, this this done directly, with examples provided in the ``05_tower_monopile`` case.  When using the yaml ontology is used for input, there are some other considerations that users should be aware of:

- The tower height determined by the :code:`reference_axis` is re-scaled to meet the user-specified hub height.  The tower grid nodes are similarly stretched.
- The foundation height is taken from the bottom z-coordinate of the :code:`reference_axis`
- The section wall thickness is taken as the average of the grid point values specified in the :code:`internal_structure.layers.thicknenss` entry.  This can either be an accepted approximation, such as in the NREL 5-MW example, or accounted for by creating tiny sections at each true section interface where the thickness changes but the outer diameter stays the same, as the IEA 15-MW example does.

To demonstrate, with the following yaml input,

.. code-block:: yaml

    assembly:
        hub_height: 90.
        ...
    components:
        tower:
            outer_shape_bem:
                reference_axis: &ref_axis_tower
                    x:
                        grid: [0.0, 1.0]
                        values: [0.0, 0.0]
                    y:
                        grid: [0.0, 1.0]
                        values: [0.0, 0.0]
                    z:
                        grid: &grid_tower [0., 0.5, 1.]
                        values: [0., 40.0, 80.0]
                outer_diameter:
                    grid: *grid_tower
                    values: [6.0, 5.0, 4.0]
            internal_structure_2d_fem:
                reference_axis: *ref_axis_tower
                layers:
                    - name: tower_wall
                      material: steel
                      thickness:
                        grid: *grid_tower
                        values: [0.03, 0.02, 0.01]

The 80m tower would be re-scaled to 90m with two sections of 45m each, one with a wall thickness of 25mm and the other with a thickness of 15mm.  To be able too set the wall thickness directly, do instead something similar to:

.. code-block:: yaml

    components:
        tower:
            outer_shape_bem:
                reference_axis: &ref_axis_tower
                    z:
                        grid: &grid_tower [0., 0.5, 0.501, 1.]
                        values: [0., 40.0, 40.001, 80.0]
                outer_diameter:
                    grid: *grid_tower
                    values: [6.0, 5.0, 5.0, 4.0]
            internal_structure_2d_fem:
                reference_axis: *ref_axis_tower
                layers:
                    - name: tower_wall
                      material: steel
                      thickness:
                        grid: *grid_tower
                        values: [0.03, 0.03, 0.02, 0.02]


Offshore Tower with Monopile
-----------------------------

The monopile discretization adheres to the same pattern as the tower discretization, with some additional assumptions.  Chief among these is that the transition piece height, where the monopile mates with the tower, is taken as the bottom z-coordinate of the tower (same point as the tower foundation height).  The monopile :code:`reference_axis` is shifted such that this is always true.  From this transition point, the monopile extends into the water column and the beneath the sea floor according to the grid and length in the monopile :code:`reference_axis` (the :code:`suctionpile_depth` parameter is an output of with this approach). If the monopile does not reach the sea floor, the suction pile depth will be output as negative value so that it can be trapped as a design constraint.  For gravity-based foundations, the user should set the monopile length to meet the sea floor exactly.

Due to numerical limitations in the soil boundary condition representation, the submerged pile segment will be represented as a single section that terminates at the mudline (sea floor).  If the user specifies multiple sections in the submerged pile or a single section that extends into the water column, these will be forcibly altered to comply with this condition.

In the following example,

.. code-block:: yaml

    components:
        tower:
            outer_shape_bem:
                reference_axis:
                    z:
                        grid: [0., 0.5, 1.]
                        values: [20., 60.0, 100.0]
        monopile:
            outer_shape_bem:
                reference_axis:
                    z:
                        grid: [0., 0.3846, 0.8462, 1.0]
                        values: [-55.0, -30.0, 0.0, 10.0]
    env:
        water_depth: 30.0

The monopile grid would be shifted to meet the transition piece height of 20m, and the 65m monopile length would extend through the 30m water column to a 15m embedded pile depth in the soil.
