.. _tutorial-label:

Tutorial
--------

Two examples are included in this section.  The first example provides the sectional properties and assumes a linear variation between the sections (see matching :ref:`constructor <beam-linear-label>`).  The example simulates a blade for the NREL 5-MW reference model.  The second example uses the convenience constructor for a beam with cylindrical shell sections (see matching :ref:`constructor <beam-tower-label>`).  The example simulates the tower for the NREL 5-MW refernece model.  A third :ref:`constructor <beam-tower-label>` is available for more advanced usage and allows for arbitrary polynomial variation in section properties.  This advanced constructor is not demonstrated in these examples, but details are available in the :ref:`documentation <documentation-label>.

Linear Variation
^^^^^^^^^^^^^^^^

This example simulates a rotor blade from the NREL 5-MW reference model in pBEAM.  First, the relevant modules are imported.

.. literalinclude:: /examples/pbeam/blade-example.py
    :start-after: # 1 ---
    :end-before: # 1 ---

Next, we define the stiffness and inertial properties.  The stiffness and inertial properties can be computed from the structural layout of the blade using a code like `PreComp <http://wind.nrel.gov/designcodes/preprocessors/precomp/>`_.  Section properties are defined using the :ref:`SectionData <section-data-linear-label>` class.

.. literalinclude:: /examples/pbeam/blade-example.py
    :start-after: # 2 ---
    :end-before: # 2 ---


Distributed loads can be computed from an aerodynamics code like CCBlade.  This example includes only distributed loads, which are defined in :ref:`Loads <loads-distributed-label>`.

.. literalinclude:: /examples/pbeam/blade-example.py
    :start-after: # 3 ---
    :end-before: # 3 ---

The tip/base data is defined with a free end in :ref:`TipData <tip-label>` and a rigid base for :ref:`BaseData <base-label>`.  The blade object is then assembled using the :ref:`beam-label` constructor that assumes linear variation in properties between sections.

.. literalinclude:: /examples/pbeam/blade-example.py
    :start-after: # 4 ---
    :end-before: # 4 ---

The constructed blade object can now be used for various computations.  For example, the mass and the first five natural frequencies

.. literalinclude:: /examples/pbeam/blade-example.py
    :start-after: # 5 ---
    :end-before: # 5 ---

which result in

>>> mass = 17170.0189
>>> natural freq = [ 0.90910346  1.13977516  2.81855826  4.23836926  6.40037864]

:num:`Figure #dy-fig` shows a plot of the blade displacement in the second principal direction generated from the following commands.

.. literalinclude:: /examples/pbeam/blade-example.py
    :start-after: # 6 ---
    :end-before: # 6 ---

.. _dy-fig:

.. figure:: /images/pbeam/dy.*
    :width: 5in
    :align: center

    Blade deflection along span in flapwise direction.

:num:`Figure #strain-fig` shows a plot of the strain along the blade at the location of maximum airfoil thickness on both the pressure and suction side of the airfoil.

.. literalinclude:: /examples/pbeam/blade-example.py
    :start-after: # 7 ---
    :end-before: # 7 ---

.. _strain-fig:

.. figure:: /images/pbeam/strain.*
    :width: 5in
    :align: center

    Strain along span at location of maximum airfoil thickness.


Cylindrical Shell Sections
^^^^^^^^^^^^^^^^^^^^^^^^^^

This example simulates the tower from the NREL 5-MW reference model in pBEAM. First, the relevant modules are imported

.. literalinclude:: /examples/pbeam/tower-example.py
    :start-after: # 1 ---
    :end-before: # 1 ---

The basic tower geometry is defined

.. literalinclude:: /examples/pbeam/tower-example.py
    :start-after: # 2 ---
    :end-before: # 2 ---

and then discretized so that it is defined at the end of every element (for convenience, a discretized definition could be supplied up front).

.. literalinclude:: /examples/pbeam/tower-example.py
    :start-after: # 3 ---
    :end-before: # 3 ---

The cylindrical shell model only allows for isotropic :ref:`material properties <material-label>`.

.. literalinclude:: /examples/pbeam/tower-example.py
    :start-after: # 4 ---
    :end-before: # 4 ---

:ref:`Distributed loads <loads-distributed-label>` in this example come from wind loading and the tower's weight.

.. literalinclude:: /examples/pbeam/tower-example.py
    :start-after: # 5 ---
    :end-before: # 5 ---

Contributions from the rotor-nacelle-assembly (RNA) include mass, moments of inertia, and transfered forces/moments.  These are added using the :ref:`TipData <tip-label>` class.

.. literalinclude:: /examples/pbeam/tower-example.py
    :start-after: # 6 ---
    :end-before: # 6 ---

The base of the tower is assumed to be rigidly mounted in this example.  This corresponds to :ref:`BaseData <base-label>` being initialized with infinite stiffness in all directions.

.. literalinclude:: /examples/pbeam/tower-example.py
    :start-after: # 7 ---
    :end-before: # 7 ---

Finally, the tower object is created using the :ref:`cylindrical shell constructor <beam-tower-label>`.

.. literalinclude:: /examples/pbeam/tower-example.py
    :start-after: # 8 ---
    :end-before: # 8 ---

Relevant properties can now be computed from this object in the same manner as in the previous example.

.. A few relevant computations are shown below.

.. .. literalinclude:: /examples/pbeam/tower-example.py
..     :start-after: # 9 ---
..     :end-before: # 9 ---


.. [TODO: add a general polynomial example]
