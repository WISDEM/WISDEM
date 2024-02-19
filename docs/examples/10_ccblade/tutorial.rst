.. _ccblade_tutorial-label:

.. currentmodule:: wisdem.ccblade

10. CCBlade Examples
--------------------

Three examples are shown below.
The first is a complete setup for the :ref:`NREL 5-MW model <5MW-example>`, the second shows how to model blade :ref:`precurvature <curvature-example>`, and the final shows how to get the provided :ref:`analytic gradients <gradient-example>`.
Each complete example is also included within the ``WISDEM/examples/10_ccblade`` directory.

.. contents:: List of Examples
   :depth: 2

.. _5MW-example:

NREL 5-MW
^^^^^^^^^

One example of a CCBlade application is the simulation of the NREL 5-MW reference model's aerodynamic performance.
First, define the geometry and atmospheric properties.

.. literalinclude:: ../../../examples/10_ccblade/example.py
    :start-after: # 1 ---
    :end-before: # 1 ---

Airfoil aerodynamic data is specified using the :class:`CCAirfoil` class.
Rather than use the default constructor, this example uses the special constructor designed to read AeroDyn files directly :meth:`CCAirfoil.initFromAerodynFile`.

.. literalinclude:: ../../../examples/10_ccblade/example.py
    :start-after: # 2 ---
    :end-before: # 2 ---


Next, construct the CCBlade object.

.. literalinclude:: ../../../examples/10_ccblade/example.py
    :start-after: # 3 ---
    :end-before: # 3 ---


Evaluate the distributed loads at a chosen set of operating conditions.

.. literalinclude:: ../../../examples/10_ccblade/example.py
    :start-after: # 4 ---
    :end-before: # 4 ---


Plot the flapwise and lead-lag aerodynamic loading

.. literalinclude:: ../../../examples/10_ccblade/example.py
    :start-after: # 5 ---
    :end-before: # 5 ---


as shown in :numref:`Figure %s <distributed-fig>`.

.. _distributed-fig:

.. figure:: /images/ccblade/distributedAeroLoads.*
    :width: 5in
    :align: center

    Flapwise and lead-lag aerodynamic loads along blade.


To get the power, thrust, and torque at the same conditions (in both absolute and coefficient form), use the :meth:`evaluate <ccblade.CCBlade.evaluate>` method.
This is generally used for generating power curves so it expects :code:`array_like` input.
For this example a list of size one is used.

.. literalinclude:: ../../../examples/10_ccblade/example.py
    :start-after: # 6 ---
    :end-before: # 6 ---


The result is

>>> CP = [ 0.46488096]
>>> CT = [ 0.76926398]
>>> CQ = [ 0.0616323]

Note that the outputs are Numpy arrays (of length 1 for this example).
To generate a nondimensional power curve (:math:`\lambda` vs :math:`c_p`):

.. literalinclude:: ../../../examples/10_ccblade/example.py
    :start-after: # 7 ---
    :end-before: # 7 ---


:numref:`Figure %s <cp-fig>` shows the resulting plot.

.. _cp-fig:

.. figure:: /images/ccblade/cp.*
    :width: 5in
    :align: center

    Power coefficient as a function of tip-speed ratio.


CCBlade provides a few additional options in its constructor.
The other options are shown in the following example with their default values.

.. code-block:: python

    # create CCBlade object
    rotor = CCBlade(r, chord, theta, af, Rhub, Rtip, B, rho, mu,
                    precone, tilt, yaw, shearExp, hubHt, nSector
                    tiploss=True, hubloss=True, wakerotation=True, usecd=True, iterRe=1)

The parameters :code:`tiploss` and :code:`hubloss` toggle Prandtl tip and hub losses respectively. The parameter :code:`wakerotation` toggles wake swirl (i.e., :math:`a^\prime = 0`).
The parameter :code:`usecd` can be used to disable the inclusion of drag in the calculation of the induction factors (it is always used in calculations of the distributed loads).
However, doing so may cause potential failure in the solution methodology (see :cite:`Ning2013A-simple-soluti`).
In practice, it should work fine, but special care for that particular case has not yet been examined, and the default implementation allows for the possibility of convergence failure.
All four of these parameters are :code:`True` by default.
The parameter :code:`iterRe` is for advanced usage.
Referring to :cite:`Ning2013A-simple-soluti`, this parameter controls the number of internal iterations on the Reynolds number.
One iteration is almost always sufficient, but for high accuracy in the Reynolds number :code:`iterRe` could be set at 2.
Anything larger than that is unnecessary.

.. _curvature-example:

Precurve
^^^^^^^^

CCBlade can also simulate blades with precurve.
This is done by using the :code:`precurve` and :code:`precurveTip` parameters.
These correspond precisely to the :code:`r` and :code:`Rtip` parameters.
Precurve is defined as the position of the blade reference axis in the x-direction of the :ref:`blade-aligned coordinate system <azimuth_blade_coord>` (r is the position in the z-direction of the same coordinate system).
Presweep can be specified in the same manner, by using the :code:`presweep` and :code:`presweepTip` parameters (position in blade-aligned y-axis).
Generally, it is advisable to set :code:`precone=0` for blades with precurve.
There is no loss of generality in defining the blade shape, and including a nonzero precone would change the rotor diameter in a nonlinear way. As an example, a downwind machine with significant curvature could be simulated using:

.. literalinclude:: ../../../examples/10_ccblade/precurve.py
    :start-after: # 1 ---
    :end-before: # 1 ---

The shape of the blade is seen in :numref:`Figure %s <shape-fig>`.
Note that the radius of the blade is preserved because we have set the precone angle to zero.

.. _shape-fig:

.. figure:: /images/ccblade/rotorshape.*
    :width: 5in
    :align: center

    Profile of an example (highly) precurved blade.

.. _gradient-example:

Gradients
^^^^^^^^^

CCBlade optionally provides analytic gradients of every output with respect to all design variables.
This is accomplished using an adjoint method (direct method is identical because there is only one state variable at each blade section).
Partial derivatives are provided by `Tapenade <http://www-tapenade.inria.fr:8080/tapenade/index.jsp>`_ and hand calculations.
Starting with the previous example for the NREL 5-MW reference model we add the keyword value :code:`derivatives=True` in the constructor.

.. literalinclude:: ../../../examples/10_ccblade/gradients.py
    :start-after: # 3 ---
    :end-before: # 3 ---

Now when we ask for the distributed loads, we also get the gradients.
The gradients are returned as a dictionary containing 2D arrays.
These can be accessed as follows:

.. literalinclude:: ../../../examples/10_ccblade/gradients.py
    :start-after: # 5 ---
    :end-before: # 5 ---

Even though many of the matrices are diagonal, the full Jacobian is returned for consistency.
We can compare against finite differencing as follows (with a randomly chosen station along the blade):

.. literalinclude:: ../../../examples/10_ccblade/gradients.py
    :start-after: # 7 ---
    :end-before: # 7 ---

The output is:

>>> (analytic) dNp_i/dr_i = 107.680395098
>>> (fin diff) dNp_i/dr_i = 107.680370762

Similarly, when we compute thrust, torque, and power we also get the gradients (for either the non-dimensional or dimensional form).
The gradients are also returned as a dictionary containing 2D Jacobians.

.. literalinclude:: ../../../examples/10_ccblade/gradients.py
    :start-after: # 6 ---
    :end-before: # 6 ---

Let us compare the derivative of power against finite differencing for one of the scalar quantities (precone):

.. literalinclude:: ../../../examples/10_ccblade/gradients.py
    :start-after: # 8 ---
    :end-before: # 8 ---

>>> (analytic) dP/dprecone = -4585.70729746
>>> (fin diff) dP/dprecone = -4585.71072668


Finally, we compare the derivative of power against finite differencing for one of the vector quantities (r) at a random index:

.. literalinclude:: ../../../examples/10_ccblade/gradients.py
    :start-after: # 9 ---
    :end-before: # 9 ---

>>> (analytic) dP/dr_i = 848.368037506
>>> (fin diff) dP/dr_i = 848.355994992

For more comprehensive comparison to finite differencing, see the unit tests contained in ``wisdem/test/test_ccblade/test_gradients.py``.
