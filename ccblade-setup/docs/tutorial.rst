.. _tutorial-label:

.. currentmodule:: ccblade

Tutorial
--------

Two examples are shown below.  The first is a complete setup for the NREL 5-MW model, and the second shows how to model blade precurvature using CCBlade.

NREL 5-MW
^^^^^^^^^

One example of a CCBlade application is the simulation of the NREL 5-MW reference model's aerodynamic performance.  First, define the geometry and atmospheric properties.

.. literalinclude:: examples/example.py
    :start-after: # 1 ---
    :end-before: # 1 ---

Airfoil aerodynamic data is specified using the :class:`CCAirfoil` class.  Rather than use the default constructor, this example uses the special constructor designed to read AeroDyn files directly :meth:`CCAirfoil.initFromAerodynFile`.

.. literalinclude:: examples/example.py
    :start-after: # 2 ---
    :end-before: # 2 ---


Next, construct the CCBlade object.

.. literalinclude:: examples/example.py
    :start-after: # 3 ---
    :end-before: # 3 ---


Evaluate the distributed loads at a chosen set of operating conditions.

.. literalinclude:: examples/example.py
    :start-after: # 4 ---
    :end-before: # 4 ---


Plot the flapwise and lead-lag aerodynamic loading

.. literalinclude:: examples/example.py
    :start-after: # 5 ---
    :end-before: # 5 ---


as shown in :num:`Figure #distributed-fig`.

.. _distributed-fig:

.. figure:: /images/distributedAeroLoads.*
    :width: 5in
    :align: center

    Flapwise and lead-lag aerodynamic loads along blade.


To get the power, thrust, and torque at the same conditions (in both absolute and coefficient form), use the :meth:`evaluate <ccblade.CCBlade.evaluate>` method.  This is generally used for generating power curves so it expects ``array_like`` input.  For this example a list of size one is used.

.. literalinclude:: examples/example.py
    :start-after: # 6 ---
    :end-before: # 6 ---


The result is

>>> CP = [ 0.48329808]
>>> CT = [ 0.7772276]
>>> CQ = [ 0.06401299]

Note that the outputs are numpy arrays (of length 1 for this example).  To generate a nondimensional power curve (:math:`\lambda` vs :math:`c_p`):

.. literalinclude:: examples/example.py
    :start-after: # 7 ---
    :end-before: # 7 ---


:num:`Figure #cp-fig` shows the resulting plot.

.. _cp-fig:

.. figure:: /images/cp.*
    :width: 5in
    :align: center

    Power coefficient as a function of tip-speed ratio.


CCBlade provides a few additional options in its constructor.  The other options are shown in the following example with their default values.

.. code-block:: python

    # create CCBlade object
    rotor = CCBlade(r, chord, theta, af, Rhub, Rtip, B, rho, mu,
                    precone, tilt, yaw, shearExp, hubHt, nSector
                    tiploss=True, hubloss=True, wakerotation=True, usecd=True, iterRe=1)

The parameters :code:`tiploss` and :code:`hubloss` toggle Prandtl tip and hub losses repsectively. The parameter :code:`wakerotation` toggles wake swirl (i.e., :math:`a^\prime = 0`).  The parameter :code:`usecd` can be used to disable the inclusion of drag in the calculation of the induction factors (it is always used in calculations of the distributed loads).  However, doing so may cause potential failure in the solution methodology (see :cite:`Ning2013A-simple-soluti`).  In practice, it should work fine, but special care for that particular case has not yet been examined, and the default implementation allows for the possibility of convergence failure.  All four of these parameters are ``True`` by default.  The parameter :code:`iterRe` is for advanced usage.  Referring to :cite:`Ning2013A-simple-soluti`, this parameter controls the number of internal iterations on the Reynolds number.  One iteration is almost always sufficient, but for high accuracy in the Reynolds number :code:`iterRe` could be set at 2.  Anything larger than that is unnecessary.


Precurve
^^^^^^^^

CCBlade can also simulate blades with precurve.  This is done by using the ``precone`` parameter and passing in an array rather than just a float.  The values in the array correspond to the angle of precurve along the blade using the same sign conventions as for :ref:`precone <azimuth_blade_coord>` For example, a downwind machine (negative precurve) with significant curvature could be simulated using:

.. literalinclude:: examples/precurve.py
    :start-after: # 1 ---
    :end-before: # 1 ---

The shape of the blade is seen in :num:`Figure #shape-fig`.  Note that the radius of the blade is *not* 63 m (it is now 58.16 m), but the blade length is preserved at 63 m.  The precurve angles are treated as (local) rotations in the same manner as the precone angle is.

.. _shape-fig:

.. figure:: /images/rotorshape.*
    :width: 5in
    :align: center

    Profile of an example (highly) precurved blade.
