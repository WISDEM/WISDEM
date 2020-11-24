.. _ccblade_theory:

Theory
------

.. note::

    Only an overview of the theory is included here; details can be found in Ning :cite:`Ning2013A-simple-soluti`.

The rotor aerodynamic analysis is based on blade element momentum (BEM) theory.
Using BEM theory in a gradient-based rotor optimization problem can be challenging because of occasional convergence difficulties of the BEM equations.
The standard approach to solving the BEM equations is to arrange the equations as functions of the axial and tangential induction factors and solve the fixed-point problem:

.. math::

    (a, a^\prime) = f_{fp}(a, a^\prime)

using either fixed-point iteration, Newton's method, or a related fixed-point algorithm.
An alternative approach is to use nonlinear optimization to minimize the sum of the squares of the residuals of the induction factors (or normal and tangential loads).
Although these approaches are generally successful, they suffer from instabilities and failure to converge in some regions of the design space.
Thus, they require increased complexity and/or heuristics (but may still not converge).

The new BEM methodology transforms the two-variable, fixed-point problem into an equivalent one-dimensional root-finding problem.
This is enormously beneficial as methods exist for one-dimensional root-finding problems that are guaranteed to converge as long as an appropriate bracket can be found.
The key insight to this reduction is to use the local inflow angle :math:`\phi` and the magnitude of the inflow velocity :math:`W` as the two unknowns in specifying the inflow conditions, rather than the traditional axial and tangential induction factors (see :numref:`Figure %s <inflow-fig>`).

.. _inflow-fig:

.. figure:: /images/ccblade/inflow.*
    :width: 5in
    :align: center

    Parameters specifying inflow conditions of a rotating blade section.


This approach allows the BEM equations to be reduced to a one-dimensional residual function as a function of :math:`\phi`:

.. math::
    R(\phi) = \frac{\sin\phi}{1-a(\phi)} - \frac{\cos\phi}{\lambda_r (1+a^\prime(\phi))}  = 0



:numref:`Figure %s <f-fig>` shows the typical behavior of :math:`R(\phi)` over the range :math:`\phi \in (0, \pi/2]`.
Almost all solutions for wind turbines fall within this range (for the provable convergence properties to be true, solutions outside of this range must also be considered).
The referenced paper :cite:`Ning2013A-simple-soluti` demonstrates through mathematical proof that the methodology will always find a bracket to a zero of :math:`R(\phi)` without any singularities in the interior.
This proof, along with existing proofs for root-finding methods like Brent's method :cite:`Brent1971An-algorithm-wi`, implies that a solution is guaranteed.
Furthermore, not only is the solution guaranteed, but it can be found efficiently and in a continuous manner.
This behavior allows the use of gradient-based algorithms to solve rotor optimization problems much more effectively than with traditional BEM solution approaches.


.. _f-fig:

.. figure:: /images/ccblade/f.*
    :width: 5in
    :align: center

    Residual function of BEM equations using new methodology.
    Solution point is where :math:`R(\phi) = 0`.



Any corrections to the BEM method can be used with this methodology (e.g., finite number of blades and skewed wake) as long as the axial induction factor can be expressed as a function of :math:`\phi` (either explicitly or through a numerical solution).
CCBlade chooses to include both hub and tip losses using Prandtl's method :cite:`glauert1935airplane` and a high-induction factor correction by Buhl :cite:`Buhl2005A-new-empirical`.
Drag is included in the computation of the induction factors.
However, all of these options can be toggled on or off.

Gradients are computed using a direct/adjoint (identical for one state variable) method.
Let us define a functional (e.g., distributed load at one section), as:

.. math::
    f = N^\prime(x_i, \phi)

Using the chain rule the total derivatives are given as

.. math::
    \frac{df}{dx_i} = \frac{\partial f}{\partial x_i} - \frac{\partial f}{\partial \phi} \frac{\partial R}{\partial x_i} / \frac{\partial R}{\partial \phi}


.. bibliography:: ../../references.bib
   :filter: docname in docnames
