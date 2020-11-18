.. _theory:

Theory
------

The tower class can be used to model a wind turbine tower or a tower/monopile configuration.
No distinction is made between the tower and foundation, and so the term tower will be used throughout to refer to the entire structure.
The current implementation assumes that the tower has cylindrical shell sections.
The underlying analysis has the capability to handle general sections should such be desired.
Dynamics for floating turbines are not included in TowerSE.

Theory for the finite element code is available at the website: `Frame3DD <http://frame3dd.sourceforge.net/>`_.
The RNA (rotor/nacelle/assembly) affects the stiffness of the structure and top loads.
It is assumed that the RNA is a rigid body with respect to the tower modes.
The RNA mass properties are transferred to the tower top using the generalized parallel axis theorem.
Two different buckling approaches are implemented.
A shell buckling method from Eurocode :cite:`Eurocode` and a global buckling method from Germanischer Lloyd :cite:`GL2005`.
The implementation of the Eurocode buckling is modified slightly so as to produce continuously differentiable output.
Since the tower is typically reinforced at shorter distances than the full tower length, the user may specify the reinforcement length.
Hoop stress is estimated using the Eurocode method.
Axial and shear stress calculations are done for cylindrical shell sections and are combined with hoop stress into a von Mises stress.
Fatigue uses supplied damage equivalent moments, which are converted to stress for the given geometry.
Using the stress, and inputs for the number of cycles and slope of the S-N curve allows for a damage calculation.

Computation of drag loads is done assuming drag over a smooth circular cylinder as a function of Reynolds number :cite:`Roshko`.
Wave drag loads are computed using Morrison's equation.
Morrison's equation predicts the hydrodynamic loads on a cylinder with three terms.
These terms correspond to a drag force and the inertial forces due to wave motion and cylinder motion.
The current analysis neglects the motion of the tower.
With that assumption the two remaining forces per unit length are given as

.. math:: {{F_i}^\prime_{max}} = \frac{\pi}{4} \rho_{water} A_{current} c_m d^2

.. math:: {{F_d}^\prime_{max}} = \frac{1}{2} \rho_{water} U_{current}^2 c_d  d

The calculation of the resulting drag is separated from the actual velocity distributions, which are handled in the commonse.environment module.
The environment model provides default implementations for power-law wind profiles, logarithmic-law wind profiles, and linear wave theory.
A textbook model is used for soil stiffness properties :cite:`Arya1979`.

.. bibliography:: ../../references.bib
   :filter: docname in docnames
