Environment
-----------

Environment contains shared wind, wave, and soil models.

.. currentmodule:: wisdem.commonse.environment

Wind
====

This module defines a wind speed profile at locations `z`, all wind speeds below `z_0` are 0.
The parameters `Uref` and `zref` allow for scaling of a profile shape.
Specific implementations of this base component include :class:`PowerWind` and :class:`LogWind`.
PowerWind assumes a power-law distribution of wind speeds of the form

.. math:: U(z) = U_{ref} \left(\frac{z - z_0}{z_{ref} - z_0}\right)^\alpha

The logarithmic profile is of the form

.. math::

    U = U_{ref} \left[\frac{\log\left(\frac{z - z_0}{z_{roughness}}\right) }{\log\left(\frac{z_{ref} - z_0}{z_{roughness}}\right)}\right]


.. class:: WindBase
.. class:: PowerWind
.. class:: LogWind


Wave
====

Hydrodynamic speed distributions are estimated using linear wave theory (:class:`LinearWaves`). According to linear wave theory, the maximum horizontal velocity of a wave is given as

.. math:: U_{current}  = \omega \frac{h}{2} \frac{\cosh(k(z+D))}{\sinh(kD)} \cos(\omega t)

and the corresponding maximum acceleration is

.. math:: A_{current} = \omega U_{current}


.. Morrison's equation predicts the hydrodynamic loads on a cylinder with three terms. These terms correspond to a drag force and the inertial forces due to wave motion and cylinder motion. The current analysis neglects the motion of the tower. With that assumption the two remaining forces per unit length are given as

.. .. math:: {{F_i}^\prime_{max}} = \frac{\pi}{4} \rho_{water} A_{current} c_m d^2

.. .. math:: {{F_d}^\prime_{max}} = \frac{1}{2} \rho_{water} U_{current}^2 c_d  d

.. Drag coefficient is estimated in the same manner as described for the wind loads.

.. class:: WaveBase
.. class:: LinearWaves

Soil
====

The soil is assumed to not contribute any inertial or applied forces and only affects the stiffness of the foundation.
The user may specify directions which are considered rigid.
For the other directions, effective spring constants are estimated based on the soil properties (:class:`TowerSoil`).
A simple textbook model is used in this implementation [1]_.
The model allows for computation of an effective spring constant for all six degrees of freedom, each computed as a function of the shear modulus and Poisson's ratio of the soil.

For example:

.. math:: k_z = \frac{4 G r}{1- \nu} \left( 1 + 0.6(1-\nu)\frac{h}{r}  \right)

where h is the depth of the foundation below the soil.


.. class:: SoilBase
.. class:: TowerSoil


.. [1] Suresh C Arya, Michael O’Neil, and George Pincus. Design of Structures and Foundations for Vibrating Machines. BPR cumulative. Gulf Publishing Co, June 1979.
