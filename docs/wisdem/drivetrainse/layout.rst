******************
Layout and Inputs
******************

The direct-drive and geared drivetrain layouts are quite different, however both use the same set of user inputs.  This was intentional to simplify the user-input burden.  The common layout parameters are,

- Overhang (:math:`L_{overhang}`)
- Height from hub to tower top (:math:`H_{htt}`)
- Shaft tilt angle (:math:`\gamma`),
- Generator length (:math:`L_{generator}`)
- Distance between the hub flange and the first main bearing (:math:`L_{h1}`)
- Distance between the main bearings (:math:`L_{12}`)

Detailed diagrams of how these parameters set the sizing and layout of the drivetrain is shown in the two subsections below.

Direct-Drive Layout
========================

.. _fig_layout_pic:
.. figure::  /images/drivetrainse/layout_picture.*
    :width: 75%
    :align: center

    Direct-drive configuration geometry

The overall layout for the direct-drive configuration is shown in :numref:`fig_layout_pic` and :numref:`fig_layout_diagram`.  The hub connects to the low-speed shaft, which is a large diameter, hollow cylinder supported by two sets of main bearings attached to the nose, also called the turret.  The nose is affixed to the bedplate, which also has a circular cross-section, but follows an elliptical curve down to the tower attachment.  The total length, parallel to the ground, from the tower center line to the rotor apex, is considered the *overhang*.  The total height difference between those same two points in the *hub to tower top height*.  The outer-rotor of the generator also attaches to the low-speed shaft, and the corresponding stator attaches to the nose.

.. _fig_layout_diagram:
.. figure::  /images/drivetrainse/layout_diagram.*
    :width: 50%
    :align: center

    Direct-drive configuration layout diagram

The detailed parameters that specify the drivetrain layout are shown in :numref:`fig_layout_detail`.

.. _fig_layout_detail:
.. figure::  /images/drivetrainse/layout_detail.*
    :width: 100%
    :align: center

    Detailed direct-drive configuration with key user inputs and derived values.

In addition to the user-defined dimensions, the other values are derived in the following way,

.. math::
   L_{grs}	&= 0.5 L_{h1}\\
   L_{gsn}	&= L_{generator} - L_{grs} - L_{12}\\
   L_{2n}	&= 2 L_{gsn}\\
   L_{lss}	&= L_{12} + L_{h1}\\
   L_{nose}	&= L_{12} + L_{2n}\\
   L_{drive}    &= L_{h1} + L_{12} + L_{2n}\\
   L_{bedplate} &= L_{overhang} - L_{drive}\cos \gamma\\
   H_{bedplate} &= H_{htt} - L_{drive}\sin \gamma

Here the length from the hub flange to the generator rotor attachment, :math:`L_{grs}`, is assumed to be at the halfway point between the flange and the first main bearing, :math:`L_{h1}`.  Similarly, the distance between the second main bearing and the nose/turret interface with the bedplate, :math:`L_{2n}`, is twice the distance as that from the same interface to the generator stator attachment, :math:`L_{gsn}`.  After adding up the total length of the low speed shaft and nose/turret, the total drivetrain length from bedplate to hub can be determined.  Then, the bedplate dimensions are determined in order to meet the target overhang and hub-to-tower top height. To ensure that these layout dimensions are adequately satisfied during a design optimization, a constraint is enforced such that :math:`L_{bed} \geq 0.5 D_{top}`.

The user must also specify diameter and thickness values for the low speed shaft (:math:`D_{lss}` and :math:`t_{lss}`) and the nose/turret (:math:`D_{nose}` and :math:`t_{nose}`).  These can also be assigned as design variables to satisfy the constraints generated in the structural analysis.

The bedplate diagram is shown in :numref:`fig_layout_bedplate`, and follows an elliptical path from the tower top to the nose/turret attachment point.  The length and height of the bedplate (major and minor half axes of the ellipse) are determined from the input user dimensions.  The bedplate diameter also follows an elliptical progression from the tower top diameter at the bedplate base to the nose/turrent diameter at the top.  The wall thickness schedule is a user defined input, or can be designated as a design variable in order to meet structural constraints.

.. _fig_layout_bedplate:
.. figure::  /images/drivetrainse/layout_bedplate.*
    :width: 65%
    :align: center

    Direct-drive configuration layout diagram

The attachment of the generator stator to the nose/turret is shown in :numref:`fig_layout_stator`.  For the direct-drive configuration, we assume an outer rotor-inner stator, radial flux topology for a permanent magnet synchronous generator. The outer rotor layout facilitates a simple and rugged structure, easy manufacturing, short end windings, and better heat transfer between windings and teeth than an inner rotor configuration.

.. _fig_layout_stator:
.. figure::  /images/drivetrainse/layout_stator.*
    :width: 70%
    :align: center

    Direct-drive configuration layout diagram


Geared Layout
========================

The overall layout for the geared configuration is shown in :numref:`fig_geared_diagram`. The hub connects to the low-speed shaft, supported by two sets of main bearings.  The low speed shaft connects to the gearbox which converts the high-torque, low-rpm input into a low-torque, high-rpm output on the high speed shaft.  The high speed shaft feeds the generator, which is assumed to be a doubly-fed induction generator (DFIG).  The bedplate is a steel platform that sits atop two parallel I-beams to provide the structural support.  The bearings and the generator are assumed to be firmly attached to the bedplate.  The gearbox attaches to the nacelle platform atop the bedplate with a trunion.

.. _fig_geared_diagram:
.. figure::  /images/drivetrainse/geared_diagram.*
    :width: 75%
    :align: center

    Geared configuration layout diagram

The detailed parameters that specify the drivetrain geared are shown in :numref:`fig_geared_detail`.

.. _fig_geared_detail:
.. figure::  /images/drivetrainse/geared_detail.*
    :width: 75%
    :align: center

    Geared configuration layout diagram

In addition to the user-defined dimensions, the other values are derived in the following way,

.. math::
   \delta       &= 0.1\\
   L_{lss}      &= L_{12} + L_{h1} + \delta\\
   L_{drive}    &= L_{lss} + L_{gearbox} + L_{hss} + L_{generator}\\
   L_{bedplate} &= L_{drive} \cos \gamma \\
   H_{bedplate} &= H_{htt} - L_{drive} \sin \gamma

The dimension, :math:`\delta` is the space between the second main bearing and the gearbox attachment where the shrink disk lies.  This is assumed to be 0.1 meters.  The bedplate height is sized to ensure that the desired height from tower top to hub is obtained.  To achieve the desired overhang distance, the tower is centered at the exact overhang distance from the hub and a constraint is enforced such that the drivetrain length is sufficient to extend past the tower, :math:`L_{drive} \cos \gamma - L_{overhang} \geq 0.5 D_{top}`.
