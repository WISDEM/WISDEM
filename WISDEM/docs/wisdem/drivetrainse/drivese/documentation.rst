.. _documentation-label:

.. currentmodule:: drivese.drive

Documentation
--------------

.. only:: latex

    An HTML version of this documentation is available which is better formatted for reading the code documentation and contains hyperlinks to the source code.


Turbine component sizing models for hub and drivetrain components are described along with mass-cost models for the full set of turbine components from the rotor to tower and foundation.

Documentation for DriveSE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following inputs and outputs are defined for DriveSE using the three-point suspension configuration:

.. literalinclude:: ../src/drivese/drive.py
    :language: python
    :start-after: Drive3pt(Assembly)
    :end-before: def configure(self)
    :prepend: class Drive3pt(Assembly):

.. module:: drivese.drive
.. class:: Drive3pt

The following inputs and outputs are defined for DriveSE using the four-point suspension configuration:

.. literalinclude:: ../src/drivese/drive.py
    :language: python
    :start-after: Drive4pt(Assembly)
    :end-before: def configure(self)
    :prepend: class Drive4pt(Assembly):

.. module:: drivese.drive
.. class:: Drive4pt

Implemented Base Model
=========================
.. module:: drivewpact.drive
.. class:: NacelleBase

Referenced Sub-System Modules 
==============================
.. module:: drivese.drivese_components
.. class:: LowSpeedShaft_drive
.. class:: LowSpeedShaft_drive4pt
.. class:: LowSpeedShaft_drive3pt
.. class:: MainBearing_drive
.. class:: SecondBearing_drive
.. class:: Gearbox_drive
.. class:: HighSpeedSide_drive
.. class:: Generator_drive
.. class:: Bedplate_drive
.. class:: AboveYawMassAdder_drive
.. class:: YawSystem_drive
.. class:: NacelleSystemAdder_drive


Supporting Functions
=====================
.. module:: drivese.drivese_utils
.. function:: seed_bearing_table
.. function:: fatigue_for_bearings
.. function:: fatigue2_for_bearings
.. function:: resize_for_bearings
.. function:: get_rotor_mass
.. function:: get_L_rb

.. currentmodule:: drivese.hub

Documentation for HubSE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following inputs and outputs are defined for HubWPACT:

.. literalinclude:: ../src/drivese/hub.py
    :language: python
    :start-after: HubSE(Assembly)
    :end-before: def configure(self)
    :prepend: class HubSE(Assembly):

.. module:: drivese.hub
.. class:: HubSE

Implemented Base Model
=========================
.. module:: drivewpact.hub
.. class:: HubBase

Referenced Sub-System Modules 
==============================
.. module:: drivese.hub
.. class:: Hub_drive
.. class:: PitchSystem_drive
.. class:: Spinner_drive
.. module:: drivewpact.hub
.. class:: HubSystemAdder

