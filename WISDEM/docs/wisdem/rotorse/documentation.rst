.. _documentation-label:

Documentation
-------------

The following inputs and outputs are defined for RotorSE:

.. literalinclude:: ../../../wisdem/rotorse/rotor.py
    :language: python
    :start-after: """rotor model"""
    :end-before: # --- Rotor Aero & Power ---


Referenced Rotor Geometry Modules
=================================

.. module:: wisdem.rotorse.rotor_geometry
.. class:: ReferenceBlade
.. class:: NREL5MW
.. class:: DTU10MW
.. class:: BladeGeometry
.. class:: Location
.. class:: TurbineClass
.. class:: RotorGeometry

Referenced Rotor Aero-Power Modules
===================================

.. module:: wisdem.rotorse.rotor_aeropower
.. class:: DrivetrainLossesBase
.. class:: SetupRunVarSpeed
.. class:: RegulatedPowerCurve
.. class:: AEP
.. class:: CSMDrivetrain
.. class:: OutputsAero
.. class:: RotorAeroPower

Referenced Rotor Structure Modules
==================================

.. module:: wisdem.rotorse.rotor_structure
.. class:: BeamPropertiesBase
.. class:: StrucBase
.. class:: ResizeCompositeSection
.. class:: PreCompSections
.. class:: BladeCurvature
.. class:: CurveFEM
.. class:: RotorWithpBEAM
.. class:: DamageLoads
.. class:: TotalLoads
.. class:: TipDeflection
.. class:: BladeDeflection
.. class:: RootMoment
.. class:: MassPropertiesExtremeLoads
.. class:: GustETM
.. class:: SetupPCModVarSpeed
.. class:: ConstraintsStructures
.. class:: OutputsStructures
.. class:: RotorStructure

Referenced Rotor Modules
========================

.. module:: wisdem.rotorse.rotor
.. class:: RotorSE

Referenced Examples
===================

.. module:: wisdem.rotorse.examples.rotorse_example1
.. class:: RotorSE_Example1
.. module:: wisdem.rotorse.examples.rotorse_example2
.. class:: RotorSE_Example2
.. module:: wisdem.rotorse.examples.rotorse_example3
.. class:: RotorSE_Example3


