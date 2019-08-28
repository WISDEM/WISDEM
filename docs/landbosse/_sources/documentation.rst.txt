.. module:: ccblade

.. _interfaces-label:

Documentation
-------------

If you have a normal Python code there are better ways to do this with autodoc.  For that case, see an example at `<https://raw.githubusercontent.com/WISDEM/CCBlade/master/docs/documentation.rst>`_.  For OpenMDAO classes there aren't any autodoc plugins (yet), so there isn't much you can do.  Fortunately, the definition of the assemblies can in many cases serve as useful documentaiton (assuming you have done a good job documenting through desc tags, units, etc.)  You can dump this out with a literalinclude

.. literalinclude:: ../src/rotorse/rotor.py
    :language: python
    :start-after: RotorSE(Assembly)
    :end-before: def configure(self)
    :prepend: class RotorSE(Assembly):

Beyond that you should provide links to important modules in your code.  This will allow you to refer to them in other modules and the source will be linked.

Referenced RotorAero Modules
============================

.. module:: rotorse.rotoraero
.. class:: VarSpeedMachine
.. class:: FixedSpeedMachine
.. class:: RatedConditions
.. class:: AeroLoads
.. class:: GeomtrySetupBase
.. class:: AeroBase
.. class:: DrivetrainLossesBase
.. class:: PDFBase
.. class:: CDFBase

Referenced RotorAeroDefaults Modules
====================================


.. module:: rotorse.rotoraerodefaults
.. class:: GeometrySpline
.. class:: CCBladeGeometry
.. class:: CCBlade
.. class:: CSMDrivetrain
.. class:: WeibullCDF
.. class:: WeibullWithMeanCDF
.. class:: RayleighCDF
.. class:: RotorAeroVSVPWithCCBlade
.. class:: RotorAeroVSFPWithCCBlade
.. class:: RotorAeroFSVPWithCCBlade
.. class:: RotorAeroFSFPWithCCBlade

Referenced Rotor Modules
====================================

.. module:: rotorse.rotor
.. class:: BeamPropertiesBase
.. class:: StrucBase
.. class:: PreCompSections
.. class:: RotorTS




