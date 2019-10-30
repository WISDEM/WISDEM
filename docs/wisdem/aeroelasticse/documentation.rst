.. module:: wisdem.aeroelasticse

.. _aeroelasticse:

Documentation
-------------
The important modules in the AeroelasticSE suite of codes include:


Run FAST
========

.. automodule:: wisdem.aeroelasticse.runFAST
  :members:
  :special-members:

Run TurbSim
===========

.. automodule:: wisdem.aeroelasticse.runTurbSim
   :members:
   :special-members:

FAST openMDAO Components
========================

.. automodule:: wisdem.aeroelasticse.FAST_component
   :members:
   :special-members:

Fused FAST
==========

.. automodule:: wisdem.aeroelasticse.FusedFAST
   :members:
   :special-members:

FAST Template Runner
========================

.. automodule:: wisdem.aeroelasticse.FSTTemplate_runner
   :members:
   :special-members:


Run IEC
=======

This code can run and process a whole study of many run cases, including in parallel on a 
compute cluster.

.. automodule:: wisdem.aeroelasticse.iecApp
   :members:
   :special-members:


FAST VariableTree Runner (FSTVT_runner)
==========================================

.. automodule:: wisdem.aeroelasticse.FSTVT_runner
   :members:
   :special-members:

Supporting Modules for FASTVT_runner
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: wisdem.aeroelasticse.FAST_VT.FST_reader
.. class:: FstInputBase
.. class:: FstInputReader

.. automodule:: wisdem.aeroelasticse.FAST_VT.FST_vartrees
.. class:: SimpleWind
.. class:: ADAirfoilPolar
.. class:: ADAirfoil
.. class:: ADBladeAeroGeometry
.. class:: ADAero
.. class:: FstBladeStrucGeometry
.. class:: FstTowerStrucGeometry
.. class:: FstPlatformModule
.. class:: FstModel

.. automodule:: wisdem.aeroelasticse.FAST_VT.FST_writer
.. class:: FstInputBuilder
.. class:: FUSEDWindInputBuilder
.. class:: FstInputWRiter

.. automodule:: wisdem.aeroelasticse.FAST_VT.FST_vartrees_out
.. class:: WindMotionsOut
.. class:: BladeMotionsOut
.. class:: HubNacelleMotionsOut
.. class:: TowerSupportMotionsOut
.. class:: BladeLoadsOut
.. class:: HubNacelleLoadsOut
.. class:: TowerSupportLoadsOut
.. class:: WaveMotionsOut
.. class:: DOFOut
.. class:: FstOutput

.. automodule:: wisdem.aeroelasticse.FAST_VT.FST_wrapper
.. class:: FstExternalCode
.. class:: FstWrapper

.. automodule:: wisdem.aeroelasticse.FAST_VT.FSTVT_runIEC
.. class:: FUSEDFSTCaseRunner



