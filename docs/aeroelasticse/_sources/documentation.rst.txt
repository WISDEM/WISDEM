.. module:: aeroelasticse

.. _aeroelasticse:

Documentation
-------------
The important modules in the AeroelasticSE suite of codes include:


Run FAST
========

.. automodule:: AeroelasticSE.runFAST
  :members:
  :special-members:

Run TurbSim
===========

.. automodule:: AeroelasticSE.runTurbSim
   :members:
   :special-members:

FAST openMDAO Components
========================

.. automodule:: AeroelasticSE.FAST_component
   :members:
   :special-members:

Fused FAST
==========

.. automodule:: AeroelasticSE.FusedFAST
   :members:
   :special-members:

FAST Template Runner
========================

.. automodule:: AeroelasticSE.FSTTemplate_runner
   :members:
   :special-members:


Run IEC
=======

This code can run and process a whole study of many run cases, including in parallel on a 
compute cluster.

.. automodule:: AeroelasticSE.iecApp
   :members:
   :special-members:


FAST VariableTree Runner (FSTVT_runner)
==========================================

.. automodule:: AeroelasticSE.FSTVT_runner
   :members:
   :special-members:

Supporting Modules for FASTVT_runner
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: AeroelasticSE.FAST_VT.FST_reader
.. class:: FstInputBase
.. class:: FstInputReader

.. automodule:: AeroelasticSE.FAST_VT.FST_vartrees
.. class:: SimpleWind
.. class:: ADAirfoilPolar
.. class:: ADAirfoil
.. class:: ADBladeAeroGeometry
.. class:: ADAero
.. class:: FstBladeStrucGeometry
.. class:: FstTowerStrucGeometry
.. class:: FstPlatformModule
.. class:: FstModel

.. automodule:: AeroelasticSE.FAST_VT.FST_writer
.. class:: FstInputBuilder
.. class:: FUSEDWindInputBuilder
.. class:: FstInputWRiter

.. automodule:: AeroelasticSE.FAST_VT.FST_vartrees_out
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

.. automodule:: AeroelasticSE.FAST_VT.FST_wrapper
.. class:: FstExternalCode
.. class:: FstWrapper

.. automodule:: AeroelasticSE.FAST_VT.FSTVT_runIEC
.. class:: FUSEDFSTCaseRunner



