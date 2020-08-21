AeroelasticSE
=============

.. only:: html

This python code forms a wrapper for the NWTC codes `FAST <https://wind.nrel.gov/designcodes/simulators/fast/>`_ and TurbSim.

The core code involved is the NWTC code `FAST <https://wind.nrel.gov/designcodes/simulators/fast/>`_.
The AeroelasticSE modules provide access to it in various ways, as detailed below:

* (:mod:`AeroelasticSE.runFAST`)--run FAST by parsing and modifying template file inputs.
* (:mod:`AeroelasticSE.runTurbSim`)--run TurbSim by parsing and modifying template file inputs.
* (:mod:`AeroelasticSE.FAST_component`)--run FAST as an openMDAO component.
* (:mod:`AeroelasticSE.FusedFAST`)--run FAST via generic run cases using the fusedwind framework. 
* (:mod:`AeroelasticSE.FSTTemplate_runner`)--run FAST via the fused wind framework's VariableTree based interface.
* (:mod:`AeroelasticSE.iecApp`)--run and process large numbers of cases using the fusedwind framework and the template based interface

In addition, there are a couple of helper modules:

* FusedFASTrunCase.py
* PeregrineClusterAllocator.py

The tutorial section is a good place to start learning about how to use these FAST wrappers.  These consist primarily of annotated
sections from the test code that is part of the modules themselves.

       .. rubric:: Table of Contents

.. toctree::

    tutorial
    documentation

