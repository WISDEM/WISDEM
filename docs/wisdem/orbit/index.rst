ORBIT
=====

Overview
--------

The Offshore Renewables Balance of system and Installation Tool (ORBIT) is a model developed by the National Renewable Energy Lab (NREL) to study the cost and times associated with Offshore Wind Balance of System (BOS) processes.

ORBIT includes many different modules that can be used to model phases within the BOS process, split into design and installation. It is highly flexible and allows the user to define which phases are needed to model their project or scenario using ProjectManager.


Documentation
-------------

ORBIT maintains its own Github `repository <https://github.com/WISDEM/ORBIT>`_ and `documentation <https://orbit-nrel.readthedocs.io/en/latest/>`_ (for now).  When ORBIT tags a release, its code and tests are copied into the WISDEM project.  Included in this code base is an OpenMDAO API that WISDEM interfaces with to provide a complete offshore cost analysis capability.


Usage
_____

ORBIT can be easily used as a standalone module.  This can be done through WISDEM or by installing ORBIT from its own `repository <https://github.com/WISDEM/ORBIT>`_ as a separate project.  For examples and documentation on ORBIT usage, see its `Tutorial <https://orbit-nrel.readthedocs.io/en/latest/source/tutorial/index.html>`_ and `API <https://orbit-nrel.readthedocs.io/en/latest/source/api.html>`_ guides.  If accessing ORBIT through WISDEM, just be sure to modify the python `import` lines from

>>> import ORBIT

to

>>> import wisdem.orbit

