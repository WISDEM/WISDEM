Introduction
------------

TowerSE is a systems engineering tower model for cylindrical shell wind turbine towers.
Wind, wave, soil, and structural analyses are designed to be modular.
Default implementations are provided, but all modules can be replaced with custom implementations.
Default implementations use beam finite element theory, simple wind profiles, cylinder drag theory, Eurocode and DNVGL shell and global buckling methods.

TowerSE is implemented as an `OpenMDAO <http://openmdao.org/>`_ group.
The underlying beam finite element code is `Frame3DD <http://frame3dd.sourceforge.net/>`_., implemented with a python wrapper in the pyFrame3DD module.
