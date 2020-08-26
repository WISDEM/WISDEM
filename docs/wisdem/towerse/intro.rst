Introduction
------------

TowerSE is a systems engineering tower model for cylindrical shell wind turbine towers.
Wind, wave, soil, and structural analyses are designed to be modular.
Default implementations are provided, but all modules can be replaced with custom implementations.
Default implementations use beam finite element theory, simple wind profiles, cylinder drag theory, Eurocode and Germanischer Lloyd shell and global buckling methods.

TowerSE is implemented as an `OpenMDAO <http://openmdao.org/>`_ group.
The beam finite element code is implemented in C++ and the rest of the implementation is in Python.
All modules are linked in Python.
