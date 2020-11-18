.. _precomp:

-------------------------------
PreComp
-------------------------------

PreComp :cite:`Bir2005` implements a modified classic laminate theory combined with a shear-flow approach to estimate equivalent sectional inertial and stiffness properties of slender and hollow composite structures. PreComp requires the geometric description of the blade (chord, twist, section profile shapes, web locations), along with the internal structural layup (laminate schedule, orientation of fibers, laminate material properties). It allows for high-flexibility in the specification of the composite layup both spanwise and chordwise.

PreComp offers the attractive advantages of running almost instantaneously and not requiring sophisticated meshing routines. However, PreComp suffers the limitation that it does not estimate the shear stiffness terms. In addition, the other stiffness and inertia terms suffer inaccuracies compared to three dimensional finite element models :cite:`precompvs3D`.

Users interested to know more about PreComp should refer to the PreComp User Guide, which is available here `https://www.nrel.gov/docs/fy06osti/38929.pdf <https://www.nrel.gov/docs/fy06osti/38929.pdf>`_.

Precomp should be handled as a preliminary/conceptual design tool. Users interested in a more accurate tool should consider using the framework SONATA, which is an Python-based open source framework that has the ability to call the commercial solver VABS and the open source solver ANBA4 :cite:`FEIL2020112755`.

Code
====

The underlying code PreComp is written in Fortran and is linked to this class with f2py. The source code is available in the file wisdem/rotorse/PreCompPy.f90

A Python wrapper to the code is implemented in the file wisdem/rotorse/rotor_elasticity.py. The wrapper itself is wrapped within an OpenMDAO explicit component in the same file.


Bibliography
============

.. bibliography:: ../../references.bib
   :filter: docname in docnames
