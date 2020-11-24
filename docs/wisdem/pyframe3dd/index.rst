pyFrame3DD
==========

Overview
---------

Frame3DD has its own `documentation <http://svn.code.sourceforge.net/p/frame3dd/code/trunk/doc/Frame3DD-manual.html>`_ that describes the theory, applications, and approach in depth.  This Python wrapper includes a few modifications to the standard code:

- Elimination of all input/output files in favor of direct variable passing
- Arbitrary stiffness values can be passed in (rather than only rigid or free).
- Frame3DD allows inclusion of concentrated masses but they only affect the modal analysis.  In pyFrame3DD they also affect the loads.

pyFrame3DD is used within WISDEM to do almost all of the structural analysis.  This includes the rotor blades, drivetrain components such as the shaft and bedplate, tower, offshore support structures (monopile, jacket, floating platform), and others.

License
-------

Frame3DD uses the GNU General Public License (GPL), which carries strong copy-left restrictions.  The standalone `pyFrame3DD repository <https://github.com/WISDEM/pyFrame3DD>`_ is therefore also released under the GNU GPL license.  For WISDEM, NREL has obtained a special dispensation from the Frame3DD author to use it within this codebase but still retain the Apache License, Version 2.0.

  
