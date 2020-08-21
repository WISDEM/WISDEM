Release Notes
=============

.. Note::
   Example input files are provided to demonstrate MAP++. 
   These fake examples do not represent realistic, practical moorings for permanent installations.

License
-------
MAP++ is licensed under Apache v 2.0 :ref:`license`. 

.. toctree::
   :maxdepth: 4
   :hidden:
   
   license.rst

Disclaimer
----------
This software is provided as-is and without warranty. 
There are no guarantees it is bug free or provides the correct answers, even if it is used for the intended purpose. 
By using this software and as a condition of the Apache license, you agree to not hold any MAP++ developer liable for damages.  

Dependencies
------------
Third party dependencies are distributed with the MAP++ archive on BitBucket. Required libraries include the following:

=====================  =================
**Library**            **Version Distributed with MAP++**
LAPACK                 `Version 3.5.0 <http://www.netlib.org/lapack/>`_
C/C++ Minpack          `Version 1.3.3 <http://devernay.free.fr/hacks/cminpack/>`_
SimCList               `Version 1.6 <http://mij.oltrelinux.com/devel/simclist/>`_
Better String Library  `Version 0.1.1 <http://mike.steinert.ca/bstring/doc/>`_
=====================  =================

Change Log
----------
v1.20.00 -- First release.

v1.20.10 -- Repaired the linearized stiffness matrix function and improved the input file parsing in python.
