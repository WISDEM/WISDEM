WISDEM Documentation
====================

.. warning::
  This documentation is currently under development and is only valid for the develop branch of WISDEM. There is no guarantee of applicability for any version of WISDEM.

The Wind-plant Integrated System Design and Engineering Model (WISDEM) includes integrated assemblies for the assessment of system behavior of wind turbines and plants. These assemblies can be used as is, but a richer use-case involves treating the assemblies as temples, modifying the source code and `OpenMDAO <https://openmdao.org/>`_ problems to answer specific research questions.  For example, any variable in these assemblies can be a design variable, an objective, or part of a constraint in a multidisciplinary optimization. WISDEM should therefore be viewed a toolbox of analysis tools and the basic structure for connecting tools across subsystems and fidelity levels, which can be extended in a multitude of directions according to the user’s needs.

License
-------
WISDEM is licensed under Apache v 2.0 :ref:`license`. 

Disclaimer
----------
This software is provided as-is and without warranty. There are no guarantees it is bug free or provides the correct answers, even if it is used for the intended purpose. By using this software and as a condition of the Apache license, you agree to not hold any WISDEM developer liable for damages.  

Important Links
---------------

- `Source Code Repository <https://github.com/WISDEM/WISDEM/tree/develop>`_ 
- `OpenMDAO <https://openmdao.org/>`_

Author: `NREL WISDEM Team <mailto:systems.engineering+WISDEM_Docs@nrel.gov>`_

.. This images is a placeholder, feel free to replace.  Would like to include an image on the landing page that captures the breadth of the tool. -EMG
.. figure:: /images/wisdem/WISDEM_Overview.*

Using WISDEM
============

.. toctree::
   :maxdepth: 2

   installation
   how_wisdem_works
   first_steps
   what_wisdem_can_do
   examples
   modules
   publications
   
   
Other Useful Docs
=================

.. toctree::
   :maxdepth: 2

   known_issues
   how_to_contribute_code
   how_to_write_docs
   
Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
