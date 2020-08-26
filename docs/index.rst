WISDEM Documentation
====================

.. warning::
  This documentation is currently under development and is only valid for the IEAontology4all branch of WISDEM. There is no guarantee of applicability for any version of WISDEM.

The Wind-plant Integrated System Design and Engineering Model (WISDEM) includes integrated assemblies for the assessment of system behavior of wind turbines and plants. These assemblies can be used as is, but a richer use-case involves treating the assemblies as temples, modifying the source code and `OpenMDAO <https://openmdao.org/>`_ problems to answer specific research questions.  For example, any variable in these assemblies can be a design variable, an objective, or part of a constraint in a multidisciplinary optimization. WISDEM should therefore be viewed a toolbox of analysis tools and the basic structure for connecting tools across subsystems and fidelity levels, which can be extended in a multitude of directions according to the user’s needs.

Important Links:

- `Source Code Repository <https://github.com/WISDEM/WISDEM/tree/IEAontology4all>`_ 
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
   examples
   modules
   theory
   
   
Other Useful Docs
=================

.. toctree::
   :maxdepth: 2

   known_issues
   how_to_write_docs
   how_to_contribute_code
   
Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
