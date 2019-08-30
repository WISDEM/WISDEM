.. figure:: nstatic/map_logo.png
    :align: center

MAP++
=====


.. toctree::
   :maxdepth: 2
   :hidden:

   notes.rst
   definitions.rst
   theory.rst
   input_file.rst
   python_example.rst
   api_doc.rst
   faq.rst
   help.rst
   ref.rst


The Mooring Analysis Program is a library to model static loads and geometry of cables.
MAP++ is designed to hook into other simulation codes, and through its API, it can be customized to do a few things:

 * Prototype a design 
 * Find the force-displacement relation for a given footprint
 * Integrate into other dynamic simulation programs to produce a nonlinear restoring force time history

A quick--start guide is available :ref:`here <quick_start>`.
We integrated MAP++ into other programs written in Python, C, C++, and Fortran.
MAP++ follows the FAST Offshore Wind Turbine Framework :cite:`jonkman2013new` software pattern.

More information on the theory behind MAP++ is described :cite:`masciola2013`, with the hopes to extend capabilities to include a heuristic finite element formulation as described :cite:`masciola2014`. MAP++ is licensed under Apache version 2.

.. figure:: nstatic/comparison.png
    :align: center


