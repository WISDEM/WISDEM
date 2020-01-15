.. figure:: nstatic/pymap/map_logo.png
    :align: center

pyMAP++
=====


.. toctree::

   notes
   definitions
   theory
   input_file
   python_example
   api_doc
   faq
   help
   ref


The Mooring Analysis Program is a library to model static loads and geometry of cables.
MAP++ is designed to hook into other simulation codes, and through its API, it can be customized to do a few things:

 * Prototype a design 
 * Find the force-displacement relation for a given footprint
 * Integrate into other dynamic simulation programs to produce a nonlinear restoring force time history

A quick--start guide is available :ref:`here <quick_start>`.
We integrated MAP++ into other programs written in Python, C, C++, and Fortran.
MAP++ follows the FAST Offshore Wind Turbine Framework :cite:`jonkman2013new` software pattern.

More information on the theory behind MAP++ is described :cite:`masciola2013`, with the hopes to extend capabilities to include a heuristic finite element formulation as described :cite:`masciola2014`. MAP++ is licensed under Apache version 2.

.. figure:: nstatic/pymap/comparison.png
    :align: center


