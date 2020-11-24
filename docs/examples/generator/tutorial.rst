.. _generator_tutorial-label:
  
7. Generator Model Examples
------------------------------------

Understanding the generator examples is most easily done with the companion report for `GeneratorSE <https://www.nrel.gov/docs/fy17osti/66462.pdf>`__.  The design variables in the examples relate to both the electromagnetic and structural design.  The five generator example files relate to the five different generator architectures available:

- **PMSG-Outer**: Permanent magnet synchronous generator (outer generator - inner stator)
- **PMSG-Disc**: Permanent magnet synchronous generator (inner generator - outer stator) with solid disc stator support
- **PMSG-Arms**: Permanent magnet synchronous generator (inner generator - outer stator) with arm/spoke stator support
- **EESG**: Electrically excited synchronous generator
- **DFIG**: Doubly fed induction generator
- **SCIG**: Squirrel-cage induction generator

Each of the technologies have slightly different sets of required inputs so while the examples follow the same pattern, the specific design variables and constraints will vary from one to the other. For brevity, only the ``pmsg_outer.py`` script is presented here.

First, we import the modules we want to use,

.. literalinclude:: /../examples/07_generator/pmsg_outer.py
    :language: python
    :start-after: # Import
    :end-before: # --

Next, we initialize the problem and set some script parameters,

.. literalinclude:: /../examples/07_generator/pmsg_outer.py
    :language: python
    :start-after: # Problem
    :end-before: # --

If running an optimization, we set the design variables, constraints, and objectives.  These are most easily understood with the report linked above.  Note that the optimization driver in the script is SLSQP from the OpenMDAO Scipy Driver, as that is available in the regular Conda-based installation of WISDEM.  However, we have had some better experience using the CONMIN driver available in the pyOptSparse library.

.. literalinclude:: /../examples/07_generator/pmsg_outer.py
    :language: python
    :start-after: # Pose
    :end-before: # --

Now, the long list of electromechanical and structural inputs are set prior to execution,

.. literalinclude:: /../examples/07_generator/pmsg_outer.py
    :language: python
    :start-after: # Input
    :end-before: # --

The script can now be executed.  If running an optimization, we activate finite differencing for the total derivatives instead of the partials,

.. literalinclude:: /../examples/07_generator/pmsg_outer.py
    :language: python
    :start-after: # Run
    :end-before: # --
