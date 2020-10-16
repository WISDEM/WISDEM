.. _what_wisdem_can_do:

WISDEM capabilities
===================

Previous sections in the documentation have focused on :ref:`how_wisdem_works` at a high-level without delving into WISDEM's modeling assumptions and coding implementation.
This page covers some of those details, explains when to use WISDEM versus other software packages, and where WISDEM's capabilities start and stop.

What WISDEM can do
------------------

WISDEM as a conceptual-level design tool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because WISDEM models the entire turbine, from wind to LCOE, it is helpful in capturing system-level effects of changes in designs.
This makes WISDEM useful at the conceptual design level to help down-select from many potential designs to the most viable designs.

Computational expense
~~~~~~~~~~~~~~~~~~~~~

Depending on the model complexity, the disciplines included, and the discretization levels, design problems take on the order of seconds to minutes to solve using WISDEM.
For example, tower optimization with fixed loads may take approximately 10 seconds, whereas full system optimization controlling the blade twist takes 5-20 minutes.

Past design problems solved using WISDEM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since its creation, WISDEM has been used in a variety of problems for many different turbine designs.
An overview of these problems is provided below with links to the studies where the results were presented.
This is not an exhaustive list and WISDEM's capabilities have changed over the years, but this table is meant as a quick glance at some of the problems that WISDEM can tackle.

.. list-table:: Previous problems solved using WISDEM
    :widths: 30 30 30 30 30
    :header-rows: 1

    * - Disciplines considered
      - Size of optimization problem
      - Design variables
      - Computational expense
      - Link to publication

    * - Entry 1
      - Entry 2
      - Entry 3
      - Entry 4
      - Entry 5
      

What WISDEM cannot do
---------------------

WISDEM cannot model time-varying effects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because WISDEM is developed to tackle preliminary and conceptual design problems, physical phenomena cannot always be captured accurately.
Specifically, WISDEM uses steady-state models, which means that time-varying aspects of turbine design cannot be examined using WISDEM.
This means that any phenomena related to cyclic, transient, stochastic, or resonance-induced loads should not be studied using WISDEM.
That being said, portions of WISDEM can be connected to other software, such as OpenFAST, to take advantage of WISDEM's computationally efficient systems-approach while also including higher-fidelity models.

WISDEM is not a push-button solution to turbine design
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
It would be nice to have a tool where you press a button and receive an optimal turbine design.
However, design optimization requires expert knowledge to perform correctly and interpret the results.
WISDEM exists to minimize time needed to evaluate possible turbine designs by providing a framework and set of models.
This enables turbine designers to more quickly assess performance trade-offs and make decisions based on the results from WISDEM.

Specific disciplinary limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Disciplinary limitations
    :widths: 30 80
    :header-rows: 1

    * - Discipline
      - Limitations or modeling assumptions

    * - Entry 1
      - Entry 2