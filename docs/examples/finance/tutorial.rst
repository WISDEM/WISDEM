.. _tutorial-label:

Tutorial
--------

Simple plant finance tutorial
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using the plant finance model only requires a short script.
The plant finance model depends on the costs of the turbine, balance of station, expected operational expenditures, energy production and a few financial parameters.
This example shows some representative values for the financial parameters, including a fixed charge rate for COE, the construction financing charge rate, the tax rate for OPEX tax deductions, the time for plant construction, the project lifetime and the sea depth (which would be 0.0 for a land-based plant).
In a full turbine analysis, some of these input values come from upstream components or from the turbine design.

.. literalinclude:: ../../../examples/plant_finance/example.py