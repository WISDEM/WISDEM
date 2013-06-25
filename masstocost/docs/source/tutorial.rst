.. _tutorial-label:

.. currentmodule:: masstocost.docs.source.examples.example


Tutorial
--------

As an example, let us simulate using masses for major wind turbine components for the NREL 5MW Reference Model :cite:`Jonkman2009` using the overall turbine cost model.  The hub and drivetrain component masses must also be provided and are calculated from Sunderland Model :cite:`Sunderland1993`.  

The first step is to import the relevant files and set up the PPI indices based on the years of interest (reference and current).

.. literalinclude:: examples/example.py
    :start-after: # 1 ---
    :end-before: # 1 ---

The turbine cost model relies on the mass inputs of all major turbine components.  These filter down to the individual component models through the rotor, nacelle and tower.

.. literalinclude:: examples/example.py
    :start-after: # 2 ---
    :end-before: # 2 ---


Next, we need to set additional inputs to the model.  The blade number must be known to get the total cost for the blade set; the advanced Boolean for the blade must be set to select which mass-cost curve for the blade to use (normal or advanced blade).  We set this to advanced to be in line with the FAST 5 MW reference model.  The machine rating and boolean flags for onboard crane and offshore project must also be set.  These are used in the determination of costs for auxiliary system components.  Finally, the drivetrain configuration (iDesign) is specified so that the proper gearbox and generator coefficients will be used.  This should always be set to 1 for the current model.

.. literalinclude:: examples/example.py
    :start-after: # 3 ---
    :end-before: # 3 ---


We can now create and evaluate the cost for the turbine and its components.

.. literalinclude:: examples/example.py
    :start-after: # 4 ---
    :end-before: # 4 ---


We then print out the resulting cost values

.. literalinclude:: examples/example.py
    :start-after: # 5 ---
    :end-before: # 5 ---

The result is

>>> Turbine cost is $5315270.08 USD
>>>
>>> Overall rotor cost with 3 advanced blades is $1471019.77 USD
>>> Advanced blade cost is $251829.54 USD
>>> Cost of 3 blades is $755488.63 USD
>>> Hub cost is $173823.07 USD
>>> Pitch cost is $531016.46 USD
>>> Spinner cost is $10691.61 USD
>>>
>>> Overall nacelle cost is $2856229.57 USD
>>> LSS cost is $174104.10 USD
>>> Main bearings cost is $56228.71 USD
>>> Gearbox cost is $641045.88 USD
>>> HSS cost is $15161.40 USD
>>> Generator cost is $432991.21 USD
>>> Bedplate cost is $136836.34 USD
>>> Yaw system cost is $137375.05 USD
>>>
>>> Tower cost is $988020.74 USD

Note that the output for the individual nacelle components do not sum to the overall nacelle cost.  There are additional costs in the overall nacelle assembly including the onboard crane, electronics and controls, HVAC, other miscellaneous hardware and the nacelle cover.

