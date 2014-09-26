.. _tutorial-label:

.. currentmodule:: wisdem.docs.examples.example


Tutorial
--------

Tutorial for WISDEM with NREL CSM
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As an example of WISDEM, let us use the NREL Cost and Scaling Model and simulate the NREL 5MW Reference Model :cite:`FAST2009` in a 500 mW offshore plant.  

The first step is to import the relevant files and set up the component.

.. literalinclude:: examples/example.py
    :start-after: # 1 ---
    :end-before: # 1 ---

The NREL CSM relies on on a large number of wind turbine and plant parameters including overall wind turbine configuration, rotor, nacelle and tower options as well as plant characteristics including the number of turbines, wind resource characteristics and financial parameters.

.. literalinclude:: examples/example.py
    :start-after: # 2 ---
    :end-before: # 2 ---


We can now simulate the overall wind plant cost of energy.

.. literalinclude:: examples/example.py
    :start-after: # 3 ---
    :end-before: # 3 ---


We then print out the resulting cost values:

.. literalinclude:: examples/example.py
    :start-after: # 4 ---
    :end-before: # 4 ---

The result is:

>>> Cost of Energy results for a 500 MW offshore wind farm using the NREL 5 MW reference turbine
>>> LCOE: $0.1113 USD/kWh
>>> COE: $0.1191 USD/kWh
>>>
>>> AEP per turbine: 16915536.836031 kWh/turbine
>>> Turbine Cost: $5950210.271159 USD
>>> BOS costs per turbine: $7664647.465834 USD/turbine
>>> OPEX per turbine: $473476.639865 USD/turbine


Tutorial for WISDEM with NREL CSM plus ECN Offshore OPEX Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you have the ECN Offshore OPEX model and license (see (Plant_CostsSE documentation for details)`http://wisdem.github.io/Plant_CostsSE/`_) then you can incorporate it into your analysis along with the NREL CSM model.

The first step is to again import the relevant files and set up the component.

.. literalinclude:: examples/example.py
    :start-after: # 5 ---
    :end-before: # 5 ---

Similar to before, the NREL CSM plus the ECN Offshore OPEX Model assembly relies on on a large number of wind turbine and plant parameters including overall wind turbine configuration, rotor, nacelle and tower options as well as plant characteristics including the number of turbines, wind resource characteristics and financial parameters.

.. literalinclude:: examples/example.py
    :start-after: # 6 ---
    :end-before: # 6 ---


We can now simulate the overall wind plant cost of energy.

.. literalinclude:: examples/example.py
    :start-after: # 9 ---
    :end-before: # 9 ---


We then print out the resulting cost values:

.. literalinclude:: examples/example.py
    :start-after: # 10 ---
    :end-before: # 10 ---

The result is:

>>> Cost of Energy results for a 500 MW offshore wind farm using the NREL 5 MW reference turbine
>>> LCOE: $0.1088 USD/kWh
>>> COE: $0.1175 USD/kWh
>>>
>>> AEP per turbine: 16949265.578077 kWh/turbine
>>> Turbine Cost: $5950210.271159 USD
>>> BOS costs per turbine: $7664647.465834 USD/turbine
>>> OPEX per turbine: $433958.907447 USD/turbine

