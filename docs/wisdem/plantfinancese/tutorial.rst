.. _tutorial-label:

.. currentmodule:: plant_financese.docs.examples.example


Tutorial
--------

Tutorial for NREL_CSM_Fin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As an example of NREL_CSM_Fin, let us calculate a wind plant cost of energy using cost inputs for a hypothetical wind plant of 50 5 MW turbines at an offshore site.    

The first step is to import the relevant files and set up the component.

.. literalinclude:: /examples/plantfinancese/example.py
    :start-after: # 1 ---
    :end-before: # 1 ---

The plant finance model depends on the costs of the turbine, balance of station, expected operational expenditures, energy production and a few financial parameters.  The default settings for the financial parameters are used - including a fixed charge rate for coe, the construction financing charge rate, the tax rate for opex tax deductions, the time for plant construction, the project lifetime and the sea depth (which would be 0.0 for a land-based plant).

.. literalinclude:: /examples/plantfinancese/example.py
    :start-after: # 2 ---
    :end-before: # 2 ---

We can now evaluate the cost of energy for the wind plant.

.. literalinclude:: /examples/plantfinancese/example.py
    :start-after: # 3 ---
    :end-before: # 3 ---


We then print out the resulting cost values:

.. literalinclude:: /examples/plantfinancese/example.py
    :start-after: # 4 ---
    :end-before: # 4 ---

The result is

>>> Offshore plant cost
>>> lcoe: 0.1231
>>> coe: 0.1307


Tutorial for Basic_Finance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let’s repeat the above example using the basic finance module which does not provide the second lcoe financial metric.    

The first step is to import the relevant files and set up the component.

.. literalinclude:: /examples/plantfinancese/example.py
    :start-after: # 5 ---
    :end-before: # 5 ---

The plant finance model depends on the costs of the turbine, balance of station, expected operational expenditures, energy production and a few financial parameters.  The default settings for the financial parameters are used - including a fixed charge rate for coe, the tax rate for opex tax deductions, and a boolean for whether it is offshore or land-based.

.. literalinclude:: /examples/plantfinancese/example.py
    :start-after: # 6 ---
    :end-before: # 6 ---

We can now evaluate the cost of energy for the wind plant.

.. literalinclude:: /examples/plantfinancese/example.py
    :start-after: # 7 ---
    :end-before: # 7 ---


We then print out the resulting cost values.

.. literalinclude:: /examples/plantfinancese/example.py
    :start-after: # 8 ---
    :end-before: # 8 ---

The result is:

>>> Offshore plant cost
>>> coe: 0.1307



