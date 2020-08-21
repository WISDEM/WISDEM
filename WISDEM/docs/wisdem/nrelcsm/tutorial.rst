.. _tutorial-label:

.. currentmodule:: wisdem.NREL_CSM.docs.examples.example


Tutorial for NREL_CSM
-----------------------

As an example of NREL_CSM, let us calculate a wind plant cost of energy using cost inputs for a hypothetical wind plant of 100 5 MW turbines at an offshore site.    

The first step is to import the relevant files.

.. literalinclude:: /examples/nrelcsm/example.py
    :start-after: # 1 ---
    :end-before: # 1 ---

The NREL Cost and Scaling Model depends on the costs of the turbine, balance of station, expected operational expenditures, energy production and a few financial parameters.  These are calculated by sub-models which take a small number of wind turbine and plant attributes as inputs.  In addition, the target year has to be spcified for the calculation.  The scaling of PPI indices is current up to 2010.

.. literalinclude:: /examples/nrelcsm/example.py
    :start-after: # 2 ---
    :end-before: # 2 ---

We can now evaluate the cost of energy for the wind plant by creating an instance of the csm class and computing cost using its compute function.

.. literalinclude:: /examples/nrelcsm/example.py
    :start-after: # 3 ---
    :end-before: # 3 ---


We then print out the resulting cost values.

.. literalinclude:: /examples/nrelcsm/example.py
    :start-after: # 4 ---
    :end-before: # 4 ---

The result is:

>>> LCOE: 0.10506401
>>> COE: 0.12217748
>>> AEP: 16850.57535
>>> BOS: 766464.74370
>>> TCC: 5950.20931
>>> OM: 360.99240
>>> LRC: 91.04839
>>> LLC: 19.96719
