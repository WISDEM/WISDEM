.. _finance_tutorial-label:

8. Plant Finance Example
------------------------

Using the plant finance model only requires a short script.
The plant finance model depends on the costs of the turbine, balance of station, expected operational expenditures, energy production and a few financial parameters.
This example shows some representative values for the financial parameters, including a fixed charge rate for COE, the construction financing charge rate, the tax rate for OPEX tax deductions, the time for plant construction, the project lifetime and the sea depth (which would be 0.0 for a land-based plant).
In a full turbine analysis, some of these input values come from upstream components or from the turbine design.

.. literalinclude:: ../../../examples/08_plant_finance/example.py
   :language: python

The screen output is,

.. code:: console

    10 Input(s) in 'model'
    ----------------------

    varname            value         units
    -----------------  ------------  -----------
    machine_rating     [2320.]       kW
    tcc_per_kW         [1093.]       USD/kW
    offset_tcc_per_kW  [0.]          USD/kW
    bos_per_kW         [517.]        USD/kW
    opex_per_kW        [43.56]       USD/kW/yr
    park_aep           [0.]          kW*h
    turbine_aep        [9915950.]    kW*h
    wake_loss_factor   [0.15]        None
    fixed_charge_rate  [0.07921664]  None
    turbine_number     87.0          Unavailable


    2 Explicit Output(s) in 'model'
    -------------------------------

    varname    value             units
    ---------  ----------------  --------
    lcoe       [0.04709575]      USD/kW/h
    plant_aep  [7.33284502e+08]  USD/kW/h
