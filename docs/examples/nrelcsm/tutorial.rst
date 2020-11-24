.. _nrelcsm_tutorial-label:

----------------------------------------
1. NREL Cost and Scaling Model Example
----------------------------------------

.. contents:: List of Examples
   :depth: 2


Turbine Component Masses Using the NREL_CSM (2015)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As an example of estimating turbine component masses (only) using the 2015 update of the NREL Cost and Scaling Model (CSM), let us simulate the NREL 5MW Reference Model :cite:`FAST2009`.

The first step is to import OpenMDAO and the model itself:

.. literalinclude:: /../examples/01_nrel_csm/mass.py
    :start-after: # 0 ---
    :end-before: # 0 ---

Next, we initialize an OpenMDAO instance and assign the model to be the `nrel_csm_mass_2015` module.  The `setup()` command completes the high-level configuration readies the model for variable inpu:

.. literalinclude:: /../examples/01_nrel_csm/mass.py
    :start-after: # 1 ---
    :end-before: # 1 ---

The turbine scaling relies on key turbine configuration parameters.  These filter down to the individual component models through the rotor, nacelle, and tower as described on the :ref:`theory` page.  The variables are set like a Python dictionary:

.. literalinclude:: /../examples/01_nrel_csm/mass.py
    :start-after: # 2 ---
    :end-before: # 2 ---

We can now run the model to compute the component masses:

.. literalinclude:: /../examples/01_nrel_csm/mass.py
    :start-after: # 3 ---
    :end-before: # 3 ---


We can then print out an exhaustive listing of the inputs and outputs to each submodule:

.. literalinclude:: /../examples/01_nrel_csm/mass.py
    :start-after: # 4 ---
    :end-before: # 4 ---

The final lines highlight the mass breakdown summaries:

>>>  turbine
>>>    hub_system_mass      [47855.49446548]   kg
>>>    rotor_mass           [95109.93575675]   kg
>>>    nacelle_mass         [165460.38774975]  kg
>>>    turbine_mass         [442906.80408368]  kg

See the full source for this example on `Github <https://github.com/WISDEM/WISDEM/blob/master/examples/01_nrel_csm/mass.py>`_.


Turbine Component Masses and Costs Using the NREL_CSM (2015)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is often desired to estimate the component costs and cost of energy of a hypothetical turbine, not just the component masses in the previous example.  To do so, all that is required is import the full 2015 Cost and Scaling model with:

.. literalinclude:: /../examples/01_nrel_csm/mass_and_cost.py
    :start-after: # 0 ---
    :end-before: # 0 ---

The OpenMDAO problem instance must also be assigned this model (`nrel_csm_2015`):

.. literalinclude:: /../examples/01_nrel_csm/mass_and_cost.py
    :start-after: # 1 ---
    :end-before: # 1 ---

The model inputs remain the same:

.. literalinclude:: /../examples/01_nrel_csm/mass_and_cost.py
    :start-after: # 2 ---
    :end-before: # 2 ---

We can now run the model to compute the component masses and costs:

.. literalinclude:: /../examples/01_nrel_csm/mass_and_cost.py
    :start-after: # 3 ---
    :end-before: # 3 ---

Then we can again print out an exhaustive listing of the inputs and outputs:

.. literalinclude:: /../examples/01_nrel_csm/mass_and_cost.py
    :start-after: # 4 ---
    :end-before: # 4 ---

The final screen output is:

>>>    turbine_c
>>>      turbine_mass_tcc                       [434686.14646457]   kg
>>>      turbine_cost                           [3543676.12253719]  USD
>>>      turbine_cost_kW                        [708.73522451]      USD/kW

See the full source for this example on `Github <https://github.com/WISDEM/WISDEM/blob/master/examples/01_nrel_csm/mass_and_cost.py>`__.


Turbine Component Costs Using the NREL_CSM (2015)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As an example of estimating turbine component costs (only), if the component masses are already known, using the 2015 update of the NREL Cost and Scaling Model (CSM), let us simulate the NREL 5MW Reference Model :cite:`FAST2009`.

The first step is to import OpenMDAO and the model itself:

.. literalinclude:: /../examples/01_nrel_csm/costs.py
    :start-after: # 0 ---
    :end-before: # 0 ---

Next, we initialize an OpenMDAO instance and assign the model to be the `Turbine_CostsSE_2015` module.  This module has a configuration option to print to the screen a nicely formatted summary of the outputs, which is accessed by setting `verbosity=True`.  The `setup()` command completes the high-level configuration readies the model for variable inpu:

.. literalinclude:: /../examples/01_nrel_csm/costs.py
    :start-after: # 1 ---
    :end-before: # 1 ---

The turbine scaling relies on key turbine configuration parameters.  These filter down to the individual component models through the rotor, nacelle, and tower as described on the :ref:`theory` page.  The variables are set like a Python dictionary:

.. literalinclude:: /../examples/01_nrel_csm/costs.py
    :start-after: # 2 ---
    :end-before: # 2 ---

Next we set the individual component masses.  These values might come from publicly available data, the other WISDEM modules, or through parametric study.  In this example, we grab the masses computed in the previous example:

.. literalinclude:: /../examples/01_nrel_csm/costs.py
    :start-after: # 3 ---
    :end-before: # 3 ---

We can now run the model to compute the component costs:

.. literalinclude:: /../examples/01_nrel_csm/costs.py
    :start-after: # 4 ---
    :end-before: # 4 ---

A formatted tabular output is printed to the screen:

>>> ######################################################################
>>> Computation of costs of the main turbine components from TurbineCostSE
>>> Blade cost              229.972 k USD       mass 15751.480 kg
>>> Pitch system cost       206.283 k USD       mass 9334.089 kg
>>> Hub cost                146.439 k USD       mass 37548.405 kg
>>> Spinner cost            10.800 k USD        mass 973.000 kg
>>> ---------------------------------------------------------------------
>>> Rotor cost              1053.437 k USD      mass 95109.936 kg
>>>
>>> LSS cost                244.771 k USD       mass 20568.963 kg
>>> Main bearing cost       10.104 k USD        mass 2245.416 kg
>>> Gearbox cost            560.741 k USD       mass 43468.321 kg
>>> HSS cost                6.764 k USD         mass 994.700 kg
>>> Generator cost          184.760 k USD       mass 14900.000 kg
>>> Bedplate cost           121.119 k USD       mass 41765.261 kg
>>> Yaw system cost         102.339 k USD       mass 12329.962 kg
>>> HVAC cost               49.600 k USD        mass 400.000 kg
>>> Nacelle cover cost      38.969 k USD        mass 6836.690 kg
>>> Electr connection cost  209.250 k USD
>>> Controls cost           105.750 k USD
>>> Other main frame cost   101.273 k USD
>>> Transformer cost        215.918 k USD       mass 11485.000 kg
>>> Converter cost          0.000 k USD         mass 0.000 kg
>>> ---------------------------------------------------------------------
>>> Nacelle cost            1961.463 k USD      mass 157239.730 kg
>>>
>>> Tower cost              528.776 k USD       mass 182336.481 kg
>>> ---------------------------------------------------------------------
>>> ---------------------------------------------------------------------
>>> Turbine cost            3543.676 k USD      mass 434686.146 kg
>>> Turbine cost per kW     708.735 k USD/kW
>>> ######################################################################


We can also print out an exhaustive listing of the inputs and outputs to each submodule:

.. literalinclude:: /../examples/01_nrel_csm/costs.py
    :start-after: # 5 ---
    :end-before: # 5 ---

See the full source for this example on `Github <https://github.com/WISDEM/WISDEM/blob/master/examples/01_nrel_csm/costs.py>`__.




Parametric Studies Using the NREL_CSM (2015)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The simplicity and rapid execution of the NREL CSM makes it well suited for parametric studies.  This example runs approximately 6000 points in a Design of Experiment (DoE) parametric analysis varying machine rating, rotor diameter (and thereby hub_height), the blade mass scaling exponent, the average wind speed, and wind shear.

As above, the first step is to import OpenMDAO and the model itself, but we will also need other Python and WISDEM packages.  In this case, the NumPy library and the annual energy production (AEP) estimator from the older (~2010) CSM code:

.. literalinclude:: /../examples/01_nrel_csm/parametric.py
    :start-after: # 0 ---
    :end-before: # 0 ---

Next, we initialize an OpenMDAO instance and assign the model to be the `nrel_csm_2015` module.  We also initialize an instance of the AEP model:

.. literalinclude:: /../examples/01_nrel_csm/parametric.py
    :start-after: # 1 ---
    :end-before: # 1 ---

The CSM model initialization is abbreviated here because some of the variables will be modified within the DoE loop.  The remaining ones are:

.. literalinclude:: /../examples/01_nrel_csm/parametric.py
    :start-after: # 2 ---
    :end-before: # 2 ---

Note that the `turbine_class` variable has been set to `-1` to allow us to override the `blade_mass_exp` value as described in the :ref:`csmsource` documentation.  Also, two variables are jointly assigned to local Python variables for use in the AEP estimation.
The AEP model requires a number of other inputs to define the turbine power curve.  To keep things simple, we focus on a single turbine, and ignore many of the other losses and options:

.. literalinclude:: /../examples/01_nrel_csm/parametric.py
    :start-after: # 3 ---
    :end-before: # 3 ---

Next we define our parametric axes using NumPy's `arange <https://numpy.org/doc/stable/reference/generated/numpy.arange.html>`_ function that provides evenly spaced intervals:

.. literalinclude:: /../examples/01_nrel_csm/parametric.py
    :start-after: # 4 ---
    :end-before: # 4 ---

To run our n-dimensional DOE, we do a "tensor" or "outer" multiplication of the arrays using NumPy's `meshgrid <https://numpy.org/doc/stable/reference/generated/numpy.arange.html>`_, but then flatten them into 1-D vectors for easy enumeration of all of the scenarios:

.. literalinclude:: /../examples/01_nrel_csm/parametric.py
    :start-after: # 5 ---
    :end-before: # 5 ---

We are now ready to loop through all of the points, and evaluate the CSM model and AEP model:

.. literalinclude:: /../examples/01_nrel_csm/parametric.py
    :start-after: # 6 ---
    :end-before: # 6 ---

To store for later postprocessing, we save everything into a large csv-file.  Flattening the arrays makes this fairly straightforward using NumPy's concatenation shortcuts:

.. literalinclude:: /../examples/01_nrel_csm/parametric.py
    :start-after: # 7 ---
    :end-before: # 7 ---

See the full source for this example on `Github <https://github.com/WISDEM/WISDEM/blob/master/examples/01_nrel_csm/parametric.py>`__.

.. bibliography:: ../../references.bib
   :filter: docname in docnames
