.. _tutorial-label:

.. currentmodule:: wisdem.docs.examples.example


Tutorial
--------

Tutorial for WISDEM with NREL CSM
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As an example of WISDEM, let us use the NREL Cost and Scaling Model and simulate the NREL 5MW Reference Model :cite:`FAST2009` in a 500 mW offshore plant.  

The first step is to import the relevant files and set up the component.

.. literalinclude:: /examples/wisdem/example.py
    :start-after: # 1 ---
    :end-before: # 1 ---

The NREL CSM relies on on a large number of wind turbine and plant parameters including overall wind turbine configuration, rotor, nacelle and tower options as well as plant characteristics including the number of turbines, wind resource characteristics and financial parameters.

.. literalinclude:: examples/example.py
    :start-after: # 2 ---
    :end-before: # 2 ---


We can now simulate the overall wind plant cost of energy.

.. literalinclude:: /examples/wisdem/example.py
    :start-after: # 3 ---
    :end-before: # 3 ---


We then print out the resulting cost values:

.. literalinclude:: /examples/wisdem/example.py
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

.. literalinclude:: /examples/wisdem/example.py
    :start-after: # 5 ---
    :end-before: # 5 ---

Similar to before, the NREL CSM plus the ECN Offshore OPEX Model assembly relies on on a large number of wind turbine and plant parameters including overall wind turbine configuration, rotor, nacelle and tower options as well as plant characteristics including the number of turbines, wind resource characteristics and financial parameters.

.. literalinclude:: /examples/wisdem/example.py
    :start-after: # 6 ---
    :end-before: # 6 ---


We can now simulate the overall wind plant cost of energy.

.. literalinclude:: /examples/wisdem/example.py
    :start-after: # 7 ---
    :end-before: # 7 ---


We then print out the resulting cost values:

.. literalinclude:: /examples/wisdem/example.py
    :start-after: # 8 ---
    :end-before: # 8 ---

The result is:

>>> Cost of Energy results for a 500 MW offshore wind farm using the NREL 5 MW reference turbine
>>> LCOE: $0.1088 USD/kWh
>>> COE: $0.1175 USD/kWh
>>>
>>> AEP per turbine: 16949265.578077 kWh/turbine
>>> Turbine Cost: $5950210.271159 USD
>>> BOS costs per turbine: $7664647.465834 USD/turbine
>>> OPEX per turbine: $433958.907447 USD/turbine



Tutorial for WISDEM with SE Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can also use the new systems engineering WISDEM modules for assessing plant cost of energy.  In this example, we add the new turbine systems engineering and cost models along with the NREL CSM plant energy production and plant cost models.  First we create the assembly with a set of options.  Here we are setting up the lcoe_se_assembly with the new DriveSE 4-pt configuration nacelle, the NREL CSM balance of station model and a rigid blade.

.. literalinclude:: /examples/wisdem/example.py
    :start-after: # 9 ---
    :end-before: # 9 ---

Next we need to set up the LCOE level inputs for the analysis and several inputs for the sub-modules.

.. literalinclude:: ../src/wisdem/lcoe/lcoe_se_csm_assembly.py
    :start-after: # === Set
    :end-before: # ====

Then we run the lcoe analysis.

.. literalinclude:: ../src/wisdem/lcoe/lcoe_se_csm_assembly.py
    :start-after: # === Run
    :end-before: # ====

Finally we print the values from the analysis.

.. literalinclude:: ../src/wisdem/lcoe/lcoe_se_csm_assembly.py
    :start-after: # === Print
    :end-before: # ====

The results should be:

>>> Key Turbine Outputs for NREL 5 MW Reference Turbine
>>> mass rotor blades:54674.80 (kg)
>>> mass hub system: 37118.36 (kg)
>>> mass nacelle: 193805.65 (kg)
>>> mass tower: 358230.15 (kg)
>>> maximum tip deflection: 10.65 (m)
>>> ground clearance: 28.44 (m)
>>> 
>>> Key Plant Outputs for wind plant with NREL 5 MW Turbine
>>> COE: $0.0447 USD/kWh
>>> 
>>> AEP per turbine: 20716963.8 kWh/turbine
>>> Turbine Cost: $5122974.19 USD
>>> BOS costs per turbine: $3084430.38 USD/turbine
>>> OPEX per turbine: $242359.84 USD/turbine




Tutorial for Turbine Assembly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The turbine assembly is a nested assembly and contains subassemblies for the rotor, hub, nacelle, and tower.  Determining what inputs need to be set, and which are already connected through the assembly, can be challenging for one of these large assemblies.  There is a helper method ``print_vars`` in commonse noted below.  This can be particularly useful as the models are udpated and different variable sets are used.

.. literalinclude:: ../src/wisdem/turbinese/turbine.py
    :start-after: # === setup
    :end-before: # ====

All of the variables from the rotor need to be set.

.. literalinclude:: ../src/wisdem/turbinese/turbine.py
    :start-after: # === rotor
    :end-before: # ====

Most of the nacelle parameters need to be set.

.. literalinclude:: ../src/wisdem/turbinese/turbine.py
    :start-after: # === nacelle
    :end-before: # ====

Some tower parameters need to be set, including those which come from the configurable slots.  Loading conditions and mass properties are connected to the rotor and nacelle and do not need to be set by the user.

.. literalinclude:: ../src/wisdem/turbinese/turbine.py
    :start-after: # === tower
    :end-before: # ====

With the model defined we can now run it. All outputs of the subassemblies are available, as are two additional outputs related to the turbine geometry.  The parameter ``turbine.maxdeflection.max_tip_deflection`` gives the clearance between the undeflected blade shape and the tower.  It represents the maximum allowable tip deflection in the +x yaw c.s. (not including safety factors) before a tower strike.  The parameter ``turbine.maxdeflection.ground_clearance`` gives the distance between the blade tip at its bottom passage and the ground.

.. literalinclude:: ../src/wisdem/turbinese/turbine.py
    :start-after: # === run
    :end-before: # ====

The results should be:

>>> mass rotor blades (kg) = 54674.7959412
>>> mass hub system (kg) = 37118.3606357
>>> mass nacelle (kg) = 193805.649876
>>> mass tower (kg) = 358230.152841
>>> maximum tip deflection (m) = 10.6526326093
>>> ground clearance (m) = 28.4361396283





