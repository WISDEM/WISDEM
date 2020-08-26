.. _tutorial-label:

.. currentmodule:: masstocost.docs.source.examples.example


Tutorial
--------

As an example, let us simulate model calculations for major wind turbine components for the NREL 5MW Reference Model :cite:`FAST2009`.  

The first step is to import the relevant files.

.. literalinclude:: examples/example.py
    :start-after: # 1 ---
    :end-before: # 1 ---

We will start with the hub system model.  The hub model relies on inputs from the rotor such as blade mass, rotor diameter, and blade number.  It also requires either the specification of variables necessary to calculate the maximum bending moment at the blade root (the wind speed necessary to achieve rated power production, air density conditions, and rotor solidity) or the root moment itself.  Specification of the hub diameter is required as well as the machine rating, the positions of the main bearing and other low speed shaft information. 

.. literalinclude:: examples/example.py
    :start-after: # 2 ---
    :end-before: # 2 ---


We now run the hub system model.

.. literalinclude:: examples/example.py
    :start-after: # 3 ---
    :end-before: # 3 ---


The resulting system and component properties can then be printed.

.. literalinclude:: examples/example.py
    :start-after: # 4 ---
    :end-before: # 4 ---

The results should appear as below:

>>> Estimate of Hub Component Sizes for the NREL 5 MW Reference Turbine
>>> Hub Components
>>>   Hub:  29852.8 kg
>>>   Pitch system:      0.0 kg
>>>   Nose cone:   1810.5 kg
>>> Hub system total:  31663.3 kg
>>>     cm  -1.30   0.00   0.22 [m, m, m]
>>>     I 89383.2 137332.2 120860.1 [kg*m^2, kg*m^2, kg*m^2]

Secondly, we will demonstrate the nacelle system model.  The nacelle model relies on inputs from the rotor and hub as well as design variables for the drivetrain. Inputs from the rotor include the rotor diameter, the rotor speed at rated power, the rotor torque at rated power, the maximum thrust from the rotor and the overall rotor mass (including blades and hub).  For the drivetrain, the overall configuration (3-stage geared, single-stage, multi-generator, or direct-drive) must be specified.  The overall gear ratio (1 for direct drive) must be specified along with the gear configuration (may be null for direct drive) and a Boolean for the presence of a bevel stage.  If an onboard crane is present, then the crane Boolean should be set true.  Finally the machine rating (in kW) must be provided.  In addition to these inputs, more specific rotor aerodynamic forces and moments must be specified and a number of additional parameters on the nacelle.

.. literalinclude:: examples/example.py
    :start-after: # 5 ---
    :end-before: # 5 ---

We now instantiate the nacelle system object which contains the low speed shaft, main bearings, gearbox, high speed shaft and brakes, bedplate, and yaw system components.  The main bearings in turn contain components for the main and a second bearing.  The initialization automatically updates the mass of the components and overall system based on the supplied inputs.  In addition, calculations of mass properties are also made.

.. literalinclude:: examples/example.py
    :start-after: # 6 ---
    :end-before: # 6 ---


The resulting system and component properties can then be printed.

.. literalinclude:: examples/example.py
    :start-after: # 7 ---
    :end-before: # 7 ---

The results should appear as below:

>>> Estimate of Nacelle Component Sizes for the NREL 5 MW Reference Turbine
>>> Low speed shaft:  18564.9 kg
>>> Main bearings:   5845.4 kg
>>> Gearbox:  55658.3 kg
>>> High speed shaft & brakes:   2414.7 kg
>>> Generator:  16699.9 kg
>>> Variable speed electronics:      0.0 kg
>>> Overall mainframe: 60785.2 kg
>>>      Bedplate:  51364.7 kg
>>> Electrical connections:      0.0 kg
>>> HVAC system:    400.0 kg
>>> Nacelle cover:   4577.4 kg
>>> Yaw system:   6044.7 kg
>>> Overall nacelle: 170990.5 kg
>>>     cm  -0.44   0.00   0.26 [m, m, m]
>>>     I 77320965.7 1033725.7 935773.7 [kg*m^2, kg*m^2, kg*m^2]

