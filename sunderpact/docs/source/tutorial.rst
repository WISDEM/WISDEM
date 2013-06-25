.. _tutorial-label:

.. currentmodule:: masstocost.docs.source.examples.example


Tutorial
--------

As an example, let us simulate model calculations for major wind turbine components for the NREL 5MW Reference Model :cite:`Jonkman2009`.  

The first step is to import the relevant files.

.. literalinclude:: examples/example.py
    :start-after: # 1 ---
    :end-before: # 1 ---

We will start with the hub system model.  The hub model relies on inputs from the rotor such as blade mass, rotor diameter, and blade number.  It also requires either the specification of variables necessary to calculate the maximum bending moment at the blade root (the wind speed necessary to achieve rated power production, air density conditions, and rotor solidity) or the root moment itself.  Specification of the hub diameter is an optional input.  If it is not supplied, then it will be calculated internal to the model. 

.. literalinclude:: examples/example.py
    :start-after: # 2 ---
    :end-before: # 2 ---


We now instantiate the hub system object which contains a hub, pitch system and spinner component.  The initialization automatically updates the mass of the components and overall system based on the supplied inputs.  In addition, calculations of mass properties are also made.

.. literalinclude:: examples/example.py
    :start-after: # 3 ---
    :end-before: # 3 ---


The resulting system and component properties can then be printed.

.. literalinclude:: examples/example.py
    :start-after: # 4 ---
    :end-before: # 4 ---

The results should appear as below:

>>>NREL 5 MW Turbine test
>>>Hub Components
>>>hub         29536.2 kg
>>>pitch mech  16670.5 kg
>>>nose cone    1810.5 kg
>>>Hub system total 48017.1 kg
>>>    cm -6.30 0.00 3.15
>>>    I 55039.9 55039.9 55039.9

Secondly, we will demonstrate the nacelle system model.  The hub model relies on inputs from the rotor and hub as well as design variables for the drivetrain. Inputs from the rotor include the rotor diameter, the rotor speed at rated power, the rotor torque at rated power, the maximum thrust from the rotor and the overall rotor mass (including blades and hub).  For the drivetrain, the overall configuration (3-stage geared, single-stage, multi-generator, or direct-drive) must be specified.  The overall gear ratio (1 for direct drive) must be specified along with the gear configuration (may be null for direct drive) and a Boolean for the presence of a bevel stage.  If an onboard crane is present, then the crane Boolean should be set true.  Finally the machine rating (in kW) must be provided.

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

>>>NREL 5 MW Turbine test
>>>Nacelle system model results
>>>Low speed shaft 34764.8 kg
>>>Main bearings   11238.5 kg
>>>Gearbox         34191.4 kg
>>>High speed shaft & brakes  1687.6 kg
>>>Generator       16699.9 kg
>>>Variable speed electronics 0.0 kg
>>>Overall mainframe 107726.9 kg
>>>     Bedplate      93090.6 kg
>>>electrical connections  0.0 kg
>>>HVAC system     400.0 kg
>>>Nacelle cover:   9097.4 kg
>>>Yaw system      12519.3 kg 
>>>Overall nacelle:  228325.7 kg cm -0.59  0.00  2.37 I 9257806.18 988966.94 1238044.62

