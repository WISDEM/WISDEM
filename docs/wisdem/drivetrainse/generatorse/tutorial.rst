.. _tutorial-label:

.. currentmodule:: masstocost.docs.source.examples.example


Tutorial
--------

<<<<<<< HEAD
As an example, let us size a direct drive Permanent Magnet synchronous Generator (PMSG) for the NREL 5MW Reference Model :cite:`FAST2009`.  
=======
As a frst example, we will size a direct-drive Permanent Magnet synchronous Generator (PMSG) for the NREL 5MW Reference Model :cite:`FAST2009`.  
>>>>>>> develop

The first step is to import the relevant files.

.. literalinclude:: examples/example.py
    :start-after: # 1 ---
    :end-before: # 1 ---

We will start with the radial flux PMSG with spoke arm arrangement.  The PMSG module relies on inputs from the rotor such as rated speed, 
power rating and rated torque.  It also requires specification of shear stress, material properties( densities), specific costs and target design efficiency.
and initialization of electromagnetic and structural design variables necessary to calculate basic design. Specification of the optimisation 
objective(Costs,Mass, Efficiency or Aspect ratio) and driver determines the final design. The designs are generated in compliance with the user-specified constraints on generator 
terminal voltage and constraints imposed on the dimensions and electrical,magnetic and structural deformations. The excitation requirement (e.g., magnet dimensions,pole pairs) 
is determined in accordance with the required voltage at no-load. 

.. literalinclude:: examples/example.py
    :start-after: # 2 ---
    :end-before: # 2 ---


We now run the PMSG_arms module.

.. literalinclude:: examples/example.py
    :start-after: # 3 ---
    :end-before: # 3 ---


The resulting system and component properties can then be printed.

.. literalinclude:: examples/example.py
    :start-after: # 4 ---
    :end-before: # 4 ---

The results should appear as below listing the parameters together with their limits where applicable:

>>>                               Parameters       Values       Limit      Units
>>>0                                  Rating     5.000000                     MW
>>>1                             Stator Arms     5.000000                   unit
>>>2              Stator Axial arm dimension   350.703167                     mm
>>>3    Stator Circumferential arm dimension   482.007741     540.354         mm
>>>4                    Stator arm Thickness    61.619914                     mm
>>>5                              Rotor arms     5.000000                     mm
>>>6               Rotor Axial arm dimension   700.262663                     mm
>>>7     Rotor Circumferential arm dimension   529.596686     540.354
>>>8                     Rotor arm Thickness    61.507975                     mm
>>>9                Stator Radial deflection     0.303023    0.336816         mm
>>>10                Stator Axial deflection     0.889166     35.5411         mm
>>>11               Stator circum deflection     2.810001     2.93928         mm
>>>12                Rotor Radial deflection     0.273438     0.32038         mm
>>>13                 Rotor Axial deflection     0.207442     35.5411         mm
>>>14                Rotor circum deflection     2.730594     2.79584         mm
>>>15                       Air gap diameter     6.528774                      m
>>>16                 Overall Outer diameter     6.735516                      m
>>>17                          Stator length     1.602079                      m
>>>18                              l/d ratio     0.245387  (0.2-0.27)
>>>19                      Slot_aspect_ratio     4.540202      (4-10)
>>>20                             Pole pitch    87.489138                     mm
>>>21                     Stator slot height    59.694191                     mm
>>>22                       Stator slotwidth    13.147915                     mm
>>>23                     Stator tooth width    16.069674                     mm
>>>24                     Stator yoke height    88.166066                     mm
>>>25                      Rotor yoke height    88.054034                     mm
>>>26                          Magnet height    10.033035                     mm
>>>27                           Magnet width    61.242397                     mm
>>>28  Peak air gap flux density fundamental     0.801267                      T
>>>29          Peak stator yoke flux density     0.310472          <2          T
>>>30           Peak rotor yoke flux density     0.279780          <2          T
>>>31              Flux density above magnet     0.706295          <2          T
>>>32            Maximum Stator flux density     0.096866    0.801267          T
>>>33             Maximum tooth flux density     1.456850                      T
>>>34                             Pole pairs   117.000000                      -
>>>35             Generator output frequency    23.595000                     Hz
>>>36         Generator output phase voltage  1949.059873                      V
>>>37         Generator Output phase current   876.527669        >500          A
>>>38                      Stator resistance     0.098504              ohm/phase
>>>39                 Synchronous inductance     0.009766                    p.u
>>>40                           Stator slots   702.000000                 A/mm^2
>>>41                           Stator turns   234.000000                  slots
>>>42                Conductor cross-section   233.712239           5      turns
>>>43                Stator Current density      3.750457         3-6       mm^2
>>>44               Specific current loading    60.000000          60       kA/m
>>>45                  Generator Efficiency     93.015124        >93%          %
>>>46                              Iron mass    58.999520                   tons
>>>47                            Magnet mass     1.897407                   tons
>>>48                            Copper mass     5.700674                   tons
>>>49                           Mass of Arms    34.452569                   tons
>>>50                             Total Mass   101.050170                   tons
>>>51                    Total Material Cost   257.614977                     k$

These structural design dimensions can be used to create CAD models and the resulting design can be validated by structural analysis. The electromagnetic design dimensions from the excel file can be read into the the MATALB script [Par_GeneratorSE_FEMM_PMSG.m] to populate the 2D finite element mesh model in FEMM and perform a magnetostatic analysis.

As a second example, we will design a gear driven Doubly fed Induction generator(DFIG).  The DFIG design 
relies on inputs such as machine rating, target overall drivetrain efficiency, gearbox efficiency,
high speed shaft speed( or gear ratio).It also requires specification of shear stress, material properties( densities), specific material costs. 
The main design variables are initialized with an objective and driver. The designs are computed analytically and checked against predefined constraints 
to meet objective functions.In addition, calculations of mass properties are also made.
We now instantiate the DFIG object which automatically updates the mass, costs, efficiency and performance variables based on the supplied inputs.  

.. literalinclude:: examples/example.py
    :start-after: # 5 ---
    :end-before: # 5 ---



.. literalinclude:: examples/example.py
    :start-after: # 6 ---
    :end-before: # 6 ---


The resulting system and component properties can then be printed. 

.. literalinclude:: examples/example.py
    :start-after: # 7 ---
    :end-before: # 7 ---

The optimized design dimensions together with the performance parameters are printed on screen and available in an output file in a Microsoft Excel format(file name [DFIG_5.0MW.xlsx]). 
The results should appear as below listing the parameters together with their limits where applicable:

>>>                               Parameters           Values         Limit				Units
>>>0                                  Rating                5												MW
>>>1                      Objective function  DFIG_Cost.Costs
>>>2                        Air gap diameter         0.986335												m
>>>3                           Stator length          1.06174												m
>>>4                                   K_rad          1.07645     (0.2-1.5)					-
>>>5                          Diameter ratio          1.36611    (1.37-1.4)					-
>>>6                       Pole pitch(tau_p)          516.444												mm
>>>7                  Number of Stator Slots               90												-
>>>8                 Stator slot height(h_s)              100												mm
>>>9                        Slots/pole/phase                5												-
>>>10                 Stator slot width(b_s)          15.4933												mm
>>>11               Stator slot aspect ratio           6.4544        (4-10)					-
>>>12                Stator tooth width(b_t)          18.9363												mm
>>>13               Stator yoke height(h_ys)          75.5535												mm
>>>14                            Rotor slots               72												-
>>>15                Rotor yoke height(h_yr)          75.5535												mm
>>>16                 Rotor slot height(h_r)          99.8798												mm
>>>17                  Rotor slot width(b_r)          19.2821												mm
>>>18                Rotor Slot aspect ratio          5.17991												-
>>>19                 Rotor tooth width(b_t)           23.567												mm
>>>20              Peak air gap flux density         0.733576     (0.7-1.2)					T
>>>21  Peak air gap flux density fundamental         0.682788												T
>>>22          Peak stator yoke flux density          1.59611            2.					T
>>>23           Peak rotor yoke flux density          1.59611            2.					T
>>>24         Peak Stator tooth flux density          1.33377            2.					T
>>>25          Peak rotor tooth flux density          1.67436            2.					T
>>>26                             Pole pairs                3												-
>>>27             Generator output frequency               60												Hz
>>>28         Generator output phase voltage          1734.26    (500-5000)					V
>>>29         Generator Output phase current          750.958												A
>>>30                           Optimal Slip             -0.3  (-0.002-0.3)					-
>>>31                           Stator Turns               30										
>>>32                Conductor cross-section          294.373												mm^2
>>>33                 Stator Current density          2.55104         (3-6)					A/mm^2
>>>34               Specific current loading          43.6228           <60					kA/m
>>>35                      Stator resistance       0.00201965												-
>>>36              Stator leakage inductance      0.000214916												-
>>>37            Excited magnetic inductance         0.034612												-
>>>38                    Rotor winding turns               98												-
>>>39                Conductor cross-section           123.21												mm^2
>>>40                  Magnetization current          40.0191												A
>>>41                               I_mag/Is         0.138807     (0.1-0.3)					-
>>>42                  Rotor Current density          5.99991         (3-6)					A/mm^2
>>>43                        Rotor resitance       0.00510091												-
>>>44               Rotor leakage inductance      0.000695358												-
>>>45                   Generator Efficiency           98.339												%
>>>46          Overall drivetrain Efficiency          93.9137            93					%
>>>47                              Iron mass          6.59953												Tons
>>>48                            Copper mass          1.09704												Tons
>>>49                  Structural Steel mass          17.4623												Tons
>>>50                             Total Mass          25.1589												Tons
>>>51                    Total Material Cost          17.6752     									$1000

These result file can be read into the the MATALB script [Par_GeneratorSE_FEMM_DFIG.m] to populate the 2D finite element mesh model in FEMM and perform a magnetostatic analysis.