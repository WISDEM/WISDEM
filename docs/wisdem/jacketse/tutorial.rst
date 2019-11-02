.. _tutorial-label: 5 MW Jacket Tower in 30 m Waters

Tutorial
--------

Two examples are included in this tutorial section: simulation of a jacket-tower, and optimization of a jacket-tower support structure.

5-MW Jacket Tower in 30 m Waters
================================

.. currentmodule:: wisdem.jacketse.jacket

This example demonstrates how to setup and run *analysis* for a jacket-tower support structure.
The structure looks as shown in Figure :ref:`jacketTut-fig` :

.. _jacketTut-fig:

.. figure:: /images/jacketse/jacket_tower_tut.*
    :width: 6in
    :align: center

    Jacket-tower structure for the tutorial example. 

First, we set up the jacket geometry parameters:

.. literalinclude:: ../../../wisdem/jacketse/jacket.py
    :language: python
    :start-after: #--- Set Jacket Input Parameters ---#
    :end-before: #Soil inputs

Then we set up various component inputs. Start with the soil stratigraphy:

.. literalinclude:: ../../../wisdem/jacketse/jacket.py
    :language: python
    :start-after: #Soil inputs
    :end-before: #Water and wind inputs

Then assign water and wind environmental parameters:

.. literalinclude:: ../../../wisdem/jacketse/jacket.py
    :language: python
    :start-after: #Water and wind inputs
    :end-before: ## if turbine_jacket

Now it is time to assign properties for the various member classes of the structure.

We will follow a bottom up approach, where the definitions are offered from the bottom members towards the top ones.

Start with the piles:

.. literalinclude:: ../../../wisdem/jacketse/jacket.py
    :language: python
    :start-after: #Pile data
    :end-before: #Legs data

Then continue with the main legs:

.. literalinclude:: ../../../wisdem/jacketse/jacket.py
    :language: python
    :start-after: #Legs data
    :end-before: #Xbrc data

Then move on to the X-braces:

.. literalinclude:: ../../../wisdem/jacketse/jacket.py
    :language: python
    :start-after: #Xbrc data
    :end-before: #Mbrc data

In this example we are considering the Mud-braces. They normally mitigate stress concentrations at the pile-leg connections and provide some torsional stiffness:

.. literalinclude:: ../../../wisdem/jacketse/jacket.py
    :language: python
    :start-after: #Mbrc data
    :end-before: #Hbrc data

In this case we are not considering a top horizontal brace below the main perimeter girder of the TP, so we will set its *ndiv* to 0:

.. literalinclude:: ../../../wisdem/jacketse/jacket.py
    :language: python
    :start-after: #Hbrc data
    :end-before: #TP data

The next step consists of declaring the transition piece properties.

.. literalinclude:: ../../../wisdem/jacketse/jacket.py
    :language: python
    :start-after: #TP data
    :end-before: #Tower data

Then the tower geometry is assigned:

.. literalinclude:: ../../../wisdem/jacketse/jacket.py
    :language: python
    :start-after: #Tower data
    :end-before: ## if turbine_jacket

Then the RNA mass properties are specified:

.. literalinclude:: ../../../wisdem/jacketse/jacket.py
    :language: python
    :start-after: #RNA loads
    :end-before: ## if turbine_jacket

Then assign the RNA aerodynamic loads:

.. literalinclude:: ../../../wisdem/jacketse/jacket.py
    :language: python
    :start-after: #RNA loads
    :end-before: ## if turbine_jacket

Finally auxiliary parameters for the Frame3DD solver may be assigned:

.. literalinclude:: ../../../wisdem/jacketse/jacket.py
    :language: python
    :start-after: #Frame3DD
    :end-before: ## if turbine_jacket

It is then time to launch the assembly and pass all the inputs to it; note that the assembly is called with parameters depending on the selected inputs:

.. literalinclude:: ../../../wisdem/jacketse/jacket.py
    :language: python
    :start-after: #-----Launch the assembly-----#
    :end-before: #--- RUN OPTIMIZATION ---#

Then run the assembly:

.. literalinclude:: ../../../wisdem/jacketse/jacket.py
    :language: python
    :start-after: #--- RUN JACKET ---#
    :end-before: # ---------------

You may print some of the results of the analysis:

.. literalinclude:: ../../../wisdem/jacketse/jacket.py
    :language: python
    :start-after: #Now show results of modal analysis
    :end-before: #Plot geometry

They should look as shown here:

::

>>>7 nodes in the constant-OD segment of the tower
>>>('>>>>>>>>>>  needed embedment Lp0=', 29.499564957098116)
>>>First two Freqs.= 0.1983 and 0.1994 Hz
>>>jacket+TP(structural+lumped) mass (no tower, no piles) [kg] = 1154582
>>>tower mass [kg] = 335018
>>>TP mass structural + lumped mass [kg] = 335354
>>>piles (all) mass (for assigned (not optimum) Lp [kg] =  48844
>>>frame3dd model mass (structural + TP lumped) [kg] = 1489600
>>>Tower Top Displacement in Global Coordinate System [m] =0.4843
>>>MAX member compression-bending utilization at joints = 0.5608
>>>MAX member tension utilization at joints = 0.6413
>>>MAX X-joint  utilization at joints = 0.2600
>>>MAX K-joint  utilization at joints = 0.3667

If you plot the utilization of the tower, you should get something as in Figure :ref:`utilization-fig`, where
the Von Mises stress, global and shell buckling utilizations are shown along the tower span.  Each curve represents material utilization and so should be <1 for feasibility.

.. _utilization-fig:

.. figure:: /images/jacketse/util_tut.*
    :width: 6in
    :align: center

    Utilization along tower height for: Von-Mises/yield; shell buckling; global buckling.

.. <!-- ________________________________________ !>

.. _MDAOSNOPT-label:

Jacket-Tower Optimization - OpenMDAO Internal Optimization Driver
=================================================================

We begin with the same setup as the previous section, but now we need to set up the optimizer and thus also offer bounds for the design variables:

.. literalinclude:: ../../../wisdem/jacketse/jacket.py
    :language: python
    :start-after:  #Set Optimization Bounds for the various variables:
    :end-before: # --- optimizer imports ---


Also import additional modules for optimization:

.. literalinclude:: ../../../wisdem/jacketse/jacket.py
    :language: python
    :start-after: # --- optimizer imports
    :end-before: # ---

The optimizer must first be selected and configured, in this example use SNOPT.

.. literalinclude:: ../../../wisdem/jacketse/jacket.py
    :language: python
    :start-after: # --- Setup Optimizer ---
    :end-before: if SNOPTflag:

.. literalinclude:: ../../../wisdem/jacketse/jacket.py
    :language: python
    :start-after: if SNOPTflag:
    :end-before: else:


We now set the objective, and in this example it is normalized to be of order 1 for better convergence behavior.

.. literalinclude:: ../../../wisdem/jacketse/jacket.py
    :language: python
    :start-after: # --- Objective ---
    :end-before: # ----------------------

The batter,pile OD and thickness, Embedment Length, Leg OD and thickness, X-brace OD and thickness,Mud-brace OD and thickness,Tower base OD and DTR, tower-top OD and DTR, and height of constant cross-section segment are added as design variables.

.. literalinclude:: ../../../wisdem/jacketse/jacket.py
    :language: python
    :start-after: # --- Design Variables
    :end-before: #--- Constraints ---#

Constraints are then added; note that we are after a target first eigenfrequeccy of 0.22 Hz:

.. literalinclude:: ../../../wisdem/jacketse/jacket.py
    :language: python
    :start-after: #--- Constraints ---#
    :end-before: # ---

A recorder is added to display each iteration to the screen.

.. literalinclude:: ../../../wisdem/jacketse/jacket.py
    :language: python
    :start-after: # --- recorder ---
    :end-before: # ---


Now the optimization can be run.

.. literalinclude:: ../../../wisdem/jacketse/jacket.py
    :language: python
    :start-after: #--- RUN JACKET ---#
    :end-before: # ---------------
    

THe final configuration skeleton is shown in Figure :ref:`jacket_tower_tutOpt`.
If you plot the utilization of the tower, you should get something as in Figure :ref:`utilopt-fig`, where
the Von Mises stress, global and shell buckling utilizations are shown along the tower span.  Each is a utilization and so should be <1 for feasibility.

::

>>>7 nodes in the constant-OD segment of the tower
>>>('>>>>>>>>>>  needed embedment Lp0=', 41.880993761055052)
>>>First two Freqs.= 0.2143 and 0.2154 Hz
>>>jacket+TP(structural+lumped) mass (no tower, no piles) [kg] = 1135683
>>>tower mass [kg] = 396546
>>>TP mass structural + lumped mass [kg] = 347611
>>>piles (all) mass (for assigned (not optimum) Lp [kg] =  94910
>>>frame3dd model mass (structural + TP lumped) [kg] = 1532229
>>>Tower Top Displacement in Global Coordinate System [m] =0.4437
>>>MAX member compression-bending utilization at joints = 0.5559
>>>MAX member tension utilization at joints = 0.6590
>>>MAX X-joint  utilization at joints = 0.2586
>>>MAX K-joint  utilization at joints = 0.3807

The total mass of the jacket, tower, and TP (both structural and lumped mass), and piles is 1,627,139 kg.
Also note that the first natural frequencies do NOT match the requirement >0.22 Hz.

.. _jacket_tower_tutOpt:

.. figure:: /images/jacketse/jacket_tower_tutOpt.*
    :width: 6in
    :align: center

    Jacket-tower structure for the tutorial example after OpenMDAO driven optimization via SNOPT. 

    
.. _utilopt-fig:

.. figure:: /images/jacketse/util_tutOpt.*
    :width: 6in
    :align: center

    Utilization along tower height for: Von-Mises/yield; shell buckling; global buckling. Optimization obtained via OpenMDAO pyOPT driver with SNOPT.
        
        
.. <!-- ________________________________________ !>
.. _my-reference-label:

Jacket-Tower Optimization - External Optimization via PyOPT
===========================================================

.. currentmodule:: wisdem.jacketse.JacketOpt_PyOPT

For this tutorial, we use an auxiliary module *JacketOpt_PyOPT.py* and the auxiliary input file *MyJacketInputs.py*. 
The new module is just a wrapper for jacketSE and contains: calls to the pyOPT optimization package to perform the optimization, 
objective function and constraints. The input file contains the same jacket input information as in the previous example.

We simply run a SNOPT optimization case by issuing: ::

    >>>python JacketOpt_PyOPT.py MyJacketInputs.py True

The run terminates with the following results:

::

>>>7 nodes in the constant-OD segment of the tower
>>>('>>>>>>>>>>  needed embedment Lp0=', 27.634632160831377)
>>>Jwrapper SOLUTION: bat= 9.48, Dpile= 2.45, tpile=0.025, Lp= 27.6 Dleg 1.14, tleg0.032
>>>        dck_width =12.76,    Dbrc= 1.00, tbrc=0.025, Dmudbrc= 1.00, tmudbrc=0.025
>>>from Jwrapper Db= 6.38, DTRb=139.88, Dt= 3.21, DTRt=139.88,H2twrfrac= 0.25, Dgir= 1.00,tgir=0.025, Twrmass=413487.941, PilesMass =187056.839, TPmass= 1.424e+05, Frame3DD+Piles Totmass=1477428.623

::

>>>Minimum mass Mjacket, MPiles, TPmass = 1290371.783880 187056.839229 142411.694364
>>>Minimum mass Tower, Jacket(no tower no piles) = 413487.941068 876883.842813
>>>Minimum found at Dpile=2.451534, tpile=0.025400  Lp=27.634661
>>>Minimum found at Dbrc=1.000000, tbrc=0.025400
>>>Minimum found at Dbrcmud=1.000000, tbrcmud=0.025400
>>>Minimum found at batter=9.476407, dckwidth=12.758457, Dleg=1.138723, tleg=0.031565,
>>>Minimum found at Dgir=1.000084, tgir=0.025400
>>>Minimum found at Db=6.379229 DTRb=139.882073 Dt=3.208883 DTRt=139.882073 H2frac=0.250000
>>>Minimum found at Freq 0.220003
>>>Minimum found at GLutil=0.734957 EUutil=0.195177
>>>Minimum found at Mudline Footprint=20.758694
>>>Elapsed time:  4514.93799996 seconds
>>>Execution count:  3422


The total mass of the jacket, tower, and TP (both structural and lumped mass), and piles is  1,677,428 kg. 
Note that in this case the frequency constraint is met.

.. _jacket_ExtPySnopt-fig:

.. figure:: /images/jacketse/jacket_ExtPySnopt.*
    :width: 6in
    :align: center

    Jacket-tower structure for the tutorial example after OpenMDAO driven optimization via SNOPT. 

    
.. _utilextopt-fig:
    
.. figure:: /images/jacketse/util_ExtPySnopt.*
    :width: 6in
    :align: center

    Utilization along tower height for: Von-Mises/yield; shell buckling; global buckling. Optimization obtained via pyOPT SNOPT.


.. <!-- ________________________________________ !>

Jacket-Tower Optimization - External Optimization via Python Cobyla
===================================================================

.. currentmodule:: wisdem.jacketse.JacketOpt_ExtCobyla

For this tutorial, we use an auxiliary module *JacketOpt_ExtCobyla.py* and the same auxiliary input file *MyJacketInputs.py* as above. 
The new module is just a wrapper for jacketSE and contains: calls to the python function *scipy.optimize.fmin_cobyla* to perform the optimization, 
objective function and constraints. The input file contains the same jacket input information as in the previous example.

We start by simply running: ::

    >>>python JacketOpt_ExtCobyla.py MyJacketInputs.py 

The run terminates with the following results:

::

>>>7 nodes in the constant-OD segment of the tower
>>>('>>>>>>>>>>  needed embedment Lp0=', 46.965381137541016)
>>>Jwrapper SOLUTION: bat=15.00, Dpile= 1.00, tpile=0.025, Lp= 47.0 Dleg 1.50, tleg 0.025
>>>            dck_width =13.92, Dbrc= 1.00, tbrc=0.025, Dmudbrc= 1.00, tmudbrc=0.025
>>>from Jwrapper Db= 6.96, DTRb=200.00, Dt= 3.62, DTRt=200.00,H2twrfrac= 0.25, Dgir= 1.00,tgir=0.025, Twrmass=349424.881, PilesMass =127705.311, TPmass= 1.353e+05, Frame3DD+Piles Totmass=1330334.683


::

>>>Minimum mass Mjacket, MPiles, TPmass = 1202629.371883 127705.310878 135269.829145
>>>Minimum mass Tower, Jacket(no tower no piles) = 349424.880736 853204.491147
>>>Minimum found at Dpile=1.000000, tpile=0.025400  Lp=46.965381
>>>Minimum found at Dbrc=1.000000, tbrc=0.025400
>>>Minimum found at Dbrcmud=1.000000, tbrcmud=0.025400
>>>Minimum found at batter=15.000000, dckwidth=13.921647, Dleg=1.497250, tleg=0.025400,
>>>Minimum found at Dgir=1.000000, tgir=0.025400
>>>Minimum found at Db=6.960824 DTRb=200.000000 Dt=3.620849 DTRt=200.000000 H2frac=0.250000
>>>Minimum found at Freq 0.220000
>>>Minimum found at GLutil=0.679969 EUutil=0.291541
>>>Minimum found at Mudline Footprint=17.809106 beta3D=54.979833


The overall mass (jacket, tower, TP (structural and lumped mass), and piles amounts to 1,530,334 kg, which is less than the optimum found in the previous optimizations.

Cobyla seems to perform better than the other optimization options.

.. _JacketOpt_CobyOptConfig-fig:
 
  .. figure:: /images/jacketse/JacketOpt_CobyOptConfig.*
     :width: 6in
     :align: center
 
     Jacket-tower structure for the tutorial example after OpenMDAO driven optimization via SNOPT. 

 
.. _util_tutCobyOpt-fig:
    
  .. figure:: /images/jacketse/util_tutCobyOpt.*
     :width: 6in
     :align: center
 
     Utilization along tower height for: Von-Mises/yield; shell buckling; global buckling. Optimization obtained via fmin_cobyla.

.. <!-- ________________________________________ !>

.. _MDAOcoby:

Jacket-Tower Optimization - Optimization via OpenMDAOs Cobyla (non-PyOPT)
=========================================================================

.. currentmodule:: wisdem.jacketse.jacket

For this tutorial, we repeat what was done in Section :ref:`MDAOSNOPT-label` 
and this time use the simple COBYLAdriver, not the PyOPTdriver. 
The input file contains the same jacket input information as in the previous example.

We start by simply running: ::

    >>>python jacket.py True Cobyla 

The run terminates with the following results:

::

>>>
>>>   Return from subroutine COBYLA because the MAXFUN limit has been reached.
>>>
>>>   NFVALS =    2000   F = 1.450805E+00    MAXCV = 1.500592E-06
>>>   X = 1.204965E+01   1.494575E+00   2.540000E-02   3.000000E+01   1.488532E+00
>>>       2.540000E-02   1.000000E+00   2.540000E-02   1.000000E+00   2.540000E-02
>>>       6.194941E+00   1.300558E+02   3.144349E+00   1.500515E+02   2.500000E-01
>>>First two Freqs.= 0.2200 and 0.2211 Hz
>>>jacket+TP(structural+lumped) mass (no tower, no piles) [kg] = 1130736
>>>tower mass [kg] = 410342
>>>TP mass structural + lumped mass [kg] = 346367
>>>piles (all) mass (for assigned (not optimum, unless optimization is run) Lp [kg] = 109657
>>>frame3dd model mass (structural + TP lumped) [kg] = 1541078
>>>frame3dd model mass (structural + TP lumped) + Pile Mass [kg] = 1650735
>>>Tower Top Displacement in Global Coordinate System [m] =0.4146
>>>MAX member compression-bending utilization at joints = 0.5647
>>>MAX member tension utilization at joints = 0.6596
>>>MAX X-joint  utilization at joints = 0.2544
>>>MAX K-joint  utilization at joints = 0.3807


The overall mass (jacket, tower, TP (structural and lumped mass), and piles amounts to 1,650,735 kg, which is still greater than the optimum found with teh external Cobyla, however, the maximum number of iterations was reached in this case.

Cobyla seems to perform better than the other optimization options.

.. _JacketOpt_MDAOCobyConfig-fig:
 
  .. figure:: /images/jacketse/JacketOpt_MDAOCobyConfig.*
     :width: 6in
     :align: center
 
     Jacket-tower structure for the tutorial example after OpenMDAO driven optimization via COBYLA driver. 

 
.. _util_tutMDAOCoby-fig:
    
  .. figure:: /images/jacketse/util_tutMDAOCoby.*
     :width: 6in
     :align: center
 
     Utilization along tower height for: Von-Mises/yield; shell buckling; global buckling. Optimization obtained via MDAO COBYLA driver.

