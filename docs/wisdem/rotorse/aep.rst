
.. _aep:

-------------------------------
Regulation trajectory and AEP
-------------------------------

A conventional variable-speed variable-pitch turbine features four region I, II, III, and IV, plus an intermediate region II1/2:

1. Region I: the turbine does not generate any power since the wind is below the cut-in speed, which is usually set at 3 or 4 m/s.
2. Region II: the turbines operates at its specified tip-speed ratio until either rated power or the maximum rotation speed is reached 
3. Region II1/2: if the maximum rotor speed is reached before rated power, the turbine maintains its rotor speed, therefore reducing the tip speed ratio, and pitches to maximize the power coefficient
4. Region III: the blades are pitched and the turbine generates its nameplate power at constant rotor speed and generator torque.
5. Region IV: the wind is beyond cut out speed, the turbine is shutdown, and no power is generated.

WISDEM implements this regulation trajectory by running multiple instances of :ref:`ccblade`. For regions II1/2 and III, sub-optimization routines run CCBlade iteratively to identify the right combinations of tip speed ratio and blade pitch angle to maximize power (region II1/2) or to maintain it constant (region III). These sub-optimization routines run at every wind speed and have a non negligible computational costs (total computational time is in the order of seconds). The code therefore offers the possibility to the user to compute the power curve only in region II or in region II and II1/2 and simply adopt constant power in region III. The disadvantage of such approach is that the regulation trajectory in terms of tip speed ratio and pitch angle in regions II1/2 and/or III is no longer available.


Once the regulation trajectory is completed, the annual energy production (AEP) (in kWh) is calculated as

.. math::
    AEP = 8760\ loss \int_{V_{in}}^{V_{out}} P(V) f(V) dV

where P is in Watts, loss is the drivetrain efficiency, and f(V) is a probability density function for the site.

Notably, WISDEM does not implement any peak shaving of the aerodynamic thrust. 
