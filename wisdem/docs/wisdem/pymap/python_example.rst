Python Example
==============
.. _quick_start:

Static Configuration
--------------------

This example will be run against the :ref:`baseline input file <baseline_example>`. 
The water depth is :math:`350` meters. 
The space preceeding the ``repeat 120 240`` solver option flag is removed to enable duplication of the mooring geometry twice with :math:`120^\circ` and :math:`240^\circ` offsets about the :math:`Z` axis. 

.. literalinclude:: /examples/pymap/sphinx_example.py
   :language: python

Output
~~~~~~

Two outputs are produced executing the stript above. 
Information explicitly requested is printed to the command line:

.. code-block:: bash

   MAP++ (Mooring Analysis Program++) Ver. 1.20.10 Mar-22-2016
   MAP++ environment properties (set externally)...
       Gravity constant          [m/s^2]  : 9.81
       Sea density               [kg/m^3] : 1025.00
       Water depth               [m]      : 350.00
       Vessel reference position [m]      : 0.00 , 0.00 , 0.00
   
   Linearized stiffness matrix with 0.0 vessel displacement:
   
   [[  1.99e+04  -3.78e-03   5.19e-03  -4.89e-02  -2.00e+05  -1.77e-02]
    [  1.18e-03   1.99e+04  -1.06e-02   2.00e+05   3.50e-02  -6.17e-01]
    [  2.49e-03  -1.01e-03   2.27e+04   2.21e-03   2.23e-01  -2.12e-01]
    [  1.95e-03   2.00e+05  -7.14e-03   2.17e+08   1.10e-02  -5.23e+01]
    [ -2.00e+05   3.33e-04   4.85e-01  -4.89e-02   2.17e+08  -2.19e-02]
    [  8.83e-04  -5.59e-01   1.12e-03  -8.53e+01  -7.90e-02   1.41e+08]]
   
   Linearized stiffness matrix with 5.00 surge vessel displacement:
   
   [[  1.96e+04  -2.58e-05   1.17e+03   9.61e-03  -2.15e+05  -1.67e-01]
    [ -4.57e-04   2.07e+04   1.41e-03   1.81e+05  -5.24e-02   1.72e+03]
    [  1.17e+03  -3.32e-04   2.32e+04  -5.38e-03  -1.19e+04   1.12e-03]
    [  1.05e-03   2.00e+05   1.51e-03   2.17e+08   4.25e-02  -5.21e+01]
    [ -2.00e+05  -8.91e-05   4.79e-01   5.43e-03   2.17e+08   6.60e-02]
    [  2.10e-03  -5.60e-01   5.77e-03  -8.52e+01   1.07e-01   1.41e+08]]
   Line 0: H = 597513.33 [N]  V = 1143438.75 [N]
   Line 0: Fx = -597513.33 [N]  Fy = -0.00 [N]  Fz = 1143438.75 [N]
   

A figure is also produced to show the mooring geometry with a :math:`5` meter vessel offset:

.. Note::
   The default units for the linearized stiffness matrix are [N/m], [N/rad], [Nm/m], and [Nm/rad]. See
   :ref:`the section on the linearized stiffness matrix <linearized_matrix_units>` in the FAQ for more information. 

.. figure:: nstatic/pymap/example.png
    :align: center
    :width: 60%

    Fig. 7
    
    .. raw:: html

	<font size="2"><center><i><b>
	Vessel kinematic breakdown to describe fairlead position relative to the origin.
	</b></i></center></font>

Time-Marching for Dynamics Simulation
-------------------------------------

.. literalinclude:: /examples/pymap/driver.py

Output
~~~~~~

.. figure:: nstatic/pymap/time_domain_1.png
    :align: center
    :width: 60%

    Fig. 8
    
    .. raw:: html

	<font size="2"><center><i><b>
	Mooring footprint
	</b></i></center></font>

.. figure:: nstatic/pymap/time_domain_2.png
    :align: center
    :width: 60%

    Fig. 9
    
    .. raw:: html

	<font size="2"><center><i><b>
	Precribed displacement of the vessel fed to MAP++.
	</b></i></center></font>

.. figure:: nstatic/pymap/time_domain_3.png
    :align: center
    :width: 60%

    Fig. 10
    
    .. raw:: html

	<font size="2"><center><i><b>
	Fairlead line tensions for the respective lines. 
	</b></i></center></font>

