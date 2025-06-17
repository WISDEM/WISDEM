.. _blade_opt_tutorial-label:

3a. Blade Optimization Example
-----------------------------

This example walks through a blade optimization problem with increasing complexity.

All of the iterations use the same geometry input file, ``BAR-USC.yaml``, which describes a baseline design from the NREL-Sandia Big Adaptive Rotor (BAR) project described in this GitHub [repository](https://github.com/NREL/BAR_Designs).
This blade uses carbon fiber-reinforced polymer in the spar cap design.  The same ``modeling_options.yaml`` file is also common to all iterations and shows that all WISDEM modules are called. The file has dozens of optional inputs hidden. The full list of inputs is available among the :ref:`modeling-options`.
The example file runs four cases one after the other for testing purposes. To run the cases one by one, make sure to comment out all cases at lines 15-18 except the case that should run.


Baseline Design
===============

Whenever conducting a design optimization, it is helpful to first run the starting point design and evaluate the baseline performance. The file, ``analysis_options_no_opt.yaml``, does not have any optimization variables activated and is meant for this purpose.  Outputs are generated in the ``outputs`` directory.

.. code-block:: bash

    $ wisdem BAR-USC.yaml modeling_options.yaml analysis_options_no_opt.yaml


Simple Aerodynamic Optimization
===============================

The file, ``analysis_options_aero.yaml``, is used to run a blade twist and chord optimization together with rotor tip speed ratio. 
This is activated by turning on the appropriate design variable flags in the file,

.. literalinclude:: /../examples/03_blade/analysis_options_aero.yaml
    :language: yaml
    :start-after: design_variables:
    :end-before: merit_figure:

TSR is allowed to vary between 8 and 11. For twist, WISDEM can optimize blade twist either by assigning design variables to it, or in an inverse approach by setting a desired angle of attack. In the first case, we set :code:`flag` to True. In the latter case, we set :code:`flag` to False and we set :code:`inverse` to True. Here we pick the latter approach. If we activated the :code:`inverse` flag, :code:`n_opt` will be used to smoothened the twist distribution with a spline with this number of points along span to mimick manufacturability constraints. Angle of attack can be set either to achieve maximum airfoil efficiency along span, in this case set the flag :code:`inverse_target: 'max_efficiency'`, or for a predefined margin to stall. In the latter case, use :code:`inverse_target: 'stall_margin'`. The value of margin to stall is set in the constraints.

If you prefer to optimize directly with assigned design variables, turn `flag` to True and next adjust the indices controlling which of these control points can be varied by the optimizer, and which are instead locked. In the twist section, we would set :code:`index_start` to 2 (this means that the first 2 of 8 spanwise control points sections are locked), whereas we would let all other 6 spanwise control points in the hands of the optimizer. We do this by setting :code:`index_end` to 8. You can also adjust the maximum decrease and increase that the optimizer can apply to the twist in radians. 

For chord, we set the :code:`flag` to True. Again, adjust :code:`n_opt`, :code:`index_start`, and :code:`index_end`. The chord bounds are set to be multiplicative starting from the initial chord. We don't optimize chord at blade tip (BEM isn't good at it), so set :code:`n_opt` to 8 and :code:`index_end` to 7.

Aero power coefficient Cp, computed by default at 5 m/s in region II, is the quantity that we maximize. An alternative is annual energy production. In that case use :code:`AEP`

.. code-block:: yaml

    merit_figure: Cp

We set a target stall margin using the same *flag* type of setting, with a value of :math:`5.7^{\circ} deg \approx 0.1 rad`. In case of direct optimization of twist, this value can also be activated as a constraint. We also often like to constrain chord to a maximum and we can set a constrain guiding the diameter of the blade root bolt circle (use this only if :code:`index_start` is set to 0 for the chord). The slope of the chord can be constrained so that chord is guaranteed to decrease monothonically along span after the section of max chord.  Lastly, the blade root moment coefficient can also constrained to design low induction rotors, look at DOI [10.1088/1742-6596/1618/4/042016](https://doi.org/10.1088/1742-6596/1618/4/042016) to learn more. Note that users can also take advantage of user defined constraints, see Example 11, and constrain any other quantity, such as rotor solidity.  

.. literalinclude:: /../examples/03_blade/analysis_options_aero.yaml
    :language: yaml
    :start-after: constraints:
    :end-before: driver:

The maximum iteration limit currently used in the file is 1, to keep the examples short.  However, if you want to see more progress in the optimization, change :code:`max_iter` to 20 (or 100, or more).

.. literalinclude:: /../examples/03_blade/analysis_options_aero.yaml
    :language: yaml
    :start-after: driver:
    :end-before: recorder:

Now to run the optimization we do,

.. code-block:: bash

    $ wisdem BAR-USC.yaml modeling_options.yaml analysis_options_aero.yaml

or adjust the list of analysis options yaml files in blade_driver.py and run:

.. code-block:: bash

    $ python blade_driver.py

or to run in parallel using multiple CPU cores (Mac or Linux only). 
We have 6 chord design variables and tsr, we use forward finite differencing, so use up to 7 cores. Remember to comment out all the unused analysis options in `blade_driver.py`!

.. code-block:: bash

    $ mpirun -np 7 python blade_driver.py

where the ``blade_driver.py`` script is:

.. literalinclude:: /../examples/03_blade/blade_driver.py
    :language: python

When run with mpi, the CPU run time is approximately 3 minutes. As the script runs, you will see some
output to your terminal, such as performance metrics and some analysis warnings.
Once the optimization terminates, type in the terminal:

.. code-block:: bash

    $ compare_designs BAR_USC.yaml outputs_aero/blade_out.yaml --labels Init Opt

This script compares the initial and optimized designs.

Another good command is 

.. code-block:: bash

    $ python load_log.py

Some screen output is generated, as well as plots (contained in the `outputs` folder), such as in :numref:`_fig_opt1_chord`.

.. _fig_opt1_chord:
.. figure:: /images/blade/chord.png
    :height: 4in
    :align: center

    Initial versus optimized chord profiles

.. _fig_opt1_twist:
.. figure:: /images/blade/twist_opt.png
    :height: 4in
    :align: center

    Initial versus optimized twist profiles

As well as convergence plots in the `outputs_aero` folder:

.. _fig_opt1_cp:
.. figure:: /images/blade/CP_trend.png
    :height: 4in
    :align: center

    Cp value being optimized

.. _fig_opt1_tsr:
.. figure:: /images/blade/tsr_trend.png
    :height: 4in
    :align: center

    TSR value being optimized and converging to 8.5


Simple Structural Optimization
==============================

Next, we shift from an aerodynamic optimization of the blade to a structural optimization.  In this case, we make the following changes,

- The design variables as the thickness of the blade structural layers :code:`Spar_cap_ss` and :code:`Spar_cap_ps`
- The thickness is parameterized in 8 locations along span and can vary between 70 and 130% of the initial value (using the :code:`max_decrease` and :code:`max_increase` options)
- The merit figure is blade mass instead of AEP
- A max allowable strain of :math:`3500 \mu\epsilon` and the blade tip deflection constrain the problem, but the latter ratio is relaxed to 1.134

To run this optimization problem, we can use the same geometry and modeling input files, and the optimization problem is captured in ``analysis_options_struct.yaml``.  The design variables are,

.. literalinclude:: /../examples/03_blade/analysis_options_struct.yaml
    :language: yaml
    :start-after: design_variables:
    :end-before: merit_figure:

Just increase :code:`n_opt` to 8 and :code:`index_end` for both suction- and pressure-side spar caps. The objective function is set to:

.. code-block:: yaml

    merit_figure: blade_mass

and the constraints are,

.. literalinclude:: /../examples/03_blade/analysis_options_struct.yaml
    :language: yaml
    :start-after: constraints:
    :end-before: driver:

To run the optimization, just be sure to increase :code:`max_iter` to 10 and pass in this new analysis options,

.. code-block:: bash

    $ wisdem BAR-USC.yaml modeling_options.yaml analysis_options_struct.yaml

or, to use the Python driver, be sure to comment out lines 15, 16, and 18 and only leave this uncommented

.. code-block:: python

    wt_opt, modeling_options, analysis_options = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options_struct)

and then do,

.. code-block:: bash

    $ python blade_driver.py

(parallel calculation is also available if desired).

Once the optimization terminates, the same ``compare_designs`` script can be used again to plot the differences:

.. code-block:: bash

    $ compare_designs BAR-USC.yaml outputs_struct/blade_out.yaml --labels Init Opt

The relaxed tip deflection constraint compared to when the baseline was created allows the spar cap thickness to come down and the overall blade mass drops from 51 metric tons to 50 metric tons.  This is shown in :numref:`fig_opt2_spar` and :numref:`fig_opt2_mass`.

.. _fig_opt2_spar:
.. figure:: /images/blade/sc_opt.png
    :height: 4in
    :align: center

    Baseline versus optimized spar cap thickness profiles

.. _fig_opt2_mass:
.. figure:: /images/blade/mass.png
    :height: 4in
    :align: center

    Baseline versus optimized blade mass profiles. The bump at 70% span corresponds to the spanwise joint of BAR-USC.

WISDEM also estimates the final blade cost, which is reported in the output file outputs_struct/blade_out.csv in the field :code:`tcc.blade_cost`.
The blade cost after the optimization is USD 538.5k. The initial cost was USD 556.7k. The reduction is larger than blade mass because spar caps are made of expensive carbon fiber.

Aero-Structural Optimization
============================

Finally, we will combine the previous two scenarios and use the levelized cost of energy, LCOE, as a way to balance the power production and minimum mass/cost objectives.  This problem formulation is represented in the file, ``analysis_options_aerostruct.yaml``. The design variables are,

.. literalinclude:: /../examples/03_blade/analysis_options_aerostruct.yaml
    :language: yaml
    :start-after: design_variables:
    :end-before: merit_figure:

with rotor diameter, blade chord and twist, and spar caps thickness activated as a design variable.
Again, increase the field :code:`n_opt` to 8 for chord, twist, and spar caps thickness. Also, set :code:`index_end` to 8 for twist (optimize twist all the way to the tip) and to 7 for chord and spar caps (lock the point at the tip). Do not forget to set :code:`index_end` to 7 also in the strain constraints.

The objective function set to:

.. code-block:: yaml

    merit_figure: LCOE

and the constraints are,

.. literalinclude:: /../examples/03_blade/analysis_options_aerostruct.yaml
    :language: yaml
    :start-after: constraints:
    :end-before: driver:

One more change for this final example is tighter optimization convergence tolerance (:math:`1e-5`), because LCOE tends to move only a very small amount from one design to the next,

.. literalinclude:: /../examples/03_blade/analysis_options_aerostruct.yaml
    :language: yaml
    :start-after: driver:
    :end-before: form:

To run the optimization, just be sure to pass in this new analysis options

.. code-block:: bash

    $ wisdem BAR-USC.yaml modeling_options.yaml analysis_options_aerostruct.yaml

or, to use the Python driver, be sure run line 18 as above to be

.. code-block:: python

    wt_opt, modeling_options, analysis_options = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options_aerostruct)

and then do,

.. code-block:: bash

    $ python blade_driver.py


We can then use the ``compare_designs`` command in the same way as above to plot the optimization results, two of which are shown in, :numref:`fig_opt3_induction` and :numref:`fig_opt3_twist`.  With more moving parts, it can be harder to interpret the results.  In the end, LCOE is increased marginally because the initial blade tip deflection constraint, which is set to 1.4175 in the analysis options yaml, is initially violated and the optimizer has to stiffen up and shorten the blade. The rotor diameter is reduced from 206 m to 202.7 and twist is simultaneously adjusted to keep performance up.

.. _fig_opt3_induction:
.. figure:: /images/blade/induction2.png
    :height: 4in
    :align: center

    Baseline versus optimized induction profiles

.. _fig_opt3_twist:
.. figure:: /images/blade/twist_opt2.png
    :height: 4in
    :align: center

    Baseline versus optimized twist profiles
