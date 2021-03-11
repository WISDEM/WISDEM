.. _blade_opt_tutorial-label:

3. Blade Optimization Example
-----------------------------

This example walks through a blade optimization problem with increasing complexity.
All of the iterations use the same geometry input file, ``BAR0.yaml``, which describes a baseline design from the NREL-Sandia Big Adaptive Rotor (BAR) project.
This blade uses glass fiber-reinforced polymer in the spar cap design.  The same ``modeling_options.yaml`` file is also common to all iterations and shows that all modules are called, the airfoil polars are discretized at 200 angles of attack, etc.
The example file runs four cases one after the other for testing purposes. To run the cases one by one, make sure to comment out all cases at lines 15-18 except the case that should run. 


Baseline Design
===============

Whenever conducting a design optimization, it is helpful to first run the starting point design and evaluate the baseline performance. The file, ``analysis_options_no_opt.yaml``, does not have any optimization variables activated and is meant for this purpose.  Outputs are generated in the ``outputs`` directory.

.. code-block:: bash

    $ wisdem BAR0.yaml modeling_options.yaml analysis_options_no_opt.yaml


Simple Aerodynamic Optimization
===============================

The file, ``analysis_options_aero.yaml``, is used first to run a blade twist optimization. This is activated by turning on the appropriate design variable flags in the file,

.. literalinclude:: /../examples/03_blade/analysis_options_aero.yaml
    :language: yaml
    :start-after: blade:
    :end-before: structure:

We are also setting the number of spline control points at 8 and the maximum decrease and increase that the optimizer can apply to the twist (in radians) at each evenly spaced control point along the blade span.  We also need to set the objective function to be AEP with,

.. code-block:: yaml

    merit_figure: AEP

To better guide the optimization, we activate a stall margin constraint using the same *flag* type of setting, with a value of :math:`5^{\circ} deg \approx 0.087 rad`.

.. literalinclude:: /../examples/03_blade/analysis_options_aero.yaml
    :language: yaml
    :start-after: constraints:
    :end-before: tower:

The maximum iteration limit currently used in the file is 2, to keep the examples short.  However, if you want to see more progress in the optimization, change the following lines from:

.. literalinclude:: /../examples/03_blade/analysis_options_aero.yaml
    :language: yaml
    :start-after: driver:
    :end-before: form:

to:

.. code-block:: yaml

    tol: 1.e-3            # Optimality tolerance
    max_iter: 10          # Maximum number of minor design iterations
    solver: SLSQP         # Optimization solver. Other options are 'SLSQP' - 'CONMIN'
    step_size: 1.e-3      # Step size for finite differencing


Now to run the optimization we do,

.. code-block:: bash

    $ wisdem BAR0.yaml modeling_options.yaml analysis_options_aero.yaml

or we comment out lines 16, 17, and 18 in blade_driver.py and we do:

.. code-block:: bash

    $ python blade_driver.py

or to run in parallel using multiple CPU cores (Mac or Linux only):

.. code-block:: bash

    $ mpirun -np 4 python blade_driver.py

where the ``blade_driver.py`` script is:

.. literalinclude:: /../examples/03_blade/blade_driver.py
    :language: python

The CPU run time is approximately 5 minutes. As the script runs, you will see some
output to your terminal, such as performance metrics and some analysis warnings.
The optimizer might report that it has failed, but we have artificially limited the number of steps it can take during optimization, so that is expected.
Once the optimization terminates, type in the terminal:

.. code-block:: bash

    $ compare_designs outputs/BAR0.yaml outputs_aero/blade_out.yaml

This script compares the initial and optimized designs.
Some screen output is generated, as well as plots (contained in the `outputs` folder), such as in :numref:`fig_opt1_induction` and :numref:`fig_opt1_twist`. The twist optimization had to cope with a wider
margin to stall than the baseline was originally designed to.  The results show higher twist angles
towards the blade tip, but the AEP is only mildly reduced by 0.18%.

.. _fig_opt1_induction:
.. figure:: /images/blade/bladeopt1_induction.*
    :height: 4in
    :align: center

    Baseline versus optimized induction profiles

.. _fig_opt1_twist:
.. figure:: /images/blade/bladeopt1_twist.*
    :height: 4in
    :align: center

    Baseline versus optimized twist profiles


Simple Structural Optimization
==============================

Next, we shift from an aerodynamic optimization of the blade to a structural optimization.  In this case, we make the following changes,

- The design variables as the thickness of the blade structural layers :code:`Spar_cap_ss` and :code:`Spar_cap_ps`
- The thickness is parameterized in 8 locations along span and can vary between 70 and 130% of the initial value (using the :code:`max_decrease` and :code:`max_increase` options)
- The merit figure is blade mass instead of AEP
- A max allowable strain of :math:`3500 \mu\epsilon` and the blade tip deflection constrain the problem, but the latter ratio is relaxed from a safety factor of 1.4175 to 1.134

To run this optimization problem, we can use the same geometry and modeling input files, and the optimization problem is captured in ``analysis_options_struct.yaml``.  The design variables are,

.. literalinclude:: /../examples/03_blade/analysis_options_struct.yaml
    :language: yaml
    :start-after: blade:
    :end-before: structure:

with the objective function set to:

.. code-block:: yaml

    merit_figure: blade_mass

and the constraints are,

.. literalinclude:: /../examples/03_blade/analysis_options_struct.yaml
    :language: yaml
    :start-after: constraints:
    :end-before: tower:

To run the optimization, just be sure to pass in this new analysis options,

.. code-block:: bash

    $ wisdem BAR0.yaml modeling_options.yaml analysis_options_struct.yaml

or, to use the Python driver, be sure to comment out lines 15, 16, and 18 and only leave this uncommented

.. code-block:: python

    wt_opt, modeling_options, analysis_options = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options_struct)

and then do,

.. code-block:: bash

    $ python blade_driver.py

(parallel calculation is also available if desired).

Once the optimization terminates, the same ``compare_designs`` script can be used again to plot the differences:

.. code-block:: bash

    $ compare_designs outputs/BAR0.yaml outputs_aero/blade_out.yaml outputs_struct/blade_out.yaml

The relaxed tip deflection constraint compared to when the baseline was created allows the spar cap thickness to come down and the overall blade mass drops from 60.3 metric tons to 54.5 metric tons.  This is shown in :numref:`fig_opt2_spar` and :numref:`fig_opt2_mass`.

.. _fig_opt2_spar:
.. figure:: /images/blade/bladeopt2_sparcap.*
    :height: 4in
    :align: center

    Baseline versus optimized spar cap thickness profiles

.. _fig_opt2_mass:
.. figure:: /images/blade/bladeopt2_mass.*
    :height: 4in
    :align: center

    Baseline versus optimized blade mass profiles


Aero-Structural Optimization
============================

Finally, we will combine the previous two scenarios and use the levelized cost of energy, LCOE, as a way to balance the power production and minimum mass/cost objectives.  This problem formulation is represented in the file, ``analysis_options_aerostruct.yaml``. The design variables are,

.. literalinclude:: /../examples/03_blade/analysis_options_aerostruct.yaml
    :language: yaml
    :start-after: blade:
    :end-before: structure:

with blade chord also activated as a design variable. The objective function set to:

.. code-block:: yaml

    merit_figure: LCOE

and the constraints are,

.. literalinclude:: /../examples/03_blade/analysis_options_aerostruct.yaml
    :language: yaml
    :start-after: constraints:
    :end-before: tower:

One more change for this final example is tighter optimization convergence tolerance (:math:`1e-5`), because LCOE tends to move only a very small amount from one design to the next,

.. literalinclude:: /../examples/03_blade/analysis_options_aerostruct.yaml
    :language: yaml
    :start-after: driver:
    :end-before: form:

To run the optimization, just be sure to pass in this new analysis options

.. code-block:: bash

    $ wisdem BAR0.yaml modeling_options.yaml analysis_options_aerostruct.yaml

or, to use the Python driver, be sure run line 18 as above to be

.. code-block:: python

    wt_opt, modeling_options, analysis_options = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options_aerostruct)

and then do,

.. code-block:: bash

    $ python blade_driver.py


We can then use the ``compare_designs`` command in the same way as above to plot the optimization results, two of which are shown in, :numref:`fig_opt3_chord` and :numref:`fig_opt3_twist`.  With more moving parts, it can be harder to interpret the results.  In the end, LCOE is reduced marginally compared to the structural optimization-only case.

.. _fig_opt3_chord:
.. figure:: /images/blade/bladeopt3_chord.*
    :height: 4in
    :align: center

    Baseline versus optimized chord profiles

.. _fig_opt3_twist:
.. figure:: /images/blade/bladeopt3_twist.*
    :height: 4in
    :align: center

    Baseline versus optimized twist profiles
