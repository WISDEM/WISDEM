.. _user_opt_tutorial-label:

3b. User-Defined Design Variables, Constraints, and Objective
--------------------------------------------------------------

The prior example demonstrated that WISDEM has a menu of design variables, constraints, and objectives for the user.  The prior example focused on blade design, and other examples will cover other components or system-level design optimization.  Nevertheless, the WISDEM team recognizes that we cannot anticipate with perfect clarify all of the wind turbine design problems that users will pose.  Therefore, we have developed a pathway for users to list any input as a design variable and any output as a constraint or the objective function / merit figure.  To demonstrate this capability, we will continue with the prior blade design example.


User-Defined Design Variables
===============================

The file, ``analysis_options_user.yaml``, specifies user-defined design variables, constraints, and an objective function.  The design variable must be one of the Independent Variable Components of the model, in the WISDEM namespace, which is OpenMDAO language for an input variable name that can be independently adjusted by an outer optimization loop.  The input variables available as design variables are listed at :ref:`wisdem_inputs_documentation`.  In the ``analysis_options`` file, this appears in the design variable section as:

.. literalinclude:: /../examples/03_blade/analysis_options_user.yaml
    :language: yaml
    :start-after: fname_output: blade_out
    :end-before: # This will

A user-defined design variable is added by specifying the variable name, lower bound, upper bound, and reference order-of-magnitude (for better numerical conditioning).  The bounds and reference values are given with brackets to support vector variables where the bounds and scaling might change for each index.  For vector design variables, you can optionally provide an ``indices:`` entry that only actives select vector indexes to be design variables.

There is no limit to the number of design variables a user can add.  Additional entries are given with the ``-`` symbol in the yaml syntax, with the same indent level.  For instance,

.. code-block:: yaml

    user_defined:
        - name: control.rated_pitch
          lower_bound: [-5]
          upper_bound: [25]
          ref: [10.]
        - name: control.rated_TSR
          lower_bound: [7.0]
          upper_bound: [11.0]
          ref: [10.]


User-Defined Objective Function
================================

The same file, ``analysis_options_user.yaml``, provides a user-defined objective function as well.  A user-provided objective function in ``merit_figure_user`` always takes precedence over the traditional ``merit_figure:`` entry, so there is no conflict if both are given in the file.  An objective function must be an output from one of the many functions (OpenMDAO Components) that are used in WISDEM.  The list of available outputs is found at :ref:`wisdem_outputs_documentation`.  In this example we see the user is conducting an optimization to minimize the rated thrust of the turbine:

.. literalinclude:: /../examples/03_blade/analysis_options_user.yaml
    :language: yaml
    :start-after: # indices
    :end-before: constraints:

As with the user-defined design variables, a reference value must be provided for numerical conditioning.  By default, the optimization is posed as a minimization problem.  If the objective function should instead be maximized, then the ``max_flag`` should be set to ``True``.


User-Defined Constraints
============================

User-specified constraints are provided with similar syntax in the file:

.. literalinclude:: /../examples/03_blade/analysis_options_user.yaml
    :language: yaml
    :start-after: max_flag:
    :end-before: driver:

As with the user-defined merit function, the variable names must be a *WISDEM output*.  The constraint can either be one-sided or two-sided, meaning that a lower bound and/or an upper bound must be specified.

As with the design variable example above, the user can specify as many customized constraints as they like.
