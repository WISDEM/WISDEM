.. _user_custom_tutorial-label:

11. User Customized Optimization Example
-----------------------------

WISDEM offers a long list of design variables, figures of merit, and constraints that users can call in their ``analysis_options.yaml``. The full list is specified in the `modeling_options.yaml <https://github.com/WISDEM/WISDEM/blob/develop/examples/02_reference_turbines/modeling_options.yaml>`. In addition, WISDEM now offers the option to build your own optimization problem by setting any available input as a design variable and any available output as either a constraint or a figure of merit. This example 11 shows how to build your customized ``analysis_options.yaml``.

In this example, we start from a 5MW land-based wind turbine that was developed within the Big Adaptive Rotor project (for more details refer to https://github.com/NREL/BAR_Designs) and we ask WISDEM to optimize the rated power of the turbine to minimize the levelized value of energy (LVOE) while keeping the turbine capital cost (TCC) within certain limits. Note that the focus of this example is on the capability of WISDEM, more than on the actual problem setup.

The focus of this example is on the file ``analysis_options_custom.yaml``. 

The field ``general`` lists the standard output folder and naming convention.

.. literalinclude:: ../../../examples/11_user_custom/analysis_options_custom.yaml
    :language: python
    :end-before: # design variables

The field ``design_variables`` shows how the user can define a list of design variables. Here the user will need to identify what input to select among the ones available in WISDEM. To do so, it might be useful to open an existing output file of WISDEM (if you do not have any, run any other example) and parse the file looking for your desired design variables. Note that for every entry, the user should set the desired values for lower and upper bounds. In addition, a reference value should be specified for quantities whose order of magnitude is not 1. Check the OpenMDAO user manual and tutorials if you are not familiar with this field. Lastly, for design variables made of arrays the user can also specify the indices of the array that contain the active design variables. Leave out this entry for scalars and for arrays where each element should be actively optimized.

.. literalinclude:: ../../../examples/11_user_custom/analysis_options_custom.yaml
    :language: python
    :start-after: # design variables
    :end-before: # figure of merit


The ``merit_figure`` consists of only one entry (not a list of entries!) and in this case we set LVOE as the metric to be minimize. If you want to maximize a metric, set the ``max_flag`` to True. Again, quantities far away from 1 should have the ``ref`` entry set. 

.. literalinclude:: ../../../examples/11_user_custom/analysis_options_custom.yaml
    :language: python
    :start-after: # figure of merit
    :end-before: # constraints

The field ``constraints`` lists the user-defined constraints. In this example we ask for the TCC to stay within 1,000 and 1,500 USD/kW.

.. literalinclude:: ../../../examples/11_user_custom/analysis_options_custom.yaml
    :language: python
    :start-after: # constraints
    :end-before: # driver

The yaml file is closed with some standard optimization options listed in the  ``driver`` section.

.. literalinclude:: ../../../examples/16_inverse_design/analysis_options_rotor.yaml
    :language: python
    :start-after: # driver

For more details about this example, please feel free to post questions on GitHub!
