.. _analysis-options:

******************************
Analysis Options Inputs
******************************
The following inputs describe the options available in the ``analysis_options`` file.  The primary sections are:

- general
- design_variables
- constraints
- merit_figure
- merit_figure_user
- inverse_design
- driver
- recorder

Of these sections, the ``design_variables`` is the most complex.  The schema guide for all other sections is:

.. jsonschema:: analysis_schema.json
   :hide_key_if_empty: /**/default
   :hide_key: /**/design_variables


Design Variables Schema
========================

The schema guide for the design variables is:

.. jsonschema:: analysis_schema.json#/definitions/design_variables
   :hide_key_if_empty: /**/default
