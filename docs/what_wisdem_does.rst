.. _documentation-label:

.. currentmodule:: wisdem.lcoe.lcoe_csm_assembly

Documentation for WISDEM
---------------------------

Documentation for WISDEM using NREL CSM
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following inputs and outputs are defined for WISDEM:

.. literalinclude:: ../../../wisdem/assemblies/lcoe/lcoe_csm_assembly.py
    :language: python
    :start-after: lcoe_csm_assembly(Assembly)
    :end-before: def configure(self)
    :prepend: class lcoe_csm_assembly(Assembly):

Referenced Model
========================
.. module:: wisdem.lcoe.lcoe_csm_assembly
.. class:: lcoe_csm_assembly

Documentation for WISDEM using NREL CSM with ECN Offshore Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following inputs and outputs are defined for WISDEM:

.. literalinclude:: ../../../wisdem/assemblies/lcoe/lcoe_csm_ecn_assembly.py
    :language: python
    :start-after: lcoe_csm_ecn_assembly(Assembly)
    :end-before: def __init__(self, ssfile_1)
    :prepend: class lcoe_csm_ecn_assembly(Assembly):


Referenced Model
========================
.. module:: wisdem.lcoe.lcoe_csm_ecn_assembly
.. class:: lcoe_csm_ecn_assembly

Documentation for WISDEM using SE Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following inputs and outputs are defined for WISDEM using SE models:

.. literalinclude:: ../../../wisdem/assemblies/lcoe/lcoe_se_assembly.py
    :language: python
    :start-after: lcoe_se_assembly(Assembly)
    :end-before: __init__(self, with_new_nacelle=False, with_landbos=False, flexible_blade=False, with_3pt_drive=False, with_ecn_opex=False, ecn_file=None)
    :prepend: class lcoe_se_assembly(Assembly):

Referenced Model
========================
.. module:: wisdem.lcoe.lcoe_se_assembly
.. class:: lcoe_se_assembly
.. function:: configure_lcoe_with_turb_costs
.. function:: configure_lcoe_with_csm_bos
.. function:: configure_lcoe_with_landbos
.. function:: configure_lcoe_with_csm_opex
.. function:: configure_lcoe_with_ecn_opex
.. function:: configure_lcoe_with_basic_aep
.. function:: configure_lcoe_with_csm_fin


Documentation for WISDEM Turbine Assembly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. module:: wisdem.turbinese.turbine

.. _interfaces-label:


For this assembly the configure process is handled in a separate function.  This allows for additional options to be passed in (as noted in the linked documentation below), but more importantly allows it to be used in a larger assembly while still retaining a flat manner.  Creating multiple levels of nested assemblies is generally not advisable.  This is what is done when linking the cost models with the turbine models.  All of the configuring can be done in one flat assembly.

.. autosummary::
    :toctree: generated

    configure_turbine

For this case where there is only a turbine (and no cost models), the actual configure method in the assembly is very simple.  For the cost model additional configuration can be done after calling ``configure_turbine``.

.. literalinclude:: ../../../wisdem/assemblies/land_based/land_based_noGenerator_noBOS_lcoe.py
    :start-after: TurbineSE(Assembly)
    :end-before: if __name__ == '__main__':
    :prepend: class TurbineSE(Assembly):
