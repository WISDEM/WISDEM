.. _documentation-label:

.. currentmodule:: wisdem.turbine_costsse.turbine_costsse

Documentation
-------------

Documentation for Turbine_CostsSE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following inputs and outputs are defined for Turbine_CostsSE:

Documentation for Turbine_CostsSE_2015
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following inputs and outputs are defined for Turbine_CostsSE:

.. literalinclude:: ../../../wisdem/turbine_costsse/turbine_costsse_2015.py
    :language: python
    :start-after: Turbine_CostsSE_2015(Assembly)
    :end-before: def configure(self)
    :prepend: class Turbine_CostsSE_2015(Assembly):


Referenced Sub-System Modules (Rotor)
=====================================
.. module:: wisdem.turbine_costsse.turbine_costsse_2015
.. class:: BladeCost2015
.. class:: HubCost2015
.. class:: PitchSystemCost2015
.. class:: SpinnerCost2015
.. class:: HubSystemCostAdder2015
.. class:: RotorCostAdder2015
.. class:: Rotor_CostsSE_2015

Referenced Sub-System Modules (Nacelle)
=======================================
.. module:: wisdem.turbine_costsse.turbine_costsse_2015
.. class:: LowSpeedShaftCost2015
.. class:: BearingsCost2015
.. class:: GearboxCost2015
.. class:: HighSpeedSideCost2015
.. class:: GeneratorCost2015
.. class:: BedplateCost2015
.. class:: YawSystemCost2015
.. class:: HydraulicCoolingCost2015
.. class:: VariableSpeedElecCost2015
.. class:: ElecConnecCost2015
.. class:: ControlsCost2015
.. class:: NacelleCoverCost2015
.. class:: OtherMainframeCost2015
.. class:: TransformerCost2015
.. class:: NacelleSystemCostAdder2015
.. class:: Nacelle_CostsSE_2015

Referenced Sub-System Modules (Tower)
=====================================
.. module:: wisdem.turbine_costsse.turbine_costsse
.. class:: TowerCost2015
.. class:: TowerCostAdder2015
.. class:: Tower_CostsSE_2015

Referenced Turbine Cost Modules
===============================
.. module:: wisdem.turbine_costsse.turbine_costsse
.. class:: Turbine_CostsSE_2015
.. class:: TurbineCostAdder2015



Documentation for NREL_CSM_TCC_2015
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following inputs and outputs are defined for NREL_CSM_TCC:

.. literalinclude:: ../../../wisdem/turbine_costsse/nrel_csm_tcc_2015.py
    :language: python
    :start-after: nrel_csm_tcc_2015(Assembly)
    :end-before: def configure(self)
    :prepend: class nrel_csm_tcc_2015(Assembly):

Referenced Sub-System Modules (Rotor)
=====================================
.. module:: wisdem.turbine_costsse.nrel_csm_tcc_2015
.. class:: BladeMass
.. class:: HubMass
.. class:: PitchSystemMass
.. class:: SpinnerMass

Referenced Sub-System Modules (Nacelle)
=======================================
.. module:: wisdem.turbine_costsse.nrel_csm_tcc_2015
.. class:: LowSpeedShaftMass
.. class:: BearingMass
.. class:: GearboxMass
.. class:: HighSpeedSideMass
.. class:: GeneratorMass
.. class:: BedplateMass
.. class:: YawSystemMass
.. class:: HydraulicCoolingMass
.. class:: NacelleCoverMass
.. class:: OtherMainframeMass
.. class:: TransformerMass

Referenced Sub-System Modules (Tower)
=====================================
.. module:: wisdem.turbine_costsse.nrel_csm_tcc_2015
.. class:: TowerMass

Referenced Turbine Cost Modules
===============================
.. module:: wisdem.turbine_costsse.nrel_csm_tcc_2015
.. class:: turbine_mass_adder
.. class:: nrel_csm_mass_2015
.. class:: nrel_csm_tcc_2015
