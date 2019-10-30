.. _documentation-label:

.. currentmodule:: turbine_costsse.turbine_costsse

Documentation
-------------

Documentation for Turbine_CostsSE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following inputs and outputs are defined for Turbine_CostsSE:

.. literalinclude:: ../src/turbine_costsse/turbine_costsse.py
    :language: python
    :start-after: Turbine_CostsSE(Assembly)
    :end-before: def configure(self)
    :prepend: class Turbine_CostsSE(Assembly):


Referenced Sub-System Modules (Rotor)
=====================================
.. module:: turbine_costsse.turbine_costsse
.. class:: BladeCost
.. class:: HubCost
.. class:: PitchSystemCost
.. class:: SpinnerCost
.. class:: HubSystemCostAdder
.. class:: RotorCostAdder
.. class:: Rotor_CostsSE

Referenced Sub-System Modules (Nacelle)
=======================================
.. module:: turbine_costsse.turbine_costsse
.. class:: LowSpeedShaftCost
.. class:: BearingsCost
.. class:: GearboxCost
.. class:: HighSpeedSideCost
.. class:: GeneratorCost
.. class:: BedplateCost
.. class:: NacelleSystemCostAdder
.. class:: Nacelle_CostsSE

Referenced Sub-System Modules (Tower)
=====================================
.. module:: turbine_costsse.turbine_costsse
.. class:: TowerCost
.. class:: TowerCostAdder
.. class:: Tower_CostsSE

Referenced Turbine Cost Modules
===============================
.. module:: turbine_costsse.turbine_costsse
.. class:: Turbine_CostsSE
.. class:: TurbineCostAdder

Referenced PPI Index Models (via commonse.config)
=================================================
.. module:: commonse.csmPPI
.. class:: PPI


Documentation for NREL_CSM_TCC
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following inputs and outputs are defined for NREL_CSM_TCC:

.. literalinclude:: ../src/turbine_costsse/nrel_csm_tcc.py
    :language: python
    :start-after: tcc_csm_assembly(Assembly)
    :end-before: def configure(self)
    :prepend: class tcc_csm_assembly(Assembly):

Referenced Sub-System Modules (Blades)
======================================
.. module:: turbine_costsse.nrel_csm_tcc
.. class:: blades_csm_component

Referenced Sub-System Modules (Hub)
===================================
.. module:: turbine_costsse.nrel_csm_tcc
.. class:: hub_csm_component


Referenced Sub-System Modules (Nacelle)
=======================================
.. module:: turbine_costsse.nrel_csm_tcc
.. class:: nacelle_csm_component

Referenced Sub-System Modules (Tower)
=====================================
.. module:: turbine_costsse.nrel_csm_tcc
.. class:: tower_csm_component

Referenced Turbine Cost Modules
===============================
.. module:: turbine_costsse.nrel_csm_tcc
.. class:: tcc_csm_assembly
.. class:: tcc_csm_component
.. class:: rotor_mass_adder

Referenced PPI Index Models (via commonse.config)
=================================================
.. module:: commonse.csmPPI
.. class:: PPI


Documentation for Turbine_CostsSE_2015
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following inputs and outputs are defined for Turbine_CostsSE:

.. literalinclude:: ../src/turbine_costsse/turbine_costsse_2015.py
    :language: python
    :start-after: Turbine_CostsSE_2015(Assembly)
    :end-before: def configure(self)
    :prepend: class Turbine_CostsSE_2015(Assembly):


Referenced Sub-System Modules (Rotor)
=====================================
.. module:: turbine_costsse.turbine_costsse_2015
.. class:: BladeCost2015
.. class:: HubCost2015
.. class:: PitchSystemCost2015
.. class:: SpinnerCost2015
.. class:: HubSystemCostAdder2015
.. class:: RotorCostAdder2015
.. class:: Rotor_CostsSE_2015

Referenced Sub-System Modules (Nacelle)
=======================================
.. module:: turbine_costsse.turbine_costsse_2015
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
.. module:: turbine_costsse.turbine_costsse
.. class:: TowerCost2015
.. class:: TowerCostAdder2015
.. class:: Tower_CostsSE_2015

Referenced Turbine Cost Modules
===============================
.. module:: turbine_costsse.turbine_costsse
.. class:: Turbine_CostsSE_2015
.. class:: TurbineCostAdder2015



Documentation for NREL_CSM_TCC_2015
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following inputs and outputs are defined for NREL_CSM_TCC:

.. literalinclude:: ../src/turbine_costsse/nrel_csm_tcc_2015.py
    :language: python
    :start-after: nrel_csm_tcc_2015(Assembly)
    :end-before: def configure(self)
    :prepend: class nrel_csm_tcc_2015(Assembly):

Referenced Sub-System Modules (Rotor)
=====================================
.. module:: turbine_costsse.nrel_csm_tcc_2015
.. class:: BladeMass
.. class:: HubMass
.. class:: PitchSystemMass
.. class:: SpinnerMass

Referenced Sub-System Modules (Nacelle)
=======================================
.. module:: turbine_costsse.nrel_csm_tcc_2015
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
.. module:: turbine_costsse.nrel_csm_tcc_2015
.. class:: TowerMass

Referenced Turbine Cost Modules
===============================
.. module:: turbine_costsse.nrel_csm_tcc_2015
.. class:: turbine_mass_adder
.. class:: nrel_csm_mass_2015
.. class:: nrel_csm_tcc_2015