"""
Copyright (c) NREL. All rights reserved.
"""

import openmdao.api as om

import numpy as np


###### Rotor
# -------------------------------------------------------------------------------
class BladeCost2015(om.ExplicitComponent):
    """
    Compute blade cost in the form of :math:`cost = k*mass`.
    Value of :math:`k` was updated in 2015 to be $14.6 USD/kg.
    Cost includes materials and manufacturing costs.
    Cost can be overridden with use of `blade_cost_external`

    Parameters
    ----------
    blade_mass : float, [kg]
        component mass
    blade_mass_cost_coeff : float, [USD/kg]
        blade mass-cost coeff
    blade_cost_external : float, [USD]
        Blade cost computed by RotorSE

    Returns
    -------
    blade_cost : float, [USD]
        Blade cost

    """

    def setup(self):
        self.add_input("blade_mass", 0.0, units="kg")
        self.add_input("blade_mass_cost_coeff", 14.6, units="USD/kg")
        self.add_input("blade_cost_external", 0.0, units="USD")

        self.add_output("blade_cost", 0.0, units="USD")

    def compute(self, inputs, outputs):
        blade_mass = inputs["blade_mass"]
        blade_mass_cost_coeff = inputs["blade_mass_cost_coeff"]

        # calculate component cost
        if inputs["blade_cost_external"] < 1.0 or np.isnan(inputs["blade_cost_external"]):
            outputs["blade_cost"] = blade_mass_cost_coeff * blade_mass
        else:
            outputs["blade_cost"] = inputs["blade_cost_external"]


# -----------------------------------------------------------------------------------------------
class HubCost2015(om.ExplicitComponent):
    """
    Compute hub cost in the form of :math:`cost = k*mass`.
    Value of :math:`k` was updated in 2015 to be $3.9 USD/kg.
    Cost includes materials and manufacturing costs.

    Parameters
    ----------
    hub_mass : float, [kg]
        component mass
    hub_mass_cost_coeff : float, [USD/kg]
        hub mass-cost coeff

    Returns
    -------
    hub_cost : float, [USD]
        Hub cost

    """

    def setup(self):
        self.add_input("hub_mass", 0.0, units="kg")
        self.add_input("hub_mass_cost_coeff", 3.9, units="USD/kg")

        self.add_output("hub_cost", 0.0, units="USD")

    def compute(self, inputs, outputs):
        hub_mass_cost_coeff = inputs["hub_mass_cost_coeff"]
        hub_mass = inputs["hub_mass"]

        # calculate component cost
        outputs["hub_cost"] = hub_mass_cost_coeff * hub_mass


# -------------------------------------------------------------------------------
class PitchSystemCost2015(om.ExplicitComponent):
    """
    Compute pitch system cost in the form of :math:`cost = k*mass`.
    Value of :math:`k` was NOT updated in 2015 and remains the same as original CSM, $22.1 USD/kg.
    Cost includes materials and manufacturing costs.

    Parameters
    ----------
    pitch_system_mass : float, [kg]
        component mass
    pitch_system_mass_cost_coeff : float, [USD/kg]
        pitch system mass-cost coeff

    Returns
    -------
    pitch_system_cost : float, [USD]
        Pitch system cost

    """

    def setup(self):
        self.add_input("pitch_system_mass", 0.0, units="kg")
        self.add_input("pitch_system_mass_cost_coeff", 22.1, units="USD/kg")

        self.add_output("pitch_system_cost", 0.0, units="USD")

    def compute(self, inputs, outputs):
        pitch_system_mass = inputs["pitch_system_mass"]
        pitch_system_mass_cost_coeff = inputs["pitch_system_mass_cost_coeff"]

        # calculate system costs
        outputs["pitch_system_cost"] = pitch_system_mass_cost_coeff * pitch_system_mass


# -------------------------------------------------------------------------------
class SpinnerCost2015(om.ExplicitComponent):
    """
    Compute spinner cost in the form of :math:`cost = k*mass`.
    Value of :math:`k` was updated in 2015 to be $11.1 USD/kg.
    Cost includes materials and manufacturing costs.

    Parameters
    ----------
    spinner_mass : float, [kg]
        component mass
    spinner_mass_cost_coeff : float, [USD/kg]
        spinner/nose cone mass-cost coeff

    Returns
    -------
    spinner_cost : float, [USD]
        Spinner cost

    """

    def setup(self):
        self.add_input("spinner_mass", 0.0, units="kg")
        self.add_input("spinner_mass_cost_coeff", 11.1, units="USD/kg")

        self.add_output("spinner_cost", 0.0, units="USD")

    def compute(self, inputs, outputs):
        spinner_mass_cost_coeff = inputs["spinner_mass_cost_coeff"]
        spinner_mass = inputs["spinner_mass"]

        # calculate system costs
        outputs["spinner_cost"] = spinner_mass_cost_coeff * spinner_mass


# -------------------------------------------------------------------------------
class HubSystemCostAdder2015(om.ExplicitComponent):
    """
    Aggregates the hub, pitch system, and spinner costs into a single component
    that is transported to the project site and could therefore incur additional
    costs.  Cost multipliers are of the form,
    :math:`c_{hubsys} = (1+kt_{hubsys}+kp_{hubsys}) (1+ko_{hubsys}+ka_{hubsys})
    (c_{hub} + c_{pitch} + c_{spinner})`

    Where conceptually, :math:`kt` is a transportation multiplier,
    :math:`kp` is a profit multiplier,
    :math:`ko` is an overhead cost multiplier, and
    :math:`ka` is an assembly cost multiplier

    By default, :math:`kt=kp=ko=ka=0`.

    Parameters
    ----------
    hub_cost : float, [USD]
        Hub component cost
    hub_mass : float, [kg]
        Hub component mass
    pitch_system_cost : float, [USD]
        Pitch system cost
    pitch_system_mass : float, [kg]
        Pitch system mass
    spinner_cost : float, [USD]
        Spinner component cost
    spinner_mass : float, [kg]
        Spinner component mass
    hub_assemblyCostMultiplier : float
        Rotor assembly cost multiplier
    hub_overheadCostMultiplier : float
        Rotor overhead cost multiplier
    hub_profitMultiplier : float
        Rotor profit multiplier
    hub_transportMultiplier : float
        Rotor transport multiplier

    Returns
    -------
    hub_system_mass_tcc : float, [kg]
        Mass of the hub system, including hub, spinner, and pitch system for the blades
    hub_system_cost : float, [USD]
        Overall wind sub-assembly capial costs including transportation costs

    """

    def setup(self):
        self.add_input("hub_cost", 0.0, units="USD")
        self.add_input("hub_mass", 0.0, units="kg")
        self.add_input("pitch_system_cost", 0.0, units="USD")
        self.add_input("pitch_system_mass", 0.0, units="kg")
        self.add_input("spinner_cost", 0.0, units="USD")
        self.add_input("spinner_mass", 0.0, units="kg")
        self.add_input("hub_assemblyCostMultiplier", 0.0)
        self.add_input("hub_overheadCostMultiplier", 0.0)
        self.add_input("hub_profitMultiplier", 0.0)
        self.add_input("hub_transportMultiplier", 0.0)

        self.add_output("hub_system_mass_tcc", 0.0, units="kg")
        self.add_output("hub_system_cost", 0.0, units="USD")

    def compute(self, inputs, outputs):
        hub_cost = inputs["hub_cost"]
        pitch_system_cost = inputs["pitch_system_cost"]
        spinner_cost = inputs["spinner_cost"]

        hub_mass = inputs["hub_mass"]
        pitch_system_mass = inputs["pitch_system_mass"]
        spinner_mass = inputs["spinner_mass"]

        hub_assemblyCostMultiplier = inputs["hub_assemblyCostMultiplier"]
        hub_overheadCostMultiplier = inputs["hub_overheadCostMultiplier"]
        hub_profitMultiplier = inputs["hub_profitMultiplier"]
        hub_transportMultiplier = inputs["hub_transportMultiplier"]

        # Updated calculations below to account for assembly, transport, overhead and profit
        outputs["hub_system_mass_tcc"] = hub_mass + pitch_system_mass + spinner_mass
        partsCost = hub_cost + pitch_system_cost + spinner_cost
        outputs["hub_system_cost"] = (1 + hub_transportMultiplier + hub_profitMultiplier) * (
            (1 + hub_overheadCostMultiplier + hub_assemblyCostMultiplier) * partsCost
        )


# -------------------------------------------------------------------------------
class RotorCostAdder2015(om.ExplicitComponent):
    """
    Sum of individual component costs to get overall rotor cost.
    No additional transport and assembly multipliers are included because it is
    assumed that each component is transported separately.

    Parameters
    ----------
    blade_cost : float, [USD]
        Individual blade cost
    blade_mass : float, [kg]
        Individual blade mass
    hub_system_cost : float, [USD]
        Cost for hub system
    hub_system_mass_tcc : float, [kg]
        Mass for hub system
    blade_number : int
        Number of rotor blades

    Returns
    -------
    rotor_cost : float, [USD]
        Rotor cost
    rotor_mass_tcc : float, [kg]
        Rotor mass, including blades, pitch system, hub, and spinner

    """

    def setup(self):
        self.add_input("blade_cost", 0.0, units="USD")
        self.add_input("blade_mass", 0.0, units="kg")
        self.add_input("hub_system_cost", 0.0, units="USD")
        self.add_input("hub_system_mass_tcc", 0.0, units="kg")
        self.add_discrete_input("blade_number", 3)

        self.add_output("rotor_cost", 0.0, units="USD")
        self.add_output("rotor_mass_tcc", 0.0, units="kg")

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        blade_cost = inputs["blade_cost"]
        blade_mass = inputs["blade_mass"]
        blade_number = discrete_inputs["blade_number"]
        hub_system_cost = inputs["hub_system_cost"]
        hub_system_mass = inputs["hub_system_mass_tcc"]

        outputs["rotor_cost"] = blade_cost * blade_number + hub_system_cost
        outputs["rotor_mass_tcc"] = blade_mass * blade_number + hub_system_mass


# -------------------------------------------------------------------------------


###### Nacelle
# -------------------------------------------------
class LowSpeedShaftCost2015(om.ExplicitComponent):
    """
    Compute low speed shaft cost in the form of :math:`cost = k*mass`.
    Value of :math:`k` was updated in 2015 to be $11.9 USD/kg.
    Cost includes materials and manufacturing costs.

    Parameters
    ----------
    lss_mass : float, [kg]
        component mass
    lss_mass_cost_coeff : float, [USD/kg]
        low speed shaft mass-cost coeff

    Returns
    -------
    lss_cost : float, [USD]
        Low speed shaft cost

    """

    def setup(self):
        self.add_input("lss_mass", 0.0, units="kg")  # mass input
        self.add_input("lss_mass_cost_coeff", 11.9, units="USD/kg")

        self.add_output("lss_cost", 0.0, units="USD")

    def compute(self, inputs, outputs):
        lss_mass_cost_coeff = inputs["lss_mass_cost_coeff"]
        lss_mass = inputs["lss_mass"]
        outputs["lss_cost"] = lss_mass_cost_coeff * lss_mass


# -------------------------------------------------------------------------------
class BearingCost2015(om.ExplicitComponent):
    """
    Compute (single) main bearing cost in the form of :math:`cost = k*mass`.
    Value of :math:`k` was updated in 2015 to be $4.5 USD/kg.
    Cost includes materials and manufacturing costs.

    Parameters
    ----------
    main_bearing_mass : float, [kg]
        component mass
    bearing_mass_cost_coeff : float, [USD/kg]
        main bearing mass-cost coeff

    Returns
    -------
    main_bearing_cost : float, [USD]
        Main bearing cost

    """

    def setup(self):
        self.add_input("main_bearing_mass", 0.0, units="kg")  # mass input
        self.add_input("bearing_mass_cost_coeff", 4.5, units="USD/kg")

        self.add_output("main_bearing_cost", 0.0, units="USD")

    def compute(self, inputs, outputs):
        main_bearing_mass = inputs["main_bearing_mass"]
        bearing_mass_cost_coeff = inputs["bearing_mass_cost_coeff"]

        outputs["main_bearing_cost"] = bearing_mass_cost_coeff * main_bearing_mass


# -------------------------------------------------------------------------------
class GearboxCost2015(om.ExplicitComponent):
    """
    Compute gearbox cost in the form of :math:`cost = k*mass`.
    Value of :math:`k` was updated in 2015 to be $12.9 USD/kg.
    Cost includes materials and manufacturing costs.

    Parameters
    ----------
    gearbox_mass : float, [kg]
        component mass
    gearbox_mass_cost_coeff : float, [USD/kg]
        gearbox mass-cost coeff

    Returns
    -------
    gearbox_cost : float, [USD]
        Gearbox cost

    """

    def setup(self):
        self.add_input("gearbox_mass", 0.0, units="kg")
        self.add_input("gearbox_mass_cost_coeff", 12.9, units="USD/kg")

        self.add_output("gearbox_cost", 0.0, units="USD")

    def compute(self, inputs, outputs):
        gearbox_mass = inputs["gearbox_mass"]
        gearbox_mass_cost_coeff = inputs["gearbox_mass_cost_coeff"]

        outputs["gearbox_cost"] = gearbox_mass_cost_coeff * gearbox_mass


# -------------------------------------------------------------------------------
class BrakeCost2020(om.ExplicitComponent):
    """
    Compute brake cost in the form of :math:`cost = k*mass`.
    Value of :math:`k` was updated in 2020 to be $3.6254 USD/kg.
    Cost includes materials and manufacturing costs.

    Parameters
    ----------
    brake_mass : float, [kg]
        component mass
    brake_mass_cost_coeff : float, [USD/kg]
        brake mass-cost coeff

    Returns
    -------
    brake_cost : float, [USD]
        Brake cost

    """

    def setup(self):
        self.add_input("brake_mass", 0.0, units="kg")
        self.add_input("brake_mass_cost_coeff", 3.6254, units="USD/kg")

        self.add_output("brake_cost", 0.0, units="USD")

    def compute(self, inputs, outputs):
        brake_mass = inputs["brake_mass"]
        brake_mass_cost_coeff = inputs["brake_mass_cost_coeff"]

        outputs["brake_cost"] = brake_mass_cost_coeff * brake_mass


# -------------------------------------------------------------------------------
class HighSpeedShaftCost2015(om.ExplicitComponent):
    """
    Compute high speed shaft cost in the form of :math:`cost = k*mass`.
    Value of :math:`k` was updated in 2015 to be $6.8 USD/kg.
    Cost includes materials and manufacturing costs.

    Parameters
    ----------
    hss_mass : float, [kg]
        component mass
    hss_mass_cost_coeff : float, [USD/kg]
        high speed shaft mass-cost coeff

    Returns
    -------
    hss_cost : float, [USD]
        High speed shaft cost

    """

    def setup(self):
        self.add_input("hss_mass", 0.0, units="kg")
        self.add_input("hss_mass_cost_coeff", 6.8, units="USD/kg")

        self.add_output("hss_cost", 0.0, units="USD")

    def compute(self, inputs, outputs):
        hss_mass = inputs["hss_mass"]
        hss_mass_cost_coeff = inputs["hss_mass_cost_coeff"]

        outputs["hss_cost"] = hss_mass_cost_coeff * hss_mass


# -------------------------------------------------------------------------------
class GeneratorCost2015(om.ExplicitComponent):
    """
    Compute generator cost in the form of :math:`cost = k*mass`.
    Value of :math:`k` was updated in 2015 to be $12.4 USD/kg.
    Cost includes materials and manufacturing costs.
    Cost can be overridden with use of `generator_cost_external`

    Parameters
    ----------
    generator_mass : float, [kg]
        component mass
    generator_mass_cost_coeff : float, [USD/kg]
        generator mass cost coeff
    generator_cost_external : float, [USD]
        Generator cost computed by GeneratorSE

    Returns
    -------
    generator_cost : float, [USD]
        Generator cost

    """

    def setup(self):
        self.add_input("generator_mass", 0.0, units="kg")
        self.add_input("generator_mass_cost_coeff", 12.4, units="USD/kg")
        self.add_input("generator_cost_external", 0.0, units="USD")

        self.add_output("generator_cost", 0.0, units="USD")

    def compute(self, inputs, outputs):
        generator_mass = inputs["generator_mass"]
        generator_mass_cost_coeff = inputs["generator_mass_cost_coeff"]

        if inputs["generator_cost_external"] < 1.0:
            outputs["generator_cost"] = generator_mass_cost_coeff * generator_mass
        else:
            outputs["generator_cost"] = inputs["generator_cost_external"]


# -------------------------------------------------------------------------------
class BedplateCost2015(om.ExplicitComponent):
    """
    Compute bedplate cost in the form of :math:`cost = k*mass`.
    Value of :math:`k` was updated in 2015 to be $2.9 USD/kg.
    Cost includes materials and manufacturing costs.

    Parameters
    ----------
    bedplate_mass : float, [kg]
        component mass
    bedplate_mass_cost_coeff : float, [USD/kg]
        bedplate mass-cost coeff

    Returns
    -------
    bedplate_cost : float, [USD]
        Bedplate cost

    """

    def setup(self):
        self.add_input("bedplate_mass", 0.0, units="kg")
        self.add_input("bedplate_mass_cost_coeff", 2.9, units="USD/kg")

        self.add_output("bedplate_cost", 0.0, units="USD")

    def compute(self, inputs, outputs):
        bedplate_mass = inputs["bedplate_mass"]
        bedplate_mass_cost_coeff = inputs["bedplate_mass_cost_coeff"]

        outputs["bedplate_cost"] = bedplate_mass_cost_coeff * bedplate_mass


# ---------------------------------------------------------------------------------
class YawSystemCost2015(om.ExplicitComponent):
    """
    Compute yaw system cost in the form of :math:`cost = k*mass`.
    Value of :math:`k` was NOT updated in 2015 and remains the same as original CSM, $8.3 USD/kg.
    Cost includes materials and manufacturing costs.

    Parameters
    ----------
    yaw_mass : float, [kg]
        component mass
    yaw_mass_cost_coeff : float, [USD/kg]
        yaw system mass cost coeff

    Returns
    -------
    yaw_system_cost : float, [USD]
        Yaw system cost

    """

    def setup(self):
        self.add_input("yaw_mass", 0.0, units="kg")
        self.add_input("yaw_mass_cost_coeff", 8.3, units="USD/kg")

        self.add_output("yaw_system_cost", 0.0, units="USD")

    def compute(self, inputs, outputs):
        yaw_mass = inputs["yaw_mass"]
        yaw_mass_cost_coeff = inputs["yaw_mass_cost_coeff"]

        outputs["yaw_system_cost"] = yaw_mass_cost_coeff * yaw_mass


# ---------------------------------------------------------------------------------
class ConverterCost2015(om.ExplicitComponent):
    """
    Compute converter cost in the form of :math:`cost = k*mass`.
    Value of :math:`k` was updated in 2015 to be $18.8 USD/kg.
    Cost includes materials and manufacturing costs.

    Parameters
    ----------
    converter_mass : float, [kg]
        component mass
    converter_mass_cost_coeff : float, [USD/kg]
        variable speed electronics mass cost coeff

    Returns
    -------
    converter_cost : float, [USD]
        Converter cost

    """

    def setup(self):
        self.add_input("converter_mass", 0.0, units="kg")
        self.add_input("converter_mass_cost_coeff", 18.8, units="USD/kg")

        self.add_output("converter_cost", 0.0, units="USD")

    def compute(self, inputs, outputs):
        converter_mass = inputs["converter_mass"]
        converter_mass_cost_coeff = inputs["converter_mass_cost_coeff"]

        outputs["converter_cost"] = converter_mass_cost_coeff * converter_mass


# ---------------------------------------------------------------------------------
class HydraulicCoolingCost2015(om.ExplicitComponent):
    """
    Compute hydraulic cooling cost in the form of :math:`cost = k*mass`.
    Value of :math:`k` was NOT updated in 2015 and remains the same as original CSM, $124.0 USD/kg.
    Cost includes materials and manufacturing costs.

    Parameters
    ----------
    hvac_mass : float, [kg]
        component mass
    hvac_mass_cost_coeff : float, [USD/kg]
        hydraulic and cooling system mass cost coeff

    Returns
    -------
    hvac_cost : float, [USD]
        HVAC cost

    """

    def setup(self):
        self.add_input("hvac_mass", 0.0, units="kg")
        self.add_input("hvac_mass_cost_coeff", 124.0, units="USD/kg")

        self.add_output("hvac_cost", 0.0, units="USD")

    def compute(self, inputs, outputs):
        hvac_mass = inputs["hvac_mass"]
        hvac_mass_cost_coeff = inputs["hvac_mass_cost_coeff"]

        # calculate cost
        outputs["hvac_cost"] = hvac_mass_cost_coeff * hvac_mass


# ---------------------------------------------------------------------------------
class NacelleCoverCost2015(om.ExplicitComponent):
    """
    Compute nacelle cover cost in the form of :math:`cost = k*mass`.
    Value of :math:`k` was NOT updated in 2015 and remains the same as original CSM, $5.7 USD/kg.
    Cost includes materials and manufacturing costs.

    Parameters
    ----------
    cover_mass : float, [kg]
        component mass
    cover_mass_cost_coeff : float, [USD/kg]
        nacelle cover mass cost coeff

    Returns
    -------
    cover_cost : float, [USD]
        Cover cost

    """

    def setup(self):
        self.add_input("cover_mass", 0.0, units="kg")
        self.add_input("cover_mass_cost_coeff", 5.7, units="USD/kg")

        self.add_output("cover_cost", 0.0, units="USD")

    def compute(self, inputs, outputs):
        cover_mass = inputs["cover_mass"]
        cover_mass_cost_coeff = inputs["cover_mass_cost_coeff"]

        outputs["cover_cost"] = cover_mass_cost_coeff * cover_mass


# ---------------------------------------------------------------------------------
class ElecConnecCost2015(om.ExplicitComponent):
    """
    Compute electrical connection cost in the form of :math:`cost = k*mass`.
    Value of :math:`k` was NOT updated in 2015 and remains the same as original CSM, $41.85 USD/kg.
    Cost includes materials and manufacturing costs.

    Parameters
    ----------
    machine_rating : float
        machine rating
    elec_connec_machine_rating_cost_coeff : float, [USD/kg]
        electrical connections cost coefficient per kW

    Returns
    -------
    elec_cost : float, [USD]
        Electrical connection costs

    """

    def setup(self):
        self.add_input("machine_rating", 0.0, units="kW")
        self.add_input("elec_connec_machine_rating_cost_coeff", 41.85, units="USD/kW")

        self.add_output("elec_cost", 0.0, units="USD")

    def compute(self, inputs, outputs):
        machine_rating = inputs["machine_rating"]
        elec_connec_machine_rating_cost_coeff = inputs["elec_connec_machine_rating_cost_coeff"]

        outputs["elec_cost"] = elec_connec_machine_rating_cost_coeff * machine_rating


# ---------------------------------------------------------------------------------
class ControlsCost2015(om.ExplicitComponent):
    """
    Compute controls cost in the form of :math:`cost = k*mass`.
    Value of :math:`k` was NOT updated in 2015 and remains the same as original CSM, $21.15 USD/kg.
    Cost includes materials and manufacturing costs.

    Parameters
    ----------
    machine_rating : float
        machine rating
    controls_machine_rating_cost_coeff : float, [USD/kg]
        controls cost coefficient per kW

    Returns
    -------
    controls_cost : float, [USD]
        Controls cost

    """

    def setup(self):
        self.add_input("machine_rating", 0.0, units="kW")
        self.add_input("controls_machine_rating_cost_coeff", 21.15, units="USD/kW")

        self.add_output("controls_cost", 0.0, units="USD")

    def compute(self, inputs, outputs):
        machine_rating = inputs["machine_rating"]
        coeff = inputs["controls_machine_rating_cost_coeff"]

        outputs["controls_cost"] = machine_rating * coeff


# ---------------------------------------------------------------------------------
class PlatformsMainframeCost2015(om.ExplicitComponent):
    """
    Compute platforms cost in the form of :math:`cost = k*mass`.
    Value of :math:`k` was NOT updated in 2015 and remains the same as original CSM, $17.1 USD/kg.
    Cost includes materials and manufacturing costs.

    Parameters
    ----------
    platforms_mass : float, [kg]
        component mass
    platforms_mass_cost_coeff : float, [USD/kg]
        nacelle platforms mass cost coeff
    crane : boolean
        flag for presence of onboard crane
    crane_cost : float, [USD]
        crane cost if present

    Returns
    -------
    platforms_cost : float, [USD]
        Platforms cost

    """

    def setup(self):
        self.add_input("platforms_mass", 0.0, units="kg")
        self.add_input("platforms_mass_cost_coeff", 17.1, units="USD/kg")
        self.add_discrete_input("crane", False)
        self.add_input("crane_cost", 12000.0, units="USD")

        self.add_output("platforms_cost", 0.0, units="USD")

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        platforms_mass = inputs["platforms_mass"]
        platforms_mass_cost_coeff = inputs["platforms_mass_cost_coeff"]
        crane = discrete_inputs["crane"]
        crane_cost = inputs["crane_cost"]
        # bedplate_cost = inputs['bedplate_cost']
        # base_hardware_cost_coeff = inputs['base_hardware_cost_coeff']

        # nacelle platform cost

        # crane cost
        if crane:
            craneCost = crane_cost
            craneMass = 3e3
            NacellePlatformsCost = platforms_mass_cost_coeff * (platforms_mass - craneMass)
        else:
            craneCost = 0.0
            NacellePlatformsCost = platforms_mass_cost_coeff * platforms_mass

        # base hardware cost
        # BaseHardwareCost = bedplate_cost * base_hardware_cost_coeff

        # aggregate all three mainframe costs
        outputs["platforms_cost"] = NacellePlatformsCost + craneCost  # + BaseHardwareCost


# -------------------------------------------------------------------------------
class TransformerCost2015(om.ExplicitComponent):
    """
    Compute transformer cost in the form of :math:`cost = k*mass`.
    Value of :math:`k` was updated in 2015 to be $18.8 USD/kg.
    Cost includes materials and manufacturing costs.

    Parameters
    ----------
    transformer_mass : float, [kg]
        component mass
    transformer_mass_cost_coeff : float, [USD/kg]
        transformer mass cost coeff

    Returns
    -------
    transformer_cost : float, [USD]
        Transformer cost

    """

    def setup(self):
        self.add_input("transformer_mass", 0.0, units="kg")
        self.add_input("transformer_mass_cost_coeff", 18.8, units="USD/kg")  # mass-cost coeff with default from ppt

        self.add_output("transformer_cost", 0.0, units="USD")

    def compute(self, inputs, outputs):
        transformer_mass = inputs["transformer_mass"]
        transformer_mass_cost_coeff = inputs["transformer_mass_cost_coeff"]

        outputs["transformer_cost"] = transformer_mass_cost_coeff * transformer_mass


# -------------------------------------------------------------------------------
class NacelleSystemCostAdder2015(om.ExplicitComponent):
    """
    Aggregates the nacelle system costs into a single component
    that is transported to the project site and could therefore incur additional
    costs.  Cost multipliers are of the form,
    :math:`c_{nacellesys} = (1+kt_{nacelle}+kp_{nacelle}) (1+ko_{nacelle}+ka_{nacelle}) c_{nacelle}`

    Where conceptually, :math:`kt` is a transportation multiplier,
    :math:`kp` is a profit multiplier,
    :math:`ko` is an overhead cost multiplier, and
    :math:`ka` is an assembly cost multiplier

    By default, :math:`kt=kp=ko=ka=0`.

    Parameters
    ----------
    lss_cost : float, [USD]
        Component cost
    lss_mass : float, [kg]
        Component mass
    main_bearing_cost : float, [USD]
        Component cost
    main_bearing_mass : float, [kg]
        Component mass
    gearbox_cost : float, [USD]
        Component cost
    gearbox_mass : float, [kg]
        Component mass
    brake_cost : float, [USD]
        Component cost
    brake_mass : float, [kg]
        Component mass
    hss_cost : float, [USD]
        Component cost
    hss_mass : float, [kg]
        Component mass
    generator_cost : float, [USD]
        Component cost
    generator_mass : float, [kg]
        Component mass
    bedplate_cost : float, [USD]
        Component cost
    bedplate_mass : float, [kg]
        Component mass
    yaw_system_cost : float, [USD]
        Component cost
    yaw_mass : float, [kg]
        Component mass
    converter_cost : float, [USD]
        Component cost
    converter_mass : float, [kg]
        Component mass
    hvac_cost : float, [USD]
        Component cost
    hvac_mass : float, [kg]
        Component mass
    cover_cost : float, [USD]
        Component cost
    cover_mass : float, [kg]
        Component mass
    elec_cost : float, [USD]
        Component cost
    controls_cost : float, [USD]
        Component cost
    platforms_mass : float, [kg]
        Component cost
    platforms_cost : float, [USD]
        Component cost
    transformer_cost : float, [USD]
        Component cost
    transformer_mass : float, [kg]
        Component mass
    main_bearing_number : int
        number of bearings
    nacelle_assemblyCostMultiplier : float
        nacelle assembly cost multiplier
    nacelle_overheadCostMultiplier : float
        nacelle overhead cost multiplier
    nacelle_profitMultiplier : float
        nacelle profit multiplier
    nacelle_transportMultiplier : float
        nacelle transport multiplier

    Returns
    -------
    nacelle_cost : float, [USD]
        component cost
    nacelle_mass_tcc : float
        Nacelle mass, with all nacelle components, without the rotor

    """

    def setup(self):
        self.add_input("lss_cost", 0.0, units="USD")
        self.add_input("lss_mass", 0.0, units="kg")
        self.add_input("main_bearing_cost", 0.0, units="USD")
        self.add_input("main_bearing_mass", 0.0, units="kg")
        self.add_input("gearbox_cost", 0.0, units="USD")
        self.add_input("gearbox_mass", 0.0, units="kg")
        self.add_input("hss_cost", 0.0, units="USD")
        self.add_input("hss_mass", 0.0, units="kg")
        self.add_input("brake_cost", 0.0, units="USD")
        self.add_input("brake_mass", 0.0, units="kg")
        self.add_input("generator_cost", 0.0, units="USD")
        self.add_input("generator_mass", 0.0, units="kg")
        self.add_input("bedplate_cost", 0.0, units="USD")
        self.add_input("bedplate_mass", 0.0, units="kg")
        self.add_input("yaw_system_cost", 0.0, units="USD")
        self.add_input("yaw_mass", 0.0, units="kg")
        self.add_input("converter_cost", 0.0, units="USD")
        self.add_input("converter_mass", 0.0, units="kg")
        self.add_input("hvac_cost", 0.0, units="USD")
        self.add_input("hvac_mass", 0.0, units="kg")
        self.add_input("cover_cost", 0.0, units="USD")
        self.add_input("cover_mass", 0.0, units="kg")
        self.add_input("elec_cost", 0.0, units="USD")
        self.add_input("controls_cost", 0.0, units="USD")
        self.add_input("platforms_mass", 0.0, units="kg")
        self.add_input("platforms_cost", 0.0, units="USD")
        self.add_input("transformer_cost", 0.0, units="USD")
        self.add_input("transformer_mass", 0.0, units="kg")
        self.add_discrete_input("main_bearing_number", 2)
        # multipliers
        self.add_input("nacelle_assemblyCostMultiplier", 0.0)
        self.add_input("nacelle_overheadCostMultiplier", 0.0)
        self.add_input("nacelle_profitMultiplier", 0.0)
        self.add_input("nacelle_transportMultiplier", 0.0)

        self.add_output("nacelle_cost", 0.0, units="USD")
        self.add_output("nacelle_mass_tcc", 0.0, units="kg")

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        lss_cost = inputs["lss_cost"]
        main_bearing_cost = inputs["main_bearing_cost"]
        gearbox_cost = inputs["gearbox_cost"]
        hss_cost = inputs["hss_cost"]
        brake_cost = inputs["brake_cost"]
        generator_cost = inputs["generator_cost"]
        bedplate_cost = inputs["bedplate_cost"]
        yaw_system_cost = inputs["yaw_system_cost"]
        converter_cost = inputs["converter_cost"]
        hvac_cost = inputs["hvac_cost"]
        cover_cost = inputs["cover_cost"]
        elec_cost = inputs["elec_cost"]
        controls_cost = inputs["controls_cost"]
        platforms_cost = inputs["platforms_cost"]
        transformer_cost = inputs["transformer_cost"]

        lss_mass = inputs["lss_mass"]
        main_bearing_mass = inputs["main_bearing_mass"]
        gearbox_mass = inputs["gearbox_mass"]
        hss_mass = inputs["hss_mass"]
        brake_mass = inputs["brake_mass"]
        generator_mass = inputs["generator_mass"]
        bedplate_mass = inputs["bedplate_mass"]
        yaw_mass = inputs["yaw_mass"]
        converter_mass = inputs["converter_mass"]
        hvac_mass = inputs["hvac_mass"]
        cover_mass = inputs["cover_mass"]
        platforms_mass = inputs["platforms_mass"]
        transformer_mass = inputs["transformer_mass"]

        main_bearing_number = discrete_inputs["main_bearing_number"]

        nacelle_assemblyCostMultiplier = inputs["nacelle_assemblyCostMultiplier"]
        nacelle_overheadCostMultiplier = inputs["nacelle_overheadCostMultiplier"]
        nacelle_profitMultiplier = inputs["nacelle_profitMultiplier"]
        nacelle_transportMultiplier = inputs["nacelle_transportMultiplier"]

        # apply multipliers for assembly, transport, overhead, and profits
        outputs["nacelle_mass_tcc"] = (
            lss_mass
            + main_bearing_number * main_bearing_mass
            + gearbox_mass
            + hss_mass
            + brake_mass
            + generator_mass
            + bedplate_mass
            + yaw_mass
            + converter_mass
            + hvac_mass
            + cover_mass
            + platforms_mass
            + transformer_mass
        )
        partsCost = (
            lss_cost
            + main_bearing_number * main_bearing_cost
            + gearbox_cost
            + hss_cost
            + brake_cost
            + generator_cost
            + bedplate_cost
            + yaw_system_cost
            + converter_cost
            + hvac_cost
            + cover_cost
            + elec_cost
            + controls_cost
            + platforms_cost
            + transformer_cost
        )
        outputs["nacelle_cost"] = (1 + nacelle_transportMultiplier + nacelle_profitMultiplier) * (
            (1 + nacelle_overheadCostMultiplier + nacelle_assemblyCostMultiplier) * partsCost
        )


###### Tower
# -------------------------------------------------------------------------------
class TowerCost2015(om.ExplicitComponent):
    """
    Compute tower cost in the form of :math:`cost = k*mass`.
    Value of :math:`k` was updated in 2015 to be $2.9 USD/kg.
    Cost includes materials and manufacturing costs.
    Cost can be overridden with use of `tower_cost_external`

    Parameters
    ----------
    tower_mass : float, [kg]
        tower mass
    tower_mass_cost_coeff : float, [USD/kg]
        tower mass-cost coeff
    tower_cost_external : float
        Tower cost computed by TowerSE

    Returns
    -------
    tower_parts_cost : float, [USD]
        Tower parts cost

    """

    def setup(self):
        self.add_input("tower_mass", 0.0, units="kg")
        self.add_input("tower_mass_cost_coeff", 2.9, units="USD/kg")
        self.add_input("tower_cost_external", 0.0, units="USD")

        self.add_output("tower_parts_cost", 0.0, units="USD")

    def compute(self, inputs, outputs):
        tower_mass = inputs["tower_mass"]
        tower_mass_cost_coeff = inputs["tower_mass_cost_coeff"]

        # calculate component cost
        if inputs["tower_cost_external"] < 1.0:
            outputs["tower_parts_cost"] = tower_mass_cost_coeff * tower_mass
        else:
            outputs["tower_parts_cost"] = inputs["tower_cost_external"]

        if outputs["tower_parts_cost"] == 0.0:
            print(
                "Warning: turbine_costsse_2015.py : TowerCost2015.compute : No tower mass provided.  Assuming $0 for tower cost, LCOE will be artificially low."
            )


# -------------------------------------------------------------------------------
class TowerCostAdder2015(om.ExplicitComponent):
    """
    The tower is not aggregated with any other component, but for consistency
    there are allowances for additional costs incurred from transportation and
    assembly complexity,
    :math:`c_{towersys} = (1+kt_{tower}+kp_{tower}) (1+ko_{tower}+ka_{tower}) c_{tower}`

    Where conceptually, :math:`kt` is a transportation multiplier,
    :math:`kp` is a profit multiplier,
    :math:`ko` is an overhead cost multiplier, and
    :math:`ka` is an assembly cost multiplier

    By default, :math:`kt=kp=ko=ka=0`.

    Parameters
    ----------
    tower_parts_cost : float, [USD]
        component cost
    tower_assemblyCostMultiplier : float
        tower assembly cost multiplier
    tower_overheadCostMultiplier : float
        tower overhead cost multiplier
    tower_profitMultiplier : float
        tower profit cost multiplier
    tower_transportMultiplier : float
        tower transport cost multiplier

    Returns
    -------
    tower_cost : float, [USD]
        tower cost

    """

    def setup(self):
        self.add_input("tower_parts_cost", 0.0, units="USD")
        self.add_input("tower_assemblyCostMultiplier", 0.0)
        self.add_input("tower_overheadCostMultiplier", 0.0)
        self.add_input("tower_profitMultiplier", 0.0)
        self.add_input("tower_transportMultiplier", 0.0)

        self.add_output("tower_cost", 0.0, units="USD")

    def compute(self, inputs, outputs):
        tower_parts_cost = inputs["tower_parts_cost"]

        tower_assemblyCostMultiplier = inputs["tower_assemblyCostMultiplier"]
        tower_overheadCostMultiplier = inputs["tower_overheadCostMultiplier"]
        tower_profitMultiplier = inputs["tower_profitMultiplier"]
        tower_transportMultiplier = inputs["tower_transportMultiplier"]

        partsCost = tower_parts_cost
        outputs["tower_cost"] = (1 + tower_transportMultiplier + tower_profitMultiplier) * (
            (1 + tower_overheadCostMultiplier + tower_assemblyCostMultiplier) * partsCost
        )


# -------------------------------------------------------------------------------
class TurbineCostAdder2015(om.ExplicitComponent):
    """
    Aggregates the turbine system costs into a single value with allowances for
    additional costs incurred from transportation and assembly complexity.  Costs
    are reported per kW.  Cost multipliers are of the form,
    :math:`c_{turbine} = (1+kt_{turbine}+kp_{turbine}) (1+ko_{turbine}+ka_{turbine})
    (c_{rotor} + c_{nacelle} + c_{tower})`

    Where conceptually, :math:`kt` is a transportation multiplier,
    :math:`kp` is a profit multiplier,
    :math:`ko` is an overhead cost multiplier, and
    :math:`ka` is an assembly cost multiplier

    By default, :math:`kt=kp=ko=ka=0`.

    Parameters
    ----------
    rotor_cost : float, [USD]
        Rotor cost
    rotor_mass_tcc : float
        Rotor mass
    nacelle_cost : float, [USD]
        Nacelle cost
    nacelle_mass_tcc : float
        Nacelle mass
    tower_cost : float, [USD]
        Tower cost
    tower_mass : float, [kg]
        Tower mass
    machine_rating : float
        Machine rating
    turbine_assemblyCostMultiplier : float
        Turbine multiplier for assembly cost in manufacturing
    turbine_overheadCostMultiplier : float
        Turbine multiplier for overhead
    turbine_profitMultiplier : float
        Turbine multiplier for profit markup
    turbine_transportMultiplier : float
        Turbine multiplier for transport costs

    Returns
    -------
    turbine_mass_tcc : float
        Turbine total mass, without foundation
    turbine_cost : float, [USD]
        Overall wind turbine capital costs including transportation costs
    turbine_cost_kW : float
        Overall wind turbine capial costs including transportation costs

    """

    def setup(self):
        self.add_input("rotor_cost", 0.0, units="USD")
        self.add_input("rotor_mass_tcc", 0.0, units="kg")
        self.add_input("nacelle_cost", 0.0, units="USD")
        self.add_input("nacelle_mass_tcc", 0.0, units="kg")
        self.add_input("tower_cost", 0.0, units="USD")
        self.add_input("tower_mass", 0.0, units="kg")
        self.add_input("machine_rating", 0.0, units="kW")
        self.add_input("turbine_assemblyCostMultiplier", 0.0)
        self.add_input("turbine_overheadCostMultiplier", 0.0)
        self.add_input("turbine_profitMultiplier", 0.0)
        self.add_input("turbine_transportMultiplier", 0.0)

        self.add_output("turbine_mass_tcc", 0.0, units="kg")
        self.add_output("turbine_cost", 0.0, units="USD")
        self.add_output("turbine_cost_kW", 0.0, units="USD/kW")

    def compute(self, inputs, outputs):
        rotor_cost = inputs["rotor_cost"]
        nacelle_cost = inputs["nacelle_cost"]
        tower_cost = inputs["tower_cost"]

        rotor_mass_tcc = inputs["rotor_mass_tcc"]
        nacelle_mass_tcc = inputs["nacelle_mass_tcc"]
        tower_mass = inputs["tower_mass"]

        turbine_assemblyCostMultiplier = inputs["turbine_assemblyCostMultiplier"]
        turbine_overheadCostMultiplier = inputs["turbine_overheadCostMultiplier"]
        turbine_profitMultiplier = inputs["turbine_profitMultiplier"]
        turbine_transportMultiplier = inputs["turbine_transportMultiplier"]

        partsCost = rotor_cost + nacelle_cost + tower_cost

        outputs["turbine_mass_tcc"] = rotor_mass_tcc + nacelle_mass_tcc + tower_mass
        outputs["turbine_cost"] = (1 + turbine_transportMultiplier + turbine_profitMultiplier) * (
            (1 + turbine_overheadCostMultiplier + turbine_assemblyCostMultiplier) * partsCost
        )
        outputs["turbine_cost_kW"] = outputs["turbine_cost"] / inputs["machine_rating"]


class Outputs2Screen(om.ExplicitComponent):
    """
    Print cost outputs to the terminal

    Parameters
    ----------
    blade_cost : float, [USD]
        blade cost
    blade_mass : float, [kg]
        Blade mass
    hub_cost : float, [USD]
        hub cost
    hub_mass : float, [kg]
        Hub mass
    pitch_system_cost : float, [USD]
        pitch_system cost
    pitch_system_mass : float, [kg]
        Pitch system mass
    spinner_cost : float, [USD]
        spinner cost
    spinner_mass : float, [kg]
        Spinner mass
    lss_cost : float, [USD]
        lss cost
    lss_mass : float, [kg]
        LSS mass
    main_bearing_cost : float, [USD]
        main_bearing cost
    main_bearing_mass : float, [kg]
        Main bearing mass
    gearbox_cost : float, [USD]
        gearbox cost
    gearbox_mass : float, [kg]
        LSS mass
    hss_cost : float, [USD]
        hss cost
    hss_mass : float, [kg]
        HSS mass
    brake_cost : float, [USD]
        brake cost
    brake_mass : float, [kg]
        brake mass
    generator_cost : float, [USD]
        generator cost
    generator_mass : float, [kg]
        Generator mass
    bedplate_cost : float, [USD]
        bedplate cost
    bedplate_mass : float, [kg]
        Bedplate mass
    yaw_system_cost : float, [USD]
        yaw_system cost
    yaw_mass : float, [kg]
        Yaw system mass
    hvac_cost : float, [USD]
        hvac cost
    hvac_mass : float, [kg]
        HVAC mass
    cover_cost : float, [USD]
        cover cost
    cover_mass : float, [kg]
        Cover mass
    elec_cost : float, [USD]
        elec cost
    controls_cost : float, [USD]
        controls cost
    platforms_cost : float, [USD]
        platforms cost
    transformer_cost : float, [USD]
        transformer cost
    transformer_mass : float, [kg]
        Transformer mass
    converter_cost : float, [USD]
        converter cost
    converter_mass : float, [kg]
        Converter mass
    rotor_cost : float, [USD]
        rotor cost
    rotor_mass_tcc : float
        Rotor mass
    nacelle_cost : float, [USD]
        nacelle cost
    nacelle_mass_tcc : float
        Nacelle mass
    tower_cost : float, [USD]
        tower cost
    tower_mass : float, [kg]
        Tower mass
    turbine_cost : float, [USD]
        Overall turbine costs
    turbine_cost_kW : float
        Overall wind turbine capital costs including transportation costs per kW
    turbine_mass_tcc : float
        Turbine mass

    """

    def initialize(self):
        self.options.declare("verbosity", default=False)

    def setup(self):

        self.add_input("blade_cost", 0.0, units="USD")
        self.add_input("blade_mass", 0.0, units="kg")
        self.add_input("hub_cost", 0.0, units="USD")
        self.add_input("hub_mass", 0.0, units="kg")
        self.add_input("pitch_system_cost", 0.0, units="USD")
        self.add_input("pitch_system_mass", 0.0, units="kg")
        self.add_input("spinner_cost", 0.0, units="USD")
        self.add_input("spinner_mass", 0.0, units="kg")
        self.add_input("lss_cost", 0.0, units="USD")
        self.add_input("lss_mass", 0.0, units="kg")
        self.add_input("main_bearing_cost", 0.0, units="USD")
        self.add_input("main_bearing_mass", 0.0, units="kg")
        self.add_input("gearbox_cost", 0.0, units="USD")
        self.add_input("gearbox_mass", 0.0, units="kg")
        self.add_input("hss_cost", 0.0, units="USD")
        self.add_input("hss_mass", 0.0, units="kg")
        self.add_input("brake_cost", 0.0, units="USD")
        self.add_input("brake_mass", 0.0, units="kg")
        self.add_input("generator_cost", 0.0, units="USD")
        self.add_input("generator_mass", 0.0, units="kg")
        self.add_input("bedplate_cost", 0.0, units="USD")
        self.add_input("bedplate_mass", 0.0, units="kg")
        self.add_input("yaw_system_cost", 0.0, units="USD")
        self.add_input("yaw_mass", 0.0, units="kg")
        self.add_input("hvac_cost", 0.0, units="USD")
        self.add_input("hvac_mass", 0.0, units="kg")
        self.add_input("cover_cost", 0.0, units="USD")
        self.add_input("cover_mass", 0.0, units="kg")
        self.add_input("elec_cost", 0.0, units="USD")
        self.add_input("controls_cost", 0.0, units="USD")
        self.add_input("platforms_cost", 0.0, units="USD")
        self.add_input("transformer_cost", 0.0, units="USD")
        self.add_input("transformer_mass", 0.0, units="kg")
        self.add_input("converter_cost", 0.0, units="USD")
        self.add_input("converter_mass", 0.0, units="kg")
        self.add_input("rotor_cost", 0.0, units="USD")
        self.add_input("rotor_mass_tcc", 0.0, units="kg")
        self.add_input("nacelle_cost", 0.0, units="USD")
        self.add_input("nacelle_mass_tcc", 0.0, units="kg")
        self.add_input("tower_cost", 0.0, units="USD")
        self.add_input("tower_mass", 0.0, units="kg")
        self.add_input("turbine_cost", 0.0, units="USD")
        self.add_input("turbine_cost_kW", 0.0, units="USD/kW")
        self.add_input("turbine_mass_tcc", 0.0, units="kg")

    def compute(self, inputs, outputs):

        if self.options["verbosity"] == True:
            print("################################################")
            print("Computation of costs of the main turbine components from TurbineCostSE")
            print(
                "Blade cost              %.3f k USD       mass %.3f kg"
                % (inputs["blade_cost"] * 1.0e-003, inputs["blade_mass"])
            )
            print(
                "Pitch system cost       %.3f k USD       mass %.3f kg"
                % (inputs["pitch_system_cost"] * 1.0e-003, inputs["pitch_system_mass"])
            )
            print(
                "Hub cost                %.3f k USD       mass %.3f kg"
                % (inputs["hub_cost"] * 1.0e-003, inputs["hub_mass"])
            )
            print(
                "Spinner cost            %.3f k USD       mass %.3f kg"
                % (inputs["spinner_cost"] * 1.0e-003, inputs["spinner_mass"])
            )
            print("------------------------------------------------")
            print(
                "Rotor cost              %.3f k USD       mass %.3f kg"
                % (inputs["rotor_cost"] * 1.0e-003, inputs["rotor_mass_tcc"])
            )
            print("")
            print(
                "LSS cost                %.3f k USD       mass %.3f kg"
                % (inputs["lss_cost"] * 1.0e-003, inputs["lss_mass"])
            )
            print(
                "Main bearing cost       %.3f k USD       mass %.3f kg"
                % (inputs["main_bearing_cost"] * 1.0e-003, inputs["main_bearing_mass"])
            )
            print(
                "Gearbox cost            %.3f k USD       mass %.3f kg"
                % (inputs["gearbox_cost"] * 1.0e-003, inputs["gearbox_mass"])
            )
            print(
                "HSS cost                %.3f k USD       mass %.3f kg"
                % (inputs["hss_cost"] * 1.0e-003, inputs["hss_mass"])
            )
            print(
                "Brake cost              %.3f k USD       mass %.3f kg"
                % (inputs["brake_cost"] * 1.0e-003, inputs["brake_mass"])
            )
            print(
                "Generator cost          %.3f k USD       mass %.3f kg"
                % (inputs["generator_cost"] * 1.0e-003, inputs["generator_mass"])
            )
            print(
                "Bedplate cost           %.3f k USD       mass %.3f kg"
                % (inputs["bedplate_cost"] * 1.0e-003, inputs["bedplate_mass"])
            )
            print(
                "Yaw system cost         %.3f k USD       mass %.3f kg"
                % (inputs["yaw_system_cost"] * 1.0e-003, inputs["yaw_mass"])
            )
            print(
                "HVAC cost               %.3f k USD       mass %.3f kg"
                % (inputs["hvac_cost"] * 1.0e-003, inputs["hvac_mass"])
            )
            print(
                "Nacelle cover cost      %.3f k USD       mass %.3f kg"
                % (inputs["cover_cost"] * 1.0e-003, inputs["cover_mass"])
            )
            print("Electr connection cost  %.3f k USD" % (inputs["elec_cost"] * 1.0e-003))
            print("Controls cost           %.3f k USD" % (inputs["controls_cost"] * 1.0e-003))
            print("Other main frame cost   %.3f k USD" % (inputs["platforms_cost"] * 1.0e-003))
            print(
                "Transformer cost        %.3f k USD       mass %.3f kg"
                % (inputs["transformer_cost"] * 1.0e-003, inputs["transformer_mass"])
            )
            print(
                "Converter cost          %.3f k USD       mass %.3f kg"
                % (inputs["converter_cost"] * 1.0e-003, inputs["converter_mass"])
            )
            print("------------------------------------------------")
            print(
                "Nacelle cost            %.3f k USD       mass %.3f kg"
                % (inputs["nacelle_cost"] * 1.0e-003, inputs["nacelle_mass_tcc"])
            )
            print("")
            print(
                "Tower cost              %.3f k USD       mass %.3f kg"
                % (inputs["tower_cost"] * 1.0e-003, inputs["tower_mass"])
            )
            print("------------------------------------------------")
            print("------------------------------------------------")
            print(
                "Turbine cost            %.3f k USD       mass %.3f kg"
                % (inputs["turbine_cost"] * 1.0e-003, inputs["turbine_mass_tcc"])
            )
            print("Turbine cost per kW     %.3f k USD/kW" % inputs["turbine_cost_kW"])
            print("################################################")


# -------------------------------------------------------------------------------
class Turbine_CostsSE_2015(om.Group):
    """
    Print cost outputs to the terminal

    """

    def initialize(self):
        self.options.declare("verbosity", default=False)

    def setup(self):
        self.verbosity = self.options["verbosity"]

        self.set_input_defaults("blade_mass_cost_coeff", units="USD/kg", val=14.6)
        self.set_input_defaults("hub_mass_cost_coeff", units="USD/kg", val=3.9)
        self.set_input_defaults("pitch_system_mass_cost_coeff", units="USD/kg", val=22.1)
        self.set_input_defaults("spinner_mass_cost_coeff", units="USD/kg", val=11.1)
        self.set_input_defaults("lss_mass_cost_coeff", units="USD/kg", val=11.9)
        self.set_input_defaults("bearing_mass_cost_coeff", units="USD/kg", val=4.5)
        self.set_input_defaults("gearbox_mass_cost_coeff", units="USD/kg", val=12.9)
        self.set_input_defaults("hss_mass_cost_coeff", units="USD/kg", val=6.8)
        self.set_input_defaults("brake_mass_cost_coeff", units="USD/kg", val=3.6254)
        self.set_input_defaults("generator_mass_cost_coeff", units="USD/kg", val=12.4)
        self.set_input_defaults("bedplate_mass_cost_coeff", units="USD/kg", val=2.9)
        self.set_input_defaults("yaw_mass_cost_coeff", units="USD/kg", val=8.3)
        self.set_input_defaults("converter_mass_cost_coeff", units="USD/kg", val=18.8)
        self.set_input_defaults("transformer_mass_cost_coeff", units="USD/kg", val=18.8)
        self.set_input_defaults("hvac_mass_cost_coeff", units="USD/kg", val=124.0)
        self.set_input_defaults("cover_mass_cost_coeff", units="USD/kg", val=5.7)
        self.set_input_defaults("elec_connec_machine_rating_cost_coeff", units="USD/kW", val=41.85)
        self.set_input_defaults("platforms_mass_cost_coeff", units="USD/kg", val=17.1)
        self.set_input_defaults("tower_mass_cost_coeff", units="USD/kg", val=2.9)
        self.set_input_defaults("controls_machine_rating_cost_coeff", units="USD/kW", val=21.15)
        self.set_input_defaults("crane_cost", units="USD", val=12e3)

        self.set_input_defaults("hub_assemblyCostMultiplier", val=0.0)
        self.set_input_defaults("hub_overheadCostMultiplier", val=0.0)
        self.set_input_defaults("nacelle_assemblyCostMultiplier", val=0.0)
        self.set_input_defaults("nacelle_overheadCostMultiplier", val=0.0)
        self.set_input_defaults("tower_assemblyCostMultiplier", val=0.0)
        self.set_input_defaults("tower_overheadCostMultiplier", val=0.0)
        self.set_input_defaults("turbine_assemblyCostMultiplier", val=0.0)
        self.set_input_defaults("turbine_overheadCostMultiplier", val=0.0)
        self.set_input_defaults("hub_profitMultiplier", val=0.0)
        self.set_input_defaults("nacelle_profitMultiplier", val=0.0)
        self.set_input_defaults("tower_profitMultiplier", val=0.0)
        self.set_input_defaults("turbine_profitMultiplier", val=0.0)
        self.set_input_defaults("hub_transportMultiplier", val=0.0)
        self.set_input_defaults("nacelle_transportMultiplier", val=0.0)
        self.set_input_defaults("tower_transportMultiplier", val=0.0)
        self.set_input_defaults("turbine_transportMultiplier", val=0.0)

        self.add_subsystem("blade_c", BladeCost2015(), promotes=["*"])
        self.add_subsystem("hub_c", HubCost2015(), promotes=["*"])
        self.add_subsystem("pitch_c", PitchSystemCost2015(), promotes=["*"])
        self.add_subsystem("spinner_c", SpinnerCost2015(), promotes=["*"])
        self.add_subsystem("hub_adder", HubSystemCostAdder2015(), promotes=["*"])
        self.add_subsystem("rotor_adder", RotorCostAdder2015(), promotes=["*"])
        self.add_subsystem("lss_c", LowSpeedShaftCost2015(), promotes=["*"])
        self.add_subsystem("bearing_c", BearingCost2015(), promotes=["*"])
        self.add_subsystem("gearbox_c", GearboxCost2015(), promotes=["*"])
        self.add_subsystem("hss_c", HighSpeedShaftCost2015(), promotes=["*"])
        self.add_subsystem("brake_c", BrakeCost2020(), promotes=["*"])
        self.add_subsystem("generator_c", GeneratorCost2015(), promotes=["*"])
        self.add_subsystem("bedplate_c", BedplateCost2015(), promotes=["*"])
        self.add_subsystem("yaw_c", YawSystemCost2015(), promotes=["*"])
        self.add_subsystem("hvac_c", HydraulicCoolingCost2015(), promotes=["*"])
        self.add_subsystem("controls_c", ControlsCost2015(), promotes=["*"])
        self.add_subsystem("converter_c", ConverterCost2015(), promotes=["*"])
        self.add_subsystem("elec_c", ElecConnecCost2015(), promotes=["*"])
        self.add_subsystem("cover_c", NacelleCoverCost2015(), promotes=["*"])
        self.add_subsystem("platforms_c", PlatformsMainframeCost2015(), promotes=["*"])
        self.add_subsystem("transformer_c", TransformerCost2015(), promotes=["*"])
        self.add_subsystem("nacelle_adder", NacelleSystemCostAdder2015(), promotes=["*"])
        self.add_subsystem("tower_c", TowerCost2015(), promotes=["*"])
        self.add_subsystem("tower_adder", TowerCostAdder2015(), promotes=["*"])
        self.add_subsystem("turbine_c", TurbineCostAdder2015(), promotes=["*"])
        self.add_subsystem("outputs", Outputs2Screen(verbosity=self.verbosity), promotes=["*"])
