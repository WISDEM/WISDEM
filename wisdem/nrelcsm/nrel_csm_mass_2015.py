"""
Copyright (c) NREL. All rights reserved.
"""

from __future__ import print_function

import numpy as np
import openmdao.api as om

from wisdem.nrelcsm.nrel_csm_cost_2015 import Turbine_CostsSE_2015


# --------------------------------------------------------------------
class BladeMass(om.ExplicitComponent):
    """
    Compute blade mass of the form :math:`mass = k*diameter^b`.
    Value of :math:`k` was updated in 2015 to be 0.5.
    Value of :math:`b` was updated to be 2.47/2.54 for turbine class I blades with/without carbon or
    2.44/2.5 for other turbine classes with/without carbon.
    Values of k and b can be overridden by the user with use of `blade_mass_coeff` (k) and/or `blade_user_exp` (b).
    To use `blade_user_exp`, the value of `turbine_class` must be less than 1.

    Parameters
    ----------
    rotor_diameter : float, [m]
        rotor diameter of the machine
    turbine_class : float
        turbine class.  Set to 1 for Class I, 2 for Class II+, or 0 for user overrides of blade_user_exp
    blade_has_carbon : boolean
        does the blade have carbon?
    blade_mass_coeff : float
        k in the blade mass equation: k*(rotor_diameter/2)^b
    blade_user_exp : float
        optional user-entered exp for the blade mass equation

    Returns
    -------
    blade_mass : float, [kg]
        component mass

    """

    def setup(self):
        self.add_input("rotor_diameter", 0.0, units="m")
        self.add_discrete_input("turbine_class", 1)
        self.add_discrete_input("blade_has_carbon", False)
        self.add_input("blade_mass_coeff", 0.5)
        self.add_input("blade_user_exp", 2.5)

        self.add_output("blade_mass", 0.0, units="kg")

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        rotor_diameter = inputs["rotor_diameter"]
        turbine_class = discrete_inputs["turbine_class"]
        blade_has_carbon = discrete_inputs["blade_has_carbon"]
        blade_mass_coeff = inputs["blade_mass_coeff"]
        blade_user_exp = inputs["blade_user_exp"]

        # select the exp for the blade mass equation
        exp = 0.0
        if turbine_class == 1:
            if blade_has_carbon:
                exp = 2.47
            else:
                exp = 2.54
        elif turbine_class > 1:
            if blade_has_carbon:
                exp = 2.44
            else:
                exp = 2.50
        else:
            exp = blade_user_exp

        # calculate the blade mass
        outputs["blade_mass"] = blade_mass_coeff * (rotor_diameter / 2) ** exp


# --------------------------------------------------------------------
class HubMass(om.ExplicitComponent):
    """
    Compute hub mass in the form of :math:`mass = k*m_{blade} + b`.
    Value of :math:`k` was updated in 2015 to be 2.3.
    Value of :math:`b` was updated in 2015 to be 1320.

    Parameters
    ----------
    blade_mass : float, [kg]
        component mass
    hub_mass_coeff : float
        k inthe hub mass equation: k*blade_mass + b
    hub_mass_intercept : float
        b in the hub mass equation: k*blade_mass + b

    Returns
    -------
    hub_mass : float, [kg]
        component mass

    """

    def setup(self):
        # Variables
        self.add_input("blade_mass", 0.0, units="kg")
        self.add_input("hub_mass_coeff", 2.3)
        self.add_input("hub_mass_intercept", 1320.0)

        self.add_output("hub_mass", 0.0, units="kg")

    def compute(self, inputs, outputs):
        blade_mass = inputs["blade_mass"]
        hub_mass_coeff = inputs["hub_mass_coeff"]
        hub_mass_intercept = inputs["hub_mass_intercept"]

        # calculate the hub mass
        outputs["hub_mass"] = hub_mass_coeff * blade_mass + hub_mass_intercept


# --------------------------------------------------------------------
class PitchSystemMass(om.ExplicitComponent):
    """
    Compute pitch bearing mass in the form of :math:`m_{bearing} = k*m_{blade}*nblade + b1`.
    Then compute pitch system mass, with bearing housing in the form of :math:`mass = (1+h)*m_{bearing} + b2`.
    The values of the constants were NOT updated in 2015 and are the same as the original CSM.
    Value of :math:`k` is 0.1295.
    Value of :math:`h` is 0.328.
    Value of :math:`b1` is 491.31.
    Value of :math:`b2` is 555.0.

    Parameters
    ----------
    blade_mass : float, [kg]
        component mass
    blade_number : float
        number of rotor blades
    pitch_bearing_mass_coeff : float
        k in the pitch bearing mass equation: k*blade_mass*blade_number + b
    pitch_bearing_mass_intercept : float
        b in the pitch bearing mass equation: k*blade_mass*blade_number + b
    bearing_housing_fraction : float
        bearing housing fraction
    mass_sys_offset : float
        mass system offset

    Returns
    -------
    pitch_system_mass : float, [kg]
        component mass

    """

    def setup(self):
        self.add_input("blade_mass", 0.0, units="kg")
        self.add_discrete_input("blade_number", 3)
        self.add_input("pitch_bearing_mass_coeff", 0.1295)
        self.add_input("pitch_bearing_mass_intercept", 491.31)
        self.add_input("bearing_housing_fraction", 0.3280)
        self.add_input("mass_sys_offset", 555.0)

        self.add_output("pitch_system_mass", 0.0, units="kg")

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        blade_mass = inputs["blade_mass"]
        blade_number = discrete_inputs["blade_number"]
        pitch_bearing_mass_coeff = inputs["pitch_bearing_mass_coeff"]
        pitch_bearing_mass_intercept = inputs["pitch_bearing_mass_intercept"]
        bearing_housing_fraction = inputs["bearing_housing_fraction"]
        mass_sys_offset = inputs["mass_sys_offset"]

        # calculate the hub mass
        pitchBearingMass = pitch_bearing_mass_coeff * blade_mass * blade_number + pitch_bearing_mass_intercept
        outputs["pitch_system_mass"] = pitchBearingMass * (1 + bearing_housing_fraction) + mass_sys_offset


# --------------------------------------------------------------------
class SpinnerMass(om.ExplicitComponent):
    """
    Compute spinner (nose cone) mass in the form of :math:`mass = k*diameter + b`.
    Value of :math:`k` was updated in 2015 to be 15.5.
    Value of :math:`b` was updated in 2015 to be -980.

    Parameters
    ----------
    rotor_diameter : float, [m]
        rotor diameter of the machine
    spinner_mass_coeff : float
        k inthe spinner mass equation: k*rotor_diameter + b
    spinner_mass_intercept : float
        b in the spinner mass equation: k*rotor_diameter + b

    Returns
    -------
    spinner_mass : float, [kg]
        component mass

    """

    def setup(self):
        self.add_input("rotor_diameter", 0.0, units="m")
        self.add_input("spinner_mass_coeff", 15.5)
        self.add_input("spinner_mass_intercept", -980.0)

        self.add_output("spinner_mass", 0.0, units="kg")

    def compute(self, inputs, outputs):
        rotor_diameter = inputs["rotor_diameter"]
        spinner_mass_coeff = inputs["spinner_mass_coeff"]
        spinner_mass_intercept = inputs["spinner_mass_intercept"]

        # calculate the spinner mass
        outputs["spinner_mass"] = spinner_mass_coeff * rotor_diameter + spinner_mass_intercept


# --------------------------------------------------------------------
class LowSpeedShaftMass(om.ExplicitComponent):
    """
    Compute low speed shaft mass in the form of :math:`mass = k*(m_{blade}*power)^b1 + b2`.
    Value of :math:`k` was updated in 2015 to be 13.
    Value of :math:`b1` was updated in 2015 to be 0.65.
    Value of :math:`b2` was updated in 2015 to be 775.

    Parameters
    ----------
    blade_mass : float, [kg]
        mass for a single wind turbine blade
    machine_rating : float, [kW]
        machine rating
    lss_mass_coeff : float
        k inthe lss mass equation: k*(blade_mass*rated_power)^b1 + b2
    lss_mass_exp : float
        b1 in the lss mass equation: k*(blade_mass*rated_power)^b1 + b2
    lss_mass_intercept : float
        b2 in the lss mass equation: k*(blade_mass*rated_power)^b1 + b2

    Returns
    -------
    lss_mass : float, [kg]
        component mass

    """

    def setup(self):
        self.add_input("blade_mass", 0.0, units="kg")
        self.add_input("machine_rating", 0.0, units="kW")
        self.add_input("lss_mass_coeff", 13.0)
        self.add_input("lss_mass_exp", 0.65)
        self.add_input("lss_mass_intercept", 775.0)

        self.add_output("lss_mass", 0.0, units="kg")

    def compute(self, inputs, outputs):
        blade_mass = inputs["blade_mass"]
        machine_rating = inputs["machine_rating"]
        lss_mass_coeff = inputs["lss_mass_coeff"]
        lss_mass_exp = inputs["lss_mass_exp"]
        lss_mass_intercept = inputs["lss_mass_intercept"]

        # calculate the lss mass
        outputs["lss_mass"] = (
            lss_mass_coeff * (blade_mass * machine_rating * 1e-3) ** lss_mass_exp + lss_mass_intercept
        )


# --------------------------------------------------------------------
class BearingMass(om.ExplicitComponent):
    """
    Compute main bearing mass (single bearing) in the form of :math:`mass = k*diameter^b`.
    Value of :math:`k` was updated in 2015 to be 1e-4.
    Value of :math:`b` was updated in 2015 to be 3.5.

    Parameters
    ----------
    rotor_diameter : float, [m]
        rotor diameter of the machine
    bearing_mass_coeff : float
        k inthe bearing mass equation: k*rotor_diameter^b
    bearing_mass_exp : float
        exp in the bearing mass equation: k*rotor_diameter^b

    Returns
    -------
    main_bearing_mass : float, [kg]
        component mass

    """

    def setup(self):
        # Variables
        self.add_input("rotor_diameter", 0.0, units="m")
        self.add_input("bearing_mass_coeff", 0.0001)
        self.add_input("bearing_mass_exp", 3.5)

        self.add_output("main_bearing_mass", 0.0, units="kg")

    def compute(self, inputs, outputs):
        rotor_diameter = inputs["rotor_diameter"]
        bearing_mass_coeff = inputs["bearing_mass_coeff"]
        bearing_mass_exp = inputs["bearing_mass_exp"]

        # calculates the mass of a SINGLE bearing
        outputs["main_bearing_mass"] = bearing_mass_coeff * rotor_diameter**bearing_mass_exp


# --------------------------------------------------------------------
class RotorTorque(om.ExplicitComponent):
    """
    Computed rated rpm and rotor torque from rated power, rotor diameter, max tip speed, and drivetrain efficiency.
    Rotor torque will be used to size other drivetrain components, such as the generator.

    Parameters
    ----------
    rotor_diameter : float, [m]
        rotor diameter of the machine
    machine_rating : float, [kW]
        machine rating
    max_tip_speed : float, [m/s]
        Maximum allowable blade tip speed
    max_efficiency : float
        Maximum possible drivetrain efficiency

    Returns
    -------
    rated_rpm : float, [rpm]
        rpm of rotor at rated power
    rotor_torque : float, [MN*m]
        torque from rotor at rated power

    """

    def setup(self):
        self.add_input("rotor_diameter", 0.0, units="m")
        self.add_input("machine_rating", 0.0, units="kW")
        self.add_input("max_tip_speed", 0.0, units="m/s")
        self.add_input("max_efficiency", 0.0)

        self.add_output("rated_rpm", 0.0, units="rpm")
        self.add_output("rotor_torque", 0.0, units="kN*m")

    def compute(self, inputs, outputs):
        # Rotor force calculations for nacelle inputs
        maxTipSpd = inputs["max_tip_speed"]
        maxEfficiency = inputs["max_efficiency"]

        ratedHubPower_kW = inputs["machine_rating"] / maxEfficiency
        rotorSpeed = maxTipSpd / (0.5 * inputs["rotor_diameter"])
        outputs["rated_rpm"] = rotorSpeed / (2 * np.pi) * 60.0
        outputs["rotor_torque"] = ratedHubPower_kW / rotorSpeed


# --------------------------------------------------------------------
class GearboxMass(om.ExplicitComponent):
    """
    Compute gearbox mass in the form of :math:`mass = k*torque^b`.
    Value of :math:`k` was updated in 2015 to be 113.
    Value of :math:`b` was updated in 2015 to be 0.71.

    Parameters
    ----------
    rotor_torque : float, [N*m]
        torque from rotor at rated power
    gearbox_torque_density : float, [N*m/kg]
        In 2024, modern 5-7MW gearboxes are able to reach 200 Nm/kg

    Returns
    -------
    gearbox_mass : float, [kg]
        component mass

    """

    def setup(self):
        self.add_input("rotor_torque", 0.0, units="kN*m")
        self.add_input("gearbox_torque_density", 200.0, units='N*m/kg', desc='In 2024, modern 5-7MW gearboxes are able to reach 200 Nm/kg')

        self.add_output("gearbox_mass", 0.0, units="kg")

    def compute(self, inputs, outputs):

        # calculate the gearbox mass
        outputs["gearbox_mass"] = inputs["rotor_torque"] * 1.e+3 / inputs["gearbox_torque_density"]


# --------------------------------------------------------------------
class BrakeMass(om.ExplicitComponent):
    """
    Compute brake mass in the form of :math:`mass = k*torque`.
    Value of :math:`k` was updated in 2020 to be 0.00122.

    Parameters
    ----------
    rotor_torque : float, [N*m]
        rotor torque at rated power
    brake_mass_coeff : float
        Mass scaling coefficient

    Returns
    -------
    brake_mass : float, [kg]
        overall component mass
    """

    def setup(self):
        self.add_input("rotor_torque", 0.0, units="N*m")
        self.add_input("brake_mass_coeff", 0.00122)

        self.add_output("brake_mass", 0.0, units="kg")

    def compute(self, inputs, outputs):
        # Unpack inputs
        rotor_torque = inputs["rotor_torque"]
        coeff = inputs["brake_mass_coeff"]

        # Regression based sizing derived by J.Keller under FOA 1981 support project
        outputs["brake_mass"] = coeff * rotor_torque


# --------------------------------------------------------------------
class HighSpeedShaftMass(om.ExplicitComponent):
    """
    Compute high speed shaft mass in the form of :math:`mass = k*power`.
    Value of :math:`k` was updated in 2015 to be 0.19894.

    Parameters
    ----------
    machine_rating : float, [kW]
        machine rating
    hss_mass_coeff : float
        NREL CSM hss equation; removing intercept since it is negligible

    Returns
    -------
    hss_mass : float, [kg]
        component mass

    """

    def setup(self):
        self.add_input("machine_rating", 0.0, units="kW")
        self.add_input("hss_mass_coeff", 0.19894)

        self.add_output("hss_mass", 0.0, units="kg")

    def compute(self, inputs, outputs):
        machine_rating = inputs["machine_rating"]
        hss_mass_coeff = inputs["hss_mass_coeff"]

        outputs["hss_mass"] = hss_mass_coeff * machine_rating


# --------------------------------------------------------------------
class GeneratorMass(om.ExplicitComponent):
    """
    Compute generator mass in the form of :math:`mass = k*power + b`.
    Value of :math:`k` was updated in 2015 to be 2.3 (for rating in kW).
    Value of :math:`b` was updated in 2015 to be 3400.

    Parameters
    ----------
    machine_rating : float, [kW]
        machine rating
    generator_mass_coeff : float
        k inthe generator mass equation: k*rated_power + b
    generator_mass_intercept : float
        b in the generator mass equation: k*rated_power + b

    Returns
    -------
    generator_mass : float, [kg]
        component mass

    """

    def setup(self):
        self.add_input("machine_rating", 0.0, units="kW")
        self.add_input("generator_mass_coeff", 2.3)
        self.add_input("generator_mass_intercept", 3400.0)

        self.add_output("generator_mass", 0.0, units="kg")

    def compute(self, inputs, outputs):
        machine_rating = inputs["machine_rating"]
        generator_mass_coeff = inputs["generator_mass_coeff"]
        generator_mass_intercept = inputs["generator_mass_intercept"]

        # calculate the generator mass
        outputs["generator_mass"] = generator_mass_coeff * machine_rating + generator_mass_intercept


# --------------------------------------------------------------------
class BedplateMass(om.ExplicitComponent):
    """
    Compute bedplate mass in the form of :math:`mass = diameter^b`.
    Value of :math:`b` was updated in 2015 to be 2.2.

    Parameters
    ----------
    rotor_diameter : float, [m]
        rotor diameter of the machine
    bedplate_mass_exp : float
        exp in the bedplate mass equation: rotor_diameter^b

    Returns
    -------
    bedplate_mass : float, [kg]
        component mass

    """

    def setup(self):
        self.add_input("rotor_diameter", 0.0, units="m")
        self.add_input("bedplate_mass_exp", 2.2)

        self.add_output("bedplate_mass", 0.0, units="kg")

    def compute(self, inputs, outputs):
        rotor_diameter = inputs["rotor_diameter"]
        bedplate_mass_exp = inputs["bedplate_mass_exp"]

        # calculate the bedplate mass
        outputs["bedplate_mass"] = rotor_diameter**bedplate_mass_exp


# --------------------------------------------------------------------
class YawSystemMass(om.ExplicitComponent):
    """
    Compute yaw system mass in the form of :math:`mass = k*diameter^b`.
    The values of the constants were NOT updated in 2015 and are the same as the original CSM.
    Value of :math:`k` is 9e-4.
    Value of :math:`b` is 3.314.

    Parameters
    ----------
    rotor_diameter : float, [m]
        rotor diameter of the machine
    yaw_mass_coeff : float
        k inthe yaw mass equation: k*rotor_diameter^b
    yaw_mass_exp : float
        exp in the yaw mass equation: k*rotor_diameter^b

    Returns
    -------
    yaw_mass : float, [kg]
        component mass

    """

    def setup(self):
        self.add_input("rotor_diameter", 0.0, units="m")
        self.add_input("yaw_mass_coeff", 0.0009)
        self.add_input("yaw_mass_exp", 3.314)

        self.add_output("yaw_mass", 0.0, units="kg")

    def compute(self, inputs, outputs):
        rotor_diameter = inputs["rotor_diameter"]
        yaw_mass_coeff = inputs["yaw_mass_coeff"]
        yaw_mass_exp = inputs["yaw_mass_exp"]

        # calculate yaw system mass #TODO - 50% adder for non-bearing mass
        outputs["yaw_mass"] = 1.5 * (
            yaw_mass_coeff * rotor_diameter**yaw_mass_exp
        )  # JMF do we really want to expose all these?


# TODO: no variable speed mass; ignore for now


# --------------------------------------------------------------------
class HydraulicCoolingMass(om.ExplicitComponent):
    """
    Compute hydraulic cooling mass in the form of :math:`mass = k*power`.
    The values of the constants were NOT updated in 2015 and are the same as the original CSM.
    Value of :math:`k` is 0.08.

    Parameters
    ----------
    machine_rating : float, [kW]
        machine rating
    hvac_mass_coeff : float
        hvac linear coeff

    Returns
    -------
    hvac_mass : float, [kg]
        component mass

    """

    def setup(self):
        self.add_input("machine_rating", 0.0, units="kW")
        self.add_input("hvac_mass_coeff", 0.08)

        self.add_output("hvac_mass", 0.0, units="kg")

    def compute(self, inputs, outputs):
        machine_rating = inputs["machine_rating"]
        hvac_mass_coeff = inputs["hvac_mass_coeff"]

        # calculate hvac system mass
        outputs["hvac_mass"] = hvac_mass_coeff * machine_rating


# --------------------------------------------------------------------
class NacelleCoverMass(om.ExplicitComponent):
    """
    Compute nacelle cover mass in the form of :math:`mass = k*power + b`.
    The values of the constants were NOT updated in 2015 and are the same as the original CSM.
    Value of :math:`k` is 1.2817.
    Value of :math:`b` is 428.19.

    Parameters
    ----------
    machine_rating : float, [kW]
        machine rating
    cover_mass_coeff : float
        k inthe spinner mass equation: k*rotor_diameter + b
    cover_mass_intercept : float
        b in the spinner mass equation: k*rotor_diameter + b

    Returns
    -------
    cover_mass : float, [kg]
        component mass

    """

    def setup(self):
        self.add_input("machine_rating", 0.0, units="kW")
        self.add_input("cover_mass_coeff", 1.2817)
        self.add_input("cover_mass_intercept", 428.19)

        self.add_output("cover_mass", 0.0, units="kg")

    def compute(self, inputs, outputs):
        machine_rating = inputs["machine_rating"]
        cover_mass_coeff = inputs["cover_mass_coeff"]
        cover_mass_intercept = inputs["cover_mass_intercept"]

        # calculate nacelle cover mass
        outputs["cover_mass"] = cover_mass_coeff * machine_rating + cover_mass_intercept


# TODO: ignoring controls and electronics mass for now


# --------------------------------------------------------------------
class PlatformsMainframeMass(om.ExplicitComponent):
    """
    Compute platforms mass in the form of :math:`mass = k*m_{bedplate}` and
    crane mass as 3000kg, if flagged by the user.
    The values of the constants were NOT updated in 2015 and are the same as the original CSM.
    Value of :math:`k` is 0.125.

    Parameters
    ----------
    bedplate_mass : float, [kg]
        component mass
    platforms_mass_coeff : float
        nacelle platforms mass coeff as a function of bedplate mass [kg/kg]
    crane : boolean
        flag for presence of onboard crane
    crane_weight : float, [kg]
        weight of onboard crane

    Returns
    -------
    platforms_mass : float, [kg]
        component mass

    """

    # nacelle platforms, service crane, base hardware

    def setup(self):
        self.add_input("bedplate_mass", 0.0, units="kg")
        self.add_input("platforms_mass_coeff", 0.125)
        self.add_discrete_input("crane", False)
        self.add_input("crane_weight", 3000.0, units="kg")

        self.add_output("platforms_mass", 0.0, units="kg")
        # TODO: there is no base hardware mass model in the old model. Cost is not dependent on mass.

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        bedplate_mass = inputs["bedplate_mass"]
        platforms_mass_coeff = inputs["platforms_mass_coeff"]
        crane = discrete_inputs["crane"]
        crane_weight = inputs["crane_weight"]

        # calculate nacelle cover mass
        platforms_mass = platforms_mass_coeff * bedplate_mass

        # --- crane ---
        if crane:
            crane_mass = crane_weight
        else:
            crane_mass = 0.0

        outputs["platforms_mass"] = platforms_mass + crane_mass


# --------------------------------------------------------------------
class TransformerMass(om.ExplicitComponent):
    """
    Compute transformer mass in the form of :math:`mass = k*power + b`.
    Value of :math:`k` was updated in 2015 to be 1.915 (for rating in kW).
    Value of :math:`b` was updated in 2015 to be 1910.

    Parameters
    ----------
    machine_rating : float, [kW]
        machine rating
    transformer_mass_coeff : float
        k inthe transformer mass equation: k*rated_power + b
    transformer_mass_intercept : float
        b in the transformer mass equation: k*rated_power + b

    Returns
    -------
    transformer_mass : float, [kg]
        component mass

    """

    def setup(self):
        self.add_input("machine_rating", 0.0, units="kW")
        self.add_input("transformer_mass_coeff", 1.9150)
        self.add_input("transformer_mass_intercept", 1910.0)

        self.add_output("transformer_mass", 0.0, units="kg")

    def compute(self, inputs, outputs):
        machine_rating = inputs["machine_rating"]
        transformer_mass_coeff = inputs["transformer_mass_coeff"]
        transformer_mass_intercept = inputs["transformer_mass_intercept"]

        # calculate the transformer mass
        outputs["transformer_mass"] = transformer_mass_coeff * machine_rating + transformer_mass_intercept


# --------------------------------------------------------------------
class TowerMass(om.ExplicitComponent):
    """
    Compute tower mass in the form of :math:`mass = k*H_{hub}^b`.
    Value of :math:`k` was updated in 2015 to be 19.828.
    Value of :math:`b` was updated in 2015 to be 2.0282.

    Parameters
    ----------
    tower_length : float, [m]
        This is the hub height of wind turbine above ground for onshore turbines.  For offshore, this should be entered as the length from transition piece to hub height.
    tower_mass_coeff : float
        k in the tower mass equation: k*tower_length^b
    tower_mass_exp : float
        b in the tower mass equation: k*tower_length^b

    Returns
    -------
    tower_mass : float, [kg]
        component mass

    """

    def setup(self):
        self.add_input("tower_length", 0.0, units="m")
        self.add_input("tower_mass_coeff", 19.828)
        self.add_input("tower_mass_exp", 2.0282)

        self.add_output("tower_mass", 0.0, units="kg")

    def compute(self, inputs, outputs):
        tower_length = inputs["tower_length"]
        tower_mass_coeff = inputs["tower_mass_coeff"]
        tower_mass_exp = inputs["tower_mass_exp"]

        # calculate the tower mass
        outputs["tower_mass"] = tower_mass_coeff * tower_length**tower_mass_exp


# Turbine mass adder
class TurbineMassAdder(om.ExplicitComponent):
    """
    Aggregates all components masses into category labels of hub system, rotor, nacelle, and tower.

    Parameters
    ----------
    blade_mass : float, [kg]
        component mass
    hub_mass : float, [kg]
        component mass
    pitch_system_mass : float, [kg]
        component mass
    spinner_mass : float, [kg]
        component mass
    lss_mass : float, [kg]
        component mass
    main_bearing_mass : float, [kg]
        component mass
    gearbox_mass : float, [kg]
        component mass
    hss_mass : float, [kg]
        component mass
    brake_mass : float, [kg]
        component mass
    generator_mass : float, [kg]
        component mass
    bedplate_mass : float, [kg]
        component mass
    yaw_mass : float, [kg]
        component mass
    hvac_mass : float, [kg]
        component mass
    cover_mass : float, [kg]
        component mass
    platforms_mass : float, [kg]
        component mass
    transformer_mass : float, [kg]
        component mass
    tower_mass : float, [kg]
        component mass
    blade_number : float
        number of rotor blades
    main_bearing_number : float
        number of main bearings

    Returns
    -------
    hub_system_mass : float, [kg]
        hub system mass
    rotor_mass : float, [kg]
        hub system mass
    nacelle_mass : float, [kg]
        nacelle mass
    turbine_mass : float, [kg]
        turbine mass

    """

    def setup(self):
        # rotor
        self.add_input("blade_mass", 0.0, units="kg")
        self.add_input("hub_mass", 0.0, units="kg")
        self.add_input("pitch_system_mass", 0.0, units="kg")
        self.add_input("spinner_mass", 0.0, units="kg")

        # nacelle
        self.add_input("lss_mass", 0.0, units="kg")
        self.add_input("main_bearing_mass", 0.0, units="kg")
        self.add_input("gearbox_mass", 0.0, units="kg")
        self.add_input("hss_mass", 0.0, units="kg")
        self.add_input("brake_mass", 0.0, units="kg")
        self.add_input("generator_mass", 0.0, units="kg")
        self.add_input("bedplate_mass", 0.0, units="kg")
        self.add_input("yaw_mass", 0.0, units="kg")
        self.add_input("hvac_mass", 0.0, units="kg")
        self.add_input("cover_mass", 0.0, units="kg")
        self.add_input("platforms_mass", 0.0, units="kg")
        self.add_input("transformer_mass", 0.0, units="kg")

        # tower
        self.add_input("tower_mass", 0.0, units="kg")
        self.add_discrete_input("blade_number", 3)
        self.add_discrete_input("main_bearing_number", 2)

        self.add_output("hub_system_mass", 0.0, units="kg")
        self.add_output("rotor_mass", 0.0, units="kg")
        self.add_output("nacelle_mass", 0.0, units="kg")
        self.add_output("turbine_mass", 0.0, units="kg")

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        blade_mass = inputs["blade_mass"]
        hub_mass = inputs["hub_mass"]
        pitch_system_mass = inputs["pitch_system_mass"]
        spinner_mass = inputs["spinner_mass"]
        lss_mass = inputs["lss_mass"]
        main_bearing_mass = inputs["main_bearing_mass"]
        gearbox_mass = inputs["gearbox_mass"]
        hss_mass = inputs["hss_mass"]
        brake_mass = inputs["brake_mass"]
        generator_mass = inputs["generator_mass"]
        bedplate_mass = inputs["bedplate_mass"]
        yaw_mass = inputs["yaw_mass"]
        hvac_mass = inputs["hvac_mass"]
        cover_mass = inputs["cover_mass"]
        platforms_mass = inputs["platforms_mass"]
        transformer_mass = inputs["transformer_mass"]
        tower_mass = inputs["tower_mass"]
        blade_number = discrete_inputs["blade_number"]
        bearing_number = discrete_inputs["main_bearing_number"]

        outputs["hub_system_mass"] = hub_mass + pitch_system_mass + spinner_mass
        outputs["rotor_mass"] = blade_mass * blade_number + outputs["hub_system_mass"]
        outputs["nacelle_mass"] = (
            lss_mass
            + bearing_number * main_bearing_mass
            + gearbox_mass
            + hss_mass
            + brake_mass
            + generator_mass
            + bedplate_mass
            + yaw_mass
            + hvac_mass
            + cover_mass
            + platforms_mass
            + transformer_mass
        )
        outputs["turbine_mass"] = outputs["rotor_mass"] + outputs["nacelle_mass"] + tower_mass


# --------------------------------------------------------------------


class nrel_csm_mass_2015(om.Group):
    def setup(self):
        self.add_subsystem("blade", BladeMass(), promotes=["*"])
        self.add_subsystem("hub", HubMass(), promotes=["*"])
        self.add_subsystem("pitch", PitchSystemMass(), promotes=["*"])
        self.add_subsystem("spinner", SpinnerMass(), promotes=["*"])
        self.add_subsystem("lss", LowSpeedShaftMass(), promotes=["*"])
        self.add_subsystem("bearing", BearingMass(), promotes=["*"])
        self.add_subsystem("torque", RotorTorque(), promotes=["*"])
        self.add_subsystem("gearbox", GearboxMass(), promotes=["*"])
        self.add_subsystem("hss", HighSpeedShaftMass(), promotes=["*"])
        self.add_subsystem("brake", BrakeMass(), promotes=["*"])
        self.add_subsystem("generator", GeneratorMass(), promotes=["*"])
        self.add_subsystem("bedplate", BedplateMass(), promotes=["*"])
        self.add_subsystem("yaw", YawSystemMass(), promotes=["*"])
        self.add_subsystem("hvac", HydraulicCoolingMass(), promotes=["*"])
        self.add_subsystem("cover", NacelleCoverMass(), promotes=["*"])
        self.add_subsystem("platforms", PlatformsMainframeMass(), promotes=["*"])
        self.add_subsystem("transformer", TransformerMass(), promotes=["*"])
        self.add_subsystem("tower", TowerMass(), promotes=["*"])
        self.add_subsystem("turbine", TurbineMassAdder(), promotes=["*"])


class nrel_csm_2015(om.Group):
    def setup(self):
        self.add_subsystem("nrel_csm_mass", nrel_csm_mass_2015(), promotes=["*"])
        self.add_subsystem("turbine_costs", Turbine_CostsSE_2015(verbosity=False), promotes=["*"])
