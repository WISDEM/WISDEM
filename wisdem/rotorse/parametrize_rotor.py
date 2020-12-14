import numpy as np
import os
from openmdao.api import ExplicitComponent, Group
from scipy.interpolate import PchipInterpolator


class ParametrizeBladeAero(ExplicitComponent):
    # Openmdao component to parameterize distributed quantities for the outer shape of the wind turbine rotor blades
    def initialize(self):
        self.options.declare("rotorse_options")
        self.options.declare("opt_options")

    def setup(self):
        rotorse_options = self.options["rotorse_options"]
        self.opt_options = self.options["opt_options"]
        n_span = rotorse_options["n_span"]
        self.n_opt_twist = n_opt_twist = self.opt_options["design_variables"]["blade"]["aero_shape"]["twist"][
            "n_opt"
        ]
        self.n_opt_chord = n_opt_chord = self.opt_options["design_variables"]["blade"]["aero_shape"]["chord"][
            "n_opt"
        ]

        # Inputs
        self.add_input(
            "s",
            val=np.zeros(n_span),
            desc="1D array of the non-dimensional spanwise grid defined along blade axis (0-blade root, 1-blade tip)",
        )
        # Blade twist
        self.add_input(
            "twist_original",
            val=np.zeros(n_span),
            units="rad",
            desc="1D array of the twist values defined along blade span. The twist is the one defined in the yaml.",
        )
        self.add_input(
            "s_opt_twist",
            val=np.zeros(n_opt_twist),
            desc="1D array of the non-dimensional spanwise grid defined along blade axis to optimize the blade twist angle",
        )
        self.add_input(
            "twist_opt_gain",
            val=np.ones(n_opt_twist),
            desc="1D array of the non-dimensional gains to optimize the blade spanwise distribution of the twist angle",
        )
        # Blade chord
        self.add_input(
            "chord_original",
            val=np.zeros(n_span),
            units="m",
            desc="1D array of the chord values defined along blade span. The chord is the one defined in the yaml.",
        )
        self.add_input(
            "s_opt_chord",
            val=np.zeros(n_opt_chord),
            desc="1D array of the non-dimensional spanwise grid defined along blade axis to optimize the blade chord",
        )
        self.add_input(
            "chord_opt_gain",
            val=np.ones(n_opt_chord),
            desc="1D array of the non-dimensional gains to optimize the blade spanwise distribution of the chord",
        )

        # Outputs
        self.add_output(
            "twist_param",
            val=np.zeros(n_span),
            units="rad",
            desc="1D array of the twist values defined along blade span. The twist is the result of the parameterization.",
        )
        self.add_output(
            "chord_param",
            val=np.zeros(n_span),
            units="m",
            desc="1D array of the chord values defined along blade span. The chord is the result of the parameterization.",
        )
        self.add_output(
            "max_chord_constr",
            val=np.zeros(n_opt_chord),
            desc="1D array of the ratio between chord values and maximum chord along blade span.",
        )

    def compute(self, inputs, outputs):

        spline = PchipInterpolator

        if self.opt_options["design_variables"]["blade"]["aero_shape"]["twist"]["flag"] == True:
            lb_twist = np.array(
                self.opt_options["design_variables"]["blade"]["aero_shape"]["twist"]["lower_bound"]
            )
            ub_twist = np.array(
                self.opt_options["design_variables"]["blade"]["aero_shape"]["twist"]["upper_bound"]
            )
            twist_opt_gain_nd = inputs["twist_opt_gain"]
            twist_upper = np.ones(self.n_opt_twist) * ub_twist
            twist_lower = np.ones(self.n_opt_twist) * lb_twist
            twist_opt_rad = twist_opt_gain_nd * (twist_upper - twist_lower) + twist_lower

            spline = PchipInterpolator
            twist_spline = spline(inputs["s_opt_twist"], twist_opt_rad)
            outputs["twist_param"] = twist_spline(inputs["s"])
        else:
            outputs["twist_param"] = inputs["twist_original"]

        chord_spline = spline(inputs["s_opt_chord"], inputs["chord_opt_gain"])
        outputs["chord_param"] = inputs["chord_original"] * chord_spline(inputs["s"])

        chord_opt = spline(inputs["s"], outputs["chord_param"])
        max_chord = self.opt_options["constraints"]["blade"]["chord"]["max"]
        outputs["max_chord_constr"] = chord_opt(inputs["s_opt_chord"]) / max_chord


class ParametrizeBladeStruct(ExplicitComponent):
    # Openmdao component to parameterize distributed quantities for the structural design of the wind turbine rotor blades
    def initialize(self):
        self.options.declare("rotorse_options")
        self.options.declare("opt_options")

    def setup(self):
        rotorse_options = self.options["rotorse_options"]
        self.opt_options = opt_options = self.options["opt_options"]
        self.n_span = n_span = rotorse_options["n_span"]
        self.n_layers = n_layers = rotorse_options["n_layers"]
        self.n_opt_spar_cap_ss = n_opt_spar_cap_ss = opt_options["design_variables"]["blade"]["structure"][
            "spar_cap_ss"
        ]["n_opt"]
        self.n_opt_spar_cap_ps = n_opt_spar_cap_ps = opt_options["design_variables"]["blade"]["structure"][
            "spar_cap_ps"
        ]["n_opt"]

        # Inputs
        self.add_input(
            "s",
            val=np.zeros(n_span),
            desc="1D array of the non-dimensional spanwise grid defined along blade axis (0-blade root, 1-blade tip)",
        )
        # Blade spar suction side
        self.add_input(
            "layer_thickness_original",
            val=np.zeros((n_layers, n_span)),
            units="m",
            desc="2D array of the thickness of the layers of the blade structure. The first dimension represents each layer, the second dimension represents each entry along blade span.",
        )
        self.add_output(
            "s_opt_spar_cap_ss",
            val=np.zeros(n_opt_spar_cap_ss),
            desc="1D array of the non-dimensional spanwise grid defined along blade axis to optimize the blade spar cap suction side",
        )
        self.add_input(
            "spar_cap_ss_opt_gain",
            val=np.ones(n_opt_spar_cap_ss),
            desc="1D array of the non-dimensional gains to optimize the blade spanwise distribution of the spar caps suction side",
        )
        # Blade spar suction side
        self.add_output(
            "s_opt_spar_cap_ps",
            val=np.zeros(n_opt_spar_cap_ps),
            desc="1D array of the non-dimensional spanwise grid defined along blade axis to optimize the blade spar cap pressure side",
        )
        self.add_input(
            "spar_cap_ps_opt_gain",
            val=np.ones(n_opt_spar_cap_ps),
            desc="1D array of the non-dimensional gains to optimize the blade spanwise distribution of the spar caps pressure side",
        )

        # Outputs
        self.add_output(
            "layer_thickness_param",
            val=np.zeros((n_layers, n_span)),
            units="m",
            desc="2D array of the thickness of the layers of the blade structure after the parametrization. The first dimension represents each layer, the second dimension represents each entry along blade span.",
        )

    def compute(self, inputs, outputs):

        spar_cap_ss_name = self.options["rotorse_options"]["spar_cap_ss"]
        spar_cap_ps_name = self.options["rotorse_options"]["spar_cap_ps"]

        layer_name = self.options["rotorse_options"]["layer_name"]

        for i in range(self.n_layers):
            if layer_name[i] == spar_cap_ss_name:
                opt_gain_m_interp = np.interp(inputs["s"], outputs["s_opt_spar_cap_ss"], inputs["spar_cap_ss_opt_gain"])
            elif layer_name[i] == spar_cap_ps_name:
                if (
                    self.opt_options["design_variables"]["blade"]["structure"]["spar_cap_ps"]["equal_to_suction"]
                    == False
                ):
                    opt_gain_m_interp = np.interp(
                        inputs["s"], outputs["s_opt_spar_cap_ps"], inputs["spar_cap_ps_opt_gain"]
                    )
                else:
                    opt_gain_m_interp = np.interp(
                        inputs["s"], outputs["s_opt_spar_cap_ss"], inputs["spar_cap_ss_opt_gain"]
                    )
            else:
                opt_gain_m_interp = np.ones(self.n_span)

            outputs["layer_thickness_param"][i, :] = inputs["layer_thickness_original"][i, :] * opt_gain_m_interp
