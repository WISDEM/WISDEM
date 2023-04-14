import logging

import openmdao.api as om

from electrolyzer import run_electrolyzer
import electrolyzer.inputs.validation as val

logger = logging.getLogger("wisdem/weis")


class ElectrolyzerModel(om.ExplicitComponent):
    """
    This is an OpenMDAO wrapper to the generic electrolyzer model above.

    It makes some assumptions about the number of electrolyzers, stack size, and
    how to distribute electricity across the different electrolyzers. These
    could be later made into WISDEM modeling options to allow for more user configuration.
    """
    def initialize(self):
        self.options.declare("h2_modeling_options_path")
        self.options.declare("h2_opt_options")

    def setup(self):
        self.add_input("p_wind", shape_by_conn=True, units="W")
        if self.options["h2_opt_options"]["control"]["system_rating_MW"]["flag"]:
            self.add_input("system_rating_MW", units="MW")
        self.add_output("h2_produced", units="kg")
        self.add_output("max_curr_density", units="A/cm**2")

        self.h2_modeling_options_dict = val.load_modeling_yaml(self.options["h2_modeling_options_path"])

    def compute(self, inputs, outputs):
        # Set electrolyzer parameters from model inputs
        power_signal = inputs["p_wind"]

        if self.options["h2_opt_options"]["control"]["system_rating_MW"]["flag"]:
            self.h2_modeling_options_dict["electrolyzer"]["control"]["system_rating_MW"] = inputs["system_rating_MW"][0]

        h2_prod, max_curr_density = run_electrolyzer(
            self.h2_modeling_options_dict,
            power_signal,
            optimize=True
        )

        msg = (
            f"\n====== Electrolyzer ======\n"
            f"  - h2 produced (kg): {h2_prod}\n"
            f"  - max current density (A/cm^2): {max_curr_density}\n"
        )

        logger.info(msg)

        outputs["h2_produced"] = h2_prod
        outputs["max_curr_density"] = max_curr_density