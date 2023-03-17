import logging

import numpy as np
import openmdao.api as om

from electrolyzer import run_electrolyzer

logger = logging.getLogger("wisdem/weis")


class ElectrolyzerModel(om.ExplicitComponent):
    """
    This is an OpenMDAO wrapper to the generic electrolyzer model above.

    It makes some assumptions about the number of electrolyzers, stack size, and
    how to distribute electricity across the different electrolyzers. These
    could be later made into WISDEM modeling options to allow for more user configuration.
    """
    def initialize(self):
        self.options.declare("h2_modeling_options")

    def setup(self):
        self.add_input("p_wind", shape_by_conn=True, units="kW")
        self.add_output("h2_produced", units="kg")
        self.add_output("max_curr_density", units="A/cm**2")

    def compute(self, inputs, outputs):
        # Set electrolyzer parameters from model inputs
        power_signal = inputs["p_wind"]

        # Create cosine test signal.
        # TODO: Remove once we merge the hourly timescale update in NREL/electrolyzer
        turbine_rating = 3.4  # MW
        test_signal_angle = np.linspace(0, 8 * np.pi, 3600 * 8 + 10)
        base_value = (turbine_rating / 2) + 0.2
        variation_value = turbine_rating - base_value
        power_test_signal = (base_value + variation_value * np.cos(test_signal_angle)) * 1e6
        # ---

        h2_prod, max_curr_density = run_electrolyzer(
            self.options["h2_modeling_options"],
            power_test_signal,
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