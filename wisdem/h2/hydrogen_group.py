import openmdao.api as om
import electrolyzer.inputs.validation as val

from .electrolyzer_comp import ElectrolyzerModel
from .read_in_wind import ReadInWind
from .compute_power import ComputePower
from .read_in_power import ReadInPower


class HydrogenProduction(om.Group):
    """
    This is an OpenMDAO group to combine all of the relevant H2 production components.

    It has some logic based on user-set options within modeling options.
    Specifically, it checks to see if users want to read in existing wind or
    power files. Otherwise this uses a sort of dummy wind timeseries.

    Nominally, this group could be quite modular and allow for different H2
    production models, electrolyzers, degradation considerations, etc.
    """

    def initialize(self):
        self.options.declare("modeling_options")
        self.options.declare("opt_options")

    def setup(self):
        modeling_options = self.options["modeling_options"]
        h2_opt_options = self.options["opt_options"]["design_variables"]["electrolyzer"]
        h2_options = modeling_options["WISDEM"]["HydrogenProduction"]

        read_in_wind_file = "wind_filename" in h2_options
        read_in_power_signal = "power_filename" in h2_options

        if read_in_power_signal:
            self.add_subsystem(
                "read_in_power",
                ReadInPower(filename=h2_options["power_filename"]),
                promotes=["*"],
            )
        elif read_in_wind_file:
            self.add_subsystem(
                "read_in_wind",
                ReadInWind(filename=h2_options["wind_filename"]),
                promotes=["*"],
            )

        self.add_subsystem("compute_power", ComputePower(modeling_options=modeling_options), promotes=["*"])

        h2_modeling_options = val.load_modeling_yaml(h2_options["modeling_options"])
        
        h2_model = ElectrolyzerModel(
            h2_modeling_options=h2_modeling_options,
            h2_opt_options=h2_opt_options,
            modeling_options=h2_options
        )
        self.add_subsystem("electrolyzer", h2_model, promotes=["*"])
