import openmdao.api as om


class TurbineClass(om.ExplicitComponent):
    """
    Compute velocities based on the chosen turbine class

    Parameters
    ----------
    turbine_class : string
        IEC turbine class
    V_mean_overwrite : float
        overwrite value for mean velocity for using user defined CDFs

    Returns
    -------
    V_mean : float, [m/s]
        IEC mean wind speed for Rayleigh distribution
    V_extreme1 : float, [m/s]
        IEC extreme wind speed at hub height for a 1-year retunr period
    V_extreme50 : float, [m/s]
        IEC extreme wind speed at hub height for a 50-year retunr period

    """

    def setup(self):
        # parameters
        self.add_discrete_input("turbine_class", val="I")
        self.add_input("V_mean_overwrite", val=0.0)
        self.add_input("V_extreme50_overwrite", val=0.0)

        self.add_output("V_mean", 0.0, units="m/s")
        self.add_output("V_extreme1", 0.0, units="m/s")
        self.add_output("V_extreme50", 0.0, units="m/s")

        # self.declare_partials('*', '*', method='fd', form='central', step=1e-6)

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        turbine_class = discrete_inputs["turbine_class"].upper()

        if turbine_class == "I":
            Vref = 50.0
        elif turbine_class == "II":
            Vref = 42.5
        elif turbine_class == "III":
            Vref = 37.5
        elif turbine_class == "IV":
            Vref = 30.0
        else:
            raise ValueError("turbine_class input must be I/II/III/IV")

        if inputs["V_mean_overwrite"] == 0.0:
            outputs["V_mean"] = 0.2 * Vref
        else:
            outputs["V_mean"] = inputs["V_mean_overwrite"]

        outputs["V_extreme1"] = 0.8 * Vref

        if inputs["V_extreme50_overwrite"] == 0.0:
            outputs["V_extreme50"] = 1.4 * Vref
        else:
            outputs["V_extreme50"] = inputs["V_extreme50_overwrite"]
