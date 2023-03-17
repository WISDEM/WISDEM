import os

import pandas as pd
import openmdao.api as om


class ReadInWind(om.ExplicitComponent):
    """
    Quick wrapper to read in wind timeseries history from TurbSim.
    This requires WEIS to be installed.

    The wind timeseries will then be passed into the powercurve-approximated
    conversion to obtain a power timeseries that can then be passed to an
    electrolyzer model. This means that turbine design changes on the WISDEM
    level will propagate through this component and affect the power and H2 produced.
    """

    def initialize(self):
        self.options.declare("filename")

    def setup(self):
        fname = self.options["filename"]
    
        _, ext = os.path.splitext(fname)

        if ext == ".bts":
            try:
                from weis.aeroelasticse.turbsim_file import TurbSimFile
            except:
                raise Exception(
                    "Trying to read in a TurbSim file but WEIS is not installed. Please install WEIS to use its file processor."
                )

            out = TurbSimFile(fname)
            iy, iz = out._iMid()
            self.wind = out["u"][0, :, iy, iz]
            self.time = out["t"]
            time_unit = "s"
        else:
            data = pd.read_csv(fname, index_col='time_index', parse_dates=True)
            wind = data.iloc[:, 0]
            self.wind = wind.values
            self.time = range(len(wind))
            time_unit = "h" # Should we make this smarter? Possibly infer from index

        self.n_timesteps = len(self.time)

        self.add_output("time", shape=self.n_timesteps, units=time_unit)
        self.add_output("wind", shape=self.n_timesteps, units="m/s")


    def compute(self, _inputs, outputs):
        outputs["time"] = self.time
        outputs["wind"] = self.wind
