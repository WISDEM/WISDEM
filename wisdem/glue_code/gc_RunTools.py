import os

import openmdao.api as om
import matplotlib.pyplot as plt
import numpy as np


class Convergence_Trends_Opt(om.ExplicitComponent):
    """
    Deprecating this for now and using OptView from PyOptSparse instead.
    """

    def initialize(self):
        self.options.declare("opt_options")

    def compute(self, inputs, outputs):
        folder_output = self.options["opt_options"]["general"]["folder_output"]
        optimization_log = os.path.join(folder_output, self.options["opt_options"]["recorder"]["file_name"])

        if os.path.exists(optimization_log):
            cr = om.CaseReader(optimization_log)
            cases = cr.get_cases()
            rec_data = {}
            design_vars = {}
            responses = {}
            iterations = []
            for i, it_data in enumerate(cases):
                iterations.append(i)

                # Collect DVs and responses separately for DOE
                for design_var in [it_data.get_design_vars()]:
                    for dv in design_var:
                        if i == 0:
                            design_vars[dv] = []
                        design_vars[dv].append(design_var[dv])

                for response in [it_data.get_responses()]:
                    for resp in response:
                        if i == 0:
                            responses[resp] = []
                        responses[resp].append(response[resp])

                # parameters = it_data.get_responses()
                for parameters in [it_data.get_responses(), it_data.get_design_vars()]:
                    for j, param in enumerate(parameters.keys()):
                        if i == 0:
                            rec_data[param] = []
                        rec_data[param].append(parameters[param])

            if self.options["opt_options"]["driver"]["optimization"]["flag"]:
                for param in rec_data.keys():
                    fig, ax = plt.subplots(1, 1, figsize=(5.3, 4))
                    np_data = np.array(rec_data[param])
                    if len(np_data.shape) < 3:
                        ax.plot(iterations, np_data)
                    elif len(np_data.shape) == 3:
                        for i in range(np_data.shape[2]):
                            ax.plot(iterations, np_data[:,:,i])
                    else:
                        print(
                                f"Warning: Iteration plot not printed for {param} as they are arrays with more than 3 dimensions. Please check plotting logic."
                            )
                    ax.set(xlabel="Number of Iterations", ylabel=param)
                    fig_name = "Convergence_trend_" + param + ".png"
                    fig.savefig(os.path.join(folder_output, fig_name))
                    plt.close(fig)


            elif self.options["opt_options"]["driver"]["design_of_experiments"]["flag"]:
                for resp in responses:
                    fig, ax = plt.subplots(1, 1, figsize=(5.3, 4))
                    for dv in design_vars:
                        try:
                            ax.scatter(design_vars[dv], responses[resp])
                            ax.set(xlabel=dv, ylabel=resp)
                            fig_name = "Experiment_result_" + resp + ".png"
                            fig.savefig(os.path.join(folder_output, fig_name))
                            plt.close(fig)
                        except ValueError:
                            print(
                                f"Warning: Design of experiments plot not printed for {dv} and {resp} as they are array values. Plot generation currently only works for scalar values."
                            )


class Outputs_2_Screen(om.ExplicitComponent):
    # Class to print outputs on screen
    def initialize(self):
        self.options.declare("verbosity", default=False)

    def setup(self):
        self.add_input("aep", val=0.0, units="GW * h")
        self.add_input("blade_mass", val=0.0, units="kg")
        self.add_input("lcoe", val=0.0, units="USD/MW/h")
        self.add_input("My_std", val=0.0, units="N*m")
        self.add_input("flp1_std", val=0.0, units="deg")
        self.add_input("PC_omega", val=0.0, units="rad/s")
        self.add_input("PC_zeta", val=0.0)
        self.add_input("VS_omega", val=0.0, units="rad/s")
        self.add_input("VS_zeta", val=0.0)
        self.add_input("Flp_omega", val=0.0, units="rad/s")
        self.add_input("Flp_zeta", val=0.0)
        self.add_input("tip_deflection", val=0.0, units="m")

    def compute(self, inputs, outputs):
        if self.options["verbosity"] == True: 
            print("########################################")
            print("Objectives")
            print("Turbine AEP: {:8.10f} GWh".format(inputs["aep"][0]))
            print("Blade Mass:  {:8.10f} kg".format(inputs["blade_mass"][0]))
            print("LCOE:        {:8.10f} USD/MWh".format(inputs["lcoe"][0]))
            print("Tip Defl.:   {:8.10f} m".format(inputs["tip_deflection"][0]))
            print("########################################")


class PlotRecorder(om.Group):
    def initialize(self):
        self.options.declare("opt_options")

    def setup(self):
        self.add_subsystem("conv_plots", Convergence_Trends_Opt(opt_options=self.options["opt_options"]))


if __name__ == "__main__":
    opt_options = {}
    opt_options["general"] = {}
    opt_options["general"]["folder_output"] = "path2outputfolder"
    opt_options["recorder"] = {}
    opt_options["recorder"]["file_name"] = "log_opt.sql"

    wt_opt = om.Problem(model=PlotRecorder(opt_options=opt_options))
    wt_opt.setup(derivatives=False)
    wt_opt.run_model()
