import os
import textwrap
import subprocess

import numpy as np


class DakotaOptimizer:
    def __init__(self, template_dir="template_dir/"):
        self.template_dir = template_dir
        self.setup_directories(template_dir)

    def optimize(self, desvars, desired_outputs, bounds, model_string, output_scalers, options=None):
        """
        Helper method that calls all of the other methods to do the full
        top-to-bottom optimization process.
        """

        desvar_labels, desvar_shapes = self.create_input_file(
            self.template_dir, desvars, desired_outputs, bounds, options
        )
        self.create_input_yaml(self.template_dir, desvar_labels)
        self.create_driver_file(self.template_dir, model_string, desvar_shapes, desired_outputs, output_scalers)
        self.run_dakota()
        results = self.postprocess()
        return results

    def setup_directories(self, template_dir):
        """
        Create the template_dir, remove previous files.
        """
        if not os.path.exists(template_dir):
            os.makedirs(template_dir)

        subprocess.call("rm *.in", shell=True)
        subprocess.call("rm *.out", shell=True)
        subprocess.call("rm *.rst", shell=True)
        subprocess.call("rm *.dat", shell=True)
        subprocess.call("rm -rf run_history", shell=True)

    def create_input_file(self, template_dir, desvars, outputs, bounds, options):
        """
        Create the Dakota input file based on the user-defined problem formulation.

        Basically, we need the number and size of design variables, their bounds,
        and the functions of interest. Here is also where you set the optimization
        method and termination criteria.
        """
        flattened_bounds = []

        for key, value in bounds.items():
            if isinstance(value, (float, list)):
                value = np.array(value)
            flattened_value = np.squeeze(value.flatten()).reshape(-1, 2)
            flattened_bounds.extend(flattened_value)

        flattened_bounds = np.array(flattened_bounds)

        desvar_labels = []
        for key, value in desvars.items():
            if isinstance(value, (float, list)):
                value = np.array(value)
            flattened_value = np.squeeze(value.flatten())
            key = key.replace(".", "_")
            for i in range(flattened_value.size):
                desvar_labels.append(f"{key}_{i}")

        desvar_shapes = {}
        total_size = 0

        for key, value in desvars.items():
            if isinstance(value, (float, list)):
                value = np.array(value)

            desvar_shapes[key] = value.shape
            total_size += value.size

        if "cobyla" in options["method"]:
            opt_options = {
                "max_iterations": 100,
                "max_function_evaluations": 200,
                "initial_delta": 0.5,
            }
            opt_options.update(options)

        elif "efficient_global" in options["method"]:
            opt_options = {
                "seed": 123,
            }
            opt_options.update(options)

        method = opt_options.pop("method")

        options_string_list = []
        for option, val in opt_options.items():
            line = f"    {option} = {val}"
            options_string_list.append(line)
        options_string_lines = """
    {}
        """.format(
            "\n".join(options_string_list)
        )

        if len(outputs) > 1:
            constraints_string = f"  nonlinear_equality_constraints {len(outputs)-1}\n"
        else:
            constraints_string = ""

        #### Flatten the DVs here and make names for each and append the names with the number according to the size of the DVs

        # Terrible string-list manipulation to get the DVs and outputs formatted correctly
        input_file = (
            textwrap.dedent(
                """\
        # Dakota input file
        environment
          tabular_data
            tabular_data_file "dakota_data.dat"

        method
        """
            )
            + f"  {method}"
            + options_string_lines
            + textwrap.dedent(
                """
        variables
        """
            )
            + f"  continuous_design {len(flattened_bounds)}\n"
            + f"  lower_bounds "
            + " ".join([str(i) for i in flattened_bounds[:, 0]])
            + "\n"
            + f"  upper_bounds "
            + " ".join([str(i) for i in flattened_bounds[:, 1]])
            + textwrap.dedent(
                """

        interface
          fork
            asynchronous
            evaluation_concurrency 1
            parameters_file "params.in"
            results_file "results.out"
        """
            )
            + f'    copy_files "{template_dir}*"'
            + textwrap.dedent(
                """
            analysis_driver "python openmdao_driver.py"

            work_directory
              named "run_history/run"
              directory_tag
              directory_save
              file_save

        responses
          objective_functions 1
        """
            )
            + constraints_string
            + "  descriptors "
            + " ".join(['"' + key + '"' for key in outputs])
            + """
      no_gradients
      no_hessians
        """
        )

        with open("dakota_input.in", "w") as text_file:
            text_file.write(input_file)

        return desvar_labels, desvar_shapes

    def create_input_yaml(self, template_dir, desvar_labels):
        """
        Create the input yaml file which is just a file with a dictionary format
        so that Dakota knows which design variables to populate in the problem.
        """
        # Populate input.yml
        input_lines = [f"cdv_{i+1}: {{cdv_{i+1}}}" for i in range(len(desvar_labels))]
        with open(template_dir + "input_template.yml", "w") as f:
            for line in input_lines:
                f.write(line + "\n")

    def create_driver_file(self, template_dir, model_string, desvar_shapes, desired_outputs, output_scalers):
        """
        Create the Python file that actually runs the model analysis.

        Once created, this file will take in the design variables as set by Dakota
        and output all functions of interest. This function that creates that file
        needs the model definition (import statement), as well as the desvar sizes,
        the list of desired outputs, and the output scalers.
        """
        desvar_shapes_lines = []
        for key in desvar_shapes:
            desvar_shapes_lines.append(f'desvar_shapes["{key}"] = {desvar_shapes[key]}')
        desvar_shapes_lines = """
{}
        """.format(
            "\n".join(desvar_shapes_lines)
        )

        desired_outputs_string = "desired_outputs = [" + " ".join(['"' + key + '"' for key in desired_outputs]) + "]"

        write_outputs_string = []
        for i, key in enumerate(desired_outputs):
            string = f'f.write(str(float(outputs["{key}"]) * {output_scalers[i]}) + "\\n")'
            write_outputs_string.append(string)

        write_outputs_string = """
    {}
        """.format(
            "\n    ".join(write_outputs_string)
        )

        # Create openmdao_driver.py
        driver_file = (
            textwrap.dedent(
                """\
        # Import  modules
        import sys
        from subprocess import call
        import numpy as np
        from yaml import safe_load


        #########################################
        #                                       #
        #    Step 1: Use Dakota created         #
        #    input files to prepare for         #
        #    model run.                         #
        #                                       #
        #########################################
        input_template = "input_template.yml"
        inputs = "inputs.yml"
        call(["dprepro", sys.argv[1], input_template, inputs])
        call(['rm', input_template])

        #########################################
        #                                       #
        #    Step 2: Run Model                  #
        #                                       #
        #########################################
        # Load parameters from the yaml formatted input.
        with open(inputs, "r") as f:
            desvars = safe_load(f)

        desvars_list = []
        for key in desvars:
            desvars_list.append(float(desvars[key]))
        flattened_desvars = np.array(desvars_list)

        desvar_shapes = {}
        """
            )
            + desvar_shapes_lines
            + textwrap.dedent(
                """
        size_counter = 0
        desvars = {}
        for key, shape in desvar_shapes.items():
            size = int(np.prod(shape))
            desvars[key] = flattened_desvars[
                size_counter : size_counter + size
            ].reshape(shape)
            size_counter += size

        print()
        print('Design variables:')
        print(desvars)
        """
            )
            + model_string
            + "\n"
            + textwrap.dedent(
                """\
        model_instance = model(desvars)
        outputs = model_instance.compute(desvars)
        #########################################
        #                                       #
        #    Step 3: Write Output in format     #
        #    Dakota expects                     #
        #                                       #
        #########################################
        """
            )
            + desired_outputs_string
            + "\n"
            + textwrap.dedent(
                """\
        print('Outputs:')
        print(outputs)
        # Write it to the expected file.
        with open(sys.argv[2], "w") as f:"""
            )
            + write_outputs_string
        )

        with open(template_dir + "openmdao_driver.py", "w") as text_file:
            text_file.write(driver_file)

    def run_dakota(self):
        """
        Actually run Dakota from the command line.
        """
        subprocess.call("dakota -i dakota_input.in -o dakota_output.out -write_restart dakota_restart.rst", shell=True)
        subprocess.call("dakota_restart_util to_tabular dakota_restart.rst dakota_data.dat", shell=True)

    def postprocess(self):
        """
        Postprocess the dakota_data.dat output file and create a dictionary
        with run results.
        """

        results = {}
        key_list = []
        with open("dakota_data.dat") as f:
            for i, line in enumerate(f):
                # Grab headers from first line
                if i == 0:
                    for key in line.split():
                        results[key] = []
                        key_list.append(key)
                if i > 0:
                    split_line = line.split()
                    for j, key in enumerate(key_list):
                        try:
                            # If it's a float, cast it as such
                            column_entry = float(split_line[j])
                        except ValueError:
                            # If it's an actual string, keep it as a string
                            column_entry = split_line[j]
                        results[key].append(column_entry)

        return results
