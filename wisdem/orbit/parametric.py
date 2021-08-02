__author__ = ["Jake Nunemaker"]
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = ["jake.nunemaker@nrel.gov"]


import time
from copy import deepcopy
from random import sample
from itertools import product

import numpy as np
import pandas as pd
from benedict import benedict
from wisdem.orbit import ProjectManager


class ParametricManager:
    """Class for configuring parametric ORBIT runs."""

    def __init__(self, base, params, funcs, weather=None, module=None, product=False):
        """
        Creates an instance of `ParametricRun`.

        Parameters
        ----------
        base : dict
            Base ORBIT configuration.
        params : dict
            Parameters and their values.
            Format: "subdict.param": [num1, num2, num3]
        funcs : dict
            List of functions used to save results.
            Format: "name": lambda project: project.attr
        weather : DataFrame

        """

        self.base = benedict(base)
        self.params = params
        self.funcs = funcs
        self.weather = weather
        self.results = None
        self.module = module
        self.product = product

    def run(self, **kwargs):
        """Run the configured parametric runs and save any requested results to
        `self.results`."""

        outputs = []
        for run in self.run_list:
            data = self._run_config(run, **kwargs)
            outputs.append(data)

        self.results = pd.DataFrame(outputs)

    def _run_config(self, run, **kwargs):
        """Run an individual config."""

        config = deepcopy(self.base)
        config.merge(run)

        if self.module is not None:
            project = self.module(config, weather=self.weather, **kwargs)
            project.run()

        else:
            project = ProjectManager(config, weather=self.weather, **kwargs)
            project.run()

        results = self.map_funcs(project, self.funcs)
        data = {**run, **results}

        return data

    @property
    def run_list(self):
        """Returns list of configured parametric runs."""

        if self.product:
            runs = list(product(*self.params.values()))

        else:
            runs = list(zip(*self.params.values()))

        return [dict(zip(self.params.keys(), run)) for run in runs]

    @property
    def num_runs(self):
        return len(self.run_list)

    @staticmethod
    def map_funcs(obj, funcs):
        """
        Map `obj` to list of `funcs`.

        Parameters
        ----------
        obj : ProjectManager
            Project instance to run through functions.
        funcs : list
            Functions used to pull results from obj.
        """

        results = {}
        for k, f in funcs.items():
            try:
                res = f(obj)

            except TypeError:
                raise TypeError(
                    f"Result function '{f}' not structured properly. " f"Correct format: 'lambda project: project.{f}'"
                )

            except AttributeError:
                res = np.NaN

            results[k] = res

        return results

    def preview(self, num=10, **kwargs):
        """
        Runs a limited set of runs to preview the results and provide an
        estimate for total run time.

        Parameters
        ----------
        num : int
            Number to run.
        """

        start = time.time()
        outputs = []
        if num > len(self.run_list):
            to_run = self.run_list

        else:
            to_run = sample(self.run_list, num)

        for run in to_run:
            data = self._run_config(run, **kwargs)
            outputs.append(data)

        elapsed = time.time() - start
        estimate = (len(self.run_list) / num) * elapsed
        print(f"{num} runs elapsed time: {elapsed:.2f}s")
        print(f"{self.num_runs} runs estimated time: {estimate:.2f}s")

        return pd.DataFrame(outputs)
