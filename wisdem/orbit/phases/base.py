"""Provides the `BasePhase` class."""

__author__ = ["Jake Nunemaker", "Rob Hammond"]
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from abc import ABC, abstractmethod
from copy import deepcopy

from wisdem.orbit.library import (
    initialize_library,
    extract_library_data,
    extract_library_specs,
)
from wisdem.orbit.core.exceptions import MissingInputs


class BasePhase(ABC):
    """
    Base Phase Class.

    This class is not intended to be instantiated, but define the required
    interfaces for all phases defined by subclasses. Many of the methods below
    should be overwritten in subclasses.

    Attributes
    ----------
    phase : str
        Name of the phase that is being used.
    total_phase_cost : float
        Calculates the total phase cost. Should be implemented in each subclass.
    detailed_output : dict
        Creates the detailed output dictionary. Should be implemented in each
        subclass.
    phase_dataframe : pd.DataFrame

    Methods
    -------
    run()
        Runs the required internal methods to complete the phase. Should be
        implemented in each subclass.
    """

    def initialize_library(self, config, **kwargs):
        """
        Initializes the library if a path is given.

        Parameters
        ----------
        config : dict
            Configuration dictionary.
        library_path : str
            Path to the data library.
        """

        initialize_library(kwargs.get("library_path", None))
        return extract_library_data(config)

    def extract_phase_kwargs(self, **kwargs):
        """
        Consistent handling of kwargs for Phase and subclasses.
        """

        phase_name = kwargs.get("phase_name", None)
        if phase_name is not None:
            self.phase = phase_name

    def extract_defaults(self):
        """
        Extracts the default data from the library.
        """

        self.defaults = extract_library_specs("defaults", "project")

    @classmethod
    def _check_keys(cls, expected, config):
        """
        Basic recursive key check.

        Parameters
        ----------
        expected : dict
            Expected config.
        config : dict
            Possible phase_config.
        """

        missing = []

        for k, v in expected.items():

            if isinstance(k, str) and "variable" in k:
                continue

            if isinstance(v, str) and "optional" in v:
                continue

            if isinstance(v, dict):
                c = config.get(k, {})
                if not isinstance(c, dict):
                    raise TypeError(f"'{k}' must be type 'dict'.")

                _m = cls._check_keys(v, c)
                m = [f"{k}.{i}" for i in _m]
                missing.extend(m)
                continue

            c = config.get(k, None)
            if c is None:
                missing.append(k)

        return missing

    def validate_config(self, config):
        """
        Validates `config` against `self.expected_config`.

        Parameters
        ----------
        config : dict
            Input config.

        Raises
        ------
        MissingInputs
        """

        expected = deepcopy(getattr(self, "expected_config", None))
        if expected is None:
            raise AttributeError(f"'expected_config' not set for '{self}'.")

        missing = self._check_keys(expected, config)

        if missing:
            raise MissingInputs(missing)

        else:
            return config

    @abstractmethod
    def run(self):
        """Main run function for phase."""

        pass

    @property
    @abstractmethod
    def total_phase_cost(self):
        """Returns total phase cost in $USD."""

        pass

    @property
    @abstractmethod
    def total_phase_time(self):
        """Returns total phase time in hours."""

        pass

    @property
    @abstractmethod
    def detailed_output(self):
        """Returns detailed phase information."""

        pass
