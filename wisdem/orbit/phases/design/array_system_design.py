"""Provides the `ArraySystemDesign` class."""

__author__ = "Rob Hammond"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Rob Hammond"
__email__ = "robert.hammond@nrel.gov"


import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from geopy import distance

from wisdem.orbit.library import export_library_specs, extract_library_specs
from wisdem.orbit.phases.design._cables import Plant, CableSystem


class ArraySystemDesign(CableSystem):
    """
    The design phase for an array cabling system.

    Attributes
    ----------
    system : `Plant`
        A `Plant` object for wind farm specifications.
    cables : `collections.OrderedDict`
        A dictionary of `Cable` objects sorted from smallest to largest.
    full_string : list
        A list of `Cable.name` that represents the longest (in turbine
        connections) possible string.
    partial_string : list
        A list of `Cable.name` that represents the leftover turbines that could
        not create another `full_string` to complete the layout.
    num_turbines_full_string : int
        Number of turbines on a `full_string`.
    num_turbines_partial_string : int
        Number of turbines on a `partial_string`.
    num_full_strings : int
        Number of full strings required to connect the wind farm.
    num_partial_strings : int
        Number of partial strings required to connect the wind farm.
    num_strings : int
        Total number of strings in the system.
    turbines_x : np.ndarray, [`num_strings`, `num_turbines_full_string`]
        The relative x-coordinates of every turbine.
    turbines_y : np.ndarray, [`num_strings`, `num_turbines_full_string`]
        The relative y-coordinates of every turbines.
    oss_x : float
        The relative x-coordinates for the OSS.
    oss_y : float
        The relative y-coordinates for the OSS.
    coordinates : np.ndarray, [`num_strings`, `num_turbines_full_string` + 1, 2]
        The relative (x, y) coordinates for the entire wind farm with the
        first column representing the OSS.
    sections_distances : np.ndarray, [`num_strings`, `num_turbines_full_string`]
        The Euclidean distance between any two points in a string extending
        outward from the OSS.
    sections_cable_lengths : np.ndarray, [`num_strings`, `num_turbines_full_string`]
        `sections_distances` + 2 * `system.site_depth` to account for the water
        depth at each turbine.
    sections_cables : np.ndarray, [`num_strings`, `num_turbines_full_string`]
        The type of cable being used to connect turbines in a string. All
        values are either ``None`` or `Cable.name`.
    """

    expected_config = {
        "site": {"depth": "m"},
        "plant": {
            "layout": "str",
            "row_spacing": "rotor diameters",
            "turbine_spacing": "rotor diameters",
            "turbine_distance": "km (optional)",
            "num_turbines": "int",
            "substation_distance": "km",
        },
        "turbine": {"rotor_diameter": "m", "turbine_rating": "MW"},
        "array_system_design": {
            "design_time": "hrs (optional)",
            "cables": "list | str",
            "average_exclusion_percent": "float (optional)",
        },
    }

    output_config = {"array_system": {"cables": "dict"}}

    def __init__(self, config, **kwargs):
        """
        Defines the cables and sections required to install an offshore wind
        farm.

        Parameters
        ----------
        config : dict
            Configuration dictionary for the array cabling system. See
            `expected_config` for details on what is required.
        """

        super().__init__(config, "array", **kwargs)

        self.exclusion = 1 + self.config["array_system_design"].get(
            "average_exclusion_percent", 0.0
        )
        self.extract_phase_kwargs(**kwargs)
        self.system = Plant(self.config)

    @property
    def total_length(self):
        """Returns total array system length."""

        return sum([v for _, v in self.total_cable_length_by_type.items()])

    @property
    def total_cable_cost(self):
        """Returns total array system cable cost."""

        return sum(self.cost_by_type.values())

    @property
    def detailed_output(self):
        """Returns array system design outputs."""

        _output = {
            **self.design_result,
            "array_system_num_strings": self.num_strings,
            "array_system_total_length": self.total_length,
            "array_system_length_by_type": self.total_cable_length_by_type,
            "array_system_total_cost": self.total_cable_cost,
            "array_system_cost_by_type": self.cost_by_type,
        }

        return _output

    def _compute_maximum_turbines_per_cable(self):
        """
        Calculates the maximum turbines that each cable can support and adds
        it to the `Cable` object.
        """

        for cable in self.cables.values():
            with np.errstate(divide="ignore", invalid="ignore"):
                cable.max_turbines = np.floor(
                    cable.cable_power / self.system.turbine_rating
                )
                if cable.max_turbines == float("inf"):
                    raise ValueError("Must be at least 1 turbine in windfarm!")

    def _compute_string(self, max_turbines_per_string):
        """
        Calculates the maximum number of turbines that each cable type
        can support and builds a string (as a list) of cable types that
        is of length of `max_turbines_per_string` or the maximum number
        of turbines that the largest cable can support, whichever is lower.

        Parameters
        ----------
        max_turbines_per_string : int
            The maximum number of turbines that can be supported on a single
            string given the cable(s) properties.

        Returns
        -------
        cable_layout : list
            A list of cables required to create an the maximum length string
            possible given `cables` and `max_turbines_per_string`. The order
            of the cabling is from largest to smallest cable type with the
            `Cable.name` as the list entry.
        """

        cable_layout = []
        _cables = list(self.cables.values())

        n = len(cable_layout)
        while n < max_turbines_per_string and _cables:
            _cable = _cables.pop(0)

            # Ensure that the most turbines in a string is is lower than the
            # string maximum and the maximum the individual cable can support,
            # then add another cable.
            while max_turbines_per_string > n < _cable.max_turbines:
                cable_layout.append(_cable.name)
                n = len(cable_layout)

        # Return the reverse to start from the largest cable to the smallest
        # cable to represent the string extending from the OSS.
        return cable_layout[::-1]

    def create_strings(self):
        """
        Calculates the required full and partial string design.

        .. note:: For custom layouts this is to provide guidance on the number
        of strings.
        """

        self._compute_maximum_turbines_per_cable()

        # Create the longest string possible with the given cable properties
        self.full_string = self._compute_string(
            list(self.cables.values())[-1].max_turbines
        )
        self.num_turbines_full_string = len(self.full_string)
        self.num_full_strings, self.num_turbines_partial_string = np.divmod(
            self.system.num_turbines, self.num_turbines_full_string
        )

        # Create a partial string constrained by the remainder
        self.partial_string = self._compute_string(
            self.num_turbines_partial_string
        )
        self.num_partial_strings = (
            1 if self.num_turbines_partial_string > 0 else 0
        )

        self.num_strings = self.num_full_strings + self.num_partial_strings

    def _design_grid_layout(self):
        """
        Makes the coordinates of a default grid layout.
        """

        # Create the relative (x, y) coordinate matrices for the turbines
        # using vector math

        # X = column vector of turbine distance
        #     * row vector of range(1, num_turbines_full_string + 1)
        self.turbines_x = np.full(
            self.num_strings, self.system.turbine_distance
        ).reshape(-1, 1) * np.add(
            np.arange(self.num_turbines_full_string, dtype=float), 1
        )

        # Y = column vector of reverse range(1, num_strings)
        #     * row vector of row_distance
        self.turbines_y = np.arange(self.num_strings, dtype=float)[
            ::-1
        ].reshape(-1, 1) * np.full(
            (1, self.num_turbines_full_string), self.system.row_distance
        )

        # If there are partial strings the default layout then null out
        # the non-existent turbines
        if self.partial_string:
            self.turbines_x[-1, self.num_turbines_partial_string :] = None
            self.turbines_y[-1, self.num_turbines_partial_string :] = None

        # Create the relative OSS coordinates
        self.oss_x = 0.0
        self.oss_y = self.turbines_y[:, 0].mean()

    def _design_ring_layout(self):
        """
        Creates the coordinates of a default ring layout.
        """

        # Calculate the radius of each turbine from the OSS
        radius = (
            np.arange(self.num_turbines_full_string, dtype=float)
            * self.system.turbine_distance
            + self.system.substation_distance
        )

        # Calculate the angle required to make rings evenly spaced
        radians = np.linspace(0, 2 * np.pi, self.num_strings + 1)

        # Calculate the relative x and y coordinates for the turbines
        x = np.sin(radians).reshape(-1, 1) * radius
        y = np.cos(radians).reshape(-1, 1) * radius

        # Remove the repeated row that completes the circle
        self.turbines_x = np.delete(x, (-1), axis=0)
        self.turbines_y = np.delete(y, (-1), axis=0)

        # Remove the extra turbines from the partial string if there is one
        if self.partial_string:
            self.turbines_x[-1, self.num_turbines_partial_string :] = None
            self.turbines_y[-1, self.num_turbines_partial_string :] = None

        # Create the relative OSS coordinates
        self.oss_x = 0.0
        self.oss_y = 0.0

    def _create_wind_farm_layout(self):
        """
        Creates a list of substation-to-string distances based off the layout
        of the wind farm.
        """

        if self.system.layout == "custom":
            raise NotImplementedError(
                "Use `CustomArraySystemDesign` for custom layouts"
            )

        if self.system.layout == "grid":
            self._design_grid_layout()

        if self.system.layout == "ring":
            self._design_ring_layout()

        # Create the relative wind farm coordinates
        self.coordinates = np.dstack(
            (
                np.insert(self.turbines_x, 0, self.oss_x, axis=1),
                np.insert(self.turbines_y, 0, self.oss_y, axis=1),
            )
        )

        # Take the norm of the difference of turbine "coordinate pairs".
        self.sections_distance = np.round_(
            np.array(
                [
                    np.linalg.norm(
                        self.coordinates[i, :-1] - self.coordinates[i, 1:],
                        axis=1,
                    )
                    for i in range(self.coordinates.shape[0])
                ]
            ),
            10,
        )

    def _create_cable_section_lengths(self):
        """
        For each cable compute the number of section lengths required to
        complete the array cabling.
        """

        if getattr(self, "sections_cable_lengths", np.zeros(1)).sum() == 0:
            self.sections_cable_lengths = (
                self.sections_distance * self.exclusion
                + (2 * self.system.site_depth)
            )
        self.sections_cables = np.full(
            (self.num_strings, self.num_turbines_full_string), None
        )

        # Create an array of the cable names for each cable section
        for i, row in enumerate(np.isnan(self.coordinates[:, :, 0])):
            ix = row.sum()
            # Fill each string (row in array) with the cables names for valid
            # positions in the layout starting from the end
            self.sections_cables[
                i, 0 : self.num_turbines_full_string - ix
            ] = self.full_string[::-1][: self.num_turbines_full_string - ix][
                ::-1
            ]

    def run(self):
        """
        Runs all the functions to create an array sytem.
        """

        self._initialize_cables()
        self.create_strings()
        self._create_wind_farm_layout()
        self._create_cable_section_lengths()

    def _plot_oss(self, ax):
        """
        Adds the offshore substation(s) to the plot.

        Colors are a selection of the colorblind pallette from Bokeh:
        https://github.com/bokeh/bokeh/blob/master/bokeh/palettes.py#L1938-L1940

        Parameters
        ----------
        ax : matplotlib.axes
            Axis object to add the substation(s) to.
        """

        #        ['orange', 'blugren', 'skyblue', 'vermill', 'redprpl']
        colors = ["#E69F00", "#009E73", "#56B4E9", "#D55E00", "#CC79A7"]
        kwargs = {"s": 300, "c": "#E69F00", "zorder": 2}

        # If this is a custom layout there could be multiple OSS to consider
        if getattr(self, "windfarm", None) is not None:
            labels_set = []
            locations = self.windfarm.drop_duplicates(
                subset=[
                    "substation_id",
                    "substation_name",
                    "substation_longitude",
                    "substation_latitude",
                ],
                keep="first",
            )
            for ix, (_, row) in enumerate(locations.iterrows()):
                name = row.substation_name
                name = row.substation_id if name == 0 else name
                labels_set.append(name)
                kwargs["c"] = colors[ix]
                ax.scatter(
                    row.substation_longitude,
                    row.substation_latitude,
                    label=name,
                    **kwargs,
                )
        else:
            labels_set = ["OSS"]
            ax.scatter(self.oss_x, self.oss_y, label="OSS", **kwargs)

        return labels_set + ["Turbine"], ax

    def plot_array_system(self, show=True, save_path_name=None):
        """
        Plot the array cabling system.

        Parameters
        ----------
        show : bool, default: True
            If True the plot will be output inline or to screen.
        save_path_name : str, default: None
            The <path_to_file>/<file_name> of where to save the created plot.
            If None then the plot will not be saved.
        """

        fig, ax = plt.subplots(figsize=(7, 7))
        plt.axis("off")

        # Plot the offshore substation and turbiness
        labels_set, ax = self._plot_oss(ax)
        ax.scatter(
            self.coordinates[:, 1:, 0],
            self.coordinates[:, 1:, 1],
            marker="1",
            s=250,
            c="#0072B2",
            linewidth=2,
            zorder=2,
            label="Turbine",
        )

        # Determine the cable section widths
        string_sets = np.unique(
            [
                list(
                    OrderedDict.fromkeys(el for el in cables if el is not None)
                )
                for cables in self.sections_cables
            ]
        )
        if isinstance(string_sets[0], list):
            max_string = max(string_sets, key=len)
        else:
            max_string = string_sets[::-1]
        string_widths = np.arange(len(max_string), 0, -1, dtype=float).tolist()

        for i, row in enumerate(self.sections_cables):
            for cable, width in zip(max_string, string_widths):

                ix_to_plot = np.where(row == cable)[0]
                if ix_to_plot.size == 0:
                    continue
                ix_to_plot = np.append(ix_to_plot, ix_to_plot.max() + 1)

                ax.plot(
                    self.coordinates[i, ix_to_plot, 0],
                    self.coordinates[i, ix_to_plot, 1],
                    linewidth=width,
                    color="k",
                    zorder=1,
                    label=cable,
                )

        # Make the plot range larger to evenly contain the legend
        lim_range = np.diff(ax.get_xlim())[0] / 20
        shift = lim_range * np.array([-1, 1])
        ax.set_xlim(np.array(ax.get_xlim()) + shift)
        ax.set_ylim(np.array(ax.get_ylim()) + shift)

        # Constrain the legend to only have on of each type of cable
        handles, labels = ax.get_legend_handles_labels()
        labels_set.extend([str(el) for el in max_string])
        ix_filter = [labels.index(el) for el in labels_set]
        handles = [handles[ix] for ix in ix_filter]
        labels = [labels[ix] for ix in ix_filter]
        ax.legend(
            handles,
            labels,
            ncol=(2 + len(max_string)),
            loc="lower center",
            mode="expand",
            borderpad=0.7,
            borderaxespad=-0.4,
            labelspacing=1.2,
        )

        plt.tight_layout()
        if save_path_name is not None:
            plt.savefig(save_path_name, bbox_inches="tight", dpi=360)
        if show:
            plt.show()


class CustomArraySystemDesign(ArraySystemDesign):
    """
    Custom array system design phase.

    Parameters
    ----------
    ArraySystemDesign : ArraySystemDesign
        Array system design phase.

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    ValueError
        [description]
    ValueError
        [description]
    ValueError
        [description]
    """

    expected_config = {
        "site": {"depth": "str"},
        "plant": {"layout": "str", "num_turbines": "int"},
        "turbine": {"turbine_rating": "int | float"},
        "array_system_design": {
            "design_time": "int | float (optional)",
            "cables": "list | str",
            "location_data": "str",
            "average_exclusion_percent": "float (optional)",
        },
    }

    # Columns that should be included in csv file.
    COLUMNS = [
        "id",
        "substation_id",
        "name",
        "latitude",
        "longitude",
        "string",
        "order",
        "cable_length",
        "bury_speed",
    ]

    # Transformed data column names
    REQUIRED = [
        "id",
        "substation_id",
        "substation_name",
        "substation_latitude",
        "substation_longitude",
        "turbine_name",
        "turbine_latitude",
        "turbine_longitude",
        "string",
        "order",
    ]
    OPTIONAL = ["cable_length", "bury_speed"]

    def __init__(self, config, **kwargs):
        """
        Initializes the configuration.

        The location data must have the following columns:
        - substation_id : int
            ID for the substation.
        - substation_latitude : float
            Y-coordinate for a turbine's corresponding substation.
        - substation_longitude : float
            X-coordinate for a turbine's corresponding substation.
        - turbine_name : str
            Identifying name/ID for a turbine.
        - turbine_latitude : float
            Y-coordinate for the turbine.
        - turbine_longitude : float
            X-coordinate for the turbine.
        - string : int
            String number, starting from 0, the turbine is connected with.
        - order : int
            Turbine position on a string, starting from the OSS, starting
            from 0.

        Optional columns:
        - substation_name : str
            Descriptive name of the substation, default `substation_id`
        - cable_length : float
            Custom cable section length to account for exclusions that are not
            able to be accounted for with `average_exclusion_percent`,
            default 0.0, in km.
        - bury_speed : float
            Custom cable burying speeds for each cable section for highly
            variable soil conditions, by default 0.0 (uses vessel properites).

        Parameters
        ----------
        config : dict
            Configuration dictionary. See `expected_config`.
        """

        super().__init__(config, **kwargs)

    def create_project_csv(self, save_name):
        """Creates a base CSV in <`library_path`>/cables/

        Parameters
        ----------
        save_name : [type]
            [description]
        """

        self._initialize_cables()
        self.create_strings()

        # Print the required output.
        rows = [
            ("N turbines full string", self.num_turbines_full_string),
            ("N full strings", self.num_full_strings),
            ("N turbines partial string", self.num_turbines_partial_string),
            ("N partial strings", self.num_partial_strings),
        ]
        border = "|"
        title = "PROJECT SPECIFICATIONS".rjust(26)
        sep = f"+{'-' * 32}+"
        print(sep)
        sep = "+".join((sep[:28], sep[29:]))
        print(f"{border} {title.ljust(30)} {border}")
        print(sep)
        for desc, n in rows:
            print(
                " ".join(
                    (border, desc.ljust(25), border, str(n).rjust(2), border)
                )
            )
        print(sep)

        # Create the string order/number information
        strings = [
            (i, j)
            for i in range(self.num_full_strings)
            for j in range(self.num_turbines_full_string)
        ]
        strings.extend(
            [
                (i, j)
                for i in range(self.num_full_strings, self.num_strings)
                for j in range(self.num_turbines_partial_string)
            ]
        )
        rows = [
            [f"t{i}", "oss1", f"turbine-{i}", 0.0, 0.0, string, order, 0, 0]
            for i, (string, order) in enumerate(strings)
        ]
        first = ["oss1", "", "offshore_substation", 0.0, 0.0, "", "", "", ""]
        rows.insert(0, first)
        rows.insert(0, self.COLUMNS)
        print(
            f"Saving custom array CSV to: <library_path>/cables/{save_name}.csv"
        )
        export_library_specs("cables", save_name, rows, file_ext="csv")

    def _format_windfarm_data(self):

        # Separate the OSS data
        oss = self.windfarm[pd.isnull(self.windfarm.substation_id)].copy()
        oss.rename(
            columns={
                "latitude": "substation_latitude",
                "longitude": "substation_longitude",
                "name": "substation_name",
            },
            inplace=True,
        )
        oss.substation_id = oss["id"]
        oss.drop(
            ["id", "string", "order", "cable_length", "bury_speed"],
            inplace=True,
            axis=1,
        )

        # Separate the turbine data
        turbines = self.windfarm[
            ~pd.isnull(self.windfarm.substation_id)
        ].copy()
        turbines.rename(
            columns={
                "latitude": "turbine_latitude",
                "longitude": "turbine_longitude",
                "name": "turbine_name",
            },
            inplace=True,
        )

        # Merge them back together
        self.windfarm = turbines.merge(oss, on="substation_id")

        self.windfarm = self.windfarm[self.REQUIRED + self.OPTIONAL]

        self.windfarm.string = self.windfarm.string.astype(int)
        self.windfarm.order = self.windfarm.order.astype(int)
        self.windfarm.sort_values(
            by=["substation_id", "string", "order"], inplace=True
        )

    def _initialize_custom_data(self):
        windfarm = self.config["array_system_design"]["location_data"]

        self.windfarm = extract_library_specs(
            "cables", windfarm, file_type="csv"
        )

        # Make sure no data is missing
        missing = set(self.COLUMNS).difference(self.windfarm.columns)
        if missing:
            raise ValueError(
                f"The following columns must be included in the location data: {missing}"
            )

        self._format_windfarm_data()

        # Ensure there is no missing data in required columns
        missing_data_cols = [
            c for c in self.REQUIRED if pd.isnull(self.windfarm[c]).sum() > 0
        ]
        if missing_data_cols:
            raise ValueError(f"Missing data in columns: {missing_data_cols}!")

        # Ensure there is no missing data in optional columns
        missing_data_cols = [
            c
            for c in self.OPTIONAL
            if (pd.isnull(self.windfarm[c]) | self.windfarm[c] == 0).sum() > 0
        ]
        if missing_data_cols:
            message = (
                f"Missing data in columns {missing_data_cols}; "
                "all values will be calculated."
            )
            warnings.warn(message)

        # Ensure the number of turbines matches what's expected
        if self.windfarm.shape[0] != self.system.num_turbines:
            raise ValueError(
                "The provided number of turbines does not match the plant data."
            )

        n_coords = self.windfarm.groupby(
            ["turbine_latitude", "turbine_longitude"]
        ).ngroups
        duplicates = self.windfarm.shape[0] - n_coords
        if duplicates > 0:
            raise ValueError(
                f"There are {duplicates} rows with duplicate coordinates."
            )

        # Ensure the number of turbines on a string is within the limits
        longest_string = self.windfarm["order"].unique().size
        self.num_strings = self.windfarm.groupby(
            ["substation_id", "string"]
        ).ngroups
        if longest_string > self.num_turbines_full_string:
            raise ValueError(
                "Strings can't contain more than "
                f"{self.num_turbines_full_string} turbines."
            )
        else:
            self.num_turbines_full_string = longest_string
            del self.num_turbines_partial_string
            del self.num_partial_strings

    def _check_optional_input(self):
        """
        Ensures that the optionally input parameters have valid data and were
        all filled out.
        """
        if np.any(self.sections_cable_lengths == 0):
            self.sections_cable_lengths = np.zeros(
                (self.num_strings, self.num_turbines_full_string), dtype=float
            )

        if np.any(self.sections_bury_speeds == 0):
            self.sections_bury_speeds = np.zeros(
                (self.num_strings, self.num_turbines_full_string), dtype=float
            )

    def _create_windfarm_layout(self):
        """
        Creates the custom windfarm layout that includes
            `windfarm_x`: x-coordinates with a corresponding OSS in the first
                column. Shape: `n_strings` x (`num_turbines_full_string` + 1).
            `windfarm_y`: y-coordinates with a corresponding OSS in the first
                column. Shape: `n_strings` x (`num_turbines_full_string` + 1).
            `sections_cables_lenghts`: custom cable lengths provided as an
                optional column in the `location_data`. Shape: `n_strings` x `num_turbines_full_string`.
            `sections_bury_speeds`: custom cable bury speeds provided as an
                optional column in the `location_data`. Shape: `n_strings` x `num_turbines_full_string`.
        """

        self.windfarm_x = np.zeros(
            (self.num_strings, self.num_turbines_full_string + 1), dtype=float
        )
        self.windfarm_y = np.zeros(
            (self.num_strings, self.num_turbines_full_string + 1), dtype=float
        )
        self.sections_cable_lengths = np.zeros(
            (self.num_strings, self.num_turbines_full_string), dtype=float
        )
        self.sections_bury_speeds = np.zeros(
            (self.num_strings, self.num_turbines_full_string), dtype=float
        )

        i = 0
        for oss in self.windfarm.substation_id.unique():
            layout = self.windfarm[self.windfarm.substation_id == oss]
            string_id = np.sort(layout.string.unique())
            string_id += 0 if i == 0 else i
            self.windfarm_x[string_id, 0] = layout.substation_longitude.values[
                0
            ]
            self.windfarm_y[string_id, 0] = layout.substation_latitude.values[
                0
            ]

            for string in string_id:
                data = layout[layout.string == string - i]
                order = data["order"].values
                self.windfarm_x[
                    string, order + 1
                ] = data.turbine_longitude.values[order]
                self.windfarm_y[
                    string, order + 1
                ] = data.turbine_latitude.values[order]
                self.sections_cable_lengths[
                    string, order - 1
                ] = data.cable_length.values[order - 1]
                self.sections_bury_speeds[
                    string, order - 1
                ] = data.bury_speed.values[order - 1]
            i += string + 1

        self._check_optional_input()

        # Ensure any point in array without a turbine is set to None
        no_turbines = self.windfarm_x == 0
        self.windfarm_x[no_turbines] = None
        self.windfarm_y[no_turbines] = None

        self.coordinates = np.dstack((self.windfarm_x, self.windfarm_y))

        # Create the distances between each subsequent turbine in a string
        self.sections_distance = np.zeros(
            (self.num_strings, self.num_turbines_full_string), dtype=float
        )
        for i in range(self.coordinates.shape[0]):
            for j in range(self.coordinates.shape[1] - 1):
                c1 = self.coordinates[i, j][::-1]
                c2 = self.coordinates[i, j + 1][::-1]
                if np.isnan(c1).sum() > 0 or np.isnan(c2).sum() > 0:
                    d = None
                else:
                    d = distance.geodesic(c1, c2, ellipsoid="WGS-84").km
                self.sections_distance[i, j] = d
        self.sections_distance = np.round_(self.sections_distance, 10)

    def run(self):

        self._initialize_cables()
        self.create_strings()
        self._initialize_custom_data()
        self._create_windfarm_layout()
        self._create_cable_section_lengths()

    @property
    def cable_lengths_by_type_speed(self):
        """
        Creates a dictionary of tuples with section lengths and cable burying
        speeds if non-zero entries were provided, otherwise this is equal to
        `cable_lengths_by_type`.

        Returns
        -------
        dict
            A dictionary of the section lengths and burying speeds required for
            each type of cable to fully connect the array cabling system.
            E.g.: {`Cable.name`: [(float, float)]}
        """

        if self.sections_bury_speeds.sum() == 0:
            return self.cable_lengths_by_type
        cables = {
            name: list(
                zip(
                    self.sections_cable_lengths[self.sections_cables == name],
                    self.sections_bury_speeds[self.sections_cables == name],
                )
            )
            for name in self.cables
        }
        return cables
