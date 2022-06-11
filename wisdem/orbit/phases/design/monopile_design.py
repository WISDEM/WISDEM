"""Provides the `MonopileDesign` class."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from math import pi, log

from scipy.optimize import fsolve

from wisdem.orbit.core.defaults import common_costs
from wisdem.orbit.phases.design import DesignPhase


class MonopileDesign(DesignPhase):
    """Monopile Design Class."""

    expected_config = {
        "site": {"depth": "m", "mean_windspeed": "m/s"},
        "plant": {"num_turbines": "int"},
        "turbine": {
            "rotor_diameter": "m",
            "hub_height": "m",
            "rated_windspeed": "m/s",
        },
        "monopile_design": {
            "yield_stress": "Pa (optional)",
            "load_factor": "float (optional)",
            "material_factor": "float (optional)",
            "monopile_density": "kg/m3 (optional)",
            "monopile_modulus": "Pa (optional)",
            "monopile_tp_connection_thickness": "m (optional)",
            "transition_piece_density": "kg/m3 (optional)",
            "transition_piece_thickness": "m (optional)",
            "transition_piece_length": "m (optional)",
            "soil_coefficient": "N/m3 (optional)",
            "air_density": "kg/m3 (optional)",
            "weibull_scale_factor": "float (optional)",
            "weibull_shape_factor": "float (optional)",
            "turb_length_scale": "m (optional)",
            "monopile_steel_cost": "USD/t (optional)",
            "tp_steel_cost": "USD/t (optional)",
        },
    }

    output_config = {
        "monopile": {
            "diameter": "m",
            "thickness": "m",
            "moment": "m4",
            "embedment_length": "m",
            "length": "m",
            "mass": "t",
            "deck_space": "m2",
            "unit_cost": "USD",
        },
        "transition_piece": {
            "length": "m",
            "mass": "t",
            "deck_space": "m2",
            "unit_cost": "USD",
        },
    }

    def __init__(self, config, **kwargs):
        """
        Creates an instance of MonopileDesign.

        Parameters
        ----------
        config : dict
        """

        config = self.initialize_library(config, **kwargs)
        self.config = self.validate_config(config)
        self._outputs = {}

    def run(self):
        """
        Main run function. Passes required config parameters to
        :py:meth:`.design_monopile`.
        """

        _kwargs = self.config.get("monopile_design", {})
        self._outputs["monopile"] = self.design_monopile(
            mean_windspeed=self.config["site"]["mean_windspeed"],
            site_depth=self.config["site"]["depth"],
            rotor_diameter=self.config["turbine"]["rotor_diameter"],
            hub_height=self.config["turbine"]["hub_height"],
            rated_windspeed=self.config["turbine"]["rated_windspeed"],
            **_kwargs,
        )

        self._outputs["transition_piece"] = self.design_transition_piece(
            self.monopile_sizing["diameter"],
            self.monopile_sizing["thickness"],
            **_kwargs,
        )

    def design_monopile(
        self,
        mean_windspeed,
        site_depth,
        rotor_diameter,
        hub_height,
        rated_windspeed,
        **kwargs,
    ):
        """
        Solves system of equations for the required pile diameter to satisfy
        the 50 year extreme operating gust moment. Using the result from the
        diameter equation, calculates the wall thickness and the required
        embedment length and other important sizing parameters.

        Parameters
        ----------
        mean_windspeed : int | float
            Mean wind speed at site (m/s).
        site_depth : int | float
            Water depth at site (m).
        rotor_diameter : int | float
            Rotor diameter (m).
        hub_height : int | float
            Hub height above mean sea level (m).
        rated_windspeed : int | float
            Rated windspeed of turbine (m/s).

        Returns
        -------
        monopile : dict
            Dictionary of monopile sizing and costs.

            - ``diameter`` - Pile diameter (m)
            - ``thickness`` - Pile wall thickness (m)
            - ``moment`` - Pile bending moment of inertia (m4)
            - ``embedment_length`` - Pile embedment length (m)
            - ``length`` - Total pile length (m)
            - ``mass`` - Pile mass (t)
            - ``type`` - `'Monopile'`

        References
        ----------
        This class was adapted from [#arany2017]_.

        .. [#arany2017] Laszlo Arany, S. Bhattacharya, John Macdonald,
           S.J. Hogan, Design of monopiles for offshore wind turbines in 10
           steps, Soil Dynamics and Earthquake Engineering,
           Volume 92, 2017, Pages 126-152, ISSN 0267-7261,
        """

        yield_stress = kwargs.get("yield_stress", 355000000)  # PA
        material_factor = kwargs.get("material_factor", 1.1)
        M_50y = self.calculate_50year_wind_moment(
            mean_windspeed=mean_windspeed,
            site_depth=site_depth,
            rotor_diameter=rotor_diameter,
            hub_height=hub_height,
            rated_windspeed=rated_windspeed,
            **kwargs,
        )

        data = (yield_stress, material_factor, M_50y)
        monopile = {}

        # Monopile sizing
        monopile["diameter"] = fsolve(self.pile_diam_equation, 10, args=data)[0]
        monopile["thickness"] = self.pile_thickness(monopile["diameter"])
        monopile["moment"] = self.pile_moment(monopile["diameter"], monopile["thickness"])
        monopile["embedment_length"] = self.pile_embedment_length(monopile["moment"], **kwargs)

        # Total length
        airgap = kwargs.get("airgap", 10)  # m
        monopile["length"] = monopile["embedment_length"] + site_depth + airgap
        monopile["mass"] = self.pile_mass(
            Dp=monopile["diameter"],
            tp=monopile["thickness"],
            Lt=monopile["length"],
            **kwargs,
        )

        # Deck space
        monopile["deck_space"] = monopile["diameter"] ** 2

        # Costs
        monopile["unit_cost"] = monopile["mass"] * self.monopile_steel_cost

        self.monopile_sizing = monopile
        return monopile

    def design_transition_piece(self, D_p, t_p, **kwargs):
        """
        Designs transition piece given the results of the monopile design.

        Based on Arany 2016, sections 2.2.7 - 2.2.8.

        Parameters
        ----------
        monopile_diameter : int | float
            Diameter of the designed monopile.

        Returns
        -------
        tp_design : dict
            Transition piece design parameters.
        """

        # Defaults to a bolted connection
        t_c = kwargs.get("monopile_tp_connection_thickness", 0.0)

        dens_tp = kwargs.get("transition_piece_density", 7860)  # kg/m3
        t_tp = kwargs.get("transition_piece_thickness", t_p)
        L_tp = kwargs.get("transition_piece_length", 25)
        D_tp = D_p + 2 * (t_c + t_tp)  # Arany 2016, Section 2.2.7

        # Arany 2016, Section 2.2.8
        m_tp = (dens_tp * (D_p + 2 * t_c + t_tp) * pi * t_tp * L_tp) / 907.185  # t

        tp_design = {
            "thickness": t_tp,
            "diameter": D_tp,
            "mass": m_tp,
            "length": L_tp,
            "deck_space": D_tp**2,
            "unit_cost": m_tp * self.tp_steel_cost,
        }

        return tp_design

    @property
    def design_result(self):
        """Returns the results of :py:meth:`.design_monopile`."""

        if not self._outputs:
            raise Exception("Has MonopileDesign been ran yet?")

        return self._outputs

    @property
    def total_cost(self):
        """Returns total cost of the substructures and transition pieces."""

        return sum([v for _, v in self.material_cost.items()])

    @property
    def detailed_output(self):
        """Returns detailed phase information."""

        _outputs = {
            "total_monopile_mass": self.total_monopile_mass,
            "total_monopile_cost": self.material_cost["monopile"],
            "total_transition_piece_mass": self.total_tp_mass,
            "total_transition_piece_cost": self.material_cost["transition_piece"],
        }

        return _outputs

    @property
    def material_cost(self):
        """Returns the material cost of the monopile and transition piece."""

        if not self._outputs:
            raise Exception("Has MonopileDesign been ran yet?")

        out = {
            "monopile": self.total_monopile_mass * self.monopile_steel_cost,
            "transition_piece": self.total_tp_mass * self.tp_steel_cost,
        }

        return out

    @property
    def total_monopile_mass(self):
        """Returns total mass of all monopiles."""

        if not self._outputs:
            raise Exception("Has MonopileDesign been ran yet?")

        num_turbines = self.config["plant"]["num_turbines"]

        return self._outputs["monopile"]["mass"] * num_turbines

    @property
    def total_tp_mass(self):
        """Returns total mass of all transition pieces."""

        if not self._outputs:
            raise Exception("Has MonopileDesign been ran yet?")

        num_turbines = self.config["plant"]["num_turbines"]

        return self._outputs["transition_piece"]["mass"] * num_turbines

    @property
    def monopile_steel_cost(self):
        """Returns the cost of monopile steel (USD/t) fully fabricated."""

        _design = self.config.get("monopile_design", {})
        _key = "monopile_steel_cost"

        try:
            cost = _design.get(_key, common_costs[_key])

        except KeyError:
            raise Exception("Cost of monopile steel not found.")

        return cost

    @property
    def tp_steel_cost(self):
        """
        Returns the cost of transition piece steel (USD/t) fully fabricated.
        """

        _design = self.config.get("monopile_design", {})
        _key = "tp_steel_cost"

        try:
            cost = _design.get(_key, common_costs[_key])

        except KeyError:
            raise Exception("Cost of transition piece steel not found.")

        return cost

    @staticmethod
    def pile_mass(Dp, tp, Lt, **kwargs):
        """
        Calculates the total monopile mass in tonnes.

        Parameters
        ----------
        Dp : int | float
            Pile diameter (m).
        tp : int | float
            Pile wall thickness (m).
        Lt : int | float
            Total pile length (m).

        Returns
        -------
        mt : float
            Total pile mass (t).
        """

        density = kwargs.get("monopile_density", 7860)  # kg/m3
        volume = (pi / 4) * (Dp**2 - (Dp - tp) ** 2) * Lt
        mass = density * volume / 907.185

        return mass

    @staticmethod
    def pile_embedment_length(Ip, **kwargs):
        """
        Calculates required pile embedment length.
        Source: Arany & Bhattacharya (2016)
        - Equation 7 (Enforces a rigid/lower aspect ratio monopile)

        Parameters
        ----------
        Ip : int | float
            Pile moment of inertia (m4)

        Returns
        -------
        Lp : float
            Required pile embedment length (m).
        """

        monopile_modulus = kwargs.get("monopile_modulus", 200e9)  # Pa
        soil_coefficient = kwargs.get("soil_coefficient", 4000000)  # N/m3

        Lp = 2 * ((monopile_modulus * Ip) / soil_coefficient) ** 0.2

        return Lp

    @staticmethod
    def pile_thickness(Dp):
        """
        Calculates pile wall thickness.
        Source: Arany & Bhattacharya (2016)
        - Equation 1

        Parameters
        ----------
        Dp : int | float
            Pile diameter (m).

        Returns
        -------
        tp : float
            Pile Wall Thickness (m)
        """

        tp = 0.00635 + Dp / 100

        return tp

    @staticmethod
    def pile_moment(Dp, tp):
        """
        Equation to calculate the pile bending moment of inertia.

        Parameters
        ----------
        Dp : int | float
            Pile diameter (m).
        tp : int | float
            Pile wall thickness (m).

        Returns
        -------
        Ip : float
            Pile bending moment of inertia
        """

        Ip = 0.125 * ((Dp - tp) ** 3) * tp * pi

        return Ip

    @staticmethod
    def pile_diam_equation(Dp, *data):
        """
        Equation to be solved for Pile Diameter. Combination of equations 99 &
        101 in this paper:
        Source: Arany & Bhattacharya (2016)
        - Equations 99 & 101

        Parameters
        ----------
        Dp : int | float
            Pile diameter (m).

        Returns
        -------
        res : float
            Reduced equation result.
        """

        yield_stress, material_factor, M_50y = data
        A = (yield_stress * pi) / (4 * material_factor * M_50y)
        res = A * ((0.99 * Dp - 0.00635) ** 3) * (0.00635 + 0.01 * Dp) - Dp

        return res

    def calculate_50year_wind_moment(
        self,
        mean_windspeed,
        site_depth,
        rotor_diameter,
        hub_height,
        rated_windspeed,
        **kwargs,
    ):
        """
        Calculates the 50 year extreme wind moment using methodology from
        DNV-GL. Source: Arany & Bhattacharya (2016)
        - Equation 30

        Parameters
        ----------
        mean_windspeed : int | float
            Mean wind speed at site (m/s).
        site_depth : int | float
            Water depth at site (m).
        rotor_diameter : int | float
            Rotor diameter (m).
        hub_height : int | float
            Hub height above mean sea level (m).
        rated_windspeed : int | float
            Rated windspeed of turbine (m/s).
        load_factor : float
            Added safety factor on the extreme wind moment.
            Default: 3.375 (2.5x DNV standard as this model does not design for buckling or fatigue)

        Returns
        -------
        M_50y : float
            50 year extreme wind moment (N-m).
        """

        load_factor = kwargs.get("load_factor", 3.375)

        F_50y = self.calculate_50year_wind_load(
            mean_windspeed=mean_windspeed,
            rotor_diameter=rotor_diameter,
            rated_windspeed=rated_windspeed,
            **kwargs,
        )

        M_50y = F_50y * (site_depth + hub_height)

        return M_50y * load_factor

    def calculate_50year_wind_load(self, mean_windspeed, rotor_diameter, rated_windspeed, **kwargs):
        """
        Calculates the 50 year extreme wind load using methodology from DNV-GL.
        Source: Arany & Bhattacharya (2016)
        - Equation 29

        Parameters
        ----------
        mean_windspeed : int | float
            Mean wind speed at site (m/s).
        rotor_diam : int | float
            Rotor diameter (m).
        rated_windspeed : int | float
            Rated windspeed of turbine (m/s).

        Returns
        -------
        F_50y : float
            50 year extreme wind load (N).
        """

        dens = kwargs.get("air_density", 1.225)
        swept_area = pi * (rotor_diameter / 2) ** 2

        ct = self.calculate_thrust_coefficient(rated_windspeed=rated_windspeed)

        U_eog = self.calculate_50year_extreme_gust(
            mean_windspeed=mean_windspeed,
            rated_windspeed=rated_windspeed,
            rotor_diameter=rotor_diameter,
            **kwargs,
        )

        F_50y = 0.5 * dens * swept_area * ct * (rated_windspeed + U_eog) ** 2

        return F_50y

    @staticmethod
    def calculate_thrust_coefficient(rated_windspeed):
        """
        Calculates the thrust coefficient using rated windspeed.
        Source: Frohboese & Schmuck (2010)

        Parameters
        ----------
        rated_windspeed : int | float
            Rated windspeed of turbine (m/s).

        Returns
        -------
        ct : float
            Coefficient of thrust.
        """

        ct = min([3.5 * (2 * rated_windspeed + 3.5) / (rated_windspeed**2), 1])

        return ct

    @staticmethod
    def calculate_50year_extreme_ws(mean_windspeed, **kwargs):
        """
        Calculates the 50 year extreme wind speed using methodology from DNV-GL.
        Source: Arany & Bhattacharya (2016)
        - Equation 27

        Parameters
        ----------
        mean_windspeed : int | float
            Mean wind speed (m/s).
        shape_factor : int | float
            Shape factor of the Weibull distribution.

        Returns
        -------
        U_50y : float
            50 year extreme wind speed (m/s).
        """

        scale_factor = kwargs.get("weibull_scale_factor", mean_windspeed)
        shape_factor = kwargs.get("weibull_shape_factor", 2)
        U_50y = scale_factor * (-log(1 - 0.98 ** (1 / 52596))) ** (1 / shape_factor)

        return U_50y

    def calculate_50year_extreme_gust(self, mean_windspeed, rotor_diameter, rated_windspeed, **kwargs):
        """
        Calculates the 50 year extreme wind gust using methodology from DNV-GL.
        Source: Arany & Bhattacharya (2016)
        - Equation 28

        Parameters
        ----------
        mean_windspeed : int | float
            Mean wind speed at site (m/s).
        rotor_diameter : int | float
            Rotor diameter (m).
        rated_windspeed : int | float
            Rated windspeed of turbine (m/s).
        turb_length_scale : int | float
            Turbulence integral length scale (m).

        Returns
        -------
        U_eog : float
            Extreme operating gust speed (m/s).
        """

        length_scale = kwargs.get("turb_length_scale", 340.2)

        U_50y = self.calculate_50year_extreme_ws(mean_windspeed, **kwargs)
        U_1y = 0.8 * U_50y

        U_eog = min(
            [
                (1.35 * (U_1y - rated_windspeed)),
                (3.3 * 0.11 * U_1y) / (1 + (0.1 * rotor_diameter) / (length_scale / 8)),
            ]
        )

        return U_eog
