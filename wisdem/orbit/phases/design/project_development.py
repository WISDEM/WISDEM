"""Provides the `ProjectDevelopment` class."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from .design_phase import DesignPhase


class ProjectDevelopment(DesignPhase):
    """Project Development Class."""

    expected_config = {
        "project_development": {
            "site_auction_duration": "h (optional)",
            "site_auction_price": "USD(optional)",
            "site_assessment_plan_duration": "h (optional)",
            "site_assessment_plan_cost": "USD (optional)",
            "site_assessment_duration": "h (optional)",
            "site_assessment_cost": "USD (optional)",
            "construction_operations_plan_duration": "h (optional)",
            "construction_operations_plan_cost": "USD (optional)",
            "boem_review_duration": "h (optional)",
            "boem_review_cost": "USD (optional)",
            "design_install_plan_duration": "h (optional)",
            "design_install_plan_cost": "USD (optional)",
        }
    }

    output_config = {}

    def __init__(self, config, **kwargs):
        """
        Creates an instance of ProjectDevelopment.

        Parameters
        ----------
        config : dict
        """

        config = self.initialize_library(config, **kwargs)
        self.config = self.validate_config(config)
        self._outputs = {}

    def run(self):
        """
        Main run function. Passes ``self.config['project_development']`` to the
        following methods:

        - :py:meth:`.site_auction`
        - :py:meth:`.site_assessment_plan_development`
        - :py:meth:`.site_assessment`
        - :py:meth:`.construction_operations_plan_development`
        - :py:meth:`.boem_review`
        - :py:meth:`.design_install_plan_development`
        """

        dev_specs = self.config.get("project_development", {})

        self.site_auction(**dev_specs)
        self.site_assessment_plan_development(**dev_specs)
        self.site_assessment(**dev_specs)
        self.construction_operations_plan_development(**dev_specs)
        self.boem_review(**dev_specs)
        self.design_install_plan_development(**dev_specs)

    @property
    def design_result(self):
        """
        Returns design results for ProjectDevelopment. This method currently
        returns an empty dictionary as ProjectDevelopment does not produce any
        additional config keys.
        """

        return {}

    @property
    def total_phase_cost(self):
        """Returns total phase cost in $USD."""

        phase_cost = sum([v["cost"] for v in self._outputs.values()])
        return phase_cost

    @property
    def total_phase_time(self):
        """Returns total phase time in hours."""

        phase_time = sum([v["duration"] for v in self._outputs.values()])
        return phase_time

    @property
    def detailed_output(self):
        """Returns detailed phase information."""

        if not self._outputs:
            raise Exception("Has ProjectDevelopment been ran yet?")

        return self._outputs

    def site_auction(self, **development_specs):
        """
        Cost and duration associated with lease area auction.

        Parameters
        ----------
        development_specs : dict
            Dicitonary containing development specifications. Keys:

            - ``site_auction_duration`` - Auction duration in hours.
            - ``site_auction_cost`` - Auction price.
        """

        t = "site_auction_duration"
        c = "site_auction_cost"
        self._outputs["site_auction"] = {
            "duration": development_specs.get(t, 0),
            "cost": development_specs.get(c, 100e6),
        }

    def site_assessment_plan_development(self, **development_specs):
        """
        Cost and duration associated with developing a site assessment plan.

        Parameters
        ----------
        development_specs : dict
            Dicitonary containing development specifications. Keys:

            - ``site_assessment_plan_duration`` - Site assesment plan
              development duration in hours.
            - ``site_assessment_plan_cost`` - Site assessment plan development
              cost.
        """

        t = "site_assessment_plan_duration"
        c = "site_assessment_plan_cost"
        self._outputs["site_assessment_plan"] = {
            "duration": development_specs.get(t, 8760),
            "cost": development_specs.get(c, 0.5e6),
        }

    def site_assessment(self, **development_specs):
        """
        Cost and duration for conducting site assessments/surveys and
        obtaining permits.

        Parameters
        ----------
        development_specs : dict
            Dicitonary containing development specifications. Keys:

            - ``site_assessment_duration`` - Site assesment duration in hours.
            - ``site_assessment_cost`` - Site assessment costs.
        """

        t = "site_assessment_duration"
        c = "site_assessment_cost"
        self._outputs["site_assessment"] = {
            "duration": development_specs.get(t, 43800),
            "cost": development_specs.get(c, 50e6),
        }

    def construction_operations_plan_development(self, **development_specs):
        """
        Cost and duration for developing the Construction and Operations Plan
        (COP). Typically occurs in parallel to Site Assessment.

        Parameters
        ----------
        development_specs : dict
            Dicitonary containing development specifications. Keys:

            - ``construction_operations_plan_duration`` - Construction and
              operations plan development duration in hours.
            - ``construction_operations_plan_cost`` - Construction and
              operations plan development cost.
        """

        t = "construction_operations_plan_duration"
        c = "construction_operations_plan_cost"
        self._outputs["construction_operations_plan"] = {
            "duration": development_specs.get(t, 43800),
            "cost": development_specs.get(c, 1e6),
        }

    def boem_review(self, **development_specs):
        """
        Cost and duration for BOEM to review the Construction and Operations
        Plan.

        Parameters
        ----------
        development_specs : dict
            Dicitonary containing development specifications. Keys:

            - ``boem_review_duration`` - BOEM review duration in hours.
            - ``boem_review_cost`` - BOEM review cost. Typically not a cost to
              developers.
        """

        t = "boem_review_duration"
        c = "boem_review_cost"
        self._outputs["boem_review"] = {
            "duration": development_specs.get(t, 8760),
            "cost": development_specs.get(c, 0),
        }

    def design_install_plan_development(self, **development_specs):
        """
        Cost and duration for developing the Design and Installation Plan.
        Typically occurs in parallel with BOEM Review.

        Parameters
        ----------
        development_specs : dict
            Dicitonary containing development specifications. Keys:

            - ``design_install_plan_duration`` - Design and installation plan
              development duration in hours.
            - ``design_install_plan_cost`` - Design and installation plan
              development cost.
        """

        t = "design_install_plan_duration"
        c = "design_install_plan_cost"
        self._outputs["design_install_plan"] = {
            "duration": development_specs.get(t, 8760),
            "cost": development_specs.get(c, 0.25e6),
        }

    @property
    def design_result(self):
        """
        Returns design results for ProjectDevelopment. This method currently
        returns an empty dictionary as ProjectDevelopment does not produce any
        additional config keys.
        """

        return {}
