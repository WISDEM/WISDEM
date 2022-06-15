__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2021, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from marmot import false, process

from wisdem.orbit.core import Cargo
from wisdem.orbit.core.defaults import process_times as pt


class Jacket(Cargo):
    """Jacket Cargo"""

    def __init__(
        self,
        height=None,
        mass=None,
        deck_space=None,
        foundation_type="piles",
        **kwargs,
    ):
        """Creats an instance of `Jacket`."""

        self.height = height
        self.mass = mass
        self.deck_space = deck_space
        self.num_legs = kwargs.get("num_legs", 4)
        self.foundation_type = foundation_type

    @staticmethod
    def fasten(**kwargs):
        """Returns time requred to fasten a jacket at port."""

        key = "jacket_fasten_time"
        time = kwargs.get(key, pt[key])

        return "Fasten Jacket", time

    @staticmethod
    def release(**kwargs):
        """Returns time required to release jacket from fastenings."""

        key = "jacket_release_time"
        time = kwargs.get(key, pt[key])

        return "Release Jacket", time


@process
def install_piles(vessel, jacket, **kwargs):
    """
    Process logic for installing piles at site.

    Parameters
    ----------
    vessel : Vessel
    jacket : dict
    """

    reposition_time = kwargs.get("jacket_vessel_reposition", pt["jacket_vessel_reposition"])
    position_pile_time = kwargs.get("jacket_position_pile", pt["jacket_position_pile"])
    drive_time = kwargs.get("jacket_pile_drive_time", pt["jacket_pile_drive_time"])
    pin_template_time = kwargs.get("jacket_pin_template_time", pt["jacket_pin_template_time"])

    yield vessel.task_wrapper(
        "Lay Pin Template",
        pin_template_time,
        constraints=vessel.operational_limits,
        **kwargs,
    )

    for i in range(jacket.num_legs):
        yield vessel.task_wrapper(
            "Position Pile",
            position_pile_time,
            constraints={**vessel.operational_limits},
            **kwargs,
        )

        yield vessel.task_wrapper(
            "Drive Pile",
            drive_time,
            constraints={**vessel.operational_limits, "night": false()},
            suspendable=True,
            **kwargs,
        )

        if i < (jacket.num_legs - 1):
            yield vessel.task_wrapper(
                "Move to Next Leg",
                reposition_time,
                constraints=vessel.transit_limits,
                suspendable=True,
                **kwargs,
            )


@process
def install_suction_buckets(vessel, jacket, **kwargs):
    """
    Process logic for installing suction buckets at site.

    Parameters
    ----------
    vessel : Vessel
    jacket : dict
    """

    reposition_time = kwargs.get("jacket_vessel_reposition", pt["jacket_vessel_reposition"])
    install_time = kwargs.get("jacket_suction_bucket", pt["jacket_suction_bucket"])

    for i in range(jacket.num_legs):
        yield vessel.task_wrapper(
            "Install Suction Bucket",
            install_time,
            constraints={**vessel.operational_limits},
            **kwargs,
        )

        if i < (jacket.num_legs - 1):
            yield vessel.task_wrapper(
                "Move to Next Leg",
                reposition_time,
                constraints=vessel.transit_limits,
                suspendable=True,
                **kwargs,
            )


@process
def install_jacket(vessel, jacket, **kwargs):
    """
    Process logic for installing a jacket at site.

    Parameters
    ----------
    vessel : Vessel
    jacket : dict
    """

    if jacket.foundation_type == "piles":
        yield install_piles(vessel, jacket, **kwargs)

    elif jacket.foundation_type == "suction":
        yield install_suction_buckets(vessel, jacket, **kwargs)

    else:
        return ValueError("Input 'jacket.foundation_type' must be 'piles' or 'suction'.")

    lift_time = kwargs.get("jacket_lift_time", pt["jacket_lift_time"])
    yield vessel.task_wrapper(
        "Lift Jacket",
        lift_time,
        constraints=vessel.operational_limits,
        **kwargs,
    )

    lower_time = kwargs.get("jacket_lower_time", pt["jacket_lower_time"])
    yield vessel.task_wrapper(
        "Lower and Position Jacket",
        lower_time,
        constraints=vessel.operational_limits,
        **kwargs,
    )

    grout_time = kwargs.get("jacket_grout_time", pt["jacket_grout_time"])
    yield vessel.task_wrapper("Grout Jacket", grout_time, constraints=vessel.transit_limits, **kwargs)
