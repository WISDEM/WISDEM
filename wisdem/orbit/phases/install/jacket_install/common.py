__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2021, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from marmot import false, process
from wisdem.orbit.core import Cargo
from wisdem.orbit.core.defaults import process_times as pt


class Jacket(Cargo):
    """Jacket Cargo"""

    def __init__(self, height=None, mass=None, deck_space=None, **kwargs):
        """Creats an instance of `Jacket`."""

        self.height = height
        self.mass = mass
        self.deck_space = deck_space
        self.num_legs = kwargs.get("num_legs", 4)

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
def install_jacket(vessel, jacket, **kwargs):
    """
    Process logic for installing a jacket at site.

    Parameters
    ----------
    env : Environment
    vessel : Vessel
    jacket : dict
    """

    reequip_time = vessel.crane.reequip(**kwargs)
    # TODO:

    pile_time = kwargs.get("drive_piles_time", 6)
    for i in range(jacket.num_legs):
        yield vessel.task_wrapper(
            "Install Pile",
            6,
            constraints={**vessel.operational_limits},
            **kwargs,
        )

        yield vessel.task_wrapper(
            "Drive Pile",
            pile_time,
            constraints={**vessel.operational_limits, "night": false()},
            suspendable=True,
            **kwargs,
        )

        if i < (jacket.num_legs - 1):
            yield vessel.task_wrapper(
                "Move Between Piles",
                4,
                constraints=vessel.transit_limits,
                suspendable=True,
                **kwargs,
            )

    yield vessel.task_wrapper("Lift Jacket", 4, constraints=vessel.operational_limits, **kwargs)

    yield vessel.task_wrapper(
        "Lower and Position Jacket",
        8,
        constraints=vessel.operational_limits,
        **kwargs,
    )

    yield vessel.task_wrapper("Grout Jacket", 8, constraints=vessel.transit_limits, **kwargs)
