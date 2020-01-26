"""
Provides the base class and simulation logic for array and export cable
installation simulations.
"""

__author__ = "Rob Hammond"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Rob Hammond"
__email__ = "robert.hammond@nrel.gov"


from numpy import isclose

from wisdem.orbit.vessels import tasks
from wisdem.orbit.simulation.logic import get_list_of_items_from_port
from wisdem.orbit.simulation.exceptions import InsufficientAmount

# Trech digging pre-installation task


def dig_trench(env, vessel, distance, **kwargs):
    """
    Subprocess to dig the trench for the export cable between landfall
    and the onshore substation.

    Parameters
    ----------
    distance : int or float
        Distance between landfall and onshore substation.
    """

    kwargs = {**vessel._transport_specs, **kwargs}
    dig_time = tasks.dig_trench(distance, **kwargs)

    task = {
        "agent": "Trench Dig Vessel",
        "action": "DigTrench",
        "duration": dig_time,
        "location": "Onshore",
        "type": "Operations",
    }

    yield env.process(env.task_handler(task))


# Basic cable laying processes


def transport(env, vessel, distance, to_port, to_site, **kwargs):
    """
    Subprocess to travel between port and site.

    Parameters
    ----------
    env : simpy.Environment
        Simulation environment.
    vessel : Vessel
        Cable laying vessel.
    distance : int or float
        Distance between port and site.
    to_port : bool
        Indicator for travelling to port (True) or to site (False).
    """

    kwargs = {**vessel._transport_specs, **kwargs}
    transit_time = vessel.transit_time(distance)

    task = {
        "agent": vessel.name,
        "action": "Transit",
        "duration": transit_time,
        "location": "At Sea",
        "type": "Operations",
        **vessel.transit_limits,
    }

    if to_port and not to_site:
        vessel.at_site = False
        vessel.storage.deck_space -= 1
    elif to_site and not to_port:
        vessel.at_port = False

    yield env.process(env.task_handler(task))

    if to_port and not to_site:
        vessel.at_port = True
    elif to_site and not to_port:
        vessel.at_site = True


def get_carousel_from_port(env, vessel, port, **kwargs):
    """
    Logic required to load a carousel onto the cable laying vessel at port.

    Parameters
    ----------
    env : simpy.Environment
        Simulation environment.
    vessel : Vessel
        Cable laying vessel.
    port : Port
        Port object.
    """

    component_list = [("type", "Carousel")]
    yield env.process(
        get_list_of_items_from_port(
            env, vessel, component_list, port, **kwargs
        )
    )


def position_onsite(env, vessel, **kwargs):
    """
    Processs to position cable laying vessel at turbine substation.

    Parameters
    ----------
    env : simpy.Environment
        Simulation environment.
    vessel : Vessel
        Cable laying vessel.
    """

    kwargs = {**vessel._transport_specs, **kwargs}
    position_time = tasks.position_onsite(**kwargs)
    task = {
        "agent": vessel.name,
        "action": "PositionOnsite",
        "duration": position_time,
        "location": "Site",
        "type": "Operations",
        **vessel.transit_limits,
    }
    yield env.process(env.task_handler(task))


def prep_cable(env, vessel, **kwargs):
    """
    Processs to prepare the cable for laying and burial at site.

    Parameters
    ----------
    env : simpy.Environment
        Simulation environment.
    vessel : Vessel
        Cable laying vessel.
    """

    prep_time = tasks.prep_cable(**kwargs)
    task = {
        "agent": vessel.name,
        "action": "PrepCable",
        "duration": prep_time,
        "location": "Site",
        "type": "Operations",
        **vessel.transit_limits,
    }
    yield env.process(env.task_handler(task))


def lower_cable(env, vessel, **kwargs):
    """
    Process to lower the cable to the seafloor at site.

    Parameters
    ----------
    env : simpy.Environment
        Simulation environment.
    vessel : Vessel
        Cable laying vessel.
    """

    kwargs = {**vessel._transport_specs, **kwargs}
    lower_time = tasks.lower_cable(**kwargs)
    task = {
        "agent": vessel.name,
        "action": "LowerCable",
        "duration": lower_time,
        "location": "Site",
        "type": "Operations",
        **vessel.transit_limits,
    }
    yield env.process(env.task_handler(task))


def pull_in_cable(env, vessel, **kwargs):
    """
    Process to pull cable into substation at site.

    Parameters
    ----------
    env : simpy.Environment
        Simulation environment.
    vessel : Vessel
        Cable laying vessel.
    """

    kwargs = {**vessel._transport_specs, **kwargs}
    pull_time = tasks.pull_in_cable(**kwargs)
    task = {
        "agent": vessel.name,
        "action": "PullInCable",
        "duration": pull_time,
        "location": "Site",
        "type": "Operations",
        **vessel.transit_limits,
    }
    yield env.process(env.task_handler(task))


def test_cable(env, vessel, **kwargs):
    """
    Process to test cable at substation to ensure it works.

    Parameters
    ----------
    env : simpy.Environment
        Simulation environment.
    vessel : Vessel
        Cable laying vessel.
    """

    kwargs = {**vessel._transport_specs, **kwargs}
    test_time = tasks.test_cable(**kwargs)

    task = {
        "agent": vessel.name,
        "action": "TestCable",
        "duration": test_time,
        "location": "Site",
        "type": "Operations",
        **vessel.transit_limits,
    }

    yield env.process(env.task_handler(task))


def lay_bury_cable_section(
    env, vessel, cable_len_km, cable_mass_tonnes, **kwargs
):
    """
    Process to lay (and bury) cable between substructures.

    Parameters
    ----------
    env : simpy.Environment
        Simulation environment.
    vessel : Vessel
        Cable laying vessel.
    cable_len_km : int or float
        Length of cable to be laid.
    simultaneous_lay_bury : bool
        Indicator for whether or not laying and burrying of cable happen
        simultaneously or separately.
    """

    kwargs = {**vessel._transport_specs, **kwargs}
    lay_bury_time = tasks.lay_bury_cable(cable_len_km, **kwargs)
    task = {
        "agent": vessel.name,
        "action": "LayBuryCable",
        "duration": lay_bury_time,
        "location": "Site",
        "type": "Operations",
        **vessel.transit_limits,
    }

    # Check that we are not just getting a float error that is some trivial
    # difference as opposed to a legitimate error
    try:
        _ = vessel.storage.get_item("cable", cable_mass_tonnes)
    except InsufficientAmount:
        try:
            _ = vessel.storage.get_item("cable", round(cable_mass_tonnes, 10))
        except InsufficientAmount:
            if isclose(vessel.storage.current_cargo_weight, cable_mass_tonnes):
                _ = vessel.storage.get_item(
                    "cable", vessel.storage.current_cargo_weight
                )

    yield env.process(env.task_handler(task))


def lay_cable_section(env, vessel, cable_len_km, cable_mass_tonnes, **kwargs):
    """
    Process to lay (and bury) cable between substructures.

    Parameters
    ----------
    env : simpy.Environment
        Simulation environment.
    vessel : Vessel
        Cable laying vessel.
    cable_len_km : int or float
        Length of cable to be laid.
    simultaneous_lay_bury : bool
        Indicator for whether or not laying and burrying of cable happen
        simultaneously or separately.
    """

    kwargs = {**vessel._transport_specs, **kwargs}
    lay_time = tasks.lay_cable(cable_len_km, **kwargs)
    task = {
        "agent": vessel.name,
        "action": "LayCable",
        "duration": lay_time,
        "location": "Site",
        "type": "Operations",
        **vessel.transit_limits,
    }

    # Check that we are not just getting a float error that is some trivial
    # difference as opposed to a legitimate error
    try:
        _ = vessel.storage.get_item("cable", cable_mass_tonnes)
    except InsufficientAmount:
        try:
            _ = vessel.storage.get_item("cable", round(cable_mass_tonnes, 10))
        except InsufficientAmount:
            if isclose(vessel.storage.current_cargo_weight, cable_mass_tonnes):
                _ = vessel.storage.get_item(
                    "cable", vessel.storage.current_cargo_weight
                )

    yield env.process(env.task_handler(task))


def bury_cable_section(env, vessel, cable_len_km, **kwargs):
    """
    Process to bury cable section between substructures.

    Parameters
    ----------
    env : simpy.Environment
        Simulation environment.
    vessel : Vessel
        Cable burying vessel.
    cable_len_km : int or float
        Length of cable to be laid.
    """

    kwargs = {**vessel._transport_specs, **kwargs}
    bury_time = tasks.bury_cable(cable_len_km, **kwargs)
    task = {
        "agent": vessel.name,
        "action": "BuryCable",
        "duration": bury_time,
        "location": "Site",
        "type": "Operations",
        **vessel.transit_limits,
    }
    yield env.process(env.task_handler(task))


# Cable splicing processes


def raise_cable(env, vessel, **kwargs):
    """
    Process to raise the unspliced cable end from the seafloor.

    Parameters
    ----------
    env : simpy.Environment
        Simulation environment.
    vessel : Vessel
        Cable laying vessel.
    """

    kwargs = {**vessel._transport_specs, **kwargs}
    raise_time = tasks.raise_cable(**kwargs)
    task = {
        "agent": vessel.name,
        "action": "RaiseCable",
        "duration": raise_time,
        "location": "Site",
        "type": "Operations",
        **vessel.transit_limits,
    }
    yield env.process(env.task_handler(task))


def splice_cable(env, vessel, **kwargs):
    """
    Process to splice two cable ends together.

    Parameters
    ----------
    env : simpy.Environment
        Simulation environment.
    vessel : Vessel
        Cable laying vessel.
    """

    kwargs = {**vessel._transport_specs, **kwargs}
    splice_time = tasks.splice_cable(**kwargs)

    task = {
        "agent": vessel.name,
        "action": "SpliceCable",
        "duration": splice_time,
        "location": "Site",
        "type": "Operations",
        **vessel.transit_limits,
    }

    yield env.process(env.task_handler(task))


# Export cable specific processes


def tow_plow(env, vessel, distance, **kwargs):
    """
    Process to tow the plow to the landfall site from the cable laying vessel.

    Parameters
    ----------
    env : simpy.Environment
        Simulation environment.
    vessel : Vessel
        Cable laying vessel.
    distance : int or float
        Distance between landfall and cable laying vessel. This is either
        where the vessel has beached itself or will be anchored while
        doing the onshore part of installation.
    """

    kwargs = {**vessel._transport_specs, **kwargs}
    tow_time = tasks.tow_plow(distance, **kwargs)
    if tow_time > 0:
        task = {
            "agent": vessel.name,
            "action": "TowPlow",
            "duration": tow_time,
            "location": "Landfall",
            "type": "Operations",
            **vessel.transit_limits,
        }
        yield env.process(env.task_handler(task))


def pull_in_winch(env, vessel, distance, **kwargs):
    """
    Subprocess to pull in the winch to the landfall site from the cable
    laying vessel.

    Parameters
    ----------
    env : simpy.Environment
        Simulation environment.
    vessel : Vessel
        Cable laying vessel.
    distance : int or float
        Distance between landfall and the onshare substation. This is either
        where the vessel has beached itself or will be anchored while
        doing the onshore part of installation.
    """

    kwargs = {**vessel._transport_specs, **kwargs}
    winch_time = tasks.pull_winch(distance, **kwargs)
    task = {
        "agent": vessel.name,
        "action": "PullWinch",
        "duration": winch_time,
        "location": "Landfall",
        "type": "Operations",
        **vessel.transit_limits,
    }
    yield env.process(env.task_handler(task))


# Grouped processes for simpler logic in the simulation


def connect_cable_section_to_target(env, vessel, **kwargs):
    """
    Subprocess to start or end a cable installation.

    Parameters
    ----------
    env : simpy.Environment
        Simulation environment.
    vessel : Vessel
        Cable laying vessel.
    """

    # Prep cable
    yield env.process(prep_cable(env, vessel, **kwargs))

    # Pull in cable to offshore substructure
    yield env.process(pull_in_cable(env, vessel, **kwargs))

    # Test and terminate cable at offshore substructure
    yield env.process(test_cable(env, vessel, **kwargs))


def lay_bury_full_array_cable_section(
    env, vessel, cable_len_km, cable_mass_tonnes, strategy, **kwargs
):
    """
    Subprocesses to lay, bury, and connect cable between turbines and offshore
    substation.

    Parameters
    ----------
    env : simpy.Environment
        Simulation environment.
    vessel : Vessel
        Cable laying vessel.
    cable_len_km : int or float
        Length of cable, in km.
    cable_mass_tonnes : int or float
        Mass of cable, tonnes.
    strategy : str
        One of "lay", "bury", "simultaneous" to indicate which of the two
        (or both) processes will be completed in the subprocess.
    """

    # Position at site
    yield env.process(position_onsite(env, vessel, **kwargs))

    # Connect cable to turbine
    yield env.process(connect_cable_section_to_target(env, vessel, **kwargs))

    # Lower cable
    yield env.process(lower_cable(env, vessel, **kwargs))

    if strategy == "simultaneous":
        yield env.process(
            lay_bury_cable_section(
                env, vessel, cable_len_km, cable_mass_tonnes, **kwargs
            )
        )
    elif strategy == "lay":
        yield env.process(
            lay_cable_section(
                env, vessel, cable_len_km, cable_mass_tonnes, **kwargs
            )
        )
    yield env.process(connect_cable_section_to_target(env, vessel, **kwargs))


def bury_full_cable_section(env, vessel, cable_len_km, **kwargs):
    """
    Subprocesses to bury cable between turbines and the offshore substation.

    Parameters
    ----------
    env : simpy.Environment
        Simulation environment.
    vessel : Vessel
        Cable laying vessel.
    cable_len_km : int or float
        Length of cable, in km.
    """

    # Position at site
    yield env.process(position_onsite(env, vessel, **kwargs))

    yield env.process(bury_cable_section(env, vessel, cable_len_km, **kwargs))


def splice_cable_process(env, vessel, **kwargs):
    """
    Subprocess for splicing two cable ends together.

    Parameters
    ----------
    env : simpy.Environment
        Simulation environment.
    vessel : Vessel
        Cable laying vessel.
    cable_len_km : int or float
        Length (in km) of the cable section to be installed.
    """

    # Position at splice site
    yield env.process(position_onsite(env, vessel, **kwargs))

    # Raise the cable end from the seafloor
    yield env.process(raise_cable(env, vessel, **kwargs))

    # Splice the cable ends
    yield env.process(splice_cable(env, vessel, **kwargs))

    # Lower cable to seafloor
    yield env.process(lower_cable(env, vessel, **kwargs))


def onshore_work(
    env,
    vessel,
    distance_to_beach,
    distance_to_interconnection,
    cable_mass,
    **kwargs,
):
    """
    Processes to connect the export cable between the offshore
    substation and landfall.

    Parameters
    ----------
    env : simpy.Environment
        Simulation environment.
    vessel : Vessel
        Cable laying vessel.
    distance_to_beach : int or float
        Distance from lanfall to where the vessel is located (could beach
        itself or remain at sea).
    distance_to_interconnection : int or float
        Distance between the landfall site and the onshore substation, in km.
    cable_mass : int or float
        Mass of cable length to be installed onshore, in tonnes.
    """

    distance_vessel_to_inter = distance_to_beach + distance_to_interconnection

    # Tow plow to landfall
    yield env.process(tow_plow(env, vessel, distance_to_beach, **kwargs))

    # Pull in winch wire
    yield env.process(
        pull_in_winch(env, vessel, distance_vessel_to_inter, **kwargs)
    )

    # Remove the onshore cable length from storage
    _ = vessel.storage.get_item("cable", cable_mass)

    # Connect cable at interconnection
    yield env.process(connect_cable_section_to_target(env, vessel, **kwargs))

    # Lower cable to seafloor
    yield env.process(lower_cable(env, vessel, **kwargs))


def lay_array_cables(
    env,
    cable_lay_vessel,
    port,
    num_cables,
    distance_to_site,
    strategy,
    **kwargs,
):
    """
    Simulation of the installation of array cables.
    NOTE: This does not support cable splicing scenarios.

    Parameters
    ----------
    env : Simpy.Environment
        Simulation environment.
    cable_lay_vessel : Vessel
        Cable laying vessel.
    port : Simpy.FilterStore
        Simulation port object.
    num_cables : int
        Number of cable sections to be installed.
    distance_to_site : int or float
        Distance between port and offshore wind site.
    strategy : str
        One of "lay" or "simultaneous" to indicate if the cable will be buried
        at the same time as laying it.

    Raises
    ------
    Exception
        Vessel is lost at sea if not at sea or at port.
    """

    while num_cables:
        if cable_lay_vessel.at_port:
            yield env.process(
                get_carousel_from_port(env, cable_lay_vessel, port, **kwargs)
            )
            cable_lay_vessel.update_trip_data(deck=False, items=False)
            yield env.process(
                transport(
                    env,
                    cable_lay_vessel,
                    distance_to_site,
                    False,
                    True,
                    **kwargs,
                )
            )
        elif cable_lay_vessel.at_site:
            while cable_lay_vessel.carousel.section_lengths:

                # Retrieve the cable section length and mass and install
                _len = cable_lay_vessel.carousel.section_lengths.pop(0)
                _mass = cable_lay_vessel.carousel.section_masses.pop(0)
                _speed = cable_lay_vessel.carousel.section_bury_speeds.pop(0)
                if _speed == -1:
                    kw = {**kwargs}
                else:
                    kw = {**kwargs, "cable_bury_speed": _speed}

                yield env.process(
                    lay_bury_full_array_cable_section(
                        env, cable_lay_vessel, _len, _mass, strategy, **kw
                    )
                )

                num_cables -= 1

            # Go back to port once the carousel is depleted.
            yield env.process(
                transport(
                    env,
                    cable_lay_vessel,
                    distance_to_site,
                    True,
                    False,
                    **kwargs,
                )
            )

        else:
            raise Exception("Vessel is lost at sea.")

    _strategy = "laying and burying"
    env.logger.debug(
        f"Array cable {_strategy} complete!",
        extra={
            "agent": cable_lay_vessel.name,
            "time": env.now,
            "type": "Status",
            "action": "Complete",
        },
    )


def bury_cables(
    env, cable_bury_vessel, num_cables, distance_to_site, system, **kwargs
):
    """
    Simulation of the burying of array cables.

    .. note:: This shouldn't actually acrue any port time or costs because it's
    not relying on the port for any operations other than going to/from site.

    Parameters
    ----------
    env : Simpy.Environment
        Simulation environment.
    cable_bury_vessel : Vessel
        Cable laying vessel.
    num_cables : int
        Number of cable sections to be installed.
    system : str
        One of "array" or "export".
    distance_to_site : int or float
        Distance between port and offshore wind site.

    Raises
    ------
    Exception
        Vessel is lost at sea if not at sea or at port.
    """

    while num_cables:
        if cable_bury_vessel.at_port:
            cable_bury_vessel.update_trip_data(deck=False, items=False)
            yield env.process(
                transport(
                    env,
                    cable_bury_vessel,
                    distance_to_site,
                    False,
                    True,
                    **kwargs,
                )
            )
        elif cable_bury_vessel.at_site:
            while cable_bury_vessel.carousel.section_lengths:

                # Retrieve the cable section length and mass and install
                _len = cable_bury_vessel.carousel.section_lengths.pop(0)
                _speed = cable_bury_vessel.carousel.section_bury_speeds.pop(0)
                if _speed == -1:
                    kw = {**kwargs}
                else:
                    kw = {**kwargs, "cable_bury_speed": _speed}

                yield env.process(
                    bury_full_cable_section(env, cable_bury_vessel, _len, **kw)
                )

                num_cables -= 1

            # Go back to port once the carousel is depleted.
            yield env.process(
                transport(
                    env,
                    cable_bury_vessel,
                    distance_to_site,
                    True,
                    False,
                    **kwargs,
                )
            )

        else:
            raise Exception("Vessel is lost at sea.")

    env.logger.debug(
        f"{system.title()} cable burying complete!",
        extra={
            "agent": cable_bury_vessel.name,
            "time": env.now,
            "type": "Status",
            "action": "Complete",
        },
    )


def separate_lay_bury_array(
    env,
    cable_lay_vessel,
    cable_bury_vessel,
    port,
    num_cables,
    distance_to_site,
    strategy,
    system,
    **kwargs,
):
    """Performs the laying and burying of array cables by calling both
    `lay_array_cables` and `bury_cables`.

    Parameters
    ----------
    env : Simpy.Environment
        Simpy environment where the simulation will be run.
    cable_lay_vessel : ORBIT.Vessel
        Cable laying vessel.
    cable_bury_vessel : ORBIT.Vessel
        Cable burying vessel.
    port : ORBIT.Port
        Port where carousels will be loaded onto the cable laying vessel.
    num_cables : int
        Number of cables to be laid.
    distance_to_site : int | float
        Distance between port and the windfarm.
    strategy : str
        Should be "separate". Not actually used.
    system : str
        Should be "array".

    Yields
    -------
    [type]
        [description]
    """
    strategy = "lay"
    yield env.process(
        lay_array_cables(
            env,
            cable_lay_vessel,
            port,
            num_cables,
            distance_to_site,
            strategy,
            **kwargs,
        )
    )
    yield env.process(
        bury_cables(
            env,
            cable_bury_vessel,
            num_cables,
            distance_to_site,
            system,
            **kwargs,
        )
    )


def install_trench(env, vessel, trench_length, **kwargs):
    """
    Simulation of the installation of array cables.
    NOTE: This does not support cable splicing scenarios.

    Parameters
    ----------
    env : Simpy.Environment
        Simulation environment
    trench_length : int or float
        Distance between onshore substation and landfall, km.
    """

    yield env.process(dig_trench(env, vessel, trench_length, **kwargs))


def lay_export_cables(
    env,
    cable_lay_vessel,
    trench_vessel,
    port,
    cable_length,
    num_sections,
    distances,
    strategy,
    **kwargs,
):
    """
    Simulation of the installation of export cables.

    Parameters
    ----------
    env : Simpy.Environment
        Simulation environment
    cable_lay_vessel : Vessel
        Cable laying vessel.
    trench_vessel : Vessel
        Trench digging operation.
    port : Simpy.FilterStore
        Simulation port object
    cable_length : float
        Length of a full cable section.
    num_sections : int
        Number of individual cable sections (could be spliced) to install to
        connect the wind farm to its onshore interconnection point.
    distances : SimpleNamespace
        The collection of distances required for export cable installation:
        site : int or float
            Distance to site, km.
        landfall : int or float
            Distance between from the offshore substation and landfall, km.
        beach : int or float
            Distance between where a vessel anchors offshore and the landfall
            site, km.
        interconnection : int or float
            Distance between landfall and the onshore substation, km.
    strategy : str
        One of "lay" or "simultaneous" to indicate if the export cable is being
        laid only or laid and buried simultaneously.

    Raises
    ------
    Exception
        Vessel is lost at sea if not at sea or at port.
    """

    STRATEGY_MAP = {
        "simultaneous": lay_bury_cable_section,
        "lay": lay_cable_section,
    }

    yield env.process(
        install_trench(
            env=env,
            vessel=trench_vessel,
            trench_length=distances.interconnection,
            **kwargs,
        )
    )

    splice_required = False
    new_start = True

    while num_sections:  # floats aren't exact
        if cable_lay_vessel.at_port:
            yield env.process(
                get_carousel_from_port(env, cable_lay_vessel, port, **kwargs)
            )
            cable_lay_vessel.update_trip_data(deck=False, items=False)
            yield env.process(
                transport(
                    env,
                    cable_lay_vessel,
                    distances.site,
                    False,
                    True,
                    **kwargs,
                )
            )
        elif cable_lay_vessel.at_site:
            while cable_lay_vessel.carousel.section_lengths:

                # Retrieve the cable section length and mass and install
                _len = cable_lay_vessel.carousel.section_lengths.pop(0)
                _mass = cable_lay_vessel.carousel.section_masses.pop(0)
                num_sections -= 1

                if new_start:
                    _pct_to_install = (
                        distances.beach + distances.interconnection
                    ) / _len
                    remaining_connection_len = cable_length

                    yield env.process(
                        onshore_work(
                            env,
                            cable_lay_vessel,
                            distances.beach,
                            distances.interconnection,
                            _mass * _pct_to_install,
                            **kwargs,
                        )
                    )
                    new_start = False

                    # Keep track of what's left overall
                    remaining_connection_len -= _len * _pct_to_install

                    # Keep track of what's left in the cable section
                    _len_remaining = _len * (1 - _pct_to_install)
                    _mass_remaining = _mass * (1 - _pct_to_install)

                    # If there is cable still left in the section, install it
                    if round(_len_remaining, 10) > 0:
                        yield env.process(
                            STRATEGY_MAP[strategy](
                                env,
                                cable_lay_vessel,
                                _len_remaining,
                                _mass_remaining,
                                **kwargs,
                            )
                        )
                        remaining_connection_len -= _len_remaining

                        # If an individual connection is complete, then finish
                        # the individual installation; otherwise return to port
                        # for the remaing cable
                        if round(remaining_connection_len, 10) == 0:
                            yield env.process(
                                position_onsite(
                                    env, cable_lay_vessel, **kwargs
                                )
                            )
                            yield env.process(
                                connect_cable_section_to_target(
                                    env, cable_lay_vessel, **kwargs
                                )
                            )
                            new_start = True
                        else:
                            splice_required = True
                            yield env.process(
                                transport(
                                    env,
                                    cable_lay_vessel,
                                    distances.site,
                                    True,
                                    False,
                                    **kwargs,
                                )
                            )
                    else:
                        splice_required = True
                        yield env.process(
                            transport(
                                env,
                                cable_lay_vessel,
                                distances.site,
                                True,
                                False,
                                **kwargs,
                            )
                        )

                elif splice_required:
                    yield env.process(
                        splice_cable_process(env, cable_lay_vessel, **kwargs)
                    )
                    yield env.process(
                        STRATEGY_MAP[strategy](
                            env, cable_lay_vessel, _len, _mass, **kwargs
                        )
                    )
                    remaining_connection_len -= _len

                    if round(remaining_connection_len, 10) == 0:
                        yield env.process(
                            position_onsite(env, cable_lay_vessel, **kwargs)
                        )
                        yield env.process(
                            connect_cable_section_to_target(
                                env, cable_lay_vessel, **kwargs
                            )
                        )
                        new_start = True
                        splice_required = False

                    else:
                        # The carousel has no more cable because there aren't
                        # incomplete sections unless the whole cable can't fit
                        # on the carousel.
                        splice_required = True  # enforcing that it stays True

                    # go back to port
                    yield env.process(
                        transport(
                            env,
                            cable_lay_vessel,
                            distances.site,
                            True,
                            False,
                            **kwargs,
                        )
                    )

            # go back to port
            yield env.process(
                transport(
                    env,
                    cable_lay_vessel,
                    distances.site,
                    True,
                    False,
                    **kwargs,
                )
            )

        else:
            raise Exception("Vessel is lost at sea.")

    _strategy = strategy.replace("_", " and ").replace("y", "ying")
    env.logger.debug(
        f"Export cable {_strategy} complete!",
        extra={
            "agent": cable_lay_vessel.name,
            "time": env.now,
            "type": "Status",
            "action": "Complete",
        },
    )


def separate_lay_bury_export(
    env,
    cable_lay_vessel,
    cable_bury_vessel,
    trench_vessel,
    port,
    cable_length,
    num_sections,
    distances,
    strategy,
    system,
    distance_to_site,
    num_cables,
    **kwargs,
):
    strategy = "lay"
    yield env.process(
        lay_export_cables(
            env,
            cable_lay_vessel,
            trench_vessel,
            port,
            cable_length,
            num_sections,
            distances,
            strategy,
            **kwargs,
        )
    )
    yield env.process(
        bury_cables(
            env,
            cable_bury_vessel,
            num_cables,
            distance_to_site,
            system,
            **kwargs,
        )
    )
