"""Provides the `CarouselSystem` and `Carousel` class."""

__author__ = "Rob Hammond"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Rob Hammond"
__email__ = "robert.hammond@nrel.gov"


import itertools
from dataclasses import dataclass

import numpy as np


@dataclass
class Carousel:
    """
    Data class that represents the cable information contained on a single
    carousel.
    """

    name: str
    length: float
    weight: float
    section_lengths: list
    section_masses: list
    section_bury_speeds: list
    deck_space: int


class CarouselSystem:
    """
    CarouselSystem to define the carousels required for an installation.

    Parameters
    ----------
    cables : dict
        Dictionary of cable names with their respective sections to be
        installed and linear densities.
    vessel : Vessel
        An initialized vessel with property vessel.storage

    Attributes
    ----------
    cables : dict
        Dictionary of cable names with their respective sections to be
        installed and linear densities.
    max_cargo_weight : int | float
        Maximum cargo weight or carousel weight allowed.
    carousels : dict
        A dictionary of `Carousel` objects to be loaded onto a cable
        installation vessel.

    Methods
    -------
    create_carousels
        Creates the carousel objects to install all required cable sections.
    """

    def __init__(self, cables, max_cargo_weight):
        """
        Carousel object that produces the carousels required for installation.

        Parameters
        ----------
        cables : dict
            Dictionary of cables section lengths to be installed and the linear
            denisty of each cable. This is the `output_config` from
            `ArraySystemDesign` or `ExportSystemDesign` that is created in the
            method `design_result` if not provided as custom input.
            cables = {
                "name": {
                    "linear_density": "int | float",
                    "cable_sections": [("float", "int")]
                }
            }
        max_cargo_weight : int | float
            Maximum weight allowed on a vessel/in a carousel.
        """

        self.cables = cables
        self.max_cargo_weight = max_cargo_weight

        self.carousels = {}

    def _create_section_list(self, cable_name):
        """
        Creates a list of section lengths and masses to be installed.

        Parameters
        ----------
        cable_name : str
            Name of the cable type to create the sections list.

        Returns
        -------
        lengths : np.ndarray
            Array of all cable section lengths to be installed in km.
        masses : np.ndarray
            Array of cable section masses to be installed in tonnes.
        """

        length_speed = np.array(
            list(
                itertools.chain.from_iterable(
                    [[length, *_]] * n
                    for length, *_, n in self.cables[cable_name][
                        "cable_sections"
                    ]
                )
            )
        )
        if length_speed.shape[1] == 1:
            bury_speeds = np.full(length_speed.shape[0], -1)
        else:
            bury_speeds = length_speed[:, 1]

        lengths = length_speed[:, 0]

        masses = np.round_(
            lengths * self.cables[cable_name]["linear_density"], 10
        )

        return lengths, masses, bury_speeds

    def _create_cable_carousel_with_splice(
        self,
        max_length,
        cable_index,
        section_lengths,
        section_masses,
        section_bury_speeds,
    ):
        """
        Creates a `Carousel` of spliced cables with only a single cable section
        on any individual carousel.
        """

        j = 1
        name = f"Carousel {cable_index}-{j}"
        section_lengths = section_lengths.tolist()
        section_masses = section_masses.tolist()
        section_bury_speeds = section_bury_speeds.tolist()

        while section_lengths:
            remaining_length = section_lengths.pop(0)
            remaining_mass = section_masses.pop(0)
            speed = section_bury_speeds.pop(0)

            while remaining_length:
                length = min(max_length, remaining_length)
                pct = length / remaining_length
                mass = remaining_mass * pct
                self.carousels[name] = Carousel(
                    name, length, mass, [length], [mass], [speed], 1
                )

                j += 1
                name = name = f"Carousel {cable_index}-{j}"
                remaining_length -= length
                remaining_mass -= mass

    def _create_cable_carousel_without_splice(
        self,
        max_length,
        cable_index,
        section_lengths,
        section_masses,
        section_bury_speeds,
    ):
        """
        Creates carousels of unspliced cables with only a single cable type on
        any individual carousel.

        Parameters
        ----------
        max_length : float
            Maximum length of cable that can fit on a carousel.
        cable_index : int
            1-indexed index of the cable that has a cable being created.
        section_lengths : np.ndarray
            Array of section lengths that need to be installed. Lengths
            correspond to`section_masses`.
        section_masses : np.ndarray
            Array of section masses that need to be installed. Masses
            correspond to`section_lengths`.
        """

        j = 1
        name = f"Carousel {cable_index}-{j}"

        while section_lengths.size > 0:
            sum_lengths = np.cumsum(section_lengths)
            max_sections_ix = np.where(sum_lengths <= max_length)[0][-1]

            self.carousels[name] = Carousel(
                name,
                sum_lengths[max_sections_ix],
                section_masses[: max_sections_ix + 1].sum(),
                section_lengths[: max_sections_ix + 1].tolist(),
                section_masses[: max_sections_ix + 1].tolist(),
                section_bury_speeds[: max_sections_ix + 1].tolist(),
                1,
            )

            j += 1
            name = f"Carousel {cable_index}-{j}"
            section_lengths = section_lengths[max_sections_ix + 1 :]
            section_masses = section_masses[max_sections_ix + 1 :]
            section_bury_speeds = section_bury_speeds[max_sections_ix + 1 :]

    def _create_cable_carousel(self, cable_name, max_length, cable_index):
        """
        Creates the individual `Carousel`s.

        Parameters
        ----------
        cable_name : str
            Dictionary key the cable.
        max_length : float
            Maximum length of cable allowed on a single carousel.
        cable_index : int
            1-indexed index of cable to keep track of which `Carousel` has what
            type of cable.
        """

        lengths, masses, bury_speeds = self._create_section_list(cable_name)
        if (lengths > max_length).sum() > 0:
            self._create_cable_carousel_with_splice(
                max_length, cable_index, lengths, masses, bury_speeds
            )
        else:
            self._create_cable_carousel_without_splice(
                max_length, cable_index, lengths, masses, bury_speeds
            )

    def create_carousels(self):
        """
        Creates the carousel information by cable type.
        """

        for i, (name, cable) in enumerate(self.cables.items()):
            max_cable_len_per_carousel = (
                self.max_cargo_weight / cable["linear_density"]
            )
            self._create_cable_carousel(
                name, max_cable_len_per_carousel, i + 1
            )
