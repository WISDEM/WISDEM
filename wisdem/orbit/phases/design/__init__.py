"""The design package contains `DesignPhase` and its subclasses."""

__author__ = ["Jake Nunemaker", "Rob Hammond"]
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = ["Jake Nunemaker", "Rob Hammond"]
__email__ = ["jake.nunemaker@nrel.gov" "robert.hammond@nrel.gov"]


from .design_phase import DesignPhase  # isort:skip
from .oss_design import OffshoreSubstationDesign
from .spar_design import SparDesign
from .monopile_design import MonopileDesign
from .array_system_design import ArraySystemDesign, CustomArraySystemDesign
from .export_system_design import ExportSystemDesign
from .mooring_system_design import MooringSystemDesign
from .scour_protection_design import ScourProtectionDesign
from .semi_submersible_design import SemiSubmersibleDesign
