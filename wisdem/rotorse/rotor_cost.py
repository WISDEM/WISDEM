import os

import numpy as np
import openmdao.api as om
from scipy.optimize import brentq

from wisdem.commonse.utilities import arc_length
from wisdem.glue_code.gc_LoadInputs import WindTurbineOntologyPython
from wisdem.glue_code.gc_WT_DataStruc import Blade, Materials, ComputeHighLevelBladeProperties
from wisdem.glue_code.gc_WT_InitModel import assign_blade_values, assign_airfoil_values, assign_material_values
from wisdem.glue_code.gc_PoseOptimization import PoseOptimization
import logging
logger = logging.getLogger("wisdem/weis")

### USING OLD NUMPY SRC FOR PMT-FUNCTION INSTEAD OF SWITCHING TO ANNOYING NUMPY-FINANCIAL
_when_to_num = {"end": 0, "begin": 1, "e": 0, "b": 1, 0: 0, 1: 1, "beginning": 1, "start": 1, "finish": 0}


def _convert_when(when):
    # Test to see if when has already been converted to ndarray
    # This will happen if one function calls another, for example ppmt
    if isinstance(when, np.ndarray):
        return when
    try:
        return _when_to_num[when]
    except (KeyError, TypeError):
        return [_when_to_num[x] for x in when]


def pmt(rate, nper, pv, fv=0, when="end"):
    when = _convert_when(when)
    (rate, nper, pv, fv, when) = map(np.array, [rate, nper, pv, fv, when])
    temp = (1 + rate) ** nper
    mask = rate == 0
    masked_rate = np.where(mask, 1, rate)
    fact = np.where(mask != 0, nper, (1 + masked_rate * when) * (temp - 1) / masked_rate)
    return -(fv + pv * temp) / fact


class blade_bom(object):
    def compute_consumables(self):

        # Consumables
        consumables = {}
        # # LE Erosion Tape
        # consumables["LE_tape"]                               = {}
        # consumables["LE_tape"]["unit_length"]                = 250. # [m] Roll length
        # consumables["LE_tape"]["unit_cost"]                  = 576. # [$/roll]
        # consumables["LE_tape"]["waste"]                      = 5.   # [%]
        # consumables["LE_tape"]["units_per_blade"]            = self.blade_specs["LE_length"] / consumables["LE_tape"]["unit_length"] # Rolls per blade
        # consumables["LE_tape"]["total_cost_wo_waste"]        = consumables["LE_tape"]["units_per_blade"] * consumables["LE_tape"]["unit_cost"]
        # consumables["LE_tape"]["total_cost_w_waste"]         = consumables["LE_tape"]["total_cost_wo_waste"] * (1 + consumables["LE_tape"]["waste"]/100)
        # if self.options["verbosity"]:
        # print("LE erosion tape cost %.2f $\t \t --- \t \t cost with waste %.2f $" % (consumables["LE_tape"]["total_cost_wo_waste"] , consumables["LE_tape"]["total_cost_w_waste"]))
        # Peel Ply
        consumables["peel_ply"] = {}
        consumables["peel_ply"]["unit_cost"] = 1.94  # [$/m2] 0.18 $/sqft
        consumables["peel_ply"]["waste"] = 15.0  # [%]
        consumables["peel_ply"]["total_cost_wo_waste"] = (
            sum(self.blade_specs["area_webs_w_flanges"])
            + self.blade_specs["area_lpskin_w_flanges"]
            + self.blade_specs["area_hpskin_w_flanges"]
            + self.blade_specs["area_sc_lp"]
            + self.blade_specs["area_sc_hp"]
            + self.blade_specs["area_lp_root"]
            + self.blade_specs["area_hp_root"]
        ) * consumables["peel_ply"]["unit_cost"]
        consumables["peel_ply"]["total_cost_w_waste"] = consumables["peel_ply"]["total_cost_wo_waste"] * (
            1 + consumables["peel_ply"]["waste"] / 100
        )
        # Non-Sanding Peel Ply
        consumables["ns_peel_ply"] = {}
        consumables["ns_peel_ply"]["unit_cost"] = 1.67  # [$/m2] 0.15 $/sqft
        consumables["ns_peel_ply"]["waste"] = 10.0  # [%]
        consumables["ns_peel_ply"]["unit_width"] = 0.127  # [m] Roll width
        consumables["ns_peel_ply"]["total_cost_wo_waste"] = (
            consumables["ns_peel_ply"]["unit_width"]
            * 2
            * (self.blade_specs["TE_length"] + self.blade_specs["LE_length"] + sum(self.blade_specs["length_webs"]))
            * consumables["ns_peel_ply"]["unit_cost"]
        )
        consumables["ns_peel_ply"]["total_cost_w_waste"] = consumables["ns_peel_ply"]["total_cost_wo_waste"] * (
            1 + consumables["ns_peel_ply"]["waste"] / 100
        )

        # Chopped Strand
        consumables["chopped_strand"] = {}
        consumables["chopped_strand"]["unit_cost"] = 2.16  # [$/kg] 0.98 $/lbs
        consumables["chopped_strand"]["mass_length"] = 0.037  # [kg/m] 0.025 lb/ft
        consumables["chopped_strand"]["waste"] = 5.0  # [%]
        consumables["chopped_strand"]["total_cost_wo_waste"] = (
            consumables["chopped_strand"]["mass_length"]
            * self.blade_specs["blade_length"]
            * consumables["chopped_strand"]["unit_cost"]
        )
        consumables["chopped_strand"]["total_cost_w_waste"] = consumables["chopped_strand"]["total_cost_wo_waste"] * (
            1 + consumables["chopped_strand"]["waste"] / 100
        )

        # 3M77 Adhesive, Bulk
        consumables["adhesive_bulk"] = {}
        consumables["adhesive_bulk"]["unit_cost"] = 10566.9  # [$/m3] 40 $/ga
        consumables["adhesive_bulk"]["volume_area"] = 3.06e-5  # [m3/m2] 0.00075 ga/sf
        consumables["adhesive_bulk"]["waste"] = 5.0  # [%]
        consumables["adhesive_bulk"]["total_cost_wo_waste"] = (
            consumables["adhesive_bulk"]["volume_area"]
            * (
                sum(self.blade_specs["area_webs_w_flanges"])
                + self.blade_specs["area_lpskin_w_flanges"]
                + self.blade_specs["area_hpskin_w_flanges"]
                + self.blade_specs["area_sc_lp"]
                + self.blade_specs["area_sc_hp"]
                + self.blade_specs["area_lp_root"]
                + self.blade_specs["area_hp_root"]
            )
            * consumables["adhesive_bulk"]["unit_cost"]
        )
        consumables["adhesive_bulk"]["total_cost_w_waste"] = consumables["adhesive_bulk"]["total_cost_wo_waste"] * (
            1 + consumables["adhesive_bulk"]["waste"] / 100
        )
        # 3M77 Adhesive, Cans
        consumables["adhesive_cans"] = {}
        consumables["adhesive_cans"]["unit_cost"] = 6.65  # [$]
        consumables["adhesive_cans"]["waste"] = 5.0  # [%]
        consumables["adhesive_cans"]["units_area"] = 0.022  # [each/m2] 0.002 each/sf
        consumables["adhesive_cans"]["units_blade"] = consumables["adhesive_cans"]["units_area"] * (
            sum(self.blade_specs["area_webs_w_flanges"])
            + self.blade_specs["area_lpskin_w_flanges"]
            + self.blade_specs["area_hpskin_w_flanges"]
            + self.blade_specs["area_sc_lp"]
            + self.blade_specs["area_sc_hp"]
            + self.blade_specs["area_lp_root"]
            + self.blade_specs["area_hp_root"]
        )
        consumables["adhesive_cans"]["total_cost_wo_waste"] = (
            consumables["adhesive_cans"]["units_blade"] * consumables["adhesive_cans"]["unit_cost"]
        )
        consumables["adhesive_cans"]["total_cost_w_waste"] = consumables["adhesive_cans"]["total_cost_wo_waste"] * (
            1 + consumables["adhesive_cans"]["waste"] / 100
        )

        # Mold Release
        consumables["release_agent"] = {}
        consumables["release_agent"]["unit_cost"] = 15691.82  # [$/m3] - 59.40 $/gal
        consumables["release_agent"]["waste"] = 5.0  # [%]
        consumables["release_agent"]["volume_area"] = 2.57e-5  # [m3/m2] 0.00063 ga/sf
        consumables["release_agent"]["total_cost_wo_waste"] = (
            consumables["release_agent"]["volume_area"]
            * (
                sum(self.blade_specs["area_webs_w_flanges"])
                + self.blade_specs["area_lpskin_w_flanges"]
                + self.blade_specs["area_hpskin_w_flanges"]
                + self.blade_specs["area_sc_lp"]
                + self.blade_specs["area_sc_hp"]
                + self.blade_specs["area_lp_root"]
                + self.blade_specs["area_hp_root"]
            )
            * consumables["release_agent"]["unit_cost"]
        )
        consumables["release_agent"]["total_cost_w_waste"] = consumables["release_agent"]["total_cost_wo_waste"] * (
            1 + consumables["release_agent"]["waste"] / 100
        )

        # Flow Medium
        consumables["flow_medium"] = {}
        consumables["flow_medium"]["unit_cost"] = 0.646  # [$/m2] 0.06 $/sqft
        consumables["flow_medium"]["waste"] = 15.0  # [%]
        consumables["flow_medium"]["coverage"] = 70.0  # [%]
        consumables["flow_medium"]["total_cost_wo_waste"] = (
            (
                sum(self.blade_specs["area_webs_w_flanges"])
                + self.blade_specs["area_lpskin_w_flanges"]
                + self.blade_specs["area_hpskin_w_flanges"]
                + self.blade_specs["area_sc_lp"]
                + self.blade_specs["area_sc_hp"]
                + self.blade_specs["area_lp_root"]
                + self.blade_specs["area_hp_root"]
            )
            * consumables["flow_medium"]["coverage"]
            / 100
            * consumables["flow_medium"]["unit_cost"]
        )
        consumables["flow_medium"]["total_cost_w_waste"] = consumables["flow_medium"]["total_cost_wo_waste"] * (
            1 + consumables["flow_medium"]["waste"] / 100
        )

        # tubing - 3/8"
        consumables["tubing3/8"] = {}
        consumables["tubing3/8"]["unit_cost"] = 0.23  # [$/m] 0.07 $/ft
        consumables["tubing3/8"]["waste"] = 10.0  # [%]
        consumables["tubing3/8"]["length_per_length_blade"] = 5  # [m/m]
        consumables["tubing3/8"]["length"] = (
            consumables["tubing3/8"]["length_per_length_blade"] * self.blade_specs["blade_length"]
        )
        consumables["tubing3/8"]["total_cost_wo_waste"] = (
            consumables["tubing3/8"]["length"] * consumables["tubing3/8"]["unit_cost"]
        )
        consumables["tubing3/8"]["total_cost_w_waste"] = consumables["tubing3/8"]["total_cost_wo_waste"] * (
            1 + consumables["tubing3/8"]["waste"] / 100
        )

        # tubing - 1/2"
        consumables["tubing1/2"] = {}
        consumables["tubing1/2"]["unit_cost"] = 0.23  # [$/m] 0.07 $/ft
        consumables["tubing1/2"]["waste"] = 10.0  # [%]
        consumables["tubing1/2"]["length_per_length_blade"] = 5  # [m/m]
        consumables["tubing1/2"]["length"] = (
            consumables["tubing1/2"]["length_per_length_blade"] * self.blade_specs["blade_length"]
        )
        consumables["tubing1/2"]["total_cost_wo_waste"] = (
            consumables["tubing1/2"]["length"] * consumables["tubing1/2"]["unit_cost"]
        )
        consumables["tubing1/2"]["total_cost_w_waste"] = consumables["tubing1/2"]["total_cost_wo_waste"] * (
            1 + consumables["tubing1/2"]["waste"] / 100
        )

        # tubing - 5/8"
        consumables["tubing5/8"] = {}
        consumables["tubing5/8"]["unit_cost"] = 0.49  # [$/m] 0.15 $/ft
        consumables["tubing5/8"]["waste"] = 10.0  # [%]
        consumables["tubing5/8"]["length_per_length_blade"] = 5  # [m/m]
        consumables["tubing5/8"]["length"] = (
            consumables["tubing5/8"]["length_per_length_blade"] * self.blade_specs["blade_length"]
        )
        consumables["tubing5/8"]["total_cost_wo_waste"] = (
            consumables["tubing5/8"]["length"] * consumables["tubing5/8"]["unit_cost"]
        )
        consumables["tubing5/8"]["total_cost_w_waste"] = consumables["tubing5/8"]["total_cost_wo_waste"] * (
            1 + consumables["tubing5/8"]["waste"] / 100
        )

        # tubing - 3/4"
        consumables["tubing3/4"] = {}
        consumables["tubing3/4"]["unit_cost"] = 0.62  # [$/m] 0.19 $/ft
        consumables["tubing3/4"]["waste"] = 10.0  # [%]
        consumables["tubing3/4"]["length_per_length_blade"] = 5  # [m/m]
        consumables["tubing3/4"]["length"] = (
            consumables["tubing3/4"]["length_per_length_blade"] * self.blade_specs["blade_length"]
        )
        consumables["tubing3/4"]["total_cost_wo_waste"] = (
            consumables["tubing3/4"]["length"] * consumables["tubing3/4"]["unit_cost"]
        )
        consumables["tubing3/4"]["total_cost_w_waste"] = consumables["tubing3/4"]["total_cost_wo_waste"] * (
            1 + consumables["tubing3/4"]["waste"] / 100
        )

        # tubing - 7/8"
        consumables["tubing7/8"] = {}
        consumables["tubing7/8"]["unit_cost"] = 0.49  # [$/m] 0.15 $/ft
        consumables["tubing7/8"]["waste"] = 10.0  # [%]
        consumables["tubing7/8"]["length_per_length_blade"] = 5  # [m/m]
        consumables["tubing7/8"]["length"] = (
            consumables["tubing7/8"]["length_per_length_blade"] * self.blade_specs["blade_length"]
        )
        consumables["tubing7/8"]["total_cost_wo_waste"] = (
            consumables["tubing7/8"]["length"] * consumables["tubing7/8"]["unit_cost"]
        )
        consumables["tubing7/8"]["total_cost_w_waste"] = consumables["tubing7/8"]["total_cost_wo_waste"] * (
            1 + consumables["tubing7/8"]["waste"] / 100
        )

        # Silicon flange tape
        consumables["tacky_tape"] = {}
        consumables["tacky_tape"]["unit_length"] = 3.5  # [m/roll]
        consumables["tacky_tape"]["unit_cost"] = 21.11  # [$/roll]
        consumables["tacky_tape"]["waste"] = 5.0  # [%]
        consumables["tacky_tape"]["units_per_blade"] = (10.0 * self.blade_specs["blade_length"]) / consumables[
            "tacky_tape"
        ][
            "unit_length"
        ]  # [-]
        consumables["tacky_tape"]["total_cost_wo_waste"] = (
            consumables["tacky_tape"]["units_per_blade"] * consumables["tacky_tape"]["unit_cost"]
        )
        consumables["tacky_tape"]["total_cost_w_waste"] = consumables["tacky_tape"]["total_cost_wo_waste"] * (
            1 + consumables["tacky_tape"]["waste"] / 100
        )

        # 2" masking tape
        consumables["masking_tape"] = {}
        consumables["masking_tape"]["unit_cost"] = 5.50  # [$/roll]
        consumables["masking_tape"]["waste"] = 10.0  # [%]
        consumables["masking_tape"]["roll_per_length"] = 0.328  # [roll/m]
        consumables["masking_tape"]["units_per_blade"] = (
            self.blade_specs["blade_length"] * consumables["masking_tape"]["roll_per_length"]
        )  # [-]
        consumables["masking_tape"]["total_cost_wo_waste"] = (
            consumables["masking_tape"]["units_per_blade"] * consumables["masking_tape"]["unit_cost"]
        )
        consumables["masking_tape"]["total_cost_w_waste"] = consumables["masking_tape"]["total_cost_wo_waste"] * (
            1 + consumables["masking_tape"]["waste"] / 100
        )

        # Chop Fiber
        consumables["chop_fiber"] = {}
        consumables["chop_fiber"]["unit_cost"] = 6.19  # [$/kg] 2.81 $/lbs
        consumables["chop_fiber"]["mass_area"] = 9.76e-3  # [kg/m2] 0.002 lb/sf
        consumables["chop_fiber"]["waste"] = 10.0  # [%]
        consumables["chop_fiber"]["total_cost_wo_waste"] = (
            consumables["chop_fiber"]["mass_area"]
            * (self.blade_specs["area_lpskin_wo_flanges"] + self.blade_specs["area_hpskin_wo_flanges"])
            * consumables["chop_fiber"]["unit_cost"]
        )
        consumables["chop_fiber"]["total_cost_w_waste"] = consumables["chop_fiber"]["total_cost_wo_waste"] * (
            1 + consumables["chop_fiber"]["waste"] / 100
        )

        # White Lightning
        consumables["white_lightning"] = {}
        consumables["white_lightning"]["unit_cost"] = 3006.278  # [$/m3] - 11.38 $/gal
        consumables["white_lightning"]["waste"] = 10.0  # [%]
        consumables["white_lightning"]["volume_area"] = 2.04e-5  # [m3/m2] 0.0005 ga/sf
        consumables["white_lightning"]["total_cost_wo_waste"] = (
            consumables["white_lightning"]["volume_area"]
            * (self.blade_specs["area_lpskin_wo_flanges"] + self.blade_specs["area_hpskin_wo_flanges"])
            * consumables["white_lightning"]["unit_cost"]
        )
        consumables["white_lightning"]["total_cost_w_waste"] = consumables["white_lightning"]["total_cost_wo_waste"] * (
            1 + consumables["white_lightning"]["waste"] / 100
        )

        # Hardener
        consumables["hardener"] = {}
        consumables["hardener"]["unit_cost"] = 1.65  # [$/tube]
        consumables["hardener"]["waste"] = 10.0  # [%]
        consumables["hardener"]["units_area"] = 0.012  # [each/m2] 0.0011 tube/sf
        consumables["hardener"]["units_blade"] = consumables["hardener"]["units_area"] * (
            self.blade_specs["area_lpskin_wo_flanges"] + self.blade_specs["area_hpskin_wo_flanges"]
        )
        consumables["hardener"]["total_cost_wo_waste"] = (
            consumables["hardener"]["units_blade"] * consumables["hardener"]["unit_cost"]
        )
        consumables["hardener"]["total_cost_w_waste"] = consumables["hardener"]["total_cost_wo_waste"] * (
            1 + consumables["hardener"]["waste"] / 100
        )

        # Putty
        consumables["putty"] = {}
        consumables["putty"]["unit_cost"] = 6.00  # [$/kg]
        consumables["putty"]["mass_area"] = 0.0244  # [kg/m2]
        consumables["putty"]["waste"] = 10.0  # [%]
        consumables["putty"]["total_cost_wo_waste"] = (
            consumables["putty"]["mass_area"]
            * (self.blade_specs["area_lpskin_wo_flanges"] + self.blade_specs["area_hpskin_wo_flanges"])
            * consumables["putty"]["unit_cost"]
        )
        consumables["putty"]["total_cost_w_waste"] = consumables["putty"]["total_cost_wo_waste"] * (
            1 + consumables["putty"]["waste"] / 100
        )

        # Putty Catalyst
        consumables["catalyst"] = {}
        consumables["catalyst"]["unit_cost"] = 7.89  # [$/kg]  3.58 $/lbs
        consumables["catalyst"]["mass_area"] = 4.88e-3  # [kg/m2] 0.001 lb/sf
        consumables["catalyst"]["waste"] = 10.0  # [%]
        consumables["catalyst"]["total_cost_wo_waste"] = (
            consumables["catalyst"]["mass_area"]
            * (self.blade_specs["area_lpskin_wo_flanges"] + self.blade_specs["area_hpskin_wo_flanges"])
            * consumables["catalyst"]["unit_cost"]
        )
        consumables["catalyst"]["total_cost_w_waste"] = consumables["catalyst"]["total_cost_wo_waste"] * (
            1 + consumables["catalyst"]["waste"] / 100
        )

        return consumables


class blade_labor_ct(object):
    def __init__(self, blade_specs, mat_dictionary, metallic_parts):

        # # Blade input parameters
        # # Material inputs
        self.materials = mat_dictionary
        # Root preform low pressure side
        self.root_parameters_lp = {}
        self.root_parameters_lp["blade_length"] = blade_specs["blade_length"]  # Length of the blade [m]
        self.root_parameters_lp["length"] = blade_specs["root_preform_length"]  # Root PF length [m]
        self.root_parameters_lp["root_D"] = blade_specs["root_D"]  # Root PF diameter [m]
        self.root_parameters_lp["half_circum"] = 0.5 * np.pi * blade_specs["root_D"]  # 1/2 root circumference [m]
        self.root_parameters_lp["area"] = (
            self.root_parameters_lp["half_circum"] * self.root_parameters_lp["length"]
        )  # Root PF area [m2]
        self.root_parameters_lp["fabric2lay"] = round(blade_specs["n_plies_root_lp"] / 2)  # Number of root plies [-]
        self.root_parameters_lp["ply_volume"] = blade_specs["volume_root_preform_lp"]  # Ply volume [m3]
        # Root preform high pressure side
        self.root_parameters_hp = {}
        self.root_parameters_hp["blade_length"] = blade_specs["blade_length"]  # Length of the blade [m]
        self.root_parameters_hp["length"] = self.root_parameters_lp[
            "length"
        ]  # Root PF length [m], currently assumed as 1% of BL
        self.root_parameters_hp["root_D"] = self.root_parameters_lp["root_D"]  # Root PF diameter [m]
        self.root_parameters_hp["half_circum"] = self.root_parameters_lp["half_circum"]  # 1/2 root circumference [m]
        self.root_parameters_hp["area"] = self.root_parameters_lp["area"]  # Root PF area [m2]
        self.root_parameters_hp["fabric2lay"] = round(blade_specs["n_plies_root_hp"] / 2)  # Number of root plies [-]
        self.root_parameters_hp["ply_volume"] = blade_specs["volume_root_preform_hp"]  # Ply volume [m3]
        # Shear webs
        self.n_webs = blade_specs["n_webs"]
        self.sw_parameters = {}
        self.sw_parameters["blade_length"] = blade_specs["blade_length"]  # Length of the blade [m]
        self.sw_parameters["length"] = blade_specs["length_webs"]  # Length of the shear webs [m]
        self.sw_parameters["height1"] = blade_specs[
            "height_webs_start"
        ]  # Heigth of the shear webs towards blade root [m]
        self.sw_parameters["height2"] = blade_specs["height_webs_end"]  # Heigth of the shear webs towards blade tip [m]
        self.sw_parameters["core_area"] = blade_specs[
            "area_webs_w_core"
        ]  # Area of the shear webs with sandwich core [m2]
        self.sw_parameters["area"] = blade_specs["area_webs_w_flanges"]  # Shear webs area [m2]
        self.sw_parameters["fabric2lay"] = blade_specs["fabric2lay_webs"]  # Total ply area [m2]
        self.sw_parameters["ply_volume"] = blade_specs["volumeskin2lay_webs"]  # Ply volume [m3]
        # Spar cap low pressure side
        self.lp_sc_parameters = {}
        self.lp_sc_parameters["blade_length"] = blade_specs["blade_length"]  # Length of the blade [m]
        self.lp_sc_parameters["length"] = blade_specs["length_sc_lp"]  # Length of the spar caps [m]
        self.lp_sc_parameters["width"] = blade_specs["width_sc_start_lp"]  # Width of the spar caps [m]
        self.lp_sc_parameters["area"] = (
            blade_specs["length_sc_lp"] * blade_specs["width_sc_start_lp"]
        )  # Spar caps area [m2]
        self.lp_sc_parameters["area_wflanges"] = (
            self.lp_sc_parameters["area"] * 1.5
        )  # Spar caps area including flanges [m2] Assume the sides and the flanges of a spar cap mold equal about 1.5 times the area of the spar cap (for tool prep purposes)
        self.lp_sc_parameters["fabric2lay"] = blade_specs["fabric2lay_sc_lp"]  # Total ply length [m]
        self.lp_sc_parameters["ply_volume"] = blade_specs["volume2lay_sc_lp"]  # Ply volume [m3]
        # Spar caps high pressure side
        self.hp_sc_parameters = {}
        self.hp_sc_parameters["blade_length"] = blade_specs["blade_length"]  # Length of the blade [m]
        self.hp_sc_parameters["length"] = blade_specs["length_sc_hp"]  # Length of the spar caps [m]
        self.hp_sc_parameters["width"] = blade_specs["width_sc_start_hp"]  # Width of the spar caps [m]
        self.hp_sc_parameters["area"] = (
            blade_specs["length_sc_hp"] * blade_specs["width_sc_start_hp"]
        )  # Spar caps area [m2]
        self.hp_sc_parameters["area_wflanges"] = (
            self.hp_sc_parameters["area"] * 1.5
        )  # Spar caps area including flanges [m2] Assume the sides and the flanges of a spar cap mold equal about 1.5 times the area of the spar cap (for tool prep purposes)
        self.hp_sc_parameters["fabric2lay"] = blade_specs["fabric2lay_sc_hp"]  # Total ply length [m]
        self.hp_sc_parameters["ply_volume"] = blade_specs["volume2lay_sc_hp"]  # Ply volume [m3]
        # Low pressure skin
        self.lp_skin_parameters = {}
        self.lp_skin_parameters["blade_length"] = blade_specs["blade_length"]  # Length of the blade [m]
        self.lp_skin_parameters["length"] = blade_specs["blade_length"]  # Length of the skin mold [m]
        self.lp_skin_parameters["area"] = blade_specs[
            "area_lpskin_wo_flanges"
        ]  # Skin area on the low pressure side [m2]
        self.lp_skin_parameters["area_wflanges"] = blade_specs[
            "area_lpskin_w_flanges"
        ]  # Skin area including flanges [m2]
        self.lp_skin_parameters["fabric2lay"] = (
            0.5 * blade_specs["fabric2lay_shell_lp"]
        )  # Total ply area, outer layers [m2]. Assumed to be 50% of the total layers
        self.lp_skin_parameters["fabric2lay_inner"] = (
            0.5 * blade_specs["fabric2lay_shell_lp"]
        )  # Total ply area, inner layers [m2]. Assumed to be 50% of the total layers
        self.lp_skin_parameters["core_area"] = blade_specs[
            "areacore2lay_shell_lp"
        ]  # Area of the outer shell with sandwich core [m2]
        self.lp_skin_parameters["n_root_plies"] = self.root_parameters_lp["fabric2lay"]  # Number of root plies [-]
        self.lp_skin_parameters["total_TE"] = blade_specs["fabric2lay_te_reinf_lp"]  # Total TE reinforcement layer [m]
        self.lp_skin_parameters["total_LE"] = blade_specs["fabric2lay_le_reinf_lp"]  # Total LE reinforcement layer [m]
        self.lp_skin_parameters["perimeter_noroot"] = blade_specs[
            "skin_perimeter_wo_root"
        ]  # Perimeter of the skin area excluding blade root [m]
        self.lp_skin_parameters["perimeter"] = blade_specs["skin_perimeter_w_root"]  # Perimeter of the skin area [m]
        self.lp_skin_parameters["sc_length"] = blade_specs["length_sc_lp"]  # Length of the spar cap [m]
        self.lp_skin_parameters["root_sect_length"] = blade_specs["root_preform_length"]  # Root section length [m]
        self.lp_skin_parameters["root_half_circumf"] = self.root_parameters_lp[
            "half_circum"
        ]  # Root half circumference [m]
        # High pressure skin
        self.hp_skin_parameters = {}
        self.hp_skin_parameters["blade_length"] = blade_specs["blade_length"]  # Length of the blade [m]
        self.hp_skin_parameters["length"] = blade_specs["blade_length"]  # Length of the skin mold [m]
        self.hp_skin_parameters["area"] = blade_specs["area_hpskin_wo_flanges"]  # Skin area [m2]
        self.hp_skin_parameters["area_wflanges"] = blade_specs[
            "area_hpskin_w_flanges"
        ]  # Skin area including flanges [m2]
        self.hp_skin_parameters["fabric2lay"] = (
            0.5 * blade_specs["fabric2lay_shell_hp"]
        )  # Total ply area, outer layers [m2]. Assumed to be 50% of the total layers
        self.hp_skin_parameters["fabric2lay_inner"] = (
            0.5 * blade_specs["fabric2lay_shell_hp"]
        )  # Total ply area, inner layers [m2]. Assumed to be 50% of the total layers
        self.hp_skin_parameters["core_area"] = blade_specs[
            "areacore2lay_shell_hp"
        ]  # Area of the outer shell with sandwich core [m2]
        self.hp_skin_parameters["n_root_plies"] = self.root_parameters_hp["fabric2lay"]  # Number of root plies [-]
        self.hp_skin_parameters["total_TE"] = blade_specs["fabric2lay_te_reinf_hp"]  # Total TE reinforcement layer [m]
        self.hp_skin_parameters["total_LE"] = blade_specs["fabric2lay_le_reinf_hp"]  # Total LE reinforcement layer [m]
        self.hp_skin_parameters["perimeter_noroot"] = blade_specs[
            "skin_perimeter_wo_root"
        ]  # Perimeter of the skin area excluding blade root [m]
        self.hp_skin_parameters["perimeter"] = blade_specs["skin_perimeter_w_root"]  # Perimeter of the skin area [m]
        self.hp_skin_parameters["sc_length"] = blade_specs["length_sc_hp"]  # Length of the spar cap [m]
        self.hp_skin_parameters["root_sect_length"] = blade_specs["root_preform_length"]  # Root section length [m]
        self.hp_skin_parameters["root_half_circumf"] = self.root_parameters_hp[
            "half_circum"
        ]  # Root half circumference [m]
        # Assembly
        self.assembly = {}
        self.assembly["sw_length"] = self.sw_parameters["length"]  # Length of the shear webs [m]
        self.assembly["perimeter_noroot"] = blade_specs[
            "skin_perimeter_wo_root"
        ]  # Perimeter of the skin area without root [m]
        self.assembly["length"] = blade_specs["blade_length"]  # Length of the blade [m]
        self.assembly["n_webs"] = blade_specs["n_webs"]  # Number of webs [-]
        # Demold
        self.demold = {}
        self.demold["length"] = blade_specs["blade_length"]  # Length of the blade [m]
        # Trim
        self.trim = {}
        self.trim["perimeter_noroot"] = blade_specs["skin_perimeter_wo_root"]  # Length of the blade [m]
        # Overlay
        self.overlay = {}
        self.overlay["length"] = blade_specs["blade_length"]  # Length of the blade [m]
        # Post curing
        self.post_cure = {}
        self.post_cure["length"] = blade_specs["blade_length"]  # Length of the blade [m]
        # Cut and drill
        self.cut_drill = {}
        self.cut_drill["length"] = blade_specs["blade_length"]  # Length of the blade [m]
        self.cut_drill["root_D"] = blade_specs["root_D"]  # Diameter of blade root [m]
        self.cut_drill["root_perim"] = self.cut_drill["root_D"] * np.pi  # Perimeter of the root [m]
        self.cut_drill["n_bolts"] = metallic_parts["n_bolts"]  # Number of root bolts [-]
        # Root installation
        self.root_install = {}
        self.root_install["length"] = blade_specs["blade_length"]  # Length of the blade [m]
        self.root_install["root_perim"] = self.cut_drill["root_D"] * np.pi  # Perimeter of the root [m]
        self.root_install["n_bolts"] = self.cut_drill["n_bolts"]  # Number of root bolts
        # Surface preparation
        self.surface_prep = {}
        self.surface_prep["area"] = (
            blade_specs["area_lpskin_wo_flanges"] + blade_specs["area_hpskin_wo_flanges"]
        )  # Outer blade surface area [m2]
        # Paint
        self.paint = {}
        self.paint["area"] = (
            blade_specs["area_lpskin_wo_flanges"] + blade_specs["area_hpskin_wo_flanges"]
        )  # Outer blade surface area [m2]
        # Surface finishing
        self.surface_finish = {}
        self.surface_finish["area"] = (
            blade_specs["area_lpskin_wo_flanges"] + blade_specs["area_hpskin_wo_flanges"]
        )  # Outer blade surface area [m2]
        # Weight and balance
        self.weight_balance = {}
        self.weight_balance["length"] = blade_specs["blade_length"]  # Length of the blade [m]
        # Inspection
        self.inspection = {}
        self.inspection["length"] = blade_specs["blade_length"]  # Length of the blade [m]
        self.inspection["area"] = (
            blade_specs["area_lpskin_wo_flanges"] + blade_specs["area_hpskin_wo_flanges"]
        )  # Outer blade surface area [m2]
        # Shipping preparation
        self.shipping_prep = {}
        self.shipping_prep["length"] = blade_specs["blade_length"]  # Length of the blade [m]
        self.shipping_prep["n_bolts"] = self.cut_drill["n_bolts"]  # Number of root bolts

    def execute_blade_labor_ct(self):

        # Run all manufacturing steps to estimate labor and cycle time
        verbosity = 0

        n_operations = 20 + self.n_webs
        labor = np.zeros(n_operations)  # [hr]
        skin_mold_gating_ct = np.zeros(n_operations)  # [hr]
        non_gating_ct = np.zeros(n_operations)  # [hr]
        operation = [[] for i in range(int(n_operations))]

        if verbosity:
            print("\n#################################\nLabor hours and cycle times")
        operation[0] = "Material cutting"
        material_cutting = material_cutting_labor(self.materials)
        material_cutting.material_cutting_steps()
        labor[0], non_gating_ct[0] = compute_total_labor_ct(material_cutting, operation[0], verbosity)

        operation[1] = "Root preform lp"
        root_lp = root_preform_labor(self.root_parameters_lp)
        root_lp.manufacturing_steps()
        labor[1], non_gating_ct[1] = compute_total_labor_ct(root_lp, operation[1], verbosity)

        operation[2] = "Root preform hp"
        root_hp = root_preform_labor(self.root_parameters_hp)
        root_hp.manufacturing_steps()
        labor[2], non_gating_ct[2] = compute_total_labor_ct(root_hp, operation[2], verbosity)

        for i_web in range(self.n_webs):
            operation[3 + i_web] = "Infusion shear web number " + str(i_web + 1)
            sw = shearweb_labor(self.sw_parameters, i_web)
            sw.manufacturing_steps(core=True)
            labor[3 + i_web], non_gating_ct[3 + i_web] = compute_total_labor_ct(sw, operation[3 + i_web], verbosity)

        operation[3 + self.n_webs] = "Infusion spar cap lp"
        sc_lp = sparcaps_labor(self.lp_sc_parameters)
        sc_lp.manufacturing_steps()
        labor[3 + self.n_webs], non_gating_ct[3 + self.n_webs] = compute_total_labor_ct(
            sc_lp, operation[3 + self.n_webs], verbosity
        )

        operation[4 + self.n_webs] = "Infusion spar cap hp"
        sc_hp = sparcaps_labor(self.hp_sc_parameters)
        sc_hp.manufacturing_steps()
        labor[4 + self.n_webs], non_gating_ct[4 + self.n_webs] = compute_total_labor_ct(
            sc_hp, operation[4 + self.n_webs], verbosity
        )

        # Gating operations
        operation[8 + self.n_webs] = "Demolding"
        demold = demold_labor(self.demold)
        demold.demold_steps()
        labor[8 + self.n_webs], skin_mold_gating_ct[8 + self.n_webs] = compute_total_labor_ct(
            demold, operation[8 + self.n_webs], verbosity, no_contribution2ct=["move2finishing"]
        )

        # LP skin infusion
        operation[5 + self.n_webs] = "Lp skin"

        def labor_ct_lp_skin(team_size):
            lp_skin = lphp_skin_labor(self.lp_skin_parameters, team_size)
            lp_skin.manufacturing_steps(core=True, Extra_Operations_Skin=True, trim_excess=False)
            labor, ct = compute_total_labor_ct(
                lp_skin,
                operation[5 + self.n_webs],
                verbosity,
                no_contribution2ct=["layup_root_layers", "insert_TE_layers", "vacuum_line", "tack_tape"],
            )

            return labor, ct

        def min_ct_lp_skin(team_size):
            _, ct = labor_ct_lp_skin(team_size)
            return ct - (23.9999 - skin_mold_gating_ct[8 + self.n_webs]) * 0.7

        try:
            team_size = brentq(lambda x: min_ct_lp_skin(x), 0.01, 100.0, xtol=1e-4)
        except:
            team_size = 100.0
            # print("WARNING: the blade cost model is used beyond its applicability range. No team can limit the main mold cycle time to 24 hours. 100 workers are assumed at the low-pressure mold, but this is incorrect.")

        labor[5 + self.n_webs], skin_mold_gating_ct[5 + self.n_webs] = labor_ct_lp_skin(team_size)

        # HP skin infusion
        operation[6 + self.n_webs] = "Hp skin"

        def labor_ct_hp_skin(team_size):
            hp_skin = lphp_skin_labor(self.hp_skin_parameters, team_size)
            hp_skin.manufacturing_steps(core=True, Extra_Operations_Skin=True, trim_excess=False)
            labor, ct = compute_total_labor_ct(
                hp_skin,
                operation[6 + self.n_webs],
                verbosity,
                no_contribution2ct=["layup_root_layers", "insert_TE_layers", "vacuum_line", "tack_tape"],
            )

            return labor, ct

        def min_ct_hp_skin(team_size):
            _, ct = labor_ct_hp_skin(team_size)

            return ct - (23.9999 - skin_mold_gating_ct[8 + self.n_webs]) * 0.7

        try:
            team_size = brentq(lambda x: min_ct_hp_skin(x), 0.01, 100.0, xtol=1e-4)
        except:
            team_size = 100.0
            # print("WARNING: the blade cost model is used beyond its applicability range. No team can limit the main mold cycle time to 24 hours. 100 workers are assumed at the high-pressure mold, but this is incorrect.")

        labor[6 + self.n_webs], non_gating_ct[6 + self.n_webs] = labor_ct_hp_skin(team_size)

        # Assembly
        operation[7 + self.n_webs] = "Assembly"

        def labor_ct_assembly(team_size):
            assembly = assembly_labor(self.assembly, team_size)
            assembly.assembly_steps()
            labor, ct = compute_total_labor_ct(
                assembly,
                operation[7 + self.n_webs],
                verbosity,
                no_contribution2ct=["remove_nonsand_prep_hp", "insert_sw", "fillet_sw_low", "shear_clips"],
            )

            return labor, ct

        def min_ct_assembly(team_size):
            _, ct = labor_ct_assembly(team_size)

            return ct - (23.9999 - skin_mold_gating_ct[5 + self.n_webs] - skin_mold_gating_ct[8 + self.n_webs])

        try:
            team_size = brentq(lambda x: min_ct_assembly(x), 0.01, 100.0, xtol=1e-4)
        except:
            team_size = 100.0
            # print("WARNING: the blade cost model is used beyond its applicability range. No team can limit the assembly cycle time to 24 hours. 100 workers are assumed at the assembly line, but this is incorrect.")

        labor[7 + self.n_webs], skin_mold_gating_ct[7 + self.n_webs] = labor_ct_assembly(team_size)

        operation[9 + self.n_webs] = "Trim"
        trim = trim_labor(self.trim)
        trim.trim_steps()
        labor[9 + self.n_webs], non_gating_ct[9 + self.n_webs] = compute_total_labor_ct(
            trim, operation[9 + self.n_webs], verbosity
        )

        operation[10 + self.n_webs] = "Overlay"
        overlay = overlay_labor(self.overlay)
        overlay.overlay_steps()
        labor[10 + self.n_webs], non_gating_ct[10 + self.n_webs] = compute_total_labor_ct(
            overlay, operation[10 + self.n_webs], verbosity
        )

        operation[11 + self.n_webs] = "Post cure"
        post_cure = post_cure_labor(self.post_cure)
        post_cure.post_cure_steps()
        labor[11 + self.n_webs], non_gating_ct[11 + self.n_webs] = compute_total_labor_ct(
            post_cure, operation[11 + self.n_webs], verbosity
        )

        operation[12 + self.n_webs] = "Root cut and drill"
        cut_drill = cut_drill_labor(self.cut_drill)
        cut_drill.cut_drill_steps()
        labor[12 + self.n_webs], non_gating_ct[12 + self.n_webs] = compute_total_labor_ct(
            cut_drill, operation[12 + self.n_webs], verbosity
        )

        operation[13 + self.n_webs] = "Root hardware installation"
        root_install = root_install_labor(self.root_install)
        root_install.root_install_steps()
        labor[13 + self.n_webs], non_gating_ct[13 + self.n_webs] = compute_total_labor_ct(
            root_install, operation[13 + self.n_webs], verbosity
        )

        operation[14 + self.n_webs] = "Surface preparation"
        surface_prep = surface_prep_labor(self.surface_prep)
        surface_prep.surface_prep_steps()
        labor[14 + self.n_webs], non_gating_ct[14 + self.n_webs] = compute_total_labor_ct(
            surface_prep, operation[14 + self.n_webs], verbosity
        )

        operation[15 + self.n_webs] = "Painting"
        paint = paint_labor(self.paint)
        paint.paint_steps()
        labor[15 + self.n_webs], non_gating_ct[15 + self.n_webs] = compute_total_labor_ct(
            paint, operation[15 + self.n_webs], verbosity
        )

        operation[16 + self.n_webs] = "Surface finishing"
        surface_finish = surface_finish_labor(self.surface_finish)
        surface_finish.surface_finish_steps()
        labor[16 + self.n_webs], non_gating_ct[16 + self.n_webs] = compute_total_labor_ct(
            surface_finish, operation[16 + self.n_webs], verbosity
        )

        operation[17 + self.n_webs] = "Weight balance"
        weight_balance = weight_balance_labor(self.weight_balance)
        weight_balance.weight_balance_steps()
        labor[17 + self.n_webs], non_gating_ct[17 + self.n_webs] = compute_total_labor_ct(
            weight_balance, operation[17 + self.n_webs], verbosity
        )

        operation[18 + self.n_webs] = "Final inspection"
        inspection = inspection_labor(self.inspection)
        inspection.inspection_steps()
        labor[18 + self.n_webs], non_gating_ct[18 + self.n_webs] = compute_total_labor_ct(
            inspection, operation[18 + self.n_webs], verbosity
        )

        operation[19 + self.n_webs] = "Shipping preparation"
        shipping_prep = shipping_prep_labor(self.shipping_prep)
        shipping_prep.shipping_prep_steps()
        labor[19 + self.n_webs], non_gating_ct[19 + self.n_webs] = compute_total_labor_ct(
            shipping_prep, operation[19 + self.n_webs], verbosity
        )

        total_labor = sum(labor)
        total_skin_mold_gating_ct = sum(skin_mold_gating_ct)
        total_non_gating_ct = sum(non_gating_ct)

        return operation, labor, skin_mold_gating_ct, non_gating_ct


class material_cutting_process(object):
    def material_cutting_steps(self):

        self.load_roll["labor_per_mat"] = []
        self.load_roll["ct_per_mat"] = []
        self.cutting["labor_per_mat"] = []
        self.cutting["ct_per_mat"] = []
        self.kitting["labor_per_mat"] = []
        self.kitting["ct_per_mat"] = []
        self.clean_up["labor_per_mat"] = []
        self.clean_up["ct_per_mat"] = []

        self.load_roll["labor"] = []
        self.load_roll["ct"] = []
        self.cutting["labor"] = []
        self.cutting["ct"] = []
        self.kitting["labor"] = []
        self.kitting["ct"] = []
        self.clean_up["labor"] = []
        self.clean_up["ct"] = []

        mat_names = self.materials["mat_name"]

        self.materials["n_rolls"] = np.zeros(len(mat_names))

        for i_mat in range(len(mat_names)):

            if (
                self.materials["orth"][i_mat] == 1
                and self.materials["component_id"][i_mat] > 1
                and self.materials["component_id"][i_mat] < 3
            ):
                # Number of rolls
                self.materials["n_rolls"][i_mat] = (
                    self.materials["total_mass_w_waste"][i_mat] / self.materials["roll_mass"][i_mat]
                )
                # Loading and Machine Prep
                self.load_roll["labor_per_mat"].append(
                    self.load_roll["unit_time"] * self.materials["n_rolls"][i_mat] * self.load_roll["n_pers"]
                )
                self.load_roll["ct_per_mat"].append(self.load_roll["unit_time"] * self.materials["n_rolls"][i_mat])
                # Cutting
                cutting_labor = (
                    self.materials["total_ply_area_w_waste"][i_mat]
                    / self.cutting["machine_rate"]
                    * self.cutting["n_pers"]
                )
                cutting_ct = self.materials["total_ply_area_w_waste"][i_mat] / self.cutting["machine_rate"]
                self.cutting["labor_per_mat"].append(cutting_labor)
                self.cutting["ct_per_mat"].append(cutting_ct)
                # Kitting
                self.kitting["labor_per_mat"].append(cutting_ct * self.kitting["n_pers"])
                self.kitting["ct_per_mat"].append(0)
                # Clean-up
                cleaning_labor = (
                    self.materials["total_ply_area_w_waste"][i_mat] - self.materials["total_ply_area_wo_waste"][i_mat]
                ) / self.clean_up["clean_rate"]
                self.clean_up["labor_per_mat"].append(cleaning_labor)
                self.clean_up["ct_per_mat"].append(cleaning_labor / self.clean_up["n_pers"])

            else:
                self.load_roll["labor_per_mat"].append(0)
                self.load_roll["ct_per_mat"].append(0)
                self.cutting["labor_per_mat"].append(0)
                self.cutting["ct_per_mat"].append(0)
                self.kitting["labor_per_mat"].append(0)
                self.kitting["ct_per_mat"].append(0)
                self.clean_up["labor_per_mat"].append(0)
                self.clean_up["ct_per_mat"].append(0)

        # Loading and Machine Prep
        self.load_roll["labor"] = sum(self.load_roll["labor_per_mat"])
        self.load_roll["ct"] = sum(self.load_roll["ct_per_mat"])
        # Cutting
        self.cutting["labor"] = sum(self.cutting["labor_per_mat"])
        self.cutting["ct"] = sum(self.cutting["ct_per_mat"])
        # Kitting
        self.kitting["labor"] = sum(self.kitting["labor_per_mat"])
        self.kitting["ct"] = sum(self.kitting["ct_per_mat"])
        # Clean-up
        self.clean_up["labor"] = sum(self.clean_up["labor_per_mat"])
        self.clean_up["ct"] = sum(self.clean_up["ct_per_mat"])

        # Remove material data structure
        del self.materials


class material_cutting_labor(material_cutting_process):
    def __init__(self, material_parameters, process={}):

        # # Material cutting - process parameters
        self.load_roll = {}
        self.cutting = {}
        self.kitting = {}
        self.clean_up = {}
        self.materials = {}

        # Load roll
        self.load_roll["n_pers"] = 2.0  # Number of personnel involved in the operation
        self.load_roll["unit_time"] = 15.0 / 60.0  # Unit time - 15 minutes [hr]
        # Cutting
        self.cutting["n_pers"] = 2.0  # Number of personnel involved in the operation
        self.cutting["machine_rate"] = 833.0 * 0.9144 ** 2  # Machine rate - 833 [yd2/hr]
        # Kitting
        self.kitting["n_pers"] = 2.0  # Number of personnel involved in the operation
        # Clean up
        self.clean_up["n_pers"] = 2.0  # Number of personnel involved in the operation
        self.clean_up["clean_rate"] = 720.0 * 0.9144 ** 2.0  # Clean rate - 720 [yd2/hr]

        self.materials = material_parameters

        for var in process.keys():
            setattr(self, var, process[var])


class infusion_process(object):
    def manufacturing_steps(self, core=False, Extra_Operations_Skin=False, trim_excess=True):
        # Preparation of the tools
        self.tool_prep["labor"], self.tool_prep["ct"] = compute_labor_ct(
            self.tool_prep["n_pers"],
            self.tool_prep["area"],
            self.tool_prep["ri_appl_rate"] * self.tool_prep["n_pulls"],
            0,
            0,
        )

        # Lay-up of the composite fabric
        self.lay_up["labor"], self.lay_up["ct"] = compute_labor_ct(
            self.lay_up["n_pers"], self.lay_up["fabric2lay"], self.lay_up["rate"], 0, 0
        )

        # Extra operations
        if Extra_Operations_Skin:
            #  Insert the preformed root in the mold
            self.insert_root["labor"], self.insert_root["ct"] = compute_labor_ct(
                self.insert_root["n_pers"], 0, 1, self.insert_root["time"], 1
            )

            #  Insert the spar caps in the mold
            if self.insert_sc["sc_length"] <= 30.0:
                self.insert_sc["labor"] = self.insert_sc["time"] * self.insert_sc["n_pers"]
            else:
                self.insert_sc["labor"] = (
                    self.insert_sc["time"] + (self.insert_sc["sc_length"] - 30.0) * self.insert_sc["rate"]
                ) * self.insert_sc["n_pers"]
            self.insert_sc["ct"] = self.insert_sc["labor"] / self.insert_sc["n_pers"]

            #  Insert the root layers in the mold
            self.layup_root_layers["labor"], self.layup_root_layers["ct"] = compute_labor_ct(
                self.layup_root_layers["n_pers"],
                self.layup_root_layers["n_plies"],
                self.layup_root_layers["rate"],
                0,
                0,
            )

            #  Insert the trailing edge reinforcement layers in the mold
            self.insert_TE_layers["labor"], self.insert_TE_layers["ct"] = compute_labor_ct(
                self.insert_TE_layers["n_pers"], self.insert_TE_layers["length"], self.insert_TE_layers["rate"], 0, 0
            )

            #  Insert the leading edge reinforcement layers in the mold
            self.insert_LE_layers["labor"], self.insert_LE_layers["ct"] = compute_labor_ct(
                self.insert_LE_layers["n_pers"], self.insert_LE_layers["length"], self.insert_LE_layers["rate"], 0, 0
            )

            #  Insert the inner layers in the mold
            self.insert_inner_layers["labor"], self.insert_inner_layers["ct"] = compute_labor_ct(
                self.insert_inner_layers["n_pers"],
                self.insert_inner_layers["fabric2lay"],
                self.insert_inner_layers["rate"],
                0,
                0,
            )

        # Core placement
        if core:
            self.core_placement["labor"], self.core_placement["ct"] = compute_labor_ct(
                self.core_placement["n_pers"], self.core_placement["area"], self.core_placement["rate"], 0, 0
            )
        else:
            self.core_placement["labor"] = 0.0
            self.core_placement["ct"] = 0.0

        # Application of the peel-ply
        self.peel_ply["labor"], self.peel_ply["ct"] = compute_labor_ct(
            self.peel_ply["n_pers"], self.peel_ply["area"], self.peel_ply["rate"], 0, 0
        )

        # Application of the flow medium
        self.fm_app["labor"], self.fm_app["ct"] = compute_labor_ct(
            self.fm_app["n_pers"], self.peel_ply["area"] * self.fm_app["coverage"], self.fm_app["rate"], 0, 0
        )

        # Application of the feed lines
        self.feed["labor"], self.feed["ct"] = compute_labor_ct(
            self.feed["n_pers"], self.feed["length"], self.feed["rate"], 0, 0
        )

        # Application of vacuum lines
        self.vacuum_line["labor"], self.vacuum_line["ct"] = compute_labor_ct(
            self.vacuum_line["n_pers"], self.vacuum_line["length"], self.vacuum_line["rate"], 0, 0
        )

        # Application of the tack-tape
        self.tack_tape["labor"], self.tack_tape["ct"] = compute_labor_ct(
            self.tack_tape["n_pers"], self.tack_tape["length"], self.tack_tape["rate"], 0, 0
        )

        # Application of the vacuum bag
        self.vacuum_bag["labor"], self.vacuum_bag["ct"] = compute_labor_ct(
            self.vacuum_bag["n_pers"], self.peel_ply["area"], self.vacuum_bag["rate"], 0, 0
        )

        # Pull of full vacuuum
        self.vacuum_pull["labor"], self.vacuum_pull["ct"] = compute_labor_ct(
            self.vacuum_pull["n_pers"], self.peel_ply["area"], self.vacuum_pull["rate"], 0, 1
        )

        # Check of vacuum leaks
        self.leak_chk["labor"], self.leak_chk["ct"] = compute_labor_ct(
            self.leak_chk["n_pers"], self.tack_tape["length"], self.leak_chk["rate"], 0, 0
        )

        # Actual infusion
        self.infusion["labor"], self.infusion["ct"] = compute_labor_ct(
            self.infusion["n_pers"], 0, 1, self.infusion["time"], 1
        )

        # Curing
        self.cure["labor"], self.cure["ct"] = compute_labor_ct(self.cure["n_pers"], 0, 1, self.cure["time"], 1)

        # Debagging operations
        self.debag["labor"], self.debag["ct"] = compute_labor_ct(
            self.debag["n_pers"], self.debag["area"], self.debag["rate"], 0, 0
        )

        # Demolding operations
        self.demold["labor"], self.demold["ct"] = compute_labor_ct(self.demold["n_pers"], 0, 1, self.demold["time"], 0)

        if trim_excess:
            #  Trim (cut) of the excess fiberglass off of the root preform edges
            self.insert_prep_trim["labor"], self.insert_prep_trim["ct"] = compute_labor_ct(
                self.insert_prep_trim["n_pers"], self.insert_prep_trim["length"], self.insert_prep_trim["rate"], 0, 0
            )
        else:
            self.insert_prep_trim["labor"] = 0.0
            self.insert_prep_trim["ct"] = 0.0


class root_preform_labor(infusion_process):
    def __init__(self, component_parameters, process={}):

        # Manufacturing process labor input data for a root preform
        self.tool_prep = {}
        self.lay_up = {}
        self.core_placement = {}
        self.peel_ply = {}
        self.fm_app = {}
        self.feed = {}
        self.vacuum_line = {}
        self.tack_tape = {}
        self.vacuum_bag = {}
        self.vacuum_pull = {}
        self.leak_chk = {}
        self.infusion = {}
        self.cure = {}
        self.debag = {}
        self.demold = {}
        self.insert_prep_trim = {}

        # Tool preparation
        self.tool_prep["n_pers"] = 2.0  # Number of personnel involved in the operation
        self.tool_prep["n_pulls"] = 5.0  # Number of pulls [-]
        self.tool_prep["ri_appl_rate"] = 12.0  # "Rls appl rate per pn [m2/hr/man]
        self.tool_prep["area"] = component_parameters["area"]  # Area to be prepared
        # Lay-up
        self.lay_up["n_pers"] = 2.0  # Number of personnel involved in the operation
        self.lay_up["fabric2lay"] = component_parameters["fabric2lay"]
        self.lay_up["rate"] = 8.0  # Rate to lay composite [m2/hr/man]
        # Infusion preparation
        # Application of the peel ply
        self.peel_ply["n_pers"] = 2.0  # Number of personnel involved in the operation
        self.peel_ply["area"] = component_parameters["area"]  # Area where to apply peel-ply
        self.peel_ply["rate"] = 20.0  # Peel-ply application rate [m2/hr/man]
        # Application of the flow medium
        self.fm_app["n_pers"] = 2.0  # Number of personnel involved in the operation
        self.fm_app["rate"] = 10.0  # Flow-medium application rate [m2/hr/man]
        self.fm_app["coverage"] = 0.85  # Flow-medium coverage [% dec]
        # Installation feeding line
        self.feed["n_pers"] = 2.0  # Number of personnel involved in the operation
        self.feed["spacing"] = 0.5  # Spanwise spacing of the radial feed lines [m]
        self.feed["length"] = (
            component_parameters["length"]
            + component_parameters["half_circum"] * component_parameters["length"] / self.feed["spacing"]
        )
        self.feed["rate"] = 7.5  # Feed application rate [m/hr/man]
        # Vacuum line application
        self.vacuum_line["n_pers"] = 1.0  # Number of personnel involved in the operation
        self.vacuum_line["rate"] = 20.0  # Vacuum line application rate [m/hr/man]
        self.vacuum_line["length"] = 2.0 * component_parameters["length"]  # Vacuum line length [m]
        # Application tack tape
        self.tack_tape["n_pers"] = 1.0  # Number of personnel involved in the operation
        self.tack_tape["length"] = (
            2.0 * component_parameters["length"] + 2.0 * component_parameters["half_circum"]
        )  # Tack tape length [m]
        self.tack_tape["rate"] = 90.0  # Tack tape application rate [m/hr/man]
        # Application vacuum bag
        self.vacuum_bag["n_pers"] = 2.0  # Number of personnel involved in the operation
        self.vacuum_bag["rate"] = 7.5  # Vacuum bag application rate [m2/hr/man]
        # Pull full vacuum
        self.vacuum_pull["n_pers"] = 2.0  # Number of personnel involved in the operation
        self.vacuum_pull["rate"] = 45.0  # Vacuum pull rate [m2/hr/man]
        # Check for leackages
        self.leak_chk["n_pers"] = 2  # Number of personnel involved in the operation
        self.leak_chk["rate"] = 30.0  # Leak_chk rate [m/hr/man]
        # Infusion
        self.infusion["n_pers"] = 1.0  # Number of personnel involved in the operation
        p0 = 15.0  # p0 of the polynomial fit
        p1 = 1.0  # p1 of the polynomial fit
        p2 = 0.0  # p2 of the polynomial fit
        p3 = 0.0  # p3 of the polynomial fit
        self.infusion["time"] = (
            p0
            + p1 * component_parameters["blade_length"]
            + p2 * component_parameters["blade_length"] ** 2
            + p3 * component_parameters["blade_length"] ** 3
        ) / 60.0  # Infusion time [hr]
        # Cure
        self.cure["n_pers"] = 1.0  # Number of personnel involved in the operation
        self.cure["time"] = 2.0  # Curing time [hr]
        # Debag
        self.debag["n_pers"] = 2.0  # Number of personnel involved in the operation
        self.debag["area"] = component_parameters["area"]  # Area to be debagged
        self.debag["rate"] = 20.0  # Debag rate [m2/hr/man]
        # Demold
        self.demold["n_pers"] = 2.0  # Number of personnel involved in the operation
        self.demold["time"] = 0.5  # Demold time [hr]
        # Insertion preparation and trim
        self.insert_prep_trim["n_pers"] = 2.0  # Number of personnel involved in the operation
        self.insert_prep_trim["length"] = 2.0 * component_parameters["length"]
        self.insert_prep_trim["rate"] = 6.0  # Trim rate [m/hr/man]

        for var in process.keys():
            setattr(self, var, process[var])


class shearweb_labor(infusion_process):
    def __init__(self, component_parameters, i_web, process={}):

        # Manufacturing process labor input data for shear webs
        self.tool_prep = {}
        self.lay_up = {}
        self.core_placement = {}
        self.peel_ply = {}
        self.fm_app = {}
        self.feed = {}
        self.vacuum_line = {}
        self.tack_tape = {}
        self.vacuum_bag = {}
        self.vacuum_pull = {}
        self.leak_chk = {}
        self.infusion = {}
        self.cure = {}
        self.debag = {}
        self.demold = {}
        self.insert_prep_trim = {}

        # Tool preparation
        self.tool_prep["n_pers"] = 10.0  # Number of personnel involved in the operation
        self.tool_prep["n_pulls"] = 5.0  # Number of pulls [-]
        self.tool_prep["ri_appl_rate"] = 12.0  # "Rls appl rate per pn [m2/hr/man]
        self.tool_prep["area"] = component_parameters["area"][i_web]  # Area to be prepared
        # Lay-up
        self.lay_up["n_pers"] = 10.0  # Number of personnel involved in the operation
        self.lay_up["fabric2lay"] = component_parameters["fabric2lay"][i_web]
        self.lay_up["rate"] = 24.0  # Rate to lay composite [m2/hr/man]
        # Core
        self.core_placement["area"] = component_parameters["core_area"][i_web]  # Area with sandwich core
        self.core_placement[
            "n_pers"
        ] = 10.0  # Number of personnel involved in the operation - Ignored if the core_area is set to 0
        self.core_placement["rate"] = 12.0  # Rate of core lay [m2/hr/man] - Ignored if the core_area is set to 0
        # Infusion preparation
        # Application of the peel ply
        self.peel_ply["n_pers"] = 10.0  # Number of personnel involved in the operation
        self.peel_ply["area"] = component_parameters["area"][i_web]  # Area where to apply peel-ply
        self.peel_ply["area"] = component_parameters["area"][i_web]
        self.peel_ply["rate"] = 20.0  # Peel-ply application rate [m2/hr/man]
        # Application of the flow medium
        self.fm_app["n_pers"] = 10.0  # Number of personnel involved in the operation
        self.fm_app["rate"] = 10.0  # Flow-medium application rate [m2/hr/man]
        self.fm_app["coverage"] = 0.85  # Flow-medium coverage [% dec]
        # Installation feeding line
        self.feed["n_pers"] = 10.0  # Number of personnel involved in the operation
        self.feed["length"] = component_parameters["length"][i_web]
        self.feed["rate"] = 7.5  # Feed application rate [m/hr/man]
        # Vacuum line application
        self.vacuum_line["n_pers"] = 5.0  # Number of personnel involved in the operation
        self.vacuum_line["rate"] = 20.0  # Vacuum line application rate [m/hr/man]
        self.vacuum_line["length"] = 2.0 * component_parameters["length"][i_web]  # Vacuum line length [m]
        # Application tack tape
        self.tack_tape["n_pers"] = 5.0  # Number of personnel involved in the operation
        self.tack_tape["length"] = (
            2.0 * component_parameters["length"][i_web]
            + component_parameters["height1"][i_web]
            + component_parameters["height2"][i_web]
        )  # Tack tape length [m]
        self.tack_tape["rate"] = 90.0  # Tack tape application rate [m/hr/man]
        # Application vacuum bag
        self.vacuum_bag["n_pers"] = 10.0  # Number of personnel involved in the operation
        self.vacuum_bag["rate"] = 7.5  # Vacuum bag application rate [m2/hr/man]
        # Pull full vacuum
        self.vacuum_pull["n_pers"] = 10.0  # Number of personnel involved in the operation
        self.vacuum_pull["rate"] = 45.0  # Vacuum pull rate [m2/hr/man]
        # Check for leackages
        self.leak_chk["n_pers"] = 10.0  # Number of personnel involved in the operation
        self.leak_chk["rate"] = 30.0  # Leak_chk rate [m/hr/man]
        # Infusion
        self.infusion["n_pers"] = 1.0  # Number of personnel involved in the operation
        p0 = 11.983  # p0 of the polynomial fit
        p1 = 0.3784  # p1 of the polynomial fit
        p2 = 0.0  # p2 of the polynomial fit
        p3 = 0.0  # p3 of the polynomial fit
        self.infusion["time"] = (
            p0
            + p1 * component_parameters["blade_length"]
            + p2 * component_parameters["blade_length"] ** 2
            + p3 * component_parameters["blade_length"] ** 3
        ) / 60.0  # Infusion time [hr]
        # Cure
        self.cure["n_pers"] = 1.0  # Number of personnel involved in the operation
        self.cure["time"] = 2.0  # Curing time [hr]
        # Debag
        self.debag["n_pers"] = 10.0  # Number of personnel involved in the operation
        self.debag["area"] = component_parameters["area"][i_web]  # Area to be debagged
        self.debag["rate"] = 20.0  # Debag rate [m2/hr/man]
        # Demold
        self.demold["n_pers"] = 10.0  # Number of personnel involved in the operation
        self.demold["time"] = 0.5  # Demold time [hr]
        # Insertion preparation and trim
        self.insert_prep_trim["n_pers"] = 10.0  # Number of personnel involved in the operation
        self.insert_prep_trim["length"] = component_parameters["length"][i_web]
        self.insert_prep_trim["rate"] = 30.0  # Trim rate [m/hr/man]

        for var in process.keys():
            setattr(self, var, process[var])


class sparcaps_labor(infusion_process):
    def __init__(self, component_parameters, process={}):

        # Manufacturing process labor input data for the spar caps
        self.tool_prep = {}
        self.lay_up = {}
        self.core_placement = {}
        self.peel_ply = {}
        self.fm_app = {}
        self.feed = {}
        self.vacuum_line = {}
        self.tack_tape = {}
        self.vacuum_bag = {}
        self.vacuum_pull = {}
        self.leak_chk = {}
        self.infusion = {}
        self.cure = {}
        self.debag = {}
        self.demold = {}
        self.insert_prep_trim = {}

        # Tool preparation
        self.tool_prep["n_pers"] = 10.0  # Number of personnel involved in the operation
        self.tool_prep["n_pulls"] = 5.0  # Number of pulls [-]
        self.tool_prep["ri_appl_rate"] = 12.0  # "Rls appl rate per pn [m2/hr/man]
        self.tool_prep["area"] = component_parameters["area_wflanges"]  # Area to be prepared
        # Lay-up
        self.lay_up["n_pers"] = 10.0  # Number of personnel involved in the operation
        self.lay_up["fabric2lay"] = component_parameters["fabric2lay"]
        self.lay_up["rate"] = 110.0  # Rate to lay composite [m/hr/man]
        # Infusion preparation
        # Application of the peel ply
        self.peel_ply["n_pers"] = 10.0  # Number of personnel involved in the operation
        self.peel_ply["area"] = component_parameters["area"]  # Area where to apply peel-ply
        self.peel_ply["rate"] = 20.0  # Peel-ply application rate [m2/hr/man]
        # Application of the flow medium
        self.fm_app["n_pers"] = 10.0  # Number of personnel involved in the operation
        self.fm_app["rate"] = 10.0  # Flow-medium application rate [m2/hr/man]
        self.fm_app["coverage"] = 0.85  # Flow-medium coverage [% dec]
        # Installation feeding line
        self.feed["n_pers"] = 10.0  # Number of personnel involved in the operation
        self.feed["length"] = component_parameters["length"]
        self.feed["rate"] = 7.5  # Feed application rate [m/hr/man]
        # Vacuum line application
        self.vacuum_line["n_pers"] = 5.0  # Number of personnel involved in the operation
        self.vacuum_line["rate"] = 20.0  # Vacuum line application rate [m/hr/man]
        self.vacuum_line["length"] = component_parameters["length"]  # Vacuum line length [m]
        # Application tack tape
        self.tack_tape["n_pers"] = 5.0  # Number of personnel involved in the operation
        self.tack_tape["length"] = (
            2.0 * component_parameters["length"] + 2.0 * component_parameters["width"]
        )  # Tack tape length [m]
        self.tack_tape["rate"] = 90.0  # Tack tape application rate [m/hr/man]
        # Application vacuum bag
        self.vacuum_bag["n_pers"] = 10.0  # Number of personnel involved in the operation
        self.vacuum_bag["rate"] = 7.5  # Vacuum bag application rate [m2/hr/man]
        # Pull full vacuum
        self.vacuum_pull["n_pers"] = 10.0  # Number of personnel involved in the operation
        self.vacuum_pull["rate"] = 45.0  # Vacuum pull rate [m2/hr/man]
        # Check for leackages
        self.leak_chk["n_pers"] = 10.0  # Number of personnel involved in the operation
        self.leak_chk["rate"] = 30.0  # Leak_chk rate [m/hr/man]
        # Infusion
        self.infusion["n_pers"] = 1.0  # Number of personnel involved in the operation
        p0 = 23.992  # p0 of the polynomial fit
        p1 = 0.0037  # p1 of the polynomial fit
        p2 = 0.0089  # p2 of the polynomial fit
        p3 = 0.0  # p3 of the polynomial fit
        self.infusion["time"] = (
            p0
            + p1 * component_parameters["blade_length"]
            + p2 * component_parameters["blade_length"] ** 2
            + p3 * component_parameters["blade_length"] ** 3
        ) / 60.0  # Infusion time [hr]
        # Cure
        self.cure["n_pers"] = 1.0  # Number of personnel involved in the operation
        self.cure["time"] = 2.0  # Curing time [hr]
        # Debag
        self.debag["n_pers"] = 10.0  # Number of personnel involved in the operation
        self.debag["area"] = component_parameters["area"]  # Area to be debagged
        self.debag["rate"] = 20.0  # Debag rate [m2/hr/man]
        # Demold
        self.demold["n_pers"] = 10.0  # Number of personnel involved in the operation
        self.demold["time"] = 0.5  # Demold time [hr]
        # Insertion preparation and trim
        self.insert_prep_trim["n_pers"] = 10.0  # Number of personnel involved in the operation
        self.insert_prep_trim["length"] = component_parameters["length"]
        self.insert_prep_trim["rate"] = 30.0  # Trim rate [m/hr/man]

        for var in process.keys():
            setattr(self, var, process[var])


class lphp_skin_labor(infusion_process):
    def __init__(self, component_parameters, team_size, process={}):

        # Manufacturing process labor input data for the low pressure and high pressure skins
        self.tool_prep = {}
        self.lay_up = {}
        self.insert_root = {}
        self.insert_sc = {}
        self.layup_root_layers = {}
        self.core_placement = {}
        self.insert_TE_layers = {}
        self.insert_LE_layers = {}
        self.insert_inner_layers = {}
        self.peel_ply = {}
        self.fm_app = {}
        self.feed = {}
        self.vacuum_line = {}
        self.tack_tape = {}
        self.vacuum_bag = {}
        self.vacuum_pull = {}
        self.leak_chk = {}
        self.infusion = {}
        self.cure = {}
        self.debag = {}
        self.demold = {}
        self.insert_prep_trim = {}

        # Tool preparation
        self.tool_prep["n_pers"] = team_size  # Number of personnel involved in the operation
        self.tool_prep["n_pulls"] = 5.0  # Number of pulls [-]
        self.tool_prep["ri_appl_rate"] = 12.0  # "Rls appl rate per pn [m2/hr/man]
        self.tool_prep["area"] = component_parameters["area_wflanges"]  # Area to be prepared
        # Lay-up
        self.lay_up["n_pers"] = team_size  # Number of personnel involved in the operation
        self.lay_up["fabric2lay"] = component_parameters["fabric2lay"]
        self.lay_up["rate"] = 30.0  # Rate to lay composite [m/hr/man]
        # Insert the preformed root
        self.insert_root["n_pers"] = 0.25 * team_size  # Number of personnel involved in the operation
        self.insert_root["time"] = 0.25  # Root Preform Insertion Time [hr]
        # Insert the pre-fabricated spar caps
        self.insert_sc["n_pers"] = 0.25 * team_size  # Number of personnel involved in the operation
        self.insert_sc["sc_length"] = component_parameters["sc_length"]
        self.insert_sc["time"] = 0.25  # Time to insert spar caps shorter than 30 meters [hr]
        self.insert_sc["rate"] = 0.00833  # Addsitional time per meter to insert spar caps longer than 30 meters [hr/m]
        # Layup of the root plies
        self.layup_root_layers["n_pers"] = 0.25 * team_size  # Number of personnel involved in the operation
        self.layup_root_layers["n_plies"] = component_parameters["n_root_plies"]
        self.layup_root_layers["rate"] = 16.0  # Root layup rate
        # Core placement
        self.core_placement["n_pers"] = (
            0.75 * team_size
        )  # Number of personnel involved in the operation - Ignored if the core_area is set to 0
        self.core_placement["area"] = component_parameters["core_area"]  # Area with core [m2]
        self.core_placement["rate"] = 12.0  # Rate of core lay [m2/hr/man] - Ignored if the core_area is set to 0
        # Layup of the trailing edge reinforcement
        self.insert_TE_layers["n_pers"] = 0.25 * team_size  # Number of personnel involved in the operation
        self.insert_TE_layers["length"] = component_parameters[
            "total_TE"
        ]  # Length of the layers of trailing edge reinforcement
        self.insert_TE_layers["rate"] = 96.0  # TE reinforcement layup rate
        # Layup of the leading edge reinforcement
        self.insert_LE_layers["n_pers"] = team_size  # Number of personnel involved in the operation
        self.insert_LE_layers["length"] = component_parameters[
            "total_LE"
        ]  # Length of the layers of leading edge reinforcement
        self.insert_LE_layers["rate"] = 96.0  # LE reinforcement layup rate
        # Layup of the inner layers
        self.insert_inner_layers["n_pers"] = team_size  # Number of personnel involved in the operation
        self.insert_inner_layers["fabric2lay"] = component_parameters["fabric2lay_inner"]
        self.insert_inner_layers["rate"] = 30.0  # Inner layers reinforcement layup rate
        # Infusion preparation
        # Application of the peel ply
        self.peel_ply["n_pers"] = team_size  # Number of personnel involved in the operation
        self.peel_ply["area"] = component_parameters["area"]  # Area where to apply peel-ply
        self.peel_ply["rate"] = 22.5  # Peel-ply application rate [m2/hr/man]
        # Application of the flow medium
        self.fm_app["n_pers"] = team_size  # Number of personnel involved in the operation
        self.fm_app["rate"] = 10.0  # Flow-medium application rate [m2/hr/man]
        self.fm_app["coverage"] = 0.85  # Flow-medium coverage [% dec]
        # Installation feeding line
        self.feed["n_pers"] = team_size  # Number of personnel involved in the operation
        self.feed["spacing"] = 0.5  # Spanwise spacing of the radial feed lines [m]
        self.feed["length"] = (
            2 * component_parameters["root_sect_length"]
            + component_parameters["root_half_circumf"]
            * component_parameters["root_sect_length"]
            / self.feed["spacing"]
            + component_parameters["length"]
            - 2
            + 4 * 0.65 * component_parameters["length"]
        )  # Vacuum line length [m]
        self.feed["rate"] = 10.0  # Feed application rate [m/hr/man]
        # Vacuum line application
        self.vacuum_line["n_pers"] = 0.5 * team_size  # Number of personnel involved in the operation
        self.vacuum_line["rate"] = 20.0  # Vacuum line application rate [m/hr/man]
        self.vacuum_line["length"] = component_parameters["perimeter_noroot"]  # Vacuum line length [m]
        # Application tack tape
        self.tack_tape["n_pers"] = 0.5 * team_size  # Number of personnel involved in the operation
        self.tack_tape["length"] = component_parameters["perimeter"]  # Tack tape length [m]
        self.tack_tape["rate"] = 360.0  # Tack tape application rate [m/hr/man]
        # Application vacuum bag
        self.vacuum_bag["n_pers"] = team_size  # Number of personnel involved in the operation
        self.vacuum_bag["rate"] = 7.5  # Vacuum bag application rate [m2/hr/man]
        # Pull full vacuum
        self.vacuum_pull["n_pers"] = team_size  # Number of personnel involved in the operation
        self.vacuum_pull["rate"] = 360.0  # Vacuum pull rate [m2/hr/man]
        # Check for leackages
        self.leak_chk["n_pers"] = team_size  # Number of personnel involved in the operation
        self.leak_chk["rate"] = 180.0  # Leak_chk rate [m/hr/man]
        # Infusion
        self.infusion["n_pers"] = 3.0  # Number of personnel involved in the operation
        p0 = 15.972  # p0 of the polynomial fit
        p1 = 3.1484  # p1 of the polynomial fit
        p2 = -0.0568  # p2 of the polynomial fit
        p3 = 0.0004  # p3 of the polynomial fit
        self.infusion["time"] = (
            p0
            + p1 * component_parameters["blade_length"]
            + p2 * component_parameters["blade_length"] ** 2
            + p3 * component_parameters["blade_length"] ** 3
        ) / 60.0  # Infusion time [hr]
        # Cure
        self.cure["n_pers"] = 1.0  # Number of personnel involved in the operation
        self.cure["time"] = 3.0  # Curing time [hr]
        # Debag
        self.debag["n_pers"] = team_size  # Number of personnel involved in the operation
        self.debag["area"] = component_parameters["area"]  # Area to be debagged
        self.debag["rate"] = team_size  # Debag rate [m2/hr/man]
        # Demold
        self.demold["n_pers"] = team_size  # Number of personnel involved in the operation
        self.demold["time"] = 0.0  # Demold time [hr]

        for var in process.keys():
            setattr(self, var, process[var])


class assembly_process(object):
    def assembly_steps(self):
        # Remove non-sand / prep (LP)
        self.remove_nonsand_prep_lp["labor"], self.remove_nonsand_prep_lp["ct"] = compute_labor_ct(
            self.remove_nonsand_prep_lp["n_pers"],
            self.remove_nonsand_prep_lp["length"],
            self.remove_nonsand_prep_lp["rate"],
            0,
            0,
        )
        # Remove non-sand / prep (HP)
        self.remove_nonsand_prep_hp["labor"], self.remove_nonsand_prep_hp["ct"] = compute_labor_ct(
            self.remove_nonsand_prep_hp["n_pers"],
            self.remove_nonsand_prep_hp["length"],
            self.remove_nonsand_prep_hp["rate"],
            0,
            0,
        )
        # Insert SWs into fixture
        self.insert_sw["labor"], self.insert_sw["ct"] = compute_labor_ct(
            self.insert_sw["n_pers"], self.insert_sw["length"], 1, 0, 1
        )
        # Dry fit - boundary mark
        self.dry_fit["labor"], self.dry_fit["ct"] = compute_labor_ct(
            self.dry_fit["n_pers"], self.dry_fit["length"], self.dry_fit["rate"], 0, 0
        )
        # Dispense lower adhesive
        self.low_adhesive["labor"], self.low_adhesive["ct"] = compute_labor_ct(
            self.low_adhesive["n_pers"], self.low_adhesive["length"], self.low_adhesive["rate"], 0, 0
        )
        # Bond SWs - lower
        self.bond_sw_low["labor"], self.bond_sw_low["ct"] = compute_labor_ct(
            self.bond_sw_low["n_pers"], 0, 1, self.bond_sw_low["time"], 1
        )
        # Fillet SW bonds
        self.fillet_sw_low["labor"], self.fillet_sw_low["ct"] = compute_labor_ct(
            self.fillet_sw_low["n_pers"], self.fillet_sw_low["length"], self.fillet_sw_low["rate"], 0, 0
        )
        # Cure - lower adhesive
        self.cure_low["labor"] = 0.0
        self.cure_low["ct"] = self.cure_low["time"]
        # Remove fixture
        self.remove_fixture["labor"], self.remove_fixture["ct"] = compute_labor_ct(
            self.remove_fixture["n_pers"], 0, 1, self.remove_fixture["time"], 1
        )
        # Dry fit - upper
        self.dry_fit_up["labor"], self.dry_fit_up["ct"] = compute_labor_ct(
            self.dry_fit_up["n_pers"], self.dry_fit_up["length"], self.dry_fit_up["rate"], 0, 0
        )
        self.dry_fit_up["ct"] = self.dry_fit_up["ct"] + 2 * self.close_mold["time"]
        # (SW height rework)
        self.sw_height_rework["labor"] = 0
        self.sw_height_rework["ct"] = 0
        # Dispense upper/perim adhesive
        self.up_adhesive["labor"], self.up_adhesive["ct"] = compute_labor_ct(
            self.up_adhesive["n_pers"], self.up_adhesive["length"], self.up_adhesive["rate"], 0.0, 0
        )
        # Close mold
        self.close_mold["labor"], self.close_mold["ct"] = compute_labor_ct(
            self.close_mold["n_pers"], 0, 1, self.close_mold["time"], 1
        )
        # Install shear clips
        self.shear_clips["labor"], self.shear_clips["ct"] = compute_labor_ct(
            self.shear_clips["n_pers"], self.shear_clips["length"], self.shear_clips["rate"], 0.0, 0
        )
        # Cure - entire
        self.cure_entire["labor"] = 0.0
        self.cure_entire["ct"] = self.cure_entire["time"]
        # Open mold
        self.open_mold["labor"], self.open_mold["ct"] = compute_labor_ct(
            self.open_mold["n_pers"], 0, 1, self.open_mold["time"], 1
        )


class assembly_labor(assembly_process):
    def __init__(self, blade_parameters, team_size, process={}):
        # # Assembly labor
        self.remove_nonsand_prep_lp = {}
        self.remove_nonsand_prep_hp = {}
        self.insert_sw = {}
        self.dry_fit = {}
        self.low_adhesive = {}
        self.bond_sw_low = {}
        self.fillet_sw_low = {}
        self.cure_low = {}
        self.remove_fixture = {}
        self.dry_fit_up = {}
        self.sw_height_rework = {}
        self.up_adhesive = {}
        self.close_mold = {}
        self.shear_clips = {}
        self.cure_entire = {}
        self.open_mold = {}

        # Remove non-sand / prep (LP)
        self.remove_nonsand_prep_lp["n_pers"] = team_size / 2.0  # Number of personnel involved in the operation
        self.remove_nonsand_prep_lp["length"] = (
            sum(blade_parameters["sw_length"]) + blade_parameters["perimeter_noroot"]
        )  # Length where to remove sand [m]
        self.remove_nonsand_prep_lp["rate"] = 60.0  # Rate of sand removal [m/hr/man]
        # Remove non-sand / prep (HP)
        self.remove_nonsand_prep_hp["n_pers"] = team_size / 2.0  # Number of personnel involved in the operation
        self.remove_nonsand_prep_hp["length"] = self.remove_nonsand_prep_lp["length"]  # Length where to remove sand [m]
        self.remove_nonsand_prep_hp["rate"] = 60.0  # Rate of sand removal [m/hr/man]
        # Insert SWs into fixture
        self.insert_sw["n_pers"] = team_size  # Number of personnel involved in the operation
        self.insert_sw["time"] = 0.25  # Time to install the shear web in the mold for webs shorter than 50 meters [hr]
        self.insert_sw[
            "rate"
        ] = 0.0167  # Extra time per meter to install the shear web in the mold for webs longer than 50 meters [hr/m]
        insert_sw_len = np.zeros(len(blade_parameters["sw_length"]))
        for i_web in range(len(blade_parameters["sw_length"])):  # Loop for all existing webs
            insert_sw_len[i_web] = self.insert_sw["time"] - (
                self.insert_sw["rate"] * (50.0 - blade_parameters["sw_length"][i_web])
            )
        self.insert_sw["length"] = sum(insert_sw_len)
        # Dry fit - boundary mark
        self.dry_fit["n_pers"] = team_size  # Number of personnel involved in the operation
        self.dry_fit["length"] = sum(blade_parameters["sw_length"])  # Length where to dry fit [m]
        self.dry_fit["rate"] = 60.0  # Rate of dry fit [m/hr/man]
        # Dispense lower adhesive
        self.low_adhesive["n_pers"] = team_size  # Number of personnel involved in the operation
        self.low_adhesive["length"] = sum(blade_parameters["sw_length"])  # Length where to dispose adhesive [m]
        self.low_adhesive["rate"] = 60.0  # Rate to dispose adhesive [m/hr/man]
        # Bond SWs - lower
        self.bond_sw_low["n_pers"] = team_size  # Number of personnel involved in the operation
        self.bond_sw_low["time"] = blade_parameters["n_webs"] * 0.25  # Time to bond the shear webs in the mold [hr]
        # Fillet SW bonds
        self.fillet_sw_low["n_pers"] = team_size  # Number of personnel involved in the operation
        self.fillet_sw_low["length"] = 2.0 * sum(blade_parameters["sw_length"])
        self.fillet_sw_low["rate"] = 180.0  # Rate to fillett the webs [fillet/hr/man]
        # Cure - lower adhesive
        self.cure_low["n_pers"] = 0.0  # Number of personnel involved in the operation
        self.cure_low["time"] = 2.0  # Time to cure the adhesive [hr]
        # Remove fixture
        self.remove_fixture["n_pers"] = team_size  # Number of personnel involved in the operation
        self.remove_fixture["time"] = 0.25  # Time to remove the fixture [hr]
        # Dry fit - upper
        self.dry_fit_up["n_pers"] = team_size  # Number of personnel involved in the operation
        self.dry_fit_up["time"] = 0.0833  # Time close or open the mold [hr]
        self.dry_fit_up["rate"] = 15.0  # Dry fit rate of the shear webs [m/hr/man]
        self.dry_fit_up["length"] = sum(blade_parameters["sw_length"])
        # (SW height rework)
        self.sw_height_rework["n_pers"] = 0.0  # Number of personnel involved in the operation
        # Dispense upper/perim adhesive
        self.up_adhesive["n_pers"] = team_size  # Number of personnel involved in the operation
        self.up_adhesive["length"] = sum(blade_parameters["sw_length"]) + blade_parameters["perimeter_noroot"]
        self.up_adhesive["rate"] = 60.0  # Rate to dispose adhesive [m/hr/man]
        # Close mold
        self.close_mold["n_pers"] = team_size  # Number of personnel involved in the operation
        self.close_mold["time"] = self.dry_fit_up["time"]  # Time close or open the mold [hr]
        # Install shear clips
        self.shear_clips["n_pers"] = team_size  # Number of personnel involved in the operation
        self.shear_clips["%vert"] = 50.0  # Vertical Fraction of the shear webs with shear clip coverage [%]
        self.shear_clips["%span"] = 10.0  # Spanwise fraction of the shear webs with shear clip coverage [%]
        self.shear_clips["rate"] = 4.0  # Shear clip lay rate [m/hr/man]
        self.shear_clips["length"] = (
            sum(blade_parameters["sw_length"]) * self.shear_clips["%vert"] * 4.0 / 100.0
            + blade_parameters["length"] * self.shear_clips["%span"] * 2.0 / 100.0
        )  # Length where to install shear clips [m]
        # Cure - entire
        self.cure_entire["n_pers"] = 0.0  # Number of personnel involved in the operation
        self.cure_entire["time"] = 2.0  # Curing time
        # Open mold
        self.open_mold["n_pers"] = 1.0  # Number of personnel involved in the operation
        self.open_mold["time"] = 0.0833  # Time close or open the mold [hr]

        for var in process.keys():
            setattr(self, var, process[var])


class demold_process(object):
    def demold_steps(self):

        # Cool-down period
        self.cool_down["ct"] = self.cool_down["time"]
        self.cool_down["labor"] = self.cool_down["ct"] * self.cool_down["n_pers"]
        # Placement of lift straps
        if self.lift_straps["length"] <= 40.0:
            self.lift_straps["ct"] = self.lift_straps["time"]
        else:
            self.lift_straps["ct"] = (
                self.lift_straps["time"] + (self.lift_straps["length"] - 40.0) * self.lift_straps["rate"]
            )
        self.lift_straps["labor"] = self.lift_straps["ct"] * self.lift_straps["n_pers"]
        # Transfer to blade cart
        if self.transfer2cart["length"] <= 60.0:
            self.transfer2cart["ct"] = self.transfer2cart["time"]
        else:
            self.transfer2cart["ct"] = (
                self.transfer2cart["time"] + (self.transfer2cart["length"] - 60.0) * self.transfer2cart["rate"]
            )
        self.transfer2cart["labor"] = self.transfer2cart["ct"] * self.transfer2cart["n_pers"]
        # Move blade to finishing area
        if self.move2finishing["length"] <= 60.0:
            self.move2finishing["ct"] = self.move2finishing["time"]
        else:
            self.move2finishing["ct"] = (
                self.move2finishing["time"] + (self.move2finishing["length"] - 60.0) * self.move2finishing["rate"]
            )
        self.move2finishing["labor"] = self.move2finishing["ct"] * self.move2finishing["n_pers"]


class demold_labor(demold_process):
    def __init__(self, blade_parameters, process={}):
        # # Demold labor
        self.cool_down = {}
        self.lift_straps = {}
        self.transfer2cart = {}
        self.move2finishing = {}

        # Cool-down period
        self.cool_down["n_pers"] = 0.0  # Number of personnel involved in the operation
        self.cool_down["time"] = 1.0  # Time to cool down [hr]
        # Placement of lift straps
        self.lift_straps["n_pers"] = 4.0  # Number of personnel involved in the operation
        self.lift_straps["length"] = blade_parameters["length"]  # Length of the blade [m]
        self.lift_straps["rate"] = 0.0083  # Premium lift straps for blades longer than 40 m [hr/m]
        self.lift_straps["time"] = 0.5  # Strap baseline [hr]
        # Transfer to blade cart
        self.transfer2cart["n_pers"] = 4.0  # Number of personnel involved in the operation
        self.transfer2cart["time"] = 0.25  # Time to move blade to the cart [hr]
        self.transfer2cart["length"] = blade_parameters["length"]  # Length of the blade [m]
        self.transfer2cart["rate"] = 0.0167  # Extra time per meter length needed to move blades longer than 60 m [hr/m]
        # Remove non-sand / prep (LP)
        self.move2finishing["n_pers"] = 4.0  # Number of personnel involved in the operation
        self.move2finishing["time"] = 0.1667  # Time to move blade to the cart [hr]
        self.move2finishing["length"] = blade_parameters["length"]  # Length of the blade [m]
        self.move2finishing[
            "rate"
        ] = 0.0056  # Extra time per meter length needed to move blades longer than 60 m [hr/m]

        for var in process.keys():
            setattr(self, var, process[var])


class trim_process(object):
    def trim_steps(self):

        # Move blade into trim booth
        self.move2trim["ct"] = self.move2trim["time"]
        self.move2trim["labor"] = self.move2trim["ct"] * self.move2trim["n_pers"]
        # Trim blade
        self.trim["labor"], self.trim["ct"] = compute_labor_ct(
            self.trim["n_pers"], self.trim["length"], self.trim["rate"], 0, 0
        )
        # Move blade out of trim booth
        self.move_out["ct"] = self.move_out["time"]
        self.move_out["labor"] = self.move_out["ct"] * self.move_out["n_pers"]


class trim_labor(trim_process):
    def __init__(self, blade_parameters, process={}):
        # # trim labor
        self.move2trim = {}
        self.trim = {}
        self.move_out = {}

        # Move blade into trim booth
        self.move2trim["n_pers"] = 3.0  # Number of personnel involved in the operation
        self.move2trim["time"] = 0.5  # Time to move the blade to the trim booth [hr]
        # Trim blade
        self.trim["n_pers"] = 6.0  # Number of personnel involved in the operation
        self.trim["length"] = blade_parameters["perimeter_noroot"]  # Length of the blade [m]
        self.trim["rate"] = 10.0  # Trim rate [m/hr/man]
        # Move blade out of trim booth
        self.move_out["n_pers"] = 3.0  # Number of personnel involved in the operation
        self.move_out["time"] = 0.5  # Time to move blade to out of the trim booth [hr]

        for var in process.keys():
            setattr(self, var, process[var])


class overlay_process(object):
    def overlay_steps(self):

        # Move blade to station
        self.move2station["ct"] = self.move2station["time"]
        self.move2station["labor"] = self.move2station["ct"] * self.move2station["n_pers"]
        # Rotate blade 90
        if self.rotate90deg["length"] <= 40.0:
            self.rotate90deg["ct"] = self.rotate90deg["time"]
        else:
            self.rotate90deg["ct"] = (
                self.rotate90deg["time"] + (self.rotate90deg["length"] - 40.0) * self.rotate90deg["rate"]
            )
        self.rotate90deg["labor"] = self.rotate90deg["ct"] * self.rotate90deg["n_pers"]
        # Place staging
        self.place_staging["ct"] = self.place_staging["time"]
        self.place_staging["labor"] = self.place_staging["ct"] * self.place_staging["n_pers"]
        # Repair over/under-bite
        self.repair["ct"] = 0.0
        self.repair["labor"] = 0.0
        # Install overlay
        self.inst_overlay["labor"] = self.inst_overlay["length"] / self.inst_overlay["rate"]
        self.inst_overlay["ct"] = self.inst_overlay["labor"] / self.inst_overlay["n_pers"]
        # Vacuum bag overlay
        self.vacuum_bag["ct"] = self.vacuum_bag["length"] / self.vacuum_bag["rate"]
        self.vacuum_bag["labor"] = self.vacuum_bag["ct"] * self.vacuum_bag["n_pers"]
        # Cure of overlay
        self.cure["ct"] = self.cure["time"]
        self.cure["labor"] = self.cure["ct"] * self.cure["n_pers"]
        # Remove vacuum bag
        self.remove_bag["ct"] = self.remove_bag["length"] / self.remove_bag["rate"]
        self.remove_bag["labor"] = self.remove_bag["ct"] * self.remove_bag["n_pers"]


class overlay_labor(overlay_process):
    def __init__(self, blade_parameters, process={}):
        # # Overlay labor
        self.move2station = {}
        self.rotate90deg = {}
        self.place_staging = {}
        self.repair = {}
        self.inst_overlay = {}
        self.vacuum_bag = {}
        self.cure = {}
        self.remove_bag = {}

        coverage = 20.0  # [%] Percentage of overlay coverage
        OL_ply = coverage / 100.0 * blade_parameters["length"]  # Longest available overlay ply [m]
        avg_taper = 0.5  # [m] Average taper on ends
        totalOL = 2.0 * (
            OL_ply
            + (OL_ply - (2.0 * avg_taper))
            + (OL_ply - (4.0 * avg_taper))
            + (OL_ply - (6.0 * avg_taper))
            + (OL_ply - (8.0 * avg_taper))
            + (OL_ply - (10.0 * avg_taper))
            + (OL_ply - (12.0 * avg_taper))
            + (OL_ply - (14.0 * avg_taper))
        )

        # Move blade to station
        self.move2station["n_pers"] = 3.0  # Number of personnel involved in the operation
        self.move2station["time"] = 0.5  # Time to move the blade to the overlay station [hr]
        # Rotate blade 90
        self.rotate90deg["n_pers"] = 3.0  # Number of personnel involved in the operation
        self.rotate90deg["length"] = blade_parameters["length"]  # Blade length [m]
        self.rotate90deg["time"] = 0.5  # Fixed time to rotate a blade shorter than 40 m [hr]
        self.rotate90deg["rate"] = 0.0083  # Extra time per meter length needed to rotate blades longer than 40 m [hr/m]
        # Place staging
        self.place_staging["n_pers"] = 6.0  # Number of personnel involved in the operation
        self.place_staging["time"] = 0.25  # Time for place staging [hr]
        # Repair over/under-bite
        self.repair["n_pers"] = 0.0  # Number of personnel involved in the operation
        # Install overlay
        self.inst_overlay["n_pers"] = 12.0  # Number of personnel involved in the operation
        self.inst_overlay["length"] = totalOL  # Length of overlay ply [m]
        self.inst_overlay["rate"] = 4.0  # Rate to install overlay [m/hr/man]
        # Vacuum bag overlay
        self.vacuum_bag["n_pers"] = 6.0  # Number of personnel involved in the operation
        self.vacuum_bag["rate"] = 30.0  # Rate to place vacuum bag [m/hr]
        self.vacuum_bag["length"] = 2 * OL_ply  # 2x longest overlay ply [m]
        # Cure of overlay
        self.cure["n_pers"] = 6.0  # Number of personnel involved in the operation
        self.cure["time"] = 1.0  # Curing time [hr]
        # Remove vacuum bag
        self.remove_bag["n_pers"] = 3.0  # Number of personnel involved in the operation
        self.remove_bag["rate"] = 60.0  # Rate to remove vacuum bag [m/hr]
        self.remove_bag["length"] = OL_ply  # 2x longest overlay ply [m]

        for var in process.keys():
            setattr(self, var, process[var])


class post_cure_process(object):
    def post_cure_steps(self):

        # Place blade in oven carts
        if self.move2cart["length"] <= 40.0:
            self.move2cart["ct"] = self.move2cart["time"]
        else:
            self.move2cart["ct"] = self.move2cart["time"] + (self.move2cart["length"] - 40.0) * self.move2cart["rate"]
        self.move2cart["labor"] = self.move2cart["ct"] * self.move2cart["n_pers"]
        # Move blade into oven
        self.move2oven["ct"] = self.move2oven["time"]
        self.move2oven["labor"] = self.move2oven["ct"] * self.move2oven["n_pers"]
        # Post cure dwell
        self.post_cure["ct"] = self.post_cure["time"]
        self.post_cure["labor"] = self.post_cure["ct"] * self.post_cure["n_pers"]
        # Move blade out of oven
        self.move_out["ct"] = self.move_out["time"]
        self.move_out["labor"] = self.move_out["ct"] * self.move_out["n_pers"]
        # Cool-down dwell
        self.cool_down["ct"] = self.cool_down["time"]
        self.cool_down["labor"] = self.cool_down["ct"] * self.cool_down["n_pers"]


class post_cure_labor(post_cure_process):
    def __init__(self, blade_parameters, process={}):
        # # Post_cure labor
        self.move2cart = {}
        self.move2oven = {}
        self.post_cure = {}
        self.move_out = {}
        self.cool_down = {}

        # Place blade in oven carts
        self.move2cart["n_pers"] = 3.0  # Number of personnel involved in the operation
        self.move2cart["length"] = blade_parameters["length"]  # Blade length [m]
        self.move2cart["time"] = 0.25  # Fixed time to move a blade shorter than 40 m [hr]
        self.move2cart["rate"] = 0.0042  # Extra time per meter length needed to move blades longer than 40 m [hr/m]
        # Move blade into oven
        self.move2oven["n_pers"] = 2.0  # Number of personnel involved in the operation
        self.move2oven["time"] = 0.1667  # Time to move blade into the oven [hr]
        # Post cure dwell
        self.post_cure["n_pers"] = 0.0  # Number of personnel involved in the operation
        self.post_cure["time"] = 8.0  # Time of post-curing [hr]
        # Move blade out of oven
        self.move_out["n_pers"] = 2.0  # Number of personnel involved in the operation
        self.move_out["time"] = 0.1667  # Time to move blade out of the oven [hr]
        # Cool-down dwell
        self.cool_down["n_pers"] = 0.0  # Number of personnel involved in the operation
        self.cool_down["time"] = 1.0  # Time needed to cool the blade down [hr]

        for var in process.keys():
            setattr(self, var, process[var])


class cut_drill_process(object):
    def cut_drill_steps(self):

        # Move blade / place in saddles
        if self.move2saddles["length"] <= 40.0:
            self.move2saddles["ct"] = self.move2saddles["time"]
        else:
            self.move2saddles["ct"] = (
                self.move2saddles["time"] + (self.move2saddles["length"] - 40.0) * self.move2saddles["rate"]
            )
        self.move2saddles["labor"] = self.move2saddles["ct"] * self.move2saddles["n_pers"]
        # Check level / point of reference
        self.checklevel["ct"] = self.checklevel["time"]
        self.checklevel["labor"] = self.checklevel["ct"] * self.checklevel["n_pers"]
        # Machine trim blade root
        self.trim_root["ct"] = self.trim_root["root_perim"] / self.trim_root["rate"]
        self.trim_root["labor"] = self.trim_root["ct"] * self.trim_root["n_pers"]
        # Clear trim excess
        self.clear_excess["ct"] = self.clear_excess["time"]
        self.clear_excess["labor"] = self.clear_excess["ct"] * self.clear_excess["n_pers"]
        # Machine cut axial and radial holes
        self.cut_holes["ct"] = self.cut_holes["n_bolts"] * self.cut_holes["time_per_hole"]
        self.cut_holes["labor"] = self.cut_holes["ct"] * self.cut_holes["n_pers"]
        # Clear drill excess
        self.clear_excess2["ct"] = self.clear_excess2["time"]
        self.clear_excess2["labor"] = self.clear_excess2["ct"] * self.clear_excess2["n_pers"]


class cut_drill_labor(cut_drill_process):
    def __init__(self, blade_parameters, process={}):
        # # Cut_drill labor
        self.move2saddles = {}
        self.checklevel = {}
        self.trim_root = {}
        self.clear_excess = {}
        self.cut_holes = {}
        self.clear_excess2 = {}

        # Move blade / place in saddles
        self.move2saddles["n_pers"] = 3.0  # Number of personnel involved in the operation
        self.move2saddles["length"] = blade_parameters["length"]  # Blade length [m]
        self.move2saddles["time"] = 0.1667  # Fixed time to move a blade shorter than 40 m [hr]
        self.move2saddles["rate"] = 0.0083  # Extra time per meter length needed to move blades longer than 40 m [hr/m]
        # Check level / point of reference
        self.checklevel["n_pers"] = 2.0  # Number of personnel involved in the operation
        self.checklevel["time"] = 0.333  # Time to check the level and the point of reference [hr]
        # Machine trim blade root
        self.trim_root["n_pers"] = 2.0  # Number of personnel involved in the operation
        self.trim_root["root_perim"] = blade_parameters["root_perim"]  # Blade root perimeter [m]
        self.trim_root["rate"] = 6.0  # Root cutting rate [m/hr]
        # Clear trim excess
        self.clear_excess["n_pers"] = 2.0  # Number of personnel involved in the operation
        self.clear_excess["time"] = 0.25  # Time to clear trim excess [hr]
        # Machine cut axial and radial holes
        self.cut_holes["n_pers"] = 2.0  # Number of personnel involved in the operation
        self.cut_holes["n_bolts"] = blade_parameters["n_bolts"]  # Number of root bolts [-]
        self.cut_holes["time_per_hole"] = 0.0333  # Time per hole [hr]
        # Clear drill excess
        self.clear_excess2["n_pers"] = 2.0  # Number of personnel involved in the operation
        self.clear_excess2["time"] = 0.25  # Time needed to clear drill excess [hr]

        for var in process.keys():
            setattr(self, var, process[var])


class root_install_process(object):
    def root_install_steps(self):

        # Move blade and place it in carts
        if self.move2cart["length"] <= 40.0:
            self.move2cart["ct"] = self.move2cart["time"]
        else:
            self.move2cart["ct"] = self.move2cart["time"] + (self.move2cart["length"] - 40.0) * self.move2cart["rate"]
        self.move2cart["labor"] = self.move2cart["ct"] * self.move2cart["n_pers"]
        # Install barrel nuts
        self.barrel_nuts["labor"] = self.barrel_nuts["n_bolts"] / self.barrel_nuts["rate"]
        self.barrel_nuts["ct"] = self.barrel_nuts["labor"] / self.barrel_nuts["n_pers"]
        # Apply root band
        self.root_band["ct"] = self.root_band["root_perim"] / self.root_band["rate"]
        self.root_band["labor"] = self.root_band["ct"] * self.root_band["n_pers"]


class root_install_labor(root_install_process):
    def __init__(self, blade_parameters, process={}):
        # # Root_install labor
        self.move2cart = {}
        self.barrel_nuts = {}
        self.root_band = {}

        # Move blade and place it in carts
        self.move2cart["n_pers"] = 3.0  # Number of personnel involved in the operation
        self.move2cart["length"] = blade_parameters["length"]  # Blade length [m]
        self.move2cart["time"] = 0.1667  # Fixed time to move a blade shorter than 40 m [hr]
        self.move2cart["rate"] = 0.0083  # Extra time per meter length needed to move blades longer than 40 m [hr/m]
        # Install barrel nuts
        self.barrel_nuts["n_pers"] = 2.0  # Number of personnel involved in the operation
        self.barrel_nuts["n_bolts"] = blade_parameters["n_bolts"]  # Number of root bolts [-]
        self.barrel_nuts["rate"] = 24.0  # Nut install rate [#/hr]
        # Apply root band
        self.root_band["n_pers"] = 2.0  # Number of personnel involved in the operation
        self.root_band["root_perim"] = blade_parameters["root_perim"]  # Blade root perimeter [m]
        self.root_band["rate"] = 6.0  # Apply root band rate [m/hr]

        for var in process.keys():
            setattr(self, var, process[var])


class surface_prep_process(object):
    def surface_prep_steps(self):

        # Move blade carts to surface preparation area
        self.move2area["ct"] = self.move2area["time"]
        self.move2area["labor"] = self.move2area["ct"] * self.move2area["n_pers"]
        # Perform surface preparation
        self.surface_prep["labor"] = self.surface_prep["area"] / self.surface_prep["rate"]
        self.surface_prep["ct"] = self.surface_prep["labor"] / self.surface_prep["n_pers"]


class surface_prep_labor(surface_prep_process):
    def __init__(self, blade_parameters, process={}):
        # # Surface preparation labor
        self.move2area = {}
        self.surface_prep = {}

        # Move blade carts to surface preparation area
        self.move2area["n_pers"] = 2.0  # Number of personnel involved in the operation
        self.move2area["time"] = 0.1667  # Fixed time to move the blade [hr]
        # Perform surface preparation
        self.surface_prep["n_pers"] = 8.0  # Number of personnel involved in the operation
        self.surface_prep["area"] = blade_parameters["area"]  # Total blade outer area [m2]
        self.surface_prep["rate"] = 6.0  # Surface preparation rate [m2/hr]

        for var in process.keys():
            setattr(self, var, process[var])


class paint_process(object):
    def paint_steps(self):

        # Move blade carts to painting area
        self.move2area["ct"] = self.move2area["time"]
        self.move2area["labor"] = self.move2area["ct"] * self.move2area["n_pers"]
        # Apply primer
        self.primer["ct"] = self.primer["area"] / self.primer["rate"]
        self.primer["labor"] = self.primer["ct"] * self.primer["n_pers"]
        # Cure / tackify
        self.cure["ct"] = self.cure["time"]
        self.cure["labor"] = self.cure["ct"] * self.cure["n_pers"]
        # Apply top coat
        self.top_coat["ct"] = self.top_coat["area"] / self.top_coat["rate"]
        self.top_coat["labor"] = self.top_coat["ct"] * self.top_coat["n_pers"]
        # Cure
        self.cure2["ct"] = self.cure2["time"]
        self.cure2["labor"] = self.cure2["ct"] * self.cure2["n_pers"]


class paint_labor(paint_process):
    def __init__(self, blade_parameters, process={}):
        # # Painting labor
        self.move2area = {}
        self.primer = {}
        self.cure = {}
        self.top_coat = {}
        self.cure2 = {}

        # Move blade carts to painting area
        self.move2area["n_pers"] = 2.0  # Number of personnel involved in the operation
        self.move2area["time"] = 0.1667  # Fixed time to move the blade [hr]
        # Apply primer
        self.primer["n_pers"] = 4.0  # Number of personnel involved in the operation
        self.primer["area"] = blade_parameters["area"]  # Total blade outer area [m2]
        self.primer["rate"] = 480.0  # Rate of application  of the primer  - spray rate [m2/hr]
        # Cure / tackify
        self.cure["n_pers"] = 4.0  # Number of personnel involved in the operation
        self.cure["time"] = 1.0  # Fixed time to cure / tackify the primer dwell
        # Apply top coat
        self.top_coat["n_pers"] = 4.0  # Number of personnel involved in the operation
        self.top_coat["area"] = blade_parameters["area"]  # Total blade outer area [m2]
        self.top_coat["rate"] = 480.0  # Rate of application  of the top coat - spray rate [m2/hr]
        # Cure
        self.cure2["n_pers"] = 4.0  # Number of personnel involved in the operation
        self.cure2["time"] = 3.0  # Fixed time for the paint to dwell

        for var in process.keys():
            setattr(self, var, process[var])


class surface_finish_process(object):
    def surface_finish_steps(self):

        # Move blade carts to surface finishing area
        self.move2area["ct"] = self.move2area["time"]
        self.move2area["labor"] = self.move2area["ct"] * self.move2area["n_pers"]
        # Perform surface finishing
        self.surface_finish["labor"] = self.surface_finish["area"] / self.surface_finish["rate"]
        self.surface_finish["ct"] = self.surface_finish["labor"] / self.surface_finish["n_pers"]


class surface_finish_labor(surface_finish_process):
    def __init__(self, blade_parameters, process={}):
        # # Surface finishing labor
        self.move2area = {}
        self.surface_finish = {}

        # Move blade carts to surface finishing area
        self.move2area["n_pers"] = 2.0  # Number of personnel involved in the operation
        self.move2area["time"] = 0.1667  # Fixed time to move the blade [hr]
        # Perform surface finishing
        self.surface_finish["n_pers"] = 3.0  # Number of personnel involved in the operation
        self.surface_finish["area"] = blade_parameters["area"]  # Total blade outer area [m2]
        self.surface_finish["rate"] = 30.0  # Surface finishing rate [m2/hr]

        for var in process.keys():
            setattr(self, var, process[var])


class weight_balance_process(object):
    def weight_balance_steps(self):

        # Move blade / place in saddles
        if self.move2saddles["length"] <= 40.0:
            self.move2saddles["ct"] = self.move2saddles["time"]
        else:
            self.move2saddles["ct"] = (
                self.move2saddles["time"] + (self.move2saddles["length"] - 40.0) * self.move2saddles["rate"]
            )
        self.move2saddles["labor"] = self.move2saddles["ct"] * self.move2saddles["n_pers"]
        # Check balance
        self.check_balance["ct"] = self.check_balance["time"]
        self.check_balance["labor"] = self.check_balance["ct"] * self.check_balance["n_pers"]
        # Drill holes into balance boxes
        self.holes_boxes["ct"] = self.holes_boxes["time"]
        self.holes_boxes["labor"] = self.holes_boxes["ct"] * self.holes_boxes["n_pers"]
        # Mix balance box filler
        self.mix_filler["ct"] = self.mix_filler["time"]
        self.mix_filler["labor"] = self.mix_filler["ct"] * self.mix_filler["n_pers"]
        # Pump filler into balance boxes
        self.pump_filler["ct"] = self.pump_filler["time"]
        self.pump_filler["labor"] = self.pump_filler["ct"] * self.pump_filler["n_pers"]
        # Plug balance box holes
        self.plug_holes["ct"] = self.plug_holes["time"]
        self.plug_holes["labor"] = self.plug_holes["ct"] * self.plug_holes["n_pers"]


class weight_balance_labor(weight_balance_process):
    def __init__(self, blade_parameters, process={}):
        # # Weight and balance labor
        self.move2saddles = {}
        self.check_balance = {}
        self.holes_boxes = {}
        self.mix_filler = {}
        self.pump_filler = {}
        self.plug_holes = {}

        # Move blade / place in saddles
        self.move2saddles["n_pers"] = 3.0  # Number of personnel involved in the operation
        self.move2saddles["length"] = blade_parameters["length"]  # Blade length [m]
        self.move2saddles["time"] = 0.1667  # Fixed time to move a blade shorter than 40 m [hr]
        self.move2saddles["rate"] = 0.0083  # Extra time per meter length needed to move blades longer than 40 m [hr/m]
        # Check balance
        self.check_balance["n_pers"] = 2.0  # Number of personnel involved in the operation
        self.check_balance["time"] = 0.25  # Time needed [hr]
        # Drill holes into balance boxes
        self.holes_boxes["n_pers"] = 2.0  # Number of personnel involved in the operation
        self.holes_boxes["time"] = 0.1667  # Time needed [hr]
        # Mix balance box filler
        self.mix_filler["n_pers"] = 2.0  # Number of personnel involved in the operation
        self.mix_filler["time"] = 0.5  # Time needed [hr]
        # Pump filler into balance boxes
        self.pump_filler["n_pers"] = 2.0  # Number of personnel involved in the operation
        self.pump_filler["time"] = 1.0  # Time needed [hr]
        # Plug balance box holes
        self.plug_holes["n_pers"] = 2.0  # Number of personnel involved in the operation
        self.plug_holes["time"] = 0.667  # Time needed [hr]

        for var in process.keys():
            setattr(self, var, process[var])


class inspection_process(object):
    def inspection_steps(self):

        # Move blade and place it in shipping saddles
        if self.move2saddles["length"] <= 40.0:
            self.move2saddles["ct"] = self.move2saddles["time"]
        else:
            self.move2saddles["ct"] = (
                self.move2saddles["time"] + (self.move2saddles["length"] - 40.0) * self.move2saddles["rate"]
            )
        self.move2saddles["labor"] = self.move2saddles["ct"] * self.move2saddles["n_pers"]
        # Perform final inspection
        self.inspection["labor"] = self.inspection["area"] / self.inspection["rate"]
        self.inspection["ct"] = self.inspection["labor"] / self.inspection["n_pers"]


class inspection_labor(inspection_process):
    def __init__(self, blade_parameters, process={}):
        # # Final inspection labor
        self.move2saddles = {}
        self.inspection = {}

        # move blade / place in shipping saddles
        self.move2saddles["n_pers"] = 3.0  # Number of personnel involved in the operation
        self.move2saddles["length"] = blade_parameters["length"]  # Blade length [m]
        self.move2saddles["time"] = 0.333  # Fixed time to move a blade shorter than 40 m [hr]
        self.move2saddles["rate"] = 0.0083  # Extra time per meter length needed to move blades longer than 40 m [hr/m]
        # Perform final inspection
        self.inspection["n_pers"] = 2.0  # Number of personnel involved in the operation
        self.inspection["area"] = blade_parameters["area"]  # Total blade outer area [m2]
        self.inspection["rate"] = 360.0  # Surface preparation rate [m2/hr]

        for var in process.keys():
            setattr(self, var, process[var])


class shipping_prep_process(object):
    def shipping_prep_steps(self):

        # Install final root bolts
        self.root_bolts["labor"] = self.root_bolts["n_bolts"] / self.root_bolts["rate"]
        self.root_bolts["ct"] = self.root_bolts["labor"] / self.root_bolts["n_pers"]
        # Install root plate
        self.root_plate["ct"] = self.root_plate["time"]
        self.root_plate["labor"] = self.root_plate["ct"] * self.root_plate["n_pers"]
        # Connect LPS
        self.connectLPS["ct"] = self.connectLPS["time"]
        self.connectLPS["labor"] = self.connectLPS["ct"] * self.connectLPS["n_pers"]
        # Install root cover
        self.root_cover["ct"] = self.root_cover["time"]
        self.root_cover["labor"] = self.root_cover["ct"] * self.root_cover["n_pers"]
        # Install 0 deg pitch plate
        self.pitch_plate["ct"] = self.pitch_plate["time"]
        self.pitch_plate["labor"] = self.pitch_plate["ct"] * self.pitch_plate["n_pers"]
        # Apply blade serial number
        self.serial_num["ct"] = self.serial_num["time"]
        self.serial_num["labor"] = self.serial_num["ct"] * self.serial_num["n_pers"]
        # Remove blade from factory
        if self.remove_blade["length"] <= 50.0:
            self.remove_blade["ct"] = self.remove_blade["time"]
        else:
            self.remove_blade["ct"] = (
                self.remove_blade["time"] + (self.remove_blade["length"] - 50.0) * self.remove_blade["rate"]
            )
        self.remove_blade["labor"] = self.remove_blade["ct"] * self.remove_blade["n_pers"]


class shipping_prep_labor(shipping_prep_process):
    def __init__(self, blade_parameters, process={}):
        # # Shipping preparation labor
        self.root_bolts = {}
        self.root_plate = {}
        self.connectLPS = {}
        self.root_cover = {}
        self.pitch_plate = {}
        self.serial_num = {}
        self.remove_blade = {}

        # Install final root bolts
        self.root_bolts["n_pers"] = 2.0  # Number of personnel involved in the operation
        self.root_bolts["n_bolts"] = blade_parameters["n_bolts"]  # Number of blade root bolts [#]
        self.root_bolts["rate"] = 120.0  # Rate of bolts installation [#/hr]
        # Install root plate
        self.root_plate["n_pers"] = 2.0  # Number of personnel involved in the operation
        self.root_plate["time"] = 0.4  # Fixed time to install the root plate [hr]
        # Connect LPS
        self.connectLPS["n_pers"] = 2.0  # Number of personnel involved in the operation
        self.connectLPS["time"] = 0.5  # Fixed time to connect the LPS [hr]
        # Install root cover
        self.root_cover["n_pers"] = 2.0  # Number of personnel involved in the operation
        self.root_cover["time"] = 0.25  # Fixed time to install the root cover [hr]
        # Install 0 pitch plate
        self.pitch_plate["n_pers"] = 2.0  # Number of personnel involved in the operation
        self.pitch_plate["time"] = 0.1667  # Fixed time to install the 0 deg pitch plate [hr]
        # Apply blade serial number
        self.serial_num["n_pers"] = 2.0  # Number of personnel involved in the operation
        self.serial_num["time"] = 0.333  # Fixed time to apply the blade serial number [hr]
        # Remove blade from factory
        self.remove_blade["n_pers"] = 3.0  # Number of personnel involved in the operation
        self.remove_blade["length"] = blade_parameters["length"]  # Blade length [m]
        self.remove_blade["time"] = 0.5  # Fixed time to move a blade shorter than 40 m [hr]
        self.remove_blade["rate"] = 0.0042  # Extra time per meter length needed to move blades longer than 40 m [hr/m]

        for var in process.keys():
            setattr(self, var, process[var])


def compute_labor_ct(n_workers, action, rate, time, flag):
    if flag:
        labor = n_workers * (action / rate + time)
    else:
        labor = action / rate + time

    ct = labor / n_workers

    return labor, ct


def compute_total_labor_ct(data_struct, name, verbose, no_contribution2ct=[]):

    process = data_struct.__dict__.keys()
    labor_total_per_process = 0.0
    ct_total_per_process = 0.0
    for var in process:
        data = getattr(data_struct, var)
        labor_total_per_process += data["labor"]
        if var not in no_contribution2ct:
            ct_total_per_process += data["ct"]

    return labor_total_per_process, ct_total_per_process


class virtual_factory(object):
    def __init__(self, blade_specs, operation, gating_ct, non_gating_ct):

        # Blade inputs
        self.n_webs = blade_specs["n_webs"]

        # Financial parameters
        self.wage = 20.0  # [$] Wage of an unskilled worker
        self.beni = 30.4  # [%] Benefits on wage and salary
        self.overhead = 30.0  # [%] Labor overhead
        self.crr = 10.0  # [%] Capital recovery rate
        self.wcp = 3.0  # [month] Working capital period - amount of time it takes to turn the net current assets and current liabilities into cash
        self.p_life = 1  # [yr] Length of production run
        self.rejr = 0.25  # [%] Part reject rate per process

        # Productive lives
        self.building_life = 30.0  # [yr] Building recovery life
        self.eq_life = 10.0  # [yr] Equipment recovery life
        self.tool_life = 4.0  # [yr] Productive tool life

        # Factory parameters
        self.n_blades = 1000  # [-] Number of blades that the factory aims at manufacturing

        self.install_cost = 10.0  # [%] Installation costs
        self.price_space = 800.0  # [$/m2] Price of building space
        self.maintenance_cost = 4.0  # [%] Maintenance costs
        self.electr = 0.08  # [$/kWh] Price of electricity
        self.hours = 24.0  # [hr] Working hours per day
        self.days = 250.0  # [day] Working days per year
        self.avg_dt = 20.0  # [%] Average downtime for workers and equipment

        # Compute cumulative rejection rate
        self.cum_rejr = np.zeros(len(operation))  # [%]
        self.cum_rejr[-1] = 1.0 - (1.0 - (self.rejr / 100.0))
        for i_op in range(1, len(operation)):
            self.cum_rejr[-i_op - 1] = 1.0 - (1.0 - (self.rejr / 100)) * (1.0 - self.cum_rejr[-i_op])

        # Calculate the number of sets of lp and hp skin molds needed
        self.n_set_molds_skins = (
            self.n_blades * sum(gating_ct) / (1 - self.cum_rejr[5 + self.n_webs]) / (self.hours * self.days)
        )  # [-] Number of skin mold sets (low and high pressure)

        # Number of parallel processes
        self.parallel_proc = np.ones(len(operation))  # [-]

        for i_op in range(0, len(operation)):
            self.parallel_proc[i_op] = (
                self.n_set_molds_skins * non_gating_ct[i_op] / sum(gating_ct) / (1 - self.cum_rejr[i_op])
            )
        n_molds_root = 2 * self.n_set_molds_skins * non_gating_ct[1] / sum(gating_ct) / (1 - self.cum_rejr[1])
        if n_molds_root < 1:
            self.parallel_proc[2] = 0
        else:
            self.parallel_proc[1] = self.n_set_molds_skins * non_gating_ct[1] / sum(gating_ct) / (1 - self.cum_rejr[1])
            self.parallel_proc[2] = self.n_set_molds_skins * non_gating_ct[2] / sum(gating_ct) / (1 - self.cum_rejr[2])
        for i_web in range(self.n_webs):
            self.parallel_proc[3 + i_web] = (
                2 * self.n_set_molds_skins * non_gating_ct[3 + i_web] / sum(gating_ct) / (1 - self.cum_rejr[3 + i_web])
            )

        self.parallel_proc[5 + self.n_webs] = self.n_set_molds_skins
        self.parallel_proc[6 + self.n_webs] = self.n_set_molds_skins
        self.parallel_proc[7 + self.n_webs] = self.n_set_molds_skins
        self.parallel_proc[8 + self.n_webs] = self.n_set_molds_skins

        # Building space per operation
        delta = 2.0  # [m] Distance between blades
        self.floor_space = np.zeros(len(operation))  # [m2]
        self.floor_space[0] = 3.0 * blade_specs["blade_length"]  # [m2] Material cutting
        self.floor_space[1] = (
            self.parallel_proc[1] * (delta + blade_specs["root_D"]) * (delta + blade_specs["root_preform_length"])
        )  # [m2] Infusion root preform lp
        self.floor_space[2] = (
            self.parallel_proc[2] * (delta + blade_specs["root_D"]) * (delta + blade_specs["root_preform_length"])
        )  # [m2] Infusion root preform hp
        for i_web in range(self.n_webs):
            self.floor_space[3 + i_web] = (
                self.parallel_proc[3 + i_web]
                * (delta + blade_specs["length_webs"][i_web])
                * (delta + blade_specs["height_webs_start"][i_web])
            )  # [m2] Infusion webs
        self.floor_space[3 + self.n_webs] = (
            self.parallel_proc[3 + self.n_webs]
            * (delta + blade_specs["length_sc_lp"])
            * (delta + blade_specs["width_sc_start_lp"])
        )  # [m2] Infusion spar caps
        self.floor_space[4 + self.n_webs] = (
            self.parallel_proc[4 + self.n_webs]
            * (delta + blade_specs["length_sc_hp"])
            * (delta + blade_specs["width_sc_start_hp"])
        )  # [m2] Infusion spar caps
        self.floor_space[5 + self.n_webs] = (
            self.parallel_proc[5 + self.n_webs]
            * (blade_specs["max_chord"] + delta)
            * (blade_specs["blade_length"] + delta)
        )  # [m2] Infusion skin shell lp
        self.floor_space[6 + self.n_webs] = (
            self.parallel_proc[6 + self.n_webs]
            * (blade_specs["max_chord"] + delta)
            * (blade_specs["blade_length"] + delta)
        )  # [m2] Infusion skin shell hp
        self.floor_space[9 + self.n_webs] = (
            self.parallel_proc[9 + self.n_webs]
            * (blade_specs["max_chord"] + delta)
            * (blade_specs["blade_length"] + delta)
        )  # [m2] Trim
        self.floor_space[10 + self.n_webs] = (
            self.parallel_proc[10 + self.n_webs]
            * (blade_specs["root_D"] + delta)
            * (blade_specs["blade_length"] + delta)
        )  # [m2] Overlay
        self.floor_space[11 + self.n_webs] = (
            self.parallel_proc[11 + self.n_webs]
            * (blade_specs["root_D"] + delta)
            * (blade_specs["blade_length"] + delta)
        )  # [m2] Post cure
        self.floor_space[12 + self.n_webs] = (
            self.parallel_proc[12 + self.n_webs]
            * (blade_specs["max_chord"] + delta)
            * (blade_specs["blade_length"] + delta)
        )  # [m2] Root cut and drill
        self.floor_space[13 + self.n_webs] = (
            self.parallel_proc[13 + self.n_webs]
            * (blade_specs["root_D"] + delta)
            * (blade_specs["blade_length"] + delta)
        )  # [m2] Root hardware install
        self.floor_space[14 + self.n_webs] = (
            self.parallel_proc[14 + self.n_webs]
            * (blade_specs["root_D"] + delta)
            * (blade_specs["blade_length"] + delta)
        )  # [m2] Surface preparation
        self.floor_space[15 + self.n_webs] = (
            self.parallel_proc[15 + self.n_webs]
            * (blade_specs["root_D"] + delta)
            * (blade_specs["blade_length"] + delta)
        )  # [m2] Paint
        self.floor_space[16 + self.n_webs] = (
            self.parallel_proc[16 + self.n_webs]
            * (blade_specs["root_D"] + delta)
            * (blade_specs["blade_length"] + delta)
        )  # [m2] Surface inspection and finish
        self.floor_space[17 + self.n_webs] = (
            self.parallel_proc[17 + self.n_webs]
            * (blade_specs["root_D"] + delta)
            * (blade_specs["blade_length"] + delta)
        )  # [m2] Weight and balance
        self.floor_space[18 + self.n_webs] = (
            self.parallel_proc[18 + self.n_webs]
            * (blade_specs["root_D"] + delta)
            * (blade_specs["blade_length"] + delta)
        )  # [m2] Inspection
        self.floor_space[19 + self.n_webs] = (
            self.parallel_proc[19 + self.n_webs]
            * (blade_specs["root_D"] + delta)
            * (blade_specs["blade_length"] + delta)
        )  # [m2] Shipping preparation

        # Average power consumption during each operation
        Cp = 1.01812  # [kJ/kg/K] Kalogiannakis et. al 2003
        Tcure = 70  # [C]
        Tamb = 22  # [C]
        OvenCycle = 7  # [hr]
        EtaOven = 0.5  # [-]

        kJ_per_kg = Cp * (Tcure - Tamb) / (OvenCycle * 3600) / EtaOven

        self.power_consumpt = self.floor_space * 250 / self.hours / self.days  # [kW] 80000 btu / sq ft
        self.power_consumpt[1] = (
            self.power_consumpt[1] + self.parallel_proc[1] * blade_specs["mass_root_preform_lp"] * kJ_per_kg
        )  # [kW] Root preform lp
        self.power_consumpt[2] = (
            self.power_consumpt[2] + self.parallel_proc[2] * blade_specs["mass_root_preform_hp"] * kJ_per_kg
        )  # [kW] Root preform hp
        for i_web in range(self.n_webs):
            self.power_consumpt[3 + i_web] = (
                self.power_consumpt[3 + i_web]
                + self.parallel_proc[3 + i_web] * blade_specs["mass_webs"][i_web] * kJ_per_kg
            )  # [kW] Root preform hp
        self.power_consumpt[3 + self.n_webs] = (
            self.power_consumpt[3 + self.n_webs]
            + self.parallel_proc[3 + self.n_webs] * blade_specs["mass_sc_lp"] * kJ_per_kg
        )  # [kW] Spar cap lp
        self.power_consumpt[4 + self.n_webs] = (
            self.power_consumpt[4 + self.n_webs]
            + self.parallel_proc[4 + self.n_webs] * blade_specs["mass_sc_hp"] * kJ_per_kg
        )  # [kW] Spar cap hp
        self.power_consumpt[5 + self.n_webs] = (
            self.power_consumpt[5 + self.n_webs]
            + self.parallel_proc[5 + self.n_webs] * (blade_specs["mass_shell_lp"]) * kJ_per_kg
        )  # [kW] Shell lp
        self.power_consumpt[6 + self.n_webs] = (
            self.power_consumpt[6 + self.n_webs]
            + self.parallel_proc[6 + self.n_webs] * (blade_specs["mass_shell_hp"]) * kJ_per_kg
        )  # [kW] Shell hp
        self.power_consumpt[11 + self.n_webs] = (
            self.power_consumpt[11 + self.n_webs]
            + self.parallel_proc[11 + self.n_webs] * blade_specs["blade_mass"] * kJ_per_kg
        )  # [kW] Post cure

        # Tooling investment per station per operation (molds)
        self.tooling_investment = np.zeros(len(operation))  # [$]
        price_mold_sqm = 5000.0
        self.tooling_investment[1] = (
            price_mold_sqm * self.parallel_proc[1] * blade_specs["area_lp_root"]
        )  # [$] Mold of the root preform - lp, cost assumed equal to 50000 $ per meter square of surface
        self.tooling_investment[2] = (
            price_mold_sqm * self.parallel_proc[2] * blade_specs["area_hp_root"]
        )  # [$] Mold of the root preform - hp, cost assumed equal to 50000 $ per meter square of surface
        for i_web in range(self.n_webs):
            self.tooling_investment[3 + i_web] = (
                price_mold_sqm * self.parallel_proc[3 + i_web] * blade_specs["area_webs_w_flanges"][i_web]
            )  # [$] Mold of the webs, cost assumed equal to 10800 $ per meter square of surface
        self.tooling_investment[3 + self.n_webs] = (
            price_mold_sqm * self.parallel_proc[3 + self.n_webs] * blade_specs["area_sc_lp"]
        )  # [$] Mold of the low pressure spar cap, cost assumed equal to 10800 $ per meter square of surface
        self.tooling_investment[4 + self.n_webs] = (
            price_mold_sqm * self.parallel_proc[4 + self.n_webs] * blade_specs["area_sc_hp"]
        )  # [$] Mold of the high pressure spar cap, cost assumed equal to 10800 $ per meter square of surface
        self.tooling_investment[5 + self.n_webs] = (
            price_mold_sqm * self.parallel_proc[5 + self.n_webs] * blade_specs["area_lpskin_w_flanges"]
        )  # [$] Mold of the low pressure skin shell, assumed equal to 9400 $ per meter square of surface
        self.tooling_investment[6 + self.n_webs] = (
            price_mold_sqm * self.parallel_proc[6 + self.n_webs] * blade_specs["area_hpskin_w_flanges"]
        )  # [$] Mold of the low pressure skin shell, assumed equal to 9400 $ per meter square of surface

        # Equipment investment per station per operation
        self.equipm_investment = np.zeros(len(operation))  # [$]
        self.equipm_investment[0] = (
            5000.0 * self.parallel_proc[0] * blade_specs["blade_length"]
        )  # [$] Equipment for material cutting is assumed at 5000 $ per meter of blade length
        self.equipm_investment[1] = (
            15000.0 * self.parallel_proc[1] * blade_specs["root_D"]
        )  # [$] Equipment for root preform infusion is assumed at 15000 $ per meter of blade root diameter
        self.equipm_investment[2] = (
            15000.0 * self.parallel_proc[2] * blade_specs["root_D"]
        )  # [$] Equipment for root preform infusion is assumed at 15000 $ per meter of blade root diameter
        for i_web in range(self.n_webs):
            self.equipm_investment[3 + i_web] = (
                1700.0 * self.parallel_proc[3 + i_web] * blade_specs["length_webs"][i_web]
            )  # [$] Equipment for webs infusion is assumed at 1700 $ per meter of web length
        self.equipm_investment[3 + self.n_webs] = (
            1700.0 * self.parallel_proc[3 + self.n_webs] * blade_specs["length_sc_lp"]
        )  # [$] Equipment for spar caps infusion is assumed at 1700 $ per meter of spar cap length
        self.equipm_investment[4 + self.n_webs] = (
            1700.0 * self.parallel_proc[4 + self.n_webs] * blade_specs["length_sc_hp"]
        )  # [$] Equipment for spar caps infusion is assumed at 1700 $ per meter of spar cap length
        self.equipm_investment[5 + self.n_webs] = (
            1600.0 * self.parallel_proc[5 + self.n_webs] * blade_specs["skin_perimeter_wo_root"]
        )  # [$] Equipment for skins infusion is assumed at 1600 $ per meter of skin perimeter
        self.equipm_investment[6 + self.n_webs] = (
            1600.0 * self.parallel_proc[6 + self.n_webs] * blade_specs["skin_perimeter_wo_root"]
        )  # [$] Equipment for skins infusion is assumed at 1600 $ per meter of skin perimeter
        self.equipm_investment[7 + self.n_webs] = (
            6600.0 * self.parallel_proc[7 + self.n_webs] * sum(blade_specs["length_webs"])
        )  # [$] Equipment for assembly is assumed equal to 6600 $ per meter of total webs length
        self.equipm_investment[9 + self.n_webs] = (
            25000.0 * self.parallel_proc[9 + self.n_webs] * blade_specs["blade_length"]
        )  # [$] Equipment for trim booth is assumed at 25000 $ per meter of blade length
        self.equipm_investment[10 + self.n_webs] = (
            250.0 * self.parallel_proc[10 + self.n_webs] * blade_specs["blade_length"]
        )  # [$] Equipment for overlay is assumed at 250 $ per meter of blade length
        self.equipm_investment[11 + self.n_webs] = (
            28500.0 * self.parallel_proc[11 + self.n_webs] * blade_specs["blade_length"]
        )  # [$] Equipment for post-cure is assumed at 28500 $ per meter of blade length
        self.equipm_investment[12 + self.n_webs] = (
            390000.0 * self.parallel_proc[12 + self.n_webs] * blade_specs["root_D"]
        )  # [$] Equipment for root cut and drill is assumed at 390000 $ per meter of root diameter
        self.equipm_investment[13 + self.n_webs] = (
            15500.0 * self.parallel_proc[13 + self.n_webs] * blade_specs["root_D"]
        )  # [$] Equipment for root hardware install is assumed at 15500 $ per meter of root diameter
        self.equipm_investment[14 + self.n_webs] = (
            160.0
            * self.parallel_proc[14 + self.n_webs]
            * (blade_specs["area_lpskin_wo_flanges"] + blade_specs["area_hpskin_wo_flanges"])
        )  # [$] Equipment for surface preparation is assumed at 160 $ per meter square of blade outer surface
        self.equipm_investment[15 + self.n_webs] = (
            57000.0 * self.parallel_proc[15 + self.n_webs] * blade_specs["blade_length"]
        )  # [$] Equipment for paint booth is assumed at 57000 $ per meter of blade length
        self.equipm_investment[16 + self.n_webs] = (
            800.0 * self.parallel_proc[16 + self.n_webs] * blade_specs["blade_length"]
        )  # [$] Equipment for surface inspection and finish is assumed at 800 $ per meter of blade length
        self.equipm_investment[17 + self.n_webs] = (
            200000.0 * self.parallel_proc[17 + self.n_webs]
        )  # [$] Weight and Balance, assumed constant
        self.equipm_investment[18 + self.n_webs] = (
            400.0 * self.parallel_proc[18 + self.n_webs] * blade_specs["blade_length"]
        )  # [$] Equipment for final inspection is assumed at 400 $ per meter of blade length
        self.equipm_investment[19 + self.n_webs] = (
            8000.0 * self.parallel_proc[19 + self.n_webs] * blade_specs["root_D"]
        )  # [$] Equipment for shipping preparation is assumed at 8000 $ per meter of root diameter

    def execute_direct_labor_cost(self, operation, labor_hours):

        verbosity = 0

        direct_labor_cost_per_blade = np.zeros(len(operation))  # [$]
        direct_labor_cost_per_year = np.zeros(len(operation))  # [$]

        if verbosity:
            print("\n#################################\nDirect labor cost")

        for i_op in range(0, len(operation)):
            direct_labor_cost_per_blade[i_op], direct_labor_cost_per_year[i_op] = compute_direct_labor_cost(
                self, labor_hours[i_op], operation[i_op], self.cum_rejr[i_op], verbosity
            )

        total_direct_labor_cost_per_blade = sum(direct_labor_cost_per_blade)
        total_direct_labor_cost_per_year = sum(direct_labor_cost_per_year)

        total_labor_overhead_per_blade = total_direct_labor_cost_per_blade * (self.overhead / 100.0)

        return total_direct_labor_cost_per_blade, total_labor_overhead_per_blade

    def execute_utility_cost(self, operation, ct):

        verbosity = 0

        utility_cost_per_blade = np.zeros(len(operation))  # [$]
        utility_cost_per_year = np.zeros(len(operation))  # [$]

        if verbosity:
            print("\n#################################\nUtility cost")

        for i_op in range(0, len(operation)):
            utility_cost_per_blade[i_op], utility_cost_per_year[i_op] = compute_utility_cost(
                self, ct[i_op], self.power_consumpt[i_op], operation[i_op], self.cum_rejr[i_op], verbosity
            )

        total_utility_cost_per_blade = sum(utility_cost_per_blade)
        total_utility_labor_cost_per_year = sum(utility_cost_per_year)

        return total_utility_cost_per_blade

    def execute_fixed_cost(self, operation, ct, blade_variable_cost_w_overhead):

        verbosity = 0

        building_cost_per_blade = np.zeros(len(operation))  # [$]
        building_cost_per_year = np.zeros(len(operation))  # [$]
        building_annuity = np.zeros(len(operation))  # [$]
        tooling_cost_per_blade = np.zeros(len(operation))  # [$]
        tooling_cost_per_year = np.zeros(len(operation))  # [$]
        tooling_annuity = np.zeros(len(operation))  # [$]
        equipment_cost_per_blade = np.zeros(len(operation))  # [$]
        equipment_cost_per_year = np.zeros(len(operation))  # [$]
        equipment_annuity = np.zeros(len(operation))  # [$]
        maintenance_cost_per_blade = np.zeros(len(operation))  # [$]
        maintenance_cost_per_year = np.zeros(len(operation))  # [$]

        for i_op in range(0, len(operation)):
            if verbosity:
                print("\nBuilding:")
            building_investment = self.floor_space[i_op] * self.price_space
            investment_bu = building_investment * self.parallel_proc[i_op]
            building_cost_per_blade[i_op], building_cost_per_year[i_op], building_annuity[i_op] = compute_cost_annuity(
                self, operation[i_op], investment_bu, self.building_life, verbosity
            )

            if verbosity:
                print("\nTooling:")
            investment_to = self.tooling_investment[i_op] * self.parallel_proc[i_op]
            tooling_cost_per_blade[i_op], tooling_cost_per_year[i_op], tooling_annuity[i_op] = compute_cost_annuity(
                self, operation[i_op], investment_to, self.tool_life, verbosity
            )

            if verbosity:
                print("\nEquipment:")
            investment_eq = self.equipm_investment[i_op] * self.parallel_proc[i_op]
            (
                equipment_cost_per_blade[i_op],
                equipment_cost_per_year[i_op],
                equipment_annuity[i_op],
            ) = compute_cost_annuity(self, operation[i_op], investment_eq, self.eq_life, verbosity)

            if verbosity:
                print("\nMaintenance:")
            maintenance_cost_per_blade[i_op], maintenance_cost_per_year[i_op] = compute_maintenance_cost(
                self, operation[i_op], investment_eq, investment_to, investment_bu, verbosity
            )

        # Sums across operations
        total_building_labor_cost_per_year = sum(building_cost_per_year)
        total_building_cost_per_blade = sum(building_cost_per_blade)

        total_tooling_labor_cost_per_year = sum(tooling_cost_per_year)
        total_tooling_cost_per_blade = sum(tooling_cost_per_blade)

        total_equipment_labor_cost_per_year = sum(equipment_cost_per_year)
        total_equipment_cost_per_blade = sum(equipment_cost_per_blade)

        total_maintenance_labor_cost_per_year = sum(maintenance_cost_per_year)
        total_maintenance_cost_per_blade = sum(maintenance_cost_per_blade)

        # Annuity
        equipment_annuity_tot = sum(equipment_annuity)
        tooling_annuity_tot = sum(tooling_annuity)
        building_annuity_tot = sum(building_annuity)

        working_annuity = (
            pmt(
                self.crr / 100.0 / 12.0,
                self.wcp,
                -(
                    self.wcp
                    / 12.0
                    * (total_maintenance_labor_cost_per_year + blade_variable_cost_w_overhead * self.n_blades)
                ),
            )
            * 12.0
        )

        annuity_tot_per_year = equipment_annuity_tot + tooling_annuity_tot + building_annuity_tot + working_annuity

        cost_of_capital_per_year = annuity_tot_per_year - (
            blade_variable_cost_w_overhead * self.n_blades
            + total_equipment_labor_cost_per_year
            + total_tooling_labor_cost_per_year
            + total_building_labor_cost_per_year
            + total_maintenance_labor_cost_per_year
        )
        cost_of_capital_per_blade = cost_of_capital_per_year / self.n_blades

        return (
            total_equipment_cost_per_blade,
            total_tooling_cost_per_blade,
            total_building_cost_per_blade,
            total_maintenance_cost_per_blade,
            cost_of_capital_per_blade,
        )


def compute_direct_labor_cost(self, labor_hours, operation, cum_rejr, verbosity):

    cost_per_blade = (
        (self.wage * (1.0 + self.beni / 100.0) * labor_hours) / (1.0 - self.avg_dt / 100.0) / (1.0 - cum_rejr)
    )
    cost_per_year = cost_per_blade * self.n_blades
    if verbosity == 1:
        print("Activity: " + operation)
        print(
            "per blade: {:8.2f} $ \t \t --- \t \t per year: {:8.2f} $".format(
                float(cost_per_blade), float(cost_per_year)
            )
        )

    return cost_per_blade, cost_per_year


def compute_utility_cost(self, ct, power_consumpt, operation, cum_rejr, verbosity):

    cost_per_blade = (self.electr * power_consumpt * ct) / (1.0 - self.avg_dt / 100.0) / (1.0 - cum_rejr)
    cost_per_year = cost_per_blade * self.n_blades

    if verbosity == 1:
        print("Activity: " + operation)
        print(
            "per blade: {:8.2f} $ \t \t --- \t \t per year: {:8.2f} $".format(
                float(cost_per_blade), float(cost_per_year)
            )
        )

    return cost_per_blade, cost_per_year


def compute_cost_annuity(self, operation, investment, life, verbosity):

    cost_per_year = investment / life
    cost_per_blade = cost_per_year / self.n_blades
    annuity = pmt(self.crr / 100.0 / 12.0, life * 12.0, -investment) * 12.0

    if verbosity == 1:
        print("Activity: " + operation)
        print(
            "per blade: {:8.2f} $ \t \t --- \t \t per year: {:8.2f} $ \t \t --- \t \t annuity: {:8.2f} $".format(
                float(cost_per_blade), float(cost_per_year), float(annuity)
            )
        )

    return cost_per_blade, cost_per_year, annuity


def compute_maintenance_cost(self, operation, investment_eq, investment_to, investment_bu, verbosity):
    cost_per_year = self.maintenance_cost / 100.0 * (investment_eq + investment_to + investment_bu)
    cost_per_blade = cost_per_year / self.n_blades

    if verbosity == 1:
        print("Activity: " + operation)
        print(
            "per blade: {:8.2f} $ \t \t --- \t \t per year: {:8.2f} $".format(
                float(cost_per_blade), float(cost_per_year)
            )
        )

    return cost_per_blade, cost_per_year


# OpenMDAO component to execute the blade cost model
class RotorCost(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("mod_options")
        self.options.declare("opt_options")

    def setup(self):
        mod_options = self.options["mod_options"]
        rotorse_options = mod_options["WISDEM"]["RotorSE"]
        self.n_span = n_span = rotorse_options["n_span"]
        self.n_span = n_span = rotorse_options["n_span"]
        self.n_webs = n_webs = rotorse_options["n_webs"]
        self.n_layers = n_layers = rotorse_options["n_layers"]
        self.n_xy = n_xy = rotorse_options["n_xy"]  # Number of coordinate points to describe the airfoil geometry
        self.layer_mat = rotorse_options["layer_mat"]
        self.layer_name = rotorse_options["layer_name"]
        self.spar_cap_ss = rotorse_options["spar_cap_ss"]
        self.spar_cap_ps = rotorse_options["spar_cap_ps"]
        mat_init_options = self.options["mod_options"]["materials"]
        self.n_mat = n_mat = mat_init_options["n_mat"]

        # Inputs - Outer blade shape
        self.add_input("blade_length", val=0.0, units="m", desc="blade length")
        self.add_input("s", val=np.zeros(n_span), desc="blade nondimensional span location")
        self.add_input("chord", val=np.zeros(n_span), units="m", desc="Chord distribution")
        self.add_input(
            "coord_xy_interp",
            val=np.zeros((n_span, n_xy, 2)),
            desc="3D array of the non-dimensional x and y airfoil coordinates of the airfoils interpolated along span for n_span stations.",
        )
        self.add_input(
            "web_start_nd",
            val=np.zeros((n_webs, n_span)),
            desc="2D array of the non-dimensional start point defined along the outer profile of a web. The TE suction side is 0, the TE pressure side is 1. The first dimension represents each web, the second dimension represents each entry along blade span.",
        )
        self.add_input(
            "web_end_nd",
            val=np.zeros((n_webs, n_span)),
            desc="2D array of the non-dimensional end point defined along the outer profile of a web. The TE suction side is 0, the TE pressure side is 1. The first dimension represents each web, the second dimension represents each entry along blade span.",
        )
        self.add_input(
            "layer_web",
            val=np.zeros(n_layers),
            desc="1D array of the web id the layer is associated to. If the layer is on the outer profile, this entry can simply stay equal to 0.",
        )
        self.add_discrete_input(
            "definition_layer",
            val=np.zeros(n_layers),
            desc="1D array of flags identifying how layers are specified in the yaml. 1) all around (skin, paint, ) 2) offset+rotation twist+width (spar caps) 3) offset+user defined rotation+width 4) midpoint TE+width (TE reinf) 5) midpoint LE+width (LE reinf) 6) layer position fixed to other layer (core fillers) 7) start and width 8) end and width 9) start and end nd 10) web layer",
        )
        self.add_input(
            "layer_thickness",
            val=np.zeros((n_layers, n_span)),
            units="m",
            desc="2D array of the thickness of the layers of the blade structure. The first dimension represents each layer, the second dimension represents each entry along blade span.",
        )
        self.add_input(
            "layer_start_nd",
            val=np.zeros((n_layers, n_span)),
            desc="2D array of the non-dimensional start point defined along the outer profile of a layer. The TE suction side is 0, the TE pressure side is 1. The first dimension represents each layer, the second dimension represents each entry along blade span.",
        )
        self.add_input(
            "layer_end_nd",
            val=np.zeros((n_layers, n_span)),
            desc="2D array of the non-dimensional end point defined along the outer profile of a layer. The TE suction side is 0, the TE pressure side is 1. The first dimension represents each layer, the second dimension represents each entry along blade span.",
        )

        # Inputs - Materials
        self.add_discrete_input("mat_name", val=n_mat * [""], desc="1D array of names of materials.")
        self.add_discrete_input(
            "orth",
            val=np.zeros(n_mat),
            desc="1D array of flags to set whether a material is isotropic (0) or orthtropic (1). Each entry represents a material.",
        )
        self.add_discrete_input(
            "component_id",
            val=np.zeros(n_mat),
            desc="1D array of flags to set whether a material is used in a blade: 0 - coating, 1 - sandwich filler , 2 - shell skin, 3 - shear webs, 4 - spar caps, 5 - TE/LE reinf.",
        )
        self.add_input(
            "rho",
            val=np.zeros(n_mat),
            units="kg/m**3",
            desc="1D array of the density of the materials. For composites, this is the density of the laminate.",
        )
        self.add_input(
            "unit_cost", val=np.zeros(n_mat), units="USD/kg", desc="1D array of the unit costs of the materials."
        )
        self.add_input(
            "waste", val=np.zeros(n_mat), desc="1D array of the non-dimensional waste fraction of the materials."
        )
        self.add_input(
            "rho_fiber",
            val=np.zeros(n_mat),
            units="kg/m**3",
            desc="1D array of the density of the fibers of the materials.",
        )
        self.add_input(
            "ply_t",
            val=np.zeros(n_mat),
            units="m",
            desc="1D array of the ply thicknesses of the materials. Non-composite materials are kept at 0.",
        )
        self.add_input(
            "fwf",
            val=np.zeros(n_mat),
            desc="1D array of the non-dimensional fiber weight- fraction of the composite materials. Non-composite materials are kept at 0.",
        )
        self.add_input(
            "roll_mass",
            val=np.zeros(n_mat),
            units="kg",
            desc="1D array of the roll mass of the composite fabrics. Non-composite materials are kept at 0.",
        )
        self.add_input(
            "flange_adhesive_squeezed",
            val=0.2,
            desc="Extra width of the adhesive once squeezed",
        )
        self.add_input(
            "flange_thick",
            val=0.008,
            units="m",
            desc="Average thickness of adhesive",
        )
        self.add_input(
            "flange_width",
            val=0.05,
            units="m",
            desc="Average width of adhesive lines",
        )

        self.add_input(
            "t_bolt_unit_cost",
            val=25.0,
            units="USD",
            desc="Cost of one t-bolt",
        )
        self.add_input(
            "t_bolt_unit_mass",
            val=2.5,
            units="kg",
            desc="Mass of one t-bolt",
        )
        self.add_input(
            "t_bolt_spacing",
            val=0.15,
            units="m",
            desc="Spacing of t-bolts along blade root circumference",
        )
        self.add_input(
            "barrel_nut_unit_cost",
            val=12.0,
            units="USD",
            desc="Cost of one barrel nut",
        )
        self.add_input(
            "barrel_nut_unit_mass",
            val=1.9,
            units="kg",
            desc="Mass of one barrel nut",
        )
        self.add_input(
            "LPS_unit_mass",
            val=1.00,
            units="kg/m",
            desc="Unit mass of the lightining protection system. Linear scaling based on the weight of 150 lbs for the 61.5 m NREL 5MW blade",
        )
        self.add_input(
            "LPS_unit_cost",
            val=40.00,
            units="USD/m",
            desc="Unit cost of the lightining protection system. Linear scaling based on the cost of 2500$ for the 61.5 m NREL 5MW blade",
        )
        self.add_input(
            "root_preform_length",
            val=0.01,
            desc="Percentage of blade length starting from blade root that is preformed and later inserted into the mold",
        )
        self.add_input(
            "joint_position",
            val=0.0,
            desc="Spanwise position of the segmentation joint.",
        )
        self.add_input("joint_mass", val=0.0, desc="Mass of the joint.")
        self.add_input("joint_cost", val=0.0, units="USD", desc="Cost of the joint.")
        # Outputs
        self.add_output(
            "sect_perimeter",
            val=np.zeros(n_span),
            units="m",
            desc="Perimeter of the section along the blade span",
        )
        self.add_output(
            "layer_volume",
            val=np.zeros(n_layers),
            units="m**3",
            desc="Volumes of each layer used in the blade, ignoring the scrap factor",
        )
        self.add_output(
            "mat_volume",
            val=np.zeros(n_mat),
            units="m**3",
            desc="Volumes of each material used in the blade, ignoring the scrap factor. For laminates, this is the wet volume",
        )
        self.add_output(
            "mat_mass",
            val=np.zeros(n_mat),
            units="kg",
            desc="Masses of each material used in the blade, ignoring the scrap factor. For laminates, this is the wet mass.",
        )
        self.add_output(
            "mat_cost",
            val=np.zeros(n_mat),
            units="USD",
            desc="Costs of each material used in the blade, ignoring the scrap factor. For laminates, this is the cost of the dry fabric.",
        )
        self.add_output(
            "mat_cost_scrap",
            val=np.zeros(n_mat),
            units="USD",
            desc="Same as mat_cost, now including the scrap factor.",
        )
        self.add_output(
            "total_labor_hours",
            val=0.0,
            units="h",
            desc="Total amount of labor hours per blade.",
        )
        self.add_output(
            "total_skin_mold_gating_ct",
            val=0.0,
            units="h",
            desc="Total amount of gating cycle time per blade. This is the cycle time required in the main mold that cannot be parallelized unless the number of molds is increased.",
        )
        self.add_output(
            "total_non_gating_ct",
            val=0.0,
            units="h",
            desc="Total amount of non-gating cycle time per blade. This cycle time can happen in parallel.",
        )
        self.add_output(
            "total_metallic_parts_cost",
            val=0.0,
            units="USD",
            desc="Cost of the metallic parts (bolts, nuts, lightining protection system), excluding the blade joint.",
        )
        self.add_output(
            "total_consumable_cost_w_waste",
            val=0.0,
            units="USD",
            desc="Cost of the consumables including the waste.",
        )
        self.add_output(
            "total_blade_mat_cost_w_waste",
            val=0.0,
            units="USD",
            desc="Total blade material costs including the waste per blade.",
        )
        self.add_output(
            "total_cost_labor",
            val=0.0,
            units="USD",
            desc="Total labor costs per blade.",
        )
        self.add_output(
            "total_cost_utility",
            val=0.0,
            units="USD",
            desc="Total utility costs per blade.",
        )
        self.add_output(
            "blade_variable_cost",
            val=0.0,
            units="USD",
            desc="Total blade variable costs per blade (material, labor, utility).",
        )
        self.add_output(
            "total_cost_equipment",
            val=0.0,
            units="USD",
            desc="Total equipment cost per blade.",
        )
        self.add_output(
            "total_cost_tooling",
            val=0.0,
            units="USD",
            desc="Total tooling cost per blade.",
        )
        self.add_output(
            "total_cost_building",
            val=0.0,
            units="USD",
            desc="Total builting cost per blade.",
        )
        self.add_output(
            "total_maintenance_cost",
            val=0.0,
            units="USD",
            desc="Total maintenance cost per blade.",
        )
        self.add_output(
            "total_labor_overhead",
            val=0.0,
            units="USD",
            desc="Total labor overhead cost per blade.",
        )
        self.add_output(
            "cost_capital",
            val=0.0,
            units="USD",
            desc="Cost of capital per blade.",
        )
        self.add_output(
            "blade_fixed_cost",
            val=0.0,
            units="USD",
            desc="Total blade fixed cost per blade (equipment, tooling, building, maintenance, labor, capital).",
        )
        self.add_output(
            "total_blade_cost",
            val=0.0,
            units="USD",
            desc="Total blade cost (variable and fixed)",
        )

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        # Inputs
        s = inputs["s"]
        blade_length = inputs["blade_length"]
        chord = inputs["chord"]
        blade_length = inputs["blade_length"]
        s = inputs["s"]
        layer_start_nd = inputs["layer_start_nd"]
        layer_end_nd = inputs["layer_end_nd"]
        web_start_nd = inputs["web_start_nd"]
        web_end_nd = inputs["web_end_nd"]
        layer_thickness = inputs["layer_thickness"]
        orth = discrete_inputs["orth"]
        component_id = discrete_inputs["component_id"]
        rho_mat = inputs["rho"]
        waste = inputs["waste"]
        layer_web = np.array(inputs["layer_web"], dtype=int)
        ply_t = inputs["ply_t"]
        roll_mass = inputs["roll_mass"]
        fwf = inputs["fwf"]
        unit_cost = inputs["unit_cost"]
        flange_adhesive_squeezed = inputs["flange_adhesive_squeezed"]
        flange_thick = inputs["flange_thick"]
        flange_width = inputs["flange_width"]
        t_bolt_spacing = inputs["t_bolt_spacing"]
        t_bolt_unit_cost = inputs["t_bolt_unit_cost"]
        barrel_nut_unit_cost = inputs["barrel_nut_unit_cost"]
        LPS_unit_cost = inputs["LPS_unit_cost"]
        root_preform_length = inputs["root_preform_length"]
        joint_cost = inputs["joint_cost"]

        # Compute arc length along blade span
        arc_L_i = np.zeros(self.n_span)
        arc_L_SS_i = np.zeros(self.n_span)
        arc_L_PS_i = np.zeros(self.n_span)
        xy_arc_nd_LE = np.zeros(self.n_span)
        web_height = np.zeros((self.n_webs, self.n_span))

        for i in range(self.n_span):
            # Compute the arc length (arc_L_i) of the non dimensional airfoil coordinates
            xy_coord_i = inputs["coord_xy_interp"][i, :, :]
            xy_arc_i = arc_length(xy_coord_i)
            arc_L_i[i] = xy_arc_i[-1]
            xy_arc_nd_i = xy_arc_i / arc_L_i[i]
            # Get the half perimeters
            idx_le = np.argmin(xy_coord_i[:, 0])
            xy_arc_nd_LE[i] = xy_arc_nd_i[idx_le]
            if np.mean(xy_coord_i[:idx_le, 1]) > 0:
                arc_L_SS_i[i] = xy_arc_i[idx_le]
                arc_L_PS_i[i] = xy_arc_i[-1] - xy_arc_i[idx_le]
            else:
                arc_L_PS_i[i] = xy_arc_i[idx_le]
                arc_L_SS_i[i] = xy_arc_i[-1] - xy_arc_i[idx_le]

            # Compute height the webs along span
            for j in range(self.n_webs):
                id_start = np.argmin(abs(xy_arc_nd_i - web_start_nd[j, i]))
                id_end = np.argmin(abs(xy_arc_nd_i - web_end_nd[j, i]))
                web_height[j, i] = abs((xy_coord_i[id_end, 1] - xy_coord_i[id_start, 0]) * chord[i])

        # Compute materials from the yaml
        layer_volume_span_ss = np.zeros((self.n_layers, self.n_span))
        layer_volume_span_ps = np.zeros((self.n_layers, self.n_span))
        layer_volume_span_webs = np.zeros((self.n_layers, self.n_span))
        layer_volume = np.zeros(self.n_layers)
        mat_volume = np.zeros(self.n_mat)
        sect_perimeter = arc_L_i * chord
        sect_perimeter_ss = arc_L_SS_i * chord
        sect_perimeter_ps = arc_L_PS_i * chord
        web_length = np.zeros(self.n_webs)
        volumeskin2lay_webs = np.zeros(self.n_webs)
        fabric2lay_webs = np.zeros(self.n_webs)
        mass_webs = np.zeros(self.n_webs)
        web_indices = np.zeros((self.n_webs, 2), dtype=int)
        spar_cap_width_ss = np.zeros(self.n_span)
        spar_cap_width_ps = np.zeros(self.n_span)
        spar_cap_length_ss = 0.0
        spar_cap_length_ps = 0.0
        width_sc_start_ss = 0.0
        width_sc_end_ss = 0.0
        width_sc_start_ps = 0.0
        width_sc_end_ps = 0.0
        fabric2lay_sc_ss = 0.0
        volume2lay_sc_ss = 0.0
        fabric2lay_sc_ps = 0.0
        volume2lay_sc_ps = 0.0
        mass_sc_ss = 0.0
        mass_sc_ps = 0.0
        fabric2lay_shell_ss = 0.0
        fabric2lay_shell_ps = 0.0
        n_plies_root_ss = 0.0
        n_plies_root_ps = 0.0
        volume_root_preform_ss = 0.0
        volume_root_preform_ps = 0.0
        areacore2lay_shell_ss = 0.0
        areacore2lay_shell_ps = 0.0
        fabric2lay_te_reinf_ss = 0.0
        fabric2lay_te_reinf_ps = 0.0
        fabric2lay_le_reinf_ss = 0.0
        fabric2lay_le_reinf_ps = 0.0
        mass_root_preform_ss = 0.0
        mass_root_preform_ps = 0.0
        mass_shell_ss = 0.0
        mass_shell_ps = 0.0
        tol_LE = 1.0e-5
        for i_lay in range(self.n_layers):
            imin, imax = np.nonzero(layer_thickness[i_lay, :])[0][[0, -1]]
            width_ss = np.zeros(self.n_span)
            width_ps = np.zeros(self.n_span)
            if layer_web[i_lay] == 0:
                # Determine on which of the two molds the layer should go
                if (
                    layer_start_nd[i_lay, imin] < xy_arc_nd_LE[imin] + tol_LE
                    and layer_end_nd[i_lay, imin] > xy_arc_nd_LE[imin] - tol_LE
                ):
                    SS = True
                    PS = True
                elif (layer_start_nd[i_lay, imin] > xy_arc_nd_LE[imin] - tol_LE
                    and layer_end_nd[i_lay, imin] <= 1. + tol_LE
                    and layer_start_nd[i_lay, imin] < layer_end_nd[i_lay, imin]):
                    SS = False
                    PS = True
                elif (layer_start_nd[i_lay, imin] < xy_arc_nd_LE[imin] + tol_LE
                    and layer_end_nd[i_lay, imin] < xy_arc_nd_LE[imin] + tol_LE):
                    SS = True
                    PS = False
                else:
                    SS = False
                    PS = False
                    logger.debug("The layer " + self.layer_name[i_lay] + " cannot be assigned " + 
                    " neither to suction nor to pressure. Please check your input geometry yaml.")

                # Compute width layer
                width = arc_L_i * chord * (layer_end_nd[i_lay, :] - layer_start_nd[i_lay, :])
                # Compute width in the suction side
                if SS and PS:
                    width_ss = arc_L_i * chord * (xy_arc_nd_LE - layer_start_nd[i_lay, :])
                    width_ps = arc_L_i * chord * (layer_end_nd[i_lay, :] - xy_arc_nd_LE)
                elif SS and not PS:
                    width_ss = width
                else:
                    width_ps = width
                # Compute the volume per unit meter for each layer split per side
                layer_volume_span_ss[i_lay, :] = layer_thickness[i_lay, :] * width_ss
                layer_volume_span_ps[i_lay, :] = layer_thickness[i_lay, :] * width_ps
            else:
                SS = False
                PS = False
                # Compute the volume per unit meter for each layer
                layer_volume_span_webs[i_lay, :] = layer_thickness[i_lay, :] * web_height[int(layer_web[i_lay]) - 1, :]
                # Compute length of shear webs
                if web_length[int(layer_web[i_lay]) - 1] == 0:
                    web_length[int(layer_web[i_lay]) - 1] = (s[imax] - s[imin]) * blade_length
                    web_indices[int(layer_web[i_lay]) - 1, :] = [imin, imax]
            # Compute volume of layer
            layer_volume_span = (
                layer_volume_span_ss[i_lay, :] + layer_volume_span_ps[i_lay, :] + layer_volume_span_webs[i_lay, :]
            )
            layer_volume[i_lay] = np.trapz(layer_volume_span, s * blade_length)

            # Assign volume to corresponding material
            mat_name = self.layer_mat[i_lay]
            i_mat = discrete_inputs["mat_name"].index(mat_name)
            mat_volume[i_mat] += layer_volume[i_lay]

            # Root plies
            if orth[i_mat] and layer_thickness[i_lay, 0] > 0.0:
                if SS:
                    n_plies_root_ss += layer_thickness[i_lay, 0] / ply_t[i_mat]
                if PS:
                    n_plies_root_ps += layer_thickness[i_lay, 0] / ply_t[i_mat]

            # Root volume
            if orth[i_mat]:
                layer_volume_span_interp_ss = np.interp(root_preform_length, s, layer_volume_span_ss[i_lay, :])
                layer_volume_span_interp_ps = np.interp(root_preform_length, s, layer_volume_span_ps[i_lay, :])
                add_volume_ss = np.trapz(
                    [layer_volume_span_ss[i_lay, 0], layer_volume_span_interp_ss],
                    [0, blade_length * root_preform_length],
                )
                add_volume_ps = np.trapz(
                    [layer_volume_span_ps[i_lay, 0], layer_volume_span_interp_ps],
                    [0, blade_length * root_preform_length],
                )
                volume_root_preform_ss += add_volume_ss
                volume_root_preform_ps += add_volume_ps
                mass_root_preform_ss += add_volume_ss * rho_mat[i_mat]
                mass_root_preform_ps += add_volume_ps * rho_mat[i_mat]
                width_ss_interp = np.interp(root_preform_length, s, width_ss)
                width_ps_interp = np.interp(root_preform_length, s, width_ps)
                area_root_ss = np.trapz([width_ss[0], width_ss_interp], [0, blade_length * root_preform_length])
                area_root_ps = np.trapz([width_ps[0], width_ps_interp], [0, blade_length * root_preform_length])

            # Fabric shear webs
            if layer_web[i_lay] != 0:
                add_volume = np.trapz(layer_volume_span_webs[i_lay, :], s * blade_length)
                mass_webs[layer_web[i_lay] - 1] += add_volume * rho_mat[i_mat]
                if orth[i_mat]:
                    volumeskin2lay_webs[layer_web[i_lay] - 1] += add_volume
                    fabric2lay_webs[layer_web[i_lay] - 1] += add_volume / ply_t[i_mat]

            # Spar caps
            elif self.layer_name[i_lay] == self.spar_cap_ss:
                spar_cap_width_ss[imin:imax] = width[imin:imax]
                spar_cap_length_ss = (s[imax] - s[imin]) * blade_length
                width_sc_start_ss = width[imin]
                width_sc_end_ss = width[imax]
                area_sc_ss = np.trapz(width[imin:imax], s[imin:imax] * blade_length)
                volume2lay_sc_ss = np.trapz(layer_volume_span_ss[i_lay, :], s * blade_length)
                fabric2lay_sc_ss = volume2lay_sc_ss / ply_t[i_mat]
                mass_sc_ss = volume2lay_sc_ss * rho_mat[i_mat]
            elif self.layer_name[i_lay] == self.spar_cap_ps:
                spar_cap_width_ps[imin:imax] = width[imin:imax]
                spar_cap_length_ps = (s[imax] - s[imin]) * blade_length
                width_sc_start_ps = width[imin]
                width_sc_end_ps = width[imax]
                area_sc_ps = np.trapz(width[imin:imax], s[imin:imax] * blade_length)
                volume2lay_sc_ps = np.trapz(layer_volume_span_ss[i_lay, :], s * blade_length)
                fabric2lay_sc_ps = volume2lay_sc_ps / ply_t[i_mat]
                mass_sc_ps = volume2lay_sc_ps * rho_mat[i_mat]

            # Shell skins
            elif component_id[i_mat] == 2:
                volume2lay_shell_ss = np.trapz(layer_volume_span_ss[i_lay, :], s * blade_length)
                volume2lay_shell_ps = np.trapz(layer_volume_span_ps[i_lay, :], s * blade_length)
                fabric2lay_shell_ss += volume2lay_shell_ss / ply_t[i_mat]
                fabric2lay_shell_ps += volume2lay_shell_ps / ply_t[i_mat]
                mass_shell_ss += volume2lay_shell_ss * rho_mat[i_mat]
                mass_shell_ps += volume2lay_shell_ps * rho_mat[i_mat]

            # Shell core
            elif component_id[i_mat] == 1:
                areacore2lay_shell_ss += np.trapz(width_ss[imin:imax], s[imin:imax] * blade_length)
                areacore2lay_shell_ps += np.trapz(width_ps[imin:imax], s[imin:imax] * blade_length)
                volume2lay_coreshell_ss = np.trapz(layer_volume_span_ss[i_lay, imin:imax], s[imin:imax] * blade_length)
                volume2lay_coreshell_ps = np.trapz(layer_volume_span_ps[i_lay, imin:imax], s[imin:imax] * blade_length)
                mass_shell_ss += volume2lay_coreshell_ss * rho_mat[i_mat]
                mass_shell_ps += volume2lay_coreshell_ps * rho_mat[i_mat]

            # TE/LE reinforcement
            elif component_id[i_mat] != 0:
                length2lay_reinf = np.trapz(layer_thickness[i_lay, imin:imax], s[imin:imax] * blade_length)
                volume2lay_reinf_ss = np.trapz(layer_volume_span_ss[i_lay, imin:imax], s[imin:imax] * blade_length)
                volume2lay_reinf_ps = np.trapz(layer_volume_span_ps[i_lay, imin:imax], s[imin:imax] * blade_length)
                if np.mean(layer_start_nd[i_lay, :]) > 0.0 and np.mean(layer_end_nd[i_lay, :]) > 0.0:
                    LE = True
                    TE = False
                else:
                    LE = False
                    TE = True
                if SS:
                    mass_shell_ss += volume2lay_reinf_ss * rho_mat[i_mat]
                    if LE:
                        fabric2lay_le_reinf_ss += length2lay_reinf / ply_t[i_mat]
                    else:
                        fabric2lay_te_reinf_ss += length2lay_reinf / ply_t[i_mat]
                else:
                    mass_shell_ps += volume2lay_reinf_ps * rho_mat[i_mat]
                    if LE:
                        fabric2lay_le_reinf_ps += length2lay_reinf / ply_t[i_mat]
                    else:
                        fabric2lay_te_reinf_ps += length2lay_reinf / ply_t[i_mat]
            # else:
            #     print("Layer not accounted for in the labor and cycle time model")

        # Compute masses of laminates with and without waste factor
        mat_mass = mat_volume * rho_mat
        mat_mass_scrap = mat_volume * rho_mat * (1.0 + waste)

        # Compute costs
        dry_laminate_mass = mat_mass * fwf
        complementary_mass = mat_mass * (1.0 - fwf)
        mat_cost = np.zeros(self.n_mat)
        i_resin = discrete_inputs["mat_name"].index("resin")
        for i_mat in range(self.n_mat):
            if fwf[i_mat] == 0:
                mat_cost[i_mat] += complementary_mass[i_mat] * unit_cost[i_mat]
            else:
                mat_cost[i_mat] = dry_laminate_mass[i_mat] * unit_cost[i_mat]
                mat_cost[i_resin] += complementary_mass[i_mat] * unit_cost[i_resin]

        mat_cost_scrap = mat_cost * (1.0 + waste)

        # Compute total fabric area, with and without scrap factor
        mat_area = np.zeros_like(mat_volume)
        mat_area[ply_t != 0.0] = mat_volume[ply_t != 0.0] / ply_t[ply_t != 0.0]
        mat_area_scrap = mat_area * (1.0 + waste)

        # Estimate adhesive mass and costs
        length_bonding_lines = 2.0 * blade_length + 2 * np.sum(web_length)
        bonding_lines_vol = length_bonding_lines * flange_thick * flange_width * (1.0 + flange_adhesive_squeezed)
        if "adhesive" not in discrete_inputs["mat_name"] and "Adhesive" not in discrete_inputs["mat_name"]:
            raise Exception(
                "Warning: a material named adhesive (or Adhesive) is not defined in the input yaml.  This is required for the blade cost model"
            )
        try:
            i_adhesive = discrete_inputs["mat_name"].index("adhesive")
        except:
            i_adhesive = discrete_inputs["mat_name"].index("Adhesive")
        mat_mass[i_adhesive] += bonding_lines_vol * rho_mat[i_adhesive]
        mat_cost[i_adhesive] += mat_mass[i_adhesive] * unit_cost[i_adhesive]
        mat_mass_scrap[i_adhesive] += bonding_lines_vol * rho_mat[i_adhesive]
        mat_cost_scrap[i_adhesive] += mat_mass_scrap[i_adhesive] * unit_cost[i_adhesive]

        # Hub connection and lightning protection system
        n_bolts = np.pi * chord[0] / t_bolt_spacing
        bolts_cost = n_bolts * t_bolt_unit_cost
        nuts_cost = n_bolts * barrel_nut_unit_cost
        mid_span_station = np.argmin(abs(s - 0.5))
        # LPS_mass = LPS_unit_mass * (blade_length + chord[mid_span_station])
        LPS_cost = LPS_unit_cost * (blade_length + chord[mid_span_station])
        # tot_metallic_mass = LPS_mass + n_bolts * (t_bolt_unit_mass + barrel_nut_unit_mass)
        total_metallic_parts_cost = bolts_cost + nuts_cost + LPS_cost

        # Consumables
        bom = blade_bom()
        web_area = np.zeros(self.n_webs)
        for i_web in range(self.n_webs):
            web_area[i_web] = np.trapz(
                web_height[i_web, web_indices[i_web, 0] : web_indices[i_web, 1]],
                blade_length * s[web_indices[i_web, 0] : web_indices[i_web, 1]],
            )
        web_area_w_flanges = web_area + 2.0 * web_length * flange_width
        ss_area = np.trapz(sect_perimeter_ss, blade_length * s)
        ps_area = np.trapz(sect_perimeter_ps, blade_length * s)
        ss_area_w_flanges = ss_area + 2.0 * flange_width * blade_length
        ps_area_w_flanges = ps_area + 2.0 * flange_width * blade_length
        spar_cap_ss_area = np.trapz(spar_cap_width_ss, blade_length * s)
        spar_cap_ps_area = np.trapz(spar_cap_width_ps, blade_length * s)
        sect_perimeter_ss_interp = np.interp(root_preform_length, s, sect_perimeter_ss)
        ss_area_root = np.trapz(
            [sect_perimeter_ss[0], sect_perimeter_ss_interp], [0, blade_length * root_preform_length]
        )
        sect_perimeter_ps_interp = np.interp(root_preform_length, s, sect_perimeter_ps)
        ps_area_root = np.trapz(
            [sect_perimeter_ps[0], sect_perimeter_ps_interp], [0, blade_length * root_preform_length]
        )
        bom.blade_specs = {}
        bom.blade_specs["area_webs_w_flanges"] = web_area_w_flanges
        bom.blade_specs["area_lpskin_wo_flanges"] = ss_area
        bom.blade_specs["area_hpskin_wo_flanges"] = ps_area
        bom.blade_specs["area_lpskin_w_flanges"] = ss_area_w_flanges
        bom.blade_specs["area_hpskin_w_flanges"] = ps_area_w_flanges
        bom.blade_specs["area_sc_lp"] = spar_cap_ss_area
        bom.blade_specs["area_sc_hp"] = spar_cap_ps_area
        bom.blade_specs["area_lp_root"] = ss_area_root
        bom.blade_specs["area_hp_root"] = ps_area_root
        bom.blade_specs["TE_length"] = blade_length
        bom.blade_specs["LE_length"] = blade_length
        bom.blade_specs["length_webs"] = web_length
        bom.blade_specs["blade_length"] = blade_length
        consumables = bom.compute_consumables()
        name_consumables = consumables.keys()
        total_consumable_cost_wo_waste = 0.0
        total_consumable_cost_w_waste = 0.0
        consumable_cost_w_waste = []
        for name in name_consumables:
            total_consumable_cost_wo_waste = total_consumable_cost_wo_waste + consumables[name]["total_cost_wo_waste"]
            total_consumable_cost_w_waste = total_consumable_cost_w_waste + consumables[name]["total_cost_w_waste"]
            consumable_cost_w_waste.append(consumables[name]["total_cost_w_waste"])

        total_blade_mat_cost_w_waste = (
            np.sum(mat_cost_scrap) + total_metallic_parts_cost + total_consumable_cost_w_waste + joint_cost
        )

        # Labor and cycle time
        blade_specs = {}
        mat_dictionary = {}
        mat_dictionary["mat_name"] = discrete_inputs["mat_name"]
        mat_dictionary["orth"] = orth
        mat_dictionary["component_id"] = component_id
        mat_dictionary["roll_mass"] = roll_mass
        mat_dictionary["total_mass_w_waste"] = mat_mass_scrap
        mat_dictionary["total_ply_area_w_waste"] = mat_area
        mat_dictionary["total_ply_area_wo_waste"] = mat_area_scrap

        metallic_parts = {}
        blade_specs["blade_length"] = blade_length
        blade_specs["root_preform_length"] = root_preform_length
        blade_specs["root_D"] = chord[0]
        blade_specs["n_plies_root_lp"] = n_plies_root_ss
        blade_specs["n_plies_root_hp"] = n_plies_root_ps
        blade_specs["volume_root_preform_lp"] = volume_root_preform_ss
        blade_specs["volume_root_preform_hp"] = volume_root_preform_ps
        blade_specs["n_webs"] = self.n_webs
        blade_specs["length_webs"] = web_length
        web_height_start = np.zeros(self.n_webs)
        web_height_end = np.zeros(self.n_webs)
        for i in range(self.n_webs):
            web_height_start[i] = web_height[i, web_indices[i, 0]]
            web_height_end[i] = web_height[i, web_indices[i, 1]]
        blade_specs["height_webs_start"] = web_height_start
        blade_specs["height_webs_end"] = web_height_end
        blade_specs["area_webs_w_core"] = web_area
        blade_specs["area_webs_w_flanges"] = web_area_w_flanges
        blade_specs["fabric2lay_webs"] = fabric2lay_webs
        blade_specs["volumeskin2lay_webs"] = volumeskin2lay_webs
        blade_specs["length_sc_lp"] = spar_cap_length_ss
        blade_specs["length_sc_hp"] = spar_cap_length_ps
        blade_specs["width_sc_start_lp"] = width_sc_start_ss
        blade_specs["width_sc_end_lp"] = width_sc_end_ss
        blade_specs["width_sc_start_hp"] = width_sc_start_ps
        blade_specs["width_sc_end_hp"] = width_sc_end_ps
        blade_specs["fabric2lay_sc_lp"] = fabric2lay_sc_ss
        blade_specs["fabric2lay_sc_hp"] = fabric2lay_sc_ps
        blade_specs["volume2lay_sc_lp"] = volume2lay_sc_ss
        blade_specs["volume2lay_sc_hp"] = volume2lay_sc_ps
        blade_specs["area_lpskin_wo_flanges"] = ss_area
        blade_specs["area_hpskin_wo_flanges"] = ps_area
        blade_specs["area_lpskin_w_flanges"] = ss_area_w_flanges
        blade_specs["area_hpskin_w_flanges"] = ps_area_w_flanges
        blade_specs["fabric2lay_shell_lp"] = fabric2lay_shell_ss
        blade_specs["fabric2lay_shell_hp"] = fabric2lay_shell_ps
        blade_specs["areacore2lay_shell_lp"] = areacore2lay_shell_ss
        blade_specs["areacore2lay_shell_hp"] = areacore2lay_shell_ps
        blade_specs["fabric2lay_te_reinf_lp"] = fabric2lay_te_reinf_ss
        blade_specs["fabric2lay_te_reinf_hp"] = fabric2lay_te_reinf_ps
        blade_specs["fabric2lay_le_reinf_lp"] = fabric2lay_le_reinf_ss
        blade_specs["fabric2lay_le_reinf_hp"] = fabric2lay_le_reinf_ps
        blade_specs["skin_perimeter_wo_root"] = 2.0 * blade_length * (1.0 - root_preform_length)
        blade_specs["skin_perimeter_w_root"] = 2.0 * blade_length
        metallic_parts["n_bolts"] = n_bolts

        labor_ct = blade_labor_ct(blade_specs, mat_dictionary, metallic_parts)
        operation, labor_hours, skin_mold_gating_ct, non_gating_ct = labor_ct.execute_blade_labor_ct()
        total_labor_hours = sum(labor_hours)
        total_skin_mold_gating_ct = sum(skin_mold_gating_ct)
        total_non_gating_ct = sum(non_gating_ct)

        # Virtual factory
        blade_specs["max_chord"] = np.max(chord)
        blade_specs["mass_root_preform_lp"] = mass_root_preform_ss
        blade_specs["mass_root_preform_hp"] = mass_root_preform_ps
        blade_specs["mass_webs"] = mass_webs
        blade_specs["mass_sc_lp"] = mass_sc_ss
        blade_specs["mass_sc_hp"] = mass_sc_ps
        blade_specs["mass_shell_lp"] = mass_shell_ss
        blade_specs["mass_shell_hp"] = mass_shell_ps
        blade_specs["blade_mass"] = (
            mass_root_preform_ss
            + mass_root_preform_ps
            + np.sum(mass_webs)
            + mass_sc_ss
            + mass_sc_ps
            + mass_shell_ss
            + mass_shell_ps
        )
        blade_specs["area_lp_root"] = area_root_ss
        blade_specs["area_hp_root"] = area_root_ps
        blade_specs["area_sc_lp"] = area_sc_ss
        blade_specs["area_sc_hp"] = area_sc_ps
        vf = virtual_factory(blade_specs, operation, skin_mold_gating_ct, non_gating_ct)
        total_cost_labor, total_labor_overhead = vf.execute_direct_labor_cost(operation, labor_hours)
        total_cost_utility = vf.execute_utility_cost(operation, skin_mold_gating_ct + non_gating_ct)
        blade_variable_cost = total_blade_mat_cost_w_waste + total_cost_labor + total_cost_utility
        (
            total_cost_equipment,
            total_cost_tooling,
            total_cost_building,
            total_maintenance_cost,
            cost_capital,
        ) = vf.execute_fixed_cost(
            operation, skin_mold_gating_ct + non_gating_ct, blade_variable_cost + total_labor_overhead
        )
        blade_fixed_cost = (
            total_cost_equipment
            + total_cost_tooling
            + total_cost_building
            + total_maintenance_cost
            + total_labor_overhead
            + cost_capital
        )

        # Total blade cost
        total_blade_cost = blade_variable_cost + blade_fixed_cost

        # Assign outputs
        outputs["sect_perimeter"] = sect_perimeter
        outputs["layer_volume"] = layer_volume
        outputs["mat_volume"] = mat_volume
        outputs["mat_mass"] = mat_mass
        outputs["mat_cost"] = mat_cost
        outputs["mat_cost_scrap"] = mat_cost_scrap
        outputs["total_metallic_parts_cost"] = total_metallic_parts_cost
        outputs["total_consumable_cost_w_waste"] = total_consumable_cost_w_waste
        outputs["total_blade_mat_cost_w_waste"] = total_blade_mat_cost_w_waste
        # Labor and cycle time
        outputs["total_labor_hours"] = total_labor_hours
        outputs["total_skin_mold_gating_ct"] = total_skin_mold_gating_ct
        outputs["total_non_gating_ct"] = total_non_gating_ct
        # Total costs
        outputs["total_cost_labor"] = total_cost_labor
        outputs["total_cost_utility"] = total_cost_utility
        outputs["blade_variable_cost"] = blade_variable_cost
        outputs["total_cost_equipment"] = total_cost_equipment
        outputs["total_cost_tooling"] = total_cost_tooling
        outputs["total_cost_building"] = total_cost_building
        outputs["total_maintenance_cost"] = total_maintenance_cost
        outputs["total_labor_overhead"] = total_labor_overhead
        outputs["cost_capital"] = cost_capital
        outputs["blade_fixed_cost"] = blade_fixed_cost
        outputs["total_blade_cost"] = total_blade_cost


# OpenMDAO group to execute the blade cost model without the rest of WISDEM
class StandaloneRotorCost(om.Group):
    def initialize(self):
        self.options.declare("modeling_options")
        self.options.declare("opt_options")

    def setup(self):
        modeling_options = self.options["modeling_options"]
        opt_options = self.options["opt_options"]

        # Material dictionary inputs
        self.add_subsystem(
            "materials",
            Materials(mat_init_options=modeling_options["materials"], composites=modeling_options["flags"]["blade"]),
        )

        # Airfoil dictionary inputs
        airfoils = om.IndepVarComp()
        rotorse_options = modeling_options["WISDEM"]["RotorSE"]
        n_af = rotorse_options["n_af"]  # Number of airfoils
        n_xy = rotorse_options["n_xy"]  # Number of coordinate points to describe the airfoil geometry
        airfoils.add_discrete_output("name", val=n_af * [""], desc="1D array of names of airfoils.")
        airfoils.add_output("r_thick", val=np.zeros(n_af), desc="1D array of the relative thicknesses of each airfoil.")
        # Airfoil coordinates
        airfoils.add_output(
            "coord_xy",
            val=np.zeros((n_af, n_xy, 2)),
            desc="3D array of the x and y airfoil coordinates of the n_af airfoils.",
        )
        self.add_subsystem("airfoils", airfoils)

        self.add_subsystem(
            "blade",
            Blade(
                rotorse_options=modeling_options["WISDEM"]["RotorSE"],
                opt_options=opt_options,
            ),
        )
        self.connect("airfoils.name", "blade.interp_airfoils.name")
        self.connect("airfoils.r_thick", "blade.interp_airfoils.r_thick")
        self.connect("airfoils.coord_xy", "blade.interp_airfoils.coord_xy")

        self.add_subsystem(
            "high_level_blade_props",
            ComputeHighLevelBladeProperties(rotorse_options=modeling_options["WISDEM"]["RotorSE"]),
        )
        self.connect("blade.outer_shape_bem.ref_axis", "high_level_blade_props.blade_ref_axis_user")

        self.add_subsystem("rc", RotorCost(mod_options=modeling_options, opt_options=opt_options))

        self.connect("high_level_blade_props.blade_length", "rc.blade_length")
        self.connect("blade.outer_shape_bem.s", "rc.s")
        self.connect("blade.pa.chord_param", "rc.chord")
        self.connect("blade.interp_airfoils.coord_xy_interp", "rc.coord_xy_interp")
        self.connect("blade.internal_structure_2d_fem.layer_thickness", "rc.layer_thickness")
        self.connect("blade.internal_structure_2d_fem.layer_start_nd", "rc.layer_start_nd")
        self.connect("blade.internal_structure_2d_fem.layer_end_nd", "rc.layer_end_nd")
        self.connect("blade.internal_structure_2d_fem.layer_web", "rc.layer_web")
        self.connect("blade.internal_structure_2d_fem.definition_layer", "rc.definition_layer")
        self.connect("blade.internal_structure_2d_fem.web_start_nd", "rc.web_start_nd")
        self.connect("blade.internal_structure_2d_fem.web_end_nd", "rc.web_end_nd")
        self.connect("blade.internal_structure_2d_fem.joint_position", "rc.joint_position")
        self.connect("blade.internal_structure_2d_fem.joint_mass", "rc.joint_mass")
        self.connect("blade.internal_structure_2d_fem.joint_cost", "rc.joint_cost")
        self.connect("materials.name", "rc.mat_name")
        self.connect("materials.orth", "rc.orth")
        self.connect("materials.rho", "rc.rho")
        self.connect("materials.component_id", "rc.component_id")
        self.connect("materials.unit_cost", "rc.unit_cost")
        self.connect("materials.waste", "rc.waste")
        self.connect("materials.rho_fiber", "rc.rho_fiber")
        self.connect("materials.ply_t", "rc.ply_t")
        self.connect("materials.fwf", "rc.fwf")
        self.connect("materials.roll_mass", "rc.roll_mass")


def initialize_omdao_prob(wt_opt, modeling_options, wt_init):

    materials = wt_init["materials"]
    wt_opt = assign_material_values(wt_opt, modeling_options, materials)

    blade = wt_init["components"]["blade"]
    wt_opt = assign_blade_values(wt_opt, modeling_options, blade)

    airfoils = wt_init["airfoils"]
    wt_opt = assign_airfoil_values(wt_opt, modeling_options, airfoils, coordinates_only=True)

    return wt_opt


if __name__ == "__main__":

    wisdem_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    example_dir = os.path.join(wisdem_dir, "examples", "02_reference_turbines")  # get path example 03_blade
    fname_wt_input = os.path.join(example_dir, "IEA-3p4-130-RWT.yaml")
    fname_modeling_options = os.path.join(example_dir, "modeling_options.yaml")
    fname_opt_options = os.path.join(example_dir, "analysis_options.yaml")
    wt_initial = WindTurbineOntologyPython(fname_wt_input, fname_modeling_options, fname_opt_options)
    wt_init, modeling_options, opt_options = wt_initial.get_input_data()
    modeling_options["WISDEM"]["RotorSE"]["flag"] = False
    wt_opt = om.Problem(model=StandaloneRotorCost(modeling_options=modeling_options, opt_options=opt_options))
    wt_opt.setup(derivatives=False)
    myopt = PoseOptimization(wt_init, modeling_options, opt_options)
    wt_opt = myopt.set_initial(wt_opt, wt_init)
    wt_opt = initialize_omdao_prob(wt_opt, modeling_options, wt_init)
    wt_opt.run_model()
