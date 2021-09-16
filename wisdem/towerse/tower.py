import numpy as np
import openmdao.api as om
import wisdem.towerse.tower_props as tp
import wisdem.towerse.tower_struct as ts
import wisdem.commonse.utilization_constraints as util_con
from wisdem.towerse import NPTS_SOIL, get_nfull
from wisdem.commonse.environment import TowerSoil
from wisdem.commonse.cross_sections import CylindricalShellProperties
from wisdem.commonse.wind_wave_drag import CylinderEnvironment


class TowerLeanSE(om.Group):
    """
    This is a geometry preprocessing group for the tower.

    This group contains components that calculate the geometric properties of
    the tower, such as mass and moments of inertia, as well as geometric
    constraints like diameter-to-thickness and taper ratio. No static or dynamic
    analysis of the tower occurs here.

    """

    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        mod_opt = self.options["modeling_options"]["WISDEM"]["TowerSE"]

        n_height_tow = mod_opt["n_height_tower"]
        n_layers_tow = mod_opt["n_layers_tower"]
        n_height_mon = mod_opt["n_height_monopile"]
        n_layers_mon = mod_opt["n_layers_monopile"]
        if "n_height" in mod_opt:
            n_height = mod_opt["n_height"]
        else:
            n_height = mod_opt["n_height"] = n_height_tow if n_height_mon == 0 else n_height_tow + n_height_mon - 1
        nFull = get_nfull(n_height, nref=mod_opt["n_refine"])

        self.set_input_defaults("gravity_foundation_mass", 0.0, units="kg")
        self.set_input_defaults("transition_piece_mass", 0.0, units="kg")
        self.set_input_defaults("tower_outer_diameter", np.ones(n_height), units="m")
        self.set_input_defaults("tower_wall_thickness", np.ones(n_height), units="m")
        self.set_input_defaults("outfitting_factor", np.zeros(n_height - 1))
        self.set_input_defaults("water_depth", 0.0, units="m")
        self.set_input_defaults("hub_height", 0.0, units="m")
        self.set_input_defaults("rho", np.zeros(n_height - 1), units="kg/m**3")
        self.set_input_defaults("unit_cost", np.zeros(n_height - 1), units="USD/kg")
        self.set_input_defaults("labor_cost_rate", 0.0, units="USD/min")
        self.set_input_defaults("painting_cost_rate", 0.0, units="USD/m**2")

        # Inputs here are the outputs from the Tower component in load_IEA_yaml
        # TODO: Use reference axis and curvature, s, instead of assuming everything is vertical on z
        self.add_subsystem(
            "yaml",
            tp.DiscretizationYAML(
                n_height_tower=n_height_tow,
                n_height_monopile=n_height_mon,
                n_layers_tower=n_layers_tow,
                n_layers_monopile=n_layers_mon,
                n_mat=self.options["modeling_options"]["materials"]["n_mat"],
            ),
            promotes=["*"],
        )

        # Promote all but foundation_height so that we can override
        self.add_subsystem(
            "geometry",
            tp.CylinderDiscretization(nPoints=n_height),
            promotes=[
                "z_param",
                "z_full",
                "d_full",
                "t_full",
                ("section_height", "tower_section_height"),
                ("diameter", "tower_outer_diameter"),
                ("wall_thickness", "tower_wall_thickness"),
            ],
        )

        self.add_subsystem(
            "props", CylindricalShellProperties(nFull=nFull), promotes=["Az", "Asx", "Asy", "Ixx", "Iyy", "Jz"]
        )
        self.add_subsystem("tgeometry", tp.TowerDiscretization(n_height=n_height, n_refine=mod_opt["n_refine"]), promotes=["*"])

        self.add_subsystem(
            "cm",
            tp.CylinderMass(nPoints=nFull),
            promotes=["z_full", "d_full", "t_full", "labor_cost_rate", "painting_cost_rate"],
        )
        self.add_subsystem(
            "tm",
            tp.TowerMass(n_height=n_height,
                         n_refine=mod_opt["n_refine"]),
            promotes=[
                "z_full",
                "d_full",
                "tower_mass",
                "tower_center_of_mass",
                "tower_section_center_of_mass",
                "tower_I_base",
                "tower_cost",
                "gravity_foundation_mass",
                "gravity_foundation_I",
                "transition_piece_mass",
                "transition_piece_cost",
                "transition_piece_height",
                "transition_piece_I",
                "monopile_mass",
                "monopile_cost",
                "structural_mass",
                "structural_cost",
            ],
        )
        self.add_subsystem(
            "gc",
            util_con.GeometricConstraints(nPoints=n_height),
            promotes=[
                "constr_taper",
                "constr_d_to_t",
                "slope",
                ("d", "tower_outer_diameter"),
                ("t", "tower_wall_thickness"),
            ],
        )

        self.add_subsystem(
            "turb",
            tp.TurbineMass(),
            promotes=[
                "turbine_mass",
                "monopile_mass",
                "tower_mass",
                "tower_center_of_mass",
                "tower_I_base",
                "rna_mass",
                "rna_cg",
                "rna_I",
                "hub_height",
            ],
        )

        # Connections for geometry and mass
        self.connect("z_start", "geometry.foundation_height")
        self.connect("d_full", "props.d")
        self.connect("t_full", "props.t")
        self.connect("rho_full", "cm.rho")
        self.connect("outfitting_full", "cm.outfitting_factor")
        self.connect("unit_cost_full", "cm.material_cost_rate")
        self.connect("cm.mass", "tm.cylinder_mass")
        self.connect("cm.cost", "tm.cylinder_cost")
        self.connect("cm.center_of_mass", "tm.cylinder_center_of_mass")
        self.connect("cm.section_center_of_mass", "tm.cylinder_section_center_of_mass")
        self.connect("cm.I_base", "tm.cylinder_I_base")


class TowerSE(om.Group):
    """
    This is the main TowerSE group that performs analysis of the tower.

    This group takes in geometric inputs from TowerLeanSE and environmental and
    loading conditions.
    """

    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        mod_opt = self.options["modeling_options"]["WISDEM"]["TowerSE"]
        monopile = self.options["modeling_options"]["flags"]["monopile"]
        nLC = mod_opt["nLC"]  # not yet supported
        wind = mod_opt["wind"]  # not yet supported
        frame3dd_opt = mod_opt["frame3dd"]
        if "n_height" in mod_opt:
            n_height = mod_opt["n_height"]
        else:
            n_height_tow = mod_opt["n_height_tower"]
            n_height_mon = mod_opt["n_height_monopile"]
            n_height = mod_opt["n_height"] = n_height_tow if n_height_mon == 0 else n_height_tow + n_height_mon - 1
        nFull = get_nfull(n_height, nref=mod_opt["n_refine"])
        self.set_input_defaults("E", np.zeros(n_height - 1), units="N/m**2")
        self.set_input_defaults("G", np.zeros(n_height - 1), units="N/m**2")
        if monopile and mod_opt["soil_springs"]:
            self.set_input_defaults("G_soil", 0.0, units="N/m**2")
            self.set_input_defaults("nu_soil", 0.0)
        self.set_input_defaults("sigma_y", np.zeros(n_height - 1), units="N/m**2")
        self.set_input_defaults("life", 0.0)

        # Load baseline discretization
        self.add_subsystem("geom", TowerLeanSE(modeling_options=self.options["modeling_options"]), promotes=["*"])

        if monopile and mod_opt["soil_springs"]:
            self.add_subsystem(
                "soil",
                TowerSoil(npts=NPTS_SOIL),
                promotes=[("G", "G_soil"), ("nu", "nu_soil"), ("depth", "suctionpile_depth")],
            )
            self.connect("d_full", "soil.d0", src_indices=[0])

        # Add in all Components that drive load cases
        # Note multiple load cases have to be handled by replicating components and not groups/assemblies.
        # Replicating Groups replicates the IndepVarComps which doesn't play nicely in OpenMDAO
        prom = [("zref", "wind_reference_height"), "shearExp", "z0", "cd_usr", "yaw", "beta_wind", "rho_air", "mu_air"]
        if monopile:
            prom += [
                "beta_wave",
                "rho_water",
                "mu_water",
                "cm",
                "Uc",
                "Hsig_wave",
                "Tsig_wave",
                "water_depth",
            ]

        for iLC in range(nLC):
            lc = "" if nLC == 1 else str(iLC + 1)

            self.add_subsystem(
                "wind" + lc, CylinderEnvironment(nPoints=nFull, water_flag=monopile, wind=wind), promotes=prom
            )

            self.add_subsystem(
                "pre" + lc,
                ts.TowerPreFrame(
                    n_height=n_height,
                    monopile=monopile,
                    soil_springs=mod_opt["soil_springs"],
                    gravity_foundation=mod_opt["gravity_foundation"],
                    n_refine=mod_opt["n_refine"]
                ),
                promotes=[
                    "transition_piece_mass",
                    "transition_piece_height",
                    "transition_piece_I",
                    "gravity_foundation_mass",
                    "gravity_foundation_I",
                    "z_full",
                    "suctionpile_depth",
                ],
            )
            self.add_subsystem(
                "tower" + lc,
                ts.CylinderFrame3DD(
                    nFull=nFull,
                    nK=4 if monopile and not mod_opt["gravity_foundation"] else 1,
                    nMass=2,
                    nPL=1,
                    frame3dd_opt=frame3dd_opt,
                ),
                promotes=["Az", "Asx", "Asy", "Ixx", "Iyy", "Jz"],
            )
            self.add_subsystem(
                "post" + lc,
                ts.TowerPostFrame(modeling_options=mod_opt),
                promotes=[
                    "life",
                    "z_full",
                    "d_full",
                    "t_full",
                    "rho_full",
                    "E_full",
                    "G_full",
                    "sigma_y_full",
                    "suctionpile_depth",
                    "Az",
                    "Asx",
                    "Asy",
                    "Ixx",
                    "Iyy",
                    "Jz",
                ],
            )

            self.connect("z_full", ["wind" + lc + ".z", "tower" + lc + ".z"])
            self.connect("d_full", ["wind" + lc + ".d", "tower" + lc + ".d"])
            self.connect("t_full", "tower" + lc + ".t")

            self.connect("rho_full", "tower" + lc + ".rho")
            self.connect("E_full", "tower" + lc + ".E")
            self.connect("G_full", "tower" + lc + ".G")

            self.connect("pre" + lc + ".kidx", "tower" + lc + ".kidx")
            self.connect("pre" + lc + ".kx", "tower" + lc + ".kx")
            self.connect("pre" + lc + ".ky", "tower" + lc + ".ky")
            self.connect("pre" + lc + ".kz", "tower" + lc + ".kz")
            self.connect("pre" + lc + ".ktx", "tower" + lc + ".ktx")
            self.connect("pre" + lc + ".kty", "tower" + lc + ".kty")
            self.connect("pre" + lc + ".ktz", "tower" + lc + ".ktz")
            self.connect("pre" + lc + ".midx", "tower" + lc + ".midx")
            self.connect("pre" + lc + ".m", "tower" + lc + ".m")
            self.connect("pre" + lc + ".mIxx", "tower" + lc + ".mIxx")
            self.connect("pre" + lc + ".mIyy", "tower" + lc + ".mIyy")
            self.connect("pre" + lc + ".mIzz", "tower" + lc + ".mIzz")
            self.connect("pre" + lc + ".mIxy", "tower" + lc + ".mIxy")
            self.connect("pre" + lc + ".mIxz", "tower" + lc + ".mIxz")
            self.connect("pre" + lc + ".mIyz", "tower" + lc + ".mIyz")
            self.connect("pre" + lc + ".mrhox", "tower" + lc + ".mrhox")
            self.connect("pre" + lc + ".mrhoy", "tower" + lc + ".mrhoy")
            self.connect("pre" + lc + ".mrhoz", "tower" + lc + ".mrhoz")

            self.connect("pre" + lc + ".plidx", "tower" + lc + ".plidx")
            self.connect("pre" + lc + ".Fx", "tower" + lc + ".Fx")
            self.connect("pre" + lc + ".Fy", "tower" + lc + ".Fy")
            self.connect("pre" + lc + ".Fz", "tower" + lc + ".Fz")
            self.connect("pre" + lc + ".Mxx", "tower" + lc + ".Mxx")
            self.connect("pre" + lc + ".Myy", "tower" + lc + ".Myy")
            self.connect("pre" + lc + ".Mzz", "tower" + lc + ".Mzz")
            if monopile and mod_opt["soil_springs"]:
                self.connect("soil.z_k", "pre" + lc + ".z_soil")
                self.connect("soil.k", "pre" + lc + ".k_soil")

            self.connect("wind" + lc + ".Px", "tower" + lc + ".Px")
            self.connect("wind" + lc + ".Py", "tower" + lc + ".Py")
            self.connect("wind" + lc + ".Pz", "tower" + lc + ".Pz")

            self.connect("wind" + lc + ".qdyn", "post" + lc + ".qdyn")

            self.connect("tower" + lc + ".tower_Fz", "post" + lc + ".tower_Fz")
            self.connect("tower" + lc + ".tower_Vx", "post" + lc + ".tower_Vx")
            self.connect("tower" + lc + ".tower_Vy", "post" + lc + ".tower_Vy")
            self.connect("tower" + lc + ".tower_Mxx", "post" + lc + ".tower_Mxx")
            self.connect("tower" + lc + ".tower_Myy", "post" + lc + ".tower_Myy")
            self.connect("tower" + lc + ".tower_Mzz", "post" + lc + ".tower_Mzz")
