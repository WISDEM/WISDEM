import copy

import numpy as np
import openmdao.api as om
import wisdem.commonse.utilities as util
import wisdem.commonse.utilization_constraints as util_con
from wisdem.commonse.environment import TowerSoil
from wisdem.commonse.cross_sections import CylindricalShellProperties
from wisdem.commonse.wind_wave_drag import CylinderEnvironment
from wisdem.commonse.vertical_cylinder import (
    NFREQ,
    RIGID,
    CylinderMass,
    CylinderFrame3DD,
    CylinderDiscretization,
    get_nfull,
)

NPTS_SOIL = 10

# -----------------
#  Components
# -----------------


class DiscretizationYAML(om.ExplicitComponent):
    """
    Process some of the tower YAML inputs.

    Parameters
    ----------
    tower_s : numpy array[n_height_tow]
        1D array of the non-dimensional grid defined along the tower axis (0-tower base,
        1-tower top)
    tower_layer_materials : list of strings
        1D array of the names of the materials of each layer modeled in the tower
        structure.
    tower_layer_thickness : numpy array[n_layers_tow, n_height_tow], [m]
        2D array of the thickness of the layers of the tower structure. The first
        dimension represents each layer, the second dimension represents each piecewise-
        constant entry of the tower sections.
    tower_height : float, [m]
        Scalar of the tower height computed along the z axis.
    tower_outer_diameter_in : numpy array[n_height_tow], [m]
        cylinder diameter at corresponding locations
    tower_outfitting_factor : float
        Multiplier that accounts for secondary structure mass inside of cylinder
    monopile_s : numpy array[n_height_mon]
        1D array of the non-dimensional grid defined along the tower axis (0-tower base,
        1-tower top)
    monopile_layer_materials : list of strings
        1D array of the names of the materials of each layer modeled in the tower
        structure.
    monopile_layer_thickness : numpy array[n_layers_mon, n_height_mon], [m]
        2D array of the thickness of the layers of the tower structure. The first
        dimension represents each layer, the second dimension represents each piecewise-
        constant entry of the tower sections.
    monopile_height : float, [m]
        Scalar of the tower height computed along the z axis.
    monopile_outer_diameter_in : numpy array[n_height_tow], [m]
        cylinder diameter at corresponding locations
    monopile_outfitting_factor : float
        Multiplier that accounts for secondary structure mass inside of cylinder
    material_names : list of strings
        1D array of names of materials.
    E_mat : numpy array[n_mat, 3], [Pa]
        2D array of the Youngs moduli of the materials. Each row represents a material,
        the three columns represent E11, E22 and E33.
    G_mat : numpy array[n_mat, 3], [Pa]
        2D array of the shear moduli of the materials. Each row represents a material,
        the three columns represent G12, G13 and G23.
    sigma_y_mat : numpy array[n_mat], [Pa]
        2D array of the yield strength of the materials. Each row represents a material,
        the three columns represent Xt12, Xt13 and Xt23.
    rho_mat : numpy array[n_mat], [kg/m**3]
        1D array of the density of the materials. For composites, this is the density of
        the laminate.
    unit_cost_mat : numpy array[n_mat], [USD/kg]
        1D array of the unit costs of the materials.

    Returns
    -------
    tower_section_height : numpy array[n_height-1], [m]
        parameterized section heights along cylinder
    tower_outer_diameter : numpy array[n_height], [m]
        cylinder diameter at corresponding locations
    tower_wall_thickness : numpy array[n_height-1], [m]
        shell thickness at corresponding locations
    outfitting_factor : numpy array[n_height-1]
        Multiplier that accounts for secondary structure mass inside of cylinder
    E : numpy array[n_height-1], [Pa]
        Isotropic Youngs modulus of the materials along the tower sections.
    G : numpy array[n_height-1], [Pa]
        Isotropic shear modulus of the materials along the tower sections.
    sigma_y : numpy array[n_height-1], [Pa]
        Isotropic yield strength of the materials along the tower sections.
    rho : numpy array[n_height-1], [kg/m**3]
        Density of the materials along the tower sections.
    unit_cost : numpy array[n_height-1], [USD/kg]
        Unit costs of the materials along the tower sections.
    outfitting_factor : numpy array[n_height-1]
        Additional outfitting multiplier in each section

    """

    def initialize(self):
        self.options.declare("n_height_tower")
        self.options.declare("n_height_monopile")
        self.options.declare("n_layers_tower")
        self.options.declare("n_layers_monopile")
        self.options.declare("n_mat")

    def setup(self):
        n_height_tow = self.options["n_height_tower"]
        n_height_mon = self.options["n_height_monopile"]
        n_layers_tow = self.options["n_layers_tower"]
        n_layers_mon = self.options["n_layers_monopile"]
        n_mat = self.options["n_mat"]
        if n_height_mon > 0:
            n_height = n_height_tow + n_height_mon - 1  # Should have one overlapping point
        else:
            n_height = n_height_tow

        # Inputs here are the outputs from the Tower component in load_IEA_yaml
        # TODO: Use reference axis and curvature, s, instead of assuming everything is vertical on z
        self.add_input("tower_s", val=np.zeros(n_height_tow))
        self.add_discrete_input("tower_layer_materials", val=n_layers_tow * [""])
        self.add_input("tower_layer_thickness", val=np.zeros((n_layers_tow, n_height_tow)), units="m")
        self.add_input("tower_height", val=0.0, units="m")
        self.add_input("tower_foundation_height", val=0.0, units="m")
        self.add_input("tower_outer_diameter_in", np.zeros(n_height_tow), units="m")
        self.add_input("tower_outfitting_factor", val=0.0)
        self.add_input("monopile_s", val=np.zeros(n_height_mon))
        self.add_discrete_input("monopile_layer_materials", val=n_layers_tow * [""])
        self.add_input("monopile_layer_thickness", val=np.zeros((n_layers_mon, n_height_mon)), units="m")
        self.add_input("monopile_foundation_height", val=0.0, units="m")
        self.add_input("monopile_height", val=0.0, units="m")
        self.add_input("monopile_outer_diameter_in", np.zeros(n_height_mon), units="m")
        self.add_input("monopile_outfitting_factor", val=0.0)
        self.add_discrete_input("material_names", val=n_mat * [""])
        self.add_input("water_depth", val=0.0, units="m")
        self.add_input("E_mat", val=np.zeros([n_mat, 3]), units="Pa")
        self.add_input("G_mat", val=np.zeros([n_mat, 3]), units="Pa")
        self.add_input("sigma_y_mat", val=np.zeros(n_mat), units="Pa")
        self.add_input("rho_mat", val=np.zeros(n_mat), units="kg/m**3")
        self.add_input("unit_cost_mat", val=np.zeros(n_mat), units="USD/kg")

        self.add_output("tower_section_height", val=np.zeros(n_height - 1), units="m")
        self.add_output("tower_outer_diameter", val=np.zeros(n_height), units="m")
        self.add_output("tower_wall_thickness", val=np.zeros(n_height - 1), units="m")
        self.add_output("transition_piece_height", 0.0, units="m")
        self.add_output("suctionpile_depth", 0.0, units="m")
        self.add_output("outfitting_factor", val=np.zeros(n_height - 1))
        self.add_output("E", val=np.zeros(n_height - 1), units="Pa")
        self.add_output("G", val=np.zeros(n_height - 1), units="Pa")
        self.add_output("sigma_y", val=np.zeros(n_height - 1), units="Pa")
        self.add_output("rho", val=np.zeros(n_height - 1), units="kg/m**3")
        self.add_output("unit_cost", val=np.zeros(n_height - 1), units="USD/kg")
        self.add_output("z_start", 0.0, units="m")

        # self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # Unpack dimensions
        n_height_tow = self.options["n_height_tower"]
        n_height_mon = self.options["n_height_monopile"]
        n_layers_tow = self.options["n_layers_tower"]
        n_layers_mon = self.options["n_layers_monopile"]
        n_height = n_height_tow if n_height_mon == 0 else n_height_tow + n_height_mon - 1

        # Unpack values
        h_mon = inputs["monopile_height"]
        h_tow = inputs["tower_height"]
        s_mon = inputs["monopile_s"]
        s_tow = inputs["tower_s"]
        lthick_mon = inputs["monopile_layer_thickness"]
        lthick_tow = inputs["tower_layer_thickness"]
        lmat_mon = copy.copy(discrete_inputs["monopile_layer_materials"])
        lmat_tow = copy.copy(discrete_inputs["tower_layer_materials"])
        fh_tow = inputs["tower_foundation_height"]
        fh_mon = inputs["monopile_foundation_height"]
        water_depth = inputs["water_depth"]

        # Enforce constant tower thickness per section, assuming rolling of a flat steel plate
        # Might not have to assume this with concrete, but can account for it in input values if necessary
        lthick_tow = 0.5 * (lthick_tow[:, :-1] + lthick_tow[:, 1:])
        if n_height_mon > 0:
            lthick_mon = 0.5 * (lthick_mon[:, :-1] + lthick_mon[:, 1:])

        outputs["transition_piece_height"] = fh_tow

        if n_height_mon > 0:
            if np.abs(fh_tow - fh_mon - h_mon) > 1.0:
                print(
                    "WARNING: Monopile length is not consistent with transition piece height and monopile base height\n",
                    "         Determining new base height value . . .",
                )
            outputs["z_start"] = fh_tow - h_mon

            pile = h_mon - fh_tow - water_depth
            outputs["suctionpile_depth"] = pile
            # Ensure that we have only one segment for pile, a current limitation
            if pile > 0:
                s1 = pile / h_mon
                icheck = np.where(s_mon > s1 + 1e-3)[0][0]
                s_mon = np.r_[0.0, np.linspace(s1, s_mon[icheck], icheck).flatten(), s_mon[(icheck + 1) :].flatten()]

            # Last monopile point and first tower point are the same
            outputs["tower_section_height"] = np.r_[np.diff(h_mon * s_mon), np.diff(h_tow * s_tow)]
            outputs["outfitting_factor"] = np.r_[
                inputs["monopile_outfitting_factor"] * np.ones(n_height_mon - 1),
                inputs["tower_outfitting_factor"] * np.ones(n_height_tow - 1),
            ]
            outputs["tower_outer_diameter"] = np.r_[
                inputs["monopile_outer_diameter_in"], inputs["tower_outer_diameter_in"][1:]
            ]

            # Combine layers into one structure
            layer_mat = []
            twall = np.zeros((1, n_height - 1))
            for k in range(n_layers_mon):
                ilayer = np.zeros(n_height - 1)

                ilayer[: (n_height_mon - 1)] = lthick_mon[k, :]

                imat_mon = lmat_mon[k]
                layer_mat.append(imat_mon)

                if imat_mon in lmat_tow:
                    ktow = lmat_tow.index(imat_mon)
                    ilayer[(n_height_mon - 1) :] = lthick_tow[ktow, :]

                    # Remove from listing so we don't double count later
                    lmat_tow.pop(ktow)
                    lthick_tow = np.delete(lthick_tow, [ktow], axis=0)

                twall = np.vstack((twall, ilayer))

            # If there any uncounted tower layers, add them in
            n_layers_tow = len(lmat_tow)
            for k in range(n_layers_tow):
                ilayer = np.zeros(n_height - 1)
                ilayer[(n_height_mon - 1) :] = lthick_tow[k, :]
                twall = np.vstack((twall, ilayer))
                imat = lmat_tow[k]
                layer_mat.append(imat)

            twall = np.delete(twall, [0], axis=0)
            outputs["tower_wall_thickness"] = np.sum(twall, axis=0)

        else:
            outputs["tower_section_height"] = np.diff(h_tow * s_tow)
            outputs["tower_wall_thickness"] = np.sum(lthick_tow, axis=0)
            outputs["outfitting_factor"] = inputs["tower_outfitting_factor"] * np.ones(n_height - 1)
            outputs["tower_outer_diameter"] = inputs["tower_outer_diameter_in"]
            twall = lthick_tow
            layer_mat = discrete_inputs["tower_layer_materials"]
            outputs["z_start"] = fh_tow
            outputs["suctionpile_depth"] = 0.0

        # Check to make sure we have good values
        if np.any(outputs["tower_section_height"] <= 0.0):
            raise ValueError("Section height values must be greater than zero, " + str(outputs["tower_section_height"]))
        if np.any(outputs["tower_wall_thickness"] <= 0.0):
            raise ValueError("Wall thickness values must be greater than zero, " + str(outputs["tower_wall_thickness"]))
        if np.any(outputs["tower_outer_diameter"] <= 0.0):
            raise ValueError("Diameter values must be greater than zero, " + str(outputs["tower_outer_diameter"]))

        # DETERMINE MATERIAL PROPERTIES IN EACH SECTION
        # Convert to isotropic material
        E = np.mean(inputs["E_mat"], axis=1)
        G = np.mean(inputs["G_mat"], axis=1)
        sigy = inputs["sigma_y_mat"]
        rho = inputs["rho_mat"]
        cost = inputs["unit_cost_mat"]
        mat_names = discrete_inputs["material_names"]

        # Initialize sectional data
        E_param = np.zeros(twall.shape)
        G_param = np.zeros(twall.shape)
        sigy_param = np.zeros(twall.shape)
        rho_param = np.zeros(n_height - 1)
        cost_param = np.zeros(n_height - 1)

        # Loop over materials and associate it with its thickness
        for k in range(len(layer_mat)):
            # Get the material name for this layer
            iname = layer_mat[k]

            # Get the index into the material list
            imat = mat_names.index(iname)

            imass = rho[imat] * twall[k, :]

            # For density, take mass weighted layer
            rho_param += imass

            # For cost, take mass weighted layer
            cost_param += imass * cost[imat]

            # Store the value associated with this thickness
            E_param[k, :] = E[imat]
            G_param[k, :] = G[imat]
            sigy_param[k, :] = sigy[imat]

        # Mass weighted cost (should really weight by radius too)
        cost_param /= rho_param

        # Thickness weighted density (should really weight by radius too)
        rho_param /= twall.sum(axis=0)

        # Mixtures of material properties: https://en.wikipedia.org/wiki/Rule_of_mixtures

        # Volume fraction
        vol_frac = twall / twall.sum(axis=0)[np.newaxis, :]

        # Average of upper and lower bounds
        E_param = 0.5 * np.sum(vol_frac * E_param, axis=0) + 0.5 / np.sum(vol_frac / E_param, axis=0)
        G_param = 0.5 * np.sum(vol_frac * G_param, axis=0) + 0.5 / np.sum(vol_frac / G_param, axis=0)
        sigy_param = 0.5 * np.sum(vol_frac * sigy_param, axis=0) + 0.5 / np.sum(vol_frac / sigy_param, axis=0)

        # Store values
        outputs["E"] = E_param
        outputs["G"] = G_param
        outputs["rho"] = rho_param
        outputs["sigma_y"] = sigy_param
        outputs["unit_cost"] = cost_param


class TowerDiscretization(om.ExplicitComponent):
    """
    Compute the full arrays for some measures along the tower by interpolating.

    Parameters
    ----------
    hub_height : float, [m]
        diameter at tower base
    z_param : numpy array[n_height], [m]
        parameterized locations along tower, linear lofting between
    z_full : numpy array[nFull], [m]
        parameterized locations along tower, linear lofting between
    rho : numpy array[n_height-1], [kg/m**3]
        Density of the materials along the tower sections.
    unit_cost : numpy array[n_height-1], [USD/kg]
        Unit costs of the materials along the tower sections.
    outfitting_factor : numpy array[n_height-1]
        Multiplier that accounts for secondary structure mass inside of cylinder
    E : numpy array[n_height-1], [Pa]
        Isotropic Youngs modulus of the materials along the tower sections.
    G : numpy array[n_height-1], [Pa]
        Isotropic shear modulus of the materials along the tower sections.
    sigma_y : numpy array[n_height-1], [Pa]
        Isotropic yield strength of the materials along the tower sections.
    Az : numpy array[nFull-1], [m**2]
        cross-sectional area
    Jz : numpy array[nFull-1], [m**4]
        polar moment of inertia
    Ixx : numpy array[nFull-1], [m**4]
        area moment of inertia about x-axis
    Iyy : numpy array[nFull-1], [m**4]
        area moment of inertia about y-axis

    Returns
    -------
    height_constraint : float, [m]
        mismatch between tower height and desired hub_height
    rho_full : numpy array[nFull-1], [kg/m**3]
        Density of the materials along the tower sections.
    unit_cost_full : numpy array[nFull-1], [USD/kg]
        Unit costs of the materials along the tower sections.
    outfitting_full : numpy array[nFull-1]
        Multiplier that accounts for secondary structure mass inside of cylinder
    E_full : numpy array[nFull-1], [Pa]
        Isotropic Youngs modulus of the materials along the tower sections.
    G_full : numpy array[nFull-1], [Pa]
        Isotropic shear modulus of the materials along the tower sections.
    sigma_y_full : numpy array[nFull-1], [Pa]
        Isotropic yield strength of the materials along the tower sections.

    """

    def initialize(self):
        self.options.declare("n_height")

    def setup(self):
        n_height = self.options["n_height"]
        nFull = get_nfull(n_height)

        self.add_input("hub_height", val=0.0, units="m")
        self.add_input("z_param", np.zeros(n_height), units="m")
        self.add_input("z_full", val=np.zeros(nFull), units="m")
        self.add_input("rho", val=np.zeros(n_height - 1), units="kg/m**3")
        self.add_input("unit_cost", val=np.zeros(n_height - 1), units="USD/kg")
        self.add_input("outfitting_factor", val=np.zeros(n_height - 1))
        self.add_input("E", val=np.zeros(n_height - 1), units="Pa")
        self.add_input("G", val=np.zeros(n_height - 1), units="Pa")
        self.add_input("sigma_y", val=np.zeros(n_height - 1), units="Pa")

        self.add_input("Az", np.zeros(nFull - 1), units="m**2")
        self.add_input("Jz", np.zeros(nFull - 1), units="m**4")
        self.add_input("Ixx", np.zeros(nFull - 1), units="m**4")
        self.add_input("Iyy", np.zeros(nFull - 1), units="m**4")

        self.add_output("height_constraint", val=0.0, units="m")
        self.add_output("rho_full", val=np.zeros(nFull - 1), units="kg/m**3")
        self.add_output("unit_cost_full", val=np.zeros(nFull - 1), units="USD/kg")
        self.add_output("outfitting_full", val=np.zeros(nFull - 1))
        self.add_output("E_full", val=np.zeros(nFull - 1), units="Pa")
        self.add_output("G_full", val=np.zeros(nFull - 1), units="Pa")
        self.add_output("sigma_y_full", val=np.zeros(nFull - 1), units="Pa")

        # Tower Distributed Beam Properties (properties needed for ElastoDyn (OpenFAST) inputs or BModes inputs for verification purposes)
        self.add_output("sec_loc", np.zeros(n_height - 1), desc="normalized sectional location")
        self.add_output("str_tw", np.zeros(n_height - 1), units="deg", desc="structural twist of section")
        self.add_output("tw_iner", np.zeros(n_height - 1), units="deg", desc="inertial twist of section")
        self.add_output("mass_den", np.zeros(n_height - 1), units="kg/m", desc="sectional mass per unit length")
        self.add_output(
            "foreaft_iner",
            np.zeros(n_height - 1),
            units="kg*m",
            desc="sectional fore-aft intertia per unit length about the Y_G inertia axis",
        )
        self.add_output(
            "sideside_iner",
            np.zeros(n_height - 1),
            units="kg*m",
            desc="sectional side-side intertia per unit length about the Y_G inertia axis",
        )
        self.add_output(
            "foreaft_stff",
            np.zeros(n_height - 1),
            units="N*m**2",
            desc="sectional fore-aft bending stiffness per unit length about the Y_E elastic axis",
        )
        self.add_output(
            "sideside_stff",
            np.zeros(n_height - 1),
            units="N*m**2",
            desc="sectional side-side bending stiffness per unit length about the Y_E elastic axis",
        )
        self.add_output("tor_stff", np.zeros(n_height - 1), units="N*m**2", desc="sectional torsional stiffness")
        self.add_output("axial_stff", np.zeros(n_height - 1), units="N", desc="sectional axial stiffness")
        self.add_output("cg_offst", np.zeros(n_height - 1), units="m", desc="offset from the sectional center of mass")
        self.add_output("sc_offst", np.zeros(n_height - 1), units="m", desc="offset from the sectional shear center")
        self.add_output("tc_offst", np.zeros(n_height - 1), units="m", desc="offset from the sectional tension center")

        self.declare_partials("height_constraint", ["hub_height", "z_param"], method="fd")
        self.declare_partials("outfitting_full", ["outfitting_factor"], method="fd")
        self.declare_partials("rho_full", ["rho"], method="fd")
        self.declare_partials("unit_cost_full", ["unit_cost"], method="fd")

    def compute(self, inputs, outputs):
        z_full = inputs["z_full"]
        z_param = inputs["z_param"]
        z_section = 0.5 * (z_full[:-1] + z_full[1:])

        outputs["height_constraint"] = inputs["hub_height"] - z_param[-1]
        outputs["rho_full"] = util.sectionalInterp(z_section, z_param, inputs["rho"])
        outputs["outfitting_full"] = util.sectionalInterp(z_section, z_param, inputs["outfitting_factor"])
        outputs["unit_cost_full"] = util.sectionalInterp(z_section, z_param, inputs["unit_cost"])
        outputs["E_full"] = util.sectionalInterp(z_section, z_param, inputs["E"])
        outputs["G_full"] = util.sectionalInterp(z_section, z_param, inputs["G"])
        outputs["sigma_y_full"] = util.sectionalInterp(z_section, z_param, inputs["sigma_y"])

        # Unpack for Elastodyn
        z = 0.5 * (z_param[:-1] + z_param[1:])
        rho = inputs["rho"]
        E = inputs["E"]
        G = inputs["G"]
        Az = util.sectionalInterp(z, z_full, inputs["Az"])
        Ixx = util.sectionalInterp(z, z_full, inputs["Ixx"])
        Iyy = util.sectionalInterp(z, z_full, inputs["Iyy"])
        Jz = util.sectionalInterp(z, z_full, inputs["Jz"])
        outputs["sec_loc"] = (z - z[0]) / (z[-1] - z[0])
        outputs["mass_den"] = rho * Az
        outputs["foreaft_iner"] = rho * Ixx
        outputs["sideside_iner"] = rho * Iyy
        outputs["foreaft_stff"] = E * Ixx
        outputs["sideside_stff"] = E * Iyy
        outputs["tor_stff"] = G * Jz
        outputs["axial_stff"] = E * Az


class TowerMass(om.ExplicitComponent):
    """
    Compute the tower and monopile masses, raw cost, and CG properties.

    Parameters
    ----------
    cylinder_mass : numpy array[nFull-1], [kg]
        Total cylinder mass
    cylinder_cost : float, [USD]
        Total cylinder cost
    cylinder_center_of_mass : float, [m]
        z position of center of mass of cylinder
    cylinder_section_center_of_mass : numpy array[nFull-1], [m]
        z position of center of mass of each can in the cylinder
    cylinder_I_base : numpy array[6], [kg*m**2]
        Mass moment of inertia of cylinder about base [xx yy zz xy xz yz]
    transition_piece_height : float, [m]
        Point mass height of transition piece above water line
    transition_piece_mass : float, [kg]
        Point mass of transition piece
    transition_piece_cost : float, [USD]
        Cost of transition piece
    gravity_foundation_mass : float, [kg]
        Extra mass of gravity foundation
    z_full : numpy array[nFull], [m]
        Parameterized locations along tower, linear lofting between
    d_full : numpy array[nFull], [m]
        diameter along tower

    Returns
    -------
    structural_cost : float, [USD]
        Total structural cost (tower+monopile)
    structural_mass : float, [kg]
        Total structural mass (tower+monopile)
    tower_raw_cost : float, [USD]
        Tower cost only
    tower_mass : float, [kg]
        Tower mass only
    tower_center_of_mass : float, [m]
        z-position of center of mass of tower
    tower_section_center_of_mass : numpy array[nFull-1], [m]
        z position of center of mass of each can in the tower
    tower_I_base : numpy array[6], [kg*m**2]
        Mass moment of inertia of tower about base [xx yy zz xy xz yz]
    monopile_mass : float, [kg]
        Mass of monopile from bottom of suction pile through transition piece
    monopile_cost : float, [USD]
        Total monopile cost

    """

    def initialize(self):
        self.options.declare("n_height")

    def setup(self):
        n_height = self.options["n_height"]
        nFull = get_nfull(n_height)

        self.add_input("cylinder_mass", val=np.zeros(nFull - 1), units="kg")
        self.add_input("cylinder_cost", val=0.0, units="USD")
        self.add_input("cylinder_center_of_mass", val=0.0, units="m")
        self.add_input("cylinder_section_center_of_mass", val=np.zeros(nFull - 1), units="m")
        self.add_input("cylinder_I_base", np.zeros(6), units="kg*m**2")
        self.add_input("transition_piece_height", 0.0, units="m")
        self.add_input("transition_piece_mass", 0.0, units="kg")
        self.add_input("transition_piece_cost", 0.0, units="USD")
        self.add_input("gravity_foundation_mass", 0.0, units="kg")
        self.add_input("z_full", val=np.zeros(nFull), units="m")
        self.add_input("d_full", val=np.zeros(nFull), units="m")

        self.add_output("structural_cost", val=0.0, units="USD")
        self.add_output("structural_mass", val=0.0, units="kg")
        self.add_output("tower_cost", val=0.0, units="USD")
        self.add_output("tower_mass", val=0.0, units="kg")
        self.add_output("tower_center_of_mass", val=0.0, units="m")
        self.add_output("tower_section_center_of_mass", val=np.zeros(nFull - 1), units="m")
        self.add_output("tower_I_base", np.zeros(6), units="kg*m**2")
        self.add_output("monopile_mass", val=0.0, units="kg")
        self.add_output("monopile_cost", val=0.0, units="USD")
        self.add_output("transition_piece_I", np.zeros(6), units="kg*m**2")
        self.add_output("gravity_foundation_I", np.zeros(6), units="kg*m**2")

    def compute(self, inputs, outputs):
        # Unpack inputs
        z = inputs["z_full"]
        d = inputs["d_full"]
        z_trans = inputs["transition_piece_height"]
        m_trans = inputs["transition_piece_mass"]
        m_grav = inputs["gravity_foundation_mass"]
        m_cyl = inputs["cylinder_mass"]

        outputs["structural_cost"] = inputs["cylinder_cost"] + inputs["transition_piece_cost"]
        outputs["structural_mass"] = m_cyl.sum() + m_trans + m_grav
        outputs["tower_center_of_mass"] = (
            inputs["cylinder_center_of_mass"] * m_cyl.sum() + m_trans * z_trans + m_grav * z[0]
        ) / (m_cyl.sum() + m_trans + m_grav)
        outputs["tower_section_center_of_mass"] = inputs["cylinder_section_center_of_mass"]

        outputs["monopile_mass"], dydx, dydxp, dydyp = util.interp_with_deriv(z_trans, z, np.r_[0.0, np.cumsum(m_cyl)])
        outputs["monopile_cost"] = (
            inputs["cylinder_cost"] * outputs["monopile_mass"] / m_cyl.sum() + inputs["transition_piece_cost"]
        )
        outputs["monopile_mass"] += m_trans + m_grav

        outputs["tower_cost"] = outputs["structural_cost"] - outputs["monopile_cost"]
        outputs["tower_mass"] = outputs["structural_mass"] - outputs["monopile_mass"]
        outputs["tower_I_base"] = inputs["cylinder_I_base"]
        outputs["tower_I_base"][:2] += m_trans * (z_trans - z[0]) ** 2

        # Mass properties for transition piece and gravity foundation
        itrans = util.find_nearest(z, z_trans)
        r_trans = 0.5 * d[itrans]
        r_grav = 0.5 * d[0]
        I_trans = m_trans * r_trans ** 2.0 * np.r_[0.5, 0.5, 1.0, np.zeros(3)]  # shell
        I_grav = m_grav * r_grav ** 2.0 * np.r_[0.25, 0.25, 0.5, np.zeros(3)]  # disk
        outputs["transition_piece_I"] = I_trans
        outputs["gravity_foundation_I"] = I_grav


class TurbineMass(om.ExplicitComponent):
    """
    Compute the turbine mass, center of mass, and mass moment of inertia.

    Parameters
    ----------
    hub_height : float, [m]
        Hub-height
    rna_mass : float, [kg]
        Total tower mass
    rna_I : numpy array[6], [kg*m**2]
        Mass moment of inertia of RNA about tower top [xx yy zz xy xz yz]
    rna_cg : numpy array[3], [m]
        xyz-location of RNA cg relative to tower top
    tower_mass : float, [kg]
        Total tower mass (not including monopile)
    monopile_mass : float, [kg]
        Monopile mass
    tower_center_of_mass : float, [m]
        z-position of center of mass of tower
    tower_I_base : numpy array[6], [kg*m**2]
        Mass moment of inertia of tower about base [xx yy zz xy xz yz]

    Returns
    -------
    turbine_mass : float, [kg]
        Total mass of tower+rna
    turbine_center_of_mass : numpy array[3], [m]
        xyz-position of tower+rna center of mass
    turbine_I_base : numpy array[6], [kg*m**2]
        mass moment of inertia of tower about base [xx yy zz xy xz yz]

    """

    def setup(self):
        self.add_input("hub_height", val=0.0, units="m")
        self.add_input("rna_mass", val=0.0, units="kg")
        self.add_input("rna_I", np.zeros(6), units="kg*m**2")
        self.add_input("rna_cg", np.zeros(3), units="m")
        self.add_input("tower_mass", val=0.0, units="kg")
        self.add_input("monopile_mass", val=0.0, units="kg")
        self.add_input("tower_center_of_mass", val=0.0, units="m")
        self.add_input("tower_I_base", np.zeros(6), units="kg*m**2")

        self.add_output("turbine_mass", val=0.0, units="kg")
        self.add_output("turbine_center_of_mass", val=np.zeros(3), units="m")
        self.add_output("turbine_I_base", np.zeros(6), units="kg*m**2")

        self.declare_partials(
            "turbine_I_base", ["hub_height", "rna_I", "rna_cg", "rna_mass", "tower_I_base"], method="fd"
        )
        self.declare_partials(
            "turbine_center_of_mass",
            ["hub_height", "monopile_mass", "rna_cg", "rna_mass", "tower_center_of_mass", "tower_mass"],
            method="fd",
        )
        self.declare_partials("turbine_mass", ["monopile_mass", "rna_mass", "tower_mass"], val=1.0)

    def compute(self, inputs, outputs):
        outputs["turbine_mass"] = inputs["rna_mass"] + inputs["tower_mass"] + inputs["monopile_mass"]

        cg_rna = inputs["rna_cg"] + np.r_[0.0, 0.0, inputs["hub_height"]]
        cg_tower = np.r_[0.0, 0.0, inputs["tower_center_of_mass"]]
        outputs["turbine_center_of_mass"] = (inputs["rna_mass"] * cg_rna + inputs["tower_mass"] * cg_tower) / outputs[
            "turbine_mass"
        ]

        R = cg_rna
        I_tower = util.assembleI(inputs["tower_I_base"])
        I_rna = util.assembleI(inputs["rna_I"]) + inputs["rna_mass"] * (np.dot(R, R) * np.eye(3) - np.outer(R, R))
        outputs["turbine_I_base"] = util.unassembleI(I_tower + I_rna)


class TowerPreFrame(om.ExplicitComponent):
    """
    Compute some properties of the tower needed for FEM analysis.

    This component can be simplified by using src_indices for data-passing.
    At the very least, we can code the sparse derivatives as-is for
    input-output relationships.

    Parameters
    ----------
    z_full : numpy array[nFull], [m]
        location along tower. start at bottom and go to top
    mass : float, [kg]
        added mass
    mI : numpy array[6], [kg*m**2]
        mass moment of inertia about some point p [xx yy zz xy xz yz]
    mrho : numpy array[3], [m]
        xyz-location of p relative to node
    transition_piece_mass : float, [kg]
        point mass of transition piece
    gravity_foundation_mass : float, [kg]
        point mass of transition piece
    transition_piece_height : float, [m]
        height of transition piece above water line
    rna_F : numpy array[3], [N]
        rna force
    rna_M : numpy array[3], [N*m]
        rna moment
    k_monopile : numpy array[6], [N/m]
        Stiffness BCs for ocean soil. Only used if monoflag inputis True

    Returns
    -------
    kidx : numpy array[np.int_]
        indices of z where external stiffness reactions should be applied.
    kx : numpy array[nK], [N/m]
        spring stiffness in x-direction
    ky : numpy array[nK], [N/m]
        spring stiffness in y-direction
    kz : numpy array[nK], [N/m]
        spring stiffness in z-direction
    ktx : numpy array[nK], [N/m]
        spring stiffness in theta_x-rotation
    kty : numpy array[nK], [N/m]
        spring stiffness in theta_y-rotation
    ktz : numpy array[nK], [N/m]
        spring stiffness in theta_z-rotation
    midx : numpy array[np.int_]
        indices where added mass should be applied.
    m : numpy array[nMass], [kg]
        added mass
    mIxx : numpy array[nMass], [kg*m**2]
        x mass moment of inertia about some point p
    mIyy : numpy array[nMass], [kg*m**2]
        y mass moment of inertia about some point p
    mIzz : numpy array[nMass], [kg*m**2]
        z mass moment of inertia about some point p
    mIxy : numpy array[nMass], [kg*m**2]
        xy mass moment of inertia about some point p
    mIxz : numpy array[nMass], [kg*m**2]
        xz mass moment of inertia about some point p
    mIyz : numpy array[nMass], [kg*m**2]
        yz mass moment of inertia about some point p
    mrhox : numpy array[nMass], [m]
        x-location of p relative to node
    mrhoy : numpy array[nMass], [m]
        y-location of p relative to node
    mrhoz : numpy array[nMass], [m]
        z-location of p relative to node
    plidx : numpy array[np.int_]
        indices where point loads should be applied.
    Fx : numpy array[nPL], [N]
        point force in x-direction
    Fy : numpy array[nPL], [N]
        point force in y-direction
    Fz : numpy array[nPL], [N]
        point force in z-direction
    Mxx : numpy array[nPL], [N*m]
        point moment about x-axis
    Myy : numpy array[nPL], [N*m]
        point moment about y-axis
    Mzz : numpy array[nPL], [N*m]
        point moment about z-axis

    """

    def initialize(self):
        self.options.declare("n_height")
        self.options.declare("monopile", default=False)
        self.options.declare("soil_springs", default=False)
        self.options.declare("gravity_foundation", default=False)

    def setup(self):
        n_height = self.options["n_height"]
        nFull = get_nfull(n_height)

        self.add_input("z_full", np.zeros(nFull), units="m")

        # extra mass
        self.add_input("mass", 0.0, units="kg")
        self.add_input("mI", np.zeros(6), units="kg*m**2")
        self.add_input("mrho", np.zeros(3), units="m")
        self.add_input("transition_piece_mass", 0.0, units="kg")
        self.add_input("transition_piece_I", np.zeros(6), units="kg*m**2")
        self.add_input("gravity_foundation_I", np.zeros(6), units="kg*m**2")
        self.add_input("gravity_foundation_mass", 0.0, units="kg")
        self.add_input("transition_piece_height", 0.0, units="m")
        self.add_input("suctionpile_depth", 0.0, units="m")

        # point loads
        self.add_input("rna_F", np.zeros(3), units="N")
        self.add_input("rna_M", np.zeros(3), units="N*m")

        # Monopile handling
        self.add_input("z_soil", np.zeros(NPTS_SOIL), units="N/m")
        self.add_input("k_soil", np.zeros((NPTS_SOIL, 6)), units="N/m")

        # spring reaction data.
        nK = 4 if self.options["monopile"] and not self.options["gravity_foundation"] else 1
        self.add_output("kidx", np.zeros(nK, dtype=np.int_))
        self.add_output("kx", np.zeros(nK), units="N/m")
        self.add_output("ky", np.zeros(nK), units="N/m")
        self.add_output("kz", np.zeros(nK), units="N/m")
        self.add_output("ktx", np.zeros(nK), units="N/m")
        self.add_output("kty", np.zeros(nK), units="N/m")
        self.add_output("ktz", np.zeros(nK), units="N/m")

        # extra mass
        nMass = 3
        self.add_output("midx", np.zeros(nMass, dtype=np.int_))
        self.add_output("m", np.zeros(nMass), units="kg")
        self.add_output("mIxx", np.zeros(nMass), units="kg*m**2")
        self.add_output("mIyy", np.zeros(nMass), units="kg*m**2")
        self.add_output("mIzz", np.zeros(nMass), units="kg*m**2")
        self.add_output("mIxy", np.zeros(nMass), units="kg*m**2")
        self.add_output("mIxz", np.zeros(nMass), units="kg*m**2")
        self.add_output("mIyz", np.zeros(nMass), units="kg*m**2")
        self.add_output("mrhox", np.zeros(nMass), units="m")
        self.add_output("mrhoy", np.zeros(nMass), units="m")
        self.add_output("mrhoz", np.zeros(nMass), units="m")

        # point loads (if addGravityLoadForExtraMass=True be sure not to double count by adding those force here also)
        nPL = 1
        self.add_output("plidx", np.zeros(nPL, dtype=np.int_))
        self.add_output("Fx", np.zeros(nPL), units="N")
        self.add_output("Fy", np.zeros(nPL), units="N")
        self.add_output("Fz", np.zeros(nPL), units="N")
        self.add_output("Mxx", np.zeros(nPL), units="N*m")
        self.add_output("Myy", np.zeros(nPL), units="N*m")
        self.add_output("Mzz", np.zeros(nPL), units="N*m")

        self.declare_partials("Fx", ["rna_F"], method="fd")
        self.declare_partials("Fy", ["rna_F"], method="fd")
        self.declare_partials("Fz", ["rna_F"], method="fd")
        self.declare_partials("G_full", [], method="fd")
        self.declare_partials("Mxx", ["rna_M"], method="fd")
        self.declare_partials("Myy", ["rna_M"], method="fd")
        self.declare_partials("Mzz", ["rna_M"], method="fd")
        self.declare_partials("kidx", [], method="fd")
        self.declare_partials("ktx", [], method="fd")
        self.declare_partials("kty", [], method="fd")
        self.declare_partials("ktz", [], method="fd")
        self.declare_partials("kx", [], method="fd")
        self.declare_partials("ky", [], method="fd")
        self.declare_partials("kz", [], method="fd")
        self.declare_partials("m", ["gravity_foundation_mass", "mass", "transition_piece_mass"], method="fd")
        self.declare_partials("mIxx", ["gravity_foundation_mass", "mI", "transition_piece_mass"], method="fd")
        self.declare_partials("mIxy", ["mI"], method="fd")
        self.declare_partials("mIxz", ["mI"], method="fd")
        self.declare_partials("mIyy", ["gravity_foundation_mass", "mI", "transition_piece_mass"], method="fd")
        self.declare_partials("mIyz", ["mI"], method="fd")
        self.declare_partials("mIzz", ["gravity_foundation_mass", "mI", "transition_piece_mass"], method="fd")
        self.declare_partials("midx", [], method="fd")
        self.declare_partials("mrhox", ["mrho"], method="fd")
        self.declare_partials("mrhoy", ["mrho"], method="fd")
        self.declare_partials("mrhoz", ["mrho"], method="fd")
        self.declare_partials("plidx", [], method="fd")

    def compute(self, inputs, outputs):
        n_height = self.options["n_height"]
        nFull = get_nfull(n_height)
        z = inputs["z_full"]

        # Prepare RNA, transition piece, and gravity foundation (if any applicable) for "extra node mass"
        itrans = util.find_nearest(z, inputs["transition_piece_height"])
        mtrans = inputs["transition_piece_mass"]
        Itrans = inputs["transition_piece_I"]
        mgrav = inputs["gravity_foundation_mass"]
        Igrav = inputs["gravity_foundation_I"]
        # Note, need len()-1 because Frame3DD crashes if mass add at end
        outputs["midx"] = np.array([nFull - 1, itrans, 0], dtype=np.int_)
        outputs["m"] = np.array([inputs["mass"], mtrans, mgrav]).flatten()
        outputs["mIxx"] = np.array([inputs["mI"][0], Itrans[0], Igrav[0]]).flatten()
        outputs["mIyy"] = np.array([inputs["mI"][1], Itrans[1], Igrav[1]]).flatten()
        outputs["mIzz"] = np.array([inputs["mI"][2], Itrans[2], Igrav[2]]).flatten()
        outputs["mIxy"] = np.array([inputs["mI"][3], Itrans[3], Igrav[3]]).flatten()
        outputs["mIxz"] = np.array([inputs["mI"][4], Itrans[4], Igrav[4]]).flatten()
        outputs["mIyz"] = np.array([inputs["mI"][5], Itrans[5], Igrav[5]]).flatten()
        outputs["mrhox"] = np.array([inputs["mrho"][0], 0.0, 0.0]).flatten()
        outputs["mrhoy"] = np.array([inputs["mrho"][1], 0.0, 0.0]).flatten()
        outputs["mrhoz"] = np.array([inputs["mrho"][2], 0.0, 0.0]).flatten()

        # Prepare point forces at RNA node
        outputs["plidx"] = np.array([nFull - 1], dtype=np.int_)  # -1 b/c same reason as above
        outputs["Fx"] = np.array([inputs["rna_F"][0]]).flatten()
        outputs["Fy"] = np.array([inputs["rna_F"][1]]).flatten()
        outputs["Fz"] = np.array([inputs["rna_F"][2]]).flatten()
        outputs["Mxx"] = np.array([inputs["rna_M"][0]]).flatten()
        outputs["Myy"] = np.array([inputs["rna_M"][1]]).flatten()
        outputs["Mzz"] = np.array([inputs["rna_M"][2]]).flatten()

        # Prepare for reactions: rigid at tower base
        if self.options["monopile"] and not self.options["gravity_foundation"]:
            if self.options["soil_springs"]:
                z_soil = inputs["z_soil"]
                k_soil = inputs["k_soil"]
                z_pile = z[z <= (z[0] + 1e-1 + np.abs(z_soil[0]))]
                if z_pile.size != 4:
                    print(z)
                    print(z_soil)
                    print(z_pile)
                    raise ValueError("Please use only one section for submerged pile for now")
                k_mono = np.zeros((z_pile.size, 6))
                for k in range(6):
                    k_mono[:, k] = np.interp(z_pile + np.abs(z_soil[0]), z_soil, k_soil[:, k])
                outputs["kidx"] = np.arange(len(z_pile), dtype=np.int_)
                outputs["kx"] = np.array([k_mono[:, 0]])
                outputs["ky"] = np.array([k_mono[:, 2]])
                outputs["kz"] = np.zeros(k_mono.shape[0])
                outputs["kz"][0] = np.array([k_mono[0, 4]])
                outputs["ktx"] = np.array([k_mono[:, 1]])
                outputs["kty"] = np.array([k_mono[:, 3]])
                outputs["ktz"] = np.array([k_mono[:, 5]])

            else:
                z_pile = z[z <= (z[0] + 1e-1 + inputs["suctionpile_depth"])]
                npile = z_pile.size
                if npile != 4:
                    print(z)
                    print(z_pile)
                    print(inputs["suctionpile_depth"])
                    raise ValueError("Please use only one section for submerged pile for now")
                outputs["kidx"] = np.arange(npile, dtype=np.int_)
                outputs["kx"] = outputs["ky"] = outputs["kz"] = RIGID * np.ones(npile)
                outputs["ktx"] = outputs["kty"] = outputs["ktz"] = RIGID * np.ones(npile)

        else:
            outputs["kidx"] = np.array([0], dtype=np.int_)
            outputs["kx"] = outputs["ky"] = outputs["kz"] = np.array([RIGID])
            outputs["ktx"] = outputs["kty"] = outputs["ktz"] = np.array([RIGID])


class TowerPostFrame(om.ExplicitComponent):
    """
    Postprocess results from Frame3DD.

    Parameters
    ----------
    z_full : numpy array[nFull], [m]
        location along tower. start at bottom and go to top
    d_full : numpy array[nFull], [m]
        effective tower diameter for section
    t_full : numpy array[nFull-1], [m]
        effective shell thickness for section
    E_full : numpy array[nFull-1], [N/m**2]
        modulus of elasticity
    sigma_y_full : numpy array[nFull-1], [N/m**2]
        yield stress
    Fz : numpy array[nFull-1], [N]
        Axial foce in vertical z-direction in cylinder structure.
    Mxx : numpy array[nFull-1], [N*m]
        Moment about x-axis in cylinder structure.
    Myy : numpy array[nFull-1], [N*m]
        Moment about y-axis in cylinder structure.
    axial_stress : numpy array[nFull-1], [N/m**2]
        axial stress in tower elements
    shear_stress : numpy array[nFull-1], [N/m**2]
        shear stress in tower elements
    hoop_stress : numpy array[nFull-1], [N/m**2]
        hoop stress in tower elements
    tower_deflection_in : numpy array[nFull], [m]
        Deflection of tower nodes in yaw-aligned +x direction
    life : float
        fatigue life of tower
    freqs : numpy array[NFREQ], [Hz]
        Natural frequencies of the structure
    x_mode_freqs : numpy array[NFREQ2]
        Frequencies associated with mode shapes in the x-direction
    y_mode_freqs : numpy array[NFREQ2]
        Frequencies associated with mode shapes in the y-direction
    x_mode_shapes : numpy array[NFREQ2, 5]
        6-degree polynomial coefficients of mode shapes in the x-direction
    y_mode_shapes : numpy array[NFREQ2, 5]
        6-degree polynomial coefficients of mode shapes in the x-direction

    Returns
    -------
    structural_frequencies : numpy array[NFREQ], [Hz]
        First and second natural frequency
    fore_aft_freqs : numpy array[NFREQ2]
        Frequencies associated with mode shapes in the tower fore-aft direction
    side_side_freqs : numpy array[NFREQ2]
        Frequencies associated with mode shapes in the tower side-side direction
    fore_aft_modes : numpy array[NFREQ2, 5]
        6-degree polynomial coefficients of mode shapes in the tower fore-aft direction
        (without constant term)
    side_side_modes : numpy array[NFREQ2, 5]
        6-degree polynomial coefficients of mode shapes in the tower side-side direction
        (without constant term)
    tower_deflection : numpy array[nFull], [m]
        Deflection of tower nodes in yaw-aligned +x direction
    top_deflection : float, [m]
        Deflection of tower top in yaw-aligned +x direction
    stress : numpy array[nFull-1]
        Von Mises stress utilization along tower at specified locations. Includes safety
        factor.
    shell_buckling : numpy array[nFull-1]
        Shell buckling constraint. Should be < 1 for feasibility. Includes safety
        factors
    global_buckling : numpy array[nFull-1]
        Global buckling constraint. Should be < 1 for feasibility. Includes safety
        factors
    turbine_F : numpy array[3], [N]
        Total force on tower+rna
    turbine_M : numpy array[3], [N*m]
        Total x-moment on tower+rna measured at base

    """

    def initialize(self):
        self.options.declare("n_height")
        self.options.declare("modeling_options")
        # self.options.declare('nDEL')

    def setup(self):
        n_height = self.options["n_height"]
        nFull = get_nfull(n_height)

        # effective geometry -- used for handbook methods to estimate hoop stress, buckling, fatigue
        self.add_input("z_full", np.zeros(nFull), units="m")
        self.add_input("d_full", np.zeros(nFull), units="m")
        self.add_input("t_full", np.zeros(nFull - 1), units="m")

        # Material properties
        self.add_input("E_full", np.zeros(nFull - 1), units="N/m**2", desc="modulus of elasticity")
        self.add_input(
            "G_full",
            np.zeros(nFull - 1),
            units="Pa",
            desc="Isotropic shear modulus of the materials along the tower sections.",
        )
        self.add_input(
            "rho_full", np.zeros(nFull - 1), units="kg/m**3", desc="Density of the materials along the tower sections."
        )
        self.add_input("sigma_y_full", np.zeros(nFull - 1), units="N/m**2", desc="yield stress")

        # Processed Frame3DD outputs
        self.add_input("Fz", np.zeros(nFull - 1), units="N")
        self.add_input("Mxx", np.zeros(nFull - 1), units="N*m")
        self.add_input("Myy", np.zeros(nFull - 1), units="N*m")
        self.add_input("axial_stress", val=np.zeros(nFull - 1), units="N/m**2")
        self.add_input("shear_stress", val=np.zeros(nFull - 1), units="N/m**2")
        self.add_input("hoop_stress", val=np.zeros(nFull - 1), units="N/m**2")
        self.add_input("tower_deflection_in", val=np.zeros(nFull), units="m")

        # safety factors
        # self.add_input('gamma_f', 1.35, desc='safety factor on loads')
        # self.add_input('gamma_m', 1.1, desc='safety factor on materials')
        # self.add_input('gamma_n', 1.0, desc='safety factor on consequence of failure')
        # self.add_input('gamma_b', 1.1, desc='buckling safety factor')
        # self.add_input('gamma_fatigue', 1.755, desc='total safety factor for fatigue')

        # fatigue parameters
        self.add_input("life", 20.0)
        # self.add_input('m_SN', 4, desc='slope of S/N curve')
        # self.add_input('DC', 80.0, desc='standard value of stress')
        # self.add_input('z_DEL', np.zeros(nDEL), units='m', desc='absolute z coordinates of corresponding fatigue parameters')
        # self.add_input('M_DEL', np.zeros(nDEL), desc='fatigue parameters at corresponding z coordinates')

        # Frequencies
        NFREQ2 = int(NFREQ / 2)

        self.add_input("freqs", val=np.zeros(NFREQ), units="Hz", desc="Natural frequencies of the structure")
        self.add_input(
            "x_mode_shapes",
            val=np.zeros((NFREQ2, 5)),
            desc="6-degree polynomial coefficients of mode shapes in the x-direction (x^2..x^6, no linear or constant term)",
        )
        self.add_input(
            "y_mode_shapes",
            val=np.zeros((NFREQ2, 5)),
            desc="6-degree polynomial coefficients of mode shapes in the x-direction (x^2..x^6, no linear or constant term)",
        )
        self.add_input(
            "x_mode_freqs", val=np.zeros(NFREQ2), desc="Frequencies associated with mode shapes in the x-direction"
        )
        self.add_input(
            "y_mode_freqs", val=np.zeros(NFREQ2), desc="Frequencies associated with mode shapes in the y-direction"
        )

        # outputs
        self.add_output(
            "structural_frequencies", np.zeros(NFREQ), units="Hz", desc="First and second natural frequency"
        )
        self.add_output(
            "fore_aft_modes",
            np.zeros((NFREQ2, 5)),
            desc="6-degree polynomial coefficients of mode shapes in the tower fore-aft direction (x^2..x^6, no linear or constant term)",
        )
        self.add_output(
            "side_side_modes",
            np.zeros((NFREQ2, 5)),
            desc="6-degree polynomial coefficients of mode shapes in the tower side-side direction (x^2..x^6, no linear or constant term)",
        )
        self.add_output(
            "fore_aft_freqs",
            np.zeros(NFREQ2),
            desc="Frequencies associated with mode shapes in the tower fore-aft direction",
        )
        self.add_output(
            "side_side_freqs",
            np.zeros(NFREQ2),
            desc="Frequencies associated with mode shapes in the tower side-side direction",
        )
        self.add_output(
            "tower_deflection", np.zeros(nFull), units="m", desc="Deflection of tower top in yaw-aligned +x direction"
        )
        self.add_output("top_deflection", 0.0, units="m", desc="Deflection of tower top in yaw-aligned +x direction")
        self.add_output(
            "stress",
            np.zeros(nFull - 1),
            desc="Von Mises stress utilization along tower at specified locations.  incudes safety factor.",
        )
        self.add_output(
            "shell_buckling",
            np.zeros(nFull - 1),
            desc="Shell buckling constraint.  Should be < 1 for feasibility.  Includes safety factors",
        )
        self.add_output(
            "global_buckling",
            np.zeros(nFull - 1),
            desc="Global buckling constraint.  Should be < 1 for feasibility.  Includes safety factors",
        )
        # self.add_output('damage', np.zeros(nFull-1), desc='Fatigue damage at each tower section')
        self.add_output("turbine_F", val=np.zeros(3), units="N", desc="Total force on tower+rna")
        self.add_output("turbine_M", val=np.zeros(3), units="N*m", desc="Total x-moment on tower+rna measured at base")

        self.declare_partials(
            "global_buckling", ["Fz", "Mxx", "Myy", "d_full", "sigma_y_full", "t_full", "z_full"], method="fd"
        )
        self.declare_partials(
            "shell_buckling",
            ["axial_stress", "d_full", "hoop_stress", "shear_stress", "sigma_y_full", "t_full"],
            method="fd",
        )
        # self.declare_partials("stress", ["axial_stress", "hoop_stress", "shear_stress", "sigma_y_full"], method="fd")
        # self.declare_partials("structural_frequencies", ["freqs"], method="fd")
        # self.declare_partials("fore_aft_freqs", ["x_mode_freqs"], method="fd")
        # self.declare_partials("side_side_freqs", ["y_mode_freqs"], method="fd")
        # self.declare_partials("fore_aft_modes", ["x_mode_shapes"], method="fd")
        # self.declare_partials("side_side_modes", ["y_mode_shapes"], method="fd")
        # self.declare_partials("turbine_F", [], method="fd")
        # self.declare_partials("turbine_M", [], method="fd")

    def compute(self, inputs, outputs):
        # Unpack some variables
        axial_stress = inputs["axial_stress"]
        shear_stress = inputs["shear_stress"]
        hoop_stress = inputs["hoop_stress"]
        sigma_y = inputs["sigma_y_full"]
        E = inputs["E_full"]
        t = inputs["t_full"]
        d, _ = util.nodal2sectional(inputs["d_full"])
        z_section, _ = util.nodal2sectional(inputs["z_full"])
        L_reinforced = self.options["modeling_options"]["buckling_length"] * np.ones(axial_stress.shape)
        gamma_f = self.options["modeling_options"]["gamma_f"]
        gamma_m = self.options["modeling_options"]["gamma_m"]
        gamma_n = self.options["modeling_options"]["gamma_n"]
        gamma_b = self.options["modeling_options"]["gamma_b"]

        # Frequencies and mode shapes (with x^2 term first)
        outputs["structural_frequencies"] = inputs["freqs"]
        outputs["fore_aft_freqs"] = inputs["x_mode_freqs"]
        outputs["side_side_freqs"] = inputs["y_mode_freqs"]
        outputs["fore_aft_modes"] = inputs["x_mode_shapes"]
        outputs["side_side_modes"] = inputs["y_mode_shapes"]

        # Tower top deflection
        outputs["tower_deflection"] = inputs["tower_deflection_in"]
        outputs["top_deflection"] = inputs["tower_deflection_in"][-1]

        # von mises stress
        outputs["stress"] = util_con.vonMisesStressUtilization(
            axial_stress, hoop_stress, shear_stress, gamma_f * gamma_m * gamma_n, sigma_y
        )

        # shell buckling
        outputs["shell_buckling"] = util_con.shellBucklingEurocode(
            d, t, axial_stress, hoop_stress, shear_stress, L_reinforced, E, sigma_y, gamma_f, gamma_b
        )

        # global buckling
        tower_height = inputs["z_full"][-1] - inputs["z_full"][0]
        M = np.sqrt(inputs["Mxx"] ** 2 + inputs["Myy"] ** 2)
        outputs["global_buckling"] = util_con.bucklingGL(
            d, t, inputs["Fz"], M, tower_height, E, sigma_y, gamma_f, gamma_b
        )

        # fatigue
        N_DEL = 365.0 * 24.0 * 3600.0 * inputs["life"] * np.ones(len(t))
        # outputs['damage'] = np.zeros(N_DEL.shape)

        # if any(inputs['M_DEL']):
        #    M_DEL = np.interp(z_section, inputs['z_DEL'], inputs['M_DEL'])

        #    outputs['damage'] = util_con.fatigue(M_DEL, N_DEL, d, inputs['t'], inputs['m_SN'],
        #                                      inputs['DC'], gamma_fatigue, stress_factor=1.0, weld_factor=True)


# -----------------
#  Groups
# -----------------


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
        monopile = self.options["modeling_options"]["flags"]["monopile"]

        n_height_tow = mod_opt["n_height_tower"]
        n_layers_tow = mod_opt["n_layers_tower"]
        n_height_mon = 0 if not monopile else mod_opt["n_height_monopile"]
        n_layers_mon = 0 if not monopile else mod_opt["n_layers_monopile"]
        n_height = (
            n_height_tow if n_height_mon == 0 else n_height_tow + n_height_mon - 1
        )  # Should have one overlapping point
        nFull = get_nfull(n_height)

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
            DiscretizationYAML(
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
            CylinderDiscretization(nPoints=n_height),
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
        self.add_subsystem("tgeometry", TowerDiscretization(n_height=n_height), promotes=["*"])

        self.add_subsystem(
            "cm",
            CylinderMass(nPoints=nFull),
            promotes=["z_full", "d_full", "t_full", "labor_cost_rate", "painting_cost_rate"],
        )
        self.add_subsystem(
            "tm",
            TowerMass(n_height=n_height),
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
            TurbineMass(),
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
        n_height_tow = mod_opt["n_height_tower"]
        n_height_mon = 0 if not monopile else mod_opt["n_height_monopile"]
        n_height = (
            n_height_tow if n_height_mon == 0 else n_height_tow + n_height_mon - 1
        )  # Should have one overlapping point
        nFull = get_nfull(n_height)
        self.set_input_defaults("E", np.zeros(n_height - 1), units="N/m**2")
        self.set_input_defaults("G", np.zeros(n_height - 1), units="N/m**2")
        if monopile and mod_opt["soil_springs"]:
            self.set_input_defaults("G_soil", 0.0, units="N/m**2")
            self.set_input_defaults("nu_soil", 0.0)
        self.set_input_defaults("sigma_y", np.zeros(n_height - 1), units="N/m**2")
        self.set_input_defaults("rna_mass", 0.0, units="kg")
        self.set_input_defaults("rna_cg", np.zeros(3), units="m")
        self.set_input_defaults("rna_I", np.zeros(6), units="kg*m**2")
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
                TowerPreFrame(
                    n_height=n_height,
                    monopile=monopile,
                    soil_springs=mod_opt["soil_springs"],
                    gravity_foundation=mod_opt["gravity_foundation"],
                ),
                promotes=[
                    "transition_piece_mass",
                    "transition_piece_height",
                    "transition_piece_I",
                    "gravity_foundation_mass",
                    "gravity_foundation_I",
                    "z_full",
                    "suctionpile_depth",
                    ("mass", "rna_mass"),
                    ("mrho", "rna_cg"),
                    ("mI", "rna_I"),
                ],
            )
            self.add_subsystem(
                "tower" + lc,
                CylinderFrame3DD(
                    npts=nFull,
                    nK=4 if monopile and not mod_opt["gravity_foundation"] else 1,
                    nMass=3,
                    nPL=1,
                    frame3dd_opt=frame3dd_opt,
                    buckling_length=mod_opt["buckling_length"],
                ),
                promotes=["Az", "Asx", "Asy", "Ixx", "Iyy", "Jz"],
            )
            self.add_subsystem(
                "post" + lc,
                TowerPostFrame(n_height=n_height, modeling_options=mod_opt),
                promotes=["life", "z_full", "d_full", "t_full", "rho_full", "E_full", "G_full", "sigma_y_full"],
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

            self.connect("tower" + lc + ".freqs", "post" + lc + ".freqs")
            self.connect("tower" + lc + ".x_mode_freqs", "post" + lc + ".x_mode_freqs")
            self.connect("tower" + lc + ".y_mode_freqs", "post" + lc + ".y_mode_freqs")
            self.connect("tower" + lc + ".x_mode_shapes", "post" + lc + ".x_mode_shapes")
            self.connect("tower" + lc + ".y_mode_shapes", "post" + lc + ".y_mode_shapes")
            self.connect("tower" + lc + ".Fz_out", "post" + lc + ".Fz")
            self.connect("tower" + lc + ".Mxx_out", "post" + lc + ".Mxx")
            self.connect("tower" + lc + ".Myy_out", "post" + lc + ".Myy")
            self.connect("tower" + lc + ".axial_stress", "post" + lc + ".axial_stress")
            self.connect("tower" + lc + ".shear_stress", "post" + lc + ".shear_stress")
            self.connect("tower" + lc + ".hoop_stress_euro", "post" + lc + ".hoop_stress")
            self.connect("tower" + lc + ".cylinder_deflection", "post" + lc + ".tower_deflection_in")

            self.connect("wind" + lc + ".Px", "tower" + lc + ".Px")
            self.connect("wind" + lc + ".Py", "tower" + lc + ".Py")
            self.connect("wind" + lc + ".Pz", "tower" + lc + ".Pz")
            self.connect("wind" + lc + ".qdyn", "tower" + lc + ".qdyn")
