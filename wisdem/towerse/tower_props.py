import copy

import numpy as np
import openmdao.api as om
import wisdem.commonse.frustum as frustum
import wisdem.commonse.utilities as util
import wisdem.commonse.manufacturing as manufacture
from wisdem.towerse import NREFINE, eps, get_nfull


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
    suctionpile_depth : float, [m]
        Depth of monopile below sea floor
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
        self.add_input("sigma_ult_mat", val=np.zeros([n_mat, 3]), units="Pa")
        self.add_input("wohler_exp_mat", val=np.zeros(n_mat))
        self.add_input("wohler_A_mat", val=np.zeros(n_mat))
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
        self.add_output("sigma_ult", val=np.zeros(n_height - 1), units="Pa")
        self.add_output("wohler_exp", val=np.zeros(n_height - 1))
        self.add_output("wohler_A", val=np.zeros(n_height - 1))
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
        sigu = inputs["sigma_ult_mat"].min(axis=1)
        m = inputs["wohler_exp_mat"]
        A = inputs["wohler_A_mat"]
        rho = inputs["rho_mat"]
        cost = inputs["unit_cost_mat"]
        mat_names = discrete_inputs["material_names"]

        # Initialize sectional data
        E_param = np.zeros(twall.shape)
        G_param = np.zeros(twall.shape)
        sigy_param = np.zeros(twall.shape)
        sigu_param = np.zeros(twall.shape)
        m_param = np.zeros(twall.shape)
        A_param = np.zeros(twall.shape)
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
            sigu_param[k, :] = sigu[imat]
            m_param[k, :] = m[imat]
            A_param[k, :] = A[imat]

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
        sigu_param = 0.5 * np.sum(vol_frac * sigu_param, axis=0) + 0.5 / np.sum(vol_frac / sigu_param, axis=0)
        m_param = 0.5 * np.sum(vol_frac * m_param, axis=0) + 0.5 / np.sum(vol_frac / m_param, axis=0)
        A_param = 0.5 * np.sum(vol_frac * A_param, axis=0) + 0.5 / np.sum(vol_frac / A_param, axis=0)

        # Store values
        outputs["E"] = E_param
        outputs["G"] = G_param
        outputs["rho"] = rho_param
        outputs["sigma_y"] = sigy_param
        outputs["sigma_ult"] = sigu_param
        outputs["wohler_exp"] = m_param
        outputs["wohler_A"] = A_param
        outputs["unit_cost"] = cost_param


class CylinderDiscretization(om.ExplicitComponent):
    """
    Discretize geometry into finite element nodes

    Parameters
    ----------
    foundation_height : float, [m]
        starting height of tower
    section_height : numpy array[nPoints-1], [m]
        parameterized section heights along cylinder
    diameter : numpy array[nPoints], [m]
        cylinder diameter at corresponding locations
    wall_thickness : numpy array[nPoints-1], [m]
        shell thickness at corresponding locations

    Returns
    -------
    z_param : numpy array[nPoints], [m]
        parameterized locations along cylinder, linear lofting between
    z_full : numpy array[nFull], [m]
        locations along cylinder
    d_full : numpy array[nFull], [m]
        cylinder diameter at corresponding locations
    t_full : numpy array[nFull-1], [m]
        shell thickness at corresponding locations

    """

    """discretize geometry into finite element nodes"""

    def initialize(self):
        self.options.declare("nPoints")
        self.options.declare("nRefine", default=NREFINE)
        self.options.declare("nPin", default=0)

    def setup(self):
        nPoints = self.options["nPoints"]
        nRefine = np.round(self.options["nRefine"])
        nFull = int(nRefine * (nPoints - 1) + 1)

        self.add_input("foundation_height", val=0.0, units="m")
        self.add_input("section_height", np.zeros(nPoints - 1), units="m")
        self.add_input("diameter", np.zeros(nPoints), units="m")
        self.add_input("wall_thickness", np.zeros(nPoints - 1), units="m")

        self.add_output("z_param", np.zeros(nPoints), units="m")
        self.add_output("z_full", np.zeros(nFull), units="m")
        self.add_output("d_full", np.zeros(nFull), units="m")
        self.add_output("t_full", np.zeros(nFull - 1), units="m")

        # self.declare_partials('*', '*', method='fd', form='central', step=1e-6)

    def compute(self, inputs, outputs):
        # Check to make sure we have good values
        if np.any(inputs["section_height"] <= 0.0):
            raise ValueError("Section height values must be greater than zero, " + str(inputs["section_height"]))
        if np.any(inputs["wall_thickness"] <= 0.0):
            raise ValueError("Wall thickness values must be greater than zero, " + str(inputs["wall_thickness"]))
        if np.any(inputs["diameter"] <= 0.0):
            raise ValueError("Diameter values must be greater than zero, " + str(inputs["diameter"]))

        nRefine = int(np.round(self.options["nRefine"]))
        z_param = float(inputs["foundation_height"]) + np.r_[0.0, np.cumsum(inputs["section_height"].flatten())]

        # Have to regine each element one at a time so that we preserve input nodes
        z_full = np.array([])
        for k in range(z_param.size - 1):
            zref = np.linspace(z_param[k], z_param[k + 1], nRefine + 1)
            z_full = np.append(z_full, zref)
        z_full = np.unique(z_full)

        outputs["z_full"] = z_full
        outputs["d_full"] = np.interp(z_full, z_param, inputs["diameter"])
        z_section = 0.5 * (z_full[:-1] + z_full[1:])
        outputs["t_full"] = util.sectionalInterp(z_section, z_param, inputs["wall_thickness"])
        outputs["z_param"] = z_param


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
    tower_outer_diameter : numpy array[n_height], [m]
        cylinder diameter at corresponding locations
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
    axial_load2stress : numpy array[nFull-1,6], [m**2]
        Linear conversion factors between loads [Fx-z; Mx-z] and axial stress
    shear_load2stress : numpy array[nFull-1,6], [m**2]
        Linear conversion factors between loads [Fx-z; Mx-z] and shear stress

    """

    def initialize(self):
        self.options.declare("n_height")
        self.options.declare("n_refine")
        self.options.declare("modeling_options")

    def setup(self):
        n_height = self.options["n_height"]
        n_refine = self.options["n_refine"]
        nFull = get_nfull(n_height, nref=n_refine)

        self.add_input("hub_height", val=0.0, units="m")
        self.add_input("z_param", np.zeros(n_height), units="m")
        self.add_input("z_full", val=np.zeros(nFull), units="m")
        self.add_input("tower_outer_diameter", val=np.zeros(n_height), units="m")
        self.add_input("rho", val=np.zeros(n_height - 1), units="kg/m**3")
        self.add_input("unit_cost", val=np.zeros(n_height - 1), units="USD/kg")
        self.add_input("outfitting_factor", val=np.zeros(n_height - 1))
        self.add_input("E", val=np.zeros(n_height - 1), units="Pa")
        self.add_input("G", val=np.zeros(n_height - 1), units="Pa")
        self.add_input("sigma_y", val=np.zeros(n_height - 1), units="Pa")

        self.add_input("Az", np.zeros(nFull - 1), units="m**2")
        self.add_input("Asx", np.zeros(nFull - 1), units="m**2")
        self.add_input("Asy", np.zeros(nFull - 1), units="m**2")
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
        self.add_output("axial_load2stress", val=np.zeros((n_height - 1, 6)), units="m**2")
        self.add_output("shear_load2stress", val=np.zeros((n_height - 1, 6)), units="m**2")

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
        of = inputs["outfitting_factor"]
        Az = util.sectionalInterp(z, z_full, inputs["Az"])
        Asx = util.sectionalInterp(z, z_full, inputs["Asx"])
        Asy = util.sectionalInterp(z, z_full, inputs["Asy"])
        Ixx = util.sectionalInterp(z, z_full, inputs["Ixx"])
        Iyy = util.sectionalInterp(z, z_full, inputs["Iyy"])
        Jz = util.sectionalInterp(z, z_full, inputs["Jz"])
        outputs["sec_loc"] = (z - z[0]) / (z[-1] - z[0])
        outputs["mass_den"] = rho * Az * of
        outputs["foreaft_iner"] = rho * Ixx
        outputs["sideside_iner"] = rho * Iyy
        outputs["foreaft_stff"] = E * Ixx
        outputs["sideside_stff"] = E * Iyy
        outputs["tor_stff"] = G * Jz
        outputs["axial_stff"] = E * Az

        d_sec, _ = util.nodal2sectional(inputs["tower_outer_diameter"])
        r_sec = 0.5 * d_sec
        ax_load2stress = np.zeros((d_sec.size, 6))
        ax_load2stress[:, 2] = 1.0 / Az
        ax_load2stress[:, 3] = r_sec / Ixx
        ax_load2stress[:, 4] = r_sec / Iyy
        sh_load2stress = np.zeros((d_sec.size, 6))
        sh_load2stress[:, 0] = 1.0 / Asx
        sh_load2stress[:, 1] = 1.0 / Asy
        sh_load2stress[:, 5] = r_sec / Jz
        outputs["axial_load2stress"] = ax_load2stress
        outputs["shear_load2stress"] = sh_load2stress


class CylinderMass(om.ExplicitComponent):
    """
    Compute cylinder cost and mass properties

    Parameters
    ----------
    d_full : numpy array[nPoints], [m]
        cylinder diameter at corresponding locations
    t_full : numpy array[nPoints-1], [m]
        shell thickness at corresponding locations
    z_full : numpy array[nPoints], [m]
        parameterized locations along cylinder, linear lofting between
    rho : numpy array[nPoints-1], [kg/m**3]
        material density
    outfitting_factor : numpy array[nPoints-1]
        Multiplier that accounts for secondary structure mass inside of cylinder
    material_cost_rate : numpy array[nPoints-1], [USD/kg]
        Raw material cost rate: steel $1.1/kg, aluminum $3.5/kg
    labor_cost_rate : float, [USD/min]
        Labor cost rate
    painting_cost_rate : float, [USD/m/m]
        Painting / surface finishing cost rate

    Returns
    -------
    cost : float, [USD]
        Total cylinder cost
    mass : numpy array[nPoints-1], [kg]
        Total cylinder mass
    center_of_mass : float, [m]
        z-position of center of mass of cylinder
    section_center_of_mass : numpy array[nPoints-1], [m]
        z position of center of mass of each can in the cylinder
    I_base : numpy array[6], [kg*m**2]
        mass moment of inertia of cylinder about base [xx yy zz xy xz yz]

    """

    def initialize(self):
        self.options.declare("nPoints")

    def setup(self):
        nPoints = self.options["nPoints"]

        self.add_input("d_full", val=np.zeros(nPoints), units="m")
        self.add_input("t_full", val=np.zeros(nPoints - 1), units="m")
        self.add_input("z_full", val=np.zeros(nPoints), units="m")
        self.add_input("rho", val=np.zeros(nPoints - 1), units="kg/m**3")
        self.add_input("outfitting_factor", val=np.zeros(nPoints - 1))
        self.add_input("material_cost_rate", val=np.zeros(nPoints - 1), units="USD/kg")
        self.add_input("labor_cost_rate", 0.0, units="USD/min")
        self.add_input("painting_cost_rate", 0.0, units="USD/m/m")

        self.add_output("cost", val=0.0, units="USD")
        self.add_output("mass", val=np.zeros(nPoints - 1), units="kg")
        self.add_output("center_of_mass", val=0.0, units="m")
        self.add_output("section_center_of_mass", val=np.zeros(nPoints - 1), units="m")
        self.add_output("I_base", np.zeros(6), units="kg*m**2")

    def compute(self, inputs, outputs):
        # Unpack variables for thickness and average radius at each can interface
        twall = inputs["t_full"]
        Rb = 0.5 * inputs["d_full"][:-1]
        Rt = 0.5 * inputs["d_full"][1:]
        zz = inputs["z_full"]
        H = np.diff(zz)
        rho = inputs["rho"]
        coeff = inputs["outfitting_factor"]
        coeff = coeff + np.where(coeff < 1.0, 1.0, 0.0)

        # Total mass of cylinder
        V_shell = frustum.frustumShellVol(Rb, Rt, twall, H)
        mass = outputs["mass"] = coeff * rho * V_shell

        # Center of mass of each can/section
        cm_section = zz[:-1] + frustum.frustumShellCG(Rb, Rt, twall, H)
        outputs["section_center_of_mass"] = cm_section

        # Center of mass of cylinder
        V_shell += eps
        outputs["center_of_mass"] = np.dot(V_shell, cm_section) / V_shell.sum()

        # Moments of inertia
        Izz_section = coeff * rho * frustum.frustumShellIzz(Rb, Rt, twall, H)
        Ixx_section = Iyy_section = coeff * rho * frustum.frustumShellIxx(Rb, Rt, twall, H)

        # Sum up each cylinder section using parallel axis theorem
        I_base = np.zeros((3, 3))
        for k in range(Izz_section.size):
            R = np.array([0.0, 0.0, cm_section[k] - zz[0]])
            Icg = util.assembleI([Ixx_section[k], Iyy_section[k], Izz_section[k], 0.0, 0.0, 0.0])

            I_base += Icg + mass[k] * (np.dot(R, R) * np.eye(3) - np.outer(R, R))

        outputs["I_base"] = util.unassembleI(I_base)

        # Compute costs based on "Optimum Design of Steel Structures" by Farkas and Jarmai
        # All dimensions for correlations based on mm, not meters.
        R_ave = 0.5 * (Rb + Rt)
        taper = np.minimum(Rb / Rt, Rt / Rb)
        nsec = twall.size
        mshell = rho * V_shell
        mshell_tot = np.sum(rho * V_shell)
        k_m = inputs["material_cost_rate"]  # 1.1 # USD / kg carbon steel plate
        k_f = inputs["labor_cost_rate"]  # 1.0 # USD / min labor
        k_p = inputs["painting_cost_rate"]  # USD / m^2 painting
        k_e = 0.064  # Industrial electricity rate $/kWh https://www.eia.gov/electricity/monthly/epm_table_grapher.php?t=epmt_5_6_a
        e_f = 15.9  # Electricity usage kWh/kg for steel
        e_fo = 26.9  # Electricity usage kWh/kg for stainless steel

        # Cost Step 1) Cutting flat plates for taper using plasma cutter
        cutLengths = 2.0 * np.sqrt((Rt - Rb) ** 2.0 + H ** 2.0)  # Factor of 2 for both sides
        # Cost Step 2) Rolling plates
        # Cost Step 3) Welding rolled plates into shells (set difficulty factor based on tapering with logistic function)
        theta_F = 4.0 - 3.0 / (1 + np.exp(-5.0 * (taper - 0.75)))
        # Cost Step 4) Circumferential welds to join cans together
        theta_A = 2.0

        # Labor-based expenses
        K_f = k_f * (
            manufacture.steel_cutting_plasma_time(cutLengths, twall)
            + manufacture.steel_rolling_time(theta_F, R_ave, twall)
            + manufacture.steel_butt_welding_time(theta_A, nsec, mshell_tot, cutLengths, twall)
            + manufacture.steel_butt_welding_time(theta_A, nsec, mshell_tot, 2 * np.pi * Rb[1:], twall[1:])
        )

        # Cost step 5) Painting- outside and inside
        theta_p = 2
        K_p = k_p * theta_p * 2 * (2 * np.pi * R_ave * H).sum()

        # Cost step 6) Outfitting with electricity usage
        K_o = np.sum(1.5 * k_m * (coeff - 1.0) * mshell)

        # Material cost with waste fraction, but without outfitting,
        K_m = 1.21 * np.sum(k_m * mshell)

        # Electricity usage
        K_e = np.sum(k_e * (e_f * mshell + e_fo * (coeff - 1.0) * mshell))

        # Assemble all costs for now
        tempSum = K_m + K_e + K_o + K_p + K_f

        # Capital cost share from BLS MFP by NAICS
        K_c = 0.118 * tempSum / (1.0 - 0.118)

        outputs["cost"] = tempSum + K_c


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
        self.options.declare("n_refine")

    def setup(self):
        n_height = self.options["n_height"]
        n_refine = self.options["n_refine"]
        nFull = get_nfull(n_height, nref=n_refine)

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
        outputs["turbine_mass"] = inputs["rna_mass"] + inputs["tower_mass"]

        cg_rna = inputs["rna_cg"] + np.r_[0.0, 0.0, inputs["hub_height"]]
        cg_tower = np.r_[0.0, 0.0, inputs["tower_center_of_mass"]]
        outputs["turbine_center_of_mass"] = (inputs["rna_mass"] * cg_rna + inputs["tower_mass"] * cg_tower) / outputs[
            "turbine_mass"
        ]

        R = cg_rna
        I_tower = util.assembleI(inputs["tower_I_base"])
        I_rna = util.assembleI(inputs["rna_I"]) + inputs["rna_mass"] * (np.dot(R, R) * np.eye(3) - np.outer(R, R))
        outputs["turbine_I_base"] = util.unassembleI(I_tower + I_rna)
