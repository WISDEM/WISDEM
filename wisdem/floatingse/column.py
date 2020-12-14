import copy

import numpy as np
import openmdao.api as om
import wisdem.commonse.frustum as frustum
import wisdem.commonse.manufacturing as manufacture
from wisdem.commonse import eps, gravity
from wisdem.commonse.utilities import assembleI, unassembleI, nodal2sectional, sectional2nodal, sectionalInterp
from wisdem.commonse.environment import PowerWind, LinearWaves
from wisdem.commonse.wind_wave_drag import AeroHydroLoads, CylinderWaveDrag, CylinderWindDrag
from wisdem.commonse.vertical_cylinder import CylinderMass, CylinderDiscretization, get_nfull
from wisdem.commonse.utilization_constraints import GeometricConstraints, shellBuckling_withStiffeners


def get_inner_radius(Ro, t):
    # Radius varies at nodes, t varies by section
    return Ro - sectional2nodal(t)


def I_tube(r_i, r_o, h, m):
    if type(r_i) == type(np.array([])):
        n = r_i.size
        r_i = r_i.flatten()
        r_o = r_o.flatten()
        h = h.flatten()
        m = m.flatten()
    else:
        n = 1
    Ixx = Iyy = (m / 12.0) * (3.0 * (r_i ** 2.0 + r_o ** 2.0) + h ** 2.0)
    Izz = 0.5 * m * (r_i ** 2.0 + r_o ** 2.0)
    return np.c_[Ixx, Iyy, Izz, np.zeros((n, 3))]


class DiscretizationYAML(om.ExplicitComponent):
    """
    Convert the YAML inputs into more native and easy to use variables.

    Parameters
    ----------
    s : numpy array[n_height_tow]
        1D array of the non-dimensional grid defined along the column axis (0-column base,
        1-column top)
    layer_materials : list of strings
        1D array of the names of the materials of each layer modeled in the column
        structure.
    layer_thickness : numpy array[n_layers_tow, n_height_tow-1], [m]
        2D array of the thickness of the layers of the column structure. The first
        dimension represents each layer, the second dimension represents each piecewise-
        constant entry of the column sections.
    height : float, [m]
        Scalar of the column height computed along the z axis.
    outer_diameter_in : numpy array[n_height_tow], [m]
        cylinder diameter at corresponding locations
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
    section_height : numpy array[n_height-1], [m]
        parameterized section heights along cylinder
    outer_diameter : numpy array[n_height], [m]
        cylinder diameter at corresponding locations
    wall_thickness : numpy array[n_height-1], [m]
        shell thickness at corresponding locations
    E : numpy array[n_height-1], [Pa]
        Isotropic Youngs modulus of the materials along the column sections.
    G : numpy array[n_height-1], [Pa]
        Isotropic shear modulus of the materials along the column sections.
    sigma_y : numpy array[n_height-1], [Pa]
        Isotropic yield strength of the materials along the column sections.
    rho : numpy array[n_height-1], [kg/m**3]
        Density of the materials along the column sections.
    unit_cost : numpy array[n_height-1], [USD/kg]
        Unit costs of the materials along the column sections.

    """

    def initialize(self):
        self.options.declare("n_height")
        self.options.declare("n_layers")
        self.options.declare("n_mat")

    def setup(self):
        n_height = self.options["n_height"]
        n_layers = self.options["n_layers"]
        n_mat = self.options["n_mat"]

        # TODO: Use reference axis and curvature, s, instead of assuming everything is vertical on z
        self.add_input("s", val=np.zeros(n_height))
        self.add_discrete_input("layer_materials", val=n_layers * [""])
        self.add_input("layer_thickness", val=np.zeros((n_layers, n_height - 1)), units="m")
        self.add_input("height", val=0.0, units="m")
        self.add_input("outer_diameter_in", np.zeros(n_height), units="m")
        self.add_discrete_input("material_names", val=n_mat * [""])
        self.add_input("E_mat", val=np.zeros([n_mat, 3]), units="Pa")
        self.add_input("G_mat", val=np.zeros([n_mat, 3]), units="Pa")
        self.add_input("sigma_y_mat", val=np.zeros(n_mat), units="Pa")
        self.add_input("rho_mat", val=np.zeros(n_mat), units="kg/m**3")
        self.add_input("unit_cost_mat", val=np.zeros(n_mat), units="USD/kg")

        self.add_output("section_height", val=np.zeros(n_height - 1), units="m")
        self.add_output("outer_diameter", val=np.zeros(n_height), units="m")
        self.add_output("wall_thickness", val=np.zeros(n_height - 1), units="m")
        self.add_output("E", val=np.zeros(n_height - 1), units="Pa")
        self.add_output("G", val=np.zeros(n_height - 1), units="Pa")
        self.add_output("sigma_y", val=np.zeros(n_height - 1), units="Pa")
        self.add_output("rho", val=np.zeros(n_height - 1), units="kg/m**3")
        self.add_output("unit_cost", val=np.zeros(n_height - 1), units="USD/kg")

        # self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # Unpack dimensions
        n_height = self.options["n_height"]

        # Unpack values
        h_col = inputs["height"]

        outputs["section_height"] = np.diff(h_col * inputs["s"])
        outputs["wall_thickness"] = np.sum(inputs["layer_thickness"], axis=0)
        outputs["outer_diameter"] = inputs["outer_diameter_in"]
        twall = inputs["layer_thickness"]
        layer_mat = copy.copy(discrete_inputs["layer_materials"])

        # Check to make sure we have good values
        if np.any(outputs["section_height"] <= 0.0):
            raise ValueError("Section height values must be greater than zero, " + str(outputs["section_height"]))
        if np.any(outputs["wall_thickness"] <= 0.0):
            raise ValueError("Wall thickness values must be greater than zero, " + str(outputs["wall_thickness"]))
        if np.any(outputs["outer_diameter"] <= 0.0):
            raise ValueError("Diameter values must be greater than zero, " + str(outputs["outer_diameter"]))

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

            # For density, take mass weighted layer
            rho_param += rho[imat] * twall[k, :]

            # For cost, take mass weighted layer
            cost_param += rho[imat] * twall[k, :] * cost[imat]

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


class ColumnGeometry(om.ExplicitComponent):
    """
    Compute geometric properties for vertical columns in substructure
    for floating offshore wind turbines.

    Parameters
    ----------
    water_depth : float, [m]
        water depth
    Hsig_wave : float, [m]
        significant wave height
    freeboard : float, [m]
        Length of column above water line
    max_draft : float, [m]
        Maxmimum length of column below water line
    z_full_in : numpy array[n_full], [m]
        z-coordinates of section nodes (length = nsection+1)
    z_param_in : numpy array[n_height], [m]
        z-coordinates of section nodes (length = nsection+1)
    stiffener_web_height : numpy array[n_sect], [m]
        height of stiffener web (base of T) within each section bottom to top
        (length = nsection)
    stiffener_web_thickness : numpy array[n_sect], [m]
        thickness of stiffener web (base of T) within each section bottom to top
        (length = nsection)
    stiffener_flange_width : numpy array[n_sect], [m]
        height of stiffener flange (top of T) within each section bottom to top
        (length = nsection)
    stiffener_flange_thickness : numpy array[n_sect], [m]
        thickness of stiffener flange (top of T) within each section bottom to top
        (length = nsection)
    stiffener_spacing : numpy array[n_sect], [m]
        Axial distance from one ring stiffener to another within each section bottom to
        top (length = nsection)
    E : numpy array[n_height-1], [Pa]
        Isotropic Youngs modulus of the materials along the column sections.
    G : numpy array[n_height-1], [Pa]
        Isotropic shear modulus of the materials along the column sections.
    sigma_y : numpy array[n_height-1], [Pa]
        Isotropic yield strength of the materials along the column sections.
    rho : numpy array[n_height-1], [kg/m**3]
        Density of the materials along the column sections.
    unit_cost : numpy array[n_height-1], [USD/kg]
        Unit costs of the materials along the column sections.

    Returns
    -------
    z_full : numpy array[n_full], [m]
        z-coordinates of section nodes (length = nsection+1)
    z_param : numpy array[n_height], [m]
        z-coordinates of section nodes (length = nsection+1)
    draft : float, [m]
        Column draft (length of body under water)
    h_web : numpy array[n_full-1], [m]
        height of stiffener web (base of T) within each section bottom to top
    t_web : numpy array[n_full-1], [m]
        thickness of stiffener web (base of T) within each section bottom to top
    w_flange : numpy array[n_full-1], [m]
        height of stiffener flange (top of T) within each section bottom to top
    t_flange : numpy array[n_full-1], [m]
        thickness of stiffener flange (top of T) within each section bottom to top
    L_stiffener : numpy array[n_full-1], [m]
        Axial distance from one ring stiffener to another within each section bottom to
        top
    E_full : numpy array[n_full-1], [Pa]
        Isotropic Youngs modulus of the materials along the column sections.
    G_full : numpy array[n_full-1], [Pa]
        Isotropic shear modulus of the materials along the column sections.
    sigma_y_full : numpy array[n_full-1], [Pa]
        Isotropic yield strength of the materials along the column sections.
    rho_full : numpy array[n_full-1], [kg/m**3]
        Density of the materials along the column sections.
    unit_cost_full : numpy array[n_full-1], [USD/kg]
        Unit costs of the materials along the column sections.
    nu_full : numpy array[n_full-1]
        Poisson's ratio assuming isotropic material
    draft_margin : float
        Ratio of draft to water depth
    wave_height_freeboard_ratio : float
        Ratio of maximum wave height (avg of top 1%) to freeboard

    """

    def initialize(self):
        self.options.declare("n_height")

    def setup(self):
        n_height = self.options["n_height"]
        n_sect = n_height - 1
        n_full = get_nfull(n_height)

        self.add_input("water_depth", 0.0, units="m")
        self.add_input("Hsig_wave", 0.0, units="m")
        self.add_input("freeboard", 0.0, units="m")
        self.add_input("max_draft", 0.0, units="m")
        self.add_input("z_full_in", np.zeros(n_full), units="m")
        self.add_input("z_param_in", np.zeros(n_height), units="m")
        self.add_input("stiffener_web_height", np.zeros(n_sect), units="m")
        self.add_input("stiffener_web_thickness", np.zeros(n_sect), units="m")
        self.add_input("stiffener_flange_width", np.zeros(n_sect), units="m")
        self.add_input("stiffener_flange_thickness", np.zeros(n_sect), units="m")
        self.add_input("stiffener_spacing", np.zeros(n_sect), units="m")
        self.add_input("E", val=np.zeros(n_sect), units="Pa")
        self.add_input("G", val=np.zeros(n_sect), units="Pa")
        self.add_input("sigma_y", val=np.zeros(n_sect), units="Pa")
        self.add_input("rho", val=np.zeros(n_sect), units="kg/m**3")
        self.add_input("unit_cost", val=np.zeros(n_sect), units="USD/kg")

        self.add_output("z_full", np.zeros(n_full), units="m")
        self.add_output("z_param", np.zeros(n_height), units="m")
        self.add_output("draft", 0.0, units="m")
        self.add_output("h_web", np.zeros(n_full - 1), units="m")
        self.add_output("t_web", np.zeros(n_full - 1), units="m")
        self.add_output("w_flange", np.zeros(n_full - 1), units="m")
        self.add_output("t_flange", np.zeros(n_full - 1), units="m")
        self.add_output("L_stiffener", np.zeros(n_full - 1), units="m")
        self.add_output("draft_margin", 0.0)
        self.add_output("wave_height_freeboard_ratio", 0.0)
        self.add_output("E_full", val=np.zeros(n_full - 1), units="Pa")
        self.add_output("G_full", val=np.zeros(n_full - 1), units="Pa")
        self.add_output("nu_full", val=np.zeros(n_full - 1))
        self.add_output("sigma_y_full", val=np.zeros(n_full - 1), units="Pa")
        self.add_output("rho_full", val=np.zeros(n_full - 1), units="kg/m**3")
        self.add_output("unit_cost_full", val=np.zeros(n_full - 1), units="USD/kg")

        self.declare_partials("*", "*", method="fd", form="central", step=1e-6)

    def compute(self, inputs, outputs):
        # Unpack variables
        freeboard = inputs["freeboard"]

        # With waterline at z=0, set the z-position of section nodes
        # Note sections and nodes start at bottom of column and move up
        draft = inputs["z_param_in"][-1] - freeboard
        z_full = inputs["z_full_in"] - draft
        z_param = inputs["z_param_in"] - draft
        outputs["draft"] = draft
        outputs["z_full"] = z_full
        outputs["z_param"] = z_param

        # Create constraint output that draft is less than water depth
        outputs["draft_margin"] = draft / inputs["max_draft"]

        # Make sure freeboard is more than 20% of Hsig_wave (DNV-OS-J101)
        outputs["wave_height_freeboard_ratio"] = inputs["Hsig_wave"] / (np.abs(freeboard) + eps)

        # Material properties
        z_section, _ = nodal2sectional(z_full)
        outputs["rho_full"] = sectionalInterp(z_section, z_param, inputs["rho"])
        outputs["E_full"] = sectionalInterp(z_section, z_param, inputs["E"])
        outputs["G_full"] = sectionalInterp(z_section, z_param, inputs["G"])
        outputs["nu_full"] = 0.5 * outputs["E_full"] / outputs["G_full"] - 1.0
        outputs["sigma_y_full"] = sectionalInterp(z_section, z_param, inputs["sigma_y"])
        outputs["unit_cost_full"] = sectionalInterp(z_section, z_param, inputs["unit_cost"])

        # Sectional stiffener properties
        outputs["t_web"] = sectionalInterp(z_section, z_param, inputs["stiffener_web_thickness"])
        outputs["t_flange"] = sectionalInterp(z_section, z_param, inputs["stiffener_flange_thickness"])
        outputs["h_web"] = sectionalInterp(z_section, z_param, inputs["stiffener_web_height"])
        outputs["w_flange"] = sectionalInterp(z_section, z_param, inputs["stiffener_flange_width"])
        outputs["L_stiffener"] = sectionalInterp(z_section, z_param, inputs["stiffener_spacing"])


class BulkheadProperties(om.ExplicitComponent):
    """
    Compute bulkhead properties

    Parameters
    ----------
    z_full : numpy array[n_full], [m]
        z-coordinates of section nodes
    d_full : numpy array[n_full], [m]
        cylinder diameter at corresponding locations
    t_full : numpy array[n_full-1], [m]
        shell thickness at corresponding locations
    rho_full : numpy array[n_full-1], [kg/m**3]
        material density
    bulkhead_locations : numpy array[n_bulkhead]
        Vector of non-dimensional values (from 0.0 at the bottom bottom to 1.0 at the top) indicating the center locations of the bulkheads
    bulkhead_thickness : numpy array[n_bulkhead], [m]
        Vector of thicknesses of the bulkheads at the locations specified (length = n_bulkhead)
    shell_mass : numpy array[n_full-1], [kg]
        mass of column shell
    unit_cost_full : numpy array[n_full-1], [USD/kg]
        Raw material cost rate: steel $1.1/kg, aluminum $3.5/kg
    labor_cost_rate : float, [USD/min]
        Labor cost rate
    painting_cost_rate : float, [USD/m/m]
        Painting / surface finishing cost rate

    Returns
    -------
    bulkhead_mass : numpy array[n_full-1], [kg]
        mass of column bulkheads
    bulkhead_z_cg : float, [m]
        z-coordinate of center of gravity for all bulkheads
    bulkhead_cost : float, [USD]
        cost of column bulkheads
    bulkhead_I_keel : numpy array[6], [kg*m**2]
        Moments of inertia of bulkheads relative to keel point

    """

    def initialize(self):
        self.options.declare("n_height")
        self.options.declare("n_bulkhead")

    def setup(self):
        n_bulk = self.options["n_bulkhead"]
        n_height = self.options["n_height"]
        n_sect = n_height - 1
        n_full = get_nfull(n_height)

        self.bulk_full = np.zeros(n_full, dtype=np.int_)

        self.add_input("z_full", np.zeros(n_full), units="m")
        self.add_input("d_full", np.zeros(n_full), units="m")
        self.add_input("t_full", np.zeros(n_full - 1), units="m")
        self.add_input("rho_full", np.zeros(n_full - 1), units="kg/m**3")
        self.add_input("bulkhead_thickness", np.zeros(n_bulk), units="m")
        self.add_input("bulkhead_locations", np.zeros(n_bulk))
        self.add_input("shell_mass", np.zeros(n_full - 1), units="kg")
        self.add_input("unit_cost_full", np.zeros(n_full - 1), units="USD/kg")
        self.add_input("labor_cost_rate", 0.0, units="USD/min")
        self.add_input("painting_cost_rate", 0.0, units="USD/m/m")

        self.add_output("bulkhead_mass", np.zeros(n_full - 1), units="kg")
        self.add_output("bulkhead_z_cg", 0.0, units="m")
        self.add_output("bulkhead_cost", 0.0, units="USD")
        self.add_output("bulkhead_I_keel", np.zeros(6), units="kg*m**2")

        self.declare_partials("*", "*", method="fd", form="central", step=1e-6)

    def compute(self, inputs, outputs):
        # Unpack variables
        z_full = inputs["z_full"]  # at section nodes
        R_od = 0.5 * inputs["d_full"]  # at section nodes
        twall = inputs["t_full"]  # at section nodes
        R_id = get_inner_radius(R_od, twall)
        s_bulk = inputs["bulkhead_locations"]
        t_bulk = inputs["bulkhead_thickness"]
        nbulk = s_bulk.size

        # Get z and R_id values of bulkhead locations
        s_full = (z_full - z_full[0]) / (z_full[-1] - z_full[0])
        z_bulk = np.interp(s_bulk, s_full, z_full)
        R_id_bulk = np.interp(s_bulk, s_full, R_id)
        rho = sectionalInterp(s_bulk, s_full, inputs["rho_full"])

        # Compute bulkhead mass
        m_bulk = rho * np.pi * R_id_bulk ** 2 * t_bulk
        z_cg = 0.0 if m_bulk.sum() == 0.0 else np.dot(z_bulk, m_bulk) / m_bulk.sum()

        # Find sectional index for each bulkhead and assign appropriate mass
        m_bulk_sec = np.zeros(z_full.size - 1)
        ibulk = []
        for k in range(nbulk):
            idx = np.where(s_bulk[k] >= s_full[:-1])[0][-1]
            ibulk.append(idx)
            m_bulk_sec[idx] += m_bulk[k]

        # Compute moments of inertia at keel
        # Assume bulkheads are just simple thin discs with radius R_od-t_wall and mass already computed
        Izz = 0.5 * m_bulk * R_id_bulk ** 2
        Ixx = Iyy = 0.5 * Izz
        dz = z_bulk - z_full[0]
        I_keel = np.zeros((3, 3))
        for k in range(nbulk):
            R = np.array([0.0, 0.0, dz[k]])
            Icg = assembleI([Ixx[k], Iyy[k], Izz[k], 0.0, 0.0, 0.0])
            I_keel += Icg + m_bulk[k] * (np.dot(R, R) * np.eye(3) - np.outer(R, R))

        # Compute costs based on "Optimum Design of Steel Structures" by Farkas and Jarmai
        # All dimensions for correlations based on mm, not meters.
        k_m = sectionalInterp(s_bulk, s_full, inputs["unit_cost_full"])
        k_f = inputs["labor_cost_rate"]  # 1.0 # USD / min labor
        k_p = inputs["painting_cost_rate"]  # USD / m^2 painting
        m_shell = inputs["shell_mass"].sum()

        # Cost Step 1) Cutting flat plates using plasma cutter
        cutLengths = 2.0 * np.pi * R_id_bulk
        # Cost Step 2) Fillet welds with GMAW-C (gas metal arc welding with CO2) of bulkheads to shell
        theta_w = 3.0  # Difficulty factor

        # Labor-based expenses
        K_f = k_f * (
            manufacture.steel_cutting_plasma_time(cutLengths, t_bulk)
            + manufacture.steel_filett_welding_time(theta_w, nbulk, m_bulk + m_shell, 2 * np.pi * R_id_bulk, t_bulk)
        )

        # Cost Step 3) Painting (two sided)
        theta_p = 1.0
        K_p = k_p * theta_p * 2 * (np.pi * R_id_bulk ** 2.0).sum()

        # Material cost, without outfitting
        K_m = np.sum(k_m * m_bulk)

        # Total cost
        c_bulk = K_m + K_f + K_p

        # Store results
        outputs["bulkhead_I_keel"] = unassembleI(I_keel)
        outputs["bulkhead_mass"] = m_bulk_sec
        outputs["bulkhead_z_cg"] = z_cg
        outputs["bulkhead_cost"] = c_bulk


class BuoyancyTankProperties(om.ExplicitComponent):
    """
    Compute buoyancy tank properties

    Parameters
    ----------
    d_full : numpy array[n_full], [m]
        cylinder diameter at corresponding locations
    z_full : numpy array[n_full], [m]
        z-coordinates of section nodes
    rho_full : numpy array[n_full-1], [kg/m**3]
        material density
    shell_mass : numpy array[n_full-1], [kg]
        mass of column shell
    unit_cost_full : numpy array[n_full-1], [USD/kg]
        Raw material cost: steel $1.1/kg, aluminum $3.5/kg
    labor_cost_rate : float, [USD/min]
        Labor cost
    painting_cost_rate : float, [USD/m/m]
        Painting / surface finishing cost rate
    buoyancy_tank_diameter : float, [m]
        Radius of heave plate at bottom of column
    buoyancy_tank_height : float, [m]
        Radius of heave plate at bottom of column
    buoyancy_tank_location : float
        Radius of heave plate at bottom of column

    Returns
    -------
    buoyancy_tank_mass : float, [kg]
        mass of buoyancy tank
    buoyancy_tank_cost : float, [USD]
        cost of buoyancy tank
    buoyancy_tank_cg : float, [m]
        z-coordinate of center of mass for buoyancy tank
    buoyancy_tank_displacement : float, [m**3]
        volume of water displaced by buoyancy tank
    buoyancy_tank_I_keel : numpy array[6], [kg*m**2]
        Moments of inertia of heave plate relative to keel point

    """

    def initialize(self):
        self.options.declare("n_height")

    def setup(self):
        n_height = self.options["n_height"]
        n_full = get_nfull(n_height)

        self.add_input("d_full", np.zeros(n_full), units="m")
        self.add_input("z_full", np.zeros(n_full), units="m")
        self.add_input("rho_full", np.zeros(n_full - 1), units="kg/m**3")
        self.add_input("shell_mass", np.zeros(n_full - 1), units="kg")
        self.add_input("unit_cost_full", np.zeros(n_full - 1), units="USD/kg")
        self.add_input("labor_cost_rate", 0.0, units="USD/min")
        self.add_input("painting_cost_rate", 0.0, units="USD/m/m")
        self.add_input("buoyancy_tank_diameter", 0.0, units="m")
        self.add_input("buoyancy_tank_height", 0.0, units="m")
        self.add_input("buoyancy_tank_location", 0.0)

        self.add_output("buoyancy_tank_mass", 0.0, units="kg")
        self.add_output("buoyancy_tank_cost", 0.0, units="USD")
        self.add_output("buoyancy_tank_cg", 0.0, units="m")
        self.add_output("buoyancy_tank_displacement", 0.0, units="m**3")
        self.add_output("buoyancy_tank_I_keel", np.zeros(6), units="kg*m**2")

        self.declare_partials("*", "*", method="fd", form="central", step=1e-6)

    def compute(self, inputs, outputs):
        # Unpack variables
        z_full = inputs["z_full"]

        R_od = 0.5 * inputs["d_full"]
        R_plate = 0.5 * float(inputs["buoyancy_tank_diameter"])
        h_box = float(inputs["buoyancy_tank_height"])

        location = float(inputs["buoyancy_tank_location"])

        # Current hard-coded, coarse specification of shell thickness
        t_plate = R_plate / 50.0

        # Z-locations of buoyancy tank
        z_lower = location * (z_full[-1] - z_full[0]) + z_full[0]
        z_cg = z_lower + 0.5 * h_box
        z_upper = z_lower + h_box

        # Mass and volume properties that subtract out central column contributions for no double-counting
        rho = sectionalInterp(z_lower, z_full, inputs["rho_full"])
        R_col = np.interp([z_lower, z_upper], z_full, R_od)
        if not np.any(R_plate > R_col):
            R_plate = 0.0
        A_plate = np.maximum(0.0, np.pi * (R_plate ** 2.0 - R_col ** 2.0))
        m_plate = rho * t_plate * A_plate
        A_box = A_plate.sum() + 2.0 * np.pi * R_plate * h_box
        m_box = rho * t_plate * A_box

        # Compute displcement for buoyancy calculations, but check for what is submerged
        V_box = np.pi * R_plate ** 2.0 * h_box
        V_box -= frustum.frustumVol(R_col[0], R_col[1], h_box)
        if z_lower >= 0.0:
            V_box = 0.0
        elif z_upper >= 0.0:
            V_box *= -z_lower / h_box
        V_box = np.maximum(0.0, V_box)

        # Now do moments of inertia
        # First find MoI at cg of all components
        R_plate += eps
        Ixx_box = frustum.frustumShellIxx(R_plate, R_plate, t_plate, h_box)
        Izz_box = frustum.frustumShellIzz(R_plate, R_plate, t_plate, h_box)
        I_plateL = 0.25 * m_plate[0] * (R_plate ** 2.0 - R_col[0] ** 2.0) * np.array([1.0, 1.0, 2.0, 0.0, 0.0, 0.0])
        I_plateU = 0.25 * m_plate[1] * (R_plate ** 2.0 - R_col[1] ** 2.0) * np.array([1.0, 1.0, 2.0, 0.0, 0.0, 0.0])

        # Move to keel for consistency
        I_keel = np.zeros((3, 3))
        if R_plate > eps:
            # Add in lower plate
            r = np.array([0.0, 0.0, z_lower])
            Icg = assembleI(I_plateL)
            I_keel += Icg + m_plate[0] * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
            # Add in upper plate
            r = np.array([0.0, 0.0, z_upper])
            Icg = assembleI(I_plateU)
            I_keel += Icg + m_plate[1] * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
            # Add in box cylinder
            r = np.array([0.0, 0.0, z_cg])
            Icg = assembleI([Ixx_box, Ixx_box, Izz_box, 0.0, 0.0, 0.0])
            I_keel += Icg + m_plate[1] * (np.dot(r, r) * np.eye(3) - np.outer(r, r))

        # Compute costs based on "Optimum Design of Steel Structures" by Farkas and Jarmai
        # All dimensions for correlations based on mm, not meters.
        k_m = sectionalInterp(z_lower, z_full, inputs["unit_cost_full"])
        k_f = inputs["labor_cost_rate"]  # 1.0 # USD / min labor
        k_p = inputs["painting_cost_rate"]  # USD / m^2 painting
        m_shell = inputs["shell_mass"].sum()

        # Cost Step 1) Cutting flat plates using plasma cutter into box plate sizes
        cutLengths = 2.0 * np.pi * (3.0 * R_plate + R_col.sum())  # x3 for two plates + side wall
        # Cost Step 2) Welding box plates together GMAW-C (gas metal arc welding with CO2) fillet welds
        theta_w = 3.0  # Difficulty factor
        # Cost Step 3) Welding box to shell GMAW-C (gas metal arc welding with CO2) fillet welds

        # Labor-based expenses
        K_f = k_f * (
            manufacture.steel_cutting_plasma_time(cutLengths, t_plate)
            + manufacture.steel_filett_welding_time(theta_w, 3.0, m_box, 2 * np.pi * R_plate, t_plate)
            + manufacture.steel_filett_welding_time(theta_w, 2.0, m_box + m_shell, 2 * np.pi * R_col, t_plate)
        )

        # Cost Step 4) Painting
        theta_p = 1.5
        K_p = k_p * theta_p * 2.0 * A_box

        # Material cost, without outfitting
        K_m = k_m * m_box

        # Total cost
        c_box = K_m + K_f + K_p

        # Store outputs
        outputs["buoyancy_tank_cost"] = c_box
        outputs["buoyancy_tank_mass"] = m_box
        outputs["buoyancy_tank_cg"] = z_cg
        outputs["buoyancy_tank_displacement"] = V_box
        outputs["buoyancy_tank_I_keel"] = unassembleI(I_keel)


class StiffenerProperties(om.ExplicitComponent):
    """
    Computes column stiffener properties by section.

    Stiffener being the ring of T-cross section members placed periodically along column
    Assumes constant stiffener spacing along the column, but allows for varying stiffener geometry
    Slicing the column lengthwise would reveal the stiffener T-geometry as:
    |              |
    |              |
    |   |      |   |
    |----      ----|
    |   |      |   |
    |              |
    |              |

    Parameters
    ----------
    d_full : numpy array[n_full], [m]
        cylinder diameter at corresponding locations
    t_full : numpy array[n_full-1], [m]
        shell thickness at corresponding locations
    z_full : numpy array[n_full], [m]
        z-coordinates of section nodes
    rho_full : numpy array[n_full-1], [kg/m**3]
        material density
    shell_mass : numpy array[n_full-1], [kg]
        mass of column shell
    unit_cost_full : numpy array[n_full-1], [USD/kg]
        Raw material cost: steel $1.1/kg, aluminum $3.5/kg
    labor_cost_rate : float, [USD/min]
        Labor cost
    painting_cost_rate : float, [USD/m/m]
        Painting / surface finishing cost rate
    h_web : numpy array[n_full-1], [m]
        height of stiffener web (base of T) within each section bottom to top
    t_web : numpy array[n_full-1], [m]
        thickness of stiffener web (base of T) within each section bottom to top
    w_flange : numpy array[n_full-1], [m]
        height of stiffener flange (top of T) within each section bottom to top
    t_flange : numpy array[n_full-1], [m]
        thickness of stiffener flange (top of T) within each section bottom to top
    L_stiffener : numpy array[n_full-1], [m]
        Axial distance from one ring stiffener to another within each section bottom to
        top

    Returns
    -------
    stiffener_mass : numpy array[n_full-1], [kg]
        mass of column stiffeners
    stiffener_cost : float, [USD]
        cost of column stiffeners
    stiffener_I_keel : numpy array[6], [kg*m**2]
        Moments of inertia of stiffeners relative to keel point
    number_of_stiffeners : numpy array[n_sect, dtype]
        number of stiffeners in each section
    flange_spacing_ratio : numpy array[n_full-1]
        ratio between flange and stiffener spacing
    stiffener_radius_ratio : numpy array[n_full-1]
        ratio between stiffener height and radius

    """

    def initialize(self):
        self.options.declare("n_height")

    def setup(self):
        n_height = self.options["n_height"]
        n_sect = n_height - 1
        n_full = get_nfull(n_height)

        self.add_input("d_full", np.zeros(n_full), units="m")
        self.add_input("t_full", np.zeros(n_full - 1), units="m")
        self.add_input("z_full", np.zeros(n_full), units="m")
        self.add_input("rho_full", np.zeros(n_full - 1), units="kg/m**3")
        self.add_input("shell_mass", np.zeros(n_full - 1), units="kg")
        self.add_input("unit_cost_full", np.zeros(n_full - 1), units="USD/kg")
        self.add_input("labor_cost_rate", 0.0, units="USD/min")
        self.add_input("painting_cost_rate", 0.0, units="USD/m/m")
        self.add_input("h_web", np.zeros(n_full - 1), units="m")
        self.add_input("t_web", np.zeros(n_full - 1), units="m")
        self.add_input("w_flange", np.zeros(n_full - 1), units="m")
        self.add_input("t_flange", np.zeros(n_full - 1), units="m")
        self.add_input("L_stiffener", np.zeros(n_full - 1), units="m")

        self.add_output("stiffener_mass", np.zeros(n_full - 1), units="kg")
        self.add_output("stiffener_cost", 0.0, units="USD")
        self.add_output("stiffener_I_keel", np.zeros(6), units="kg*m**2")
        self.add_output("number_of_stiffeners", np.zeros(n_sect, dtype=np.int_))
        self.add_output("flange_spacing_ratio", np.zeros(n_full - 1))
        self.add_output("stiffener_radius_ratio", np.zeros(n_full - 1))

        self.declare_partials("*", "*", method="fd", form="central", step=1e-6)

    def compute(self, inputs, outputs):
        # Unpack variables
        R_od = 0.5 * inputs["d_full"]
        t_wall = inputs["t_full"]
        z_full = inputs["z_full"]  # at section nodes
        h_section = np.diff(z_full)
        V_shell = frustum.frustumShellVol(R_od[:-1], R_od[1:], t_wall, h_section)
        R_od, _ = nodal2sectional(R_od)  # at section nodes

        t_web = inputs["t_web"]
        t_flange = inputs["t_flange"]
        h_web = inputs["h_web"]
        w_flange = inputs["w_flange"]
        L_stiffener = inputs["L_stiffener"]

        rho = inputs["rho_full"]

        # Outer and inner radius of web by section
        R_wo = R_od - t_wall
        R_wi = R_wo - h_web
        # Outer and inner radius of flange by section
        R_fo = R_wi
        R_fi = R_fo - t_flange

        # Material volumes by section
        V_web = np.pi * (R_wo ** 2 - R_wi ** 2) * t_web
        V_flange = np.pi * (R_fo ** 2 - R_fi ** 2) * w_flange

        # Ring mass by volume by section
        m_web = rho * V_web
        m_flange = rho * V_flange
        m_ring = m_web + m_flange
        n_stiff = np.zeros(h_web.shape, dtype=np.int_)

        # Compute moments of inertia for stiffeners (lumped by section for simplicity) at keel
        I_web = I_tube(R_wi, R_wo, t_web, m_web)
        I_flange = I_tube(R_fi, R_fo, w_flange, m_flange)
        I_ring = I_web + I_flange
        I_keel = np.zeros((3, 3))

        # Now march up the column, adding stiffeners at correct spacing until we are done
        z_stiff = []
        isection = 0
        epsilon = 1e-6
        while True:
            if len(z_stiff) == 0:
                z_march = np.minimum(z_full[isection + 1], z_full[0] + 0.5 * L_stiffener[isection]) + epsilon
            else:
                z_march = np.minimum(z_full[isection + 1], z_stiff[-1] + L_stiffener[isection]) + epsilon
            if z_march >= z_full[-1]:
                break

            isection = np.searchsorted(z_full, z_march) - 1

            if len(z_stiff) == 0:
                add_stiff = (z_march - z_full[0]) >= 0.5 * L_stiffener[isection]
            else:
                add_stiff = (z_march - z_stiff[-1]) >= L_stiffener[isection]

            if add_stiff:
                z_stiff.append(z_march)
                n_stiff[isection] += 1

                R = np.array([0.0, 0.0, (z_march - z_full[0])])
                Icg = assembleI(I_ring[isection, :])
                I_keel += Icg + m_ring[isection] * (np.dot(R, R) * np.eye(3) - np.outer(R, R))

        # Number of stiffener rings per section (height of section divided by spacing)
        outputs["stiffener_mass"] = n_stiff * m_ring

        # Find total number of stiffeners in each original section
        n_sect = self.options["n_height"] - 1
        npts_per = int(h_web.size / n_sect)
        n_stiff_sec = np.zeros(n_sect)
        for k in range(npts_per):
            n_stiff_sec += n_stiff[k::npts_per]
        outputs["number_of_stiffeners"] = n_stiff_sec

        # Compute costs based on "Optimum Design of Steel Structures" by Farkas and Jarmai
        # All dimensions for correlations based on mm, not meters.
        k_m = inputs["unit_cost_full"]  # 1.1 # USD / kg carbon steel plate
        k_f = inputs["labor_cost_rate"]  # 1.0 # USD / min labor
        k_p = inputs["painting_cost_rate"]  # USD / m^2 painting
        m_shell = inputs["shell_mass"].sum()

        # Cost Step 1) Cutting stiffener strips from flat plates using plasma cutter
        cutLengths_w = 2.0 * np.pi * 0.5 * (R_wo + R_wi)
        cutLengths_f = 2.0 * np.pi * R_fo
        # Cost Step 2) Welding T-stiffeners together GMAW-C (gas metal arc welding with CO2) fillet welds
        theta_w = 3.0  # Difficulty factor
        # Cost Step 3) Welding stiffeners to shell GMAW-C (gas metal arc welding with CO2) fillet welds
        # Will likely fillet weld twice (top & bottom), so factor of 2 on second welding terms

        # Labor-based expenses
        K_f = k_f * (
            manufacture.steel_cutting_plasma_time(n_stiff * cutLengths_w, t_web)
            + manufacture.steel_cutting_plasma_time(n_stiff * cutLengths_f, t_flange)
            + manufacture.steel_filett_welding_time(theta_w, n_stiff, m_ring, 2 * np.pi * R_fo, t_web)
            + manufacture.steel_filett_welding_time(theta_w, n_stiff, m_ring + m_shell, 2 * np.pi * R_wo, t_web)
        )

        # Cost Step 4) Painting
        theta_p = 2.0
        K_p = (
            k_p
            * theta_p
            * (
                n_stiff
                * (
                    2 * np.pi * (R_wo ** 2.0 - R_wi ** 2.0)
                    + 2 * np.pi * 0.5 * (R_fo + R_fi) * (2 * w_flange + 2 * t_flange)
                    - 2 * np.pi * R_fo * t_web
                )
            ).sum()
        )

        # Material cost, without outfitting
        K_m = np.sum(k_m * outputs["stiffener_mass"])

        # Total cost
        c_ring = K_m + K_f + K_p

        # Store results
        outputs["stiffener_cost"] = c_ring
        outputs["stiffener_I_keel"] = unassembleI(I_keel)

        # Create some constraints for reasonable stiffener designs for an optimizer
        outputs["flange_spacing_ratio"] = w_flange / (0.5 * L_stiffener)
        outputs["stiffener_radius_ratio"] = (h_web + t_flange + t_wall) / R_od


class BallastProperties(om.ExplicitComponent):
    """
    Compute ballast properties

    Parameters
    ----------
    rho_water : float, [kg/m**3]
        density of water
    d_full : numpy array[n_full], [m]
        cylinder diameter at corresponding locations
    t_full : numpy array[n_full-1], [m]
        shell thickness at corresponding locations
    z_full : numpy array[n_full], [m]
        z-coordinates of section nodes
    permanent_ballast_density : float, [kg/m**3]
        density of permanent ballast
    permanent_ballast_height : float, [m]
        height of permanent ballast
    ballast_cost_rate : float, [USD/kg]
        Cost per unit mass of ballast

    Returns
    -------
    ballast_cost : float, [USD]
        cost of permanent ballast
    ballast_mass : numpy array[n_full-1], [kg]
        mass of permanent ballast
    ballast_z_cg : float, [m]
        z-coordinate or permanent ballast center of gravity
    ballast_I_keel : numpy array[6], [kg*m**2]
        Moments of inertia of permanent ballast relative to keel point
    variable_ballast_interp_zpts : numpy array[n_full], [m]
        z-points of potential ballast mass
    variable_ballast_interp_radius : numpy array[n_full], [m]
        inner radius of column at potential ballast mass

    """

    def initialize(self):
        self.options.declare("n_height")

    def setup(self):
        n_height = self.options["n_height"]
        n_full = get_nfull(n_height)

        self.add_input("rho_water", 0.0, units="kg/m**3")
        self.add_input("d_full", np.zeros(n_full), units="m")
        self.add_input("t_full", np.zeros(n_full - 1), units="m")
        self.add_input("z_full", np.zeros(n_full), units="m")
        self.add_input("permanent_ballast_density", 0.0, units="kg/m**3")
        self.add_input("permanent_ballast_height", 0.0, units="m")
        self.add_input("ballast_cost_rate", 0.0, units="USD/kg")

        self.add_output("ballast_cost", 0.0, units="USD")
        self.add_output("ballast_mass", np.zeros(n_full - 1), units="kg")
        self.add_output("ballast_z_cg", 0.0, units="m")
        self.add_output("ballast_I_keel", np.zeros(6), units="kg*m**2")
        self.add_output("variable_ballast_interp_zpts", np.zeros(n_full), units="m")
        self.add_output("variable_ballast_interp_radius", np.zeros(n_full), units="m")

    def compute(self, inputs, outputs):
        # Unpack variables
        R_od = 0.5 * inputs["d_full"]
        t_wall = inputs["t_full"]
        z_nodes = inputs["z_full"]
        h_ballast = float(inputs["permanent_ballast_height"])
        rho_ballast = float(inputs["permanent_ballast_density"])
        rho_water = float(inputs["rho_water"])
        R_id_orig = get_inner_radius(R_od, t_wall)

        npts = R_od.size
        section_mass = np.zeros(npts - 1)

        # Geometry of the column in our coordinate system (z=0 at waterline)
        z_draft = z_nodes[0]

        # Fixed and total ballast mass and cg
        # Assume they are bottled in columns a the keel of the column- first the permanent then the fixed
        zpts = np.linspace(z_draft, z_draft + h_ballast, npts)
        R_id = np.interp(zpts, z_nodes, R_id_orig)
        V_perm = np.pi * np.trapz(R_id ** 2, zpts)
        m_perm = rho_ballast * V_perm
        z_cg_perm = rho_ballast * np.pi * np.trapz(zpts * R_id ** 2, zpts) / m_perm if m_perm > 0.0 else 0.0
        for k in range(npts - 1):
            ind = np.logical_and(zpts >= z_nodes[k], zpts <= z_nodes[k + 1])
            section_mass[k] += rho_ballast * np.pi * np.trapz(R_id[ind] ** 2, zpts[ind])

        Ixx = Iyy = frustum.frustumIxx(R_id[:-1], R_id[1:], np.diff(zpts))
        Izz = frustum.frustumIzz(R_id[:-1], R_id[1:], np.diff(zpts))
        V_slice = frustum.frustumVol(R_id[:-1], R_id[1:], np.diff(zpts))
        I_keel = np.zeros((3, 3))
        dz = frustum.frustumCG(R_id[:-1], R_id[1:], np.diff(zpts)) + zpts[:-1] - z_draft
        for k in range(V_slice.size):
            R = np.array([0.0, 0.0, dz[k]])
            Icg = assembleI([Ixx[k], Iyy[k], Izz[k], 0.0, 0.0, 0.0])
            I_keel += Icg + V_slice[k] * (np.dot(R, R) * np.eye(3) - np.outer(R, R))
        I_keel = rho_ballast * unassembleI(I_keel)

        # Water ballast will start at top of fixed ballast
        z_water_start = z_draft + h_ballast
        # z_water_start = z_water_start + inputs['variable_ballast_start'] * (z_nodes[-1] - z_water_start)

        # Find height of water ballast numerically by finding the height that integrates to the mass we want
        # This step is completed in column.py or semi.py because we must account for other substructure elements too
        zpts = np.linspace(z_water_start, 0.0, npts)
        R_id = np.interp(zpts, z_nodes, R_id_orig)
        outputs["variable_ballast_interp_zpts"] = zpts
        outputs["variable_ballast_interp_radius"] = R_id

        # Save permanent ballast mass and variable height
        outputs["ballast_mass"] = section_mass
        outputs["ballast_I_keel"] = I_keel
        outputs["ballast_z_cg"] = z_cg_perm
        outputs["ballast_cost"] = inputs["ballast_cost_rate"] * m_perm


class ColumnProperties(om.ExplicitComponent):
    """
    Compute column substructure elements in floating offshore wind turbines.

    Parameters
    ----------
    rho_water : float, [kg/m**3]
        density of water
    z_full : numpy array[n_full], [m]
        z-coordinates of section nodes (length = nsection+1)
    z_section : numpy array[n_full-1], [m]
        z-coordinates of section centers of mass (length = nsection)
    d_full : numpy array[n_full], [m]
        outer diameter at each section node bottom to top (length = nsection + 1)
    t_full : numpy array[n_full-1], [m]
        shell wall thickness at each section node bottom to top (length = nsection + 1)
    buoyancy_tank_diameter : float, [m]
        Radius of heave plate at bottom of column
    shell_mass : numpy array[n_full-1], [kg]
        mass of column shell
    stiffener_mass : numpy array[n_full-1], [kg]
        mass of column stiffeners
    bulkhead_mass : numpy array[n_full-1], [kg]
        mass of column bulkheads
    buoyancy_tank_mass : float, [kg]
        mass of heave plate
    ballast_mass : numpy array[n_full-1], [kg]
        mass of permanent ballast
    buoyancy_tank_cg : float, [m]
        z-coordinate of center of mass for buoyancy tank
    ballast_z_cg : float, [m]
        z-coordinate or permanent ballast center of gravity
    outfitting_factor : float
        Mass fraction added for outfitting
    shell_I_keel : numpy array[6], [kg*m**2]
        Moments of inertia of outer shell relative to keel point
    bulkhead_I_keel : numpy array[6], [kg*m**2]
        Moments of inertia of bulkheads relative to keel point
    stiffener_I_keel : numpy array[6], [kg*m**2]
        Moments of inertia of stiffeners relative to keel point
    buoyancy_tank_I_keel : numpy array[6], [kg*m**2]
        Moments of inertia of heave plate relative to keel point
    ballast_I_keel : numpy array[6], [kg*m**2]
        Moments of inertia of permanent ballast relative to keel point
    buoyancy_tank_displacement : float, [m**3]
        volume of water displaced by buoyancy tank
    shell_cost : float, [USD]
        mass of column shell
    stiffener_cost : float, [USD]
        mass of column stiffeners
    bulkhead_cost : float, [USD]
        mass of column bulkheads
    ballast_cost : float, [USD]
        cost of permanent ballast
    buoyancy_tank_cost : float, [USD]
        mass of heave plate
    outfitting_cost_rate : float, [USD/kg]
        Cost per unit mass for outfitting column

    Returns
    -------
    z_center_of_mass : float, [m]
        z-position CofG of column
    z_center_of_buoyancy : float, [m]
        z-position CofB of column
    Awater : float, [m**2]
        Area of waterplace cross section
    Iwater : float, [m**4]
        Second moment of area of waterplace cross section
    I_column : numpy array[6], [kg*m**2]
        Moments of inertia of whole column relative to keel point
    displaced_volume : numpy array[n_full-1], [m**3]
        Volume of water displaced by column by section
    hydrostatic_force : numpy array[n_full-1], [N]
        Net z-force on column sections
    column_structural_mass : float, [kg]
        mass of column structure
    column_outfitting_cost : float, [USD]
        cost of outfitting the column
    column_outfitting_mass : float, [kg]
        cost of outfitting the column
    column_added_mass : numpy array[6], [kg]
        hydrodynamic added mass matrix diagonal
    column_total_mass : numpy array[n_full-1], [kg]
        total mass of column by section
    column_total_cost : float, [USD]
        total cost of column
    column_structural_cost : float, [USD]
        Cost of column without ballast or outfitting
    tapered_column_cost_rate : float, [USD/t]
        Cost rate of finished column

    """

    def initialize(self):
        self.options.declare("n_height")

    def setup(self):
        n_height = self.options["n_height"]
        n_full = get_nfull(n_height)

        # Variables local to the class and not OpenMDAO
        self.ibox = None

        self.add_input("rho_water", 0.0, units="kg/m**3")

        # Inputs from geometry
        self.add_input("z_full", np.zeros(n_full), units="m")
        self.add_input("z_section", np.zeros(n_full - 1), units="m")

        # Design variables
        self.add_input("d_full", np.zeros(n_full), units="m")
        self.add_input("t_full", np.zeros(n_full - 1), units="m")
        self.add_input("buoyancy_tank_diameter", 0.0, units="m")

        # Mass correction factors from simple rules here to real life
        self.add_input("shell_mass", np.zeros(n_full - 1), units="kg")
        self.add_input("stiffener_mass", np.zeros(n_full - 1), units="kg")
        self.add_input("bulkhead_mass", np.zeros(n_full - 1), units="kg")
        self.add_input("buoyancy_tank_mass", 0.0, units="kg")
        self.add_input("ballast_mass", np.zeros(n_full - 1), units="kg")

        self.add_input("buoyancy_tank_cg", 0.0, units="m")
        self.add_input("bulkhead_z_cg", 0.0, units="m")
        self.add_input("ballast_z_cg", 0.0, units="m")
        self.add_input("outfitting_factor", 0.0)

        # Moments of inertia
        self.add_input("shell_I_keel", np.zeros(6), units="kg*m**2")
        self.add_input("bulkhead_I_keel", np.zeros(6), units="kg*m**2")
        self.add_input("stiffener_I_keel", np.zeros(6), units="kg*m**2")
        self.add_input("buoyancy_tank_I_keel", np.zeros(6), units="kg*m**2")
        self.add_input("ballast_I_keel", np.zeros(6), units="kg*m**2")

        # For buoyancy
        self.add_input("buoyancy_tank_displacement", 0.0, units="m**3")

        # Costs and cost rates
        self.add_input("shell_cost", 0.0, units="USD")
        self.add_input("stiffener_cost", 0.0, units="USD")
        self.add_input("bulkhead_cost", 0.0, units="USD")
        self.add_input("ballast_cost", 0.0, units="USD")
        self.add_input("buoyancy_tank_cost", 0.0, units="USD")
        self.add_input("outfitting_cost_rate", 0.0, units="USD/kg")

        self.add_output("z_center_of_mass", 0.0, units="m")
        self.add_output("z_center_of_buoyancy", 0.0, units="m")
        self.add_output("Awater", 0.0, units="m**2")
        self.add_output("Iwater", 0.0, units="m**4")
        self.add_output("I_column", np.zeros(6), units="kg*m**2")
        self.add_output("displaced_volume", np.zeros(n_full - 1), units="m**3")
        self.add_output("hydrostatic_force", np.zeros(n_full - 1), units="N")

        self.add_output("column_structural_mass", 0.0, units="kg")
        self.add_output("column_outfitting_cost", 0.0, units="USD")
        self.add_output("column_outfitting_mass", 0.0, units="kg")

        self.add_output("column_added_mass", np.zeros(6), units="kg")
        self.add_output("column_total_mass", np.zeros(n_full - 1), units="kg")
        self.add_output("column_total_cost", 0.0, units="USD")
        self.add_output("column_structural_cost", 0.0, units="USD")
        self.add_output("tapered_column_cost_rate", 0.0, units="USD/t")

        self.declare_partials("*", "*", method="fd", form="central", step=1e-6)

    def compute(self, inputs, outputs):
        self.compute_column_mass_cg(inputs, outputs)
        self.balance_column(inputs, outputs)
        self.compute_cost(inputs, outputs)

    def compute_column_mass_cg(self, inputs, outputs):
        """Computes column mass from components: Shell, Stiffener rings, Bulkheads
        Also computes center of mass of the shell by weighted sum of the components' position

        INPUTS:
        ----------
        inputs   : dictionary of input parameters
        outputs : dictionary of output parameters

        OUTPUTS:
        ----------
        section_mass class variable set
        m_column   : column mass
        z_cg     : center of mass along z-axis for the column
        column_mass       in 'outputs' dictionary set
        shell_mass      in 'outputs' dictionary set
        stiffener_mass  in 'outputs' dictionary set
        bulkhead_mass   in 'outputs' dictionary set
        outfitting_mass in 'outputs' dictionary set
        """
        # Unpack variables
        out_frac = inputs["outfitting_factor"]
        z_nodes = inputs["z_full"]
        z_section = inputs["z_section"]
        z_box = inputs["buoyancy_tank_cg"]
        z_ballast = inputs["ballast_z_cg"]
        z_bulkhead = inputs["bulkhead_z_cg"]
        m_shell = inputs["shell_mass"]
        m_stiffener = inputs["stiffener_mass"]
        m_bulkhead = inputs["bulkhead_mass"]
        m_box = inputs["buoyancy_tank_mass"]
        m_ballast = inputs["ballast_mass"]
        I_shell = inputs["shell_I_keel"]
        I_stiffener = inputs["stiffener_I_keel"]
        I_bulkhead = inputs["bulkhead_I_keel"]
        I_box = inputs["buoyancy_tank_I_keel"]
        I_ballast = inputs["ballast_I_keel"]

        # Consistency check
        if out_frac < 1.0:
            out_frac += 1.0

        # Initialize summations
        m_column = 0.0
        z_cg = 0.0

        # Find mass of all of the sub-components of the column
        # Masses assumed to be focused at section centroids
        m_column += (m_shell + m_stiffener).sum()
        z_cg += np.dot(m_shell + m_stiffener, z_section)

        # Mass with variable location
        m_column += m_box + m_bulkhead.sum()
        z_cg += m_box * z_box + m_bulkhead.sum() * z_bulkhead

        z_cg /= m_column

        # Now calculate outfitting mass, evenly distributed so cg doesn't change
        m_outfit = (out_frac - 1.0) * m_column

        # Add in ballast
        m_total = m_column + m_outfit + m_ballast.sum()
        z_cg = ((m_column + m_outfit) * z_cg + m_ballast.sum() * z_ballast) / m_total

        # Find sections for ballast and buoyancy tank
        ibox = 0
        try:
            ibox = np.where(z_box >= z_nodes)[0][-1]
        except:
            print(z_box, z_ballast, z_nodes)
        self.ibox = ibox

        # Now do tally by section
        m_sections = (m_shell + m_stiffener + m_bulkhead) + m_ballast + m_outfit / m_shell.size
        m_sections[ibox] += m_box

        # Add up moments of inertia at keel, make sure to scale mass appropriately
        I_total = out_frac * (I_shell + I_stiffener + I_bulkhead + I_box) + I_ballast

        # Move moments of inertia from keel to cg
        I_total -= m_total * ((z_cg - z_nodes[0]) ** 2.0) * np.r_[1.0, 1.0, np.zeros(4)]
        I_total = np.maximum(I_total, 0.0)

        # Store outputs addressed so far
        outputs["column_total_mass"] = m_sections
        outputs["column_structural_mass"] = m_column + m_outfit
        outputs["column_outfitting_mass"] = m_outfit
        outputs["z_center_of_mass"] = z_cg
        outputs["I_column"] = I_total

    def balance_column(self, inputs, outputs):
        # Unpack variables
        R_od = 0.5 * inputs["d_full"]
        R_plate = 0.5 * inputs["buoyancy_tank_diameter"]
        z_nodes = inputs["z_full"]
        z_box = inputs["buoyancy_tank_cg"]
        V_box = inputs["buoyancy_tank_displacement"]
        rho_water = inputs["rho_water"]
        nsection = R_od.size - 1

        # Compute volume of each section and mass of displaced water by section
        # Find the radius at the waterline so that we can compute the submerged volume as a sum of frustum sections
        if z_nodes[-1] > 0.0:
            r_waterline = np.interp(0.0, z_nodes, R_od)
            z_under = np.r_[z_nodes[z_nodes < 0.0], 0.0]
            r_under = np.r_[R_od[z_nodes < 0.0], r_waterline]
        else:
            r_waterline = R_od[-1]
            r_under = R_od
            z_under = z_nodes

        # Submerged volume (with zero-padding)
        V_under = frustum.frustumVol(r_under[:-1], r_under[1:], np.diff(z_under))
        add0 = np.maximum(0, nsection - V_under.size)
        outputs["displaced_volume"] = np.r_[V_under, np.zeros(add0)]
        outputs["displaced_volume"][self.ibox] += V_box

        # Compute Center of Buoyancy in z-coordinates (0=waterline)
        # First get z-coordinates of CG of all frustums
        z_cg_under = frustum.frustumCG(r_under[:-1], r_under[1:], np.diff(z_under))
        z_cg_under += z_under[:-1]
        # Now take weighted average of these CG points with volume
        z_cb = ((V_box * z_box) + np.dot(V_under, z_cg_under)) / (outputs["displaced_volume"].sum() + eps)
        outputs["z_center_of_buoyancy"] = z_cb

        # Find total hydrostatic force by section- sign says in which direction force acts
        # Since we are working on z_under grid, need to redefine z_section, ibox, etc.
        z_undersec, _ = nodal2sectional(z_under)
        if z_box > 0.0 and V_box == 0.0:
            ibox = 0
        else:
            ibox = np.where(z_box >= z_under)[0][-1]
        F_hydro = np.pi * np.diff(r_under ** 2.0) * np.maximum(0.0, -z_undersec)  # cg_under))
        if F_hydro.size > 0:
            F_hydro[0] += np.pi * r_under[0] ** 2 * (-z_under[0])
            if z_nodes[-1] < 0.0:
                F_hydro[-1] -= np.pi * r_under[-1] ** 2 * (-z_under[-1])
            F_hydro[ibox] += V_box
            F_hydro *= rho_water * gravity
        outputs["hydrostatic_force"] = np.r_[F_hydro, np.zeros(add0)]

        # 2nd moment of area for circular cross section
        # Note: Assuming Iwater here depends on "water displacement" cross-section
        # and not actual moment of inertia type of cross section (thin hoop)
        outputs["Iwater"] = 0.25 * np.pi * r_waterline ** 4.0
        outputs["Awater"] = np.pi * r_waterline ** 2.0

        # Calculate diagonal entries of added mass matrix
        # Prep for integrals too
        npts = 100 * R_od.size
        zpts = np.linspace(z_under[0], z_under[-1], npts)
        r_under = np.interp(zpts, z_under, r_under)
        m_a = np.zeros(6)
        m_a[:2] = rho_water * outputs["displaced_volume"].sum()  # A11 surge, A22 sway
        m_a[2] = 0.5 * (8.0 / 3.0) * rho_water * np.maximum(R_plate, r_under.max()) ** 3.0  # A33 heave
        m_a[3:5] = np.pi * rho_water * np.trapz((zpts - z_cb) ** 2.0 * r_under ** 2.0, zpts)  # A44 roll, A55 pitch
        m_a[5] = 0.0  # A66 yaw
        outputs["column_added_mass"] = m_a

    def compute_cost(self, inputs, outputs):
        outputs["column_structural_cost"] = (
            inputs["shell_cost"] + inputs["stiffener_cost"] + inputs["bulkhead_cost"] + inputs["buoyancy_tank_cost"]
        )
        outputs["column_outfitting_cost"] = inputs["outfitting_cost_rate"] * outputs["column_outfitting_mass"]
        outputs["column_total_cost"] = (
            outputs["column_structural_cost"] + outputs["column_outfitting_cost"] + inputs["ballast_cost"]
        )
        outputs["tapered_column_cost_rate"] = 1e3 * outputs["column_total_cost"] / outputs["column_total_mass"].sum()


class ColumnBuckling(om.ExplicitComponent):
    """
    Compute the applied axial and hoop stresses in a column and compare that to
    limits established by the API standard. Some physcial geometry checks are also performed.

    Parameters
    ----------
    stack_mass_in : float, [kg]
        Weight above the cylinder column
    section_mass : numpy array[n_full-1], [kg]
        total mass of column by section
    pressure : numpy array[n_full], [N/m**2]
        Dynamic (and static)? pressure
    d_full : numpy array[n_full], [m]
        cylinder diameter at corresponding locations
    t_full : numpy array[n_full-1], [m]
        shell thickness at corresponding locations
    z_full : numpy array[n_full], [m]
        z-coordinates of section nodes (length = nsection+1)
    h_web : numpy array[n_full-1], [m]
        height of stiffener web (base of T) within each section bottom to top
    t_web : numpy array[n_full-1], [m]
        thickness of stiffener web (base of T) within each section bottom to top
    w_flange : numpy array[n_full-1], [m]
        height of stiffener flange (top of T) within each section bottom to top
    t_flange : numpy array[n_full-1], [m]
        thickness of stiffener flange (top of T) within each section bottom to top
    L_stiffener : numpy array[n_full-1], [m]
        Axial distance from one ring stiffener to another within each section bottom to
        top
    E : float, [Pa]
        Modulus of elasticity (Youngs) of material
    nu : float
        poissons ratio of column material
    yield_stress : float, [Pa]
        yield stress of material
    loading : string
        Loading type in API checks [hydro/radial]

    Returns
    -------
    flange_compactness : numpy array[n_full-1]
        check for flange compactness
    web_compactness : numpy array[n_full-1]
        check for web compactness
    axial_local_api : numpy array[n_full-1]
        unity check for axial load with API safety factors - local buckling
    axial_general_api : numpy array[n_full-1]
        unity check for axial load with API safety factors- genenral instability
    external_local_api : numpy array[n_full-1]
        unity check for external pressure with API safety factors- local buckling
    external_general_api : numpy array[n_full-1]
        unity check for external pressure with API safety factors- general instability
    axial_local_utilization : numpy array[n_full-1]
        utilization check for axial load - local buckling
    axial_general_utilization : numpy array[n_full-1]
        utilization check for axial load - genenral instability
    external_local_utilization : numpy array[n_full-1]
        utilization check for external pressure - local buckling
    external_general_utilization : numpy array[n_full-1]
        utilization check for external pressure - general instability

    """

    def initialize(self):
        self.options.declare("n_height")
        self.options.declare("modeling_options")

    def setup(self):
        n_height = self.options["n_height"]
        n_full = get_nfull(n_height)

        self.add_input("stack_mass_in", eps, units="kg")
        self.add_input("section_mass", np.zeros(n_full - 1), units="kg")
        self.add_input("pressure", np.zeros(n_full), units="N/m**2")
        self.add_input("d_full", np.zeros(n_full), units="m")
        self.add_input("t_full", np.zeros(n_full - 1), units="m")
        self.add_input("z_full", np.zeros(n_full), units="m")
        self.add_input("h_web", np.zeros(n_full - 1), units="m")
        self.add_input("t_web", np.zeros(n_full - 1), units="m")
        self.add_input("w_flange", np.zeros(n_full - 1), units="m")
        self.add_input("t_flange", np.zeros(n_full - 1), units="m")
        self.add_input("L_stiffener", np.zeros(n_full - 1), units="m")
        self.add_input("E_full", np.zeros(n_full - 1), units="Pa")
        self.add_input("nu_full", np.zeros(n_full - 1))
        self.add_input("sigma_y_full", np.zeros(n_full - 1), units="Pa")
        self.add_discrete_input("loading", "hydro")

        self.add_output("flange_compactness", np.zeros(n_full - 1))
        self.add_output("web_compactness", np.zeros(n_full - 1))
        self.add_output("axial_local_api", np.zeros(n_full - 1))
        self.add_output("axial_general_api", np.zeros(n_full - 1))
        self.add_output("external_local_api", np.zeros(n_full - 1))
        self.add_output("external_general_api", np.zeros(n_full - 1))
        self.add_output("axial_local_utilization", np.zeros(n_full - 1))
        self.add_output("axial_general_utilization", np.zeros(n_full - 1))
        self.add_output("external_local_utilization", np.zeros(n_full - 1))
        self.add_output("external_general_utilization", np.zeros(n_full - 1))

        # Derivatives
        self.declare_partials("*", "*", method="fd", form="central", step=1e-6)

    def compute_applied_axial(self, inputs):
        """Compute axial stress for column from z-axis loading

        INPUTS:
        ----------
        inputs       : dictionary of input parameters
        section_mass : float (scalar/vector),  mass of each column section as axial loading increases with column depth

        OUTPUTS:
        -------
        stress   : float (scalar/vector),  axial stress
        """
        # Unpack variables
        R_od, _ = nodal2sectional(inputs["d_full"])
        R_od *= 0.5
        t_wall = inputs["t_full"]
        section_mass = inputs["section_mass"]
        m_stack = inputs["stack_mass_in"]

        # Middle radius
        R_m = R_od - 0.5 * t_wall
        # Add in weight of sections above it
        axial_load = m_stack + np.r_[0.0, np.cumsum(section_mass[:-1])]
        # Divide by shell cross sectional area to get stress
        return gravity * axial_load / (2.0 * np.pi * R_m * t_wall)

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # Unpack variables
        R_od, _ = nodal2sectional(inputs["d_full"])
        R_od *= 0.5
        h_section = np.diff(inputs["z_full"])
        t_wall = inputs["t_full"]

        t_web = inputs["t_web"]
        t_flange = inputs["t_flange"]
        h_web = inputs["h_web"]
        w_flange = inputs["w_flange"]
        L_stiffener = inputs["L_stiffener"]

        gamma_f = self.options["modeling_options"]["gamma_f"]
        gamma_b = self.options["modeling_options"]["gamma_b"]

        E = inputs["E_full"]  # Young's modulus
        nu = inputs["nu_full"]  # Poisson ratio
        sigma_y = inputs["sigma_y_full"]
        loading = discrete_inputs["loading"]
        nodalP, _ = nodal2sectional(inputs["pressure"])
        pressure = 1e-12 if loading in ["ax", "axial", "testing", "test"] else nodalP + 1e-12

        # Apply quick "compactness" check on stiffener geometry
        # Constraint is that these must be >= 1
        flange_compactness = 0.375 * (t_flange / (0.5 * w_flange)) * np.sqrt(E / sigma_y)
        web_compactness = 1.0 * (t_web / h_web) * np.sqrt(E / sigma_y)

        # Compute applied axial stress simply, like API guidelines (as opposed to running frame3dd)
        sigma_ax = self.compute_applied_axial(inputs)
        (
            axial_local_api,
            axial_general_api,
            external_local_api,
            external_general_api,
            axial_local_raw,
            axial_general_raw,
            external_local_raw,
            external_general_raw,
        ) = shellBuckling_withStiffeners(
            pressure,
            sigma_ax,
            R_od,
            t_wall,
            h_section,
            h_web,
            t_web,
            w_flange,
            t_flange,
            L_stiffener,
            E,
            nu,
            sigma_y,
            loading,
        )

        outputs["flange_compactness"] = flange_compactness
        outputs["web_compactness"] = web_compactness

        outputs["axial_local_api"] = axial_local_api
        outputs["axial_general_api"] = axial_general_api
        outputs["external_local_api"] = external_local_api
        outputs["external_general_api"] = external_general_api

        outputs["axial_local_utilization"] = axial_local_raw * gamma_f * gamma_b
        outputs["axial_general_utilization"] = axial_general_raw * gamma_f * gamma_b
        outputs["external_local_utilization"] = external_local_raw * gamma_f * gamma_b
        outputs["external_general_utilization"] = external_general_raw * gamma_f * gamma_b


class Column(om.Group):
    def initialize(self):
        self.options.declare("n_mat")
        self.options.declare("column_options")
        self.options.declare("modeling_options")

    def setup(self):
        opt = self.options["modeling_options"]
        colopt = self.options["column_options"]
        n_layers = colopt["n_layers"]
        n_height = colopt["n_height"]
        n_bulk = colopt["n_bulkhead"]
        n_mat = self.options["n_mat"]
        n_sect = n_height - 1
        n_full = get_nfull(n_height)

        self.set_input_defaults("cd_usr", -1.0)
        self.set_input_defaults("Tsig_wave", 10.0)
        self.set_input_defaults("layer_materials", ["steel"])
        self.set_input_defaults("material_names", ["steel"])

        # TODO: Use reference axis and curvature, s, instead of assuming everything is vertical on z
        self.add_subsystem(
            "yaml", DiscretizationYAML(n_height=n_height, n_layers=n_layers, n_mat=n_mat), promotes=["*"]
        )

        self.add_subsystem(
            "gc", GeometricConstraints(nPoints=n_height, diamFlag=True), promotes=["constr_taper", "constr_d_to_t"]
        )

        self.add_subsystem(
            "cyl_geom",
            CylinderDiscretization(nPoints=n_height),
            promotes=["section_height", "diameter", "wall_thickness", "foundation_height", "d_full", "t_full"],
        )

        self.add_subsystem("col_geom", ColumnGeometry(n_height=n_height), promotes=["*"])

        self.add_subsystem(
            "cyl_mass",
            CylinderMass(nPoints=n_full),
            promotes=["z_full", "d_full", "t_full", "labor_cost_rate", "painting_cost_rate"],
        )

        self.connect("rho_full", "cyl_mass.rho")
        self.connect("unit_cost_full", "cyl_mass.material_cost_rate")

        self.add_subsystem("bulk", BulkheadProperties(n_height=n_height, n_bulkhead=n_bulk), promotes=["*"])

        self.add_subsystem("stiff", StiffenerProperties(n_height=n_height), promotes=["*"])

        self.add_subsystem("plate", BuoyancyTankProperties(n_height=n_height), promotes=["*"])

        self.add_subsystem("ball", BallastProperties(n_height=n_height), promotes=["*"])

        self.add_subsystem("col", ColumnProperties(n_height=n_height), promotes=["*"])

        self.add_subsystem("wind", PowerWind(nPoints=n_full), promotes=["Uref", "zref", "shearExp", ("z0", "wind_z0")])
        self.add_subsystem(
            "wave",
            LinearWaves(nPoints=n_full),
            promotes=[
                "Uc",
                "Hsig_wave",
                "Tsig_wave",
                "rho_water",
                ("z_floor", "water_depth"),
                ("z_surface", "wave_z0"),
            ],
        )
        self.add_subsystem(
            "windLoads", CylinderWindDrag(nPoints=n_full), promotes=["cd_usr", "beta_wind", "rho_air", "mu_air"]
        )
        self.add_subsystem(
            "waveLoads",
            CylinderWaveDrag(nPoints=n_full),
            promotes=["cm", "cd_usr", "beta_wave", "rho_water", "mu_water"],
        )
        self.add_subsystem("distLoads", AeroHydroLoads(nPoints=n_full), promotes=["Px", "Py", "Pz", "qdyn", "yaw"])

        self.add_subsystem("buck", ColumnBuckling(n_height=n_height, modeling_options=opt), promotes=["*"])

        self.connect("outer_diameter", ["diameter", "gc.d"])
        self.connect("wall_thickness", "gc.t")
        self.connect("cyl_geom.z_param", "z_param_in")
        self.connect("cyl_geom.z_full", "z_full_in")

        # self.connect('cyl_mass.section_center_of_mass', 'col_geom.section_center_of_mass')

        self.connect("cyl_mass.mass", "shell_mass")
        self.connect("cyl_mass.cost", "shell_cost")
        self.connect("cyl_mass.I_base", "shell_I_keel")
        self.connect("cyl_mass.section_center_of_mass", "z_section")

        self.connect("column_total_mass", "section_mass")

        self.connect("z_full", ["wind.z", "wave.z", "windLoads.z", "waveLoads.z", "distLoads.z"])
        self.connect("d_full", ["windLoads.d", "waveLoads.d"])

        self.connect("wind.U", "windLoads.U")

        self.connect("wave.U", "waveLoads.U")
        self.connect("wave.A", "waveLoads.A")
        self.connect("wave.p", "waveLoads.p")

        # connections to distLoads1
        self.connect("windLoads.windLoads_Px", "distLoads.windLoads_Px")
        self.connect("windLoads.windLoads_Py", "distLoads.windLoads_Py")
        self.connect("windLoads.windLoads_Pz", "distLoads.windLoads_Pz")
        self.connect("windLoads.windLoads_qdyn", "distLoads.windLoads_qdyn")
        self.connect("windLoads.windLoads_beta", "distLoads.windLoads_beta")
        self.connect("windLoads.windLoads_z", "distLoads.windLoads_z")
        self.connect("windLoads.windLoads_d", "distLoads.windLoads_d")

        self.connect("waveLoads.waveLoads_Px", "distLoads.waveLoads_Px")
        self.connect("waveLoads.waveLoads_Py", "distLoads.waveLoads_Py")
        self.connect("waveLoads.waveLoads_Pz", "distLoads.waveLoads_Pz")
        self.connect("waveLoads.waveLoads_pt", "distLoads.waveLoads_qdyn")
        self.connect("waveLoads.waveLoads_beta", "distLoads.waveLoads_beta")
        self.connect("waveLoads.waveLoads_z", "distLoads.waveLoads_z")
        self.connect("waveLoads.waveLoads_d", "distLoads.waveLoads_d")

        self.connect("qdyn", "pressure")
