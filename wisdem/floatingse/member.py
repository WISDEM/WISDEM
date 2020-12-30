import collections

import numpy as np
import openmdao.api as om
import wisdem.commonse.frustum as frustum
import wisdem.commonse.utilities as util
import wisdem.commonse.manufacturing as manufacture
import wisdem.commonse.cross_sections as cs
from wisdem.commonse import eps, gravity
from sortedcontainers import SortedDict
from wisdem.commonse.wind_wave_drag import CylinderEnvironment
from wisdem.commonse.utilization_constraints import GeometricConstraints

CrossSection = collections.namedtuple("CrossSection", ["A", "rho", "Ixx", "Iyy", "Izz", "E", "G"])


NREFINE = 2


def get_nfull(npts):
    nFull = int(1 + NREFINE * (npts - 1))
    return nFull


def I_cyl(r_i, r_o, h, m):
    if type(r_i) == type(np.array([])):
        n = r_i.size
        r_i = r_i.flatten()
        r_o = r_o.flatten()
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
        1D array of the non-dimensional grid defined along the member axis (0-member base,
        1-member top)
    layer_materials : list of strings
        1D array of the names of the materials of each layer modeled in the member
        structure.
    layer_thickness : numpy array[n_layers_tow, n_height_tow], [m]
        2D array of the thickness of the layers of the member structure. The first
        dimension represents each layer, the second dimension represents each piecewise-
        constant entry of the member sections.
    height : float, [m]
        Scalar of the member height computed along the z axis.
    outer_diameter_in : numpy array[n_height_tow], [m]
        cylinder diameter at corresponding locations
    material_names : list of strings
        1D array of names of materials.
    E_mat : numpy array[n_mat, 3], [Pa]
        2D array of the Youngs moduli of the materials. Each row represents a material,
        the three members represent E11, E22 and E33.
    G_mat : numpy array[n_mat, 3], [Pa]
        2D array of the shear moduli of the materials. Each row represents a material,
        the three members represent G12, G13 and G23.
    sigma_y_mat : numpy array[n_mat], [Pa]
        2D array of the yield strength of the materials. Each row represents a material,
        the three members represent Xt12, Xt13 and Xt23.
    rho_mat : numpy array[n_mat], [kg/m**3]
        1D array of the density of the materials. For composites, this is the density of
        the laminate.
    unit_cost_mat : numpy array[n_mat], [USD/kg]
        1D array of the unit costs of the materials.
    outfitting_factor : float
        Multiplier that accounts for secondary structure mass inside of member

    Returns
    -------
    section_height : numpy array[n_height-1], [m]
        parameterized section heights along cylinder
    outer_diameter : numpy array[n_height], [m]
        cylinder diameter at corresponding locations
    wall_thickness : numpy array[n_height-1], [m]
        shell thickness at corresponding locations
    E : numpy array[n_height-1], [Pa]
        Isotropic Youngs modulus of the materials along the member sections.
    G : numpy array[n_height-1], [Pa]
        Isotropic shear modulus of the materials along the member sections.
    sigma_y : numpy array[n_height-1], [Pa]
        Isotropic yield strength of the materials along the member sections.
    rho : numpy array[n_height-1], [kg/m**3]
        Density of the materials along the member sections.
    unit_cost : numpy array[n_height-1], [USD/kg]
        Unit costs of the materials along the member sections.
    outfitting_factor : numpy array[n_height-1]
        Additional outfitting multiplier in each section

    """

    def initialize(self):
        self.options.declare("n_height")
        self.options.declare("n_layers")
        self.options.declare("n_mat")

    def setup(self):
        n_height = self.options["n_height"]
        n_layers = self.options["n_layers"]
        n_ballast = self.options["n_ballast"]
        n_mat = self.options["n_mat"]

        # TODO: Use reference axis and curvature, s, instead of assuming everything is vertical on z
        self.add_input("s", val=np.zeros(n_height))
        self.add_discrete_input("layer_materials", val=n_layers * [""])
        self.add_discrete_input("ballast_materials", val=n_ballast * [""])
        self.add_input("layer_thickness", val=np.zeros((n_layers, n_height)), units="m")
        self.add_input("height", val=0.0, units="m")
        self.add_input("outer_diameter_in", np.zeros(n_height), units="m")
        self.add_discrete_input("material_names", val=n_mat * [""])
        self.add_input("E_mat", val=np.zeros([n_mat, 3]), units="Pa")
        self.add_input("G_mat", val=np.zeros([n_mat, 3]), units="Pa")
        self.add_input("sigma_y_mat", val=np.zeros(n_mat), units="Pa")
        self.add_input("rho_mat", val=np.zeros(n_mat), units="kg/m**3")
        self.add_input("unit_cost_mat", val=np.zeros(n_mat), units="USD/kg")
        self.add_input("outfitting_factor_in", val=1.0)

        self.add_output("section_height", val=np.zeros(n_height - 1), units="m")
        self.add_output("outer_diameter", val=np.zeros(n_height), units="m")
        self.add_output("wall_thickness", val=np.zeros(n_height - 1), units="m")
        self.add_output("E", val=np.zeros(n_height - 1), units="Pa")
        self.add_output("G", val=np.zeros(n_height - 1), units="Pa")
        self.add_output("sigma_y", val=np.zeros(n_height - 1), units="Pa")
        self.add_output("rho", val=np.zeros(n_height - 1), units="kg/m**3")
        self.add_output("unit_cost", val=np.zeros(n_height - 1), units="USD/kg")
        self.add_output("outfitting_factor", val=np.ones(n_height - 1))
        self.add_output("ballast_density", val=np.zeros(n_ballast), units="kg/m**3")
        self.add_output("ballast_unit_cost", val=np.zeros(n_ballast), units="USD/kg")

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # Unpack dimensions
        n_height = self.options["n_height"]
        n_ballast = self.options["n_ballast"]

        # Unpack values
        h_col = inputs["height"]

        outputs["section_height"] = np.diff(h_col * inputs["s"])
        outputs["wall_thickness"] = np.sum(inputs["layer_thickness"], axis=0)
        outputs["outer_diameter"] = inputs["outer_diameter_in"]
        outputs["outfitting_factor"] = inputs["outfitting_factor_in"] * np.ones(n_height - 1)
        twall = inputs["layer_thickness"]

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
        layer_mat = discrete_inputs["layer_materials"]
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

        # Loop over materials and associate it with its thickness
        rho_ballast = np.zeros(n_ballast)
        cost_ballast = np.zeros(n_ballast)
        ballast_mat = discrete_inputs["ballast_materials"]
        for k in range(n_ballast):
            # Get the material name for this layer
            iname = ballast_mat[k]

            # Get the index into the material list
            imat = mat_names.index(iname)

            # Store values
            rho_ballast[k] = rho[imat]
            cost_ballast[k] = cost[imat]

        outputs["ballast_density"] = rho_ballast
        outputs["ballast_unit_cost"] = cost_ballast


class MemberDiscretization(om.ExplicitComponent):
    """
    Discretize geometry into finite element nodes

    Parameters
    ----------
    s : numpy array[n_height_tow]
        1D array of the non-dimensional grid defined along the member axis (0-member base,
        1-member top)
    outer_diameter : numpy array[n_height], [m]
        cylinder diameter at corresponding locations
    wall_thickness : numpy array[n_height-1], [m]
        shell thickness at corresponding locations
    E : numpy array[n_height-1], [Pa]
        Isotropic Youngs modulus of the materials along the member sections.
    G : numpy array[n_height-1], [Pa]
        Isotropic shear modulus of the materials along the member sections.
    sigma_y : numpy array[n_height-1], [Pa]
        Isotropic yield strength of the materials along the member sections.
    rho : numpy array[n_height-1], [kg/m**3]
        Density of the materials along the member sections.
    unit_cost : numpy array[n_height-1], [USD/kg]
        Unit costs of the materials along the member sections.
    outfitting_factor : numpy array[n_height-1]
        Additional outfitting multiplier in each section

    Returns
    -------
    s_full : numpy array[nFull]
        non-dimensional locations along member
    z_full : numpy array[nFull], [m]
        dimensional locations along member axis
    d_full : numpy array[nFull], [m]
        cylinder diameter at corresponding locations
    t_full : numpy array[nFull-1], [m]
        shell thickness at corresponding locations
    E_full : numpy array[nFull-1], [Pa]
        Isotropic Youngs modulus of the materials along the member sections.
    G_full : numpy array[nFull-1], [Pa]
        Isotropic shear modulus of the materials along the member sections.
    sigma_y_full : numpy array[nFull-1], [Pa]
        Isotropic yield strength of the materials along the member sections.
    rho_full : numpy array[nFull-1], [kg/m**3]
        Density of the materials along the member sections.
    unit_cost_full : numpy array[nFull-1], [USD/kg]
        Unit costs of the materials along the member sections.
    nu_full : numpy array[nFull-1]
        Poisson's ratio assuming isotropic material
    outfitting_full : numpy array[nFull-1]
        Additional outfitting multiplier in each section

    """

    """discretize geometry into finite element nodes"""

    def initialize(self):
        self.options.declare("n_height")
        self.options.declare("nRefine", default=NREFINE)

    def setup(self):
        n_height = self.options["n_height"]
        nFull = get_nfull(n_height)

        self.add_input("s", val=np.zeros(n_height))
        self.add_input("outer_diameter", np.zeros(n_height), units="m")
        self.add_input("wall_thickness", np.zeros(n_height - 1), units="m")
        self.add_input("E", val=np.zeros(n_height - 1), units="Pa")
        self.add_input("G", val=np.zeros(n_height - 1), units="Pa")
        self.add_input("sigma_y", val=np.zeros(n_height - 1), units="Pa")
        self.add_input("rho", val=np.zeros(n_height - 1), units="kg/m**3")
        self.add_input("unit_cost", val=np.zeros(n_height - 1), units="USD/kg")
        self.add_input("outfitting_factor", val=np.ones(n_height - 1))

        self.add_output("s_full", np.zeros(nFull), units="m")
        self.add_output("z_full", np.zeros(nFull), units="m")
        self.add_output("d_full", np.zeros(nFull), units="m")
        self.add_output("t_full", np.zeros(nFull - 1), units="m")
        self.add_output("E_full", val=np.zeros(nFull - 1), units="Pa")
        self.add_output("G_full", val=np.zeros(nFull - 1), units="Pa")
        self.add_output("nu_full", val=np.zeros(nFull - 1))
        self.add_output("sigma_y_full", val=np.zeros(nFull - 1), units="Pa")
        self.add_output("rho_full", val=np.zeros(nFull - 1), units="kg/m**3")
        self.add_output("unit_cost_full", val=np.zeros(nFull - 1), units="USD/kg")
        self.add_output("outfitting_full", val=np.ones(nFull - 1))

        # self.declare_partials('*', '*', method='fd', form='central', step=1e-6)

    def compute(self, inputs, outputs):
        # Unpack inputs
        s_param = inputs["s"]
        nRefine = int(np.round(self.options["nRefine"]))

        # TODO: Put these somewhere
        # Create constraint output that draft is less than water depth
        # outputs["draft_margin"] = draft / inputs["max_draft"]

        # Make sure freeboard is more than 20% of Hsig_wave (DNV-OS-J101)
        # outputs["wave_height_freeboard_ratio"] = inputs["Hsig_wave"] / (np.abs(freeboard) + eps)

        # Have to regine each element one at a time so that we preserve input nodes
        s_full = np.array([])
        for k in range(s_param.size - 1):
            sref = np.linspace(s_param[k], s_param[k + 1], nRefine + 1)
            s_full = np.append(s_full, sref)
        s_full = np.unique(s_full)
        s_section = 0.5 * (s_full[:-1] + s_full[1:])

        # Assuming straight (non-curved) members, set dimensional z along the axis
        outputs["s_full"] = s_full
        outputs["z_full"] = s_full * inputs["height"]

        # All other parameters
        outputs["d_full"] = np.interp(s_full, s_param, inputs["outer_diameter"])
        outputs["t_full"] = util.sectionalInterp(s_section, s_param, inputs["wall_thickness"])
        outputs["rho_full"] = util.sectionalInterp(s_section, s_param, inputs["rho"])
        outputs["E_full"] = util.sectionalInterp(s_section, s_param, inputs["E"])
        outputs["G_full"] = util.sectionalInterp(s_section, s_param, inputs["G"])
        outputs["sigma_y_full"] = util.sectionalInterp(s_section, s_param, inputs["sigma_y"])
        outputs["unit_cost_full"] = util.sectionalInterp(s_section, s_param, inputs["unit_cost"])
        outputs["outfitting_full"] = util.sectionalInterp(s_section, s_param, inputs["outfitting_factor"])
        outputs["nu_full"] = 0.5 * outputs["E_full"] / outputs["G_full"] - 1.0


class MemberComponent(om.ExplicitComponent):
    """
    Convert the YAML inputs into more native and easy to use variables.

    Parameters
    ----------
    joint0 : numpy array[3], [m]
        Global dimensional coordinates (x-y-z) for bottom node of member
    joint1 : numpy array[3], [m]
        Global dimensional coordinates (x-y-z) for top node of member
    grid_axial_joints : numpy array[n_axial]
        non-dimensional locations along member for named axial joints
    height : float, [m]
        Scalar of the member height computed along the z axis.
    s_full : numpy array[nFull]
        non-dimensional locations along member
    z_full : numpy array[nFull], [m]
        dimensional locations along member axis
    d_full : numpy array[nFull], [m]
        cylinder diameter at corresponding locations
    t_full : numpy array[nFull-1], [m]
        shell thickness at corresponding locations
    E_full : numpy array[nFull-1], [Pa]
        Isotropic Youngs modulus of the materials along the member sections.
    G_full : numpy array[nFull-1], [Pa]
        Isotropic shear modulus of the materials along the member sections.
    rho_full : numpy array[nFull-1], [kg/m**3]
        Density of the materials along the member sections.
    unit_cost_full : numpy array[nFull-1], [USD/kg]
        Unit costs of the materials along the member sections.
    outfitting_full : numpy array[nFull-1]
        Additional outfitting multiplier in each section
    labor_cost_rate : float, [USD/min]
        Labor cost rate
    painting_cost_rate : float, [USD/m/m]
        Painting / surface finishing cost rate
    bulkhead_grid : numpy array[n_bulk]
        Non-dimensional locations of the bulkheads
    bulkhead_thickness : numpy array[n_bulk], [m]
        Thickness of the bulkheads at the gridded locations
    ring_stiffener_web_height : float, [m]
        height of stiffener web (base of T) within each section bottom to top
        (length = nsection)
    ring_stiffener_web_thickness : float, [m]
        thickness of stiffener web (base of T) within each section bottom to top
        (length = nsection)
    ring_stiffener_flange_width : float, [m]
        height of stiffener flange (top of T) within each section bottom to top
        (length = nsection)
    ring_stiffener_flange_thickness : float, [m]
        thickness of stiffener flange (top of T) within each section bottom to top
        (length = nsection)
    ring_stiffener_spacing : float, [m]
        Axial distance from one ring stiffener to another within each section bottom to
        top (length = nsection)
    ballast_grid : numpy array[n_ballast,2]
        Non-dimensional start and end points for each ballast segment
    ballast_density : numpy array[n_ballast], [kg/m**3]
        density of ballast material
    ballast_volume : numpy array[n_ballast], [m**3]
        Volume of ballast segments.  Should be non-zero for permanent ballast, zero for variable ballast
    ballast_unit_cost : numpy array[n_ballast], [USD/kg]
        Cost per unit mass of ballast

    Returns
    -------
    shell_cost : float, [USD]
        Outer shell cost
    shell_mass : float, [kg]
        Outer shell mass
    shell_z_cg : float, [m]
        z-position of center of mass of member shell
    shell_I_base : numpy array[6], [kg*m**2]
        mass moment of inertia of shell about base [xx yy zz xy xz yz]
    bulkhead_mass : float, [kg]
        mass of column bulkheads
    bulkhead_z_cg : float, [m]
        z-coordinate of center of gravity for all bulkheads
    bulkhead_cost : float, [USD]
        cost of column bulkheads
    bulkhead_I_base : numpy array[6], [kg*m**2]
        Moments of inertia of bulkheads relative to keel point
    stiffener_mass : float, [kg]
        mass of column stiffeners
    stiffener_cost : float, [USD]
        cost of column stiffeners
    stiffener_z_cg : float, [m]
        z-coordinate of center of gravity for all ring stiffeners
    stiffener_I_base : numpy array[6], [kg*m**2]
        Moments of inertia of stiffeners relative to base point
    flange_spacing_ratio : numpy array[n_full-1]
        ratio between flange and stiffener spacing
    stiffener_radius_ratio : numpy array[n_full-1]
        ratio between stiffener height and radius
    ballast_cost : float, [USD]
        cost of permanent ballast
    ballast_mass : float, [kg]
        mass of permanent ballast
    ballast_z_cg : float, [m]
        z-coordinate or permanent ballast center of gravity
    ballast_I_base : numpy array[6], [kg*m**2]
        Moments of inertia of permanent ballast relative to bottom point
    variable_ballast_capacity : float, [m]
        inner radius of column at potential ballast mass
    constr_ballast_capacity : numpy array[n_ballast]
        Used ballast volume relative to total capacity, must be <= 1.0
    s_all : numpy array[npts]
        Final non-dimensional points of all internal member nodes
    center_of_mass : numpy array[3], [m]
        Global dimensional coordinates (x-y-z) for member center of mass / gravity
    nodes_xyz : numpy array[npts,3], [m]
        Global dimensional coordinates (x-y-z) for all member nodes in s_all output
    section_area : numpy array[npts-1], [m**2]
        Cross-sectional area of all member segments
    section_Ixx : numpy array[npts-1], [kg*m**2]
        Cross-sectional moment of inertia about x-axis in member c.s. of all member segments
    section_Iyy : numpy array[npts-1], [kg*m**2]
        Cross-sectional moment of inertia about y-axis in member c.s. of all member segments
    section_Izz : numpy array[npts-1], [kg*m**2]
        Cross-sectional moment of inertia about z-axis in member c.s. of all member segments
    section_rho : numpy array[npts-1], [kg/m**3]
        Cross-sectional density of all member segments
    section_E : numpy array[npts-1], [Pa]
        Cross-sectional Young's modulus (of elasticity) of all member segments
    section_G : numpy array[npts-1], [Pa]
        Cross-sectional shear modulus all member segments

    """

    def initialize(self):
        self.options.declare("member_options")

    def setup(self):
        modopt = self.options["member_options"]
        n_height = modopt["n_height"]
        n_full = get_nfull(n_height)
        n_axial = modopt["n_axial"]
        n_bulk = modopt["n_bulkhead"]
        n_ball = modopt["n_ballast"]
        n_ring = modopt["n_ring"]
        # 2 points added for bulkheads and stiffeners
        # Bulkheads at 0,1 only get 1 new point
        # No separate members for ballast
        npts = n_full + n_axial + 2 * n_bulk - 2 + 2 * n_ring

        # Initialize dictionary that will keep our member nodes so we can convert to OpenFAST format
        self.sections = SortedDict()

        # Inputs
        self.add_input("joint0", val=np.zeros(3), units="m")
        self.add_input("joint1", val=np.zeros(3), units="m")
        self.add_input("height", val=0.0, units="m")
        self.add_input("s_full", np.zeros(n_full), units="m")
        self.add_input("z_full", np.zeros(n_full), units="m")
        self.add_input("d_full", np.zeros(n_full), units="m")
        self.add_input("t_full", np.zeros(n_full - 1), units="m")
        self.add_input("E_full", val=np.zeros(n_full - 1), units="Pa")
        self.add_input("G_full", val=np.zeros(n_full - 1), units="Pa")
        self.add_input("rho_full", val=np.zeros(n_full - 1), units="kg/m**3")
        self.add_input("unit_cost_full", val=np.zeros(n_full - 1), units="USD/kg")
        self.add_input("outfitting_full", val=np.ones(n_full - 1))
        self.add_input("labor_cost_rate", 0.0, units="USD/min")
        self.add_input("painting_cost_rate", 0.0, units="USD/m/m")

        self.add_input("grid_axial_joints", np.zeros(n_axial))

        self.add_input("bulkhead_grid", np.zeros(n_bulk))
        self.add_input("bulkhead_thickness", np.zeros(n_bulk), units="m")

        self.add_input("ring_stiffener_web_height", 0.0, units="m")
        self.add_input("ring_stiffener_web_thickness", 0.0, units="m")
        self.add_input("ring_stiffener_flange_width", 0.0, units="m")
        self.add_input("ring_stiffener_flange_thickness", 0.0, units="m")
        self.add_input("ring_stiffener_spacing", 0.0, units="m")

        self.add_input("ballast_grid", np.zeros((n_ball, 2)))
        self.add_input("ballast_density", np.zeros(n_ball), units="kg/m**3")
        self.add_input("ballast_volume", np.zeros(n_ball), units="m**3")
        self.add_input("ballast_unit_cost", np.zeros(n_ball), units="USD/kg")

        # Outputs
        self.add_output("shell_cost", val=0.0, units="USD")
        self.add_output("shell_mass", val=0.0, units="kg")
        self.add_output("shell_z_cg", val=0.0, units="m")
        self.add_output("shell_I_base", np.zeros(6), units="kg*m**2")

        self.add_output("bulkhead_mass", 0.0, units="kg")
        self.add_output("bulkhead_z_cg", 0.0, units="m")
        self.add_output("bulkhead_cost", 0.0, units="USD")
        self.add_output("bulkhead_I_base", np.zeros(6), units="kg*m**2")

        self.add_output("stiffener_mass", np.zeros(n_full - 1), units="kg")
        self.add_output("stiffener_z_cg", 0.0, units="m")
        self.add_output("stiffener_cost", 0.0, units="USD")
        self.add_output("stiffener_I_base", np.zeros(6), units="kg*m**2")
        self.add_output("flange_spacing_ratio", np.zeros(n_ring))
        self.add_output("stiffener_radius_ratio", np.zeros(n_ring))

        self.add_output("ballast_cost", 0.0, units="USD")
        self.add_output("ballast_mass", 0.0, units="kg")
        self.add_output("ballast_z_cg", 0.0, units="m")
        self.add_output("ballast_I_base", np.zeros(6), units="kg*m**2")
        self.add_output("variable_ballast_capacity", 0.0, units="m")
        self.add_output("constr_ballast_capacity", np.zeros(n_ball), units="m")

        self.add_output("s_all", np.zeros(npts), units="m**2")
        self.add_output("center_of_mass", np.zeros(3), units="m")
        self.add_output("nodes_xyz", np.zeros((npts, 3)), units="m**2")
        self.add_output("section_area", np.zeros(npts - 1), units="m**2")
        self.add_output("section_Ixx", np.zeros(npts - 1), units="kg*m**2")
        self.add_output("section_Iyy", np.zeros(npts - 1), units="kg*m**2")
        self.add_output("section_Izz", np.zeros(npts - 1), units="kg*m**2")
        self.add_output("section_rho", np.zeros(npts - 1), units="kg/m**3")
        self.add_output("section_E", np.zeros(npts - 1), units="Pa")
        self.add_output("section_G", np.zeros(npts - 1), units="Pa")

    def add_node(self, s_new):
        # Quit if node already exists
        if s_new in self.sections:
            return

        # Find section we will be interrupting
        idx = self.sections.bisect_left(s_new) - 1
        if idx < 0:
            raise ValueError("Cannot insert node before start of list")

        keys_orig = self.sections.keys()
        self.sections[s_new] = self.sections[keys_orig[idx]]

    def insert_section(self, s0, s1, cross_section0):

        idx0 = self.sections.bisect_left(s0)
        idx1 = self.sections.bisect_left(s1)
        keys_orig = self.sections.keys()

        # Be sure to add new node with old section before adding new section
        self.add_node(s1)

        # If we are straddling an old point, have to convert that to the new section
        if idx0 != idx1:
            self.sections[keys_orig[idx0]] = cross_section0

        # Add new section
        self.sections[s0] = cross_section0

    def add_section(self, s0, s1, cross_section0):
        self.sections[s0] = cross_section0
        self.sections[s1] = None

    def compute(self, inputs, outputs):

        self.add_main_sections(inputs, outputs)
        self.add_bulkhead_sections(inputs, outputs)
        self.add_ring_stiffener_sections(inputs, outputs)
        self.add_ballast_sections(inputs, outputs)
        self.compute_mass_properties(inputs, outputs)
        self.nodal_discretization(inputs, outputs)

    def add_main_sections(self, inputs, outputs):
        # Unpack inputs
        s_full = inputs["s_full"]
        t_full = inputs["t_full"]
        d_full = inputs["d_full"]
        zz = inputs["z_full"]
        Rb = 0.5 * d_full[:-1]
        Rt = 0.5 * d_full[1:]
        H = np.diff(zz)
        rho = inputs["rho_full"]
        Emat = inputs["E_full"]
        Gmat = inputs["G_full"]
        coeff = inputs["outfitting_full"]

        # Add sections for structural analysis
        # TODO: Longitudinal stiffeners
        d_sec, _ = util.nodal2sectional(d_full)
        for k in range(len(s_full) - 1):
            itube = cs.Tube(d_sec[k], t_full[k])
            iprop = CrossSection(
                A=coeff[k] * itube.Area,
                Ixx=coeff[k] * itube.Jxx,
                Iyy=coeff[k] * itube.Jyy,
                Izz=coeff[k] * itube.J0,
                rho=rho[k],
                E=Emat[k],
                G=Gmat[k],
            )
            self.add_section(s_full[k], s_full[k + 1], iprop)

        # Total mass of cylinder
        V_shell = frustum.frustumShellVol(Rb, Rt, t_full, H)
        mass = coeff * rho * V_shell
        outputs["shell_mass"] = mass.sum()

        # Center of mass
        cm_section = zz[:-1] + frustum.frustumShellCG(Rb, Rt, t_full, H)
        outputs["shell_z_cg"] = np.dot(cm_section, mass) / mass.sum()

        # Moments of inertia
        Izz_section = coeff * rho * frustum.frustumShellIzz(Rb, Rt, t_full, H)
        Ixx_section = Iyy_section = coeff * rho * frustum.frustumShellIxx(Rb, Rt, t_full, H)

        # Sum up each cylinder section using parallel axis theorem
        I_base = np.zeros((3, 3))
        for k in range(Izz_section.size):
            R = np.array([0.0, 0.0, cm_section[k] - zz[0]])
            Icg = util.assembleI([Ixx_section[k], Iyy_section[k], Izz_section[k], 0.0, 0.0, 0.0])

            I_base += Icg + mass[k] * (np.dot(R, R) * np.eye(3) - np.outer(R, R))

        outputs["shell_I_base"] = util.unassembleI(I_base)

        # Compute costs based on "Optimum Design of Steel Structures" by Farkas and Jarmai
        # All dimensions for correlations based on mm, not meters.
        R_ave = 0.5 * (Rb + Rt)
        taper = np.minimum(Rb / Rt, Rt / Rb)
        nsec = t_full.size
        mshell = rho * V_shell
        mshell_tot = np.sum(rho * V_shell)
        k_m = inputs["unit_cost_full"]  # 1.1 # USD / kg carbon steel plate
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
            manufacture.steel_cutting_plasma_time(cutLengths, t_full)
            + manufacture.steel_rolling_time(theta_F, R_ave, t_full)
            + manufacture.steel_butt_welding_time(theta_A, nsec, mshell_tot, cutLengths, t_full)
            + manufacture.steel_butt_welding_time(theta_A, nsec, mshell_tot, 2 * np.pi * Rb[1:], t_full[1:])
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

        outputs["shell_cost"] = tempSum + K_c

    def add_bulkhead_sections(self, inputs, outputs):
        # Unpack variables
        s_full = inputs["s_full"]
        z_full = inputs["z_full"]
        R_od = 0.5 * inputs["d_full"]
        twall = inputs["t_full"]
        rho = inputs["rho_full"]
        E = inputs["E_full"]
        G = inputs["G_full"]
        s_bulk = inputs["bulkhead_grid"]
        t_bulk = inputs["bulkhead_thickness"]
        coeff = inputs["outfitting_full"]
        nbulk = s_bulk.size
        L = inputs["height"]

        # Get z and R_id values of bulkhead locations
        z_bulk = np.interp(s_bulk, s_full, z_full)
        twall_bulk = util.sectionalInterp(s_bulk, s_full, twall)
        rho_bulk = util.sectionalInterp(s_bulk, s_full, rho)
        E_bulk = util.sectionalInterp(s_bulk, s_full, E)
        G_bulk = util.sectionalInterp(s_bulk, s_full, G)
        coeff_bulk = util.sectionalInterp(s_bulk, s_full, coeff)
        R_od_bulk = np.interp(s_bulk, s_full, R_od)
        R_id_bulk = R_od_bulk - twall_bulk

        # Add bulkhead sections: assumes bulkhead and shell are made of same material!
        s0 = s_bulk - 0.5 * t_bulk / L
        s1 = s_bulk + 0.5 * t_bulk / L
        if s0[0] < 0.0:
            s0[0] = 0.0
            s1[0] = t_bulk[0] / L
        if s1[-1] > 1.0:
            s0[-1] = 1 - t_bulk[-1] / L
            s1[-1] = 1.0
        for k in range(nbulk):
            itube = cs.Tube(2 * R_od_bulk[k], R_od_bulk[k])  # thickness=radius for solid disk
            iprop = CrossSection(
                A=coeff_bulk[k] * itube.Area,
                Ixx=coeff_bulk[k] * itube.Jxx,
                Iyy=coeff_bulk[k] * itube.Jyy,
                Izz=coeff_bulk[k] * itube.J0,
                rho=rho_bulk[k],
                E=E_bulk[k],
                G=G_bulk[k],
            )
            self.insert_section(s0[k], s1[k], iprop)

        # Compute bulkhead mass independent of shell
        m_bulk = coeff_bulk * rho_bulk * np.pi * R_id_bulk ** 2 * t_bulk
        outputs["bulkhead_mass"] = m_bulk.sum()

        z_cg = 0.0 if outputs["bulkhead_mass"] == 0.0 else np.dot(z_bulk, m_bulk) / m_bulk.sum()
        outputs["bulkhead_z_cg"] = z_cg

        # Compute moments of inertia at keel
        # Assume bulkheads are just simple thin discs with radius R_od-t_wall and mass already computed
        Izz = 0.5 * m_bulk * R_id_bulk ** 2
        Ixx = Iyy = 0.5 * Izz
        dz = z_bulk - z_full[0]
        I_keel = np.zeros((3, 3))
        for k in range(nbulk):
            R = np.array([0.0, 0.0, dz[k]])
            Icg = util.assembleI([Ixx[k], Iyy[k], Izz[k], 0.0, 0.0, 0.0])
            I_keel += Icg + m_bulk[k] * (np.dot(R, R) * np.eye(3) - np.outer(R, R))

        outputs["bulkhead_I_base"] = util.unassembleI(I_keel)

        # Compute costs based on "Optimum Design of Steel Structures" by Farkas and Jarmai
        # All dimensions for correlations based on mm, not meters.
        k_m = util.sectionalInterp(s_bulk, s_full, inputs["unit_cost_full"])
        k_f = inputs["labor_cost_rate"]  # 1.0 # USD / min labor
        k_p = inputs["painting_cost_rate"]  # USD / m^2 painting
        m_shell = outputs["shell_mass"]

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
        outputs["bulkhead_cost"] = c_bulk

    def add_ring_stiffener_sections(self, inputs, outputs):
        # Unpack variables
        s_full = inputs["s_full"]
        z_full = inputs["z_full"]
        L = inputs["height"]
        R_od = 0.5 * inputs["d_full"]
        twall = inputs["t_full"]
        rho = inputs["rho_full"]
        E = inputs["E_full"]
        G = inputs["G_full"]
        coeff = inputs["outfitting_full"]

        t_web = inputs["ring_stiffener_web_thickness"]
        t_flange = inputs["ring_stiffener_flange_thickness"]
        h_web = inputs["ring_stiffener_web_height"]
        w_flange = inputs["ring_stiffener_flange_width"]
        L_stiffener = inputs["ring_stiffener_spacing"]
        web_frac = t_web / w_flange

        n_stiff = int(np.floor(L / L_stiffener))
        if n_stiff == 0:
            return

        # Calculate stiffener spots along the member axis
        s_stiff = (np.arange(1, n_stiff + 0.1) - 0.5) * (L_stiffener / L)
        s0 = s_stiff - 0.5 * w_flange / L
        s1 = s_stiff + 0.5 * w_flange / L
        if s0[0] < 0.0:
            s0[0] = 0.0
            s1[0] = w_flange / L
        if s1[-1] > 1.0:
            s0[-1] = 1 - w_flange / L
            s1[-1] = 1.0

        # Get z and R_id values of bulkhead locations
        z_stiff = np.interp(s_stiff, s_full, z_full)
        twall_stiff = util.sectionalInterp(s_stiff, s_full, twall)
        rho_stiff = util.sectionalInterp(s_stiff, s_full, rho)
        E_stiff = util.sectionalInterp(s_stiff, s_full, E)
        G_stiff = util.sectionalInterp(s_stiff, s_full, G)
        coeff_stiff = util.sectionalInterp(s_stiff, s_full, coeff)
        R_od_stiff = np.interp(s_stiff, s_full, R_od)
        R_id_stiff = R_od_stiff - twall_stiff

        # Create some constraints for reasonable stiffener designs for an optimizer
        outputs["flange_spacing_ratio"] = w_flange / (0.5 * L_stiffener)
        outputs["stiffener_radius_ratio"] = (h_web + t_flange + twall_stiff) / R_od_stiff

        # Outer and inner radius of web by section
        R_wo = R_id_stiff
        R_wi = R_wo - h_web
        # Outer and inner radius of flange by section
        R_fo = R_wi
        R_fi = R_fo - t_flange

        # Make stiffener sections
        for k in range(n_stiff):
            ishell = cs.Tube(2 * R_od_stiff[k], twall_stiff[k])
            iweb = cs.Tube(2 * R_wo[k], h_web)
            iflange = cs.Tube(2 * R_fo[k], t_flange)
            iprop = CrossSection(
                A=coeff_stiff[k] * ishell.Area + iflange.Area + iweb.Area * web_frac,
                Ixx=coeff_stiff[k] * ishell.Jxx + iflange.Jxx + iweb.Jxx * web_frac,
                Iyy=coeff_stiff[k] * ishell.Jyy + iflange.Jyy + iweb.Jyy * web_frac,
                Izz=coeff_stiff[k] * ishell.J0 + iflange.J0 + iweb.J0 * web_frac,
                rho=rho_stiff[k],
                E=E_stiff[k],
                G=G_stiff[k],
            )
            self.insert_section(s0[k], s1[k], iprop)

        # Material volumes by section
        V_web = np.pi * (R_wo ** 2 - R_wi ** 2) * t_web
        V_flange = np.pi * (R_fo ** 2 - R_fi ** 2) * w_flange

        # Ring mass by volume by section
        m_web = rho_stiff * V_web
        m_flange = rho_stiff * V_flange
        m_ring = m_web + m_flange
        outputs["stiffener_mass"] = m_ring.sum()
        outputs["stiffener_z_cg"] = np.dot(z_stiff, m_ring) / m_ring.sum()

        # Compute moments of inertia for stiffeners (lumped by section for simplicity) at keel
        I_web = I_cyl(R_wi, R_wo, t_web, m_web)
        I_flange = I_cyl(R_fi, R_fo, w_flange, m_flange)
        I_keel = np.zeros((3, 3))
        for k in range(n_stiff):
            R = np.array([0.0, 0.0, (z_stiff[k] - z_full[0])])
            I_ring = util.assembleI(I_web[k, :] + I_flange[k, :])
            I_keel += I_ring + m_ring[k] * (np.dot(R, R) * np.eye(3) - np.outer(R, R))
        outputs["stiffener_I_base"] = util.unassembleI(I_keel)

        # Compute costs based on "Optimum Design of Steel Structures" by Farkas and Jarmai
        # All dimensions for correlations based on mm, not meters.
        k_m = util.sectionalInterp(s_stiff, s_full, inputs["unit_cost_full"])
        k_f = inputs["labor_cost_rate"]  # 1.0 # USD / min labor
        k_p = inputs["painting_cost_rate"]  # USD / m^2 painting
        m_shell = outputs["shell_mass"]

        # Cost Step 1) Cutting stiffener strips from flat plates using plasma cutter
        cutLengths_w = 2.0 * np.pi * 0.5 * (R_wo + R_wi)
        cutLengths_f = 2.0 * np.pi * R_fo
        # Cost Step 2) Welding T-stiffeners together GMAW-C (gas metal arc welding with CO2) fillet welds
        theta_w = 3.0  # Difficulty factor
        # Cost Step 3) Welding stiffeners to shell GMAW-C (gas metal arc welding with CO2) fillet welds
        # Will likely fillet weld twice (top & bottom), so factor of 2 on second welding terms

        # Labor-based expenses
        K_f = k_f * (
            manufacture.steel_cutting_plasma_time(cutLengths_w.sum(), t_web)
            + manufacture.steel_cutting_plasma_time(cutLengths_f.sum(), t_flange)
            + manufacture.steel_filett_welding_time(theta_w, 1, m_ring, 2 * np.pi * R_fo, t_web)
            + manufacture.steel_filett_welding_time(theta_w, 1, m_ring + m_shell, 2 * np.pi * R_wo, t_web)
        )

        # Cost Step 4) Painting
        theta_p = 2.0
        K_p = (
            k_p
            * theta_p
            * 2
            * np.pi
            * np.sum(R_wo ** 2.0 - R_wi ** 2.0 + 0.5 * (R_fo + R_fi) * (2 * w_flange + 2 * t_flange) - R_fo * t_web)
        )

        # Material cost, without outfitting
        K_m = np.sum(k_m * m_ring)

        # Total cost
        c_ring = K_m + K_f + K_p
        outputs["stiffener_cost"] = c_ring

    def add_ballast_sections(self, inputs, outputs):
        # Unpack variables
        s_full = inputs["s_full"]
        z_full = inputs["z_full"]
        R_od = 0.5 * inputs["d_full"]
        twall = inputs["t_full"]
        s_ballast = inputs["ballast_grid"]
        rho_ballast = inputs["ballast_density"]
        V_ballast = inputs["ballast_volume"]
        km_ballast = inputs["ballast_unit_cost"]
        n_ballast = len(V_ballast)
        if n_ballast == 0:
            return

        # Number of points for volume integration
        npts = 10

        I_ballast = np.zeros(6)
        z_cg = np.zeros(n_ballast)
        V_avail = np.zeros(n_ballast)
        for k in range(n_ballast):
            # Find geometry of the ballast space
            sinterp = np.linspace(s_ballast[k, 0], s_ballast[k, 1], npts)
            zpts = np.interp(sinterp, s_full, z_full)
            R_od_pts = np.interp(sinterp, s_full, R_od)
            twall_pts = util.sectionalInterp(sinterp, s_full, twall)
            R_id_pts = R_od_pts - twall_pts

            # Available volume in this ballast space
            V_pts = frustum.frustumVol(R_id_pts[:-1], R_id_pts[1:], np.diff(zpts))
            V_avail[k] = V_pts.sum()

            # If permanent ballast, compute mass properties, but have to find where ballast extends to in cavity
            if V_ballast[k] > 0.0:
                z_end = np.interp(V_ballast[k], np.cumsum(np.r_[0, V_pts]), zpts)
                zpts = np.linspace(zpts[0], z_end, npts)
                H = np.diff(zpts)

                R_od_pts = np.interp(zpts, z_full, R_od)
                twall_pts = util.sectionalInterp(zpts, z_full, twall)
                R_id_pts = R_od_pts - twall_pts

                V_pts = frustum.frustumVol(R_id_pts[:-1], R_id_pts[1:], H)
                cg_pts = frustum.frustumCG(R_id_pts[:-1], R_id_pts[1:], H) + zpts[:-1]
                z_cg[k] = np.dot(cg_pts, V_pts) / V_pts.sum()

                Ixx = Iyy = frustum.frustumIxx(R_id_pts[:-1], R_id_pts[1:], H)
                Izz = frustum.frustumIzz(R_id_pts[:-1], R_id_pts[1:], H)
                I_temp = np.zeros((3, 3))
                for ii in range(npts - 1):
                    R = np.array([0.0, 0.0, cg_pts[ii]])
                    Icg = util.assembleI([Ixx[ii], Iyy[ii], Izz[ii], 0.0, 0.0, 0.0])
                    I_temp += Icg + V_pts[ii] * (np.dot(R, R) * np.eye(3) - np.outer(R, R))
                I_ballast += rho_ballast[k] * util.unassembleI(I_temp)
            else:
                outputs["variable_ballast_capacity"] = V_avail[k]

        # Save permanent ballast mass and variable height
        m_ballast = rho_ballast * V_ballast
        outputs["ballast_mass"] = m_ballast.sum()
        outputs["ballast_I_base"] = I_ballast
        outputs["ballast_z_cg"] = np.dot(z_cg, m_ballast) / m_ballast.sum()
        outputs["ballast_cost"] = np.dot(km_ballast, m_ballast)
        outputs["constr_ballast_capacity"] = V_ballast / V_avail

    def compute_mass_properties(self, inputs, outputs):
        # Unpack variables
        z_full = inputs["z_full"]

        z_shell = outputs["shell_z_cg"]
        z_ballast = outputs["ballast_z_cg"]
        z_bulkhead = outputs["bulkhead_z_cg"]
        z_stiffener = outputs["stiffener_z_cg"]

        m_shell = outputs["shell_mass"]
        m_ballast = outputs["ballast_mass"]
        m_bulkhead = outputs["bulkhead_mass"]
        m_stiffener = outputs["stiffener_mass"]

        c_shell = outputs["shell_cost"]
        c_ballast = outputs["ballast_cost"]
        c_bulkhead = outputs["bulkhead_cost"]
        c_stiffener = outputs["stiffener_cost"]

        I_shell = outputs["shell_I_base"]
        I_ballast = outputs["ballast_I_base"]
        I_bulkhead = outputs["bulkhead_I_base"]
        I_stiffener = outputs["stiffener_I_base"]

        # Find mass of all of the sub-components of the member
        m_total = m_shell + m_ballast + m_bulkhead + m_stiffener
        c_total = c_shell + c_ballast + c_bulkhead + c_stiffener

        # Masses assumed to be focused at section centroids
        z_cg = (
            m_shell * z_shell + m_ballast * z_ballast + m_bulkhead * z_bulkhead + m_stiffener * z_stiffener
        ) / m_total

        # Add up moments of inertia at keel, make sure to scale mass appropriately
        I_total = I_shell + I_ballast + I_bulkhead + I_stiffener

        # Move moments of inertia from keel to cg
        I_total -= m_total * ((z_cg - z_full[0]) ** 2.0) * np.r_[1.0, 1.0, np.zeros(4)]

        # Store outputs addressed so far
        outputs["total_mass"] = m_total
        outputs["structural_mass"] = m_total - m_ballast
        outputs["z_cg"] = z_cg
        outputs["I_total"] = I_total
        outputs["total_cost"] = c_total
        outputs["structural_cost"] = c_total - c_ballast
        outputs["cost_rate"] = c_total / m_total

    def nodal_discretization(self, inputs, outputs):
        # Unpack inputs
        z_full = inputs["z_full"]
        s_axial = inputs["grid_axial_joints"]
        xyz0 = inputs["joint0"]
        xyz1 = inputs["joint1"]
        dxyz = xyz1 - xyz0

        # Add in axial nodes
        for s in s_axial:
            self.add_node(s)

        # Convert non-dimensional to dimensional
        s_grid = self.sections.keys()
        n_nodes = len(s_grid)
        nodes = np.outer(s_grid, dxyz) + xyz0[np.newaxis, :]

        # Convert axial to absolute
        outputs["center_of_mass"] = (outputs["z_cg"] / z_full[-1]) * dxyz + xyz0

        # Store all nodes and sections
        outputs["s_all"] = np.array(list(s_grid))
        outputs["nodes_xyz"] = nodes

        outputs["section_area"] = np.zeros(n_nodes - 1)
        outputs["section_rho"] = np.zeros(n_nodes - 1)
        outputs["section_Ixx"] = np.zeros(n_nodes - 1)
        outputs["section_Iyy"] = np.zeros(n_nodes - 1)
        outputs["section_Izz"] = np.zeros(n_nodes - 1)
        outputs["section_E"] = np.zeros(n_nodes - 1)
        outputs["section_G"] = np.zeros(n_nodes - 1)
        for k, s in enumerate(s_grid):
            if s == s_grid[-1]:
                continue
            outputs["section_area"][k] = self.sections[s].A
            outputs["section_rho"][k] = self.sections[s].rho
            outputs["section_Ixx"][k] = self.sections[s].Ixx
            outputs["section_Iyy"][k] = self.sections[s].Iyy
            outputs["section_Izz"][k] = self.sections[s].Izz
            outputs["section_E"][k] = self.sections[s].E
            outputs["section_G"][k] = self.sections[s].G


class MemberHydro(om.ExplicitComponent):
    """
    Compute member substructure elements in floating offshore wind turbines.

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
        Radius of heave plate at bottom of member
    shell_mass : numpy array[n_full-1], [kg]
        mass of member shell
    stiffener_mass : numpy array[n_full-1], [kg]
        mass of member stiffeners
    bulkhead_mass : numpy array[n_full-1], [kg]
        mass of member bulkheads
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
        mass of member shell
    stiffener_cost : float, [USD]
        mass of member stiffeners
    bulkhead_cost : float, [USD]
        mass of member bulkheads
    ballast_cost : float, [USD]
        cost of permanent ballast
    buoyancy_tank_cost : float, [USD]
        mass of heave plate
    outfitting_cost_rate : float, [USD/kg]
        Cost per unit mass for outfitting member

    Returns
    -------
    z_center_of_mass : float, [m]
        z-position CofG of member
    z_center_of_buoyancy : float, [m]
        z-position CofB of member
    Awater : float, [m**2]
        Area of waterplace cross section
    Iwater : float, [m**4]
        Second moment of area of waterplace cross section
    I_member : numpy array[6], [kg*m**2]
        Moments of inertia of whole member relative to keel point
    displaced_volume : numpy array[n_full-1], [m**3]
        Volume of water displaced by member by section
    hydrostatic_force : numpy array[n_full-1], [N]
        Net z-force on member sections
    member_structural_mass : float, [kg]
        mass of member structure
    member_outfitting_cost : float, [USD]
        cost of outfitting the member
    member_outfitting_mass : float, [kg]
        cost of outfitting the member
    member_added_mass : numpy array[6], [kg]
        hydrodynamic added mass matrix diagonal
    member_total_mass : numpy array[n_full-1], [kg]
        total mass of member by section
    member_total_cost : float, [USD]
        total cost of member
    member_structural_cost : float, [USD]
        Cost of member without ballast or outfitting
    tapered_member_cost_rate : float, [USD/t]
        Cost rate of finished member

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
        self.add_output("I_member", np.zeros(6), units="kg*m**2")
        self.add_output("displaced_volume", np.zeros(n_full - 1), units="m**3")
        self.add_output("hydrostatic_force", np.zeros(n_full - 1), units="N")

        self.add_output("member_structural_mass", 0.0, units="kg")
        self.add_output("member_outfitting_cost", 0.0, units="USD")
        self.add_output("member_outfitting_mass", 0.0, units="kg")

        self.add_output("member_added_mass", np.zeros(6), units="kg")
        self.add_output("member_total_mass", np.zeros(n_full - 1), units="kg")
        self.add_output("member_total_cost", 0.0, units="USD")
        self.add_output("member_structural_cost", 0.0, units="USD")
        self.add_output("tapered_member_cost_rate", 0.0, units="USD/t")

        self.declare_partials("*", "*", method="fd", form="central", step=1e-6)

    def compute(self, inputs, outputs):
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
        outputs["member_added_mass"] = m_a


class Member(om.Group):
    def initialize(self):
        self.options.declare("n_mat")
        self.options.declare("member_options")
        self.options.declare("modeling_options")

    def setup(self):
        opt = self.options["modeling_options"]
        colopt = self.options["member_options"]
        n_layers = colopt["n_layers"]
        n_height = colopt["n_height"]
        n_mat = self.options["n_mat"]
        n_full = get_nfull(n_height)

        self.set_input_defaults("layer_materials", ["steel"])
        self.set_input_defaults("material_names", ["steel"])

        # TODO: Use reference axis and curvature, s, instead of assuming everything is vertical on z
        self.add_subsystem(
            "yaml", DiscretizationYAML(n_height=n_height, n_layers=n_layers, n_mat=n_mat), promotes=["*"]
        )

        self.add_subsystem(
            "gc", GeometricConstraints(nPoints=n_height, diamFlag=True), promotes=["constr_taper", "constr_d_to_t"]
        )

        self.add_subsystem("geom", MemberDiscretization(n_height=n_height), promotes=["*"])

        self.add_subsystem("comp", MemberComponent(n_height=n_height), promotes=["*"])

        # self.add_subsystem("hydro", MemberHydro(nPoints=n_full), promotes=["*"])

        prom = [
            "Uref",
            "zref",
            "shearExp",
            "z0",
            "cd_usr",
            "cm",
            "beta_wind",
            "rho_air",
            "mu_air",
            "beta_water",
            "rho_water",
            "mu_water",
            "Uc",
            "Hsig_wave",
            "Tsig_wave",
            "rho_water",
            "water_depth",
            "Px",
            "Py",
            "Pz",
            "qdyn",
            "yaw",
        ]
        self.add_subsystem("env", CylinderEnvironment(nPoints=n_full, water_flag=True), promotes=prom)

        self.connect("outer_diameter", "gc.d")
        self.connect("wall_thickness", "gc.t")
        self.connect("z_full", "env.z")
        self.connect("d_full", "env.d")
