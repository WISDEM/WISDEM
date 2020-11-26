import numpy as np
from scipy.integrate import cumtrapz
import openmdao.api as om

from wisdem.commonse import gravity, eps, DirectionVector, NFREQ
from wisdem.commonse.utilities import assembleI, unassembleI
from wisdem.commonse.vertical_cylinder import get_nfull
from .map_mooring import NLINES_MAX


class SubstructureGeometry(om.ExplicitComponent):
    """
    Component for substructure geometry for floating offshore wind turbines.

    Parameters
    ----------
    main_d_full : numpy array[n_full_main], [m]
        outer radius at each section node bottom to top (length = nsection + 1)
    main_z_nodes : numpy array[n_full_main], [m]
        z-coordinates of section nodes (length = nsection+1)
    offset_d_full : numpy array[n_full_off], [m]
        outer radius at each section node bottom to top (length = nsection + 1)
    offset_z_nodes : numpy array[n_full_off], [m]
        z-coordinates of section nodes (length = nsection+1)
    offset_freeboard : float, [m]
        Length of column above water line
    offset_draft : float, [m]
        Length of column below water line
    fairlead_location : float
        Fractional length from column bottom to top for mooring line attachment
    fairlead_offset_from_shell : float, [m]
        fairlead offset from shell
    radius_to_offset_column : float, [m]
        Distance from main column centerpoint to offset column centerpoint
    number_of_offset_columns : float
        Number of offset columns evenly spaced around main column
    tower_d_base : float, [m]
        base diameter of the tower
    hsig_wave : float, [m]
        significant wave height
    max_survival_heel : float, [deg]
        max heel angle for turbine survival

    Returns
    -------
    fairlead : float, [m]
        Depth below water line for mooring line attachment
    fairlead_radius : float, [m]
        Outer spar radius at fairlead depth (point of mooring attachment)
    main_offset_spacing : float
        Radius of main and offset columns relative to spacing
    tower_transition_buffer : float, [m]
        Buffer between substructure main and tower main
    nacelle_transition_buffer : float, [m]
        Buffer between tower top and nacelle main
    offset_freeboard_heel_margin : float, [m]
        Margin so offset column does not submerge during max heel
    offset_draft_heel_margin : float, [m]
        Margin so offset column does not leave water during max heel
    wave_height_fairlead_ratio : float
        Ratio of maximum wave height (avg of top 1%) to fairlead

    """

    def initialize(self):
        self.options.declare("n_height_main")
        self.options.declare("n_height_off")

    def setup(self):
        n_height_main = self.options["n_height_main"]
        n_height_off = self.options["n_height_off"]
        n_full_main = get_nfull(n_height_main)
        n_full_off = get_nfull(n_height_off)

        self.add_input("main_d_full", np.zeros(n_full_main), units="m")
        self.add_input("main_z_nodes", np.zeros(n_full_main), units="m")
        self.add_input("offset_d_full", np.zeros(n_full_off), units="m")
        self.add_input("offset_z_nodes", np.zeros(n_full_off), units="m")
        self.add_input("offset_freeboard", 0.0, units="m")
        self.add_input("offset_draft", 0.0, units="m")
        self.add_input("fairlead_location", 0.0)
        self.add_input("fairlead_offset_from_shell", 0.0, units="m")
        self.add_input("radius_to_offset_column", 0.0, units="m")
        self.add_input("number_of_offset_columns", 0)
        self.add_input("tower_d_base", 0.0, units="m")
        self.add_input("hsig_wave", 0.0, units="m")
        self.add_input("max_survival_heel", 0.0, units="deg")

        self.add_output("fairlead", 0.0, units="m")
        self.add_output("fairlead_radius", 0.0, units="m")
        self.add_output("main_offset_spacing", 0.0)
        self.add_output("tower_transition_buffer", 0.0, units="m")
        self.add_output("offset_freeboard_heel_margin", 0.0, units="m")
        self.add_output("offset_draft_heel_margin", 0.0, units="m")
        self.add_output("wave_height_fairlead_ratio", 0.0)

        # Derivatives
        self.declare_partials("*", "*", method="fd", form="central", step=1e-6)

    def compute(self, inputs, outputs):
        """Sets nodal points and sectional centers of mass in z-coordinate system with z=0 at the waterline.
        Nodal points are the beginning and end points of each section.
        Nodes and sections start at bottom and move upwards.

        INPUTS:
        ----------
        params   : dictionary of input parameters
        outputs : dictionary of output parameters

        OUTPUTS  : none (all unknown dictionary values set)
        """
        # Unpack variables
        ncolumns = int(inputs["number_of_offset_columns"])
        R_od_main = 0.5 * inputs["main_d_full"]
        R_od_offset = 0.5 * inputs["offset_d_full"]
        R_semi = inputs["radius_to_offset_column"]
        R_tower_base = 0.5 * inputs["tower_d_base"]

        z_nodes_offset = inputs["offset_z_nodes"]
        z_nodes_main = inputs["main_z_nodes"]

        location = inputs["fairlead_location"]
        fair_off = inputs["fairlead_offset_from_shell"]
        off_freeboard = inputs["offset_freeboard"]
        off_draft = inputs["offset_draft"]
        max_heel = inputs["max_survival_heel"]

        # Set spacing constraint
        outputs["main_offset_spacing"] = R_semi - R_od_main.max() - R_od_offset.max()

        # Determine location and radius at mooring connection point (fairlead)
        if ncolumns > 0:
            z_fairlead = location * (z_nodes_offset[-1] - z_nodes_offset[0]) + z_nodes_offset[0]
            outputs["fairlead_radius"] = R_semi + fair_off + np.interp(z_fairlead, z_nodes_offset, R_od_offset)
        else:
            z_fairlead = location * (z_nodes_main[-1] - z_nodes_main[0]) + z_nodes_main[0]
            outputs["fairlead_radius"] = fair_off + np.interp(z_fairlead, z_nodes_main, R_od_main)
        outputs["fairlead"] = -z_fairlead  # Fairlead defined as positive below waterline
        outputs["wave_height_fairlead_ratio"] = inputs["hsig_wave"] / np.abs(z_fairlead)

        # Constrain spar top to be at least greater than tower main
        outputs["tower_transition_buffer"] = R_od_main[-1] - R_tower_base

        # Make sure semi columns don't get submerged
        heel_deflect = R_semi * np.sin(np.deg2rad(max_heel))
        outputs["offset_freeboard_heel_margin"] = off_freeboard - heel_deflect
        outputs["offset_draft_heel_margin"] = off_draft - heel_deflect


class Substructure(om.ExplicitComponent):
    """
    Calculate substructure properties

    Parameters
    ----------
    water_density : float, [kg/m**3]
        density of water
    wave_period_range_low : float, [s]
        Lower bound of typical ocean wavve period
    wave_period_range_high : float, [s]
        Upper bound of typical ocean wavve period
    operational_heel : float, [deg]
        Maximum angle of heel allowable
    mooring_mass : float, [kg]
        Mass of mooring lines
    mooring_moments_of_inertia : numpy array[6], [kg*m**2]
        mass moment of inertia of mooring system about fairlead-centerline point [xx yy
        zz xy xz yz]
    mooring_neutral_load : numpy array[NLINES_MAX, 3], [N]
        z-force of mooring lines on structure
    mooring_surge_restoring_force : float, [N]
        Restoring force from mooring system after surge motion
    mooring_pitch_restoring_force : numpy array[NLINES_MAX, 3], [N]
        Restoring force from mooring system after pitch motion
    mooring_cost : float, [USD]
        Cost of mooring system
    mooring_stiffness : numpy array[6, 6], [N/m]
        Linearized stiffness matrix of mooring system at neutral (no offset) conditions.
    fairlead : float, [m]
        Depth below water for mooring line attachment
    fairlead_radius : float, [m]
        Outer spar radius at fairlead depth (point of mooring attachment)
    number_of_offset_columns : float
        Number of offset columns evenly spaced around main column
    radius_to_offset_column : float, [m]
        Distance from main column centerpoint to offset column centerpoint
    main_Iwaterplane : float, [m**4]
        Second moment of area of waterplane cross-section
    main_Awaterplane : float, [m**2]
        Area of waterplane cross-section
    main_cost : float, [USD]
        Cost of spar structure
    main_mass : numpy array[n_full_main-1], [kg]
        mass of main column by section
    main_freeboard : float, [m]
        Length of spar above water line
    main_center_of_buoyancy : float, [m]
        z-position of center of column buoyancy force
    main_center_of_mass : float, [m]
        z-position of center of column mass
    main_moments_of_inertia : numpy array[6], [kg*m**2]
        mass moment of inertia of column about main [xx yy zz xy xz yz]
    main_added_mass : numpy array[6], [kg]
        Diagonal of added mass matrix- masses are first 3 entries, moments are last 3
    offset_Iwaterplane : float, [m**4]
        Second moment of area of waterplane cross-section
    offset_Awaterplane : float, [m**2]
        Area of waterplane cross-section
    offset_cost : float, [USD]
        Cost of spar structure
    offset_mass : numpy array[n_full_off-1], [kg]
        mass of offset column by section
    offset_center_of_buoyancy : float, [m]
        z-position of center of column buoyancy force
    offset_center_of_mass : float, [m]
        z-position of center of column mass
    offset_moments_of_inertia : numpy array[6], [kg*m**2]
        mass moment of inertia of column about main [xx yy zz xy xz yz]
    offset_added_mass : numpy array[6], [kg]
        Diagonal of added mass matrix- masses are first 3 entries, moments are last 3
    tower_mass : float, [kg]
        Mass of tower
    tower_shell_cost : float, [USD]
        Cost of tower
    tower_I_base : numpy array[6], [kg*m**2]
        Moments about tower main
    tower_z_full : numpy array[n_full_tow], [m]
        z-coordinates of section nodes (length = nsection+1)
    rna_mass : float, [kg]
        Mass of RNA
    rna_cg : numpy array[3], [m]
        Location of RNA center of mass relative to tower top
    rna_I : numpy array[6], [kg*m**2]
        Moments about turbine main
    water_ballast_zpts_vector : numpy array[n_full_main], [m]
        z-points of potential ballast mass
    water_ballast_radius_vector : numpy array[n_full_main], [m]
        Inner radius of potential ballast mass
    structural_mass : float, [kg]
        Mass of whole turbine except for mooring lines
    structure_center_of_mass : numpy array[3], [m]
        xyz-position of center of gravity of whole turbine
    structural_frequencies : numpy array[NFREQ], [Hz]
        Structural frequencies outputted from FEM calculation
    z_center_of_buoyancy : float, [m]
        z-position of center of gravity (x,y = 0,0)
    total_displacement : float, [m**3]
        Total volume of water displaced by floating turbine (except for mooring lines)
    total_force : numpy array[3], [N]
        Net forces on turbine
    total_moment : numpy array[3], [N*m]
        Moments on whole turbine
    pontoon_cost : float, [USD]
        Cost of pontoon elements and connecting truss

    Returns
    -------
    substructure_moments_of_inertia : numpy array[6], [kg*m**2]
        mass moment of inertia of substructure (no tower or rna or mooring)
        [xx yy zz xy xz yz]
    total_mass : float, [kg]
        total mass of spar and moorings
    total_cost : float, [USD]
        total cost of spar and moorings
    metacentric_height : float, [m]
        measure of static overturning stability
    buoyancy_to_gravity : float
        static stability margin based on position of centers of gravity and buoyancy
    offset_force_ratio : float
        total surge force divided by restoring force
    heel_moment_ratio : float
        total pitch moment divided by restoring moment
    Iwaterplane_system : float, [m**4]
        Second moment of area of waterplane cross-section for whole structure
    center_of_mass : numpy array[3], [m]
        xyz-position of center of gravity (x,y = 0,0)
    variable_ballast_mass : float, [kg]
        Amount of variable water ballast
    variable_ballast_center_of_mass : float, [m]
        Center of mass for variable ballast
    variable_ballast_moments_of_inertia : numpy array[6], [kg*m**2]
        mass moment of inertia of variable ballast [xx yy zz xy xz yz]
    variable_ballast_height : float, [m]
        height of water ballast to balance spar
    variable_ballast_height_ratio : float
        Ratio of water ballast height to available height
    mass_matrix : numpy array[6], [kg]
        Summary mass matrix of structure (minus pontoons)
    added_mass_matrix : numpy array[6], [kg]
        Summary hydrodynamic added mass matrix of structure (minus pontoons)
    hydrostatic_stiffness : numpy array[6], [N/m]
        Summary hydrostatic stiffness of structure
    rigid_body_periods : numpy array[6], [s]
        Natural periods of oscillation in 6 DOF
    period_margin_low : numpy array[6]
        Margin between natural periods and 2 second wave period
    period_margin_high : numpy array[6]
        Margin between natural periods and 20 second wave period
    modal_margin_low : numpy array[NFREQ]
        Margin between structural modes and 2 second wave period
    modal_margin_high : numpy array[NFREQ]
        Margin between structural modes and 20 second wave period

    """

    def initialize(self):
        self.options.declare("n_height_main")
        self.options.declare("n_height_off")
        self.options.declare("n_height_tow")

    def setup(self):
        n_height_main = self.options["n_height_main"]
        n_height_off = self.options["n_height_off"]
        n_height_tow = self.options["n_height_tow"]
        n_full_main = get_nfull(n_height_main)
        n_full_off = get_nfull(n_height_off)
        n_full_tow = get_nfull(n_height_tow)

        self.add_input("rho_water", 0.0, units="kg/m**3")
        self.add_input("wave_period_range_low", 2.0, units="s")
        self.add_input("wave_period_range_high", 20.0, units="s")
        self.add_input("operational_heel", 0.0, units="deg")
        self.add_input("mooring_mass", 0.0, units="kg")
        self.add_input("mooring_moments_of_inertia", np.zeros(6), units="kg*m**2")
        self.add_input("mooring_neutral_load", np.zeros((NLINES_MAX, 3)), units="N")
        self.add_input("mooring_surge_restoring_force", 0.0, units="N")
        self.add_input("mooring_pitch_restoring_force", np.zeros((NLINES_MAX, 3)), units="N")
        self.add_input("mooring_cost", 0.0, units="USD")
        self.add_input("mooring_stiffness", np.zeros((6, 6)), units="N/m")
        self.add_input("fairlead", 1.0, units="m")
        self.add_input("fairlead_radius", 0.0, units="m")
        self.add_input("number_of_offset_columns", 0)
        self.add_input("radius_to_offset_column", 0.0, units="m")

        self.add_input("main_Iwaterplane", 0.0, units="m**4")
        self.add_input("main_Awaterplane", 0.0, units="m**2")
        self.add_input("main_cost", 0.0, units="USD")
        self.add_input("main_mass", np.zeros(n_full_main - 1), units="kg")
        self.add_input("main_freeboard", 0.0, units="m")
        self.add_input("main_center_of_buoyancy", 0.0, units="m")
        self.add_input("main_center_of_mass", 0.0, units="m")
        self.add_input("main_moments_of_inertia", np.zeros(6), units="kg*m**2")
        self.add_input("main_added_mass", np.zeros(6), units="kg")

        self.add_input("offset_Iwaterplane", 0.0, units="m**4")
        self.add_input("offset_Awaterplane", 0.0, units="m**2")
        self.add_input("offset_cost", 0.0, units="USD")
        self.add_input("offset_mass", np.zeros(n_full_off - 1), units="kg")
        self.add_input("offset_center_of_buoyancy", 0.0, units="m")
        self.add_input("offset_center_of_mass", 0.0, units="m")
        self.add_input("offset_moments_of_inertia", np.zeros(6), units="kg*m**2")
        self.add_input("offset_added_mass", np.zeros(6), units="kg")

        self.add_input("tower_mass", 0.0, units="kg")
        self.add_input("tower_shell_cost", 0.0, units="USD")
        self.add_input("tower_I_base", np.zeros(6), units="kg*m**2")
        self.add_input("tower_z_full", np.zeros(n_full_tow), units="m")
        self.add_input("rna_mass", 0.0, units="kg")
        self.add_input("rna_cg", np.zeros(3), units="m")
        self.add_input("rna_I", np.zeros(6), units="kg*m**2")

        self.add_input("water_ballast_zpts_vector", np.zeros(n_full_main), units="m")
        self.add_input("water_ballast_radius_vector", np.zeros(n_full_main), units="m")

        self.add_input("structural_mass", 0.0, units="kg")
        self.add_input("structure_center_of_mass", np.zeros(3), units="m")
        self.add_input("structural_frequencies", np.zeros(NFREQ), units="Hz")
        self.add_input("z_center_of_buoyancy", 0.0, units="m")
        self.add_input("total_displacement", 0.0, units="m**3")
        self.add_input("total_force", np.zeros(3), units="N")
        self.add_input("total_moment", np.zeros(3), units="N*m")
        self.add_input("pontoon_cost", 0.0, units="USD")

        self.add_output("substructure_moments_of_inertia", np.zeros(6), units="kg*m**2")
        self.add_output("total_mass", 0.0, units="kg")
        self.add_output("total_cost", 0.0, units="USD")
        self.add_output("metacentric_height", 0.0, units="m")
        self.add_output("buoyancy_to_gravity", 0.0)
        self.add_output("offset_force_ratio", 0.0)
        self.add_output("heel_moment_ratio", 0.0)
        self.add_output("Iwaterplane_system", 0.0, units="m**4")
        self.add_output("center_of_mass", np.zeros(3), units="m")
        self.add_output("variable_ballast_mass", 0.0, units="kg")
        self.add_output("variable_ballast_center_of_mass", 0.0, units="m")
        self.add_output("variable_ballast_moments_of_inertia", np.zeros(6), units="kg*m**2")
        self.add_output("variable_ballast_height", 0.0, units="m")
        self.add_output("variable_ballast_height_ratio", 0.0)
        self.add_output("mass_matrix", np.zeros(6), units="kg")
        self.add_output("added_mass_matrix", np.zeros(6), units="kg")
        self.add_output("hydrostatic_stiffness", np.zeros(6), units="N/m")
        self.add_output("rigid_body_periods", np.zeros(6), units="s")
        self.add_output("period_margin_low", np.zeros(6))
        self.add_output("period_margin_high", np.zeros(6))
        self.add_output("modal_margin_low", np.zeros(NFREQ))
        self.add_output("modal_margin_high", np.zeros(NFREQ))

        self.declare_partials("*", "*", method="fd", form="central", step=1e-6)

    def compute(self, inputs, outputs):
        # TODO: Get centerlines right- in sparGeometry?
        # Determine ballast and cg of system
        self.balance(inputs, outputs)

        # Determine stability, metacentric height from waterplane profile, displaced volume
        self.compute_stability(inputs, outputs)

        # Compute natural periods of osciallation
        self.compute_rigid_body_periods(inputs, outputs)

        # Check margins of natural and eigenfrequencies against waves
        self.check_frequency_margins(inputs, outputs)

        # Sum all costs
        self.compute_costs(inputs, outputs)

    def balance(self, inputs, outputs):
        # Unpack variables
        m_struct = inputs["structural_mass"]
        Fz_mooring = np.sum(inputs["mooring_neutral_load"][:, -1])
        m_mooring = inputs["mooring_mass"]

        V_system = inputs["total_displacement"]

        cg_struct = inputs["structure_center_of_mass"]

        z_water_data = inputs["water_ballast_zpts_vector"]
        r_water_data = inputs["water_ballast_radius_vector"]
        rhoWater = inputs["rho_water"]

        # SEMI TODO: Make water_ballast in main only?  columns too?  How to apportion?

        # Make sure total mass of system with variable water ballast balances against displaced volume
        # Water ballast should be buried in m_column
        m_water = V_system * rhoWater - (m_struct + Fz_mooring / gravity)
        m_system = m_struct + m_water

        # Output substructure total turbine mass
        outputs["total_mass"] = m_struct + m_mooring

        # Find height given interpolant functions from columns
        m_water_data = rhoWater * np.pi * cumtrapz(r_water_data ** 2, z_water_data)
        m_water_data = np.r_[0.0, m_water_data]  # cumtrapz has length-1

        if m_water_data[-1] < m_water:
            # Don't have enough space, so max out variable balast here and constraints will catch this
            z_end = z_water_data[-1]
            coeff = m_water / m_water_data[-1]
        elif m_water < 0.0:
            z_end = z_water_data[0]
            coeff = 0.0
        else:
            z_end = np.interp(m_water, m_water_data, z_water_data)
            coeff = 1.0
        h_water = z_end - z_water_data[0]
        outputs["variable_ballast_mass"] = m_water
        outputs["variable_ballast_height"] = coeff * h_water
        outputs["variable_ballast_height_ratio"] = coeff * h_water / (z_water_data[-1] - z_water_data[0])

        # Find cg of whole system
        # First find cg of water variable ballast by finding midpoint of mass sum
        z_cg = np.interp(0.5 * coeff * m_water, m_water_data, z_water_data)
        outputs["center_of_mass"] = (m_struct * cg_struct + m_water * np.r_[0.0, 0.0, z_cg]) / m_system
        outputs["variable_ballast_center_of_mass"] = z_cg

        # Integrate for moment of inertia of variable ballast
        npts = 100
        z_int = np.linspace(float(z_water_data[0]), float(z_end), npts)
        r_int = np.interp(z_int, z_water_data, r_water_data)
        Izz = 0.5 * rhoWater * np.pi * np.trapz(r_int ** 4, z_int)
        Ixx = rhoWater * np.pi * np.trapz(0.25 * r_int ** 4 + r_int ** 2 * (z_int - z_cg) ** 2, z_int)
        outputs["variable_ballast_moments_of_inertia"] = np.r_[Ixx, Ixx, Izz, 0.0, 0.0, 0.0]

    def compute_stability(self, inputs, outputs):
        # Unpack variables
        ncolumn = int(inputs["number_of_offset_columns"])
        z_cb = inputs["z_center_of_buoyancy"]
        z_cg = outputs["center_of_mass"][-1]
        V_system = inputs["total_displacement"]

        Iwater_main = inputs["main_Iwaterplane"]
        Iwater_column = inputs["offset_Iwaterplane"]
        Awater_column = inputs["offset_Awaterplane"]

        F_surge = inputs["total_force"][0]
        M_pitch = inputs["total_moment"][1]
        F_restore = inputs["mooring_surge_restoring_force"]
        rhoWater = inputs["rho_water"]
        R_semi = inputs["radius_to_offset_column"]

        F_restore_pitch = inputs["mooring_pitch_restoring_force"]
        z_fairlead = inputs["fairlead"] * (-1)
        R_fairlead = inputs["fairlead_radius"]
        oper_heel = float(inputs["operational_heel"])

        # Compute the distance from the center of buoyancy to the metacentre (BM is naval architecture)
        # BM = Iw / V where V is the displacement volume (just computed)
        # Iw is the area moment of inertia (meters^4) of the water-plane cross section about the heel axis
        # For a spar, we assume this is just the I of a circle about x or y
        # See https://en.wikipedia.org/wiki/Metacentric_height
        # https://en.wikipedia.org/wiki/List_of_second_moments_of_area
        # and http://farside.ph.utexas.edu/teaching/336L/Fluidhtml/node30.html

        # Water plane area of all components with parallel axis theorem
        Iwater_system = Iwater_main
        radii = R_semi * np.cos(np.linspace(0, 2 * np.pi, ncolumn + 1))
        for k in range(ncolumn):
            Iwater_system += Iwater_column + Awater_column * radii[k] ** 2
        outputs["Iwaterplane_system"] = Iwater_column

        # Measure static stability:
        # 1. Center of buoyancy should be above CG (difference should be positive)
        # 2. Metacentric height should be positive
        buoyancy2metacentre_BM = Iwater_system / V_system
        outputs["buoyancy_to_gravity"] = z_cg - z_cb
        outputs["metacentric_height"] = buoyancy2metacentre_BM - outputs["buoyancy_to_gravity"]

        F_buoy = V_system * rhoWater * gravity
        M_restore = outputs["metacentric_height"] * np.sin(np.deg2rad(oper_heel)) * F_buoy

        # Convert mooring restoring force after pitch to a restoring moment
        nlines = np.count_nonzero(F_restore_pitch[:, 2])
        F_restore_pitch = F_restore_pitch[:nlines, :]
        moorx = R_fairlead * np.cos(np.linspace(0, 2 * np.pi, nlines + 1)[:-1])
        moory = R_fairlead * np.sin(np.linspace(0, 2 * np.pi, nlines + 1)[:-1])
        r_moor = np.c_[moorx, moory, (z_fairlead - z_cg) * np.ones(moorx.shape)]
        Msum = 0.0
        for k in range(nlines):
            dvF = DirectionVector.fromArray(F_restore_pitch[k, :])
            dvR = DirectionVector.fromArray(r_moor[k, :]).yawToHub(oper_heel)
            M = dvR.cross(dvF)
            Msum += M.y

        M_restore += Msum

        # Comput heel angle, scaling overturning moment by defect of inflow velocity
        # TODO: Make this another load case in Frame3DD
        outputs["heel_moment_ratio"] = np.abs(np.cos(np.deg2rad(oper_heel)) ** 2.0 * M_pitch / M_restore)

        # Now compute offsets from the applied force
        # First use added mass (the mass of the water that must be displaced in movement)
        # http://www.iaea.org/inis/collection/NCLCollectionStore/_Public/09/411/9411273.pdf
        # mass_add_surge = rhoWater * np.pi * R_od.max() * draft
        # T_surge        = 2*np.pi*np.sqrt( (outputs['total_mass']+mass_add_surge) / kstiff_horiz_mooring)

        # Compare restoring force from mooring to force of worst case spar displacement
        outputs["offset_force_ratio"] = np.abs(F_surge / F_restore)

    def compute_rigid_body_periods(self, inputs, outputs):
        # Unpack variables
        ncolumn = int(inputs["number_of_offset_columns"])
        R_semi = inputs["radius_to_offset_column"]

        m_main = np.sum(inputs["main_mass"])
        m_column = np.sum(inputs["offset_mass"])
        m_tower = np.sum(inputs["tower_mass"])
        m_rna = inputs["rna_mass"]
        m_mooring = inputs["mooring_mass"]
        m_total = outputs["total_mass"]
        m_water = np.maximum(0.0, outputs["variable_ballast_mass"])
        m_a_main = inputs["main_added_mass"]
        m_a_column = inputs["offset_added_mass"]

        rhoWater = inputs["rho_water"]
        V_system = inputs["total_displacement"]
        h_metacenter = outputs["metacentric_height"]

        Awater_main = inputs["main_Awaterplane"]
        Awater_column = inputs["offset_Awaterplane"]
        I_main = inputs["main_moments_of_inertia"]
        I_column = inputs["offset_moments_of_inertia"]
        I_mooring = inputs["mooring_moments_of_inertia"]
        I_water = outputs["variable_ballast_moments_of_inertia"]
        I_tower = inputs["tower_I_base"]
        I_rna = inputs["rna_I"]
        I_waterplane = outputs["Iwaterplane_system"]

        z_cg_main = float(inputs["main_center_of_mass"])
        z_cb_main = float(inputs["main_center_of_buoyancy"])
        z_cg_column = float(inputs["offset_center_of_mass"])
        z_cb_column = float(inputs["offset_center_of_buoyancy"])
        z_cb = float(inputs["z_center_of_buoyancy"])
        z_cg_water = float(outputs["variable_ballast_center_of_mass"])
        z_fairlead = float(inputs["fairlead"] * (-1))

        r_cg = outputs["center_of_mass"]
        cg_rna = inputs["rna_cg"]
        z_tower = inputs["tower_z_full"]

        K_moor = np.diag(inputs["mooring_stiffness"])

        # Number of degrees of freedom
        nDOF = 6

        # Compute elements on mass matrix diagonal
        M_mat = np.zeros((nDOF,))
        # Surge, sway, heave just use normal inertia (without mooring according to Senu)
        M_mat[:3] = m_total + m_water - m_mooring
        # Add in moments of inertia of primary column
        I_total = assembleI(np.zeros(6))
        I_main = assembleI(I_main)
        R = np.array([0.0, 0.0, z_cg_main]) - r_cg
        I_total += I_main + m_main * (np.dot(R, R) * np.eye(3) - np.outer(R, R))
        # Add up moments of intertia of other columns
        radii_x = R_semi * np.cos(np.linspace(0, 2 * np.pi, ncolumn + 1))
        radii_y = R_semi * np.sin(np.linspace(0, 2 * np.pi, ncolumn + 1))
        I_column = assembleI(I_column)
        for k in range(ncolumn):
            R = np.array([radii_x[k], radii_y[k], z_cg_column]) - r_cg
            I_total += I_column + m_column * (np.dot(R, R) * np.eye(3) - np.outer(R, R))
        # Add in variable ballast
        R = np.array([0.0, 0.0, z_cg_water]) - r_cg
        I_total += assembleI(I_water) + m_water * (np.dot(R, R) * np.eye(3) - np.outer(R, R))

        # Save what we have so far as m_substructure & I_substructure and move to its own CM
        m_subs = m_main + ncolumn * m_column + m_water
        z_cg_subs = (m_main * z_cg_main + ncolumn * m_column * z_cg_column + m_water * z_cg_water) / m_subs
        R = r_cg - np.r_[0.0, 0.0, z_cg_subs]
        I_substructure = I_total + m_subs * (np.dot(R, R) * np.eye(3) - np.outer(R, R))
        outputs["substructure_moments_of_inertia"] = unassembleI(I_total)

        # Now go back to the total
        # Add in mooring system- Not needed according to Senu
        # R         = np.array([0.0, 0.0, z_fairlead]) - r_cg
        # I_total  += assembleI(I_mooring) + m_mooring*(np.dot(R, R)*np.eye(3) - np.outer(R, R))
        # Add in tower
        R = np.array([0.0, 0.0, z_tower[0]]) - r_cg
        I_total += assembleI(I_tower) + m_tower * (np.dot(R, R) * np.eye(3) - np.outer(R, R))
        # Add in RNA
        R = np.array([0.0, 0.0, z_tower[-1]]) + cg_rna - r_cg
        I_total += assembleI(I_rna) + m_rna * (np.dot(R, R) * np.eye(3) - np.outer(R, R))
        # Stuff moments of inertia into mass matrix
        M_mat[3:] = unassembleI(I_total)[:3]
        outputs["mass_matrix"] = M_mat

        # Add up all added mass entries in a similar way
        A_mat = np.zeros((nDOF,))
        # Surge, sway, heave just use normal inertia
        A_mat[:3] = m_a_main[:3] + ncolumn * m_a_column[:3]
        # Add up moments of inertia, move added mass moments from CofB to CofG
        I_main = assembleI(np.r_[m_a_main[3:], np.zeros(3)])
        R = np.array([0.0, 0.0, z_cb_main]) - r_cg
        I_total = I_main + m_a_main[0] * (np.dot(R, R) * np.eye(3) - np.outer(R, R))
        # Add up added moments of intertia of all columns for other entries
        I_column = assembleI(np.r_[m_a_column[3:], np.zeros(3)])
        for k in range(ncolumn):
            R = np.array([radii_x[k], radii_y[k], z_cb_column]) - r_cg
            I_total += I_column + m_a_column[0] * (np.dot(R, R) * np.eye(3) - np.outer(R, R))
        A_mat[3:] = unassembleI(I_total)[:3]
        outputs["added_mass_matrix"] = A_mat

        # Hydrostatic stiffness has contributions in heave (K33) and roll/pitch (K44/55)
        # See DNV-RP-H103: Modeling and Analyis of Marine Operations
        K_hydro = np.zeros((nDOF,))
        K_hydro[2] = rhoWater * gravity * (Awater_main + ncolumn * Awater_column)
        K_hydro[3:5] = rhoWater * gravity * V_system * h_metacenter  # FAST eqns: (I_waterplane + V_system * z_cb)
        outputs["hydrostatic_stiffness"] = K_hydro

        # Now compute all six natural periods at once
        epsilon = 1e-6  # Avoids numerical issues
        K_total = np.maximum(K_hydro + K_moor, 0.0)
        outputs["rigid_body_periods"] = 2 * np.pi * np.sqrt((M_mat + A_mat) / (K_total + epsilon))

    def check_frequency_margins(self, inputs, outputs):
        # Unpack variables
        T_sys = outputs["rigid_body_periods"]
        T_wave_low = inputs["wave_period_range_low"]
        T_wave_high = inputs["wave_period_range_high"]
        f_struct = inputs["structural_frequencies"]
        T_struct = np.zeros(f_struct.shape)
        for k in range(T_struct.size):
            T_struct[k] = 0.0 if f_struct[k] == 0.0 else 1 / f_struct[k]

        # Waves cannot excite yaw, so removing that constraint

        # Compute margins between wave forcing and natural periods
        indicator_high = T_wave_high * np.ones(T_sys.shape)
        indicator_high[T_sys < T_wave_low] = 1e-16
        indicator_high[-1] = 1e-16  # Not yaw
        outputs["period_margin_high"] = T_sys / indicator_high

        indicator_low = T_wave_low * np.ones(T_sys.shape)
        indicator_low[T_sys > T_wave_high] = 1e30
        indicator_low[-1] = 1e30  # Not yaw
        outputs["period_margin_low"] = T_sys / indicator_low

        # Compute margins bewteen wave forcing and structural frequencies
        indicator_high = T_wave_high * np.ones(T_struct.shape)
        indicator_high[T_struct < T_wave_low] = 1e-16
        outputs["modal_margin_high"] = T_struct / indicator_high

        indicator_low = T_wave_low * np.ones(T_struct.shape)
        indicator_low[T_struct > T_wave_high] = 1e30
        outputs["modal_margin_low"] = T_struct / indicator_low

    def compute_costs(self, inputs, outputs):
        # Unpack variables
        ncolumn = int(inputs["number_of_offset_columns"])
        c_mooring = inputs["mooring_cost"]
        c_aux = inputs["offset_cost"]
        c_main = inputs["main_cost"]
        c_pontoon = inputs["pontoon_cost"]
        c_tower = inputs["tower_shell_cost"]

        outputs["total_cost"] = c_mooring + ncolumn * c_aux + c_main + c_pontoon + c_tower
