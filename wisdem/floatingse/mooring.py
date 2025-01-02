import numpy as np
import openmdao.api as om

import moorpy as mp
import moorpy.MoorProps as props

NLINES_MAX = 15
NPTS_PLOT = 101


class Mooring(om.ExplicitComponent):
    """
    Sets mooring line properties then writes MAP input file and executes MAP.

    Component for mooring system attached to sub-structure of floating offshore wind turbines.
    Should be tightly coupled with Spar class for full system representation.

    Parameters
    ----------
    water_density : float, [kg/m**3]
        density of water
    water_depth : float, [m]
        water depth
    fairlead_radius : float, [m]
        Mooring attachment distance from vessel centerline
    fairlead : float, [m]
        Depth below water for mooring line attachment
    line_length : float, [m]
        Unstretched total mooring line length
    line_diameter : float, [m]
        diameter of mooring line
    line_type : string
        Type of mooring line: chain, chain_stud, nylon, polyester, polypropylene, fiber, wire, iwrc, or custom
    line_mass_density_coeff: float, [kg/m/m^2]
        Line mass density per unit length per diameter^2, if line type is custom
    line_stiffness_coeff: float, [N/m^2]
        Line stiffness (E*A) per diameter^2, if line type is custom
    line_breaking_load_coeff: float, [N/m^2]
        Line minimumum breaking load (MBL) per diameter^2, if line type is custom
    line_cost_rate_coeff: float, [USD/m/m^2]
        Line cost per unit length per diameter^2, if line type is custom
    anchor_radius : float, [m]
        radius from center of spar to mooring anchor point
    anchor_type : string
        Type of anchor for sizing: drag_embedment, suction, plate, micropile, sepla, or custom
    anchor_mass : float, [USD]
        mass for one anchor, if anchor type is custom
    anchor_cost : float, [USD]
        cost for one anchor, if anchor type is custom
    anchor_max_vertical_load : float, [N]
        Maximum tolerable vertical force on anchor, if anchor type is custom
    anchor_max_lateral_load : float, [N]
        Maximum tolerable lateral force on anchor, if anchor type is custom
    max_surge_fraction : float
        Maximum allowable surge offset as a fraction of water depth (0-1)
    operational_heel : float, [deg]
        Maximum angle of heel allowable during operation
    survival_heel : float, [deg]
        max heel angle for turbine survival

    Returns
    -------
    line_mass : float, [kg]
        mass of single mooring line
    mooring_mass : float, [kg]
        total mass of mooring
    mooring_cost : float, [USD]
        total cost for anchor + legs + miscellaneous costs
    mooring_stiffness : numpy array[6, 6], [N/m]
        Linearized stiffness matrix of mooring system at neutral (no offset) conditions.
    mooring_neutral_load : numpy array[NLINES_MAX, 3], [N]
        mooring vertical load in all mooring lines
    max_offset_restoring_force : float, [N]
        sum of forces in x direction after max offset
    operational_heel_restoring_force : numpy array[NLINES_MAX, 3], [N]
        forces for all mooring lines after operational heel
    survival_heel_restoring_force : numpy array[NLINES_MAX, 3], [N]
        forces for all mooring lines after max survival heel
    mooring_plot_matrix : numpy array[NLINES_MAX, NPTS_PLOT, 3], [m]
        data matrix for plotting
    constr_axial_load : float
        range of damaged mooring
    constr_mooring_length : float
        mooring line length ratio to nodal distance
    constr_anchor_vertical : numpy array[n_lines]
        Maximum allowable vertical anchor force minus vertical line tension times safety factor (must be >= 0)
    constr_anchor_lateral : numpy array[n_lines]
        Maximum allowable lateral anchor force minus lateral line tension times safety factor (must be >= 0)

    """

    def initialize(self):
        self.options.declare("options")
        self.options.declare("gamma")

    def setup(self):
        n_lines = self.options["options"]["n_anchors"]
        n_attach = self.options["options"]["n_attach"]

        # Variables local to the class and not OpenMDAO
        self.finput = None
        self.tlpFlag = False

        self.add_input("water_depth", 0.0, units="m")

        # Design variables
        self.add_input("fairlead_radius", 0.0, units="m")
        self.add_input("fairlead", 0.0, units="m")
        self.add_input("line_length", 0.0, units="m")
        self.add_input("line_diameter", 0.0, units="m")

        self.add_input("anchor_radius", 0.0, units="m")
        self.add_input("anchor_mass", 0.0, units="kg")
        self.add_input("anchor_cost", 0.0, units="USD")
        self.add_input("anchor_max_vertical_load", 1e30, units="N")
        self.add_input("anchor_max_lateral_load", 1e30, units="N")

        self.add_input("line_mass_density_coeff", 0.0, units="kg/m**3")
        self.add_input("line_stiffness_coeff", 0.0, units="N/m**2")
        self.add_input("line_breaking_load_coeff", 0.0, units="N/m**2")
        self.add_input("line_cost_rate_coeff", 0.0, units="USD/m**3")

        # User inputs (could be design variables)
        self.add_input("max_surge_fraction", 0.1)
        self.add_input("operational_heel", 0.0, units="rad")
        self.add_input("survival_heel", 0.0, units="rad")

        self.add_output("line_mass", 0.0, units="kg")
        self.add_output("mooring_mass", 0.0, units="kg")
        self.add_output("mooring_cost", 0.0, units="USD")
        self.add_output("mooring_stiffness", np.zeros((6, 6)), units="N/m")
        self.add_output("mooring_neutral_load", np.zeros((n_attach, 3)), units="N")
        self.add_output("max_surge_restoring_force", 0.0, units="N")
        self.add_output("operational_heel_restoring_force", np.zeros(6), units="N")
        self.add_output("survival_heel_restoring_force", np.zeros(6), units="N")
        self.add_output("mooring_plot_matrix", np.zeros((n_lines, NPTS_PLOT, 3)), units="m")

        # Constraints
        self.add_output("constr_axial_load", 0.0)
        self.add_output("constr_mooring_length", 0.0)
        self.add_output("constr_anchor_vertical", np.zeros(n_lines))
        self.add_output("constr_anchor_lateral", np.zeros(n_lines))

    def compute(self, inputs, outputs):
        # Write MAP input file and analyze the system at every angle
        self.evaluate_mooring(inputs, outputs)

        # Compute costs for the system
        self.compute_cost(inputs, outputs)

    def evaluate_mooring(self, inputs, outputs):
        # Unpack variables
        water_depth = float(inputs["water_depth"][0])
        fairlead_depth = float(inputs["fairlead"][0])
        R_fairlead = float(inputs["fairlead_radius"][0])
        R_anchor = float(inputs["anchor_radius"][0])
        heel = float(inputs["operational_heel"][0])
        max_heel = float(inputs["survival_heel"][0])
        d = float(inputs["line_diameter"][0])
        L_mooring = inputs["line_length"]
        gamma = self.options["gamma"]
        n_attach = self.options["options"]["n_attach"]
        n_lines = self.options["options"]["n_anchors"]
        offset = float(inputs["max_surge_fraction"][0]) * water_depth
        n_anchors = self.options["options"]["n_anchors"]
        ratio = int(n_anchors / n_attach)

        line_obj = None
        line_mat = self.options["options"]["line_material"][0]
        if line_mat == "custom":
            min_break_load = float(inputs["line_breaking_load_coeff"][0]) * d**2
            mass_den = float(inputs["line_mass_density_coeff"][0]) * d**2
            ea_stiff = float(inputs["line_stiffness_coeff"][0]) * d**2
            cost_rate = float(inputs["line_cost_rate_coeff"][0]) * d**2
        elif line_mat == "chain_stud":
            line_obj = props.getLineProps(1e3 * d, type="chain", stud="stud")
        else:
            line_obj = props.getLineProps(1e3 * d, type=line_mat)

        if not line_obj is None:
            min_break_load = line_obj.MBL
            mass_den = line_obj.m
            ea_stiff = line_obj.EA
            cost_rate = line_obj.cost

        # Geometric constraints on line length
        if L_mooring > (water_depth - fairlead_depth):
            self.tlpFlag = False

            # Create constraint that line isn't too long that there is no catenary hang
            outputs["constr_mooring_length"] = L_mooring / (0.95 * (R_anchor + water_depth - fairlead_depth))
        else:
            self.tlpFlag = True
            # Create constraint that we don't lose line tension
            outputs["constr_mooring_length"] = L_mooring / (
                (water_depth - fairlead_depth - gamma * R_fairlead * np.sin(max_heel))
            )

        # Create MoorPy input dictionary
        config = {}
        config["water_depth"] = water_depth

        config["points"] = [dict() for k in range(n_attach + n_anchors)]
        angles = np.linspace(0, 2 * np.pi, n_attach + 1)[:n_attach]
        angles -= np.mean(angles)
        fair_x = R_fairlead * np.cos(angles)
        fair_y = R_fairlead * np.sin(angles)
        angles = np.linspace(0, 2 * np.pi, n_anchors + 1)[:n_anchors]
        angles -= np.mean(angles)
        anchor_x = R_anchor * np.cos(angles)
        anchor_y = R_anchor * np.sin(angles)
        for k in range(n_attach):
            config["points"][k]["name"] = f"fairlead{k}"
            config["points"][k]["type"] = "vessel"
            config["points"][k]["location"] = [fair_x[k], fair_y[k], -fairlead_depth]
        for k in range(n_anchors):
            config["points"][k + n_attach]["name"] = f"anchor{k}"
            config["points"][k + n_attach]["type"] = "fixed"
            config["points"][k + n_attach]["location"] = [anchor_x[k], anchor_y[k], -water_depth]

        config["lines"] = [dict() for i in range(n_lines)]
        for k in range(n_lines):
            ifair = np.int_(k / ratio)
            config["lines"][k]["name"] = f"line{k}"
            config["lines"][k]["endA"] = f"fairlead{ifair}"
            config["lines"][k]["endB"] = f"anchor{k}"
            config["lines"][k]["type"] = "myline"
            config["lines"][k]["length"] = L_mooring

        config["line_types"] = [{}]
        config["line_types"][0]["name"] = "myline"
        config["line_types"][0]["diameter"] = d
        config["line_types"][0]["mass_density"] = mass_den
        config["line_types"][0]["stiffness"] = ea_stiff
        config["line_types"][0]["breaking_load"] = min_break_load
        config["line_types"][0]["cost"] = cost_rate
        config["line_types"][0]["transverse_added_mass"] = 0.0
        config["line_types"][0]["tangential_added_mass"] = 0.0
        config["line_types"][0]["transverse_drag"] = 0.0
        config["line_types"][0]["tangential_drag"] = 0.0

        # Create a MoorPy system
        ms = mp.System()
        ms.parseYAML(config)
        ms.bodyList[0].type = -1  # need to make sure it's set to a coupled type
        ms.initialize()

        # Get the stiffness matrix at neutral position
        ms.bodyList[0].setPosition(np.zeros(6))
        ms.solveEquilibrium3()
        outputs["mooring_stiffness"] = ms.getCoupledStiffness(lines_only=True)

        # Get the vertical load in the neutral position
        F_neut = np.zeros((n_attach, 3))
        for k in range(n_attach):
            if np.abs(ms.pointList[k].r[-1] + fairlead_depth) < 0.1:
                F_neut[k, :] = ms.pointList[k].getForces(lines_only=True)
        outputs["mooring_neutral_load"] = F_neut

        # Plotting data
        plotMat = np.zeros((n_lines, NPTS_PLOT, 3))
        for k in range(n_lines):
            Xs, Ys, Zs, _ = ms.lineList[k].getLineCoords(0.0)
            plotMat[k, :, 0] = Xs
            plotMat[k, :, 1] = Ys
            plotMat[k, :, 2] = Zs
        outputs["mooring_plot_matrix"] = plotMat

        # Get the restoring moment at maximum angle of heel
        # Since we don't know the substucture CG, have to just get the forces of the lines now and do the cross product later
        # We also want to allow for arbitraty wind direction and yaw of rotor relative to mooring lines, so we will compare
        # pitch and roll forces as extremes
        F_heel = ms.mooringEq([0, 0, 0, 0, heel, 0], DOFtype="coupled")
        outputs["operational_heel_restoring_force"] = F_heel

        F_maxheel = ms.mooringEq([0, 0, 0, 0, max_heel, 0], DOFtype="coupled")
        outputs["survival_heel_restoring_force"] = F_maxheel

        # Anchor load limits
        F_anch = np.zeros((n_anchors, 3))
        for k in range(n_anchors):
            if np.abs(ms.pointList[k + n_attach].r[0] + water_depth) < 0.1:
                F_anch[k, :] = ms.pointList[k].getForces()
        outputs["constr_anchor_lateral"] = (
            inputs["anchor_max_lateral_load"] - np.sqrt(np.sum(F_anch[:, :-1] ** 2, axis=1)) * gamma
        )
        outputs["constr_anchor_vertical"] = inputs["anchor_max_vertical_load"] - F_anch[:, -1] * gamma

        # Get angles by which to find the weakest line
        dangle = 5.0
        angles = np.deg2rad(np.arange(0.0, 360.0, dangle))
        nangles = angles.size

        # Get restoring force at weakest line at maximum allowable offset
        # Will global minimum always be along mooring angle?
        Frestore = np.zeros(nangles)
        Tmax = np.zeros(nangles)
        Fa = np.zeros(n_lines)
        # Loop around all angles to find weakest point
        for ia, a in enumerate(angles):
            # Unit vector and offset in x-y components
            idir = np.array([np.cos(a), np.sin(a)])
            surge = offset * idir[0]
            sway = offset * idir[1]

            # Get restoring force of offset at this angle
            fbody = ms.mooringEq([surge, sway, 0, 0, 0, 0], DOFtype="coupled")
            Frestore[ia] = np.dot(fbody[:2], idir)
            for k in range(n_lines):
                f = ms.lineList[0].TB
                Fa[k] = np.sqrt(np.sum(f**2))

            Tmax[ia] = np.abs(Fa).max()

        # Store the weakest restoring force when the vessel is offset the maximum amount
        outputs["max_surge_restoring_force"] = np.abs(Frestore).min()

        # Check the highest line tension in those offsets
        outputs["constr_axial_load"] = gamma * Tmax.max() / min_break_load

    def compute_cost(self, inputs, outputs):
        # Unpack variables
        L_mooring = float(inputs["line_length"][0])
        d = float(inputs["line_diameter"][0])
        gamma = self.options["gamma"]

        anchor_type = self.options["options"]["line_anchor"][0]
        if anchor_type == "custom":
            anchor_rate = float(inputs["anchor_cost"][0])
            anchor_mass = float(inputs["anchor_mass"][0])
        else:
            # Do empirical sizing with MoorPy
            fx = (inputs["anchor_max_lateral_load"] - outputs["constr_anchor_lateral"].min()) / gamma
            fz = (inputs["anchor_max_vertical_load"] - outputs["constr_anchor_vertical"].min()) / gamma
            anchor_rate, _, _ = props.getAnchorProps(fx, fz, type=anchor_type.replace("_", "-"))
            anchor_mass = 0.0  # TODO
        n_anchors = n_lines = self.options["options"]["n_anchors"]

        line_obj = None
        line_mat = self.options["options"]["line_material"][0]
        if line_mat == "custom":
            mass_den = float(inputs["line_mass_density_coeff"][0]) * d**2
            cost_rate = float(inputs["line_cost_rate_coeff"][0]) * d**2
        elif line_mat == "chain_stud":
            line_obj = props.getLineProps(1e3 * d, type="chain", stud="stud")
        else:
            line_obj = props.getLineProps(1e3 * d, type=line_mat)

        if not line_obj is None:
            mass_den = line_obj.m
            cost_rate = line_obj.cost

        # Cost of anchors
        anchor_total = anchor_rate * n_anchors

        # Cost of all of the mooring lines
        legs_total = n_lines * cost_rate * L_mooring

        # Total summations
        outputs["mooring_cost"] = legs_total + anchor_total
        outputs["line_mass"] = mass_den * L_mooring
        outputs["mooring_mass"] = (outputs["line_mass"] + anchor_mass) * n_lines
