import numpy as np
import openmdao.api as om
from wisdem.pymap import pyMAP
from wisdem.commonse import gravity

NLINES_MAX = 15
NPTS_PLOT = 20


def lines2nodes(F, nnode):
    nline = F.shape[0]
    ratio = int(nline / nnode)
    if ratio == 1:
        return F
    Fnode = np.zeros((nnode, 3))
    for k in range(nnode):
        Fnode[k, :] = F[(ratio * k) : (ratio * (k + 1)), :].sum(axis=0)
    return Fnode


class MapMooring(om.ExplicitComponent):
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
    anchor_radius : float, [m]
        radius from center of spar to mooring anchor point
    line_diameter : float, [m]
        diameter of mooring line
    anchor_type : string
        SUCTIONPILE or DRAGEMBEDMENT
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
    anchor_cost : float, [USD]
        total cost for anchor
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
    constr_axial_load : float, [m]
        range of damaged mooring
    constr_mooring_length : float
        mooring line length ratio to nodal distance

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

        self.add_input("rho_water", 0.0, units="kg/m**3")
        self.add_input("water_depth", 0.0, units="m")

        # Design variables
        self.add_input("fairlead_radius", 0.0, units="m")
        self.add_input("fairlead", 0.0, units="m")
        self.add_input("line_length", 0.0, units="m")
        self.add_input("line_diameter", 0.0, units="m")
        self.add_input("anchor_radius", 0.0, units="m")
        self.add_input("anchor_cost", 0.0, units="USD")

        self.add_input("line_mass_density_coeff", 0.0, units="kg/m**3")
        self.add_input("line_stiffness_coeff", 0.0, units="N/m**2")
        self.add_input("line_breaking_load_coeff", 0.0, units="N/m**2")
        self.add_input("line_cost_rate_coeff", 0.0, units="USD/m**3")

        # User inputs (could be design variables)
        # self.add_discrete_input("mooring_type", "CHAIN")
        # self.add_discrete_input("anchor_type", "DRAGEMBEDMENT")
        self.add_input("max_surge_fraction", 0.1)
        self.add_input("operational_heel", 0.0, units="deg")
        self.add_input("survival_heel", 0.0, units="deg")

        self.add_output("line_mass", 0.0, units="kg")
        self.add_output("mooring_mass", 0.0, units="kg")
        self.add_output("mooring_cost", 0.0, units="USD")
        self.add_output("mooring_stiffness", np.zeros((6, 6)), units="N/m")
        self.add_output("mooring_neutral_load", np.zeros((n_attach, 3)), units="N")
        self.add_output("max_surge_restoring_force", 0.0, units="N")
        self.add_output("operational_heel_restoring_force", np.zeros((n_lines, 3)), units="N")
        self.add_output("survival_heel_restoring_force", np.zeros((n_lines, 3)), units="N")
        self.add_output("mooring_plot_matrix", np.zeros((n_lines, NPTS_PLOT, 3)), units="m")

        # Constraints
        self.add_output("constr_axial_load", 0.0, units="m")
        self.add_output("constr_mooring_length", 0.0)

    def compute(self, inputs, outputs):
        # Set characteristics based on regressions / empirical data
        # self.set_properties(inputs, discrete_inputs)

        # Set geometry profile
        self.set_geometry(inputs, outputs)

        # Write MAP input file and analyze the system at every angle
        self.runMAP(inputs, outputs)

        # Compute costs for the system
        self.compute_cost(inputs, outputs)

    def set_properties(self, inputs):
        """
        THIS IS NOT USED BUT REMAINS FOR REFERENCE
        Sets mooring line properties: Minimum Breaking Load, Mass per Length,
        Axial Stiffness, Cross-Sectional Area, Cost-per-Length.

        INPUTS:
        ----------
        inputs   : dictionary of input parameters

        OUTPUTS  : Parameters are class variables and are set internally

        References:
        https://daim.idi.ntnu.no/masteroppgaver/015/15116/masteroppgave.pdf
        http://offshoremechanics.asmedigitalcollection.asme.org/article.aspx?articleid=2543338
        https://www.orcina.com/SoftwareProducts/OrcaFlex/Documentation/Help/Content/html/
        Chain.htm
        Chain,AxialandBendingStiffness.htm
        Chain,MechanicalProperties.htm
        RopeWire.htm
        RopeWire,MinimumBreakingLoads.htm
        RopeWire,Massperunitlength.htm
        RopeWire,AxialandBendingStiffness.htm
        """

        # Unpack variables
        Dmooring = inputs["line_diameter"]
        lineType = discrete_inputs["mooring_type"].upper()

        # Set parameters based on regressions for different mooring line type
        Dmooring2 = Dmooring ** 2

        # TODO: Costs per unit length are not synced with new input sources
        if lineType == "CHAIN":
            self.min_break_load = 2.74e7 * Dmooring2 * (44.0 - 80.0 * Dmooring)
            # Use a linear fit to the other fit becuase it is poorly conditioned for optimization
            # self.min_break_load      = 1e3*np.maximum(1.0, -5445.2957034820683+176972.68498888266*Dmooring)
            self.wet_mass_per_length = 19.9e3 * Dmooring2  # From Orca, 7983.34117 OC3 definiton doc
            self.axial_stiffness = 8.54e10 * Dmooring2  # From Orca, 4.74374e10 OC3 definiton doc,
            self.area = 2.0 * 0.25 * np.pi * Dmooring2
            self.cost_per_length = 3.415e4 * Dmooring2  # 0.58*1e-3*self.min_break_load/gravity - 87.6

        elif lineType == "NYLON":
            self.min_break_load = 139357e3 * Dmooring2
            self.wet_mass_per_length = 0.6476e3 * Dmooring2
            self.axial_stiffness = 1.18e8 * Dmooring2
            self.area = 0.25 * np.pi * Dmooring2
            self.cost_per_length = 3.415e4 * Dmooring2  # 0.42059603*1e-3*self.min_break_load/gravity + 109.5

        elif lineType == "POLYESTER":
            self.min_break_load = 170466e3 * Dmooring2
            self.wet_mass_per_length = 0.7978e3 * Dmooring2
            self.axial_stiffness = 1.09e9 * Dmooring2
            self.area = 0.25 * np.pi * Dmooring2
            self.cost_per_length = 3.415e4 * Dmooring2  # 0.42059603*1e-3*self.min_break_load/gravity + 109.5

        elif lineType == "FIBER":  # Wire rope with fiber rope
            self.min_break_load = 584175e3 * Dmooring2
            self.wet_mass_per_length = 3.6109e3 * Dmooring2
            self.axial_stiffness = 3.67e10 * Dmooring2
            self.area = 0.455 * 0.25 * np.pi * Dmooring2
            self.cost_per_length = 2.0 * 6.32e4 * Dmooring2  # 0.53676471*1e-3*self.min_break_load/gravity

        elif lineType == "IWRC":  # Wire rope with steel core
            self.min_break_load = 633358e3 * Dmooring2
            self.wet_mass_per_length = 3.9897e3 * Dmooring2
            self.axial_stiffness = 4.04e10 * Dmooring2
            self.area = 0.455 * 0.25 * np.pi * Dmooring2
            self.cost_per_length = 6.32e4 * Dmooring2  # 0.33*1e-3*self.min_break_load/gravity + 139.5

        else:
            raise ValueError("Available line types are: chain nylon polyester fiber iwrc")

    def set_geometry(self, inputs, outputs):
        # Unpack variables
        fairleadDepth = inputs["fairlead"]
        R_fairlead = inputs["fairlead_radius"]
        R_anchor = inputs["anchor_radius"]
        waterDepth = inputs["water_depth"]
        L_mooring = inputs["line_length"]
        max_heel = inputs["survival_heel"]
        gamma = self.options["gamma"]

        if L_mooring > (waterDepth - fairleadDepth):
            self.tlpFlag = False

            # Create constraint that line isn't too long that there is no catenary hang
            outputs["constr_mooring_length"] = L_mooring / (0.95 * (R_anchor + waterDepth - fairleadDepth))
        else:
            self.tlpFlag = True
            # Create constraint that we don't lose line tension
            outputs["constr_mooring_length"] = L_mooring / (
                (waterDepth - fairleadDepth - gamma * R_fairlead * np.sin(np.deg2rad(max_heel)))
            )

    def write_line_dictionary(self, inputs, cable_sea_friction_coefficient=0.65):
        """Writes LINE DICTIONARY section of input.map file

        INPUTS:
        ----------
        inputs   : dictionary of input parameters
        cable_sea_friction_coefficient : coefficient of friction with sea floor (defaults to 0.65)

        OUTPUTS  : none
        """
        # Unpack variables
        Dmooring = inputs["line_diameter"]
        wet_mass_per_length = inputs["line_mass_density_coeff"] * Dmooring ** 2
        axial_stiffness = inputs["line_stiffness_coeff"] * Dmooring ** 2

        self.finput.append("---------------------- LINE DICTIONARY ---------------------------------------")
        self.finput.append("LineType  Diam      MassDenInAir   EA            CB   CIntDamp  Ca   Cdn    Cdt")
        self.finput.append("(-)       (m)       (kg/m)        (N)           (-)   (Pa-s)    (-)  (-)    (-)")
        self.finput.append(
            "myline   %.5f   %.5f   %.5f   %.5f   1.0E8   0.6   -1.0   0.05"
            % (Dmooring, wet_mass_per_length, axial_stiffness, cable_sea_friction_coefficient)
        )

    def write_node_properties_header(self):
        """Writes NODE PROPERTIES section header of input.map file

        INPUTS: none
        ----------

        OUTPUTS  : none
        """
        self.finput.append("---------------------- NODE PROPERTIES ---------------------------------------")
        # Doesn't like some weird character here somewhere
        # self.finput.append('Node  Type       X       Y       Z      M     B     FX      FY      FZ')
        # self.finput.append('(-)   (-)       (m)     (m)     (m)    (kg)  (m^3)  (N)     (N)     (N)')
        self.finput.append("Node Type X     Y    Z   M     V FX FY FZ")
        self.finput.append("(-)  (-) (m)   (m)  (m) (kg) (m^3) (kN) (kN) (kN)")

    def write_node_properties(
        self,
        number,
        node_type,
        x_pos,
        y_pos,
        z_pos,
        point_mass=0,
        displaced_volume=0,
        x_force=None,
        y_force=None,
        z_force=None,
    ):
        """Writes NODE PROPERTIES data of input.map file.  Nodes are connections between mooring lines and bridles, vessels, and anchors

        INPUTS:
        ----------
        number           : The node number listing in the input file
        node_type        : fix / vessel / connect (fix=anchor)
        x_, y_, z_pos    : position of node in coordinate system (separate inputs)
        point_mass       : see MAP reference (defaults to 0)
        displaced_volume : see MAP reference (defaults to 0)
        x_, y_, z_force  : see MAP reference (defaults to None)

        OUTPUTS  : none
        """
        # Ensure this connection is something MAP understands
        nodeStr = node_type.lower()
        if not nodeStr in ["fix", "connect", "vessel"]:
            raise ValueError("%s is not a valid node type for node %d" % (node_type, number))

        # If this is a node between two lines have to specify connection details
        if nodeStr == "connect":
            try:
                x_force = float(x_force)
                y_force = float(y_force)
                z_force = float(z_force)
            except:
                raise ValueError("%s must have numerical force applied values." % node_type)

        # Set location strings
        forceStr = "#   #   #"
        if nodeStr == "connect":
            forceStr = "%f   %f   %f" % (x_force, y_force, z_force)
            posStr = "#%f   #%f   #%f   " % (x_pos, y_pos, z_pos)
        elif nodeStr == "fix":
            posStr = "%f   %f   depth   " % (x_pos, y_pos)
        elif nodeStr == "vessel":
            posStr = "%f   %f   %f   " % (x_pos, y_pos, z_pos)

        # Write the connection line
        line = "%d   " % number
        line += "%s   " % node_type
        line += posStr
        line += "%f   %f   " % (point_mass, displaced_volume)
        line += forceStr
        self.finput.append(line)

    def write_line_properties(self, inputs, line_number=1, anchor_node=2, fairlead_node=1, flags=""):
        """Writes LINE PROPERTIES section of input.map file that connects multiple nodes

        INPUTS:
        ----------
        inputs        : dictionary of input parameters
        line_number   : Line ID number (defaults to 1)
        anchor_node   : Node number corresponding to anchor (defaults to 1)
        fairlead_node : Node number corresponding to fairlead (vessel) node (defaults to 2)
        flags         : see MAP reference (defaults to empty string ' ')

        OUTPUTS  : none
        """
        # Add flag for taut lines
        if self.tlpFlag:
            flags += " LINEAR SPRING"

        self.finput.append("---------------------- LINE PROPERTIES ---------------------------------------")
        self.finput.append("Line    LineType  UnstrLen  NodeAnch  NodeFair  Flags")
        self.finput.append("(-)      (-)       (m)       (-)       (-)       (-)")
        for k in range(len(line_number)):
            self.finput.append(
                "%d   %s   %f   %d   %d   %s"
                % (
                    line_number[k],
                    "myline",
                    inputs["line_length"],
                    anchor_node[k],
                    fairlead_node[k],
                    flags,
                )
            )

    def write_solver_options(self, inputs):
        """Writes SOLVER OPTIONS section of input.map file,
        which includes repeating node/line arrangement in even angular spacing around structure.

        INPUTS:
        ----------
        inputs        : dictionary of input parameters

        OUTPUTS  : none
        """
        # Unpack variables
        n_attach = self.options["options"]["n_attach"]

        self.finput.append("---------------------- SOLVER OPTIONS-----------------------------------------")
        self.finput.append("Option")
        self.finput.append("(-)")
        self.finput.append("help")
        self.finput.append(" integration_dt 0")
        self.finput.append(" kb_default 3.0e6")
        self.finput.append(" cb_default 3.0e5")
        self.finput.append(" wave_kinematics ")
        self.finput.append("inner_ftol 1e-5")
        self.finput.append("inner_gtol 1e-5")
        self.finput.append("inner_xtol 1e-5")
        self.finput.append("outer_tol 1e-3")
        self.finput.append(" pg_cooked 10000 1")
        self.finput.append(" outer_fd")
        self.finput.append(" outer_bd")
        self.finput.append(" outer_cd")
        self.finput.append(" inner_max_its 200")
        self.finput.append(" outer_max_its 600")
        # Repeat the details for the one mooring line multiple times
        angles = np.linspace(0, 360, n_attach + 1)[1:-1]
        line = "repeat"
        for degree in angles:
            line += " %d" % degree
        self.finput.append(line)
        self.finput.append(" krylov_accelerator 3")
        self.finput.append(" ref_position 0.0 0.0 0.0")

    def write_input_file(self, inputs):
        """Writes SOLVER OPTIONS section of input.map file,
        which includes repeating node/line arrangement in even angular spacing around structure.

        INPUTS:
        ----------
        inputs        : dictionary of input parameters

        OUTPUTS  : none
        """
        # Unpack variables
        fairleadDepth = float(inputs["fairlead"])
        R_fairlead = float(inputs["fairlead_radius"])
        R_anchor = float(inputs["anchor_radius"])
        n_attach = self.options["options"]["n_attach"]
        n_anchors = self.options["options"]["n_anchors"]
        ratio = int(n_anchors / n_attach)

        # Open the map input file
        self.finput = []

        # Write the "Line Dictionary" section
        self.write_line_dictionary(inputs)

        # Write the "Node Properties" section
        self.write_node_properties_header()
        # One end on sea floor the other at fairlead
        self.write_node_properties(1, "VESSEL", R_fairlead, 0, -fairleadDepth)
        if ratio > 1:
            angles = np.linspace(0, 2 * np.pi, n_anchors + 1)[:ratio]
            angles -= np.mean(angles)
            anchorx = R_anchor * np.cos(angles)
            anchory = R_anchor * np.sin(angles)
            for k in range(ratio):
                self.write_node_properties(k + 2, "FIX", anchorx[k], anchory[k], None)
        else:
            self.write_node_properties(2, "FIX", R_anchor, 0, None)

        # Write the "Line Properties" section
        iline = np.arange(ratio) + 1
        ianch = np.arange(ratio) + 2
        ifair = np.ones(iline.shape)
        self.write_line_properties(inputs, line_number=iline, anchor_node=ianch, fairlead_node=ifair)

        # Write the "Solve Options" section
        self.write_solver_options(inputs)

    def runMAP(self, inputs, outputs):
        """Writes MAP input file, executes, and then queries MAP to find
        maximum loading and displacement from vessel displacement around all 360 degrees

        INPUTS:
        ----------
        inputs   : dictionary of input parameters
        outputs : dictionary of output parameters

        OUTPUTS  : none (multiple unknown dictionary values set)
        """
        # Unpack variables
        rhoWater = float(inputs["rho_water"])
        waterDepth = float(inputs["water_depth"])
        fairleadDepth = float(inputs["fairlead"])
        heel = float(inputs["operational_heel"])
        max_heel = inputs["survival_heel"]
        d = inputs["line_diameter"]
        min_break_load = inputs["line_breaking_load_coeff"] * d ** 2
        gamma = self.options["gamma"]
        n_attach = self.options["options"]["n_attach"]
        n_lines = self.options["options"]["n_anchors"]
        offset = float(inputs["max_surge_fraction"]) * waterDepth

        # Write the mooring system input file for this design
        self.write_input_file(inputs)

        # Initiate MAP++ for this design
        mymap = pyMAP()
        # mymap.ierr = 0
        mymap.map_set_sea_depth(waterDepth)
        mymap.map_set_gravity(gravity)
        mymap.map_set_sea_density(rhoWater)
        mymap.read_list_input(self.finput)
        mymap.init()

        # Get the stiffness matrix at neutral position
        mymap.displace_vessel(0, 0, 0, 0, 0, 0)
        mymap.update_states(0.0, 0)
        K = mymap.linear(1e-4)  # Input finite difference epsilon
        outputs["mooring_stiffness"] = np.array(K)
        mymap.displace_vessel(0, 0, 0, 0, 0, 0)
        mymap.update_states(0.0, 0)

        # Get the vertical load on the structure and plotting data
        F_neutral = np.zeros((n_lines, 3))
        plotMat = np.zeros((n_lines, NPTS_PLOT, 3))
        for k in range(n_lines):
            (F_neutral[k, 0], F_neutral[k, 1], F_neutral[k, 2]) = mymap.get_fairlead_force_3d(k)
            plotMat[k, :, 0] = mymap.plot_x(k, NPTS_PLOT)
            plotMat[k, :, 1] = mymap.plot_y(k, NPTS_PLOT)
            plotMat[k, :, 2] = mymap.plot_z(k, NPTS_PLOT)
            if self.tlpFlag:
                # Seems to be a bug in the plot arrays from MAP++ for plotting output with taut lines
                plotMat[k, :, 2] = np.linspace(-fairleadDepth, -waterDepth, NPTS_PLOT)
        outputs["mooring_neutral_load"] = lines2nodes(F_neutral, n_attach)
        outputs["mooring_plot_matrix"] = plotMat

        # Get the restoring moment at maximum angle of heel
        # Since we don't know the substucture CG, have to just get the forces of the lines now and do the cross product later
        # We also want to allow for arbitraty wind direction and yaw of rotor relative to mooring lines, so we will compare
        # pitch and roll forces as extremes
        # TODO: This still isgn't quite the same as clocking the mooring lines in different directions,
        # which is what we want to do, but that requires multiple input files and solutions
        Fh = np.zeros((n_lines, 3))
        mymap.displace_vessel(0, 0, 0, 0, heel, 0)
        mymap.update_states(0.0, 0)
        for k in range(n_lines):
            Fh[k, 0], Fh[k, 1], Fh[k, 2] = mymap.get_fairlead_force_3d(k)
        outputs["operational_heel_restoring_force"] = lines2nodes(Fh, n_attach)

        mymap.displace_vessel(0, 0, 0, 0, max_heel, 0)
        mymap.update_states(0.0, 0)
        for k in range(n_lines):
            Fh[k, 0], Fh[k, 1], Fh[k, 2] = mymap.get_fairlead_force_3d(k)
        outputs["survival_heel_restoring_force"] = lines2nodes(Fh, n_attach)

        # Get angles by which to find the weakest line
        dangle = 5.0
        angles = np.deg2rad(np.arange(0.0, 360.0, dangle))
        nangles = angles.size

        # Get restoring force at weakest line at maximum allowable offset
        # Will global minimum always be along mooring angle?
        Frestore = np.zeros(nangles)
        Tmax = np.zeros(nangles)
        Fa = np.zeros((n_lines, 3))
        # Loop around all angles to find weakest point
        for ia, a in enumerate(angles):
            # Unit vector and offset in x-y components
            idir = np.array([np.cos(a), np.sin(a)])
            surge = offset * idir[0]
            sway = offset * idir[1]

            # Get restoring force of offset at this angle
            mymap.displace_vessel(surge, sway, 0, 0, 0, 0)  # 0s for z, angles
            mymap.update_states(0.0, 0)
            for k in range(n_lines):
                Fa[k, 0], Fa[k, 1], Fa[k, 2] = mymap.get_fairlead_force_3d(k)

            Frestore[ia] = np.dot(Fa[:, :2].sum(axis=0), idir)
            Tmax[ia] = np.sqrt(np.sum(Fa ** 2, axis=1)).max()

        # Store the weakest restoring force when the vessel is offset the maximum amount
        outputs["max_surge_restoring_force"] = Frestore.min()

        # Check for good convergence
        outputs["constr_axial_load"] = gamma * Tmax.max() / min_break_load

        mymap.end()

    def compute_cost(self, inputs, outputs):
        """Computes cost, based on mass scaling, of mooring system.

        INPUTS:
        ----------
        inputs   : dictionary of input parameters
        outputs : dictionary of output parameters

        OUTPUTS  : none (mooring_cost/mass unknown dictionary values set)
        """
        # Unpack variables
        L_mooring = float(inputs["line_length"])
        # anchorType = discrete_inputs["anchor_type"]
        d = float(inputs["line_diameter"])
        cost_per_length = float(inputs["line_cost_rate_coeff"]) * d ** 2
        # min_break_load = inputs['line_breaking_load_coeff'] * d**2
        wet_mass_per_length = float(inputs["line_mass_density_coeff"]) * d ** 2
        anchor_rate = float(inputs["anchor_cost"])
        n_anchors = n_lines = self.options["options"]["n_anchors"]

        # Cost of anchors
        # if anchorType.upper() == "DRAGEMBEDMENT":
        #    anchor_rate = 1e-3 * min_break_load / gravity / 20 * 2000
        # elif anchorType.upper() == "SUCTIONPILE":
        #    anchor_rate = 150000.0 * np.sqrt(1e-3 * min_break_load / gravity / 1250.0)
        # else:
        #    raise ValueError("Anchor Type must be DRAGEMBEDMENT or SUCTIONPILE")
        anchor_total = anchor_rate * n_anchors

        # Cost of all of the mooring lines
        legs_total = n_lines * cost_per_length * L_mooring

        # Total summations
        outputs["mooring_cost"] = legs_total + anchor_total
        outputs["line_mass"] = wet_mass_per_length * L_mooring
        outputs["mooring_mass"] = wet_mass_per_length * L_mooring * n_lines
