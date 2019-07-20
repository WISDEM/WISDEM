from openmdao.api import Component
import numpy as np
import os
import sys
from pymap import pyMAP

from commonse import gravity, Enum
from commonse.utilities import assembleI, unassembleI

Anchor    = Enum('DRAGEMBEDMENT SUCTIONPILE')
NLINES_MAX = 15
NPTS_PLOT = 20

class MapMooring(Component):
    """
    OpenMDAO Component class for mooring system attached to sub-structure of floating offshore wind turbines.
    Should be tightly coupled with Spar class for full system representation.
    """

    def __init__(self):
        super(MapMooring,self).__init__()
    
        # Variables local to the class and not OpenMDAO
        self.min_break_load      = None
        self.wet_mass_per_length = None
        self.axial_stiffness     = None
        self.area                = None
        self.cost_per_length     = None
        self.finput              = None
        self.tlpFlag             = False

        # Environment
        self.add_param('water_density', val=0.0, units='kg/m**3', desc='density of water')
        self.add_param('water_depth', val=0.0, units='m', desc='water depth')

        # Material properties

        # Inputs from SparGeometry
        self.add_param('fairlead_radius', val=0.0, units='m', desc='Outer spar radius at fairlead depth (point of mooring attachment)')
        
        # Design variables
        self.add_param('fairlead', val=0.0, units='m', desc='Depth below water for mooring line attachment')
        self.add_param('mooring_line_length', val=0.0, units='m',desc='Unstretched total mooring line length')
        self.add_param('anchor_radius', val=0.0, units='m', desc='radius from center of spar to mooring anchor point')
        self.add_param('mooring_diameter', val=0.0, units='m',desc='diameter of mooring line')

        # User inputs (could be design variables)
        self.add_param('number_of_mooring_connections', val=3, desc='number of mooring connections on vessel')
        self.add_param('mooring_lines_per_connection', val=1, desc='number of mooring lines per connection')
        self.add_param('mooring_type', val='CHAIN', desc='chain, nylon, polyester, fiber, or iwrc', pass_by_obj=True)
        self.add_param('anchor_type', val='DRAGEMBEDMENT', desc='SUCTIONPILE or DRAGEMBEDMENT', pass_by_obj=True)
        self.add_param('max_offset', val=0.0, units='m',desc='X offsets in discretization')
        self.add_param('operational_heel', val=0.0, units='deg',desc='Maximum angle of heel allowable during operation')
        self.add_param('max_survival_heel', val=0.0, units='deg', desc='max heel angle for turbine survival')
        self.add_param('gamma_f', val=0.0, desc='Safety factor for mooring line tension')

        # Cost rates
        self.add_param('mooring_cost_rate', val=0.0, desc='miscellaneous cost factor in percent')

        # Outputs
        self.add_output('number_of_mooring_lines', val=0, desc='total number of mooring lines')
        self.add_output('mooring_mass', val=0.0, units='kg',desc='total mass of mooring')
        self.add_output('mooring_moments_of_inertia', val=np.zeros(6), units='kg*m**2', desc='mass moment of inertia of mooring system about fairlead-centerline point [xx yy zz xy xz yz]')
        self.add_output('mooring_cost', val=0.0, units='USD',desc='total cost for anchor + legs + miscellaneous costs')
        self.add_output('mooring_stiffness', val=np.zeros((6,6)), units='N/m', desc='Linearized stiffness matrix of mooring system at neutral (no offset) conditions.')
        self.add_output('anchor_cost', val=0.0, units='USD',desc='total cost for anchor')
        self.add_output('mooring_neutral_load', val=np.zeros((NLINES_MAX,3)), units='N',desc='mooring vertical load in all mooring lines')
        self.add_output('max_offset_restoring_force', val=0.0, units='N',desc='sum of forces in x direction after max offset')
        self.add_output('operational_heel_restoring_force', val=np.zeros((NLINES_MAX,3)), units='N',desc='forces for all mooring lines after operational heel')
        self.add_output('mooring_plot_matrix', val=np.zeros((NLINES_MAX, NPTS_PLOT, 3)), units='m', desc='data matrix for plotting') 

        # Output constriants
        self.add_output('axial_unity', val=0.0, units='m',desc='range of damaged mooring')
        self.add_output('mooring_length_max', val=0.0, desc='mooring line length ratio to nodal distance')
        
        # Derivatives
        self.deriv_options['type'] = 'fd'
        self.deriv_options['form'] = 'central'
        self.deriv_options['check_form'] = 'central'
        self.deriv_options['step_calc'] = 'relative'
        self.deriv_options['step_size'] = 1e-5
        
    def solve_nonlinear(self, params, unknowns, resids):
        """Sets mooring line properties then writes MAP input file and executes MAP.
        
        INPUTS:
        ----------
        params   : dictionary of input parameters
        unknowns : dictionary of output parameters
        
        OUTPUTS  : none (all unknown dictionary values set)
        """

        # Set characteristics based on regressions / empirical data
        self.set_properties(params)

        # Set geometry profile
        self.set_geometry(params, unknowns)

        # Write MAP input file and analyze the system at every angle
        self.runMAP(params, unknowns)

        # Compute costs for the system
        self.compute_cost(params, unknowns)


    def set_properties(self, params):
        """Sets mooring line properties: Minimum Breaking Load, Mass per Length, Axial Stiffness, Cross-Sectional Area, Cost-per-Length.
        
        INPUTS:
        ----------
        params   : dictionary of input parameters
        
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
        Dmooring       = params['mooring_diameter']
        lineType       = params['mooring_type'].upper()

        # Set parameters based on regressions for different mooring line type
        Dmooring2 = Dmooring**2

        # TODO: Costs per unit length are not synced with new input sources
        if lineType == 'CHAIN':
            self.min_break_load      = 2.74e7  * Dmooring2 * (44.0 - 80.0*Dmooring)
            # Use a linear fit to the other fit becuase it is poorly conditioned for optimization
            #self.min_break_load      = 1e3*np.maximum(1.0, -5445.2957034820683+176972.68498888266*Dmooring)
            self.wet_mass_per_length = 19.9e3  * Dmooring2 # From Orca, 7983.34117 OC3 definiton doc
            self.axial_stiffness     = 8.54e10 * Dmooring2 # From Orca, 4.74374e10 OC3 definiton doc, 
            self.area                = 2.0 * 0.25 * np.pi * Dmooring2
            self.cost_per_length     = 3.415e4  * Dmooring2 #0.58*1e-3*self.min_break_load/gravity - 87.6

        elif lineType == 'NYLON':
            self.min_break_load      = 139357e3 * Dmooring2
            self.wet_mass_per_length = 0.6476e3 * Dmooring2
            self.axial_stiffness     = 1.18e8   * Dmooring2
            self.area                = 0.25 * np.pi * Dmooring2
            self.cost_per_length     = 3.415e4  * Dmooring2 #0.42059603*1e-3*self.min_break_load/gravity + 109.5

        elif lineType == 'POLYESTER':
            self.min_break_load      = 170466e3 * Dmooring2
            self.wet_mass_per_length = 0.7978e3 * Dmooring2
            self.axial_stiffness     = 1.09e9   * Dmooring2
            self.area                = 0.25 * np.pi * Dmooring2
            self.cost_per_length     = 3.415e4  * Dmooring2 #0.42059603*1e-3*self.min_break_load/gravity + 109.5

        elif lineType == 'FIBER': # Wire rope with fiber rope
            self.min_break_load      = 584175e3 * Dmooring2
            self.wet_mass_per_length = 3.6109e3 * Dmooring2
            self.axial_stiffness     = 3.67e10  * Dmooring2
            self.area                = 0.455 * 0.25 * np.pi * Dmooring2
            self.cost_per_length     = 2.0 * 6.32e4  * Dmooring2 #0.53676471*1e-3*self.min_break_load/gravity

        elif lineType == 'IWRC': # Wire rope with steel core
            self.min_break_load      = 633358e3 * Dmooring2
            self.wet_mass_per_length = 3.9897e3 * Dmooring2
            self.axial_stiffness     = 4.04e10  * Dmooring2
            self.area                = 0.455 * 0.25 * np.pi * Dmooring2
            self.cost_per_length     = 6.32e4  * Dmooring2 #0.33*1e-3*self.min_break_load/gravity + 139.5

        else:
            raise ValueError('Available line types are: chain nylon polyester fiber iwrc')

            
    def set_geometry(self, params, unknowns):
        # Unpack variables
        fairleadDepth = params['fairlead']
        R_fairlead    = params['fairlead_radius']
        R_anchor      = params['anchor_radius']
        waterDepth    = params['water_depth']
        L_mooring     = params['mooring_line_length']
        max_heel      = params['max_survival_heel']
        gamma         = params['gamma_f']
        
        if L_mooring > (waterDepth - fairleadDepth):
            self.tlpFlag = False
            
            # Create constraint that line isn't too long that there is no catenary hang
            unknowns['mooring_length_max'] = L_mooring / (0.95 * (R_anchor + waterDepth - fairleadDepth) )
        else:
            self.tlpFlag = True
            # Create constraint that we don't lose line tension
            unknowns['mooring_length_max'] = L_mooring / ( (waterDepth - fairleadDepth - gamma*R_fairlead*np.sin(np.deg2rad(max_heel))) )

    
    def write_line_dictionary(self, params, cable_sea_friction_coefficient=0.65):
        """Writes LINE DICTIONARY section of input.map file
        
        INPUTS:
        ----------
        params   : dictionary of input parameters
        cable_sea_friction_coefficient : coefficient of friction with sea floor (defaults to 0.65)
        
        OUTPUTS  : none
        """
        # Unpack variables
        rhoWater = params['water_density']
        lineType = params['mooring_type'].lower()
        Dmooring = params['mooring_diameter']
        
        self.finput.append('---------------------- LINE DICTIONARY ---------------------------------------')
        self.finput.append('LineType  Diam      MassDenInAir   EA            CB   CIntDamp  Ca   Cdn    Cdt')
        self.finput.append('(-)       (m)       (kg/m)        (N)           (-)   (Pa-s)    (-)  (-)    (-)')
        self.finput.append('%s   %.5f   %.5f   %.5f   %.5f   1.0E8   0.6   -1.0   0.05' %
                          (lineType, Dmooring, self.wet_mass_per_length, self.axial_stiffness, cable_sea_friction_coefficient) )

        
    def write_node_properties_header(self):
        """Writes NODE PROPERTIES section header of input.map file
        
        INPUTS: none 
        ----------
        
        OUTPUTS  : none
        """
        self.finput.append('---------------------- NODE PROPERTIES ---------------------------------------')
        # Doesn't like some weird character here somewhere
        #self.finput.append('Node  Type       X       Y       Z      M     B     FX      FY      FZ')
        #self.finput.append('(-)   (-)       (m)     (m)     (m)    (kg)  (m^3)  (N)     (N)     (N)')
        self.finput.append('Node Type X     Y    Z   M     V FX FY FZ')
        self.finput.append('(-)  (-) (m)   (m)  (m) (kg) (m^3) (kN) (kN) (kN)')



    def write_node_properties(self, number, node_type, x_pos, y_pos, z_pos,
                              point_mass=0, displaced_volume=0,
                              x_force=None, y_force=None, z_force=None):
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
        if not nodeStr in ['fix', 'connect', 'vessel']:
            raise ValueError('%s is not a valid node type for node %d' % (node_type, number))

        # If this is a node between two lines have to specify connection details
        if nodeStr == 'connect':
            try:
                x_force = float(x_force)
                y_force = float(y_force)
                z_force = float(z_force)
            except:
                raise ValueError('%s must have numerical force applied values.' % node_type)

        # Set location strings
        forceStr = '#   #   #' 
        if nodeStr == 'connect':
            forceStr = '%f   %f   %f' % (x_force, y_force, z_force)
            posStr   = '#%f   #%f   #%f   ' % (x_pos, y_pos, z_pos)
        elif nodeStr == 'fix':
            posStr   = '%f   %f   depth   ' % (x_pos, y_pos)
        elif nodeStr == 'vessel':
            posStr   = '%f   %f   %f   ' % (x_pos, y_pos, z_pos)

        # Write the connection line
        line = ('%d   ' % number)
        line += ('%s   ' % node_type)
        line += (posStr)
        line += ('%f   %f   ' % (point_mass, displaced_volume) )
        line += (forceStr)
        self.finput.append(line)

    def write_line_properties(self, params, line_number=1, anchor_node=2, fairlead_node=1, flags=''):
        """Writes LINE PROPERTIES section of input.map file that connects multiple nodes
        
        INPUTS:
        ----------
        params        : dictionary of input parameters- only 'mooring_type' is used
        line_number   : Line ID number (defaults to 1)
        anchor_node   : Node number corresponding to anchor (defaults to 1)
        fairlead_node : Node number corresponding to fairlead (vessel) node (defaults to 2)
        flags         : see MAP reference (defaults to empty string ' ')
        
        OUTPUTS  : none
        """
        # Add flag for taut lines
        if self.tlpFlag:
            flags += ' LINEAR SPRING'
            
        self.finput.append('---------------------- LINE PROPERTIES ---------------------------------------')
        self.finput.append('Line    LineType  UnstrLen  NodeAnch  NodeFair  Flags')
        self.finput.append('(-)      (-)       (m)       (-)       (-)       (-)')
        self.finput.append('%d   %s   %f   %d   %d   %s' %
                          (line_number, params['mooring_type'], params['mooring_line_length'], anchor_node, fairlead_node, flags) )

        
    def write_solver_options(self, params):
        """Writes SOLVER OPTIONS section of input.map file, 
        which includes repeating node/line arrangement in even angular spacing around structure.
        
        INPUTS:
        ----------
        params        : dictionary of input parameters
        
        OUTPUTS  : none
        """
        # Unpack variables
        n_connect     = max(1, int(params['number_of_mooring_connections']))
        
        self.finput.append('---------------------- SOLVER OPTIONS-----------------------------------------')
        self.finput.append('Option')
        self.finput.append('(-)')
        self.finput.append('help')
        self.finput.append(' integration_dt 0')
        self.finput.append(' kb_default 3.0e6')
        self.finput.append(' cb_default 3.0e5')
        self.finput.append(' wave_kinematics ')
        self.finput.append('inner_ftol 1e-5')
        self.finput.append('inner_gtol 1e-5')
        self.finput.append('inner_xtol 1e-5')
        self.finput.append('outer_tol 1e-3')
        self.finput.append(' pg_cooked 10000 1')
        self.finput.append(' outer_fd')
        self.finput.append(' outer_bd')
        self.finput.append(' outer_cd')
        self.finput.append(' inner_max_its 200')
        self.finput.append(' outer_max_its 600')
        # Repeat the details for the one mooring line multiple times
        angles = np.linspace(0, 360, n_connect+1)[1:-1]
        line = 'repeat'
        for degree in angles:
            line += (' %d' % degree)
        self.finput.append(line)
        self.finput.append(' krylov_accelerator 3')
        self.finput.append(' ref_position 0.0 0.0 0.0')

        
    def write_input_file(self, params):
        """Writes SOLVER OPTIONS section of input.map file, 
        which includes repeating node/line arrangement in even angular spacing around structure.
        
        INPUTS:
        ----------
        params        : dictionary of input parameters
        
        OUTPUTS  : none
        """
        # Unpack variables
        fairleadDepth = params['fairlead']
        R_fairlead    = params['fairlead_radius']
        R_anchor      = params['anchor_radius']
        n_connect     = int(params['number_of_mooring_connections'])
        n_lines       = int(params['mooring_lines_per_connection'])
        ntotal        = n_connect * n_lines
            
        
        # Open the map input file
        self.finput = []

        # Write the "Line Dictionary" section
        self.write_line_dictionary(params)

        # Write the "Node Properties" section
        self.write_node_properties_header()
        # One end on sea floor the other at fairlead
        self.write_node_properties(1, "VESSEL", R_fairlead, 0, -fairleadDepth)
        if n_lines > 1:
            angles  = np.linspace(0, 2*np.pi, ntotal+1)[:n_lines]
            angles -= np.mean(angles)
            anchorx = R_anchor * np.cos( angles )
            anchory = R_anchor * np.sin( angles )
            for k in range(n_lines):
                self.write_node_properties(k+2, "FIX", anchorx[k], anchory[k], None)
        else:
            self.write_node_properties(2, "FIX", R_anchor, 0, None)

        # Write the "Line Properties" section
        for k in range(n_lines):
            self.write_line_properties(params, line_number=k+1, anchor_node=k+2, fairlead_node=1)

        # Write the "Solve Options" section
        self.write_solver_options(params)
        
        
    def runMAP(self, params, unknowns):
        """Writes MAP input file, executes, and then queries MAP to find 
        maximum loading and displacement from vessel displacement around all 360 degrees
        
        INPUTS:
        ----------
        params   : dictionary of input parameters
        unknowns : dictionary of output parameters
        
        OUTPUTS  : none (multiple unknown dictionary values set)
        """
        # Unpack variables
        rhoWater      = params['water_density']
        waterDepth    = params['water_depth']
        fairleadDepth = params['fairlead']
        Dmooring      = params['mooring_diameter']
        offset        = params['max_offset']
        heel          = params['operational_heel']
        gamma         = params['gamma_f']
        n_connect     = int(params['number_of_mooring_connections'])
        n_lines       = int(params['mooring_lines_per_connection'])
        ntotal        = n_connect * n_lines

        # Write the mooring system input file for this design
        self.write_input_file(params)

        # Initiate MAP++ for this design
        mymap = pyMAP( )
        #mymap.ierr = 0
        mymap.map_set_sea_depth(waterDepth)
        mymap.map_set_gravity(gravity)
        mymap.map_set_sea_density(rhoWater)
        mymap.read_list_input(self.finput)
        mymap.init( )

        # Get the stiffness matrix at neutral position
        mymap.displace_vessel(0, 0, 0, 0, 0, 0)
        mymap.update_states(0.0, 0)
        K = mymap.linear(1e-4) # Input finite difference epsilon
        unknowns['mooring_stiffness'] = np.array( K )
        mymap.displace_vessel(0, 0, 0, 0, 0, 0)
        mymap.update_states(0.0, 0)
        
        # Get the vertical load on the structure and plotting data
        F_neutral = np.zeros((NLINES_MAX, 3))
        plotMat   = np.zeros((NLINES_MAX, NPTS_PLOT, 3))
        nptsMOI   = 100
        xyzpts    = np.zeros((ntotal, nptsMOI, 3)) # For MOI calculation
        for k in range(ntotal):
            (F_neutral[k,0], F_neutral[k,1], F_neutral[k,2]) = mymap.get_fairlead_force_3d(k)
            plotMat[k,:,0] = mymap.plot_x(k, NPTS_PLOT)
            plotMat[k,:,1] = mymap.plot_y(k, NPTS_PLOT)
            plotMat[k,:,2] = mymap.plot_z(k, NPTS_PLOT)
            xyzpts[k,:,0]  = mymap.plot_x(k, nptsMOI)
            xyzpts[k,:,1]  = mymap.plot_y(k, nptsMOI)
            xyzpts[k,:,2]  = mymap.plot_z(k, nptsMOI)
            if self.tlpFlag:
                # Seems to be a bug in the plot arrays from MAP++ for plotting output with taut lines
                plotMat[k,:,2] = np.linspace(-fairleadDepth, -waterDepth, NPTS_PLOT)
                xyzpts[k,:,2]  = np.linspace(-fairleadDepth, -waterDepth, nptsMOI)
        unknowns['mooring_neutral_load'] = F_neutral
        unknowns['mooring_plot_matrix']  = plotMat
        
        # Fine line segment length, ds = sqrt(dx^2 + dy^2 + dz^2)
        xyzpts_dx = np.gradient(xyzpts[:,:,0], axis=1)
        xyzpts_dy = np.gradient(xyzpts[:,:,1], axis=1)
        xyzpts_dz = np.gradient(xyzpts[:,:,2], axis=1)
        xyzpts_ds = np.sqrt(xyzpts_dx**2 + xyzpts_dy**2 + xyzpts_dz**2)

        # Initialize inertia tensor integrands in https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor
        # Taking MOI relative to body centerline at fairlead depth
        r0 = np.array([0.0, 0.0, -fairleadDepth])
        R  = np.zeros((ntotal, nptsMOI, 6))
        for ii in range(nptsMOI):
            for k in range(ntotal):
                r = xyzpts[k,ii,:] - r0
                R[k,ii,:] = unassembleI(np.dot(r,r)*np.eye(3) - np.outer(r,r))
        Imat = self.wet_mass_per_length * np.trapz(R, x=xyzpts_ds[:,:,np.newaxis], axis=1)
        unknowns['mooring_moments_of_inertia'] = np.abs( Imat.sum(axis=0) )

        # Get the restoring moment at maximum angle of heel
        # Since we don't know the substucture CG, have to just get the forces of the lines now and do the cross product later
        # We also want to allow for arbitraty wind direction and yaw of rotor relative to mooring lines, so we will compare
        # pitch and roll forces as extremes
        # TODO: This still isgn't quite the same as clocking the mooring lines in different directions,
        # which is what we want to do, but that requires multiple input files and solutions
        Fh = np.zeros((NLINES_MAX,3))
        mymap.displace_vessel(0, 0, 0, 0, heel, 0)
        mymap.update_states(0.0, 0)
        for k in range(ntotal):
            Fh[k][0], Fh[k][1], Fh[k][2] = mymap.get_fairlead_force_3d(k)

        unknowns['operational_heel_restoring_force'] = Fh

        # Get angles by which to find the weakest line
        dangle  = 2.0
        angles  = np.deg2rad( np.arange(0.0, 360.0, dangle) )
        nangles = len(angles)

        # Get restoring force at weakest line at maximum allowable offset
        # Will global minimum always be along mooring angle?
        max_tension   = 0.0
        max_angle     = None
        min_angle     = None
        F_max_tension = None
        F_min         = np.inf
        T = np.zeros((NLINES_MAX,))
        F = np.zeros((NLINES_MAX,))
        # Loop around all angles to find weakest point
        for a in angles:
            # Unit vector and offset in x-y components
            idir  = np.array([np.cos(a), np.sin(a)])
            surge = offset * idir[0]
            sway  = offset * idir[1]

            # Get restoring force of offset at this angle
            mymap.displace_vessel(surge, sway, 0, 0, 0, 0) # 0s for z, angles
            mymap.update_states(0.0, 0)
            for k in range(ntotal):
                # Force in x-y-z coordinates
                fx,fy,fz = mymap.get_fairlead_force_3d(k)
                T[k]     = np.sqrt(fx*fx + fy*fy + fz*fz)
                # Total restoring force
                F[k]     = np.dot([fx, fy], idir)
                
            # Check if this is the weakest direction (highest tension)
            tempMax = T.max()
            if tempMax > max_tension:
                max_tension   = tempMax
                F_max_tension = F.sum()
                max_angle     = a
            if F.sum() < F_min:
                F_min     = F.sum()
                min_angle = a
                
        # Store the weakest restoring force when the vessel is offset the maximum amount
        unknowns['max_offset_restoring_force'] = F_min

        # Check for good convergence
        if (plotMat[0,-1,-1] + fairleadDepth) > 1.0:
            unknowns['axial_unity'] = 1e30
        else:
            unknowns['axial_unity'] = gamma * max_tension / self.min_break_load

        mymap.end()

        
    def compute_cost(self, params, unknowns):
        """Computes cost, based on mass scaling, of mooring system.
        
        INPUTS:
        ----------
        params   : dictionary of input parameters
        unknowns : dictionary of output parameters
        
        OUTPUTS  : none (mooring_cost/mass unknown dictionary values set)
        """
        # Unpack variables
        rhoWater      = params['water_density']
        L_mooring     = params['mooring_line_length']
        anchorType    = params['anchor_type']
        costFact      = params['mooring_cost_rate']
        n_connect     = int(params['number_of_mooring_connections'])
        n_lines       = int(params['mooring_lines_per_connection'])
        ntotal        = n_connect * n_lines
        
        # Cost of anchors
        if type(anchorType) == type(''): anchorType = Anchor[anchorType.upper()]
        if anchorType == Anchor['DRAGEMBEDMENT']:
            anchor_rate = 1e-3 * self.min_break_load / gravity / 20*2000
        elif anchorType  == Anchor['SUCTIONPILE']:
            anchor_rate = 150000.* np.sqrt(1e-3*self.min_break_load/gravity/1250.)
        else:
            raise ValueError('Anchor Type must be DRAGEMBEDMENT or SUCTIONPILE')
        anchor_total = anchor_rate*ntotal

        # Cost of all of the mooring lines
        legs_total = ntotal * self.cost_per_length * L_mooring

        # Total summations
        unknowns['anchor_cost']  = anchor_total
        unknowns['mooring_cost'] = costFact*(legs_total + anchor_total)
        unknowns['mooring_mass'] = self.wet_mass_per_length*L_mooring*ntotal
        unknowns['number_of_mooring_lines'] = ntotal
