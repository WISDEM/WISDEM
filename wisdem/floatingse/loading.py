
from openmdao.api import Component, Group, IndepVarComp
import numpy as np
import pyframe3dd.frame3dd as frame3dd
from commonse.utilities import nodal2sectional

from commonse import gravity, eps, Tube, NFREQ
import commonse.UtilizationSupplement as util
import commonse.manufacturing as manufacture
from commonse.WindWaveDrag import AeroHydroLoads, CylinderWindDrag, CylinderWaveDrag
from commonse.environment import WaveBase, PowerWind
from commonse.vertical_cylinder import CylinderDiscretization, CylinderMass
from .map_mooring import NLINES_MAX


def find_nearest(array,value):
    return (np.abs(array-value)).argmin() 

        
def ghostNodes(x1, x2, r1, r2):
    dx = x2 - x1
    L = np.sqrt( np.sum( dx**2 ) )
    dr1 = (      r1/L) * dx + x1
    dr2 = (1.0 - r2/L) * dx + x1
    return dr1, dr2

class FloatingFrame(Component):
    """
    OpenMDAO Component class for semisubmersible pontoon / truss structure for floating offshore wind turbines.
    Should be tightly coupled with Semi and Mooring classes for full system representation.
    """

    def __init__(self, nFull, nFullTow):
        super(FloatingFrame,self).__init__()

        # Keep Frame3DD data object for easy testing and debugging
        self.myframe = None
        
        # Environment
        self.add_param('water_density', val=0.0, units='kg/m**3', desc='density of water')

        # Material properties
        self.add_param('material_density', val=0., units='kg/m**3', desc='density of material')
        self.add_param('E', val=0.0, units='Pa', desc='Modulus of elasticity (Youngs) of material')
        self.add_param('G', val=0.0, units='Pa', desc='Shear modulus of material')
        self.add_param('yield_stress', val=0.0, units='Pa', desc='yield stress of material')
        self.add_param('Hs', val=0.0, units='m', desc='wave significant height')

        # Base column
        self.add_param('main_z_full', val=np.zeros((nFull,)), units='m', desc='z-coordinates of section nodes (length = nsection+1)')
        self.add_param('main_d_full', val=np.zeros((nFull,)), units='m', desc='outer radius at each section node bottom to top (length = nsection + 1)')
        self.add_param('main_t_full', val=np.zeros((nFull-1,)), units='m', desc='shell wall thickness at each section node bottom to top (length = nsection + 1)')
        self.add_param('main_mass', val=np.zeros((nFull-1,)), units='kg', desc='mass of main column by section')
        self.add_param('main_buckling_length', val=np.zeros((nFull-1,)), units='m', desc='distance between ring stiffeners')
        self.add_param('main_displaced_volume', val=np.zeros((nFull-1,)), units='m**3', desc='column volume of water displaced by section')
        self.add_param('main_hydrostatic_force', val=np.zeros((nFull-1,)), units='N', desc='Net z-force of hydrostatic pressure by section')
        self.add_param('main_center_of_buoyancy', val=0.0, units='m', desc='z-position of center of column buoyancy force')
        self.add_param('main_center_of_mass', val=0.0, units='m', desc='z-position of center of column mass')
        self.add_param('main_Px', np.zeros(nFull), units='N/m', desc='force per unit length in x-direction on main')
        self.add_param('main_Py', np.zeros(nFull), units='N/m', desc='force per unit length in y-direction on main')
        self.add_param('main_Pz', np.zeros(nFull), units='N/m', desc='force per unit length in z-direction on main')
        self.add_param('main_qdyn', np.zeros(nFull), units='N/m**2', desc='dynamic pressure on main')

        self.add_param('main_pontoon_attach_upper', val=0.0, desc='Fraction of main column for upper truss attachment on main column')
        self.add_param('main_pontoon_attach_lower', val=0.0, desc='Fraction of main column lower truss attachment on main column')

        # offset columns
        self.add_param('offset_z_full', val=np.zeros((nFull,)), units='m', desc='z-coordinates of section nodes (length = nsection+1)')
        self.add_param('offset_d_full', val=np.zeros((nFull,)), units='m', desc='outer radius at each section node bottom to top (length = nsection + 1)')
        self.add_param('offset_t_full', val=np.zeros((nFull-1,)), units='m', desc='shell wall thickness at each section node bottom to top (length = nsection + 1)')
        self.add_param('offset_mass', val=np.zeros((nFull-1,)), units='kg', desc='mass of offset column by section')
        self.add_param('offset_buckling_length', val=np.zeros((nFull-1,)), units='m', desc='distance between ring stiffeners')
        self.add_param('offset_displaced_volume', val=np.zeros((nFull-1,)), units='m**3', desc='column volume of water displaced by section')
        self.add_param('offset_hydrostatic_force', val=np.zeros((nFull-1,)), units='N', desc='Net z-force of hydrostatic pressure by section')
        self.add_param('offset_center_of_buoyancy', val=0.0, units='m', desc='z-position of center of column buoyancy force')
        self.add_param('offset_center_of_mass', val=0.0, units='m', desc='z-position of center of column mass')
        self.add_param('offset_Px', np.zeros(nFull), units='N/m', desc='force per unit length in x-direction on offset')
        self.add_param('offset_Py', np.zeros(nFull), units='N/m', desc='force per unit length in y-direction on offset')
        self.add_param('offset_Pz', np.zeros(nFull), units='N/m', desc='force per unit length in z-direction on offset')
        self.add_param('offset_qdyn', np.zeros(nFull), units='N/m**2', desc='dynamic pressure on offset')

        # Tower
        self.add_param('tower_z_full', val=np.zeros((nFullTow,)), units='m', desc='z-coordinates of section nodes (length = nsection+1)')
        self.add_param('tower_d_full', val=np.zeros((nFullTow,)), units='m', desc='outer radius at each section node bottom to top (length = nsection + 1)')
        self.add_param('tower_t_full', val=np.zeros((nFullTow-1,)), units='m', desc='shell wall thickness at each section node bottom to top (length = nsection + 1)')
        self.add_param('tower_mass_section', val=np.zeros((nFullTow-1,)), units='kg', desc='mass of tower column by section')
        self.add_param('tower_buckling_length', 0.0, units='m', desc='buckling length')
        self.add_param('tower_center_of_mass', val=0.0, units='m', desc='z-position of center of tower mass')
        self.add_param('tower_Px', np.zeros(nFullTow), units='N/m', desc='force per unit length in x-direction on tower')
        self.add_param('tower_Py', np.zeros(nFullTow), units='N/m', desc='force per unit length in y-direction on tower')
        self.add_param('tower_Pz', np.zeros(nFullTow), units='N/m', desc='force per unit length in z-direction on tower')
        self.add_param('tower_qdyn', np.zeros(nFullTow), units='N/m**2', desc='dynamic pressure on tower')
        
        # Semi geometry
        self.add_param('radius_to_offset_column', val=0.0, units='m',desc='Distance from main column centerpoint to offset column centerpoint')
        self.add_param('number_of_offset_columns', val=3, desc='Number of offset columns evenly spaced around main column')

        # Pontoon properties
        self.add_param('pontoon_outer_diameter', val=0.0, units='m',desc='Outer radius of tubular pontoon that connects offset or main columns')
        self.add_param('pontoon_wall_thickness', val=0.0, units='m',desc='Inner radius of tubular pontoon that connects offset or main columns')
        self.add_param('cross_attachment_pontoons', val=True, desc='Inclusion of pontoons that connect the bottom of the central main to the tops of the outer offset columns', pass_by_obj=True)
        self.add_param('lower_attachment_pontoons', val=True, desc='Inclusion of pontoons that connect the central main to the outer offset columns at their bottoms', pass_by_obj=True)
        self.add_param('upper_attachment_pontoons', val=True, desc='Inclusion of pontoons that connect the central main to the outer offset columns at their tops', pass_by_obj=True)
        self.add_param('lower_ring_pontoons', val=True, desc='Inclusion of pontoons that ring around outer offset columns at their bottoms', pass_by_obj=True)
        self.add_param('upper_ring_pontoons', val=True, desc='Inclusion of pontoons that ring around outer offset columns at their tops', pass_by_obj=True)
        self.add_param('outer_cross_pontoons', val=True, desc='Inclusion of pontoons that ring around outer offset columns at their tops', pass_by_obj=True)
        
        # Turbine parameters
        self.add_param('rna_mass', val=0.0, units='kg', desc='mass of tower')
        self.add_param('rna_cg', val=np.zeros(3), units='m', desc='Location of RNA center of mass relative to tower top')
        self.add_param('rna_force', val=np.zeros(3), units='N', desc='Force in xyz-direction on turbine')
        self.add_param('rna_moment', val=np.zeros(3), units='N*m', desc='Moments about turbine main')
        self.add_param('rna_I', val=np.zeros(6), units='kg*m**2', desc='Moments about turbine main')

        # Mooting parameters for loading
        self.add_param('number_of_mooring_connections', val=3, desc='number of mooring connections on vessel')
        self.add_param('mooring_lines_per_connection', val=1, desc='number of mooring lines per connection')
        self.add_param('mooring_neutral_load', val=np.zeros((NLINES_MAX,3)), units='N', desc='z-force of mooring lines on structure')
        self.add_param('mooring_stiffness', val=np.zeros((6,6)), units='N/m', desc='Linearized stiffness matrix of mooring system at neutral (no offset) conditions.')
        self.add_param('mooring_moments_of_inertia', val=np.zeros(6), units='kg*m**2', desc='mass moment of inertia of mooring system about fairlead-centerline point [xx yy zz xy xz yz]')
        self.add_param('fairlead', val=0.0, units='m', desc='Depth below water for mooring line attachment')
        self.add_param('fairlead_radius', val=0.0, units='m',desc='Radius from center of structure to fairlead connection points')
        self.add_param('fairlead_support_outer_diameter', val=0.0, units='m',desc='fairlead support outer diameter')
        self.add_param('fairlead_support_wall_thickness', val=0.0, units='m',desc='fairlead support wall thickness')
        
        # safety factors
        self.add_param('gamma_f', 0.0, desc='safety factor on loads')
        self.add_param('gamma_m', 0.0, desc='safety factor on materials')
        self.add_param('gamma_n', 0.0, desc='safety factor on consequence of failure')
        self.add_param('gamma_b', 0.0, desc='buckling safety factor')
        self.add_param('gamma_fatigue', 0.0, desc='total safety factor for fatigue')

        # Manufacturing
        self.add_param('connection_ratio_max', val=0.0, desc='Maximum ratio of pontoon outer diameter to main/offset outer diameter')
        
        # Costing
        self.add_param('material_cost_rate', 0.0, units='USD/kg', desc='Raw material cost rate: steel $1.1/kg, aluminum $3.5/kg')
        self.add_param('labor_cost_rate', 0.0, units='USD/min', desc='Labor cost rate')
        self.add_param('painting_cost_rate', 0.0, units='USD/m/m', desc='Painting / surface finishing cost rate')
        
        # Outputs
        self.add_output('pontoon_wave_height_depth_margin', val=np.zeros(2), units='m', desc='Distance between attachment point of pontoons and wave crest- both above and below waterline')
        self.add_output('pontoon_cost', val=0.0, units='USD', desc='Cost of pontoon elements and connecting truss')
        self.add_output('pontoon_cost_rate', val=0.0, units='USD/t', desc='Cost rate of finished pontoon and truss')
        self.add_output('pontoon_mass', val=0.0, units='kg', desc='Mass of pontoon elements and connecting truss')
        self.add_output('pontoon_displacement', val=0.0, units='m**3', desc='Buoyancy force of submerged pontoon elements')
        self.add_output('pontoon_center_of_buoyancy', val=0.0, units='m', desc='z-position of center of pontoon buoyancy force')
        self.add_output('pontoon_center_of_mass', val=0.0, units='m', desc='z-position of center of pontoon mass')

        self.add_output('top_deflection', 0.0, units='m', desc='Deflection of tower top in yaw-aligned +x direction')
        self.add_output('pontoon_stress', val=np.zeros((70,)), desc='Utilization (<1) of von Mises stress by yield stress and safety factor for all pontoon elements')

        self.add_output('main_stress', np.zeros(nFull-1), desc='Von Mises stress utilization along main column at specified locations. Incudes safety factor.')
        self.add_output('main_stress:axial', np.zeros(nFull-1), desc='Axial stress along main column at specified locations.')
        self.add_output('main_stress:shear', np.zeros(nFull-1), desc='Shear stress along main column at specified locations.')
        self.add_output('main_stress:hoop', np.zeros(nFull-1), desc='Hoop stress along main column at specified locations.')
        self.add_output('main_stress:hoopStiffen', np.zeros(nFull-1), desc='Hoop stress along main column at specified locations.')
        self.add_output('main_shell_buckling', np.zeros(nFull-1), desc='Shell buckling constraint. Should be < 1 for feasibility. Includes safety factors')
        self.add_output('main_global_buckling', np.zeros(nFull-1), desc='Global buckling constraint. Should be < 1 for feasibility. Includes safety factors')

        self.add_output('offset_stress', np.zeros(nFull-1), desc='Von Mises stress utilization along offset column at specified locations. Incudes safety factor.')
        self.add_output('offset_stress:axial', np.zeros(nFull-1), desc='Axial stress along offset column at specified locations.')
        self.add_output('offset_stress:shear', np.zeros(nFull-1), desc='Shear stress along offset column at specified locations.')
        self.add_output('offset_stress:hoop', np.zeros(nFull-1), desc='Hoop stress along offset column at specified locations.')
        self.add_output('offset_stress:hoopStiffen', np.zeros(nFull-1), desc='Hoop stress along offset column at specified locations.')
        self.add_output('offset_shell_buckling', np.zeros(nFull-1), desc='Shell buckling constraint. Should be < 1 for feasibility. Includes safety factors')
        self.add_output('offset_global_buckling', np.zeros(nFull-1), desc='Global buckling constraint. Should be < 1 for feasibility. Includes safety factors')

        self.add_output('tower_stress', np.zeros(nFullTow-1), desc='Von Mises stress utilization along tower at specified locations.  incudes safety factor.')
        self.add_output('tower_stress:axial', np.zeros(nFullTow-1), desc='Axial stress along tower column at specified locations.')
        self.add_output('tower_stress:shear', np.zeros(nFullTow-1), desc='Shear stress along tower column at specified locations.')
        self.add_output('tower_stress:hoop', np.zeros(nFullTow-1), desc='Hoop stress along tower column at specified locations.')
        self.add_output('tower_stress:hoopStiffen', np.zeros(nFullTow-1), desc='Hoop stress along tower column at specified locations.')
        self.add_output('tower_shell_buckling', np.zeros(nFullTow-1), desc='Shell buckling constraint.  Should be < 1 for feasibility.  Includes safety factors')
        self.add_output('tower_global_buckling', np.zeros(nFullTow-1), desc='Global buckling constraint.  Should be < 1 for feasibility.  Includes safety factors')

        self.add_output('plot_matrix', val=np.array([]), desc='Ratio of shear stress to yield stress for all pontoon elements', pass_by_obj=True)
        self.add_output('main_connection_ratio', val=np.zeros((nFull,)), desc='Ratio of pontoon outer diameter to main outer diameter')
        self.add_output('offset_connection_ratio', val=np.zeros((nFull,)), desc='Ratio of pontoon outer diameter to main outer diameter')

        self.add_output('structural_frequencies', np.zeros(NFREQ), units='Hz', desc='First six natural frequencies')
        self.add_output('substructure_mass', val=0.0, units='kg', desc='Mass of substructure elements and connecting truss')
        self.add_output('structural_mass', val=0.0, units='kg', desc='Mass of whole turbine except for mooring lines')
        self.add_output('total_displacement', val=0.0, units='m**3', desc='Total volume of water displaced by floating turbine (except for mooring lines)')
        self.add_output('z_center_of_buoyancy', val=0.0, units='m', desc='z-position of center of buoyancy of whole turbine')
        self.add_output('substructure_center_of_mass', val=np.zeros(3), units='m', desc='xyz-position of center of gravity of substructure only')
        self.add_output('structure_center_of_mass', val=np.zeros(3), units='m', desc='xyz-position of center of gravity of whole turbine')
        self.add_output('total_force', val=np.zeros(3), units='N', desc='Net forces on turbine')
        self.add_output('total_moment', val=np.zeros(3), units='N*m', desc='Moments on whole turbine')
        
        # Derivatives
        self.deriv_options['type'] = 'fd'
        self.deriv_options['form'] = 'central'
        self.deriv_options['check_form'] = 'central'
        self.deriv_options['step_calc'] = 'relative'
        self.deriv_options['step_size'] = 1e-5
         
    def solve_nonlinear(self, params, unknowns, resids):
        
        # Unpack variables
        ncolumn         = int(params['number_of_offset_columns'])
        crossAttachFlag = params['cross_attachment_pontoons']
        lowerAttachFlag = params['lower_attachment_pontoons']
        upperAttachFlag = params['upper_attachment_pontoons']
        lowerRingFlag   = params['lower_ring_pontoons']
        upperRingFlag   = params['upper_ring_pontoons']
        outerCrossFlag  = params['outer_cross_pontoons']
        
        R_semi         = params['radius_to_offset_column'] if ncolumn>0 else 0.0
        R_od_pontoon   = 0.5*params['pontoon_outer_diameter']
        R_od_main      = 0.5*params['main_d_full']
        R_od_offset   = 0.5*params['offset_d_full']
        R_od_tower     = 0.5*params['tower_d_full']
        R_od_fairlead  = 0.5*params['fairlead_support_outer_diameter']

        t_wall_main     = params['main_t_full']
        t_wall_offset  = params['offset_t_full']
        t_wall_pontoon  = params['pontoon_wall_thickness']
        t_wall_tower    = params['tower_t_full']
        t_wall_fairlead = params['fairlead_support_wall_thickness']

        E              = params['E']
        G              = params['G']
        rho            = params['material_density']
        sigma_y        = params['yield_stress']
        
        z_main         = params['main_z_full']
        z_offset       = params['offset_z_full']
        z_tower        = params['tower_z_full']
        z_attach_upper = params['main_pontoon_attach_upper']*(z_main[-1] - z_main[0]) + z_main[0]
        z_attach_lower = params['main_pontoon_attach_lower']*(z_main[-1] - z_main[0]) + z_main[0]
        z_fairlead     = -params['fairlead']

        m_main         = params['main_mass']
        m_offset      = params['offset_mass']
        m_tower        = params['tower_mass_section']
        
        m_rna          = params['rna_mass']
        F_rna          = params['rna_force']
        M_rna          = params['rna_moment']
        I_rna          = params['rna_I']
        cg_rna         = params['rna_cg']
        
        rhoWater       = params['water_density']
        
        V_main         = params['main_displaced_volume']
        V_offset      = params['offset_displaced_volume']

        F_hydro_main    = params['main_hydrostatic_force']
        F_hydro_offset = params['offset_hydrostatic_force']

        z_cb_main      = params['main_center_of_buoyancy']
        z_cb_offset   = params['offset_center_of_buoyancy']
        
        cg_main        = np.r_[0.0, 0.0, params['main_center_of_mass']]
        cg_offset     = np.r_[0.0, 0.0, params['offset_center_of_mass']]
        cg_tower       = np.r_[0.0, 0.0, params['tower_center_of_mass']]
        
        n_connect      = int(params['number_of_mooring_connections'])
        n_lines        = int(params['mooring_lines_per_connection'])
        K_mooring      = np.diag( params['mooring_stiffness'] )
        I_mooring      = params['mooring_moments_of_inertia']
        F_mooring      = params['mooring_neutral_load']
        R_fairlead     = params['fairlead_radius']
        
        gamma_f        = params['gamma_f']
        gamma_m        = params['gamma_m']
        gamma_n        = params['gamma_n']
        gamma_b        = params['gamma_b']
        gamma_fatigue  = params['gamma_fatigue']

        # Quick ratio for unknowns
        unknowns['main_connection_ratio']    = params['connection_ratio_max'] - R_od_pontoon/R_od_main
        unknowns['offset_connection_ratio'] = params['connection_ratio_max'] - R_od_pontoon/R_od_offset
        unknowns['pontoon_wave_height_depth_margin'] = np.abs(np.array([z_attach_lower, z_attach_upper])) - np.abs(params['Hs'])

        
        # --- INPUT CHECKS -----
        # If something fails, we have to tell the optimizer this design is no good
        def bad_input():
            unknowns['structural_frequencies'] = 1e30 * np.ones(NFREQ)
            unknowns['top_deflection'] = 1e30
            unknowns['substructure_mass']  = 1e30
            unknowns['structural_mass']    = 1e30
            unknowns['total_displacement'] = 1e30
            unknowns['z_center_of_buoyancy'] = 0.0
            unknowns['substructure_center_of_mass'] = 1e30 * np.ones(3)
            unknowns['structure_center_of_mass'] = 1e30 * np.ones(3)
            unknowns['total_force'] =  1e30 * np.ones(3)
            unknowns['total_moment'] = 1e30 * np.ones(3)
            unknowns['tower_stress'] = 1e30 * np.ones(m_tower.shape)
            unknowns['tower_shell_buckling'] = 1e30 * np.ones(m_tower.shape)
            unknowns['tower_global_buckling'] = 1e30 * np.ones(m_tower.shape)
            unknowns['main_stress'] = 1e30 * np.ones(m_main.shape)
            unknowns['main_shell_buckling'] = 1e30 * np.ones(m_main.shape)
            unknowns['main_global_buckling'] = 1e30 * np.ones(m_main.shape)
            unknowns['offset_stress'] = 1e30 * np.ones(m_offset.shape)
            unknowns['offset_shell_buckling'] = 1e30 * np.ones(m_offset.shape)
            unknowns['offset_global_buckling'] = 1e30 * np.ones(m_offset.shape)
            return
        
        # There is no truss if not offset columns
        if ncolumn == 0:
            crossAttachFlag = lowerAttachFlag = upperAttachFlag = False
            lowerRingFlag = upperRingFlag = outerCrossFlag  = False

        # Must have symmetry for the substructure to work out
        if ncolumn in [1, 2] or ncolumn > 7:
            bad_input()
            return

        # Must have symmetry in moorning loading too
        if (ncolumn > 0) and (n_connect > 0) and (ncolumn != n_connect):
            bad_input()
            return

        # If there are offset columns, must have attachment pontoons (only have ring pontoons doesn't make sense)
        if (ncolumn > 0) and (not crossAttachFlag) and (not lowerAttachFlag) and (not upperAttachFlag):
            bad_input()
            return

        # Must have lower ring if have cross braces
        if (ncolumn > 0) and outerCrossFlag and (not lowerRingFlag):
            bad_input()
            return

        # ---GEOMETRY---
        # Compute frustum angles
        angle_tower   = np.arctan( np.diff(R_od_tower)   / np.diff(z_tower)   )
        angle_main    = np.arctan( np.diff(R_od_main)    / np.diff(z_main)    )
        angle_offset = np.arctan( np.diff(R_od_offset) / np.diff(z_offset) )
        
        # ---NODES---
        # Add nodes for main column: Using 4 nodes/3 elements per section
        # Make sure there is a node at upper and lower attachment points
        mainBeginID = 0 + 1
        if ncolumn > 0:
            idx = find_nearest(z_main, z_attach_lower)
            z_main[idx] = z_attach_lower
            mainLowerID = idx + 1
            
            idx = find_nearest(z_main, z_attach_upper)
            z_main[idx] = z_attach_upper
            mainUpperID = idx + 1
        
        mainEndID = z_main.size
        freeboard = z_main[-1]

        fairleadID  = []
        # Need mooring attachment point if just running a spar
        if ncolumn == 0:
            idx = find_nearest(z_main, z_fairlead)
            z_main[idx] = z_fairlead
            fairleadID.append( idx + 1 )
        
        znode = np.copy( z_main )
        xnode = np.zeros(znode.shape)
        ynode = np.zeros(znode.shape)
        rnode = np.copy( R_od_main)

        towerBeginID = mainEndID
        myz = np.zeros(len(z_tower)-1)
        xnode = np.append(xnode, myz)
        ynode = np.append(ynode, myz)
        znode = np.append(znode, z_tower[1:] + freeboard )
        rnode = np.append(rnode, R_od_tower[1:])
        towerEndID = xnode.size

        # Create dummy node so that the tower isn't the last in a chain.
        # This avoids a Frame3DD bug
        dummyID = xnode.size + 1
        xnode = np.append(xnode, 0.0)
        ynode = np.append(ynode, 0.0)
        znode = np.append(znode, znode[-1]+1.0 )
        rnode = np.append(rnode, 0.0)
        
        # Get x and y positions of surrounding offset columns
        offsetLowerID = []
        offsetUpperID = []
        offsetx = R_semi * np.cos( np.linspace(0, 2*np.pi, ncolumn+1) )
        offsety = R_semi * np.sin( np.linspace(0, 2*np.pi, ncolumn+1) )
        offsetx = offsetx[:-1]
        offsety = offsety[:-1]

        # Add in offset column nodes around the circle, make sure there is a node at the fairlead
        idx = find_nearest(z_offset, z_fairlead)
        myones = np.ones(z_offset.shape)
        for k in range(ncolumn):
            offsetLowerID.append( xnode.size + 1 )
            fairleadID.append( xnode.size + idx + 1 )
            xnode = np.append(xnode, offsetx[k]*myones)
            ynode = np.append(ynode, offsety[k]*myones)
            znode = np.append(znode, z_offset )
            rnode = np.append(rnode, R_od_offset )
            offsetUpperID.append( xnode.size )

        # Add nodes where mooring lines attach, which may be offset from columns
        mooringx  = R_fairlead * np.cos( np.linspace(0, 2*np.pi, n_connect+1) )[:-1]
        mooringy  = R_fairlead * np.sin( np.linspace(0, 2*np.pi, n_connect+1) )[:-1]
        mooringID = xnode.size + 1 + np.arange(n_connect, dtype=np.int32)
        xnode     = np.append(xnode, mooringx)
        ynode     = np.append(ynode, mooringy)
        znode     = np.append(znode, z_fairlead*np.ones(n_connect) )
        rnode     = np.append(rnode, np.zeros(n_connect))
            
        # Add nodes midway around outer ring for cross bracing
        if outerCrossFlag and ncolumn > 0:
            crossx = 0.5*(offsetx + np.roll(offsetx,1))
            crossy = 0.5*(offsety + np.roll(offsety,1))

            crossOuterLowerID = xnode.size + np.arange(ncolumn) + 1
            crossOuterLowerID = crossOuterLowerID.tolist()
            xnode = np.append(xnode, crossx)
            ynode = np.append(ynode, crossy)
            znode = np.append(znode, z_offset[0]*np.ones(ncolumn))
            rnode = np.append(rnode, np.zeros(ncolumn))

            #crossOuterUpperID = xnode.size + np.arange(ncolumn) + 1
            #xnode = np.append(xnode, crossx)
            #ynode = np.append(ynode, crossy)
            #znode = np.append(znode, z_offset[-1]*np.ones(ncolumn))

        # Create matrix for easy referencing
        nodeMat = np.c_[xnode, ynode, znode]

        # To aid in wrap-around references
        if ncolumn > 0:
            offsetLowerID.append( offsetLowerID[0] )
            offsetUpperID.append( offsetUpperID[0] )
            if outerCrossFlag:
                crossOuterLowerID.append( crossOuterLowerID[0] )
        
        
        # ---ELEMENTS / EDGES---
        # To accurately capture pontoon length and stiffness, for each connection we create 2 additional nodes,
        # where the pontoon "line" intersects the main and offset shells.  Highly stiff "ghost" elements are created
        # from the column centerline to the shell.  These are not calculated for pontoon weight.
        # The actual pontoon only extends from shell boundary to shell boundary.
        N1  = np.array([], dtype=np.int32)
        N2  = np.array([], dtype=np.int32)
        gN1 = np.array([], dtype=np.int32)
        gN2 = np.array([], dtype=np.int32)
        
        # Lower connection from central main column to offset columns
        if lowerAttachFlag:
            lowerAttachEID = N1.size + 1
            for k in range(ncolumn):
                tempID1 = mainLowerID
                tempID2 = offsetLowerID[k]
                add1, add2 = ghostNodes(nodeMat[tempID1-1,:], nodeMat[tempID2-1,:], rnode[tempID1-1], rnode[tempID2-1])
                if ( (add1[-1] > z_main[0]) and (add1[-1] < z_main[-1]) ):
                    tempID1 = xnode.size + 1
                    xnode   = np.append(xnode, add1[0])
                    ynode   = np.append(ynode, add1[1])
                    znode   = np.append(znode, add1[2])
                    gN1     = np.append(gN1, mainLowerID)
                    gN2     = np.append(gN2, tempID1)
                if ( (add2[-1] > z_offset[0]) and (add2[-1] < z_offset[-1]) ):
                    tempID2 = xnode.size + 1
                    xnode   = np.append(xnode, add2[0])
                    ynode   = np.append(ynode, add2[1])
                    znode   = np.append(znode, add2[2])
                    gN1     = np.append(gN1, offsetLowerID[k])
                    gN2     = np.append(gN2, tempID2)
                # Pontoon connection
                N1 = np.append(N1, tempID1 )
                N2 = np.append(N2, tempID2 )
                
        # Upper connection from central main column to offset columns
        if upperAttachFlag:
            upperAttachEID = N1.size + 1
            for k in range(ncolumn):
                tempID1 = mainUpperID
                tempID2 = offsetUpperID[k]
                add1, add2 = ghostNodes(nodeMat[tempID1-1,:], nodeMat[tempID2-1,:], rnode[tempID1-1], rnode[tempID2-1])
                if ( (add1[-1] > z_main[0]) and (add1[-1] < z_main[-1]) ):
                    tempID1 = xnode.size + 1
                    xnode   = np.append(xnode, add1[0])
                    ynode   = np.append(ynode, add1[1])
                    znode   = np.append(znode, add1[2])
                    gN1     = np.append(gN1, mainUpperID)
                    gN2     = np.append(gN2, tempID1)
                if ( (add2[-1] > z_offset[0]) and (add2[-1] < z_offset[-1]) ):
                    tempID2 = xnode.size + 1
                    xnode   = np.append(xnode, add2[0])
                    ynode   = np.append(ynode, add2[1])
                    znode   = np.append(znode, add2[2])
                    gN1     = np.append(gN1, offsetUpperID[k])
                    gN2     = np.append(gN2, tempID2)
                # Pontoon connection
                N1 = np.append(N1, tempID1 )
                N2 = np.append(N2, tempID2 )
                
        # Cross braces from lower central main column to upper offset columns
        if crossAttachFlag:
            crossAttachEID = N1.size + 1
            for k in range(ncolumn):
                tempID1 = mainLowerID
                tempID2 = offsetUpperID[k]
                add1, add2 = ghostNodes(nodeMat[tempID1-1,:], nodeMat[tempID2-1,:], rnode[tempID1-1], rnode[tempID2-1])
                if ( (add1[-1] > z_main[0]) and (add1[-1] < z_main[-1]) ):
                    tempID1 = xnode.size + 1
                    xnode   = np.append(xnode, add1[0])
                    ynode   = np.append(ynode, add1[1])
                    znode   = np.append(znode, add1[2])
                    gN1     = np.append(gN1, mainLowerID)
                    gN2     = np.append(gN2, tempID1)
                if ( (add2[-1] > z_offset[0]) and (add2[-1] < z_offset[-1]) ):
                    tempID2 = xnode.size + 1
                    xnode   = np.append(xnode, add2[0])
                    ynode   = np.append(ynode, add2[1])
                    znode   = np.append(znode, add2[2])
                    gN1     = np.append(gN1, offsetUpperID[k])
                    gN2     = np.append(gN2, tempID2)
                # Pontoon connection
                N1 = np.append(N1, tempID1 )
                N2 = np.append(N2, tempID2 )
                
            # Will be used later to convert from local member c.s. to global
            cross_angle = np.arctan( (z_attach_upper - z_attach_lower) / R_semi )
            
        # Lower ring around offset columns
        if lowerRingFlag:
            lowerRingEID = N1.size + 1
            for k in range(ncolumn):
                tempID1 = offsetLowerID[k]
                tempID2 = offsetLowerID[k+1]
                add1, add2 = ghostNodes(nodeMat[tempID1-1,:], nodeMat[tempID2-1,:], rnode[tempID1-1], rnode[tempID2-1])
                if ( (add1[-1] > z_offset[0]) and (add1[-1] < z_offset[-1]) ):
                    tempID1 = xnode.size + 1
                    xnode   = np.append(xnode, add1[0])
                    ynode   = np.append(ynode, add1[1])
                    znode   = np.append(znode, add1[2])
                    gN1     = np.append(gN1, offsetLowerID[k])
                    gN2     = np.append(gN2, tempID1)
                if ( (add2[-1] > z_offset[0]) and (add2[-1] < z_offset[-1]) ):
                    tempID2 = xnode.size + 1
                    xnode   = np.append(xnode, add2[0])
                    ynode   = np.append(ynode, add2[1])
                    znode   = np.append(znode, add2[2])
                    gN1     = np.append(gN1, offsetLowerID[k+1])
                    gN2     = np.append(gN2, tempID2)
                # Pontoon connection
                N1 = np.append(N1, tempID1 )
                N2 = np.append(N2, tempID2 )

        # Upper ring around offset columns
        if upperRingFlag:
            upperRingEID = N1.size + 1
            for k in range(ncolumn):
                tempID1 = offsetUpperID[k]
                tempID2 = offsetUpperID[k+1]
                add1, add2 = ghostNodes(nodeMat[tempID1-1,:], nodeMat[tempID2-1,:], rnode[tempID1-1], rnode[tempID2-1])
                if ( (add1[-1] > z_offset[0]) and (add1[-1] < z_offset[-1]) ):
                    tempID1 = xnode.size + 1
                    xnode   = np.append(xnode, add1[0])
                    ynode   = np.append(ynode, add1[1])
                    znode   = np.append(znode, add1[2])
                    gN1     = np.append(gN1, offsetUpperID[k])
                    gN2     = np.append(gN2, tempID1)
                if ( (add2[-1] > z_offset[0]) and (add2[-1] < z_offset[-1]) ):
                    tempID2 = xnode.size + 1
                    xnode   = np.append(xnode, add2[0])
                    ynode   = np.append(ynode, add2[1])
                    znode   = np.append(znode, add2[2])
                    gN1     = np.append(gN1, offsetUpperID[k+1])
                    gN2     = np.append(gN2, tempID2)
                # Pontoon connection
                N1 = np.append(N1, tempID1 )
                N2 = np.append(N2, tempID2 )
                
        # Outer cross braces (only one ghost node per connection)
        if outerCrossFlag:
            outerCrossEID = N1.size + 1
            for k in range(ncolumn):
                tempID1 = crossOuterLowerID[k]
                tempID2 = offsetUpperID[k]
                _, add2 = ghostNodes(nodeMat[tempID1-1,:], nodeMat[tempID2-1,:], rnode[tempID1-1], rnode[tempID2-1])
                if ( (add2[-1] > z_offset[0]) and (add2[-1] < z_offset[-1]) ):
                    tempID2 = xnode.size + 1
                    xnode   = np.append(xnode, add2[0])
                    ynode   = np.append(ynode, add2[1])
                    znode   = np.append(znode, add2[2])
                    gN1     = np.append(gN1, offsetUpperID[k])
                    gN2     = np.append(gN2, tempID2)
                # Pontoon connection
                N1 = np.append(N1, tempID1 )
                N2 = np.append(N2, tempID2 )

                _, add2 = ghostNodes(nodeMat[crossOuterLowerID[k+1]-1,:], nodeMat[offsetUpperID[k]-1,:], rnode[crossOuterLowerID[k+1]-1], rnode[offsetUpperID[k]-1])
                tempID     = xnode.size + 1
                xnode      = np.append(xnode, add2[0])
                ynode      = np.append(ynode, add2[1])
                znode      = np.append(znode, add2[2])
                gN1        = np.append(gN1, offsetUpperID[k])
                gN2        = np.append(gN2, tempID)
                N1         = np.append(N1, crossOuterLowerID[k+1] )
                N2         = np.append(N2, tempID )

                tempID1 = crossOuterLowerID[k+1]
                tempID2 = offsetUpperID[k]
                _, add2 = ghostNodes(nodeMat[tempID1-1,:], nodeMat[tempID2-1,:], rnode[tempID1-1], rnode[tempID2-1])
                if ( (add2[-1] > z_offset[0]) and (add2[-1] < z_offset[-1]) ):
                    tempID2 = xnode.size + 1
                    xnode   = np.append(xnode, add2[0])
                    ynode   = np.append(ynode, add2[1])
                    znode   = np.append(znode, add2[2])
                    gN1     = np.append(gN1, offsetUpperID[k])
                    gN2     = np.append(gN2, tempID2)
                # Pontoon connection
                N1 = np.append(N1, tempID1 )
                N2 = np.append(N2, tempID2 )
                
            
        # TODO: Parameterize these for upper, lower, cross connections
        # Properties for the inner connectors
        mytube = Tube(2.0*R_od_pontoon, t_wall_pontoon)
        Ax    = mytube.Area * np.ones(N1.shape)
        As    = mytube.Asx  * np.ones(N1.shape)
        Jx    = mytube.J0   * np.ones(N1.shape)
        I     = mytube.Jxx  * np.ones(N1.shape)
        S     = mytube.S    * np.ones(N1.shape)
        C     = mytube.C    * np.ones(N1.shape)
        modE  = E   * np.ones(N1.shape)
        modG  = G   * np.ones(N1.shape)
        roll  = 0.0 * np.ones(N1.shape)
        dens  = rho * np.ones(N1.shape)

        # Add in fairlead support elements
        mooringEID = N1.size + 1
        mytube  = Tube(2.0*R_od_fairlead, t_wall_fairlead)
        for k in range(n_connect):
            kfair   = 0 if ncolumn==0 else k
            
            add1, _ = ghostNodes(nodeMat[fairleadID[kfair]-1,:], nodeMat[mooringID[k]-1,:], rnode[fairleadID[kfair]-1], rnode[mooringID[k]-1])
            tempID  = xnode.size + 1
            xnode   = np.append(xnode, add1[0])
            ynode   = np.append(ynode, add1[1])
            znode   = np.append(znode, add1[2])
            gN1     = np.append(gN1, fairleadID[kfair])
            gN2     = np.append(gN2, tempID)
            N1      = np.append(N1, tempID )
            N2      = np.append(N2, mooringID[k] )

            Ax      = np.append(Ax  , mytube.Area )
            As      = np.append(As  , mytube.Asx )
            Jx      = np.append(Jx  , mytube.J0 )
            I       = np.append(I   , mytube.Jxx )
            S       = np.append(S   , mytube.S )
            C       = np.append(C   , mytube.C )
            modE    = np.append(modE, E )
            modG    = np.append(modG, G )
            roll    = np.append(roll, 0.0 )
            dens    = np.append(dens, rho )

        # Now mock up cylindrical columns as truss members even though long, slender assumption breaks down
        # Will set density = 0.0 so that we don't double count the mass
        # First get geometry in each of the elements
        R_od_main,_     = nodal2sectional( R_od_main )
        R_od_offset,_   = nodal2sectional( R_od_offset )
        R_od_tower,_    = nodal2sectional( R_od_tower )

        # Main column
        mainEID = N1.size + 1
        mytube  = Tube(2.0*R_od_main, t_wall_main)
        myrange = np.arange(R_od_main.size)
        myones  = np.ones(myrange.shape)
        mydens  = m_main / mytube.Area / np.diff(z_main) + eps
        N1   = np.append(N1  , myrange + mainBeginID    )
        N2   = np.append(N2  , myrange + mainBeginID + 1)
        Ax   = np.append(Ax  , mytube.Area )
        As   = np.append(As  , mytube.Asx )
        Jx   = np.append(Jx  , mytube.J0 )
        I    = np.append(I   , mytube.Jxx )
        S    = np.append(S   , mytube.S )
        C    = np.append(C   , mytube.C )
        modE = np.append(modE, E*myones )
        modG = np.append(modG, G*myones )
        roll = np.append(roll, np.zeros(myones.shape) )
        dens = np.append(dens, mydens )

        # Tower column
        towerEID = N1.size + 1
        mytube  = Tube(2.0*R_od_tower, t_wall_tower)
        myrange = np.arange(R_od_tower.size)
        myones  = np.ones(myrange.shape)
        mydens  = m_tower / mytube.Area / np.diff(z_tower) + eps
        N1   = np.append(N1  , myrange + towerBeginID    )
        N2   = np.append(N2  , myrange + towerBeginID + 1)
        Ax   = np.append(Ax  , mytube.Area )
        As   = np.append(As  , mytube.Asx )
        Jx   = np.append(Jx  , mytube.J0 )
        I    = np.append(I   , mytube.Jxx )
        S    = np.append(S   , mytube.S )
        C    = np.append(C   , mytube.C )
        modE = np.append(modE, E*myones )
        modG = np.append(modG, G*myones )
        roll = np.append(roll, np.zeros(myones.shape) )
        dens = np.append(dens, mydens ) 

        # Dummy element
        dummyEID = N1.size + 1
        N1   = np.append(N1  , towerEndID )
        N2   = np.append(N2  , dummyID )
        Ax   = np.append(Ax  , Ax[-1] )
        As   = np.append(As  , As[-1] )
        Jx   = np.append(Jx  , Jx[-1] )
        I    = np.append(I   , I[-1] )
        S    = np.append(S   , S[-1] )
        C    = np.append(C   , C[-1] )
        modE = np.append(modE, 1e20 )
        modG = np.append(modG, 1e20 )
        roll = np.append(roll, 0.0 )
        dens = np.append(dens, 1e-6 ) 

        # Offset column
        offsetEID = []
        mytube     = Tube(2.0*R_od_offset, t_wall_offset)
        myrange    = np.arange(R_od_offset.size)
        myones     = np.ones(myrange.shape)
        mydens     = m_offset / mytube.Area / np.diff(z_offset) + eps
        for k in range(ncolumn):
            offsetEID.append( N1.size + 1 )
            
            N1   = np.append(N1  , myrange + offsetLowerID[k]    )
            N2   = np.append(N2  , myrange + offsetLowerID[k] + 1)
            Ax   = np.append(Ax  , mytube.Area )
            As   = np.append(As  , mytube.Asx )
            Jx   = np.append(Jx  , mytube.J0 )
            I    = np.append(I   , mytube.Jxx )
            S    = np.append(S   , mytube.S )
            C    = np.append(C   , mytube.C )
            modE = np.append(modE, E*myones )
            modG = np.append(modG, G*myones )
            roll = np.append(roll, np.zeros(myones.shape) )
            dens = np.append(dens, mydens ) # Mass added below

        # Ghost elements between centerline nodes and column shells
        ghostEID = N1.size + 1
        myones   = np.ones(gN1.shape)
        N1   = np.append(N1  , gN1 )
        N2   = np.append(N2  , gN2 )
        Ax   = np.append(Ax  , 1e-1*myones )
        As   = np.append(As  , 1e-1*myones )
        Jx   = np.append(Jx  , 1e-1*myones )
        I    = np.append(I   , 1e-1*myones )
        S    = np.append(S   , 1e-1*myones )
        C    = np.append(C   , 1e-1*myones )
        modE = np.append(modE, 1e20*myones )
        modG = np.append(modG, 1e20*myones )
        roll = np.append(roll, 0.0 *myones )
        dens = np.append(dens, 1e-6*myones )

        # Create Node Data object
        nnode   = 1 + np.arange(xnode.size)
        myrnode = np.zeros(xnode.shape) # z-spacing too narrow for use of rnodes
        nodes   = frame3dd.NodeData(nnode, xnode, ynode, znode, myrnode)
        nodeMat = np.c_[xnode, ynode, znode]

        # Create Element Data object
        nelem    = 1 + np.arange(N1.size)
        elements = frame3dd.ElementData(nelem, N1, N2, Ax, As, As, Jx, I, I, modE, modG, roll, dens)

        # Store data for plotting, also handy for operations below
        plotMat = np.zeros((mainEID, 3, 2))
        myn1 = N1[:mainEID]
        myn2 = N2[:mainEID]
        plotMat[:,:,0] = nodeMat[myn1-1,:]
        plotMat[:,:,1] = nodeMat[myn2-1,:]
        unknowns['plot_matrix'] = plotMat
        
        # Compute length and center of gravity for each element for use below
        elemL   = np.sqrt( np.sum( np.diff(plotMat, axis=2)**2.0, axis=1) ).flatten()
        elemCoG = 0.5*np.sum(plotMat, axis=2)
        # Get vertical angle as a measure of welding prep difficulty
        elemAng = np.arccos( np.diff(plotMat[:,-1,:], axis=-1).flatten() / elemL )

        # ---Options object---
        shear = True               # 1: include shear deformation
        geom = False               # 1: include geometric stiffness
        dx = -1                    # x-axis increment for internal forces, -1 to skip
        other = frame3dd.Options(shear, geom, dx)

        # ---LOAD CASES---
        # Extreme loading
        gx = 0.0
        gy = 0.0
        gz = -gravity
        load = frame3dd.StaticLoadCase(gx, gy, gz)

        # Wind + Wave loading in local main / offset / tower c.s.
        Px_main,    Py_main,    Pz_main    = params['main_Pz'], params['main_Py'], -params['main_Px']  # switch to local c.s.
        Px_offset, Py_offset, Pz_offset = params['offset_Pz'], params['offset_Py'], -params['offset_Px']  # switch to local c.s.
        Px_tower,   Py_tower,   Pz_tower   = params['tower_Pz'], params['tower_Py'], -params['tower_Px']  # switch to local c.s.
        epsOff = 1e-5
        # Get mass right- offsets, stiffeners, tower, rna, etc.
        # Also account for buoyancy loads
        # Also apply wind/wave loading as trapezoidal on each element
        # NOTE: Loading is in local element coordinates 0-L, x is along element
        # Base
        nrange  = np.arange(R_od_main.size, dtype=np.int32)
        EL      = mainEID + nrange
        Ux      = F_hydro_main / np.diff(z_main)
        x1 = np.zeros(nrange.shape)
        x2 = np.diff(z_main) - epsOff  # subtract small number b.c. of precision
        wx1, wx2 = Px_main[:-1], Px_main[1:]
        wy1, wy2 = Py_main[:-1], Py_main[1:]
        wz1, wz2 = Pz_main[:-1], Pz_main[1:]
        # Tower
        nrange  = np.arange(R_od_tower.size, dtype=np.int32)
        EL      = np.append(EL, towerEID + nrange)
        Ux      = np.append(Ux, np.zeros(nrange.shape))
        x1      = np.append(x1, np.zeros(nrange.shape))
        x2      = np.append(x2, np.diff(z_tower) - epsOff)
        wx1     = np.append(wx1, Px_tower[:-1])
        wx2     = np.append(wx2, Px_tower[1:])
        wy1     = np.append(wy1, Py_tower[:-1])
        wy2     = np.append(wy2, Py_tower[1:])
        wz1     = np.append(wz1, Pz_tower[:-1])
        wz2     = np.append(wz2, Pz_tower[1:])
        # Buoyancy- offset columns
        nrange  = np.arange(R_od_offset.size, dtype=np.int32)
        for k in range(ncolumn):
            EL      = np.append(EL, offsetEID[k] + nrange)
            Ux      = np.append(Ux,  F_hydro_offset / np.diff(z_offset) )
            x1      = np.append(x1, np.zeros(nrange.shape))
            x2      = np.append(x2, np.diff(z_offset) - epsOff)
            wx1     = np.append(wx1, Px_offset[:-1])
            wx2     = np.append(wx2, Px_offset[1:])
            wy1     = np.append(wy1, Py_offset[:-1])
            wy2     = np.append(wy2, Py_offset[1:])
            wz1     = np.append(wz1, Pz_offset[:-1])
            wz2     = np.append(wz2, Pz_offset[1:])
            
        # Add mass of main and offset columns while we've already done the element enumeration
        Uz = Uy = np.zeros(Ux.shape)
        xx1 = xy1 = xz1 = x1
        xx2 = xy2 = xz2 = x2
        load.changeTrapezoidalLoads(EL, xx1, xx2, wx1, wx2, xy1, xy2, wy1, wy2, xz1, xz2, wz1, wz2)

        # Buoyancy for fully submerged members
        nrange  = np.arange(ncolumn, dtype=np.int32)
        Frange  = np.pi * R_od_pontoon**2 * rhoWater * gravity
        F_truss = 0.0
        z_cb    = np.zeros((3,))
        if ncolumn > 0 and znode[offsetLowerID[0]-1] < 0.0:
            if lowerAttachFlag:
                EL       = np.append(EL, lowerAttachEID + nrange)
                Ux       = np.append(Ux, np.zeros(nrange.shape))
                Uy       = np.append(Uy, np.zeros(nrange.shape))
                Uz       = np.append(Uz, Frange * np.ones(nrange.shape))
                F_truss += Frange * elemL[lowerAttachEID-1] * ncolumn
                z_cb    += Frange * elemL[lowerAttachEID-1] * ncolumn * elemCoG[lowerAttachEID-1,:]
            if lowerRingFlag:
                EL       = np.append(EL, lowerRingEID + nrange)
                Ux       = np.append(Ux, np.zeros(nrange.shape))
                Uy       = np.append(Uy, np.zeros(nrange.shape))
                Uz       = np.append(Uz, Frange * np.ones(nrange.shape))
                F_truss += Frange * elemL[lowerRingEID-1] * ncolumn
                z_cb    += Frange * elemL[lowerRingEID-1] * ncolumn * elemCoG[lowerRingEID-1]
            if crossAttachFlag:
                factor   = np.minimum(1.0, (0.0 - z_attach_lower) / (znode[offsetUpperID[0]-1] - z_attach_lower) )
                EL       = np.append(EL, crossAttachEID + nrange)
                Ux       = np.append(Ux,  factor * Frange * np.sin(cross_angle) * np.ones(nrange.shape))
                Uy       = np.append(Uy, np.zeros(nrange.shape))
                Uz       = np.append(Uz, factor * Frange * np.cos(cross_angle) * np.ones(nrange.shape))
                F_truss += factor * Frange * elemL[crossAttachEID-1] * ncolumn
                z_cb    += factor * Frange * elemL[crossAttachEID-1] * ncolumn * elemCoG[crossAttachEID-1,:]
            if outerCrossFlag:
                factor   = np.minimum(1.0, (0.0 - znode[mainLowerID-1]) / (znode[offsetUpperID[0]-1] - znode[mainLowerID-1]) )
                # TODO: This one will take a little more math
                #EL       = np.append(EL, outerCrossEID + np.arange(2*ncolumn, dtype=np.int32))
                #Ux       = np.append(Ux, np.zeros(nrange.shape))
                #Uy       = np.append(Uy, np.zeros(nrange.shape))
                #Uz       = np.append(Uz, factor * Frange * np.ones(nrange.shape))
                F_truss += factor * Frange * elemL[outerCrossEID-1] * ncolumn
                z_cb    += factor * Frange * elemL[outerCrossEID-1] * ncolumn * elemCoG[outerCrossEID-1,:]
        if ncolumn > 0 and znode[offsetUpperID[0]-1] < 0.0:
            if upperAttachFlag:
                EL       = np.append(EL, upperAttachEID + nrange)
                Ux       = np.append(Ux, np.zeros(nrange.shape))
                Uy       = np.append(Uy, np.zeros(nrange.shape))
                Uz       = np.append(Uz, Frange * np.ones(nrange.shape))
                F_truss += Frange * elemL[upperAttachEID-1] * ncolumn
                z_cb    += Frange * elemL[upperAttachEID-1] * ncolumn * elemCoG[upperAttachEID-1,:]
            if upperRingFlag:
                EL       = np.append(EL, upperRingEID + nrange)
                Ux       = np.append(Ux, np.zeros(nrange.shape))
                Uy       = np.append(Uy, np.zeros(nrange.shape))
                Uz       = np.append(Uz, Frange * np.ones(nrange.shape))
                F_truss += Frange * elemL[upperRingEID-1] * ncolumn
                z_cb    += Frange * elemL[upperRingEID-1] * ncolumn * elemCoG[upperRingEID-1,:]
        # Now do fairlead supports
        nrange   = np.arange(n_connect, dtype=np.int32)
        Frange   = np.pi * R_od_fairlead**2 * rhoWater * gravity
        EL       = np.append(EL, mooringEID + nrange)
        Ux       = np.append(Ux, np.zeros(nrange.shape))
        Uy       = np.append(Uy, np.zeros(nrange.shape))
        Uz       = np.append(Uz, Frange * np.ones(nrange.shape))
        F_truss += Frange * elemL[mooringEID-1] * n_connect
        z_cb    += Frange * elemL[mooringEID-1] * n_connect * elemCoG[mooringEID-1,:]
        # Finally add in all the uniform loads on buoyancy
        load.changeUniformLoads(EL, Ux, Uy, Uz)

        # Point loading for rotor thrust and mooring lines
        # Point loads for mooring loading
        nnode_connect = len(fairleadID)
        nF  = np.array(fairleadID, dtype=np.int32)
        Fx  = np.zeros(nnode_connect)
        Fy  = np.zeros(nnode_connect)
        Fz  = np.zeros(nnode_connect)
        Mxx = np.zeros(nnode_connect)
        Myy = np.zeros(nnode_connect)
        Mzz = np.zeros(nnode_connect)
        for k in range(n_connect):
            iline = 0 if nnode_connect==1 else k
            idx = k*n_lines + np.arange(n_lines)
            Fx[iline] += F_mooring[idx,0].sum()
            Fy[iline] += F_mooring[idx,1].sum()
            Fz[iline] += F_mooring[idx,2].sum()
        # Note: extra momemt from mass accounted for below
        nF  = np.append(nF , towerEndID)
        Fx  = np.append(Fx , F_rna[0] )
        Fy  = np.append(Fy , F_rna[1] )
        Fz  = np.append(Fz , F_rna[2] )
        Mxx = np.append(Mxx, M_rna[0] )
        Myy = np.append(Myy, M_rna[1] )
        Mzz = np.append(Mzz, M_rna[2] )
        
        # Add in all point loads
        load.changePointLoads(nF, Fx, Fy, Fz, Mxx, Myy, Mzz)



        # ---MASS SUMMARIES---
        # Mass summaries now that we've tabulated all of the pontoons
        m_substructure = m_main.sum() + ncolumn*m_offset.sum()
        if mainEID > 1: # Have some pontoons or fairlead supports
            # Buoyancy assembly from incremental calculations above
            V_pontoon = F_truss/rhoWater/gravity
            z_cb      = z_cb[-1] / F_truss if F_truss > 0.0 else 0.0
            unknowns['pontoon_displacement'] = V_pontoon
            unknowns['pontoon_center_of_buoyancy'] = z_cb

            # Sum up mass and compute CofG.  Frame3DD does mass, but not CG
            ind             = mainEID-1
            m_total         = Ax[:ind] * rho * elemL[:ind]
            m_pontoon       = m_total.sum() #mass.struct_mass
            m_substructure += m_pontoon
            cg_pontoon      = np.sum( m_total[:,np.newaxis] * elemCoG[:ind,:], axis=0 ) / m_total.sum()

            # Compute costs based on "Optimum Design of Steel Structures" by Farkas and Jarmai
            # All dimensions for correlations based on mm, not meters.
            k_m     = params['material_cost_rate'] #1.1 # USD / kg carbon steel plate
            k_f     = params['labor_cost_rate'] #1.0 # USD / min labor
            k_p     = params['painting_cost_rate'] #USD / m^2 painting
            npont   = m_total.size

            # Cost Step 1) Cutting and grinding tube ends
            theta_g = 3.0 # Difficulty factor
            # Cost Step 2) Fillet welds with SMAW (shielded metal arc welding)
            # Multiply by 2 for both ends worth of work
            theta_w = 3.0 # Difficulty factor

            # Labor-based expenses
            K_f = k_f * 2 * ( manufacture.steel_tube_cutgrind_time(theta_g, R_od_pontoon, t_wall_pontoon, elemAng[:ind]) +
                              manufacture.steel_tube_welding_time(theta_w, npont+ncolumn+1, m_substructure, 2*np.pi*R_od_pontoon, t_wall_pontoon) )

            # Cost Step 3) Painting
            theta_p = 2.0
            S_pont  = 2.0 * np.pi * R_od_pontoon * elemL[:ind]
            K_p     = k_p * theta_p * S_pont.sum()

            # Material cost
            K_m = k_m * m_pontoon

            # Total cost
            c_pontoon = K_m + K_f + K_p
            
            unknowns['pontoon_mass'] = m_pontoon
            unknowns['pontoon_cost'] = c_pontoon
            unknowns['pontoon_cost_rate'] = 1e3*c_pontoon/m_pontoon
            unknowns['pontoon_center_of_mass'] = cg_pontoon[-1]
        else:
            V_pontoon = z_cb = m_pontoon = 0.0
            cg_pontoon = np.zeros(3)
            
        # Summary of mass and volumes
        unknowns['total_displacement'] = V_main.sum() + ncolumn*V_offset.sum() + V_pontoon
        unknowns['substructure_mass']  = m_substructure
        unknowns['substructure_center_of_mass'] = (ncolumn*m_offset.sum()*cg_offset + m_main.sum()*cg_main +
                                                   m_pontoon*cg_pontoon) / unknowns['substructure_mass']
        m_total = unknowns['substructure_mass'] + m_rna + m_tower.sum()
        unknowns['structural_mass'] = m_total
        unknowns['structure_center_of_mass']  = (m_rna*cg_rna + m_tower.sum()*cg_tower +
                                       unknowns['substructure_mass']*unknowns['substructure_center_of_mass']) / m_total

        # Find cb (center of buoyancy) for whole system
        z_cb = (V_main.sum()*z_cb_main + ncolumn*V_offset.sum()*z_cb_offset + V_pontoon*z_cb) / unknowns['total_displacement']
        unknowns['z_center_of_buoyancy'] = z_cb


        # ---REACTIONS---
        # Find node closest to CG
        cg_dist = np.sum( (nodeMat - unknowns['structure_center_of_mass'][np.newaxis,:])**2, axis=1 )
        cg_node = np.argmin(cg_dist)
        # Free=0, Rigid=inf
        rid = np.array([mainBeginID]) #np.array(fairleadID) #np.array([cg_node+1]) #
        Rx  = np.inf * np.ones(rid.shape)
        Ry  = np.inf * np.ones(rid.shape)
        Rz  = np.inf * np.ones(rid.shape)
        Rxx = np.inf * np.ones(rid.shape)
        Ryy = np.inf * np.ones(rid.shape)
        Rzz = np.inf * np.ones(rid.shape)
        rid = np.append(rid, fairleadID)
        Rx  = np.append(Rx,  K_mooring[0] /nnode_connect * np.ones(nnode_connect) )
        Ry  = np.append(Ry,  K_mooring[1] /nnode_connect * np.ones(nnode_connect) )
        Rz  = np.append(Rz,  K_mooring[2] /nnode_connect * np.ones(nnode_connect) )
        Rxx = np.append(Rxx,  K_mooring[3]/nnode_connect * np.ones(nnode_connect) )
        Ryy = np.append(Ryy,  K_mooring[4]/nnode_connect * np.ones(nnode_connect) )
        Rzz = np.append(Rzz,  K_mooring[5]/nnode_connect * np.ones(nnode_connect) )
        # Get reactions object from frame3dd
        reactions = frame3dd.ReactionData(rid, Rx, Ry, Rz, Rxx, Ryy, Rzz, rigid=np.inf)

        
        # ---FRAME3DD INSTANCE---

        # Initialize frame3dd object
        self.myframe = frame3dd.Frame(nodes, reactions, elements, other)
        
        # Add in extra mass of rna
        inode   = np.array([towerEndID], dtype=np.int32) # rna
        m_extra = np.array([m_rna])
        Ixx = np.array([ I_rna[0] ])
        Iyy = np.array([ I_rna[1] ])
        Izz = np.array([ I_rna[2] ])
        Ixy = np.array([ I_rna[3] ])
        Ixz = np.array([ I_rna[4] ])
        Iyz = np.array([ I_rna[5] ])
        rhox = np.array([ cg_rna[0] ])
        rhoy = np.array([ cg_rna[1] ])
        rhoz = np.array([ cg_rna[2] ])
        self.myframe.changeExtraNodeMass(inode, m_extra, Ixx, Iyy, Izz, Ixy, Ixz, Iyz, rhox, rhoy, rhoz, True)
        
        # Store load case into frame 3dd object
        self.myframe.addLoadCase(load)


        # ---DYNAMIC ANALYSIS---
        # This needs to be compared to FAST until I trust it enough to use it.
        # Have to test BCs, results, mooring stiffness, mooring mass/MOI, etc
        nM = 0 #NFREQ          # number of desired dynamic modes of vibration
        Mmethod = 1         # 1: subspace Jacobi     2: Stodola
        lump = 0            # 0: consistent mass ... 1: lumped mass matrix
        tol = 1e-5          # mode shape tolerance
        shift = 0.0        # shift value ... for unrestrained or partially restrained structures
        
        #self.myframe.enableDynamics(nM, Mmethod, lump, tol, shift)

        # ---DEBUGGING---
        #self.myframe.write('debug.3dd') # For debugging

        # ---RUN ANALYSIS---
        try:
            displacements, forces, reactions, internalForces, mass, modal = self.myframe.run()
        except:
            bad_input()
            return
            
        # --OUTPUTS--
        nE    = nelem.size
        iCase = 0

        # natural frequncies- catch nans and zeros
        temp = np.zeros(NFREQ) #np.array( modal.freq )
        temp[np.isnan(temp)] = 0.0
        unknowns['structural_frequencies'] = temp + eps

        # deflections due to loading (from cylinder top and wind/wave loads)
        unknowns['top_deflection'] = displacements.dx[iCase, towerEndID-1]  # in yaw-aligned direction

        # Find cg (center of gravity) for whole system
        F_main = -1.0 * np.array([reactions.Fx.sum(), reactions.Fy.sum(), reactions.Fz.sum()])
        M_main = -1.0 * np.array([reactions.Mxx.sum(), reactions.Myy.sum(), reactions.Mzz.sum()])
        r_cg_main = np.array([0.0, 0.0, (znode[mainBeginID] - unknowns['structure_center_of_mass'][-1])])
        delta     = np.cross(r_cg_main, F_main)

        unknowns['total_force'] = F_main
        unknowns['total_moment'] = M_main + delta
        myM= np.cross(np.array([0.0, 0.0, (z_tower[-1] - unknowns['structure_center_of_mass'][-1])]), F_rna)
        
        # shear and bending (convert from local to global c.s.)
        Nx = forces.Nx[iCase, 1::2]
        Vy = forces.Vy[iCase, 1::2]
        Vz = forces.Vz[iCase, 1::2]

        Tx = forces.Txx[iCase, 1::2]
        My = forces.Myy[iCase, 1::2]
        Mz = forces.Mzz[iCase, 1::2]

        # Compute axial and shear stresses in elements given Frame3DD outputs and some geomtry data
        # Method comes from Section 7.14 of Frame3DD documentation
        # http://svn.code.sourceforge.net/p/frame3dd/code/trunk/doc/Frame3DD-manual.html#structuralmodeling
        M = np.sqrt(My*My + Mz*Mz)
        sigma_ax = Nx/Ax - M/S
        sigma_sh = np.sqrt(Vy*Vy + Vz*Vz)/As + Tx/C

        # Extract pontoon for stress check
        idx  = range(mainEID-1)
        npon = len(idx)
        if len(idx) > 0:
            qdyn_pontoon = np.max( np.abs( np.r_[params['main_qdyn'], params['offset_qdyn']] ) )
            sigma_ax_pon = sigma_ax[idx]
            sigma_sh_pon = sigma_sh[idx]
            sigma_h_pon  = util.hoopStress(2*R_od_pontoon, t_wall_pontoon, qdyn_pontoon) * np.ones(sigma_ax_pon.shape)

            unknowns['pontoon_stress'][:npon] = util.vonMisesStressUtilization(sigma_ax_pon, sigma_h_pon, sigma_sh_pon,
                                                                               gamma_f*gamma_m*gamma_n, sigma_y)
        
        # Extract tower for Eurocode checks
        idx = towerEID-1 + np.arange(R_od_tower.size, dtype=np.int32)
        L_reinforced   = params['tower_buckling_length'] * np.ones(idx.shape)
        sigma_ax_tower = sigma_ax[idx]
        sigma_sh_tower = sigma_sh[idx]
        qdyn_tower,_   = nodal2sectional( params['tower_qdyn'] )
        sigma_h_tower  = util.hoopStress(2*R_od_tower, t_wall_tower*np.cos(angle_tower), qdyn_tower)

        unknowns['tower_stress:axial'] = sigma_ax_tower
        unknowns['tower_stress:shear'] = sigma_sh_tower
        unknowns['tower_stress:hoop']  = sigma_h_tower
        unknowns['tower_stress:hoopStiffen'] = util.hoopStressEurocode(z_tower, 2*R_od_tower, t_wall_tower, L_reinforced, qdyn_tower)
        unknowns['tower_stress'] = util.vonMisesStressUtilization(sigma_ax_tower, sigma_h_tower, sigma_sh_tower,
                                                                  gamma_f*gamma_m*gamma_n, sigma_y)

        sigma_y_vec = sigma_y * np.ones(idx.shape)
        unknowns['tower_shell_buckling'] = util.shellBucklingEurocode(2*R_od_tower, t_wall_tower, sigma_ax_tower, sigma_h_tower, sigma_sh_tower,
                                                                      L_reinforced, modE[idx], sigma_y_vec, gamma_f, gamma_b)

        tower_height = z_tower[-1] - z_tower[0]
        unknowns['tower_global_buckling'] = util.bucklingGL(2*R_od_tower, t_wall_tower, Nx[idx], M[idx], tower_height, modE[idx], sigma_y_vec, gamma_f, gamma_b)
        
        # Extract main column for Eurocode checks
        idx = mainEID-1 + np.arange(R_od_main.size, dtype=np.int32)
        L_reinforced  = params['main_buckling_length']
        sigma_ax_main = sigma_ax[idx]
        sigma_sh_main = sigma_sh[idx]
        qdyn_main,_   = nodal2sectional( params['main_qdyn'] )
        sigma_h_main  = util.hoopStress(2*R_od_main, t_wall_main*np.cos(angle_main), qdyn_main)

        unknowns['main_stress:axial'] = sigma_ax_main
        unknowns['main_stress:shear'] = sigma_sh_main
        unknowns['main_stress:hoop']  = sigma_h_main
        unknowns['main_stress:hoopStiffen'] = util.hoopStressEurocode(z_main, 2*R_od_main, t_wall_main, L_reinforced, qdyn_main)
        unknowns['main_stress'] = util.vonMisesStressUtilization(sigma_ax_main, sigma_h_main, sigma_sh_main,
                                                                        gamma_f*gamma_m*gamma_n, sigma_y)

        sigma_y_vec = sigma_y * np.ones(idx.shape)
        unknowns['main_shell_buckling'] = util.shellBucklingEurocode(2*R_od_main, t_wall_main, sigma_ax_main, sigma_h_main, sigma_sh_main,
                                                                            L_reinforced, modE[idx], sigma_y_vec, gamma_f, gamma_b)

        main_height = z_main[-1] - z_main[0]
        unknowns['main_global_buckling'] = util.bucklingGL(2*R_od_main, t_wall_main, Nx[idx], M[idx], main_height, modE[idx], sigma_y_vec, gamma_f, gamma_b)

        
        # Extract offset column for Eurocode checks
        if ncolumn > 0:
            idx = offsetEID[0]-1 + np.arange(R_od_offset.size, dtype=np.int32)
            L_reinforced     = params['offset_buckling_length']
            sigma_ax_offset = sigma_ax[idx]
            sigma_sh_offset = sigma_sh[idx]
            qdyn_offset,_   = nodal2sectional( params['offset_qdyn'] )
            sigma_h_offset  = util.hoopStress(2*R_od_offset, t_wall_offset*np.cos(angle_offset), qdyn_offset)

            unknowns['offset_stress:axial'] = sigma_ax_offset
            unknowns['offset_stress:shear'] = sigma_sh_offset
            unknowns['offset_stress:hoop']  = sigma_h_offset
            unknowns['offset_stress:hoopStiffen'] = util.hoopStressEurocode(z_offset, 2*R_od_offset, t_wall_offset, L_reinforced, qdyn_offset)
            unknowns['offset_stress'] = util.vonMisesStressUtilization(sigma_ax_offset, sigma_h_offset, sigma_sh_offset,
                                                                            gamma_f*gamma_m*gamma_n, sigma_y)

            sigma_y_vec = sigma_y * np.ones(idx.shape)
            unknowns['offset_shell_buckling'] = util.shellBucklingEurocode(2*R_od_offset, t_wall_offset, sigma_ax_offset, sigma_h_offset, sigma_sh_offset,
                                                                                L_reinforced, modE[idx], sigma_y_vec, gamma_f, gamma_b)

            offset_height = z_offset[-1] - z_offset[0]
            unknowns['offset_global_buckling'] = util.bucklingGL(2*R_od_offset, t_wall_offset, Nx[idx], M[idx], offset_height, modE[idx], sigma_y_vec, gamma_f, gamma_b)
        
        # TODO: FATIGUE
        # Base and offset columns get API stress/buckling checked in Column Group because that takes into account stiffeners



class TrussIntegerToBoolean(Component):
    def __init__(self):
        super(TrussIntegerToBoolean,self).__init__()
        self.add_param('cross_attachment_pontoons_int', val=1, desc='Inclusion of pontoons that connect the bottom of the central main to the tops of the outer offset columns')
        self.add_param('lower_attachment_pontoons_int', val=1, desc='Inclusion of pontoons that connect the central main to the outer offset columns at their bottoms')
        self.add_param('upper_attachment_pontoons_int', val=1, desc='Inclusion of pontoons that connect the central main to the outer offset columns at their tops')
        self.add_param('lower_ring_pontoons_int', val=1, desc='Inclusion of pontoons that ring around outer offset columns at their bottoms')
        self.add_param('upper_ring_pontoons_int', val=1, desc='Inclusion of pontoons that ring around outer offset columns at their tops')
        self.add_param('outer_cross_pontoons_int', val=1, desc='Inclusion of pontoons that ring around outer offset columns at their tops')

        self.add_output('cross_attachment_pontoons', val=True, desc='Inclusion of pontoons that connect the bottom of the central main to the tops of the outer offset columns', pass_by_obj=True)
        self.add_output('lower_attachment_pontoons', val=True, desc='Inclusion of pontoons that connect the central main to the outer offset columns at their bottoms', pass_by_obj=True)
        self.add_output('upper_attachment_pontoons', val=True, desc='Inclusion of pontoons that connect the central main to the outer offset columns at their tops', pass_by_obj=True)
        self.add_output('lower_ring_pontoons', val=True, desc='Inclusion of pontoons that ring around outer offset columns at their bottoms', pass_by_obj=True)
        self.add_output('upper_ring_pontoons', val=True, desc='Inclusion of pontoons that ring around outer offset columns at their tops', pass_by_obj=True)
        self.add_output('outer_cross_pontoons', val=True, desc='Inclusion of pontoons that ring around outer offset columns at their tops', pass_by_obj=True)

    def solve_nonlinear(self, params, unknowns, resids):
        unknowns['cross_attachment_pontoons'] = (int(params['cross_attachment_pontoons_int']) == 1)
        unknowns['lower_attachment_pontoons'] = (int(params['lower_attachment_pontoons_int']) == 1)
        unknowns['upper_attachment_pontoons'] = (int(params['upper_attachment_pontoons_int']) == 1)
        unknowns['lower_ring_pontoons']       = (int(params['lower_ring_pontoons_int']) == 1)
        unknowns['upper_ring_pontoons']       = (int(params['upper_ring_pontoons_int']) == 1)
        unknowns['outer_cross_pontoons']      = (int(params['outer_cross_pontoons_int']) == 1)

        
# -----------------
#  Assembly
# -----------------

class Loading(Group):

    def __init__(self, nFull, nFullTow):
        super(Loading, self).__init__()
        
        # Independent variables that are unique to TowerSE
        self.add('main_pontoon_attach_lower',  IndepVarComp('main_pontoon_attach_lower', 0.0), promotes=['*'])
        self.add('main_pontoon_attach_upper',  IndepVarComp('main_pontoon_attach_upper', 0.0), promotes=['*'])
        self.add('pontoon_outer_diameter',     IndepVarComp('pontoon_outer_diameter', 0.0), promotes=['*'])
        self.add('pontoon_wall_thickness',     IndepVarComp('pontoon_wall_thickness', 0.0), promotes=['*'])
        self.add('outer_cross_pontoons_int',       IndepVarComp('outer_cross_pontoons_int', 1), promotes=['*'])
        self.add('cross_attachment_pontoons_int',  IndepVarComp('cross_attachment_pontoons_int', 1), promotes=['*'])
        self.add('lower_attachment_pontoons_int',  IndepVarComp('lower_attachment_pontoons_int', 1), promotes=['*'])
        self.add('upper_attachment_pontoons_int',  IndepVarComp('upper_attachment_pontoons_int', 1), promotes=['*'])
        self.add('lower_ring_pontoons_int',        IndepVarComp('lower_ring_pontoons_int', 1), promotes=['*'])
        self.add('upper_ring_pontoons_int',        IndepVarComp('upper_ring_pontoons_int', 1), promotes=['*'])
        self.add('connection_ratio_max',       IndepVarComp('connection_ratio_max', 0.0), promotes=['*'])
        self.add('fairlead_support_outer_diameter',     IndepVarComp('fairlead_support_outer_diameter', 0.0), promotes=['*'])
        self.add('fairlead_support_wall_thickness',     IndepVarComp('fairlead_support_wall_thickness', 0.0), promotes=['*'])

        # All the components
        self.add('loadingWind', PowerWind(nFullTow), promotes=['z0','Uref','shearExp','zref'])
        self.add('windLoads', CylinderWindDrag(nFullTow), promotes=['cd_usr','beta'])
        self.add('intbool', TrussIntegerToBoolean(), promotes=['*'])
        self.add('frame', FloatingFrame(nFull, nFullTow), promotes=['*'])
        
        # Connections for geometry and mass
        self.connect('loadingWind.z', ['windLoads.z', 'tower_z_full'])
        self.connect('windLoads.d', ['tower_d_full'])
        self.connect('loadingWind.U', 'windLoads.U')

        # connections to distLoads1
        self.connect('windLoads.windLoads_Px', 'tower_Px')
        self.connect('windLoads.windLoads_Py', 'tower_Py')
        self.connect('windLoads.windLoads_Pz', 'tower_Pz')
        self.connect('windLoads.windLoads_qdyn', 'tower_qdyn')

