import numpy as np
import wisdem.pyframe3dd.pyframe3dd as pyframe3dd
import openmdao.api as om
from wisdem.commonse.utilities import nodal2sectional

from wisdem.commonse import gravity, eps, NFREQ
import wisdem.commonse.utilization_constraints as util
from wisdem.commonse.utilities import get_modal_coefficients
import wisdem.commonse.manufacturing as manufacture
from wisdem.commonse.wind_wave_drag import CylinderWindDrag
from wisdem.commonse.environment import PowerWind
from wisdem.commonse.vertical_cylinder import get_nfull, RIGID
from wisdem.commonse.cross_sections import Tube
from .map_mooring import NLINES_MAX


def find_nearest(array,value):
    return (np.abs(array-value)).argmin() 

        
def ghostNodes(x1, x2, r1, r2):
    dx = x2 - x1
    L = np.sqrt( np.sum( dx**2 ) )
    dr1 = (      r1/L) * dx + x1
    dr2 = (1.0 - r2/L) * dx + x1
    return dr1, dr2

class FloatingFrame(om.ExplicitComponent):
    """
    Component for semisubmersible pontoon / truss structure for floating offshore wind turbines.
    Should be tightly coupled with Semi and Mooring classes for full system representation.
    
    Parameters
    ----------
    rho_water : float, [kg/m**3]
        density of water
    hsig_wave : float, [m]
        wave significant height
    main_z_full : numpy array[n_full_main], [m]
        z-coordinates of section nodes (length = nsection+1)
    main_d_full : numpy array[n_full_main], [m]
        outer radius at each section node bottom to top (length = nsection + 1)
    main_t_full : numpy array[n_full_main-1], [m]
        shell wall thickness at each section node bottom to top (length = nsection + 1)
    main_rho : numpy array[n_full_main-1], [kg/m**3]
        density of material
    main_E : numpy array[n_full_main-1], [Pa]
        Modulus of elasticity (Youngs) of material
    main_G : numpy array[n_full_main-1], [Pa]
        Shear modulus of material
    main_sigma_y : numpy array[n_full_main-1], [Pa]
        yield stress of material
    main_mass : numpy array[n_full_main-1], [kg]
        mass of main column by section
    main_displaced_volume : numpy array[n_full_main-1], [m**3]
        column volume of water displaced by section
    main_hydrostatic_force : numpy array[n_full_main-1], [N]
        Net z-force of hydrostatic pressure by section
    main_center_of_buoyancy : float, [m]
        z-position of center of column buoyancy force
    main_center_of_mass : float, [m]
        z-position of center of column mass
    main_Px : numpy array[n_full_main], [N/m]
        force per unit length in x-direction on main
    main_Py : numpy array[n_full_main], [N/m]
        force per unit length in y-direction on main
    main_Pz : numpy array[n_full_main], [N/m]
        force per unit length in z-direction on main
    main_qdyn : numpy array[n_full_main], [N/m**2]
        dynamic pressure on main
    main_pontoon_attach_upper : float
        Fraction of main column for upper truss attachment on main column
    main_pontoon_attach_lower : float
        Fraction of main column lower truss attachment on main column
    offset_z_full : numpy array[n_full_off], [m]
        z-coordinates of section nodes (length = nsection+1)
    offset_d_full : numpy array[n_full_off], [m]
        outer radius at each section node bottom to top (length = nsection + 1)
    offset_t_full : numpy array[n_full_off-1], [m]
        shell wall thickness at each section node bottom to top (length = nsection + 1)
    offset_rho : numpy array[n_full_off-1], [kg/m**3]
        density of material
    offset_E : numpy array[n_full_off-1], [Pa]
        Modulus of elasticity (Youngs) of material
    offset_G : numpy array[n_full_off-1], [Pa]
        Shear modulus of material
    offset_sigma_y : numpy array[n_full_off-1], [Pa]
        yield stress of material
    offset_mass : numpy array[n_full_off-1], [kg]
        mass of offset column by section
    offset_displaced_volume : numpy array[n_full_off-1], [m**3]
        column volume of water displaced by section
    offset_hydrostatic_force : numpy array[n_full_off-1], [N]
        Net z-force of hydrostatic pressure by section
    offset_center_of_buoyancy : float, [m]
        z-position of center of column buoyancy force
    offset_center_of_mass : float, [m]
        z-position of center of column mass
    offset_Px : numpy array[n_full_off], [N/m]
        force per unit length in x-direction on offset
    offset_Py : numpy array[n_full_off], [N/m]
        force per unit length in y-direction on offset
    offset_Pz : numpy array[n_full_off], [N/m]
        force per unit length in z-direction on offset
    offset_qdyn : numpy array[n_full_off], [N/m**2]
        dynamic pressure on offset
    tower_z_full : numpy array[n_full_tow], [m]
        z-coordinates of section nodes (length = nsection+1)
    tower_d_full : numpy array[n_full_tow], [m]
        outer radius at each section node bottom to top (length = nsection + 1)
    tower_t_full : numpy array[n_full_tow-1], [m]
        shell wall thickness at each section node bottom to top (length = nsection + 1)
    tower_rho : numpy array[n_full_tow-1], [kg/m**3]
        density of material
    tower_E : numpy array[n_full_tow-1], [Pa]
        Modulus of elasticity (Youngs) of material
    tower_G : numpy array[n_full_tow-1], [Pa]
        Shear modulus of material
    tower_sigma_y : numpy array[n_full_tow-1], [Pa]
        yield stress of material
    tower_mass_section : numpy array[n_full_tow-1], [kg]
        mass of tower column by section
    tower_center_of_mass : float, [m]
        z-position of center of tower mass
    tower_Px : numpy array[n_full_tow], [N/m]
        force per unit length in x-direction on tower
    tower_Py : numpy array[n_full_tow], [N/m]
        force per unit length in y-direction on tower
    tower_Pz : numpy array[n_full_tow], [N/m]
        force per unit length in z-direction on tower
    tower_qdyn : numpy array[n_full_tow], [N/m**2]
        dynamic pressure on tower
    radius_to_offset_column : float, [m]
        Distance from main column centerpoint to offset column centerpoint
    number_of_offset_columns : float
        Number of offset columns evenly spaced around main column
    pontoon_outer_diameter : float, [m]
        Outer radius of tubular pontoon that connects offset or main columns
    pontoon_wall_thickness : float, [m]
        Inner radius of tubular pontoon that connects offset or main columns
    cross_attachment_pontoons : TODO: add type by hand, could not be parsed automatically
        Inclusion of pontoons that connect the bottom of the central main to the tops of
        the outer offset columns
    lower_attachment_pontoons : TODO: add type by hand, could not be parsed automatically
        Inclusion of pontoons that connect the central main to the outer offset columns
        at their bottoms
    upper_attachment_pontoons : TODO: add type by hand, could not be parsed automatically
        Inclusion of pontoons that connect the central main to the outer offset columns
        at their tops
    lower_ring_pontoons : TODO: add type by hand, could not be parsed automatically
        Inclusion of pontoons that ring around outer offset columns at their bottoms
    upper_ring_pontoons : TODO: add type by hand, could not be parsed automatically
        Inclusion of pontoons that ring around outer offset columns at their tops
    outer_cross_pontoons : TODO: add type by hand, could not be parsed automatically
        Inclusion of pontoons that ring around outer offset columns at their tops
    rna_mass : float, [kg]
        mass of tower
    rna_cg : numpy array[3], [m]
        Location of RNA center of mass relative to tower top
    rna_force : numpy array[3], [N]
        Force in xyz-direction on turbine
    rna_moment : numpy array[3], [N*m]
        Moments about turbine main
    rna_I : numpy array[6], [kg*m**2]
        Moments about turbine main
    number_of_mooring_connections : float
        number of mooring connections on vessel
    mooring_lines_per_connection : float
        number of mooring lines per connection
    mooring_neutral_load : numpy array[NLINES_MAX, 3], [N]
        z-force of mooring lines on structure
    mooring_stiffness : numpy array[6, 6], [N/m]
        Linearized stiffness matrix of mooring system at neutral (no offset) conditions.
    mooring_moments_of_inertia : numpy array[6], [kg*m**2]
        mass moment of inertia of mooring system about fairlead-centerline point [xx yy
        zz xy xz yz]
    fairlead : float, [m]
        Depth below water for mooring line attachment
    fairlead_radius : float, [m]
        Radius from center of structure to fairlead connection points
    fairlead_support_outer_diameter : float, [m]
        fairlead support outer diameter
    fairlead_support_wall_thickness : float, [m]
        fairlead support wall thickness
    connection_ratio_max : float
        Maximum ratio of pontoon outer diameter to main/offset outer diameter
    material_cost_rate : float, [USD/kg]
        Raw material cost rate: steel $1.1/kg, aluminum $3.5/kg
    labor_cost_rate : float, [USD/min]
        Labor cost rate
    painting_cost_rate : float, [USD/m/m]
        Painting / surface finishing cost rate
    
    Returns
    -------
    pontoon_wave_height_depth_margin : numpy array[2], [m]
        Distance between attachment point of pontoons and wave crest- both above and
        below waterline
    pontoon_cost : float, [USD]
        Cost of pontoon elements and connecting truss
    pontoon_cost_rate : float, [USD/t]
        Cost rate of finished pontoon and truss
    pontoon_mass : float, [kg]
        Mass of pontoon elements and connecting truss
    pontoon_displacement : float, [m**3]
        Buoyancy force of submerged pontoon elements
    pontoon_center_of_buoyancy : float, [m]
        z-position of center of pontoon buoyancy force
    pontoon_center_of_mass : float, [m]
        z-position of center of pontoon mass
    top_deflection : float, [m]
        Deflection of tower top in yaw-aligned +x direction
    pontoon_stress : numpy array[70, ]
        Utilization (<1) of von Mises stress by yield stress and safety factor for all
        pontoon elements
    main_stress : numpy array[n_full_main-1]
        Von Mises stress utilization along main column at specified locations. Incudes
        safety factor.
    main_stress:axial : numpy array[n_full_main-1]
        Axial stress along main column at specified locations.
    main_stress:shear : numpy array[n_full_main-1]
        Shear stress along main column at specified locations.
    main_stress:hoop : numpy array[n_full_main-1]
        Hoop stress along main column at specified locations.
    main_stress:hoopStiffen : numpy array[n_full_main-1]
        Hoop stress along main column at specified locations.
    main_shell_buckling : numpy array[n_full_main-1]
        Shell buckling constraint. Should be < 1 for feasibility. Includes safety
        factors
    main_global_buckling : numpy array[n_full_main-1]
        Global buckling constraint. Should be < 1 for feasibility. Includes safety
        factors
    offset_stress : numpy array[n_full_off-1]
        Von Mises stress utilization along offset column at specified locations. Incudes
        safety factor.
    offset_stress:axial : numpy array[n_full_off-1]
        Axial stress along offset column at specified locations.
    offset_stress:shear : numpy array[n_full_off-1]
        Shear stress along offset column at specified locations.
    offset_stress:hoop : numpy array[n_full_off-1]
        Hoop stress along offset column at specified locations.
    offset_stress:hoopStiffen : numpy array[n_full_off-1]
        Hoop stress along offset column at specified locations.
    offset_shell_buckling : numpy array[n_full_off-1]
        Shell buckling constraint. Should be < 1 for feasibility. Includes safety
        factors
    offset_global_buckling : numpy array[n_full_off-1]
        Global buckling constraint. Should be < 1 for feasibility. Includes safety
        factors
    tower_stress : numpy array[n_full_tow-1]
        Von Mises stress utilization along tower at specified locations. incudes safety
        factor.
    tower_stress:axial : numpy array[n_full_tow-1]
        Axial stress along tower column at specified locations.
    tower_stress:shear : numpy array[n_full_tow-1]
        Shear stress along tower column at specified locations.
    tower_stress:hoop : numpy array[n_full_tow-1]
        Hoop stress along tower column at specified locations.
    tower_stress:hoopStiffen : numpy array[n_full_tow-1]
        Hoop stress along tower column at specified locations.
    tower_shell_buckling : numpy array[n_full_tow-1]
        Shell buckling constraint. Should be < 1 for feasibility. Includes safety
        factors
    tower_global_buckling : numpy array[n_full_tow-1]
        Global buckling constraint. Should be < 1 for feasibility. Includes safety
        factors
    plot_matrix : numpy array[]
        Ratio of shear stress to yield stress for all pontoon elements
    main_connection_ratio : numpy array[n_full_main]
        Ratio of pontoon outer diameter to main outer diameter
    offset_connection_ratio : numpy array[n_full_off]
        Ratio of pontoon outer diameter to main outer diameter
    structural_frequencies : numpy array[NFREQ], [Hz]
        First six natural frequencies
    x_mode_shapes : numpy array[NFREQ/2]
        6-degree polynomial coefficients of mode shapes in the x-direction
    y_mode_shapes : numpy array[NFREQ/2]
        6-degree polynomial coefficients of mode shapes in the y-direction
    substructure_mass : float, [kg]
        Mass of substructure elements and connecting truss
    structural_mass : float, [kg]
        Mass of whole turbine except for mooring lines
    total_displacement : float, [m**3]
        Total volume of water displaced by floating turbine (except for mooring lines)
    z_center_of_buoyancy : float, [m]
        z-position of center of buoyancy of whole turbine
    substructure_center_of_mass : numpy array[3], [m]
        xyz-position of center of gravity of substructure only
    structure_center_of_mass : numpy array[3], [m]
        xyz-position of center of gravity of whole turbine
    total_force : numpy array[3], [N]
        Net forces on turbine
    total_moment : numpy array[3], [N*m]
        Moments on whole turbine
    
    """

    def initialize(self):
        self.options.declare('n_height_main')
        self.options.declare('n_height_off')
        self.options.declare('n_height_tow')
        self.options.declare('modeling_options')
        
    def setup(self):
        n_height_main = self.options['n_height_main']
        n_height_off  = self.options['n_height_off']
        n_height_tow  = self.options['n_height_tow']
        n_full_main   = get_nfull(n_height_main)
        n_full_off    = get_nfull(n_height_off)
        n_full_tow    = get_nfull(n_height_tow)

        # Keep Frame3DD data object for easy testing and debugging
        self.myframe = None

        self.add_input('rho_water', val=0.0, units='kg/m**3')
        self.add_input('hsig_wave', val=0.0, units='m')
        
        # Base column
        self.add_input('main_z_full', val=np.zeros(n_full_main), units='m')
        self.add_input('main_d_full', val=np.zeros(n_full_main), units='m')
        self.add_input('main_t_full', val=np.zeros(n_full_main-1), units='m')
        self.add_input('main_rho_full', val=np.zeros(n_full_main-1), units='kg/m**3')
        self.add_input('main_E_full', val=np.zeros(n_full_main-1), units='Pa')
        self.add_input('main_G_full', val=np.zeros(n_full_main-1), units='Pa')
        self.add_input('main_sigma_y_full', val=np.zeros(n_full_main-1), units='Pa')
        self.add_input('main_mass', val=np.zeros(n_full_main-1), units='kg')
        self.add_input('main_displaced_volume', val=np.zeros(n_full_main-1), units='m**3')
        self.add_input('main_hydrostatic_force', val=np.zeros(n_full_main-1), units='N')
        self.add_input('main_center_of_buoyancy', val=0.0, units='m')
        self.add_input('main_center_of_mass', val=0.0, units='m')
        self.add_input('main_Px', np.zeros(n_full_main), units='N/m')
        self.add_input('main_Py', np.zeros(n_full_main), units='N/m')
        self.add_input('main_Pz', np.zeros(n_full_main), units='N/m')
        self.add_input('main_qdyn', np.zeros(n_full_main), units='N/m**2')
        
        self.add_input('main_pontoon_attach_upper', val=0.0)
        self.add_input('main_pontoon_attach_lower', val=0.0)
        
        # offset columns
        self.add_input('offset_z_full', val=np.zeros(n_full_off), units='m')
        self.add_input('offset_d_full', val=np.zeros(n_full_off), units='m')
        self.add_input('offset_t_full', val=np.zeros(n_full_off-1), units='m')
        self.add_input('offset_rho_full', val=np.zeros(n_full_main-1), units='kg/m**3')
        self.add_input('offset_E_full', val=np.zeros(n_full_main-1), units='Pa')
        self.add_input('offset_G_full', val=np.zeros(n_full_main-1), units='Pa')
        self.add_input('offset_sigma_y_full', val=np.zeros(n_full_main-1), units='Pa')
        self.add_input('offset_mass', val=np.zeros(n_full_off-1), units='kg')
        self.add_input('offset_displaced_volume', val=np.zeros(n_full_off-1), units='m**3')
        self.add_input('offset_hydrostatic_force', val=np.zeros(n_full_off-1), units='N')
        self.add_input('offset_center_of_buoyancy', val=0.0, units='m')
        self.add_input('offset_center_of_mass', val=0.0, units='m')
        self.add_input('offset_Px', np.zeros(n_full_off), units='N/m')
        self.add_input('offset_Py', np.zeros(n_full_off), units='N/m')
        self.add_input('offset_Pz', np.zeros(n_full_off), units='N/m')
        self.add_input('offset_qdyn', np.zeros(n_full_off), units='N/m**2')
        
        # Tower
        self.add_input('tower_z_full', val=np.zeros(n_full_tow), units='m')
        self.add_input('tower_d_full', val=np.zeros(n_full_tow), units='m')
        self.add_input('tower_t_full', val=np.zeros(n_full_tow-1), units='m')
        self.add_input('tower_rho_full', val=np.zeros(n_full_main-1), units='kg/m**3')
        self.add_input('tower_E_full', val=np.zeros(n_full_main-1), units='Pa')
        self.add_input('tower_G_full', val=np.zeros(n_full_main-1), units='Pa')
        self.add_input('tower_sigma_y_full', val=np.zeros(n_full_main-1), units='Pa')
        self.add_input('tower_mass_section', val=np.zeros(n_full_tow-1), units='kg')
        self.add_input('tower_center_of_mass', val=0.0, units='m')
        self.add_input('tower_Px', np.zeros(n_full_tow), units='N/m')
        self.add_input('tower_Py', np.zeros(n_full_tow), units='N/m')
        self.add_input('tower_Pz', np.zeros(n_full_tow), units='N/m')
        self.add_input('tower_qdyn', np.zeros(n_full_tow), units='N/m**2')
        
        # Semi geometry
        self.add_input('radius_to_offset_column', val=0.0, units='m')
        self.add_input('number_of_offset_columns', val=3)
        
        # Pontoon properties
        self.add_input('pontoon_outer_diameter', val=0.0, units='m')
        self.add_input('pontoon_wall_thickness', val=0.0, units='m')
        self.add_discrete_input('cross_attachment_pontoons', val=True)
        self.add_discrete_input('lower_attachment_pontoons', val=True)
        self.add_discrete_input('upper_attachment_pontoons', val=True)
        self.add_discrete_input('lower_ring_pontoons', val=True)
        self.add_discrete_input('upper_ring_pontoons', val=True)
        self.add_discrete_input('outer_cross_pontoons', val=True)
        
        # Turbine parameters
        self.add_input('rna_mass', val=0.0, units='kg')
        self.add_input('rna_cg', val=np.zeros(3), units='m')
        self.add_input('rna_force', val=np.zeros(3), units='N')
        self.add_input('rna_moment', val=np.zeros(3), units='N*m')
        self.add_input('rna_I', val=np.zeros(6), units='kg*m**2')
        
        # Mooring parameters for loading
        self.add_input('number_of_mooring_connections', val=3)
        self.add_input('mooring_lines_per_connection', val=1)
        self.add_input('mooring_neutral_load', val=np.zeros((NLINES_MAX, 3)), units='N')
        self.add_input('mooring_stiffness', val=np.zeros((6, 6)), units='N/m')
        self.add_input('mooring_moments_of_inertia', val=np.zeros(6), units='kg*m**2')
        self.add_input('fairlead', val=0.0, units='m')
        self.add_input('fairlead_radius', val=0.0, units='m')
        self.add_input('fairlead_support_outer_diameter', val=0.0, units='m')
        self.add_input('fairlead_support_wall_thickness', val=0.0, units='m')
        
        # Manufacturing
        self.add_input('connection_ratio_max', val=0.0)
        
        # Costing
        self.add_input('material_cost_rate', 0.0, units='USD/kg')
        self.add_input('labor_cost_rate', 0.0, units='USD/min')
        self.add_input('painting_cost_rate', 0.0, units='USD/m/m')

        self.add_output('pontoon_wave_height_depth_margin', val=np.zeros(2), units='m')
        self.add_output('pontoon_cost', val=0.0, units='USD')
        self.add_output('pontoon_cost_rate', val=0.0, units='USD/t')
        self.add_output('pontoon_mass', val=0.0, units='kg')
        self.add_output('pontoon_displacement', val=0.0, units='m**3')
        self.add_output('pontoon_center_of_buoyancy', val=0.0, units='m')
        self.add_output('pontoon_center_of_mass', val=0.0, units='m')
        
        self.add_output('top_deflection', 0.0, units='m')
        self.add_output('pontoon_stress', val=np.zeros(70))
        
        self.add_output('main_stress', np.zeros(n_full_main-1))
        self.add_output('main_stress:axial', np.zeros(n_full_main-1))
        self.add_output('main_stress:shear', np.zeros(n_full_main-1))
        self.add_output('main_stress:hoop', np.zeros(n_full_main-1))
        self.add_output('main_stress:hoopStiffen', np.zeros(n_full_main-1))
        self.add_output('main_shell_buckling', np.zeros(n_full_main-1))
        self.add_output('main_global_buckling', np.zeros(n_full_main-1))
        
        self.add_output('offset_stress', np.zeros(n_full_off-1))
        self.add_output('offset_stress:axial', np.zeros(n_full_off-1))
        self.add_output('offset_stress:shear', np.zeros(n_full_off-1))
        self.add_output('offset_stress:hoop', np.zeros(n_full_off-1))
        self.add_output('offset_stress:hoopStiffen', np.zeros(n_full_off-1))
        self.add_output('offset_shell_buckling', np.zeros(n_full_off-1))
        self.add_output('offset_global_buckling', np.zeros(n_full_off-1))
        
        self.add_output('tower_stress', np.zeros(n_full_tow-1))
        self.add_output('tower_stress:axial', np.zeros(n_full_tow-1))
        self.add_output('tower_stress:shear', np.zeros(n_full_tow-1))
        self.add_output('tower_stress:hoop', np.zeros(n_full_tow-1))
        self.add_output('tower_stress:hoopStiffen', np.zeros(n_full_tow-1))
        self.add_output('tower_shell_buckling', np.zeros(n_full_tow-1))
        self.add_output('tower_global_buckling', np.zeros(n_full_tow-1))
        
        self.add_discrete_output('plot_matrix', val=np.array([]))
        self.add_output('main_connection_ratio', val=np.zeros(n_full_main))
        self.add_output('offset_connection_ratio', val=np.zeros(n_full_off))
        
        NFREQ2 = int(NFREQ/2)
        self.add_output('structural_frequencies', np.zeros(NFREQ), units='Hz')
        self.add_output('x_mode_shapes', val=np.zeros((NFREQ2,5)), desc='6-degree polynomial coefficients of mode shapes in the x-direction')
        self.add_output('y_mode_shapes', val=np.zeros((NFREQ2,5)), desc='6-degree polynomial coefficients of mode shapes in the y-direction')
        self.add_output('substructure_mass', val=0.0, units='kg')
        self.add_output('structural_mass', val=0.0, units='kg')
        self.add_output('total_displacement', val=0.0, units='m**3')
        self.add_output('z_center_of_buoyancy', val=0.0, units='m')
        self.add_output('substructure_center_of_mass', val=np.zeros(3), units='m')
        self.add_output('structure_center_of_mass', val=np.zeros(3), units='m')
        self.add_output('total_force', val=np.zeros(3), units='N')
        self.add_output('total_moment', val=np.zeros(3), units='N*m')

        self.declare_partials('*', '*', method='fd', form='central', step=1e-6)
         
    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):


        ncolumn         = int(inputs['number_of_offset_columns'])
        crossAttachFlag = discrete_inputs['cross_attachment_pontoons']
        lowerAttachFlag = discrete_inputs['lower_attachment_pontoons']
        upperAttachFlag = discrete_inputs['upper_attachment_pontoons']
        lowerRingFlag   = discrete_inputs['lower_ring_pontoons']
        upperRingFlag   = discrete_inputs['upper_ring_pontoons']
        outerCrossFlag  = discrete_inputs['outer_cross_pontoons']
        
        R_semi         = inputs['radius_to_offset_column'] if ncolumn>0 else 0.0
        R_od_pontoon   = 0.5*inputs['pontoon_outer_diameter']
        R_od_main      = 0.5*inputs['main_d_full']
        R_od_offset    = 0.5*inputs['offset_d_full']
        R_od_tower     = 0.5*inputs['tower_d_full']
        R_od_fairlead  = 0.5*inputs['fairlead_support_outer_diameter']

        t_wall_main     = inputs['main_t_full']
        t_wall_offset   = inputs['offset_t_full']
        t_wall_pontoon  = inputs['pontoon_wall_thickness']
        t_wall_tower    = inputs['tower_t_full']
        t_wall_fairlead = inputs['fairlead_support_wall_thickness']

        E_main         = inputs['main_E_full']
        E_offset       = inputs['offset_E_full']
        E_tower        = inputs['tower_E_full']

        G_main         = inputs['main_G_full']
        G_offset       = inputs['offset_G_full']
        G_tower        = inputs['tower_G_full']
        
        rho_main       = inputs['main_rho_full']
        rho_offset     = inputs['offset_rho_full']
        rho_tower      = inputs['tower_rho_full']
        
        sigma_y_main   = inputs['main_sigma_y_full']
        sigma_y_offset = inputs['offset_sigma_y_full']
        sigma_y_tower  = inputs['tower_sigma_y_full']
        
        z_main         = inputs['main_z_full']
        z_offset       = inputs['offset_z_full']
        z_tower        = inputs['tower_z_full']
        z_attach_upper = inputs['main_pontoon_attach_upper']*(z_main[-1] - z_main[0]) + z_main[0]
        z_attach_lower = inputs['main_pontoon_attach_lower']*(z_main[-1] - z_main[0]) + z_main[0]
        z_fairlead     = -inputs['fairlead']

        m_main         = inputs['main_mass']
        m_offset       = inputs['offset_mass']
        m_tower        = inputs['tower_mass_section']
        
        m_rna          = float(inputs['rna_mass'])
        F_rna          = inputs['rna_force']
        M_rna          = inputs['rna_moment']
        I_rna          = inputs['rna_I']
        cg_rna         = inputs['rna_cg']
        
        rhoWater       = float(inputs['rho_water'])
        
        V_main         = inputs['main_displaced_volume']
        V_offset       = inputs['offset_displaced_volume']

        F_hydro_main    = inputs['main_hydrostatic_force']
        F_hydro_offset  = inputs['offset_hydrostatic_force']

        z_cb_main      = inputs['main_center_of_buoyancy']
        z_cb_offset    = inputs['offset_center_of_buoyancy']
        
        cg_main        = np.r_[0.0, 0.0, inputs['main_center_of_mass']]
        cg_offset      = np.r_[0.0, 0.0, inputs['offset_center_of_mass']]
        cg_tower       = np.r_[0.0, 0.0, inputs['tower_center_of_mass']]
        
        n_connect      = int(inputs['number_of_mooring_connections'])
        n_lines        = int(inputs['mooring_lines_per_connection'])
        K_mooring      = np.diag( inputs['mooring_stiffness'] )
        I_mooring      = inputs['mooring_moments_of_inertia']
        F_mooring      = inputs['mooring_neutral_load']
        R_fairlead     = inputs['fairlead_radius']

        opt = self.options['modeling_options']
        gamma_f      = opt['gamma_f']
        gamma_m      = opt['gamma_m']
        gamma_n      = opt['gamma_n']
        gamma_b      = opt['gamma_b']
        frame3dd_opt = opt['frame3dd']
        
        # Unpack variables
        # Quick ratio for unknowns
        outputs['main_connection_ratio']    = inputs['connection_ratio_max'] - R_od_pontoon/R_od_main
        outputs['offset_connection_ratio'] = inputs['connection_ratio_max'] - R_od_pontoon/R_od_offset
        outputs['pontoon_wave_height_depth_margin'] = np.abs(np.r_[z_attach_lower, z_attach_upper]) - np.abs(inputs['hsig_wave'])

        
        # --- INPUT CHECKS -----
        # If something fails, we have to tell the optimizer this design is no good
        def bad_input():
            outputs['structural_frequencies'] = 1e30 * np.ones(NFREQ)
            outputs['top_deflection'] = 1e30
            outputs['substructure_mass']  = 1e30
            outputs['structural_mass']    = 1e30
            outputs['total_displacement'] = 1e30
            outputs['z_center_of_buoyancy'] = 0.0
            outputs['substructure_center_of_mass'] = 1e30 * np.ones(3)
            outputs['structure_center_of_mass'] = 1e30 * np.ones(3)
            outputs['total_force'] =  1e30 * np.ones(3)
            outputs['total_moment'] = 1e30 * np.ones(3)
            outputs['tower_stress'] = 1e30 * np.ones(m_tower.shape)
            outputs['tower_shell_buckling'] = 1e30 * np.ones(m_tower.shape)
            outputs['tower_global_buckling'] = 1e30 * np.ones(m_tower.shape)
            outputs['main_stress'] = 1e30 * np.ones(m_main.shape)
            outputs['main_shell_buckling'] = 1e30 * np.ones(m_main.shape)
            outputs['main_global_buckling'] = 1e30 * np.ones(m_main.shape)
            outputs['offset_stress'] = 1e30 * np.ones(m_offset.shape)
            outputs['offset_shell_buckling'] = 1e30 * np.ones(m_offset.shape)
            outputs['offset_global_buckling'] = 1e30 * np.ones(m_offset.shape)
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
                if ( (add1[-1] >= z_main[0]) and (add1[-1] <= z_main[-1]) ):
                    tempID1 = xnode.size + 1
                    xnode   = np.append(xnode, add1[0])
                    ynode   = np.append(ynode, add1[1])
                    znode   = np.append(znode, add1[2])
                    gN1     = np.append(gN1, mainLowerID)
                    gN2     = np.append(gN2, tempID1)
                if ( (add2[-1] >= z_offset[0]) and (add2[-1] <= z_offset[-1]) ):
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
                if ( (add1[-1] >= z_main[0]) and (add1[-1] <= z_main[-1]) ):
                    tempID1 = xnode.size + 1
                    xnode   = np.append(xnode, add1[0])
                    ynode   = np.append(ynode, add1[1])
                    znode   = np.append(znode, add1[2])
                    gN1     = np.append(gN1, mainUpperID)
                    gN2     = np.append(gN2, tempID1)
                if ( (add2[-1] >= z_offset[0]) and (add2[-1] <= z_offset[-1]) ):
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
                if ( (add1[-1] >= z_main[0]) and (add1[-1] <= z_main[-1]) ):
                    tempID1 = xnode.size + 1
                    xnode   = np.append(xnode, add1[0])
                    ynode   = np.append(ynode, add1[1])
                    znode   = np.append(znode, add1[2])
                    gN1     = np.append(gN1, mainLowerID)
                    gN2     = np.append(gN2, tempID1)
                if ( (add2[-1] >= z_offset[0]) and (add2[-1] <= z_offset[-1]) ):
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
                if ( (add1[-1] >= z_offset[0]) and (add1[-1] <= z_offset[-1]) ):
                    tempID1 = xnode.size + 1
                    xnode   = np.append(xnode, add1[0])
                    ynode   = np.append(ynode, add1[1])
                    znode   = np.append(znode, add1[2])
                    gN1     = np.append(gN1, offsetLowerID[k])
                    gN2     = np.append(gN2, tempID1)
                if ( (add2[-1] >= z_offset[0]) and (add2[-1] <= z_offset[-1]) ):
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
                if ( (add1[-1] >= z_offset[0]) and (add1[-1] <= z_offset[-1]) ):
                    tempID1 = xnode.size + 1
                    xnode   = np.append(xnode, add1[0])
                    ynode   = np.append(ynode, add1[1])
                    znode   = np.append(znode, add1[2])
                    gN1     = np.append(gN1, offsetUpperID[k])
                    gN2     = np.append(gN2, tempID1)
                if ( (add2[-1] >= z_offset[0]) and (add2[-1] <= z_offset[-1]) ):
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
                if ( (add2[-1] >= z_offset[0]) and (add2[-1] <= z_offset[-1]) ):
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
        # Assumes same material as main column here
        mytube = Tube(2.0*R_od_pontoon, t_wall_pontoon)
        Ax    = mytube.Area * np.ones(N1.shape)
        As    = mytube.Asx  * np.ones(N1.shape)
        Jx    = mytube.J0   * np.ones(N1.shape)
        I     = mytube.Jxx  * np.ones(N1.shape)
        S     = mytube.S    * np.ones(N1.shape)
        C     = mytube.C    * np.ones(N1.shape)
        modE  = E_main[0]   * np.ones(N1.shape)
        modG  = G_main[0]   * np.ones(N1.shape)
        roll  = 0.0 * np.ones(N1.shape)
        dens  = rho_main[0] * np.ones(N1.shape)

        # Add in fairlead support elements
        mooringEID = N1.size + 1
        mytube  = Tube(2.0*R_od_fairlead, t_wall_fairlead)
        rho_fair = rho_main[0] if ncolumn==0 else rho_offset[0]
        E_fair   = E_main[0]   if ncolumn==0 else E_offset[0]
        G_fair   = G_main[0]   if ncolumn==0 else G_offset[0]
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
            modE    = np.append(modE, E_fair )
            modG    = np.append(modG, G_fair )
            roll    = np.append(roll, 0.0 )
            dens    = np.append(dens, rho_fair )

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
        mydens  = m_main / mytube.Area / np.diff(z_main) + eps # includes stiffeners, bulkheads, outfitting
        N1   = np.append(N1  , myrange + mainBeginID    )
        N2   = np.append(N2  , myrange + mainBeginID + 1)
        Ax   = np.append(Ax  , mytube.Area )
        As   = np.append(As  , mytube.Asx )
        Jx   = np.append(Jx  , mytube.J0 )
        I    = np.append(I   , mytube.Jxx )
        S    = np.append(S   , mytube.S )
        C    = np.append(C   , mytube.C )
        modE = np.append(modE, E_main )
        modG = np.append(modG, G_main )
        roll = np.append(roll, np.zeros(myones.shape) )
        dens = np.append(dens, mydens )

        # Tower column
        towerEID = N1.size + 1
        mytube  = Tube(2.0*R_od_tower, t_wall_tower)
        myrange = np.arange(R_od_tower.size)
        myones  = np.ones(myrange.shape)
        mydens  = m_tower / mytube.Area / np.diff(z_tower) + eps # includes stiffeners, bulkheads, outfitting
        N1   = np.append(N1  , myrange + towerBeginID    )
        N2   = np.append(N2  , myrange + towerBeginID + 1)
        Ax   = np.append(Ax  , mytube.Area )
        As   = np.append(As  , mytube.Asx )
        Jx   = np.append(Jx  , mytube.J0 )
        I    = np.append(I   , mytube.Jxx )
        S    = np.append(S   , mytube.S )
        C    = np.append(C   , mytube.C )
        modE = np.append(modE, E_tower )
        modG = np.append(modG, G_tower )
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
        mydens     = m_offset / mytube.Area / np.diff(z_offset) + eps # includes stiffeners, bulkheads, outfitting
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
            modE = np.append(modE, E_offset )
            modG = np.append(modG, G_offset )
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
        nodes   = pyframe3dd.NodeData(nnode, xnode, ynode, znode, myrnode)
        nodeMat = np.c_[xnode, ynode, znode]

        # Create Element Data object
        nelem    = 1 + np.arange(N1.size)
        elements = pyframe3dd.ElementData(nelem, N1, N2, Ax, As, As, Jx, I, I, modE, modG, roll, dens)

        # Store data for plotting, also handy for operations below
        plotMat = np.zeros((mainEID, 3, 2))
        myn1 = N1[:mainEID]
        myn2 = N2[:mainEID]
        plotMat[:,:,0] = nodeMat[myn1-1,:]
        plotMat[:,:,1] = nodeMat[myn2-1,:]
        discrete_outputs['plot_matrix'] = plotMat
        
        # Compute length and center of gravity for each element for use below
        elemL   = np.sqrt( np.sum( np.diff(plotMat, axis=2)**2.0, axis=1) ).flatten()
        elemCoG = 0.5*np.sum(plotMat, axis=2)
        # Get vertical angle as a measure of welding prep difficulty
        elemAng = np.arccos( np.diff(plotMat[:,-1,:], axis=-1).flatten() / elemL )

        # ---Options object---
        other = pyframe3dd.Options(frame3dd_opt['shear'], frame3dd_opt['geom'], float(frame3dd_opt['dx']))

        # ---LOAD CASES---
        # Extreme loading
        gx = gy = 0.0
        gz = -gravity
        load = pyframe3dd.StaticLoadCase(gx, gy, gz)
        load0 = pyframe3dd.StaticLoadCase(gx, gy, gz)

        # Wind + Wave loading in local main / offset / tower c.s.
        Px_main,    Py_main,    Pz_main    = inputs['main_Pz'], inputs['main_Py'], -inputs['main_Px']  # switch to local c.s.
        Px_offset, Py_offset, Pz_offset = inputs['offset_Pz'], inputs['offset_Py'], -inputs['offset_Px']  # switch to local c.s.
        Px_tower,   Py_tower,   Pz_tower   = inputs['tower_Pz'], inputs['tower_Py'], -inputs['tower_Px']  # switch to local c.s.
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
        z_cb    = np.zeros(3)
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
            outputs['pontoon_displacement'] = V_pontoon
            outputs['pontoon_center_of_buoyancy'] = z_cb

            # Sum up mass and compute CofG.  Frame3DD does mass, but not CG
            ind             = mainEID-1
            m_total         = Ax[:ind] * rho_main[0] * elemL[:ind]
            m_pontoon       = m_total.sum() #mass.struct_mass
            m_substructure += m_pontoon
            cg_pontoon      = np.sum( m_total[:,np.newaxis] * elemCoG[:ind,:], axis=0 ) / m_total.sum()

            # Compute costs based on "Optimum Design of Steel Structures" by Farkas and Jarmai
            # All dimensions for correlations based on mm, not meters.
            k_m     = inputs['material_cost_rate'] #1.1 # USD / kg carbon steel plate
            k_f     = inputs['labor_cost_rate'] #1.0 # USD / min labor
            k_p     = inputs['painting_cost_rate'] #USD / m^2 painting
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
            
            outputs['pontoon_mass'] = m_pontoon
            outputs['pontoon_cost'] = c_pontoon
            outputs['pontoon_cost_rate'] = 1e3*c_pontoon/m_pontoon
            outputs['pontoon_center_of_mass'] = cg_pontoon[-1]
        else:
            V_pontoon = z_cb = m_pontoon = 0.0
            cg_pontoon = np.zeros(3)
            
        # Summary of mass and volumes
        outputs['total_displacement'] = V_main.sum() + ncolumn*V_offset.sum() + V_pontoon
        outputs['substructure_mass']  = m_substructure
        outputs['substructure_center_of_mass'] = (ncolumn*m_offset.sum()*cg_offset + m_main.sum()*cg_main +
                                                   m_pontoon*cg_pontoon) / outputs['substructure_mass']
        m_total = outputs['substructure_mass'] + m_rna + m_tower.sum()
        outputs['structural_mass'] = m_total
        outputs['structure_center_of_mass']  = (m_rna*cg_rna + m_tower.sum()*cg_tower +
                                       outputs['substructure_mass']*outputs['substructure_center_of_mass']) / m_total

        # Find cb (center of buoyancy) for whole system
        z_cb = (V_main.sum()*z_cb_main + ncolumn*V_offset.sum()*z_cb_offset + V_pontoon*z_cb) / outputs['total_displacement']
        outputs['z_center_of_buoyancy'] = z_cb


        # ---REACTIONS---
        # Will first compute unrestained mode shapes and then impose reactions to compute stress loads
        # Find node closest to CG
        cg_dist = np.sum( (nodeMat - outputs['structure_center_of_mass'][np.newaxis,:])**2, axis=1 )
        cg_node = np.argmin(cg_dist)
        rigid = RIGID
        rid = np.array([mainBeginID]) #np.array(fairleadID) #np.array([cg_node+1]) #
        Rx  = rigid * np.ones(rid.shape)
        Ry  = rigid * np.ones(rid.shape)
        Rz  = rigid * np.ones(rid.shape)
        Rxx = rigid * np.ones(rid.shape)
        Ryy = rigid * np.ones(rid.shape)
        Rzz = rigid * np.ones(rid.shape)
        rid = np.append(rid, fairleadID)
        Rx  = np.append(Rx,  K_mooring[0] /nnode_connect * np.ones(nnode_connect) )
        Ry  = np.append(Ry,  K_mooring[1] /nnode_connect * np.ones(nnode_connect) )
        Rz  = np.append(Rz,  K_mooring[2] /nnode_connect * np.ones(nnode_connect) )
        Rxx = np.append(Rxx,  K_mooring[3]/nnode_connect * np.ones(nnode_connect) )
        Ryy = np.append(Ryy,  K_mooring[4]/nnode_connect * np.ones(nnode_connect) )
        Rzz = np.append(Rzz,  K_mooring[5]/nnode_connect * np.ones(nnode_connect) )
        # Get reactions object from frame3dd
        reactions  = pyframe3dd.ReactionData(rid, Rx, Ry, Rz, Rxx, Ryy, Rzz, rigid=rigid)
        R0 = np.zeros(rid.shape)
        reactions0 = pyframe3dd.ReactionData([], R0, R0, R0, R0, R0, R0, rigid=1)

        # Initialize two frame3dd objects
        myframe0 = pyframe3dd.Frame(nodes, reactions0, elements, other)
        self.myframe = pyframe3dd.Frame(nodes, reactions, elements, other)

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
        myframe0.changeExtraNodeMass(inode, m_extra, Ixx, Iyy, Izz, Ixy, Ixz, Iyz, rhox, rhoy, rhoz, True)
        self.myframe.changeExtraNodeMass(inode, m_extra, Ixx, Iyy, Izz, Ixy, Ixz, Iyz, rhox, rhoy, rhoz, True)

        # Add in load cases
        myframe0.addLoadCase(load0)
        self.myframe.addLoadCase(load)
        
        # ---RUN MODAL ANALYSIS---
        if opt['run_modal']:

            # ---DYNAMIC ANALYSIS---
            shift = 1e1
            myframe0.enableDynamics(3*NFREQ, frame3dd_opt['Mmethod'], frame3dd_opt['lump'], float(frame3dd_opt['tol']), shift)

            # ---DEBUGGING---
            #myframe0.write('debug.3dd') # For debugging

            try:
                _, _, _, _, _, modal = myframe0.run()
            except:
                bad_input()
                return

            # Get mode shapes in batch
            mpfs   = np.abs( np.c_[modal.xmpf, modal.ympf, modal.zmpf] )
            polys  = get_modal_coefficients(znode, np.vstack((modal.xdsp, modal.ydsp)).T, 6)
            xpolys = polys[:,:(3*NFREQ)].T
            ypolys = polys[:,(3*NFREQ):].T

            NFREQ2    = int(NFREQ/2)
            mshapes_x = np.zeros((NFREQ2, 5))
            mshapes_y = np.zeros((NFREQ2, 5))
            myfreqs   = np.zeros(NFREQ)
            ix = 0
            iy = 0
            im = 0
            for m in range(len(modal.freq)):
                if mpfs[m,:].max() < 1e-11: continue
                imode = np.argmax(mpfs[m,:])
                if imode == 0 and ix<NFREQ2:
                    mshapes_x[ix,:] = xpolys[m,:]
                    myfreqs[im] = modal.freq[m]
                    ix += 1
                    im += 1
                elif imode == 1 and iy<NFREQ2:
                    mshapes_y[iy,:] = ypolys[m,:]
                    myfreqs[im] = modal.freq[m]
                    iy += 1
                    im += 1
                #else:
                #    print('Warning: Unknown mode shape')
            outputs['x_mode_shapes'] = mshapes_x
            outputs['y_mode_shapes'] = mshapes_y
            outputs['structural_frequencies'] = myfreqs


        # ---RUN STRESS ANALYSIS---
        
        # Initialize frame3dd object again with reactions and rerun with 
        try:
            displacements, forces, reactions, internalForces, mass, _ = self.myframe.run()
        except:
            bad_input()
            return
            
        # --OUTPUTS--
        nE    = nelem.size
        iCase = 0

        # deflections due to loading (from cylinder top and wind/wave loads)
        outputs['top_deflection'] = displacements.dx[iCase, towerEndID-1]  # in yaw-aligned direction

        # Find cg (center of gravity) for whole system
        F_main = -1.0 * np.array([reactions.Fx.sum(), reactions.Fy.sum(), reactions.Fz.sum()])
        M_main = -1.0 * np.array([reactions.Mxx.sum(), reactions.Myy.sum(), reactions.Mzz.sum()])
        r_cg_main = np.array([0.0, 0.0, (znode[mainBeginID] - outputs['structure_center_of_mass'][-1])])
        delta     = np.cross(r_cg_main, F_main)

        outputs['total_force'] = F_main
        outputs['total_moment'] = M_main + delta
        myM= np.cross(np.array([0.0, 0.0, (z_tower[-1] - outputs['structure_center_of_mass'][-1])]), F_rna)
        
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
            qdyn_pontoon = np.max( np.abs( np.r_[inputs['main_qdyn'], inputs['offset_qdyn']] ) )
            sigma_ax_pon = sigma_ax[idx]
            sigma_sh_pon = sigma_sh[idx]
            sigma_h_pon  = util.hoopStress(2*R_od_pontoon, t_wall_pontoon, qdyn_pontoon) * np.ones(sigma_ax_pon.shape)

            outputs['pontoon_stress'][:npon] = util.vonMisesStressUtilization(sigma_ax_pon, sigma_h_pon, sigma_sh_pon,
                                                                               gamma_f*gamma_m*gamma_n, sigma_y_main[0])
        
        # Extract tower for Eurocode checks
        idx = towerEID-1 + np.arange(R_od_tower.size, dtype=np.int32)
        L_reinforced   = opt['tower']['buckling_length'] * np.ones(idx.shape)
        sigma_ax_tower = sigma_ax[idx]
        sigma_sh_tower = sigma_sh[idx]
        qdyn_tower,_   = nodal2sectional( inputs['tower_qdyn'] )
        sigma_h_tower  = util.hoopStress(2*R_od_tower, t_wall_tower*np.cos(angle_tower), qdyn_tower)

        outputs['tower_stress:axial'] = sigma_ax_tower
        outputs['tower_stress:shear'] = sigma_sh_tower
        outputs['tower_stress:hoop']  = sigma_h_tower
        outputs['tower_stress:hoopStiffen'] = util.hoopStressEurocode(z_tower, 2*R_od_tower, t_wall_tower, L_reinforced, qdyn_tower)
        outputs['tower_stress'] = util.vonMisesStressUtilization(sigma_ax_tower, sigma_h_tower, sigma_sh_tower,
                                                                  gamma_f*gamma_m*gamma_n, sigma_y_tower)

        outputs['tower_shell_buckling'] = util.shellBucklingEurocode(2*R_od_tower, t_wall_tower, sigma_ax_tower, sigma_h_tower, sigma_sh_tower,
                                                                      L_reinforced, E_tower, sigma_y_tower, gamma_f, gamma_b)

        tower_height = z_tower[-1] - z_tower[0]
        outputs['tower_global_buckling'] = util.bucklingGL(2*R_od_tower, t_wall_tower, Nx[idx], M[idx], tower_height, E_tower, sigma_y_tower, gamma_f, gamma_b)
        
        # Extract main column for Eurocode checks
        idx = mainEID-1 + np.arange(R_od_main.size, dtype=np.int32)
        L_reinforced  = opt['columns']['main']['buckling_length'] * np.ones(idx.shape)
        sigma_ax_main = sigma_ax[idx]
        sigma_sh_main = sigma_sh[idx]
        qdyn_main,_   = nodal2sectional( inputs['main_qdyn'] )
        sigma_h_main  = util.hoopStress(2*R_od_main, t_wall_main*np.cos(angle_main), qdyn_main)

        outputs['main_stress:axial'] = sigma_ax_main
        outputs['main_stress:shear'] = sigma_sh_main
        outputs['main_stress:hoop']  = sigma_h_main
        outputs['main_stress:hoopStiffen'] = util.hoopStressEurocode(z_main, 2*R_od_main, t_wall_main, L_reinforced, qdyn_main)
        outputs['main_stress'] = util.vonMisesStressUtilization(sigma_ax_main, sigma_h_main, sigma_sh_main,
                                                                        gamma_f*gamma_m*gamma_n, sigma_y_main)

        outputs['main_shell_buckling'] = util.shellBucklingEurocode(2*R_od_main, t_wall_main, sigma_ax_main, sigma_h_main, sigma_sh_main,
                                                                            L_reinforced, E_main, sigma_y_main, gamma_f, gamma_b)

        main_height = z_main[-1] - z_main[0]
        outputs['main_global_buckling'] = util.bucklingGL(2*R_od_main, t_wall_main, Nx[idx], M[idx], main_height, E_main, sigma_y_main, gamma_f, gamma_b)

        
        # Extract offset column for Eurocode checks
        if ncolumn > 0:
            idx = offsetEID[0]-1 + np.arange(R_od_offset.size, dtype=np.int32)
            L_reinforced    = opt['columns']['offset']['buckling_length'] * np.ones(idx.shape)
            sigma_ax_offset = sigma_ax[idx]
            sigma_sh_offset = sigma_sh[idx]
            qdyn_offset,_   = nodal2sectional( inputs['offset_qdyn'] )
            sigma_h_offset  = util.hoopStress(2*R_od_offset, t_wall_offset*np.cos(angle_offset), qdyn_offset)

            outputs['offset_stress:axial'] = sigma_ax_offset
            outputs['offset_stress:shear'] = sigma_sh_offset
            outputs['offset_stress:hoop']  = sigma_h_offset
            outputs['offset_stress:hoopStiffen'] = util.hoopStressEurocode(z_offset, 2*R_od_offset, t_wall_offset, L_reinforced, qdyn_offset)
            outputs['offset_stress'] = util.vonMisesStressUtilization(sigma_ax_offset, sigma_h_offset, sigma_sh_offset,
                                                                            gamma_f*gamma_m*gamma_n, sigma_y_offset)

            outputs['offset_shell_buckling'] = util.shellBucklingEurocode(2*R_od_offset, t_wall_offset, sigma_ax_offset, sigma_h_offset, sigma_sh_offset,
                                                                                L_reinforced, E_offset, sigma_y_offset, gamma_f, gamma_b)

            offset_height = z_offset[-1] - z_offset[0]
            outputs['offset_global_buckling'] = util.bucklingGL(2*R_od_offset, t_wall_offset, Nx[idx], M[idx], offset_height, E_offset, sigma_y_offset, gamma_f, gamma_b)
        



class TrussIntegerToBoolean(om.ExplicitComponent):
    """
    Get booleans from truss integers
    
    Parameters
    ----------
    cross_attachment_pontoons_int : float
        Inclusion of pontoons that connect the bottom of the central main to the tops of
        the outer offset columns
    lower_attachment_pontoons_int : float
        Inclusion of pontoons that connect the central main to the outer offset columns
        at their bottoms
    upper_attachment_pontoons_int : float
        Inclusion of pontoons that connect the central main to the outer offset columns
        at their tops
    lower_ring_pontoons_int : float
        Inclusion of pontoons that ring around outer offset columns at their bottoms
    upper_ring_pontoons_int : float
        Inclusion of pontoons that ring around outer offset columns at their tops
    outer_cross_pontoons_int : float
        Inclusion of pontoons that ring around outer offset columns at their tops
    
    Returns
    -------
    cross_attachment_pontoons : boolean
        Inclusion of pontoons that connect the bottom of the central main to the tops of
        the outer offset columns
    lower_attachment_pontoons : boolean
        Inclusion of pontoons that connect the central main to the outer offset columns
        at their bottoms
    upper_attachment_pontoons : boolean
        Inclusion of pontoons that connect the central main to the outer offset columns
        at their tops
    lower_ring_pontoons : boolean
        Inclusion of pontoons that ring around outer offset columns at their bottoms
    upper_ring_pontoons : boolean
        Inclusion of pontoons that ring around outer offset columns at their tops
    outer_cross_pontoons : boolean
        Inclusion of pontoons that ring around outer offset columns at their tops
    
    """
    def setup(self):
        self.add_input('cross_attachment_pontoons_int', val=1)
        self.add_input('lower_attachment_pontoons_int', val=1)
        self.add_input('upper_attachment_pontoons_int', val=1)
        self.add_input('lower_ring_pontoons_int', val=1)
        self.add_input('upper_ring_pontoons_int', val=1)
        self.add_input('outer_cross_pontoons_int', val=1)

        self.add_discrete_output('cross_attachment_pontoons', val=True)
        self.add_discrete_output('lower_attachment_pontoons', val=True)
        self.add_discrete_output('upper_attachment_pontoons', val=True)
        self.add_discrete_output('lower_ring_pontoons', val=True)
        self.add_discrete_output('upper_ring_pontoons', val=True)
        self.add_discrete_output('outer_cross_pontoons', val=True)


    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        discrete_outputs['cross_attachment_pontoons'] = (int(inputs['cross_attachment_pontoons_int']) == 1)
        discrete_outputs['lower_attachment_pontoons'] = (int(inputs['lower_attachment_pontoons_int']) == 1)
        discrete_outputs['upper_attachment_pontoons'] = (int(inputs['upper_attachment_pontoons_int']) == 1)
        discrete_outputs['lower_ring_pontoons']       = (int(inputs['lower_ring_pontoons_int']) == 1)
        discrete_outputs['upper_ring_pontoons']       = (int(inputs['upper_ring_pontoons_int']) == 1)
        discrete_outputs['outer_cross_pontoons']      = (int(inputs['outer_cross_pontoons_int']) == 1)

        
# -----------------
#  Assembly
# -----------------

class Loading(om.Group):

    def initialize(self):
        self.options.declare('n_height_main')
        self.options.declare('n_height_off')
        self.options.declare('n_height_tow')
        self.options.declare('modeling_options')
        
    def setup(self):
        n_height_main = self.options['n_height_main']
        n_height_off  = self.options['n_height_off']
        n_height_tow  = self.options['n_height_tow']
        n_full_main   = get_nfull(n_height_main)
        n_full_off    = get_nfull(n_height_off)
        n_full_tow    = get_nfull(n_height_tow)
        
        self.set_input_defaults('outer_cross_pontoons_int', 1)
        self.set_input_defaults('cross_attachment_pontoons_int', 1)
        self.set_input_defaults('lower_attachment_pontoons_int', 1)
        self.set_input_defaults('upper_attachment_pontoons_int', 1)
        self.set_input_defaults('lower_ring_pontoons_int', 1)
        self.set_input_defaults('upper_ring_pontoons_int', 1)
        
        # All the components
        self.add_subsystem('loadingWind', PowerWind(nPoints=n_full_tow), promotes=['z0','Uref','shearExp','zref'])
        self.add_subsystem('windLoads', CylinderWindDrag(nPoints=n_full_tow), promotes=['cd_usr','beta_wind','rho_air','mu_air'])
        self.add_subsystem('intbool', TrussIntegerToBoolean(), promotes=['*'])
        self.add_subsystem('frame', FloatingFrame(n_height_main=n_height_main,
                                                  n_height_off=n_height_off,
                                                  n_height_tow=n_height_tow,
                                                  modeling_options=self.options['modeling_options']), promotes=['*'])
        
        self.connect('loadingWind.U', 'windLoads.U')
        self.connect('windLoads.windLoads_Px', 'tower_Px')
        self.connect('windLoads.windLoads_Py', 'tower_Py')
        self.connect('windLoads.windLoads_Pz', 'tower_Pz')
        self.connect('windLoads.windLoads_qdyn', 'tower_qdyn')

