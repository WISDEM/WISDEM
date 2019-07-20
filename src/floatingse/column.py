from openmdao.api import Component, Group
import numpy as np

from commonse.utilities import nodal2sectional, sectional2nodal, assembleI, unassembleI, sectionalInterp
import commonse.frustum as frustum
import commonse.manufacturing as manufacture
from commonse.UtilizationSupplement import shellBuckling_withStiffeners, GeometricConstraints
from commonse import gravity, eps, AeroHydroLoads, CylinderWindDrag, CylinderWaveDrag
from commonse.vertical_cylinder import CylinderDiscretization, CylinderMass
from commonse.environment import PowerWind, LinearWaves

def get_inner_radius(Ro, t):
    # Radius varies at nodes, t varies by section
    return (Ro-sectional2nodal(t))

def I_tube(r_i, r_o, h, m):
    if type(r_i) == type(np.array([])):
        n = r_i.size
        r_i = r_i.flatten()
        r_o = r_o.flatten()
        h   = h.flatten()
        m   = m.flatten()
    else:
        n = 1
    Ixx = Iyy = (m/12.0) * (3.0*(r_i**2.0 + r_o**2.0) + h**2.0)
    Izz = 0.5 * m * (r_i**2.0 + r_o**2.0)
    return np.c_[Ixx, Iyy, Izz, np.zeros((n,3))]
    

class BulkheadProperties(Component):

    def __init__(self, nSection, nFull):
        super(BulkheadProperties,self).__init__()
        self.bulk_full = np.zeros( nFull, dtype=np.int_)

        self.add_param('z_full', val=np.zeros(nFull), units='m', desc='z-coordinates of section nodes (length = nsection+1)')
        self.add_param('z_param', val=np.zeros((nSection+1,)), units='m', desc='z-coordinates of section nodes (length = nsection+1)')
        self.add_param('d_full', val=np.zeros(nFull), units='m', desc='cylinder diameter at corresponding locations')
        self.add_param('t_full', val=np.zeros(nFull-1), units='m', desc='shell thickness at corresponding locations')
        self.add_param('rho', val=0.0, units='kg/m**3', desc='material density')
        self.add_param('bulkhead_thickness', val=np.zeros(nSection+1), units='m', desc='Nodal locations of bulkhead thickness, zero meaning no bulkhead, bottom to top (length = nsection + 1)')
        self.add_param('bulkhead_mass_factor', val=0.0, desc='Bulkhead mass correction factor')
        
        self.add_param('shell_mass', val=np.zeros(nFull-1), units='kg', desc='mass of column shell')
        self.add_param('material_cost_rate', 0.0, units='USD/kg', desc='Raw material cost rate: steel $1.1/kg, aluminum $3.5/kg')
        self.add_param('labor_cost_rate', 0.0, units='USD/min', desc='Labor cost rate')
        self.add_param('painting_cost_rate', 0.0, units='USD/m/m', desc='Painting / surface finishing cost rate')

        self.add_output('bulkhead_mass', val=np.zeros(nFull), units='kg', desc='mass of column bulkheads')
        self.add_output('bulkhead_cost', val=0.0, units='USD', desc='cost of column bulkheads')
        self.add_output('bulkhead_I_keel', val=np.zeros(6), units='kg*m**2', desc='Moments of inertia of bulkheads relative to keel point')
        
        # Derivatives
        self.deriv_options['type'] = 'fd'
        self.deriv_options['form'] = 'central'
        self.deriv_options['check_form'] = 'central'
        self.deriv_options['step_calc'] = 'relative'
        
    def solve_nonlinear(self, params, unknowns, resids):
        # Unpack variables
        z_full     = params['z_full'] # at section nodes
        z_param    = params['z_param']
        R_od       = 0.5*params['d_full'] # at section nodes
        twall      = params['t_full'] # at section nodes
        R_id       = get_inner_radius(R_od, twall)
        t_bulk     = params['bulkhead_thickness'] # at section nodes
        rho        = params['rho']
        
        # Map bulkhead locations to finer computation grid
        Zf,Zp = np.meshgrid(z_full, z_param)
        idx = np.argmin( np.abs(Zf-Zp), axis=1 )
        t_bulk_full = np.zeros( z_full.shape )
        t_bulk_full[idx] = t_bulk
        # Make sure top and bottom are capped
        if (t_bulk_full[ 0] == 0.0): t_bulk_full[ 0] = twall[ 0]
        if (t_bulk_full[-1] == 0.0): t_bulk_full[-1] = twall[-1]
        
        # Compute bulkhead volume at every section node
        # Assume bulkheads are same thickness as shell wall
        V_bulk = np.pi * R_id**2 * t_bulk_full

        # Convert to mass with fudge factor for design features not captured in this simple approach
        m_bulk = params['bulkhead_mass_factor'] * rho * V_bulk

        # Compute moments of inertia at keel
        # Assume bulkheads are just simple thin discs with radius R_od-t_wall and mass already computed
        Izz = 0.5 * m_bulk * R_id**2
        Ixx = Iyy = 0.5 * Izz
        I_keel = np.zeros((3,3))
        dz  = z_full - z_full[0]
        for k in range(m_bulk.size):
            R = np.array([0.0, 0.0, dz[k]])
            Icg = assembleI( [Ixx[k], Iyy[k], Izz[k], 0.0, 0.0, 0.0] )
            I_keel += Icg + m_bulk[k]*(np.dot(R, R)*np.eye(3) - np.outer(R, R))


        # Compute costs based on "Optimum Design of Steel Structures" by Farkas and Jarmai
        # All dimensions for correlations based on mm, not meters.
        k_m     = params['material_cost_rate'] #1.1 # USD / kg carbon steel plate
        k_f     = params['labor_cost_rate'] #1.0 # USD / min labor
        k_p     = params['painting_cost_rate'] #USD / m^2 painting
        m_shell = params['shell_mass'].sum()
        nbulk   = np.count_nonzero(V_bulk)
        bulkind = (V_bulk > 0.0)
        
        # Cost Step 1) Cutting flat plates using plasma cutter
        cutLengths = 2.0 * np.pi * R_id * bulkind
        # Cost Step 2) Fillet welds with GMAW-C (gas metal arc welding with CO2) of bulkheads to shell
        theta_w = 3.0 # Difficulty factor

        # Labor-based expenses
        K_f = k_f * ( manufacture.steel_cutting_plasma_time(cutLengths, t_bulk_full) +
                      manufacture.steel_filett_welding_time(theta_w, nbulk, m_bulk+m_shell, 2*np.pi*R_id, t_bulk_full) )
        
        # Cost Step 3) Painting (two sided)
        theta_p = 1.0
        K_p  = k_p * theta_p * 2 * (np.pi * R_id**2.0 * bulkind).sum()

        # Material cost, without outfitting
        K_m = k_m * m_bulk.sum()

        # Total cost
        c_bulk = K_m + K_f + K_p
        
        
        # Store results
        unknowns['bulkhead_I_keel'] = unassembleI(I_keel)
        unknowns['bulkhead_mass'] = m_bulk
        unknowns['bulkhead_cost'] = c_bulk

    

class BuoyancyTankProperties(Component):
    def __init__(self, nFull):
        super(BuoyancyTankProperties,self).__init__()
        
        self.add_param('d_full', val=np.zeros(nFull), units='m', desc='cylinder diameter at corresponding locations')
        self.add_param('z_full', val=np.zeros(nFull), units='m', desc='z-coordinates of section nodes')
        self.add_param('rho', val=0.0, units='kg/m**3', desc='material density')

        self.add_param('shell_mass', val=np.zeros(nFull-1), units='kg', desc='mass of column shell')
        self.add_param('material_cost_rate', 0.0, units='USD/kg', desc='Raw material cost: steel $1.1/kg, aluminum $3.5/kg')
        self.add_param('labor_cost_rate', 0.0, units='USD/min', desc='Labor cost')
        self.add_param('painting_cost_rate', 0.0, units='USD/m/m', desc='Painting / surface finishing cost rate')
        
        self.add_param('buoyancy_tank_diameter', val=0.0, units='m', desc='Radius of heave plate at bottom of column')
        self.add_param('buoyancy_tank_height', val=0.0, units='m', desc='Radius of heave plate at bottom of column')
        self.add_param('buoyancy_tank_location', val=0.0, units='m', desc='Radius of heave plate at bottom of column')
        self.add_param('buoyancy_tank_mass_factor', val=0.0, desc='Heave plate mass correction factor')

        self.add_output('buoyancy_tank_mass', val=0.0, units='kg', desc='mass of buoyancy tank')
        self.add_output('buoyancy_tank_cost', val=0.0, units='USD', desc='cost of buoyancy tank')
        self.add_output('buoyancy_tank_cg', val=0.0, units='m', desc='z-coordinate of center of mass for buoyancy tank')
        self.add_output('buoyancy_tank_displacement', val=0.0, units='m**3', desc='volume of water displaced by buoyancy tank')
        self.add_output('buoyancy_tank_I_keel', val=np.zeros(6), units='kg*m**2', desc='Moments of inertia of heave plate relative to keel point')

        # Derivatives
        self.deriv_options['type'] = 'fd'
        self.deriv_options['form'] = 'central'
        self.deriv_options['check_form'] = 'central'
        self.deriv_options['step_calc'] = 'relative'

        
    def solve_nonlinear(self, params, unknowns, resids):
        # Unpack variables
        z_full    = params['z_full']

        R_od      = 0.5*params['d_full']
        R_plate   = 0.5*params['buoyancy_tank_diameter']
        h_box     = params['buoyancy_tank_height']

        location  = params['buoyancy_tank_location']
        
        coeff     = params['buoyancy_tank_mass_factor']
        rho       = params['rho']

        # Current hard-coded, coarse specification of shell thickness
        t_plate   = R_plate / 50.0

        # Z-locations of buoyancy tank
        z_lower   = location * (z_full[-1] - z_full[0]) + z_full[0]
        z_cg      = z_lower + 0.5*h_box
        z_upper   = z_lower +     h_box

        # Mass and volume properties that subtract out central column contributions for no double-counting
        R_col     = np.interp([z_lower, z_upper], z_full, R_od)
        if not np.any(R_plate > R_col): R_plate = 0.0
        A_plate   = np.maximum(0.0, np.pi * (R_plate**2.0 - R_col**2.0))
        m_plate   = coeff * rho * t_plate * A_plate
        A_box     = A_plate.sum() + 2.0 * np.pi * R_plate * h_box
        m_box     = coeff * rho * t_plate * A_box

        # Compute displcement for buoyancy calculations, but check for what is submerged
        V_box      = np.pi * R_plate**2.0 * h_box
        V_box     -= frustum.frustumVol(R_col[0], R_col[1], h_box)
        if z_lower >= 0.0:
            V_box  = 0.0
        elif z_upper >= 0.0:
            V_box *= (- z_lower / h_box)
        V_box      = np.maximum(0.0, V_box)

        # Now do moments of inertia
        # First find MoI at cg of all components
        R_plate += eps
        Ixx_box   = frustum.frustumShellIxx(R_plate, R_plate, t_plate, h_box)
        Izz_box   = frustum.frustumShellIzz(R_plate, R_plate, t_plate, h_box)
        I_plateL  = 0.25 * m_plate[0] * (R_plate**2.0 - R_col[0]**2.0) * np.array([1.0, 1.0, 2.0, 0.0, 0.0, 0.0])
        I_plateU  = 0.25 * m_plate[1] * (R_plate**2.0 - R_col[1]**2.0) * np.array([1.0, 1.0, 2.0, 0.0, 0.0, 0.0])

        # Move to keel for consistency
        I_keel    = np.zeros((3,3))
        if R_plate > eps:
            # Add in lower plate
            r         = np.array([0.0, 0.0, z_lower])
            Icg       = assembleI( I_plateL )
            I_keel   += Icg + m_plate[0]*(np.dot(r, r)*np.eye(3) - np.outer(r, r))
            # Add in upper plate
            r         = np.array([0.0, 0.0, z_upper])
            Icg       = assembleI( I_plateU )
            I_keel   += Icg + m_plate[1]*(np.dot(r, r)*np.eye(3) - np.outer(r, r))
            # Add in box cylinder
            r         = np.array([0.0, 0.0, z_cg])
            Icg       = assembleI( [Ixx_box, Ixx_box, Izz_box, 0.0, 0.0, 0.0] )
            I_keel   += Icg + m_plate[1]*(np.dot(r, r)*np.eye(3) - np.outer(r, r))


        # Compute costs based on "Optimum Design of Steel Structures" by Farkas and Jarmai
        # All dimensions for correlations based on mm, not meters.
        k_m     = params['material_cost_rate'] #1.1 # USD / kg carbon steel plate
        k_f     = params['labor_cost_rate'] #1.0 # USD / min labor
        k_p     = params['painting_cost_rate'] #USD / m^2 painting
        m_shell = params['shell_mass'].sum()

        # Cost Step 1) Cutting flat plates using plasma cutter into box plate sizes
        cutLengths = 2.0 * np.pi * (3.0*R_plate + R_col.sum()) # x3 for two plates + side wall
        # Cost Step 2) Welding box plates together GMAW-C (gas metal arc welding with CO2) fillet welds
        theta_w = 3.0 # Difficulty factor
        # Cost Step 3) Welding box to shell GMAW-C (gas metal arc welding with CO2) fillet welds

        # Labor-based expenses
        K_f = k_f * ( manufacture.steel_cutting_plasma_time(cutLengths, t_plate) +
                      manufacture.steel_filett_welding_time(theta_w, 3.0, m_box, 2*np.pi*R_plate, t_plate) +
                      manufacture.steel_filett_welding_time(theta_w, 2.0, m_box+m_shell, 2*np.pi*R_col, t_plate) )
        
        # Cost Step 4) Painting
        theta_p = 1.5
        K_p  = k_p * theta_p * 2.0 * A_box
        
        # Material cost, without outfitting
        K_m = k_m * m_box

        # Total cost
        c_box = K_m + K_f + K_p
            
        # Store outputs
        unknowns['buoyancy_tank_cost']         = c_box
        unknowns['buoyancy_tank_mass']         = m_box
        unknowns['buoyancy_tank_cg']           = z_cg
        unknowns['buoyancy_tank_displacement'] = V_box
        unknowns['buoyancy_tank_I_keel']       = unassembleI(I_keel)
        
        
class StiffenerProperties(Component):
    """Computes column stiffener properties by section.  
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
    """
    def __init__(self, nSection, nFull):
        super(StiffenerProperties,self).__init__()

        self.nSection = nSection
        self.add_param('d_full', val=np.zeros(nFull), units='m', desc='cylinder diameter at corresponding locations')
        self.add_param('t_full', val=np.zeros(nFull-1), units='m', desc='shell thickness at corresponding locations')
        self.add_param('z_full', val=np.zeros(nFull), units='m', desc='z-coordinates of section nodes')
        self.add_param('rho', val=0.0, units='kg/m**3', desc='material density')
        
        self.add_param('shell_mass', val=np.zeros(nFull-1), units='kg', desc='mass of column shell')
        self.add_param('material_cost_rate', 0.0, units='USD/kg', desc='Raw material cost: steel $1.1/kg, aluminum $3.5/kg')
        self.add_param('labor_cost_rate', 0.0, units='USD/min', desc='Labor cost')
        self.add_param('painting_cost_rate', 0.0, units='USD/m/m', desc='Painting / surface finishing cost rate')

        self.add_param('h_web', val=np.zeros((nFull-1,)), units='m', desc='height of stiffener web (base of T) within each section bottom to top')
        self.add_param('t_web', val=np.zeros((nFull-1,)), units='m', desc='thickness of stiffener web (base of T) within each section bottom to top')
        self.add_param('w_flange', val=np.zeros((nFull-1,)), units='m', desc='height of stiffener flange (top of T) within each section bottom to top')
        self.add_param('t_flange', val=np.zeros((nFull-1,)), units='m', desc='thickness of stiffener flange (top of T) within each section bottom to top')
        self.add_param('L_stiffener', val=np.zeros((nFull-1,)), units='m', desc='Axial distance from one ring stiffener to another within each section bottom to top')

        self.add_param('ring_mass_factor', val=0.0, desc='Stiffener ring mass correction factor')
        
        self.add_output('stiffener_mass', val=np.zeros(nFull-1), units='kg', desc='mass of column stiffeners')
        self.add_output('stiffener_cost', val=0.0, units='USD', desc='cost of column stiffeners')
        self.add_output('stiffener_I_keel', val=np.zeros(6), units='kg*m**2', desc='Moments of inertia of stiffeners relative to keel point')
        self.add_output('number_of_stiffeners', val=np.zeros(nSection, dtype=np.int_), desc='number of stiffeners in each section')
        self.add_output('flange_spacing_ratio', val=np.zeros((nFull-1,)), desc='ratio between flange and stiffener spacing')
        self.add_output('stiffener_radius_ratio', val=np.zeros((nFull-1,)), desc='ratio between stiffener height and radius')

        # Derivatives
        self.deriv_options['type'] = 'fd'
        self.deriv_options['form'] = 'central'
        self.deriv_options['check_form'] = 'central'
        self.deriv_options['step_calc'] = 'relative'
        
    def solve_nonlinear(self, params, unknowns, resids):
        # Unpack variables
        R_od         = 0.5*params['d_full']
        t_wall       = params['t_full']
        z_full       = params['z_full'] # at section nodes
        h_section    = np.diff(z_full)
        V_shell      = frustum.frustumShellVol(R_od[:-1], R_od[1:], t_wall, h_section)
        R_od,_       = nodal2sectional( R_od ) # at section nodes
        
        t_web        = params['t_web']
        t_flange     = params['t_flange']
        h_web        = params['h_web']
        w_flange     = params['w_flange']
        L_stiffener  = params['L_stiffener']

        rho          = params['rho']
        
        # Outer and inner radius of web by section
        R_wo = R_od - t_wall
        R_wi = R_wo - h_web
        # Outer and inner radius of flange by section
        R_fo = R_wi
        R_fi = R_fo - t_flange

        # Material volumes by section
        V_web    = np.pi*(R_wo**2 - R_wi**2) * t_web
        V_flange = np.pi*(R_fo**2 - R_fi**2) * w_flange

        # Ring mass by volume by section 
        # Include fudge factor for design features not captured in this simple approach
        m_web    = params['ring_mass_factor'] * rho * V_web
        m_flange = params['ring_mass_factor'] * rho * V_flange
        m_ring   = m_web + m_flange
        n_stiff  = np.zeros(h_web.shape, dtype=np.int_)
        
        # Compute moments of inertia for stiffeners (lumped by section for simplicity) at keel
        I_web     = I_tube(R_wi, R_wo, t_web   , m_web)
        I_flange  = I_tube(R_fi, R_fo, w_flange, m_flange)
        I_ring    = I_web + I_flange
        I_keel    = np.zeros((3,3))

        # Now march up the column, adding stiffeners at correct spacing until we are done
        z_stiff  = []
        isection = 0
        epsilon  = 1e-6
        while True:
            if len(z_stiff) == 0:
                z_march = np.minimum(z_full[isection+1], z_full[0] + 0.5*L_stiffener[isection]) + epsilon
            else:
                z_march = np.minimum(z_full[isection+1], z_stiff[-1] + L_stiffener[isection]) + epsilon
            if z_march >= z_full[-1]: break
            
            isection = np.searchsorted(z_full, z_march) - 1
            
            if len(z_stiff) == 0:
                add_stiff = (z_march - z_full[0]) >= 0.5*L_stiffener[isection]
            else:
                add_stiff = (z_march - z_stiff[-1]) >= L_stiffener[isection]
                
            if add_stiff:
                z_stiff.append(z_march)
                n_stiff[isection] += 1
                
                R       = np.array([0.0, 0.0, (z_march - z_full[0])])
                Icg     = assembleI( I_ring[isection,:] )
                I_keel += Icg + m_ring[isection]*(np.dot(R, R)*np.eye(3) - np.outer(R, R))

        # Number of stiffener rings per section (height of section divided by spacing)
        unknowns['stiffener_mass'] =  n_stiff * m_ring

        # Find total number of stiffeners in each original section
        npts_per    = int(h_web.size / self.nSection)
        n_stiff_sec = np.zeros(self.nSection)
        for k in range(npts_per):
            n_stiff_sec += n_stiff[k::npts_per]
        unknowns['number_of_stiffeners'] = n_stiff_sec


        # Compute costs based on "Optimum Design of Steel Structures" by Farkas and Jarmai
        # All dimensions for correlations based on mm, not meters.
        k_m     = params['material_cost_rate'] #1.1 # USD / kg carbon steel plate
        k_f     = params['labor_cost_rate'] #1.0 # USD / min labor
        k_p     = params['painting_cost_rate'] #USD / m^2 painting
        m_shell = params['shell_mass'].sum()
        
        # Cost Step 1) Cutting stiffener strips from flat plates using plasma cutter
        cutLengths_w = 2.0 * np.pi * 0.5 * (R_wo + R_wi)
        cutLengths_f = 2.0 * np.pi * R_fo
        # Cost Step 2) Welding T-stiffeners together GMAW-C (gas metal arc welding with CO2) fillet welds
        theta_w = 3.0 # Difficulty factor
        # Cost Step 3) Welding stiffeners to shell GMAW-C (gas metal arc welding with CO2) fillet welds
        # Will likely fillet weld twice (top & bottom), so factor of 2 on second welding terms

        # Labor-based expenses
        K_f = k_f * ( manufacture.steel_cutting_plasma_time(n_stiff * cutLengths_w, t_web) +
                      manufacture.steel_cutting_plasma_time(n_stiff * cutLengths_f, t_flange) +
                      manufacture.steel_filett_welding_time(theta_w, n_stiff, m_ring, 2*np.pi*R_fo, t_web) +
                      manufacture.steel_filett_welding_time(theta_w, n_stiff, m_ring+m_shell, 2*np.pi*R_wo, t_web) )
        
        # Cost Step 4) Painting
        theta_p = 2.0
        K_p  = k_p * theta_p * (n_stiff*(2*np.pi*(R_wo**2.0-R_wi**2.0) + 2*np.pi*0.5*(R_fo+R_fi)*(2*w_flange + 2*t_flange) - 2*np.pi*R_fo*t_web)).sum()
        
        # Material cost, without outfitting
        K_m = k_m * unknowns['stiffener_mass'].sum()

        # Total cost
        c_ring = K_m + K_f + K_p
        
        # Store results
        unknowns['stiffener_cost'] = c_ring
        unknowns['stiffener_I_keel'] = unassembleI(I_keel)
        
        # Create some constraints for reasonable stiffener designs for an optimizer
        unknowns['flange_spacing_ratio']   = w_flange / (0.5*L_stiffener)
        unknowns['stiffener_radius_ratio'] = (h_web + t_flange + t_wall) / R_od



        
class BallastProperties(Component):

    def __init__(self, nFull):
        super(BallastProperties,self).__init__()

        self.add_param('water_density', val=0.0, units='kg/m**3', desc='density of water')
        self.add_param('d_full', val=np.zeros(nFull), units='m', desc='cylinder diameter at corresponding locations')
        self.add_param('t_full', val=np.zeros(nFull-1), units='m', desc='shell thickness at corresponding locations')
        self.add_param('z_full', val=np.zeros(nFull), units='m', desc='z-coordinates of section nodes')
        self.add_param('permanent_ballast_density', val=0.0, units='kg/m**3', desc='density of permanent ballast')
        self.add_param('permanent_ballast_height', val=0.0, units='m', desc='height of permanent ballast')
        self.add_param('ballast_cost_rate', val=0.0, units='USD/kg', desc='Cost per unit mass of ballast')

        self.add_output('ballast_cost', val=0.0, units='USD', desc='cost of permanent ballast')
        self.add_output('ballast_mass', val=np.zeros(nFull-1), units='kg', desc='mass of permanent ballast')
        self.add_output('ballast_z_cg', val=0.0, units='m', desc='z-coordinate or permanent ballast center of gravity')
        self.add_output('ballast_I_keel', val=np.zeros(6), units='kg*m**2', desc='Moments of inertia of permanent ballast relative to keel point')
        self.add_output('variable_ballast_interp_zpts', val=np.zeros((nFull,)), units='m', desc='z-points of potential ballast mass')
        self.add_output('variable_ballast_interp_radius', val=np.zeros((nFull,)), units='m', desc='inner radius of column at potential ballast mass')

        
    def solve_nonlinear(self, params, unknowns, resids):
        # Unpack variables
        R_od        = 0.5*params['d_full']
        t_wall      = params['t_full']
        z_nodes     = params['z_full']
        h_ballast   = params['permanent_ballast_height']
        rho_ballast = params['permanent_ballast_density']
        rho_water   = params['water_density']
        R_id_orig   = get_inner_radius(R_od, t_wall)

        npts = R_od.size
        section_mass = np.zeros(npts-1)
        
        # Geometry of the column in our coordinate system (z=0 at waterline)
        z_draft   = z_nodes[0]

        # Fixed and total ballast mass and cg
        # Assume they are bottled in columns a the keel of the column- first the permanent then the fixed
        zpts      = np.linspace(z_draft, z_draft+h_ballast, npts)
        R_id      = np.interp(zpts, z_nodes, R_id_orig)
        V_perm    = np.pi * np.trapz(R_id**2, zpts)
        m_perm    = rho_ballast * V_perm
        z_cg_perm = rho_ballast * np.pi * np.trapz(zpts*R_id**2, zpts) / m_perm if m_perm > 0.0 else 0.0
        for k in range(npts-1):
            ind = np.logical_and(zpts>=z_nodes[k], zpts<=z_nodes[k+1]) 
            section_mass[k] += rho_ballast * np.pi * np.trapz(R_id[ind]**2, zpts[ind])

        Ixx = Iyy = frustum.frustumIxx(R_id[:-1], R_id[1:], np.diff(zpts))
        Izz = frustum.frustumIzz(R_id[:-1], R_id[1:], np.diff(zpts))
        V_slice = frustum.frustumVol(R_id[:-1], R_id[1:], np.diff(zpts))
        I_keel = np.zeros((3,3))
        dz  = frustum.frustumCG(R_id[:-1], R_id[1:], np.diff(zpts)) + zpts[:-1] - z_draft
        for k in range(V_slice.size):
            R = np.array([0.0, 0.0, dz[k]])
            Icg = assembleI( [Ixx[k], Iyy[k], Izz[k], 0.0, 0.0, 0.0] )
            I_keel += Icg + V_slice[k]*(np.dot(R, R)*np.eye(3) - np.outer(R, R))
        I_keel = rho_ballast * unassembleI(I_keel)
        
        # Water ballast will start at top of fixed ballast
        z_water_start = (z_draft + h_ballast)
        #z_water_start = z_water_start + params['variable_ballast_start'] * (z_nodes[-1] - z_water_start)
        
        # Find height of water ballast numerically by finding the height that integrates to the mass we want
        # This step is completed in column.py or semi.py because we must account for other substructure elements too
        zpts    = np.linspace(z_water_start, 0.0, npts)
        R_id    = np.interp(zpts, z_nodes, R_id_orig)
        unknowns['variable_ballast_interp_zpts']   = zpts
        unknowns['variable_ballast_interp_radius'] = R_id
        
        # Save permanent ballast mass and variable height
        unknowns['ballast_mass']   = section_mass
        unknowns['ballast_I_keel'] = I_keel
        unknowns['ballast_z_cg']   = z_cg_perm
        unknowns['ballast_cost']   = params['ballast_cost_rate'] * m_perm


        
        
class ColumnGeometry(Component):
    """
    OpenMDAO Component class for vertical columns in substructure for floating offshore wind turbines.
    """

    def __init__(self, nSection, nFull):
        super(ColumnGeometry,self).__init__()

        # Design variables
        self.add_param('water_depth', val=0.0, units='m', desc='water depth')
        self.add_param('Hs', val=0.0, units='m', desc='significant wave height')
        self.add_param('freeboard', val=0.0, units='m', desc='Length of column above water line')
        self.add_param('max_draft', val=0.0, units='m', desc='Maxmimum length of column below water line')
        self.add_param('z_full_in', val=np.zeros((nFull,)), units='m', desc='z-coordinates of section nodes (length = nsection+1)')
        self.add_param('z_param_in', val=np.zeros((nSection+1,)), units='m', desc='z-coordinates of section nodes (length = nsection+1)')
        self.add_param('section_center_of_mass', val=np.zeros(nFull-1), units='m', desc='z position of center of mass of each can in the cylinder')

        self.add_param('stiffener_web_height', val=np.zeros((nSection,)), units='m', desc='height of stiffener web (base of T) within each section bottom to top (length = nsection)')
        self.add_param('stiffener_web_thickness', val=np.zeros((nSection,)), units='m', desc='thickness of stiffener web (base of T) within each section bottom to top (length = nsection)')
        self.add_param('stiffener_flange_width', val=np.zeros((nSection,)), units='m', desc='height of stiffener flange (top of T) within each section bottom to top (length = nsection)')
        self.add_param('stiffener_flange_thickness', val=np.zeros((nSection,)), units='m', desc='thickness of stiffener flange (top of T) within each section bottom to top (length = nsection)')
        self.add_param('stiffener_spacing', val=np.zeros((nSection,)), units='m', desc='Axial distance from one ring stiffener to another within each section bottom to top (length = nsection)')

        # Outputs
        self.add_output('z_full', val=np.zeros((nFull,)), units='m', desc='z-coordinates of section nodes (length = nsection+1)')
        self.add_output('z_param', val=np.zeros((nSection+1,)), units='m', desc='z-coordinates of section nodes (length = nsection+1)')
        self.add_output('draft', val=0.0, units='m', desc='Column draft (length of body under water)')
        self.add_output('z_section', val=np.zeros((nFull-1,)), units='m', desc='z-coordinates of section centers of mass (length = nsection)')


        self.add_output('h_web', val=np.zeros((nFull-1,)), units='m', desc='height of stiffener web (base of T) within each section bottom to top')
        self.add_output('t_web', val=np.zeros((nFull-1,)), units='m', desc='thickness of stiffener web (base of T) within each section bottom to top')
        self.add_output('w_flange', val=np.zeros((nFull-1,)), units='m', desc='height of stiffener flange (top of T) within each section bottom to top')
        self.add_output('t_flange', val=np.zeros((nFull-1,)), units='m', desc='thickness of stiffener flange (top of T) within each section bottom to top')
        self.add_output('L_stiffener', val=np.zeros((nFull-1,)), units='m', desc='Axial distance from one ring stiffener to another within each section bottom to top')
        
        # Output constraints
        self.add_output('draft_margin', val=0.0, desc='Ratio of draft to water depth')
        self.add_output('wave_height_freeboard_ratio', val=0.0, desc='Ratio of maximum wave height (avg of top 1%) to freeboard')

        # Derivatives
        self.deriv_options['type'] = 'fd'
        self.deriv_options['form'] = 'central'
        self.deriv_options['check_form'] = 'central'
        self.deriv_options['step_calc'] = 'relative'

    def solve_nonlinear(self, params, unknowns, resids):
        # Unpack variables
        freeboard = params['freeboard']

        # With waterline at z=0, set the z-position of section nodes
        # Note sections and nodes start at bottom of column and move up
        draft     = params['z_param_in'][-1] - freeboard
        z_full    = params['z_full_in'] - draft 
        z_param   = params['z_param_in'] - draft 
        z_section = params['section_center_of_mass'] - draft 
        unknowns['draft']     = draft
        unknowns['z_full']    = z_full
        unknowns['z_param']   = z_param
        unknowns['z_section'] = z_section

        # Create constraint output that draft is less than water depth
        unknowns['draft_margin'] = draft / params['max_draft']

        # Make sure freeboard is more than 20% of Hs (DNV-OS-J101)
        unknowns['wave_height_freeboard_ratio'] = params['Hs'] / np.abs(freeboard)

        # Sectional stiffener properties
        unknowns['t_web']        = sectionalInterp(z_section, z_param, params['stiffener_web_thickness'])
        unknowns['t_flange']     = sectionalInterp(z_section, z_param, params['stiffener_flange_thickness'])
        unknowns['h_web']        = sectionalInterp(z_section, z_param, params['stiffener_web_height'])
        unknowns['w_flange']     = sectionalInterp(z_section, z_param, params['stiffener_flange_width'])
        unknowns['L_stiffener']  = sectionalInterp(z_section, z_param, params['stiffener_spacing'])
        


class ColumnProperties(Component):
    """
    OpenMDAO Component class for column substructure elements in floating offshore wind turbines.
    """

    def __init__(self, nFull):
        super(ColumnProperties,self).__init__()

        # Variables local to the class and not OpenMDAO
        self.ibox = None
        
        # Environment
        self.add_param('water_density', val=0.0, units='kg/m**3', desc='density of water')

        # Inputs from Geometry
        self.add_param('z_full', val=np.zeros((nFull,)), units='m', desc='z-coordinates of section nodes (length = nsection+1)')
        self.add_param('z_section', val=np.zeros((nFull-1,)), units='m', desc='z-coordinates of section centers of mass (length = nsection)')

        # Design variables
        self.add_param('d_full', val=np.zeros((nFull,)), units='m', desc='outer diameter at each section node bottom to top (length = nsection + 1)')
        self.add_param('t_full', val=np.zeros((nFull-1,)), units='m', desc='shell wall thickness at each section node bottom to top (length = nsection + 1)')
        self.add_param('buoyancy_tank_diameter', val=0.0, units='m', desc='Radius of heave plate at bottom of column')
        
        # Mass correction factors from simple rules here to real life
        self.add_param('shell_mass', val=np.zeros(nFull-1), units='kg', desc='mass of column shell')
        self.add_param('stiffener_mass', val=np.zeros(nFull-1), units='kg', desc='mass of column stiffeners')
        self.add_param('bulkhead_mass', val=np.zeros(nFull), units='kg', desc='mass of column bulkheads')
        self.add_param('buoyancy_tank_mass', val=0.0, units='kg', desc='mass of heave plate')
        self.add_param('ballast_mass', val=np.zeros(nFull-1), units='kg', desc='mass of permanent ballast')

        self.add_param('buoyancy_tank_cg', val=0.0, units='m', desc='z-coordinate of center of mass for buoyancy tank')
        self.add_param('ballast_z_cg', val=0.0, units='m', desc='z-coordinate or permanent ballast center of gravity')
        self.add_param('column_mass_factor', val=0.0, desc='Overall column mass correction factor')
        self.add_param('outfitting_mass_fraction', val=0.0, desc='Mass fraction added for outfitting')

        # Moments of inertia
        self.add_param('shell_I_keel', val=np.zeros(6), units='kg*m**2', desc='Moments of inertia of outer shell relative to keel point')
        self.add_param('bulkhead_I_keel', val=np.zeros(6), units='kg*m**2', desc='Moments of inertia of bulkheads relative to keel point')
        self.add_param('stiffener_I_keel', val=np.zeros(6), units='kg*m**2', desc='Moments of inertia of stiffeners relative to keel point')
        self.add_param('buoyancy_tank_I_keel', val=np.zeros(6), units='kg*m**2', desc='Moments of inertia of heave plate relative to keel point')
        self.add_param('ballast_I_keel', val=np.zeros(6), units='kg*m**2', desc='Moments of inertia of permanent ballast relative to keel point')

        # For buoyancy
        self.add_param('buoyancy_tank_displacement', val=0.0, units='m**3', desc='volume of water displaced by buoyancy tank')
        
        # Costs and cost rates
        self.add_param('shell_cost', val=0.0, units='USD', desc='mass of column shell')
        self.add_param('stiffener_cost', val=0.0, units='USD', desc='mass of column stiffeners')
        self.add_param('bulkhead_cost', val=0.0, units='USD', desc='mass of column bulkheads')
        self.add_param('ballast_cost', val=0.0, units='USD', desc='cost of permanent ballast')
        self.add_param('buoyancy_tank_cost', val=0.0, units='USD', desc='mass of heave plate')
        self.add_param('material_cost_rate', 0.0, units='USD/kg', desc='Raw material cost: steel $1.1/kg, aluminum $3.5/kg')
        self.add_param('outfitting_cost_rate', val=0.0, units='USD/kg', desc='Cost per unit mass for outfitting column')

        # Outputs
        self.add_output('z_center_of_mass', val=0.0, units='m', desc='z-position CofG of column')
        self.add_output('z_center_of_buoyancy', val=0.0, units='m', desc='z-position CofB of column')
        self.add_output('Awater', val=0.0, units='m**2', desc='Area of waterplace cross section')
        self.add_output('Iwater', val=0.0, units='m**4', desc='Second moment of area of waterplace cross section')
        self.add_output('I_column', val=np.zeros(6), units='kg*m**2', desc='Moments of inertia of whole column relative to keel point')
        self.add_output('displaced_volume', val=np.zeros((nFull-1,)), units='m**3', desc='Volume of water displaced by column by section')
        self.add_output('hydrostatic_force', val=np.zeros((nFull-1,)), units='N', desc='Net z-force on column sections')
 
        self.add_output('column_structural_mass', val=0.0, units='kg', desc='mass of column structure')
        
        self.add_output('column_outfitting_cost', val=0.0, units='USD', desc='cost of outfitting the column')
        self.add_output('column_outfitting_mass', val=0.0, units='kg', desc='cost of outfitting the column')

        self.add_output('column_added_mass', val=np.zeros(6), units='kg', desc='hydrodynamic added mass matrix diagonal')
        self.add_output('column_total_mass', val=np.zeros((nFull-1,)), units='kg', desc='total mass of column by section')
        self.add_output('column_total_cost', val=0.0, units='USD', desc='total cost of column')
        self.add_output('column_structural_cost', val=0.0, units='USD', desc='Cost of column without ballast or outfitting')
        self.add_output('tapered_column_cost_rate', val=0.0, units='USD/t', desc='Cost rate of finished column')
        
        # Derivatives
        self.deriv_options['type'] = 'fd'
        self.deriv_options['form'] = 'central'
        self.deriv_options['check_form'] = 'central'
        self.deriv_options['step_calc'] = 'relative'
        self.deriv_options['step_size'] = 1e-5
        
        
    def solve_nonlinear(self, params, unknowns, resids):

        self.compute_column_mass_cg(params, unknowns)
        self.balance_column(params, unknowns)
        self.compute_cost(params, unknowns)

        

    def compute_column_mass_cg(self, params, unknowns):
        """Computes column mass from components: Shell, Stiffener rings, Bulkheads
        Also computes center of mass of the shell by weighted sum of the components' position
        
        INPUTS:
        ----------
        params   : dictionary of input parameters
        unknowns : dictionary of output parameters
        
        OUTPUTS:
        ----------
        section_mass class variable set
        m_column   : column mass
        z_cg     : center of mass along z-axis for the column
        column_mass       in 'unknowns' dictionary set
        shell_mass      in 'unknowns' dictionary set
        stiffener_mass  in 'unknowns' dictionary set
        bulkhead_mass   in 'unknowns' dictionary set
        outfitting_mass in 'unknowns' dictionary set
        """
        # Unpack variables
        out_frac     = params['outfitting_mass_fraction']
        coeff        = params['column_mass_factor']
        z_nodes      = params['z_full']
        z_section    = params['z_section']
        z_box        = params['buoyancy_tank_cg']
        z_ballast    = params['ballast_z_cg']
        m_shell      = params['shell_mass']
        m_stiffener  = params['stiffener_mass']
        m_bulkhead   = params['bulkhead_mass']
        m_box        = params['buoyancy_tank_mass']
        m_ballast    = params['ballast_mass']
        I_shell      = params['shell_I_keel']
        I_stiffener  = params['stiffener_I_keel']
        I_bulkhead   = params['bulkhead_I_keel']
        I_box        = params['buoyancy_tank_I_keel']
        I_ballast    = params['ballast_I_keel']

        # Consistency check
        if out_frac > 1.0: out_frac -= 1.0
        
        # Initialize summations
        m_column  = 0.0
        z_cg      = 0.0
        
        # Find mass of all of the sub-components of the column
        # Masses assumed to be focused at section centroids
        m_column += (m_shell + m_stiffener).sum()
        z_cg     += np.dot(m_shell+m_stiffener, z_section)

        # Masses assumed to be centered at nodes
        m_column += m_bulkhead.sum()
        z_cg     += np.dot(m_bulkhead, z_nodes)

        # Mass with variable location
        m_column += m_box
        z_cg     += m_box*z_box

        # Account for components not explicitly calculated here
        m_column *= coeff

        # Compute CG position of the column
        z_cg     *= coeff / m_column

        # Now calculate outfitting mass, evenly distributed so cg doesn't change
        m_outfit  = out_frac * m_column

        # Add in ballast
        m_total   = m_column + m_outfit + m_ballast.sum()
        z_cg      = ( (m_column+m_outfit)*z_cg + m_ballast.sum()*z_ballast ) / m_total

        # Find sections for ballast and buoyancy tank
        ibox  = 0
        try:
            ibox  = np.where(z_box >= z_nodes)[0][-1]
        except:
            print(z_box, z_ballast, z_nodes)
        self.ibox = ibox

        # Now do tally by section
        m_sections         = coeff*(m_shell + m_stiffener + m_bulkhead[:-1]) + m_ballast
        m_sections        += m_outfit / m_shell.size
        m_sections[-1]    += coeff*m_bulkhead[-1]
        m_sections[ibox]  += coeff*m_box

        # Add up moments of inertia at keel, make sure to scale mass appropriately
        I_total   = ((1+out_frac) * coeff) * (I_shell + I_stiffener + I_bulkhead + I_box) + I_ballast

        # Move moments of inertia from keel to cg
        I_total  -= m_total*((z_cg-z_nodes[0])**2.0) * np.r_[1.0, 1.0, np.zeros(4)]
        I_total   = np.maximum(I_total, 0.0)

        # Store outputs addressed so far
        unknowns['column_total_mass']      = m_sections
        unknowns['column_structural_mass'] = m_column + m_outfit
        unknowns['column_outfitting_mass'] = m_outfit
        unknowns['z_center_of_mass']       = z_cg
        unknowns['I_column']               = I_total
        

        
    def balance_column(self, params, unknowns):
        # Unpack variables
        R_od              = 0.5*params['d_full']
        R_plate           = 0.5*params['buoyancy_tank_diameter']
        z_nodes           = params['z_full']
        z_box             = params['buoyancy_tank_cg']
        V_box             = params['buoyancy_tank_displacement']
        rho_water         = params['water_density']
        nsection          = R_od.size - 1

        # Compute volume of each section and mass of displaced water by section
        # Find the radius at the waterline so that we can compute the submerged volume as a sum of frustum sections
        if z_nodes[-1] > 0.0:
            r_waterline = np.interp(0.0, z_nodes, R_od)
            z_under     = np.r_[z_nodes[z_nodes < 0.0], 0.0]
            r_under     = np.r_[R_od[z_nodes < 0.0], r_waterline]
        else:
            r_waterline = R_od[-1]
            r_under     = R_od
            z_under     = z_nodes

        # Submerged volume (with zero-padding)
        V_under = frustum.frustumVol(r_under[:-1], r_under[1:], np.diff(z_under))
        add0    = np.maximum(0, nsection-V_under.size)
        unknowns['displaced_volume'] = np.r_[V_under, np.zeros(add0)]
        unknowns['displaced_volume'][self.ibox] += V_box

        # Compute Center of Buoyancy in z-coordinates (0=waterline)
        # First get z-coordinates of CG of all frustums
        z_cg_under  = frustum.frustumCG(r_under[:-1], r_under[1:], np.diff(z_under))
        z_cg_under += z_under[:-1]
        # Now take weighted average of these CG points with volume
        z_cb     = ( (V_box*z_box) + np.dot(V_under, z_cg_under)) / (unknowns['displaced_volume'].sum() + eps)
        unknowns['z_center_of_buoyancy'] = z_cb

        # Find total hydrostatic force by section- sign says in which direction force acts
        # Since we are working on z_under grid, need to redefine z_section, ibox, etc.
        z_undersec,_ = nodal2sectional(z_under)
        if z_box > 0.0 and V_box == 0.0:
            ibox = 0
        else:
            ibox = np.where(z_box >= z_under)[0][-1]
        F_hydro      = np.pi * np.diff(r_under**2.0) * np.maximum(0.0, -z_undersec) #cg_under))
        if F_hydro.size > 0:
            F_hydro[0] += np.pi * r_under[0]**2 * (-z_under[0])
            if z_nodes[-1] < 0.0:
                F_hydro[-1] -= np.pi * r_under[-1]**2 * (-z_under[-1])
            F_hydro[ibox] += V_box
            F_hydro    *= rho_water * gravity
        unknowns['hydrostatic_force'] = np.r_[F_hydro, np.zeros(add0)]
        
        # 2nd moment of area for circular cross section
        # Note: Assuming Iwater here depends on "water displacement" cross-section
        # and not actual moment of inertia type of cross section (thin hoop)
        unknowns['Iwater'] = 0.25 * np.pi * r_waterline**4.0
        unknowns['Awater'] = np.pi * r_waterline**2.0

        # Calculate diagonal entries of added mass matrix
        # Prep for integrals too
        npts     = 1e2 * R_od.size
        zpts     = np.linspace(z_under[0], z_under[-1], npts)
        r_under  = np.interp(zpts, z_under, r_under)
        m_a      = np.zeros(6)
        m_a[:2]  = rho_water * unknowns['displaced_volume'].sum() # A11 surge, A22 sway
        m_a[2]   = 0.5 * (8.0/3.0) * rho_water * np.maximum(R_plate, r_under.max())**3.0# A33 heave
        m_a[3:5] = np.pi * rho_water * np.trapz((zpts-z_cb)**2.0 * r_under**2.0, zpts)# A44 roll, A55 pitch
        m_a[5]   = 0.0 # A66 yaw
        unknowns['column_added_mass'] = m_a

        
    def compute_cost(self, params, unknowns):
        unknowns['column_structural_cost']   = params['column_mass_factor']*(params['shell_cost'] + params['stiffener_cost'] +
                                                                             params['bulkhead_cost'] + params['buoyancy_tank_cost'])
        unknowns['column_outfitting_cost']   = params['outfitting_cost_rate'] * unknowns['column_outfitting_mass']
        unknowns['column_total_cost']        = unknowns['column_structural_cost'] + unknowns['column_outfitting_cost'] + params['ballast_cost']
        unknowns['tapered_column_cost_rate'] = 1e3*unknowns['column_total_cost']/unknowns['column_total_mass'].sum()

        
class ColumnBuckling(Component):
    '''
    This function computes the applied axial and hoop stresses in a column and compares that to 
    limits established by the API standard.  Some physcial geometry checks are also performed.
    '''
    def __init__(self, nSection, nFull):
        super(ColumnBuckling,self).__init__()

        # From other modules
        self.add_param('stack_mass_in', val=eps, units='kg', desc='Weight above the cylinder column')
        self.add_param('section_mass', val=np.zeros((nFull-1,)), units='kg', desc='total mass of column by section')
        self.add_param('pressure', np.zeros(nFull), units='N/m**2', desc='Dynamic (and static)? pressure')
        
        self.add_param('d_full', np.zeros(nFull), units='m', desc='cylinder diameter at corresponding locations')
        self.add_param('t_full', np.zeros(nFull-1), units='m', desc='shell thickness at corresponding locations')
        self.add_param('z_full', val=np.zeros(nFull), units='m', desc='z-coordinates of section nodes (length = nsection+1)')

        self.add_param('h_web', val=np.zeros((nFull-1,)), units='m', desc='height of stiffener web (base of T) within each section bottom to top')
        self.add_param('t_web', val=np.zeros((nFull-1,)), units='m', desc='thickness of stiffener web (base of T) within each section bottom to top')
        self.add_param('w_flange', val=np.zeros((nFull-1,)), units='m', desc='height of stiffener flange (top of T) within each section bottom to top')
        self.add_param('t_flange', val=np.zeros((nFull-1,)), units='m', desc='thickness of stiffener flange (top of T) within each section bottom to top')
        self.add_param('L_stiffener', val=np.zeros((nFull-1,)), units='m', desc='Axial distance from one ring stiffener to another within each section bottom to top')

        self.add_param('E', val=0.0, units='Pa', desc='Modulus of elasticity (Youngs) of material')
        self.add_param('nu', val=0.0, desc='poissons ratio of column material')
        self.add_param('yield_stress', val=0.0, units='Pa', desc='yield stress of material')

        self.add_param('loading', val='hydro', desc='Loading type in API checks [hydro/radial]', pass_by_obj=True)
        self.add_param('gamma_f', 0.0, desc='safety factor on loads')
        self.add_param('gamma_b', 0.0, desc='buckling safety factor')
        
        # Output constraints
        self.add_output('flange_compactness', val=np.zeros((nFull-1,)), desc='check for flange compactness')
        self.add_output('web_compactness', val=np.zeros((nFull-1,)), desc='check for web compactness')
        
        self.add_output('axial_local_api', val=np.zeros((nFull-1,)), desc='unity check for axial load with API safety factors - local buckling')
        self.add_output('axial_general_api', val=np.zeros((nFull-1,)), desc='unity check for axial load with API safety factors- genenral instability')
        self.add_output('external_local_api', val=np.zeros((nFull-1,)), desc='unity check for external pressure with API safety factors- local buckling')
        self.add_output('external_general_api', val=np.zeros((nFull-1,)), desc='unity check for external pressure with API safety factors- general instability')

        self.add_output('axial_local_utilization', val=np.zeros((nFull-1,)), desc='utilization check for axial load - local buckling')
        self.add_output('axial_general_utilization', val=np.zeros((nFull-1,)), desc='utilization check for axial load - genenral instability')
        self.add_output('external_local_utilization', val=np.zeros((nFull-1,)), desc='utilization check for external pressure - local buckling')
        self.add_output('external_general_utilization', val=np.zeros((nFull-1,)), desc='utilization check for external pressure - general instability')
        
        # Derivatives
        self.deriv_options['type'] = 'fd'
        self.deriv_options['form'] = 'central'
        self.deriv_options['check_form'] = 'central'
        self.deriv_options['step_calc'] = 'relative'
        self.deriv_options['step_size'] = 1e-5

        
    def compute_applied_axial(self, params):
        """Compute axial stress for column from z-axis loading

        INPUTS:
        ----------
        params       : dictionary of input parameters
        section_mass : float (scalar/vector),  mass of each column section as axial loading increases with column depth

        OUTPUTS:
        -------
        stress   : float (scalar/vector),  axial stress
        """
        # Unpack variables
        R_od,_         = nodal2sectional(params['d_full'])
        R_od          *= 0.5
        t_wall         = params['t_full']
        section_mass   = params['section_mass']
        m_stack        = params['stack_mass_in']
        
        # Middle radius
        R_m = R_od - 0.5*t_wall
        # Add in weight of sections above it
        axial_load = m_stack + np.r_[0.0, np.cumsum(section_mass[:-1])]
        # Divide by shell cross sectional area to get stress
        return (gravity * axial_load / (2.0 * np.pi * R_m * t_wall))

    
    def solve_nonlinear(self, params, unknowns, resids):
        # Unpack variables
        R_od,_       = nodal2sectional( params['d_full'] )
        R_od        *= 0.5
        h_section    = np.diff( params['z_full'] )
        t_wall       = params['t_full']
        
        t_web        = params['t_web']
        t_flange     = params['t_flange']
        h_web        = params['h_web']
        w_flange     = params['w_flange']
        L_stiffener  = params['L_stiffener']

        gamma_f      = params['gamma_f']
        gamma_b      = params['gamma_b']
        
        E            = params['E'] # Young's modulus
        nu           = params['nu'] # Poisson ratio
        sigma_y      = params['yield_stress']
        loading      = params['loading']
        nodalP,_     = nodal2sectional( params['pressure'] )
        pressure     = 1e-12 if loading in ['ax','axial','testing','test'] else nodalP+1e-12

        # Apply quick "compactness" check on stiffener geometry
        # Constraint is that these must be >= 1
        flange_compactness = 0.375 * (t_flange / (0.5*w_flange)) * np.sqrt(E / sigma_y)
        web_compactness    = 1.0   * (t_web    / h_web         ) * np.sqrt(E / sigma_y)

        # Compute applied axial stress simply, like API guidelines (as opposed to running frame3dd)
        sigma_ax = self.compute_applied_axial(params)
        (axial_local_api, axial_general_api, external_local_api, external_general_api,
         axial_local_raw, axial_general_raw, external_local_raw, external_general_raw) = shellBuckling_withStiffeners(
             pressure, sigma_ax, R_od, t_wall, h_section,
             h_web, t_web, w_flange, t_flange,
             L_stiffener, E, nu, sigma_y, loading)
        
        unknowns['flange_compactness']     = flange_compactness
        unknowns['web_compactness']        = web_compactness
        
        unknowns['axial_local_api']      = axial_local_api
        unknowns['axial_general_api']    = axial_general_api
        unknowns['external_local_api']   = external_local_api
        unknowns['external_general_api'] = external_general_api

        unknowns['axial_local_utilization']      = axial_local_raw * gamma_f*gamma_b
        unknowns['axial_general_utilization']    = axial_general_raw * gamma_f*gamma_b
        unknowns['external_local_utilization']   = external_local_raw * gamma_f*gamma_b
        unknowns['external_general_utilization'] = external_general_raw * gamma_f*gamma_b


class Column(Group):
    def __init__(self, nSection, nFull):
        super(Column,self).__init__()

        nRefine = (nFull-1)/nSection
        
        self.add('cyl_geom', CylinderDiscretization(nSection+1, nRefine), promotes=['section_height','diameter','wall_thickness',
                                                                                    'd_full','t_full','foundation_height'])
        
        self.add('gc', GeometricConstraints(nSection+1, diamFlag=True), promotes=['max_taper','min_d_to_t','manufacturability','weldability'])

        self.add('cyl_mass', CylinderMass(nFull), promotes=['d_full','t_full','material_density',
                                                            'material_cost_rate','labor_cost_rate','painting_cost_rate',
                                                            'section_center_of_mass'])

        self.add('col_geom', ColumnGeometry(nSection, nFull), promotes=['*'])

        self.add('bulk', BulkheadProperties(nSection, nFull), promotes=['*'])

        self.add('stiff', StiffenerProperties(nSection,nFull), promotes=['*'])

        self.add('plate', BuoyancyTankProperties(nFull), promotes=['*'])

        self.add('ball', BallastProperties(nFull), promotes=['*'])

        self.add('col', ColumnProperties(nFull), promotes=['*'])

        self.add('wind', PowerWind(nFull), promotes=['Uref','zref','shearExp','z0'])
        self.add('wave', LinearWaves(nFull), promotes=['Uc','hmax','T'])
        self.add('windLoads', CylinderWindDrag(nFull), promotes=['cd_usr','beta'])
        self.add('waveLoads', CylinderWaveDrag(nFull), promotes=['cm','cd_usr'])
        self.add('distLoads', AeroHydroLoads(nFull), promotes=['Px','Py','Pz','qdyn','yaw'])

        self.add('buck', ColumnBuckling(nSection, nFull), promotes=['*'])
        
        self.connect('diameter', 'gc.d')
        self.connect('wall_thickness', 'gc.t')
        self.connect('cyl_geom.z_param', 'z_param_in')
        self.connect('cyl_geom.z_full', ['cyl_mass.z_full','z_full_in'])
        
        #self.connect('cyl_mass.section_center_of_mass', 'col_geom.section_center_of_mass')
        
        self.connect('cyl_mass.mass', 'shell_mass')
        self.connect('cyl_mass.cost', 'shell_cost')
        self.connect('cyl_mass.I_base', 'shell_I_keel')
        self.connect('material_density','rho')
        
        self.connect('column_total_mass', 'section_mass')

        self.connect('water_depth','wave.z_floor')
        self.connect('z_full', ['wind.z', 'wave.z', 'windLoads.z','waveLoads.z','distLoads.z'])
        self.connect('d_full', ['windLoads.d','waveLoads.d'])
        self.connect('beta','waveLoads.beta')
        self.connect('z0', 'wave.z_surface')

        self.connect('wind.U', 'windLoads.U')
        self.connect('Hs', 'hmax')

        self.connect('water_density',['wave.rho','waveLoads.rho'])
        self.connect('wave.U', 'waveLoads.U')
        self.connect('wave.A', 'waveLoads.A')
        self.connect('wave.p', 'waveLoads.p')
        
        # connections to distLoads1
        self.connect('windLoads.windLoads_Px', 'distLoads.windLoads_Px')
        self.connect('windLoads.windLoads_Py', 'distLoads.windLoads_Py')
        self.connect('windLoads.windLoads_Pz', 'distLoads.windLoads_Pz')
        self.connect('windLoads.windLoads_qdyn', 'distLoads.windLoads_qdyn')
        self.connect('windLoads.windLoads_beta', 'distLoads.windLoads_beta')
        self.connect('windLoads.windLoads_z', 'distLoads.windLoads_z')
        self.connect('windLoads.windLoads_d', 'distLoads.windLoads_d')
        
        self.connect('waveLoads.waveLoads_Px', 'distLoads.waveLoads_Px')
        self.connect('waveLoads.waveLoads_Py', 'distLoads.waveLoads_Py')
        self.connect('waveLoads.waveLoads_Pz', 'distLoads.waveLoads_Pz')
        self.connect('waveLoads.waveLoads_pt', 'distLoads.waveLoads_qdyn')
        self.connect('waveLoads.waveLoads_beta', 'distLoads.waveLoads_beta')
        self.connect('waveLoads.waveLoads_z', 'distLoads.waveLoads_z')
        self.connect('waveLoads.waveLoads_d', 'distLoads.waveLoads_d')

        self.connect('qdyn', 'pressure')
