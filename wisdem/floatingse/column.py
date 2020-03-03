from openmdao.api import Group, IndepVarComp, ExplicitComponent
import numpy as np

from wisdem.commonse.utilities import nodal2sectional, sectional2nodal, assembleI, unassembleI, sectionalInterp
import wisdem.commonse.frustum as frustum
import wisdem.commonse.manufacturing as manufacture
from wisdem.commonse.UtilizationSupplement import shellBuckling_withStiffeners, GeometricConstraints
from wisdem.commonse import gravity, eps, AeroHydroLoads, CylinderWindDrag, CylinderWaveDrag
from wisdem.commonse.vertical_cylinder import CylinderDiscretization, CylinderMass
from wisdem.commonse.environment import PowerWind, LinearWaves

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
    

class BulkheadProperties(ExplicitComponent):

    def initialize(self):
        self.options.declare('nSection')
        self.options.declare('nFull')
        
    def setup(self):
        nSection = self.options['nSection']
        nFull    = self.options['nFull']
        
        self.bulk_full = np.zeros( nFull, dtype=np.int_)

        self.add_input('z_full', val=np.zeros(nFull), units='m', desc='z-coordinates of section nodes (length = nsection+1)')
        self.add_input('z_param', val=np.zeros((nSection+1,)), units='m', desc='z-coordinates of section nodes (length = nsection+1)')
        self.add_input('d_full', val=np.zeros(nFull), units='m', desc='cylinder diameter at corresponding locations')
        self.add_input('t_full', val=np.zeros(nFull-1), units='m', desc='shell thickness at corresponding locations')
        self.add_input('rho', val=0.0, units='kg/m**3', desc='material density')
        self.add_input('bulkhead_thickness', val=np.zeros(nSection+1), units='m', desc='Nodal locations of bulkhead thickness, zero meaning no bulkhead, bottom to top (length = nsection + 1)')
        self.add_input('bulkhead_mass_factor', val=0.0, desc='Bulkhead mass correction factor')
        
        self.add_input('shell_mass', val=np.zeros(nFull-1), units='kg', desc='mass of column shell')
        self.add_input('material_cost_rate', 0.0, units='USD/kg', desc='Raw material cost rate: steel $1.1/kg, aluminum $3.5/kg')
        self.add_input('labor_cost_rate', 0.0, units='USD/min', desc='Labor cost rate')
        self.add_input('painting_cost_rate', 0.0, units='USD/m/m', desc='Painting / surface finishing cost rate')

        self.add_output('bulkhead_mass', val=np.zeros(nFull), units='kg', desc='mass of column bulkheads')
        self.add_output('bulkhead_cost', val=0.0, units='USD', desc='cost of column bulkheads')
        self.add_output('bulkhead_I_keel', val=np.zeros(6), units='kg*m**2', desc='Moments of inertia of bulkheads relative to keel point')
        
        # Derivatives
        self.declare_partials('*', '*', method='fd', form='central', step=1e-6)

        
    def compute(self, inputs, outputs):
        # Unpack variables
        z_full     = inputs['z_full'] # at section nodes
        z_param    = inputs['z_param']
        R_od       = 0.5*inputs['d_full'] # at section nodes
        twall      = inputs['t_full'] # at section nodes
        R_id       = get_inner_radius(R_od, twall)
        t_bulk     = inputs['bulkhead_thickness'] # at section nodes
        rho        = inputs['rho']
        
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
        m_bulk = inputs['bulkhead_mass_factor'] * rho * V_bulk

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
        k_m     = inputs['material_cost_rate'] #1.1 # USD / kg carbon steel plate
        k_f     = inputs['labor_cost_rate'] #1.0 # USD / min labor
        k_p     = inputs['painting_cost_rate'] #USD / m^2 painting
        m_shell = inputs['shell_mass'].sum()
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
        outputs['bulkhead_I_keel'] = unassembleI(I_keel)
        outputs['bulkhead_mass'] = m_bulk
        outputs['bulkhead_cost'] = c_bulk

    

class BuoyancyTankProperties(ExplicitComponent):

    def initialize(self):
        self.options.declare('nFull')
        
    def setup(self):
        nFull    = self.options['nFull']
        
        self.add_input('d_full', val=np.zeros(nFull), units='m', desc='cylinder diameter at corresponding locations')
        self.add_input('z_full', val=np.zeros(nFull), units='m', desc='z-coordinates of section nodes')
        self.add_input('rho', val=0.0, units='kg/m**3', desc='material density')

        self.add_input('shell_mass', val=np.zeros(nFull-1), units='kg', desc='mass of column shell')
        self.add_input('material_cost_rate', 0.0, units='USD/kg', desc='Raw material cost: steel $1.1/kg, aluminum $3.5/kg')
        self.add_input('labor_cost_rate', 0.0, units='USD/min', desc='Labor cost')
        self.add_input('painting_cost_rate', 0.0, units='USD/m/m', desc='Painting / surface finishing cost rate')
        
        self.add_input('buoyancy_tank_diameter', val=0.0, units='m', desc='Radius of heave plate at bottom of column')
        self.add_input('buoyancy_tank_height', val=0.0, units='m', desc='Radius of heave plate at bottom of column')
        self.add_input('buoyancy_tank_location', val=0.0, units='m', desc='Radius of heave plate at bottom of column')
        self.add_input('buoyancy_tank_mass_factor', val=0.0, desc='Heave plate mass correction factor')

        self.add_output('buoyancy_tank_mass', val=0.0, units='kg', desc='mass of buoyancy tank')
        self.add_output('buoyancy_tank_cost', val=0.0, units='USD', desc='cost of buoyancy tank')
        self.add_output('buoyancy_tank_cg', val=0.0, units='m', desc='z-coordinate of center of mass for buoyancy tank')
        self.add_output('buoyancy_tank_displacement', val=0.0, units='m**3', desc='volume of water displaced by buoyancy tank')
        self.add_output('buoyancy_tank_I_keel', val=np.zeros(6), units='kg*m**2', desc='Moments of inertia of heave plate relative to keel point')

        # Derivatives
        self.declare_partials('*', '*', method='fd', form='central', step=1e-6)

        
    def compute(self, inputs, outputs):
        # Unpack variables
        z_full    = inputs['z_full']

        R_od      = 0.5*inputs['d_full']
        R_plate   = 0.5*inputs['buoyancy_tank_diameter']
        h_box     = inputs['buoyancy_tank_height']

        location  = inputs['buoyancy_tank_location']
        
        coeff     = inputs['buoyancy_tank_mass_factor']
        rho       = inputs['rho']

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
        k_m     = inputs['material_cost_rate'] #1.1 # USD / kg carbon steel plate
        k_f     = inputs['labor_cost_rate'] #1.0 # USD / min labor
        k_p     = inputs['painting_cost_rate'] #USD / m^2 painting
        m_shell = inputs['shell_mass'].sum()

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
        outputs['buoyancy_tank_cost']         = c_box
        outputs['buoyancy_tank_mass']         = m_box
        outputs['buoyancy_tank_cg']           = z_cg
        outputs['buoyancy_tank_displacement'] = V_box
        outputs['buoyancy_tank_I_keel']       = unassembleI(I_keel)
        
        
class StiffenerProperties(ExplicitComponent):
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

    def initialize(self):
        self.options.declare('nSection')
        self.options.declare('nFull')
        
    def setup(self):
        nSection = self.options['nSection']
        nFull    = self.options['nFull']

        self.add_input('d_full', val=np.zeros(nFull), units='m', desc='cylinder diameter at corresponding locations')
        self.add_input('t_full', val=np.zeros(nFull-1), units='m', desc='shell thickness at corresponding locations')
        self.add_input('z_full', val=np.zeros(nFull), units='m', desc='z-coordinates of section nodes')
        self.add_input('rho', val=0.0, units='kg/m**3', desc='material density')
        
        self.add_input('shell_mass', val=np.zeros(nFull-1), units='kg', desc='mass of column shell')
        self.add_input('material_cost_rate', 0.0, units='USD/kg', desc='Raw material cost: steel $1.1/kg, aluminum $3.5/kg')
        self.add_input('labor_cost_rate', 0.0, units='USD/min', desc='Labor cost')
        self.add_input('painting_cost_rate', 0.0, units='USD/m/m', desc='Painting / surface finishing cost rate')

        self.add_input('h_web', val=np.zeros((nFull-1,)), units='m', desc='height of stiffener web (base of T) within each section bottom to top')
        self.add_input('t_web', val=np.zeros((nFull-1,)), units='m', desc='thickness of stiffener web (base of T) within each section bottom to top')
        self.add_input('w_flange', val=np.zeros((nFull-1,)), units='m', desc='height of stiffener flange (top of T) within each section bottom to top')
        self.add_input('t_flange', val=np.zeros((nFull-1,)), units='m', desc='thickness of stiffener flange (top of T) within each section bottom to top')
        self.add_input('L_stiffener', val=np.zeros((nFull-1,)), units='m', desc='Axial distance from one ring stiffener to another within each section bottom to top')

        self.add_input('ring_mass_factor', val=0.0, desc='Stiffener ring mass correction factor')
        
        self.add_output('stiffener_mass', val=np.zeros(nFull-1), units='kg', desc='mass of column stiffeners')
        self.add_output('stiffener_cost', val=0.0, units='USD', desc='cost of column stiffeners')
        self.add_output('stiffener_I_keel', val=np.zeros(6), units='kg*m**2', desc='Moments of inertia of stiffeners relative to keel point')
        self.add_output('number_of_stiffeners', val=np.zeros(nSection, dtype=np.int_), desc='number of stiffeners in each section')
        self.add_output('flange_spacing_ratio', val=np.zeros((nFull-1,)), desc='ratio between flange and stiffener spacing')
        self.add_output('stiffener_radius_ratio', val=np.zeros((nFull-1,)), desc='ratio between stiffener height and radius')

        # Derivatives
        self.declare_partials('*', '*', method='fd', form='central', step=1e-6)
        
    def compute(self, inputs, outputs):
        # Unpack variables
        R_od         = 0.5*inputs['d_full']
        t_wall       = inputs['t_full']
        z_full       = inputs['z_full'] # at section nodes
        h_section    = np.diff(z_full)
        V_shell      = frustum.frustumShellVol(R_od[:-1], R_od[1:], t_wall, h_section)
        R_od,_       = nodal2sectional( R_od ) # at section nodes
        
        t_web        = inputs['t_web']
        t_flange     = inputs['t_flange']
        h_web        = inputs['h_web']
        w_flange     = inputs['w_flange']
        L_stiffener  = inputs['L_stiffener']

        rho          = inputs['rho']
        
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
        m_web    = inputs['ring_mass_factor'] * rho * V_web
        m_flange = inputs['ring_mass_factor'] * rho * V_flange
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
        outputs['stiffener_mass'] =  n_stiff * m_ring

        # Find total number of stiffeners in each original section
        nSection    = self.options['nSection']
        npts_per    = int(h_web.size / nSection)
        n_stiff_sec = np.zeros(nSection)
        for k in range(npts_per):
            n_stiff_sec += n_stiff[k::npts_per]
        outputs['number_of_stiffeners'] = n_stiff_sec


        # Compute costs based on "Optimum Design of Steel Structures" by Farkas and Jarmai
        # All dimensions for correlations based on mm, not meters.
        k_m     = inputs['material_cost_rate'] #1.1 # USD / kg carbon steel plate
        k_f     = inputs['labor_cost_rate'] #1.0 # USD / min labor
        k_p     = inputs['painting_cost_rate'] #USD / m^2 painting
        m_shell = inputs['shell_mass'].sum()
        
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
        K_m = k_m * outputs['stiffener_mass'].sum()

        # Total cost
        c_ring = K_m + K_f + K_p
        
        # Store results
        outputs['stiffener_cost'] = c_ring
        outputs['stiffener_I_keel'] = unassembleI(I_keel)
        
        # Create some constraints for reasonable stiffener designs for an optimizer
        outputs['flange_spacing_ratio']   = w_flange / (0.5*L_stiffener)
        outputs['stiffener_radius_ratio'] = (h_web + t_flange + t_wall) / R_od



        
class BallastProperties(ExplicitComponent):

    def initialize(self):
        self.options.declare('nFull')
        
    def setup(self):
        nFull    = self.options['nFull']

        self.add_input('water_density', val=0.0, units='kg/m**3', desc='density of water')
        self.add_input('d_full', val=np.zeros(nFull), units='m', desc='cylinder diameter at corresponding locations')
        self.add_input('t_full', val=np.zeros(nFull-1), units='m', desc='shell thickness at corresponding locations')
        self.add_input('z_full', val=np.zeros(nFull), units='m', desc='z-coordinates of section nodes')
        self.add_input('permanent_ballast_density', val=0.0, units='kg/m**3', desc='density of permanent ballast')
        self.add_input('permanent_ballast_height', val=0.0, units='m', desc='height of permanent ballast')
        self.add_input('ballast_cost_rate', val=0.0, units='USD/kg', desc='Cost per unit mass of ballast')

        self.add_output('ballast_cost', val=0.0, units='USD', desc='cost of permanent ballast')
        self.add_output('ballast_mass', val=np.zeros(nFull-1), units='kg', desc='mass of permanent ballast')
        self.add_output('ballast_z_cg', val=0.0, units='m', desc='z-coordinate or permanent ballast center of gravity')
        self.add_output('ballast_I_keel', val=np.zeros(6), units='kg*m**2', desc='Moments of inertia of permanent ballast relative to keel point')
        self.add_output('variable_ballast_interp_zpts', val=np.zeros((nFull,)), units='m', desc='z-points of potential ballast mass')
        self.add_output('variable_ballast_interp_radius', val=np.zeros((nFull,)), units='m', desc='inner radius of column at potential ballast mass')

        
    def compute(self, inputs, outputs):
        # Unpack variables
        R_od        = 0.5*inputs['d_full']
        t_wall      = inputs['t_full']
        z_nodes     = inputs['z_full']
        h_ballast   = float(inputs['permanent_ballast_height'])
        rho_ballast = float(inputs['permanent_ballast_density'])
        rho_water   = float(inputs['water_density'])
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
        #z_water_start = z_water_start + inputs['variable_ballast_start'] * (z_nodes[-1] - z_water_start)
        
        # Find height of water ballast numerically by finding the height that integrates to the mass we want
        # This step is completed in column.py or semi.py because we must account for other substructure elements too
        zpts    = np.linspace(z_water_start, 0.0, npts)
        R_id    = np.interp(zpts, z_nodes, R_id_orig)
        outputs['variable_ballast_interp_zpts']   = zpts
        outputs['variable_ballast_interp_radius'] = R_id
        
        # Save permanent ballast mass and variable height
        outputs['ballast_mass']   = section_mass
        outputs['ballast_I_keel'] = I_keel
        outputs['ballast_z_cg']   = z_cg_perm
        outputs['ballast_cost']   = inputs['ballast_cost_rate'] * m_perm


        
        
class ColumnGeometry(ExplicitComponent):
    """
    OpenMDAO Component class for vertical columns in substructure for floating offshore wind turbines.
    """

    def initialize(self):
        self.options.declare('nSection')
        self.options.declare('nFull')
        
    def setup(self):
        nSection = self.options['nSection']
        nFull    = self.options['nFull']

        # Design variables
        self.add_input('water_depth', val=0.0, units='m', desc='water depth')
        self.add_input('Hs', val=0.0, units='m', desc='significant wave height')
        self.add_input('freeboard', val=0.0, units='m', desc='Length of column above water line')
        self.add_input('max_draft', val=0.0, units='m', desc='Maxmimum length of column below water line')
        self.add_input('z_full_in', val=np.zeros((nFull,)), units='m', desc='z-coordinates of section nodes (length = nsection+1)')
        self.add_input('z_param_in', val=np.zeros((nSection+1,)), units='m', desc='z-coordinates of section nodes (length = nsection+1)')
        self.add_input('section_center_of_mass', val=np.zeros(nFull-1), units='m', desc='z position of center of mass of each can in the cylinder')

        self.add_input('stiffener_web_height', val=np.zeros((nSection,)), units='m', desc='height of stiffener web (base of T) within each section bottom to top (length = nsection)')
        self.add_input('stiffener_web_thickness', val=np.zeros((nSection,)), units='m', desc='thickness of stiffener web (base of T) within each section bottom to top (length = nsection)')
        self.add_input('stiffener_flange_width', val=np.zeros((nSection,)), units='m', desc='height of stiffener flange (top of T) within each section bottom to top (length = nsection)')
        self.add_input('stiffener_flange_thickness', val=np.zeros((nSection,)), units='m', desc='thickness of stiffener flange (top of T) within each section bottom to top (length = nsection)')
        self.add_input('stiffener_spacing', val=np.zeros((nSection,)), units='m', desc='Axial distance from one ring stiffener to another within each section bottom to top (length = nsection)')

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
        self.declare_partials('*', '*', method='fd', form='central', step=1e-6)

    def compute(self, inputs, outputs):
        # Unpack variables
        freeboard = inputs['freeboard']

        # With waterline at z=0, set the z-position of section nodes
        # Note sections and nodes start at bottom of column and move up
        draft     = inputs['z_param_in'][-1] - freeboard
        z_full    = inputs['z_full_in'] - draft 
        z_param   = inputs['z_param_in'] - draft 
        z_section = inputs['section_center_of_mass'] - draft 
        outputs['draft']     = draft
        outputs['z_full']    = z_full
        outputs['z_param']   = z_param
        outputs['z_section'] = z_section

        # Create constraint output that draft is less than water depth
        outputs['draft_margin'] = draft / inputs['max_draft']

        # Make sure freeboard is more than 20% of Hs (DNV-OS-J101)
        outputs['wave_height_freeboard_ratio'] = inputs['Hs'] / np.abs(freeboard)

        # Sectional stiffener properties
        outputs['t_web']        = sectionalInterp(z_section, z_param, inputs['stiffener_web_thickness'])
        outputs['t_flange']     = sectionalInterp(z_section, z_param, inputs['stiffener_flange_thickness'])
        outputs['h_web']        = sectionalInterp(z_section, z_param, inputs['stiffener_web_height'])
        outputs['w_flange']     = sectionalInterp(z_section, z_param, inputs['stiffener_flange_width'])
        outputs['L_stiffener']  = sectionalInterp(z_section, z_param, inputs['stiffener_spacing'])
        


class ColumnProperties(ExplicitComponent):
    """
    OpenMDAO Component class for column substructure elements in floating offshore wind turbines.
    """

    def initialize(self):
        self.options.declare('nFull')
        
    def setup(self):
        nFull    = self.options['nFull']

        # Variables local to the class and not OpenMDAO
        self.ibox = None
        
        # Environment
        self.add_input('water_density', val=0.0, units='kg/m**3', desc='density of water')

        # Inputs from Geometry
        self.add_input('z_full', val=np.zeros((nFull,)), units='m', desc='z-coordinates of section nodes (length = nsection+1)')
        self.add_input('z_section', val=np.zeros((nFull-1,)), units='m', desc='z-coordinates of section centers of mass (length = nsection)')

        # Design variables
        self.add_input('d_full', val=np.zeros((nFull,)), units='m', desc='outer diameter at each section node bottom to top (length = nsection + 1)')
        self.add_input('t_full', val=np.zeros((nFull-1,)), units='m', desc='shell wall thickness at each section node bottom to top (length = nsection + 1)')
        self.add_input('buoyancy_tank_diameter', val=0.0, units='m', desc='Radius of heave plate at bottom of column')
        
        # Mass correction factors from simple rules here to real life
        self.add_input('shell_mass', val=np.zeros(nFull-1), units='kg', desc='mass of column shell')
        self.add_input('stiffener_mass', val=np.zeros(nFull-1), units='kg', desc='mass of column stiffeners')
        self.add_input('bulkhead_mass', val=np.zeros(nFull), units='kg', desc='mass of column bulkheads')
        self.add_input('buoyancy_tank_mass', val=0.0, units='kg', desc='mass of heave plate')
        self.add_input('ballast_mass', val=np.zeros(nFull-1), units='kg', desc='mass of permanent ballast')

        self.add_input('buoyancy_tank_cg', val=0.0, units='m', desc='z-coordinate of center of mass for buoyancy tank')
        self.add_input('ballast_z_cg', val=0.0, units='m', desc='z-coordinate or permanent ballast center of gravity')
        self.add_input('column_mass_factor', val=0.0, desc='Overall column mass correction factor')
        self.add_input('outfitting_mass_fraction', val=0.0, desc='Mass fraction added for outfitting')

        # Moments of inertia
        self.add_input('shell_I_keel', val=np.zeros(6), units='kg*m**2', desc='Moments of inertia of outer shell relative to keel point')
        self.add_input('bulkhead_I_keel', val=np.zeros(6), units='kg*m**2', desc='Moments of inertia of bulkheads relative to keel point')
        self.add_input('stiffener_I_keel', val=np.zeros(6), units='kg*m**2', desc='Moments of inertia of stiffeners relative to keel point')
        self.add_input('buoyancy_tank_I_keel', val=np.zeros(6), units='kg*m**2', desc='Moments of inertia of heave plate relative to keel point')
        self.add_input('ballast_I_keel', val=np.zeros(6), units='kg*m**2', desc='Moments of inertia of permanent ballast relative to keel point')

        # For buoyancy
        self.add_input('buoyancy_tank_displacement', val=0.0, units='m**3', desc='volume of water displaced by buoyancy tank')
        
        # Costs and cost rates
        self.add_input('shell_cost', val=0.0, units='USD', desc='mass of column shell')
        self.add_input('stiffener_cost', val=0.0, units='USD', desc='mass of column stiffeners')
        self.add_input('bulkhead_cost', val=0.0, units='USD', desc='mass of column bulkheads')
        self.add_input('ballast_cost', val=0.0, units='USD', desc='cost of permanent ballast')
        self.add_input('buoyancy_tank_cost', val=0.0, units='USD', desc='mass of heave plate')
        self.add_input('material_cost_rate', 0.0, units='USD/kg', desc='Raw material cost: steel $1.1/kg, aluminum $3.5/kg')
        self.add_input('outfitting_cost_rate', val=0.0, units='USD/kg', desc='Cost per unit mass for outfitting column')

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
        self.declare_partials('*', '*', method='fd', form='central', step=1e-6)
        
        
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
        out_frac     = inputs['outfitting_mass_fraction']
        coeff        = inputs['column_mass_factor']
        z_nodes      = inputs['z_full']
        z_section    = inputs['z_section']
        z_box        = inputs['buoyancy_tank_cg']
        z_ballast    = inputs['ballast_z_cg']
        m_shell      = inputs['shell_mass']
        m_stiffener  = inputs['stiffener_mass']
        m_bulkhead   = inputs['bulkhead_mass']
        m_box        = inputs['buoyancy_tank_mass']
        m_ballast    = inputs['ballast_mass']
        I_shell      = inputs['shell_I_keel']
        I_stiffener  = inputs['stiffener_I_keel']
        I_bulkhead   = inputs['bulkhead_I_keel']
        I_box        = inputs['buoyancy_tank_I_keel']
        I_ballast    = inputs['ballast_I_keel']

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
        outputs['column_total_mass']      = m_sections
        outputs['column_structural_mass'] = m_column + m_outfit
        outputs['column_outfitting_mass'] = m_outfit
        outputs['z_center_of_mass']       = z_cg
        outputs['I_column']               = I_total
        

        
    def balance_column(self, inputs, outputs):
        # Unpack variables
        R_od              = 0.5*inputs['d_full']
        R_plate           = 0.5*inputs['buoyancy_tank_diameter']
        z_nodes           = inputs['z_full']
        z_box             = inputs['buoyancy_tank_cg']
        V_box             = inputs['buoyancy_tank_displacement']
        rho_water         = inputs['water_density']
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
        outputs['displaced_volume'] = np.r_[V_under, np.zeros(add0)]
        outputs['displaced_volume'][self.ibox] += V_box

        # Compute Center of Buoyancy in z-coordinates (0=waterline)
        # First get z-coordinates of CG of all frustums
        z_cg_under  = frustum.frustumCG(r_under[:-1], r_under[1:], np.diff(z_under))
        z_cg_under += z_under[:-1]
        # Now take weighted average of these CG points with volume
        z_cb     = ( (V_box*z_box) + np.dot(V_under, z_cg_under)) / (outputs['displaced_volume'].sum() + eps)
        outputs['z_center_of_buoyancy'] = z_cb

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
        outputs['hydrostatic_force'] = np.r_[F_hydro, np.zeros(add0)]
        
        # 2nd moment of area for circular cross section
        # Note: Assuming Iwater here depends on "water displacement" cross-section
        # and not actual moment of inertia type of cross section (thin hoop)
        outputs['Iwater'] = 0.25 * np.pi * r_waterline**4.0
        outputs['Awater'] = np.pi * r_waterline**2.0

        # Calculate diagonal entries of added mass matrix
        # Prep for integrals too
        npts     = 100 * R_od.size
        zpts     = np.linspace(z_under[0], z_under[-1], npts)
        r_under  = np.interp(zpts, z_under, r_under)
        m_a      = np.zeros(6)
        m_a[:2]  = rho_water * outputs['displaced_volume'].sum() # A11 surge, A22 sway
        m_a[2]   = 0.5 * (8.0/3.0) * rho_water * np.maximum(R_plate, r_under.max())**3.0# A33 heave
        m_a[3:5] = np.pi * rho_water * np.trapz((zpts-z_cb)**2.0 * r_under**2.0, zpts)# A44 roll, A55 pitch
        m_a[5]   = 0.0 # A66 yaw
        outputs['column_added_mass'] = m_a

        
    def compute_cost(self, inputs, outputs):
        outputs['column_structural_cost']   = inputs['column_mass_factor']*(inputs['shell_cost'] + inputs['stiffener_cost'] +
                                                                             inputs['bulkhead_cost'] + inputs['buoyancy_tank_cost'])
        outputs['column_outfitting_cost']   = inputs['outfitting_cost_rate'] * outputs['column_outfitting_mass']
        outputs['column_total_cost']        = outputs['column_structural_cost'] + outputs['column_outfitting_cost'] + inputs['ballast_cost']
        outputs['tapered_column_cost_rate'] = 1e3*outputs['column_total_cost']/outputs['column_total_mass'].sum()

        
class ColumnBuckling(ExplicitComponent):
    '''
    This function computes the applied axial and hoop stresses in a column and compares that to 
    limits established by the API standard.  Some physcial geometry checks are also performed.
    '''

    def initialize(self):
        self.options.declare('nSection')
        self.options.declare('nFull')
        
    def setup(self):
        nSection = self.options['nSection']
        nFull    = self.options['nFull']

        # From other modules
        self.add_input('stack_mass_in', val=eps, units='kg', desc='Weight above the cylinder column')
        self.add_input('section_mass', val=np.zeros((nFull-1,)), units='kg', desc='total mass of column by section')
        self.add_input('pressure', np.zeros(nFull), units='N/m**2', desc='Dynamic (and static)? pressure')
        
        self.add_input('d_full', np.zeros(nFull), units='m', desc='cylinder diameter at corresponding locations')
        self.add_input('t_full', np.zeros(nFull-1), units='m', desc='shell thickness at corresponding locations')
        self.add_input('z_full', val=np.zeros(nFull), units='m', desc='z-coordinates of section nodes (length = nsection+1)')

        self.add_input('h_web', val=np.zeros((nFull-1,)), units='m', desc='height of stiffener web (base of T) within each section bottom to top')
        self.add_input('t_web', val=np.zeros((nFull-1,)), units='m', desc='thickness of stiffener web (base of T) within each section bottom to top')
        self.add_input('w_flange', val=np.zeros((nFull-1,)), units='m', desc='height of stiffener flange (top of T) within each section bottom to top')
        self.add_input('t_flange', val=np.zeros((nFull-1,)), units='m', desc='thickness of stiffener flange (top of T) within each section bottom to top')
        self.add_input('L_stiffener', val=np.zeros((nFull-1,)), units='m', desc='Axial distance from one ring stiffener to another within each section bottom to top')

        self.add_input('E', val=0.0, units='Pa', desc='Modulus of elasticity (Youngs) of material')
        self.add_input('nu', val=0.0, desc='poissons ratio of column material')
        self.add_input('yield_stress', val=0.0, units='Pa', desc='yield stress of material')

        self.add_discrete_input('loading', val='hydro', desc='Loading type in API checks [hydro/radial]')
        self.add_input('gamma_f', 0.0, desc='safety factor on loads')
        self.add_input('gamma_b', 0.0, desc='buckling safety factor')
        
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
        self.declare_partials('*', '*', method='fd', form='central', step=1e-6)

        
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
        R_od,_         = nodal2sectional(inputs['d_full'])
        R_od          *= 0.5
        t_wall         = inputs['t_full']
        section_mass   = inputs['section_mass']
        m_stack        = inputs['stack_mass_in']
        
        # Middle radius
        R_m = R_od - 0.5*t_wall
        # Add in weight of sections above it
        axial_load = m_stack + np.r_[0.0, np.cumsum(section_mass[:-1])]
        # Divide by shell cross sectional area to get stress
        return (gravity * axial_load / (2.0 * np.pi * R_m * t_wall))

    
    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # Unpack variables
        R_od,_       = nodal2sectional( inputs['d_full'] )
        R_od        *= 0.5
        h_section    = np.diff( inputs['z_full'] )
        t_wall       = inputs['t_full']
        
        t_web        = inputs['t_web']
        t_flange     = inputs['t_flange']
        h_web        = inputs['h_web']
        w_flange     = inputs['w_flange']
        L_stiffener  = inputs['L_stiffener']

        gamma_f      = inputs['gamma_f']
        gamma_b      = inputs['gamma_b']
        
        E            = inputs['E'] # Young's modulus
        nu           = inputs['nu'] # Poisson ratio
        sigma_y      = inputs['yield_stress']
        loading      = discrete_inputs['loading']
        nodalP,_     = nodal2sectional( inputs['pressure'] )
        pressure     = 1e-12 if loading in ['ax','axial','testing','test'] else nodalP+1e-12

        # Apply quick "compactness" check on stiffener geometry
        # Constraint is that these must be >= 1
        flange_compactness = 0.375 * (t_flange / (0.5*w_flange)) * np.sqrt(E / sigma_y)
        web_compactness    = 1.0   * (t_web    / h_web         ) * np.sqrt(E / sigma_y)

        # Compute applied axial stress simply, like API guidelines (as opposed to running frame3dd)
        sigma_ax = self.compute_applied_axial(inputs)
        (axial_local_api, axial_general_api, external_local_api, external_general_api,
         axial_local_raw, axial_general_raw, external_local_raw, external_general_raw) = shellBuckling_withStiffeners(
             pressure, sigma_ax, R_od, t_wall, h_section,
             h_web, t_web, w_flange, t_flange,
             L_stiffener, E, nu, sigma_y, loading)
        
        outputs['flange_compactness']     = flange_compactness
        outputs['web_compactness']        = web_compactness
        
        outputs['axial_local_api']      = axial_local_api
        outputs['axial_general_api']    = axial_general_api
        outputs['external_local_api']   = external_local_api
        outputs['external_general_api'] = external_general_api

        outputs['axial_local_utilization']      = axial_local_raw * gamma_f*gamma_b
        outputs['axial_general_utilization']    = axial_general_raw * gamma_f*gamma_b
        outputs['external_local_utilization']   = external_local_raw * gamma_f*gamma_b
        outputs['external_general_utilization'] = external_general_raw * gamma_f*gamma_b


class Column(Group):

    def initialize(self):
        self.options.declare('nSection')
        self.options.declare('nFull')
        self.options.declare('topLevelFlag', default=False)
        
    def setup(self):
        nSection = self.options['nSection']
        nFull    = self.options['nFull']
        nRefine  = (nFull-1)/nSection
        topLevelFlag = self.options['topLevelFlag']

        columnIndeps = IndepVarComp()
        columnIndeps.add_output('freeboard', 0.0, units='m')
        columnIndeps.add_output('section_height', np.zeros((nSection,)), units='m')
        columnIndeps.add_output('outer_diameter', np.zeros((nSection+1,)), units='m')
        columnIndeps.add_output('wall_thickness', np.zeros((nSection,)), units='m')
        columnIndeps.add_output('stiffener_web_height', np.zeros((nSection,)), units='m')
        columnIndeps.add_output('stiffener_web_thickness', np.zeros((nSection,)), units='m')
        columnIndeps.add_output('stiffener_flange_width', np.zeros((nSection,)), units='m')
        columnIndeps.add_output('stiffener_flange_thickness', np.zeros((nSection,)), units='m')
        columnIndeps.add_output('stiffener_spacing', np.zeros((nSection,)), units='m')
        columnIndeps.add_output('bulkhead_thickness', np.zeros((nSection+1,)), units='m')
        columnIndeps.add_output('permanent_ballast_height', 0.0, units='m')
        columnIndeps.add_output('buoyancy_tank_diameter', 0.0, units='m')
        columnIndeps.add_output('buoyancy_tank_height', 0.0, units='m')
        columnIndeps.add_output('buoyancy_tank_location', 0.0, units='m')
        columnIndeps.add_output('material_density', 0.0, units='kg/m**3')
        self.add_subsystem('columnIndeps', columnIndeps, promotes=['*'])

        if topLevelFlag:
            sharedIndeps = IndepVarComp()
            sharedIndeps.add_output('water_density', 0.0, units='kg/m**3')
            sharedIndeps.add_output('water_depth', 0.0, units='m')
            sharedIndeps.add_output('wave_beta', 0.0, units='deg')
            sharedIndeps.add_output('wave_z0', 0.0, units='m')
            sharedIndeps.add_output('significant_wave_height', 0.0, units='m')
            self.add_subsystem('sharedIndeps', sharedIndeps, promotes=['*'])
            
        
        self.add_subsystem('cyl_geom', CylinderDiscretization(nPoints=nSection+1, nRefine=nRefine), promotes=['section_height','diameter','wall_thickness',
                                                                                    'd_full','t_full','foundation_height'])
        
        self.add_subsystem('gc', GeometricConstraints(nPoints=nSection+1, diamFlag=True), promotes=['max_taper','min_d_to_t','manufacturability','weldability'])

        self.add_subsystem('cyl_mass', CylinderMass(nPoints=nFull), promotes=['d_full','t_full','material_density',
                                                            'material_cost_rate','labor_cost_rate','painting_cost_rate',
                                                            'section_center_of_mass'])

        self.add_subsystem('col_geom', ColumnGeometry(nSection=nSection, nFull=nFull), promotes=['*'])

        self.add_subsystem('bulk', BulkheadProperties(nSection=nSection, nFull=nFull), promotes=['*'])

        self.add_subsystem('stiff', StiffenerProperties(nSection=nSection, nFull=nFull), promotes=['*'])

        self.add_subsystem('plate', BuoyancyTankProperties(nFull=nFull), promotes=['*'])

        self.add_subsystem('ball', BallastProperties(nFull=nFull), promotes=['*'])

        self.add_subsystem('col', ColumnProperties(nFull=nFull), promotes=['*'])

        self.add_subsystem('wind', PowerWind(nPoints=nFull), promotes=['Uref','zref','shearExp','z0'])
        self.add_subsystem('wave', LinearWaves(nPoints=nFull), promotes=['Uc','hmax','T'])
        self.add_subsystem('windLoads', CylinderWindDrag(nPoints=nFull), promotes=['cd_usr','beta'])
        self.add_subsystem('waveLoads', CylinderWaveDrag(nPoints=nFull), promotes=['cm','cd_usr'])
        self.add_subsystem('distLoads', AeroHydroLoads(nPoints=nFull), promotes=['Px','Py','Pz','qdyn','yaw'])

        self.add_subsystem('buck', ColumnBuckling(nSection=nSection, nFull=nFull), promotes=['*'])

        self.connect('outer_diameter', ['diameter', 'gc.d'])
        self.connect('wall_thickness', 'gc.t')
        self.connect('cyl_geom.z_param', 'z_param_in')
        self.connect('cyl_geom.z_full', ['cyl_mass.z_full','z_full_in'])
        
        #self.connect('cyl_mass.section_center_of_mass', 'col_geom.section_center_of_mass')
        
        self.connect('cyl_mass.mass', 'shell_mass')
        self.connect('cyl_mass.cost', 'shell_cost')
        self.connect('cyl_mass.I_base', 'shell_I_keel')
        self.connect('material_density','rho')
        
        self.connect('column_total_mass', 'section_mass')

        if topLevelFlag:
            self.connect('water_depth','wave.z_floor')
            self.connect('wave_beta','waveLoads.beta')
            self.connect('wave_z0', 'wave.z_surface')
            self.connect('significant_wave_height',['Hs', 'hmax'])
            self.connect('water_density',['wave.rho','waveLoads.rho'])
        self.connect('z_full', ['wind.z', 'wave.z', 'windLoads.z','waveLoads.z','distLoads.z'])
        self.connect('d_full', ['windLoads.d','waveLoads.d'])

        self.connect('wind.U', 'windLoads.U')

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
