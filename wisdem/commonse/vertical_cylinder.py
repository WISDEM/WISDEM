
import numpy as np
from openmdao.api import ExplicitComponent
from wisdem.commonse.tube import CylindricalShellProperties

from wisdem.commonse import gravity, eps
import wisdem.commonse.frustum as frustum
import wisdem.commonse.manufacturing as manufacture
from wisdem.commonse.UtilizationSupplement import hoopStressEurocode, hoopStress
from wisdem.commonse.utilities import assembleI, unassembleI, sectionalInterp, nodal2sectional
import wisdem.pyframe3dd.frame3dd as frame3dd

RIGID = 1e30

# -----------------
#  Components
# -----------------

#TODO need to check the length of each array
class CylinderDiscretization(ExplicitComponent):
    """discretize geometry into finite element nodes"""

    def initialize(self):
        self.options.declare('nPoints')
        self.options.declare('nRefine')

    def setup(self):
        nPoints = self.options['nPoints']
        nRefine = np.round( self.options['nRefine'] )
        nFull   = int(nRefine * (nPoints-1) + 1)
        
        # variables
        self.add_input('foundation_height', val=0.0, units='m', desc='starting height of tower') 
        self.add_input('section_height', np.zeros(nPoints-1), units='m', desc='parameterized section heights along cylinder')
        self.add_input('diameter', np.zeros(nPoints), units='m', desc='cylinder diameter at corresponding locations')
        self.add_input('wall_thickness', np.zeros(nPoints-1), units='m', desc='shell thickness at corresponding locations')

        #out
        self.add_output('z_param', np.zeros(nPoints), units='m', desc='parameterized locations along cylinder, linear lofting between')
        self.add_output('z_full', np.zeros(nFull), units='m', desc='locations along cylinder')
        self.add_output('d_full', np.zeros(nFull), units='m', desc='cylinder diameter at corresponding locations')
        self.add_output('t_full', np.zeros(nFull-1), units='m', desc='shell thickness at corresponding locations')
        # Convenience outputs for export to other modules
        
        # Derivatives
        # self.declare_partials('*', '*', method='fd', form='central', step=1e-6)

    def compute(self, inputs, outputs):
        nRefine = int(np.round( self.options['nRefine'] ))
        z_param = float(inputs['foundation_height']) + np.r_[0.0, np.cumsum(inputs['section_height'].flatten())]
        # Have to regine each element one at a time so that we preserve input nodes
        z_full = np.array([])
        for k in range(z_param.size-1):
            zref = np.linspace(z_param[k], z_param[k+1], nRefine+1)
            z_full = np.append(z_full, zref)
        z_full = np.unique(z_full)
        outputs['z_full']  = z_full
        outputs['d_full']  = np.interp(z_full, z_param, inputs['diameter'])
        z_section = 0.5*(z_full[:-1] + z_full[1:])
        outputs['t_full']  = sectionalInterp(z_section, z_param, inputs['wall_thickness'])
        outputs['z_param'] = z_param

class CylinderMass(ExplicitComponent):

    def initialize(self):
        self.options.declare('nPoints')
        
    def setup(self):
        nPoints = self.options['nPoints']
        
        self.add_input('d_full', val=np.zeros(nPoints), units='m', desc='cylinder diameter at corresponding locations')
        self.add_input('t_full', val=np.zeros(nPoints-1), units='m', desc='shell thickness at corresponding locations')
        self.add_input('z_full', val=np.zeros(nPoints), units='m', desc='parameterized locations along cylinder, linear lofting between')
        self.add_input('material_density', 0.0, units='kg/m**3', desc='material density')
        self.add_input('outfitting_factor', val=0.0, desc='Multiplier that accounts for secondary structure mass inside of cylinder')
        
        self.add_input('material_cost_rate', 0.0, units='USD/kg', desc='Raw material cost rate: steel $1.1/kg, aluminum $3.5/kg')
        self.add_input('labor_cost_rate', 0.0, units='USD/min', desc='Labor cost rate')
        self.add_input('painting_cost_rate', 0.0, units='USD/m/m', desc='Painting / surface finishing cost rate')
        
        self.add_output('cost', val=0.0, units='USD', desc='Total cylinder cost')
        self.add_output('mass', val=np.zeros(nPoints-1), units='kg', desc='Total cylinder mass')
        self.add_output('center_of_mass', val=0.0, units='m', desc='z-position of center of mass of cylinder')
        self.add_output('section_center_of_mass', val=np.zeros(nPoints-1), units='m', desc='z position of center of mass of each can in the cylinder')
        self.add_output('I_base', np.zeros((6,)), units='kg*m**2', desc='mass moment of inertia of cylinder about base [xx yy zz xy xz yz]')
        
        # Derivatives
        # self.declare_partials('*', '*', method='fd', form='central', step=1e-6)
        
    def compute(self, inputs, outputs):
        # Unpack variables for thickness and average radius at each can interface
        twall = inputs['t_full']
        Rb    = 0.5*inputs['d_full'][:-1]
        Rt    = 0.5*inputs['d_full'][1:]
        zz    = inputs['z_full']
        H     = np.diff(zz)
        rho   = inputs['material_density']
        coeff = inputs['outfitting_factor']
        if coeff < 1.0: coeff += 1.0

        # Total mass of cylinder
        V_shell = frustum.frustumShellVol(Rb, Rt, twall, H)
        outputs['mass'] = coeff * rho * V_shell
        
        # Center of mass of each can/section
        cm_section = zz[:-1] + frustum.frustumShellCG(Rb, Rt, twall, H)
        outputs['section_center_of_mass'] = cm_section

        # Center of mass of cylinder
        V_shell += eps
        outputs['center_of_mass'] = np.dot(V_shell, cm_section) / V_shell.sum()

        # Moments of inertia
        Izz_section = frustum.frustumShellIzz(Rb, Rt, twall, H)
        Ixx_section = Iyy_section = frustum.frustumShellIxx(Rb, Rt, twall, H)

        # Sum up each cylinder section using parallel axis theorem
        I_base = np.zeros((3,3))
        for k in range(Izz_section.size):
            R = np.array([0.0, 0.0, cm_section[k]])
            Icg = assembleI( [Ixx_section[k], Iyy_section[k], Izz_section[k], 0.0, 0.0, 0.0] )

            I_base += Icg + V_shell[k]*(np.dot(R, R)*np.eye(3) - np.outer(R, R))
            
        # All of the mass and volume terms need to be multiplied by density
        I_base *= coeff * rho

        outputs['I_base'] = unassembleI(I_base)
        

        # Compute costs based on "Optimum Design of Steel Structures" by Farkas and Jarmai
        # All dimensions for correlations based on mm, not meters.
        R_ave  = 0.5*(Rb + Rt)
        taper  = np.minimum(Rb/Rt, Rt/Rb)
        nsec   = twall.size
        mplate = rho * V_shell.sum()
        k_m    = inputs['material_cost_rate'] #1.1 # USD / kg carbon steel plate
        k_f    = inputs['labor_cost_rate'] #1.0 # USD / min labor
        k_p    = inputs['painting_cost_rate'] #USD / m^2 painting
        
        # Cost Step 1) Cutting flat plates for taper using plasma cutter
        cutLengths = 2.0 * np.sqrt( (Rt-Rb)**2.0 + H**2.0 ) # Factor of 2 for both sides
        # Cost Step 2) Rolling plates 
        # Cost Step 3) Welding rolled plates into shells (set difficulty factor based on tapering with logistic function)
        theta_F = 4.0 - 3.0 / (1 + np.exp(-5.0*(taper-0.75)))
        # Cost Step 4) Circumferential welds to join cans together
        theta_A = 2.0

        # Labor-based expenses
        K_f = k_f * ( manufacture.steel_cutting_plasma_time(cutLengths, twall) +
                      manufacture.steel_rolling_time(theta_F, R_ave, twall) +
                      manufacture.steel_butt_welding_time(theta_A, nsec, mplate, cutLengths, twall) +
                      manufacture.steel_butt_welding_time(theta_A, nsec, mplate, 2*np.pi*Rb[1:], twall[1:]) )
        
        # Cost step 5) Painting- outside and inside
        theta_p = 2
        K_p  = k_p * theta_p * 2 * (2 * np.pi * R_ave * H).sum()

        # Cost step 6) Outfitting
        K_o = 1.5 * k_m * (coeff - 1.0) * mplate
        
        # Material cost, without outfitting
        K_m = k_m * mplate

        # Assemble all costs
        outputs['cost'] = K_m + K_o + K_p + K_f

        

#@implement_base(CylinderFromCSProps)
class CylinderFrame3DD(ExplicitComponent):
    def initialize(self):
        self.options.declare('npts')
        self.options.declare('nK')
        self.options.declare('nMass')
        self.options.declare('nPL')
        
    def setup(self):
        npts  = self.options['npts']
        nK    = self.options['nK']
        nMass = self.options['nMass']
        nPL   = self.options['nPL']

        # cross-sectional data along cylinder.
        self.add_input('z', np.zeros(npts), units='m', desc='location along cylinder. start at bottom and go to top')
        self.add_input('Az', np.zeros(npts-1), units='m**2', desc='cross-sectional area')
        self.add_input('Asx', np.zeros(npts-1), units='m**2', desc='x shear area')
        self.add_input('Asy', np.zeros(npts-1), units='m**2', desc='y shear area')
        self.add_input('Jz', np.zeros(npts-1), units='m**4', desc='polar moment of inertia')
        self.add_input('Ixx', np.zeros(npts-1), units='m**4', desc='area moment of inertia about x-axis')
        self.add_input('Iyy', np.zeros(npts-1), units='m**4', desc='area moment of inertia about y-axis')

        self.add_input('E', val=0.0, units='N/m**2', desc='modulus of elasticity')
        self.add_input('G', val=0.0, units='N/m**2', desc='shear modulus')
        self.add_input('rho', val=0.0, units='kg/m**3', desc='material density')
        self.add_input('sigma_y', val=0.0, units='N/m**2', desc='yield stress')
        self.add_input('L_reinforced', val=0.0, units='m')

        # effective geometry -- used for handbook methods to estimate hoop stress, buckling, fatigue
        self.add_input('d', np.zeros(npts), units='m', desc='effective cylinder diameter for section')
        self.add_input('t', np.zeros(npts-1), units='m', desc='effective shell thickness for section')

        # spring reaction data.  Use global RIGID for rigid constraints.
        self.add_input('kidx', np.zeros(nK, dtype=np.int_), desc='indices of z where external stiffness reactions should be applied.')
        self.add_input('kx', np.zeros(nK), units='N/m', desc='spring stiffness in x-direction')
        self.add_input('ky', np.zeros(nK), units='N/m', desc='spring stiffness in y-direction')
        self.add_input('kz', np.zeros(nK), units='N/m', desc='spring stiffness in z-direction')
        self.add_input('ktx', np.zeros(nK), units='N/m', desc='spring stiffness in theta_x-rotation')
        self.add_input('kty', np.zeros(nK), units='N/m', desc='spring stiffness in theta_y-rotation')
        self.add_input('ktz', np.zeros(nK), units='N/m', desc='spring stiffness in theta_z-rotation')

        # extra mass
        self.add_input('midx', np.zeros(nMass, dtype=np.int_), desc='indices where added mass should be applied.')
        self.add_input('m', np.zeros(nMass), units='kg', desc='added mass')
        self.add_input('mIxx', np.zeros(nMass), units='kg*m**2', desc='x mass moment of inertia about some point p')
        self.add_input('mIyy', np.zeros(nMass), units='kg*m**2', desc='y mass moment of inertia about some point p')
        self.add_input('mIzz', np.zeros(nMass), units='kg*m**2', desc='z mass moment of inertia about some point p')
        self.add_input('mIxy', np.zeros(nMass), units='kg*m**2', desc='xy mass moment of inertia about some point p')
        self.add_input('mIxz', np.zeros(nMass), units='kg*m**2', desc='xz mass moment of inertia about some point p')
        self.add_input('mIyz', np.zeros(nMass), units='kg*m**2', desc='yz mass moment of inertia about some point p')
        self.add_input('mrhox', np.zeros(nMass), units='m', desc='x-location of p relative to node')
        self.add_input('mrhoy', np.zeros(nMass), units='m', desc='y-location of p relative to node')
        self.add_input('mrhoz', np.zeros(nMass), units='m', desc='z-location of p relative to node')
        self.add_discrete_input('addGravityLoadForExtraMass', True, desc='add gravitational load')

        # point loads (if addGravityLoadForExtraMass=True be sure not to double count by adding those force here also)
        self.add_input('plidx', np.zeros(nPL, dtype=np.int_), desc='indices where point loads should be applied.')
        self.add_input('Fx', np.zeros(nPL), units='N', desc='point force in x-direction')
        self.add_input('Fy', np.zeros(nPL), units='N', desc='point force in y-direction')
        self.add_input('Fz', np.zeros(nPL), units='N', desc='point force in z-direction')
        self.add_input('Mxx', np.zeros(nPL), units='N*m', desc='point moment about x-axis')
        self.add_input('Myy', np.zeros(nPL), units='N*m', desc='point moment about y-axis')
        self.add_input('Mzz', np.zeros(nPL), units='N*m', desc='point moment about z-axis')

        # combined wind-water distributed loads
        self.add_input('Px', np.zeros(npts), units='N/m', desc='force per unit length in x-direction')
        self.add_input('Py', np.zeros(npts), units='N/m', desc='force per unit length in y-direction')
        self.add_input('Pz', np.zeros(npts), units='N/m', desc='force per unit length in z-direction')
        self.add_input('qdyn', np.zeros(npts), units='N/m**2', desc='dynamic pressure')

        # options
        self.add_discrete_input('shear', True, desc='include shear deformation')
        self.add_discrete_input('geom', False, desc='include geometric stiffness')
        self.add_input('dx', 5.0, desc='z-axis increment for internal forces')
        self.add_discrete_input('nM', 2, desc='number of desired dynamic modes of vibration (below only necessary if nM > 0)')
        self.add_discrete_input('Mmethod', 1, desc='1: subspace Jacobi, 2: Stodola')
        self.add_discrete_input('lump', 0, desc='0: consistent mass, 1: lumped mass matrix')
        self.add_input('tol', 1e-9, desc='mode shape tolerance')
        self.add_input('shift', 0.0, desc='shift value ... for unrestrained structures')

        # outputs
        self.add_output('mass', 0.0)
        self.add_output('f1', 0.0, units='Hz', desc='First natural frequency')
        self.add_output('f2', 0.0, units='Hz', desc='Second natural frequency')
        self.add_output('top_deflection', 0.0, units='m', desc='Deflection of cylinder top in yaw-aligned +x direction')
        self.add_output('Fz_out', np.zeros(npts-1), units='N', desc='Axial foce in vertical z-direction in cylinder structure.')
        self.add_output('Vx_out', np.zeros(npts-1), units='N', desc='Shear force in x-direction in cylinder structure.')
        self.add_output('Vy_out', np.zeros(npts-1), units='N', desc='Shear force in y-direction in cylinder structure.')
        self.add_output('Mxx_out', np.zeros(npts-1), units='N*m', desc='Moment about x-axis in cylinder structure.')
        self.add_output('Myy_out', np.zeros(npts-1), units='N*m', desc='Moment about y-axis in cylinder structure.')
        self.add_output('Mzz_out', np.zeros(npts-1), units='N*m', desc='Moment about z-axis in cylinder structure.')
        self.add_output('base_F', val=np.zeros(3), units='N', desc='Total force on cylinder')
        self.add_output('base_M', val=np.zeros(3), units='N*m', desc='Total moment on cylinder measured at base')

        self.add_output('axial_stress', np.zeros(npts-1), units='N/m**2', desc='Axial stress in cylinder structure')
        self.add_output('shear_stress', np.zeros(npts-1), units='N/m**2', desc='Shear stress in cylinder structure')
        self.add_output('hoop_stress', np.zeros(npts-1), units='N/m**2', desc='Hoop stress in cylinder structure calculated with simple method used in API standards')
        self.add_output('hoop_stress_euro', np.zeros(npts-1), units='N/m**2', desc='Hoop stress in cylinder structure calculated with Eurocode method')
        
        # Derivatives
        # self.declare_partials('*', '*', method='fd', form='central', step=1e-6)

        
    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        # ------- node data ----------------
        z = inputs['z']
        n = len(z)
        node = np.arange(1, n+1)
        x = np.zeros(n)
        y = np.zeros(n)
        r = np.zeros(n)

        nodes = frame3dd.NodeData(node, x, y, z, r)
        # -----------------------------------

        # ------ reaction data ------------

        # rigid base
        node = inputs['kidx'] + np.ones(len(inputs['kidx']))   # add one because 0-based index but 1-based node numbering
        rigid = RIGID

        reactions = frame3dd.ReactionData(node, inputs['kx'], inputs['ky'], inputs['kz'], inputs['ktx'], inputs['kty'], inputs['ktz'], rigid)
        # -----------------------------------

        # ------ frame element data ------------
        element = np.arange(1, n)
        N1 = np.arange(1, n)
        N2 = np.arange(2, n+1)

        roll = np.zeros(n-1)

        # average across element b.c. frame3dd uses constant section elements
        # TODO: Use nodal2sectional
        Az  = inputs['Az']
        Asx = inputs['Asx']
        Asy = inputs['Asy']
        Jz  = inputs['Jz']
        Ixx = inputs['Ixx']
        Iyy = inputs['Iyy']
        E   = inputs['E']*np.ones(Az.shape)
        G   = inputs['G']*np.ones(Az.shape)
        rho = inputs['rho']*np.ones(Az.shape)

        elements = frame3dd.ElementData(element, N1, N2, Az, Asx, Asy, Jz, Ixx, Iyy, E, G, roll, rho)
        # -----------------------------------


        # ------ options ------------
        options = frame3dd.Options(discrete_inputs['shear'], discrete_inputs['geom'], float(inputs['dx']))
        # -----------------------------------

        # initialize frame3dd object
        cylinder = frame3dd.Frame(nodes, reactions, elements, options)


        # ------ add extra mass ------------

        # extra node inertia data
        N = inputs['midx'] + np.ones(len(inputs['midx']))

        cylinder.changeExtraNodeMass(N, inputs['m'], inputs['mIxx'], inputs['mIyy'], inputs['mIzz'], inputs['mIxy'], inputs['mIxz'], inputs['mIyz'],
            inputs['mrhox'], inputs['mrhoy'], inputs['mrhoz'], discrete_inputs['addGravityLoadForExtraMass'])

        # ------------------------------------

        # ------- enable dynamic analysis ----------
        cylinder.enableDynamics(discrete_inputs['nM'], discrete_inputs['Mmethod'], discrete_inputs['lump'], float(inputs['tol']), float(inputs['shift']))
        # ----------------------------

        # ------ static load case 1 ------------

        # gravity in the X, Y, Z, directions (global)
        gx = 0.0
        gy = 0.0
        gz = -gravity

        load = frame3dd.StaticLoadCase(gx, gy, gz)

        # point loads
        nF = inputs['plidx'] + np.ones(len(inputs['plidx']))
        load.changePointLoads(nF, inputs['Fx'], inputs['Fy'], inputs['Fz'], inputs['Mxx'], inputs['Myy'], inputs['Mzz'])

        # distributed loads
        Px, Py, Pz = inputs['Pz'], inputs['Py'], -inputs['Px']  # switch to local c.s.
        z = inputs['z']

        # trapezoidally distributed loads
        EL = np.arange(1, n)
        xx1 = xy1 = xz1 = np.zeros(n-1)
        xx2 = xy2 = xz2 = np.diff(z) - 1e-6  # subtract small number b.c. of precision
        wx1 = Px[:-1]
        wx2 = Px[1:]
        wy1 = Py[:-1]
        wy2 = Py[1:]
        wz1 = Pz[:-1]
        wz2 = Pz[1:]

        load.changeTrapezoidalLoads(EL, xx1, xx2, wx1, wx2, xy1, xy2, wy1, wy2, xz1, xz2, wz1, wz2)

        cylinder.addLoadCase(load)
        # Debugging
        #cylinder.write('temp.3dd')
        # -----------------------------------
        # run the analysis
        displacements, forces, reactions, internalForces, mass, modal = cylinder.run()
        iCase = 0

        # mass
        outputs['mass'] = mass.struct_mass

        # natural frequncies
        outputs['f1'] = modal.freq[0]
        outputs['f2'] = modal.freq[1]

        # deflections due to loading (from cylinder top and wind/wave loads)
        outputs['top_deflection'] = displacements.dx[iCase, n-1]  # in yaw-aligned direction

        # shear and bending, one per element (convert from local to global c.s.)
        Fz = forces.Nx[iCase, 1::2]
        Vy = forces.Vy[iCase, 1::2]
        Vx = -forces.Vz[iCase, 1::2]

        Mzz = forces.Txx[iCase, 1::2]
        Myy = forces.Myy[iCase, 1::2]
        Mxx = -forces.Mzz[iCase, 1::2]

        # Record total forces and moments
        outputs['base_F'] = -1.0 * np.array([reactions.Fx.sum(), reactions.Fy.sum(), reactions.Fz.sum()])
        outputs['base_M'] = -1.0 * np.array([reactions.Mxx.sum(), reactions.Myy.sum(), reactions.Mzz.sum()])

        outputs['Fz_out']  = Fz
        outputs['Vx_out']  = Vx
        outputs['Vy_out']  = Vy
        outputs['Mxx_out'] = Mxx
        outputs['Myy_out'] = Myy
        outputs['Mzz_out'] = Mzz

        # axial and shear stress
        d,_    = nodal2sectional(inputs['d'])
        qdyn,_ = nodal2sectional(inputs['qdyn'])
        
        ##R = self.d/2.0
        ##x_stress = R*np.cos(self.theta_stress)
        ##y_stress = R*np.sin(self.theta_stress)
        ##axial_stress = Fz/self.Az + Mxx/self.Ixx*y_stress - Myy/self.Iyy*x_stress
#        V = Vy*x_stress/R - Vx*y_stress/R  # shear stress orthogonal to direction x,y
#        shear_stress = 2. * V / self.Az  # coefficient of 2 for a hollow circular section, but should be conservative for other shapes
        outputs['axial_stress'] = Fz/inputs['Az'] - np.sqrt(Mxx**2+Myy**2)/inputs['Iyy']*d/2.0  #More conservative, just use the tilted bending and add total max shear as well at the same point, if you do not like it go back to the previous lines

        outputs['shear_stress'] = 2. * np.sqrt(Vx**2+Vy**2) / inputs['Az'] # coefficient of 2 for a hollow circular section, but should be conservative for other shapes

        # hoop_stress (Eurocode method)
        L_reinforced = inputs['L_reinforced'] * np.ones(Fz.shape)
        outputs['hoop_stress_euro'] = hoopStressEurocode(inputs['z'], d, inputs['t'], L_reinforced, qdyn)

        # Simpler hoop stress used in API calculations
        outputs['hoop_stress'] = hoopStress(d, inputs['t'], qdyn)
