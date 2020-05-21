import numpy as np
import scipy.constants as spc
from scipy.optimize import brentq, minimize_scalar, minimize
from openmdao.api import ExplicitComponent
import wisdem.pyframe3dd.pyframe3dd as pyframe3dd
import wisdem.commonse.utilities as util
from wisdem.commonse.constants import gravity


def find_nearest(array, value):
    return (np.abs(array-value)).argmin() 


class RailTransport(ExplicitComponent):
    # Openmdao component to simulate a rail transport of a wind turbine blade
    def initialize(self):
        self.options.declare('analysis_options')

    def setup(self):
        blade_init_options = self.options['analysis_options']['blade']
        af_init_options    = self.options['analysis_options']['airfoils']
        self.n_span        = n_span    = blade_init_options['n_span']
        self.n_xy          = n_xy      = af_init_options['n_xy'] # Number of coordinate points to describe the airfoil geometry

        # Rail configuration
        self.add_input('horizontal_angle_deg', val=13.,         units='deg', desc='Angle of horizontal turn (defined for an chord of 100 feet)')
        self.add_input('min_vertical_radius',  val=609.6,       units='m',   desc='Minimum radius of a vertical curvature (hill or sag) (2000 feet)')
        self.add_input('lateral_clearance',    val=6.7056,      units='m',   desc='Clearance profile horizontal (22 feet)')
        self.add_input('vertical_clearance',   val=7.0104,      units='m',   desc='Clearance profile vertical (23 feet)')
        self.add_input('deck_height',          val=1.19,        units='m',   desc='Height of the deck of the flatcar from the rails (4 feet)')
        self.add_input('max_strains',          val=3500.*1.e-6,              desc='Max allowable strains during transport')
        self.add_input('max_LV',               val=0.5,                      desc='Max allowable ratio between lateral and vertical forces')
        self.add_input('max_flatcar_weight_4axle',   val=129727.31,   units='kg',  desc='Max mass of an 4-axle flatcar (286000 lbm)')
        self.add_input('max_flatcar_weight_8axle',   val=217724.16,   units='kg',  desc='Max mass of an 8-axle flatcar (480000 lbm)')
        self.add_input('max_root_rot_deg',     val=15.,         units='deg', desc='Max degree of angle at blade root')
        self.add_input('flatcar_tc_length',    val=20.12,       units='m',   desc='Flatcar truck center to truck center lenght')

        # Input - Outer blade geometry
        self.add_input('blade_ref_axis',        val=np.zeros((n_span,3)),units='m',   desc='2D array of the coordinates (x,y,z) of the blade reference axis, defined along blade span. The coordinate system is the one of BeamDyn: it is placed at blade root with x pointing the suction side of the blade, y pointing the trailing edge and z along the blade span. A standard configuration will have negative x values (prebend), if swept positive y values, and positive z values.')
        self.add_input('theta',         val=np.zeros(n_span), units='deg', desc='Twist angle at each section (positive decreases angle of attack)')
        self.add_input('chord',         val=np.zeros(n_span), units='m',   desc='chord length at each section')
        self.add_input('pitch_axis',    val=np.zeros(n_span),                 desc='1D array of the chordwise position of the pitch axis (0-LE, 1-TE), defined along blade span.')
        self.add_input('coord_xy_interp',  val=np.zeros((n_span, n_xy, 2)),              desc='3D array of the non-dimensional x and y airfoil coordinates of the airfoils interpolated along span for n_span stations. The leading edge is place at x=0 and y=0.')
        self.add_input('coord_xy_dim',     val=np.zeros((n_span, n_xy, 2)), units = 'm', desc='3D array of the dimensional x and y airfoil coordinates of the airfoils interpolated along span for n_span stations. The origin is placed at the pitch axis.')

        # Inputs - Distributed beam properties
        self.add_input('A',            val=np.zeros(n_span), units='m**2',   desc='airfoil cross section material area')
        self.add_input('EA',           val=np.zeros(n_span), units='N',      desc='axial stiffness')
        self.add_input('EIxx',         val=np.zeros(n_span), units='N*m**2', desc='edgewise stiffness (bending about :ref:`x-axis of airfoil aligned coordinate system <blade_airfoil_coord>`)')
        self.add_input('EIyy',         val=np.zeros(n_span), units='N*m**2', desc='flatwise stiffness (bending about y-axis of airfoil aligned coordinate system)')
        self.add_input('EIxy',         val=np.zeros(n_span), units='N*m**2',   desc='coupled flap-edge stiffness')
        self.add_input('GJ',           val=np.zeros(n_span), units='N*m**2', desc='torsional stiffness (about axial z-axis of airfoil aligned coordinate system)')
        self.add_input('rhoA',         val=np.zeros(n_span), units='kg/m',   desc='mass per unit length')
        self.add_input('rhoJ',         val=np.zeros(n_span), units='kg*m',   desc='polar mass moment of inertia per unit length')
        self.add_input('x_ec_abs',     val=np.zeros(n_span), units='m',      desc='x-distance to elastic center from point about which above structural properties are computed (airfoil aligned coordinate system)')
        self.add_input('y_ec_abs',     val=np.zeros(n_span), units='m',      desc='y-distance to elastic center from point about which above structural properties are computed')
        self.add_input('x_ec',  val=np.zeros(n_span), units='m',        desc='x-distance to elastic center from point about which above structural properties are computed (airfoil aligned coordinate system)')
        self.add_input('y_ec',  val=np.zeros(n_span), units='m', desc='y-distance to elastic center from point about which above structural properties are computed')
        
        # Outputs
        self.add_output('constr_LV_4axle_horiz', val=np.zeros(2), desc='Constraint for max L/V for a 4-axle flatcar on horiz curves, violated when bigger than 1')
        self.add_output('constr_LV_8axle_horiz', val=np.zeros(2), desc='Constraint for max L/V for an 8-axle flatcar on horiz curves, violated when bigger than 1')
        self.add_output('constr_LV_4axle_vert', val=np.zeros(2), desc='Constraint for max L/V for a 4-axle flatcar on vert curves, violated when bigger than 1')
        self.add_output('constr_LV_8axle_vert', val=np.zeros(2), desc='Constraint for max L/V for an 8-axle flatcar on vert curves, violated when bigger than 1')
        self.add_output('constr_strainPS', val=np.zeros((n_span,2)), desc='Strain along pressure side of blade on a horizontal curve')
        self.add_output('constr_strainSS', val=np.zeros((n_span,2)), desc='Strain along suction side of blade on a horizontal curve')
        self.add_output('constr_strainLE', val=np.zeros((n_span,2)), desc='Strain along leading edge side of blade on a vertical curve')
        self.add_output('constr_strainTE', val=np.zeros((n_span,2)), desc='Strain along trailing edge side of blade on a vertical curve')


    def compute(self, inputs, outputs):

        PBEAM = False
        
        # Unpack inputs
        x_ref = inputs['blade_ref_axis'][:,0] # from PS to SS
        y_ref = inputs['blade_ref_axis'][:,1] # from LE to TE
        z_ref = inputs['blade_ref_axis'][:,2] # from root to tip
        r     = util.arc_length(inputs['blade_ref_axis'])
        blade_length = r[-1]
        theta = inputs['theta']
        chord = inputs['chord']
        x_ec  = inputs['x_ec']
        y_ec  = inputs['y_ec']
        A     = inputs['A']
        rhoA  = inputs['rhoA']
        rhoJ  = inputs['rhoJ']
        GJ    = inputs['GJ']
        EA    = inputs['EA']
        EIxx  = inputs['EIxx'] # edge (rotation about x)
        EIyy  = inputs['EIyy'] # flap (rotation about y)
        EIxy  = inputs['EIxy']
        lateral_clearance = 0.5*inputs['lateral_clearance'][0]
        vertical_clearance = inputs['vertical_clearance'][0]
        #n_points          = 10000
        max_strains       = inputs['max_strains'][0]
        #n_opt             = 21
        max_LV            = inputs['max_LV'][0]
        mass_car_4axle    = inputs['max_flatcar_weight_4axle'][0]
        mass_car_8axle    = inputs['max_flatcar_weight_8axle'][0]
        #max_root_rot_deg  = inputs['max_root_rot_deg'][0]
        flatcar_tc_length = inputs['flatcar_tc_length'][0]
        #np.savez('nrel5mw_test.npz',r=r,x_az=x_az,y_az=y_az,z_az=z_az,theta=theta,x_ec=x_ec,y_ec=y_ec,A=A,rhoA=rhoA,rhoJ=rhoJ,GJ=GJ,EA=EA,EIxx=EIxx,EIyy=EIyy,EIxy=EIxy,Px_af=Px_af,Py_af=Py_af,Pz_af=Pz_af,xu_strain_spar=xu_strain_spar,xl_strain_spar=xl_strain_spar,yu_strain_spar=yu_strain_spar,yl_strain_spar=yl_strain_spar,xu_strain_te=xu_strain_te,xl_strain_te=xl_strain_te,yu_strain_te=yu_strain_te,yl_strain_te=yl_strain_te)


        #------- Get turn radius geometry for horizontal and vertical curves
        # Horizontal turns- defined as a degree of arc assuming a 100ft "chord"
        # https://trn.trains.com/railroads/ask-trains/2011/01/measuring-track-curvature
        angleH_rad = np.deg2rad(inputs['horizontal_angle_deg'][0])
        r_curveH   = spc.foot * 100. /(2.*np.sin(0.5*angleH_rad))
        arcsH      = r / r_curveH
        
        # Vertical curves on hills and sags defined directly by radius
        r_curveV   = inputs['min_vertical_radius'][0]
        arcsV      = r / r_curveV
        # ----------


        #---------- Put airfoil cross sections into principle axes
        # Determine principal C.S. (with swap of x, y for profile c.s.)
        EIxx_cs , EIyy_cs = EIyy.copy() , EIxx.copy()
        x_ec_cs , y_ec_cs = y_ec.copy() , x_ec.copy()
        EIxy_cs = EIxy.copy()

        # translate to elastic center
        EIxx_cs -= y_ec_cs**2 * EA
        EIyy_cs -= x_ec_cs**2 * EA
        EIxy_cs -= x_ec_cs * y_ec_cs * EA

        # get rotation angle
        alpha = 0.5*np.arctan2(2*EIxy_cs, EIyy_cs-EIxx_cs)

        # get moments and positions in principal axes
        EI11 = EIxx_cs - EIxy_cs*np.tan(alpha)
        EI22 = EIyy_cs + EIxy_cs*np.tan(alpha)
        ca   = np.cos(alpha)
        sa   = np.sin(alpha)
        def rotate(x,y):
            x2 =  x*ca + y*sa
            y2 = -x*sa + y*ca
            return x2, y2

        # Now store alpha for later use in degrees
        alpha = np.rad2deg(alpha)
        # -------------------

        
        # ---------- Frame3dd blade prep
        # Nodes: Prep data, but node x,y,z will shift for vertical and horizontal curves
        rad   = np.zeros(self.n_span) # 'radius' of rigidity at node- set to zero
        inode = np.arange(self.n_span) # Node numbers (1-based indexing)
        L     = np.diff(r)

        # Reactions: prep data for 3 attachment points
        rigid     = 1e16
        pin_pin   = rigid*np.ones(3)
        pin_free  = np.array([rigid, 0.0, 0.0])
        
        # Element data
        elem = np.arange(1, self.n_span) # Element Numbers
        N1   = np.arange(1, self.n_span) # Node number start
        N2   = np.arange(2, self.n_span+1) # Node number finish
        E    = EA   / A
        rho  = rhoA / A
        J    = rhoJ / rho
        G    = GJ   / J
        Ix   = EIyy / E if PBEAM else EI22 / E
        Iy   = EIxx / E if PBEAM else EI11 / E
        Asx  = Asy = 1e-6*np.ones(elem.shape) # Unused when shear=False

        # Have to convert nodal values to find average at center of element
        Abar,_   = util.nodal2sectional(A)
        Ebar,_   = util.nodal2sectional(E)
        rhobar,_ = util.nodal2sectional(rho)
        Jbar,_   = util.nodal2sectional(J)
        Gbar,_   = util.nodal2sectional(G)
        Ixbar,_  = util.nodal2sectional(Ix)
        Iybar,_  = util.nodal2sectional(Iy)

        # Angle of element principal axes relative to global coordinate system
        # Global c.s. is blade with z from root to tip, y from ss to ps, and x from LE to TE (TE points up)
        # Local element c.s. is airfoil (twist + principle rotation)
        if PBEAM:
            roll = np.zeros(theta.shape)
        else:
            roll,_ = util.nodal2sectional(theta + alpha)

        elements = pyframe3dd.ElementData(elem, N1, N2, Abar, Asx, Asy, Jbar, Ixbar, Iybar, Ebar, Gbar, roll, rhobar)

        # Frame3dd options: no need for shear, axial stiffening, or higher resolution force calculations
        options = pyframe3dd.Options(False, False, -1)
        #-----------

        
        #------ Airfoil positions at which to measure strain
        # Find the cross sectional points furthest from the elastic center at each spanwise location to be used for strain measurement
        xps = np.zeros(self.n_span)
        xss = np.zeros(self.n_span)
        yps = np.zeros(self.n_span)
        yss = np.zeros(self.n_span)
        xle = np.zeros(self.n_span)
        xte = np.zeros(self.n_span)
        yle = np.zeros(self.n_span)
        yte = np.zeros(self.n_span)

        for i in range(self.n_span):        
            ## Rotate the profiles to the blade reference system
            profile_i = inputs['coord_xy_interp'][i,:,:]
            profile_i_rot = np.column_stack(util.rotate(inputs['pitch_axis'][i], 0., profile_i[:,0], profile_i[:,1], np.radians(theta[i])))
            # normalize
            profile_i_rot[:,0] -= min(profile_i_rot[:,0])
            profile_i_rot = profile_i_rot/ max(profile_i_rot[:,0])
            profile_i_rot_precomp = profile_i_rot.copy()
            idx_s = 0
            idx_le_precomp = np.argmax(profile_i_rot_precomp[:,0])
            if idx_le_precomp != 0:
                if profile_i_rot_precomp[0,0] == profile_i_rot_precomp[-1,0]:
                     idx_s = 1
                profile_i_rot_precomp = np.row_stack((profile_i_rot_precomp[idx_le_precomp:], profile_i_rot_precomp[idx_s:idx_le_precomp,:]))
            profile_i_rot_precomp[:,1] -= profile_i_rot_precomp[np.argmin(profile_i_rot_precomp[:,0]),1]

            # # renormalize
            profile_i_rot_precomp[:,0] -= min(profile_i_rot_precomp[:,0])
            profile_i_rot_precomp = profile_i_rot_precomp/ max(profile_i_rot_precomp[:,0])

            if profile_i_rot_precomp[-1,0] != 1.:
                profile_i_rot_precomp = np.row_stack((profile_i_rot_precomp, profile_i_rot_precomp[0,:]))

            # 'web' at trailing edge needed for flatback airfoils
            if profile_i_rot_precomp[0,1] != profile_i_rot_precomp[-1,1] and profile_i_rot_precomp[0,0] == profile_i_rot_precomp[-1,0]:
                flatback = True
            else:
                flatback = False

            xnode          = profile_i_rot_precomp[:,0]
            xnode_pa       = xnode - inputs['pitch_axis'][i]
            ynode          = profile_i_rot_precomp[:,1]
            theta_rad      = theta[i] * np.pi / 180.

            xnode_no_theta = xnode_pa * np.cos(-theta_rad) - ynode * np.sin(-theta_rad)
            ynode_no_theta = xnode_pa * np.sin(-theta_rad) + ynode * np.cos(-theta_rad)

            xnode_dim_no_theta = xnode_no_theta * chord[i]
            ynode_dim_no_theta = ynode_no_theta * chord[i]

            xnode_dim = xnode_dim_no_theta * np.cos(theta_rad) - ynode_dim_no_theta * np.sin(theta_rad)
            ynode_dim = xnode_dim_no_theta * np.sin(theta_rad) + ynode_dim_no_theta * np.cos(theta_rad)

            yss[i] = max(ynode_dim) - y_ec[i]
            yps[i] = y_ec[i] - min(ynode_dim)
            xte[i] = max(xnode_dim) - x_ec[i]
            xle[i] = x_ec[i] - min(xnode_dim)

        # Put these sectional points in airfoil principle directions
        xps_cs, yps_cs = yps, xps
        xss_cs, yss_cs = yss, xss
        
        ps1, ps2 = rotate(xps_cs, yps_cs)
        ss1, ss2 = rotate(xss_cs, yss_cs)

        xle_cs, yle_cs = yle, xle
        xte_cs, yte_cs = yte, xte
        
        le1, le2 = rotate(xle_cs, yle_cs)
        te1, te2 = rotate(xte_cs, yte_cs)
        #----------------


        #-------- Horizontal curve where we select blade support nodes on flat cars
        # Gravity field orientation
        gy = -gravity
        gx = gz = 0.0

        ireact = inode.copy() #np.unique(np.r_[0, np.where(node_dr)[0]])
        pin_pin  = rigid*np.ones(ireact.size)
        pin_free = np.zeros(ireact.size)
        pin_free[0] = rigid
        reactions = pyframe3dd.ReactionData(ireact+1, pin_pin, pin_pin, pin_pin, pin_free, pin_free, pin_free, float(rigid))

        RF_derailH = np.zeros((ireact.size, 2))
        strainPS = np.zeros((self.n_span, 2))
        strainSS = np.zeros((self.n_span, 2))

        def run_hcurve(rot_angles):
            # Curve towards SS (towards the left with LE pointed down and standing at the root)
            x_rot1, z_rot1 = util.rotate(r_curveH, 0.0, r_curveH + x_ref, z_ref, rot_angles[0])

            # Curve towards PS (towards the right with LE pointed down and standing at the root)
            x_rot2, z_rot2 = util.rotate(-r_curveH, 0.0, -r_curveH + x_ref, z_ref, rot_angles[1])

            # Set nodes to be convenient for coordinate system with center of curvature 0,0 in y-z plane
            nodes1 = pyframe3dd.NodeData(inode+1, x_rot1, y_ref, z_rot1, rad)
            nodes2 = pyframe3dd.NodeData(inode+1, x_rot2, y_ref, z_rot2, rad)
            r_blade1 = np.sqrt(nodes1.x**2 + nodes1.z**2)
            r_blade2 = np.sqrt(nodes2.x**2 + nodes2.z**2)

            # Initialize frame3dd object
            blade1 = pyframe3dd.Frame(nodes1, reactions, elements, options)
            blade2 = pyframe3dd.Frame(nodes2, reactions, elements, options)

            # Load case1: gravity + blade bending towards SS
            blade_xmin = x_ref - xss
            blade_xmax = x_ref + xps
            r_envelopeH = r_curveH + lateral_clearance*np.array([-1, 1])
            r_envelopeH_inner = r_envelopeH.min() + blade_xmin
            r_envelopeH_outer = r_envelopeH.max() - blade_xmax
            node_dr_inner = np.maximum(r_envelopeH_inner - r_blade1, 0)
            node_dr_outer = np.minimum(r_envelopeH_outer - r_blade1, 0)
            node_dr = node_dr_inner + node_dr_outer
            #node_dr_inner = r_envelopeH_inner - r_blade1
            #node_dr_outer = r_envelopeH_outer - r_blade1
            #node_dr = rot_angles[2:]*(node_dr_outer-node_dr_inner) + node_dr_inner
            node_dx = node_dr*np.cos(arcsH)
            node_dz = node_dr*np.sin(arcsH)

            dy = dM = np.zeros(ireact.size)
            load1 = pyframe3dd.StaticLoadCase(gx, gy, gz)
            load1.changePrescribedDisplacements(ireact+1, node_dx[ireact], dy, node_dz[ireact], dM, dM, dM)

            # Load case2: gravity + blade bending towards PS
            blade_xmin = x_ref - xps
            blade_xmax = x_ref + xss
            r_envelopeH = r_curveH + lateral_clearance*np.array([-1, 1])
            r_envelopeH_inner = r_envelopeH.min() + blade_xmin
            r_envelopeH_outer = r_envelopeH.max() - blade_xmax
            node_dr_inner = np.maximum(r_envelopeH_inner - r_blade2, 0)
            node_dr_outer = np.minimum(r_envelopeH_outer - r_blade2, 0)
            node_dr = node_dr_inner + node_dr_outer
            node_dx = node_dr*np.cos(np.pi - arcsH)
            node_dz = node_dr*np.sin(np.pi - arcsH)

            load2 = pyframe3dd.StaticLoadCase(gx, gy, gz)
            load2.changePrescribedDisplacements(ireact+1, node_dx[ireact], dy, node_dz[ireact], dM, dM, dM)

            # Store this load case
            blade1.addLoadCase(load1)
            blade2.addLoadCase(load2)

            # Debugging
            #blade.write('blade.3dd')

            for k in range(2):
                blade = blade1 if k==0 else blade2

                # Run the case
                displacements, forces, forces_rxn, internalForces, mass, modal = blade.run()
                #r_check = np.sqrt( (nodes.x+displacements.dx[0,:])**2 + (nodes.z+displacements.dz[0,:])**2)

                # Reaction forces for derailment:
                #  - Lateral force on wheels (multiply by 0.5 for 2 wheel sets)
                #  - Moment around axis perpendicular to ground
                RF_derailH[:,k] = 0.5*np.abs(forces_rxn.Fy) + np.abs(forces_rxn.Mxx)/flatcar_tc_length

                # Element shear and bending, one per element, which are already in principle directions in Hansen's notation
                iCase = 0
                Fz = np.r_[-forces.Nx[ iCase,0],  forces.Nx[ iCase, 1::2]]
                M1 = np.r_[-forces.Myy[iCase,0],  forces.Myy[iCase, 1::2]]
                M2 = np.r_[ forces.Mzz[iCase,0], -forces.Mzz[iCase, 1::2]]

                # compute strain at the two points: pressure/suction side extremes
                strainPS[:,k] = -(M1/EI11*ps2 - M2/EI22*ps1 + Fz/EA)  # negative sign because Hansen c3 is opposite of Precomp z
                strainSS[:,k] = -(M1/EI11*ss2 - M2/EI22*ss1 + Fz/EA)
                
            return RF_derailH, strainPS, strainSS
        
        # Assume root rotates to max point that still keeps blade within clearance envelope: have to find that rotation angle
        def opt_rot_blade(anglesIn):
            RF_derailH, strainPS, strainSS = run_hcurve(anglesIn)
            #obj1 = RF_derailH[0,:].mean() / (0.5 * mass_car_8axle * gravity) / max_LV
            mystrainPS = np.maximum(np.abs(strainPS) - max_strains, 0.0)
            mystrainSS = np.maximum(np.abs(strainSS) - max_strains, 0.0)
            obj2 = (mystrainPS.mean() + mystrainSS.mean()) / 2
            return obj2
        
        def con_rot_blade(anglesIn):
            RF_derailH, strainPS, strainSS = run_hcurve(anglesIn)
            obj1 = RF_derailH[0,:] / (0.5 * mass_car_8axle * gravity) / max_LV
            #mystrainPS = np.maximum(np.abs(strainPS) - max_strains, 0.0)
            #mystrainSS = np.maximum(np.abs(strainSS) - max_strains, 0.0)
            #obj2 = (mystrainPS.mean() + mystrainSS.mean()) / 2
            return 1-obj1
        
        const         = {}
        const['type'] = 'ineq'
        const['fun']  = con_rot_blade
        bounds = [np.pi/10.0*np.r_[-1,1]]*2 #+ [[0,1]]*self.n_span
        x0     = np.r_[np.deg2rad([15, -15])]#, np.zeros(self.n_span)]
        result = minimize(opt_rot_blade, x0, method='slsqp', bounds=bounds, tol=1e-6, constraints=const)
        
        if result.success or result.status==9:
            print(result)
            print(np.rad2deg(result.x))
            RF_derailH, strainPS, strainSS = run_hcurve(result.x)
        else:
            breakpoint()
            
        # Express derailing force as a constraint
        constr_derailH_4axle = RF_derailH / (0.5 * mass_car_4axle * gravity) / max_LV
        constr_derailH_8axle = RF_derailH / (0.5 * mass_car_8axle * gravity) / max_LV
        outputs['constr_LV_4axle_horiz'] = constr_derailH_4axle[0,:]
        outputs['constr_LV_8axle_horiz'] = constr_derailH_8axle[0,:]
        print(outputs['constr_LV_8axle_horiz'])
        # Strain constraint outputs
        outputs['constr_strainPS'] = np.abs(strainPS) / max_strains
        outputs['constr_strainSS'] = np.abs(strainSS) / max_strains
        print(outputs['constr_strainPS'].sum(),outputs['constr_strainSS'].sum())
        
        # ------- Vertical hills/sag using best attachment points
        # Set up Frame3DD blade for vertical analysis

        # Set nodes to be convenient for coordinate system with center of curvature 0,0 in x-z plane
        nodes = pyframe3dd.NodeData(inode+1, x_ref, r_curveV+y_ref, z_ref, rad)
        r_blade = np.sqrt(nodes.y**2 + nodes.z**2)
        
        # Initialize frame3dd object
        blade = pyframe3dd.Frame(nodes, reactions, elements, options)

        # Hill
        r_envelopeV = r_curveV + vertical_clearance
        blade_ymax = y_ref + yte
        r_envelopeV_outer = r_envelopeV - blade_ymax
        node_dr = np.minimum(r_envelopeV_outer - r_blade, 0)
        node_dy = node_dr*np.cos(arcsV)
        node_dz = node_dr*np.sin(arcsV)
        
        # Load case 1: gravity + hill
        dx = dM = np.zeros(ireact.size)
        load1 = pyframe3dd.StaticLoadCase(gx, gy, gz)
        load1.changePrescribedDisplacements(ireact+1, dx, node_dy[ireact], node_dz[ireact], dM, dM, dM)

        # Sag
        r_envelopeV = r_curveV - vertical_clearance
        blade_ymin = y_ref - yte
        r_envelopeV_inner = r_envelopeV.min() + blade_ymin
        node_dr = np.maximum(r_envelopeV_inner - r_blade, 0)
        node_dy = node_dr*np.cos(arcsV)
        node_dz = node_dr*np.sin(arcsV)
        
        # Load case 2: gravity + sag
        load2 = pyframe3dd.StaticLoadCase(gx, gy, gz)
        load2.changePrescribedDisplacements(ireact+1, dx, node_dy[ireact], node_dz[ireact], dM, dM, dM)

        # Store these load cases and run
        blade.addLoadCase(load1)
        blade.addLoadCase(load2)
        #blade.write('blade.3dd')
        displacements, forces, forces_rxn, internalForces, mass, modal = blade.run()

        # Reaction forces for derailment:
        #  - Lateral force on wheels (multiply by 0.5 for 2 wheel sets)
        #  - Moment around axis perpendicular to ground
        # Should have 2 cases X 3 rxn nodes
        RF_derailV = -0.5*forces_rxn.Fy - forces_rxn.Mxx/flatcar_tc_length

        # Loop over hill & sag cases, then take worst strain case
        strainLE = np.zeros((self.n_span, 2))
        strainTE = np.zeros((self.n_span, 2))
        for k in range(2):
            # Element shear and bending, one per element, with conversion to profile c.s. using Hansen's notation
            Fz = np.r_[-forces.Nx[ k, 0],  forces.Nx[ k, 1::2]]
            M1 = np.r_[-forces.Myy[k, 0],  forces.Myy[k, 1::2]]
            M2 = np.r_[ forces.Mzz[k, 0], -forces.Mzz[k, 1::2]]

            # compute strain at the two points
            strainLE[:,k] = -(M1/EI11*le2 - M2/EI22*le1 + Fz/EA)
            strainTE[:,k] = -(M1/EI11*te2 - M2/EI22*te1 + Fz/EA)
            
        # Find best points for middle reaction and formulate as constraints
        constr_derailV_8axle = (RF_derailV.T / (0.5 * mass_car_8axle * gravity)) / max_LV
        constr_derailV_4axle = (RF_derailV.T / (0.5 * mass_car_4axle * gravity)) / max_LV

        outputs['constr_LV_4axle_vert'] = constr_derailV_4axle[0,:]
        outputs['constr_LV_8axle_vert'] = constr_derailV_8axle[0,:]

        # Strain constraint outputs
        outputs['constr_strainLE'] = strainLE / max_strains
        outputs['constr_strainTE'] = strainTE / max_strains
