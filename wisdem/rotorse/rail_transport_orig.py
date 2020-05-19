import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import wisdem.pBeam._pBEAM as _pBEAM
from openmdao.api import ExplicitComponent
from wisdem.commonse.utilities import rotate
from wisdem.commonse.constants import gravity
import copy

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
        self.add_input('max_flatcar_weight_4axle',   val=129727.31,   units='kg',  desc='Max weight of an 4-axle flatcar (286000 lbm)')
        self.add_input('max_flatcar_weight_8axle',   val=217724.16,   units='kg',  desc='Max weight of an 8-axle flatcar (480000 lbm)')
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
        self.add_input('EA',           val=np.zeros(n_span), units='N',      desc='axial stiffness')
        self.add_input('EIxx',         val=np.zeros(n_span), units='N*m**2', desc='edgewise stiffness (bending about :ref:`x-direction of airfoil aligned coordinate system <blade_airfoil_coord>`)')
        self.add_input('EIyy',         val=np.zeros(n_span), units='N*m**2', desc='flatwise stiffness (bending about y-direction of airfoil aligned coordinate system)')
        self.add_input('GJ',           val=np.zeros(n_span), units='N*m**2', desc='torsional stiffness (about axial z-direction of airfoil aligned coordinate system)')
        self.add_input('rhoA',         val=np.zeros(n_span), units='kg/m',   desc='mass per unit length')
        self.add_input('rhoJ',         val=np.zeros(n_span), units='kg*m',   desc='polar mass moment of inertia per unit length')
        self.add_input('x_ec_abs',     val=np.zeros(n_span), units='m',      desc='x-distance to elastic center from point about which above structural properties are computed (airfoil aligned coordinate system)')
        self.add_input('y_ec_abs',     val=np.zeros(n_span), units='m',      desc='y-distance to elastic center from point about which above structural properties are computed')
        
        # Outputs
        self.add_output('LV_constraint_4axle_horiz', val=0.0, desc='Constraint for max L/V for a 4-axle flatcar, violated when bigger than 1')
        self.add_output('LV_constraint_8axle_horiz', val=0.0, desc='Constraint for max L/V for an 8-axle flatcar, violated when bigger than 1')
        self.add_output('LV_constraint_4axle_vert', val=0.0, desc='Constraint for max L/V for a 4-axle flatcar, violated when bigger than 1')
        self.add_output('LV_constraint_8axle_vert', val=0.0, desc='Constraint for max L/V for an 8-axle flatcar, violated when bigger than 1')


    def compute(self, inputs, outputs):
       
        # Horizontal turns
        # Inputs
        blade_length            = inputs['blade_ref_axis'][-1,2]
        if max(abs(inputs['blade_ref_axis'][:,1])) > 0.:
            exit('The script currently does not support swept blades')

        lateral_clearance       = inputs['lateral_clearance'][0]
        n_points                = 10000
        max_strains             = inputs['max_strains'][0]
        n_opt                   = 21
        max_LV                  = inputs['max_LV'][0]
        weight_car_4axle        = inputs['max_flatcar_weight_4axle'][0]
        weight_car_8axle        = inputs['max_flatcar_weight_8axle'][0]
        max_root_rot_deg        = inputs['max_root_rot_deg'][0]
        flatcar_tc_length       = inputs['flatcar_tc_length'][0]
        #########

        def arc_length(x, y):
            arc = np.sqrt( np.diff(x)**2 + np.diff(y)**2 )
            return np.r_[0.0, np.cumsum(arc)]

        angle_rad    = inputs['horizontal_angle_deg'][0] / 180. * np.pi
        radius = sp.constants.foot * 100. /(2.*np.sin(angle_rad/2.))

        r        = inputs['blade_ref_axis'][:,2]
        EIflap   = inputs['EIyy']
        EA       = inputs['EA']
        EIedge   = inputs['EIxx']
        GJ       = inputs['GJ']
        rhoA     = inputs['rhoA']
        rhoJ     = inputs['rhoJ']

        AE = np.zeros(self.n_span)
        DE = np.zeros(self.n_span)
        EC = np.zeros(self.n_span)
        EB = np.zeros(self.n_span)


        # Elastic center
        ## Spanwise
        for i in range(self.n_span):        
            ## Rotate the profiles to the blade reference system
            profile_i = inputs['coord_xy_interp'][i,:,:]
            profile_i_rot = np.column_stack(rotate(inputs['pitch_axis'][i], 0., profile_i[:,0], profile_i[:,1], np.radians(inputs['theta'][i])))
            # normalize
            profile_i_rot[:,0] -= min(profile_i_rot[:,0])
            profile_i_rot = profile_i_rot/ max(profile_i_rot[:,0])
            profile_i_rot_precomp = copy.copy(profile_i_rot)
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
            theta_rad      = inputs['theta'][i] * np.pi / 180.

            xnode_no_theta = xnode_pa * np.cos(-theta_rad) - ynode * np.sin(-theta_rad)
            ynode_no_theta = xnode_pa * np.sin(-theta_rad) + ynode * np.cos(-theta_rad)

            xnode_dim_no_theta = xnode_no_theta * inputs['chord'][i]
            ynode_dim_no_theta = ynode_no_theta * inputs['chord'][i]

            xnode_dim = xnode_dim_no_theta * np.cos(theta_rad) - ynode_dim_no_theta * np.sin(theta_rad)
            ynode_dim = xnode_dim_no_theta * np.sin(theta_rad) + ynode_dim_no_theta * np.cos(theta_rad)

            # Compute the points farthest from the elastic center in the blade reference system
            x_ec = inputs['x_ec_abs'][i]
            y_ec = inputs['y_ec_abs'][i]

            AE[i] = max(ynode_dim) - y_ec
            EB[i] = y_ec - min(ynode_dim)
            EC[i] = max(xnode_dim) - x_ec
            DE[i] = x_ec - min(xnode_dim)

        dist_ss  = AE
        dist_ps  = EB

        # Reconstruct the distributed loading q along the blade corresponding to the maximum allowable strains. This is done by computing the moment distribution M and deriving it twice
        M        = np.array(max_strains * EIflap / dist_ss)
        V        = -np.gradient(M,r)
        q        = -np.gradient(V,r)    

        # Interpolate the moment at the optimization points and recompute the distributed loading q
        r_opt    = np.linspace(0., blade_length, n_opt)
        pb_opt   = np.interp(r_opt, r, inputs['blade_ref_axis'][:,0])
        M_opt_h  = np.interp(r_opt, r, M)
        V_opt_h  = np.gradient(M_opt_h,r_opt)
        q_opt_h  = np.max([np.zeros(n_opt), np.gradient(V_opt_h,r_opt)], axis=0)

        # Draw the rail lines given the lateral curvature radius
        r_midline = radius
        r_outer   = r_midline + 0.5*lateral_clearance
        r_inner   = r_midline - 0.5*lateral_clearance

        x_rail_h  = np.linspace(0., 2.*r_midline, n_points)
        y_rail_h  = np.sqrt(r_midline**2. - (x_rail_h-r_midline)**2.)

        # Draw the lateral clearance given the rail lines
        x_outer   = np.linspace(- 0.5*lateral_clearance, 2.*r_midline + 0.5*lateral_clearance, n_points)
        y_outer   = np.sqrt(r_outer**2. - (x_outer-r_midline)**2.)

        x_inner   = np.linspace(0.5*lateral_clearance, 2.*r_midline - 0.5*lateral_clearance, n_points)
        y_inner   = np.sqrt(r_inner**2. - (x_inner-r_midline)**2.)

        # Interpolate the blade elastic properties
        dist_ss_interp   = np.interp(r_opt, r, dist_ss)
        dist_ps_interp   = np.interp(r_opt, r, dist_ps)
        EIflap_interp    = np.interp(r_opt, r, EIflap)
        EIedge_interp    = np.interp(r_opt, r, EIedge)
        GJ_interp        = np.interp(r_opt, r, GJ)
        rhoA_interp      = np.interp(r_opt, r, rhoA)
        EA_interp        = np.interp(r_opt, r, EA)
        rhoJ_interp      = np.interp(r_opt, r, rhoJ)

        # pbeam initialization
        p_section    = _pBEAM.SectionData(n_opt, r_opt, EA_interp, EIedge_interp, EIflap_interp, GJ_interp, rhoA_interp, rhoJ_interp)
        p_tip        = _pBEAM.TipData()  # no tip mass
        p_base       = _pBEAM.BaseData(np.ones(6), 1.0)  # rigid base
        
        def get_max_force_h(inputs):
            # Objective function to minimize the reaction force of the first flatcat, which holds blade root, during a lateral curve
            q_iter    = q_opt_h * inputs[:-1]
            V_iter    = np.zeros(n_opt)
            M_iter    = np.zeros(n_opt)
            for i in range(n_opt):
                V_iter[i] = np.trapz(q_iter[i:],r_opt[i:])
            for i in range(n_opt):
                M_iter[i] = np.trapz(V_iter[i:],r_opt[i:])
            
            RF_flatcar_1 = 0.5 * V_iter[0] + M_iter[0] / flatcar_tc_length

            return RF_flatcar_1*1.e-5

        def get_constraints_h(inputs):
            # Constraint function to make sure the blade does not exceed the maximum strains while staying within the lateral clearance
            q_iter = q_opt_h * inputs[:-1] #np.gradient(V_iter,r_opt)
            V_iter = np.zeros(n_opt)
            M_iter = np.zeros(n_opt)
            for i in range(n_opt):
                V_iter[i] = np.trapz(q_iter[i:],r_opt[i:])
            for i in range(n_opt):
                M_iter[i]    = np.trapz(V_iter[i:],r_opt[i:])

            root_rot_rad_iter = inputs[-1]

            eps            = M_iter * dist_ss_interp / EIflap_interp
            consts_strains = (max_strains - abs(eps))*1.e+3

            p_loads      = _pBEAM.Loads(n_opt, q_iter, np.zeros_like(r_opt), np.zeros_like(r_opt))
            blade = _pBEAM.Beam(p_section, p_loads, p_tip, p_base)
            dx, dy, dz, dtheta_r1, dtheta_r2, dtheta_z = blade.displacement()

            x_blade_transport = (dx + pb_opt)*blade_length/arc_length(r_opt, dx)[-1]
            y_blade_transport = r_opt*blade_length/arc_length(r_opt, dx)[-1]

            ps_x = x_blade_transport + dist_ps_interp
            ss_x = x_blade_transport - dist_ss_interp
            ps_y = ss_y = y_blade_transport

            ps_x_rot  = ps_x*np.cos(root_rot_rad_iter) - ps_y*np.sin(root_rot_rad_iter)
            ps_y_rot  = ps_y*np.cos(root_rot_rad_iter) + ps_x*np.sin(root_rot_rad_iter)
            
            ss_x_rot  = ss_x*np.cos(root_rot_rad_iter) - ss_y*np.sin(root_rot_rad_iter)
            ss_y_rot  = ss_y*np.cos(root_rot_rad_iter) + ss_x*np.sin(root_rot_rad_iter)

            id_outer = np.zeros(n_opt, dtype = int)
            id_inner = np.zeros(n_opt, dtype = int)
            for i in range(n_opt):
                id_outer[i] = np.argmin(abs(y_outer[:int(np.ceil(n_points*0.5))] - ps_y_rot[i]))
                id_inner[i] = np.argmin(abs(y_inner[:int(np.ceil(n_points*0.5))]  - ss_y_rot[i]))

            consts_envelope_outer = ss_x_rot - x_outer[id_outer]
            consts_envelope_inner = x_inner[id_inner] - ps_x_rot

            # Constraints on maximum strains, and outer and inner limits of the lateral clearance
            consts = np.hstack((consts_strains, consts_envelope_outer, consts_envelope_inner))

            return consts

        # Run a sub-optimization to find the distributed loading that respects the constraints in maximum strains, and outer and inner limits of the lateral clearance, while minimizing the reaction force of the first flatcar
        x0    = np.hstack((np.ones(n_opt), 0.))
        # To-do initialize the tuple given the n_opt
        bnds = ((0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(-max_root_rot_deg / 180. * np.pi, max_root_rot_deg / 180. * np.pi))
        const           = {}
        const['type']   = 'ineq'
        const['fun']    = get_constraints_h
        res    = minimize(get_max_force_h, x0, method='SLSQP', bounds=bnds, constraints=const)

        if res.success == False:
            # If the optimization does not find a solution, assign a high value to the the fields LV_constraint_8axle, which are typically imposed lower than 1 in an outer blade design loop
            outputs['LV_constraint_8axle_horiz'] = 2.
            outputs['LV_constraint_4axle_horiz'] = 2.
            print('The optimization cannot satisfy the constraint on max strains of 3500 mu eps')
        else:
            # If the optimization does converge, integrate the distributed loading twice to obtain the bending moment and the strains along span
            q_final    = q_opt_h * res.x[:-1]
            V_final    = np.zeros(n_opt)
            M_final    = np.zeros(n_opt)
            for i in range(n_opt):
                V_final[i] = np.trapz(q_final[i:],r_opt[i:])
            for i in range(n_opt):
                M_final[i] = np.trapz(V_final[i:],r_opt[i:])
            
            root_rot_rad_final = res.x[-1]

            # print('The optimizer finds a solution for the lateral curves!')
            # print('Prescribed rotation angle: ' + str(root_rot_rad_final * 180. / np.pi) + ' deg')
            
            # Compute the reaction force on the first flatcar
            RF_flatcar_1 = 0.5 * V_final[0] + M_final[0] / flatcar_tc_length

            # print('Max reaction force lateral turn: ' + str(RF_flatcar_1) + ' N')

            # Constraint the lateral reaction force to respect the max L/V ratio, typically imposed to be lower than 0.5 
            outputs['LV_constraint_8axle_horiz'] = (RF_flatcar_1 / (0.5 * weight_car_8axle * gravity)) / max_LV
            outputs['LV_constraint_4axle_horiz'] = (RF_flatcar_1 / (0.5 * weight_car_4axle * gravity)) / max_LV

            print('L/V constraint 8-axle: ' + str(outputs['LV_constraint_8axle_horiz']))
            # print('L/V constraint 4-axle: ' + str(outputs['LV_constraint_4axle_horiz']))


        # Vertical turns - hill
        vertical_clearance      = inputs['vertical_clearance'][0]
        deck_height             = inputs['deck_height'][0]
        min_vertical_radius     = inputs['min_vertical_radius'][0]

        dist_le  = DE
        dist_te  = EC

        # The blade is transported with the trailing edge pointing upwards. Compute the moment that generates max strains at the trailing edge and derive it twice to obtain the correspnding distributed loading q
        M        = np.array(max_strains * EIedge / dist_te)
        V        = -np.gradient(M,r)
        q        = -np.gradient(V,r)    

        M_opt_v  = np.interp(r_opt, r, M)
        V_opt_v  = np.gradient(M_opt_v,r_opt)
        q_opt_v  = np.max([np.zeros(n_opt), np.gradient(V_opt_v,r_opt)], axis=0)

        dist_le_interp   = np.interp(r_opt, r, dist_le)
        dist_te_interp   = np.interp(r_opt, r, dist_te)


        r_rail    = min_vertical_radius
        r_deck_hill    = r_rail + deck_height
        r_upper_hill   = r_rail + vertical_clearance
        r_deck_sag     = r_rail - deck_height
        r_upper_sag    = r_rail - vertical_clearance

        x_rail_v  = np.linspace(0., 2.*r_rail, n_points)
        y_rail_v  = np.sqrt(r_rail**2. - (x_rail_v-r_rail)**2.)

        # Draw the verticl clearance given the rail lines for hill and sag cases
        x_deck_hill   = np.linspace(-deck_height, 2.*r_deck_hill - deck_height, n_points)
        y_deck_hill   = np.sqrt(r_deck_hill**2. - (x_deck_hill-r_rail)**2.)
        x_upper_hill  = np.linspace(-vertical_clearance, 2.*r_upper_hill - vertical_clearance, n_points)
        y_upper_hill  = np.sqrt(r_upper_hill**2. - (x_upper_hill-r_rail)**2. + 1.e-5)
        x_deck_sag    = np.linspace(deck_height, 2.*r_deck_sag + deck_height, n_points)
        y_deck_sag   = np.sqrt(r_deck_sag**2. - (x_deck_sag-r_rail)**2.)
        x_upper_sag   = np.linspace(vertical_clearance, 2.*r_upper_sag + vertical_clearance, n_points)
        y_upper_sag   = np.sqrt(r_upper_sag**2. - (x_upper_sag-r_rail)**2. + 1.e-5)

        def get_max_force_v(inputs):
            # Objective function to minimize the reaction force of the first flatcat, which holds blade root, during a vertical curve 
            q_iter    = q_opt_h * inputs[:-2]
            V_iter    = np.zeros(n_opt)
            M_iter    = np.zeros(n_opt)
            for i in range(n_opt):
                V_iter[i] = np.trapz(q_iter[i:],r_opt[i:])
            for i in range(n_opt):
                M_iter[i] = np.trapz(V_iter[i:],r_opt[i:])
            
            RF_flatcar_1 = V_iter[0] + M_iter[0] / (flatcar_tc_length * 0.5)

            return RF_flatcar_1*1.e-4

        def get_constraints_hill(inputs):
            # Constraint function to make sure the blade does not exceed the maximum strains while staying within the vertical clearance during a summit curve
            q_iter = q_opt_v * inputs[:-2]
            V_iter = np.zeros(n_opt)
            M_iter = np.zeros(n_opt)
            for i in range(n_opt):
                V_iter[i] = np.trapz(q_iter[i:],r_opt[i:])
            for i in range(n_opt):
                M_iter[i] = np.trapz(V_iter[i:],r_opt[i:])

            eps            = M_iter * dist_te_interp / EIedge_interp
            consts_strains = (max_strains - abs(eps))*1.e+3

            p_loads      = _pBEAM.Loads(n_opt, np.zeros_like(r_opt), q_iter, np.zeros_like(r_opt))
            blade = _pBEAM.Beam(p_section, p_loads, p_tip, p_base)
            dx, dy, dz, dtheta_r1, dtheta_r2, dtheta_z = blade.displacement()

            x_blade_transport = dy*blade_length/arc_length(r_opt, dy)[-1] - dist_le[0] - deck_height
            y_blade_transport = r_opt*blade_length/arc_length(r_opt, dy)[-1]

            le_x = x_blade_transport + dist_le_interp
            te_x = x_blade_transport - dist_te_interp
            le_y = te_y = y_blade_transport

            # Rotation
            root_rot_rad_iter = inputs[-1]
            le_x_rot  = le_x*np.cos(root_rot_rad_iter) - le_y*np.sin(root_rot_rad_iter)
            le_y_rot  = le_y*np.cos(root_rot_rad_iter) + le_x*np.sin(root_rot_rad_iter)
            
            te_x_rot  = te_x*np.cos(root_rot_rad_iter) - te_y*np.sin(root_rot_rad_iter)
            te_y_rot  = te_y*np.cos(root_rot_rad_iter) + te_x*np.sin(root_rot_rad_iter)

            # Translation
            le_x_transl = le_x_rot - inputs[-2]
            te_x_transl = te_x_rot - inputs[-2]
            le_y_transl = te_y_transl = le_y_rot = te_y_rot


            id_upper = np.zeros(n_opt, dtype = int)
            id_deck  = np.zeros(n_opt, dtype = int)
            for i in range(n_opt):
                id_upper[i] = np.argmin(abs(y_upper_hill[:int(np.ceil(n_points*0.5))] - te_y_transl[i]))
                id_deck[i]  = np.argmin(abs(y_deck_hill[:int(np.ceil(n_points*0.5))]  - le_y_transl[i]))

            consts_envelope_upper = te_x_transl - x_upper_hill[id_upper]
            consts_envelope_deck  = x_deck_hill[id_deck] - le_x_transl

            consts = np.hstack((consts_strains, consts_envelope_upper, consts_envelope_deck))

            return consts

        def get_constraints_sag(inputs):
            # Constraint function to make sure the blade does not exceed the maximum strains while staying within the vertical clearance during a sag curve
            q_iter = q_opt_v * inputs[:-2]
            V_iter = np.zeros(n_opt)
            M_iter = np.zeros(n_opt)
            for i in range(n_opt):
                V_iter[i] = np.trapz(q_iter[i:],r_opt[i:])
            for i in range(n_opt):
                M_iter[i] = np.trapz(V_iter[i:],r_opt[i:])

            eps            = - M_iter * dist_te_interp / EIedge_interp
            consts_strains = (max_strains - abs(eps))*1.e+3

            p_section    = _pBEAM.SectionData(n_opt, r_opt, EA_interp, EIedge_interp, EIflap_interp, GJ_interp, rhoA_interp, rhoJ_interp)
            p_tip        = _pBEAM.TipData()  # no tip mass
            p_base       = _pBEAM.BaseData(np.ones(6), 1.0)  # rigid base
            p_loads      = _pBEAM.Loads(n_opt, np.zeros_like(r_opt), q_iter, np.zeros_like(r_opt))
            blade = _pBEAM.Beam(p_section, p_loads, p_tip, p_base)
            dx, dy, dz, dtheta_r1, dtheta_r2, dtheta_z = blade.displacement()

            x_blade_transport = dy*blade_length/arc_length(r_opt, dy)[-1] + dist_le[0] + deck_height
            y_blade_transport = r_opt*blade_length/arc_length(r_opt, dy)[-1]

            le_x = x_blade_transport - dist_le_interp
            te_x = x_blade_transport + dist_te_interp
            le_y = te_y = y_blade_transport

            # Rotation
            root_rot_rad_iter = inputs[-1]
            le_x_rot  = le_x*np.cos(root_rot_rad_iter) - le_y*np.sin(root_rot_rad_iter)
            le_y_rot  = le_y*np.cos(root_rot_rad_iter) + le_x*np.sin(root_rot_rad_iter)
            
            te_x_rot  = te_x*np.cos(root_rot_rad_iter) - te_y*np.sin(root_rot_rad_iter)
            te_y_rot  = te_y*np.cos(root_rot_rad_iter) + te_x*np.sin(root_rot_rad_iter)

            # Translation
            le_x_transl = le_x_rot + inputs[-2]
            te_x_transl = te_x_rot + inputs[-2]
            le_y_transl = te_y_transl = le_y_rot = te_y_rot


            id_upper = np.zeros(n_opt, dtype = int)
            id_deck  = np.zeros(n_opt, dtype = int)
            for i in range(n_opt):
                id_upper[i] = np.argmin(abs(y_upper_sag[:int(np.ceil(n_points*0.5))] - te_y_transl[i]))
                id_deck[i]  = np.argmin(abs(y_deck_sag[:int(np.ceil(n_points*0.5))]  - le_y_transl[i]))

            consts_envelope_upper = x_upper_sag[id_upper] - te_x_transl
            consts_envelope_deck  = le_x_transl - x_deck_sag[id_deck]

            consts = np.hstack((consts_strains, consts_envelope_upper, consts_envelope_deck))

            return consts

        x0    = np.hstack((np.ones(n_opt), np.zeros(2)))

        bnds = ((0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.), (0., vertical_clearance - deck_height - 2*dist_le[0]), (-max_root_rot_deg / 180. * np.pi, max_root_rot_deg / 180. * np.pi))
        const           = {}
        const['type']   = 'ineq'
        const['fun']    = get_constraints_hill
        res_hill        = minimize(get_max_force_v, x0, method='SLSQP', bounds=bnds, constraints=const)

        const['fun']    = get_constraints_sag
        res_sag         = minimize(get_max_force_v, x0, method='SLSQP', bounds=bnds, constraints=const)
        
        if res_hill.success == False:
            # If the optimization does not find a solution, assign a high value to the the fields LV_constraint_8axle, which are typically imposed lower than 1 in an outer blade design loop
            outputs['LV_constraint_8axle_vert'] = 2.
            print('The optimization cannot satisfy the constraint on max strains of 3500 mu eps for the hill case.')
        elif res_sag.success == False:
            # If the optimization does not find a solution, assign a high value to the the fields LV_constraint_8axle, which are typically imposed lower than 1 in an outer blade design loop
            outputs['LV_constraint_8axle_vert'] = 2.
            print('The optimization cannot satisfy the constraint on max strains of 3500 mu eps for the sag case')
        else:
             # If the optimization does converge, integrate the distributed loading twice to obtain the bending moment and the strains in the trailing edge along span
            q_hill    = q_opt_v * res_hill.x[:-2]
            V_hill    = np.zeros(n_opt)
            M_hill    = np.zeros(n_opt)
            for i in range(n_opt):
                V_hill[i] = np.trapz(q_hill[i:],r_opt[i:])
            for i in range(n_opt):
                M_hill[i] = np.trapz(V_hill[i:],r_opt[i:])

            q_sag    = q_opt_v * res_sag.x[:-2]
            V_sag    = np.zeros(n_opt)
            M_sag    = np.zeros(n_opt)
            for i in range(n_opt):
                V_sag[i] = np.trapz(q_sag[i:],r_opt[i:])
            for i in range(n_opt):
                M_sag[i] = np.trapz(V_sag[i:],r_opt[i:])
            
            # print('The optimizer finds a solution for the vertical curves!')
            
            # Compute the reaction forces for hill and sag cases
            RF_flatcar_1_hill = 0.5 * V_hill[0] + M_hill[0] / (flatcar_tc_length)
            RF_flatcar_1_sag  = 0.5 * V_sag[0] + M_sag[0] / (flatcar_tc_length)

            # print('Max reaction force from hill: ' + str(RF_flatcar_1_hill) + ' N')
            # print('Max reaction force from sag: '  + str(RF_flatcar_1_sag) + ' N')

            outputs['LV_constraint_8axle_vert'] = (RF_flatcar_1_hill / (weight_car_8axle * gravity)) / max_LV
            outputs['LV_constraint_4axle_vert'] = (RF_flatcar_1_hill / (weight_car_4axle * gravity)) / max_LV

            # print('L/V constraint 8-axle: ' + str(outputs['LV_constraint_8axle_vert']))
            # print('L/V constraint 4-axle: ' + str(outputs['LV_constraint_4axle_vert']))
