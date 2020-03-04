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
    # Openmdao component to run precomp and generate the elastic properties of a wind turbine blade
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
        self.add_output('LV_constraint_4axle', val=0.0, desc='Constraint for max L/V for a 4-axle flatcar, violated when bigger than 1')
        self.add_output('LV_constraint_8axle', val=0.0, desc='Constraint for max L/V for an 8-axle flatcar, violated when bigger than 1')


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
            # time0 = time.time()
        
            ## Profiles
            # rotate            
            profile_i = inputs['coord_xy_interp'][i,:,:]
            profile_i_rot = np.column_stack(rotate(inputs['pitch_axis'][i], 0., profile_i[:,0], profile_i[:,1], np.radians(inputs['theta'][i])))

            # import matplotlib.pyplot as plt
            # plt.plot(profile_i[:,0], profile_i[:,1])
            # plt.plot(profile_i_rot[:,0], profile_i_rot[:,1])
            # plt.axis('equal')
            # plt.title(i)
            # plt.show()

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

            # plt.plot(profile_i_rot_precomp[:,0], profile_i_rot_precomp[:,1])
            # plt.axis('equal')
            # plt.title(i)
            # plt.show()

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

            # plt.figure()
            # plt.plot(xnode, ynode, label='original')
            # # plt.plot(xnode_no_theta, ynode_no_theta, label='original no twist')
            # plt.plot(0.,0., 'ro', label='PA')
            # plt.plot(xnode_dim, ynode_dim, label='Profile around pa')
            # plt.plot(xnode_dim_no_theta, ynode_dim_no_theta, label='Profile around pa no twist')
            # plt.xlabel('x')
            # plt.ylabel('y')
            # plt.axis('equal')
            # plt.legend()

            # plt.show()

            x_ec = inputs['x_ec_abs'][i]
            y_ec = inputs['y_ec_abs'][i]

            AE[i] = max(ynode_dim) - y_ec
            EB[i] = y_ec - min(ynode_dim)
            EC[i] = max(xnode_dim) - x_ec
            DE[i] = x_ec - min(xnode_dim)

            # plt.figure()
            # plt.plot(xnode_dim, ynode_dim, label='Profile around pa')
            # plt.plot(0.,0., 'ro', label='PA')
            # plt.plot(x_ec,y_ec, 'r*', label='EC')
            # plt.plot(x_ec,y_ec + AE[i], 'g+', label='A')
            # plt.plot(x_ec,y_ec - EB[i], 'go', label='B')
            # plt.plot(x_ec + EC[i],y_ec, 'gv', label='C')
            # plt.plot(x_ec - DE[i],y_ec, 'g*', label='D')
            # plt.xlabel('x')
            # plt.ylabel('y')
            # plt.axis('equal')
            # plt.legend()

            # plt.show()


        dist_ss  = AE
        dist_ps  = EB

        M        = np.array(max_strains * EIflap / dist_ss)
        V        = -np.gradient(M,r)
        q        = -np.gradient(V,r)    

        r_opt    = np.linspace(0., blade_length, n_opt)
        pb_opt   = np.interp(r_opt, r, inputs['blade_ref_axis'][:,0])
        M_opt_h  = np.interp(r_opt, r, M)
        V_opt_h  = np.gradient(M_opt_h,r_opt)
        q_opt_h  = np.max([np.zeros(n_opt), np.gradient(V_opt_h,r_opt)], axis=0)

        # f, axes = plt.subplots(3,1,figsize=(5.3, 5.3))
        # axes[0].plot(r, M * 1.e-6)
        # axes[0].plot(r_opt, M_opt_h * 1.e-6)
        # axes[0].set_ylabel('Moment [MNm]')
        # axes[1].plot(r, V*1.e-6)
        # axes[1].plot(r_opt, V_opt_h_h*1.e-6)
        # axes[1].set_ylabel('Shear Forces [MN]')
        # axes[2].plot(r, q*1.e-3)
        # axes[2].plot(r_opt, q_opt_h_h*1.e-3)
        # axes[2].set_ylabel('Distributed Forces [kN/m]')
        # plt.xlabel("Blade span [m]")
        # plt.show()

        r_midline = radius
        r_outer   = r_midline + 0.5*lateral_clearance
        r_inner   = r_midline - 0.5*lateral_clearance

        x_rail_h  = np.linspace(0., 2.*r_midline, n_points)
        y_rail_h  = np.sqrt(r_midline**2. - (x_rail_h-r_midline)**2.)

        x_outer   = np.linspace(- 0.5*lateral_clearance, 2.*r_midline + 0.5*lateral_clearance, n_points)
        y_outer   = np.sqrt(r_outer**2. - (x_outer-r_midline)**2.)

        x_inner   = np.linspace(0.5*lateral_clearance, 2.*r_midline - 0.5*lateral_clearance, n_points)
        y_inner   = np.sqrt(r_inner**2. - (x_inner-r_midline)**2.)

        dist_ss_interp   = np.interp(r_opt, r, dist_ss)
        dist_ps_interp   = np.interp(r_opt, r, dist_ps)
        EIflap_interp    = np.interp(r_opt, r, EIflap)
        EIedge_interp    = np.interp(r_opt, r, EIedge)
        GJ_interp        = np.interp(r_opt, r, GJ)
        rhoA_interp      = np.interp(r_opt, r, rhoA)
        EA_interp        = np.interp(r_opt, r, EA)
        rhoJ_interp      = np.interp(r_opt, r, rhoJ)

        def get_max_force_h(inputs):
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

            # fs=10
            # f, ax = plt.subplots(1,1,figsize=(5.3, 5.3))
            # ax.plot(r_opt, eps)
            # ax.legend(fontsize=fs)
            # plt.xlabel('blade length [m]', fontsize=fs+2, fontweight='bold')
            # plt.ylabel('eps [-]', fontsize=fs+2, fontweight='bold')
            # plt.xticks(fontsize=fs)
            # plt.yticks(fontsize=fs)
            # plt.grid(color=[0.8,0.8,0.8], linestyle='--')
            # plt.subplots_adjust(bottom = 0.15, left = 0.18)
            # plt.show()

            

            p_section    = _pBEAM.SectionData(n_opt, r_opt, EA_interp, EIedge_interp, EIflap_interp, GJ_interp, rhoA_interp, rhoJ_interp)
            p_tip        = _pBEAM.TipData()  # no tip mass
            p_base       = _pBEAM.BaseData(np.ones(6), 1.0)  # rigid base
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

            consts = np.hstack((consts_strains, consts_envelope_outer, consts_envelope_inner))

            # fs=10
            # f, ax = plt.subplots(1,1,figsize=(5.3, 5.3))
            # ax.plot(x_rail_h, y_rail_h,   color=[0.8,0.8,0.8], linestyle='--', label='rail midline')
            # ax.plot(x_outer, y_outer, color=[0.8,0.8,0.8], linestyle=':', label='clearance envelope')
            # ax.plot(x_inner, y_inner, color=[0.8,0.8,0.8], linestyle=':')
            # ax.plot(x_blade_transport, y_blade_transport, label='blade max strains')
            # ax.plot(ps_x_rot, ps_y_rot, label='pressured side')
            # ax.plot(ss_x_rot, ss_y_rot, label='suction side')
            # plt.xlim(left=-10, right=110)
            # plt.ylim(bottom=0, top=120)
            # ax.legend(fontsize=fs)
            # plt.xlabel('x [m]', fontsize=fs+2, fontweight='bold')
            # plt.ylabel('y [m]', fontsize=fs+2, fontweight='bold')
            # plt.xticks(fontsize=fs)
            # plt.yticks(fontsize=fs)
            # plt.grid(color=[0.8,0.8,0.8], linestyle='--')
            # plt.subplots_adjust(bottom = 0.15, left = 0.18)
            # plt.show()
            # exit()


            return consts

        x0    = np.hstack((np.ones(n_opt), 0.))

        bnds = ((0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(-max_root_rot_deg / 180. * np.pi, max_root_rot_deg / 180. * np.pi))
        const           = {}
        const['type']   = 'ineq'
        const['fun']    = get_constraints_h
        res    = minimize(get_max_force_h, x0, method='SLSQP', bounds=bnds, constraints=const)

        if res.success == False:
            outputs['LV_constraint_8axle'] = 2.
            outputs['LV_constraint_4axle'] = 2.
            exit('The optimization cannot satisfy the constraint on max strains of 3500 mu eps')
        else:
            
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
            
            RF_flatcar_1 = 0.5 * V_final[0] + M_final[0] / flatcar_tc_length

            # print('Max reaction force lateral turn: ' + str(RF_flatcar_1) + ' N')

            outputs['LV_constraint_8axle'] = (RF_flatcar_1 / (0.5 * weight_car_8axle * gravity)) / max_LV
            outputs['LV_constraint_4axle'] = (RF_flatcar_1 / (0.5 * weight_car_4axle * gravity)) / max_LV

            print('L/V constraint 8-axle: ' + str(outputs['LV_constraint_8axle']))
            # print('L/V constraint 4-axle: ' + str(outputs['LV_constraint_4axle']))

            eps            = M_final * dist_ss_interp / EIflap_interp

            p_section    = _pBEAM.SectionData(n_opt, r_opt, EA_interp, EIedge_interp, EIflap_interp, GJ_interp, rhoA_interp, rhoJ_interp)
            p_tip        = _pBEAM.TipData()  # no tip mass
            p_base       = _pBEAM.BaseData(np.ones(6), 1.0)  # rigid base
            p_loads      = _pBEAM.Loads(n_opt, q_final, np.zeros_like(r_opt), np.zeros_like(r_opt))
            blade = _pBEAM.Beam(p_section, p_loads, p_tip, p_base)
            dx, dy, dz, dtheta_r1, dtheta_r2, dtheta_z = blade.displacement()

            x_blade_transport = (dx +  + pb_opt)*blade_length/arc_length(r_opt, dx)[-1]
            y_blade_transport = r_opt*blade_length/arc_length(r_opt, dx)[-1]


            ps_x = x_blade_transport + dist_ps_interp
            ss_x = x_blade_transport - dist_ss_interp
            ps_y = ss_y = y_blade_transport

            ps_x_rot  = ps_x*np.cos(root_rot_rad_final) - ps_y*np.sin(root_rot_rad_final)
            ps_y_rot  = ps_y*np.cos(root_rot_rad_final) + ps_x*np.sin(root_rot_rad_final)
            
            ss_x_rot  = ss_x*np.cos(root_rot_rad_final) - ss_y*np.sin(root_rot_rad_final)
            ss_y_rot  = ss_y*np.cos(root_rot_rad_final) + ss_x*np.sin(root_rot_rad_final)

            x_blade_transport_rot = x_blade_transport*np.cos(root_rot_rad_final) - y_blade_transport*np.sin(root_rot_rad_final)
            y_blade_transport_rot = y_blade_transport*np.cos(root_rot_rad_final) + x_blade_transport*np.sin(root_rot_rad_final)


            fs=12

            # f, ax = plt.subplots(1,1,figsize=(5.3, 5.3))
            # ax.bar(np.array([1,2,3,4]), RF_carts*1.e-3)
            # plt.xlabel('Cart [-]', fontsize=fs+2, fontweight='bold')
            # plt.ylabel('Lateral Forces [kN]', fontsize=fs+2, fontweight='bold')
            # plt.xticks(fontsize=fs)
            # plt.yticks(fontsize=fs)
            # from matplotlib.ticker import MaxNLocator
            # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            # plt.ylim(bottom=0, top=300)
            # plt.grid(color=[0.8,0.8,0.8], linestyle='--')
            # plt.subplots_adjust(bottom = 0.15, left = 0.18)


            # f, ax = plt.subplots(1,1,figsize=(5.3, 5.3))
            # ax.plot(r_opt, eps)
            # ax.legend(fontsize=fs)
            # plt.xlabel('Blade span position [m]', fontsize=fs+2, fontweight='bold')
            # plt.ylabel('Strains [-]', fontsize=fs+2, fontweight='bold')
            # plt.xticks(fontsize=fs)
            # plt.yticks(fontsize=fs)
            # plt.grid(color=[0.8,0.8,0.8], linestyle='--')
            # plt.subplots_adjust(bottom = 0.15, left = 0.18)

            rail_spacing = 1.4351
            x_r_outer  = np.linspace(- 0.5*rail_spacing, 2.*r_midline + 0.5*rail_spacing, n_points)
            r_r_outer  = r_midline + 0.5*rail_spacing
            y_r_outer    = np.sqrt(r_r_outer**2. - (x_r_outer-r_midline)**2. + 1.e-5)

            x_r_inner  = np.linspace( 0.5*rail_spacing, 2.*r_midline - 0.5*rail_spacing, n_points)
            r_r_inner  = r_midline - 0.5*rail_spacing
            y_r_inner  = np.sqrt(r_r_inner**2. - (x_r_inner-r_midline)**2. + 1.e-5)

            # f, ax = plt.subplots(1,1,figsize=(5.3, 5.3))
            # # ax.plot(x_rail_h, y_rail_h,   color='dimgray', linestyle='--', label='Rail midline')
            # ax.plot(x_r_outer, y_r_outer,   color='dimgray', linestyle='--',linewidth=1., label='Railway')
            # ax.plot(x_r_inner, y_r_inner,   color='dimgray', linestyle='--',linewidth=1.,)
            # ax.plot(x_outer, y_outer, color='dimgray', linestyle=':',linewidth=1., label='Lateral Clearance Profile')
            # ax.plot(x_inner, y_inner, color='dimgray', linestyle=':',linewidth=1.)
            # # ax.plot(x_blade_transport_rot, y_blade_transport_rot, label='Blade pitch axis')
            # ax.plot(ps_x_rot, ps_y_rot, color='tab:red', label='Blade Pressure Side')
            # ax.plot(ss_x_rot, ss_y_rot, color='tab:blue', label='Blade Suction Side')
            # plt.xlim(left=-10, right=110)
            # plt.ylim(bottom=0, top=120)
            # ax.legend(fontsize=fs)
            # plt.xlabel('x [m]', fontsize=fs+2, fontweight='bold')
            # plt.ylabel('y [m]', fontsize=fs+2, fontweight='bold')
            # plt.xticks(fontsize=fs)
            # plt.yticks(fontsize=fs)
            # plt.grid(color=[0.8,0.8,0.8], linestyle='--')
            # plt.subplots_adjust(bottom = 0.15, left = 0.18)
            # plt.show()
            # f.savefig('/Users/pbortolo/Dropbox/Writing/TORQUE2020/rail/' + 'h' + str(int(inputs['horizontal_angle_deg'][0])) + '.pdf')
            # exit()
            # f, axes = plt.subplots(3,1,figsize=(5.3, 5.3))
            # axes[0].plot(r, M * 1.e-6, 'b-', label = '3500 mu eps')
            # axes[0].plot(r_opt, M_final * 1.e-6, 'r--', label = 'Final')
            # axes[0].set_ylabel('Moment [MNm]')
            # axes[0].legend(fontsize=fs)
            # axes[1].plot(r, V*1.e-6, 'b-', label = '3500 mu eps')
            # axes[1].plot(r_opt, V_final*1.e-6, 'r--', label = 'Final')
            # axes[1].legend(fontsize=fs)
            # axes[1].set_ylabel('Shear Forces [MN]')
            # axes[2].plot(r, q*1.e-3, 'b-', label = '3500 mu eps')
            # axes[2].plot(r_opt, q_final*1.e-3, 'r--', label = 'Final')
            # axes[2].set_ylabel('Distributed Forces [kN/m]')
            # axes[2].legend(fontsize=fs)
            # plt.xlabel("Blade span [m]")
            # plt.show()


        # Vertical turns - hill
        vertical_clearance      = inputs['vertical_clearance'][0]
        deck_height             = inputs['deck_height'][0]
        min_vertical_radius     = inputs['min_vertical_radius'][0]

        dist_le  = DE
        dist_te  = EC

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

        x_deck_hill   = np.linspace(-deck_height, 2.*r_deck_hill - deck_height, n_points)
        y_deck_hill   = np.sqrt(r_deck_hill**2. - (x_deck_hill-r_rail)**2.)

        x_upper_hill  = np.linspace(-vertical_clearance, 2.*r_upper_hill - vertical_clearance, n_points)
        y_upper_hill  = np.sqrt(r_upper_hill**2. - (x_upper_hill-r_rail)**2. + 1.e-5)

        x_deck_sag    = np.linspace(deck_height, 2.*r_deck_sag + deck_height, n_points)
        y_deck_sag   = np.sqrt(r_deck_sag**2. - (x_deck_sag-r_rail)**2.)

        x_upper_sag   = np.linspace(vertical_clearance, 2.*r_upper_sag + vertical_clearance, n_points)
        y_upper_sag   = np.sqrt(r_upper_sag**2. - (x_upper_sag-r_rail)**2. + 1.e-5)

        def get_max_force_v(inputs):
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
            q_iter = q_opt_v * inputs[:-2]
            V_iter = np.zeros(n_opt)
            M_iter = np.zeros(n_opt)
            for i in range(n_opt):
                V_iter[i] = np.trapz(q_iter[i:],r_opt[i:])
            for i in range(n_opt):
                M_iter[i] = np.trapz(V_iter[i:],r_opt[i:])

            eps            = M_iter * dist_te_interp / EIedge_interp
            consts_strains = (max_strains - abs(eps))*1.e+3

            # fs=10
            # f, ax = plt.subplots(1,1,figsize=(5.3, 5.3))
            # ax.plot(r_opt, eps)
            # ax.legend(fontsize=fs)
            # plt.xlabel('blade length [m]', fontsize=fs+2, fontweight='bold')
            # plt.ylabel('eps [-]', fontsize=fs+2, fontweight='bold')
            # plt.xticks(fontsize=fs)
            # plt.yticks(fontsize=fs)
            # plt.grid(color=[0.8,0.8,0.8], linestyle='--')
            # plt.subplots_adjust(bottom = 0.15, left = 0.18)
            # plt.show()

            p_section    = _pBEAM.SectionData(n_opt, r_opt, EA_interp, EIedge_interp, EIflap_interp, GJ_interp, rhoA_interp, rhoJ_interp)
            p_tip        = _pBEAM.TipData()  # no tip mass
            p_base       = _pBEAM.BaseData(np.ones(6), 1.0)  # rigid base
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

            # fs=10
            # f, ax = plt.subplots(1,1,figsize=(5.3, 5.3))
            # ax.plot(x_rail_v, y_rail_v,   color=[0.8,0.8,0.8], linestyle='--', label='rail line')
            # ax.plot(x_deck_hill, y_deck_hill, color=[0.8,0.8,0.8], linestyle=':', label='deck line')
            # ax.plot(x_upper_hill, y_upper_hill, color=[0.8,0.8,0.8], linestyle=':', label='upper envelope')
            # ax.plot(le_x_transl, le_y_transl, label='leading edge')
            # ax.plot(te_x_transl, te_y_transl, label='trailing edge')
            # plt.xlim(left=-10, right=110)
            # plt.ylim(bottom=0, top=120)
            # ax.legend(fontsize=fs)
            # plt.xlabel('x [m]', fontsize=fs+2, fontweight='bold')
            # plt.ylabel('y [m]', fontsize=fs+2, fontweight='bold')
            # plt.xticks(fontsize=fs)
            # plt.yticks(fontsize=fs)
            # plt.grid(color=[0.8,0.8,0.8], linestyle='--')
            # plt.subplots_adjust(bottom = 0.15, left = 0.18)
            # plt.show()


            return consts

        def get_constraints_sag(inputs):
            q_iter = q_opt_v * inputs[:-2]
            V_iter = np.zeros(n_opt)
            M_iter = np.zeros(n_opt)
            for i in range(n_opt):
                V_iter[i] = np.trapz(q_iter[i:],r_opt[i:])
            for i in range(n_opt):
                M_iter[i] = np.trapz(V_iter[i:],r_opt[i:])

            eps            = - M_iter * dist_te_interp / EIedge_interp
            consts_strains = (max_strains - abs(eps))*1.e+3

            # fs=10
            # f, ax = plt.subplots(1,1,figsize=(5.3, 5.3))
            # ax.plot(r_opt, eps)
            # ax.legend(fontsize=fs)
            # plt.xlabel('blade length [m]', fontsize=fs+2, fontweight='bold')
            # plt.ylabel('eps [-]', fontsize=fs+2, fontweight='bold')
            # plt.xticks(fontsize=fs)
            # plt.yticks(fontsize=fs)
            # plt.grid(color=[0.8,0.8,0.8], linestyle='--')
            # plt.subplots_adjust(bottom = 0.15, left = 0.18)
            # plt.show()

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

            # fs=10
            # f, ax = plt.subplots(1,1,figsize=(5.3, 5.3))
            # ax.plot(x_rail_v, y_rail_v,   color=[0.8,0.8,0.8], linestyle='--', label='rail line')
            # ax.plot(x_deck_sag, y_deck_sag, color=[0.8,0.8,0.8], linestyle=':', label='deck line')
            # ax.plot(x_upper_sag, y_upper_sag, color=[0.8,0.8,0.8], linestyle=':', label='upper envelope')
            # ax.plot(le_x_transl, le_y_transl, label='leading edge')
            # ax.plot(te_x_transl, te_y_transl, label='trailing edge')
            # plt.xlim(left=-10, right=110)
            # plt.ylim(bottom=0, top=120)
            # ax.legend(fontsize=fs)
            # plt.xlabel('x [m]', fontsize=fs+2, fontweight='bold')
            # plt.ylabel('y [m]', fontsize=fs+2, fontweight='bold')
            # plt.xticks(fontsize=fs)
            # plt.yticks(fontsize=fs)
            # plt.grid(color=[0.8,0.8,0.8], linestyle='--')
            # plt.subplots_adjust(bottom = 0.15, left = 0.18)
            # plt.show()


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
            outputs['LV_constraint_8axle'] = 2.
            print('The optimization cannot satisfy the constraint on max strains of 3500 mu eps for the hill case.')
        elif res_sag.success == False:
            outputs['LV_constraint_8axle'] = 2.
            print('The optimization cannot satisfy the constraint on max strains of 3500 mu eps for the sag case')
        else:
            
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
            
            RF_flatcar_1_hill = 0.5 * V_hill[0] + M_hill[0] / (flatcar_tc_length)
            RF_flatcar_1_sag  = 0.5 * V_sag[0] + M_sag[0] / (flatcar_tc_length)

            # print('Max reaction force from hill: ' + str(RF_flatcar_1_hill) + ' N')
            # print('Max reaction force from sag: '  + str(RF_flatcar_1_sag) + ' N')

            # outputs['LV_constraint_8axle'] = (RF_flatcar_1_hill / (weight_car_8axle * gravity)) / max_LV
            # outputs['LV_constraint_4axle'] = (RF_flatcar_1_hill / (weight_car_4axle * gravity)) / max_LV

            # print('L/V constraint 8-axle: ' + str(outputs['LV_constraint_8axle']))
            # print('L/V constraint 4-axle: ' + str(outputs['LV_constraint_4axle']))

            eps            = M_hill * dist_te_interp / EIedge_interp

            p_section    = _pBEAM.SectionData(n_opt, r_opt, EA_interp, EIedge_interp, EIflap_interp, GJ_interp, rhoA_interp, rhoJ_interp)
            p_tip        = _pBEAM.TipData()  # no tip mass
            p_base       = _pBEAM.BaseData(np.ones(6), 1.0)  # rigid base
            p_loads      = _pBEAM.Loads(n_opt, np.zeros_like(r_opt), q_hill, np.zeros_like(r_opt))
            blade = _pBEAM.Beam(p_section, p_loads, p_tip, p_base)
            dx, dy, dz, dtheta_r1, dtheta_r2, dtheta_z = blade.displacement()

            x_blade_transport = dy*blade_length/arc_length(r_opt, dy)[-1] - dist_le[0] - deck_height
            y_blade_transport = r_opt*blade_length/arc_length(r_opt, dy)[-1]

            le_x = x_blade_transport + dist_le_interp
            te_x = x_blade_transport - dist_te_interp
            le_y = te_y = y_blade_transport

            # Rotation
            root_rot_rad_hill = res_hill.x[-1]
            # print('Prescribed rotation angle hill-case: ' + str(root_rot_rad_hill * 180. / np.pi) + ' deg')
            le_x_rot  = le_x*np.cos(root_rot_rad_hill) - le_y*np.sin(root_rot_rad_hill)
            le_y_rot  = le_y*np.cos(root_rot_rad_hill) + le_x*np.sin(root_rot_rad_hill)
            
            te_x_rot  = te_x*np.cos(root_rot_rad_hill) - te_y*np.sin(root_rot_rad_hill)
            te_y_rot  = te_y*np.cos(root_rot_rad_hill) + te_x*np.sin(root_rot_rad_hill)

            # Translation
            # print('Prescribed translation from deck hill-case: ' + str(res_hill.x[-2]) + ' m')
            le_x_transl = le_x_rot - res_hill.x[-2]
            te_x_transl = te_x_rot - res_hill.x[-2]
            le_y_transl = te_y_transl = le_y_rot = te_y_rot

            id_upper = np.zeros(n_opt, dtype = int)
            id_deck  = np.zeros(n_opt, dtype = int)
            for i in range(n_opt):
                id_upper[i] = np.argmin(abs(y_upper_hill[:int(np.ceil(n_points*0.5))] - te_y_transl[i]))
                id_deck[i]  = np.argmin(abs(y_deck_hill[:int(np.ceil(n_points*0.5))]  - le_y_transl[i]))

            consts_envelope_upper = te_x_transl - x_upper_hill[id_upper]
            consts_envelope_deck  = x_deck_hill[id_deck] - le_x_transl


            # ang = -np.pi*0.46
            # off_y = 20
            # off_x = 0
            # fs=12
            # f, ax = plt.subplots(1,1,figsize=(5.3, 5.3))
            # ax.plot(x_rail_v*np.cos(ang) - y_rail_v*np.sin(ang) + off_x, y_rail_v*np.cos(ang) + x_rail_v*np.sin(ang) + off_y,   color='dimgray', linestyle='--', linewidth=1.,label='Railway')
            # ax.plot(x_deck_hill*np.cos(ang) - y_deck_hill*np.sin(ang) + off_x, y_deck_hill*np.cos(ang) + x_deck_hill*np.sin(ang) + off_y,   color='dimgray', linestyle=':',linewidth=1., label='Deck Line')
            # ax.plot(x_upper_hill*np.cos(ang) - y_upper_hill*np.sin(ang) + off_x, y_upper_hill*np.cos(ang) + x_upper_hill*np.sin(ang) + off_y,   color='dimgray', linestyle='-.',linewidth=1., label='Vertical Clearance Profile')
            # ax.plot(le_x_transl*np.cos(ang) - le_y_transl*np.sin(ang) + off_x, le_y_transl*np.cos(ang) + le_x_transl*np.sin(ang) + off_y,   color='tab:red', label='Leading Edge')
            # ax.plot(te_x_transl*np.cos(ang) - te_y_transl*np.sin(ang) + off_x, te_y_transl*np.cos(ang) + te_x_transl*np.sin(ang) + off_y,   color='tab:blue', label='Trailing Edge')
            # plt.xlim(left=0, right=120)
            # plt.ylim(bottom=0, top=120)
            # ax.legend(fontsize=fs)
            # plt.xlabel('x [m]', fontsize=fs+2, fontweight='bold')
            # plt.ylabel('y [m]', fontsize=fs+2, fontweight='bold')
            # plt.xticks(fontsize=fs)
            # plt.yticks(fontsize=fs)
            # plt.grid(color=[0.8,0.8,0.8], linestyle='--')
            # plt.subplots_adjust(bottom = 0.15, left = 0.18)
            # f.savefig('/Users/pbortolo/Dropbox/Writing/TORQUE2020/rail/' + 'v_hill_rot.pdf')
            # plt.show()
            # exit()

            eps            = M_sag * dist_te_interp / EIedge_interp

            p_section    = _pBEAM.SectionData(n_opt, r_opt, EA_interp, EIedge_interp, EIflap_interp, GJ_interp, rhoA_interp, rhoJ_interp)
            p_tip        = _pBEAM.TipData()  # no tip mass
            p_base       = _pBEAM.BaseData(np.ones(6), 1.0)  # rigid base
            p_loads      = _pBEAM.Loads(n_opt, np.zeros_like(r_opt), q_sag, np.zeros_like(r_opt))
            blade = _pBEAM.Beam(p_section, p_loads, p_tip, p_base)
            dx, dy, dz, dtheta_r1, dtheta_r2, dtheta_z = blade.displacement()

            x_blade_transport = dy*blade_length/arc_length(r_opt, dy)[-1] + dist_le[0] + deck_height
            y_blade_transport = r_opt*blade_length/arc_length(r_opt, dy)[-1]

            le_x = x_blade_transport - dist_le_interp
            te_x = x_blade_transport + dist_te_interp
            le_y = te_y = y_blade_transport

            # Rotation
            root_rot_rad_sag = res_sag.x[-1]
            # print('Prescribed rotation angle sag-case: ' + str(root_rot_rad_sag * 180. / np.pi) + ' deg')
            le_x_rot  = le_x*np.cos(root_rot_rad_sag) - le_y*np.sin(root_rot_rad_sag)
            le_y_rot  = le_y*np.cos(root_rot_rad_sag) + le_x*np.sin(root_rot_rad_sag)
            
            te_x_rot  = te_x*np.cos(root_rot_rad_sag) - te_y*np.sin(root_rot_rad_sag)
            te_y_rot  = te_y*np.cos(root_rot_rad_sag) + te_x*np.sin(root_rot_rad_sag)

            # Translation
            # print('Prescribed translation from deck sag-case: ' + str(res_sag.x[-2]) + ' m')
            le_x_transl = le_x_rot + res_sag.x[-2]
            te_x_transl = te_x_rot + res_sag.x[-2]
            le_y_transl = te_y_transl = le_y_rot = te_y_rot

            # fs=10
            # f, ax = plt.subplots(1,1,figsize=(5.3, 5.3))
            # ax.plot(x_rail_v, y_rail_v,   color=[0.8,0.8,0.8], linestyle='--', label='rail line')
            # ax.plot(x_deck_sag, y_deck_sag, color=[0.8,0.8,0.8], linestyle=':', label='deck line')
            # ax.plot(x_upper_sag, y_upper_sag, color=[0.8,0.8,0.8], linestyle=':', label='upper envelope')
            # ax.plot(le_x_transl, le_y_transl, label='leading edge')
            # ax.plot(te_x_transl, te_y_transl, label='trailing edge')
            # plt.xlim(left=-10, right=110)
            # plt.ylim(bottom=0, top=120)
            # ax.legend(fontsize=fs)
            # plt.xlabel('x [m]', fontsize=fs+2, fontweight='bold')
            # plt.ylabel('y [m]', fontsize=fs+2, fontweight='bold')
            # plt.xticks(fontsize=fs)
            # plt.yticks(fontsize=fs)
            # plt.grid(color=[0.8,0.8,0.8], linestyle='--')
            # plt.subplots_adjust(bottom = 0.15, left = 0.18)
            # plt.show()

            # fs=10
            # f, ax = plt.subplots(1,1,figsize=(5.3, 5.3))
            # ax.plot(x_rail_v, y_rail_v,   color='dimgray', linestyle='--', linewidth=1.,label='Railway')
            # ax.plot(x_deck_sag, y_deck_sag, color='dimgray', linestyle=':',linewidth=1., label='Deck line')
            # ax.plot(x_upper_sag, y_upper_sag, color='dimgray', linestyle='-.',linewidth=1., label='Upper envelope')
            # ax.plot(le_x_transl, le_y_transl, color='tab:red', label='Leading edge')
            # ax.plot(te_x_transl, te_y_transl, color='tab:blue', label='Trailing edge')
            # plt.xlim(left=-10, right=110)
            # plt.ylim(bottom=0, top=120)
            # # ax.legend(fontsize=fs)
            # plt.xlabel('x [m]', fontsize=fs+2, fontweight='bold')
            # plt.ylabel('y [m]', fontsize=fs+2, fontweight='bold')
            # plt.xticks(fontsize=fs)
            # plt.yticks(fontsize=fs)
            # plt.grid(color=[0.8,0.8,0.8], linestyle='--')
            # plt.subplots_adjust(bottom = 0.15, left = 0.18)
            # f.savefig('/Users/pbortolo/Dropbox/Writing/TORQUE2020/rail/' + 'v_sag.pdf')
            # plt.show()
            # exit()

            # ang = np.pi*0.53
            # off_y = 20
            # off_x = 100
            # fs=12
            # f, ax = plt.subplots(1,1,figsize=(5.3, 5.3))
            # ax.plot(x_rail_v*np.cos(ang) - y_rail_v*np.sin(ang) + off_x, y_rail_v*np.cos(ang) + x_rail_v*np.sin(ang) + off_y,   color='dimgray', linestyle='--', linewidth=1.,label='Railway')
            # ax.plot(x_deck_sag*np.cos(ang) - y_deck_sag*np.sin(ang) + off_x, y_deck_sag*np.cos(ang) + x_deck_sag*np.sin(ang) + off_y,   color='dimgray', linestyle=':',linewidth=1., label='Deck Line')
            # ax.plot(x_upper_sag*np.cos(ang) - y_upper_sag*np.sin(ang) + off_x, y_upper_sag*np.cos(ang) + x_upper_sag*np.sin(ang) + off_y,   color='dimgray', linestyle='-.',linewidth=1., label='Vertical Clearance Envelope')
            # ax.plot(le_x_transl*np.cos(ang) - le_y_transl*np.sin(ang) + off_x, le_y_transl*np.cos(ang) + le_x_transl*np.sin(ang) + off_y,   color='tab:red', label='Leading Edge')
            # ax.plot(te_x_transl*np.cos(ang) - te_y_transl*np.sin(ang) + off_x, te_y_transl*np.cos(ang) + te_x_transl*np.sin(ang) + off_y,   color='tab:blue', label='Trailing Edge')
            # plt.xlim(left=-20, right=100)
            # plt.ylim(bottom=0, top=120)
            # # ax.legend(fontsize=fs)
            # plt.xlabel('x [m]', fontsize=fs+2, fontweight='bold')
            # plt.ylabel('y [m]', fontsize=fs+2, fontweight='bold')
            # plt.xticks(fontsize=fs)
            # plt.yticks(fontsize=fs)
            # plt.grid(color=[0.8,0.8,0.8], linestyle='--')
            # plt.subplots_adjust(bottom = 0.15, left = 0.18)
            # f.savefig('/Users/pbortolo/Dropbox/Writing/TORQUE2020/rail/' + 'v_sag_rot.pdf')
            # plt.show()
            # exit()

            # exit()

    


