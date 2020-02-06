import copy
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import PchipInterpolator
from openmdao.api import ExplicitComponent, Group
from wisdem.ccblade.ccblade_component import CCBladeLoads
import wisdem.ccblade._bem as _bem
from wisdem.commonse.utilities import rotate, arc_length
from wisdem.commonse.akima import Akima
from wisdem.commonse import gravity
from wisdem.commonse.csystem import DirectionVector
from wisdem.rotorse.precomp import PreComp, Profile, Orthotropic2DMaterial, CompositeSection
from wisdem.rotorse import RPM2RS, RS2RPM
import wisdem.pBeam._pBEAM as _pBEAM

from wisdem.rotorse.rotor_cost import blade_cost_model

class RunPreComp(ExplicitComponent):
    # Openmdao component to run precomp and generate the elastic properties of a wind turbine blade
    def initialize(self):
        self.options.declare('wt_init_options')
        self.options.declare('opt_options')

    def setup(self):
        blade_init_options = self.options['wt_init_options']['blade']
        self.n_span        = n_span    = blade_init_options['n_span']
        self.n_webs        = n_webs    = blade_init_options['n_webs']
        self.n_layers      = n_layers  = blade_init_options['n_layers']
        af_init_options    = self.options['wt_init_options']['airfoils']
        self.n_xy          = n_xy      = af_init_options['n_xy'] # Number of coordinate points to describe the airfoil geometry
        mat_init_options = self.options['wt_init_options']['materials']
        self.n_mat = n_mat = mat_init_options['n_mat']

        opt_options   = self.options['opt_options']
        self.te_ss_var   = opt_options['blade_struct']['te_ss_var']
        self.te_ps_var   = opt_options['blade_struct']['te_ps_var']
        self.spar_ss_var = opt_options['blade_struct']['spar_ss_var']
        self.spar_ps_var = opt_options['blade_struct']['spar_ps_var']

        # Outer geometry
        self.add_input('r',             val=np.zeros(n_span), units='m',   desc='radial locations where blade is defined (should be increasing and not go all the way to hub or tip)')
        self.add_input('theta',         val=np.zeros(n_span), units='deg', desc='Twist angle at each section (positive decreases angle of attack)')
        self.add_input('chord',         val=np.zeros(n_span), units='m',   desc='chord length at each section')
        self.add_input('pitch_axis',    val=np.zeros(n_span),                 desc='1D array of the chordwise position of the pitch axis (0-LE, 1-TE), defined along blade span.')
        self.add_input('precurve',      val=np.zeros(n_span),    units='m', desc='precurve at each section')
        self.add_input('presweep',      val=np.zeros(n_span),    units='m', desc='presweep at each section')
        self.add_input('coord_xy_interp',  val=np.zeros((n_span, n_xy, 2)), desc='3D array of the non-dimensional x and y airfoil coordinates of the airfoils interpolated along span for n_span stations.')

        # Inner structure
        self.add_discrete_input('web_name', val=n_webs * [''],                          desc='1D array of the names of the shear webs defined in the blade structure.')
        self.add_input('web_start_nd',   val=np.zeros((n_webs, n_span)),                desc='2D array of the non-dimensional start point defined along the outer profile of a web. The TE suction side is 0, the TE pressure side is 1. The first dimension represents each web, the second dimension represents each entry along blade span.')
        self.add_input('web_end_nd',     val=np.zeros((n_webs, n_span)),                desc='2D array of the non-dimensional end point defined along the outer profile of a web. The TE suction side is 0, the TE pressure side is 1. The first dimension represents each web, the second dimension represents each entry along blade span.')
        self.add_discrete_input('layer_web',  val=n_layers * [''],                      desc='1D array of the names of the webs the layer is associated to. If the layer is on the outer profile this entry can simply stay empty.')
        self.add_discrete_input('layer_name', val=n_layers * [''],                      desc='1D array of the names of the layers modeled in the blade structure.')
        self.add_discrete_input('layer_mat',  val=n_layers * [''],                      desc='1D array of the names of the materials of each layer modeled in the blade structure.')
        self.add_discrete_input('definition_layer', val=np.zeros(n_layers),                 desc='1D array of flags identifying how layers are specified in the yaml. 1) all around (skin, paint, ) 2) offset+rotation twist+width (spar caps) 3) offset+user defined rotation+width 4) midpoint TE+width (TE reinf) 5) midpoint LE+width (LE reinf) 6) layer position fixed to other layer (core fillers) 7) start and width 8) end and width 9) start and end nd 10) web layer')
        self.add_input('layer_thickness',   val=np.zeros((n_layers, n_span)), units='m',    desc='2D array of the thickness of the layers of the blade structure. The first dimension represents each layer, the second dimension represents each entry along blade span.')
        self.add_input('layer_start_nd',    val=np.zeros((n_layers, n_span)),               desc='2D array of the non-dimensional start point defined along the outer profile of a layer. The TE suction side is 0, the TE pressure side is 1. The first dimension represents each layer, the second dimension represents each entry along blade span.')
        self.add_input('layer_end_nd',      val=np.zeros((n_layers, n_span)),               desc='2D array of the non-dimensional end point defined along the outer profile of a layer. The TE suction side is 0, the TE pressure side is 1. The first dimension represents each layer, the second dimension represents each entry along blade span.')
        self.add_input('fiber_orientation',   val=np.zeros((n_layers, n_span)), units='deg',    desc='2D array of the orientation of the layers of the blade structure. The first dimension represents each layer, the second dimension represents each entry along blade span.')

        # Materials
        self.add_discrete_input('mat_name', val=n_mat * [''],                         desc='1D array of names of materials.')
        self.add_discrete_input('orth', val=np.zeros(n_mat),                      desc='1D array of flags to set whether a material is isotropic (0) or orthtropic (1). Each entry represents a material.')
        self.add_input('E',             val=np.zeros([n_mat, 3]), units='Pa',     desc='2D array of the Youngs moduli of the materials. Each row represents a material, the three columns represent E11, E22 and E33.')
        self.add_input('G',             val=np.zeros([n_mat, 3]), units='Pa',     desc='2D array of the shear moduli of the materials. Each row represents a material, the three columns represent G12, G13 and G23.')
        self.add_input('nu',            val=np.zeros([n_mat, 3]),                 desc='2D array of the Poisson ratio of the materials. Each row represents a material, the three columns represent nu12, nu13 and nu23.')
        self.add_input('rho',           val=np.zeros(n_mat),      units='kg/m**3',desc='1D array of the density of the materials. For composites, this is the density of the laminate.')


        # Outputs - Beam properties
        self.add_output('z',            val=np.zeros(n_span), units='m',      desc='locations of properties along beam')
        self.add_output('EA',           val=np.zeros(n_span), units='N',      desc='axial stiffness')
        self.add_output('EIxx',         val=np.zeros(n_span), units='N*m**2', desc='edgewise stiffness (bending about :ref:`x-direction of airfoil aligned coordinate system <blade_airfoil_coord>`)')
        self.add_output('EIyy',         val=np.zeros(n_span), units='N*m**2', desc='flatwise stiffness (bending about y-direction of airfoil aligned coordinate system)')
        self.add_output('EIxy',         val=np.zeros(n_span), units='N*m**2', desc='coupled flap-edge stiffness')
        self.add_output('GJ',           val=np.zeros(n_span), units='N*m**2', desc='torsional stiffness (about axial z-direction of airfoil aligned coordinate system)')
        self.add_output('rhoA',         val=np.zeros(n_span), units='kg/m',   desc='mass per unit length')
        self.add_output('rhoJ',         val=np.zeros(n_span), units='kg*m',   desc='polar mass moment of inertia per unit length')
        self.add_output('Tw_iner',      val=np.zeros(n_span), units='m',      desc='y-distance to elastic center from point about which above structural properties are computed')
        self.add_output('x_ec',         val=np.zeros(n_span), units='m',      desc='x-distance to elastic center from point about which above structural properties are computed (airfoil aligned coordinate system)')
        self.add_output('y_ec',         val=np.zeros(n_span), units='m',      desc='y-distance to elastic center from point about which above structural properties are computed')
        self.add_output('flap_iner',    val=np.zeros(n_span), units='kg/m',   desc='Section flap inertia about the Y_G axis per unit length.')
        self.add_output('edge_iner',    val=np.zeros(n_span), units='kg/m',   desc='Section lag inertia about the X_G axis per unit length')
        # self.add_output('eps_crit_spar',    val=np.zeros(n_span), desc='critical strain in spar from panel buckling calculation')
        # self.add_output('eps_crit_te',      val=np.zeros(n_span), desc='critical strain in trailing-edge panels from panel buckling calculation')
        self.add_output('xu_strain_spar',   val=np.zeros(n_span), desc='x-position of midpoint of spar cap on upper surface for strain calculation')
        self.add_output('xl_strain_spar',   val=np.zeros(n_span), desc='x-position of midpoint of spar cap on lower surface for strain calculation')
        self.add_output('yu_strain_spar',   val=np.zeros(n_span), desc='y-position of midpoint of spar cap on upper surface for strain calculation')
        self.add_output('yl_strain_spar',   val=np.zeros(n_span), desc='y-position of midpoint of spar cap on lower surface for strain calculation')
        self.add_output('xu_strain_te',     val=np.zeros(n_span), desc='x-position of midpoint of trailing-edge panel on upper surface for strain calculation')
        self.add_output('xl_strain_te',     val=np.zeros(n_span), desc='x-position of midpoint of trailing-edge panel on lower surface for strain calculation')
        self.add_output('yu_strain_te',     val=np.zeros(n_span), desc='y-position of midpoint of trailing-edge panel on upper surface for strain calculation')
        self.add_output('yl_strain_te',     val=np.zeros(n_span), desc='y-position of midpoint of trailing-edge panel on lower surface for strain calculation')


        # Placeholder - rotor cost
        self.add_discrete_input('component_id', val=np.zeros(n_mat),              desc='1D array of flags to set whether a material is used in a blade: 0 - coating, 1 - sandwich filler , 2 - shell skin, 3 - shear webs, 4 - spar caps, 5 - TE reinf.isotropic.')
        self.add_input('unit_cost',     val=np.zeros(n_mat),      units='USD/kg', desc='1D array of the unit costs of the materials.')
        self.add_input('waste',         val=np.zeros(n_mat),                      desc='1D array of the non-dimensional waste fraction of the materials.')
        self.add_input('rho_fiber',     val=np.zeros(n_mat),      units='kg/m**3',desc='1D array of the density of the fibers of the materials.')
        self.add_input('rho_area_dry',  val=np.zeros(n_mat),      units='kg/m**2',desc='1D array of the dry aerial density of the composite fabrics. Non-composite materials are kept at 0.')
        self.add_input('ply_t',        val=np.zeros(n_mat),      units='m',      desc='1D array of the ply thicknesses of the materials. Non-composite materials are kept at 0.')
        self.add_input('fvf',          val=np.zeros(n_mat),                      desc='1D array of the non-dimensional fiber volume fraction of the composite materials. Non-composite materials are kept at 0.')
        self.add_input('fwf',          val=np.zeros(n_mat),                      desc='1D array of the non-dimensional fiber weight- fraction of the composite materials. Non-composite materials are kept at 0.')
        self.add_input('roll_mass',    val=np.zeros(n_mat),      units='kg',     desc='1D array of the roll mass of the composite fabrics. Non-composite materials are kept at 0.')

        # Outputs
        self.add_output('total_blade_cost', val=0.0, units='USD', desc='total blade cost')
        self.add_output('total_blade_mass', val=0.0, units='USD', desc='total blade cost')




    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        
                ##############################
        def region_stacking(i, idx, start_nd_arc, end_nd_arc, layer_name, layer_thickness, fiber_orientation, layer_mat, material_dict, materials, region_loc):
            # Recieve start and end of composite sections chordwise, find which composites layers are in each
            # chordwise regions, generate the precomp composite class instance

            # error handling to makes sure there were no numeric errors causing values very close too, but not exactly, 0 or 1
            start_nd_arc = [0. if start_nd_arci!=0. and np.isclose(start_nd_arci,0.) else start_nd_arci for start_nd_arci in start_nd_arc]
            end_nd_arc = [0. if end_nd_arci!=0. and np.isclose(end_nd_arci,0.) else end_nd_arci for end_nd_arci in end_nd_arc]
            start_nd_arc = [1. if start_nd_arci!=1. and np.isclose(start_nd_arci,1.) else start_nd_arci for start_nd_arci in start_nd_arc]
            end_nd_arc = [1. if end_nd_arci!=1. and np.isclose(end_nd_arci,1.) else end_nd_arci for end_nd_arci in end_nd_arc]

            # region end points
            dp = sorted(list(set(start_nd_arc+end_nd_arc)))

            #initialize
            n_plies = []
            thk = []
            theta = []
            mat_idx = []

            # loop through division points, find what layers make up the stack between those bounds
            for i_reg, (dp0, dp1) in enumerate(zip(dp[0:-1], dp[1:])):
                n_pliesi = []
                thki     = []
                thetai   = []
                mati     = []
                for i_sec, start_nd_arci, end_nd_arci in zip(idx, start_nd_arc, end_nd_arc):
                    name = layer_name[i_sec]
                    if start_nd_arci <= dp0 and end_nd_arci >= dp1:
                        
                        if name in region_loc.keys():
                            if region_loc[name][i] == None:
                                region_loc[name][i] = [i_reg]
                            else:
                                region_loc[name][i].append(i_reg)

                        n_pliesi.append(1.)
                        thki.append(layer_thickness[i_sec])
                        if fiber_orientation[i_sec] == None:
                            thetai.append(0.)
                        else:
                            thetai.append(fiber_orientation[i_sec])
                        mati.append(material_dict[layer_mat[i_sec]])

                n_plies.append(np.array(n_pliesi))
                thk.append(np.array(thki))
                theta.append(np.array(thetai))
                mat_idx.append(np.array(mati))

            # print('----------------------')
            # print('dp', dp)
            # print('n_plies', n_plies)
            # print('thk', thk)
            # print('theta', theta)
            # print('mat_idx', mat_idx)
            # print('materials', materials)

            sec = CompositeSection(dp, n_plies, thk, theta, mat_idx, materials)
            return sec, region_loc
            ##############################

        def web_stacking(i, web_idx, web_start_nd_arc, web_end_nd_arc, layer_thickness, fiber_orientation, layer_mat, material_dict, materials, flatback, upperCSi):
            dp = []
            n_plies = []
            thk = []
            theta = []
            mat_idx = []

            if len(web_idx)>0:
                dp = np.mean((np.abs(web_start_nd_arc), np.abs(web_start_nd_arc)), axis=0).tolist()

                dp_all = [[-1.*start_nd_arci, -1.*end_nd_arci] for start_nd_arci, end_nd_arci in zip(web_start_nd_arc, web_end_nd_arc)]
                web_dp, web_ids = np.unique(dp_all, axis=0, return_inverse=True)
                for webi in np.unique(web_ids):
                    # store variable values (thickness, orientation, material) for layers that make up each web, based on the mapping array web_ids
                    n_pliesi = [1. for i_reg, web_idi in zip(web_idx, web_ids) if web_idi==webi]
                    thki     = [layer_thickness[i_reg] for i_reg, web_idi in zip(web_idx, web_ids) if web_idi==webi]
                    thetai   = [fiber_orientation[i_reg] for i_reg, web_idi in zip(web_idx, web_ids) if web_idi==webi]
                    thetai   = [0. if theta_ij==None else theta_ij for theta_ij in thetai]
                    mati     = [material_dict[layer_mat[i_reg]] for i_reg, web_idi in zip(web_idx, web_ids) if web_idi==webi]

                    n_plies.append(np.array(n_pliesi))
                    thk.append(np.array(thki))
                    theta.append(np.array(thetai))
                    mat_idx.append(np.array(mati))

            if flatback:
                dp.append(1.)
                n_plies.append(upperCSi.n_plies[-1])
                thk.append(upperCSi.t[-1])
                theta.append(upperCSi.theta[-1])
                mat_idx.append(upperCSi.mat_idx[-1])

            dp_out = sorted(list(set(dp)))

            sec = CompositeSection(dp_out, n_plies, thk, theta, mat_idx, materials)
            return sec
            ##############################



        upperCS = [None]*self.n_span
        lowerCS = [None]*self.n_span
        websCS  = [None]*self.n_span
        profile = [None]*self.n_span

        # Check that the layer to be optimized actually exist
        te_ss_var_ok   = False
        te_ps_var_ok   = False
        spar_ss_var_ok = False
        spar_ps_var_ok = False
        for i_layer in range(self.n_layers):
            if discrete_inputs['layer_name'][i_layer] == self.te_ss_var:
                te_ss_var_ok = True
            if discrete_inputs['layer_name'][i_layer] == self.te_ps_var:
                te_ps_var_ok = True
            if discrete_inputs['layer_name'][i_layer] == self.spar_ss_var:
                spar_ss_var_ok = True
            if discrete_inputs['layer_name'][i_layer] == self.spar_ps_var:
                spar_ps_var_ok = True

        if te_ss_var_ok == False:
            print('The layer at the trailing edge suction side is set to be optimized, but does not exist in the input yaml. Please check.')
        if te_ps_var_ok == False:
            print('The layer at the trailing edge pressure side is set to be optimized, but does not exist in the input yaml. Please check.')
        if spar_ss_var_ok == False:
            print('The layer at the spar cap suction side is set to be optimized, but does not exist in the input yaml. Please check.')
        if spar_ps_var_ok == False:
            print('The layer at the spar cap pressure side is set to be optimized, but does not exist in the input yaml. Please check.')
        region_loc_vars = [self.te_ss_var, self.te_ps_var, self.spar_ss_var, self.spar_ps_var]

        region_loc_ss = {} # track precomp regions for user selected composite layers
        region_loc_ps = {}
        for var in region_loc_vars:
            region_loc_ss[var] = [None]*self.n_span
            region_loc_ps[var] = [None]*self.n_span


        ## Materials
        material_dict = {}
        materials     = []
        for i_mat in range(self.n_mat):
            materials.append(Orthotropic2DMaterial(inputs['E'][i_mat,0], inputs['E'][i_mat,1], inputs['G'][i_mat,0], inputs['nu'][i_mat,0], inputs['rho'][i_mat], discrete_inputs['mat_name'][i_mat]))
            material_dict[discrete_inputs['mat_name'][i_mat]] = i_mat

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

            profile[i] = Profile.initWithTEtoTEdata(profile_i_rot_precomp[:,0], profile_i_rot_precomp[:,1])

            # import matplotlib.pyplot as plt
            # plt.plot(profile_i_rot_precomp[:,0], profile_i_rot_precomp[:,1])
            # plt.axis('equal')
            # plt.title(i)
            # plt.show()

            idx_le = np.argmin(profile_i_rot[:,0])

            profile_i_arc = arc_length(profile_i_rot[:,0], profile_i_rot[:,1])
            arc_L = profile_i_arc[-1]
            profile_i_arc /= arc_L

            loc_LE = profile_i_arc[idx_le]
            len_PS = 1.-loc_LE

            ## Composites
            ss_idx           = []
            ss_start_nd_arc  = []
            ss_end_nd_arc    = []
            ps_idx           = []
            ps_start_nd_arc  = []
            ps_end_nd_arc    = []
            web_start_nd_arc = []
            web_end_nd_arc   = []
            web_idx          = []

            # Determine spanwise composite layer elements that are non-zero at this spanwise location,
            # determine their chord-wise start and end location on the pressure and suctions side

            spline_arc2xnd = PchipInterpolator(profile_i_arc, profile_i_rot[:,0])

            # time1 = time.time()
            for idx_sec in range(self.n_layers):            
                if discrete_inputs['definition_layer'][idx_sec] != 10:
                    if inputs['layer_thickness'][idx_sec,i] != 0.:
                        if inputs['layer_start_nd'][idx_sec,i] < loc_LE or inputs['layer_end_nd'][idx_sec,i] < loc_LE:
                            ss_idx.append(idx_sec)
                            if inputs['layer_start_nd'][idx_sec,i] < loc_LE:
                                # ss_start_nd_arc.append(sec['start_nd_arc']['values'][i])
                                ss_end_nd_arc_temp = float(spline_arc2xnd(inputs['layer_start_nd'][idx_sec,i]))
                                if ss_end_nd_arc_temp > 1 or ss_end_nd_arc_temp < 0:
                                    exit('Error in the definition of material ' + discrete_inputs['layer_name'][idx_sec] + '. It cannot fit in the section')
                                if ss_end_nd_arc_temp == profile_i_rot[0,0] and profile_i_rot[0,0] != 1.:
                                    ss_end_nd_arc_temp = 1.
                                ss_end_nd_arc.append(ss_end_nd_arc_temp)
                            else:
                                ss_end_nd_arc.append(1.)
                            # ss_end_nd_arc.append(min(sec['end_nd_arc']['values'][i], loc_LE)/loc_LE)
                            if inputs['layer_end_nd'][idx_sec,i] < loc_LE:
                                ss_start_nd_arc.append(float(spline_arc2xnd(inputs['layer_end_nd'][idx_sec,i])))
                            else:
                                ss_start_nd_arc.append(0.)
                            
                        if inputs['layer_start_nd'][idx_sec,i] > loc_LE or inputs['layer_end_nd'][idx_sec,i] > loc_LE:
                            ps_idx.append(idx_sec)
                            # ps_start_nd_arc.append((max(sec['start_nd_arc']['values'][i], loc_LE)-loc_LE)/len_PS)
                            # ps_end_nd_arc.append((min(sec['end_nd_arc']['values'][i], 1.)-loc_LE)/len_PS)

                            if inputs['layer_start_nd'][idx_sec,i] > loc_LE and inputs['layer_end_nd'][idx_sec,i] < loc_LE:
                                # ps_start_nd_arc.append(float(remap2grid(profile_i_arc, profile_i_rot[:,0], sec['start_nd_arc']['values'][i])))
                                ps_end_nd_arc.append(1.)
                            else:
                                ps_end_nd_arc_temp = float(spline_arc2xnd(inputs['layer_end_nd'][idx_sec,i]))
                                if np.isclose(ps_end_nd_arc_temp, profile_i_rot[-1,0]) and profile_i_rot[-1,0] != 1.:
                                    ps_end_nd_arc_temp = 1.
                                ps_end_nd_arc.append(ps_end_nd_arc_temp)
                            if inputs['layer_start_nd'][idx_sec,i] < loc_LE:
                                ps_start_nd_arc.append(0.)
                            else:
                                ps_start_nd_arc.append(float(spline_arc2xnd(inputs['layer_start_nd'][idx_sec,i])))
                else: 
                    target_name  = discrete_inputs['layer_web'][idx_sec]
                    for i_web in range(self.n_webs):
                        if target_name == discrete_inputs['web_name'][i_web]:
                            target_idx = i_web
                            break

                    if inputs['layer_thickness'][idx_sec,i] != 0.:
                        web_idx.append(idx_sec)

                        start_nd_arc = float(spline_arc2xnd(inputs['web_start_nd'][target_idx,i]))
                        end_nd_arc   = float(spline_arc2xnd(inputs['web_end_nd'][target_idx,i]))

                        web_start_nd_arc.append(start_nd_arc)
                        web_end_nd_arc.append(end_nd_arc)

            # time1 = time.time() - time1
            # print(time1)

            # generate the Precomp composite stacks for chordwise regions
            if np.min([ss_start_nd_arc, ss_end_nd_arc]) < 0 or np.max([ss_start_nd_arc, ss_end_nd_arc]) > 1:
                print('Error in the layer definition at station number ' + str(i))
                exit()
            upperCS[i], region_loc_ss = region_stacking(i, ss_idx, ss_start_nd_arc, ss_end_nd_arc, discrete_inputs['layer_name'],inputs['layer_thickness'][:,i], inputs['fiber_orientation'][:,i], discrete_inputs['layer_mat'], material_dict, materials, region_loc_ss)
            lowerCS[i], region_loc_ps = region_stacking(i, ps_idx, ps_start_nd_arc, ps_end_nd_arc, discrete_inputs['layer_name'],inputs['layer_thickness'][:,i], inputs['fiber_orientation'][:,i], discrete_inputs['layer_mat'], material_dict, materials, region_loc_ps)
            if len(web_idx)>0 or flatback:
                websCS[i] = web_stacking(i, web_idx, web_start_nd_arc, web_end_nd_arc, inputs['layer_thickness'][:,i], inputs['fiber_orientation'][:,i], discrete_inputs['layer_mat'], material_dict, materials, flatback, upperCS[i])
            else:
                websCS[i] = CompositeSection([], [], [], [], [], [])
        
        sector_idx_strain_spar_ss = [None if regs==None else regs[int(len(regs)/2)] for regs in region_loc_ss[self.spar_ss_var]]
        sector_idx_strain_spar_ps = [None if regs==None else regs[int(len(regs)/2)] for regs in region_loc_ps[self.spar_ps_var]]
        sector_idx_strain_te_ss   = [None if regs==None else regs[int(len(regs)/2)] for regs in region_loc_ss[self.te_ss_var]]
        sector_idx_strain_te_ps   = [None if regs==None else regs[int(len(regs)/2)] for regs in region_loc_ps[self.te_ps_var]]

        # Get Beam Properties        
        beam = PreComp(inputs['r'], inputs['chord'], inputs['theta'], inputs['pitch_axis'], 
                       inputs['precurve'], inputs['presweep'], profile, materials, upperCS, lowerCS, websCS, 
                       sector_idx_strain_spar_ps, sector_idx_strain_spar_ss, sector_idx_strain_te_ps, sector_idx_strain_te_ss)
        EIxx, EIyy, GJ, EA, EIxy, x_ec, y_ec, rhoA, rhoJ, Tw_iner, flap_iner, edge_iner = beam.sectionProperties()

        # outputs['eps_crit_spar'] = beam.panelBucklingStrain(sector_idx_strain_spar_ss)
        # outputs['eps_crit_te'] = beam.panelBucklingStrain(sector_idx_strain_te_ss)

        xu_strain_spar, xl_strain_spar, yu_strain_spar, yl_strain_spar = beam.criticalStrainLocations(sector_idx_strain_spar_ss, sector_idx_strain_spar_ps)
        xu_strain_te, xl_strain_te, yu_strain_te, yl_strain_te = beam.criticalStrainLocations(sector_idx_strain_te_ss, sector_idx_strain_te_ps)
        
        outputs['z']         = inputs['r']
        outputs['EIxx']      = EIxx
        outputs['EIyy']      = EIyy
        outputs['GJ']        = GJ
        outputs['EA']        = EA
        outputs['EIxy']      = EIxy
        outputs['x_ec']      = x_ec
        outputs['y_ec']      = y_ec
        outputs['rhoA']      = rhoA
        outputs['rhoJ']      = rhoJ
        outputs['Tw_iner']   = Tw_iner
        outputs['flap_iner'] = flap_iner
        outputs['edge_iner'] = edge_iner

        outputs['xu_strain_spar'] = xu_strain_spar
        outputs['xl_strain_spar'] = xl_strain_spar
        outputs['yu_strain_spar'] = yu_strain_spar
        outputs['yl_strain_spar'] = yl_strain_spar
        outputs['xu_strain_te']   = xu_strain_te
        outputs['xl_strain_te']   = xl_strain_te
        outputs['yu_strain_te']   = yu_strain_te
        outputs['yl_strain_te']   = yl_strain_te

        # Placeholder - rotor cost
        bcm             = blade_cost_model(verbosity = True)
        bcm.name        = ''
        bcm.materials   = {}
        bcm.mat_options = {}

        bcm.mat_options['core_mat_id'] = np.zeros(self.n_mat)
        bcm.mat_options['coating_mat_id']   = -1
        bcm.mat_options['le_reinf_mat_id']  = -1
        bcm.mat_options['te_reinf_mat_id']  = -1


        for i_mat in range(self.n_mat):
            name = discrete_inputs['mat_name'][i_mat]
            # if name != 'resin':
            bcm.materials[name]             = {}
            bcm.materials[name]['id']       = i_mat + 1
            bcm.materials[name]['name']     = discrete_inputs['mat_name'][i_mat]
            bcm.materials[name]['density']  = inputs['rho'][i_mat]
            bcm.materials[name]['unit_cost']= inputs['unit_cost'][i_mat]
            bcm.materials[name]['waste']    = inputs['waste'][i_mat] * 100.
            if discrete_inputs['component_id'][i_mat] > 1: # It's a composite
                bcm.materials[name]['fiber_density']  = inputs['rho_fiber'][i_mat]
                bcm.materials[name]['area_density_dry']  = inputs['rho_area_dry'][i_mat]
                bcm.materials[name]['fvf']  = inputs['fvf'][i_mat] * 100.
                bcm.materials[name]['fwf']  = inputs['fwf'][i_mat] * 100.
                bcm.materials[name]['ply_t']  = inputs['ply_t'][i_mat]
                if discrete_inputs['component_id'][i_mat] > 3: # The material does not need to be cut@station
                    bcm.materials[name]['cut@station'] = 'N'
                else:
                    bcm.materials[name]['cut@station'] = 'Y'
                    bcm.materials[name]['roll_mass']   = inputs['roll_mass'][i_mat]
            else:
                bcm.materials[name]['fvf']  = 100.
                bcm.materials[name]['fwf']  = 100.
                bcm.materials[name]['cut@station'] = 'N'
                if discrete_inputs['component_id'][i_mat] <= 0:
                    bcm.materials[name]['ply_t']  = inputs['ply_t'][i_mat]
            
            if discrete_inputs['component_id'][i_mat] == 0:
                bcm.mat_options['coating_mat_id'] = bcm.materials[name]['id']        # Assigning the material to the coating
            elif discrete_inputs['component_id'][i_mat] == 1:    
                bcm.mat_options['core_mat_id'][bcm.materials[name]['id'] - 1]  = 1   # Assigning the material to the core
            elif discrete_inputs['component_id'][i_mat] == 2:    
                bcm.mat_options['skin_mat_id'] = bcm.materials[name]['id']     # Assigning the material to the shell skin
            elif discrete_inputs['component_id'][i_mat] == 3:    
                bcm.mat_options['skinwebs_mat_id'] = bcm.materials[name]['id'] # Assigning the material to the webs skin 
            elif discrete_inputs['component_id'][i_mat] == 4:    
                bcm.mat_options['sc_mat_id'] = bcm.materials[name]['id']   # Assigning the material to the spar caps
            elif discrete_inputs['component_id'][i_mat] == 5:
                bcm.mat_options['le_reinf_mat_id'] = bcm.materials[name]['id']   # Assigning the material to the le reinf
                bcm.mat_options['te_reinf_mat_id'] = bcm.materials[name]['id']   # Assigning the material to the te reinf

        bcm.upperCS     = upperCS
        bcm.lowerCS     = lowerCS
        bcm.websCS      = websCS
        bcm.profile     = profile
        bcm.chord       = inputs['chord']
        bcm.r           = inputs['r'] - inputs['r'][0]
        bcm.bladeLength         = inputs['r'][-1] - inputs['r'][0]
        bcm.le_location         = inputs['pitch_axis']
        blade_cost, blade_mass  = bcm.execute_blade_cost_model()
        
        outputs['total_blade_cost'] = blade_cost
        outputs['total_blade_mass'] = blade_mass

class RunCurveFEM(ExplicitComponent):
    # OpenMDAO component that computes the natural frequencies for curved blades using _pBEAM
    def initialize(self):
        self.options.declare('wt_init_options')

    def setup(self):
        blade_init_options = self.options['wt_init_options']['blade']
        self.n_span = n_span = blade_init_options['n_span']
        self.n_freq = n_freq = blade_init_options['n_freq']

        # Inputs
        self.add_input('Omega',         val=0.0,                units='rpm',    desc='rotor rotation frequency')
        self.add_input('r',             val=np.zeros(n_span),   units='m',      desc='locations of properties along beam')
        self.add_input('EA',            val=np.zeros(n_span),   units='N',      desc='axial stiffness')
        self.add_input('EIxx',          val=np.zeros(n_span),   units='N*m**2', desc='edgewise stiffness (bending about :ref:`x-direction of airfoil aligned coordinate system <blade_airfoil_coord>`)')
        self.add_input('EIyy',          val=np.zeros(n_span),   units='N*m**2', desc='flatwise stiffness (bending about y-direction of airfoil aligned coordinate system)')
        self.add_input('GJ',            val=np.zeros(n_span),   units='N*m**2', desc='torsional stiffness (about axial z-direction of airfoil aligned coordinate system)')
        self.add_input('rhoA',          val=np.zeros(n_span),   units='kg/m',   desc='mass per unit length')
        self.add_input('rhoJ',          val=np.zeros(n_span),   units='kg*m',   desc='polar mass moment of inertia per unit length')
        self.add_input('Tw_iner',       val=np.zeros(n_span),   units='m',      desc='y-distance to elastic center from point about which above structural properties are computed')
        self.add_input('precurve',      val=np.zeros(n_span),   units='m',      desc='structural precuve (see FAST definition)')
        self.add_input('presweep',      val=np.zeros(n_span),   units='m',      desc='structural presweep (see FAST definition)')

        # Outputs
        self.add_output('freq',         val=np.zeros(n_freq),   units='Hz',     desc='first nF natural frequencies')
        self.add_output('modes_coef',   val=np.zeros((3, 5)),                   desc='mode shapes as 6th order polynomials, in the format accepted by ElastoDyn, [[c_x2, c_],..]')


    def compute(self, inputs, outputs):

        mycurve = _pBEAM.CurveFEM(inputs['Omega'], inputs['Tw_iner'], inputs['r'], inputs['precurve'], inputs['presweep'], inputs['rhoA'], True)
        freq, eig_vec = mycurve.frequencies(inputs['EA'], inputs['EIxx'], inputs['EIyy'], inputs['GJ'], inputs['rhoJ'], self.n_span)
        outputs['freq'] = freq[:self.n_freq]
        
        # Parse eigen vectors
        R = inputs['r']
        R = np.asarray([(Ri-R[0])/(R[-1]-R[0]) for Ri in R])
        ndof = 6

        flap = np.zeros((self.n_freq, self.n_span))
        edge = np.zeros((self.n_freq, self.n_span))
        for i in range(self.n_freq):
            eig_vec_i = eig_vec[:,i]
            for j in range(self.n_span):
                flap[i,j] = eig_vec_i[0+j*ndof]
                edge[i,j] = eig_vec_i[1+j*ndof]


        # Mode shape polynomial fit
        def mode_fit(x, a, b, c, d, e):
            return a*x**2. + b*x**3. + c*x**4. + d*x**5. + e*x**6.
        # First Flapwise
        coef, pcov = curve_fit(mode_fit, R, flap[0,:])
        coef_norm = [c/sum(coef) for c in coef]
        outputs['modes_coef'][0,:] = coef_norm
        # Second Flapwise
        coef, pcov = curve_fit(mode_fit, R, flap[1,:])
        coef_norm = [c/sum(coef) for c in coef]
        outputs['modes_coef'][1,:] = coef_norm
        # First Edgewise
        coef, pcov = curve_fit(mode_fit, R, edge[0,:])
        coef_norm = [c/sum(coef) for c in coef]
        outputs['modes_coef'][2,:] = coef_norm


        # # temp
        # from bmodes import BModes_tools
        # r = np.asarray([(ri-inputs['z'][0])/(inputs['z'][-1]-inputs['z'][0]) for ri in inputs['z']])
        # prop = np.column_stack((r, inputs['theta'], inputs['Tw_iner'], inputs['rhoA'], inputs['flap_iner'], inputs['edge_iner'], inputs['EIyy'], \
        #         inputs['EIxx'], inputs['GJ'], inputs['EA'], np.zeros_like(r), np.zeros_like(r), np.zeros_like(r)))
        
        # bm = BModes_tools()
        # bm.setup.radius = inputs['z'][-1]
        # bm.setup.hub_rad = inputs['z'][0]
        # bm.setup.precone = -2.5
        # bm.prop = prop
        # bm.exe_BModes = 'C:/Users/egaertne/WT_Codes/bModes/BModes.exe'
        # bm.execute()
        # print(bm.freq)

        # import matplotlib.pyplot as plt

        # fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(12., 6.), sharex=True, sharey=True)
        # # fig.subplots_adjust(bottom=0.2, top=0.9)
        # fig.subplots_adjust(bottom=0.15, left=0.1, hspace=0, wspace=0)
        # i = 0
        # k_flap = bm.flap_disp[i,-1]/flap[i,-1]
        # k_edge = bm.lag_disp[i,-1]/edge[i,-1]
        # ax[0,0].plot(R, flap[i,:]*k_flap ,'k',label='CurveFEM')
        # ax[0,0].plot(bm.r[i,:], bm.flap_disp[i,:],'bx',label='BModes')
        # ax[0,0].set_ylabel('Flapwise Disp.')
        # ax[0,0].set_title('1st Mode')
        # ax[1,0].plot(R, edge[i,:]*k_edge ,'k')
        # ax[1,0].plot(bm.r[i,:], bm.lag_disp[i,:],'bx')
        # ax[1,0].set_ylabel('Edgewise Disp.')

        # i = 1
        # k_flap = bm.flap_disp[i,-1]/flap[i,-1]
        # k_edge = bm.lag_disp[i,-1]/edge[i,-1]
        # ax[0,1].plot(R, flap[i,:]*k_flap ,'k')
        # ax[0,1].plot(bm.r[i,:], bm.flap_disp[i,:],'bx')
        # ax[0,1].set_title('2nd Mode')
        # ax[1,1].plot(R, edge[i,:]*k_edge ,'k')
        # ax[1,1].plot(bm.r[i,:], bm.lag_disp[i,:],'bx')

        # i = 2
        # k_flap = bm.flap_disp[i,-1]/flap[i,-1]
        # k_edge = bm.lag_disp[i,-1]/edge[i,-1]
        # ax[0,2].plot(R, flap[i,:]*k_flap ,'k')
        # ax[0,2].plot(bm.r[i,:], bm.flap_disp[i,:],'bx')
        # ax[0,2].set_title('3rd Mode')
        # ax[1,2].plot(R, edge[i,:]*k_edge ,'k')
        # ax[1,2].plot(bm.r[i,:], bm.lag_disp[i,:],'bx')
        # fig.legend(loc='lower center', ncol=2)

        # i = 3
        # k_flap = bm.flap_disp[i,-1]/flap[i,-1]
        # k_edge = bm.lag_disp[i,-1]/edge[i,-1]
        # ax[0,3].plot(R, flap[i,:]*k_flap ,'k')
        # ax[0,3].plot(bm.r[i,:], bm.flap_disp[i,:],'bx')
        # ax[0,3].set_title('4th Mode')
        # ax[1,3].plot(R, edge[i,:]*k_edge ,'k')
        # ax[1,3].plot(bm.r[i,:], bm.lag_disp[i,:],'bx')
        # fig.legend(loc='lower center', ncol=2)
        # fig.text(0.5, 0.075, 'Blade Spanwise Position, $r/R$', ha='center')

        # (n,m)=np.shape(ax)
        # for i in range(n):
        #     for j in range(m):
        #         ax[i,j].tick_params(axis='both', which='major', labelsize=8)
        #         ax[i,j].grid(True, linestyle=':')

        # plt.show()

class GustETM(ExplicitComponent):
    # OpenMDAO component that generates an "equivalent gust" wind speed by summing an user-defined wind speed at hub height with 3 times sigma. sigma is the turbulent wind speed standard deviation for the extreme turbulence model, see IEC-61400-1 Eq. 19 paragraph 6.3.2.3
    
    def setup(self):
        # Inputs
        self.add_input('V_mean', val=0.0, units='m/s', desc='IEC average wind speed for turbine class')
        self.add_input('V_hub', val=0.0, units='m/s', desc='hub height wind speed')
        self.add_discrete_input('turbulence_class', val='A', desc='IEC turbulence class')
        self.add_discrete_input('std', val=3, desc='number of standard deviations for strength of gust')

        # Output
        self.add_output('V_gust', val=0.0, units='m/s', desc='gust wind speed')

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        V_mean = inputs['V_mean']
        V_hub = inputs['V_hub']
        turbulence_class = discrete_inputs['turbulence_class']
        std = discrete_inputs['std']

        if turbulence_class == 'A':
            Iref = 0.16
        elif turbulence_class == 'B':
            Iref = 0.14
        elif turbulence_class == 'C':
            Iref = 0.12

        c = 2.0
        sigma = c * Iref * (0.072*(V_mean/c + 3)*(V_hub/c - 4) + 10)
        V_gust = V_hub + std*sigma
        outputs['V_gust'] = V_gust

class BladeCurvature(ExplicitComponent):
    # OpenMDAO component that computes the 3D curvature of the blade
    def initialize(self):
        self.options.declare('wt_init_options')

    def setup(self):
        blade_init_options = self.options['wt_init_options']['blade']
        n_span    = blade_init_options['n_span']

        # Inputs
        self.add_input('r',         val=np.zeros(n_span), units='m',      desc='location in blade z-coordinate')
        self.add_input('precurve',  val=np.zeros(n_span), units='m',      desc='location in blade x-coordinate')
        self.add_input('presweep',  val=np.zeros(n_span), units='m',      desc='location in blade y-coordinate')
        self.add_input('precone',   val=0.0,              units='deg',    desc='precone angle')

        # Outputs
        self.add_output('3d_curv',  val=np.zeros(n_span),units='deg',    desc='total cone angle from precone and curvature')
        self.add_output('x_az',     val=np.zeros(n_span), units='m',      desc='location of blade in azimuth x-coordinate system')
        self.add_output('y_az',     val=np.zeros(n_span), units='m',      desc='location of blade in azimuth y-coordinate system')
        self.add_output('z_az',     val=np.zeros(n_span), units='m',      desc='location of blade in azimuth z-coordinate system')
        self.add_output('s',        val=np.zeros(n_span), units='m',      desc='cumulative path length along blade')

    def compute(self, inputs, outputs):

        r = inputs['r']
        precurve = inputs['precurve']
        presweep = inputs['presweep']
        precone = inputs['precone']

        n = len(r)
        dx_dx = np.eye(3*n)

        x_az, x_azd, y_az, y_azd, z_az, z_azd, cone, coned, s, sd = _bem.definecurvature_dv2(r, dx_dx[:, :n],
                                                                                             precurve, dx_dx[:, n:2*n],
                                                                                             presweep, dx_dx[:, 2*n:],
                                                                                             0.0, np.zeros(3*n))

        totalCone = precone + np.degrees(cone)
        s = r[0] + s

        outputs['3d_curv'] = totalCone
        outputs['x_az'] = x_az
        outputs['y_az'] = y_az
        outputs['z_az'] = z_az
        outputs['s'] = s

class TotalLoads(ExplicitComponent):
    # OpenMDAO component that takes as input the rotor configuration (tilt, cone), the blade twist and mass distributions, and the blade aerodynamic loading, and computes the total loading including gravity and centrifugal forces
    def initialize(self):
        self.options.declare('wt_init_options')

    def setup(self):
        blade_init_options = self.options['wt_init_options']['blade']
        n_span    = blade_init_options['n_span']

        # Inputs
        self.add_input('r',                 val=np.zeros(n_span),   units='m',      desc='radial positions along blade going toward tip')
        self.add_input('aeroloads_Px',      val=np.zeros(n_span),   units='N/m',    desc='distributed loads in blade-aligned x-direction')
        self.add_input('aeroloads_Py',      val=np.zeros(n_span),   units='N/m',    desc='distributed loads in blade-aligned y-direction')
        self.add_input('aeroloads_Pz',      val=np.zeros(n_span),   units='N/m',    desc='distributed loads in blade-aligned z-direction')
        self.add_input('aeroloads_Omega',   val=0.0,                units='rpm',    desc='rotor rotation speed')
        self.add_input('aeroloads_pitch',   val=0.0,                units='deg',    desc='pitch angle')
        self.add_input('aeroloads_azimuth', val=0.0,                units='deg',    desc='azimuthal angle')
        self.add_input('theta',             val=np.zeros(n_span),   units='deg',    desc='structural twist')
        self.add_input('tilt',              val=0.0,                units='deg',    desc='tilt angle')
        self.add_input('3d_curv',           val=np.zeros(n_span),   units='deg',    desc='total cone angle from precone and curvature')
        self.add_input('z_az',              val=np.zeros(n_span),   units='m',      desc='location of blade in azimuth z-coordinate system')
        self.add_input('rhoA',              val=np.zeros(n_span),   units='kg/m',   desc='mass per unit length')
        self.add_input('dynamicFactor',     val=1.0,                                desc='a dynamic amplification factor to adjust the static deflection calculation') #)

        # Outputs
        self.add_output('Px_af', val=np.zeros(n_span), desc='total distributed loads in airfoil x-direction')
        self.add_output('Py_af', val=np.zeros(n_span), desc='total distributed loads in airfoil y-direction')
        self.add_output('Pz_af', val=np.zeros(n_span), desc='total distributed loads in airfoil z-direction')


    def compute(self, inputs, outputs):

        dynamicFactor = inputs['dynamicFactor']
        r = inputs['r']
        theta = inputs['theta']
        tilt = inputs['tilt']
        totalCone = inputs['3d_curv']
        z_az = inputs['z_az']
        rhoA = inputs['rhoA']


        # totalCone = precone
        # z_az = r*cosd(precone)
        totalCone = totalCone
        z_az = z_az

        # keep all in blade c.s. then rotate all at end

        # rename
        # aero = aeroloads

        # --- aero loads ---

        # interpolate aerodynamic loads onto structural grid
        P_a = DirectionVector(0, 0, 0)
        myakima = Akima(inputs['r'], inputs['aeroloads_Px'])
        P_a.x, dPax_dr, dPax_daeror, dPax_daeroPx = myakima(r)

        myakima = Akima(inputs['r'], inputs['aeroloads_Py'])
        P_a.y, dPay_dr, dPay_daeror, dPay_daeroPy = myakima(r)

        myakima = Akima(inputs['r'], inputs['aeroloads_Pz'])
        P_a.z, dPaz_dr, dPaz_daeror, dPaz_daeroPz = myakima(r)


        # --- weight loads ---

        # yaw c.s.
        weight = DirectionVector(0.0, 0.0, -rhoA*gravity)

        P_w = weight.yawToHub(tilt).hubToAzimuth(inputs['aeroloads_azimuth'])\
            .azimuthToBlade(totalCone)


        # --- centrifugal loads ---

        # azimuthal c.s.
        Omega = inputs['aeroloads_Omega']*RPM2RS
        load = DirectionVector(0.0, 0.0, rhoA*Omega**2*z_az)

        P_c = load.azimuthToBlade(totalCone)


        # --- total loads ---
        P = P_a + P_w + P_c

        # rotate to airfoil c.s.
        theta = np.array(theta) + inputs['aeroloads_pitch']
        P = P.bladeToAirfoil(theta)

        Px_af = dynamicFactor * P.x
        Py_af = dynamicFactor * P.y
        Pz_af = dynamicFactor * P.z

        outputs['Px_af'] = Px_af
        outputs['Py_af'] = Py_af
        outputs['Pz_af'] = Pz_af

class RunpBEAM(ExplicitComponent):
    def initialize(self):
        self.options.declare('wt_init_options')

    def setup(self):
        blade_init_options = self.options['wt_init_options']['blade']
        self.n_span = n_span = blade_init_options['n_span']
        self.n_freq = n_freq = blade_init_options['n_freq']

        # all inputs/outputs in airfoil coordinate system
        self.add_input('Px_af', val=np.zeros(n_span), desc='distributed load (force per unit length) in airfoil x-direction')
        self.add_input('Py_af', val=np.zeros(n_span), desc='distributed load (force per unit length) in airfoil y-direction')
        self.add_input('Pz_af', val=np.zeros(n_span), desc='distributed load (force per unit length) in airfoil z-direction')

        self.add_input('xu_strain_spar',    val=np.zeros(n_span), desc='x-position of midpoint of spar cap on upper surface for strain calculation')
        self.add_input('xl_strain_spar',    val=np.zeros(n_span), desc='x-position of midpoint of spar cap on lower surface for strain calculation')
        self.add_input('yu_strain_spar',    val=np.zeros(n_span), desc='y-position of midpoint of spar cap on upper surface for strain calculation')
        self.add_input('yl_strain_spar',    val=np.zeros(n_span), desc='y-position of midpoint of spar cap on lower surface for strain calculation')
        self.add_input('xu_strain_te',      val=np.zeros(n_span), desc='x-position of midpoint of trailing-edge panel on upper surface for strain calculation')
        self.add_input('xl_strain_te',      val=np.zeros(n_span), desc='x-position of midpoint of trailing-edge panel on lower surface for strain calculation')
        self.add_input('yu_strain_te',      val=np.zeros(n_span), desc='y-position of midpoint of trailing-edge panel on upper surface for strain calculation')
        self.add_input('yl_strain_te',      val=np.zeros(n_span), desc='y-position of midpoint of trailing-edge panel on lower surface for strain calculation')

        self.add_input('r',     val=np.zeros(n_span), units='m',        desc='locations of properties along beam')
        self.add_input('EA',    val=np.zeros(n_span), units='N',        desc='axial stiffness')
        self.add_input('EIxx',  val=np.zeros(n_span), units='N*m**2',   desc='edgewise stiffness (bending about :ref:`x-direction of airfoil aligned coordinate system <blade_airfoil_coord>`)')
        self.add_input('EIyy',  val=np.zeros(n_span), units='N*m**2',   desc='flatwise stiffness (bending about y-direction of airfoil aligned coordinate system)')
        self.add_input('EIxy',  val=np.zeros(n_span), units='N*m**2',   desc='coupled flap-edge stiffness')
        self.add_input('GJ',    val=np.zeros(n_span), units='N*m**2',   desc='torsional stiffness (about axial z-direction of airfoil aligned coordinate system)')
        self.add_input('rhoA',  val=np.zeros(n_span), units='kg/m',     desc='mass per unit length')
        self.add_input('rhoJ',  val=np.zeros(n_span), units='kg*m',     desc='polar mass moment of inertia per unit length')
        self.add_input('x_ec',  val=np.zeros(n_span), units='m',        desc='x-distance to elastic center from point about which above structural properties are computed (airfoil aligned coordinate system)')
        self.add_input('y_ec',  val=np.zeros(n_span), units='m', desc='y-distance to elastic center from point about which above structural properties are computed')

        # outputs
        self.add_output('blade_mass',       val=0.0,              units='kg',       desc='mass of one blades')
        self.add_output('blade_moment_of_inertia', val=0.0,       units='kg*m**2',  desc='out of plane moment of inertia of a blade')
        self.add_output('freq_pbeam',       val=np.zeros(n_freq), units='Hz',       desc='first nF natural frequencies of blade')
        self.add_output('freq_distance',    val=0.0,              desc='ration of 2nd and 1st natural frequencies, should be ratio of edgewise to flapwise')
        self.add_output('dx',               val=np.zeros(n_span), desc='deflection of blade section in airfoil x-direction')
        self.add_output('dy',               val=np.zeros(n_span), desc='deflection of blade section in airfoil y-direction')
        self.add_output('dz',               val=np.zeros(n_span), desc='deflection of blade section in airfoil z-direction')
        self.add_output('strainU_spar',     val=np.zeros(n_span), desc='strain in spar cap on upper surface at location xu,yu_strain with loads P_strain')
        self.add_output('strainL_spar',     val=np.zeros(n_span), desc='strain in spar cap on lower surface at location xl,yl_strain with loads P_strain')
        self.add_output('strainU_te',       val=np.zeros(n_span), desc='strain in trailing-edge panels on upper surface at location xu,yu_te with loads P_te')
        self.add_output('strainL_te',       val=np.zeros(n_span), desc='strain in trailing-edge panels on lower surface at location xl,yl_te with loads P_te')
        
    def principalCS(self, EIyy_in, EIxx_in, y_ec_in, x_ec_in, EA, EIxy):

        # rename (with swap of x, y for profile c.s.)
        EIxx , EIyy = EIyy_in , EIxx_in
        x_ec , y_ec = y_ec_in , x_ec_in
        self.EA     = EA
        EIxy        = EIxy

        # translate to elastic center
        EIxx -= y_ec**2*EA
        EIyy -= x_ec**2*EA
        EIxy -= x_ec*y_ec*EA

        # get rotation angle
        alpha = 0.5*np.arctan2(2*EIxy, EIyy-EIxx)

        self.EI11 = EIxx - EIxy*np.tan(alpha)
        self.EI22 = EIyy + EIxy*np.tan(alpha)

        # get moments and positions in principal axes
        self.ca = np.cos(alpha)
        self.sa = np.sin(alpha)

    def strain(self, blade, xu, yu, xl, yl):

        Vx, Vy, Fz, Mx, My, Tz = blade.shearAndBending()

        # use profile c.s. to use Hansen's notation
        Vx, Vy = Vy, Vx
        Mx, My = My, Mx
        xu, yu = yu, xu
        xl, yl = yl, xl

        # convert to principal xes
        M1 = Mx*self.ca + My*self.sa
        M2 = -Mx*self.sa + My*self.ca

        x = xu*self.ca + yu*self.sa
        y = -xu*self.sa + yu*self.ca

        # compute strain
        strainU = -(M1/self.EI11*y - M2/self.EI22*x + Fz/self.EA)  # negative sign because 3 is opposite of z

        x = xl*self.ca + yl*self.sa
        y = -xl*self.sa + yl*self.ca

        strainL = -(M1/self.EI11*y - M2/self.EI22*x + Fz/self.EA)

        return strainU, strainL

    def compute(self, inputs, outputs):

        Px = inputs['Px_af']
        Py = inputs['Py_af']
        Pz = inputs['Pz_af']
        xu_strain_spar = inputs['xu_strain_spar']
        xl_strain_spar = inputs['xl_strain_spar']
        yu_strain_spar = inputs['yu_strain_spar']
        yl_strain_spar = inputs['yl_strain_spar']
        xu_strain_te = inputs['xu_strain_te']
        xu_strain_te = inputs['xu_strain_te']
        xl_strain_te = inputs['xl_strain_te']
        yu_strain_te = inputs['yu_strain_te']
        yl_strain_te = inputs['yl_strain_te']

        # outputs
        nsec = self.n_span
        
        # create finite element objects
        p_section = _pBEAM.SectionData(nsec, inputs['r'], inputs['EA'], inputs['EIxx'],
            inputs['EIyy'], inputs['GJ'], inputs['rhoA'], inputs['rhoJ'])
        p_tip = _pBEAM.TipData()  # no tip mass
        p_base = _pBEAM.BaseData(np.ones(6), 1.0)  # rigid base


        # ----- tip deflection -----

        # evaluate displacements
        p_loads = _pBEAM.Loads(nsec, Px, Py, Pz)
        blade = _pBEAM.Beam(p_section, p_loads, p_tip, p_base)
        dx, dy, dz, dtheta_r1, dtheta_r2, dtheta_z = blade.displacement()

        # --- moments of inertia
        blade_moment_of_inertia = blade.outOfPlaneMomentOfInertia()
        # --- mass ---
        blade_mass = blade.mass()
        # ----- natural frequencies ----
        freq = blade.naturalFrequencies(self.n_freq)

        # ----- strain -----
        self.principalCS(inputs['EIyy'], inputs['EIxx'], inputs['y_ec'], inputs['x_ec'], inputs['EA'], inputs['EIxy'])
        strainU_spar, strainL_spar = self.strain(blade, xu_strain_spar, yu_strain_spar, xl_strain_spar, yl_strain_spar)
        strainU_te, strainL_te = self.strain(blade, xu_strain_te, yu_strain_te, xl_strain_te, yl_strain_te)

        outputs['blade_mass'] = blade_mass
        outputs['blade_moment_of_inertia'] = blade_moment_of_inertia
        outputs['freq_pbeam'] = freq
        outputs['freq_distance'] = np.float(freq[1]/freq[0])
        outputs['dx'] = dx
        outputs['dy'] = dy
        outputs['dz'] = dz
        outputs['strainU_spar'] = strainU_spar
        outputs['strainL_spar'] = strainL_spar
        outputs['strainU_te'] = strainU_te
        outputs['strainL_te'] = strainL_te

class TipDeflection(ExplicitComponent):
    # OpenMDAO component that computes the blade deflection at tip in yaw x-direction
    def setup(self):
        # Inputs
        self.add_input('dx_tip',        val=0.0,                    desc='deflection at tip in airfoil x-direction')
        self.add_input('dy_tip',        val=0.0,                    desc='deflection at tip in airfoil y-direction')
        self.add_input('dz_tip',        val=0.0,                    desc='deflection at tip in airfoil z-direction')
        self.add_input('theta_tip',     val=0.0,    units='deg',    desc='twist at tip section')
        self.add_input('pitch_load',    val=0.0,    units='deg',    desc='blade pitch angle')
        self.add_input('tilt',          val=0.0,    units='deg',    desc='tilt angle')
        self.add_input('3d_curv_tip',   val=0.0,    units='deg',    desc='total coning angle including precone and curvature')
        self.add_input('dynamicFactor', val=1.0,                    desc='a dynamic amplification factor to adjust the static deflection calculation') #)
        # Outputs
        self.add_output('tip_deflection', val=0.0,  units='m',      desc='deflection at tip in yaw x-direction')

    def compute(self, inputs, outputs):

        dx            = inputs['dx_tip']
        dy            = inputs['dy_tip']
        dz            = inputs['dz_tip']
        theta         = inputs['theta_tip']
        pitch         = inputs['pitch_load']
        azimuth       = 180.0 # The blade is assumed in front of the tower, although the loading may correspond to another azimuthal position
        tilt          = inputs['tilt']
        totalConeTip  = inputs['3d_curv_tip']
        dynamicFactor = inputs['dynamicFactor']

        theta = theta + pitch

        dr = DirectionVector(dx, dy, dz)
        delta = dr.airfoilToBlade(theta).bladeToAzimuth(totalConeTip).azimuthToHub(azimuth).hubToYaw(tilt)

        tip_deflection = dynamicFactor * delta.x

        outputs['tip_deflection'] = tip_deflection

class DesignConstraints(ExplicitComponent):
    # OpenMDAO component that formulates constraints on user-defined maximum strains, frequencies   
    def initialize(self):
        self.options.declare('wt_init_options')
        self.options.declare('opt_options')

    def setup(self):
        blade_init_options = self.options['wt_init_options']['blade']
        self.n_span = n_span = blade_init_options['n_span']
        self.n_freq = n_freq = blade_init_options['n_freq']
        self.opt_options   = opt_options   = self.options['opt_options']
        self.n_opt_spar_ss = n_opt_spar_ss = opt_options['blade_struct']['n_opt_spar_ss']
        self.n_opt_spar_ps = n_opt_spar_ps = opt_options['blade_struct']['n_opt_spar_ps']
        # Inputs strains
        self.add_input('strainU_spar',     val=np.zeros(n_span), desc='strain in spar cap on upper surface at location xu,yu_strain with loads P_strain')
        self.add_input('strainL_spar',     val=np.zeros(n_span), desc='strain in spar cap on lower surface at location xl,yl_strain with loads P_strain')

        self.add_input('min_strainU_spar', val=0.0, desc='minimum strain in spar cap suction side')
        self.add_input('max_strainU_spar', val=0.0, desc='minimum strain in spar cap pressure side')
        self.add_input('min_strainL_spar', val=0.0, desc='maximum strain in spar cap suction side')
        self.add_input('max_strainL_spar', val=0.0, desc='maximum strain in spar cap pressure side')
        
        self.add_input('s',                     val=np.zeros(n_span),       desc='1D array of the non-dimensional spanwise grid defined along blade axis (0-blade root, 1-blade tip)')
        self.add_input('s_opt_spar_ss',         val=np.zeros(n_opt_spar_ss),desc='1D array of the non-dimensional spanwise grid defined along blade axis to optimize the blade spar cap suction side')
        self.add_input('s_opt_spar_ps',         val=np.zeros(n_opt_spar_ss),desc='1D array of the non-dimensional spanwise grid defined along blade axis to optimize the blade spar cap suction side')

        # Input frequencies
        self.add_input('rated_Omega', val=0.0,                units='rpm', desc='rotor rotation speed at rated')
        self.add_input('delta_f',     val=1.1,                             desc='minimum margin between 3P and edge frequency')
        self.add_input('freq',        val=np.zeros(n_freq),   units='Hz',  desc='first nF natural frequencies')

        # Outputs
        self.add_output('constr_min_strainU_spar',     val=np.zeros(n_opt_spar_ss), desc='constraint for minimum strain in spar cap suction side')
        self.add_output('constr_max_strainU_spar',     val=np.zeros(n_opt_spar_ss), desc='constraint for maximum strain in spar cap suction side')
        self.add_output('constr_min_strainL_spar',     val=np.zeros(n_opt_spar_ps), desc='constraint for minimum strain in spar cap pressure side')
        self.add_output('constr_max_strainL_spar',     val=np.zeros(n_opt_spar_ps), desc='constraint for maximum strain in spar cap pressure side')
        self.add_output('constr_flap_f_above_3P',      val=0.0,                     desc='constraint on flap blade frequency to stay above 3P + delta')
        self.add_output('constr_edge_f_above_3P',      val=0.0,                     desc='constraint on edge blade frequency to stay above 3P + delta')

    def compute(self, inputs, outputs):
        
        # Constraints on blade strains
        s               = inputs['s']
        s_opt_spar_ss   = inputs['s_opt_spar_ss']
        s_opt_spar_ps   = inputs['s_opt_spar_ps']
        
        strainU_spar    = inputs['strainU_spar']
        strainL_spar    = inputs['strainL_spar']
        min_strainU_spar= inputs['min_strainU_spar']
        max_strainU_spar= inputs['max_strainU_spar']
        min_strainL_spar= inputs['min_strainL_spar']
        max_strainL_spar= inputs['max_strainL_spar']
        
        outputs['constr_min_strainU_spar'] = abs(np.interp(s_opt_spar_ss, s, strainU_spar)) / abs(min_strainU_spar)
        outputs['constr_max_strainU_spar'] = abs(np.interp(s_opt_spar_ss, s, strainU_spar)) / max_strainU_spar
        outputs['constr_min_strainL_spar'] = abs(np.interp(s_opt_spar_ps, s, strainL_spar)) / abs(min_strainL_spar)
        outputs['constr_max_strainL_spar'] = abs(np.interp(s_opt_spar_ps, s, strainL_spar)) / max_strainL_spar

        # Constraints on blade frequencies
        threeP = 3. * inputs['rated_Omega'] / 60.
        flap_f = inputs['freq'][0] # assuming the flap frequency is the first lowest
        edge_f = inputs['freq'][1] # assuming the edge frequency is the second lowest
        delta  = inputs['delta_f']
        outputs['constr_flap_f_above_3P'] = (threeP * delta) / flap_f
        outputs['constr_edge_f_above_3P'] = (threeP * delta) / edge_f
        
class RotorStructure(Group):
    # OpenMDAO group to compute the blade elastic properties, deflections, and loading
    def initialize(self):
        self.options.declare('wt_init_options')
        self.options.declare('opt_options')
    def setup(self):
        wt_init_options = self.options['wt_init_options']
        opt_options     = self.options['opt_options']

        # Get elastic properties by running precomp
        self.add_subsystem('precomp',   RunPreComp(wt_init_options = wt_init_options, opt_options = opt_options),    promotes=['r','theta','chord','EA','EIxx','EIyy','GJ','rhoA','rhoJ','x_ec','y_ec','Tw_iner','precurve','presweep','xu_strain_spar','xl_strain_spar','yu_strain_spar','yl_strain_spar','xu_strain_te','xl_strain_te','yu_strain_te','yl_strain_te'])
        # Compute frequencies
        self.add_subsystem('curvefem', RunCurveFEM(wt_init_options = wt_init_options), promotes=['r','EA','EIxx','EIyy','GJ','rhoA','rhoJ','Tw_iner','precurve','presweep'])
        # Load blade with rated conditions and compute aerodynamic forces
        promoteListAeroLoads =  ['r', 'theta', 'chord', 'Rtip', 'Rhub', 'hub_height', 'precone', 'tilt', 'airfoils_aoa', 'airfoils_Re', 'airfoils_cl', 'airfoils_cd', 'airfoils_cm', 'nBlades', 'rho', 'mu', 'Omega_load','pitch_load']
        # self.add_subsystem('aero_rated',        CCBladeLoads(wt_init_options = wt_init_options), promotes=promoteListAeroLoads)
        self.add_subsystem('gust',              GustETM())
        self.add_subsystem('aero_gust',         CCBladeLoads(wt_init_options = wt_init_options), promotes=promoteListAeroLoads)
        # self.add_subsystem('aero_storm_1yr',    CCBladeLoads(wt_init_options = wt_init_options), promotes=promoteListAeroLoads)
        # self.add_subsystem('aero_storm_50yr',   CCBladeLoads(wt_init_options = wt_init_options), promotes=promoteListAeroLoads)
        # Add centrifugal and gravity loading to aero loading
        promotes=['tilt','theta','rhoA','z','totalCone','z_az']
        self.add_subsystem('curvature',         BladeCurvature(wt_init_options = wt_init_options),  promotes=['r','precone','precurve','presweep','3d_curv','z_az'])
        promoteListTotalLoads = ['r', 'theta', 'tilt', 'rhoA', '3d_curv', 'z_az', 'aeroloads_Omega', 'aeroloads_pitch']
        # self.add_subsystem('tot_loads_rated',       TotalLoads(wt_init_options = wt_init_options),      promotes=promoteListTotalLoads)
        self.add_subsystem('tot_loads_gust',        TotalLoads(wt_init_options = wt_init_options),      promotes=promoteListTotalLoads)
        # self.add_subsystem('tot_loads_storm_1yr',   TotalLoads(wt_init_options = wt_init_options),      promotes=promoteListTotalLoads)
        # self.add_subsystem('tot_loads_storm_50yr',  TotalLoads(wt_init_options = wt_init_options),      promotes=promoteListTotalLoads)
        promoteListpBeam = ['r','EA','EIxx','EIyy','EIxy','GJ','rhoA','rhoJ','x_ec','y_ec','xu_strain_spar','xl_strain_spar','yu_strain_spar','yl_strain_spar','xu_strain_te','xl_strain_te','yu_strain_te','yl_strain_te','blade_mass']
        self.add_subsystem('pbeam',     RunpBEAM(wt_init_options = wt_init_options),      promotes=promoteListpBeam)
        self.add_subsystem('tip_pos',   TipDeflection(),                                  promotes=['tilt','pitch_load'])
        self.add_subsystem('constr',    DesignConstraints(wt_init_options = wt_init_options, opt_options = opt_options))

        # Aero loads to total loads
        # self.connect('aero_rated.loads_Px',     'tot_loads_rated.aeroloads_Px')
        # self.connect('aero_rated.loads_Py',     'tot_loads_rated.aeroloads_Py')
        # self.connect('aero_rated.loads_Pz',     'tot_loads_rated.aeroloads_Pz')
        self.connect('aero_gust.loads_Px',      'tot_loads_gust.aeroloads_Px')
        self.connect('aero_gust.loads_Py',      'tot_loads_gust.aeroloads_Py')
        self.connect('aero_gust.loads_Pz',      'tot_loads_gust.aeroloads_Pz')
        # self.connect('aero_storm_1yr.loads_Px', 'tot_loads_storm_1yr.aeroloads_Px')
        # self.connect('aero_storm_1yr.loads_Py', 'tot_loads_storm_1yr.aeroloads_Py')
        # self.connect('aero_storm_1yr.loads_Pz', 'tot_loads_storm_1yr.aeroloads_Pz')
        # self.connect('aero_storm_50yr.loads_Px', 'tot_loads_storm_50yr.aeroloads_Px')
        # self.connect('aero_storm_50yr.loads_Py', 'tot_loads_storm_50yr.aeroloads_Py')
        # self.connect('aero_storm_50yr.loads_Pz', 'tot_loads_storm_50yr.aeroloads_Pz')

        # Total loads to strains
        self.connect('tot_loads_gust.Px_af', 'pbeam.Px_af')
        self.connect('tot_loads_gust.Py_af', 'pbeam.Py_af')
        self.connect('tot_loads_gust.Pz_af', 'pbeam.Pz_af')

        # Blade distributed deflections to tip deflection
        self.connect('pbeam.dx', 'tip_pos.dx_tip', src_indices=[-1])
        self.connect('pbeam.dy', 'tip_pos.dy_tip', src_indices=[-1])
        self.connect('pbeam.dz', 'tip_pos.dz_tip', src_indices=[-1])
        self.connect('3d_curv',  'tip_pos.3d_curv_tip', src_indices=[-1])

        # Strains from pbeam to constraint
        self.connect('pbeam.strainU_spar', 'constr.strainU_spar')
        self.connect('pbeam.strainL_spar', 'constr.strainL_spar')

        # Frequencies from curvefem to constraint
        self.connect('curvefem.freq',      'constr.freq')  