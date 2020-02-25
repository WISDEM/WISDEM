import copy
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import PchipInterpolator
from openmdao.api import ExplicitComponent, Group
from wisdem.commonse.utilities import rotate, arc_length
from wisdem.rotorse.precomp import PreComp, Profile, Orthotropic2DMaterial, CompositeSection
import wisdem.pBeam._pBEAM as _pBEAM
from wisdem.commonse.csystem import DirectionVector
from wisdem.rotorse.rotor_cost import blade_cost_model

class RunPreComp(ExplicitComponent):
    # Openmdao component to run precomp and generate the elastic properties of a wind turbine blade
    def initialize(self):
        self.options.declare('analysis_options')
        self.options.declare('opt_options')

    def setup(self):
        blade_init_options = self.options['analysis_options']['blade']
        self.n_span        = n_span    = blade_init_options['n_span']
        self.n_webs        = n_webs    = blade_init_options['n_webs']
        self.n_layers      = n_layers  = blade_init_options['n_layers']
        af_init_options    = self.options['analysis_options']['airfoils']
        self.n_xy          = n_xy      = af_init_options['n_xy'] # Number of coordinate points to describe the airfoil geometry
        mat_init_options = self.options['analysis_options']['materials']
        self.n_mat = n_mat = mat_init_options['n_mat']
        self.verbosity     = self.options['analysis_options']['general']['verbosity']

        opt_options   = self.options['opt_options']
        self.te_ss_var   = opt_options['optimization_variables']['blade']['structure']['te_ss']['name']
        self.te_ps_var   = opt_options['optimization_variables']['blade']['structure']['te_ps']['name']
        self.spar_cap_ss_var = opt_options['optimization_variables']['blade']['structure']['spar_cap_ss']['name']
        self.spar_cap_ps_var = opt_options['optimization_variables']['blade']['structure']['spar_cap_ps']['name']

        # Outer geometry
        self.add_input('r',             val=np.zeros(n_span), units='m',   desc='radial locations where blade is defined (should be increasing and not go all the way to hub or tip)')
        self.add_input('theta',         val=np.zeros(n_span), units='deg', desc='Twist angle at each section (positive decreases angle of attack)')
        self.add_input('chord',         val=np.zeros(n_span), units='m',   desc='chord length at each section')
        self.add_input('pitch_axis',    val=np.zeros(n_span),                 desc='1D array of the chordwise position of the pitch axis (0-LE, 1-TE), defined along blade span.')
        self.add_input('precurve',      val=np.zeros(n_span),    units='m', desc='precurve at each section')
        self.add_input('presweep',      val=np.zeros(n_span),    units='m', desc='presweep at each section')
        self.add_input('coord_xy_interp',  val=np.zeros((n_span, n_xy, 2)), desc='3D array of the non-dimensional x and y airfoil coordinates of the airfoils interpolated along span for n_span stations.')

        # Rotor configuration
        self.add_input('uptilt',           val=0.0, units='deg',    desc='Nacelle uptilt angle. A standard machine has positive values.')
        self.add_discrete_input('n_blades',val=3,                   desc='Number of blades of the rotor.')

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

        # Outputs - Distributed beam properties
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

        # Outputs - Overall beam properties
        self.add_output('blade_mass',                val=0.0, units='kg',        desc='mass of one blade')
        self.add_output('blade_moment_of_inertia',   val=0.0, units='kg*m**2',   desc='mass moment of inertia of blade about hub')
        self.add_output('mass_all_blades',           val=0.0, units='kg',        desc='mass of all blades')
        self.add_output('I_all_blades',              shape=6, units='kg*m**2',   desc='mass moments of inertia of all blades in yaw c.s. order:Ixx, Iyy, Izz, Ixy, Ixz, Iyz')

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
        spar_cap_ss_var_ok = False
        spar_cap_ps_var_ok = False
        for i_layer in range(self.n_layers):
            if discrete_inputs['layer_name'][i_layer] == self.te_ss_var:
                te_ss_var_ok = True
            if discrete_inputs['layer_name'][i_layer] == self.te_ps_var:
                te_ps_var_ok = True
            if discrete_inputs['layer_name'][i_layer] == self.spar_cap_ss_var:
                spar_cap_ss_var_ok = True
            if discrete_inputs['layer_name'][i_layer] == self.spar_cap_ps_var:
                spar_cap_ps_var_ok = True

        if te_ss_var_ok == False:
            print('The layer at the trailing edge suction side is set to be optimized, but does not exist in the input yaml. Please check.')
        if te_ps_var_ok == False:
            print('The layer at the trailing edge pressure side is set to be optimized, but does not exist in the input yaml. Please check.')
        if spar_cap_ss_var_ok == False:
            print('The layer at the spar cap suction side is set to be optimized, but does not exist in the input yaml. Please check.')
        if spar_cap_ps_var_ok == False:
            print('The layer at the spar cap pressure side is set to be optimized, but does not exist in the input yaml. Please check.')
        region_loc_vars = [self.te_ss_var, self.te_ps_var, self.spar_cap_ss_var, self.spar_cap_ps_var]

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
                                    exit('Error in the definition of material ' + discrete_inputs['layer_name'][idx_sec] + '. It cannot fit in the section number ' + str(i) + ' at span location ' + str(inputs['r'][i]/inputs['r'][-1]*100.) + ' %.')
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
        
        sector_idx_strain_spar_cap_ss = [None if regs==None else regs[int(len(regs)/2)] for regs in region_loc_ss[self.spar_cap_ss_var]]
        sector_idx_strain_spar_cap_ps = [None if regs==None else regs[int(len(regs)/2)] for regs in region_loc_ps[self.spar_cap_ps_var]]
        sector_idx_strain_te_ss   = [None if regs==None else regs[int(len(regs)/2)] for regs in region_loc_ss[self.te_ss_var]]
        sector_idx_strain_te_ps   = [None if regs==None else regs[int(len(regs)/2)] for regs in region_loc_ps[self.te_ps_var]]

        # Get Beam Properties
        beam = PreComp(inputs['r'], inputs['chord'], inputs['theta'], inputs['pitch_axis'], 
                       inputs['precurve'], inputs['presweep'], profile, materials, upperCS, lowerCS, websCS, 
                       sector_idx_strain_spar_cap_ps, sector_idx_strain_spar_cap_ss, sector_idx_strain_te_ps, sector_idx_strain_te_ss)
        EIxx, EIyy, GJ, EA, EIxy, x_ec, y_ec, rhoA, rhoJ, Tw_iner, flap_iner, edge_iner = beam.sectionProperties()

        # outputs['eps_crit_spar'] = beam.panelBucklingStrain(sector_idx_strain_spar_cap_ss)
        # outputs['eps_crit_te'] = beam.panelBucklingStrain(sector_idx_strain_te_ss)

        xu_strain_spar, xl_strain_spar, yu_strain_spar, yl_strain_spar = beam.criticalStrainLocations(sector_idx_strain_spar_cap_ss, sector_idx_strain_spar_cap_ps)
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

        # Compute mass and inertia of blade and rotor
        blade_mass = np.trapz(rhoA, inputs['r'])
        blade_moment_of_inertia = np.trapz(rhoA * inputs['r']**2., inputs['r'])
        tilt    = inputs['uptilt']
        n_blades = discrete_inputs['n_blades']
        mass_all_blades = n_blades * blade_mass
        Ibeam = n_blades * blade_moment_of_inertia
        Ixx = Ibeam
        Iyy = Ibeam/2.0  # azimuthal average for 2 blades, exact for 3+
        Izz = Ibeam/2.0
        Ixy = 0.0
        Ixz = 0.0
        Iyz = 0.0  # azimuthal average for 2 blades, exact for 3+
        # rotate to yaw c.s.
        I = DirectionVector(Ixx, Iyy, Izz).hubToYaw(tilt)  # because off-diagonal components are all zero
        I_all_blades = np.array([I.x, I.y, I.z, Ixy, Ixz, Iyz])

        outputs['blade_mass']              = blade_mass
        outputs['blade_moment_of_inertia'] = blade_moment_of_inertia
        outputs['mass_all_blades']         = mass_all_blades
        outputs['I_all_blades']            = I_all_blades

        # Placeholder - rotor cost
        bcm             = blade_cost_model(verbosity = self.verbosity)
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
        self.options.declare('analysis_options')

    def setup(self):
        blade_init_options = self.options['analysis_options']['blade']
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

        # mycurve = _pBEAM.CurveFEM(inputs['Omega'], inputs['Tw_iner'], inputs['r'], inputs['precurve'], inputs['presweep'], inputs['rhoA'], True)
        mycurve = _pBEAM.CurveFEM(0., inputs['Tw_iner'], inputs['r'], inputs['precurve'], inputs['presweep'], inputs['rhoA'], True)
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

class RotorElasticity(Group):
    # OpenMDAO group to compute the blade elastic properties and natural frequencies
    def initialize(self):
        self.options.declare('analysis_options')
        self.options.declare('opt_options')
    def setup(self):
        analysis_options = self.options['analysis_options']
        opt_options     = self.options['opt_options']

        # Get elastic properties by running precomp
        self.add_subsystem('precomp',   RunPreComp(analysis_options = analysis_options, opt_options = opt_options),    promotes=['r','chord','theta','EA','EIxx','EIyy','GJ','rhoA','rhoJ','Tw_iner','precurve','presweep'])
        # Compute frequencies
        self.add_subsystem('curvefem', RunCurveFEM(analysis_options = analysis_options), promotes=['r','EA','EIxx','EIyy','GJ','rhoA','rhoJ','Tw_iner','precurve','presweep'])