
import openmdao.api as om
import numpy as np
import wisdem.commonse.utilities as util
import sys

class HubShell(om.ExplicitComponent):
    """
    Size the wind turbine rotor hub, modeled as a spherical shell
    
    Parameters
    ----------
    n_blades : integer
        Number of rotor blades
    flange_t2shell_t : float
        Ratio of flange thickness to shell thickness
    flange_OD2hub_D : float
        Ratio of flange outer diameter to hub diameter
    flange_ID2flange_OD : float
        Ratio of flange inner diameter to flange outer diameter
        (adjust to match shaft ID if necessary)
    rho : float, [kg/m**3]
        Density metal
    blade_root_diameter : float, [m]
        Outer diameter of blade root
    in2out_circ : float
        Safety factor applied on hub circumference. This factor determines the
        extra material needed between blade cutouts/holes in the hub to provide
        enough load carrying material. Good values are usually 1.15/1.2
    max_torque : float, [N*m]
        Max torque that the hub needs to resist (Mx in a hub aliged reference system)
    Xy : float, [Pa]
        Yield strength metal for hub (200MPa is a good value for SN Cast Iron
        GJS-350 for thick sections)
    stress_concentration : float
        Stress concentration factor. Stress concentration occurs at all fillets,
        notches, lifting lugs, hatches and are accounted for by assigning a
        stress concentration factor
    gamma : float
        Design safety factor
    metal_cost : float, [USD/kg]
        Unit cost metal

    Returns
    -------
    hub_mass : float, [kg]
        Total mass of the hub shell, including the flanges
    hub_diameter : float, [m]
        Outer diameter of the hub
    hub_cost : float, [USD]
        Cost of the hub shell, including flanges
    hub_cm : float, [m]
        Distance between hub/shaft flange and hub center of mass
    hub_I : numpy array[3], [kg*m**2]
        Total mass moment of inertia of the hub about its cm
    
    """
    def initialize(self):
        self.options.declare('verbosity', default=False)
        
    def setup(self):

        # Inputs
        self.add_discrete_input('n_blades',         val = 0)
        self.add_input('flange_t2shell_t',          val = 0.0)
        self.add_input('flange_OD2hub_D',           val = 0.0)
        self.add_input('flange_ID2flange_OD',       val = 0.0)
        self.add_input('rho',                       val = 0.0, units = 'kg/m**3')
        self.add_input('blade_root_diameter',       val = 0.0, units = 'm')
        self.add_input('in2out_circ',               val = 0.0)
        self.add_input('max_torque',                val = 0.0, units = 'N*m')
        self.add_input('Xy',                        val = 0.0, units = 'Pa')
        self.add_input('stress_concentration',      val = 0.0)
        self.add_input('gamma',                     val = 0.0)
        self.add_input('metal_cost',                val = 0.0, units = 'USD/kg')

        # Outputs
        self.add_output('hub_mass',                 val = 0.0, units = 'kg')
        self.add_output('hub_diameter',             val = 0.0, units = 'm')
        self.add_output('hub_cost',                 val = 0.0, units = 'USD')
        self.add_output('hub_cm',                   val = 0.0, units = 'm')
        self.add_output('hub_I',                    val = np.zeros(3), units = 'kg*m**2')

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        
        # Estimate diameter of sphere based on blade root diameter
        angle_btw_blades        = np.pi / discrete_inputs['n_blades']
        init_hub_diam           = inputs['blade_root_diameter'] / np.sin(angle_btw_blades)                                   
        # Compute minimum circumference hub
        init_hub_circ           = np.pi * init_hub_diam
        # Scale it up with safety factor
        dsgn_hub_circ           = init_hub_circ * inputs['in2out_circ']
        # Compute diameter and radius of the hub including safety factor
        dsgn_hub_diam           = dsgn_hub_circ / np.pi
        dsgn_hub_rad            = dsgn_hub_diam * 0.5
        # Determine max stress
        stress_allow_pa = inputs['Xy'] / (inputs['stress_concentration'] * inputs['gamma']) 
        # Size shell thickness, assuming max torsional stress from torque can not exceed design allowable stress, and solving, torsional stress (t=Tr/J), and polar moment of inertia (J=PI/32(Do^4-Di^4), for thickness.
        sph_hub_shell_thick = (((dsgn_hub_diam**4. - 32. / np.pi*inputs['max_torque']*dsgn_hub_rad/stress_allow_pa)**(1./4.)) - dsgn_hub_diam) / (-2.)
        # Compute volume and mass of the shell  
        sph_hub_vol = 4./3. * np.pi * (dsgn_hub_rad**3. - (dsgn_hub_rad - sph_hub_shell_thick)**3.)
        sph_hub_mass = sph_hub_vol * inputs['rho']
        # Assume outer (OD) and inner diameter (ID) of the flanges based on hub diameter
        main_flange_OD    = dsgn_hub_diam  * inputs['flange_OD2hub_D']
        main_flange_ID    = main_flange_OD * inputs['flange_ID2flange_OD']
        # Compute thickness, volume, and mass of main flange
        main_flange_thick = inputs['flange_t2shell_t'] * sph_hub_shell_thick
        main_flange_vol   = np.pi * main_flange_thick * ((main_flange_OD/2.)**2. - (main_flange_ID/2.)**2.)
        main_flange_mass  = main_flange_vol * inputs['rho']
        # Sum masses flange and hub
        hub_mass = main_flange_mass + sph_hub_mass
        # Compute cost
        hub_cost = hub_mass * inputs['metal_cost']
        # Compute distance between hub/shaft flange and hub center of mass
        hub_cm  = (main_flange_mass*main_flange_thick*0.5 + sph_hub_mass*dsgn_hub_rad) / (main_flange_mass + sph_hub_mass) 

        # Assign values to openmdao outputs 
        outputs['hub_diameter'] = dsgn_hub_diam
        outputs['hub_mass']     = hub_mass
        outputs['hub_cost']     = hub_cost
        outputs['hub_cm']       = hub_cm
        outputs['hub_I']        = (2./3.) * hub_mass * (0.5*dsgn_hub_diam)**2 * np.ones(3) # Spherical shell

        if self.options['verbosity']:
            sys.stderr.write('Spherical_Hub:\n')
            sys.stderr.write('  Sizing scale parameters: Flange_2_Hub {:.2f}   Flange_ID_2_OD {:.2f}\n'.format(inputs['flange_OD2hub_D'][0], inputs['flange_ID2flange_OD'][0]))
            sys.stderr.write('  Shell:   Mass {:8.1f} kg Diam {:4.2f} m             Thick {:.3f} m == {:5.1f} mm Vol {:5.2f} m^3\n'.format(sph_hub_mass[0], float(dsgn_hub_diam[0]), float(sph_hub_shell_thick[0]), 1000.*float(sph_hub_shell_thick[0]), sph_hub_vol[0]))
            sys.stderr.write('  Flange:  Mass {:8.1f} kg OD   {:4.2f} m ID   {:4.2f} m Thick {:.3f} m == {:5.1f} mm Vol {:5.2f} m^3\n'.format(float(main_flange_mass), float(main_flange_OD), float(main_flange_ID), float(main_flange_thick), 1000*float(main_flange_thick),float(main_flange_vol)))
            sys.stderr.write('  HubAssy: Mass {:8.1f} kg CM {:6.2f} m COST ${:.2f}\n'.format(float(hub_mass), float(hub_cm), float(hub_cost)))

class Spinner(om.ExplicitComponent):
    """
    Size the wind turbine rotor hub spinner, modeled as a spherical shell that wraps the hub 
    
    Parameters
    ----------
    n_blades : integer
        Number of rotor blades
    n_front_brackets : integer
        Number of front spinner brackets
    n_rear_brackets : integer
        Number of rear spinner brackets
    blade_root_diameter : float, [m]
        Outer diameter of blade root
    hub_diameter : float, [m]
        Outer diameter of the hub
    clearance_hub_spinner : float, [m]
        Clearance between spinner and hub
    spin_hole_incr : float
        Ratio between access hole diameter in the spinner and blade root diameter.
        Typical value 1.2
    gust_ws : float, [m/s]
        Extreme gust wind speed
    gamma : float
        Scaling factor of the thrust forces on spinner
    composite_Xt : float, [Pa]
        Tensile strength of the composite material of the shell.
        A glass CFM (continuous fiber mat) is often used.
    composite_SF : float
        Safety factor composite shell
    composite_rho : float, [kg/m**3]
        Density of composite of the shell
    Xy : float, [Pa]
        Yield strength metal
    metal_SF : float
        Safety factor metal
    metal_rho : float, [kg/m**3]
        Density metal
    composite_cost : float, [USD/kg]
        Unit cost composite of the shell
    metal_cost : float, [USD/kg]
        Unit cost metal

    Returns
    -------
    spinner_mass : float, [m]
        Total mass of the spinner
    spinner_diameter : float, [kg]
        Outer diameter of the spinner
    spinner_cost : float, [kg]
        Cost of the spinner
    spinner_cm : float, [m]
        Distance between center of mass of the spinner and main flange
    spinner_I : numpy array[3], [kg*m**2]
        Total mass moment of inertia of the spinner about its cm
    
    """
    def initialize(self):
        self.options.declare('verbosity', default=False)
        
    def setup(self):

        # Inputs
        self.add_discrete_input('n_blades',         val = 0)
        self.add_discrete_input('n_front_brackets', val = 0)
        self.add_discrete_input('n_rear_brackets',  val = 0)
        self.add_input('blade_root_diameter',       val = 0.0, units = 'm')
        self.add_input('hub_diameter',              val = 0.0, units = 'm')
        self.add_input('clearance_hub_spinner',     val = 0.0, units = 'm')
        self.add_input('spin_hole_incr',            val = 0.0)
        self.add_input('gust_ws',                   val = 0.0, units = 'm/s')
        self.add_input('gamma',              val = 0.0)
        self.add_input('composite_Xt',              val = 0.0, units = 'Pa')
        self.add_input('composite_SF',              val = 0.0)
        self.add_input('composite_rho',             val = 0.0, units = 'kg/m**3')
        self.add_input('Xy',                        val = 0.0, units = 'Pa')
        self.add_input('metal_SF',                  val = 0.0)
        self.add_input('metal_rho',                 val = 0.0, units = 'kg/m**3')
        self.add_input('composite_cost',            val = 0.0, units = 'USD/kg')
        self.add_input('metal_cost',                val = 0.0, units = 'USD/kg')

        # Outputs
        self.add_output('spinner_diameter',         val = 0.0, units = 'm')
        self.add_output('spinner_mass',             val = 0.0, units = 'kg')
        self.add_output('spinner_cost',             val = 0.0, units = 'kg')
        self.add_output('spinner_cm',               val = 0.0, units = 'm')
        self.add_output('spinner_I',                val = np.zeros(3), units = 'kg*m**2')

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        
        # Increase hub diameter by twice the clearance between hub and spinner. This gives us the spherical spinner diameter, radius, and circumference
        sph_spin_diam           = inputs['hub_diameter'] + (2.*inputs['clearance_hub_spinner'])
        sph_spin_rad            = 0.5 * sph_spin_diam
        sph_spin_circ           = np.pi * sph_spin_diam
        dsgn_hub_circ           = np.pi * inputs['hub_diameter']
        # Compute the width of the panel between blade cutouts
        spin_panel_width        = (sph_spin_circ - dsgn_hub_circ) / 3.
        # Compute the access hole diameter given blade root diameter
        spin_acc_hole_diam      = inputs['blade_root_diameter'] * inputs['spin_hole_incr']
        # Estimate thrust pressure on spinner given a wind speed and a multiplicative factor
        extr_gust_dsgn_pressure = 0.5 * 1.225 * (inputs['gust_ws'] ** 2.)* inputs['gamma'] 
        # Compute max allowable tensile strength of composite and max allowable yield strength metal given material properties and safety factors
        allow_tensile_strength  = inputs['composite_Xt'] / inputs['composite_SF']
        allow_yield_strength    = inputs['Xy'] / inputs['metal_SF']
        # Estimate thickness of the shell of the spinner. Stress equation for a flat plate with simply supported edges, with a load equal to the extreme gust pressure load.  
        # The equation is [Stress=(.75*P*b^2)/(t^2*(1.61*(b/a)^3 +1)).  See Roarks for reference.  Shell is curved and not flat but simplifying for calculation purposes.
        spin_shell_thickness    = np.sqrt((0.75 * extr_gust_dsgn_pressure * spin_panel_width ** 2.) / (allow_tensile_strength*(1.61*(spin_panel_width/sph_spin_diam) ** 3. + 1.)))
        # Compute volume and mass of the spinner shell
        spin_shell_volume       = (4./3.) * np.pi * (sph_spin_rad ** 3. - ((sph_spin_diam - 2.*spin_shell_thickness)/2.) ** 3.)
        spin_shell_mass         = spin_shell_volume * inputs['composite_rho']
        # Estimate area, volume, and mass of the spherical caps that are removed because of blade access
        sph_cap_area            = 2.  * np.pi * sph_spin_rad * (sph_spin_rad - np.sqrt(sph_spin_rad ** 2. - (spin_acc_hole_diam/2.) ** 2.))
        sph_caps_volume         = discrete_inputs['n_blades'] * sph_cap_area * spin_shell_thickness
        sph_caps_mass           = sph_caps_volume * inputs['composite_rho']
        # Estimate main flange diameter, area, volume, and mass
        main_flange_diam        = 0.6 * inputs['hub_diameter']
        main_flange_area        = 2. * np.pi * sph_spin_rad * (sph_spin_rad - np.sqrt(sph_spin_rad ** 2. - (main_flange_diam/2.) ** 2.))
        main_flange_volume      = main_flange_area * spin_shell_thickness
        main_flange_mass        = main_flange_volume * inputs['composite_rho']
        spin_shell_mass         = spin_shell_mass - sph_caps_mass - main_flange_mass

        # Compute frontal area of spherical spinner
        spin_frontal_area = np.pi * (sph_spin_diam ** 2.)/4.
        # Compute load given frontal area
        frontal_gust_load = spin_frontal_area * extr_gust_dsgn_pressure
        # Compute load on single bracket
        bracket_load = frontal_gust_load / (discrete_inputs['n_front_brackets'] + discrete_inputs['n_rear_brackets'])
        # Compute bending moment on bracket
        bracket_bending_moment = bracket_load * inputs['clearance_hub_spinner']
        # Assume bracket width is half of the spinner panel width
        bracket_width = spin_panel_width / 2.
        # Compute bracket thickness given loads and metal properties
        bracket_thickness = np.sqrt((6. * bracket_bending_moment) / (bracket_width * allow_yield_strength))
        # Sent warning if bracket thickness is small than 16mm. This is a switch between material properties, not implemented here to prevent discontinuities
        if bracket_thickness < 0.016:
            print('The thickness of the bracket of the hub spinner is smaller than 16 mm. You may increase the Yield strength of the metal.')
            print('The standard approach adopted 235 MPa below 16 mm and 225 above 16 mm.') 
        # Assume flang is 25% of bracket length
        bracket_flange_length = inputs['clearance_hub_spinner'] * 0.25
        # Compute bracket volume and mass and total mass of all brackets
        bracket_volume = (inputs['clearance_hub_spinner'] + bracket_flange_length + bracket_flange_length) * bracket_width * bracket_thickness
        bracket_mass = bracket_volume * inputs['metal_rho']
        bracket_mass_total = bracket_mass * (discrete_inputs['n_front_brackets'] + discrete_inputs['n_rear_brackets'])

        mass = spin_shell_mass + bracket_mass_total
        
        # Compute outputs and assign them to openmdao outputs
        outputs['spinner_diameter']         = sph_spin_diam
        outputs['spinner_mass']             = mass
         # Spinner and hub are assumed to be concentric (spinner wraps hub)
        outputs['spinner_cm']               = inputs['hub_diameter'] / 2.
        outputs['spinner_cost']             = spin_shell_mass * inputs['composite_cost'] + bracket_mass_total * inputs['metal_cost']
        outputs['spinner_I']                = (2./3.) * mass * (0.5*sph_spin_diam)**2 * np.ones(3) # Spherical shell
        
        # Print to screen if verbosity option is on
        if self.options['verbosity']:
            sys.stderr.write('Spherical spinner: mass {:.1f} kg = Shell {:.1f} kg + Bracket structures {:.1f} kg\n'.format(float(outputs['total_mass']), float(spin_shell_mass), float(bracket_mass_total)))
            sys.stderr.write('Spherical spinner: spinner outer diameter {:.1f} m\n'.format(float(sph_spin_diam)))
            sys.stderr.write('Spherical spinner: cost ${:.2f}  center of mass {:.2f} m\n'.format(float(outputs['cost']), float(outputs['cm'])))

class PitchSystem(om.ExplicitComponent):
    """
    Semi-empirical model to size the pitch system of the wind turbine rotor.
    
    Parameters
    ----------
    n_blades : integer
        Number of rotor blades
    blade_mass : float, [kg]
        Total mass of one blade
    rho : float, [kg/m**3]
        Density of the material used for the pitch system
    Xy : float, [Pa]
        Yield strength metal
    scaling_factor : float, [kg/m**3]
        Scaling factor to tune the total mass (0.54 is recommended for modern designs)
    BRFM : float, [N*m]
        Flapwise bending moment at blade root
    hub_diameter : float, [m]
        Outer diameter of the hub

    Returns
    -------
    pitch_mass : float, [kg]
        Total mass of the pitch system
    pitch_cost : float, [USD]
        Cost of the pitch system
    pitch_I : float, [kg*m**2]
        Total mass moment of inertia of the pitch system about central point
    
    """
    def initialize(self):
        self.options.declare('verbosity', default=False)
        
    def setup(self):

        # Inputs
        self.add_discrete_input('n_blades',     val = 0)
        self.add_input('blade_mass',            val = 0.0, units = 'kg')
        self.add_input('rho',                   val = 0.0, units = 'kg/m**3')
        self.add_input('Xy',                    val = 0.0, units = 'Pa')
        self.add_input('scaling_factor',        val = 0.0, units = 'kg/m**3')
        self.add_input('BRFM',                  val = 0.0, units = 'N*m')
        self.add_input('hub_diameter',          val = 0.0, units = 'm')
        # Outputs
        self.add_output('pitch_mass',           val = 0.0, units = 'kg')
        self.add_output('pitch_cost',           val = 0.0, units = 'USD')
        self.add_output('pitch_I',              val = np.zeros(3), units = 'kg*m**2')

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        mass  = inputs['scaling_factor'] * (0.22 * inputs['blade_mass'] * discrete_inputs['n_blades'] + 12.6 * np.abs(inputs['BRFM']) * inputs['rho'] / inputs['Xy'])
        r_hub = 0.5*inputs['hub_diameter']
        I     = mass * r_hub**2 * np.array([1.0, 0.5, 0.5])

        outputs['pitch_mass'] = mass
        outputs['pitch_cost'] = 0.0
        outputs['pitch_I'] = I

        if self.options['verbosity']:
            sys.stderr.write('PitchSystem IN : blade mass {:.1f} kg rbmy {:.1f} Nm\n'.format(float(inputs['blade_mass']), float(inputs['BRFM'])))
            sys.stderr.write('PitchSystem OUT: mass {:.1f} kg\n'.format(float(mass)))
    
class Hub_Adder(om.ExplicitComponent):
    """
    Aggregates components for total hub system mass, center of mass, and mass moment of inertia
    
    Parameters
    ----------
    pitch_mass : float, [kg]
        Total mass of the pitch system
    pitch_cost : float, [kg]
        Cost of the pitch system
    pitch_I : numpy array[3], [kg*m**2]
        Total mass moment of inertia of the pitch system about central point
    hub_mass : float, [kg]
        Total mass of the hub shell, including the flanges
    hub_cost : float, [USD]
        Cost of the hub shell, including flanges
    hub_cm : float, [m]
        Distance between hub/shaft flange and hub center of mass
    hub_I : numpy array[3], [kg*m**2]
        Total mass moment of inertia of the hub about its cm
    spinner_mass : float, [kg]
        Total mass of the spinner
    spinner_cost : float, [kg]
        Cost of the spinner
    spinner_cm : float, [m]
        Radius / Distance between center of mass of the spinner and outer surface
    spinner_I : numpy array[3], [kg*m**2]
        Total mass moment of inertia of the spinner about its cm

    Returns
    -------
    hub_system_mass : float, [kg]
        Total mass of the hub system
    hub_system_cost : float, [USD]
        Cost of the hub system
    hub_system_cm : float, [m]
        Distance between hub/main shaft flange and center of mass of the hub system
    hub_system_I : numpy array[3], [kg*m**2]
        Total mass moment of inertia of the hub system about its center of mass

    """
    def setup(self):
        self.add_input('pitch_mass',           val = 0.0, units = 'kg')
        self.add_input('pitch_cost',           val = 0.0, units = 'USD')
        self.add_input('pitch_I',              val = np.zeros(3), units = 'kg*m**2')

        self.add_input('hub_mass',             val = 0.0, units = 'kg')
        self.add_input('hub_cost',             val = 0.0, units = 'USD')
        self.add_input('hub_cm',               val = 0.0, units = 'm')
        self.add_input('hub_I',                val = np.zeros(3), units = 'kg*m**2')

        self.add_input('spinner_mass',         val = 0.0, units = 'kg')
        self.add_input('spinner_cost',         val = 0.0, units = 'kg')
        self.add_input('spinner_cm',           val = 0.0, units = 'm')
        self.add_input('spinner_I',            val = np.zeros(3), units = 'kg*m**2')

        self.add_output('hub_system_mass',     val = 0.0, units = 'kg')
        self.add_output('hub_system_cost',     val = 0.0, units = 'USD')
        self.add_output('hub_system_cm',       val = 0.0, units = 'm')
        self.add_output('hub_system_I',        val = np.zeros(3), units = 'kg*m**2')

    def compute(self, inputs, outputs):
        # Unpack inputs
        m_pitch = float(inputs['pitch_mass'])
        m_hub   = float(inputs['hub_mass'])
        m_spin  = float(inputs['spinner_mass'])
        cm_hub  = float(inputs['hub_cm'])
        cm_spin = float(inputs['spinner_cm'])
        
        # Mass and cost totals
        m_total  = m_pitch + m_hub + m_spin
        c_total  = inputs['pitch_cost'] + inputs['hub_cost'] + inputs['spinner_cost']

        # CofM total
        cm_total = ( (m_pitch + m_hub)*cm_hub + m_spin*cm_spin ) / m_total

        # I total
        I_total  = util.assembleI( np.zeros(6) )
        r        = np.array([cm_hub-cm_total, 0.0, 0.0])
        I_total += util.assembleI(np.r_[inputs['hub_I'], np.zeros(3)])     + m_hub   * (np.dot(r, r)*np.eye(3) - np.outer(r, r))
        I_total += util.assembleI(np.r_[inputs['pitch_I'], np.zeros(3)])   + m_pitch * (np.dot(r, r)*np.eye(3) - np.outer(r, r))
        r        = np.array([cm_spin-cm_total, 0.0, 0.0])
        I_total += util.assembleI(np.r_[inputs['spinner_I'], np.zeros(3)]) + m_spin  * (np.dot(r, r)*np.eye(3) - np.outer(r, r))

        # Outputs
        outputs['hub_system_mass'] = m_total
        outputs['hub_system_cost'] = c_total
        outputs['hub_system_cm']   = cm_total
        outputs['hub_system_I']    = util.unassembleI( I_total )[:3]
    
class Hub_System(om.Group):
    """
    Openmdao group to model the hub system, which includes pitch system, spinner, and hub
    """
    # def initialize(self):
        
    def setup(self):
        # analysis_options = self.options['analysis_options']
        # opt_options     = self.options['opt_options']
        ivc = om.IndepVarComp()
        ivc.add_output('flange_t2shell_t',          val = 0.0)
        ivc.add_output('flange_OD2hub_D',           val = 0.0)
        ivc.add_output('flange_ID2flange_OD',       val = 0.0)
        ivc.add_output('in2out_circ',               val = 0.0)
        ivc.add_output('stress_concentration',      val = 0.0)
        ivc.add_discrete_output('n_front_brackets', val = 0)
        ivc.add_discrete_output('n_rear_brackets',  val = 0)
        ivc.add_output('clearance_hub_spinner',     val = 0.0, units = 'm')
        ivc.add_output('spin_hole_incr',            val = 0.0)
        
        self.add_subsystem('ivc', ivc, promotes=['*'])
        self.add_subsystem('hub_shell',      HubShell(),    promotes=['n_blades', 'hub_mass', 'hub_diameter', 'hub_cost', 'hub_cm', 'hub_I',
                                                                      'flange_t2shell_t','flange_OD2hub_D','flange_ID2flange_OD','in2out_circ',
                                                                      'stress_concentration'])
        self.add_subsystem('spinner',        Spinner(),     promotes=['n_blades', 'hub_diameter', 'spinner_mass', 'spinner_cost', 'spinner_cm', 'spinner_I',
                                                                      'n_front_brackets','n_rear_brackets','clearance_hub_spinner','spin_hole_incr'])
        self.add_subsystem('pitch_system',   PitchSystem(), promotes=['n_blades', 'hub_diameter', 'pitch_mass', 'pitch_cost', 'pitch_I'])
        self.add_subsystem('adder',          Hub_Adder(), promotes=['hub_mass', 'hub_cost', 'hub_cm', 'hub_I',
                                                                    'spinner_mass', 'spinner_cost', 'spinner_cm', 'spinner_I',
                                                                    'pitch_mass', 'pitch_cost', 'pitch_I',
                                                                    'hub_system_mass', 'hub_system_cost', 'hub_system_cm', 'hub_system_I'])

if __name__ == "__main__":

    hub_prob = om.Problem(model=Hub_System())
    hub_prob.setup()
    
    hub_prob['pitch_system.blade_mass']         = 17000.
    hub_prob['pitch_system.BRFM']               = 1.e+6
    hub_prob['pitch_system.n_blades']           = 3
    hub_prob['pitch_system.scaling_factor']     = 0.54
    hub_prob['pitch_system.rho']                = 7850.
    hub_prob['pitch_system.Xy']                 = 371.e+6

    hub_prob['hub_shell.blade_root_diameter']   = 4.5
    hub_prob['hub_shell.n_blades']              = 3
    hub_prob['hub_shell.flange_t2shell_t']      = 4.
    hub_prob['hub_shell.flange_OD2hub_D']       = 0.5
    hub_prob['hub_shell.flange_ID2flange_OD']   = 0.8
    hub_prob['hub_shell.rho']                   = 7200.
    hub_prob['hub_shell.in2out_circ']           = 1.2 
    hub_prob['hub_shell.max_torque']            = 199200777.51 # Value assumed during model development
    hub_prob['hub_shell.Xy']                    = 200.e+6
    hub_prob['hub_shell.stress_concentration']  = 2.5
    hub_prob['hub_shell.gamma']                 = 2.0
    hub_prob['hub_shell.metal_cost']            = 3.00

    hub_prob['spinner.n_front_brackets']        = 3
    hub_prob['spinner.n_rear_brackets']         = 3
    hub_prob['spinner.n_blades']                = 3
    hub_prob['spinner.blade_root_diameter']     = hub_prob['hub_shell.blade_root_diameter']
    hub_prob['spinner.clearance_hub_spinner']   = 0.5
    hub_prob['spinner.spin_hole_incr']          = 1.2
    hub_prob['spinner.gust_ws']                 = 70
    hub_prob['spinner.gamma']                   = 1.5
    hub_prob['spinner.composite_Xt']            = 60.e6
    hub_prob['spinner.composite_SF']            = 1.5
    hub_prob['spinner.composite_rho']           = 1600.
    hub_prob['spinner.Xy']                      = 225.e+6
    hub_prob['spinner.metal_SF']                = 1.5
    hub_prob['spinner.metal_rho']               = 7850.
    hub_prob['spinner.composite_cost']          = 7.00
    hub_prob['spinner.metal_cost']              = 3.00

    hub_prob.run_model()

    print('Pitch system mass: ' + str(hub_prob['pitch_system.pitch_mass'][0]) + ' kg')
    print('Hub shell mass: ' + str(hub_prob['hub_shell.hub_mass'][0]) + ' kg')
    print('Hub shell outer diameter: ' + str(hub_prob['hub_shell.hub_diameter'][0]) + ' m')
    print('Hub shell cost: ' + str(hub_prob['hub_shell.hub_cost'][0]) + ' USD')
    print('Distance btw flange and cm of hub shell: ' + str(hub_prob['hub_shell.hub_cm'][0]) + ' m')
    print('Mass moment of inertia of hub shell: ' + str(hub_prob['hub_shell.hub_I']) + 'kg * m2')
    print('Spinner mass: ' + str(hub_prob['spinner.spinner_mass'][0]) + ' kg')
    print('Spinner outer diameter: ' + str(hub_prob['spinner.spinner_diameter'][0]) + ' m')
    print('Spinner cost: ' + str(hub_prob['spinner.spinner_cost'][0]) + ' USD')
    print('Distance btw flange and cm of spinner: ' + str(hub_prob['spinner.spinner_cm'][0]) + ' m')
    print('Mass moment of inertia of spinner: ' + str(hub_prob['spinner.spinner_I']) + 'kg * m2')
    print('Overall hub system mass: ' + str(hub_prob['adder.hub_system_mass'][0]) + ' kg')
    print('Overall hub system cost: ' + str(hub_prob['adder.hub_system_cost'][0]) + ' USD')
    print('Distance btw shaft flange and cm of overall hub system: ' + str(hub_prob['adder.hub_system_cm'][0]) + ' m')
    print('Mass moment of inertia of the overall hub system: ' + str(hub_prob['adder.hub_system_I']) + 'kg * m2')

