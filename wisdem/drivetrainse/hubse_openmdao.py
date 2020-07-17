
import openmdao.api as om
import numpy as np
import sys

class PitchSystem(om.ExplicitComponent):
    """
    Semi-empirical model to size the pitch system of the wind turbine rotor.
    
    Parameters
    ----------
    n_blades : integer
        Number of rotor blades
    blade_mass : float
        Total mass of one blade
    rho : float
        Density of the material used for the pitch system
    scaling_factor : float
        Scaling factor to tune the total mass (0.54 is recommended for modern designs)
    BRFM : float
        Flapwise bending moment at blade root

    Returns
    -------
    total_mass : float
        Total mass of the pitch system
    
    """
    def initialize(self):
        self.options.declare('verbosity', default=False)
        
    def setup(self):

        # Inputs
        self.add_discrete_input('n_blades',     val = 0,                      desc = 'Number of rotor blades')
        self.add_input('blade_mass',            val = 0.0, units = 'kg',      desc = 'Total mass of one blade')
        self.add_input('rho',                   val = 0.0, units = 'kg/m**3', desc = 'Density of the metal used for the pitch system')
        self.add_input('Xy',                    val = 0.0, units = 'Pa',      desc = 'Yield strength of the metal used for the pitch system')
        self.add_input('scaling_factor',        val = 0.0, units = 'kg/m**3', desc = 'Scaling factor to tune the total mass (0.54 is recommended for modern designs)')
        self.add_input('BRFM',                  val = 0.0, units = 'N*m',     desc = 'Flapwise bending moment at blade root')
        # Outputs
        self.add_output('mass',                 val = 0.0, units = 'kg',      desc = 'Total mass of the pitch system')

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        mass            = inputs['scaling_factor'] * (0.22 * inputs['blade_mass'] * discrete_inputs['n_blades'] + 12.6 * np.abs(inputs['BRFM']) * inputs['rho'] / inputs['Xy'])
        outputs['mass'] = mass        

        if self.options['verbosity']:
            sys.stderr.write('PitchSystem IN : blade mass {:.1f} kg rbmy {:.1f} Nm\n'.format(float(inputs['blade_mass']), float(inputs['BRFM'])))
            sys.stderr.write('PitchSystem OUT: mass {:.1f} kg\n'.format(float(mass)))

class HubShell(om.ExplicitComponent):
    """
    Size the shell of the wind turbine rotor hub
    
    Parameters
    ----------
    n_blades : integer
        Number of rotor blades
    flange_t2shell_t : float
        Ratio of flange thickness to shell thickness
    flange_OD2hub_D : float
        Ratio of flange outer diameter to hub diameter
    flange_ID2flange_OD : float
        Ratio of flange inner diameter to flange outer diameter (adjust to match shaft ID if necessary)
    rho : float
        Density metal
    blade_root_diameter : float
        Outer diameter of blade root
    in2out_circ : float
        Safety factor applied on hub circumference
    max_torque : float
        Safety factor applied on hub circumference
    Xy : float
        Yield strength metal for hub (200MPa)
    stress_concentration : float
        Stress concentration factor
    reserve_factor : float
        Safety factor
    metal_cost : float
        Unit cost metal

    Returns
    -------
    total_mass : float
        Total mass of the hub shell, including the flanges
    outer_diameter : float
        Outer diameter of the hub
    cost : float
        Cost of the hub shell, including flanges
    cm : float
        Distance between hub/shaft flange and hub center of mass
    
    """
    def initialize(self):
        self.options.declare('verbosity', default=False)
        
    def setup(self):

        # Inputs
        self.add_discrete_input('n_blades',         val = 0,                        desc = 'Number of rotor blades')
        self.add_input('flange_t2shell_t',          val = 0.0,                      desc = 'Ratio of flange thickness to shell thickness')
        self.add_input('flange_OD2hub_D',           val = 0.0,                      desc = 'Ratio of flange outer diameter to hub diameter')
        self.add_input('flange_ID2flange_OD',       val = 0.0,                      desc = 'Ratio of flange inner diameter to flange outer diameter (adjust to match shaft ID if necessary)')
        self.add_input('rho',                       val = 0.0, units = 'kg/m**3',   desc = 'Density of the material of the hub')
        self.add_input('blade_root_diameter',       val = 0.0, units = 'm',         desc = 'Blade root diameter')
        self.add_input('in2out_circ',               val = 0.0,                      desc = 'Safety factor applied on hub circumference')
        self.add_input('max_torque',                val = 0.0, units = 'N*m',       desc = 'Safety factor applied on hub circumference')
        self.add_input('Xy',                        val = 0.0, units = 'Pa',        desc = 'Yield strength metal for hub (200MPa)')
        self.add_input('stress_concentration',      val = 0.0,                      desc = 'Stress concentration factor')
        self.add_input('reserve_factor',            val = 0.0,                      desc = 'Safety factor')
        self.add_input('metal_cost',                val = 0.0, units = 'USD/kg',    desc = 'Unit cost metal')

        # Outputs
        self.add_output('total_mass',               val = 0.0, units = 'kg',        desc = 'Total mass of the hub shell, including the flanges.')
        self.add_output('outer_diameter',           val = 0.0, units = 'm',         desc = 'Outer diameter of the hub')
        self.add_output('cost',                     val = 0.0, units = 'USD',       desc = 'Cost of the hub shell, including flanges')
        self.add_output('cm',                       val = 0.0, units = 'm',         desc = 'Distance between hub/shaft flange and hub center of mass')

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
        stress_allow_pa = inputs['Xy'] / (inputs['stress_concentration'] * inputs['reserve_factor']) 
        # Size shell thickness
        sph_hub_shell_thick = (((dsgn_hub_diam**4. - 32. / np.pi*inputs['max_torque']*dsgn_hub_rad/stress_allow_pa)**(1./4.)) - dsgn_hub_diam) / (-2.)
        # Compute volume and mass of the shell  
        sph_hub_vol = 4./3. * np.pi * (dsgn_hub_rad**3. - (dsgn_hub_rad - sph_hub_shell_thick)**3.)
        sph_hub_mass = sph_hub_vol * inputs['rho']
        # Compute outer (OD) and inner diameter (ID) of the flanges
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
        outputs['outer_diameter'] = dsgn_hub_diam
        outputs['total_mass']     = hub_mass
        outputs['cost']           = hub_cost
        outputs['cm']             = hub_cm

        if self.options['verbosity']:
            sys.stderr.write('Spherical_Hub:\n')
            sys.stderr.write('  Sizing scale parameters: Flange_2_Hub {:.2f}   Flange_ID_2_OD {:.2f}\n'.format(inputs['flange_OD2hub_D'][0], inputs['flange_ID2flange_OD'][0]))
            sys.stderr.write('  Shell:   Mass {:8.1f} kg Diam {:4.2f} m             Thick {:.3f} m == {:5.1f} mm Vol {:5.2f} m^3\n'.format(sph_hub_mass[0], float(dsgn_hub_diam[0]), float(sph_hub_shell_thick[0]), 1000.*float(sph_hub_shell_thick[0]), sph_hub_vol[0]))
            sys.stderr.write('  Flange:  Mass {:8.1f} kg OD   {:4.2f} m ID   {:4.2f} m Thick {:.3f} m == {:5.1f} mm Vol {:5.2f} m^3\n'.format(float(main_flange_mass), float(main_flange_OD), float(main_flange_ID), float(main_flange_thick), 1000*float(main_flange_thick),float(main_flange_vol)))
            sys.stderr.write('  HubAssy: Mass {:8.1f} kg CM {:6.2f} m COST ${:.2f}\n'.format(float(hub_mass), float(hub_cm), float(hub_cost)))

class Spinner(om.ExplicitComponent):
    """
    Size wind turbine rotor spinner
    
    Parameters
    ----------
    n_blades : integer
        Number of rotor blades
    n_front_brackets : integer
        Number of front spinner brackets
    n_rear_brackets : integer
        Number of rear spinner brackets
    blade_root_diameter : float
        Outer diameter of blade root
    hub_diameter : float
        Outer diameter of the hub
    clearance_hub_spinner : float
        Clearance between spinner and hub
    spin_hole_incr : float
        Ratio between access hole diameter in the spinner and blade root diameter
    gust_ws : float
        Extreme gust wind speed
    load_scaling : float
        Scaling factor of the thrust forces on spinner
    composite_Xt : float
        Tensile strength of the composite material of the shell
    composite_SF : float
        Safety factor composite shell
    composite_rho : float
        Density of composite of the shell
    Xy : float
        Yield strength metal
    metal_SF : float
        Safety factor metal
    metal_rho : float
        Density metal
    composite_cost : float
        Unit cost composite of the shell
    metal_cost : float
        Unit cost metal

    Returns
    -------
    total_mass : float
        Total mass of the spinner
    diameter : float
        Outer diameter of the spinner
    cost : float
        Cost of the spinner
    cm : float
        Radius / Distance between center of mass of the spinner and outer surface
    
    """
    def initialize(self):
        self.options.declare('verbosity', default=False)
        
    def setup(self):

        # Inputs
        self.add_discrete_input('n_blades',         val = 0,                        desc = 'Number of rotor blades')
        self.add_discrete_input('n_front_brackets', val = 0,                        desc = 'Number of front spinner brackets')
        self.add_discrete_input('n_rear_brackets',  val = 0,                        desc = 'Number of rear spinner brackets')
        self.add_input('blade_root_diameter',       val = 0.0, units = 'm',         desc = 'Blade root diameter')
        self.add_input('hub_diameter',              val = 0.0, units = 'm',         desc = 'Outer diameter of the hub')
        self.add_input('clearance_hub_spinner',     val = 0.0, units = 'm',         desc = 'Clearance between spinner and hub')
        self.add_input('spin_hole_incr',            val = 0.0,                      desc = 'Ratio between access hole diameter in the spinner and blade root diameter')
        self.add_input('gust_ws',                   val = 0.0, units = 'm/s',       desc = 'Extreme gust wind speed')
        self.add_input('load_scaling',              val = 0.0,                      desc = 'Scaling factor of the thrust forces on spinner')
        self.add_input('composite_Xt',              val = 0.0, units = 'Pa',        desc = 'Tensile strength of the composite material of the shell')
        self.add_input('composite_SF',              val = 0.0,                      desc = 'Safety factor composite shell')
        self.add_input('composite_rho',             val = 0.0, units = 'kg/m**3',   desc = 'Density of composite of the shell')
        self.add_input('Xy',                        val = 0.0, units = 'Pa',        desc = 'Yield strength metal')
        self.add_input('metal_SF',                  val = 0.0,                      desc = 'Safety factor metal')
        self.add_input('metal_rho',                 val = 0.0, units = 'kg/m**3',   desc = 'Density metal')
        self.add_input('composite_cost',            val = 0.0, units = 'USD/kg',    desc = 'Unit cost composite of the shell')
        self.add_input('metal_cost',                val = 0.0, units = 'USD/kg',    desc = 'Unit cost metal')

        # Outputs
        self.add_output('diameter',                 val = 0.0, units = 'm',         desc = 'Outer diameter of the spinner')
        self.add_output('total_mass',               val = 0.0, units = 'kg',        desc = 'Total mass of the spinner, which includes composite shell and steel hardware')
        self.add_output('cost',                     val = 0.0, units = 'kg',        desc = 'Cost of the spinner')
        self.add_output('cm',                       val = 0.0, units = 'm',         desc = 'Distance between center of mass of the spinner and outer surface. Equal to the radius of the spinner')

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
        extr_gust_dsgn_pressure = 0.5 * 1.225 * (inputs['gust_ws'] ** 2.)* inputs['load_scaling'] 
        # Compute max allowable tensile strength of composite and max allowable yield strength metal given material properties and safety factors
        allow_tensile_strength  = inputs['composite_Xt'] / inputs['composite_SF']
        allow_yield_strength    = inputs['Xy'] / inputs['metal_SF']
        # Estimate thickness of the shell of the spinner
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
        
        # Compute outputs and assign them to openmdao outputs
        outputs['diameter']         = sph_spin_diam
        outputs['total_mass']       = spin_shell_mass + bracket_mass_total
        outputs['cm']               = sph_spin_diam / 2.
        outputs['cost']             = spin_shell_mass * inputs['composite_cost'] + bracket_mass_total * inputs['metal_cost']

        # Print to screen if verbosity option is on
        if self.options['verbosity']:
            sys.stderr.write('Spherical spinner: mass {:.1f} kg = Shell {:.1f} kg + Bracket structures {:.1f} kg\n'.format(float(outputs['total_mass']), float(spin_shell_mass), float(bracket_mass_total)))
            sys.stderr.write('Spherical spinner: spinner outer diameter {:.1f} m\n'.format(float(sph_spin_diam)))
            sys.stderr.write('Spherical spinner: cost ${:.2f}  center of mass {:.2f} m\n'.format(float(outputs['cost']), float(outputs['cm'])))

class Hub_System(om.Group):
    """
    # Openmdao group to model the hub system, which includes pitch system, spinner, and hub
    """
    # def initialize(self):
        
    def setup(self):
        # analysis_options = self.options['analysis_options']
        # opt_options     = self.options['opt_options']

        self.add_subsystem('pitch_system',   PitchSystem())
        self.add_subsystem('hub_shell',      HubShell())
        self.add_subsystem('spinner',        Spinner())

        self.connect('hub_shell.outer_diameter' , 'spinner.hub_diameter')

if __name__ == "__main__":

    hub_prob = om.Problem(model=Hub_System())
    hub_prob.setup()
    
    hub_prob['pitch_system.blade_mass']         = 17000.
    hub_prob['pitch_system.BRFM']               = 1.e+6
    hub_prob['pitch_system.n_blades']           = 3
    hub_prob['pitch_system.scaling_factor']     = 0.54
    hub_prob['pitch_system.rho']                = 7850.
    hub_prob['pitch_system.Xy']                 = 371.e+6

    hub_prob['hub_shell.blade_root_diameter']   = 4.
    hub_prob['hub_shell.n_blades']              = 3
    hub_prob['hub_shell.flange_t2shell_t']      = 4.
    hub_prob['hub_shell.flange_OD2hub_D']       = 0.5
    hub_prob['hub_shell.flange_ID2flange_OD']   = 0.8
    hub_prob['hub_shell.rho']                   = 7200.
    hub_prob['hub_shell.in2out_circ']           = 1.2 
    hub_prob['hub_shell.max_torque']            = 30.e+6
    hub_prob['hub_shell.Xy']                    = 200.e+6
    hub_prob['hub_shell.stress_concentration']  = 2.5
    hub_prob['hub_shell.reserve_factor']        = 2.0
    hub_prob['hub_shell.metal_cost']            = 3.00

    hub_prob['spinner.n_front_brackets']        = 3
    hub_prob['spinner.n_rear_brackets']         = 3
    hub_prob['spinner.n_blades']                = 3
    hub_prob['spinner.blade_root_diameter']     = 4.
    hub_prob['spinner.clearance_hub_spinner']   = 0.5
    hub_prob['spinner.spin_hole_incr']          = 1.2
    hub_prob['spinner.gust_ws']                 = 70
    hub_prob['spinner.load_scaling']            = 1.5
    hub_prob['spinner.composite_Xt']            = 60.e6
    hub_prob['spinner.composite_SF']            = 1.5
    hub_prob['spinner.composite_rho']           = 1600.
    hub_prob['spinner.Xy']                      = 225.e+6
    hub_prob['spinner.metal_SF']                = 1.5
    hub_prob['spinner.metal_rho']               = 7850.
    hub_prob['spinner.composite_cost']          = 7.00
    hub_prob['spinner.metal_cost']              = 3.00

    hub_prob.run_model()

    print('Pitch system mass: ' + str(hub_prob['pitch_system.mass'][0]) + ' kg')
    print('Hub shell mass: ' + str(hub_prob['hub_shell.total_mass'][0]) + ' kg')
    print('Hub shell outer diameter: ' + str(hub_prob['hub_shell.outer_diameter'][0]) + ' m')
    print('Hub shell cost: ' + str(hub_prob['hub_shell.cost'][0]) + ' USD')
    print('Distance btw flange and cm of hub shell: ' + str(hub_prob['hub_shell.cm'][0]) + ' m')
    print('Spinner mass: ' + str(hub_prob['spinner.total_mass'][0]) + ' kg')
    print('Spinner outer diameter: ' + str(hub_prob['spinner.diameter'][0]) + ' m')
    print('Spinner cost: ' + str(hub_prob['spinner.cost'][0]) + ' USD')
    print('Distance between center of mass of the spinner and outer surface: ' + str(hub_prob['spinner.cm'][0]) + ' m')
