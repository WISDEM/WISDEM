import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.integrate import odeint

from openmdao.api import ExplicitComponent, Group
from wisdem.commonse.csystem import DirectionVector
from wisdem.ccblade.ccblade import CCBlade, CCAirfoil
import wisdem.ccblade._bem as _bem

class ODEsolveExtremeLoads(ExplicitComponent):
    # OpenMDAO component that runs a reduced order ODE simulation to get the time domain extreme loads
    def initialize(self):
        self.options.declare('modeling_options')

    def setup(self):
        self.modeling_options = self.options['modeling_options']
        self.n_span     = n_span       = self.modeling_options['blade']['n_span']
        self.n_aoa      = n_aoa        = self.modeling_options['airfoils']['n_aoa']# Number of angle of attacks
        self.n_Re       = n_Re         = self.modeling_options['airfoils']['n_Re'] # Number of Reynolds, so far hard set at 1
        self.n_tab      = n_tab        = self.modeling_options['airfoils']['n_tab']# Number of tabulated data. For distributed aerodynamic control this could be > 1
        self.n_pc       = n_pc         = self.modeling_options['servose']['n_pc']
        self.n_pitch    = n_pitch      = self.modeling_options['servose']['n_pitch_perf_surfaces']

        self.verbosity  = self.options['modeling_options']['general']['verbosity']
        
        # Inputs strains
        self.add_input('r',                val=np.zeros(n_span),  units='m',        desc='radial locations where blade is defined (should be increasing and not go all the way to hub or tip)')
        self.add_input('theta',            val=np.zeros(n_span),  units='deg',      desc='structural twist')
        self.add_input('chord',            val=np.zeros(n_span),  units='m',        desc='chord length at each section')
        self.add_input('Rtip',             val=0.0,               units='m',        desc='tip radius')
        self.add_input('Rhub',             val=0.0,               units='m',        desc='hub radius')
        self.add_input('hub_height',       val=0.0,               units='m',        desc='hub height')            
        self.add_input('precone',          val=0.0,               units='deg',      desc='precone angle', )
        self.add_input('tilt',             val=0.0,               units='deg',      desc='shaft tilt', )
        self.add_input('yaw',              val=0.0,               units='deg',      desc='yaw error', )
        self.add_input('precurve',         val=np.zeros(n_span),  units='m',        desc='precurve at each section')
        self.add_input('precurveTip',      val=0.0,               units='m',        desc='precurve at tip')
        self.add_input('presweep',         val=np.zeros(n_span),  units='m',        desc='presweep at each section')
        self.add_input('presweepTip',      val=0.0,               units='m',        desc='presweep at tip')

        self.add_input('airfoils_aoa',     val=np.zeros((n_aoa)), units='deg',         desc='angle of attack grid for polars')
        self.add_input('airfoils_Re',      val=np.zeros((n_Re)),                       desc='Reynolds numbers of polars')
        self.add_input('airfoils_cl',      val=np.zeros((n_span, n_aoa, n_Re, n_tab)), desc='lift coefficients, spanwise')
        self.add_input('airfoils_cd',      val=np.zeros((n_span, n_aoa, n_Re, n_tab)), desc='drag coefficients, spanwise')
        self.add_input('airfoils_cm',      val=np.zeros((n_span, n_aoa, n_Re, n_tab)), desc='moment coefficients, spanwise')

        self.add_input('V',                val=np.zeros(n_pc),    units='m/s',      desc='wind vector')
        self.add_input('Omega',            val=np.zeros(n_pc),    units='rpm',      desc='rotor rotational speed')
        self.add_input('pitch',            val=np.zeros(n_pc),    units='deg',      desc='rotor pitch schedule')
        self.add_input('rated_V',          val=0.0,               units='m/s',      desc='rated wind speed')
        self.add_input('rated_Omega',      val=0.0,               units='rpm',      desc='rotor rotation speed at rated')
        self.add_input('rated_pitch',      val=0.0,               units='deg',      desc='pitch setting at rated')
        self.add_input('rated_Q',          val=0.0,               units='N*m',      desc='rotor aerodynamic torque at rated')
        self.add_input('max_pitch_rate',   val=0.0,               units='rad/s',    desc='Maximum allowed blade pitch rate')
        self.add_input('max_pitch',        val=0.0,               units='rad',      desc='')
        self.add_input('min_pitch',        val=0.0,               units='rad',      desc='')

        self.add_input('gearbox_efficiency',   val=0.0,                             desc='Gearbox efficiency')
        self.add_input('generator_efficiency', val=0.0,                             desc='Generator efficiency')
        self.add_input('gear_ratio',       val=0.0,                                 desc='Gearbox Ratio')

        self.add_input('PC_GS_angles',     val=np.zeros(n_pitch+1), units='rad',    desc='Gain-schedule table: pitch angles')
        self.add_input('PC_GS_KP',         val=np.zeros(n_pitch+1),                 desc='Gain-schedule table: pitch controller kp gains')
        self.add_input('PC_GS_KI',         val=np.zeros(n_pitch+1),                 desc='Gain-schedule table: pitch controller ki gains')
        self.add_input('VS_Rgn2K',         val=0.0,               units='N*m/(rad/s)**2', desc='Generator torque constant in Region 2 (HSS side), [N-m/(rad/s)^2]')

        self.add_input('I_all_blades',     shape=6,               units='kg*m**2',  desc='mass moments of inertia of all blades in yaw c.s. order:Ixx, Iyy, Izz, Ixy, Ixz, Iyz')

        self.add_discrete_input('nBlades', val=0,                                   desc='number of blades')
        self.add_input('rho',              val=1.225,             units='kg/m**3',  desc='density of air')
        self.add_input('mu',               val=1.81e-5,           units='kg/(m*s)', desc='dynamic viscosity of air')
        self.add_input('shearExp',         val=0.0,                                 desc='shear exponent')
        self.add_discrete_input('nSector', val=4,                                   desc='number of sectors to divide rotor face into in computing thrust and power')
        self.add_discrete_input('tiploss', val=True,                                desc='include Prandtl tip loss model')
        self.add_discrete_input('hubloss', val=True,                                desc='include Prandtl hub loss model')
        self.add_discrete_input('wakerotation', val=True,                           desc='include effect of wake rotation (i.e., tangential induction factor is nonzero)')
        self.add_discrete_input('usecd',   val=True,                                desc='use drag coefficient in computing induction factors')

        self.add_output('loads_r',         val=np.zeros(n_span),  units='m')
        self.add_output('loads_Px',        val=np.zeros(n_span),  units='N/m')
        self.add_output('loads_Py',        val=np.zeros(n_span),  units='N/m')
        self.add_output('loads_Pz',        val=np.zeros(n_span),  units='N/m')

        self.add_output('Fxyz_hub_aero',    val=np.zeros(3),      units='N')
        self.add_output('Mxyz_hub_aero',    val=np.zeros(3),      units='N*m')


    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        
        ### Inputs
        PitchControl = True
        ## Rotor Properties
        r              = inputs['r']
        chord          = inputs['chord']
        Rhub           = inputs['Rhub']
        Rtip           = inputs['Rtip']
        theta          = inputs['theta']
        precurve       = inputs['precurve']
        precurveTip    = inputs['precurveTip']
        presweep       = inputs['presweep']
        presweepTip    = inputs['presweepTip']
        precone        = inputs['precone']
        tilt           = inputs['tilt']
        hub_height     = inputs['hub_height']
        J              = inputs['I_all_blades'][0] # todo, using the blade inertia b/c that's what ServoSE is using for the tuning, should include the hub and drivetrain as well in both places
        nBlades        = discrete_inputs['nBlades']

        ## Airfoil properties
        aoa            = inputs['airfoils_aoa']
        af             = [CCAirfoil(aoa, inputs['airfoils_Re'], inputs['airfoils_cl'][i,:,:,0], inputs['airfoils_cd'][i,:,:,0], cm=inputs['airfoils_cm'][i,:,:,0]) for i in range(len(r))]

        ## Controller Properties
        V_pc           = inputs['V']
        Omega_pc       = inputs['Omega']
        pitch_pc       = inputs['pitch']
        rated_V        = inputs['rated_V']
        rated_Omega    = inputs['rated_Omega']
        rated_pitch    = inputs['rated_pitch']
        rated_Q        = inputs['rated_Q']
        max_pitch_rate = np.degrees(inputs['max_pitch_rate'])
        max_pitch      = inputs['max_pitch']
        min_pitch      = inputs['min_pitch']

        PC_GS_angles   = inputs['PC_GS_angles']
        PC_GS_KP       = inputs['PC_GS_KP']
        PC_GS_KI       = inputs['PC_GS_KI']
        VS_Rgn2K       = inputs['VS_Rgn2K']

        # Drivetrain Properties
        eff_gen        = inputs['gearbox_efficiency']
        eff_gb         = inputs['generator_efficiency']
        N_gear         = inputs['gear_ratio']

        ## Aerodynamic Settings
        rho            = inputs['rho']
        mu             = inputs['mu']
        shearExp       = 0. #inputs['shearExp']
        nSector        = 4
        tiploss        = True
        hubloss        = True
        wakerotation   = True
        usecd          = True

        ## Saturate Controls
        def saturate(val, valmin, valmax):
            return min(max(val, valmin), valmax)

        ## Rotor ODE model with Region 2 and 3 torque and pitch control
        t_ode     = []
        pitch_ode = []
        Iterm_ode = []
        def rotor(y, t, t_all, u, yaw):

            # Inputs
            ui     = [np.interp(t, t_all, u)]
            yawi   = np.interp(t, t_all, yaw)
            omega  = y*60./(2.*np.pi)
            bem    = CCBlade(r, chord, theta, af, Rhub, Rtip, B=nBlades, rho=rho, mu=mu, precone=precone, tilt=tilt, yaw=yawi, shearExp=shearExp, hubHt=hub_height, nSector=nSector, precurve=precurve, precurveTip=precurveTip, presweep=presweep, presweepTip=presweepTip, tiploss=tiploss, hubloss=hubloss, wakerotation=wakerotation, usecd=usecd)

            # Blade pitch
            pitch  = np.interp([u[0]], V_pc, pitch_pc)
            Iterm  = np.radians(pitch)
            Pterm  = 0.

            if t>0. and PitchControl:
                if omega > rated_Omega or pitch_ode[-1]>min_pitch:
                    t_old = t_ode[-1]
                    dt = t - t_old
                    if dt <= 0.:
                        t_old = max([ti for ti in sorted(list(set(t_ode))) if ti < t])
                        dt = t - t_old

                    pitch_steady = np.interp(ui, V_pc, pitch_pc)
                    PC_GS_KP_i   = np.interp(pitch_steady, PC_GS_angles, PC_GS_KP)
                    PC_GS_KI_i   = np.interp(pitch_steady, PC_GS_angles, PC_GS_KI)

                    y_err        = (rated_Omega - omega) * (2.*np.pi) / 60. * N_gear

                    if t_old == t_ode[-1]:
                        Iterm_old = Iterm_ode[-1]
                        pitch_old = pitch_ode[-1]
                    else:
                        Iterm_old = np.interp(t_old, t_ode, Iterm_ode)
                        pitch_old = np.interp(t_old, t_ode, pitch_ode)

                    Pterm = PC_GS_KP_i * y_err
                    Iterm = Iterm_old + dt * PC_GS_KI_i * y_err
                    Iterm = saturate(Iterm, min_pitch, max_pitch)
                    pitch = np.degrees(Pterm + Iterm)

                    dpitch = (pitch - pitch_old) / dt
                    if dpitch < -1.*max_pitch_rate:
                        pitch = pitch_old - (dt*max_pitch_rate)
                    elif dpitch > max_pitch_rate:
                        pitch = pitch_old + (dt*max_pitch_rate)
                    pitch = saturate(pitch, np.degrees(min_pitch), np.degrees(max_pitch))
                    Iterm = np.radians(pitch) - Pterm
            

            # Aero Torque
            bem_outputs, derivs = bem.evaluate(ui, omega, pitch)
            Q_aero = bem_outputs['Q'] * eff_gb * eff_gen

            # Controller Torque
            Q_c    = VS_Rgn2K*y**2 * N_gear**3 / (eff_gb * eff_gen)
            # Cp_max = 0.48763972038134035
            # tsr    = 11.18918918918919
            # Q_c = 0.5*rho*np.pi*(Rtip**5)*Cp_max/(tsr**3)*y**2 / (eff_gb * eff_gen)
            if Q_c > rated_Q and PitchControl:
                Q_c = rated_Q

            # EOM
            dydt   = (Q_aero-Q_c)/J

            store_history = True
            if t > 0:
                if t <= t_ode[-1]:
                    store_history = False

            if store_history:
                t_ode.append(float(t))
                pitch_ode.append(float(pitch[-1]))
                Iterm_ode.append(float(Iterm))

            return dydt


        ## Initialize simulation
        t_min = 0.  # time 0
        t_max = 15. # final time
        fs = 10.    # sample frequency

        n = int((t_max-t_min)*fs + 1)       # number of time steps
        t = np.linspace(t_min,t_max,num=n)  # time step vector

        ## Extreme coherent gust with direction change (ECD) wind and yaw
        T_start = 2.  #s         # time to start gust
        T       = 10  #s         # length of transient event
        V_cg    = 15  #m/s       # coherent gust magnitude

        V_hub   = rated_V  # standard calls for rated and rated +/- 2 m/s. testing this showed higher loads, more consistent with OpenFAST when using slightly below rated wind speeds, resultsing in slightly below rated rotor speeds when the gust occurs
        if V_hub < 4:
            Theta_cg = 180
        else:
            Theta_cg = 720/V_hub
        # calc time series
        u   = np.zeros_like(t)
        yaw = np.zeros_like(t)
        for i, ti in enumerate(t):
            if ti >= T_start:
                if ti<T+T_start:
                    u[i]   = V_hub + 0.5*V_cg*(1-np.cos(np.pi*(ti-T_start)/T))
                    yaw[i] = 0.5*Theta_cg*(1-np.cos(np.pi*(ti-T_start)/T))
                else:
                    u[i]   = V_hub+V_cg
                    yaw[i] = Theta_cg
            else:
                u[i]   = V_hub
                yaw[i] = 0.
        # u = np.ones_like(t)*7.
        # yaw = np.zeros_like(t)

        ## Solve ODE
        if self.verbosity:
            print('Solving ODE')
            t1 = time.time()

        y0        = np.interp(u[0], V_pc, Omega_pc)*2.*np.pi/60. # initial rotor speed
        sol       = odeint(rotor, y0, t, args=(t, u, yaw))       # call solver

        # format outputs        
        omega_out = sol*60/(2.*np.pi)                            # rotor speed from ODE simulation
        if PitchControl: # blade pitch from ODE simulation
            pitch_out = np.interp(t, t_ode, pitch_ode)
        else:
            pitch_out = np.zeros_like(t)

        # back calculate loads from ODE wind speed, yaw, pitch, and rotor speed, determine time of max loads
        outputs_all = []
        for ui, omegai, pitchi, yawi in zip(u, omega_out, pitch_out, yaw):
            bem    = CCBlade(r, chord, theta, af, Rhub, Rtip, B=nBlades, rho=rho, mu=mu, precone=precone, tilt=tilt, yaw=yawi, shearExp=inputs['shearExp'], hubHt=hub_height, nSector=nSector, precurve=precurve, precurveTip=precurveTip, presweep=presweep, presweepTip=presweepTip, tiploss=tiploss, hubloss=hubloss, wakerotation=wakerotation, usecd=usecd)
            bem_outputs, derivs = bem.evaluate([ui], [omegai], [pitchi])
            outputs_all.append(bem_outputs)

        Q = [out['Q'] for out in outputs_all]
        T = [out['T'] for out in outputs_all]

        idx_max   = np.argmax(T)
        u_max     = u[idx_max]
        pitch_max = pitch_out[idx_max]
        omega_max = omega_out[idx_max]
        yaw_max   = yaw[idx_max]
        azimuth   = 0.

        # calc distributed loads
        dx_dx = np.eye(3*self.n_span)
        x_az, x_azd, y_az, y_azd, z_az, z_azd, cone, coned, s, sd = _bem.definecurvature_dv2(r, dx_dx[:, :self.n_span], precurve, dx_dx[:, self.n_span:2*self.n_span], presweep, dx_dx[:, 2*self.n_span:], 0.0, np.zeros(3*self.n_span))
        totalCone = precone + np.degrees(cone)

        Fxyz_blade_aero = np.zeros((nBlades, 6))
        Mxyz_blade_aero = np.zeros((nBlades, 6))
        azimuth_blades  = np.linspace(0, 360, nBlades+1)
        for i_blade in range(nBlades):
            bem = CCBlade(r, chord, theta, af, Rhub, Rtip, B=nBlades, rho=rho, mu=mu, precone=precone, tilt=tilt, yaw=yaw_max, shearExp=inputs['shearExp'], hubHt=hub_height, nSector=nSector, precurve=precurve, precurveTip=precurveTip, presweep=presweep, presweepTip=presweepTip, tiploss=tiploss, hubloss=hubloss, wakerotation=wakerotation, usecd=usecd)
            loads, derivs = bem.distributedAeroLoads(u_max, omega_max, pitch_max, azimuth)
            
            Np = loads['Np']
            Tp = loads['Tp']
            # conform to blade-aligned coordinate system
            Px = Np
            Py = -Tp
            Pz = 0.*Np
            # Integrate to get shear forces
            Fx = np.trapz(Px, r)
            Fy = np.trapz(Py, r)
            Fz = np.trapz(Pz, r)
            Fxy= np.sqrt(Fx**2 + Fy**2)
            Fyz= np.sqrt(Fy**2 + Fz**2)
            Fxz= np.sqrt(Fx**2 + Fz**2)
            # loads in azimuthal c.s.
            P = DirectionVector(Px, Py, Pz).bladeToAzimuth(totalCone)
            # distributed bending load in azimuth coordinate ysstem
            az = DirectionVector(x_az, y_az, z_az)
            Mp = az.cross(P)
            # Integrate to obtain moments
            Mx = np.trapz(Mp.x, r)
            My = np.trapz(Mp.y, r)
            Mz = np.trapz(Mp.z, r)
            Mxy= np.sqrt(Mx**2 + My**2)
            Myz= np.sqrt(My**2 + Mz**2)
            Mxz= np.sqrt(Mx**2 + Mz**2)
            Fxyz_blade_aero[i_blade, :] = np.array([Fx, Fy, Fz, Fxy, Fyz, Fxz])
            Mxyz_blade_aero[i_blade, :] = np.array([Mx, My, Mz, Mxy, Myz, Mxz])


        # calc hub loads
        F_hub_tot = np.zeros((3,))
        M_hub_tot = np.zeros((3,))
        for i_blade in range(nBlades): # Convert from blade to hub c.s.
            myF = DirectionVector.fromArray(Fxyz_blade_aero[i_blade,:]).azimuthToHub(azimuth_blades[i_blade])
            myM = DirectionVector.fromArray(Mxyz_blade_aero[i_blade,:]).azimuthToHub(azimuth_blades[i_blade])
            F_hub_tot += myF.toArray()
            M_hub_tot += myM.toArray()

        outputs['Fxyz_hub_aero'] = F_hub_tot
        outputs['Mxyz_hub_aero'] = M_hub_tot
        outputs['loads_r']       = r
        outputs['loads_Px']      = Np
        outputs['loads_Py']      = -Tp

        if self.verbosity:
            print('ODE Solve complete, run time:', time.time()-t1)

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1,2, figsize=(10., 6.))
        # plt.subplots_adjust(wspace=0.3, left=0.1)
        # ax[0].plot(t, u, label='Wind Speed, m/s')
        # ax[0].plot(t, omega_out, label='Rotor Speed, rpm')
        # ax[0].plot([t[0], t[-1]], [rated_Omega, rated_Omega], ':', label='Rated Rotor Speed')
        # axa = ax[0].twinx()
        # color = 'tab:red'
        # axa.plot(t_ode, pitch_ode, label='Blade Pitch, deg', color=color)
        # axa.tick_params(axis='y', labelcolor=color)
        # axa.set_ylabel('Blade Pitch')
        # ax[0].set_xlabel('Time')
        # ax[0].set_title('Rotor Response')
        # handles1, labels1 = ax[0].get_legend_handles_labels()
        # handles2, labels2 = axa.get_legend_handles_labels()
        # ax[0].legend(handles1+handles2, labels1+labels2)

        # color = 'tab:red'
        # ax[1].plot(t, Q, label='Torque, Nm', color=color)
        # ax[1].tick_params(axis='y', labelcolor=color)
        # color = 'tab:blue'
        # axb = ax[1].twinx()
        # axb.plot(t, T, label='Thrust, N', color=color)
        # axb.tick_params(axis='y', labelcolor=color)
        # ax[1].set_xlabel('Time')
        # ax[1].set_title('Loads')
        # ax[1].set_ylabel('Torque')
        # axb.set_ylabel('Thrust')

        # handles1, labels1 = ax[1].get_legend_handles_labels()
        # handles2, labels2 = axb.get_legend_handles_labels()
        # ax[1].legend(handles1+handles2, labels1+labels2)

        # plt.show()
