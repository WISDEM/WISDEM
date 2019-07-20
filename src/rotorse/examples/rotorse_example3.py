from __future__ import print_function

class RotorSE_Example3():

	def execute(self):

		# --- Import Modules

		import numpy as np
		import os
		from openmdao.api import IndepVarComp, Component, Group, Problem, Brent, ScipyGMRES, ScipyOptimizer, DumpRecorder
		from rotorse.rotor_aeropower import RotorAeroPower
		from rotorse.rotor_geometry import RotorGeometry, NREL5MW, DTU10MW, TUM3_35MW, NINPUT
		from rotorse import RPM2RS, RS2RPM, TURBULENCE_CLASS, DRIVETRAIN_TYPE
		from rotorse.rotor import RotorSE

		myref = TUM3_35MW()

		rotor = Problem()
		npts_coarse_power_curve = 20 # (Int): number of points to evaluate aero analysis at
		npts_spline_power_curve = 200  # (Int): number of points to use in fitting spline to power curve

		rotor.root = RotorSE(myref, npts_coarse_power_curve, npts_spline_power_curve)
		rotor.setup()

		# ---
		# === blade grid ===
		rotor['hubFraction'] = myref.hubFraction #0.023785  # (Float): hub location as fraction of radius
		rotor['bladeLength'] = myref.bladeLength #96.7  # (Float, m): blade length (if not precurved or swept) otherwise length of blade before curvature
		rotor['precone'] = myref.precone #4.  # (Float, deg): precone angle
		rotor['tilt'] = myref.tilt #6.0  # (Float, deg): shaft tilt
		rotor['yaw'] = 0.0  # (Float, deg): yaw error
		rotor['nBlades'] = myref.nBlades #3  # (Int): number of blades
		# ---

		# === blade geometry ===
		rotor['r_max_chord'] =  myref.r_max_chord  # 0.2366 #(Float): location of max chord on unit radius
		rotor['chord_in'] = myref.chord # np.array([4.6, 4.869795, 5.990629, 3.00785428, 0.0962])  # (Array, m): chord at control points. defined at hub, then at linearly spaced locations from r_max_chord to tip
		rotor['theta_in'] = myref.theta # np.array([14.5, 12.874, 6.724, -0.03388039, -0.037]) # (Array, deg): twist at control points.  defined at linearly spaced locations from r[idx_cylinder] to tip
		rotor['precurve_in'] = myref.precurve #np.array([-0., -0.054497, -0.175303, -0.84976143, -6.206217])  # (Array, m): precurve at control points.  defined at same locations at chord, starting at 2nd control point (root must be zero precurve)
		rotor['presweep_in'] = myref.presweep #np.array([0., 0., 0., 0., 0.])  # (Array, m): precurve at control points.  defined at same locations at chord, starting at 2nd control point (root must be zero precurve)
		rotor['sparT_in'] = myref.spar_thickness # np.array([0.03200042 0.07038508 0.08515644 0.07777004 0.01181032])  # (Array, m): spar cap thickness parameters
		rotor['teT_in'] = myref.te_thickness # np.array([0.04200055 0.08807739 0.05437378 0.01610219 0.00345225])  # (Array, m): trailing-edge thickness parameters
		# ---

		# === atmosphere ===
		rotor['rho'] = 1.225  # (Float, kg/m**3): density of air
		rotor['mu'] = 1.81206e-5  # (Float, kg/m/s): dynamic viscosity of air
		rotor['wind.shearExp'] = 0.25  # (Float): shear exponent
		rotor['hub_height'] = myref.hub_height #119.0  # (Float, m): hub height
		rotor['turbine_class'] = myref.turbine_class #TURBINE_CLASS['I']  # (Enum): IEC turbine class
		rotor['turbulence_class'] = TURBULENCE_CLASS['B']  # (Enum): IEC turbulence class class
		rotor['wind.zref'] = myref.hub_height #119.0  # (Float): reference hub height for IEC wind speed (used in CDF calculation)
		rotor['gust_stddev'] = 3
		# ---

		# === control ===
		rotor['control_Vin'] = myref.control_Vin #4.0  # (Float, m/s): cut-in wind speed
		rotor['control_Vout'] = myref.control_Vout #25.0  # (Float, m/s): cut-out wind speed
		rotor['control_minOmega'] = myref.control_minOmega #6.0  # (Float, rpm): minimum allowed rotor rotation speed
		rotor['control_maxOmega'] = myref.control_maxOmega #8.88766  # (Float, rpm): maximum allowed rotor rotation speed
		rotor['control_tsr'] = myref.control_tsr #10.58  # (Float): tip-speed ratio in Region 2 (should be optimized externally)
		rotor['control_pitch'] = myref.control_pitch #0.0  # (Float, deg): pitch angle in region 2 (and region 3 for fixed pitch machines)
		rotor['control_maxTS'] = myref.control_maxTS
		rotor['machine_rating'] = myref.rating #10e6  # (Float, W): rated power
		rotor['pitch_extreme'] = 0.0  # (Float, deg): worst-case pitch at survival wind condition
		rotor['azimuth_extreme'] = 0.0  # (Float, deg): worst-case azimuth at survival wind condition
		rotor['VfactorPC'] = 0.7  # (Float): fraction of rated speed at which the deflection is assumed to representative throughout the power curve calculation
		# ---

		# === aero and structural analysis options ===
		rotor['nSector'] = 4  # (Int): number of sectors to divide rotor face into in computing thrust and power
		rotor['AEP_loss_factor'] = 1.0  # (Float): availability and other losses (soiling, array, etc.)
		rotor['drivetrainType'] = myref.drivetrain #DRIVETRAIN_TYPE['GEARED']  # (Enum)
		rotor['dynamic_amplication_tip_deflection'] = 1.35  # (Float): a dynamic amplification factor to adjust the static deflection calculation
		# ---


		# === fatigue ===
		r_aero = np.array([0.02222276, 0.06666667, 0.11111057, 0.2, 0.23333333, 0.3, 0.36666667, 0.43333333,
		               0.5, 0.56666667, 0.63333333, 0.64, 0.7, 0.83333333, 0.88888943, 0.93333333,
		               0.97777724])  # (Array): new aerodynamic grid on unit radius
		rstar_damage = np.array([0.000, 0.022, 0.067, 0.111, 0.167, 0.233, 0.300, 0.367, 0.433, 0.500,
		    0.567, 0.633, 0.700, 0.767, 0.833, 0.889, 0.933, 0.978])  # (Array): nondimensional radial locations of damage equivalent moments
		Mxb_damage = 1e3*np.array([2.3743E+003, 2.0834E+003, 1.8108E+003, 1.5705E+003, 1.3104E+003,
		    1.0488E+003, 8.2367E+002, 6.3407E+002, 4.7727E+002, 3.4804E+002, 2.4458E+002, 1.6339E+002,
		    1.0252E+002, 5.7842E+001, 2.7349E+001, 1.1262E+001, 3.8549E+000, 4.4738E-001])  # (Array, N*m): damage equivalent moments about blade c.s. x-direction
		Myb_damage = 1e3*np.array([2.7732E+003, 2.8155E+003, 2.6004E+003, 2.3933E+003, 2.1371E+003,
		    1.8459E+003, 1.5582E+003, 1.2896E+003, 1.0427E+003, 8.2015E+002, 6.2449E+002, 4.5229E+002,
		    3.0658E+002, 1.8746E+002, 9.6475E+001, 4.2677E+001, 1.5409E+001, 1.8426E+000])  # (Array, N*m): damage equivalent moments about blade c.s. y-direction
		xp = np.r_[0.0, r_aero]
		xx = np.r_[0.0, myref.r]
		rotor['rstar_damage'] = np.interp(xx, xp, rstar_damage)
		rotor['Mxb_damage'] = np.interp(xx, xp, Mxb_damage)
		rotor['Myb_damage'] = np.interp(xx, xp, Myb_damage)
		rotor['strain_ult_spar'] = 1.0e-2  # (Float): ultimate strain in spar cap
		rotor['strain_ult_te'] = 2500*1e-6 * 2   # (Float): uptimate strain in trailing-edge panels, note that I am putting a factor of two for the damage part only.
		rotor['gamma_fatigue'] = 1.755 # (Float): safety factor for fatigue
		rotor['gamma_f'] = 1.35 # (Float): safety factor for loads/stresses
		rotor['gamma_m'] = 1.1 # (Float): safety factor for materials
		rotor['gamma_freq'] = 1.1 # (Float): safety factor for resonant frequencies
		rotor['m_damage'] = 10.0  # (Float): slope of S-N curve for fatigue analysis
		rotor['struc.lifetime'] = 20.0  # (Float): number of cycles used in fatigue analysis  TODO: make function of rotation speed
		# ----------------

		# === run and outputs ===
		rotor.run()

		print('AEP =', rotor['AEP'])
		print('diameter =', rotor['diameter'])
		print('rated_V =', rotor['rated_V'])
		print('rated_Omega =', rotor['rated_Omega'])
		print('rated_pitch =', rotor['rated_pitch'])
		print('rated_T =', rotor['rated_T'])
		print('rated_Q =', rotor['rated_Q'])
		print('mass_one_blade =', rotor['mass_one_blade'])
		print('mass_all_blades =', rotor['mass_all_blades'])
		print('I_all_blades =', rotor['I_all_blades'])
		print('freq =', rotor['freq'])
		print('tip_deflection =', rotor['tip_deflection'])
		print('root_bending_moment =', rotor['root_bending_moment'])

		outpath = '..\..\..\docs\images'
		import matplotlib.pyplot as plt
		plt.figure()
		plt.plot(rotor['V'], rotor['P']/1e6)
		plt.xlabel('wind speed (m/s)')
		plt.xlabel('power (W)')

		plt.figure()

		plt.plot(rotor['r_pts'], rotor['strainU_spar'], label='suction')
		plt.plot(rotor['r_pts'], rotor['strainL_spar'], label='pressure')
		plt.plot(rotor['r_pts'], rotor['eps_crit_spar'], label='critical')
		# plt.ylim([-6e-3, 6e-3])
		plt.xlabel('r')
		plt.ylabel('strain')
		plt.legend()
		plt.savefig(os.path.abspath(os.path.join(outpath,'strain_spar_dtu10mw.png')))
		plt.savefig(os.path.abspath(os.path.join(outpath,'strain_spar_dtu10mw.pdf')))

		plt.figure()

		plt.plot(rotor['r_pts'], rotor['strainU_te'], label='suction')
		plt.plot(rotor['r_pts'], rotor['strainL_te'], label='pressure')
		plt.plot(rotor['r_pts'], rotor['eps_crit_te'], label='critical')
		# plt.ylim([-5e-3, 5e-3])
		plt.xlabel('r')
		plt.ylabel('strain')
		plt.legend()
		plt.savefig(os.path.abspath(os.path.join(outpath,'strain_te_dtu10mw.png')))
		plt.savefig(os.path.abspath(os.path.join(outpath,'strain_te_dtu10mw.pdf')))

		plt.show()
		# ----------------


if __name__ == "__main__":

	rotor = RotorSE_Example3()
	rotor.execute()