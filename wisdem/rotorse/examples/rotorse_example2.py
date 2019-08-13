
class RotorSE_Example2():

	def execute(self):

		# --- Import Modules
		import numpy as np
		import os
		from openmdao.api import IndepVarComp, Component, Group, Problem, Brent, ScipyGMRES, ScipyOptimizer, DumpRecorder
		from rotorse.rotor_aeropower import RotorAeroPower
		from rotorse.rotor_geometry import RotorGeometry, NREL5MW, DTU10MW, NINPUT
		from rotorse import RPM2RS, RS2RPM, TURBULENCE_CLASS, DRIVETRAIN_TYPE

		rotor = Problem()
		myref = DTU10MW()

		npts_coarse_power_curve = 20 # (Int): number of points to evaluate aero analysis at
		npts_spline_power_curve = 200  # (Int): number of points to use in fitting spline to power curve

		rotor.root = RotorAeroPower(myref, npts_coarse_power_curve, npts_spline_power_curve)
		# ---

		# --- Optimizer
		rotor.driver = ScipyOptimizer()
		rotor.driver.options['optimizer'] = 'SLSQP'
		rotor.driver.options['tol'] = 1.0e-8
		# ---
		# --- Objective
		# AEP0 = 47147596.2911617
		AEP0 = 48113504.25433461
		rotor.driver.add_objective('AEP', scaler=-1./AEP0)
		# --- Design Variables
		rotor.driver.add_desvar('r_max_chord', lower=0.15, upper=0.4) # 0.2366
		rotor.driver.add_desvar('chord_in', low=0.4, high=7.)
		rotor.driver.add_desvar('theta_in', low=-10.0, high=30.0)
		rotor.driver.add_desvar('control_tsr', low=3.0, high=14.0)

		T0 = 1448767.9705024462
		rotor.driver.add_constraint('rated_T',upper=T0*1.02)
		# ---
		# --- Recorder
		rec = DumpRecorder()
		rotor.driver.add_recorder(rec)
		rec.options['record_metadata'] = False
		rec.options['record_unknowns'] = False
		rec.options['record_params'] = False
		rec.options['includes'] = ['r_max_chord', 'chord_in', 'theta_in', 'control_tsr']
		# ---
		# --- Setup
		rotor.setup()

		# === blade grid ===
		rotor['hubFraction'] = myref.hubFraction #0.025  # (Float): hub location as fraction of radius
		rotor['bladeLength'] = myref.bladeLength #61.5  # (Float, m): blade length (if not precurved or swept) otherwise length of blade before curvature
		rotor['precone'] = myref.precone #2.5  # (Float, deg): precone angle
		rotor['tilt'] = myref.tilt #5.0  # (Float, deg): shaft tilt
		# ...
		# ---
		rotor['yaw'] = 0.0  # (Float, deg): yaw error
		rotor['nBlades'] = myref.nBlades #3  # (Int): number of blades

		# === blade geometry ===
		rotor['r_max_chord'] = myref.r_max_chord #0.23577  # (Float): location of max chord on unit radius
		rotor['chord_in'] = myref.chord #np.array([3.2612, 4.5709, 3.3178, 1.4621])  # (Array, m): chord at control points. defined at hub, then at linearly spaced locations from r_max_chord to tip
		rotor['theta_in'] = myref.theta #np.array([13.2783, 7.46036, 2.89317, -0.0878099])  # (Array, deg): twist at control points.  defined at linearly spaced locations from r[idx_cylinder] to tip
		rotor['precurve_in'] = myref.precurve #np.array([0.0, 0.0, 0.0])  # (Array, m): precurve at control points.  defined at same locations at chord, starting at 2nd control point (root must be zero precurve)
		rotor['presweep_in'] = myref.presweep #np.array([0.0, 0.0, 0.0])  # (Array, m): precurve at control points.  defined at same locations at chord, starting at 2nd control point (root must be zero precurve)
		rotor['sparT_in'] = myref.spar_thickness #np.array([0.05, 0.047754, 0.045376, 0.031085, 0.0061398])  # (Array, m): spar cap thickness parameters
		rotor['teT_in'] = myref.te_thickness #np.array([0.1, 0.09569, 0.06569, 0.02569, 0.00569])  # (Array, m): trailing-edge thickness parameters

		# === atmosphere ===
		rotor['analysis.rho'] = 1.225  # (Float, kg/m**3): density of air
		rotor['analysis.mu'] = 1.81206e-5  # (Float, kg/m/s): dynamic viscosity of air
		rotor['hub_height'] = myref.hub_height #90.0
		rotor['analysis.shearExp'] = 0.25  # (Float): shear exponent
		rotor['turbine_class'] = myref.turbine_class #TURBINE_CLASS['I']  # (Enum): IEC turbine class
		rotor['cdf_reference_height_wind_speed'] = myref.hub_height #90.0  # (Float): reference hub height for IEC wind speed (used in CDF calculation)

		# === control ===
		rotor['control_Vin'] = myref.control_Vin #3.0  # (Float, m/s): cut-in wind speed
		rotor['control_Vout'] = myref.control_Vout #25.0  # (Float, m/s): cut-out wind speed
		rotor['control_ratedPower'] = myref.rating #5e6  # (Float, W): rated power
		rotor['control_minOmega'] = myref.control_minOmega #0.0  # (Float, rpm): minimum allowed rotor rotation speed
		rotor['control_maxOmega'] = myref.control_maxOmega #12.0  # (Float, rpm): maximum allowed rotor rotation speed
		rotor['control_tsr'] = myref.control_tsr #7.55  # (Float): tip-speed ratio in Region 2 (should be optimized externally)
		rotor['control_pitch'] = myref.control_pitch #0.0  # (Float, deg): pitch angle in region 2 (and region 3 for fixed pitch machines)

		# === aero and structural analysis options ===
		rotor['nSector'] = 4  # (Int): number of sectors to divide rotor face into in computing thrust and power
		rotor['AEP_loss_factor'] = 1.0  # (Float): availability and other losses (soiling, array, etc.)
		rotor['drivetrainType'] = myref.drivetrain #DRIVETRAIN_TYPE['GEARED']  # (Enum)
		# ---

		# --- run and outputs
		rotor.run()
		# ---

		print 'Max Chord Radius = ', rotor['r_max_chord']
		print 'Chord Control Points = ', rotor['chord_in']
		print 'Twist Control Points = ', rotor['theta_in']
		print 'TSR = ', rotor['control_tsr']

		print'----------------'
		print 'Objective = ', -1*rotor['AEP']/AEP0
		print 'AEP = ', rotor['AEP']
		print 'Rated Thrust =', rotor['rated_T']
		print 'Percent change in thrust =', (rotor['rated_T']-T0)/T0 *100





		# import matplotlib.pyplot as plt
		# # plt.plot(rotor['V'], rotor['P']/1e6)
		# # plt.xlabel('Wind Speed (m/s)')
		# # plt.ylabel('Power (MW)')
		# # plt.show()
		# # # ---

		# # -----------------------------------------------------------------------
		# rotor2 = Problem()
		# myref = DTU10MW()

		# npts_coarse_power_curve = 20 # (Int): number of points to evaluate aero analysis at
		# npts_spline_power_curve = 200  # (Int): number of points to use in fitting spline to power curve

		# rotor2.root = RotorAeroPower(myref, npts_coarse_power_curve, npts_spline_power_curve)
		# # ---

		# rotor2.setup()

		# # === blade grid ===
		# rotor2['hubFraction'] = myref.hubFraction #0.025  # (Float): hub location as fraction of radius
		# rotor2['bladeLength'] = myref.bladeLength #61.5  # (Float, m): blade length (if not precurved or swept) otherwise length of blade before curvature
		# rotor2['precone'] = myref.precone #2.5  # (Float, deg): precone angle
		# rotor2['tilt'] = myref.tilt #5.0  # (Float, deg): shaft tilt
		# # ...
		# # ---
		# rotor2['yaw'] = 0.0  # (Float, deg): yaw error
		# rotor2['nBlades'] = myref.nBlades #3  # (Int): number of blades

		# # === blade geometry ===
		# rotor2['r_max_chord'] = myref.r_max_chord #0.23577  # (Float): location of max chord on unit radius
		# rotor2['chord_in'] = myref.chord #np.array([3.2612, 4.5709, 3.3178, 1.4621])  # (Array, m): chord at control points. defined at hub, then at linearly spaced locations from r_max_chord to tip
		# rotor2['theta_in'] = myref.theta #np.array([13.2783, 7.46036, 2.89317, -0.0878099])  # (Array, deg): twist at control points.  defined at linearly spaced locations from r[idx_cylinder] to tip
		# rotor2['precurve_in'] = myref.precurve #np.array([0.0, 0.0, 0.0])  # (Array, m): precurve at control points.  defined at same locations at chord, starting at 2nd control point (root must be zero precurve)
		# rotor2['presweep_in'] = myref.presweep #np.array([0.0, 0.0, 0.0])  # (Array, m): precurve at control points.  defined at same locations at chord, starting at 2nd control point (root must be zero precurve)
		# rotor2['sparT_in'] = myref.spar_thickness #np.array([0.05, 0.047754, 0.045376, 0.031085, 0.0061398])  # (Array, m): spar cap thickness parameters
		# rotor2['teT_in'] = myref.te_thickness #np.array([0.1, 0.09569, 0.06569, 0.02569, 0.00569])  # (Array, m): trailing-edge thickness parameters

		# # === atmosphere ===
		# rotor2['analysis.rho'] = 1.225  # (Float, kg/m**3): density of air
		# rotor2['analysis.mu'] = 1.81206e-5  # (Float, kg/m/s): dynamic viscosity of air
		# rotor2['hub_height'] = myref.hub_height #90.0
		# rotor2['analysis.shearExp'] = 0.25  # (Float): shear exponent
		# rotor2['turbine_class'] = myref.turbine_class #TURBINE_CLASS['I']  # (Enum): IEC turbine class
		# rotor2['cdf_reference_height_wind_speed'] = myref.hub_height #90.0  # (Float): reference hub height for IEC wind speed (used in CDF calculation)

		# # === control ===
		# rotor2['control_Vin'] = myref.control_Vin #3.0  # (Float, m/s): cut-in wind speed
		# rotor2['control_Vout'] = myref.control_Vout #25.0  # (Float, m/s): cut-out wind speed
		# rotor2['control_ratedPower'] = myref.rating #5e6  # (Float, W): rated power
		# rotor2['control_minOmega'] = myref.control_minOmega #0.0  # (Float, rpm): minimum allowed rotor rotation speed
		# rotor2['control_maxOmega'] = myref.control_maxOmega #12.0  # (Float, rpm): maximum allowed rotor rotation speed
		# rotor2['control_tsr'] = myref.control_tsr #7.55  # (Float): tip-speed ratio in Region 2 (should be optimized externally)
		# rotor2['control_pitch'] = myref.control_pitch #0.0  # (Float, deg): pitch angle in region 2 (and region 3 for fixed pitch machines)

		# # === aero and structural analysis options ===
		# rotor2['nSector'] = 4  # (Int): number of sectors to divide rotor face into in computing thrust and power
		# rotor2['AEP_loss_factor'] = 1.0  # (Float): availability and other losses (soiling, array, etc.)
		# rotor2['drivetrainType'] = myref.drivetrain #DRIVETRAIN_TYPE['GEARED']  # (Enum)
		# # ---

		# # --- run and outputs
		# rotor2.run()
		# # -----------------------------------------------------------------------


		outpath = '..\..\..\docs\images'
		# Power Curve
		f, ax = plt.subplots(1,1,figsize=(5.3, 4))
		ax.plot(rotor['V'], rotor['P']/1e6)
		ax.set(xlabel='Wind Speed (m/s)' , ylabel='Power (MW)')
		ax.set_ylim([0, 10.3])
		ax.set_xlim([0, 25])
		f.tight_layout()
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		# f.savefig(os.path.abspath(os.path.join(outpath,'power_curve_dtu10mw.png')))
		# f.savefig(os.path.abspath(os.path.join(outpath,'power_curve_dtu10mw.pdf')))

		# Chord
		fc, axc = plt.subplots(1,1,figsize=(5.3, 4))
		rc_c = np.r_[0.0, myref.r_cylinder, np.linspace(rotor['r_max_chord'], 1.0, NINPUT-2)]
		rc_c2 = np.r_[0.0, myref.r_cylinder, np.linspace(rotor2['r_max_chord'], 1.0, NINPUT-2)]
		r = (rotor['r_pts'] - rotor['Rhub'])/rotor['bladeLength']
		# axc.plot(r, rotor2['chord'], c='k', label='Initial')
		# axc.plot(rc_c2, rotor2['chord_in'], '.', c='k')
		axc.plot(r, rotor['chord'], c='b', label='Optimized')
		axc.plot(rc_c, rotor['chord_in'], '.', c='b')
		# for i, (x, y) in enumerate(zip(rc_c, rotor['chord_in'])):
		#     txt = '$c_%d$' % i
		#     if i<=1:
		#         axc.annotate(txt, (x,y), xytext=(x+0.01,y-0.4), textcoords='data')
		#     else:
		#         axc.annotate(txt, (x,y), xytext=(x+0.01,y+0.2), textcoords='data')
		# for i, (x, y) in enumerate(zip(rc_c2, rotor2['chord_in'])):
		#     txt = '$c_%d$' % i
		#     if i==0:
		#         axc.annotate(txt, (x,y), xytext=(x+0.01,y-0.4), textcoords='data', color='blue')
		#     else:
		#         axc.annotate(txt, (x,y), xytext=(x+0.01,y+0.2), textcoords='data', color='blue')
		axc.set(xlabel='Blade Fraction, $r/R$' , ylabel='Chord (m)')
		axc.set_ylim([0, 7.5])
		axc.set_xlim([0, 1.1])
		fc.tight_layout()
		axc.spines['right'].set_visible(False)
		axc.spines['top'].set_visible(False)
		# fc.savefig(os.path.abspath(os.path.join(outpath,'chord_opt_10mw.png')))
		# fc.savefig(os.path.abspath(os.path.join(outpath,'chord_opt_10mw.pdf')))

		# Twist
		ft, axt = plt.subplots(1,1,figsize=(5.3, 4))
		rc_t = rc_c#np.linspace(myref.r_cylinder, 1.0, NINPUT)
		rc_t2 = rc_c2#np.linspace(myref.r_cylinder, 1.0, NINPUT)
		r = (rotor['r_pts'] - rotor['Rhub'])/rotor['bladeLength']
		# axt.plot(r, rotor2['theta'], c='k', label='Initial')
		# axt.plot(rc_t2, rotor2['theta_in'], '.', c='k')
		axt.plot(r, rotor['theta'], c='b', label='Optimized')
		axt.plot(rc_t, rotor['theta_in'], '.', c='b')
		# for i, (x, y) in enumerate(zip(rc_t, rotor['theta_in'])):
		#     txt = '$\Theta_%d$' % i
		#     axt.annotate(txt, (x,y), xytext=(x+0.01,y+0.1), textcoords='data')
		# for i, (x, y) in enumerate(zip(rc_t2, rotor2['theta_in'])):
		#     txt = '$\Theta_%d$' % i
		#     axt.annotate(txt, (x,y), xytext=(x+0.01,y+0.1), textcoords='data', color='blue')
		axt.set(xlabel='Blade Fraction, $r/R$' , ylabel='Twist ($\deg$)')
		axt.set_ylim([-4, 16])
		axt.set_xlim([0, 1.1])
		ft.tight_layout()
		axt.spines['right'].set_visible(False)
		axt.spines['top'].set_visible(False)
		# ft.savefig(os.path.abspath(os.path.join(outpath,'theta_opt_10mw.png')))
		# ft.savefig(os.path.abspath(os.path.join(outpath,'theta_opt_10mw.pdf')))

		plt.show()


if __name__ == "__main__":

	rotor = RotorSE_Example2()
	rotor.execute()