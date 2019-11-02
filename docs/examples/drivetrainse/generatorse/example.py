# 1 ---------
import generatorse.PMSG_arms
import generatorse.DFIG


# 1 ---------
# 2 ---------

## Initial design variables for a DD PMSG designed for a 5MW turbine

opt_problem = PMSG_arms_Opt('CONMINdriver','PMSG_Cost.Costs',1) # Optimiser and Objective function
opt_problem.P_rated=5.0e6															# Rated power
opt_problem.T_rated=4.143289e6												# Rated torque (Nm)
opt_problem.N=12.1																		# Rated speed (rpm)
opt_problem.Eta_target = 93														# Target design efficiency %
opt_problem.PMSG_r_s= 3.26														# Air gap radius (meter)
opt_problem.PMSG_l_s= 1.6															# core length (meter)
opt_problem.PMSG_h_s = 0.07														# Stator slot height (meter)
opt_problem.PMSG_tau_p = 0.08													# Pole pitch (meter)
opt_problem.PMSG_h_m = 0.009													# Magnet height (meter)
opt_problem.PMSG_h_ys = 0.075												# Stator yoke height (meter)
opt_problem.PMSG_h_yr = 0.075													# Rotor yoke height (meter)
opt_problem.PMSG_n_s = 5															# Stator arms
opt_problem.PMSG_b_st = 0.480													# Stator circumferential arm dimension (meter)
opt_problem.PMSG_n_r =5																# Rotor arms
opt_problem.PMSG_b_r = 0.530													# Rotor circumferential arm dimension (meter)
opt_problem.PMSG_d_r = 0.7														# Rotor axial arm dimension (meter)
opt_problem.PMSG_d_s= 0.35															# Stator axial arm dimension (meter)
opt_problem.PMSG_t_wr =0.06														# Rotor arm thickness (meter)
opt_problem.PMSG_t_ws =0.06														# Stator arm thickness (meter)
opt_problem.PMSG_R_o =0.43														# Main shaft radius (meter)

#Specific costs
opt_problem.C_Cu   =4.786                  						# Unit cost of Copper $/kg
opt_problem.C_Fe	= 0.556                    					# Unit cost of Iron/magnetic steel $/kg
opt_problem.C_Fes =0.50139                   					# specific cost of structural steel

#Material properties
opt_problem.rho_Fe = 7700                 						# magnetic Steel density
opt_problem.rho_Copper =8900              						# Kg/m3 copper density
opt_problem.rho_PM =7450                  						# Kg/m3 magnet density
opt_problem.rho_Fes =7850                  						# Kg/m3 structural steel density

# 2 ----------
# 3 ----------

opt_problem.run()

# 3 ----------
# 4 ----------

raw_data = {'Parameters': ['Rating','Stator Arms', 'Stator Axial arm dimension','Stator Circumferential arm dimension',' Stator arm Thickness' ,'Rotor arms','Rotor Axial arm dimension','Rotor Circumferential arm dimension' ,'Rotor arm Thickness',' Stator Radial deflection', 'Stator Axial deflection','Stator circum deflection',' Rotor Radial deflection', 'Rotor Axial deflection','Rotor circum deflection', 'Air gap diameter','Overall Outer diameter', 'Stator length', 'l/d ratio','Slot_aspect_ratio','Pole pitch', 'Stator slot height','Stator slotwidth','Stator tooth width', 'Stator yoke height', 'Rotor yoke height', 'Magnet height', 'Magnet width', 'Peak air gap flux density fundamental','Peak stator yoke flux density','Peak rotor yoke flux density','Flux density above magnet','Maximum Stator flux density','Maximum tooth flux density','Pole pairs', 'Generator output frequency', 'Generator output phase voltage', 'Generator Output phase current', 'Stator resistance','Synchronous inductance', 'Stator slots','Stator turns','Conductor cross-section','Stator Current density ','Specific current loading','Generator Efficiency ','Iron mass','Magnet mass','Copper mass','Mass of Arms', 'Total Mass','Total Material Cost'],
			'Values': [opt_problem.PMSG.P_gennom/1000000,opt_problem.PMSG.n_s,opt_problem.PMSG.d_s*1000,opt_problem.PMSG.b_st*1000,opt_problem.PMSG.t_ws*1000,opt_problem.PMSG.n_r,opt_problem.PMSG.d_r*1000,opt_problem.PMSG.b_r*1000,opt_problem.PMSG.t_wr*1000,opt_problem.PMSG.Stator_delta_radial*1000,opt_problem.PMSG.Stator_delta_axial*1000,opt_problem.PMSG.Stator_circum*1000,opt_problem.PMSG.Rotor_delta_radial*1000,opt_problem.PMSG.Rotor_delta_axial*1000,opt_problem.PMSG.Rotor_circum*1000,2*opt_problem.PMSG.r_s,opt_problem.PMSG.R_out*2,opt_problem.PMSG.l_s,opt_problem.PMSG.K_rad,opt_problem.PMSG.Slot_aspect_ratio,opt_problem.PMSG.tau_p*1000,opt_problem.PMSG.h_s*1000,opt_problem.PMSG.b_s*1000,opt_problem.PMSG.b_t*1000,opt_problem.PMSG.t_s*1000,opt_problem.PMSG.t*1000,opt_problem.PMSG.h_m*1000,opt_problem.PMSG.b_m*1000,opt_problem.PMSG.B_g,opt_problem.PMSG.B_symax,opt_problem.PMSG.B_rymax,opt_problem.PMSG.B_pm1,opt_problem.PMSG.B_smax,opt_problem.PMSG.B_tmax,opt_problem.PMSG.p,opt_problem.PMSG.f,opt_problem.PMSG.E_p,opt_problem.PMSG.I_s,opt_problem.PMSG.R_s,opt_problem.PMSG.L_s,opt_problem.PMSG.S,opt_problem.PMSG.N_s,opt_problem.PMSG.A_Cuscalc,opt_problem.PMSG.J_s,opt_problem.PMSG.A_1/1000,opt_problem.PMSG.gen_eff,opt_problem.PMSG.Iron/1000,opt_problem.PMSG.mass_PM/1000,opt_problem.PMSG.Copper/1000,opt_problem.PMSG.Structural_mass/1000,opt_problem.PMSG.Mass/1000,opt_problem.PMSG_Cost.Costs/1000],
				'Limit': ['','','',opt_problem.PMSG.b_all_s*1000,'','','',opt_problem.PMSG.b_all_r*1000,'',opt_problem.PMSG.u_all_s*1000,opt_problem.PMSG.y_all*1000,opt_problem.PMSG.z_all_s*1000,opt_problem.PMSG.u_all_r*1000,opt_problem.PMSG.y_all*1000,opt_problem.PMSG.z_all_r*1000,'','','','(0.2-0.27)','(4-10)','','','','','','','','','','<2','<2','<2',opt_problem.PMSG.B_g,'','','','','>500','','','','','5','3-6','60','>93%','','','','','',''],
				'Units':['MW','unit','mm','mm','mm','mm','mm','','mm','mm','mm','mm','mm','mm','mm','m','m','m','','','mm','mm','mm','mm','mm','mm','mm','mm','T','T','T','T','T','T','-','Hz','V','A','ohm/phase','p.u','A/mm^2','slots','turns','mm^2','kA/m','%','tons','tons','tons','tons','tons','k$']}
	df=pd.DataFrame(raw_data, columns=['Parameters','Values','Limit','Units'])
print df

# 4 ---------
# 5 ---------

# # Initial design values for a DFIG designed for a 5MW turbine

opt_problem = DFIG_Opt('CONMINdriver','DFIG_Cost.Costs',1)    # Optimiser and Objective function
opt_problem.Eta_target=93														# Target overall drivetrain efficiency
opt_problem.DFIG_P_rated=5e6												# Rated power
opt_problem.DFIG_N_rated=1200												# Rated speed
opt_problem.Gearbox_efficiency=0.955								# Gearbox efficiency
opt_problem.DFIG_r_s= 0.61                          # Air gap radius (meter)
opt_problem.DFIG_l_s= 0.49 													# Core length (meter)
opt_problem.DFIG_h_s = 0.08 													# Stator Slot height (meter)
opt_problem.DFIG_h_r = 0.100 												# Rotor Slot height (meter)
opt_problem.DFIG_I_0 = 40 													# No-load magnetization current (Ampere)
opt_problem.DFIG_B_symax = 1.3 											# Peak Stator yoke flux density (Tesla)
opt_problem.DFIG_S_Nmax = -0.2 										# Maximum slip

# Specific costs
opt_problem.C_Cu   =4.786														# Unit cost of Copper $/kg
opt_problem.C_Fe	= 0.556                    				# Unit cost of Iron $/kg
opt_problem.C_Fes =0.50139                   				# specific cost of structure

#Material properties
opt_problem.rho_Fe = 7700                 					#Steel density
opt_problem.rho_Copper =8900                  			# Kg/m3 copper density

# 5 ---------
# 6 ---------

#Run optimization
opt_problem.run()

# 6 ---------
# 7 ---------
raw_data = {'Parameters': ['Rating','Objective function','Air gap diameter', "Stator length","Kra","Diameter ratio", "Pole pitch(tau_p)", " Number of Stator Slots","Stator slot height(h_s)","Slots/pole/phase","Stator slot width(b_s)", " Stator slot aspect ratio","Stator tooth width(b_t)", "Stator yoke height(h_ys)","Rotor slots", "Rotor yoke height(h_yr)", "Rotor slot height(h_r)", "Rotor slot width(b_r)"," Rotor Slot aspect ratio", "Rotor tooth width(b_t)", "Peak air gap flux density","Peak air gap flux density fundamental","Peak stator yoke flux density","Peak rotor yoke flux density","Peak Stator tooth flux density","Peak rotor tooth flux density","Pole pairs", "Generator output frequency", "Generator output phase voltage", "Generator Output phase current","Optimal Slip","Stator Turns","Conductor cross-section","Stator Current density","Specific current loading","Stator resistance", "Stator leakage inductance", "Excited magnetic inductance"," Rotor winding turns","Conductor cross-section","Magnetization current","I_mag/Is"," Rotor Current density","Rotor resitance", " Rotor leakage inductance", "Generator Efficiency","Overall drivetrain Efficiency","Iron mass","Copper mass","Structural Steel mass","Total Mass","Total Material Cost"],
		'Values': [opt_problem.DFIG.machine_rating/1e6,opt_problem.Objective_function,2*opt_problem.DFIG.r_s,opt_problem.DFIG.l_s,opt_problem.DFIG.K_rad,opt_problem.DFIG.D_ratio,opt_problem.DFIG.tau_p*1000,opt_problem.DFIG.N_slots,opt_problem.DFIG.h_s*1000,opt_problem.DFIG.q1,opt_problem.DFIG.b_s*1000,opt_problem.DFIG.Slot_aspect_ratio1,opt_problem.DFIG.b_t*1000,opt_problem.DFIG.h_ys*1000,opt_problem.DFIG.Q_r,opt_problem.DFIG.h_yr*1000,opt_problem.DFIG.h_r*1000,opt_problem.DFIG.b_r*1000,opt_problem.DFIG.Slot_aspect_ratio2,opt_problem.DFIG.b_tr*1000,opt_problem.DFIG.B_g,opt_problem.DFIG.B_g1,opt_problem.DFIG.B_symax,opt_problem.DFIG.B_rymax,opt_problem.DFIG.B_tsmax,opt_problem.DFIG.B_trmax,opt_problem.DFIG.p,opt_problem.DFIG.f,opt_problem.DFIG.E_p,opt_problem.DFIG.I_s,opt_problem.DFIG.S_Nmax,opt_problem.DFIG.N_s,opt_problem.DFIG.A_Cuscalc,opt_problem.DFIG.J_s,opt_problem.DFIG.A_1/1000,opt_problem.DFIG.R_s,opt_problem.DFIG.L_s,opt_problem.DFIG.L_sm,opt_problem.DFIG.N_r,opt_problem.DFIG.A_Curcalc,opt_problem.DFIG.I_0,opt_problem.DFIG.Current_ratio,opt_problem.DFIG.J_r,opt_problem.DFIG.R_R,opt_problem.DFIG.L_r,opt_problem.DFIG.gen_eff,opt_problem.DFIG.Overall_eff,opt_problem.DFIG.Iron/1000,opt_problem.DFIG.Copper/1000,opt_problem.DFIG.Structural_mass/1000,opt_problem.DFIG.Mass/1000,opt_problem.DFIG_Cost.Costs/1000],
			'Limit': ['','','','','(0.2-1.5)','(1.37-1.4)','','','','','','(4-10)','','','','','','','(4-10)','','(0.7-1.2)','','2','2.1','2.1','2.1','','','(500-5000)','','(-0.002-0.3)','','','(3-6)','<60','','','','','','','(0.1-0.3)','(3-6)','','','',opt_problem.Eta_target,'','','','',''],
				'Units':['MW','','m','m','-','-','mm','-','mm','','mm','','mm','mm','-','mm','mm','mm','-','mm','T','T','T','T','T','T','-','Hz','V','A','','turns','mm^2','A/mm^2','kA/m','ohms','p.u','p.u','turns','mm^2','A','','A/mm^2','ohms','p.u','%','%','Tons','Tons','Tons','Tons','$1000']}
	df=pandas.DataFrame(raw_data, columns=['Parameters','Values','Limit','Units'])
	print(df)

# 7 ---------