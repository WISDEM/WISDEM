	d = PMSG_Disc()
	
	rad_ag  = 3.49 #3.494618182
	len_s   = 1.5 #1.506103927
	h_s     = 0.06 #0.06034976
	tau_p   = 0.07 #0.07541515 
	h_m     = 0.0105 #0.0090100202 
	h_ys    = 0.085 #0.084247994 #
	h_yr    = 0.055 #0.0545789687
	machine_rating = 10000000.0
	n_nom          = 7.54
	Torque  = 12.64e6
	b_st    = 0.460 #0.46381
	d_s     = 0.350 #0.35031 #
	t_ws    = 0.150 #=0.14720 #
	n_s     = 5.0 #5.0
	t_d     = 0.105 #0.10 
	R_o     = 0.43 #0.43
	rho_Fe       = 7700.0        # Steel density Kg/m3
	rho_Fes      = 7850          # structural Steel density Kg/m3
	rho_Copper   = 8900.0        # copper density Kg/m3
	rho_PM       = 7450.0        # typical density Kg/m3 of neodymium magnets (added 2019 09 18) - for pmsg_[disc|arms]
	
	shaft_cm     = np.zeros(3)
	shaft_length = 2.0
	    
	B_symax, B_tmax, B_rymax, B_smax, B_pm1, B_g, N_s, b_s, b_t, A_Cuscalc, b_m, p, E_p, \
	        f, I_s, R_s, L_s, A_1, J_s, Losses, K_rad, gen_eff, S, Slot_aspect_ratio, Copper, Iron, \
	        u_Ar, y_Ar, u_As, y_As, z_A_s, u_all_r, u_all_s, y_all, z_all_s, z_all_r, b_all_s, \
	        TC1, TC2, TC3, R_out, Structural_mass, Mass, mass_PM, cm, I	= d.compute(rad_ag, 
	        len_s, h_s, tau_p, h_m, h_ys, h_yr, machine_rating, n_nom, Torque,          
	        b_st, d_s, t_ws, n_s, t_d, R_o, rho_Fe, rho_Copper, rho_Fes, rho_PM, shaft_cm, shaft_length)
            
	sys.stderr.write('B_symax           {:15.7f}\n'.format(B_symax))
	sys.stderr.write('B_tmax            {:15.7f}\n'.format(B_tmax))
	sys.stderr.write('B_rymax           {:15.7f}\n'.format(B_rymax))
	sys.stderr.write('B_smax            {:15.7f}\n'.format(B_smax))
	sys.stderr.write('B_pm1             {:15.7f}\n'.format(B_pm1))
	sys.stderr.write('B_g               {:15.7f}\n'.format(B_g))
	sys.stderr.write('N_s               {:15.7f}\n'.format(N_s))
	sys.stderr.write('b_s               {:15.7f}\n'.format(b_s))
	sys.stderr.write('b_t               {:15.7f}\n'.format(b_t))
	sys.stderr.write('A_Cuscalc         {:15.7f}\n'.format(A_Cuscalc))
	sys.stderr.write('b_m               {:15.7f}\n'.format(b_m))
	sys.stderr.write('p                 {:15.7f}\n'.format(p))
	sys.stderr.write('E_p               {:15.7f}\n'.format(E_p))
	sys.stderr.write('f                 {:15.7f}\n'.format(f))
	sys.stderr.write('I_s               {:15.7f}\n'.format(I_s))
	sys.stderr.write('R_s               {:15.7f}\n'.format(R_s))
	sys.stderr.write('L_s               {:15.7f}\n'.format(L_s))
	sys.stderr.write('A_1               {:15.7f}\n'.format(A_1))
	sys.stderr.write('J_s               {:15.7f}\n'.format(J_s))
	sys.stderr.write('Losses            {:15.7f}\n'.format(Losses))
	sys.stderr.write('K_rad             {:15.7f}\n'.format(K_rad))
	sys.stderr.write('gen_eff           {:15.7f}\n'.format(gen_eff))
	sys.stderr.write('S                 {:15.7f}\n'.format(S))
	sys.stderr.write('Slot_aspect_ratio {:15.7f}\n'.format(Slot_aspect_ratio))
	sys.stderr.write('Copper            {:15.7f}\n'.format(Copper))
	sys.stderr.write('Iron              {:15.7f}\n'.format(Iron))
	sys.stderr.write('u_Ar              {:15.7f}\n'.format(u_Ar))
	sys.stderr.write('y_Ar              {:15.7f}\n'.format(y_Ar))
	sys.stderr.write('u_As              {:15.7f}\n'.format(u_As))
	sys.stderr.write('y_As              {:15.7f}\n'.format(y_As))
	sys.stderr.write('z_A_s             {:15.7f}\n'.format(z_A_s))
	sys.stderr.write('u_all_r           {:15.7f}\n'.format(u_all_r))
	sys.stderr.write('u_all_s           {:15.7f}\n'.format(u_all_s))
	sys.stderr.write('y_all             {:15.7f}\n'.format(y_all))
	sys.stderr.write('z_all_s           {:15.7f}\n'.format(z_all_s))
	sys.stderr.write('z_all_r           {:15.7f}\n'.format(z_all_r))
	sys.stderr.write('b_all_s           {:15.7f}\n'.format(b_all_s))
	sys.stderr.write('TC1               {:15.7f}\n'.format(TC1))
	sys.stderr.write('TC2               {:15.7f}\n'.format(TC2))
	sys.stderr.write('TC3               {:15.7f}\n'.format(TC3))
	sys.stderr.write('R_out             {:15.7f}\n'.format(R_out))
	sys.stderr.write('Structural_mass   {:15.7f}\n'.format(Structural_mass))
	sys.stderr.write('Mass              {:15.7f}\n'.format(Mass))
	sys.stderr.write('mass_PM           {:15.7f}\n'.format(mass_PM))
	sys.stderr.write('cm                {:15.7f} {:15.7f} {:15.7f}\n'.format(cm[0], cm[1], cm[2]))
	sys.stderr.write('I		            {:15.7f} {:15.7f} {:15.7f}\\n'.format(I[0], I[1], I[2]))






    scig = SCIG()
    
    r_s                = 0.55    # meter
                                 
    len_s                = 1.30    # meter
    h_s                = 0.090   # meter
    h_r                = 0.050   # meter
    machine_rating     = 5000000.0
    n_nom              = 1200.0
    Gearbox_efficiency = 0.955        
    I_0                = 140     # Ampere
    rho_Fe             = 7700.0  # Steel density Kg/m3
    rho_Copper         = 8900.0  # copper density Kg/m3       
    B_symax            = 1.4     # Tesla
    shaft_cm           = np.array([0.0, 0.0, 0.0])
    shaft_length       = 2.0
        
    rad_ag = r_s
        
    outputs = scig.compute(rad_ag, len_s, h_s, h_r, machine_rating, n_nom, Gearbox_efficiency, I_0, 
                rho_Fe, rho_Copper, B_symax, shaft_cm, shaft_length)

        
