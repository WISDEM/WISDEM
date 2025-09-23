import numpy as np

eps = 1e-10

def properties(chord, tw_aero_d, tw_prime_d, le_loc, xnode, ynode, e1, e2, g12, anu12, density, xsec_nodeU, n_laminaU, n_pliesU, t_lamU, tht_lamU, mat_lamU, xsec_nodeL, n_laminaL, n_pliesL, t_lamL, tht_lamL, mat_lamL, nweb, loc_web, n_laminaW, n_pliesW, t_lamW, tht_lamW, mat_lamW):

    chord = np.array(chord)
    tw_aero_d = np.array(tw_aero_d)
    tw_prime_d = np.array(tw_prime_d)
    le_loc = np.array(le_loc)
    xnode = np.array(xnode)
    ynode = np.array(ynode)
    e1 = np.array(e1)
    e2 = np.array(e2)
    g12 = np.array(g12)
    anu12 = np.array(anu12)
    density = np.array(density)
    xsec_nodeU = np.array(xsec_nodeU)
    t_lamU = np.array(t_lamU)
    tht_lamU = np.array(tht_lamU)
    xsec_nodeL = np.array(xsec_nodeL)
    t_lamL = np.array(t_lamL)
    tht_lamL = np.array(tht_lamL)
    loc_web = np.array(loc_web)
    t_lamW = np.array(t_lamW)
    tht_lamW = np.array(tht_lamW)
    
    n_laminaU = np.array(n_laminaU, dtype=np.int_)
    n_pliesU = np.array(n_pliesU, dtype=np.int_)
    mat_lamU = np.array(mat_lamU, dtype=np.int_)
    n_laminaL = np.array(n_laminaL, dtype=np.int_)
    n_pliesL = np.array(n_pliesL, dtype=np.int_)
    mat_lamL = np.array(mat_lamL, dtype=np.int_)
    n_laminaW = np.array(n_laminaW, dtype=np.int_)
    n_pliesW = np.array(n_pliesW, dtype=np.int_)
    mat_lamW = np.array(mat_lamW, dtype=np.int_)

    max_sectors = np.int_(np.max([n_laminaU.size, n_laminaL.size, n_laminaW.size]))
    max_laminatesUL = np.int_(np.max([n_laminaU.max(), n_laminaL.max()]))
    max_laminatesW = np.int_(np.max(n_laminaW.max()))

    n_materials = len(e1)
    n_af_nodes = len(xnode)
    n_sctU = len(n_laminaU)
    n_sctL = len(n_laminaL)
    
    r2d = 180.0 / np.pi

    webs_exist = nweb > 0

    if le_loc < 0.0:
        print(' WARNING** leading edge aft of reference axis **')

    if np.any(anu12 > np.sqrt(e1/e2)):
        idx = np.where(anu12 > np.sqrt(e1/e2))[0]
        raise ValueError(f'**ERROR** material {idx+1} properties not consistent')

    if n_af_nodes <= 2:
        raise ValueError(' ERROR** min 3 nodes reqd to define airfoil geom')

    location = np.argmin(xnode)
    if location != 0:
        raise ValueError(' ERROR** the first airfoil node not a leading node')

    if np.abs(xnode[0]) > eps or np.abs(ynode[0]) > eps:
        raise ValueError(' ERROR** leading-edge node not located at (0,0)')

    location = np.argmax(xnode)
    if xnode.max() > 1.0:
        raise ValueError(' ERROR** trailing-edge node exceeds chord boundary')

    tenode_u = location
    ncounter = np.where(np.abs(xnode[location:] - xnode[location]) < eps)[0].max()

    tenode_l = tenode_u + ncounter
    nodes_u = xnode[:(tenode_u+1)].size
    nodes_l = xnode[tenode_l:].size

    xnode_u = xnode[:nodes_u]
    ynode_u = ynode[:nodes_u]
    xnode_l = np.r_[xnode[0], np.flipud(xnode[tenode_l:])]
    ynode_l = np.r_[ynode[0], np.flipud(ynode[tenode_l:])]

    if np.any(np.abs(np.diff(xnode_u) <= eps)):
        raise ValueError(' ERROR** upper surface not single-valued')

    if np.any(np.abs(np.diff(xnode_l) <= eps)):
        raise ValueError(' ERROR** lower surface not single-valued')

    if ynode_u[1]/xnode_u[1] <= ynode_l[1]/xnode_l[1]:
        raise ValueError(' ERROR** airfoil node numbering not clockwise')

    yinterp_lu = np.interp(xnode_l, xnode_u, ynode_u)
    if np.any(ynode_l[1:-1] >= yinterp_lu[1:-1]):
        raise ValueError(' ERROR** airfoil shape self-crossing')

    if webs_exist:
        # Vectorize the embed
        weby_u = np.interp(loc_web, xnode_u, ynode_u)
        weby_l = np.interp(loc_web, xnode_l, ynode_l)
        xnode_u, idx = np.unique(np.r_[xnode_u, loc_web], return_index=True)
        ynode_u = (np.r_[ynode_u, weby_u])[idx]
        xnode_l, idx = np.unique(np.r_[xnode_l, loc_web], return_index=True)
        ynode_l = (np.r_[ynode_l, weby_l])[idx]
    else:
        weby_u = weby_l = np.array([])

    n_scts = np.array([n_sctU, n_sctL])
    xsec_node = np.zeros((2, max_sectors+1))
    xsec_node[0,:xsec_nodeU.size] = xsec_nodeU
    xsec_node[1,:xsec_nodeL.size] = xsec_nodeL
    if np.any(n_scts <= 0):
        raise ValueError(' ERROR** no of sectors not positive')
    if np.any(xsec_node[:, 0] < 0.0):
        raise ValueError(' ERROR** sector node x-location not positive')
    if np.any(np.diff(xsec_nodeU) <= 0.0):
        raise ValueError(' ERROR** upper sector nodal x-locations not in ascending order')
    if np.any(np.diff(xsec_nodeL) <= 0.0):
        raise ValueError(' ERROR** lower sector nodal x-locations not in ascending order')

    n_weblams = n_laminaW    
    n_laminas = np.zeros((2, max_sectors), dtype=np.int_)
    n_laminas[0,:n_laminaU.size] = n_laminaU
    n_laminas[1,:n_laminaL.size] = n_laminaL
    
    tht_lam = np.zeros((2, max_sectors, max_laminatesUL))
    tlam = np.zeros((2, max_sectors, max_laminatesUL))
    mat_id = np.zeros((2, max_sectors, max_laminatesUL), dtype=np.int_)
    wmat_id = np.zeros((nweb, max_laminatesW), dtype=np.int_)
    tht_wlam = np.zeros((nweb, max_laminatesW))
    twlam = np.zeros((nweb, max_laminatesW))

    k = 0
    for i in range(n_sctU):
        for j in range(n_laminaU[i]):
            tlam[0, i, j] = n_pliesU[k] * t_lamU[k]
            tht_lam[0, i, j] = tht_lamU[k] / r2d
            mat_id[0, i, j] = mat_lamU[k] - 1 # Input is 1-based indexing for Fortran
            k += 1

    k = 0
    for i in range(n_sctL):
        for j in range(n_laminaL[i]):
            tlam[1, i, j] = n_pliesL[k] * t_lamL[k]
            tht_lam[1, i, j] = tht_lamL[k] / r2d
            mat_id[1, i, j] = mat_lamL[k] - 1 # Input is 1-based indexing for Fortran
            k += 1

    k = 0
    for i in range(nweb):
        for j in range(n_laminaW[i]):
            twlam[i, j] = n_pliesW[k] * t_lamW[k]
            tht_wlam[i, j] = tht_lamW[k] / r2d
            wmat_id[i, j] = mat_lamW[k] - 1 # Input is 1-based indexing for Fortran
            k += 1

    xu1 = xsec_node[0, 0]
    xu2 = xsec_node[0, n_sctU]
    if xu2 > xnode_u[-1]:
        raise ValueError(f' ERROR** upper-surf last sector node out of bounds {xu2} {xnode_u[-1]}')
    xl1 = xsec_node[1, 0]
    xl2 = xsec_node[1, n_sctL]
    if xl2 > xnode_l[-1]:
        raise ValueError(f' ERROR** lower-surf last sector node out of bounds {xl2} {xnode_l[-1]}')
            
    # Vectorize the embed
    yinterp_u = np.interp(xsec_nodeU, xnode_u, ynode_u)
    xnode_u, idx = np.unique(np.r_[xnode_u, xsec_nodeU], return_index=True)
    ynode_u = (np.r_[ynode_u, yinterp_u])[idx]
    yu1 = yinterp_u[0]
    yu2 = yinterp_u[-1]
    ndu1 = np.searchsorted(xnode_u, xu1)
    ndu2 = np.searchsorted(xnode_u, xu2)

    yinterp_l = np.interp(xsec_nodeL, xnode_l, ynode_l)
    xnode_l, idx = np.unique(np.r_[xnode_l, xsec_nodeL], return_index=True)
    ynode_l = (np.r_[ynode_l, yinterp_l])[idx]
    yl1 = yinterp_l[0]
    yl2 = yinterp_l[-1]
    ndl1 = np.searchsorted(xnode_l, xl1)
    ndl2 = np.searchsorted(xnode_l, xl2)

    nseg_u = ndu2 - ndu1
    nseg_l = ndl2 - ndl1
    nseg_p = nseg_u + nseg_l
    nseg = nseg_p + nweb if webs_exist else nseg_p
    
    if np.abs(xu1 - xl1) > eps:
        print(' WARNING** the leading edge may be open; check closure')
    else:
        if (yu1 - yl1) > eps:
            wreq = 1
            if webs_exist:
                if np.abs(xu1 - loc_web[0]) < eps:
                    wreq = 0
            if wreq == 1:
                print(' WARNING** open leading edge; check web requirement')

    if np.abs(xu2 - xl2) > eps:
        print(' WARNING** the trailing edge may be open; check closure')
    else:
        if (yu2 - yl2) > eps:
            wreq = 1
            if webs_exist:
                if np.abs(xu2 - loc_web[-1]) < eps:
                    wreq = 0
            if wreq == 1:
                print(' WARNING** open trailing edge; check web requirement')

    if webs_exist:
        if loc_web[0] < xu1 or loc_web[0] < xl1:
            print('ERROR** first web out of sectors-bounded airfoil')
        if loc_web[-1] > xu2 or loc_web[-1] > xl2:
            print(' ERROR** last web out of sectors-bounded airfoil')

    # Begin segment calculations
    tw_aero = tw_aero_d / r2d
    tw_prime = tw_prime_d / r2d
    tphip = tw_prime

    anud = 1.0 - anu12 * anu12 * e2 / e1
    q11 = e1 / anud
    q22 = e2 / anud
    q12 = anu12 * e2 / anud
    q66 = g12

    isur, idsect, yseg, zseg, wseg, sthseg, cthseg, s2thseg, c2thseg = seg_info(chord, le_loc, nseg, nseg_u, nseg_p, xnode_u, ynode_u, xnode_l, ynode_l, ndl1, ndu1, loc_web, weby_u, weby_l, n_scts, xsec_node)

    # Initialize variables
    sigma = 0.0
    eabar = 0.0
    q11ya = 0.0
    q11za = 0.0
    
    #   segments loop for sc
    for iseg in range(nseg_p):
        # Initialize segment properties
        ks = isur[iseg]
        idsec = idsect[iseg]
        ysg = yseg[iseg]
        zsg = zseg[iseg]
        w = wseg[iseg]
        sths = sthseg[iseg]
        cths = cthseg[iseg]
        nlam = n_laminas[ks, idsec]

        # initialization for seg (sc)
        tbar = 0.0
        q11t = 0.0
        q11yt_u = 0.0
        q11zt_u = 0.0
        q11yt_l = 0.0
        q11zt_l = 0.0
        
        # Vectorized
        t = tlam[ks, idsec, :]       # thickness
        thp = tht_lam[ks, idsec, :]  # ply angle
        mat = mat_id[ks, idsec, :]   # material no.
        tbar = np.cumsum(np.r_[0, t[:-1]]) + (t / 2.0) # Just need half of the current increment
        y0 = ysg - ((-1.0) ** (ks+1)) * tbar * sths
        z0 = zsg + ((-1.0) ** (ks+1)) * tbar * cths
        
        qbar11, qbar22, qbar12, qbar16, qbar26, qbar66 = q_bars(thp, q11[mat], q22[mat], q12[mat], q66[mat])
        qtil = q_tildas(qbar11, qbar22, qbar12, qbar16, qbar26, qbar66)

        qtil11t = np.squeeze(qtil[0, 0, :] * t)
        q11t = np.sum(qtil11t)

        if iseg < nseg_u:
            q11yt_u = np.sum(qtil11t * y0)
            q11zt_u = np.sum(qtil11t * z0)
        else:
            q11yt_l = np.sum(qtil11t * y0)
            q11zt_l = np.sum(qtil11t * z0)

        tbar = t.sum() # Now need the full thing
        '''
        # Original python port with inner loop
        for ilam in range(nlam):
            t = tlam[ks, idsec, ilam]       # thickness
            thp = tht_lam[ks, idsec, ilam]  # ply angle
            mat = mat_id[ks, idsec, ilam]   # material no.

            tbar = tbar + t / 2.0
            y0 = ysg - ((-1.0) ** (ks+1)) * tbar * sths
            z0 = zsg + ((-1.0) ** (ks+1)) * tbar * cths

            # Call function q_bars and q_tildas
            qbar11, qbar22, qbar12, qbar16, qbar26, qbar66 = q_bars(thp, q11[mat], q22[mat], q12[mat], q66[mat])
            qtil = q_tildas(qbar11, qbar22, qbar12, qbar16, qbar26, qbar66)

            qtil11t = qtil[0, 0] * t
            q11t = q11t + qtil11t

            if iseg < nseg_u:
                q11yt_u = q11yt_u + qtil11t * y0
                q11zt_u = q11zt_u + qtil11t * z0
            else:
                q11yt_l = q11yt_l + qtil11t * y0
                q11zt_l = q11zt_l + qtil11t * z0

            tbar = tbar + t / 2.0
        '''

        # add seg contributions (sc)
        sigma = sigma + w * np.abs(zsg + ((-1.0) ** (ks+1)) * 0.5 * tbar * cths) * cths
        eabar = eabar + q11t * w
        q11ya = q11ya + (q11yt_u + q11yt_l) * w
        q11za = q11za + (q11zt_u + q11zt_l) * w

    # get section sc
    y_sc = q11ya / eabar
    z_sc = q11za / eabar
    #---------------- end section sc -----------

    #   initializations for section (properties)
    eabar = 0.0
    q11ya = 0.0
    q11za = 0.0
    ap = 0.0
    bp = 0.0
    cp = 0.0
    dp = 0.0
    ep = 0.0
    q11ysqa = 0.0
    q11zsqa = 0.0
    q11yza = 0.0
    mass = 0.0
    area = 0.0
    rhoya = 0.0
    rhoza = 0.0
    rhoysqa = 0.0
    rhozsqa = 0.0
    rhoyza = 0.0
    
    #   segments loop (for properties)
    for iseg in range(nseg):
        # Initialize segment properties
        ks = isur[iseg]
        idsec = idsect[iseg]
        ysg = yseg[iseg]
        zsg = zseg[iseg]
        w = wseg[iseg]
        sths = sthseg[iseg]
        cths = cthseg[iseg]
        s2ths = s2thseg[iseg]
        c2ths = c2thseg[iseg]

        nlam = n_laminas[ks, idsec] if ks >= 0 else n_weblams[idsec]

        tbar = 0.0
        q11t = 0.0
        q11yt = 0.0
        q11zt = 0.0
        dtbar = 0.0
        q2bar = 0.0
        zbart = 0.0
        ybart = 0.0
        tbart = 0.0
        q11ysqt = 0.0
        q11zsqt = 0.0
        q11yzt = 0.0
        rhot = 0.0
        rhoyt = 0.0
        rhozt = 0.0
        rhoysqt = 0.0
        rhozsqt = 0.0
        rhoyzt = 0.0

        # Vectorized
        if ks >= 0:
            t = tlam[ks, idsec, :]       # thickness
            thp = tht_lam[ks, idsec, :]  # ply angle
            mat = mat_id[ks, idsec, :]   # material no.
            tbar = np.cumsum(np.r_[0, t[:-1]]) + (t / 2.0) # Just need half of the current increment
            y0 = ysg - ((-1.0) ** (ks+1)) * tbar * sths - y_sc
            z0 = zsg + ((-1.0) ** (ks+1)) * tbar * cths - z_sc
        else:
            t = twlam[idsec, :]
            thp = tht_wlam[idsec, :]
            mat = wmat_id[idsec, :]
            tbar = np.cumsum(np.r_[0, t[:-1]]) + (t / 2.0) # Just need half of the current increment
            y0 = ysg - tbar / 2.0 - y_sc
            z0 = zsg - z_sc

        y0sq = y0 * y0
        z0sq = z0 * z0

        qbar11, qbar22, qbar12, qbar16, qbar26, qbar66 = q_bars(thp, q11[mat], q22[mat], q12[mat], q66[mat])
        qtil = q_tildas(qbar11, qbar22, qbar12, qbar16, qbar26, qbar66)

        ieta1 = (t ** 2) / 12.0
        izeta1 = (w ** 2) / 12.0
        iepz = 0.5 * (ieta1 + izeta1)
        iemz = 0.5 * (ieta1 - izeta1)
        ipp = iepz + iemz * c2ths
        iqq = iepz - iemz * c2ths
        ipq = iemz * s2ths
        qtil11t = np.squeeze(qtil[0, 0, :] * t)
        rot = density[mat] * t

        if ks >= 0:
            qtil12t = np.squeeze(qtil[0, 1, :]) * t
            qtil22t = np.squeeze(qtil[1, 1, :]) * t

            q11t = np.sum(qtil11t)
            q11yt = np.sum(qtil11t * y0)
            q11zt = np.sum(qtil11t * z0)
            dtbar = np.sum(qtil12t * (y0sq + z0sq) * tphip * t)
            q2bar = np.sum(qtil22t)
            zbart = np.sum(z0 * qtil12t)
            ybart = np.sum(y0 * qtil12t)
            tbart = np.sum(qtil12t)
            q11ysqt = np.sum(qtil11t * (y0sq + iqq))
            q11zsqt = np.sum(qtil11t * (z0sq + ipp))
            q11yzt = np.sum(qtil11t * (y0 * z0 + ipq))
            rhot = np.sum(rot)
            rhoyt = np.sum(rot * y0)
            rhozt = np.sum(rot * z0)
            rhoysqt = np.sum(rot * (y0sq + iqq))
            rhozsqt = np.sum(rot * (z0sq + ipp))
            rhoyzt = np.sum(rot * (y0 * z0 + ipq))
        else:
            q11t = np.sum(qtil11t)
            q11yt = np.sum(qtil11t * y0)
            q11zt = np.sum(qtil11t * z0)
            q11ysqt = np.sum(qtil11t * (y0sq + iqq))
            q11zsqt = np.sum(qtil11t * (z0sq + ipp))
            q11yzt = np.sum(qtil11t * (y0 * z0 + ipq))
            rhot = np.sum(rot)
            rhoyt = np.sum(rot * y0)
            rhozt = np.sum(rot * z0)
            rhoysqt = np.sum(rot * (y0sq + iqq))
            rhozsqt = np.sum(rot * (z0sq + ipp))
            rhoyzt = np.sum(rot * (y0 * z0 + ipq))
            
        tbar = t.sum() # Now need the full thing

        # Loop over laminates
        '''
        for ilam in range(nlam):
            if ks >= 0:
                t = tlam[ks, idsec, ilam]
                thp = tht_lam[ks, idsec, ilam]
                mat = mat_id[ks, idsec, ilam]
                tbar = tbar + t / 2.0
                y0 = ysg - ((-1.0) ** (ks+1)) * tbar * sths - y_sc
                z0 = zsg + ((-1.0) ** (ks+1)) * tbar * cths - z_sc
            else:
                t = twlam[idsec, ilam]
                thp = tht_wlam[idsec, ilam]
                mat = wmat_id[idsec, ilam]
                tbar = tbar + t / 2.0
                y0 = ysg - tbar / 2.0 - y_sc
                z0 = zsg - z_sc

            y0sq = y0 * y0
            z0sq = z0 * z0

            # Call function q_bars and q_tildas
            qbar11, qbar22, qbar12, qbar16, qbar26, qbar66 = q_bars(thp, q11[mat], q22[mat], q12[mat], q66[mat])
            qtil = q_tildas(qbar11, qbar22, qbar12, qbar16, qbar26, qbar66)

            ieta1 = (t ** 2) / 12.0
            izeta1 = (w ** 2) / 12.0
            iepz = 0.5 * (ieta1 + izeta1)
            iemz = 0.5 * (ieta1 - izeta1)
            ipp = iepz + iemz * c2ths
            iqq = iepz - iemz * c2ths
            ipq = iemz * s2ths
            qtil11t = qtil[0, 0] * t
            rot = density[mat] * t

            if ks >= 0:
                qtil12t = qtil[0, 1] * t
                qtil22t = qtil[1, 1] * t

                q11t = q11t + qtil11t
                q11yt = q11yt + qtil11t * y0
                q11zt = q11zt + qtil11t * z0
                dtbar = dtbar + qtil12t * (y0sq + z0sq) * tphip * t
                q2bar = q2bar + qtil22t
                zbart = zbart + z0 * qtil12t
                ybart = ybart + y0 * qtil12t
                tbart = tbart + qtil12t
                q11ysqt = q11ysqt + qtil11t * (y0sq + iqq)
                q11zsqt = q11zsqt + qtil11t * (z0sq + ipp)
                q11yzt = q11yzt + qtil11t * (y0 * z0 + ipq)
                rhot = rhot + rot
                rhoyt = rhoyt + rot * y0
                rhozt = rhozt + rot * z0
                rhoysqt = rhoysqt + rot * (y0sq + iqq)
                rhozsqt = rhozsqt + rot * (z0sq + ipp)
                rhoyzt = rhoyzt + rot * (y0 * z0 + ipq)
            else:
                q11t = q11t + qtil11t
                q11yt = q11yt + qtil11t * y0
                q11zt = q11zt + qtil11t * z0
                q11ysqt = q11ysqt + qtil11t * (y0sq + iqq)
                q11zsqt = q11zsqt + qtil11t * (z0sq + ipp)
                q11yzt = q11yzt + qtil11t * (y0 * z0 + ipq)
                rhot = rhot + rot
                rhoyt = rhoyt + rot * y0
                rhozt = rhozt + rot * z0
                rhoysqt = rhoysqt + rot * (y0sq + iqq)
                rhozsqt = rhozsqt + rot * (z0sq + ipp)
                rhoyzt = rhoyzt + rot * (y0 * z0 + ipq)

            tbar = tbar + t / 2.0
        '''

        eabar = eabar + q11t * w
        q11ya = q11ya + q11yt * w
        q11za = q11za + q11zt * w
        q11ysqa = q11ysqa + q11ysqt * w
        q11zsqa = q11zsqa + q11zsqt * w
        q11yza = q11yza + q11yzt * w

        if ks >= 0:
            wdq2bar = w / q2bar
            ap = ap + wdq2bar
            bp = bp + wdq2bar * tbart
            cp = cp + wdq2bar * dtbar
            dp = dp + wdq2bar * zbart
            ep = ep + wdq2bar * ybart

        area = area + w
        mass = mass + rhot * w
        rhoya = rhoya + rhoyt * w
        rhoza = rhoza + rhozt * w
        rhoysqa = rhoysqa + rhoysqt * w
        rhozsqa = rhozsqa + rhozsqt * w
        rhoyza = rhoyza + rhoyzt * w

    y_tc = q11ya / eabar
    z_tc = q11za / eabar
    
    sfbar = q11za
    slbar = q11ya
    eifbar = q11zsqa
    eilbar = q11ysqa
    eiflbar = q11yza
    
    sigm2 = sigma*2.0
    gjbar = sigm2*(sigm2+cp)/ap
    sftbar = -sigm2*dp/ap
    sltbar = -sigm2*ep/ap
    satbar = sigm2*bp/ap
    
    ycm_sc =   rhoya/mass #wrt sc
    zcm_sc =   rhoza/mass #wrt sc
    
    iflap_sc = rhozsqa #wrt sc
    ilag_sc = rhoysqa   #wrt sc
    ifl_sc = rhoyza     #wrt sc
    
    # get section tc and cm
    
    ytc_ref =  y_tc + y_sc  #wrt the ref axes
    ztc_ref =  z_tc + z_sc  #wrt the ref axes
    
    ycm_ref =  ycm_sc + y_sc    #wrt the ref axes
    zcm_ref =  zcm_sc + z_sc    #wrt the ref axes
    
    # moments of inertia # about ref_parallel axes at cm
    
    iflap_cm = iflap_sc - mass*zcm_sc**2
    ilag_cm = ilag_sc - mass*ycm_sc**2
    ifl_cm = ifl_sc - mass*ycm_sc*zcm_sc
    
    # inertia principal axes orientation and moments of inertia
    
    m_inertia = 0.5*(ilag_cm + iflap_cm)
    r_inertia = np.sqrt(0.25*((ilag_cm-iflap_cm)**2) + ifl_cm**2)
    
    if (iflap_cm <= ilag_cm):
        iflap_eta = m_inertia - r_inertia
        ilag_zeta = m_inertia + r_inertia
    else:
        iflap_eta = m_inertia + r_inertia
        ilag_zeta = m_inertia - r_inertia

    if (ilag_cm == iflap_cm):
        th_pa = np.pi/4.0
        if (np.abs(ifl_cm/iflap_cm) < 1e-6):
            th_pa = 0.0
    else:
        th_pa = 0.5*np.abs(np.arctan(2.0*ifl_cm/(ilag_cm-iflap_cm)))

    if (np.abs(ifl_cm) < eps):
        th_pa = 0.0
    else:
        if (iflap_cm >= ilag_cm and ifl_cm > 0.):
            th_pa = -th_pa
        if (iflap_cm >= ilag_cm and ifl_cm < 0.):
            th_pa = th_pa
        if (iflap_cm < ilag_cm and ifl_cm > 0.):
            th_pa = th_pa
        if (iflap_cm < ilag_cm and ifl_cm < 0.):
            th_pa = -th_pa

    # elastic principal axes orientation and principal bending stiffneses

    em_stiff = 0.5*(eilbar + eifbar)
    er_stiff = np.sqrt(0.25*((eilbar-eifbar)**2) + eiflbar**2)
    
    if (eifbar <= eilbar):
        pflap_stff = em_stiff - er_stiff
        plag_stff = em_stiff + er_stiff
    else:
        pflap_stff = em_stiff + er_stiff
        plag_stff = em_stiff - er_stiff

    if (eilbar == eifbar):
        the_pa = np.pi/4.0
    else:
        the_pa = 0.5*np.abs(np.arctan(2.0*eiflbar/(eilbar-eifbar)))

    if (np.abs(eiflbar) < eps):
        the_pa = 0.0
    else:
        if (eifbar >= eilbar and eiflbar > 0.):
            the_pa = -the_pa
        if (eifbar >= eilbar and eiflbar < 0.):
            the_pa = the_pa
        if (eifbar < eilbar and eiflbar > 0.):
            the_pa = the_pa
        if (eifbar < eilbar and eiflbar < 0.):
            the_pa = -the_pa

    #---------------- end properties computation -----------

    # ---------- prepare outputs --------------
    id_form = 1  # hardwired for wt's

    if (id_form == 1):
        tw_iner = tw_aero - th_pa
        str_tw =  tw_aero - the_pa
        y_sc = -y_sc
        ytc_ref = -ytc_ref
        ycm_ref = -ycm_ref
    else:         # for h/c
        #       note: for h/c, th_aero input is +ve acc to h/c convention
        tw_iner = tw_aero + th_pa
        str_tw =  tw_aero + the_pa

    # conversions
    eiflbar = -eiflbar
    sfbar = -sfbar
    sltbar = -sltbar
    tw_iner = tw_iner*r2d

    return (eifbar, eilbar, gjbar, eabar, eiflbar, sfbar, slbar, sftbar, sltbar, satbar, z_sc, y_sc, ztc_ref, ytc_ref, mass, area, iflap_eta, ilag_zeta, tw_iner, zcm_ref, ycm_ref)

def seg_info(ch, rle, nseg, nseg_u, nseg_p, xnode_u, ynode_u, xnode_l, ynode_l, ndl1, ndu1, loc_web, weby_u, weby_l, n_scts, xsec_node):
    # NOTE: coord transformation from xaf-yaf to yre-zref and seg info

    # inputs
    # real(dbp), intent(in) :: ch, rle  # chord length, loc of l.e. (non-d wrt chord)
    # integer, intent(in) :: nseg, nseg_u, nseg_p  # total number of segs, no of segs on the upper surface, no of segs for both upper and lower surfaces
    # real(dbp), intent(in), dimension(300) :: xnode_u, ynode_u, xnode_l, ynode_l  # x,y nodes on upper/lower
    # integer, intent(in) :: ndl1, ndu1 # 1st seg lhs node number lower/upper surface
    # real(dbp), intent(in), dimension(:) :: loc_web, weby_u, weby_l  # x coord of web, y coord of web upper/lower
    # integer, intent(in), dimension(2) :: n_scts  # no of sectors on 'is' surf
    # integer, intent(in) :: nsecnode
    # real(dbp), dimension(2, nsecnode) :: xsec_node  # x coord of sect-i lhs on 's' surf

    # outputs
    '''
    isur = -1 * np.ones(nseg, dtype=np.int_)  # surf id
    thseg = -np.pi / 2.0 * np.ones(nseg)

    # Vectorized
    iseg = np.arange(nseg, dtype=np.int_)
    isur[iseg < nseg_p] = 1 # Do this first
    isur[iseg < nseg_u] = 0
    nd_a = np.zeros(nseg, dtype=np.int_)
    nd_a[iseg < nseg_p] = ndl1 + iseg[iseg < nseg_p] - nseg_u
    nd_a[iseg < nseg_u] = ndu1 + iseg[iseg < nseg_u]
    xa = xnode_u[nd_a]
    ya = ynode_u[nd_a]
    xb = xnode_u[nd_a + 1]
    yb = ynode_u[nd_a + 1]
    
    iweb = iseg[iseg >= nseg_p] - nseg_p
    xa[iseg >= nseg_p] = loc_web[iweb]
    xb[iseg >= nseg_p] = xa[iseg >= nseg_p]
    ya[iseg >= nseg_p] = weby_u[iweb]
    yb[iseg >= nseg_p] = weby_l[iweb]

    xba = xb - xa
    yba = ya - yb
    yseg = ch * (2. * rle - xa - xb) / 2.0  # yref coord of mid-seg pt (in r-frame)
    zseg = ch * (ya + yb) / 2.0  # zref coord of mid-seg pt (in r-frame)
    wseg = ch * np.sqrt(xba ** 2 + yba ** 2)

    thseg[isur >= 0] = np.arctan(yba[isur >= 0] / xba[isur >= 0])  # thseg +ve in new y-z ref frame
    
    idsect = np.zeros(nseg, dtype=np.int_)  # associated sect or web number
    idsect[iseg >= nseg_p] = iweb
    for iseg in range(nseg_p):  # seg numbering from le clockwise
        ks = isur[iseg]
        for i in range(n_scts[ks]):
            if xa[iseg] > (xsec_node[ks, i] - eps) and xb[iseg] < (xsec_node[ks, i + 1] + eps):
                idsect[iseg] = i
                break

    '''
    isur = np.zeros(nseg, dtype=np.int_)  # surf id
    thseg = np.zeros(nseg)
    idsect = np.zeros(nseg, dtype=np.int_)  # associated sect or web number
    yseg = np.zeros(nseg)  # y-ref of mid-seg point
    zseg = np.zeros(nseg)  # z-ref of mid-seg point
    wseg = np.zeros(nseg)  # seg width
    
    for iseg in range(nseg):  # seg numbering from le clockwise
        ks = -2
        if iseg < nseg_u:  # upper surface segs
            nd_a = ndu1 + iseg
            xa = xnode_u[nd_a]
            ya = ynode_u[nd_a]
            xb = xnode_u[nd_a + 1]
            yb = ynode_u[nd_a + 1]
            ks = 0
        elif iseg < nseg_p:  # lower surface segs
            nd_a = ndl1 + iseg - nseg_u
            xa = xnode_l[nd_a]  # xref of node toward le (in a/f ref frame)
            ya = ynode_l[nd_a]  # yref of node toward le (in new ref frame)
            xb = xnode_l[nd_a + 1]  # xref of node toward te (in a/f ref frame)
            yb = ynode_l[nd_a + 1]  # yref of node toward te (in new ref frame)
            ks = 1
        else:
            iweb = iseg - nseg_p
            xa = loc_web[iweb]
            xb = xa
            ya = weby_u[iweb]
            yb = weby_l[iweb]
            ks = -1

        if ks == -2:
            print(f'iseg={iseg}')
            raise ValueError('ERROR** unknown, contact NREL')

        isur[iseg] = ks

        if ks >= 0:  # id associated sect number
            icheck = True
            for i in range(n_scts[ks]):
                if xa > (xsec_node[ks, i] - eps) and xb < (xsec_node[ks, i + 1] + eps):
                    idsect[iseg] = i
                    icheck = False
                    break

            if icheck:
                print('ERROR** unknown, contact NREL')
        else:
            idsect[iseg] = iweb  # id associated web number

        xba = xb - xa
        yba = ya - yb
        yseg[iseg] = ch * (2. * rle - xa - xb) / 2.0  # yref coord of mid-seg pt (in r-frame)
        zseg[iseg] = ch * (ya + yb) / 2.0  # zref coord of mid-seg pt (in r-frame)
        wseg[iseg] = ch * np.sqrt(xba ** 2 + yba ** 2)

        if ks >= 0:  # id associated sect number
            thseg[iseg] = np.arctan(yba / xba)  # thseg +ve in new y-z ref frame
        else:
            thseg[iseg] = -np.pi / 2.0

    sthseg = np.sin(thseg)
    cthseg = np.cos(thseg)
    s2thseg = np.sin(2.0 * thseg)
    c2thseg = np.cos(2.0 * thseg)

    return isur, idsect, yseg, zseg, wseg, sthseg, cthseg, s2thseg, c2thseg


def tw_rate(sloc, tw_aero):
    th_prime = np.zeros(sloc.shape)  # vector of twist rates

    f0 = tw_aero[1:-1]
    f1 = tw_aero[0:-2]
    f2 = tw_aero[2:]
    h1 = sloc[1:-1] - sloc[0:-2]
    h2 = sloc[2:] - sloc[1:-1]
    th_prime[1:-1] = (h1*(f2-f0) + h2*(f0-f1))/(2.*h1*h2)

    #for i in range(1, naf-1):
    #    f0 = tw_aero[i]
    #    f1 = tw_aero[i-1]
    #    f2 = tw_aero[i+1]
    #    h1 = sloc[i] - sloc[i-1]
    #    h2 = sloc[i+1] - sloc[i]
    #    th_prime[i] = (h1*(f2-f0) + h2*(f0-f1))/(2.*h1*h2)

    th_prime[0] = (tw_aero[1]-tw_aero[0])/(sloc[1]-sloc[0])
    th_prime[-1] = (tw_aero[-1]-tw_aero[-2])/(sloc[-1]-sloc[-2])

    return th_prime


def q_bars(thp, q11, q22, q12, q66):
    ct = np.cos(thp)
    st = np.sin(thp)

    c2t = ct**2
    c3t = c2t * ct
    c4t = c3t * ct
    s2t = st**2
    s3t = s2t * st
    s4t = s3t * st
    s2thsq = 4.0 * s2t * c2t

    k11 = q11
    k22 = q22
    k12 = q12
    k66 = q66
    kmm = k11 - k12 - 2.0 * k66
    kmp = k12 - k22 + 2.0 * k66

    qbar11 = k11 * c4t + 0.5 * (k12 + 2.0 * k66) * s2thsq + k22 * s4t
    qbar22 = k11 * s4t + 0.5 * (k12 + 2.0 * k66) * s2thsq + k22 * c4t
    qbar12 = 0.25 * (k11 + k22 - 4.0 * k66) * s2thsq + k12 * (s4t + c4t)
    qbar16 = kmm * st * c3t + kmp * s3t * ct
    qbar26 = kmm * s3t * ct + kmp * st * c3t
    qbar66 = 0.25 * (kmm + k22 - k12) * s2thsq + k66 * (s4t + c4t)

    return qbar11, qbar22, qbar12, qbar16, qbar26, qbar66


def q_tildas(qbar11, qbar22, qbar12, qbar16, qbar26, qbar66):
    n = qbar11.size if isinstance(qbar11, type(np.array([]))) else 1        
    qtil = np.zeros((2, 2, n))

    qtil[0, 0, :] = qbar11 - qbar12**2 / qbar22
    qtil[0, 1, :] = qbar16 - qbar12 * qbar26 / qbar22
    qtil[1, 1, :] = qbar66 - qbar26**2 / qbar22
    
    return qtil
