!   PreComp v1.0.0a

!   Changes:
!   v2.0    07/26/2013  S. Andrew Ning      complete reorganization to allow calling from Python

!
!   This code computes structural properties of a composite blade.

!   Given blade airfoil geometry, twist distribution, composite plies layup,
!   and material properties, PreComp computes cross-sectional stiffness
!   (flap bending, lag bending, torsion, axial, coupled bending-twist,
!   axial-twist, axial-bending), mass per unit length , cross-sectional area, flap inertia,
!   lag inertia, elastic-axis offset, tension center offset, and c.m. offset)
!   at user-specified span locations.

!     Developed at NWTC/NREL (by Gunjit Bir, 1617, Cole Blvd., Golden, Colorado.
!   phone: 303-384-6953, fax: 303-384-6901)
!
!   NOTE: This code is the property of NREL.  The code, or any part of it, may
!   not be transferred, altered or copied, directly or indirecly, without approval
!   from NWTC/NREL.

!   This code is still in a developmental stage and, therefore, no guarantee
!   is given on its accuracy and proper working. Also, the code is not fully
!   commented yet.
!.................................................................................



module precomp

    implicit none

    integer, parameter :: dbp = kind(0.d0)
    real(dbp), parameter   :: pi = 3.14159265358979323846_dbp
    real(dbp), parameter   :: r2d = 57.29577951308232_dbp
    real(dbp), parameter   :: eps = 1.0d-10

contains

    subroutine properties(chord, tw_aero_d, tw_prime_d, le_loc, &
        xnode, ynode, e1, e2, g12, anu12, density, &
        xsec_nodeU, n_laminaU, n_pliesU, t_lamU, tht_lamU, mat_lamU, &
        xsec_nodeL, n_laminaL, n_pliesL, t_lamL, tht_lamL, mat_lamL, &
        nweb, loc_web, n_laminaW, n_pliesW, t_lamW, tht_lamW, mat_lamW, &
        eifbar, eilbar, gjbar, eabar, eiflbar, &
        sfbar, slbar, sftbar, sltbar, satbar, &
        z_sc, y_sc, ztc_ref, ytc_ref, mass, area, iflap_eta, &
        ilag_zeta, tw_iner, zcm_ref, ycm_ref, &
        n_af_nodes, n_materials, n_sctU, n_sctL, nwebin, &
        n_laminaTotalU, n_laminaTotalL, n_laminaTotalW)


        implicit none
        integer, parameter :: dbp = kind(0.d0)


        ! ----- inputs ------
        ! geometry
        real(dbp), intent(in) :: chord        ! section chord length (m)
        real(dbp), intent(in) :: tw_aero_d    ! section twist angle (deg)
        real(dbp), intent(in) :: tw_prime_d   ! derivative of section twist angle w.r.t. span location (deg/m)
        real(dbp), intent(in) :: le_loc       ! leading edge location relative to reference axis (normalized by chord)

        ! airfoil
        real(dbp), intent(in), dimension(n_af_nodes)  :: xnode, ynode  ! x, y airfoil coordinates starting at leading edge traversing upper surface and back around lower surface

        ! material properties
        real(dbp), intent(in), dimension(n_materials) :: e1, e2, g12, anu12, density  ! material properties: E1, E2, G12, Nu12, density

        ! laminates upper
        real(dbp), intent(in), dimension(n_sctU+1)        :: xsec_nodeU  ! normalized chord location of sector boundaries
        integer, intent(in), dimension(n_sctU)          :: n_laminaU  ! number of lamina in each sector
        integer, intent(in), dimension(n_laminaTotalU)  :: n_pliesU, mat_lamU  ! number of plies, material id for the lamina
        real(dbp), intent(in), dimension(n_laminaTotalU)  :: t_lamU, tht_lamU  ! thickness (m) and orientation (deg) for the lamina

        ! laminates lower
        real(dbp), intent(in), dimension(n_sctL+1)        :: xsec_nodeL  ! all the same quantities on the lower surface
        integer, intent(in), dimension(n_sctL)          :: n_laminaL
        integer, intent(in), dimension(n_laminaTotalL)  :: n_pliesL, mat_lamL
        real(dbp), intent(in), dimension(n_laminaTotalL)  :: t_lamL, tht_lamL

        ! laminates web
        integer :: nweb  ! required to specify this b.c. of a bug in f2py
        real(dbp), intent(in), dimension(nwebin) :: loc_web  ! same quantities for the webs
        integer, intent(in), dimension(nwebin) :: n_laminaW
        integer, intent(in), dimension(n_laminaTotalW) :: n_pliesW, mat_lamW
        real(dbp), intent(in), dimension(n_laminaTotalW) :: t_lamW, tht_lamW
        ! -------------

        ! implicitly assigned inputs (length of arrays)
        integer, intent(in) :: n_af_nodes, n_materials  ! number of airfoil nodes, materials
        integer, intent(in) :: n_sctU, n_sctL, nwebin  ! number of sectors on upper/lower, number of webs
        integer, intent(in) :: n_laminaTotalU, n_laminaTotalL, n_laminaTotalW ! total number of lamina on upper/lower/webs


        ! outputs
        real(dbp), intent(out) :: eifbar      ! EI_flap, Section flap bending stiffness about the YE axis (Nm2)
        real(dbp), intent(out) :: eilbar      ! EI_lag, Section lag (edgewise) bending stiffness about the XE axis (Nm2)
        real(dbp), intent(out) :: gjbar       ! GJ, Section torsion stiffness (Nm2)
        real(dbp), intent(out) :: eabar       ! EA, Section axial stiffness (N)
        real(dbp), intent(out) :: eiflbar     ! S_f, Coupled flap-lag stiffness with respect to the XE-YE frame (Nm2)
        real(dbp), intent(out) :: sfbar       ! S_airfoil, Coupled axial-flap stiffness with respect to the XE-YE frame (Nm)
        real(dbp), intent(out) :: slbar       ! S_al, Coupled axial-lag stiffness with respect to the XE-YE frame (Nm.)
        real(dbp), intent(out) :: sftbar      ! S_ft, Coupled flap-torsion stiffness with respect to the XE-YE frame (Nm2)
        real(dbp), intent(out) :: sltbar      ! S_lt, Coupled lag-torsion stiffness with respect to the XE-YE frame (Nm2)
        real(dbp), intent(out) :: satbar      ! S_at, Coupled axial-torsion stiffness (Nm)
        real(dbp), intent(out) :: z_sc        ! X_sc, X-coordinate of the shear-center offset with respect to the XR-YR axes (m)
        real(dbp), intent(out) :: y_sc        ! Y_sc, Chordwise offset of the section shear-center with respect to the reference frame, XR-YR (m)
        real(dbp), intent(out) :: ztc_ref     ! X_tc, X-coordinate of the tension-center offset with respect to the XR-YR axes (m)
        real(dbp), intent(out) :: ytc_ref     ! Y_tc, Chordwise offset of the section tension-center with respect to the XR-YR axes (m)
        real(dbp), intent(out) :: mass        ! Mass, Section mass per unit length (Kg/m)
        real(dbp), intent(out) :: area        ! Area, Cross-Sectional area (m)
        real(dbp), intent(out) :: iflap_eta   ! Flap_iner, Section flap inertia about the YG axis per unit length (Kg-m)
        real(dbp), intent(out) :: ilag_zeta   ! Lag_iner, Section lag inertia about the XG axis per unit length (Kg-m)
        real(dbp), intent(out) :: tw_iner     ! Tw_iner, Orientation of the section principal inertia axes with respect the blade reference plane, θ (deg)
        real(dbp), intent(out) :: zcm_ref     ! X_cm, X-coordinate of the center-of-mass offset with respect to the XR-YR axes (m)
        real(dbp), intent(out) :: ycm_ref     ! Y_cm, Chordwise offset of the section center of mass with respect to the XR-YR axes (m)

        ! local
        real(dbp) :: tw_aero, tw_prime, tphip

        integer, dimension(:, :), allocatable :: n_laminas
        real(dbp), dimension(:, :, :), allocatable :: tht_lam, tlam
        integer, dimension(:, :, :), allocatable :: mat_id

        integer :: max_sectors, max_laminates
        integer :: allocateStatus

        real(dbp), dimension(nweb, 6) :: tht_wlam, twlam
        integer, dimension(nweb, 6) :: wmat_id
        integer, dimension(nweb) :: n_weblams

        integer, dimension(1) :: location

        integer :: webs_exist, tenode_u, tenode_l

        real(dbp) :: ieta1, izeta1, iepz, iemz, ipp, &
            iqq, ipq, iflap_sc, ilag_sc, ifl_sc, iflap_cm, ilag_cm, ifl_cm, &
            m_inertia, r_inertia

        real(dbp) :: x, xl, xr, y, yl, yr

        real(dbp) :: sths, s2ths, cths, c2ths, em_stiff, er_stiff, y_tc, z_tc, &
            sigm2, ycm_sc, zcm_sc, sigma, tbar, q11t, q11yt, q11zt, dtbar, &
            q2bar, zbart, ybart, tbart, q11ysqt, q11zsqt, q11yzt, rhot, rhoyt, &
            rhozt, rhoysqt, rhozsqt, rhoyzt, pflap_stff, plag_stff, &
            q11ya, q11za, ap, bp, cp, dp, ep, q11ysqa, q11zsqa, q11yza, &
            rhoya, rhoza, rhoysqa, rhozsqa, rhoyza, q11yt_u, q11zt_u, &
            q11yt_l, q11zt_l, qtil12t, qtil11t, qtil22t, rot, t, the_pa, th_pa, &
            wdq2bar, w, xu1, xu2, xl1, xl2, yu1, yu2, yl1, yl2, y0, &
            y0sq, ysg, zsg, z0, z0sq, ynd, str_tw

        integer :: idsec, ilam, is, iseg, iweb, nlam, ncounter, ndl2, &
            ndu2, nsects, wreq, id_form

        integer :: i, j, k


        ! ---- Embed ----
        real(dbp), dimension(n_af_nodes+40) :: xnode_u, xnode_l, ynode_u, ynode_l
        integer :: newnode, nodes_u, nodes_l
        ! --------

        ! -- seg info ---
        real(dbp), dimension(nweb) :: weby_u, weby_l
        real(dbp), dimension(:, :), allocatable :: xsec_node
        real(dbp), dimension(n_af_nodes+40) :: yseg, zseg, wseg, sthseg, &
            cthseg, s2thseg, c2thseg

        integer :: nseg, nseg_l, nseg_u, nseg_p, ndl1, ndu1
        integer, dimension(2) :: n_scts
        integer, dimension(n_af_nodes+40) :: isur, idsect
        ! -----------

        ! --- QBars ----
        real(dbp) :: rho_m, thp, qbar11, qbar22, qbar12, qbar16, qbar26, qbar66
        real(dbp), dimension(n_materials) :: q11, q22, q12, q66, anud  ! composite matrices
        real(dbp), dimension(2, 2) :: qtil
        integer :: mat
        ! -------------

        ! allocate and initialize

        max_sectors = max(n_sctU, n_sctL, nweb)
        max_laminates = max(maxval(n_laminaU), maxval(n_laminaL), maxval(n_laminaW))

        ALLOCATE(n_laminas(2, max_sectors), STAT = allocateStatus)
           IF (allocateStatus /= 0) STOP "*** n_laminas not enough memory ***"
        ALLOCATE(tht_lam(2, max_sectors, max_laminates), STAT = allocateStatus)
           IF (allocateStatus /= 0) STOP "*** tht_lam not enough memory ***"
        ALLOCATE(tlam(2, max_sectors, max_laminates), STAT = allocateStatus)
           IF (allocateStatus /= 0) STOP "*** tlam not enough memory ***"
        ALLOCATE(mat_id(2, max_sectors, max_laminates), STAT = allocateStatus)
           IF (allocateStatus /= 0) STOP "*** mat_id not enough memory ***"
        ALLOCATE(xsec_node(2, max_sectors+1), STAT = allocateStatus)
           IF (allocateStatus /= 0) STOP "*** xsec_node not enough memory ***"

        xsec_node = 0.0_dbp
        xnode_u = 0.0_dbp
        xnode_l = 0.0_dbp
        ynode_u = 0.0_dbp
        ynode_l = 0.0_dbp
        yseg = 0.0_dbp
        zseg = 0.0_dbp
        wseg = 0.0_dbp
        sthseg = 0.0_dbp
        cthseg = 0.0_dbp
        s2thseg = 0.0_dbp
        c2thseg = 0.0_dbp
        isur = 0
        idsect = 0


        ! convert twist angle to radians
        tw_aero = tw_aero_d / r2d
        tw_prime = tw_prime_d / r2d

        ! webs?

        webs_exist = 1

        !     check number of webs
        if (nweb .eq. 0) then
!             write(*,*) ' ** no webs in this blade **'
            webs_exist = 0
        end if


        ! ---- checks --------------
        !  check leading edge location
        if (le_loc .lt. 0.) then
            write(*,*) ' WARNING** leading edge aft of reference axis **'
        end if


        ! check materials
        do i = 1, n_materials

            if (anu12(i) .gt. sqrt(e1(i)/e2(i))) then
                write(*,*) '**ERROR** material', i, 'properties not consistent'
            end if

        end do

        ! check airfoil nodes
        if (n_af_nodes .le. 2) then
            write(*,*)' ERROR** min 3 nodes reqd to define airfoil geom'
            stop
        end if


        !   check if the first airfoil node is a leading-edge node and at (0,0)
        location = minloc(xnode)
        if (location(1) .ne. 1) then
            write(*,*) ' ERROR** the first airfoil node not a leading node'
            stop
        endif

        if (abs(xnode(1)) .gt. eps .or. abs(ynode(1)) .gt. eps) then
            write(*,*) ' ERROR** leading-edge node not located at (0,0)'
            stop
        endif

        !   identify trailing-edge end nodes on upper and lower surfaces
        location = maxloc(xnode)
        if(abs(xnode(location(1))) .gt. 1.) then
            write(*,*) ' ERROR** trailing-edge node exceeds chord boundary'
            stop
        endif



        ! ----------------


        !   get th_prime and phi_prime
        !call tw_rate(naf, sloc, tw_aero, th_prime)

        !do i = 1, naf
        !    phi_prime(i) = 0.  ! later: get it from aeroelastic code
        !    tphip(i) = th_prime(i) + 0.5*phi_prime(i)
        !end do

        tphip = tw_prime


        ! material properties
        anud = 1.0_dbp - anu12*anu12*e2/e1
        q11 = e1 / anud
        q22 = e2 / anud
        q12 = anu12*e2 / anud
        q66 = g12



        ! begin blade sections loop sec-sec-sec-sec-sec-sec-sec-sec-sec-sec--------


        ! ----------- airfoil data -------------------

        !   identify trailing-edge end nodes on upper and lower surfaces
        location = maxloc(xnode)
        tenode_u = location(1)

        do i = location(1), n_af_nodes
            if (abs(xnode(i)-xnode(location(1))).lt. eps) then
                ncounter = i - location(1)
            end if
        end do

        tenode_l = tenode_u + ncounter


        !   renumber airfoil nodes
        !   (modify later using equivalence or pointers)
        nodes_u = tenode_u
        nodes_l = n_af_nodes - tenode_l + 2


        do i = 1, nodes_u
            xnode_u(i) = xnode(i)
            ynode_u(i) = ynode(i)
        end do

        xnode_l(1) = xnode(1)
        ynode_l(1) = ynode(1)

        do i = 2, tenode_l
            xnode_l(i) = xnode(n_af_nodes+2-i)
            ynode_l(i) = ynode(n_af_nodes+2-i)
        end do

        ! ----------------------------------------------


        ! ------ more checks -------------
        !   ensure surfaces are single-valued functions

        do i = 2, nodes_u
            if ((xnode_u(i) - xnode_u(i-1)) .le. eps ) then
                write(*,*) ' ERROR** upper surface not single-valued'
                stop
            endif
        end do

        do i = 2, nodes_l
            if ((xnode_l(i) - xnode_l(i-1)) .le. eps ) then
                write(*,*) ' ERROR** lower surface not single-valued'
                stop
            endif
        end do

        !   check clockwise node numbering at the le

        if (ynode_u(2)/xnode_u(2) .le. ynode_l(2)/xnode_l(2)) then
            write(*,*) ' ERROR** airfoil node numbering not clockwise'
            stop
        endif

        !   check for single-connectivity of the airfoil shape
        !   (modify later using binary search)

        do j = 2, nodes_l - 1   ! loop over lower-surf nodes
            x = xnode_l(j)

            do i = 1, nodes_u - 1   ! loop over upper-surf nodes

                xl = xnode_u(i)
                xr = xnode_u(i+1)

                if(x .ge. xl .and. x .le. xr) then
                    yl = ynode_u(i)
                    yr = ynode_u(i+1)
                    y = yl + (yr-yl)*(x-xl)/(xr-xl)

                    if(ynode_l(j) .ge. y) then
                        write(*,*) ' ERROR** airfoil shape self-crossing'
                        stop
                    endif
                endif

            end do    ! end loop over upper-surf nodes
        end do   ! end loop over lower-surf nodes

        ! ---------- end checks ---------------------


        ! -------------- webs ------------------


        !   embed airfoil nodes at web-to-airfoil intersections

        if (webs_exist .eq. 1) then
            do i = 1, nweb
                call embed_us(loc_web(i), ynd, nodes_u, newnode, &
                    xnode_u, ynode_u)
                weby_u(i) = ynd

                call embed_ls(loc_web(i), ynd, nodes_l, newnode, &
                    xnode_l, ynode_l)
                weby_l(i) = ynd
            end do
        end if

        ! ----------------------------------------------


        ! ------ internal srtucture data ------------
        n_scts(1) = n_sctU
        n_scts(2) = n_sctL
        xsec_node(1, :) = xsec_nodeU
        xsec_node(2, :) = xsec_nodeL

        ! unpack data
        k = 1
        do i = 1, n_sctU
            n_laminas(1, i) = n_laminaU(i)

            do j = 1, n_laminaU(i)
                tlam(1, i, j) = n_pliesU(k) * t_lamU(k)
                tht_lam(1, i, j) = tht_lamU(k) / r2d
                mat_id(1, i, j) = mat_lamU(k)

                k = k + 1
            end do
        end do

        k = 1
        do i = 1, n_sctL
            n_laminas(2, i) = n_laminaL(i)

            do j = 1, n_laminaL(i)
                tlam(2, i, j) = n_pliesL(k) * t_lamL(k)
                tht_lam(2, i, j) = tht_lamL(k) / r2d
                mat_id(2, i, j) = mat_lamL(k)

                k = k + 1
            end do
        end do

        k = 1
        do i = 1, nweb
            n_weblams(i) = n_laminaW(i)

            do j = 1, n_laminaW(i)
                twlam(i, j) = n_pliesW(k) * t_lamW(k)
                tht_wlam(i, j) = tht_lamW(k) / r2d
                wmat_id(i, j) = mat_lamW(k)

                k = k + 1
            end do
        end do




        do is = 1, 2  ! begin loop for blade surfaces

            nsects = n_scts(is)

            if (nsects .le. 0) then
                write(*,*) ' ERROR** no of sectors not positive'
                stop
            endif

            if (xsec_node(is,1) .lt. 0.) then
                write(*,*) ' ERROR** sector node x-location not positive'
            endif

            if (is .eq. 1) then
                xu1 = xsec_node(is,1)
                xu2 = xsec_node(is,nsects+1)
                if (xu2 .gt. xnode_u(nodes_u)) then
                   write(*,*) &
                        ' ERROR** upper-surf last sector node out of bounds', xu2, xnode_u(nodes_u)
                    stop
                endif
            else
                xl1 = xsec_node(is,1)
                xl2 = xsec_node(is,nsects+1)
                if (xl2 .gt. xnode_l(nodes_l)) then
                   write(*,*) &
                        ' ERROR** lower-surf last sector node out of bounds', xl2, xnode_l(nodes_l)
                    stop
                endif
            endif

            do i = 1, nsects
                if (xsec_node(is,i+1) .le. xsec_node(is,i)) then
                    write(*,*) &
                        ' ERROR** sector nodal x-locations not in ascending order'
                    stop
                endif
            end do


    !       embed airfoil nodes representing sectors bounds

            do i = 1, nsects+1

              if(is .eq. 1) then

                call embed_us(xsec_node(is,i), ynd, nodes_u, &
                    newnode, xnode_u, ynode_u)

                if(i .eq. 1) then
                    yu1 = ynd
                    ndu1 = newnode
                endif
                if(i .eq. nsects+1) then
                    yu2 = ynd
                    ndu2 = newnode
                endif

              endif

              if(is .eq. 2) then

                call embed_ls(xsec_node(is,i), ynd, nodes_l, &
                    newnode, xnode_l, ynode_l)

                if(i .eq. 1) then
                    yl1 = ynd
                    ndl1 = newnode
                endif
                if(i .eq. nsects+1) then
                    yl2 = ynd
                    ndl2 = newnode
                endif

              endif

            enddo

        end do      ! end blade surfaces loop


        !.... check for le and te non-closures and issue warning ....

        if (abs(xu1-xl1) .gt. eps) then

            write(*,*) ' WARNING** the leading edge may be open; check closure'

        else

            if ((yu1-yl1) .gt. eps) then
                wreq = 1

                if (webs_exist .ne. 0) then
                    if (abs(xu1-loc_web(1)) .lt. eps) then
                        wreq = 0
                    endif
                endif


                if (wreq .eq. 1) then
                    write(*,*) ' WARNING** open leading edge; check web requirement'
                endif

            endif

        endif

    !
        if (abs(xu2-xl2) .gt. eps) then

            write(*,*) ' WARNING** the trailing edge may be open; check closure'

        else

            if ((yu2-yl2) .gt. eps) then
                wreq = 1

                if (webs_exist .ne. 0) then
                    if (abs(xu2-loc_web(nweb)) .lt. eps) then
                        wreq = 0
                    endif
                endif


                if (wreq .eq. 1) then
                    write(*,*) ' WARNING** open trailing edge; check web requirement'
                endif

            endif

        endif
        !................

        if (webs_exist .eq. 1) then

            if(loc_web(1) .lt. xu1 .or. loc_web(1) .lt. xl1) then
                write(*,*) 'ERROR** first web out of sectors-bounded airfoil'
                !stop
            endif

            if(loc_web(nweb) .gt. xu2 .or. loc_web(nweb) .gt. xl2) then
                write(*,*) ' ERROR** last web out of sectors-bounded airfoil'
                !stop
            endif

        endif


        ! ------------- Done Processing Inputs ----------------------


        ! ----------- Start Computations ------------------


        !   identify segments groupings and get segs attributes
        nseg_u = ndu2 - ndu1
        nseg_l = ndl2 - ndl1
        nseg_p = nseg_u + nseg_l    ! no of peripheral segments

        if(webs_exist .eq. 1) then
            nseg = nseg_p + nweb  ! total no of segments (webs in section)
        else
            nseg = nseg_p      ! total no of segments (no webs in section)
        endif

        call seg_info(chord, le_loc, nseg, nseg_u, nseg_p, xnode_u, ynode_u, &
            xnode_l, ynode_l, ndl1, ndu1, loc_web, weby_u, weby_l, n_scts, &
            max_sectors+1, xsec_node, &
            isur, idsect, yseg, zseg, wseg, sthseg, cthseg, s2thseg, c2thseg)

    !------------------------------------------

        !   initialize for section (sc)
        sigma = 0.0_dbp
        eabar = 0.0_dbp
        q11ya = 0.0_dbp
        q11za = 0.0_dbp

        !   segments loop for sc


        do iseg = 1, nseg_p !begin paeripheral segments loop (sc)

        !     retrieve seg attributes
            is = isur(iseg)
            idsec = idsect(iseg)
            ysg = yseg(iseg)
            zsg = zseg(iseg)
            w = wseg(iseg)
            sths = sthseg(iseg)
            cths = cthseg(iseg)
    !       s2ths = s2thseg(iseg)
    !       c2ths = c2thseg(iseg)  ! commented out


            nlam = n_laminas(is,idsec)    ! for sector seg

            !     initialization for seg (sc)
            tbar = 0.0_dbp
            q11t = 0.0_dbp
            q11yt_u = 0.0_dbp
            q11zt_u = 0.0_dbp
            q11yt_l = 0.0_dbp
            q11zt_l = 0.0_dbp

            do ilam = 1, nlam !laminas loop (sc)

                t = tlam(is,idsec,ilam)          !thickness
                thp = tht_lam(is,idsec,ilam)  ! ply angle
                mat = mat_id(is,idsec,ilam)    !material

                tbar = tbar + t/2.0_dbp
                y0 = ysg - ((-1.0_dbp)**(is))*tbar*sths
                z0 = zsg + ((-1.0_dbp)**(is))*tbar*cths


                ! obtain qtil for specified mat
                call q_bars(mat, thp, density, q11, q22, q12, q66, &
                    qbar11, qbar22, qbar12, qbar16, qbar26, qbar66, rho_m)
                call q_tildas(qbar11, qbar22, qbar12, qbar16, qbar26, qbar66, &
                    mat, qtil)

                ! add seg-laminas contributions (sc)
                qtil11t = qtil(1,1)*t
                q11t = q11t + qtil11t
                if(iseg .le. nseg_u) then
                    q11yt_u = q11yt_u + qtil11t*y0
                    q11zt_u = q11zt_u + qtil11t*z0
                else
                    q11yt_l = q11yt_l + qtil11t*y0
                    q11zt_l = q11zt_l + qtil11t*z0
                endif

                tbar = tbar + t/2.0_dbp

            enddo         ! end laminas loop

            ! add seg contributions (sc)
            sigma = sigma + w*abs(zsg + ((-1.0_dbp)**is)*0.5_dbp*tbar*cths)*cths
            eabar = eabar + q11t*w
            q11ya = q11ya + (q11yt_u+q11yt_l)*w
            q11za = q11za + (q11zt_u+q11zt_l)*w


        enddo           !end af_periph segment loop (sc)

        ! get section sc
        y_sc = q11ya/eabar     !wrt r-frame
        z_sc = q11za/eabar     !wrt r-frame


    !---------------- end section sc -----------

        !   initializations for section (properties)

        eabar = 0.0_dbp
        q11ya = 0.0_dbp
        q11za = 0.0_dbp
        ap = 0.0_dbp
        bp = 0.0_dbp
        cp = 0.0_dbp
        dp = 0.0_dbp
        ep = 0.0_dbp
        q11ysqa = 0.0_dbp
        q11zsqa = 0.0_dbp
        q11yza = 0.0_dbp

        mass = 0.0_dbp
        area = 0.0_dbp
        rhoya = 0.0_dbp
        rhoza = 0.0_dbp
        rhoysqa = 0.0_dbp
        rhozsqa = 0.0_dbp
        rhoyza = 0.0_dbp

        !   segments loop (for properties)

        do iseg = 1, nseg   !begin segment loop (properties)

            ! retrieve seg attributes
            is = isur(iseg)
            idsec = idsect(iseg)
            ysg = yseg(iseg)
            zsg = zseg(iseg)
            w = wseg(iseg)
            sths = sthseg(iseg)
            cths = cthseg(iseg)
            s2ths = s2thseg(iseg)
            c2ths = c2thseg(iseg)

            if(is .gt. 0) then
                nlam = n_laminas(is,idsec)  ! for sector seg
            else
                iweb = idsec
                nlam = n_weblams(iweb)      ! for web seg
            endif

            ! initialization for seg (properties)
            tbar = 0.0_dbp
            q11t = 0.0_dbp
            q11yt = 0.0_dbp
            q11zt = 0.0_dbp
            dtbar = 0.0_dbp
            q2bar = 0.0_dbp
            zbart = 0.0_dbp
            ybart = 0.0_dbp
            tbart = 0.0_dbp
            q11ysqt = 0.0_dbp
            q11zsqt = 0.0_dbp
            q11yzt = 0.0_dbp

            rhot = 0.0_dbp
            rhoyt = 0.0_dbp
            rhozt = 0.0_dbp
            rhoysqt = 0.0_dbp
            rhozsqt = 0.0_dbp
            rhoyzt = 0.0_dbp

            do ilam = 1, nlam !laminas loop (properties)

                if(is .gt. 0) then
                    t = tlam(is,idsec,ilam)          !thickness
                    thp = tht_lam(is,idsec,ilam)  ! ply angle
                    mat = mat_id(is,idsec,ilam)      ! material
                    tbar = tbar + t/2.0_dbp
                    y0 = ysg - ((-1.0_dbp)**(is))*tbar*sths - y_sc
                    z0 = zsg + ((-1.0_dbp)**(is))*tbar*cths - z_sc
                else
                    t = twlam(iweb,ilam)
                    thp = tht_wlam(iweb,ilam)
                    mat = wmat_id(iweb,ilam)
                    tbar = tbar + t/2.0_dbp
                    y0 = ysg - tbar/2.0_dbp - y_sc
                    z0 = zsg - z_sc
                endif

                y0sq = y0*y0
                z0sq = z0*z0

                ! obtain qtil and rho for specified mat
                call q_bars(mat, thp, density, q11, q22, q12, q66, &
                    qbar11, qbar22, qbar12, qbar16, qbar26, qbar66, rho_m)
                call q_tildas(qbar11, qbar22, qbar12, qbar16, qbar26, qbar66, &
                    mat, qtil)


                ieta1 = (t**2)/12.0_dbp
                izeta1 = (w**2)/12.0_dbp
                iepz = 0.5_dbp*(ieta1+izeta1)
                iemz = 0.5_dbp*(ieta1-izeta1)
                ipp = iepz + iemz*c2ths   ! check this block later
                iqq = iepz - iemz*c2ths
                ipq = iemz*s2ths

                qtil11t = qtil(1,1)*t
                rot = rho_m*t

                !add laminas contributions (properties) at current segment

                if(is .gt. 0) then ! peripheral segs contribution

                    qtil12t = qtil(1,2)*t
                    qtil22t = qtil(2,2)*t

                    q11t = q11t + qtil11t
                    q11yt = q11yt + qtil11t*y0
                    q11zt = q11zt + qtil11t*z0

                    dtbar = dtbar + qtil12t*(y0sq+z0sq)*tphip*t
                    q2bar = q2bar + qtil22t    ! later: retain only this block
                    zbart = zbart + z0*qtil12t
                    ybart = ybart + y0*qtil12t
                    tbart = tbart + qtil12t

                    q11ysqt = q11ysqt + qtil11t*(y0sq+iqq)
                    q11zsqt = q11zsqt + qtil11t*(z0sq+ipp)
                    q11yzt = q11yzt + qtil11t*(y0*z0+ipq)

                    rhot = rhot + rot
                    rhoyt = rhoyt + rot*y0
                    rhozt = rhozt + rot*z0
                    rhoysqt = rhoysqt + rot*(y0sq+iqq)
                    rhozsqt = rhozsqt + rot*(z0sq+ipp)
                    rhoyzt = rhoyzt + rot*(y0*z0+ipq)

                else            !web segs contribution

                    q11t = q11t + qtil11t
                    q11yt = q11yt + qtil11t*y0
                    q11zt = q11zt + qtil11t*z0
                    q11ysqt = q11ysqt + qtil11t*(y0sq+iqq)
                    q11zsqt = q11zsqt + qtil11t*(z0sq+ipp)
                    q11yzt = q11yzt + qtil11t*(y0*z0+ipq)

                    rhot = rhot + rot
                    rhoyt = rhoyt + rot*y0
                    rhozt = rhozt + rot*z0
                    rhoysqt = rhoysqt + rot*(y0sq+iqq)
                    rhozsqt = rhozsqt + rot*(z0sq+ipp)
                    rhoyzt = rhoyzt + rot*(y0*z0+ipq)

                endif

                tbar = tbar + t/2.0_dbp

            enddo         ! end laminas loop


            ! add seg contributions to obtain sec props about ref_parallel axes at sc
            eabar = eabar + q11t*w
            q11ya = q11ya + q11yt*w
            q11za = q11za + q11zt*w
            q11ysqa = q11ysqa + q11ysqt*w
            q11zsqa = q11zsqa + q11zsqt*w
            q11yza = q11yza + q11yzt*w

            if(is .gt. 0) then
                wdq2bar = w/q2bar
                ap = ap + wdq2bar
                bp = bp + wdq2bar*tbart
                cp = cp + wdq2bar*dtbar
                dp = dp + wdq2bar*zbart
                ep = ep + wdq2bar*ybart
            endif

            area = area + w
            mass = mass + rhot*w
            rhoya = rhoya + rhoyt*w
            rhoza = rhoza + rhozt*w
            rhoysqa = rhoysqa + rhoysqt*w
            rhozsqa = rhozsqa + rhozsqt*w
            rhoyza = rhoyza + rhoyzt*w

        enddo       !end af_periph segment loop (properties)

        !  get more section properties ! about ref_parallel axes at sc

        y_tc = q11ya/eabar
        z_tc = q11za/eabar

        sfbar = q11za
        slbar = q11ya
        eifbar = q11zsqa
        eilbar = q11ysqa
        eiflbar = q11yza

        sigm2 = sigma*2.0_dbp
        gjbar = sigm2*(sigm2+cp)/ap
        sftbar = -sigm2*dp/ap
        sltbar = -sigm2*ep/ap
        satbar = sigm2*bp/ap

        ycm_sc =   rhoya/mass !wrt sc
        zcm_sc =   rhoza/mass !wrt sc

        iflap_sc = rhozsqa !wrt sc
        ilag_sc = rhoysqa   !wrt sc
        ifl_sc = rhoyza     !wrt sc

        ! get section tc and cm

        ytc_ref =  y_tc + y_sc  !wrt the ref axes
        ztc_ref =  z_tc + z_sc  !wrt the ref axes

        ycm_ref =  ycm_sc + y_sc    !wrt the ref axes
        zcm_ref =  zcm_sc + z_sc    !wrt the ref axes

        ! moments of inertia ! about ref_parallel axes at cm

        iflap_cm = iflap_sc - mass*zcm_sc**2
        ilag_cm = ilag_sc - mass*ycm_sc**2
        ifl_cm = ifl_sc - mass*ycm_sc*zcm_sc

        ! inertia principal axes orientation and moments of inertia

        m_inertia = 0.5_dbp*(ilag_cm + iflap_cm)
        r_inertia = sqrt(0.25_dbp*((ilag_cm-iflap_cm)**2) + ifl_cm**2)

        if(iflap_cm .le. ilag_cm) then
            iflap_eta = m_inertia - r_inertia
            ilag_zeta = m_inertia + r_inertia
        else
            iflap_eta = m_inertia + r_inertia
            ilag_zeta = m_inertia - r_inertia
        endif

        if(ilag_cm .eq. iflap_cm) then
            th_pa = pi/4.0_dbp
            if(abs(ifl_cm/iflap_cm) .lt. 1.d-6) th_pa = 0.0_dbp
        else
            th_pa = 0.5_dbp*abs(atan(2.0_dbp*ifl_cm/(ilag_cm-iflap_cm)))
        endif

        if(abs(ifl_cm) .lt. eps) then
            th_pa = 0.0_dbp
        else          ! check this block later
            if(iflap_cm .ge. ilag_cm .and. ifl_cm .gt. 0.) th_pa = -th_pa
            if(iflap_cm .ge. ilag_cm .and. ifl_cm .lt. 0.) th_pa = th_pa
            if(iflap_cm .lt. ilag_cm .and. ifl_cm .gt. 0.) th_pa = th_pa
            if(iflap_cm .lt. ilag_cm .and. ifl_cm .lt. 0.) th_pa = -th_pa
        endif

        ! elastic principal axes orientation and principal bending stiffneses

        em_stiff = 0.5_dbp*(eilbar + eifbar)
        er_stiff = sqrt(0.25_dbp*((eilbar-eifbar)**2) + eiflbar**2)

        if(eifbar .le. eilbar) then
            pflap_stff = em_stiff - er_stiff
            plag_stff = em_stiff + er_stiff
        else
            pflap_stff = em_stiff + er_stiff
            plag_stff = em_stiff - er_stiff
        endif

        if(eilbar .eq. eifbar) then
            the_pa = pi/4.0_dbp
        else
            the_pa = 0.5_dbp*abs(atan(2.0_dbp*eiflbar/(eilbar-eifbar)))
        endif

        if(abs(eiflbar) .lt. eps) then
            the_pa = 0.0_dbp
        else          ! check this block later
            if(eifbar .ge. eilbar .and. eiflbar .gt. 0.) the_pa = -the_pa
            if(eifbar .ge. eilbar .and. eiflbar .lt. 0.) the_pa = the_pa
            if(eifbar .lt. eilbar .and. eiflbar .gt. 0.) the_pa = the_pa
            if(eifbar .lt. eilbar .and. eiflbar .lt. 0.) the_pa = -the_pa
        endif

    !---------------- end properties computation -----------


    ! ---------- prepare outputs --------------



        id_form = 1  ! hardwired for wt's

        if (id_form .eq. 1) then
            tw_iner = tw_aero - th_pa
            str_tw =  tw_aero - the_pa
            y_sc = -y_sc
            ytc_ref = -ytc_ref
            ycm_ref = -ycm_ref
        else         ! for h/c
            !       note: for h/c, th_aero input is +ve acc to h/c convention
            tw_iner = tw_aero + th_pa
            str_tw =  tw_aero + the_pa
        endif


        ! conversions
        eiflbar = -eiflbar
        sfbar = -sfbar
        sltbar = -sltbar
        tw_iner = tw_iner*r2d



        deallocate(n_laminas, STAT = allocateStatus)
        deallocate(tht_lam, STAT = allocateStatus)
        deallocate(tlam, STAT = allocateStatus)
        deallocate(mat_id, STAT = allocateStatus)
        deallocate(xsec_node, STAT = allocateStatus)


        return



    end subroutine properties

















    !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    subroutine embed_us(x, y, nodes_u, newnode, xnode_u, ynode_u)
        !   purpose: embed a node in the upper-surface airfoil section nodes
        !   NOTE: nodal x coordinates must be in ascending order


        implicit none
        integer, parameter :: dbp = kind(0.d0)


        real(dbp), intent(in) :: x  ! x-coordinate of node to be embedded in the u-surf

        integer, intent(inout) :: nodes_u  ! no of current nodes on the upper surface / revised no of current nodes on upper surface
        real(dbp), intent(inout), dimension(300) :: xnode_u, ynode_u

        real(dbp), intent(out) :: y  ! y-coordinate of node embedded in the u-surf
        integer, intent(out) :: newnode  ! number of the embedded node

        ! local
        real(dbp) :: xl, xr, yl, yr
        integer :: isave, i


        newnode = -1

        do i = 1, nodes_u - 1   ! loop over upper-surf nodes
            xl = xnode_u(i)
            xr = xnode_u(i+1)
            yl = ynode_u(i)

            if(abs(x-xl) .le. eps) then
                newnode = 0
                isave = i
                y = yl
                exit
            else if(x .lt. (xr-eps)) then
                yr = ynode_u(i+1)
                y = yl + (yr-yl)*(x-xl)/(xr-xl)
                newnode = i+1
                exit
            endif
        enddo       ! end loop over upper-surf nodes

        if(newnode .eq. -1) then
            if(abs(x-xnode_u(nodes_u)) .le. eps) then
                newnode = 0
                isave = nodes_u
                y =  ynode_u(nodes_u)
            else
                write(*,*) ' ERROR unknown, consult NWTC'
                !stop 1111
            endif
        endif

        if(newnode .gt. 0) then
            nodes_u = nodes_u +1

            do i = nodes_u, (newnode+1), -1
                xnode_u(i) = xnode_u(i-1)
                ynode_u(i) = ynode_u(i-1)
            enddo

            xnode_u(newnode) = x
            ynode_u(newnode) = y

        else
            newnode = isave
        endif

    end subroutine embed_us

    !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


    subroutine embed_ls(x, y, nodes_l, newnode, xnode_l, ynode_l)
        !   purpose: embed a node in the lower-surface airfoil section nodes
        !   NOTE: nodal x coordinates must be in ascending order


        implicit none
        integer, parameter :: dbp = kind(0.d0)


        real(dbp), intent(in) :: x  ! x-coordinate of node to be embedded in the u-surf

        integer, intent(inout) :: nodes_l  ! no of current nodes on the lower surface / revised no of current nodes on lower surface
        real(dbp), intent(inout), dimension(300) :: xnode_l, ynode_l

        real(dbp), intent(out) :: y  ! y-coordinate of node embedded in the u-surf
        integer, intent(out) :: newnode  ! number of the embedded node

        ! local
        real(dbp) :: xl, xr, yl, yr
        integer :: isave, i


        newnode = -1

        do i = 1, nodes_l - 1   ! loop over lower-surf nodes
            xl = xnode_l(i)
            xr = xnode_l(i+1)
            yl = ynode_l(i)

            if(abs(x-xl) .le. eps) then
                newnode = 0
                isave = i
                y = yl
                exit
            else if(x .lt. (xr-eps)) then
                yr = ynode_l(i+1)
                y = yl + (yr-yl)*(x-xl)/(xr-xl)
                newnode = i+1
                exit

            endif

        enddo       ! end loop over lower-surf nodes

        if(newnode .eq. -1) then
            if(abs(x-xnode_l(nodes_l)) .le. eps) then
                newnode = 0
                isave = nodes_l
                y =  ynode_l(nodes_l)
            else
                write(*,*) ' ERROR unknown, consult NWTC'
                !stop 1111
            endif
        endif

        if(newnode .gt. 0) then
            nodes_l = nodes_l +1

            do i = nodes_l, (newnode+1), -1
                xnode_l(i) = xnode_l(i-1)
                ynode_l(i) = ynode_l(i-1)
            enddo

            xnode_l(newnode) = x
            ynode_l(newnode) = y

        else
            newnode = isave
        endif

    end subroutine embed_ls

    !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    subroutine seg_info(ch, rle, nseg, nseg_u, nseg_p, xnode_u, ynode_u, &
        xnode_l, ynode_l, ndl1, ndu1, loc_web, weby_u, weby_l, n_scts, &
        nsecnode, xsec_node, &
        isur, idsect, yseg, zseg, wseg, sthseg, cthseg, s2thseg, c2thseg)
        !   NOTE: coord transformation from xaf-yaf to yre-zref and seg info


        implicit none
        integer, parameter :: dbp = kind(0.d0)

        ! inputs
        real(dbp), intent(in) :: ch, rle  ! chord length, loc of l.e. (non-d wrt chord)
        integer, intent(in) :: nseg, nseg_u, nseg_p  ! total number of segs, no of segs on the upper surface, no of segs for both upper and lower surfaces
        real(dbp), intent(in), dimension(300) :: xnode_u, ynode_u, xnode_l, ynode_l  ! x,y nodes on upper/lower
        integer, intent(in) :: ndl1, ndu1 ! 1st seg lhs node number lower/upper surface
        real(dbp), intent(in), dimension(:) :: loc_web, weby_u, weby_l  ! x coord of web, y coord of web upper/lower
        integer, intent(in), dimension(2) :: n_scts  ! no of sectors on 'is' surf
        integer, intent(in) :: nsecnode
        real(dbp), dimension(2, nsecnode) :: xsec_node  ! x coord of sect-i lhs on 's' surf

        ! outputs
        integer, intent(out), dimension(:) :: isur  ! surf id
        integer, intent(out), dimension(:) :: idsect  ! associated sect or web number
        real(dbp), intent(out), dimension(:) :: yseg  ! y-ref of mid-seg point
        real(dbp), intent(out), dimension(:) :: zseg  ! z-ref of mid-seg point
        real(dbp), intent(out), dimension(:) :: wseg  ! seg width
        real(dbp), intent(out), dimension(:) :: sthseg  ! sin(th_seg)
        real(dbp), intent(out), dimension(:) :: cthseg  ! cos(th_seg)
        real(dbp), intent(out), dimension(:) :: s2thseg  ! sin(2*th_seg)
        real(dbp), intent(out), dimension(:) :: c2thseg  ! cos(2*th_seg)

        ! local
        integer :: iseg, is, i, icheck, iweb, nd_a
        real(dbp) :: xa, ya, xb, yb, xba, yba, thseg


        do iseg = 1, nseg   ! seg numbering from le clockwise
            is = -1
            if(iseg .le. nseg_u) then  ! upper surface segs

                nd_a = ndu1+iseg-1
                xa = xnode_u(nd_a)
                ya = ynode_u(nd_a)
                xb = xnode_u(nd_a+1)
                yb = ynode_u(nd_a+1)
                is = 1
            else
                if(iseg .le. nseg_p) then   ! lower surface segs
                    nd_a = ndl1+iseg-nseg_u-1
                    xa = xnode_l(nd_a)        !xref of node toward le (in a/f ref frame)
                    ya = ynode_l(nd_a)        !yref of node toward le (in new ref frame)
                    xb = xnode_l(nd_a+1)      !xref of node toward te (in a/f ref frame)
                    yb = ynode_l(nd_a+1)      !yref of node toward te (in new ref frame)
                    is = 2
                endif

                if(iseg .gt. nseg_p ) then  ! web segs
                    iweb = iseg-nseg_p
                    xa = loc_web(iweb)
                    xb = xa
                    ya = weby_u(iweb)
                    yb = weby_l(iweb)
                    is = 0
                endif

            endif  ! end seg group identification


            if(is .eq. -1) then
                write(*,*) 'iseg=', iseg
                write(*,*) ' ERROR** unknown, contact NREL'
                stop
            endif

            isur(iseg) = is


            if(is .gt. 0) then !id assocaited sect number
                icheck = 0
                do i = 1, n_scts(is)
                    if(xa .gt. (xsec_node(is,i)-eps) .and. &
                        xb .lt. (xsec_node(is,i+1)+eps)) then
                        idsect(iseg) = i
                        icheck = 1
                        exit
                    endif
                enddo
            endif


            if(icheck .eq. 0) then
                write(*,*) ' ERROR** unknown, contact NREL'
                !stop 2222
            endif

            if(is .eq. 0) idsect(iseg) = iweb   !id assocaited web number

            xba = xb - xa
            yba = ya - yb
            yseg(iseg) = ch*(2.*rle-xa-xb)/2.0_dbp !yref coord of mid-seg pt (in r-frame)
            zseg(iseg) = ch*(ya+yb)/2.0_dbp    !zref coord of mid-seg pt (in r-frame)
            wseg(iseg) = ch*sqrt(xba**2 + yba**2)


            if (is .eq. 0) then
                thseg = -pi/2.0_dbp
            else
                thseg = atan(yba/xba) ! thseg +ve in new y-z ref frame
            endif

            sthseg(iseg) = sin(thseg)
            cthseg(iseg) = cos(thseg)
            s2thseg(iseg) = sin(2.0_dbp*thseg)
            c2thseg(iseg) = cos(2.0_dbp*thseg)


        enddo   ! end seg loop


    end subroutine seg_info

    !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


    subroutine tw_rate(naf, sloc, tw_aero, th_prime)


        implicit none
        integer, parameter :: dbp = kind(0.d0)

        ! inputs
        integer, intent(in) :: naf  ! no of blade stations
        real(dbp), intent(in), dimension(naf) :: sloc ! vector of station locations
        real(dbp), intent(in), dimension(naf) :: tw_aero ! vector of twist distribution

        ! outputs
        real(dbp), intent(out), dimension(naf) :: th_prime ! vector of twist rates

        ! local
        real(dbp) :: f0, f1, f2, h1, h2
        integer :: i

        do i = 2, naf-1
            f0 = tw_aero(i)
            f1 = tw_aero(i-1)
            f2 = tw_aero(i+1)
            h1 = sloc(i) - sloc(i-1)
            h2 = sloc(i+1) - sloc(i)
            th_prime(i) = (h1*(f2-f0) + h2*(f0-f1))/(2.*h1*h2)
        enddo

        th_prime(1) = (tw_aero(2)-tw_aero(1))/(sloc(2)-sloc(1))
        th_prime(naf)=(tw_aero(naf)-tw_aero(naf-1))/(sloc(naf)-sloc(naf-1))

    end subroutine tw_rate

    !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    subroutine q_bars(mat, thp, density, q11, q22, q12, q66, &
        qbar11, qbar22, qbar12, qbar16, qbar26, qbar66, rho_m)


        implicit none
        integer, parameter :: dbp = kind(0.d0)

        ! input
        integer, intent(in) :: mat  ! material id
        real(dbp), intent(in) :: thp  ! ply orientation
        real(dbp), intent(in), dimension(:) :: density, q11, q22, q12, q66

        ! outputs
        real(dbp), intent(out) :: qbar11, qbar22, qbar12, qbar16, qbar26, qbar66, rho_m

        ! local
        real(dbp) :: k11, k22, k12, k66, kmm, kmp
        real(dbp) :: ct, st, c2t, c3t, c4t, s2t, s3t, s4t, s2thsq

        ct = cos(thp)
        st = sin(thp)

        c2t = ct*ct
        c3t = c2t*ct
        c4t = c3t*ct
        s2t = st*st
        s3t = s2t*st
        s4t = s3t*st
        s2thsq = 4.0_dbp*s2t*c2t

        k11 = q11(mat)
        k22 = q22(mat)
        k12 = q12(mat)
        k66 = q66(mat)
        kmm = k11 -k12 -2.0_dbp*k66
        kmp = k12 -k22 +2.0_dbp*k66

        qbar11 = k11*c4t + 0.5_dbp*(k12+2.0_dbp*k66)*s2thsq + k22*s4t
        qbar22 = k11*s4t + 0.5_dbp*(k12+2.0_dbp*k66)*s2thsq + k22*c4t
        qbar12 = 0.25_dbp*(k11+k22-4.0_dbp*k66)*s2thsq + k12*(s4t+c4t)
        qbar16 = kmm*st*c3t + kmp*s3t*ct
        qbar26 = kmm*s3t*ct + kmp*st*c3t
        qbar66 = 0.25_dbp*(kmm+k22-k12)*s2thsq  + k66*(s4t+c4t)

        rho_m = density(mat)

    end subroutine q_bars

    !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    subroutine q_tildas(qbar11, qbar22, qbar12, qbar16, qbar26, qbar66, mat, qtil)


        implicit none
        integer, parameter :: dbp = kind(0.d0)

        real(dbp), intent(in) :: qbar11, qbar22, qbar12, qbar16, qbar26, qbar66
        integer, intent(in) :: mat
        real(dbp), intent(out), dimension(2, 2) :: qtil


        qtil(1,1) = qbar11 - qbar12*qbar12/qbar22
        if (qtil(1,1) .lt. 0.) then
            write(*,*) '**ERROR: check material no', mat, &
                'properties; these are not physically realizable.'

        end if

        qtil(1,2) = qbar16 - qbar12*qbar26/qbar22
        qtil(2,2) = qbar66 - qbar26*qbar26/qbar22


    end subroutine q_tildas

    !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

end module precomp
