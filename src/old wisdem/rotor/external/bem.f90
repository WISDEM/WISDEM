subroutine inductionFactors(r, c, Rhub, Rtip, phi, cl, cd, B, &
    Vx, Vy, useCd, hubLoss, tipLoss, wakerotation, &
    fzero, a, ap)

    implicit none

    integer, parameter :: ReKi = selected_real_kind(15, 307)

    ! in
    real(ReKi), intent(in) :: r, c, Rhub, Rtip, phi, cl, cd
    integer, intent(in) :: B
    real(ReKi), intent(in) :: Vx, Vy
    logical, intent(in) :: useCd, hubLoss, tipLoss, wakerotation
    !f2py logical, optional, intent(in) :: useCd = 1, hubLoss = 1, tipLoss = 1, wakerotation = 1

    ! out
    real(ReKi), intent(out) :: fzero, a, ap

    ! local
    real(ReKi) :: pi, sigma_p, sphi, cphi, lambda_r
    real(ReKi) :: factortip, Ftip, factorhub, Fhub
    real(ReKi) :: k, kp, cn, ct, F
    real(ReKi) :: g1, g2, g3


    ! constants
    pi = 3.1415926535897932
    sigma_p = B/2.0/pi*c/r
    sphi = sin(phi)
    cphi = cos(phi)

    ! resolve into normal and tangential forces
    if ( .not. useCd ) then
        cn = cl*cphi
        ct = cl*sphi
    else
        cn = cl*cphi + cd*sphi
        ct = cl*sphi - cd*cphi
    end if

    ! Prandtl's tip and hub loss factor
    Ftip = 1.0
    if ( tipLoss ) then
        factortip = B/2.0*(Rtip - r)/(r*abs(sphi))
        Ftip = 2.0/pi*acos(exp(-factortip))
    end if

    Fhub = 1.0
    if ( hubLoss ) then
        factorhub = B/2.0*(r - Rhub)/(Rhub*abs(sphi))
        Fhub = 2.0/pi*acos(exp(-factorhub))
    end if

    F = Ftip * Fhub

    ! bem parameters
    k = sigma_p*cn/4/F/sphi/sphi
    kp = sigma_p*ct/4/F/sphi/cphi

    ! compute axial induction factor
    if (phi > 0) then  ! momentum/empirical

        ! update axial induction factor
        if (k <= 2.0/3) then  ! momentum state
            a = k/(1+k)

        else  ! Glauert(Buhl) correction
            if ( abs(k - (25.0/18/F - 1.0)) < 1e-6) then
                k = k + 1.0e-5  ! avoid singularity
            end if

            g1 = 2*F*k - (10.0/9-F)
            g2 = 2*F*k - (4.0/3-F)*F
            g3 = 2*F*k - (25.0/9-2*F)

            a = (g1 - sqrt(g2)) / g3

        end if

    else  ! propeller brake region (a and ap not directly used but update anyway)

        if (k > 1.0) then
            a = k/(k-1)
        else
            a = 0.0  ! dummy value
        end if

    end if

    ! compute tangential induction factor
    ap = kp/(1-kp)

    if (.not. wakerotation) then
        ap = 0.0
        kp = 0.0
    end if

    ! error function
    lambda_r = Vy/Vx
    if (phi > 0) then  ! momentum/empirical
        fzero = sphi/(1-a) - cphi/lambda_r*(1-kp)
    else  ! propeller brake region
        fzero = sphi*(1-k) - cphi/lambda_r*(1-kp)
    end if

end subroutine inductionFactors




subroutine relativeWind(phi, a, ap, Vx, Vy, pitch, &
    chord, theta, rho, mu, alpha, W, Re)

    implicit none

    integer, parameter :: ReKi = selected_real_kind(15, 307)

    ! in
    real(ReKi), intent(in) :: phi, a, ap, Vx, Vy, pitch
    real(ReKi), intent(in) :: chord, theta, rho, mu

    ! out
    real(ReKi), intent(out) :: alpha, W, Re

    ! angle of attack
    alpha = phi - (theta + pitch)

    ! avoid numerical errors when angle is close to 0 or 90 deg
    ! and other induction factor is at some ridiculous value
    ! this only occurs when iterating on Reynolds number
    ! during the phi sweep where a solution has not been found yet
    if ( abs(a) > 10 ) then
        W = Vy*(1+ap)/cos(phi)
    else if ( abs(ap) > 10 ) then
        W = Vx*(1-a)/sin(phi)
    else
        W = sqrt((Vx*(1-a))**2 + (Vy*(1+ap))**2)
    end if

    Re = rho * W * chord / mu


end subroutine relativeWind



subroutine defineCurvature(n, r, precurve, presweep, precone, x_az, y_az, z_az, cone, s)

    implicit none

    integer, parameter :: ReKi = selected_real_kind(15, 307)

    ! in
    integer, intent(in) :: n
    real(ReKi), dimension(n), intent(in) :: r, precurve, presweep
    real(ReKi), intent(in) :: precone

    ! out
    real(ReKi), dimension(n), intent(out) :: x_az, y_az, z_az, cone, s

    ! local
    integer :: i


    ! coordinate in azimuthal coordinate system
    ! az_coords = DirectionVector(precurve, presweep, r).bladeToAzimuth(precone)

    x_az = -r*sin(precone) + precurve*cos(precone)
    z_az = r*cos(precone) + precurve*sin(precone)
    y_az = presweep


    ! compute total coning angle for purposes of relative velocity
    cone(1) = atan2(-(x_az(2) - x_az(1)), z_az(2) - z_az(1))
    cone(2:n-1) = 0.5*(atan2(-(x_az(2:n-1) - x_az(1:n-2)), z_az(2:n-1) - z_az(1:n-2)) &
                       + atan2(-(x_az(3:n) - x_az(2:n-1)), z_az(3:n) - z_az(2:n-1)))
    cone(n) = atan2(-(x_az(n) - x_az(n-1)), z_az(n) - z_az(n-1))


    ! total path length of blade
    s(1) = 0.0
    do i = 2, n
        s(i) = s(i-1) + sqrt((precurve(i) - precurve(i-1))**2 + &
            (presweep(i) - presweep(i-1))**2 + (r(i) - r(i-1))**2)
    end do


end subroutine defineCurvature




subroutine windComponents(n, r, precurve, presweep, precone, yaw, tilt, azimuth, &
    Uinf, OmegaRPM, hubHt, shearExp, Vx, Vy)

    implicit none

    integer, parameter :: ReKi = selected_real_kind(15, 307)

    ! in
    integer, intent(in) :: n
    real(ReKi), dimension(n), intent(in) :: r, precurve, presweep
    real(ReKi), intent(in) :: precone, yaw, tilt, azimuth, Uinf, OmegaRPM, hubHt, shearExp

    ! out
    real(ReKi), dimension(n), intent(out) :: Vx, Vy

    ! local
    real(ReKi) :: sy, cy, st, ct, sa, ca, pi, Omega
    real(ReKi), dimension(n) :: cone, sc, cc, x_az, y_az, z_az, sint
    real(ReKi), dimension(n) :: heightFromHub, V, Vwind_x, Vwind_y, Vrot_x, Vrot_y


    ! rename
    sy = sin(yaw)
    cy = cos(yaw)
    st = sin(tilt)
    ct = cos(tilt)
    sa = sin(azimuth)
    ca = cos(azimuth)
    pi = 3.1415926535897932
    Omega = OmegaRPM * pi/30.0


    call defineCurvature(n, r, precurve, presweep, precone, x_az, y_az, z_az, cone, sint)
    sc = sin(cone)
    cc = cos(cone)


    ! get section heights in wind-aligned coordinate system
    ! heightFromHub = az_coords.azimuthToHub(azimuth).hubToYaw(tilt).z

    heightFromHub = (y_az*sa + z_az*ca)*ct - x_az*st


    ! velocity with shear

    V = Uinf*(1 + heightFromHub/hubHt)**shearExp


    ! transform wind to blade c.s.
    ! Vwind = DirectionVector(V, 0*V, 0*V).windToYaw(yaw).yawToHub(tilt).hubToAzimuth(azimuth).azimuthToBlade(cone)

    Vwind_x = V * ((cy*st*ca + sy*sa)*sc + cy*ct*cc)
    Vwind_y = V * (cy*st*sa - sy*ca)


    ! wind from rotation to blade c.s.
    ! OmegaV = DirectionVector(Omega, 0.0, 0.0)
    ! Vrot = -OmegaV.cross(az_coords)  # negative sign because relative wind opposite to rotation
    ! Vrot = Vrot.azimuthToBlade(cone)

    Vrot_x = -Omega*y_az*sc
    Vrot_y = Omega*z_az


    ! total velocity
    Vx = Vwind_x + Vrot_x
    Vy = Vwind_y + Vrot_y



end subroutine windComponents







subroutine thrustTorque(n, Np, Tp, r, precurve, presweep, precone, &
    Rhub, Rtip, precurveTip, presweepTip, T, Q)

    implicit none

    integer, parameter :: ReKi = selected_real_kind(15, 307)

    ! in
    integer, intent(in) :: n
    real(ReKi), dimension(n), intent(in) :: Np, Tp, r, precurve, presweep
    real(ReKi), intent(in) :: precone, Rhub, Rtip, precurveTip, presweepTip

    ! out
    real(ReKi), intent(out) :: T, Q

    ! local
    real(ReKi) :: ds
    real(ReKi), dimension(n+2) :: rfull, curvefull, sweepfull, Npfull, Tpfull
    real(ReKi), dimension(n+2) :: thrust, torque, x_az, y_az, z_az, cone, s
    integer :: i


    ! add hub/tip for complete integration.  loads go to zero at hub/tip.
    rfull(1) = Rhub
    rfull(2:n+1) = r
    rfull(n+2) = Rtip

    curvefull(1) = 0.0
    curvefull(2:n+1) = precurve
    curvefull(n+2) = precurveTip

    sweepfull(1) = 0
    sweepfull(2:n+1) = presweep
    sweepfull(n+2) = presweepTip

    Npfull(1) = 0.0
    Npfull(2:n+1) = Np
    Npfull(n+2) = 0.0

    Tpfull(1) = 0.0
    Tpfull(2:n+1) = Tp
    Tpfull(n+2) = 0.0


    ! get z_az and total cone angle
    call defineCurvature(n+2, rfull, curvefull, sweepfull, precone, x_az, y_az, z_az, cone, s)


    ! integrate Thrust and Torque (trapezoidal)
    thrust = Npfull*cos(cone)
    torque = Tpfull*z_az

    T = 0.0
    do i = 1, n+1
        ds = s(i+1) - s(i)
        T = T + 0.5*(thrust(i) + thrust(i+1))*ds
        Q = Q + 0.5*(torque(i) + torque(i+1))*ds
    end do


end subroutine thrustTorque





!        Generated by TAPENADE     (INRIA, Tropics team)
!  Tapenade 3.7 (r4888) - 28 May 2013 10:47
!
!  Differentiation of inductionfactors in reverse (adjoint) mode:
!   gradient     of useful results: ap fzero a
!   with respect to varying inputs: r rtip ap rhub fzero phi cd
!                cl vx vy a c
!   RW status of diff variables: r:out rtip:out ap:in-zero rhub:out
!                fzero:in-zero phi:out cd:out cl:out vx:out vy:out
!                a:in-zero c:out
SUBROUTINE INDUCTIONFACTORS_B(r, rb, c, cb, rhub, rhubb, rtip, rtipb, &
&  phi, phib, cl, clb, cd, cdb, b, vx, vxb, vy, vyb, usecd, hubloss, &
&  tiploss, wakerotation, fzero, fzerob, a, ab, ap, apb)
  IMPLICIT NONE
  INTEGER, PARAMETER :: reki=SELECTED_REAL_KIND(15, 307)
! in
  REAL(reki), INTENT(IN) :: r, c, rhub, rtip, phi, cl, cd
  REAL(reki), intent(out) :: rb, cb, rhubb, rtipb, phib, clb, cdb
  INTEGER, INTENT(IN) :: b
  REAL(reki), INTENT(IN) :: vx, vy
  REAL(reki), intent(out) :: vxb, vyb
  LOGICAL, INTENT(IN) :: usecd, hubloss, tiploss, wakerotation
!f2py logical, optional, intent(in) :: useCd = 1, hubLoss = 1, tipLoss = 1, wakerotation = 1
! out
  REAL(reki) :: fzero, a, ap
  REAL(reki), intent(inout) :: fzerob, ab, apb
! local
  REAL(reki) :: pi, sigma_p, sphi, cphi, lambda_r
  REAL(reki) :: sigma_pb, sphib, cphib, lambda_rb
  REAL(reki) :: factortip, ftip, factorhub, fhub
  REAL(reki) :: factortipb, ftipb, factorhubb, fhubb
  REAL(reki) :: k, kp, cn, ct, f
  REAL(reki) :: kb, kpb, cnb, ctb, fb
  REAL(reki) :: g1, g2, g3
  REAL(reki) :: g1b, g2b, g3b
  INTEGER :: branch
  REAL(reki) :: temp3
  REAL(reki) :: temp2
  REAL(reki) :: temp1
  REAL(reki) :: temp0
  INTRINSIC COS
  INTRINSIC EXP
  REAL(reki) :: abs1b
  REAL(reki) :: temp4b1
  REAL(reki) :: temp4b0
  INTRINSIC SIN
  REAL(reki) :: temp0b
  REAL(reki) :: temp3b
  INTRINSIC ABS
  REAL(reki) :: temp2b0
  REAL(reki) :: abs0b
  REAL(reki) :: temp5b3
  REAL(reki) :: temp5b2
  REAL(reki) :: temp5b1
  REAL(reki) :: temp5b0
  REAL(reki) :: tempb
  REAL(reki) :: temp0b0
  REAL(reki) :: temp2b
  INTRINSIC SELECTED_REAL_KIND
  REAL(reki) :: temp5b
  REAL(reki) :: temp3b0
  INTRINSIC ACOS
  REAL(reki) :: abs2
  REAL(reki) :: abs1
  REAL(reki) :: abs0
  REAL(reki) :: temp1b
  INTRINSIC SQRT
  REAL(reki) :: temp
  REAL(reki) :: temp1b0
  REAL(reki) :: temp4b
  REAL(reki) :: temp4
! constants
  pi = 3.1415926535897932
  sigma_p = b/2.0/pi*c/r
  sphi = SIN(phi)
  cphi = COS(phi)
! resolve into normal and tangential forces
  IF (.NOT.usecd) THEN
    cn = cl*cphi
    ct = cl*sphi
    CALL PUSHCONTROL1B(0)
  ELSE
    cn = cl*cphi + cd*sphi
    ct = cl*sphi - cd*cphi
    CALL PUSHCONTROL1B(1)
  END IF
! Prandtl's tip and hub loss factor
  ftip = 1.0
  IF (tiploss) THEN
    IF (sphi .GE. 0.) THEN
      abs0 = sphi
      CALL PUSHCONTROL1B(0)
    ELSE
      abs0 = -sphi
      CALL PUSHCONTROL1B(1)
    END IF
    factortip = b/2.0*(rtip-r)/(r*abs0)
    ftip = 2.0/pi*ACOS(EXP(-factortip))
    CALL PUSHCONTROL1B(0)
  ELSE
    CALL PUSHCONTROL1B(1)
  END IF
  fhub = 1.0
  IF (hubloss) THEN
    IF (sphi .GE. 0.) THEN
      abs1 = sphi
      CALL PUSHCONTROL1B(0)
    ELSE
      abs1 = -sphi
      CALL PUSHCONTROL1B(1)
    END IF
    factorhub = b/2.0*(r-rhub)/(rhub*abs1)
    fhub = 2.0/pi*ACOS(EXP(-factorhub))
    CALL PUSHCONTROL1B(0)
  ELSE
    CALL PUSHCONTROL1B(1)
  END IF
  f = ftip*fhub
! bem parameters
  k = sigma_p*cn/4/f/sphi/sphi
  kp = sigma_p*ct/4/f/sphi/cphi
! compute axial induction factor
  IF (phi .GT. 0) THEN
! momentum/empirical
! update axial induction factor
    IF (k .LE. 2.0/3) THEN
! momentum state
      a = k/(1+k)
      CALL PUSHCONTROL2B(0)
    ELSE
      IF (k - (25.0/18/f-1.0) .GE. 0.) THEN
        abs2 = k - (25.0/18/f-1.0)
      ELSE
        abs2 = -(k-(25.0/18/f-1.0))
      END IF
! Glauert(Buhl) correction
      IF (abs2 .LT. 1e-6) k = k + 1.0e-5
! avoid singularity
      g1 = 2*f*k - (10.0/9-f)
      g2 = 2*f*k - (4.0/3-f)*f
      g3 = 2*f*k - (25.0/9-2*f)
      a = (g1-SQRT(g2))/g3
      CALL PUSHCONTROL2B(1)
    END IF
  ELSE IF (k .GT. 1.0) THEN
! propeller brake region (a and ap not directly used but update anyway)
    a = k/(k-1)
    CALL PUSHCONTROL2B(2)
  ELSE
    CALL PUSHCONTROL2B(3)
! dummy value
    a = 0.0
  END IF
! compute tangential induction factor
  IF (.NOT.wakerotation) THEN
    kp = 0.0
    CALL PUSHCONTROL1B(0)
  ELSE
    CALL PUSHCONTROL1B(1)
  END IF
! error function
  lambda_r = vy/vx
  IF (phi .GT. 0) THEN
    temp5b1 = fzerob/(1-a)
    temp5b2 = -((1-kp)*fzerob/lambda_r)
    sphib = temp5b1
    ab = ab + sphi*temp5b1/(1-a)
    kpb = cphi*fzerob/lambda_r
    cphib = temp5b2
    lambda_rb = -(cphi*temp5b2/lambda_r)
    kb = 0.0
  ELSE
    temp5b3 = -((1-kp)*fzerob/lambda_r)
    sphib = (1-k)*fzerob
    kb = -(sphi*fzerob)
    kpb = cphi*fzerob/lambda_r
    cphib = temp5b3
    lambda_rb = -(cphi*temp5b3/lambda_r)
  END IF
  vyb = lambda_rb/vx
  vxb = -(vy*lambda_rb/vx**2)
  CALL POPCONTROL1B(branch)
  IF (branch .EQ. 0) THEN
    kp = sigma_p*ct/4/f/sphi/cphi
    apb = 0.0
    kpb = 0.0
  END IF
  temp5b0 = apb/(1-kp)
  kpb = kpb + (kp/(1-kp)+1.0)*temp5b0
  CALL POPCONTROL2B(branch)
  IF (branch .LT. 2) THEN
    IF (branch .EQ. 0) THEN
      temp4b0 = ab/(k+1)
      kb = kb + (1.0-k/(k+1))*temp4b0
      fb = 0.0
    ELSE
      temp4b1 = ab/g3
      temp4 = SQRT(g2)
      g1b = temp4b1
      IF (g2 .EQ. 0.0) THEN
        g2b = 0.0
      ELSE
        g2b = -(temp4b1/(2.0*temp4))
      END IF
      g3b = -((g1-temp4)*temp4b1/g3)
      fb = (2*f-4.0/3+2*k)*g2b + (2*k+1.0)*g1b + (2*k+2)*g3b
      kb = kb + 2*f*g2b + 2*f*g1b + 2*f*g3b
    END IF
  ELSE
    IF (branch .EQ. 2) THEN
      temp5b = ab/(k-1)
      kb = kb + (1.0-k/(k-1))*temp5b
    END IF
    fb = 0.0
  END IF
  temp2 = 4*f*sphi**2
  temp2b = kb/temp2
  temp2b0 = -(sigma_p*cn*temp2b/temp2)
  temp3 = 4*f*sphi*cphi
  temp3b = kpb/temp3
  temp3b0 = -(sigma_p*ct*temp3b/temp3)
  temp4b = 4*f*temp3b0
  sigma_pb = cn*temp2b + ct*temp3b
  ctb = sigma_p*temp3b
  fb = fb + sphi**2*4*temp2b0 + sphi*cphi*4*temp3b0
  sphib = sphib + 4*f*2*sphi*temp2b0 + cphi*temp4b
  cphib = cphib + sphi*temp4b
  cnb = sigma_p*temp2b
  ftipb = fhub*fb
  fhubb = ftip*fb
  CALL POPCONTROL1B(branch)
  IF (branch .EQ. 0) THEN
    IF (EXP(-factorhub) .EQ. 1.0 .OR. EXP(-factorhub) .EQ. (-1.0)) THEN
      factorhubb = 0.0
    ELSE
      factorhubb = EXP(-factorhub)*2.0*fhubb/(SQRT(1.0-EXP(-factorhub)**&
&        2)*pi)
    END IF
    temp1 = 2.0*rhub*abs1
    temp1b = b*factorhubb/temp1
    temp1b0 = -((r-rhub)*temp1b/temp1)
    rb = temp1b
    rhubb = abs1*2.0*temp1b0 - temp1b
    abs1b = 2.0*rhub*temp1b0
    CALL POPCONTROL1B(branch)
    IF (branch .EQ. 0) THEN
      sphib = sphib + abs1b
    ELSE
      sphib = sphib - abs1b
    END IF
  ELSE
    rb = 0.0
    rhubb = 0.0
  END IF
  CALL POPCONTROL1B(branch)
  IF (branch .EQ. 0) THEN
    IF (EXP(-factortip) .EQ. 1.0 .OR. EXP(-factortip) .EQ. (-1.0)) THEN
      factortipb = 0.0
    ELSE
      factortipb = EXP(-factortip)*2.0*ftipb/(SQRT(1.0-EXP(-factortip)**&
&        2)*pi)
    END IF
    temp0 = 2.0*r*abs0
    temp0b = b*factortipb/temp0
    temp0b0 = -((rtip-r)*temp0b/temp0)
    rtipb = temp0b
    rb = rb + abs0*2.0*temp0b0 - temp0b
    abs0b = 2.0*r*temp0b0
    CALL POPCONTROL1B(branch)
    IF (branch .EQ. 0) THEN
      sphib = sphib + abs0b
    ELSE
      sphib = sphib - abs0b
    END IF
  ELSE
    rtipb = 0.0
  END IF
  CALL POPCONTROL1B(branch)
  IF (branch .EQ. 0) THEN
    clb = cphi*cnb + sphi*ctb
    sphib = sphib + cl*ctb
    cphib = cphib + cl*cnb
    cdb = 0.0
  ELSE
    clb = cphi*cnb + sphi*ctb
    sphib = sphib + cd*cnb + cl*ctb
    cdb = sphi*cnb - cphi*ctb
    cphib = cphib + cl*cnb - cd*ctb
  END IF
  phib = COS(phi)*sphib - SIN(phi)*cphib
  temp = 2.0*pi*r
  tempb = b*sigma_pb/temp
  cb = tempb
  rb = rb - c*2.0*pi*tempb/temp
  apb = 0.0
  fzerob = 0.0
  ab = 0.0
END SUBROUTINE INDUCTIONFACTORS_B



!        Generated by TAPENADE     (INRIA, Tropics team)
!  Tapenade 3.7 (r4888) - 28 May 2013 10:47
!
!  Differentiation of inductionfactors in forward (tangent) mode:
!   variations   of useful results: ap fzero a
!   with respect to varying inputs: r rtip rhub phi cd cl vx vy
!                c
!   RW status of diff variables: r:in rtip:in ap:out rhub:in fzero:out
!                phi:in cd:in cl:in vx:in vy:in a:out c:in
SUBROUTINE INDUCTIONFACTORS_D(r, rd, c, cd0, rhub, rhubd, rtip, rtipd, &
&  phi, phid, cl, cld, cd, cdd, b, vx, vxd, vy, vyd, usecd, hubloss, &
&  tiploss, wakerotation, fzero, fzerod, a, ad, ap, apd)
  IMPLICIT NONE
  INTEGER, PARAMETER :: reki=SELECTED_REAL_KIND(15, 307)
! in
  REAL(reki), INTENT(IN) :: r, c, rhub, rtip, phi, cl, cd
  REAL(reki), INTENT(IN) :: rd, cd0, rhubd, rtipd, phid, cld, cdd
  INTEGER, INTENT(IN) :: b
  REAL(reki), INTENT(IN) :: vx, vy
  REAL(reki), INTENT(IN) :: vxd, vyd
  LOGICAL, INTENT(IN) :: usecd, hubloss, tiploss, wakerotation
!f2py logical, optional, intent(in) :: useCd = 1, hubLoss = 1, tipLoss = 1, wakerotation = 1
! out
  REAL(reki), INTENT(OUT) :: fzero, a, ap
  REAL(reki), INTENT(OUT) :: fzerod, ad, apd
! local
  REAL(reki) :: pi, sigma_p, sphi, cphi, lambda_r
  REAL(reki) :: sigma_pd, sphid, cphid, lambda_rd
  REAL(reki) :: factortip, ftip, factorhub, fhub
  REAL(reki) :: factortipd, ftipd, factorhubd, fhubd
  REAL(reki) :: k, kp, cn, ct, f
  REAL(reki) :: kd, kpd, cnd, ctd, fd
  REAL(reki) :: g1, g2, g3
  REAL(reki) :: g1d, g2d, g3d
  REAL(reki) :: arg1
  REAL(reki) :: arg1d
  REAL(reki) :: result1
  REAL(reki) :: result1d
  INTRINSIC COS
  INTRINSIC EXP
  REAL(reki) :: abs1d
  INTRINSIC SIN
  INTRINSIC ABS
  REAL(reki) :: abs0d
  INTRINSIC SELECTED_REAL_KIND
  INTRINSIC ACOS
  REAL(reki) :: abs2
  REAL(reki) :: abs1
  REAL(reki) :: abs0
  INTRINSIC SQRT
! constants
  pi = 3.1415926535897932
  sigma_pd = (b*cd0*r/(2.0*pi)-b*c*rd/(2.0*pi))/r**2
  sigma_p = b/2.0/pi*c/r
  sphid = phid*COS(phi)
  sphi = SIN(phi)
  cphid = -(phid*SIN(phi))
  cphi = COS(phi)
! resolve into normal and tangential forces
  IF (.NOT.usecd) THEN
    cnd = cld*cphi + cl*cphid
    cn = cl*cphi
    ctd = cld*sphi + cl*sphid
    ct = cl*sphi
  ELSE
    cnd = cld*cphi + cl*cphid + cdd*sphi + cd*sphid
    cn = cl*cphi + cd*sphi
    ctd = cld*sphi + cl*sphid - cdd*cphi - cd*cphid
    ct = cl*sphi - cd*cphi
  END IF
! Prandtl's tip and hub loss factor
  ftip = 1.0
  IF (tiploss) THEN
    IF (sphi .GE. 0.) THEN
      abs0d = sphid
      abs0 = sphi
    ELSE
      abs0d = -sphid
      abs0 = -sphi
    END IF
    factortipd = (b*(rtipd-rd)*r*abs0/2.0-b*(rtip-r)*(rd*abs0+r*abs0d)/&
&      2.0)/(r*abs0)**2
    factortip = b/2.0*(rtip-r)/(r*abs0)
    arg1d = -(factortipd*EXP(-factortip))
    arg1 = EXP(-factortip)
    IF (arg1 .EQ. 1.0 .OR. arg1 .EQ. (-1.0)) THEN
      result1d = 0.0
    ELSE
      result1d = -(arg1d/SQRT(1.0-arg1**2))
    END IF
    result1 = ACOS(arg1)
    ftipd = 2.0*result1d/pi
    ftip = 2.0/pi*result1
  ELSE
    ftipd = 0.0
  END IF
  fhub = 1.0
  IF (hubloss) THEN
    IF (sphi .GE. 0.) THEN
      abs1d = sphid
      abs1 = sphi
    ELSE
      abs1d = -sphid
      abs1 = -sphi
    END IF
    factorhubd = (b*(rd-rhubd)*rhub*abs1/2.0-b*(r-rhub)*(rhubd*abs1+rhub&
&      *abs1d)/2.0)/(rhub*abs1)**2
    factorhub = b/2.0*(r-rhub)/(rhub*abs1)
    arg1d = -(factorhubd*EXP(-factorhub))
    arg1 = EXP(-factorhub)
    IF (arg1 .EQ. 1.0 .OR. arg1 .EQ. (-1.0)) THEN
      result1d = 0.0
    ELSE
      result1d = -(arg1d/SQRT(1.0-arg1**2))
    END IF
    result1 = ACOS(arg1)
    fhubd = 2.0*result1d/pi
    fhub = 2.0/pi*result1
  ELSE
    fhubd = 0.0
  END IF
  fd = ftipd*fhub + ftip*fhubd
  f = ftip*fhub
! bem parameters
  kd = ((((sigma_pd*cn+sigma_p*cnd)*f/4-sigma_p*cn*fd/4)*sphi/f**2-&
&    sigma_p*cn*sphid/(4*f))/sphi-sigma_p*cn*sphid/(4*f*sphi))/sphi**2
  k = sigma_p*cn/4/f/sphi/sphi
  kpd = ((((sigma_pd*ct+sigma_p*ctd)*f/4-sigma_p*ct*fd/4)*sphi/f**2-&
&    sigma_p*ct*sphid/(4*f))*cphi/sphi**2-sigma_p*ct*cphid/(4*f*sphi))/&
&    cphi**2
  kp = sigma_p*ct/4/f/sphi/cphi
! compute axial induction factor
  IF (phi .GT. 0) THEN
! momentum/empirical
! update axial induction factor
    IF (k .LE. 2.0/3) THEN
! momentum state
      ad = (kd*(1+k)-k*kd)/(1+k)**2
      a = k/(1+k)
    ELSE
      IF (k - (25.0/18/f-1.0) .GE. 0.) THEN
        abs2 = k - (25.0/18/f-1.0)
      ELSE
        abs2 = -(k-(25.0/18/f-1.0))
      END IF
! Glauert(Buhl) correction
      IF (abs2 .LT. 1e-6) k = k + 1.0e-5
! avoid singularity
      g1d = 2*(fd*k) + 2*(f*kd) + fd
      g1 = 2*f*k - (10.0/9-f)
      g2d = 2*(fd*k) + 2*(f*kd) - (4.0/3-f)*fd + fd*f
      g2 = 2*f*k - (4.0/3-f)*f
      g3d = 2*(fd*k) + 2*(f*kd) + 2*fd
      g3 = 2*f*k - (25.0/9-2*f)
      IF (g2 .EQ. 0.0) THEN
        result1d = 0.0
      ELSE
        result1d = g2d/(2.0*SQRT(g2))
      END IF
      result1 = SQRT(g2)
      ad = ((g1d-result1d)*g3-(g1-result1)*g3d)/g3**2
      a = (g1-result1)/g3
    END IF
  ELSE IF (k .GT. 1.0) THEN
! propeller brake region (a and ap not directly used but update anyway)
    ad = (kd*(k-1)-k*kd)/(k-1)**2
    a = k/(k-1)
  ELSE
! dummy value
    a = 0.0
    ad = 0.0
  END IF
! compute tangential induction factor
  apd = (kpd*(1-kp)+kp*kpd)/(1-kp)**2
  ap = kp/(1-kp)
  IF (.NOT.wakerotation) THEN
    ap = 0.0
    kp = 0.0
    apd = 0.0
    kpd = 0.0
  END IF
! error function
  lambda_rd = (vyd*vx-vy*vxd)/vx**2
  lambda_r = vy/vx
  IF (phi .GT. 0) THEN
! momentum/empirical
    fzerod = (sphid*(1-a)+sphi*ad)/(1-a)**2 - (cphid*lambda_r-cphi*&
&      lambda_rd)*(1-kp)/lambda_r**2 + cphi*kpd/lambda_r
    fzero = sphi/(1-a) - cphi/lambda_r*(1-kp)
  ELSE
! propeller brake region
    fzerod = sphid*(1-k) - sphi*kd - (cphid*lambda_r-cphi*lambda_rd)*(1-&
&      kp)/lambda_r**2 + cphi*kpd/lambda_r
    fzero = sphi*(1-k) - cphi/lambda_r*(1-kp)
  END IF
END SUBROUTINE INDUCTIONFACTORS_D




!        Generated by TAPENADE     (INRIA, Tropics team)
!  Tapenade 3.7 (r4888) - 28 May 2013 10:47
!
!  Differentiation of inductionfactors in reverse (adjoint) mode:
!   gradient     of useful results: ap fzero a
!   with respect to varying inputs: r rtip ap rhub fzero phi cd
!                cl vx vy a c
!   RW status of diff variables: r:out rtip:out ap:in-zero rhub:out
!                fzero:in-zero phi:out cd:out cl:out vx:out vy:out
!                a:in-zero c:out
SUBROUTINE INDUCTIONFACTORS_BV(r, rb, c, cb, rhub, rhubb, rtip, rtipb, &
&  phi, phib, cl, clb, cd, cdb, b, vx, vxb, vy, vyb, usecd, hubloss, &
&  tiploss, wakerotation, fzero, fzerob, a, ab, ap, apb, nbdirsmax)
  !USE DIFFSIZES
!  Hint: nbdirsmax should be the maximum number of differentiation directions
  IMPLICIT NONE
  INTEGER, PARAMETER :: reki=SELECTED_REAL_KIND(15, 307)
! in
  integer, intent(in) :: nbdirsmax
  REAL(reki), INTENT(IN) :: r, c, rhub, rtip, phi, cl, cd
  REAL(reki), DIMENSION(nbdirsmax), intent(out) :: rb, cb, rhubb, rtipb, phib, clb, &
&  cdb
  INTEGER, INTENT(IN) :: b
  REAL(reki), INTENT(IN) :: vx, vy
  REAL(reki), DIMENSION(nbdirsmax), intent(out) :: vxb, vyb
  LOGICAL, INTENT(IN) :: usecd, hubloss, tiploss, wakerotation
!f2py logical, optional, intent(in) :: useCd = 1, hubLoss = 1, tipLoss = 1, wakerotation = 1
! out
  REAL(reki) :: fzero, a, ap
  REAL(reki), DIMENSION(nbdirsmax), intent(inout) :: fzerob, ab, apb
! local
  REAL(reki) :: pi, sigma_p, sphi, cphi, lambda_r
  REAL(reki), DIMENSION(nbdirsmax) :: sigma_pb, sphib, cphib, lambda_rb
  REAL(reki) :: factortip, ftip, factorhub, fhub
  REAL(reki), DIMENSION(nbdirsmax) :: factortipb, ftipb, factorhubb, &
&  fhubb
  REAL(reki) :: k, kp, cn, ct, f
  REAL(reki), DIMENSION(nbdirsmax) :: kb, kpb, cnb, ctb, fb
  REAL(reki) :: g1, g2, g3
  REAL(reki), DIMENSION(nbdirsmax) :: g1b, g2b, g3b
  INTEGER :: nd
  INTEGER :: branch
  INTEGER :: nbdirs
  REAL(reki) :: temp3
  REAL(reki) :: temp2
  REAL(reki) :: temp1
  REAL(reki) :: temp0
  INTRINSIC COS
  INTRINSIC EXP
  REAL(reki) :: abs1b(nbdirsmax)
  REAL(reki) :: temp4b1(nbdirsmax)
  REAL(reki) :: temp4b0(nbdirsmax)
  INTRINSIC SIN
  REAL(reki) :: temp0b(nbdirsmax)
  REAL(reki) :: temp3b(nbdirsmax)
  INTRINSIC ABS
  REAL(reki) :: temp2b0(nbdirsmax)
  REAL(reki) :: abs0b(nbdirsmax)
  REAL(reki) :: temp5b3(nbdirsmax)
  REAL(reki) :: temp5b2(nbdirsmax)
  REAL(reki) :: temp5b1(nbdirsmax)
  REAL(reki) :: temp5b0(nbdirsmax)
  REAL(reki) :: tempb(nbdirsmax)
  REAL(reki) :: temp0b0(nbdirsmax)
  REAL(reki) :: temp2b(nbdirsmax)
  INTRINSIC SELECTED_REAL_KIND
  REAL(reki) :: temp5b(nbdirsmax)
  REAL(reki) :: temp3b0(nbdirsmax)
  INTRINSIC ACOS
  REAL(reki) :: abs2
  REAL(reki) :: abs1
  REAL(reki) :: abs0
  REAL(reki) :: temp1b(nbdirsmax)
  INTRINSIC SQRT
  REAL(reki) :: temp
  REAL(reki) :: temp1b0(nbdirsmax)
  REAL(reki) :: temp4b(nbdirsmax)
  REAL(reki) :: temp4
! constants
  pi = 3.1415926535897932
  sigma_p = b/2.0/pi*c/r
  sphi = SIN(phi)
  cphi = COS(phi)
! resolve into normal and tangential forces
  IF (.NOT.usecd) THEN
    cn = cl*cphi
    ct = cl*sphi
    CALL PUSHCONTROL1B(0)
  ELSE
    cn = cl*cphi + cd*sphi
    ct = cl*sphi - cd*cphi
    CALL PUSHCONTROL1B(1)
  END IF
! Prandtl's tip and hub loss factor
  ftip = 1.0
  IF (tiploss) THEN
    IF (sphi .GE. 0.) THEN
      abs0 = sphi
      CALL PUSHCONTROL1B(0)
    ELSE
      abs0 = -sphi
      CALL PUSHCONTROL1B(1)
    END IF
    factortip = b/2.0*(rtip-r)/(r*abs0)
    ftip = 2.0/pi*ACOS(EXP(-factortip))
    CALL PUSHCONTROL1B(0)
  ELSE
    CALL PUSHCONTROL1B(1)
  END IF
  fhub = 1.0
  IF (hubloss) THEN
    IF (sphi .GE. 0.) THEN
      abs1 = sphi
      CALL PUSHCONTROL1B(0)
    ELSE
      abs1 = -sphi
      CALL PUSHCONTROL1B(1)
    END IF
    factorhub = b/2.0*(r-rhub)/(rhub*abs1)
    fhub = 2.0/pi*ACOS(EXP(-factorhub))
    CALL PUSHCONTROL1B(0)
  ELSE
    CALL PUSHCONTROL1B(1)
  END IF
  f = ftip*fhub
! bem parameters
  k = sigma_p*cn/4/f/sphi/sphi
  kp = sigma_p*ct/4/f/sphi/cphi
! compute axial induction factor
  IF (phi .GT. 0) THEN
! momentum/empirical
! update axial induction factor
    IF (k .LE. 2.0/3) THEN
! momentum state
      a = k/(1+k)
      CALL PUSHCONTROL2B(0)
    ELSE
      IF (k - (25.0/18/f-1.0) .GE. 0.) THEN
        abs2 = k - (25.0/18/f-1.0)
      ELSE
        abs2 = -(k-(25.0/18/f-1.0))
      END IF
! Glauert(Buhl) correction
      IF (abs2 .LT. 1e-6) k = k + 1.0e-5
! avoid singularity
      g1 = 2*f*k - (10.0/9-f)
      g2 = 2*f*k - (4.0/3-f)*f
      g3 = 2*f*k - (25.0/9-2*f)
      a = (g1-SQRT(g2))/g3
      CALL PUSHCONTROL2B(1)
    END IF
  ELSE IF (k .GT. 1.0) THEN
! propeller brake region (a and ap not directly used but update anyway)
    a = k/(k-1)
    CALL PUSHCONTROL2B(2)
  ELSE
    CALL PUSHCONTROL2B(3)
! dummy value
    a = 0.0
  END IF
! compute tangential induction factor
  IF (.NOT.wakerotation) THEN
    kp = 0.0
    CALL PUSHCONTROL1B(0)
  ELSE
    CALL PUSHCONTROL1B(1)
  END IF
! error function
  lambda_r = vy/vx
  IF (phi .GT. 0) THEN
    DO nd=1,nbdirsmax
      temp5b1(nd) = fzerob(nd)/(1-a)
      temp5b2(nd) = -((1-kp)*fzerob(nd)/lambda_r)
      sphib(nd) = temp5b1(nd)
      ab(nd) = ab(nd) + sphi*temp5b1(nd)/(1-a)
      kpb(nd) = cphi*fzerob(nd)/lambda_r
      cphib(nd) = temp5b2(nd)
      lambda_rb(nd) = -(cphi*temp5b2(nd)/lambda_r)
    END DO
    DO nd=1,nbdirsmax
      kb(nd) = 0.0
    END DO
  ELSE
    DO nd=1,nbdirsmax
      temp5b3(nd) = -((1-kp)*fzerob(nd)/lambda_r)
      sphib(nd) = (1-k)*fzerob(nd)
      kb(nd) = -(sphi*fzerob(nd))
      kpb(nd) = cphi*fzerob(nd)/lambda_r
      cphib(nd) = temp5b3(nd)
      lambda_rb(nd) = -(cphi*temp5b3(nd)/lambda_r)
    END DO
  END IF
  DO nd=1,nbdirsmax
    vyb(nd) = lambda_rb(nd)/vx
    vxb(nd) = -(vy*lambda_rb(nd)/vx**2)
  END DO
  CALL POPCONTROL1B(branch)
  IF (branch .EQ. 0) THEN
    kp = sigma_p*ct/4/f/sphi/cphi
    DO nd=1,nbdirsmax
      apb(nd) = 0.0
      kpb(nd) = 0.0
    END DO
  END IF
  DO nd=1,nbdirsmax
    temp5b0(nd) = apb(nd)/(1-kp)
    kpb(nd) = kpb(nd) + (kp/(1-kp)+1.0)*temp5b0(nd)
  END DO
  CALL POPCONTROL2B(branch)
  IF (branch .LT. 2) THEN
    IF (branch .EQ. 0) THEN
      DO nd=1,nbdirsmax
        temp4b0(nd) = ab(nd)/(k+1)
        kb(nd) = kb(nd) + (1.0-k/(k+1))*temp4b0(nd)
      END DO
      DO nd=1,nbdirsmax
        fb(nd) = 0.0
      END DO
    ELSE
      temp4 = SQRT(g2)
      DO nd=1,nbdirsmax
        temp4b1(nd) = ab(nd)/g3
        g1b(nd) = temp4b1(nd)
        IF (g2 .EQ. 0.0) THEN
          g2b(nd) = 0.0
        ELSE
          g2b(nd) = -(temp4b1(nd)/(2.0*temp4))
        END IF
        g3b(nd) = -((g1-temp4)*temp4b1(nd)/g3)
        fb(nd) = (2*f-4.0/3+2*k)*g2b(nd) + (2*k+1.0)*g1b(nd) + (2*k+2)*&
&          g3b(nd)
        kb(nd) = kb(nd) + 2*f*g2b(nd) + 2*f*g1b(nd) + 2*f*g3b(nd)
      END DO
    END IF
  ELSE
    IF (branch .EQ. 2) THEN
      DO nd=1,nbdirsmax
        temp5b(nd) = ab(nd)/(k-1)
        kb(nd) = kb(nd) + (1.0-k/(k-1))*temp5b(nd)
      END DO
    END IF
    DO nd=1,nbdirsmax
      fb(nd) = 0.0
    END DO
  END IF
  temp3 = 4*f*sphi*cphi
  temp2 = 4*f*sphi**2
  DO nd=1,nbdirsmax
    temp2b(nd) = kb(nd)/temp2
    temp2b0(nd) = -(sigma_p*cn*temp2b(nd)/temp2)
    temp3b(nd) = kpb(nd)/temp3
    temp3b0(nd) = -(sigma_p*ct*temp3b(nd)/temp3)
    temp4b(nd) = 4*f*temp3b0(nd)
    sigma_pb(nd) = cn*temp2b(nd) + ct*temp3b(nd)
    ctb(nd) = sigma_p*temp3b(nd)
    fb(nd) = fb(nd) + sphi**2*4*temp2b0(nd) + sphi*cphi*4*temp3b0(nd)
    sphib(nd) = sphib(nd) + 4*f*2*sphi*temp2b0(nd) + cphi*temp4b(nd)
    cphib(nd) = cphib(nd) + sphi*temp4b(nd)
    cnb(nd) = sigma_p*temp2b(nd)
    ftipb(nd) = fhub*fb(nd)
    fhubb(nd) = ftip*fb(nd)
  END DO
  CALL POPCONTROL1B(branch)
  IF (branch .EQ. 0) THEN
    temp1 = 2.0*rhub*abs1
    DO nd=1,nbdirsmax
      IF (EXP(-factorhub) .EQ. 1.0 .OR. EXP(-factorhub) .EQ. (-1.0)) &
&      THEN
        factorhubb(nd) = 0.0
      ELSE
        factorhubb(nd) = EXP(-factorhub)*2.0*fhubb(nd)/(SQRT(1.0-EXP(-&
&          factorhub)**2)*pi)
      END IF
      temp1b(nd) = b*factorhubb(nd)/temp1
      temp1b0(nd) = -((r-rhub)*temp1b(nd)/temp1)
      rb(nd) = temp1b(nd)
      rhubb(nd) = abs1*2.0*temp1b0(nd) - temp1b(nd)
      abs1b(nd) = 2.0*rhub*temp1b0(nd)
    END DO
    CALL POPCONTROL1B(branch)
    IF (branch .EQ. 0) THEN
      DO nd=1,nbdirsmax
        sphib(nd) = sphib(nd) + abs1b(nd)
      END DO
    ELSE
      DO nd=1,nbdirsmax
        sphib(nd) = sphib(nd) - abs1b(nd)
      END DO
    END IF
  ELSE
    DO nd=1,nbdirsmax
      rb(nd) = 0.0
      rhubb(nd) = 0.0
    END DO
  END IF
  CALL POPCONTROL1B(branch)
  IF (branch .EQ. 0) THEN
    temp0 = 2.0*r*abs0
    DO nd=1,nbdirsmax
      IF (EXP(-factortip) .EQ. 1.0 .OR. EXP(-factortip) .EQ. (-1.0)) &
&      THEN
        factortipb(nd) = 0.0
      ELSE
        factortipb(nd) = EXP(-factortip)*2.0*ftipb(nd)/(SQRT(1.0-EXP(-&
&          factortip)**2)*pi)
      END IF
      temp0b(nd) = b*factortipb(nd)/temp0
      temp0b0(nd) = -((rtip-r)*temp0b(nd)/temp0)
      rtipb(nd) = temp0b(nd)
      rb(nd) = rb(nd) + abs0*2.0*temp0b0(nd) - temp0b(nd)
      abs0b(nd) = 2.0*r*temp0b0(nd)
    END DO
    CALL POPCONTROL1B(branch)
    IF (branch .EQ. 0) THEN
      DO nd=1,nbdirsmax
        sphib(nd) = sphib(nd) + abs0b(nd)
      END DO
    ELSE
      DO nd=1,nbdirsmax
        sphib(nd) = sphib(nd) - abs0b(nd)
      END DO
    END IF
  ELSE
    DO nd=1,nbdirsmax
      rtipb(nd) = 0.0
    END DO
  END IF
  CALL POPCONTROL1B(branch)
  IF (branch .EQ. 0) THEN
    DO nd=1,nbdirsmax
      clb(nd) = cphi*cnb(nd) + sphi*ctb(nd)
      sphib(nd) = sphib(nd) + cl*ctb(nd)
      cphib(nd) = cphib(nd) + cl*cnb(nd)
    END DO
    DO nd=1,nbdirsmax
      cdb(nd) = 0.0
    END DO
  ELSE
    DO nd=1,nbdirsmax
      clb(nd) = cphi*cnb(nd) + sphi*ctb(nd)
      sphib(nd) = sphib(nd) + cd*cnb(nd) + cl*ctb(nd)
      cdb(nd) = sphi*cnb(nd) - cphi*ctb(nd)
      cphib(nd) = cphib(nd) + cl*cnb(nd) - cd*ctb(nd)
    END DO
  END IF
  temp = 2.0*pi*r
  DO nd=1,nbdirsmax
    phib(nd) = COS(phi)*sphib(nd) - SIN(phi)*cphib(nd)
    tempb(nd) = b*sigma_pb(nd)/temp
    cb(nd) = tempb(nd)
    rb(nd) = rb(nd) - c*2.0*pi*tempb(nd)/temp
  END DO
  DO nd=1,nbdirsmax
    apb(nd) = 0.0
    fzerob(nd) = 0.0
    ab(nd) = 0.0
  END DO
END SUBROUTINE INDUCTIONFACTORS_BV




!        Generated by TAPENADE     (INRIA, Tropics team)
!  Tapenade 3.7 (r4888) - 28 May 2013 10:47
!
!  Differentiation of inductionfactors in forward (tangent) mode:
!   variations   of useful results: ap fzero a
!   with respect to varying inputs: r rtip rhub phi cd cl vx vy
!                c
!   RW status of diff variables: r:in rtip:in ap:out rhub:in fzero:out
!                phi:in cd:in cl:in vx:in vy:in a:out c:in
SUBROUTINE INDUCTIONFACTORS_DV(r, rd, c, cd0, rhub, rhubd, rtip, rtipd, &
&  phi, phid, cl, cld, cd, cdd, b, vx, vxd, vy, vyd, usecd, hubloss, &
&  tiploss, wakerotation, fzero, fzerod, a, ad, ap, apd, nbdirsmax)
  !USE DIFFSIZES
!  Hint: nbdirsmax should be the maximum number of differentiation directions
  IMPLICIT NONE
  INTEGER, PARAMETER :: reki=SELECTED_REAL_KIND(15, 307)
! in
  integer, intent(in) :: nbdirsmax
  REAL(reki), INTENT(IN) :: r, c, rhub, rtip, phi, cl, cd
  REAL(reki), DIMENSION(nbdirsmax), INTENT(IN) :: rd, cd0, rhubd, rtipd&
&  , phid, cld, cdd
  INTEGER, INTENT(IN) :: b
  REAL(reki), INTENT(IN) :: vx, vy
  REAL(reki), DIMENSION(nbdirsmax), INTENT(IN) :: vxd, vyd
  LOGICAL, INTENT(IN) :: usecd, hubloss, tiploss, wakerotation
!f2py logical, optional, intent(in) :: useCd = 1, hubLoss = 1, tipLoss = 1, wakerotation = 1
! out
  REAL(reki), INTENT(OUT) :: fzero, a, ap
  REAL(reki), DIMENSION(nbdirsmax), INTENT(OUT) :: fzerod, ad, apd
! local
  REAL(reki) :: pi, sigma_p, sphi, cphi, lambda_r
  REAL(reki), DIMENSION(nbdirsmax) :: sigma_pd, sphid, cphid, lambda_rd
  REAL(reki) :: factortip, ftip, factorhub, fhub
  REAL(reki), DIMENSION(nbdirsmax) :: factortipd, ftipd, factorhubd, &
&  fhubd
  REAL(reki) :: k, kp, cn, ct, f
  REAL(reki), DIMENSION(nbdirsmax) :: kd, kpd, cnd, ctd, fd
  REAL(reki) :: g1, g2, g3
  REAL(reki), DIMENSION(nbdirsmax) :: g1d, g2d, g3d
  REAL(reki) :: arg1
  REAL(reki), DIMENSION(nbdirsmax) :: arg1d
  REAL(reki) :: result1
  REAL(reki), DIMENSION(nbdirsmax) :: result1d
  INTEGER :: nd
  INTEGER :: nbdirs
  INTRINSIC COS
  INTRINSIC EXP
  REAL(reki) :: abs1d(nbdirsmax)
  INTRINSIC SIN
  INTRINSIC ABS
  REAL(reki) :: abs0d(nbdirsmax)
  INTRINSIC SELECTED_REAL_KIND
  INTRINSIC ACOS
  REAL(reki) :: abs2
  REAL(reki) :: abs1
  REAL(reki) :: abs0
  INTRINSIC SQRT
! constants
  pi = 3.1415926535897932
  DO nd=1,nbdirsmax
    sigma_pd(nd) = (b*cd0(nd)*r/(2.0*pi)-b*c*rd(nd)/(2.0*pi))/r**2
    sphid(nd) = phid(nd)*COS(phi)
    cphid(nd) = -(phid(nd)*SIN(phi))
  END DO
  sigma_p = b/2.0/pi*c/r
  sphi = SIN(phi)
  cphi = COS(phi)
! resolve into normal and tangential forces
  IF (.NOT.usecd) THEN
    DO nd=1,nbdirsmax
      cnd(nd) = cld(nd)*cphi + cl*cphid(nd)
      ctd(nd) = cld(nd)*sphi + cl*sphid(nd)
    END DO
    cn = cl*cphi
    ct = cl*sphi
  ELSE
    DO nd=1,nbdirsmax
      cnd(nd) = cld(nd)*cphi + cl*cphid(nd) + cdd(nd)*sphi + cd*sphid(nd&
&        )
      ctd(nd) = cld(nd)*sphi + cl*sphid(nd) - cdd(nd)*cphi - cd*cphid(nd&
&        )
    END DO
    cn = cl*cphi + cd*sphi
    ct = cl*sphi - cd*cphi
  END IF
! Prandtl's tip and hub loss factor
  ftip = 1.0
  IF (tiploss) THEN
    IF (sphi .GE. 0.) THEN
      DO nd=1,nbdirsmax
        abs0d(nd) = sphid(nd)
      END DO
      abs0 = sphi
    ELSE
      DO nd=1,nbdirsmax
        abs0d(nd) = -sphid(nd)
      END DO
      abs0 = -sphi
    END IF
    factortip = b/2.0*(rtip-r)/(r*abs0)
    arg1 = EXP(-factortip)
    DO nd=1,nbdirsmax
      factortipd(nd) = (b*(rtipd(nd)-rd(nd))*r*abs0/2.0-b*(rtip-r)*(rd(&
&        nd)*abs0+r*abs0d(nd))/2.0)/(r*abs0)**2
      arg1d(nd) = -(factortipd(nd)*EXP(-factortip))
      IF (arg1 .EQ. 1.0 .OR. arg1 .EQ. (-1.0)) THEN
        result1d(nd) = 0.0
      ELSE
        result1d(nd) = -(arg1d(nd)/SQRT(1.0-arg1**2))
      END IF
      ftipd(nd) = 2.0*result1d(nd)/pi
    END DO
    result1 = ACOS(arg1)
    ftip = 2.0/pi*result1
  ELSE
    DO nd=1,nbdirsmax
      ftipd(nd) = 0.0
    END DO
  END IF
  fhub = 1.0
  IF (hubloss) THEN
    IF (sphi .GE. 0.) THEN
      DO nd=1,nbdirsmax
        abs1d(nd) = sphid(nd)
      END DO
      abs1 = sphi
    ELSE
      DO nd=1,nbdirsmax
        abs1d(nd) = -sphid(nd)
      END DO
      abs1 = -sphi
    END IF
    factorhub = b/2.0*(r-rhub)/(rhub*abs1)
    arg1 = EXP(-factorhub)
    DO nd=1,nbdirsmax
      factorhubd(nd) = (b*(rd(nd)-rhubd(nd))*rhub*abs1/2.0-b*(r-rhub)*(&
&        rhubd(nd)*abs1+rhub*abs1d(nd))/2.0)/(rhub*abs1)**2
      arg1d(nd) = -(factorhubd(nd)*EXP(-factorhub))
      IF (arg1 .EQ. 1.0 .OR. arg1 .EQ. (-1.0)) THEN
        result1d(nd) = 0.0
      ELSE
        result1d(nd) = -(arg1d(nd)/SQRT(1.0-arg1**2))
      END IF
      fhubd(nd) = 2.0*result1d(nd)/pi
    END DO
    result1 = ACOS(arg1)
    fhub = 2.0/pi*result1
  ELSE
    DO nd=1,nbdirsmax
      fhubd(nd) = 0.0
    END DO
  END IF
  f = ftip*fhub
  DO nd=1,nbdirsmax
    fd(nd) = ftipd(nd)*fhub + ftip*fhubd(nd)
    kd(nd) = ((((sigma_pd(nd)*cn+sigma_p*cnd(nd))*f/4-sigma_p*cn*fd(nd)/&
&      4)*sphi/f**2-sigma_p*cn*sphid(nd)/(4*f))/sphi-sigma_p*cn*sphid(nd)&
&      /(4*f*sphi))/sphi**2
    kpd(nd) = ((((sigma_pd(nd)*ct+sigma_p*ctd(nd))*f/4-sigma_p*ct*fd(nd)&
&      /4)*sphi/f**2-sigma_p*ct*sphid(nd)/(4*f))*cphi/sphi**2-sigma_p*ct*&
&      cphid(nd)/(4*f*sphi))/cphi**2
  END DO
! bem parameters
  k = sigma_p*cn/4/f/sphi/sphi
  kp = sigma_p*ct/4/f/sphi/cphi
! compute axial induction factor
  IF (phi .GT. 0) THEN
! momentum/empirical
! update axial induction factor
    IF (k .LE. 2.0/3) THEN
      DO nd=1,nbdirsmax
        ad(nd) = (kd(nd)*(1+k)-k*kd(nd))/(1+k)**2
      END DO
! momentum state
      a = k/(1+k)
    ELSE
      IF (k - (25.0/18/f-1.0) .GE. 0.) THEN
        abs2 = k - (25.0/18/f-1.0)
      ELSE
        abs2 = -(k-(25.0/18/f-1.0))
      END IF
! Glauert(Buhl) correction
      IF (abs2 .LT. 1e-6) k = k + 1.0e-5
! avoid singularity
      g1 = 2*f*k - (10.0/9-f)
      g2 = 2*f*k - (4.0/3-f)*f
      g3 = 2*f*k - (25.0/9-2*f)
      result1 = SQRT(g2)
      DO nd=1,nbdirsmax
        g1d(nd) = 2*(fd(nd)*k) + 2*(f*kd(nd)) + fd(nd)
        g2d(nd) = 2*(fd(nd)*k) + 2*(f*kd(nd)) - (4.0/3-f)*fd(nd) + fd(nd&
&          )*f
        g3d(nd) = 2*(fd(nd)*k) + 2*(f*kd(nd)) + 2*fd(nd)
        IF (g2 .EQ. 0.0) THEN
          result1d(nd) = 0.0
        ELSE
          result1d(nd) = g2d(nd)/(2.0*SQRT(g2))
        END IF
        ad(nd) = ((g1d(nd)-result1d(nd))*g3-(g1-result1)*g3d(nd))/g3**2
      END DO
      a = (g1-result1)/g3
    END IF
  ELSE IF (k .GT. 1.0) THEN
! propeller brake region (a and ap not directly used but update anyway)
    DO nd=1,nbdirsmax
      ad(nd) = (kd(nd)*(k-1)-k*kd(nd))/(k-1)**2
    END DO
    a = k/(k-1)
  ELSE
! dummy value
    a = 0.0
    DO nd=1,nbdirsmax
      ad(nd) = 0.0
    END DO
  END IF
  DO nd=1,nbdirsmax
    apd(nd) = (kpd(nd)*(1-kp)+kp*kpd(nd))/(1-kp)**2
  END DO
! compute tangential induction factor
  ap = kp/(1-kp)
  IF (.NOT.wakerotation) THEN
    ap = 0.0
    kp = 0.0
    DO nd=1,nbdirsmax
      apd(nd) = 0.0
      kpd(nd) = 0.0
    END DO
  END IF
  DO nd=1,nbdirsmax
    lambda_rd(nd) = (vyd(nd)*vx-vy*vxd(nd))/vx**2
  END DO
! error function
  lambda_r = vy/vx
  IF (phi .GT. 0) THEN
    DO nd=1,nbdirsmax
      fzerod(nd) = (sphid(nd)*(1-a)+sphi*ad(nd))/(1-a)**2 - (cphid(nd)*&
&        lambda_r-cphi*lambda_rd(nd))*(1-kp)/lambda_r**2 + cphi*kpd(nd)/&
&        lambda_r
    END DO
! momentum/empirical
    fzero = sphi/(1-a) - cphi/lambda_r*(1-kp)
  ELSE
    DO nd=1,nbdirsmax
      fzerod(nd) = sphid(nd)*(1-k) - sphi*kd(nd) - (cphid(nd)*lambda_r-&
&        cphi*lambda_rd(nd))*(1-kp)/lambda_r**2 + cphi*kpd(nd)/lambda_r
    END DO
! propeller brake region
    fzero = sphi*(1-k) - cphi/lambda_r*(1-kp)
  END IF
END SUBROUTINE INDUCTIONFACTORS_DV
