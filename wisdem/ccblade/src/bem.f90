! Created by S. Andrew Ning
!
! Copyright 2011 NREL
!
! Licensed under the Apache License, Version 2.0 (the "License");
! you may not use this file except in compliance with the License.
! You may obtain a copy of the License at
!
!    http://www.apache.org/licenses/LICENSE-2.0
!
! Unless required by applicable law or agreed to in writing, software
! distributed under the License is distributed on an "AS IS" BASIS,
! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
! See the License for the specific language governing permissions and
! limitations under the License.

subroutine inductionFactors(r, chord, Rhub, Rtip, phi, cl, cd, B, &
    Vx, Vy, useCd, hubLoss, tipLoss, wakerotation, &
    fzero, a, ap)

    implicit none

    integer, parameter :: dp = kind(0.d0)

    ! in
    real(dp), intent(in) :: r, chord, Rhub, Rtip, phi, cl, cd
    integer, intent(in) :: B
    real(dp), intent(in) :: Vx, Vy
    logical, intent(in) :: useCd, hubLoss, tipLoss, wakerotation
    !f2py logical, optional, intent(in) :: useCd = 1, hubLoss = 1, tipLoss = 1, wakerotation = 1

    ! out
    real(dp), intent(out) :: fzero, a, ap

    ! local
    real(dp) :: pi, sigma_p, sphi, cphi, lambda_r
    real(dp) :: factortip, Ftip, factorhub, Fhub
    real(dp) :: k, kp, cn, ct, F
    real(dp) :: g1, g2, g3


    ! constants
    pi = 3.1415926535897932_dp
    sigma_p = B/2.0_dp/pi*chord/r
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
    Ftip = 1.0_dp
    if ( tipLoss ) then
        ! factortip = B/2.0_dp*(Rtip - r)/(r*abs(sphi))
        factortip = B/2.0_dp*(Rtip - r)/(r*sphi)
        Ftip = 2.0_dp/pi*acos(exp(-factortip))
    end if

    Fhub = 1.0_dp
    if ( hubLoss ) then
        ! factorhub = B/2.0_dp*(r - Rhub)/(Rhub*abs(sphi))
        factorhub = B/2.0_dp*(r - Rhub)/(Rhub*sphi)
        Fhub = 2.0_dp/pi*acos(exp(-factorhub))
    end if

    F = Ftip * Fhub

    ! bem parameters
    k = sigma_p*cn/4.0_dp/F/sphi/sphi
    kp = sigma_p*ct/4.0_dp/F/sphi/cphi

    ! compute axial induction factor
    if (phi > 0) then  ! momentum/empirical

        ! update axial induction factor
        if (k <= 2.0_dp/3.0) then  ! momentum state
            a = k/(1+k)

        else  ! Glauert(Buhl) correction

            g1 = 2.0_dp*F*k - (10.0_dp/9-F)
            g2 = 2.0_dp*F*k - (4.0_dp/3-F)*F
            g3 = 2.0_dp*F*k - (25.0_dp/9-2*F)

            if (abs(g3) < 1e-6_dp) then  ! avoid singularity
                a = 1.0_dp - 1.0_dp/2.0/sqrt(g2)
            else
                a = (g1 - sqrt(g2)) / g3
            end if

        end if

    else  ! propeller brake region (a and ap not directly used but update anyway)

        if (k > 1) then
            a = k/(k-1)
        else
            a = 0.0_dp  ! dummy value
        end if

    end if

    ! compute tangential induction factor
    ap = kp/(1-kp)

    if (.not. wakerotation) then
        ap = 0.0_dp
        kp = 0.0_dp
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

    integer, parameter :: dp = kind(0.d0)

    ! in
    real(dp), intent(in) :: phi, a, ap, Vx, Vy, pitch
    real(dp), intent(in) :: chord, theta, rho, mu

    ! out
    real(dp), intent(out) :: alpha, W, Re

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

    integer, parameter :: dp = kind(0.d0)

    ! in
    integer, intent(in) :: n
    real(dp), dimension(n), intent(in) :: r, precurve, presweep
    real(dp), intent(in) :: precone

    ! out
    real(dp), dimension(n), intent(out) :: x_az, y_az, z_az, cone, s

    ! local
    integer :: i


    ! coordinate in azimuthal coordinate system
    ! az_coords = DirectionVector(precurve, presweep, r).bladeToAzimuth(precone)

    x_az = -r*sin(precone) + precurve*cos(precone)
    z_az = r*cos(precone) + precurve*sin(precone)
    y_az = presweep


    ! compute total coning angle for purposes of relative velocity
    cone(1) = atan2(-(x_az(2) - x_az(1)), z_az(2) - z_az(1))
    cone(2:n-1) = 0.5_dp*(atan2(-(x_az(2:n-1) - x_az(1:n-2)), z_az(2:n-1) - z_az(1:n-2)) &
                       + atan2(-(x_az(3:n) - x_az(2:n-1)), z_az(3:n) - z_az(2:n-1)))
    cone(n) = atan2(-(x_az(n) - x_az(n-1)), z_az(n) - z_az(n-1))


    ! total path length of blade
    s(1) = 0.0_dp
    do i = 2, n
        s(i) = s(i-1) + sqrt((precurve(i) - precurve(i-1))**2 + &
            (presweep(i) - presweep(i-1))**2 + (r(i) - r(i-1))**2)
    end do


end subroutine defineCurvature




subroutine windComponents(n, r, precurve, presweep, precone, yaw, tilt, azimuth, &
    Uinf, OmegaRPM, hubHt, shearExp, Vx, Vy)

    implicit none

    integer, parameter :: dp = kind(0.d0)

    ! in
    integer, intent(in) :: n
    real(dp), dimension(n), intent(in) :: r, precurve, presweep
    real(dp), intent(in) :: precone, yaw, tilt, azimuth, Uinf, OmegaRPM, hubHt, shearExp

    ! out
    real(dp), dimension(n), intent(out) :: Vx, Vy

    ! local
    real(dp) :: sy, cy, st, ct, sa, ca, pi, Omega
    real(dp), dimension(n) :: cone, sc, cc, x_az, y_az, z_az, sint
    real(dp), dimension(n) :: heightFromHub, V, Vwind_x, Vwind_y, Vrot_x, Vrot_y


    ! rename
    sy = sin(yaw)
    cy = cos(yaw)
    st = sin(tilt)
    ct = cos(tilt)
    sa = sin(azimuth)
    ca = cos(azimuth)
    pi = 3.1415926535897932_dp
    Omega = OmegaRPM * pi/30.0_dp


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
    Rhub, Rtip, precurveTip, presweepTip, T, Q, M)

    implicit none

    integer, parameter :: dp = kind(0.d0)

    ! in
    integer, intent(in) :: n
    real(dp), dimension(n), intent(in) :: Np, Tp, r, precurve, presweep
    real(dp), intent(in) :: precone, Rhub, Rtip, precurveTip, presweepTip

    ! out
    real(dp), intent(out) :: T, Q, M

    ! local
    real(dp) :: ds
    real(dp), dimension(n+2) :: rfull, curvefull, sweepfull, Npfull, Tpfull
    real(dp), dimension(n+2) :: thrust, torque, flap_moment, x_az, y_az, z_az, cone, s
    integer :: i


    ! add hub/tip for complete integration.  loads go to zero at hub/tip.
    rfull(1) = Rhub
    rfull(2:n+1) = r
    rfull(n+2) = Rtip

    curvefull(1) = 0.0_dp
    curvefull(2:n+1) = precurve
    curvefull(n+2) = precurveTip

    sweepfull(1) = 0.0_dp
    sweepfull(2:n+1) = presweep
    sweepfull(n+2) = presweepTip

    Npfull(1) = 0.0_dp
    Npfull(2:n+1) = Np
    Npfull(n+2) = 0.0_dp

    Tpfull(1) = 0.0_dp
    Tpfull(2:n+1) = Tp
    Tpfull(n+2) = 0.0_dp


    ! get z_az and total cone angle
    call defineCurvature(n+2, rfull, curvefull, sweepfull, precone, x_az, y_az, z_az, cone, s)


    ! integrate Thrust and Torque (trapezoidal)
    thrust      = Npfull*cos(cone)
    torque      = Tpfull*z_az
    flap_moment = Npfull*z_az
    
    T = 0.0_dp
    do i = 1, n+1
        ds = s(i+1) - s(i)
        T = T + 0.5_dp*(thrust(i) + thrust(i+1))*ds
        Q = Q + 0.5_dp*(torque(i) + torque(i+1))*ds
        M = M + 0.5_dp*(flap_moment(i) + flap_moment(i+1))*ds
    end do


end subroutine thrustTorque






!        Generated by TAPENADE     (INRIA, Tropics team)
!  Tapenade 3.9 (r5096) - 24 Feb 2014 16:54
!
!  Differentiation of inductionfactors in forward (tangent) mode:
!   variations   of useful results: ap fzero a
!   with respect to varying inputs: r rtip rhub chord phi cd cl
!                vx vy
!   RW status of diff variables: r:in rtip:in ap:out rhub:in chord:in
!                fzero:out phi:in cd:in cl:in vx:in vy:in a:out

SUBROUTINE INDUCTIONFACTORS_DV(r, chord, rhub, rtip, phi, cl, cd, b, &
  vx, vy, usecd, hubloss, tiploss, wakerotation, &
  rd, chordd, rhubd, rtipd, phid, cld, cdd, vxd, vyd, &
  fzero, a, ap, fzerod, ad, apd, nbdirs)
!  Hint: nbdirsmax should be the maximum number of differentiation directions
  IMPLICIT NONE
  INTEGER, PARAMETER :: dp=KIND(0.d0)
! in
  REAL(dp), INTENT(IN) :: r, chord, rhub, rtip, phi, cl, cd
  REAL(dp), DIMENSION(nbdirs), INTENT(IN) :: rd, chordd, rhubd, rtipd&
& , phid, cld, cdd
  INTEGER, INTENT(IN) :: b
  REAL(dp), INTENT(IN) :: vx, vy
  REAL(dp), DIMENSION(nbdirs), INTENT(IN) :: vxd, vyd
  LOGICAL, INTENT(IN) :: usecd, hubloss, tiploss, wakerotation
  INTEGER, intent(in) :: nbdirs
!f2py logical, optional, intent(in) :: useCd = 1, hubLoss = 1, tipLoss = 1, wakerotation = 1
! out
  REAL(dp), INTENT(OUT) :: fzero, a, ap
  REAL(dp), DIMENSION(nbdirs), INTENT(OUT) :: fzerod, ad, apd
! local
  REAL(dp) :: pi, sigma_p, sphi, cphi, lambda_r
  REAL(dp), DIMENSION(nbdirs) :: sigma_pd, sphid, cphid, lambda_rd
  REAL(dp) :: factortip, ftip, factorhub, fhub
  REAL(dp), DIMENSION(nbdirs) :: factortipd, ftipd, factorhubd, fhubd
  REAL(dp) :: k, kp, cn, ct, f
  REAL(dp), DIMENSION(nbdirs) :: kd, kpd, cnd, ctd, fd
  REAL(dp) :: g1, g2, g3
  REAL(dp), DIMENSION(nbdirs) :: g1d, g2d, g3d
  INTRINSIC KIND
  INTRINSIC SIN
  INTRINSIC COS
  INTRINSIC ABS
  INTRINSIC EXP
  INTRINSIC ACOS
  INTRINSIC SQRT
  REAL(dp) :: arg1
  REAL(dp), DIMENSION(nbdirs) :: arg1d
  REAL(dp) :: result1
  REAL(dp), DIMENSION(nbdirs) :: result1d
  INTEGER :: nd
  REAL(dp) :: abs1d(nbdirs)
  REAL(dp) :: abs0d(nbdirs)
  REAL(dp) :: abs2
  REAL(dp) :: abs1
  REAL(dp) :: abs0
! constants
  pi = 3.1415926535897932_dp
  DO nd=1,nbdirs
    sigma_pd(nd) = (b*chordd(nd)*r/(2.0_dp*pi)-b*chord*rd(nd)/(2.0_dp*pi&
&     ))/r**2
    sphid(nd) = phid(nd)*COS(phi)
    cphid(nd) = -(phid(nd)*SIN(phi))
  END DO
  sigma_p = b/2.0_dp/pi*chord/r
  sphi = SIN(phi)
  cphi = COS(phi)
! resolve into normal and tangential forces
  IF (.NOT.usecd) THEN
    DO nd=1,nbdirs
      cnd(nd) = cld(nd)*cphi + cl*cphid(nd)
      ctd(nd) = cld(nd)*sphi + cl*sphid(nd)
    END DO
    cn = cl*cphi
    ct = cl*sphi
  ELSE
    DO nd=1,nbdirs
      cnd(nd) = cld(nd)*cphi + cl*cphid(nd) + cdd(nd)*sphi + cd*sphid(nd&
&       )
      ctd(nd) = cld(nd)*sphi + cl*sphid(nd) - cdd(nd)*cphi - cd*cphid(nd&
&       )
    END DO
    cn = cl*cphi + cd*sphi
    ct = cl*sphi - cd*cphi
  END IF
! Prandtl's tip and hub loss factor
  ftip = 1.0_dp
  IF (tiploss) THEN
    IF (sphi .GE. 0.) THEN
      DO nd=1,nbdirs
        abs0d(nd) = sphid(nd)
      END DO
      abs0 = sphi
    ELSE
      DO nd=1,nbdirs
        abs0d(nd) = -sphid(nd)
      END DO
      abs0 = -sphi
    END IF
    factortip = b/2.0_dp*(rtip-r)/(r*abs0)
    arg1 = EXP(-factortip)
    DO nd=1,nbdirs
      factortipd(nd) = (b*(rtipd(nd)-rd(nd))*r*abs0/2.0_dp-b*(rtip-r)*(&
&       rd(nd)*abs0+r*abs0d(nd))/2.0_dp)/(r*abs0)**2
      arg1d(nd) = -(factortipd(nd)*EXP(-factortip))
      IF (arg1 .EQ. 1.0 .OR. arg1 .EQ. (-1.0)) THEN
        result1d(nd) = 0.0
      ELSE
        result1d(nd) = -(arg1d(nd)/SQRT(1.0-arg1**2))
      END IF
      ftipd(nd) = 2.0_dp*result1d(nd)/pi
    END DO
    result1 = ACOS(arg1)
    ftip = 2.0_dp/pi*result1
  ELSE
    DO nd=1,nbdirs
      ftipd(nd) = 0.0
    END DO
  END IF
  fhub = 1.0_dp
  IF (hubloss) THEN
    IF (sphi .GE. 0.) THEN
      DO nd=1,nbdirs
        abs1d(nd) = sphid(nd)
      END DO
      abs1 = sphi
    ELSE
      DO nd=1,nbdirs
        abs1d(nd) = -sphid(nd)
      END DO
      abs1 = -sphi
    END IF
    factorhub = b/2.0_dp*(r-rhub)/(rhub*abs1)
    arg1 = EXP(-factorhub)
    DO nd=1,nbdirs
      factorhubd(nd) = (b*(rd(nd)-rhubd(nd))*rhub*abs1/2.0_dp-b*(r-rhub)&
&       *(rhubd(nd)*abs1+rhub*abs1d(nd))/2.0_dp)/(rhub*abs1)**2
      arg1d(nd) = -(factorhubd(nd)*EXP(-factorhub))
      IF (arg1 .EQ. 1.0 .OR. arg1 .EQ. (-1.0)) THEN
        result1d(nd) = 0.0
      ELSE
        result1d(nd) = -(arg1d(nd)/SQRT(1.0-arg1**2))
      END IF
      fhubd(nd) = 2.0_dp*result1d(nd)/pi
    END DO
    result1 = ACOS(arg1)
    fhub = 2.0_dp/pi*result1
  ELSE
    DO nd=1,nbdirs
      fhubd(nd) = 0.0
    END DO
  END IF
  f = ftip*fhub
  DO nd=1,nbdirs
    fd(nd) = ftipd(nd)*fhub + ftip*fhubd(nd)
! bem parameters
    kd(nd) = ((((sigma_pd(nd)*cn+sigma_p*cnd(nd))*f/4.0_dp-sigma_p*cn*fd&
&     (nd)/4.0_dp)*sphi/f**2-sigma_p*cn*sphid(nd)/(4.0_dp*f))/sphi-&
&     sigma_p*cn*sphid(nd)/(4.0_dp*f*sphi))/sphi**2
    kpd(nd) = ((((sigma_pd(nd)*ct+sigma_p*ctd(nd))*f/4.0_dp-sigma_p*ct*&
&     fd(nd)/4.0_dp)*sphi/f**2-sigma_p*ct*sphid(nd)/(4.0_dp*f))*cphi/&
&     sphi**2-sigma_p*ct*cphid(nd)/(4.0_dp*f*sphi))/cphi**2
  END DO
  k = sigma_p*cn/4.0_dp/f/sphi/sphi
  kp = sigma_p*ct/4.0_dp/f/sphi/cphi
! compute axial induction factor
  IF (phi .GT. 0) THEN
! momentum/empirical
! update axial induction factor
    IF (k .LE. 2.0_dp/3.0) THEN
      DO nd=1,nbdirs
! momentum state
        ad(nd) = (kd(nd)*(1+k)-k*kd(nd))/(1+k)**2
      END DO
      a = k/(1+k)
    ELSE
      DO nd=1,nbdirs
! Glauert(Buhl) correction
        g1d(nd) = 2.0_dp*(fd(nd)*k+f*kd(nd)) + fd(nd)
        g2d(nd) = 2.0_dp*(fd(nd)*k+f*kd(nd)) - (4.0_dp/3-f)*fd(nd) + fd(&
&         nd)*f
        g3d(nd) = 2.0_dp*(fd(nd)*k+f*kd(nd)) + 2*fd(nd)
      END DO
      g1 = 2.0_dp*f*k - (10.0_dp/9-f)
      g2 = 2.0_dp*f*k - (4.0_dp/3-f)*f
      g3 = 2.0_dp*f*k - (25.0_dp/9-2*f)
      IF (g3 .GE. 0.) THEN
        abs2 = g3
      ELSE
        abs2 = -g3
      END IF
      IF (abs2 .LT. 1e-6_dp) THEN
        result1 = SQRT(g2)
        DO nd=1,nbdirs
! avoid singularity
          IF (g2 .EQ. 0.0) THEN
            result1d(nd) = 0.0
          ELSE
            result1d(nd) = g2d(nd)/(2.0*SQRT(g2))
          END IF
          ad(nd) = result1d(nd)/2.0/result1**2
        END DO
        a = 1.0_dp - 1.0_dp/2.0/result1
      ELSE
        result1 = SQRT(g2)
        DO nd=1,nbdirs
          IF (g2 .EQ. 0.0) THEN
            result1d(nd) = 0.0
          ELSE
            result1d(nd) = g2d(nd)/(2.0*SQRT(g2))
          END IF
          ad(nd) = ((g1d(nd)-result1d(nd))*g3-(g1-result1)*g3d(nd))/g3**&
&           2
        END DO
        a = (g1-result1)/g3
      END IF
    END IF
  ELSE IF (k .GT. 1) THEN
! propeller brake region (a and ap not directly used but update anyway)
    DO nd=1,nbdirs
      ad(nd) = (kd(nd)*(k-1)-k*kd(nd))/(k-1)**2
    END DO
    a = k/(k-1)
  ELSE
! dummy value
    a = 0.0_dp
    DO nd=1,nbdirs
      ad(nd) = 0.0
    END DO
  END IF
  DO nd=1,nbdirs
! compute tangential induction factor
    apd(nd) = (kpd(nd)*(1-kp)+kp*kpd(nd))/(1-kp)**2
  END DO
  ap = kp/(1-kp)
  IF (.NOT.wakerotation) THEN
    ap = 0.0_dp
    kp = 0.0_dp
    DO nd=1,nbdirs
      apd(nd) = 0.0
      kpd(nd) = 0.0
    END DO
  END IF
  DO nd=1,nbdirs
! error function
    lambda_rd(nd) = (vyd(nd)*vx-vy*vxd(nd))/vx**2
  END DO
  lambda_r = vy/vx
  IF (phi .GT. 0) THEN
    DO nd=1,nbdirs
! momentum/empirical
      fzerod(nd) = (sphid(nd)*(1-a)+sphi*ad(nd))/(1-a)**2 - (cphid(nd)*&
&       lambda_r-cphi*lambda_rd(nd))*(1-kp)/lambda_r**2 + cphi*kpd(nd)/&
&       lambda_r
    END DO
    fzero = sphi/(1-a) - cphi/lambda_r*(1-kp)
  ELSE
    DO nd=1,nbdirs
! propeller brake region
      fzerod(nd) = sphid(nd)*(1-k) - sphi*kd(nd) - (cphid(nd)*lambda_r-&
&       cphi*lambda_rd(nd))*(1-kp)/lambda_r**2 + cphi*kpd(nd)/lambda_r
    END DO
    fzero = sphi*(1-k) - cphi/lambda_r*(1-kp)
  END IF
END SUBROUTINE INDUCTIONFACTORS_DV




!        Generated by TAPENADE     (INRIA, Tropics team)
!  Tapenade 3.9 (r5096) - 24 Feb 2014 16:54
!
!  Differentiation of relativewind in forward (tangent) mode:
!   variations   of useful results: alpha w re
!   with respect to varying inputs: ap chord theta pitch phi vx
!                vy a
!   RW status of diff variables: alpha:out w:out ap:in re:out chord:in
!                theta:in pitch:in phi:in vx:in vy:in a:in

SUBROUTINE RELATIVEWIND_DV(phi, phid, a, ad, ap, apd, vx, vxd, vy, vyd, &
& pitch, pitchd, chord, chordd, theta, thetad, rho, mu, alpha, alphad, w&
& , wd, re, red, nbdirs)
!  Hint: nbdirsmax should be the maximum number of differentiation directions
  IMPLICIT NONE
  INTEGER, PARAMETER :: dp=KIND(0.d0)
! in
  REAL(dp), INTENT(IN) :: phi, a, ap, vx, vy, pitch
  REAL(dp), DIMENSION(nbdirs), INTENT(IN) :: phid, ad, apd, vxd, vyd&
& , pitchd
  REAL(dp), INTENT(IN) :: chord, theta, rho, mu
  REAL(dp), DIMENSION(nbdirs), INTENT(IN) :: chordd, thetad
  INTEGER, intent(in) :: nbdirs
! out
  REAL(dp), INTENT(OUT) :: alpha, w, re
  REAL(dp), DIMENSION(nbdirs), INTENT(OUT) :: alphad, wd, red
  INTRINSIC KIND
  INTRINSIC ABS
  INTRINSIC COS
  INTRINSIC SIN
  INTRINSIC SQRT
  REAL(dp) :: arg1
  REAL(dp), DIMENSION(nbdirs) :: arg1d
  INTEGER :: nd
  REAL(dp) :: abs1
  REAL(dp) :: abs0
  DO nd=1,nbdirs
! angle of attack
    alphad(nd) = phid(nd) - thetad(nd) - pitchd(nd)
  END DO
  alpha = phi - (theta+pitch)
  IF (a .GE. 0.) THEN
    abs0 = a
  ELSE
    abs0 = -a
  END IF
! avoid numerical errors when angle is close to 0 or 90 deg
! and other induction factor is at some ridiculous value
! this only occurs when iterating on Reynolds number
! during the phi sweep where a solution has not been found yet
  IF (abs0 .GT. 10) THEN
    DO nd=1,nbdirs
      wd(nd) = ((vyd(nd)*(1+ap)+vy*apd(nd))*COS(phi)+vy*(1+ap)*phid(nd)*&
&       SIN(phi))/COS(phi)**2
    END DO
    w = vy*(1+ap)/COS(phi)
  ELSE
    IF (ap .GE. 0.) THEN
      abs1 = ap
    ELSE
      abs1 = -ap
    END IF
    IF (abs1 .GT. 10) THEN
      DO nd=1,nbdirs
        wd(nd) = ((vxd(nd)*(1-a)-vx*ad(nd))*SIN(phi)-vx*(1-a)*phid(nd)*&
&         COS(phi))/SIN(phi)**2
      END DO
      w = vx*(1-a)/SIN(phi)
    ELSE
      arg1 = (vx*(1-a))**2 + (vy*(1+ap))**2
      DO nd=1,nbdirs
        arg1d(nd) = 2*vx*(1-a)*(vxd(nd)*(1-a)-vx*ad(nd)) + 2*vy*(1+ap)*(&
&         vyd(nd)*(1+ap)+vy*apd(nd))
        IF (arg1 .EQ. 0.0) THEN
          wd(nd) = 0.0
        ELSE
          wd(nd) = arg1d(nd)/(2.0*SQRT(arg1))
        END IF
      END DO
      w = SQRT(arg1)
    END IF
  END IF
  DO nd=1,nbdirs
    red(nd) = rho*(wd(nd)*chord+w*chordd(nd))/mu
  END DO
  re = rho*w*chord/mu
END SUBROUTINE RELATIVEWIND_DV



!        Generated by TAPENADE     (INRIA, Tropics team)
!  Tapenade 3.9 (r5096) - 24 Feb 2014 16:54
!
!  Differentiation of windcomponents in forward (tangent) mode:
!   variations   of useful results: vx vy
!   with respect to varying inputs: yaw r azimuth precurve tilt
!                presweep hubht omegarpm uinf precone
!   RW status of diff variables: yaw:in r:in azimuth:in precurve:in
!                tilt:in presweep:in hubht:in omegarpm:in uinf:in
!                vx:out vy:out precone:in

SUBROUTINE WINDCOMPONENTS_DV(n, r, rd, precurve, precurved, presweep, &
& presweepd, precone, preconed, yaw, yawd, tilt, tiltd, azimuth, &
& azimuthd, uinf, uinfd, omegarpm, omegarpmd, hubht, hubhtd, shearexp, &
& vx, vxd, vy, vyd, nbdirs)
!  Hint: nbdirsmax should be the maximum number of differentiation directions
  IMPLICIT NONE
  INTEGER, PARAMETER :: dp=KIND(0.d0)
! in
  INTEGER, INTENT(IN) :: n
  REAL(dp), DIMENSION(n), INTENT(IN) :: r, precurve, presweep
  REAL(dp), DIMENSION(nbdirs, n), INTENT(IN) :: rd, precurved, &
& presweepd
  REAL(dp), INTENT(IN) :: precone, yaw, tilt, azimuth, uinf, omegarpm, &
& hubht, shearexp
  REAL(dp), DIMENSION(nbdirs), INTENT(IN) :: preconed, yawd, tiltd, &
& azimuthd, uinfd, omegarpmd, hubhtd
  INTEGER, intent(in) :: nbdirs
! out
  REAL(dp), DIMENSION(n), INTENT(OUT) :: vx, vy
  REAL(dp), DIMENSION(nbdirs, n), INTENT(OUT) :: vxd, vyd
! local
  REAL(dp) :: sy, cy, st, ct, sa, ca, pi, omega
  REAL(dp), DIMENSION(nbdirs) :: syd, cyd, std, ctd, sad, cad, omegad
  REAL(dp), DIMENSION(n) :: cone, sc, cc, x_az, y_az, z_az, sint
  REAL(dp), DIMENSION(nbdirs, n) :: coned, scd, ccd, x_azd, y_azd, &
& z_azd
  REAL(dp), DIMENSION(n) :: heightfromhub, v, vwind_x, vwind_y, vrot_x, &
& vrot_y
  REAL(dp), DIMENSION(nbdirs, n) :: heightfromhubd, vd, vwind_xd, &
& vwind_yd, vrot_xd, vrot_yd
  INTRINSIC KIND
  INTRINSIC SIN
  INTRINSIC COS
  REAL(dp), DIMENSION(n) :: pwx1
  REAL(dp), DIMENSION(nbdirs, n) :: pwx1d
  REAL(dp), DIMENSION(n) :: pwr1
  REAL(dp), DIMENSION(nbdirs, n) :: pwr1d
  INTEGER :: nd
  sy = SIN(yaw)
  cy = COS(yaw)
  st = SIN(tilt)
  ct = COS(tilt)
  sa = SIN(azimuth)
  ca = COS(azimuth)
  pi = 3.1415926535897932_dp
  omega = omegarpm*pi/30.0_dp
  CALL DEFINECURVATURE_DV(n, r, rd, precurve, precurved, presweep, &
&                   presweepd, precone, preconed, x_az, x_azd, y_az, &
&                   y_azd, z_az, z_azd, cone, coned, sint, nbdirs)
  sc = SIN(cone)
  cc = COS(cone)
  heightfromhub = (y_az*sa+z_az*ca)*ct - x_az*st
  pwx1 = 1 + heightfromhub/hubht
  pwr1 = pwx1**shearexp
  v = uinf*pwr1
  DO nd=1,nbdirs
! rename
    syd(nd) = yawd(nd)*COS(yaw)
    cyd(nd) = -(yawd(nd)*SIN(yaw))
    std(nd) = tiltd(nd)*COS(tilt)
    ctd(nd) = -(tiltd(nd)*SIN(tilt))
    sad(nd) = azimuthd(nd)*COS(azimuth)
    cad(nd) = -(azimuthd(nd)*SIN(azimuth))
    omegad(nd) = pi*omegarpmd(nd)/30.0_dp
    scd(nd, :) = coned(nd, :)*COS(cone)
    ccd(nd, :) = -(coned(nd, :)*SIN(cone))
! get section heights in wind-aligned coordinate system
! heightFromHub = az_coords.azimuthToHub(azimuth).hubToYaw(tilt).z
    heightfromhubd(nd, :) = (y_azd(nd, :)*sa+y_az*sad(nd)+z_azd(nd, :)*&
&     ca+z_az*cad(nd))*ct + (y_az*sa+z_az*ca)*ctd(nd) - x_azd(nd, :)*st &
&     - x_az*std(nd)
! velocity with shear
    pwx1d(nd, :) = (heightfromhubd(nd, :)*hubht-heightfromhub*hubhtd(nd)&
&     )/hubht**2
    WHERE (pwx1 .GT. 0.0 .OR. (pwx1 .LT. 0.0 .AND. shearexp .EQ. INT(&
&       shearexp)))
      pwr1d(nd, :) = shearexp*pwx1**(shearexp-1)*pwx1d(nd, :)
    ELSEWHERE (pwx1 .EQ. 0.0 .AND. shearexp .EQ. 1.0)
      pwr1d(nd, :) = pwx1d(nd, :)
    ELSEWHERE
      pwr1d(nd, :) = 0.0
    END WHERE
    vd(nd, :) = uinfd(nd)*pwr1 + uinf*pwr1d(nd, :)
! transform wind to blade c.s.
! Vwind = DirectionVector(V, 0*V, 0*V).windToYaw(yaw).yawToHub(tilt).hubToAzimuth(azimuth).azimuthToBlade(cone)
    vwind_xd(nd, :) = vd(nd, :)*((cy*st*ca+sy*sa)*sc+cy*ct*cc) + v*(((&
&     cyd(nd)*st+cy*std(nd))*ca+cy*st*cad(nd)+syd(nd)*sa+sy*sad(nd))*sc+&
&     (cy*st*ca+sy*sa)*scd(nd, :)+(cyd(nd)*ct+cy*ctd(nd))*cc+cy*ct*ccd(&
&     nd, :))
    vwind_yd(nd, :) = vd(nd, :)*(cy*st*sa-sy*ca) + v*((cyd(nd)*st+cy*std&
&     (nd))*sa+cy*st*sad(nd)-syd(nd)*ca-sy*cad(nd))
! wind from rotation to blade c.s.
! OmegaV = DirectionVector(Omega, 0.0, 0.0)
! Vrot = -OmegaV.cross(az_coords)  # negative sign because relative wind opposite to rotation
! Vrot = Vrot.azimuthToBlade(cone)
    vrot_xd(nd, :) = -((omegad(nd)*y_az+omega*y_azd(nd, :))*sc+omega*&
&     y_az*scd(nd, :))
    vrot_yd(nd, :) = omegad(nd)*z_az + omega*z_azd(nd, :)
! total velocity
    vxd(nd, :) = vwind_xd(nd, :) + vrot_xd(nd, :)
    vyd(nd, :) = vwind_yd(nd, :) + vrot_yd(nd, :)
  END DO
  vwind_x = v*((cy*st*ca+sy*sa)*sc+cy*ct*cc)
  vwind_y = v*(cy*st*sa-sy*ca)
  vrot_x = -(omega*y_az*sc)
  vrot_y = omega*z_az
  vx = vwind_x + vrot_x
  vy = vwind_y + vrot_y
END SUBROUTINE WINDCOMPONENTS_DV

!  Differentiation of definecurvature in forward (tangent) mode:
!   variations   of useful results: z_az y_az x_az cone
!   with respect to varying inputs: r precurve presweep precone

SUBROUTINE DEFINECURVATURE_DV(n, r, rd, precurve, precurved, presweep, &
& presweepd, precone, preconed, x_az, x_azd, y_az, y_azd, z_az, z_azd, &
& cone, coned, s, nbdirs)
!  Hint: nbdirs should be the maximum number of differentiation directions
  IMPLICIT NONE
  INTEGER, PARAMETER :: dp=KIND(0.d0)
! in
  INTEGER, INTENT(IN) :: n
  REAL(dp), DIMENSION(n), INTENT(IN) :: r, precurve, presweep
  REAL(dp), DIMENSION(nbdirs, n), INTENT(IN) :: rd, precurved, &
& presweepd
  REAL(dp), INTENT(IN) :: precone
  REAL(dp), DIMENSION(nbdirs), INTENT(IN) :: preconed
  INTEGER, intent(in) :: nbdirs
! out
  REAL(dp), DIMENSION(n), INTENT(OUT) :: x_az, y_az, z_az, cone, s
  REAL(dp), DIMENSION(nbdirs, n), INTENT(OUT) :: x_azd, y_azd, z_azd&
& , coned
! local
  INTEGER :: i
  INTRINSIC KIND
  INTRINSIC SIN
  INTRINSIC COS
  INTRINSIC ATAN2
  INTRINSIC SQRT
  REAL(dp) :: arg1
  REAL(dp), DIMENSION(nbdirs) :: arg1d
  REAL(dp) :: arg2
  REAL(dp), DIMENSION(nbdirs) :: arg2d
  REAL(dp), DIMENSION(n-2) :: arg10
  REAL(dp), DIMENSION(nbdirs, n-2) :: arg10d
  REAL(dp), DIMENSION(n-2) :: arg20
  REAL(dp), DIMENSION(nbdirs, n-2) :: arg20d
  REAL(dp), DIMENSION(n-2) :: arg3
  REAL(dp), DIMENSION(nbdirs, n-2) :: arg3d
  REAL(dp), DIMENSION(n-2) :: arg4
  REAL(dp), DIMENSION(nbdirs, n-2) :: arg4d
  REAL(dp) :: result1
  INTEGER :: nd
  x_az = -(r*SIN(precone)) + precurve*COS(precone)
  z_az = r*COS(precone) + precurve*SIN(precone)
  arg1 = -(x_az(2)-x_az(1))
  arg2 = z_az(2) - z_az(1)
  arg10(:) = -(x_az(2:n-1)-x_az(1:n-2))
  arg20(:) = z_az(2:n-1) - z_az(1:n-2)
  arg3(:) = -(x_az(3:n)-x_az(2:n-1))
  arg4(:) = z_az(3:n) - z_az(2:n-1)
  DO nd=1,nbdirs
! coordinate in azimuthal coordinate system
! az_coords = DirectionVector(precurve, presweep, r).bladeToAzimuth(precone)
    x_azd(nd, :) = precurved(nd, :)*COS(precone) - r*preconed(nd)*COS(&
&     precone) - rd(nd, :)*SIN(precone) - precurve*preconed(nd)*SIN(&
&     precone)
    z_azd(nd, :) = rd(nd, :)*COS(precone) - r*preconed(nd)*SIN(precone) &
&     + precurved(nd, :)*SIN(precone) + precurve*preconed(nd)*COS(&
&     precone)
    y_azd(nd, :) = presweepd(nd, :)
! compute total coning angle for purposes of relative velocity
    arg1d(nd) = -(x_azd(nd, 2)-x_azd(nd, 1))
    arg2d(nd) = z_azd(nd, 2) - z_azd(nd, 1)
    coned(nd, :) = 0.0
    coned(nd, 1) = (arg1d(nd)*arg2-arg2d(nd)*arg1)/(arg1**2+arg2**2)
    arg10d(nd, :) = -(x_azd(nd, 2:n-1)-x_azd(nd, 1:n-2))
    arg20d(nd, :) = z_azd(nd, 2:n-1) - z_azd(nd, 1:n-2)
    arg3d(nd, :) = -(x_azd(nd, 3:n)-x_azd(nd, 2:n-1))
    arg4d(nd, :) = z_azd(nd, 3:n) - z_azd(nd, 2:n-1)
    coned(nd, 2:n-1) = 0.5_dp*((arg10d(nd, :)*arg20(:)-arg20d(nd, :)*&
&     arg10(:))/(arg10(:)**2+arg20(:)**2)+(arg3d(nd, :)*arg4(:)-arg4d(nd&
&     , :)*arg3(:))/(arg3(:)**2+arg4(:)**2))
    arg1d(nd) = -(x_azd(nd, n)-x_azd(nd, n-1))
    arg2d(nd) = z_azd(nd, n) - z_azd(nd, n-1)
  END DO
  y_az = presweep
  cone(1) = ATAN2(arg1, arg2)
  cone(2:n-1) = 0.5_dp*(ATAN2(arg10(:), arg20(:))+ATAN2(arg3(:), arg4(:)&
&   ))
  arg1 = -(x_az(n)-x_az(n-1))
  arg2 = z_az(n) - z_az(n-1)
  DO nd=1,nbdirs
    coned(nd, n) = (arg1d(nd)*arg2-arg2d(nd)*arg1)/(arg1**2+arg2**2)
  END DO
  cone(n) = ATAN2(arg1, arg2)
! total path length of blade
  s(1) = 0.0_dp
  DO i=2,n
    arg1 = (precurve(i)-precurve(i-1))**2 + (presweep(i)-presweep(i-1))&
&     **2 + (r(i)-r(i-1))**2
    result1 = SQRT(arg1)
    s(i) = s(i-1) + result1
  END DO
END SUBROUTINE DEFINECURVATURE_DV






!        Generated by TAPENADE     (INRIA, Tropics team)
!  Tapenade 3.9 (r5096) - 24 Feb 2014 16:54
!
!  Differentiation of thrusttorque in reverse (adjoint) mode:
!   gradient     of useful results: q t
!   with respect to varying inputs: tp precurvetip q r t rtip np
!                precurve presweep presweeptip rhub precone
!   RW status of diff variables: tp:out precurvetip:out q:in-out
!                r:out t:in-zero rtip:out np:out precurve:out presweep:out
!                presweeptip:out rhub:out precone:out

SUBROUTINE THRUSTTORQUE_BV(n, np, tp, r, precurve, presweep, precone, &
  rhub, rtip, precurvetip, presweeptip, tb, qb, &
  npb, tpb, rb, precurveb, presweepb, preconeb, rhubb, rtipb, &
  precurvetipb, presweeptipb, nbdirs)
!  Hint: nbdirsmax should be the maximum number of differentiation directions
  IMPLICIT NONE
  INTEGER, PARAMETER :: dp=KIND(0.d0)
! in
  INTEGER, INTENT(IN) :: n
  REAL(dp), DIMENSION(n), INTENT(IN) :: np, tp, r, precurve, presweep
  REAL(dp), INTENT(IN) :: precone, rhub, rtip, precurvetip, presweeptip
  REAL(dp), DIMENSION(nbdirs), intent(in) :: tb, qb
  INTEGER, intent(in) :: nbdirs
! out
!   REAL(dp) :: t, q
  REAL(dp), DIMENSION(nbdirs, n), intent(out) :: npb, tpb, rb, precurveb, &
& presweepb
  REAL(dp), DIMENSION(nbdirs), intent(out) :: preconeb, rhubb, rtipb, precurvetipb&
& , presweeptipb
! local
  REAL(dp) :: ds
  REAL(dp), DIMENSION(nbdirs) :: dsb
  REAL(dp), DIMENSION(n + 2) :: rfull, curvefull, sweepfull, npfull, &
& tpfull
  REAL(dp), DIMENSION(nbdirs, n+2) :: rfullb, curvefullb, sweepfullb&
& , npfullb, tpfullb
  REAL(dp), DIMENSION(n + 2) :: thrust, torque, x_az, y_az, z_az, cone, &
& s
  REAL(dp), DIMENSION(nbdirs, n+2) :: thrustb, torqueb, x_azb, z_azb&
& , coneb, sb
  INTEGER :: i
  INTRINSIC KIND
  INTRINSIC COS
  INTEGER :: arg1
  INTEGER :: nd
  REAL(dp) :: tempb0(nbdirs)
  REAL(dp) :: tempb(nbdirs)
! add hub/tip for complete integration.  loads go to zero at hub/tip.
  rfull(1) = rhub
  rfull(2:n+1) = r
  rfull(n+2) = rtip
  curvefull(1) = 0.0_dp
  curvefull(2:n+1) = precurve
  curvefull(n+2) = precurvetip
  sweepfull(1) = 0.0_dp
  sweepfull(2:n+1) = presweep
  sweepfull(n+2) = presweeptip
  npfull(1) = 0.0_dp
  npfull(2:n+1) = np
  npfull(n+2) = 0.0_dp
  tpfull(1) = 0.0_dp
  tpfull(2:n+1) = tp
  tpfull(n+2) = 0.0_dp
! get z_az and total cone angle
  arg1 = n + 2
  CALL DEFINECURVATURE(arg1, rfull, curvefull, sweepfull, precone, x_az&
&                , y_az, z_az, cone, s)
! integrate Thrust and Torque (trapezoidal)
  thrust = npfull*COS(cone)
  torque = tpfull*z_az
  DO nd=1,nbdirs
    sb(nd, :) = 0.0
    torqueb(nd, :) = 0.0
    thrustb(nd, :) = 0.0
  END DO
  DO i=n+1,1,-1
    ds = s(i+1) - s(i)
    DO nd=1,nbdirs
      tempb(nd) = 0.5_dp*ds*qb(nd)
      torqueb(nd, i) = torqueb(nd, i) + tempb(nd)
      torqueb(nd, i+1) = torqueb(nd, i+1) + tempb(nd)
      dsb(nd) = 0.5_dp*(thrust(i)+thrust(i+1))*tb(nd) + 0.5_dp*(torque(i&
&       )+torque(i+1))*qb(nd)
      tempb0(nd) = 0.5_dp*ds*tb(nd)
      thrustb(nd, i) = thrustb(nd, i) + tempb0(nd)
      thrustb(nd, i+1) = thrustb(nd, i+1) + tempb0(nd)
      sb(nd, i+1) = sb(nd, i+1) + dsb(nd)
      sb(nd, i) = sb(nd, i) - dsb(nd)
    END DO
  END DO
  DO nd=1,nbdirs
    z_azb(nd, :) = 0.0
    tpfullb(nd, :) = 0.0
    tpfullb(nd, :) = z_az*torqueb(nd, :)
    z_azb(nd, :) = tpfull*torqueb(nd, :)
    npfullb(nd, :) = 0.0
    coneb(nd, :) = 0.0
    npfullb(nd, :) = COS(cone)*thrustb(nd, :)
    coneb(nd, :) = -(SIN(cone)*npfull*thrustb(nd, :))
    tpfullb(nd, n+2) = 0.0
    tpb(nd, :) = 0.0
    tpb(nd, :) = tpfullb(nd, 2:n+1)
    npfullb(nd, n+2) = 0.0
    npb(nd, :) = 0.0
    npb(nd, :) = npfullb(nd, 2:n+1)
    presweepb(nd, :) = 0.0
    precurveb(nd, :) = 0.0
    rb(nd, :) = 0.0
  END DO
  CALL DEFINECURVATURE_BV(arg1, rfull, rfullb, curvefull, curvefullb, &
&                   sweepfull, sweepfullb, precone, preconeb, x_az, &
&                   x_azb, y_az, z_az, z_azb, cone, coneb, s, sb, nbdirs&
&                  )
  DO nd=1,nbdirs
    presweeptipb(nd) = sweepfullb(nd, n+2)
    sweepfullb(nd, n+2) = 0.0
    presweepb(nd, :) = sweepfullb(nd, 2:n+1)
    precurvetipb(nd) = curvefullb(nd, n+2)
    curvefullb(nd, n+2) = 0.0
    precurveb(nd, :) = curvefullb(nd, 2:n+1)
    rtipb(nd) = rfullb(nd, n+2)
    rfullb(nd, n+2) = 0.0
    rb(nd, :) = rfullb(nd, 2:n+1)
    rfullb(nd, 2:n+1) = 0.0
    rhubb(nd) = rfullb(nd, 1)
  END DO
!   DO nd=1,nbdirs
!     tb(nd) = 0.0
!   END DO
END SUBROUTINE THRUSTTORQUE_BV

!  Differentiation of definecurvature in reverse (adjoint) mode:
!   gradient     of useful results: s z_az cone
!   with respect to varying inputs: r precurve presweep precone

SUBROUTINE DEFINECURVATURE_BV(n, r, rb, precurve, precurveb, presweep, &
& presweepb, precone, preconeb, x_az, x_azb, y_az, z_az, z_azb, cone, &
& coneb, s, sb, nbdirs)
!  Hint: nbdirsmax should be the maximum number of differentiation directions
  IMPLICIT NONE
  INTEGER, PARAMETER :: dp=KIND(0.d0)
! in
  INTEGER, INTENT(IN) :: n
  REAL(dp), DIMENSION(n), INTENT(IN) :: r, precurve, presweep
  REAL(dp), DIMENSION(nbdirs, n) :: rb, precurveb, presweepb
  REAL(dp), INTENT(IN) :: precone
  REAL(dp), DIMENSION(nbdirs) :: preconeb
  INTEGER, intent(in) :: nbdirs
! out
  REAL(dp), DIMENSION(n) :: x_az, y_az, z_az, cone, s
  REAL(dp), DIMENSION(nbdirs, n) :: x_azb, z_azb, coneb, sb
! local
  INTEGER :: i
  INTRINSIC KIND
  INTRINSIC SIN
  INTRINSIC COS
  INTRINSIC ATAN2
  INTRINSIC SQRT
  INTEGER :: nd
  REAL(dp) :: temp0(n-2)
  REAL(dp) :: tempb9(nbdirs)
  REAL(dp) :: tempb8(nbdirs)
  REAL(dp) :: tempb7(nbdirs)
  REAL(dp) :: tempb6(nbdirs)
  REAL(dp) :: tempb5(nbdirs, n-2)
  REAL(dp) :: tempb4(nbdirs, n-2)
  REAL(dp) :: tempb3(nbdirs, n-2)
  REAL(dp) :: tempb2(nbdirs, n-2)
  REAL(dp) :: tempb1(nbdirs, n)
  REAL(dp) :: tempb0(nbdirs)
  REAL(dp) :: tempb11(nbdirs)
  REAL(dp) :: tempb10(nbdirs)
  REAL(dp) :: tempb(nbdirs)
  REAL(dp) :: temp(n-2)
! coordinate in azimuthal coordinate system
! az_coords = DirectionVector(precurve, presweep, r).bladeToAzimuth(precone)
  x_az = -(r*SIN(precone)) + precurve*COS(precone)
  z_az = r*COS(precone) + precurve*SIN(precone)
! compute total coning angle for purposes of relative velocity
! total path length of blade
  DO nd=1,nbdirs
    rb(nd, :) = 0.0
    precurveb(nd, :) = 0.0
    presweepb(nd, :) = 0.0
  END DO
  DO i=n,2,-1
    DO nd=1,nbdirs
      IF ((precurve(i)-precurve(i-1))**2 + (presweep(i)-presweep(i-1))**&
&         2 + (r(i)-r(i-1))**2 .EQ. 0.0) THEN
        tempb8(nd) = 0.0
      ELSE
        tempb8(nd) = sb(nd, i)/(2.0*SQRT((precurve(i)-precurve(i-1))**2+&
&         (presweep(i)-presweep(i-1))**2+(r(i)-r(i-1))**2))
      END IF
      tempb9(nd) = 2*(precurve(i)-precurve(i-1))*tempb8(nd)
      tempb10(nd) = 2*(presweep(i)-presweep(i-1))*tempb8(nd)
      tempb11(nd) = 2*(r(i)-r(i-1))*tempb8(nd)
      sb(nd, i-1) = sb(nd, i-1) + sb(nd, i)
      precurveb(nd, i) = precurveb(nd, i) + tempb9(nd)
      precurveb(nd, i-1) = precurveb(nd, i-1) - tempb9(nd)
      presweepb(nd, i) = presweepb(nd, i) + tempb10(nd)
      presweepb(nd, i-1) = presweepb(nd, i-1) - tempb10(nd)
      rb(nd, i) = rb(nd, i) + tempb11(nd)
      rb(nd, i-1) = rb(nd, i-1) - tempb11(nd)
      sb(nd, i) = 0.0
    END DO
  END DO
  temp0 = z_az(2:n-1) - z_az(1:n-2)
  temp = x_az(1:n-2) - x_az(2:n-1)
  DO nd=1,nbdirs
    x_azb(nd, :) = 0.0
    tempb(nd) = (z_az(n)-z_az(n-1))*coneb(nd, n)/((x_az(n-1)-x_az(n))**2&
&     +(z_az(n)-z_az(n-1))**2)
    tempb0(nd) = -((x_az(n-1)-x_az(n))*coneb(nd, n)/((x_az(n-1)-x_az(n))&
&     **2+(z_az(n)-z_az(n-1))**2))
    x_azb(nd, n-1) = x_azb(nd, n-1) + tempb(nd)
    x_azb(nd, n) = x_azb(nd, n) - tempb(nd)
    z_azb(nd, n) = z_azb(nd, n) + tempb0(nd)
    z_azb(nd, n-1) = z_azb(nd, n-1) - tempb0(nd)
    coneb(nd, n) = 0.0
    tempb1(nd, :) = 0.5_dp*coneb(nd, 2:n-1)
    tempb2(nd, :) = temp0*tempb1(nd, :)/(temp**2+temp0**2)
    tempb3(nd, :) = -(temp*tempb1(nd, :)/(temp**2+temp0**2))
    tempb4(nd, :) = (z_az(3:n)-z_az(2:n-1))*tempb1(nd, :)/((x_az(2:n-1)-&
&     x_az(3:n))**2+(z_az(3:n)-z_az(2:n-1))**2)
    tempb5(nd, :) = -((x_az(2:n-1)-x_az(3:n))*tempb1(nd, :)/((x_az(2:n-1&
&     )-x_az(3:n))**2+(z_az(3:n)-z_az(2:n-1))**2))
    x_azb(nd, 1:n-2) = x_azb(nd, 1:n-2) + tempb2(nd, :)
    x_azb(nd, 2:n-1) = x_azb(nd, 2:n-1) + tempb4(nd, :) - tempb2(nd, :)
    z_azb(nd, 2:n-1) = z_azb(nd, 2:n-1) + tempb3(nd, :) - tempb5(nd, :)
    z_azb(nd, 1:n-2) = z_azb(nd, 1:n-2) - tempb3(nd, :)
    x_azb(nd, 3:n) = x_azb(nd, 3:n) - tempb4(nd, :)
    z_azb(nd, 3:n) = z_azb(nd, 3:n) + tempb5(nd, :)
    coneb(nd, 2:n-1) = 0.0
    tempb6(nd) = (z_az(2)-z_az(1))*coneb(nd, 1)/((x_az(1)-x_az(2))**2+(&
&     z_az(2)-z_az(1))**2)
    tempb7(nd) = -((x_az(1)-x_az(2))*coneb(nd, 1)/((x_az(1)-x_az(2))**2+&
&     (z_az(2)-z_az(1))**2))
    x_azb(nd, 1) = x_azb(nd, 1) + tempb6(nd)
    x_azb(nd, 2) = x_azb(nd, 2) - tempb6(nd)
    z_azb(nd, 2) = z_azb(nd, 2) + tempb7(nd)
    z_azb(nd, 1) = z_azb(nd, 1) - tempb7(nd)
    rb(nd, :) = rb(nd, :) + COS(precone)*z_azb(nd, :) - SIN(precone)*&
&     x_azb(nd, :)
    preconeb(nd) = COS(precone)*SUM(precurve*z_azb(nd, :)) - SIN(precone&
&     )*SUM(precurve*x_azb(nd, :)) - COS(precone)*SUM(r*x_azb(nd, :)) - &
&     SIN(precone)*SUM(r*z_azb(nd, :))
    precurveb(nd, :) = precurveb(nd, :) + COS(precone)*x_azb(nd, :) + &
&     SIN(precone)*z_azb(nd, :)
  END DO
END SUBROUTINE DEFINECURVATURE_BV



!        Generated by TAPENADE     (INRIA, Tropics team)
!  Tapenade 3.9 (r5096) - 24 Feb 2014 16:54
!
!  Differentiation of definecurvature in forward (tangent) mode:
!   variations   of useful results: s z_az y_az x_az cone
!   with respect to varying inputs: r precurve presweep precone
!   RW status of diff variables: r:in s:out precurve:in z_az:out
!                presweep:in y_az:out x_az:out cone:out precone:in

SUBROUTINE DEFINECURVATURE_DV2(n, r, rd, precurve, precurved, presweep, &
& presweepd, precone, preconed, x_az, x_azd, y_az, y_azd, z_az, z_azd, &
& cone, coned, s, sd, nbdirs)
!   USE DIFFSIZES
!  Hint: nbdirsmax should be the maximum number of differentiation directions
  IMPLICIT NONE
  INTEGER, PARAMETER :: dp=KIND(0.d0)
! in
  INTEGER, INTENT(IN) :: n
  REAL(dp), DIMENSION(n), INTENT(IN) :: r, precurve, presweep
  REAL(dp), DIMENSION(nbdirs, n), INTENT(IN) :: rd, precurved, presweepd
  INTEGER, intent(in) :: nbdirs
  REAL(dp), INTENT(IN) :: precone
  REAL(dp), DIMENSION(nbdirs), INTENT(IN) :: preconed
! out
  REAL(dp), DIMENSION(n), INTENT(OUT) :: x_az, y_az, z_az, cone, s
  REAL(dp), DIMENSION(nbdirs, n), INTENT(OUT) :: x_azd, y_azd, z_azd&
& , coned, sd
! local
  INTEGER :: i
  INTRINSIC KIND
  INTRINSIC SIN
  INTRINSIC COS
  INTRINSIC ATAN2
  INTRINSIC SQRT
  REAL(dp) :: arg1
  REAL(dp), DIMENSION(nbdirs) :: arg1d
  REAL(dp) :: arg2
  REAL(dp), DIMENSION(nbdirs) :: arg2d
  REAL(dp), DIMENSION(n-2) :: arg10
  REAL(dp), DIMENSION(nbdirs, n-2) :: arg10d
  REAL(dp), DIMENSION(n-2) :: arg20
  REAL(dp), DIMENSION(nbdirs, n-2) :: arg20d
  REAL(dp), DIMENSION(n-2) :: arg3
  REAL(dp), DIMENSION(nbdirs, n-2) :: arg3d
  REAL(dp), DIMENSION(n-2) :: arg4
  REAL(dp), DIMENSION(nbdirs, n-2) :: arg4d
  REAL(dp) :: result1
  REAL(dp), DIMENSION(nbdirs) :: result1d
  INTEGER :: nd
  x_az = -(r*SIN(precone)) + precurve*COS(precone)
  z_az = r*COS(precone) + precurve*SIN(precone)
  arg1 = -(x_az(2)-x_az(1))
  arg2 = z_az(2) - z_az(1)
  arg10(:) = -(x_az(2:n-1)-x_az(1:n-2))
  arg20(:) = z_az(2:n-1) - z_az(1:n-2)
  arg3(:) = -(x_az(3:n)-x_az(2:n-1))
  arg4(:) = z_az(3:n) - z_az(2:n-1)
  DO nd=1,nbdirs
! coordinate in azimuthal coordinate system
! az_coords = DirectionVector(precurve, presweep, r).bladeToAzimuth(precone)
    x_azd(nd, :) = precurved(nd, :)*COS(precone) - r*preconed(nd)*COS(&
&     precone) - rd(nd, :)*SIN(precone) - precurve*preconed(nd)*SIN(&
&     precone)
    z_azd(nd, :) = rd(nd, :)*COS(precone) - r*preconed(nd)*SIN(precone) &
&     + precurved(nd, :)*SIN(precone) + precurve*preconed(nd)*COS(&
&     precone)
    y_azd(nd, :) = presweepd(nd, :)
! compute total coning angle for purposes of relative velocity
    arg1d(nd) = -(x_azd(nd, 2)-x_azd(nd, 1))
    arg2d(nd) = z_azd(nd, 2) - z_azd(nd, 1)
    coned(nd, :) = 0.0
    coned(nd, 1) = (arg1d(nd)*arg2-arg2d(nd)*arg1)/(arg1**2+arg2**2)
    arg10d(nd, :) = -(x_azd(nd, 2:n-1)-x_azd(nd, 1:n-2))
    arg20d(nd, :) = z_azd(nd, 2:n-1) - z_azd(nd, 1:n-2)
    arg3d(nd, :) = -(x_azd(nd, 3:n)-x_azd(nd, 2:n-1))
    arg4d(nd, :) = z_azd(nd, 3:n) - z_azd(nd, 2:n-1)
    coned(nd, 2:n-1) = 0.5_dp*((arg10d(nd, :)*arg20(:)-arg20d(nd, :)*&
&     arg10(:))/(arg10(:)**2+arg20(:)**2)+(arg3d(nd, :)*arg4(:)-arg4d(nd&
&     , :)*arg3(:))/(arg3(:)**2+arg4(:)**2))
    arg1d(nd) = -(x_azd(nd, n)-x_azd(nd, n-1))
    arg2d(nd) = z_azd(nd, n) - z_azd(nd, n-1)
! total path length of blade
    sd(nd, 1) = 0.0
  END DO
  y_az = presweep
  cone(1) = ATAN2(arg1, arg2)
  cone(2:n-1) = 0.5_dp*(ATAN2(arg10(:), arg20(:))+ATAN2(arg3(:), arg4(:)&
&   ))
  arg1 = -(x_az(n)-x_az(n-1))
  arg2 = z_az(n) - z_az(n-1)
  DO nd=1,nbdirs
    coned(nd, n) = (arg1d(nd)*arg2-arg2d(nd)*arg1)/(arg1**2+arg2**2)
  END DO
  cone(n) = ATAN2(arg1, arg2)
  s(1) = 0.0_dp
  DO nd=1,nbdirs
    sd(nd, :) = 0.0
  END DO
  DO i=2,n
    arg1 = (precurve(i)-precurve(i-1))**2 + (presweep(i)-presweep(i-1))&
&     **2 + (r(i)-r(i-1))**2
    DO nd=1,nbdirs
      arg1d(nd) = 2*(precurve(i)-precurve(i-1))*(precurved(nd, i)-&
&       precurved(nd, i-1)) + 2*(presweep(i)-presweep(i-1))*(presweepd(&
&       nd, i)-presweepd(nd, i-1)) + 2*(r(i)-r(i-1))*(rd(nd, i)-rd(nd, i&
&       -1))
      IF (arg1 .EQ. 0.0) THEN
        result1d(nd) = 0.0
      ELSE
        result1d(nd) = arg1d(nd)/(2.0*SQRT(arg1))
      END IF
      sd(nd, i) = sd(nd, i-1) + result1d(nd)
    END DO
    result1 = SQRT(arg1)
    s(i) = s(i-1) + result1
  END DO
END SUBROUTINE DEFINECURVATURE_DV2

