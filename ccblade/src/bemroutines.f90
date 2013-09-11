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


