

!****************************
SUBROUTINE band2full_dbl(n,nb,a)
!****************************
! Purpose:
!   Transformation of a symmetric, banded matrix into a full matrix in
!   double precision
!
! Record of revisions:
!    Date        Programmer      Description of change
!    ====       =============   ==================================
! 01/04/2008    S. Larwood      Original code
! 07/19/2013    S. Andrew Ning  Removed symmetry mirroring (LAPACK only needs upper triangle filled)

IMPLICIT NONE

! Data dictionary: declare calling parameter types & definitions
INTEGER, INTENT(IN)                             :: n    ! Rows of matrix A
INTEGER, INTENT(IN)                             :: nb   ! Semi-bandwidth of matrix A
DOUBLE PRECISION, INTENT(INOUT), DIMENSION(n,n) :: a    ! Matrix A in banded, out full

! Data dictionary: declare local variable types & definitions
DOUBLE PRECISION, DIMENSION(n,n)    :: full     ! Temporary full matrix where A will be stored
INTEGER                             :: i        ! Index
INTEGER                             :: j        ! Index
INTEGER                             :: ipjm1    ! i + j - 1

! Initialize full matrix
full = 0.0d0

! Build upper triangle of full
DO i = 1, n
    DO j = 1, nb

        ipjm1 = i + j - 1

        IF (ipjm1 <= n) THEN

            full(i,ipjm1) = a(i,j)

        END IF

    END DO
END DO

! Set A equal to the full matrix
a = full

! ! Add symmetric terms
! DO i = 1, n
!     DO j = 1, n

!         a(j,i) = a(i,j)

!     END DO
! END DO

RETURN
END SUBROUTINE band2full_dbl



!****************************
SUBROUTINE taper_axial_force_dbl(nn,ne,loc,cx,cy,cz,mu,ivert,omega,f1)
!****************************
! Purpose:
!   Calculates axial force for spinning finite elements in double precision.
!   Coordinates are in CurveFAST Lj system.
!
! Reference:
!   for transformation matrices-
!   Rao, S. S. (2005). "The Finite Element Method in Engineering"
!   Fourth Edition, Pergamon Press, Section 9.4
!
!   and
!
!   for axial force calculations-
!   Leung, A. Y. T. and T. C. Fung (1988). "Spinning Finite Elements."
!   Journal of Sound and Vibration 125(3): pp. 523-537.
!
! Record of revisions:
!    Date        Programmer      Description of change
!    ====       =============   ==================================
! 01/18/2008    S. Larwood      Original code based on axial_force_dbl from 01/07/2008
! 07/19/2013    S. Andrew Ning  commented out unused variables
IMPLICIT NONE

! Data dictionary: declare calling parameter types & definitions
INTEGER, INTENT(IN)                             :: ne       ! Number of finite elements
INTEGER, INTENT(IN)                             :: nn       ! Number of nodes in structure
DOUBLE PRECISION, INTENT(IN), DIMENSION(nn)     :: cx       ! x-coordinate of node
DOUBLE PRECISION, INTENT(IN), DIMENSION(nn)     :: cy       ! y-coordinate of node
DOUBLE PRECISION, INTENT(IN), DIMENSION(nn)     :: cz       ! z-coordinate of node
INTEGER, INTENT(IN), DIMENSION(ne)              :: ivert    ! Orientation of element, 1 if vertical,
                                                            ! 0 if otherwise
INTEGER, INTENT(IN), DIMENSION(ne,2)            :: loc      ! Global node number, loc(i,j), jth corner of element i
DOUBLE PRECISION,INTENT(IN), DIMENSION(nn)      :: mu       ! Element lineal density
DOUBLE PRECISION, INTENT(IN)                    :: omega    ! Rotational rate about global y-axis
DOUBLE PRECISION, INTENT(OUT), DIMENSION(ne)    :: f1       ! Element axial force

! Data dictionary: declare local variable types & definitions
DOUBLE PRECISION, DIMENSION(ne)     :: al       ! Element length
DOUBLE PRECISION, DIMENSION(ne)     :: b32      ! Element 1,1 of transformation matrix
DOUBLE PRECISION, DIMENSION(ne)     :: b33      ! Element 1,3 of transformation matrix
! DOUBLE PRECISION                    :: d        ! Factor used in transformation
INTEGER                             :: ie       ! First corner of element
INTEGER                             :: i        ! Index
INTEGER                             :: je       ! Second corner of element
INTEGER                             :: j        ! Index

! Calculate required direction cosines and lengths for each element
elemloop1: DO i = 1, ne

    ! Find element node number
    ie = loc(i,1)
    je = loc(i,2)

    ! Compute length
    al(i) = DSQRT((cx(je)-cx(ie))**2 + (cy(je)-cy(ie))**2 &
                    +(cz(je)-cz(ie))**2)

    !Compute elements of (3x3)transformation matrix
    ! If element is vertical there is special handling
    IF (ivert(i) == 1) THEN

        b32(i) = 0.0d0
        b33(i) = 0.0d0

    ELSE

        b32(i) = (cy(je) - cy(ie))/al(i)
        b33(i) = (cz(je) - cz(ie))/al(i)

    END IF

END DO elemloop1

! Initialize axial force values
f1 = 0.0d0

elemloop2: DO i = 1, ne

    ! Add up contribution from outer elements, except for last element
    IF (i /= ne) THEN

        DO j = i + 1, ne

            ! Find element node numbers
            ie = loc(j,1)
            je = loc(j,2)

            f1(i) = f1(i) + omega**2 * al(j)/2.0 * ( &
                     (b32(i) * cy(ie) + b33(i) * cz(ie) ) * (mu(ie) + mu(je)) &
                    + al(j) * (b32(i) * b32(j) + b33(i) * b33(j)) &
                    * (mu(ie) + 2.0 * mu(je))/3.0   )

        END DO

    END IF

    ! Find element node numbers
    ie = loc(i,1)
    je = loc(i,2)

    ! Add contribution of current element to outer elements (if any)
    f1(i) = f1(i) + omega**2 * al(i)/2.0 * ( &
            (b32(i) * cy(ie) + b33(i) * cz(ie) ) * (mu(ie) + mu(je)) &
            + al(i) * (b32(i)**2 + b33(i)**2) &
            * (mu(ie) + 2.0 * mu(je))/3.0   )

END DO elemloop2

RETURN
END SUBROUTINE taper_axial_force_dbl

!****************************
SUBROUTINE taper_frame_spin_dbl(nn,ne,nd,nb,loc,cx,cy,cz,mu, &
                    alpha,ivert,omega,gyro,cf,kspin)
!****************************
! Purpose:
!   Builds banded matrices for structure with tapered space frame elements in a
!   rotating reference frame in double precision.
!
! Reference:
!   Rao, S. S. (2005). "The Finite Element Method in Engineering"
!   Fourth Edition, Pergamon Press, Section 9.4
!
!   and
!
!   Leung, A. Y. T. and T. C. Fung (1988). "Spinning Finite Elements."
!   Journal of Sound and Vibration 125(3): pp. 523-537.
!
! Record of revisions:
!    Date        Programmer      Description of change
!    ====       =============   ==================================
! 01/25/2008    S. Larwood      Original code based on frame_spin_dbl 01/16/2008
! 07/19/2013    S. Andrew Ning  Removed matmul_dbl to use intrinsic MATMUL.  commented out unused vars

IMPLICIT NONE

! Data dictionary: declare calling parameter types & definitions
INTEGER, INTENT(IN)                             :: nb       ! Bandwidth of overall stiffness matrix
INTEGER, INTENT(IN)                             :: nd       ! Total number of degrees of freedom
INTEGER, INTENT(IN)                             :: ne       ! Number of finite elements
INTEGER, INTENT(IN)                             :: nn       ! Number of nodes in structure
DOUBLE PRECISION, INTENT(IN), DIMENSION(ne)     :: alpha    ! Element angular alignment
DOUBLE PRECISION, INTENT(OUT), DIMENSION(nd,nb) :: cf       ! Global centrifugal force matrix
DOUBLE PRECISION, INTENT(IN), DIMENSION(nn)     :: cx       ! x-coordinate of node
DOUBLE PRECISION, INTENT(IN), DIMENSION(nn)     :: cy       ! y-coordinate of node
DOUBLE PRECISION, INTENT(IN), DIMENSION(nn)     :: cz       ! z-coordinate of node
DOUBLE PRECISION, INTENT(OUT), DIMENSION(nd,nb) :: gyro     ! Global gyroscopic matrix
INTEGER, INTENT(IN), DIMENSION(ne)              :: ivert    ! Orientation of element, 1 if vertical,
                                                            ! 0 if otherwise
DOUBLE PRECISION, INTENT(OUT), DIMENSION(nd,nb) :: kspin    ! Global gyroscopic matrix
INTEGER, INTENT(IN), DIMENSION(ne,2)            :: loc      ! Global node number, loc(i,j), jth corner of element i
DOUBLE PRECISION,INTENT(IN), DIMENSION(nn)      :: mu       ! Element lineal density
DOUBLE PRECISION, INTENT(IN)                    :: omega    ! Rotational rate about global y-axis

! Data dictionary: declare local variable types & definitions
DOUBLE PRECISION                    :: a0       ! Variable in cf matrix
DOUBLE PRECISION, DIMENSION(12,12)  :: aa       ! Dummy matrix
DOUBLE PRECISION                    :: a11      ! Variable in matrices
DOUBLE PRECISION                    :: a12      ! Variable in matrices
DOUBLE PRECISION                    :: a13      ! Variable in matrices
DOUBLE PRECISION                    :: a21      ! Variable in transformation matrix
DOUBLE PRECISION                    :: a22      ! Variable in matrices
DOUBLE PRECISION                    :: a23      ! Variable in matrices
! DOUBLE PRECISION                    :: a31      ! Variable in transformation matrix
! DOUBLE PRECISION                    :: a32      ! Variable in transformation matrix
DOUBLE PRECISION                    :: a33      ! Variable in matrices
DOUBLE PRECISION                    :: al       ! Element length
DOUBLE PRECISION                    :: alz      ! dx/dl
DOUBLE PRECISION                    :: amz      ! dy/dl
DOUBLE PRECISION                    :: anz      ! dz/dl
DOUBLE PRECISION, DIMENSION(3,3)    :: b        ! Transformation matrix
DOUBLE PRECISION                    :: b0       ! Variable in matrices
DOUBLE PRECISION                    :: b1       ! Variable in matrices
DOUBLE PRECISION                    :: b2       ! Variable in matrices
DOUBLE PRECISION                    :: b3       ! Variable in matrices
DOUBLE PRECISION                    :: c0       ! Variable in cf matrix
DOUBLE PRECISION, DIMENSION(12,12)  :: cf_e     ! Element local/global centrifugal force matrix
DOUBLE PRECISION                    :: cs       ! cos(alpha)
DOUBLE PRECISION                    :: d        ! Factor used in transformation
DOUBLE PRECISION                    :: d0       ! Variable in cf matrix
DOUBLE PRECISION                    :: e0       ! Variable in cf matrix
DOUBLE PRECISION, DIMENSION(ne)     :: f1       ! Element axial force from rotation
DOUBLE PRECISION, DIMENSION(12,12)  :: gyro_e   ! Element local/global gyroscopic matrix
INTEGER                             :: ie       ! First corner of element
INTEGER                             :: i        ! Index
INTEGER                             :: i3       ! i + 3
INTEGER                             :: i6       ! i + 6
INTEGER                             :: i9       ! i + 9
INTEGER                             :: ii       ! Index
INTEGER                             :: ik       ! Dummy variable
INTEGER                             :: in       ! Dummy variable
! INTEGER                             :: ix       ! Number of fixed DOF
INTEGER                             :: j        ! Index
INTEGER                             :: j3       ! j + 3
INTEGER                             :: j6       ! j + 6
INTEGER                             :: j9       ! j + 9
INTEGER                             :: je       ! Second corner of element
INTEGER                             :: jk       ! Dummy variable
DOUBLE PRECISION, DIMENSION(12,12)  :: lambda   ! Transformation matrix
DOUBLE PRECISION, DIMENSION(12,12)  :: lambdat  ! Transpose of lambda
DOUBLE PRECISION, DIMENSION(12,12)  :: kspin_e  ! Element local/global spin stiffness matrix
DOUBLE PRECISION, DIMENSION(12)     :: n        ! Used for assembling global stiffness matrix
DOUBLE PRECISION                    :: ss       ! Sin(alpha)

! Initialize global matrices
cf = 0.0d0
gyro = 0.0d0
kspin = 0.0d0

! Calculate axial forces for the elements
CALL taper_axial_force_dbl(nn,ne,loc,cx,cy,cz,mu,ivert,omega,f1)

elemloop: DO ii = 1, ne

    ! Initialize local element matrices
    cf_e = 0.0d0
    gyro_e = 0.0d0
    kspin_e = 0.0d0

    ! Find node element node number
    ie = loc(ii,1)
    je = loc(ii,2)

    ! Compute length
    al=DSQRT((cx(je)-cx(ie))**2 + (cy(je)-cy(ie))**2 &
            +(cz(je)-cz(ie))**2)

    ! Compute slopes
    alz = (cx(je) - cx(ie))/al
    amz = (cy(je) - cy(ie))/al
    anz = (cz(je) - cz(ie))/al

    !Compute elements of (3x3)transformation matrix
    ! If element is vertical there is special handling
    IF (ivert(ii) == 1) THEN

        b(1,1) = 0.0d0
        b(1,2) = -DSIN(alpha(ii))
        b(1,3) = -DCOS(alpha(ii))
        b(2,1) = 0.0d0
        b(2,2) = DCOS(alpha(ii))
        b(2,3) = -DSIN(alpha(ii))
        b(3,1) = 1.0d0
        b(3,2) = 0.0d0
        b(3,3) = 0.0d0

    ELSE

        d = DSQRT(amz**2 + anz**2)
        a11 = (amz**2 + anz**2)/d
        a12 = -(alz * amz) /d
        a13 = -(alz * anz) /d
        a21 = 0.0d0
        a22 = anz/d
        a23 = -amz/d
        cs = DCOS(alpha(ii))
        ss = DSIN(alpha(ii))
        b(1,1) = a11 * cs - a21 * ss
        b(1,2) = a12 * cs - a22 * ss
        b(1,3) = a13 * cs - a23 * ss
        b(2,1) = a11 * ss + a21 * cs
        b(2,2) = a12 * ss + a22 * cs
        b(2,3) = a13 * ss + a23 * cs
        b(3,1) = alz
        b(3,2) = amz
        b(3,3) = anz

    END IF

    ! Compute various parameters
    a0 = omega**2*(b(3,2) * cy(ie) + b(3,3) * cz(ie))
    b0 = omega**2*(b(3,2)**2 + b(3,3)**2)
    c0 = mu(ie) * a0
    d0 = ((mu(je) - mu(ie)) * a0 + mu(ie) * b0 * al )/ al
    e0 = (mu(je) - mu(ie)) * b0 / al
    a11 = b(1,2)**2 + b(1,3)**2
    a12 = b(1,2)*b(2,2) + b(1,3)*b(2,3)
    a13 = b(1,2)*b(3,2) + b(1,3)*b(3,3)
    a22 = b(2,2)**2 + b(2,3)**2
    a23 = b(2,2)*b(3,2) + b(2,3)*b(3,3)
    a33 = b(3,2)**2 + b(3,3)**2
    b1 = b(1,3)*b(2,2) - b(1,2)*b(2,3)
    b2 = b(1,3)*b(3,2) - b(1,2)*b(3,3)
    b3 = b(2,3)*b(3,2) - b(2,2)*b(3,3)

    ! Build centrifugal force matrix
    cf_e(1,1)   =   f1(ii)/(30.0*al)    * 36.0          &
                    -c0/60.0            * 36.0          &
                    -d0*al/420.0        * 72.0          &
                    -e0*al**2/2520.0    * 180.0

    cf_e(1,5)   =   f1(ii)/(30.0*al)    * 3.0*al        &
                    -c0/60.0            * 6.0*al        &
                    -d0*al/420.0        * 15.0*al       &
                    -e0*al**2/2520.0    * 42.0*al

    cf_e(1,7)   =   f1(ii)/(30.0*al)    * -36.0         &
                    -c0/60.0            * -36.0         &
                    -d0*al/420.0        * -72.0         &
                    -e0*al**2/2520.0    * -180.0

    cf_e(1,11)  =   f1(ii)/(30.0*al)    * 3.0*al        &
                    -c0/60.0            * 0.0*al        &
                    -d0*al/420.0        * -6.0*al       &
                    -e0*al**2/2520.0    * -30.0*al

    cf_e(2,2)   =   f1(ii)/(30.0*al)    * 36.0          &
                    -c0/60.0            * 36.0          &
                    -d0*al/420.0        * 72.0          &
                    -e0*al**2/2520.0    * 180.0

    cf_e(2,4)   =   f1(ii)/(30.0*al)    * -3.0*al       &
                    -c0/60.0            * -6.0*al       &
                    -d0*al/420.0        * -15.0*al      &
                    -e0*al**2/2520.0    * -42.0*al

    cf_e(2,8)   =   f1(ii)/(30.0*al)    * -36.0         &
                    -c0/60.0            * -36.0         &
                    -d0*al/420.0        * -72.0         &
                    -e0*al**2/2520.0    * -180.0

    cf_e(2,10)  =   f1(ii)/(30.0*al)    * -3.0*al       &
                    -c0/60.0            * 0.0*al        &
                    -d0*al/420.0        * 6.0*al        &
                    -e0*al**2/2520.0    * 30.0*al

    cf_e(4,4)   =   f1(ii)/(30.0*al)    * 4.0*al**2     &
                    -c0/60.0            * 2.0*al**2     &
                    -d0*al/420.0        * 4.0*al**2     &
                    -e0*al**2/2520.0    * 11.0*al**2

    cf_e(4,8)   =   f1(ii)/(30.0*al)    * 3.0*al        &
                    -c0/60.0            * 6.0*al        &
                    -d0*al/420.0        * 15.0*al       &
                    -e0*al**2/2520.0    * 42.0*al

    cf_e(4,10)  =   f1(ii)/(30.0*al)    * -al**2        &
                    -c0/60.0            * -al**2        &
                    -d0*al/420.0        * -3.0*al**2    &
                    -e0*al**2/2520.0    * -11.0*al**2

    cf_e(5,5)   =   f1(ii)/(30.0*al)    * 4.0*al**2     &
                    -c0/60.0            * 2.0*al**2     &
                    -d0*al/420.0        * 4.0*al**2     &
                    -e0*al**2/2520.0    * 11.0*al**2

    cf_e(5,7)   =   f1(ii)/(30.0*al)    * -3.0*al       &
                    -c0/60.0            * -6.0*al       &
                    -d0*al/420.0        * -15.0*al      &
                    -e0*al**2/2520.0    * -42.0*al

    cf_e(5,11)  =   f1(ii)/(30.0*al)    * -al**2        &
                    -c0/60.0            * -al**2        &
                    -d0*al/420.0        * -3.0*al**2    &
                    -e0*al**2/2520.0    * -11.0*al**2

    cf_e(7,7)   =   f1(ii)/(30.0*al)    * 36.0          &
                    -c0/60.0            * 36.0          &
                    -d0*al/420.0        * 72.0          &
                    -e0*al**2/2520.0    * 180.0

    cf_e(7,11)  =   f1(ii)/(30.0*al)    * -3.0*al       &
                    -c0/60.0            * 0.0*al        &
                    -d0*al/420.0        * 6.0*al        &
                    -e0*al**2/2520.0    * 30.0*al

    cf_e(8,8)   =   f1(ii)/(30.0*al)    * 36.0          &
                    -c0/60.0            * 36.0          &
                    -d0*al/420.0        * 72.0          &
                    -e0*al**2/2520.0    * 180.0

    cf_e(8,10)  =   f1(ii)/(30.0*al)    * 3.0*al        &
                    -c0/60.0            * 0.0*al        &
                    -d0*al/420.0        * -6.0*al       &
                    -e0*al**2/2520.0    * -30.0*al

    cf_e(10,10) =   f1(ii)/(30.0*al)    * 4.0*al**2     &
                    -c0/60.0            * 6.0*al**2     &
                    -d0*al/420.0        * 18.0*al**2        &
                    -e0*al**2/2520.0    * 65.0*al**2

    cf_e(11,11) =   f1(ii)/(30.0*al)    * 4.0*al**2     &
                    -c0/60.0            * 6.0*al**2     &
                    -d0*al/420.0        * 18.0*al**2    &
                    -e0*al**2/2520.0    * 65.0*al**2

    ! Add symmetric terms
    DO i = 1, 12
        DO j = 1, 12

            cf_e(j,i) = cf_e(i,j)

        END DO
    END DO

    ! Build local gyroscopic matrix
    gyro_e(1,2)     =   mu(ie)/420.0            * 156.0*b1      + &
                        (mu(je)-mu(ie))/840.0   * 72.0*b1

    gyro_e(1,3)     =   mu(ie)/420.0            * 147.0*b2      + &
                        (mu(je)-mu(ie))/840.0   * 70.0*b2

    gyro_e(1,4)     =   mu(ie)/420.0            * -22.0*al*b1   + &
                        (mu(je)-mu(ie))/840.0   * -14.0*al*b1

    gyro_e(1,8)     =   mu(ie)/420.0            * 54.0*b1       + &
                        (mu(je)-mu(ie))/840.0   * 54.0*b1

    gyro_e(1,9)     =   mu(ie)/420.0            * 63.0*b2       + &
                        (mu(je)-mu(ie))/840.0   * 56.0*b2

    gyro_e(1,10)    =   mu(ie)/420.0            * 13.0*al*b1    + &
                        (mu(je)-mu(ie))/840.0   * 12.0*al*b1

    gyro_e(2,3)     =   mu(ie)/420.0            * 147.0*b3      + &
                        (mu(je)-mu(ie))/840.0   * 70.0*b3

    gyro_e(2,5)     =   mu(ie)/420.0            * -22.0*al*b1   + &
                        (mu(je)-mu(ie))/840.0   * -14.0*al*b1

    gyro_e(2,7)     =   mu(ie)/420.0            * -54.0*b1      + &
                        (mu(je)-mu(ie))/840.0   * -54.0*b1

    gyro_e(2,9)     =   mu(ie)/420.0            * 63.0*b3       + &
                        (mu(je)-mu(ie))/840.0   * 56.0*b3

    gyro_e(2,11)    =   mu(ie)/420.0            * 13.0*al*b1    + &
                        (mu(je)-mu(ie))/840.0   * 12.0*al*b1

    gyro_e(3,4)     =   mu(ie)/420.0            * 21.0*al*b3    + &
                        (mu(je)-mu(ie))/840.0   * 14.0*al*b3

    gyro_e(3,5)     =   mu(ie)/420.0            * -21.0*al*b2   + &
                        (mu(je)-mu(ie))/840.0   * -14.0*al*b2

    gyro_e(3,7)     =   mu(ie)/420.0            * -63.0*b2      + &
                        (mu(je)-mu(ie))/840.0   * -70.0*b2

    gyro_e(3,8)     =   mu(ie)/420.0            * -63.0*b3      + &
                        (mu(je)-mu(ie))/840.0   * -70.0*b3

    gyro_e(3,10)    =   mu(ie)/420.0            * -14.0*al*b3   + &
                        (mu(je)-mu(ie))/840.0   * -14.0*al*b3

    gyro_e(3,11)    =   mu(ie)/420.0            * 14.0*al*b2    + &
                        (mu(je)-mu(ie))/840.0   * 14.0*al*b2

    gyro_e(4,5)     =   mu(ie)/420.0            * 4.0*al**2*b1  + &
                        (mu(je)-mu(ie))/840.0   * 3.0*al**2*b1

    gyro_e(4,7)     =   mu(ie)/420.0            * 13.0*al*b1    + &
                        (mu(je)-mu(ie))/840.0   * 14.0*al*b1

    gyro_e(4,9)     =   mu(ie)/420.0            * -14.0*al*b3   + &
                        (mu(je)-mu(ie))/840.0   * -14.0*al*b3

    gyro_e(4,11)    =   mu(ie)/420.0            * -3.0*al**2*b1 + &
                        (mu(je)-mu(ie))/840.0   * -3.0*al**2*b1

    gyro_e(5,8)     =   mu(ie)/420.0            * 13.0*al*b1    + &
                        (mu(je)-mu(ie))/840.0   * 14.0*al*b1

    gyro_e(5,9)     =   mu(ie)/420.0            * 14.0*al*b2    + &
                        (mu(je)-mu(ie))/840.0   * 14.0*al*b2

    gyro_e(5,10)    =   mu(ie)/420.0            * 3.0*al**2*b1  + &
                        (mu(je)-mu(ie))/840.0   * 3.0*al**2*b1

    gyro_e(7,8)     =   mu(ie)/420.0            * 156.0*b1      + &
                        (mu(je)-mu(ie))/840.0   * 240.0*b1

    gyro_e(7,9)     =   mu(ie)/420.0            * 147.0*b2      + &
                        (mu(je)-mu(ie))/840.0   * 224.0*b2

    gyro_e(7,10)    =   mu(ie)/420.0            * 22.0*al*b1    + &
                        (mu(je)-mu(ie))/840.0   * 30.0*al*b1

    gyro_e(8,9)     =   mu(ie)/420.0            * 147.0*b3      + &
                        (mu(je)-mu(ie))/840.0   * 224.0*b3

    gyro_e(8,11)    =   mu(ie)/420.0            * 22.0*al*b1    + &
                        (mu(je)-mu(ie))/840.0   * 30.0*al*b1

    gyro_e(9,10)    =   mu(ie)/420.0            * -21.0*al*b3   + &
                        (mu(je)-mu(ie))/840.0   * -22.0*al*b3

    gyro_e(9,11)    =   mu(ie)/420.0            * 21.0*al*b2    + &
                        (mu(je)-mu(ie))/840.0   * 22.0*al*b2

    gyro_e(10,11)   =   mu(ie)/420.0            * 4.0*al**2*b1  + &
                        (mu(je)-mu(ie))/840.0   * 5.0*al**2*b1

    ! Add skew-symmetric terms
    DO i = 1, 12
        DO j = 1, 12

            gyro_e(j,i) = -gyro_e(i,j)

        END DO
    END DO

    ! Multiply matrices by common factors
    gyro_e = gyro_e * al * omega

    ! Build local spin stiffness matrix
    kspin_e(1,1)    =   mu(ie)/420.0            * 156.0*a11         + &
                        (mu(je)-mu(ie))/840.0   * 72.0*a11

    kspin_e(1,2)    =   mu(ie)/420.0            * 156.0*a12         + &
                        (mu(je)-mu(ie))/840.0   * 72.0*a12

    kspin_e(1,3)    =   mu(ie)/420.0            * 147.0*a13         + &
                        (mu(je)-mu(ie))/840.0   * 70.0*a13

    kspin_e(1,4)    =   mu(ie)/420.0            * -22.0*al*a12      + &
                        (mu(je)-mu(ie))/840.0   * -14.0*al*a12

    kspin_e(1,5)    =   mu(ie)/420.0            * 22.0*al*a11       + &
                        (mu(je)-mu(ie))/840.0   * 14.0*al*a11

    kspin_e(1,7)    =   mu(ie)/420.0            * 54.0*a11          + &
                        (mu(je)-mu(ie))/840.0   * 54.0*a11

    kspin_e(1,8)    =   mu(ie)/420.0            * 54.0*a12          + &
                        (mu(je)-mu(ie))/840.0   * 54.0*a12

    kspin_e(1,9)    =   mu(ie)/420.0            * 63.0*a13          + &
                        (mu(je)-mu(ie))/840.0   * 56.0*a13

    kspin_e(1,10)   =   mu(ie)/420.0            * 13.0*al*a12       + &
                        (mu(je)-mu(ie))/840.0   * 12.0*al*a12

    kspin_e(1,11)   =   mu(ie)/420.0            * -13.0*al*a11      + &
                        (mu(je)-mu(ie))/840.0   * -12.0*al*a11

    kspin_e(2,2)    =   mu(ie)/420.0            * 156.0*a22         + &
                        (mu(je)-mu(ie))/840.0   * 72.0*a22

    kspin_e(2,3)    =   mu(ie)/420.0            * 147.0*a23         + &
                        (mu(je)-mu(ie))/840.0   * 70.0*a23

    kspin_e(2,4)    =   mu(ie)/420.0            * -22.0*al*a22      + &
                        (mu(je)-mu(ie))/840.0   * -14.0*al*a22

    kspin_e(2,5)    =   mu(ie)/420.0            * 22.0*al*a12       + &
                        (mu(je)-mu(ie))/840.0   * 14.0*al*a12

    kspin_e(2,7)    =   mu(ie)/420.0            * 54.0*a12          + &
                        (mu(je)-mu(ie))/840.0   * 54.0*a12

    kspin_e(2,8)    =   mu(ie)/420.0            * 54.0*a22          + &
                        (mu(je)-mu(ie))/840.0   * 54.0*a22

    kspin_e(2,9)    =   mu(ie)/420.0            * 63.0*a23          + &
                        (mu(je)-mu(ie))/840.0   * 56.0*a23

    kspin_e(2,10)   =   mu(ie)/420.0            * 13.0*al*a22       + &
                        (mu(je)-mu(ie))/840.0   * 12.0*al*a22

    kspin_e(2,11)   =   mu(ie)/420.0            * -13.0*al*a12      + &
                        (mu(je)-mu(ie))/840.0   * -12.0*al*a12

    kspin_e(3,3)    =   mu(ie)/420.0            * 140.0*a33         + &
                        (mu(je)-mu(ie))/840.0   * 70.0*a33

    kspin_e(3,4)    =   mu(ie)/420.0            * -21.0*al*a23      + &
                        (mu(je)-mu(ie))/840.0   * -14.0*al*a23

    kspin_e(3,5)    =   mu(ie)/420.0            * 21.0*al*a13       + &
                        (mu(je)-mu(ie))/840.0   * 14.0*al*a13

    kspin_e(3,7)    =   mu(ie)/420.0            * 63.0*a13          + &
                        (mu(je)-mu(ie))/840.0   * 70.0*a13

    kspin_e(3,8)    =   mu(ie)/420.0            * 63.0*a23          + &
                        (mu(je)-mu(ie))/840.0   * 70.0*a23

    kspin_e(3,9)    =   mu(ie)/420.0            * 70.0*a33          + &
                        (mu(je)-mu(ie))/840.0   * 70.0*a33

    kspin_e(3,10)   =   mu(ie)/420.0            * 14.0*al*a23       + &
                        (mu(je)-mu(ie))/840.0   * 14.0*al*a23

    kspin_e(3,11)   =   mu(ie)/420.0            * -14.0*al*a13      + &
                        (mu(je)-mu(ie))/840.0   * -14.0*al*a13

    kspin_e(4,4)    =   mu(ie)/420.0            * 4.0*al**2*a22     + &
                        (mu(je)-mu(ie))/840.0   * 3.0*al**2*a22

    kspin_e(4,5)    =   mu(ie)/420.0            * -4.0*al**2*a12    + &
                        (mu(je)-mu(ie))/840.0   * -3.0*al**2*a12

    kspin_e(4,7)    =   mu(ie)/420.0            * -13.0*al*a12      + &
                        (mu(je)-mu(ie))/840.0   * -14.0*al*a12

    kspin_e(4,8)    =   mu(ie)/420.0            * -13.0*al*a22      + &
                        (mu(je)-mu(ie))/840.0   * -14.0*al*a22

    kspin_e(4,9)    =   mu(ie)/420.0            * -14.0*al*a23      + &
                        (mu(je)-mu(ie))/840.0   * -14.0*al*a23

    kspin_e(4,10)   =   mu(ie)/420.0            * -3.0*al**2*a22    + &
                        (mu(je)-mu(ie))/840.0   * -3.0*al**2*a22

    kspin_e(4,11)   =   mu(ie)/420.0            * 3.0*al**2*a12     + &
                        (mu(je)-mu(ie))/840.0   * 3.0*al**2*a12

    kspin_e(5,5)    =   mu(ie)/420.0            * 4.0*al**2*a11     + &
                        (mu(je)-mu(ie))/840.0   * 3.0*al**2*a11

    kspin_e(5,7)    =   mu(ie)/420.0            * 13.0*al*a11       + &
                        (mu(je)-mu(ie))/840.0   * 14.0*al*a11

    kspin_e(5,8)    =   mu(ie)/420.0            * 13.0*al*a12       + &
                        (mu(je)-mu(ie))/840.0   * 14.0*al*a12

    kspin_e(5,9)    =   mu(ie)/420.0            * 14.0*al*a13       + &
                        (mu(je)-mu(ie))/840.0   * 14.0*al*a13

    kspin_e(5,10)   =   mu(ie)/420.0            * 3.0*al**2*a12     + &
                        (mu(je)-mu(ie))/840.0   * 3.0*al**2*a12

    kspin_e(5,11)   =   mu(ie)/420.0            * -3.0*al**2*a11    + &
                        (mu(je)-mu(ie))/840.0   * -3.0*al**2*a11

    kspin_e(7,7)    =   mu(ie)/420.0            * 156.0*a11         + &
                        (mu(je)-mu(ie))/840.0   * 240.0*a11

    kspin_e(7,8)    =   mu(ie)/420.0            * 156.0*a12         + &
                        (mu(je)-mu(ie))/840.0   * 240.0*a12

    kspin_e(7,9)    =   mu(ie)/420.0            * 147.0*a13         + &
                        (mu(je)-mu(ie))/840.0   * 224.0*a13

    kspin_e(7,10)   =   mu(ie)/420.0            * 22.0*al*a12       + &
                        (mu(je)-mu(ie))/840.0   * 30.0*al*a12

    kspin_e(7,11)   =   mu(ie)/420.0            * -22.0*al*a11      + &
                        (mu(je)-mu(ie))/840.0   * -30.0*al*a11

    kspin_e(8,8)    =   mu(ie)/420.0            * 156.0*a22         + &
                        (mu(je)-mu(ie))/840.0   * 240.0*a22

    kspin_e(8,9)    =   mu(ie)/420.0            * 147.0*a23         + &
                        (mu(je)-mu(ie))/840.0   * 224.0*a23

    kspin_e(8,10)   =   mu(ie)/420.0            * 22.0*al*a22       + &
                        (mu(je)-mu(ie))/840.0   * 30.0*al*a22

    kspin_e(8,11)   =   mu(ie)/420.0            * -22.0*al*a12      + &
                        (mu(je)-mu(ie))/840.0   * -30.0*al*a12

    kspin_e(9,9)    =   mu(ie)/420.0            * 140.0*a33         + &
                        (mu(je)-mu(ie))/840.0   * 210.0*a33

    kspin_e(9,10)   =   mu(ie)/420.0            * 21.0*al*a23       + &
                        (mu(je)-mu(ie))/840.0   * 22.0*al*a23

    kspin_e(9,11)   =   mu(ie)/420.0            * -21.0*al*a13      + &
                        (mu(je)-mu(ie))/840.0   * -22.0*al*a13

    kspin_e(10,10)  =   mu(ie)/420.0            * 4.0*al**2*a22     + &
                        (mu(je)-mu(ie))/840.0   * 5.0*al**2*a22

    kspin_e(10,11)  =   mu(ie)/420.0            * -4.0*al**2*a12    + &
                        (mu(je)-mu(ie))/840.0   * -5.0*al**2*a12

    kspin_e(11,11)  =   mu(ie)/420.0            * 4.0*al**2*a11     + &
                        (mu(je)-mu(ie))/840.0   * 5.0*al**2*a11

    ! Add symmetric terms
    DO i = 1, 12
        DO j = 1, 12

            kspin_e(j,i) = kspin_e(i,j)

        END DO
    END DO

    ! Multiply matrices by common factors
    kspin_e = kspin_e * al * omega**2

    ! Initialize transformation matrix
    lambda = 0.0d0

    ! Build transformation matrix
    DO i = 1, 3
        DO j = 1, 3

        ! Set index values
        i3 = i + 3
        j3 = j + 3
        i6 = i + 6
        j6 = j + 6
        i9 = i + 9
        j9 = j + 9

        ! Build 12x12 lambda matrix from 3x3 b matrix
        lambda(i,j) = b(i,j)
        lambda(i3,j3) = b(i,j)
        lambda(i6,j6) = b(i,j)
        lambda(i9,j9) = b(i,j)

        END DO
    END DO

    ! Build transpose of lambda
    DO i = 1, 12
        DO j = 1, 12

        lambdat(i,j) = lambda(j,i)

        END DO
    END DO

    ! Multiply local centrifugal force matrix by transformation
    aa = MATMUL(cf_e, lambda)

    ! Multiply previous result by transpose of transformation to obtain
    ! element global centrifugal force matrix
    cf_e = MATMUL(lambdat, aa)

    ! Multiply local gyroscopic matrix by transformation
    aa = MATMUL(gyro_e, lambda)

    ! Multiply previous result by transpose of transformation to obtain
    ! element global gyroscopic matrix
    gyro_e = MATMUL(lambdat, aa)

    ! Multiply local spin stiffness matrix by transformation
    aa = MATMUL(kspin_e, lambda)

    ! Multiply previous result by transpose of transformation to obtain
    ! element global spin stiffness matrix
    kspin_e = MATMUL(lambdat, aa)

    ! Steps to assemble global stiffness matrices
    DO i = 1, 6

        n(i)= 6 * ie - 6 + i
        n(i+6) = 6 * je - 6 + i

    END DO

    ! Place this elements contribution into the global stiffness matrix
    DO i = 1, 12
        DO j = 1, 12

            ik = n(i)
            jk = n(j)
            in = jk - ik + 1

            IF (in > 0) THEN

                cf(ik,in) = cf(ik,in) + cf_e(i,j)
                gyro(ik,in) = gyro(ik,in) + gyro_e(i,j)
                kspin(ik,in) = kspin(ik,in) + kspin_e(i,j)

            END IF

        END DO
    END DO


END DO elemloop

RETURN
END SUBROUTINE taper_frame_spin_dbl

!****************************
SUBROUTINE taper_mass_dbl(nn,ne,nd,nb,loc,cx,cy,cz,mu,jprime, &
                    alpha,ivert,nfix,kfix,gm)
!****************************
! Purpose:
!   Builds banded mass matrix for structure with space frame elements in double precision
!
! Reference:
!   Rao, S. S. (2005). "The Finite Element Method in Engineering"
!   Fourth Edition, Pergamon Press, Section 12.3.2, p. 428
!
! Record of revisions:
!    Date        Programmer      Description of change
!    ====       =============   ==================================
! 09/05/2008    S. Larwood      Fixed errors in ml(3,3) and ml(6,6)
! 01/25/2008    S. Larwood      Original code based on taper_mass_dbl 09/19/2007
! 07/19/2013    S. Andrew Ning  Removed matmul_dbl to use intrinsic MATMUL.  commented out unused vars.

IMPLICIT NONE

! Data dictionary: declare calling parameter types & definitions
INTEGER, INTENT(IN)                             :: nb       ! Bandwidth of overall stiffness matrix
INTEGER, INTENT(IN)                             :: nd       ! Total number of degrees of freedom
INTEGER, INTENT(IN)                             :: ne       ! Number of finite elements
INTEGER, INTENT(IN)                             :: nfix     ! Number of fixed degrees of freedom
INTEGER, INTENT(IN)                             :: nn       ! Number of nodes in structure
DOUBLE PRECISION, INTENT(IN), DIMENSION(ne)     :: alpha    ! Element angular alignment
DOUBLE PRECISION, INTENT(IN), DIMENSION(nn)     :: cx       ! x-coordinate of node
DOUBLE PRECISION, INTENT(IN), DIMENSION(nn)     :: cy       ! y-coordinate of node
DOUBLE PRECISION, INTENT(IN), DIMENSION(nn)     :: cz       ! z-coordinate of node
DOUBLE PRECISION, INTENT(OUT), DIMENSION(nd,nb) :: gm       ! Global mass matrix
INTEGER, INTENT(IN), DIMENSION(ne)              :: ivert    ! Orientation of element, 1 if vertical,
                                                            ! 0 if otherwise
DOUBLE PRECISION, INTENT(IN), DIMENSION(nn)     :: jprime   ! Element lineal rotational mass moment of inertia
INTEGER, INTENT(IN), DIMENSION(nfix)            :: kfix     ! Fixed degree of freedom number
DOUBLE PRECISION, INTENT(IN), DIMENSION(nn)     :: mu       ! Element lineal density
INTEGER, INTENT(IN), DIMENSION(ne,2)            :: loc      ! Global node number, loc(i,j), jth corner of element i

! Data dictionary: declare local variable types & definitions
DOUBLE PRECISION, DIMENSION(3,3)    :: b        ! matrix used for transformation
DOUBLE PRECISION                    :: a11      ! Element of transformation matrix
DOUBLE PRECISION                    :: a12      ! Element of transformation matrix
DOUBLE PRECISION                    :: a13      ! Element of transformation matrix
DOUBLE PRECISION                    :: a21      ! Element of transformation matrix
DOUBLE PRECISION                    :: a22      ! Element of transformation matrix
DOUBLE PRECISION                    :: a23      ! Element of transformation matrix
DOUBLE PRECISION, DIMENSION(12,12)  :: aa       ! Dummy matrix
DOUBLE PRECISION                    :: al       ! Element length
DOUBLE PRECISION                    :: alz      ! dx/dl
DOUBLE PRECISION                    :: amz      ! dy/dl
DOUBLE PRECISION                    :: anz      ! dz/dl
DOUBLE PRECISION                    :: cs       ! cos(alpha)
DOUBLE PRECISION                    :: d        ! Factor used in transformation
INTEGER                             :: ie       ! First corner of element
INTEGER                             :: i        ! Index
INTEGER                             :: i3       ! i + 3
INTEGER                             :: i6       ! i + 6
INTEGER                             :: i9       ! i + 9
INTEGER                             :: ii       ! Index
INTEGER                             :: ik       ! Dummy variable
INTEGER                             :: in       ! Dummy variable
INTEGER                             :: ix       ! Number of fixed DOF
INTEGER                             :: j        ! Index
INTEGER                             :: j3       ! j + 3
INTEGER                             :: j6       ! j + 6
INTEGER                             :: j9       ! j + 9
INTEGER                             :: je       ! Second corner of element
INTEGER                             :: jk       ! Dummy variable
DOUBLE PRECISION, DIMENSION(12,12)  :: ml       ! Element local mass matrix
DOUBLE PRECISION, DIMENSION(12,12)  :: lamda    ! Transformation matrix
DOUBLE PRECISION, DIMENSION(12,12)  :: lamdat   ! Transpose of lamda
DOUBLE PRECISION, DIMENSION(12,12)  :: mg       ! Element global mass matrix
DOUBLE PRECISION, DIMENSION(12)     :: n        ! Used for assembling global mass matrix
! DOUBLE PRECISION                    :: mul      ! Mu * Length
DOUBLE PRECISION                    :: ss       ! Sin(alpha)

! Initialize global stiffness matrix
gm = 0.0d0

elemloop: DO ii = 1, ne

    ! Initialize local element stiffness matrix
    ml = 0.0d0

    ! Find node element node number
    ie = loc(ii,1)
    je = loc(ii,2)

    ! Compute length
    al=DSQRT((cx(je)-cx(ie))**2 + (cy(je)-cy(ie))**2 &
            +(cz(je)-cz(ie))**2)

    ! Build local stiffness matrix
    ml(1,1)     = al/35.0 * (10.0 * mu(ie) + 3.0 * mu(je) )
    ml(1,5)     = al**2/420.0 * (17.0 * mu(ie) + 5.0 * mu(je) )
    ml(1,7)     = 9.0 * al/140.0 * (mu(ie) + mu(je) )
    ml(1,11)    = -al**2/420.0 * (7.0 * mu(ie) + 6.0 * mu(je) )
    ml(2,2)     = al/35.0 * (10.0 * mu(ie) + 3.0 * mu(je) )
    ml(2,4)     = -al**2/420.0 * (17.0 * mu(ie) + 5.0 * mu(je) )
    ml(2,8)     = 9.0 * al/140.0 * (mu(ie) + mu(je) )
    ml(2,10)    = al**2/420.0 * (7.0 * mu(ie) + 6.0 * mu(je) )
    ml(3,3)     = al/12.0 * (3.0 * mu(ie) + mu(je) )
    ml(3,9)     = al/12.0 * (mu(ie) + mu(je) )
    ml(4,4)     = al**3/840.0 * (5.0 * mu(ie) + 3.0 * mu(je) )
    ml(4,8)     = -al**2/420.0 * (6.0 * mu(ie) + 7.0 * mu(je) )
    ml(4,10)    = -al**3/280.0 * (mu(ie) + mu(je) )
    ml(5,5)     = al**3/840.0 * (5.0 * mu(ie) + 3.0 * mu(je) )
    ml(5,7)     = al**2/420.0 * (6.0 * mu(ie) + 7.0 * mu(je) )
    ml(5,11)    = -al**3/280.0 * (mu(ie) + mu(je) )
    ml(6,6)     = al/12.0 * (3.0 * jprime(ie) + jprime(je) )
    ml(6,12)    = al/12.0 * (jprime(ie) + jprime(je) )
    ml(7,7)     = al/35.0 * (3.0 * mu(ie) + 10.0 * mu(je) )
    ml(7,11)    = -al**2/420.0 * (7.0 * mu(ie) + 15.0 * mu(je) )
    ml(8,8)     = al/35.0 * (3.0 * mu(ie) + 10.0 * mu(je) )
    ml(8,10)    = al**2/420.0 * (7.0 * mu(ie) + 15.0 * mu(je) )
    ml(9,9)     = al/4.0 * (mu(ie)/3.0 + mu(je) )
    ml(10,10)   = al**3/840.0 * (3.0 * mu(ie) + 5.0 * mu(je) )
    ml(11,11)   = al**3/840.0 * (3.0 * mu(ie) + 5.0 * mu(je) )
    ml(12,12)   = al/4.0 * (jprime(ie)/3.0 + jprime(je) )

    ! Add symmetric terms
    DO i = 1, 12
        DO j = 1, 12

            ml(j,i) = ml(i,j)

        END DO
    END DO

    ! Initialize transformation matrix
    lamda = 0.0d0

    ! Compute slopes
    alz = (cx(je) - cx(ie))/al
    amz = (cy(je) - cy(ie))/al
    anz = (cz(je) - cz(ie))/al

    ! If element is vertical there is special handling
    IF (ivert(ii) == 1) THEN

        b(1,1) = 0.0d0
        b(1,2) = -DSIN(alpha(ii))
        b(1,3) = -DCOS(alpha(ii))
        b(2,1) = 0.0d0
        b(2,2) = DCOS(alpha(ii))
        b(2,3) = -DSIN(alpha(ii))
        b(3,1) = 1.0d0
        b(3,2) = 0.0d0
        b(3,3) = 0.0d0

    ELSE

        d = DSQRT(amz**2 + anz**2)
        a11 = (amz**2 + anz**2)/d
        a12 = -(alz * amz) /d
        a13 = -(alz * anz) /d
        a21 = 0.0d0
        a22 = anz/d
        a23 = -amz/d
        cs = DCOS(alpha(ii))
        ss = DSIN(alpha(ii))
        b(1,1) = a11 * cs - a21 * ss
        b(1,2) = a12 * cs - a22 * ss
        b(1,3) = a13 * cs - a23 * ss
        b(2,1) = a11 * ss + a21 * cs
        b(2,2) = a12 * ss + a22 * cs
        b(2,3) = a13 * ss + a23 * cs
        b(3,1) = alz
        b(3,2) = amz
        b(3,3) = anz

    END IF

    ! Build transformation matrix
    DO i = 1, 3
        DO j = 1, 3

        ! Set index values
        i3 = i + 3
        j3 = j + 3
        i6 = i + 6
        j6 = j + 6
        i9 = i + 9
        j9 = j + 9

        ! Build 12x12 lamda matrix from 3x3 b matrix
        lamda(i,j) = b(i,j)
        lamda(i3,j3) = b(i,j)
        lamda(i6,j6) = b(i,j)
        lamda(i9,j9) = b(i,j)

        END DO
    END DO

    ! Build transpose of lamda
    DO i = 1, 12
        DO j = 1, 12

        lamdat(i,j) = lamda(j,i)

        END DO
    END DO

    ! Multiply local stiffness by transformation
    aa = MATMUL(ml, lamda)

    ! Multiply previous result by transpose of transformation to obtain
    ! element global stiffness matrix
    mg = MATMUL(lamdat, aa)

    ! Steps to assemble global stiffness matrix
    DO i = 1, 6

        n(i)= 6 * ie - 6 + i
        n(i+6) = 6 * je - 6 + i

    END DO

    ! Place this elements contribution into the global stiffness matrix
    DO i = 1, 12
        DO j = 1, 12

            ik = n(i)
            jk = n(j)
            in = jk - ik + 1

            IF (in > 0) THEN

                gm(ik,in) = gm(ik,in) + mg(i,j)

            END IF

        END DO
    END DO


END DO elemloop

! Incorporate boundary conditions
DO i = 1, nfix

    ix = kfix(i)
    gm(ix,1)= gm(ix,1)* 1.0E6

END DO

RETURN
END SUBROUTINE taper_mass_dbl

!****************************
SUBROUTINE taper_stiff_dbl(nn,ne,nd,nb,loc,cx,cy,cz,ea,eix,eiy,gj, &
                    alpha,ivert,nfix,kfix,gs)
!****************************
! Purpose:
!   Builds banded stiffness matrix for structure with space frame elements
!   and linearly tapered properties in double precision.
!
! Reference:
!   Rao, S. S. (2005). "The Finite Element Method in Engineering"
!   Fourth Edition, Pergamon Press, Section 9.6
!
! Record of revisions:
!    Date        Programmer      Description of change
!    ====       =============   ==================================
! 01/25/2008    S. Larwood      Original code.based on taper_stiff 09/05/2007
!                                   Values for matrix from Dissertation Journal p. 6-25
! 07/19/2013    S. Andrew Ning  Removed matmul_dbl to use intrinsic MATMUL
!
IMPLICIT NONE

! Data dictionary: declare calling parameter types & definitions
INTEGER, INTENT(IN)                             :: nb       ! Bandwidth of overall stiffness matrix
INTEGER, INTENT(IN)                             :: nd       ! Total number of degrees of freedom
INTEGER, INTENT(IN)                             :: ne       ! Number of finite elements
INTEGER, INTENT(IN)                             :: nfix     ! Number of fixed degrees of freedom
INTEGER, INTENT(IN)                             :: nn       ! Number of nodes in structure
DOUBLE PRECISION, INTENT(IN), DIMENSION(ne)     :: alpha    ! Element angular alignment
DOUBLE PRECISION, INTENT(IN), DIMENSION(nn)     :: cx       ! x-coordinate of node
DOUBLE PRECISION, INTENT(IN), DIMENSION(nn)     :: cy       ! y-coordinate of node
DOUBLE PRECISION, INTENT(IN), DIMENSION(nn)     :: cz       ! z-coordinate of node
DOUBLE PRECISION, INTENT(IN), DIMENSION(nn)     :: ea       ! Element EA extensional stiffness
DOUBLE PRECISION, INTENT(IN), DIMENSION(nn)     :: eix      ! Element EIxx bending stiffness
DOUBLE PRECISION, INTENT(IN), DIMENSION(nn)     :: eiy      ! Element EIyy bending stiffness
DOUBLE PRECISION, INTENT(IN), DIMENSION(nn)     :: gj       ! Element GJ torsional stiffness
DOUBLE PRECISION, INTENT(OUT), DIMENSION(nd,nb) :: gs       ! Global stiffness matrix
INTEGER, INTENT(IN), DIMENSION(ne)              :: ivert    ! Orientation of element, 1 if vertical,
                                                            ! 0 if otherwise
INTEGER, INTENT(IN), DIMENSION(nfix)            :: kfix     ! Fixed degree of freedom number
INTEGER, INTENT(IN), DIMENSION(ne,2)            :: loc      ! Global node number, loc(i,j), jth corner of element i

! Data dictionary: declare local variable types & definitions
DOUBLE PRECISION, DIMENSION(3,3)    :: b        ! matrix used for transformation
DOUBLE PRECISION                    :: a11      ! Element of transformation matrix
DOUBLE PRECISION                    :: a12      ! Element of transformation matrix
DOUBLE PRECISION                    :: a13      ! Element of transformation matrix
DOUBLE PRECISION                    :: a21      ! Element of transformation matrix
DOUBLE PRECISION                    :: a22      ! Element of transformation matrix
DOUBLE PRECISION                    :: a23      ! Element of transformation matrix
DOUBLE PRECISION, DIMENSION(12,12)  :: aa       ! Dummy matrix
DOUBLE PRECISION                    :: al       ! Element length
DOUBLE PRECISION                    :: alz      ! dx/dl
DOUBLE PRECISION                    :: amz      ! dy/dl
DOUBLE PRECISION                    :: anz      ! dz/dl
DOUBLE PRECISION                    :: cs       ! cos(alpha)
DOUBLE PRECISION                    :: d        ! Factor used in transformation
INTEGER                             :: ie       ! First corner of element
INTEGER                             :: i        ! Index
INTEGER                             :: i3       ! i + 3
INTEGER                             :: i6       ! i + 6
INTEGER                             :: i9       ! i + 9
INTEGER                             :: ii       ! Index
INTEGER                             :: ik       ! Dummy variable
INTEGER                             :: in       ! Dummy variable
INTEGER                             :: ix       ! Number of fixed DOF
INTEGER                             :: j        ! Index
INTEGER                             :: j3       ! j + 3
INTEGER                             :: j6       ! j + 6
INTEGER                             :: j9       ! j + 9
INTEGER                             :: je       ! Second corner of element
INTEGER                             :: jk       ! Dummy variable
DOUBLE PRECISION, DIMENSION(12,12)  :: kl       ! Element local stiffness matrix
DOUBLE PRECISION, DIMENSION(12,12)  :: lamda    ! Transformation matrix
DOUBLE PRECISION, DIMENSION(12,12)  :: lamdat   ! Transpose of lamda
DOUBLE PRECISION, DIMENSION(12,12)  :: kg       ! Element global stiffness matrix
DOUBLE PRECISION, DIMENSION(12)     :: n        ! Used for assembling global stiffness matrix
DOUBLE PRECISION                    :: ss       ! Sin(alpha)

! Initialize global stiffness matrix
gs = 0.0d0

elemloop: DO ii = 1, ne

    ! Initialize local element stiffness matrix
    kl = 0.0d0

    ! Find node element node number
    ie = loc(ii,1)
    je = loc(ii,2)

    ! Compute length
    al=DSQRT((cx(je)-cx(ie))**2 + (cy(je)-cy(ie))**2 &
            +(cz(je)-cz(ie))**2)

    ! Build local stiffness matrix- terms in upper triangle
    kl(1,1)     = 6.0/al**3 * (eiy(ie) + eiy(je))
    kl(1,5)     = 2.0/al**2 * (2.0 * eiy(ie) + eiy(je))
    kl(1,7)     = -6.0/al**3 * (eiy(ie) + eiy(je))
    kl(1,11)    = 2.0/al**2 * (eiy(ie) + 2.0 * eiy(je))
    kl(2,2)     = 6.0/al**3 * (eix(ie) + eix(je))
    kl(2,4)     = -2.0/al**2 * (2.0* eix(ie) + eix(je))
    kl(2,8)     = -6.0/al**3 * (eix(ie) + eix(je))
    kl(2,10)    = -2.0/al**2 * (eix(ie) + 2.0 * eix(je))
    kl(3,3)     = (ea(ie) + ea(je))/(2.0 * al)
    kl(3,9)     = -(ea(ie) + ea(je))/(2.0 * al)
    kl(4,4)     = 1.0/al * (3.0 * eix(ie) + eix(je))
    kl(4,8)     = 2.0/al**2 * (2.0 * eix(ie) + eix(je))
    kl(4,10)    = 1.0/al * (eix(ie) + eix(je))
    kl(5,5)     = 1.0/al * (3.0 * eiy(ie) + eiy(je))
    kl(5,7)     = -2.0/al**2 * (2.0 * eiy(ie) + eiy(je))
    kl(5,11)    = 1.0/al * (eiy(ie) + eiy(je))
    kl(6,6)     = (gj(ie) + gj(je))/(2.0 * al)
    kl(6,12)    = -(gj(ie) + gj(je))/(2.0 * al)
    kl(7,7)     = 6.0/al**3 * (eiy(ie) + eiy(je))
    kl(7,11)    = -2.0/al**2 * (eiy(ie) + 2.0 * eiy(je))
    kl(8,8)     = 6.0/al**3 * (eix(ie) + eix(je))
    kl(8,10)    = 2.0/al**2 * (eix(ie) + 2.0 * eix(je))
    kl(9,9)     = (ea(ie) + ea(je))/(2.0 * al)
    kl(10,10)   = 1.0/al * (eix(ie) + 3.0 * eix(je))
    kl(11,11)   = 1.0/al * (eiy(ie) + 3.0 * eiy(je))
    kl(12,12)   = (gj(ie) + gj(je))/(2.0 * al)

    ! Add symmetric terms in lower triangle
    DO i = 1, 12
        DO j = 1, 12

            kl(j,i) = kl(i,j)

        END DO
    END DO

    ! Initialize transformation matrix
    lamda = 0.0d0

    ! Compute slopes
    alz = (cx(je) - cx(ie))/al
    amz = (cy(je) - cy(ie))/al
    anz = (cz(je) - cz(ie))/al

    ! If element is vertical there is special handling
    IF (ivert(ii) == 1) THEN

        b(1,1) = 0.0d0
        b(1,2) = -DSIN(alpha(ii))
        b(1,3) = -DCOS(alpha(ii))
        b(2,1) = 0.0d0
        b(2,2) = DCOS(alpha(ii))
        b(2,3) = -DSIN(alpha(ii))
        b(3,1) = 1.0d0
        b(3,2) = 0.0d0
        b(3,3) = 0.0d0

    ELSE

        d = DSQRT(amz**2 + anz**2)
        a11 = (amz**2 + anz**2)/d
        a12 = -(alz * amz) /d
        a13 = -(alz * anz) /d
        a21 = 0.0d0
        a22 = anz/d
        a23 = -amz/d
        cs = DCOS(alpha(ii))
        ss = DSIN(alpha(ii))
        b(1,1) = a11 * cs - a21 * ss
        b(1,2) = a12 * cs - a22 * ss
        b(1,3) = a13 * cs - a23 * ss
        b(2,1) = a11 * ss + a21 * cs
        b(2,2) = a12 * ss + a22 * cs
        b(2,3) = a13 * ss + a23 * cs
        b(3,1) = alz
        b(3,2) = amz
        b(3,3) = anz

    END IF


    ! Build transformation matrix
    DO i = 1, 3
        DO j = 1, 3

        ! Set index values
        i3 = i + 3
        j3 = j + 3
        i6 = i + 6
        j6 = j + 6
        i9 = i + 9
        j9 = j + 9

        ! Build 12x12 lamda matrix from 3x3 b matrix
        lamda(i,j) = b(i,j)
        lamda(i3,j3) = b(i,j)
        lamda(i6,j6) = b(i,j)
        lamda(i9,j9) = b(i,j)

        END DO
    END DO

    ! Build transpose of lamda
    DO i = 1, 12
        DO j = 1, 12

        lamdat(i,j) = lamda(j,i)

        END DO
    END DO

    ! Multiply local stiffness by transformation
    aa = MATMUL(kl, lamda)

    ! Multiply previous result by transpose of transformation to obtain
    ! element global stiffness matrix
    kg = MATMUL(lamdat, aa)

    ! Steps to assemble global stiffness matrix
    DO i = 1, 6

        n(i)= 6 * ie - 6 + i
        n(i+6) = 6 * je - 6 + i

    END DO

    ! Place this elements contribution into the global stiffness matrix
    DO i = 1, 12
        DO j = 1, 12

            ik = n(i)
            jk = n(j)
            in = jk - ik + 1

            IF (in > 0) THEN

                gs(ik,in) = gs(ik,in) + kg(i,j)

            END IF

        END DO
    END DO


END DO elemloop

! Incorporate boundary conditions
DO i = 1, nfix

    ix = kfix(i)
    gs(ix,1)= gs(ix,1)* 1.0E6

END DO

RETURN
END SUBROUTINE taper_stiff_dbl





SUBROUTINE eigensolve(A, B, n, sd, eig)

    ! Description:
    !   solves generalized eigenvalue problem Ax = lambda * Bx
    !
    ! Changelog:
    !   07/18/2013      S. Andrew Ning      uses LAPACK (much faster than previous method in CurveFEM)
    !

    IMPLICIT NONE

    ! in
    INTEGER, INTENT(IN) :: n        ! Rows/columns of matrix A and B
    INTEGER, INTENT(IN) :: sd       ! number of super diagonals of matrix A and B
    DOUBLE PRECISION, INTENT(IN), DIMENSION(n, n) :: A, B

    ! out
    DOUBLE PRECISION, INTENT(OUT), DIMENSION(n) :: eig  ! eigenvalues in ascending order

    ! local
    CHARACTER *1 :: jobz, uplo
    INTEGER :: lda, ldz, info
    DOUBLE PRECISION, DIMENSION(:, :), ALLOCATABLE :: AB, BB
    DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: work
    INTEGER     :: i, j, status
    DOUBLE PRECISION, DIMENSION(1) :: z


    jobz = 'N'  ! compute eigenvalues only
    uplo = 'U'  ! data stored in upper triangle

    lda = sd + 1
    ldz = 1

    ! allocate
    ALLOCATE (AB(lda, n), STAT=status)
    IF (status /= 0) WRITE(*,*) 'Error allocating AB'
    ALLOCATE (BB(lda, n), STAT=status)
    IF (status /= 0) WRITE(*,*) 'Error allocating BB'
    ALLOCATE (work(3*n), STAT=status)
    IF (status /= 0) WRITE(*,*) 'Error allocating work'

    ! if UPLO = 'U', AB(ka+1+i-j,j) = A(i,j) for max(1,j-ka)<=i<=j;
    do j = 1, n
        do i = max(1, j-sd), j
            AB(sd+1+i-j, j) = A(i, j)
            BB(sd+1+i-j, j) = B(i, j)
        end do
    end do

    CALL DSBGV(jobz, uplo, n, sd, sd, AB, lda, BB, lda, eig, z, &
        ldz, work, info)

    ! deallocate
    DEALLOCATE (AB, STAT=status)
    IF (status /= 0) WRITE(*,*) 'Error deallocating AB'
    DEALLOCATE (BB, STAT=status)
    IF (status /= 0) WRITE(*,*) 'Error deallocating BB'
    DEALLOCATE (work, STAT=status)
    IF (status /= 0) WRITE(*,*) 'Error deallocating work'

    RETURN

END SUBROUTINE eigensolve



SUBROUTINE frequencies(NBlInpSt, omegaRPM, BldFlexL, HubRad, BlFract, &
    StrcTwst, BMassDen, FlpStff, EdgStff, GJStff, EAStff, &
    rhoJIner, PrecrvRef, PreswpRef, freq)

    !
    ! Description:
    !   Compute natural frequencies of structure (Hz)
    !   Allows for direct access (e.g., from Python) without needing to write an input file
    !   Substituted in LAPACK call to compute eigenvalues for much faster speed
    !   Streamlined input set.
    !
    ! ChangeLog:
    !   07/18/2013      S. Andrew Ning      Simplified version of S. Larwood's main routine and read_input
    !

    IMPLICIT NONE

    ! parameters
!     INTEGER, PARAMETER      :: ReKi = selected_real_kind(15, 307)
    DOUBLE PRECISION, PARAMETER :: RPM2RPS = 0.10471975511965977d0 ! Factor to convert revolutions per minute to radians per second.
    REAL, PARAMETER             :: D2R = 0.017453292519943295d0    ! Factor to convert degrees to radians.
    REAL, PARAMETER             :: RPS2HZ = 0.15915494309189535d0 ! Factor to convert radians/sec to Hertz
    INTEGER, PARAMETER          :: nb = 12              ! Bandwidth of overall stiffness matrix
    INTEGER, PARAMETER          :: nfix = 6             ! Number of fixed degrees of freedom
    DOUBLE PRECISION,PARAMETER  :: SmllNmbr  = 9.999E-4 ! A small number used to define rotational inertia to avoid singularities

    ! inputs
    INTEGER, INTENT(IN)                               :: NBlInpSt  ! Number of blade input stations
    DOUBLE PRECISION, INTENT(IN)                      :: omegaRPM  ! Rotational speed, RPM
    DOUBLE PRECISION, INTENT(IN)                      :: BldFlexL  ! Blade flexible length
    DOUBLE PRECISION, INTENT(IN)                      :: HubRad    ! Hub radius
    DOUBLE PRECISION, DIMENSION(NBlInpSt), INTENT(IN) :: BlFract   ! Non-dimensional blade fraction
    DOUBLE PRECISION, DIMENSION(NBlInpSt), INTENT(IN) :: StrcTwst  ! Structural twist
    DOUBLE PRECISION, DIMENSION(NBlInpSt), INTENT(IN) :: BMassDen  ! Blade lineal density
    DOUBLE PRECISION, DIMENSION(NBlInpSt), INTENT(IN) :: FlpStff   ! Blade flapwise stiffness
    DOUBLE PRECISION, DIMENSION(NBlInpSt), INTENT(IN) :: EdgStff   ! Blade edgewise stiffness
    DOUBLE PRECISION, DIMENSION(NBlInpSt), INTENT(IN) :: GJStff    ! Blade torsional stiffness
    DOUBLE PRECISION, DIMENSION(NBlInpSt), INTENT(IN) :: EAStff    ! Blade extensional stiffness
    DOUBLE PRECISION, DIMENSION(NBlInpSt), INTENT(IN) :: rhoJIner  ! Blade axial mass moment of inertia per unit length
    DOUBLE PRECISION, DIMENSION(NBlInpSt), INTENT(IN) :: PrecrvRef ! Blade out of plane curvature
    DOUBLE PRECISION, DIMENSION(NBlInpSt), INTENT(IN) :: PreswpRef ! Blade in-plane sweep


    ! outputs
    DOUBLE PRECISION, DIMENSION(NBlInpSt * 6), INTENT(OUT) :: freq  ! natural frequencies in Hz


    ! local
    INTEGER                     :: i        ! Loop index
    DOUBLE PRECISION            :: omega    ! Rotational speed, rad/s
    INTEGER, DIMENSION(nfix)    :: kfix     ! Fixed degree of freedom number

    DOUBLE PRECISION, DIMENSION(NBlInpSt - 1)                 :: alpha2  ! Element angular alignment in radians
    DOUBLE PRECISION, DIMENSION(NBlInpSt * 6, NBlInpSt * 6)   :: cf      ! Global centrifugal force matrix
    DOUBLE PRECISION, DIMENSION(NBlInpSt * 6, NBlInpSt * 6)   :: gm      ! Generalized mass matrix
    DOUBLE PRECISION, DIMENSION(NBlInpSt * 6, NBlInpSt * 6)   :: gs      ! Global stiffness matrix
    DOUBLE PRECISION, DIMENSION(NBlInpSt * 6, NBlInpSt * 6)   :: gyro    ! Global gyroscopic matrix
    INTEGER, DIMENSION(NBlInpSt - 1)                          :: ivert   ! Orientation of element, 1 if vertical, 0 if otherwise
    DOUBLE PRECISION, DIMENSION(NBlInpSt * 6, NBlInpSt * 6)   :: kspin   ! Global spin-stiffness matrix
    INTEGER, DIMENSION(NBlInpSt - 1, 2)                       :: loc     ! Global node number, loc(i,j), jth corner of element i
    DOUBLE PRECISION, DIMENSION(NBlInpSt * 6)                 :: eig     ! vector of actual eigenvectors


    ! Convert to radians
    omega = omegaRPM * RPM2RPS

    ! Set values of global node numbers
    DO i = 1, NBlInpSt - 1
        ! Number of inboard node
        loc(i,1) = i

        ! Number of outboard node
        loc(i,2) = i + 1

    END DO

    ! Set values for alpha, whose array size is the number of elements (tip not used)
    DO i = 1, NBlInpSt - 1
        alpha2(i) = StrcTwst(i) * D2R

    END DO

    ! Set value of ivert
    ivert = 0

    ! Set values for fixed degrees of freedom
    kfix = (/1, 2, 3, 4, 5, 6/)

    ! Obtain stiffness matrix, gs
    CALL taper_stiff_dbl(   NBlInpSt,NBlInpSt - 1,NBlInpSt * 6,nb,loc,PrecrvRef,PreswpRef, &
                            BlFract * BldFlexL + HubRad,EAStff,EdgStff,FlpStff,GJStff, &
                            alpha2,ivert,nfix,kfix,gs)

    ! Build the mass matrix
    CALL taper_mass_dbl(NBlInpSt,NBlInpSt - 1,NBlInpSt * 6,nb,loc,PrecrvRef, &
                        PreswpRef,BlFract * BldFlexL + HubRad,BMassDen, &
                        rhoJIner + SmllNmbr,alpha2,ivert,nfix,kfix,gm)

    ! Build the gyroscopic matrices
    CALL taper_frame_spin_dbl(  NBlInpSt,NBlInpSt - 1,NBlInpSt * 6,nb,loc, &
                                PrecrvRef,PreswpRef,BlFract * BldFlexL + HubRad, &
                                BMassDen,alpha2,ivert,omega,gyro,cf,kspin)

    ! Convert various matrices from banded to full
    CALL band2full_dbl(NBlInpSt * 6,nb,gs)
    CALL band2full_dbl(NBlInpSt * 6,nb,gm)
    CALL band2full_dbl(NBlInpSt * 6,nb,cf)
    CALL band2full_dbl(NBlInpSt * 6,nb,kspin)

    ! solve eigenvalues
    CALL eigensolve(gs + cf - kspin, gm, NBlInpSt*6, nb-1, eig)

    ! convert to frequencies (Hz)
    DO i = 1, NBlInpSt *6

        freq(i) = DSQRT(eig(i)) * RPS2HZ

    END DO



END SUBROUTINE frequencies



