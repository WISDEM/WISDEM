! Copyright 2019 NREL

! Licensed under the Apache License, Version 2.0 (the "License"); you may not use
! this file except in compliance with the License. You may obtain a copy of the
! License at http://www.apache.org/licenses/LICENSE-2.0

! Unless required by applicable law or agreed to in writing, software distributed
! under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
! CONDITIONS OF ANY KIND, either express or implied. See the License for the
! specific language governing permissions and limitations under the License.
! -------------------------------------------------------------------------------------------
! This module contains basic functions used by the controller

! Functions:
!       AeroDynTorque: Calculate aerodynamic torque
!       ColemanTransform: Perform Colemant transform
!       ColemanTransformInverse: Perform inverse Colemant transform
!       CPFunction: Find Cp using parameterized surface
!       Debug: Debug the controller
!       DFController: DF control
!       identity: Make identity matrix
!       interp1d: 1-d interpolation
!       interp2d: 2-d interpolation
!       matinv3: 3x3 matrix inverse
!       PIController: implement a PI controller
!       ratelimit: Rate limit signal
!       saturate: Saturate signal

MODULE Functions

USE Constants
USE Filters

IMPLICIT NONE

CONTAINS
!-------------------------------------------------------------------------------------------------------------------------------
    REAL FUNCTION saturate(inputValue, minValue, maxValue)
    ! Saturates inputValue. Makes sure it is not smaller than minValue and not larger than maxValue

        IMPLICIT NONE

        REAL(4), INTENT(IN)     :: inputValue
        REAL(4), INTENT(IN)     :: minValue
        REAL(4), INTENT(IN)     :: maxValue

        saturate = MIN(MAX(inputValue,minValue), maxValue)

    END FUNCTION saturate
!-------------------------------------------------------------------------------------------------------------------------------
    REAL FUNCTION ratelimit(inputSignal, inputSignalPrev, minRate, maxRate, DT)
    ! Saturates inputValue. Makes sure it is not smaller than minValue and not larger than maxValue
        IMPLICIT NONE

        REAL(4), INTENT(IN)     :: inputSignal
        REAL(4), INTENT(IN)     :: inputSignalPrev
        REAL(4), INTENT(IN)     :: minRate
        REAL(4), INTENT(IN)     :: maxRate
        REAL(4), INTENT(IN)     :: DT
        ! Local variables
        REAL(4)                 :: rate

        rate = (inputSignal - inputSignalPrev)/DT                       ! Signal rate (unsaturated)
        rate = saturate(rate, minRate, maxRate)                 ! Saturate the signal rate
        ratelimit = inputSignalPrev + rate*DT                       ! Saturate the overall command using the rate limit

    END FUNCTION ratelimit
!-------------------------------------------------------------------------------------------------------------------------------
    REAL FUNCTION PIController(error, kp, ki, minValue, maxValue, DT, I0, reset, inst)
    ! PI controller, with output saturation

        IMPLICIT NONE
        ! Allocate Inputs
        REAL(4), INTENT(IN)         :: error
        REAL(4), INTENT(IN)         :: kp
        REAL(4), INTENT(IN)         :: ki
        REAL(4), INTENT(IN)         :: minValue
        REAL(4), INTENT(IN)         :: maxValue
        REAL(4), INTENT(IN)         :: DT
        INTEGER(4), INTENT(INOUT)   :: inst
        REAL(4), INTENT(IN)         :: I0
        LOGICAL, INTENT(IN)         :: reset     
        ! Allocate local variables
        INTEGER(4)                      :: i                                            ! Counter for making arrays
        REAL(4)                         :: PTerm                                        ! Proportional term
        REAL(4), DIMENSION(99), SAVE    :: ITerm = (/ (real(9999.9), i = 1,99) /)       ! Integral term, current.
        REAL(4), DIMENSION(99), SAVE    :: ITermLast = (/ (real(9999.9), i = 1,99) /)   ! Integral term, the last time this controller was called. Supports 99 separate instances.
        INTEGER(4), DIMENSION(99), SAVE :: FirstCall = (/ (1, i=1,99) /)                ! First call of this function?
        
        ! Initialize persistent variables/arrays, and set inital condition for integrator term
        IF ((FirstCall(inst) == 1) .OR. reset) THEN
            ITerm(inst) = I0
            ITermLast(inst) = I0
            
            FirstCall(inst) = 0
            PIController = I0
        ELSE
            PTerm = kp*error
            ITerm(inst) = ITerm(inst) + DT*ki*error
            ITerm(inst) = saturate(ITerm(inst), minValue, maxValue)
            PIController = PTerm + ITerm(inst)
        
            ITermLast(inst) = ITerm(inst)
        END IF
        inst = inst + 1
        
    END FUNCTION PIController
!-------------------------------------------------------------------------------------------------------------------------------
    REAL FUNCTION PIIController(error, error2, kp, ki, ki2, minValue, maxValue, DT, I0, reset, inst)
    ! PI controller, with output saturation. 
    ! Added error2 term for additional integral control input

        IMPLICIT NONE
        ! Allocate Inputs
        REAL(4), INTENT(IN)         :: error
        REAL(4), INTENT(IN)         :: error2
        REAL(4), INTENT(IN)         :: kp
        REAL(4), INTENT(IN)         :: ki2
        REAL(4), INTENT(IN)         :: ki
        REAL(4), INTENT(IN)         :: minValue
        REAL(4), INTENT(IN)         :: maxValue
        REAL(4), INTENT(IN)         :: DT
        INTEGER(4), INTENT(INOUT)   :: inst
        REAL(4), INTENT(IN)         :: I0
        LOGICAL, INTENT(IN)         :: reset     
        ! Allocate local variables
        INTEGER(4)                      :: i                                            ! Counter for making arrays
        REAL(4)                         :: PTerm                                        ! Proportional term
        REAL(4), DIMENSION(99), SAVE    :: ITerm = (/ (real(9999.9), i = 1,99) /)       ! Integral term, current.
        REAL(4), DIMENSION(99), SAVE    :: ITermLast = (/ (real(9999.9), i = 1,99) /)   ! Integral term, the last time this controller was called. Supports 99 separate instances.
        REAL(4), DIMENSION(99), SAVE    :: ITerm2 = (/ (real(9999.9), i = 1,99) /)       ! Second Integral term, current.
        REAL(4), DIMENSION(99), SAVE    :: ITermLast2 = (/ (real(9999.9), i = 1,99) /)   ! Second Integral term, the last time this controller was called. Supports 99 separate instances.
        INTEGER(4), DIMENSION(99), SAVE :: FirstCall = (/ (1, i=1,99) /)                ! First call of this function?
        
        ! Initialize persistent variables/arrays, and set inital condition for integrator term
        IF ((FirstCall(inst) == 1) .OR. reset) THEN
            ITerm(inst) = I0
            ITermLast(inst) = I0
            ITerm2(inst) = I0
            ITermLast2(inst) = I0
            
            FirstCall(inst) = 0
            PIIController = I0
        ELSE
            PTerm = kp*error
            ITerm(inst) = ITerm(inst) + DT*ki*error
            ITerm2(inst) = ITerm2(inst) + DT*ki2*error2
            ITerm(inst) = saturate(ITerm(inst), minValue, maxValue)
            ITerm2(inst) = saturate(ITerm2(inst), minValue, maxValue)
            PIIController = PTerm + ITerm(inst) + ITerm2(inst)
            PIIController = saturate(PIIController, minValue, maxValue)
        
            ITermLast(inst) = ITerm(inst)
        END IF
        inst = inst + 1
        
    END FUNCTION PIIController
!-------------------------------------------------------------------------------------------------------------------------------
    REAL FUNCTION interp1d(xData, yData, xq)
    ! interp1d 1-D interpolation (table lookup), xData should be monotonically increasing

        IMPLICIT NONE
        ! Inputs
        REAL(4), DIMENSION(:), INTENT(IN)       :: xData        ! Provided x data (vector), to be interpolated
        REAL(4), DIMENSION(:), INTENT(IN)       :: yData        ! Provided y data (vector), to be interpolated
        REAL(4), INTENT(IN)                     :: xq           ! x-value for which the y value has to be interpolated
        INTEGER(4)                              :: I            ! Iteration index
        
        ! Interpolate
        IF (xq <= MINVAL(xData)) THEN
            interp1d = yData(1)
        ELSEIF (xq >= MAXVAL(xData)) THEN
            interp1d = yData(SIZE(xData))
        ELSE
            DO I = 1, SIZE(xData)
                IF (xq <= xData(I)) THEN
                    interp1d = yData(I-1) + (yData(I) - yData(I-1))/(xData(I) - xData(I-1))*(xq - xData(I-1))
                    EXIT
                ELSE
                    CONTINUE
                END IF
            END DO
        END IF
        
    END FUNCTION interp1d
!-------------------------------------------------------------------------------------------------------------------------------
    REAL FUNCTION interp2d(xData, yData, zData, xq, yq)
    ! interp2d 2-D interpolation (table lookup). Query done using bilinear interpolation. 
    ! Note that the interpolated matrix with associated query vectors may be different than "standard", - zData should be formatted accordingly
    ! - xData follows the matrix from left to right
    ! - yData follows the matrix from top to bottom
    ! A simple case: xData = [1 2 3], yData = [4 5 6]
    !        | 1    2   3
    !       -------------
    !       4| a    b   c
    !       5| d    e   f
    !       6| g    H   i

        IMPLICIT NONE
        ! Inputs
        REAL(4), DIMENSION(:),   INTENT(IN)     :: xData        ! Provided x data (vector), to find query point (should be monotonically increasing)
        REAL(4), DIMENSION(:),   INTENT(IN)     :: yData        ! Provided y data (vector), to find query point (should be monotonically increasing)
        REAL(4), DIMENSION(:,:), INTENT(IN)     :: zData        ! Provided z data (vector), to be interpolated
        REAL(4),                 INTENT(IN)     :: xq           ! x-value for which the z value has to be interpolated
        REAL(4),                 INTENT(IN)     :: yq           ! y-value for which the z value has to be interpolated
        ! Allocate variables
        INTEGER(4)                              :: i            ! Iteration index & query index, x-direction
        INTEGER(4)                              :: ii           ! Iteration index & second que .  ry index, x-direction
        INTEGER(4)                              :: j            ! Iteration index & query index, y-direction
        INTEGER(4)                              :: jj           ! Iteration index & second query index, y-direction
        REAL(4), DIMENSION(2,2)                 :: fQ           ! zData value at query points for bilinear interpolation            
        REAL(4), DIMENSION(1)                   :: fxy           ! Interpolated z-data point to be returned
        REAL(4)                                 :: fxy1          ! zData value at query point for bilinear interpolation            
        REAL(4)                                 :: fxy2          ! zData value at query point for bilinear interpolation            
        
        ! ---- Find corner indices surrounding desired interpolation point -----
            ! x-direction
        IF (xq <= MINVAL(xData)) THEN       ! On lower x-bound, just need to find zData(yq)
            j = 1
            jj = 1
            interp2d = interp1d(yData,zData(:,j),yq)
            RETURN
        ELSEIF (xq >= MAXVAL(xData)) THEN   ! On upper x-bound, just need to find zData(yq)
            j = size(xData)
            jj = size(xData)
            interp2d = interp1d(yData,zData(:,j),yq)
            RETURN
        ELSE
            DO j = 1,size(xData)            
                IF (xq == xData(j)) THEN ! On axis, just need 1d interpolation
                    jj = j
                    interp2d = interp1d(yData,zData(:,j),yq)
                    RETURN
                ELSEIF (xq < xData(j)) THEN
                    jj = j
                    EXIT
                ELSE
                    CONTINUE
                END IF
            END DO
        ENDIF
        j = j-1 ! Move j back one
            ! y-direction
        IF (yq <= MINVAL(yData)) THEN       ! On lower y-bound, just need to find zData(xq)
            i = 1
            ii = 1
            interp2d = interp1d(xData,zData(i,:),xq)
            RETURN
        ELSEIF (yq >= MAXVAL(yData)) THEN   ! On upper y-bound, just need to find zData(xq)
            i = size(yData)
            ii = size(yData)
            interp2d = interp1d(xData,zData(i,:),xq)
            RETURN
        ELSE
            DO i = 1,size(yData)
                IF (yq == yData(i)) THEN    ! On axis, just need 1d interpolation
                    ii = i
                    interp2d = interp1d(xData,zData(i,:),xq)
                    RETURN
                ELSEIF (yq < yData(i)) THEN
                    ii = i
                    EXIT
                ELSE
                    CONTINUE
                END IF
            END DO
        ENDIF
        i = i-1 ! move i back one
        
        ! ---- Do bilinear interpolation ----
        ! Find values at corners 
        fQ(1,1) = zData(i,j)
        fQ(2,1) = zData(ii,j)
        fQ(1,2) = zData(i,jj)
        fQ(2,2) = zData(ii,jj)
        ! Interpolate
        fxy1 = (xData(jj) - xq)/(xData(jj) - xData(j))*fQ(1,1) + (xq - xData(j))/(xData(jj) - xData(j))*fQ(1,2)
        fxy2 = (xData(jj) - xq)/(xData(jj) - xData(j))*fQ(2,1) + (xq - xData(j))/(xData(jj) - xData(j))*fQ(2,2)
        fxy = (yData(ii) - yq)/(yData(ii) - yData(i))*fxy1 + (yq - yData(i))/(yData(ii) - yData(i))*fxy2

        interp2d = fxy(1)

    END FUNCTION interp2d
!-------------------------------------------------------------------------------------------------------------------------------
    FUNCTION matinv3(A) RESULT(B)
    ! Performs a direct calculation of the inverse of a 3×3 matrix.
    ! Source: http://fortranwiki.org/fortran/show/Matrix+inversion
        REAL(4), INTENT(IN) :: A(3,3)   !! Matrix
        REAL(4)             :: B(3,3)   !! Inverse matrix
        REAL(4)             :: detinv

        ! Calculate the inverse determinant of the matrix
        detinv = 1/(A(1,1)*A(2,2)*A(3,3) - A(1,1)*A(2,3)*A(3,2)&
                - A(1,2)*A(2,1)*A(3,3) + A(1,2)*A(2,3)*A(3,1)&
                + A(1,3)*A(2,1)*A(3,2) - A(1,3)*A(2,2)*A(3,1))

        ! Calculate the inverse of the matrix
        B(1,1) = +detinv * (A(2,2)*A(3,3) - A(2,3)*A(3,2))
        B(2,1) = -detinv * (A(2,1)*A(3,3) - A(2,3)*A(3,1))
        B(3,1) = +detinv * (A(2,1)*A(3,2) - A(2,2)*A(3,1))
        B(1,2) = -detinv * (A(1,2)*A(3,3) - A(1,3)*A(3,2))
        B(2,2) = +detinv * (A(1,1)*A(3,3) - A(1,3)*A(3,1))
        B(3,2) = -detinv * (A(1,1)*A(3,2) - A(1,2)*A(3,1))
        B(1,3) = +detinv * (A(1,2)*A(2,3) - A(1,3)*A(2,2))
        B(2,3) = -detinv * (A(1,1)*A(2,3) - A(1,3)*A(2,1))
        B(3,3) = +detinv * (A(1,1)*A(2,2) - A(1,2)*A(2,1))
    END FUNCTION matinv3
!-------------------------------------------------------------------------------------------------------------------------------
    FUNCTION identity(n) RESULT(A)
    ! Produces an identity matrix of size n x n

        INTEGER, INTENT(IN)         :: n
        REAL(4), DIMENSION(n, n)    :: A
        INTEGER                     :: i
        INTEGER                     :: j

        ! Build identity matrix 
        DO i=1,n  
            DO j = 1,n
                IF (i == j) THEN 
                    A(i,j) = 1.0
                ELSE
                    A(i,j) = 0.0
                ENDIF
            ENDDO
        ENDDO
    
    END FUNCTION identity
!-------------------------------------------------------------------------------------------------------------------------------  
    REAL FUNCTION DFController(error, Kd, Tf, DT, inst)
    ! DF controller, with output saturation
    
        IMPLICIT NONE
        ! Inputs
        REAL(4), INTENT(IN)     :: error
        REAL(4), INTENT(IN)     :: kd
        REAL(4), INTENT(IN)     :: tf
        REAL(4), INTENT(IN)     :: DT
        INTEGER(4), INTENT(IN)  :: inst
        ! Local
        REAL(4)                         :: B                                    ! 
        INTEGER(4)                      :: i                                    ! Counter for making arrays
        REAL(4), DIMENSION(99), SAVE    :: errorLast = (/ (0, i=1,99) /)        ! 
        REAL(4), DIMENSION(99), SAVE    :: DFControllerLast = (/ (0, i=1,99) /) ! 
        INTEGER(4), DIMENSION(99), SAVE :: FirstCall = (/ (1, i=1,99) /)        ! First call of this function?
        
        ! Initialize persistent variables/arrays, and set inital condition for integrator term
        ! IF (FirstCall(inst) == 1) THEN
            ! FirstCall(inst) = 0
        ! END IF
        
        B = 2.0/DT
        DFController = (Kd*B)/(B*Tf+1.0)*error - (Kd*B)/(B*Tf+1.0)*errorLast(inst) - (1.0-B*Tf)/(B*Tf+1.0)*DFControllerLast(inst)

        errorLast(inst) = error
        DFControllerLast(inst) = DFController
    END FUNCTION DFController
!-------------------------------------------------------------------------------------------------------------------------------
    SUBROUTINE ColemanTransform(rootMOOP, aziAngle, nHarmonic, axTOut, axYOut)
    ! The Coleman or d-q axis transformation transforms the root out of plane bending moments of each turbine blade
    ! to a direct axis and a quadrature axis

        IMPLICIT NONE
        ! Inputs
        REAL(4), INTENT(IN)     :: rootMOOP(3)                      ! Root out of plane bending moments of each blade
        REAL(4), INTENT(IN)     :: aziAngle                         ! Rotor azimuth angle
        INTEGER(4), INTENT(IN)  :: nHarmonic                        ! The harmonic number, nP
        ! Outputs
        REAL(4), INTENT(OUT)    :: axTOut, axYOut               ! Direct axis and quadrature axis outputted by this transform
        ! Local
        REAL(4), PARAMETER      :: phi2 = 2.0/3.0*PI                ! Phase difference from first to second blade
        REAL(4), PARAMETER      :: phi3 = 4.0/3.0*PI                ! Phase difference from first to third blade

        ! Body
        axTOut  = 2.0/3.0 * (cos(nHarmonic*(aziAngle))*rootMOOP(1) + cos(nHarmonic*(aziAngle+phi2))*rootMOOP(2) + cos(nHarmonic*(aziAngle+phi3))*rootMOOP(3))
        axYOut  = 2.0/3.0 * (sin(nHarmonic*(aziAngle))*rootMOOP(1) + sin(nHarmonic*(aziAngle+phi2))*rootMOOP(2) + sin(nHarmonic*(aziAngle+phi3))*rootMOOP(3))
        
    END SUBROUTINE ColemanTransform
!-------------------------------------------------------------------------------------------------------------------------------
    SUBROUTINE ColemanTransformInverse(axTIn, axYIn, aziAngle, nHarmonic, aziOffset, PitComIPC)
    ! The inverse Coleman or d-q axis transformation transforms the direct axis and quadrature axis
    ! back to root out of plane bending moments of each turbine blade
        IMPLICIT NONE
        ! Inputs
        REAL(4), INTENT(IN)     :: axTIn, axYIn         ! Direct axis and quadrature axis
        REAL(4), INTENT(IN)     :: aziAngle                     ! Rotor azimuth angle
        REAL(4), INTENT(IN)     :: aziOffset                    ! Phase shift added to the azimuth angle
        INTEGER(4), INTENT(IN)  :: nHarmonic                    ! The harmonic number, nP
        ! Outputs
        REAL(4), INTENT(OUT)    :: PitComIPC(3)                 ! Root out of plane bending moments of each blade
        ! Local
        REAL(4), PARAMETER      :: phi2 = 2.0/3.0*PI                ! Phase difference from first to second blade
        REAL(4), PARAMETER      :: phi3 = 4.0/3.0*PI                ! Phase difference from first to third blade

        ! Body
        PitComIPC(1) = cos(nHarmonic*(aziAngle+aziOffset))*axTIn + sin(nHarmonic*(aziAngle+aziOffset))*axYIn
        PitComIPC(2) = cos(nHarmonic*(aziAngle+aziOffset+phi2))*axTIn + sin(nHarmonic*(aziAngle+aziOffset+phi2))*axYIn
        PitComIPC(3) = cos(nHarmonic*(aziAngle+aziOffset+phi3))*axTIn + sin(nHarmonic*(aziAngle+aziOffset+phi3))*axYIn

    END SUBROUTINE ColemanTransformInverse
!-------------------------------------------------------------------------------------------------------------------------------
    REAL FUNCTION CPfunction(CP, lambda)
    ! Paremeterized Cp(lambda) function for a fixed pitch angle. Circumvents the need of importing a look-up table
        IMPLICIT NONE
        
        ! Inputs
        REAL(4), INTENT(IN) :: CP(4)    ! Parameters defining the parameterizable Cp(lambda) function
        REAL(4), INTENT(IN) :: lambda    ! Estimated or measured tip-speed ratio input
        
        ! Lookup
        CPfunction = exp(-CP(1)/lambda)*(CP(2)/lambda-CP(3))+CP(4)*lambda
        CPfunction = saturate(CPfunction, 0.001, 1.0)
        
    END FUNCTION CPfunction
!-------------------------------------------------------------------------------------------------------------------------------
    REAL FUNCTION AeroDynTorque(LocalVar, CntrPar, PerfData)
    ! Function for computing the aerodynamic torque, divided by the effective rotor torque of the turbine, for use in wind speed estimation
        
        USE ROSCO_Types, ONLY : LocalVariables, ControlParameters, PerformanceData
        IMPLICIT NONE
    
        ! Inputs
        TYPE(ControlParameters), INTENT(IN) :: CntrPar
        TYPE(LocalVariables), INTENT(IN) :: LocalVar
        TYPE(PerformanceData), INTENT(IN) :: PerfData
            
        ! Local
        REAL(4) :: RotorArea
        REAL(4) :: Cp
        REAL(4) :: Lambda
        
        ! Find Torque
        RotorArea = PI*CntrPar%WE_BladeRadius**2
        Lambda = LocalVar%RotSpeedF*CntrPar%WE_BladeRadius/LocalVar%WE_Vw
        ! Cp = CPfunction(CntrPar%WE_CP, Lambda)
        Cp = interp2d(PerfData%Beta_vec,PerfData%TSR_vec,PerfData%Cp_mat, LocalVar%PC_PitComTF*R2D, Lambda)
        AeroDynTorque = 0.5*(CntrPar%WE_RhoAir*RotorArea)*(LocalVar%WE_Vw**3/LocalVar%RotSpeedF)*Cp
        AeroDynTorque = MAX(AeroDynTorque, 0.0)
        
    END FUNCTION AeroDynTorque
!-------------------------------------------------------------------------------------------------------------------------------
    SUBROUTINE Debug(LocalVar, CntrPar, DebugVar, avrSWAP, RootName, size_avcOUTNAME)
    ! Debug routine, defines what gets printed to DEBUG.dbg if LoggingLevel = 1
    
        USE, INTRINSIC  :: ISO_C_Binding
        USE ROSCO_Types, ONLY : LocalVariables, ControlParameters, DebugVariables
        
        IMPLICIT NONE
    
        TYPE(ControlParameters), INTENT(IN)     :: CntrPar
        TYPE(LocalVariables), INTENT(IN)        :: LocalVar
        TYPE(DebugVariables), INTENT(IN)        :: DebugVar
    
        INTEGER(4), INTENT(IN)                      :: size_avcOUTNAME
        INTEGER(4)                                  :: I , nDebugOuts               ! Generic index.
        CHARACTER(1), PARAMETER                     :: Tab = CHAR(9)                        ! The tab character.
        CHARACTER(29), PARAMETER                    :: FmtDat = "(F10.3,TR5,99(ES10.3E2,TR5:))"   ! The format of the debugging data
        INTEGER(4), PARAMETER                       :: UnDb = 85        ! I/O unit for the debugging information
        INTEGER(4), PARAMETER                       :: UnDb2 = 86       ! I/O unit for the debugging information, avrSWAP
        REAL(C_FLOAT), INTENT(INOUT)                :: avrSWAP(*)   ! The swap array, used to pass data to, and receive data from, the DLL controller.
        CHARACTER(size_avcOUTNAME-1), INTENT(IN)    :: RootName     ! a Fortran version of the input C string (not considered an array here)    [subtract 1 for the C null-character]
        
        CHARACTER(10)                               :: DebugOutStr1,  DebugOutStr2, DebugOutStr3, DebugOutStr4, DebugOutStr5, &
                                                         DebugOutStr6, DebugOutStr7, DebugOutStr8, DebugOutStr9, DebugOutStr10, &
                                                         DebugOutStr11, DebugOutStr12, DebugOutStr13, DebugOutStr14, DebugOutStr15                                                         
        CHARACTER(10)                               :: DebugOutUni1,  DebugOutUni2, DebugOutUni3, DebugOutUni4, DebugOutUni5, &
                                                         DebugOutUni6, DebugOutUni7, DebugOutUni8, DebugOutUni9, DebugOutUni10, &
                                                         DebugOutUni11, DebugOutUni12, DebugOutUni13, DebugOutUni14, DebugOutUni15 
        CHARACTER(10), ALLOCATABLE                  :: DebugOutStrings(:), DebugOutUnits(:)
        REAL(4), ALLOCATABLE                        :: DebugOutData(:)

        ! Set up Debug Strings and Data
        ! Note that Debug strings have 10 character limit
        nDebugOuts = 12
        ALLOCATE(DebugOutData(nDebugOuts))
        !                 Header                            Unit                                Variable
        DebugOutStr1  = 'FA_AccF';         DebugOutUni1  = '(m/s)';      DebugOutData(1)  = LocalVar%NacIMU_FA_AccF
        DebugOutStr2  = 'WE_Vw';           DebugOutUni2  = '(rad)';      DebugOutData(2)  = LocalVar%WE_Vw
        DebugOutStr3  = 'FA_AccR';          DebugOutUni3  = '(rad/s^2)';  DebugOutData(3)  = LocalVar%NacIMU_FA_Acc
        DebugOutStr4  = 'FA_Acc';          DebugOutUni4  = '(m/s^2)';    DebugOutData(4)  = LocalVar%FA_Acc
        DebugOutStr5  = 'Fl_Pitcom';       DebugOutUni5  = '(rad)';      DebugOutData(5)  = LocalVar%Fl_Pitcom
        DebugOutStr6  = 'WE_Cp';           DebugOutUni6  = '(-)';        DebugOutData(6)  = DebugVar%WE_Cp
        DebugOutStr7  = 'PC_MinPit';       DebugOutUni7  = '(rad)';      DebugOutData(7)  = LocalVar%PC_MinPit
        DebugOutStr8  = 'SS_dOmF';         DebugOutUni8  = '(rad/s)';    DebugOutData(8)  = LocalVar%SS_DelOmegaF
        DebugOutStr9  = 'WE_b';            DebugOutUni9  = '(deg)';      DebugOutData(9)  = DebugVar%WE_b
        DebugOutStr10  = 'WE_t';            DebugOutUni10  = '(Nm)';      DebugOutData(10)  = DebugVar%WE_t
        DebugOutStr11  = 'WE_w';            DebugOutUni11  = '(rad/s)';      DebugOutData(11)  = DebugVar%WE_w
        DebugOutStr12  = 'WE_D';            DebugOutUni12  = '()';      DebugOutData(12)  = DebugVar%WE_D

        Allocate(DebugOutStrings(nDebugOuts))
        Allocate(DebugOutUnits(nDebugOuts))
        DebugOutStrings =   [CHARACTER(10)  :: DebugOutStr1, DebugOutStr2, DebugOutStr3, DebugOutStr4, &
                                                DebugOutStr5, DebugOutStr6, DebugOutStr7, DebugOutStr8, &
                                                DebugOutStr9, DebugOutStr10, DebugOutStr11, DebugOutStr12]
        DebugOutUnits =     [CHARACTER(10)  :: DebugOutUni1, DebugOutUni2, DebugOutUni3, DebugOutUni4, &
                                                DebugOutUni5, DebugOutUni6, DebugOutUni7, DebugOutUni8, &
                                                DebugOutUni9, DebugOutUni10, DebugOutUni11, DebugOutUni12]
        
        ! Initialize debug file
        IF (LocalVar%iStatus == 0)  THEN  ! .TRUE. if we're on the first call to the DLL
        ! If we're debugging, open the debug file and write the header:
            ! Note that the headers will be Truncated to 10 characters!!
            IF (CntrPar%LoggingLevel > 0) THEN
                OPEN(unit=UnDb, FILE=RootName(1:size_avcOUTNAME-5)//'RO.dbg')
                WRITE (UnDb,'(99(a10,TR5:))') 'Time',   DebugOutStrings
                WRITE (UnDb,'(99(a10,TR5:))') '(sec)',  DebugOutUnits
            END IF
            
            IF (CntrPar%LoggingLevel > 1) THEN
                OPEN(unit=UnDb2, FILE='DEBUG2.dbg')
                WRITE(UnDb2,'(/////)')
                WRITE(UnDb2,'(A,85("'//Tab//'AvrSWAP(",I2,")"))')  'LocalVar%Time ', (i,i=1,85)
                WRITE(UnDb2,'(A,85("'//Tab//'(-)"))')  '(s)'
            END IF
        ELSE
            ! Print simulation status, every 10 seconds
            IF (MODULO(LocalVar%Time, 10.0) == 0) THEN
                WRITE(*, 100) LocalVar%GenSpeedF*RPS2RPM, LocalVar%BlPitch(1)*R2D, avrSWAP(15)/1000.0, LocalVar%WE_Vw ! LocalVar%Time !/1000.0
                100 FORMAT('Generator speed: ', f6.1, ' RPM, Pitch angle: ', f5.1, ' deg, Power: ', f7.1, ' kW, Est. wind Speed: ', f5.1, ' m/s')
            END IF
            
        ENDIF

        ! Write debug files
        IF (CntrPar%LoggingLevel > 0) THEN
            WRITE (UnDb,FmtDat)  LocalVar%Time, DebugOutData
        END IF

        IF (CntrPar%LoggingLevel > 1) THEN
            WRITE (UnDb2,FmtDat)    LocalVar%Time, avrSWAP(1:85)
        END IF

    END SUBROUTINE Debug
END MODULE Functions
