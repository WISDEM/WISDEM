! Copyright 2019 NREL

! Licensed under the Apache License, Version 2.0 (the "License"); you may not use
! this file except in compliance with the License. You may obtain a copy of the
! License at http://www.apache.org/licenses/LICENSE-2.0

! Unless required by applicable law or agreed to in writing, software distributed
! under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
! CONDITIONS OF ANY KIND, either express or implied. See the License for the
! specific language governing permissions and limitations under the License.
! -------------------------------------------------------------------------------------------

! This module contains the primary controller routines

! Subroutines:
!           PitchControl: Blade pitch control high level subroutine
!           VariableSpeedControl: Variable speed generator torque control
!           YawRateControl: Nacelle yaw control
!           IPC: Individual pitch control
!           ForeAftDamping: Tower fore-aft damping control
!           FloatingFeedback: Tower fore-aft feedback for floating offshore wind turbines

MODULE Controllers

    USE, INTRINSIC :: ISO_C_Binding
    USE Functions
    USE Filters
    USE ControllerBlocks

    IMPLICIT NONE

CONTAINS
!-------------------------------------------------------------------------------------------------------------------------------
    SUBROUTINE PitchControl(avrSWAP, CntrPar, LocalVar, objInst)
    ! Blade pitch controller, generally maximizes rotor speed below rated (region 2) and regulates rotor speed above rated (region 3)
    !       PC_State = 0, fix blade pitch to fine pitch angle (PC_FinePit)
    !       PC_State = 1, is gain scheduled PI controller 
    ! Additional loops/methods (enabled via switches in DISCON.IN):
    !       Individual pitch control
    !       Tower fore-aft damping 
    !       Sine excitation on pitch    
        USE ROSCO_Types, ONLY : ControlParameters, LocalVariables, ObjectInstances
        
        ! Inputs
        TYPE(ControlParameters), INTENT(INOUT)  :: CntrPar
        TYPE(LocalVariables), INTENT(INOUT)     :: LocalVar
        TYPE(ObjectInstances), INTENT(INOUT)    :: objInst
        ! Allocate Variables:
        REAL(C_FLOAT), INTENT(INOUT)    :: avrSWAP(*)   ! The swap array, used to pass data to, and receive data from the DLL controller.
        INTEGER(4)                      :: K            ! Index used for looping through blades.
        REAL(4), Save                :: PitComT_Last 

        ! ------- Blade Pitch Controller --------
        ! Load PC State
        IF (LocalVar%PC_State == 1) THEN ! PI BldPitch control
            LocalVar%PC_MaxPit = CntrPar%PC_MaxPit
        ELSE ! debug mode, fix at fine pitch
            LocalVar%PC_MaxPit = CntrPar%PC_FinePit
        END IF
        
        ! Compute (interpolate) the gains based on previously commanded blade pitch angles and lookup table:
        LocalVar%PC_KP = interp1d(CntrPar%PC_GS_angles, CntrPar%PC_GS_KP, LocalVar%PC_PitComT) ! Proportional gain
        LocalVar%PC_KI = interp1d(CntrPar%PC_GS_angles, CntrPar%PC_GS_KI, LocalVar%PC_PitComT) ! Integral gain
        LocalVar%PC_KD = interp1d(CntrPar%PC_GS_angles, CntrPar%PC_GS_KD, LocalVar%PC_PitComT) ! Derivative gain
        LocalVar%PC_TF = interp1d(CntrPar%PC_GS_angles, CntrPar%PC_GS_TF, LocalVar%PC_PitComT) ! TF gains (derivative filter) !NJA - need to clarify
        
        ! Compute the collective pitch command associated with the proportional and integral gains:
        IF (LocalVar%iStatus == 0) THEN
            LocalVar%PC_PitComT = PIController(LocalVar%PC_SpdErr, LocalVar%PC_KP, LocalVar%PC_KI, CntrPar%PC_FinePit, LocalVar%PC_MaxPit, LocalVar%DT, LocalVar%PitCom(1), .TRUE., objInst%instPI)
        ELSE
            LocalVar%PC_PitComT = PIController(LocalVar%PC_SpdErr, LocalVar%PC_KP, LocalVar%PC_KI, LocalVar%PC_MinPit, LocalVar%PC_MaxPit, LocalVar%DT, LocalVar%BlPitch(1), .FALSE., objInst%instPI)
        END IF
        
        ! Find individual pitch control contribution
        IF ((CntrPar%IPC_ControlMode >= 1) .OR. (CntrPar%Y_ControlMode == 2)) THEN
            CALL IPC(CntrPar, LocalVar, objInst)
        ELSE
            LocalVar%IPC_PitComF = 0.0 ! THIS IS AN ARRAY!!
        END IF
        
        ! Include tower fore-aft tower vibration damping control
        IF ((CntrPar%FA_KI > 0.0) .OR. (CntrPar%Y_ControlMode == 2)) THEN
            CALL ForeAftDamping(CntrPar, LocalVar, objInst)
        ELSE
            LocalVar%FA_PitCom = 0.0 ! THIS IS AN ARRAY!!
        ENDIF
        
        ! Pitch Saturation
        IF (CntrPar%PS_Mode == 1) THEN
            LocalVar%PC_MinPit = PitchSaturation(LocalVar,CntrPar,objInst)
            LocalVar%PC_MinPit = max(LocalVar%PC_MinPit, CntrPar%PC_FinePit)
        ELSE
            LocalVar%PC_MinPit = CntrPar%PC_FinePit
        ENDIF

        ! Shutdown
        IF (CntrPar%SD_Mode == 1) THEN
            LocalVar%PC_PitComT = Shutdown(LocalVar, CntrPar, objInst)
        ENDIF

        ! FloatingFeedback
        IF (CntrPar%Fl_Mode == 1) THEN
            CALL FloatingFeedback(LocalVar, CntrPar, objInst)
            LocalVar%PC_PitComT = LocalVar%PC_PitComT + LocalVar%Fl_PitCom
        ENDIF

        ! Saturate collective pitch commands:
        LocalVar%PC_PitComT = saturate(LocalVar%PC_PitComT, LocalVar%PC_MinPit, CntrPar%PC_MaxPit)                    ! Saturate the overall command using the pitch angle limits
        LocalVar%PC_PitComT = ratelimit(LocalVar%PC_PitComT, PitComT_Last, CntrPar%PC_MinRat, CntrPar%PC_MaxRat, LocalVar%DT) ! Saturate the overall command of blade K using the pitch rate limit
        PitComT_Last = LocalVar%PC_PitComT

        ! Combine and saturate all individual pitch commands:
        ! Filter to emulate pitch actuator
        DO K = 1,LocalVar%NumBl ! Loop through all blades, add IPC contribution and limit pitch rate
            LocalVar%PitCom(K) = LocalVar%PC_PitComT + LocalVar%IPC_PitComF(K) + LocalVar%FA_PitCom(K) 
            ! LocalVar%PitCom(K) = SecLPFilter(LocalVar%PitCom(K), LocalVar%DT, , Damp, iStatus, reset, inst)
            LocalVar%PitCom(K) = saturate(LocalVar%PitCom(K), LocalVar%PC_MinPit, CntrPar%PC_MaxPit)                    ! Saturate the overall command using the pitch angle limits
            LocalVar%PitCom(K) = ratelimit(LocalVar%PitCom(K), LocalVar%BlPitch(K), CntrPar%PC_MinRat, CntrPar%PC_MaxRat, LocalVar%DT) ! Saturate the overall command of blade K using the pitch rate limit
        END DO

        ! Command the pitch demanded from the last
        ! call to the controller (See Appendix A of Bladed User's Guide):
        avrSWAP(42) = LocalVar%PitCom(1)    ! Use the command angles of all blades if using individual pitch
        avrSWAP(43) = LocalVar%PitCom(2)    ! "
        avrSWAP(44) = LocalVar%PitCom(3)    ! "
        avrSWAP(45) = LocalVar%PitCom(1)    ! Use the command angle of blade 1 if using collective pitch
    END SUBROUTINE PitchControl
!-------------------------------------------------------------------------------------------------------------------------------  
    SUBROUTINE VariableSpeedControl(avrSWAP, CntrPar, LocalVar, objInst)
    ! Generator torque controller
    !       VS_State = 0, Error state, for debugging purposes, GenTq = VS_RtTq
    !       VS_State = 1, Region 1(.5) operation, torque control to keep the rotor at cut-in speed towards the Cp-max operational curve
    !       VS_State = 2, Region 2 operation, maximum rotor power efficiency (Cp-max) tracking using K*omega^2 law, fixed fine-pitch angle in BldPitch controller
    !       VS_State = 3, Region 2.5, transition between below and above-rated operating conditions (near-rated region) using PI torque control
    !       VS_State = 4, above-rated operation using pitch control (constant torque mode)
    !       VS_State = 5, above-rated operation using pitch and torque control (constant power mode)
    !       VS_State = 6, Tip-Speed-Ratio tracking PI controller
        USE ROSCO_Types, ONLY : ControlParameters, LocalVariables, ObjectInstances
        ! Inputs
        TYPE(ControlParameters), INTENT(INOUT)  :: CntrPar
        TYPE(LocalVariables), INTENT(INOUT)     :: LocalVar
        TYPE(ObjectInstances), INTENT(INOUT)    :: objInst
        ! Allocate Variables
        REAL(C_FLOAT), INTENT(INOUT)            :: avrSWAP(*)    ! The swap array, used to pass data to, and receive data from, the DLL controller.
        REAL(4)                                 :: VS_MaxTq      ! Locally allocated maximum torque saturation limits
        
        ! -------- Variable-Speed Torque Controller --------
        ! Define max torque
        IF (LocalVar%VS_State == 4) THEN
            VS_MaxTq = CntrPar%VS_RtTq
        ELSE
            ! VS_MaxTq = CntrPar%VS_MaxTq           ! NJA: May want to boost max torque
            VS_MaxTq = CntrPar%VS_RtTq
        ENDIF

        ! Optimal Tip-Speed-Ratio tracking controller
        IF (CntrPar%VS_ControlMode == 2) THEN
            ! PI controller
            LocalVar%GenTq = PIController(LocalVar%VS_SpdErr, CntrPar%VS_KP(1), CntrPar%VS_KI(1), CntrPar%VS_MinTq, VS_MaxTq, LocalVar%DT, LocalVar%VS_LastGenTrq, .FALSE., objInst%instPI)
            LocalVar%GenTq = saturate(LocalVar%GenTq, CntrPar%VS_MinTq, VS_MaxTq)
        
        ! K*Omega^2 control law with PI torque control in transition regions
        ELSE
            ! Update PI loops for region 1.5 and 2.5 PI control
            LocalVar%GenArTq = PIController(LocalVar%VS_SpdErrAr, CntrPar%VS_KP(1), CntrPar%VS_KI(1), CntrPar%VS_MaxOMTq, CntrPar%VS_ArSatTq, LocalVar%DT, CntrPar%VS_MaxOMTq, .FALSE., objInst%instPI)
            LocalVar%GenBrTq = PIController(LocalVar%VS_SpdErrBr, CntrPar%VS_KP(1), CntrPar%VS_KI(1), CntrPar%VS_MinTq, CntrPar%VS_MinOMTq, LocalVar%DT, CntrPar%VS_MinOMTq, .FALSE., objInst%instPI)
            
            ! The action
            IF (LocalVar%VS_State == 1) THEN ! Region 1.5
                LocalVar%GenTq = LocalVar%GenBrTq
            ELSEIF (LocalVar%VS_State == 2) THEN ! Region 2
                LocalVar%GenTq = CntrPar%VS_Rgn2K*LocalVar%GenSpeedF*LocalVar%GenSpeedF
            ELSEIF (LocalVar%VS_State == 3) THEN ! Region 2.5
                LocalVar%GenTq = LocalVar%GenArTq
            ELSEIF (LocalVar%VS_State == 4) THEN ! Region 3, constant torque
                LocalVar%GenTq = CntrPar%VS_RtTq
            ELSEIF (LocalVar%VS_State == 5) THEN ! Region 3, constant power
                LocalVar%GenTq = (CntrPar%VS_RtPwr/(CntrPar%VS_GenEff/100.0))/LocalVar%GenSpeedF
            END IF
            
            ! Saturate
            LocalVar%GenTq = saturate(LocalVar%GenTq, CntrPar%VS_MinTq, VS_MaxTq)
        ENDIF


        ! Saturate the commanded torque using the maximum torque limit:
        LocalVar%GenTq = MIN(LocalVar%GenTq, CntrPar%VS_MaxTq)                    ! Saturate the command using the maximum torque limit
        
        ! Saturate the commanded torque using the torque rate limit:
        LocalVar%GenTq = ratelimit(LocalVar%GenTq, LocalVar%VS_LastGenTrq, -CntrPar%VS_MaxRat, CntrPar%VS_MaxRat, LocalVar%DT)    ! Saturate the command using the torque rate limit
        
        ! Reset the value of LocalVar%VS_LastGenTrq to the current values:
        LocalVar%VS_LastGenTrq = LocalVar%GenTq
        
        ! Set the command generator torque (See Appendix A of Bladed User's Guide):
        avrSWAP(47) = LocalVar%VS_LastGenTrq   ! Demanded generator torque
    END SUBROUTINE VariableSpeedControl
!-------------------------------------------------------------------------------------------------------------------------------
    SUBROUTINE YawRateControl(avrSWAP, CntrPar, LocalVar, objInst)
        ! Yaw rate controller
        !       Y_ControlMode = 0, No yaw control
        !       Y_ControlMode = 1, Simple yaw rate control using yaw drive
        !       Y_ControlMode = 2, Yaw by IPC (accounted for in IPC subroutine)
        USE ROSCO_Types, ONLY : ControlParameters, LocalVariables, ObjectInstances
    
        REAL(C_FLOAT), INTENT(INOUT) :: avrSWAP(*) ! The swap array, used to pass data to, and receive data from, the DLL controller.
    
        TYPE(ControlParameters), INTENT(INOUT)    :: CntrPar
        TYPE(LocalVariables), INTENT(INOUT)       :: LocalVar
        TYPE(ObjectInstances), INTENT(INOUT)      :: objInst
        
        !..............................................................................................................................
        ! Yaw control
        !..............................................................................................................................
        
        IF (CntrPar%Y_ControlMode == 1) THEN
            avrSWAP(29) = 0                                      ! Yaw control parameter: 0 = yaw rate control
            IF (LocalVar%Time >= LocalVar%Y_YawEndT) THEN        ! Check if the turbine is currently yawing
                avrSWAP(48) = 0.0                                ! Set yaw rate to zero
            
                LocalVar%Y_ErrLPFFast = LPFilter(LocalVar%Y_MErr, LocalVar%DT, CntrPar%Y_omegaLPFast, LocalVar%iStatus, .FALSE., objInst%instLPF)        ! Fast low pass filtered yaw error with a frequency of 1
                LocalVar%Y_ErrLPFSlow = LPFilter(LocalVar%Y_MErr, LocalVar%DT, CntrPar%Y_omegaLPSlow, LocalVar%iStatus, .FALSE., objInst%instLPF)        ! Slow low pass filtered yaw error with a frequency of 1/60
            
                LocalVar%Y_AccErr = LocalVar%Y_AccErr + LocalVar%DT*SIGN(LocalVar%Y_ErrLPFFast**2, LocalVar%Y_ErrLPFFast)    ! Integral of the fast low pass filtered yaw error
            
                IF (ABS(LocalVar%Y_AccErr) >= CntrPar%Y_ErrThresh) THEN                                   ! Check if accumulated error surpasses the threshold
                    LocalVar%Y_YawEndT = ABS(LocalVar%Y_ErrLPFSlow/CntrPar%Y_Rate) + LocalVar%Time        ! Yaw to compensate for the slow low pass filtered error
                END IF
            ELSE
                avrSWAP(48) = SIGN(CntrPar%Y_Rate, LocalVar%Y_MErr)        ! Set yaw rate to predefined yaw rate, the sign of the error is copied to the rate
                LocalVar%Y_ErrLPFFast = LPFilter(LocalVar%Y_MErr, LocalVar%DT, CntrPar%Y_omegaLPFast, LocalVar%iStatus, .TRUE., objInst%instLPF)        ! Fast low pass filtered yaw error with a frequency of 1
                LocalVar%Y_ErrLPFSlow = LPFilter(LocalVar%Y_MErr, LocalVar%DT, CntrPar%Y_omegaLPSlow, LocalVar%iStatus, .TRUE., objInst%instLPF)        ! Slow low pass filtered yaw error with a frequency of 1/60
                LocalVar%Y_AccErr = 0.0    ! "
            END IF
        END IF
    END SUBROUTINE YawRateControl
!-------------------------------------------------------------------------------------------------------------------------------
    SUBROUTINE IPC(CntrPar, LocalVar, objInst)
        ! Individual pitch control subroutine
        !   - Calculates the commanded pitch angles for IPC employed for blade fatigue load reductions at 1P and 2P
        !   - Includes yaw by IPC

        USE ROSCO_Types, ONLY : ControlParameters, LocalVariables, ObjectInstances
        
        ! Local variables
        REAL(4)                  :: PitComIPC(3), PitComIPCF(3), PitComIPC_1P(3), PitComIPC_2P(3)
        INTEGER(4)               :: K                                       ! Integer used to loop through turbine blades
        REAL(4)                  :: axisTilt_1P, axisYaw_1P, axisYawF_1P    ! Direct axis and quadrature axis outputted by Coleman transform, 1P
        REAL(4), SAVE            :: IntAxisTilt_1P, IntAxisYaw_1P           ! Integral of the direct axis and quadrature axis, 1P
        REAL(4)                  :: axisTilt_2P, axisYaw_2P, axisYawF_2P    ! Direct axis and quadrature axis outputted by Coleman transform, 1P
        REAL(4), SAVE            :: IntAxisTilt_2P, IntAxisYaw_2P           ! Integral of the direct axis and quadrature axis, 1P
        REAL(4)                  :: IntAxisYawIPC_1P                        ! IPC contribution with yaw-by-IPC component
        REAL(4)                  :: Y_MErrF, Y_MErrF_IPC                    ! Unfiltered and filtered yaw alignment error [rad]
        
        TYPE(ControlParameters), INTENT(INOUT)  :: CntrPar
        TYPE(LocalVariables), INTENT(INOUT)     :: LocalVar
        TYPE(ObjectInstances), INTENT(INOUT)    :: objInst
        
        ! Body
        ! Initialization
        ! Set integrals to be 0 in the first time step
        IF (LocalVar%iStatus==0) THEN
            IntAxisTilt_1P = 0.0
            IntAxisYaw_1P = 0.0
            IntAxisTilt_2P = 0.0
            IntAxisYaw_2P = 0.0
        END IF
        
        ! Pass rootMOOPs through the Coleman transform to get the tilt and yaw moment axis
        CALL ColemanTransform(LocalVar%rootMOOP, LocalVar%Azimuth, NP_1, axisTilt_1P, axisYaw_1P)
        CALL ColemanTransform(LocalVar%rootMOOP, LocalVar%Azimuth, NP_2, axisTilt_2P, axisYaw_2P)
        
        ! High-pass filter the MBC yaw component and filter yaw alignment error, and compute the yaw-by-IPC contribution
        IF (CntrPar%Y_ControlMode == 2) THEN
            Y_MErrF = SecLPFilter(LocalVar%Y_MErr, LocalVar%DT, CntrPar%Y_IPC_omegaLP, CntrPar%Y_IPC_zetaLP, LocalVar%iStatus, .FALSE., objInst%instSecLPF)
            Y_MErrF_IPC = PIController(Y_MErrF, CntrPar%Y_IPC_KP(1), CntrPar%Y_IPC_KI(1), -CntrPar%Y_IPC_IntSat, CntrPar%Y_IPC_IntSat, LocalVar%DT, 0.0, .FALSE., objInst%instPI)
        ELSE
            axisYawF_1P = axisYaw_1P
            Y_MErrF = 0.0
            Y_MErrF_IPC = 0.0
        END IF
        
        ! Integrate the signal and multiply with the IPC gain
        IF ((CntrPar%IPC_ControlMode >= 1) .AND. (CntrPar%Y_ControlMode /= 2)) THEN
            IntAxisTilt_1P = IntAxisTilt_1P + LocalVar%DT * CntrPar%IPC_KI(1) * axisTilt_1P
            IntAxisYaw_1P = IntAxisYaw_1P + LocalVar%DT * CntrPar%IPC_KI(1) * axisYawF_1P
            IntAxisTilt_1P = saturate(IntAxisTilt_1P, -CntrPar%IPC_IntSat, CntrPar%IPC_IntSat)
            IntAxisYaw_1P = saturate(IntAxisYaw_1P, -CntrPar%IPC_IntSat, CntrPar%IPC_IntSat)
            
            IF (CntrPar%IPC_ControlMode >= 2) THEN
                IntAxisTilt_2P = IntAxisTilt_2P + LocalVar%DT * CntrPar%IPC_KI(2) * axisTilt_2P
                IntAxisYaw_2P = IntAxisYaw_2P + LocalVar%DT * CntrPar%IPC_KI(2) * axisYawF_2P
                IntAxisTilt_2P = saturate(IntAxisTilt_2P, -CntrPar%IPC_IntSat, CntrPar%IPC_IntSat)
                IntAxisYaw_2P = saturate(IntAxisYaw_2P, -CntrPar%IPC_IntSat, CntrPar%IPC_IntSat)
            END IF
        ELSE
            IntAxisTilt_1P = 0.0
            IntAxisYaw_1P = 0.0
            IntAxisTilt_2P = 0.0
            IntAxisYaw_2P = 0.0
        END IF
        
        ! Add the yaw-by-IPC contribution
        IntAxisYawIPC_1P = IntAxisYaw_1P + Y_MErrF_IPC
        
        ! Pass direct and quadrature axis through the inverse Coleman transform to get the commanded pitch angles
        CALL ColemanTransformInverse(IntAxisTilt_1P, IntAxisYawIPC_1P, LocalVar%Azimuth, NP_1, CntrPar%IPC_aziOffset(1), PitComIPC_1P)
        CALL ColemanTransformInverse(IntAxisTilt_2P, IntAxisYaw_2P, LocalVar%Azimuth, NP_2, CntrPar%IPC_aziOffset(2), PitComIPC_2P)
        
        ! Sum nP IPC contributions and store to LocalVar data type
        DO K = 1,LocalVar%NumBl
            PitComIPC(K) = PitComIPC_1P(K) + PitComIPC_2P(K)
            
            ! Optionally filter the resulting signal to induce a phase delay
            IF (CntrPar%IPC_CornerFreqAct > 0.0) THEN
                PitComIPCF(K) = LPFilter(PitComIPC(K), LocalVar%DT, CntrPar%IPC_CornerFreqAct, LocalVar%iStatus, .FALSE., objInst%instLPF)
            ELSE
                PitComIPCF(K) = PitComIPC(K)
            END IF
            
            LocalVar%IPC_PitComF(K) = PitComIPCF(K)
        END DO
    END SUBROUTINE IPC
!-------------------------------------------------------------------------------------------------------------------------------
    SUBROUTINE ForeAftDamping(CntrPar, LocalVar, objInst)
        ! Fore-aft damping controller, reducing the tower fore-aft vibrations using pitch

        USE ROSCO_Types, ONLY : ControlParameters, LocalVariables, ObjectInstances
        
        ! Local variables
        INTEGER(4) :: K    ! Integer used to loop through turbine blades

        TYPE(ControlParameters), INTENT(INOUT)  :: CntrPar
        TYPE(LocalVariables), INTENT(INOUT)     :: LocalVar
        TYPE(ObjectInstances), INTENT(INOUT)    :: objInst
        
        ! Body
        LocalVar%FA_AccHPFI = PIController(LocalVar%FA_AccHPF, 0.0, CntrPar%FA_KI, -CntrPar%FA_IntSat, CntrPar%FA_IntSat, LocalVar%DT, 0.0, .FALSE., objInst%instPI)
        
        ! Store the fore-aft pitch contribution to LocalVar data type
        DO K = 1,LocalVar%NumBl
            LocalVar%FA_PitCom(K) = LocalVar%FA_AccHPFI
        END DO
        
    END SUBROUTINE ForeAftDamping
!-------------------------------------------------------------------------------------------------------------------------------
    SUBROUTINE FloatingFeedback(LocalVar, CntrPar, objInst) 
    ! FloatingFeedback defines a minimum blade pitch angle based on a lookup table provided by DISON.IN
    !       Fl_Mode = 0, No feedback
    !       Fl_Mode = 1, Proportional feedback of nacelle velocity
        USE ROSCO_Types, ONLY : LocalVariables, ControlParameters, ObjectInstances
        IMPLICIT NONE
        ! Inputs
        TYPE(ControlParameters), INTENT(IN)     :: CntrPar
        TYPE(LocalVariables), INTENT(INOUT)     :: LocalVar 
        TYPE(ObjectInstances), INTENT(INOUT)    :: objInst
        ! Allocate Variables 
        REAL(4)                      :: NacIMU_FA_vel ! Tower fore-aft velocity
        
        ! Calculate floating contribution to pitch command
        NacIMU_FA_vel = PIController(LocalVar%NacIMU_FA_AccF, 0.0, 1.0, -100.0 , 100.0 ,LocalVar%DT, 0.0, .FALSE., objInst%instPI) ! NJA: should never reach saturation limits....
        LocalVar%Fl_PitCom = (0.0 - NacIMU_FA_vel) * CntrPar%Fl_Kp !* LocalVar%PC_KP/maxval(CntrPar%PC_GS_KP)

    END SUBROUTINE FloatingFeedback
!-------------------------------------------------------------------------------------------------------------------------------
    SUBROUTINE FlapControl(avrSWAP, CntrPar, LocalVar, objInst)
        ! Yaw rate controller
        !       Y_ControlMode = 0, No yaw control
        !       Y_ControlMode = 1, Simple yaw rate control using yaw drive
        !       Y_ControlMode = 2, Yaw by IPC (accounted for in IPC subroutine)
        USE ROSCO_Types, ONLY : ControlParameters, LocalVariables, ObjectInstances
    
        REAL(C_FLOAT), INTENT(INOUT) :: avrSWAP(*) ! The swap array, used to pass data to, and receive data from, the DLL controller.
    
        TYPE(ControlParameters), INTENT(INOUT)    :: CntrPar
        TYPE(LocalVariables), INTENT(INOUT)       :: LocalVar
        TYPE(ObjectInstances), INTENT(INOUT)      :: objInst
        ! Internal Variables
        Integer(4)                                :: K
        REAL(4)                                   :: rootMOOP_F(3)
        REAL(4)                                   :: RootMyb_Vel(3)
        REAL(4), SAVE                             :: RootMyb_Last(3)
        REAL(4)                                   :: RootMyb_VelErr(3)

        ! Flap control
        IF (CntrPar%Flp_Mode >= 1) THEN
            IF ((LocalVar%iStatus == 0) .AND. (CntrPar%Flp_Mode >= 1)) THEN
                RootMyb_Last(1) = 0 - LocalVar%rootMOOP(1)
                RootMyb_Last(2) = 0 - LocalVar%rootMOOP(2)
                RootMyb_Last(3) = 0 - LocalVar%rootMOOP(3)
                ! Initial Flap angle
                LocalVar%Flp_Angle(1) = CntrPar%Flp_Angle
                LocalVar%Flp_Angle(2) = CntrPar%Flp_Angle
                LocalVar%Flp_Angle(3) = CntrPar%Flp_Angle
                ! Initialize filter
                RootMOOP_F(1) = SecLPFilter(LocalVar%rootMOOP(1),LocalVar%DT, CntrPar%F_FlpCornerFreq, CntrPar%F_FlpDamping, LocalVar%iStatus, .FALSE.,objInst%instSecLPF)
                RootMOOP_F(2) = SecLPFilter(LocalVar%rootMOOP(2),LocalVar%DT, CntrPar%F_FlpCornerFreq, CntrPar%F_FlpDamping, LocalVar%iStatus, .FALSE.,objInst%instSecLPF)
                RootMOOP_F(3) = SecLPFilter(LocalVar%rootMOOP(3),LocalVar%DT, CntrPar%F_FlpCornerFreq, CntrPar%F_FlpDamping, LocalVar%iStatus, .FALSE.,objInst%instSecLPF)
                ! Initialize controller
                IF (CntrPar%Flp_Mode == 2) THEN
                    LocalVar%Flp_Angle(K) = PIIController(RootMyb_VelErr(K), 0 - LocalVar%Flp_Angle(K), CntrPar%Flp_Kp, CntrPar%Flp_Ki, 0.05, -CntrPar%Flp_MaxPit , CntrPar%Flp_MaxPit , LocalVar%DT, 0.0, .TRUE., objInst%instPI)
                ENDIF
            
            ! Steady flap angle
            ELSEIF (CntrPar%Flp_Mode == 1) THEN
                LocalVar%Flp_Angle(1) = LocalVar%Flp_Angle(1) 
                LocalVar%Flp_Angle(2) = LocalVar%Flp_Angle(2) 
                LocalVar%Flp_Angle(3) = LocalVar%Flp_Angle(3) 
                ! IF (MOD(LocalVar%Time,10.0) == 0) THEN
                !     LocalVar%Flp_Angle(1) = LocalVar%Flp_Angle(1) + 1*D2R
                !     LocalVar%Flp_Angle(2) = LocalVar%Flp_Angle(2) + 1*D2R
                !     LocalVar%Flp_Angle(3) = LocalVar%Flp_Angle(3) + 1*D2R
                ! ENDIF

            ! PII flap control
            ELSEIF (CntrPar%Flp_Mode == 2) THEN
                DO K = 1,LocalVar%NumBl
                    ! LPF Blade root bending moment
                    RootMOOP_F(K) = SecLPFilter(LocalVar%rootMOOP(K),LocalVar%DT, CntrPar%F_FlpCornerFreq, CntrPar%F_FlpDamping, LocalVar%iStatus, .FALSE.,objInst%instSecLPF)
                    
                    ! Find derivative and derivative error of blade root bending moment
                    RootMyb_Vel(K) = (RootMOOP_F(K) - RootMyb_Last(K))/LocalVar%DT
                    RootMyb_VelErr(K) = 0 - RootMyb_Vel(K)
                    
                    ! Find flap angle command - includes an integral term to encourage zero flap angle
                    LocalVar%Flp_Angle(K) = PIIController(RootMyb_VelErr(K), 0 - LocalVar%Flp_Angle(K), CntrPar%Flp_Kp, CntrPar%Flp_Ki, 0.05, -CntrPar%Flp_MaxPit , CntrPar%Flp_MaxPit , LocalVar%DT, 0.0, .FALSE., objInst%instPI)
                    ! Saturation Limits
                    LocalVar%Flp_Angle(K) = saturate(LocalVar%Flp_Angle(K), -CntrPar%Flp_MaxPit, CntrPar%Flp_MaxPit)
                    
                    ! Save some data for next iteration
                    RootMyb_Last(K) = RootMOOP_F(K)
                END DO
            ENDIF

            ! Send to AVRSwap
            avrSWAP(120) = LocalVar%Flp_Angle(1) * R2D   ! Needs to be sent to openfast in degrees
            avrSWAP(121) = LocalVar%Flp_Angle(2) * R2D   ! Needs to be sent to openfast in degrees
            avrSWAP(122) = LocalVar%Flp_Angle(3) * R2D   ! Needs to be sent to openfast in degrees
        ELSE
            RETURN
        ENDIF
    END SUBROUTINE FlapControl
END MODULE Controllers
