function [R,F] = load_ROSCO_params(P,simu)
%% Load ROSCO Controller Parameters
% This script is required to load ROSCO control parameters into workspace
% Uses *.IN file parameters as input via P.SD_dllP  (ServoDyn Dll params)


% keyboard;


%% Simulation and controller setup


%% Turbine Parameters

R.RotorRad = GetFASTPar(P.EDP,'TipRad');        % Rotor radius
R.GBRatio = GetFASTPar(P.EDP,'GBRatio');        % Gearbox ratio

R.GenEff    = GetFASTPar(P.SvDP,'GenEff')/100;


%% Pitch Actuator Parameters
% No input yet, define here

R.PitchActBW    = 0.25* 2 * pi;   % rad/s
F_PitchAct      = Af_LPF(R.PitchActBW,0.707,simu.dt);
F.F_PitchAct.b  = F_PitchAct.num{1};
F.F_PitchAct.a  = F_PitchAct.den{1};

%% Torque Control Parameters

R.VS_RefSpd     = GetFASTPar(P.SD_dllP,'VS_RefSpd');  % reference speed for torque control
R.VS_MinOMSpd   = GetFASTPar(P.SD_dllP,'VS_MinOMSpd');  % Minimum rotor speed

R.VS_TSRopt     = GetFASTPar(P.SD_dllP,'VS_TSRopt');  % Minimum rotor speed

R.VS_KP         = GetFASTPar(P.SD_dllP,'VS_KP');        % PI gain schedule
R.VS_KI         = GetFASTPar(P.SD_dllP,'VS_KI');

R.VS_MaxTq      = GetFASTPar(P.SD_dllP,'VS_MaxTq');      % max torque
R.VS_RtTq       = GetFASTPar(P.SD_dllP,'VS_RtTq');      % rated torque
R.VS_MaxRat     = GetFASTPar(P.SD_dllP,'VS_MaxRat');    % max torque rate

R.VS_Rgn3MP     = deg2rad(3); %GetFASTPar(P.SD_dllP,'VS_Rgn3MP');        % torque ratchet? not in .IN file, hard code for now


%% Pitch Control Parameters

R.PC_RefSpd     = GetFASTPar(P.SD_dllP,'PC_RefSpd');
R.PC_GS_angles     = GetFASTPar(P.SD_dllP,'PC_GS_angles');
R.PC_GS_KP     = GetFASTPar(P.SD_dllP,'PC_GS_KP');
R.PC_GS_KI     = GetFASTPar(P.SD_dllP,'PC_GS_KI');


R.PC_MaxPit     = GetFASTPar(P.SD_dllP,'PC_MaxPit');
R.PC_MinPit     = GetFASTPar(P.SD_dllP,'PC_MinPit');
R.PC_MaxRat     = GetFASTPar(P.SD_dllP,'PC_MaxPit');

R.PC_IC         = GetFASTPar(P.EDP,'BlPitch(1)');

%% Setpoint Smoothing Control Parameters

R.SS_VSGain     = GetFASTPar(P.SD_dllP,'SS_VSGain');
R.SS_PCGain     = GetFASTPar(P.SD_dllP,'SS_PCGain');

%% Filter Parameters

R.F_LPFType         = GetFASTPar(P.SD_dllP,'F_LPFType');
R.F_LPFCornerFreq   = GetFASTPar(P.SD_dllP,'F_LPFCornerFreq');

if R.F_LPFType == 2
    F_HSS           = Af_LPF(R.F_LPFCornerFreq,GetFASTPar(P.SD_dllP,'F_LPFDamping'),simu.dt);
else
    F_HSS           = Af_LPF(R.F_LPFCornerFreq,GetFASTPar(P.SD_dllP,'F_LPFDamping'),simu.dt,1);
end
F.HSS.b         = F_HSS.num{1};
F.HSS.a         = F_HSS.den{1};

F_SS            = Af_LPF(GetFASTPar(P.SD_dllP,'F_SSCornerFreq'),1,simu.dt,1);
F.F_SS.b        = F_SS.num{1};
F.F_SS.a        = F_SS.den{1};

F_Wind          = Af_LPF(0.20944,1,simu.dt,1);
F.Wind.b     = F_Wind.num{1};
F.Wind.a     = F_Wind.den{1};


%% Wind Speed Estimator Parameters
% Only the EKF is implemented, for meow

R.WE_Mode           = GetFASTPar(P.SD_dllP,'WE_Mode');

R.WE_BladeRadius    = GetFASTPar(P.SD_dllP,'WE_BladeRadius');
R.WE_CP_n           = GetFASTPar(P.SD_dllP,'WE_CP_n');
R.WE_CP             = GetFASTPar(P.SD_dllP,'WE_CP');
R.WE_Gamma          = GetFASTPar(P.SD_dllP,'WE_Gamma');
R.WE_GearboxRatio   = GetFASTPar(P.SD_dllP,'WE_GearboxRatio');
R.WE_Jtot           = GetFASTPar(P.SD_dllP,'WE_Jtot');
R.WE_RhoAir         = GetFASTPar(P.SD_dllP,'WE_RhoAir');
R.PerfTableSize     = GetFASTPar(P.SD_dllP,'PerfTableSize');
R.WE_FOPoles_N      = GetFASTPar(P.SD_dllP,'WE_FOPoles_N');
R.WE_FOPoles_v      = GetFASTPar(P.SD_dllP,'WE_FOPoles_v');
R.WE_FOPoles        = GetFASTPar(P.SD_dllP,'WE_FOPoles');



% Initial condition
R.WE_v0             = 12;
R.WE_om0            = GetFASTPar(P.EDP,'RotSpeed') * R.WE_GearboxRatio;

% Pitch Input
R.NumBl              = GetFASTPar(P.EDP,'NumBl');
F_WSEBlPitch         = Af_LPF(R.F_LPFCornerFreq/2,1,simu.dt,1);
F.F_WSEBlPitch.b     = F_WSEBlPitch.num{1};
F.F_WSEBlPitch.a     = F_WSEBlPitch.den{1};

% Torque Input
F_GenTq             = Af_LPF(R.F_LPFCornerFreq,0.7,simu.dt);
F.F_GenTq.b         = F_GenTq.num{1};
F.F_GenTq.a         = F_GenTq.den{1};

%% Floating Platform Damper

% Enable
R.Fl_Mode           = GetFASTPar(P.SD_dllP,'Fl_Mode');

% Filters
R.F_FlCornerFreq    = GetFASTPar(P.SD_dllP,'F_FlCornerFreq');
F_Fl_LPF            = Af_LPF(R.F_FlCornerFreq(1),R.F_FlCornerFreq(2),simu.dt) * Af_HPF(R.F_FlCornerFreq(1)/20,1,simu.dt,1);
F.F_Fl.b            = F_Fl_LPF.num{1};
F.F_Fl.a            = F_Fl_LPF.den{1};

% High Pass Filter
F_Fl_HPF            = Af_LPF(1/60,1,simu.dt,1);
F.F_Fl_HPF.b        = F_Fl_HPF.num{1};
F.F_Fl_HPF.a        = F_Fl_HPF.den{1};

% Optional Notch
F.F_NotchType       = GetFASTPar(P.SD_dllP,'F_NotchType');

if F.F_NotchType == 2
    
    F.F_NotchCornerFreq     = GetFASTPar(P.SD_dllP,'F_NotchCornerFreq');
    F.F_NotchBetaNumDen     = GetFASTPar(P.SD_dllP,'F_NotchBetaNumDen');
       
    F_Fl_Notch  = Af_MovingNotch(F.F_NotchCornerFreq,F.F_NotchBetaNumDen(2),F.F_NotchBetaNumDen(1),simu.dt);
    
    F.F_Fl_Notch.b          = F_Fl_Notch.num{1};
    F.F_Fl_Notch.a          = F_Fl_Notch.den{1};
        
else
    F.F_Fl_Notch.b          = 1;
    F.F_Fl_Nothc.a          = 1;
end

% Gain
R.Fl_Kp             = GetFASTPar(P.SD_dllP,'Fl_Kp');

if 0
    
    F_Fl_HPF    = Af_HPF(R.F_FlCornerFreq(1)/50,1,simu.dt,1);
    
    figure(900)
    set(gcf,'Name','Fl Filts');
    bodemag(F_Fl_LPF*F_Fl_HPF)
    
    ylim([-50,2]);

            
end


%% Min Pitch Saturation

% Peak shaving if PS_Mode == 1
R.PS_Mode           = GetFASTPar(P.SD_dllP,'PS_Mode');

% Peak shaving lookup table
R.PS_WindSpeeds     = GetFASTPar(P.SD_dllP,'PS_WindSpeeds');
R.PS_BldPitchMin    = GetFASTPar(P.SD_dllP,'PS_BldPitchMin');

% Filter (hard coded)
F_PS                = Af_LPF(0.21,1,simu.dt,1);
F.F_PS.a            = F_PS.den{1};
F.F_PS.b            = F_PS.num{1};

if 0
    
    
    figure(1000);
    set(gcf,'Name','MP Table');
    
    subplot(211);
    plot(R.PS_WindSpeeds,R.PS_BldPitchMin);
    
    subplot(212);
    bodemag(F_PS);
    ylim([-50,2]);
    
end


