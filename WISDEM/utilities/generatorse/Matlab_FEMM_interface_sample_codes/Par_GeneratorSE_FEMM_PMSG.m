clear all
clc
close all
addpath('c:\femm42\mfiles');
openfemm;
newdocument(0)

disp('GeneratorSE Validation for PMSG by FEMM');
disp('Latha Sethuraman and Katherine Dykes')
disp('Copyright (c) NREL. All rights reserved.')
disp(' For queries contact : Latha Sethuraman@nrel.gov')
disp(' ');
[Parameters,txt,raw] =xlsread('C:\GeneratorSE\src\generatorse\PMDD_5.0MW.xlsx'); % Specify the path of the GeneratorSE output excel files
depth=raw{17,3};
mi_probdef(0, 'meters', 'planar', 1e-8,depth,30);
mi_getmaterial('NdFeB 40 MGOe') %fetches the material specified by materialname from
mi_modifymaterial('NdFeB 40 MGOe',0,'Magnet')
mi_getmaterial('Pure Iron') 
mi_modifymaterial('Pure Iron',0,'Iron')
mi_addmaterial('Steel',5000,5000,0,0,0,0,0,0,0,0)
mi_addmaterial('Air', 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0);
mi_getmaterial('20 SWG') %fetches the material specified by materialname from
mi_addmaterial('20 SWG')

mi_addboundprop('A=0', 0, 0, 0, 0, 0, 0, 0, 0, 0)
% mi_addboundprop('apbc1', 0, 0, 0, 0, 0, 0, 0, 0, 5)
% mi_addboundprop('apbc2', 0, 0, 0, 0, 0, 0, 0, 0, 5)
% mi_addboundprop('apbc3', 0, 0, 0, 0, 0, 0, 0, 0, 5)
% mi_addboundprop('apbc4', 0, 0, 0, 0, 0, 0, 0, 0, 5)
% mi_addboundprop('apbc5', 0, 0, 0, 0, 0, 0, 0, 0, 5)
% mi_addboundprop('apbc6', 0, 0, 0, 0, 0, 0, 0, 0, 5)

 
 
 %read data example: Import columns as column vectors 
% [X Y Z] = csvimport('vectors.csv', 'columns', {'X, 'Y', 'Z'});

%remove headers

% M = csvread(filename,R1,C1,[R1 C1 R2 C2]) 
 
 tau_p=raw{22,3}/1000;
 b_p=raw{29,3}/1000;
 h_s=raw{23,3}/1000;
 h_ys=raw{26,3}/1000;
 h_yr=raw{27,3}/1000;
 N_st=raw{43,3};
 
 
 Poles=raw{36,3}*2;
 Magnet_height=raw{28,3}*0.001;
 Magnets=zeros(Poles,19);
 A=length(Magnets);
 Current=raw{1,3}*1e6/(0.85*3*raw{38,3})*0;
 
 % Circuit properties
 mi_addcircprop('A+', Current, 2)
mi_addcircprop('A-', Current, 2)
mi_addcircprop('B+', Current, 2)
mi_addcircprop('B-', Current, 2)
mi_addcircprop('C+', Current, 2)
mi_addcircprop('C-', Current, 2)
 
 
 
 
 Magnet_outer_radius=raw{17,3}*0.5-0.001*raw{17,3};
 Magnet_inner_radius=Magnet_outer_radius-Magnet_height;
 Magnet_hratio=Magnet_outer_radius/Magnet_inner_radius;
 Stator_tooth_rad=(raw{17,3})/2;
 Stator_yoke_irad=Stator_tooth_rad+raw{23,3}*0.001;
 tau_p_new=Magnet_inner_radius*2*pi/Poles;
 b_t=raw{25,3}*0.001;
 b_s=raw{24,3}*0.001;
 Stator_ratio=Stator_yoke_irad/Stator_tooth_rad;
 Magnets(1,1)=b_p*0.5/Magnet_inner_radius; 
 Magnets(1,2)= b_p*0.5*Magnet_hratio/Magnet_outer_radius;%Angle 1
 Magnets(1,3)=tau_p_new/Magnet_inner_radius; 
 Magnets(1,4)=Magnet_inner_radius*cos(Magnets(1,1));
 Magnets(1,5)=Magnet_inner_radius*sin(Magnets(1,1));
 Magnets(1,6)=Magnet_inner_radius*cos(Magnets(1,1));
 Magnets(1,7)=-Magnet_inner_radius*sin(Magnets(1,1));
 Magnets(1,8)=Magnet_outer_radius*cos(Magnets(1,2));
 Magnets(1,9)=Magnet_outer_radius*sin(Magnets(1,2));
 Magnets(1,10)=Magnet_outer_radius*cos(Magnets(1,2));
 Magnets(1,11)=-Magnet_outer_radius*sin(Magnets(1,2));
 Magnets(1,12)=(tau_p_new-b_p)/2+b_p*0.5;
 Magnets(1,13)=Magnets(1,12)/(Magnet_inner_radius);
 Magnets(1,14)=Magnet_inner_radius*cos(Magnets(1,13));
 Magnets(1,15)=Magnet_inner_radius*sin(Magnets(1,13));
 Magnets(1,16)=Magnet_inner_radius*cos(Magnets(1,13));
 Magnets(1,17)=-Magnet_inner_radius*sin(Magnets(1,13));
 Magnets(1,18)=0;
 Magnets(1,19)=0;
 Magnets(1,20)=(Magnet_inner_radius+0.005)*cos(Magnets(1,18));
 Magnets(1,21)=(Magnet_inner_radius+0.005)*sin(Magnets(1,18));
 
 
 for i=1:1
    mi_addnode(Magnets(i,4),Magnets(i,5))
        mi_selectnode(Magnets(i,4),Magnets(i,5))
       mi_setgroup(1)
         mi_addnode(Magnets(i,6),Magnets(i,7))
        mi_selectnode(Magnets(i,6),Magnets(i,7))
       mi_setgroup(1)
       mi_addnode(Magnets(i,8),Magnets(i,9))
        mi_selectnode(Magnets(i,8),Magnets(i,9))
       mi_setgroup(1)
        mi_addnode(Magnets(i,10),Magnets(i,11))
        mi_selectnode(Magnets(i,10),Magnets(i,11))
       mi_setgroup(1)
        mi_addnode(Magnets(i,14),Magnets(i,15))
        mi_selectnode(Magnets(i,14),Magnets(i,15))
       mi_setgroup(1)
         mi_addnode(Magnets(i,16),Magnets(i,17))
        mi_selectnode(Magnets(i,16),Magnets(i,17))
       mi_setgroup(1)
 end
 
 mi_addarc(Magnets(1,6),Magnets(1,7),Magnets(1,4),Magnets(1,5),1,1)
 
 mi_addarc(Magnets(1,10),Magnets(1,11),Magnets(1,8),Magnets(1,9),1,1)
 mi_addarc(Magnets(1,14),Magnets(1,15),Magnets(1,4),Magnets(1,5),1,1)
 mi_addarc(Magnets(1,6),Magnets(1,7),Magnets(1,16),Magnets(1,17),1,1)
 mi_addsegment(Magnets(1,4),Magnets(1,5),Magnets(1,8),Magnets(1,9))
 mi_addsegment(Magnets(1,6),Magnets(1,7),Magnets(1,10),Magnets(1,11))
 mi_selectarcsegment(Magnets(1,14),Magnets(1,15))
  mi_selectarcsegment(Magnets(1,16),Magnets(1,17))
 mi_selectarcsegment(Magnet_inner_radius,0)
 mi_selectarcsegment(Magnets(1,8),Magnets(1,9))
 
 
 mi_setgroup(1)
 mi_selectsegment(Magnets(1,6),Magnets(1,7))
 mi_selectsegment(Magnets(1,8),Magnets(1,9))
 mi_setgroup(1)
 
 mi_selectgroup(1)
 Angle=(tau_p_new/Magnet_inner_radius)*180/pi;
 mi_copyrotate(0,0,Angle,A-1)
 
 for p=1:A;
 Magnets(p+1,19)=Magnets(p,19)+(tau_p_new*((Magnet_inner_radius+0.005)/(Magnet_inner_radius))/(Magnet_inner_radius+0.005));
 Magnets(p+1,20)=(Magnet_inner_radius+0.005)*cos(Magnets(p+1,19));
 Magnets(p+1,21)=(Magnet_inner_radius+0.005)*sin(Magnets(p+1,19));
 end
 
  
 Magnets(1,22)=b_p*0.5/2;
 Magnets(2,22)=tau_p;
 Magnets(1,23)=-1*(180-Magnets(1,22)*180/Magnet_inner_radius/pi);
 Magnets(2,23)=Magnets(2,22)*180/Magnet_inner_radius/pi;
 
   m=3;  
while m < A+1   %Magnet angles
    Magnets(m,22)=Magnets(m-1,22)+tau_p;
    Magnets(m+1,22)=Magnets(m,22)+tau_p;
    Magnets(m,23)=-1*(180-Magnets(m,22)*180/Magnet_inner_radius/pi);
    Magnets(m+1,23)=(Magnets(m+1,22)/Magnet_inner_radius)*180/pi;
    m=m+2;
end


 h_w=0.005;
 b_so=0.004;
 Stator_inner_rad=raw{17,3}*0.5;
 Wedge_rad=Stator_inner_rad+h_w;
 Stator_yoke_irad=Wedge_rad+h_s-h_w;
 Stator(1,1)=b_so*0.5;
 Stator(1,2)=Stator(1,1)/Stator_inner_rad;
 Stator(1,3)=Stator_inner_rad*cos(Stator(1,2));
 Stator(1,4)=Stator_inner_rad*sin(Stator(1,2));
  Stator(1,5)=Stator_inner_rad*cos(Stator(1,2));
 Stator(1,6)=-Stator_inner_rad*sin(Stator(1,2));
 Stator(1,7)=b_s*0.5*Wedge_rad/Stator_inner_rad;
 Stator(1,8)=Stator(1,7)/Wedge_rad;
 Stator(1,9)=Wedge_rad*cos(Stator(1,8));
 Stator(1,10)=Wedge_rad*sin(Stator(1,8));
 Stator(1,11)=Wedge_rad*cos(Stator(1,8));
 Stator(1,12)=-Wedge_rad*sin(Stator(1,8));
 Stator(1,13)=Stator(1,7)*Stator_yoke_irad/Stator_yoke_irad/Wedge_rad;
 Stator(1,14)=Stator_yoke_irad*cos(Stator(1,13));
 Stator(1,15)=Stator_yoke_irad*sin(Stator(1,13));
 Stator(1,16)=Stator_yoke_irad*cos(Stator(1,13));
 Stator(1,17)=-Stator_yoke_irad*sin(Stator(1,13));
 Stator(1,18)=b_s*0.5+b_t*0.5;
 Stator(1,19)=Stator(1,18)/Stator_inner_rad;
 Stator(1,20)=Stator_inner_rad*cos(Stator(1,19));
 Stator(1,21)=Stator_inner_rad*sin(Stator(1,19));
 Stator(1,22)=Stator_inner_rad*cos(Stator(1,19));
 Stator(1,23)=-Stator_inner_rad*sin(Stator(1,19));
 Stator(1,24)=Wedge_rad;
 Stator(1,30)=Stator(1,24);
 Stator(1,31)=0;
  
 
 
 
 mi_addnode(Stator(1,3),Stator(1,4))
  mi_selectnode(Stator(1,3),Stator(1,4))
       mi_setgroup(2)
 mi_addnode(Stator(1,5),Stator(1,6))
 mi_selectnode(Stator(1,5),Stator(1,6))
       mi_setgroup(2)
  mi_addnode(Stator(1,9),Stator(1,10))
  mi_selectnode(Stator(1,9),Stator(1,10))
       mi_setgroup(2)
 mi_addnode(Stator(1,11),Stator(1,12))
 mi_selectnode(Stator(1,11),Stator(1,12))
       mi_setgroup(2)
  mi_addnode(Stator(1,14),Stator(1,15))
  mi_selectnode(Stator(1,14),Stator(1,15))
       mi_setgroup(2)
 mi_addnode(Stator(1,16),Stator(1,17))
 mi_selectnode(Stator(1,16),Stator(1,17))
       mi_setgroup(2)
 mi_addnode(Stator(1,20),Stator(1,21))
 mi_selectnode(Stator(1,20),Stator(1,21))
       mi_setgroup(2)
 mi_addnode(Stator(1,22),Stator(1,23))
 mi_selectnode(Stator(1,22),Stator(1,23))
       mi_setgroup(2)
       
        mi_addnode(Stator(1,26),Stator(1,27))
 mi_selectnode(Stator(1,26),Stator(1,27))
       mi_setgroup(2)
       
       mi_addnode(Stator(1,28),Stator(1,29))
 mi_selectnode(Stator(1,28),Stator(1,29))
       mi_setgroup(2)
       
                  
       
  mi_addsegment(Stator(1,3),Stator(1,4),Stator(1,9),Stator(1,10))
  mi_selectsegment(Stator(1,3),Stator(1,4))
  mi_setgroup(2)
  mi_addsegment(Stator(1,5),Stator(1,6),Stator(1,11),Stator(1,12))
  mi_selectsegment(Stator(1,5),Stator(1,6))
  mi_setgroup(2)
  mi_addsegment(Stator(1,9),Stator(1,10),Stator(1,14),Stator(1,15))
  mi_selectsegment(Stator(1,14),Stator(1,15))
  mi_setgroup(2)
   mi_addsegment(Stator(1,9),Stator(1,10),Stator(1,11),Stator(1,12))
  mi_selectsegment(Stator(1,30),Stator(1,31))
  mi_setgroup(2)
   mi_addsegment(Stator(1,11),Stator(1,12),Stator(1,16),Stator(1,17))
   mi_selectsegment(Stator(1,16),Stator(1,17))
   mi_setgroup(2)
   mi_addarc(Stator(1,14),Stator(1,15),Stator(1,16),Stator(1,17),1,1)
   mi_selectarcsegment(Stator(1,14),Stator(1,15))
  mi_setgroup(2)
   mi_addarc(Stator(1,3),Stator(1,4),Stator(1,20),Stator(1,21),1,1)
   mi_selectarcsegment(Stator(1,20),Stator(1,21))
  mi_setgroup(2)
   mi_addarc(Stator(1,5),Stator(1,6),Stator(1,22),Stator(1,23),1,1)
   mi_selectarcsegment(Stator(1,22),Stator(1,23))
  mi_setgroup(2)
 
  
  mi_selectgroup(2)
 Angle=((b_s+b_t)/Stator_inner_rad)*180/pi ;
 Slots=raw{42,3};
   mi_copyrotate(0,0,Angle,Slots-1)
  
  SLabel(1,1)=0;
  SLabel(1,2)=(Stator_inner_rad+0.5*h_s)*cos(SLabel(1,1));
  SLabel(1,3)=(Stator_inner_rad+0.5*h_s)*sin(SLabel(1,1));
  
  
   
  
  
   t=1;
  while t <Slots+1  % circuit labels
  SLabel(t+1,1)=SLabel(t,1)+((b_s+b_t)*(Stator_inner_rad+0.5*h_s)/Stator_inner_rad)/(Stator_inner_rad+0.5*h_s);
  SLabel(t+1,2)=(Stator_inner_rad+0.5*h_s)*cos(SLabel(t+1,1));
  SLabel(t+1,3)=(Stator_inner_rad+0.5*h_s)*sin(SLabel(t+1,1)); 
%   mi_addnode(Stator(t+1,13),Stator(t+1,14));
  t=t+1;
  end
%   
  u=1;
  while u <Slots+1 
  mi_addblocklabel(SLabel(u,2),SLabel(u,3));
  mi_setgroup(A*100+u);
mi_selectlabel(SLabel(u,2),SLabel(u,3)) ;
 mi_setblockprop('20 SWG', 0, 0, 'A+', 0,A*100+u,N_st);
 mi_addblocklabel(SLabel(u+1,2),SLabel(u+1,3));
  mi_setgroup(A*100+u+1);
mi_selectlabel(SLabel(u+1,2),SLabel(u+1,3)) ;
 mi_setblockprop('20 SWG', 0, 0, 'C-', 0,A*100+u+1,N_st);
 mi_addblocklabel(SLabel(u+2,2),SLabel(u+2,3));
  mi_setgroup(A*100+u+2);
mi_selectlabel(SLabel(u+2,2),SLabel(u+2,3)) ;
 mi_setblockprop('20 SWG', 0, 0, 'B+', 0,A*100+u+2,N_st);
 mi_addblocklabel(SLabel(u+3,2),SLabel(u+3,3));
  mi_setgroup(A*100+u+3);
mi_selectlabel(SLabel(u+3,2),SLabel(u+3,3)) ;
 mi_setblockprop('20 SWG', 0, 0, 'A-', 0,A*100+u+3,N_st);
 mi_addblocklabel(SLabel(u+4,2),SLabel(u+4,3));
  mi_setgroup(A*100+u+4);
mi_selectlabel(SLabel(u+4,2),SLabel(u+4,3)) ;
 mi_setblockprop('20 SWG', 0, 0, 'C+', 0,A*100+u+4,N_st);
 mi_addblocklabel(SLabel(u+5,2),SLabel(u+5,3));
  mi_setgroup(A*100+u+5);
mi_selectlabel(SLabel(u+5,2),SLabel(u+5,3)) ;
 mi_setblockprop('20 SWG', 0, 0, 'B-', 0,A*100+u+5,N_st);
 u=u+6;
  end
Outer_dia=Stator_inner_rad+h_s+h_ys;  
  
 mi_addnode(Outer_dia,0);
 mi_addnode(-Outer_dia,0);
 mi_addarc(Outer_dia,0,-Outer_dia,0,180,1); 
 mi_selectarcsegment(Outer_dia,0)
 mi_setarcsegmentprop(0.8257, 'A=0', 0, 0); 
  mi_addarc(-Outer_dia,0,Outer_dia,0,180,1); 
  mi_selectarcsegment(0,-Outer_dia)
   mi_setarcsegmentprop(0.8257, 'A=0', 0, 0); 
Inner_rad=Magnet_inner_radius-raw{27,3}*0.001;
mi_addnode(Inner_rad,0);
mi_addnode(-Inner_rad,0);

mi_addarc(-Inner_rad,0,Inner_rad,0,180,1);
mi_addarc(Inner_rad,0,-Inner_rad,0,180,1);
mi_selectarcsegment(0,-Inner_rad)
mi_selectarcsegment(0,Inner_rad)
 mi_setarcsegmentprop(0.8257, 'A=0', 0, 0);

 Iron_block=Stator_yoke_irad +raw{24,3}*0.001*0.5;
 mi_addblocklabel( Iron_block*cosd(45),Iron_block*sind(45));
 mi_setgroup(10000);
 mi_selectlabel( Iron_block*cosd(45),Iron_block*sind(45));
  
 mi_setblockprop('Iron', 0, 0, 'None', 0,10000,0);
 
RY=Inner_rad+raw{26,3}*0.001*0.5;
mi_addblocklabel( RY*cosd(45),RY*sind(45));
mi_setgroup(10002);
  mi_selectlabel( RY*cosd(45),RY*sind(45));
  mi_setblockprop('Iron', 0, 0, 'None', 0,10002,0);


 mi_addblocklabel( Inner_rad*0.5*cosd(45),Inner_rad*0.5*sind(45));
mi_setgroup(10001);
  mi_selectlabel( Inner_rad*0.5*cosd(45),Inner_rad*0.5*sind(45));
  
mi_setblockprop('Air', 0, 0, 'None', 0,10001,0);


Air_gap_rad=(raw{17,3})/2-0.001*0.5*(raw{17,3});
mi_addblocklabel( Air_gap_rad*cosd(45),Air_gap_rad*sind(45));
mi_setgroup(10003);
  mi_selectlabel( Air_gap_rad*cosd(45),Air_gap_rad*sind(45));
  
mi_setblockprop('Air', 0, 0, 'None', 0,10003,0);

for n=1:A+1
 mi_addblocklabel(Magnets(n,20),Magnets(n,21));
 mi_setgroup(500*n);
 mi_selectlabel(Magnets(n,20),Magnets(n,21)) ;
 mi_setblockprop('Magnet', 0, 0, '<None>', Magnets(n,23),500*n,0);
  end
mi_saveas('5MW_PMSG_new.fem')
%  mi_analyze
%  mi_loadsolution
