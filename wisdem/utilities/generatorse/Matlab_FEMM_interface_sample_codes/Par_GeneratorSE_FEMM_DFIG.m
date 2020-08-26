clear all
close all
clc

addpath('c:\femm42\\mfiles');
openfemm;
newdocument(0)

disp('GeneratorSE Validation for DFIG by FEMM');
disp('Latha Sethuraman and Katherine Dykes')
disp('Copyright (c) NREL. All rights reserved.')
disp(' For queries,please contact : Latha Sethuraman@nrel.gov or Katherine.Dykes @nrel.gov')
disp(' ');

[Parameters,txt,raw] =xlsread('C:\GeneratorSE\src\generatorse\DFIG_5.0MW.xlsx'); % Specify the path of the GeneratorSE output excel files
depth=raw{4,3};
mi_probdef(0, 'meters', 'planar', 1e-8,depth,30);
mi_getmaterial('Pure Iron') 
mi_modifymaterial('Pure Iron',0,'Iron')
mi_addmaterial('Steel',5000,5000,0,0,0,0,0,0,0,0)
mi_addmaterial('Air', 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0);
mi_getmaterial('20 SWG') %fetches the material specified by materialname from
mi_modifymaterial('20 SWG',0,'Stator')
mi_getmaterial('20 SWG')
mi_modifymaterial('20 SWG',0,'Rotor')


mi_addboundprop('A=0', 0, 0, 0, 0, 0, 0, 0, 0, 0)
mi_addboundprop('apbc1', 0, 0, 0, 0, 0, 0, 0, 0, 5)
mi_addboundprop('apbc2', 0, 0, 0, 0, 0, 0, 0, 0, 5)
mi_addboundprop('apbc3', 0, 0, 0, 0, 0, 0, 0, 0, 5)
mi_addboundprop('apbc4', 0, 0, 0, 0, 0, 0, 0, 0, 5)
mi_addboundprop('apbc5', 0, 0, 0, 0, 0, 0, 0, 0, 5)
mi_addboundprop('apbc6', 0, 0, 0, 0, 0, 0, 0, 0, 5)
mi_addboundprop('apbc7', 0, 0, 0, 0, 0, 0, 0, 0, 5)

 
 
 %read data example: Import columns as column vectors 
% [X Y Z] = csvimport('vectors.csv', 'columns', {'X, 'Y', 'Z'});

%remove headers

% M = csvread(filename,R1,C1,[R1 C1 R2 C2]) 
 
 tau_p=raw{8,3}/1000;
 h_s=raw{10,3}/1000;
 b_s=raw{12,3}/1000;
 h_w=0.005;
 b_so =0.004;
 b_t=raw{14,3}/1000;
 h_ys=raw{15,3}/1000;
 b_ro=0.004;
 h_r=raw{18,3}/1000;
 b_r=raw{19,3}/1000;
 b_tr=raw{21,3}/1000;
 h_yr=raw{17,3}/1000;
 Slots=raw{9,3};
 N_st=2; %turns per coil
 A_s=raw{34,3};
%  N_st=raw{31,3};
 A_r=raw{41,3};
 N_rslots=raw{16,3};
 N_rt=round(raw{40,3})/(N_rslots/3);
 N_s=round(sqrt(A_s*4/pi)/0.9144);
 N_r=round(sqrt(A_r*4/pi)/0.9144);
 mi_modifymaterial('Stator',12,N_s)
mi_modifymaterial('Rotor',9,4)
mi_modifymaterial('Rotor',12,N_r)
 
 
 Stator_Current=raw{31,3}*0;
 Field_Current=raw{42,3};
 % Circuit properties
 mi_addcircprop('A+', Stator_Current, 1)
mi_addcircprop('A-', Stator_Current, 1)
mi_addcircprop('B+', Stator_Current, 1)
mi_addcircprop('B-', Stator_Current,1)
mi_addcircprop('C+', Stator_Current, 1)
mi_addcircprop('C-', Stator_Current, 1)
  I=Field_Current;
mi_addcircprop('A1+', Field_Current, 1)
mi_addcircprop('A1-',-Field_Current, 1)
mi_addcircprop('B1+',Field_Current*sind(120) , 1)
mi_addcircprop('B1-', -Field_Current*sind(120),1)
mi_addcircprop('C1+', Field_Current*sind(240), 1)
mi_addcircprop('C1-', -Field_Current*sind(240), 1)



  
g=(0.1+0.012*(raw{2,3}*1e6)^(1/3))*0.001;
 Rotor_outer_radius=raw{4,3}*0.5-g;
 Stator_inner_rad=raw{4,3}*0.5;
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
%  Stator(1,24)=Wedge_rad+0.15*h_s;
%  Stator(1,25)=Stator(1,7)*Stator(1,24)/Wedge_rad/Stator(1,24);
%  Stator(1,26)=Stator(1,24)*cos(Stator(1,25));
%  Stator(1,27)=Stator(1,24)*sin(Stator(1,25));
%  Stator(1,28)=Stator(1,24)*cos(Stator(1,25));
%  Stator(1,29)=-Stator(1,24)*sin(Stator(1,25));
 Stator(1,30)=Wedge_rad+0.5*h_s;
 Stator(1,31)=Stator(1,7)*Stator(1,30)/Wedge_rad/Stator(1,30);
  Stator(1,32)=Stator(1,30)*cos(Stator(1,31));
 Stator(1,33)=Stator(1,30)*sin(Stator(1,31));
 Stator(1,34)=Stator(1,30)*cos(Stator(1,31));
 Stator(1,35)=-Stator(1,30)*sin(Stator(1,31));
 
 Stator(1,36)=Stator(1,24);
 Stator(1,37)=0;
 Stator(1,38)=Stator(1,30);
 Stator(1,39)=0;


 
 
 
 mi_addnode(Stator(1,3),Stator(1,4))
  mi_selectnode(Stator(1,3),Stator(1,4))
       mi_setgroup(1)
 mi_addnode(Stator(1,5),Stator(1,6))
 mi_selectnode(Stator(1,5),Stator(1,6))
       mi_setgroup(1)
  mi_addnode(Stator(1,9),Stator(1,10))
  mi_selectnode(Stator(1,9),Stator(1,10))
       mi_setgroup(1)
 mi_addnode(Stator(1,11),Stator(1,12))
 mi_selectnode(Stator(1,11),Stator(1,12))
       mi_setgroup(1)
  mi_addnode(Stator(1,14),Stator(1,15))
  mi_selectnode(Stator(1,14),Stator(1,15))
       mi_setgroup(1)
 mi_addnode(Stator(1,16),Stator(1,17))
 mi_selectnode(Stator(1,16),Stator(1,17))
       mi_setgroup(1)
 mi_addnode(Stator(1,20),Stator(1,21))
 mi_selectnode(Stator(1,20),Stator(1,21))
       mi_setgroup(1)
 mi_addnode(Stator(1,22),Stator(1,23))
 mi_selectnode(Stator(1,22),Stator(1,23))
       mi_setgroup(1)
       
%         mi_addnode(Stator(1,26),Stator(1,27))
%  mi_selectnode(Stator(1,26),Stator(1,27))
%        mi_setgroup(1)
%        
%        mi_addnode(Stator(1,28),Stator(1,29))
%  mi_selectnode(Stator(1,28),Stator(1,29))
%        mi_setgroup(1)
       
         mi_addnode(Stator(1,32),Stator(1,33))
 mi_selectnode(Stator(1,32),Stator(1,33))
       mi_setgroup(1)
       
        mi_addnode(Stator(1,34),Stator(1,35))
 mi_selectnode(Stator(1,34),Stator(1,35))
       mi_setgroup(1)
       
       
       
  mi_addsegment(Stator(1,3),Stator(1,4),Stator(1,9),Stator(1,10))
  mi_selectsegment(Stator(1,3),Stator(1,4))
  mi_setgroup(1)
  mi_addsegment(Stator(1,5),Stator(1,6),Stator(1,11),Stator(1,12))
  mi_selectsegment(Stator(1,5),Stator(1,6))
  mi_setgroup(1)
   mi_addsegment(Stator(1,9),Stator(1,10),Stator(1,11),Stator(1,12))
   mi_selectsegment(Wedge_rad,0)
   mi_setgroup(1)
%    mi_addsegment(Stator(1,11),Stator(1,12),Stator(1,28),Stator(1,29))
%    mi_selectsegment(Stator(1,26),Stator(1,27))
%    mi_selectsegment(Stator(1,28),Stator(1,29))
%    mi_setgroup(1)
   mi_addsegment(Stator(1,9),Stator(1,10),Stator(1,32),Stator(1,33))
   mi_selectsegment(Stator(1,32),Stator(1,33))
   mi_setgroup(1)
   mi_addsegment(Stator(1,11),Stator(1,12),Stator(1,34),Stator(1,35))
   mi_selectsegment(Stator(1,34),Stator(1,35))
   mi_setgroup(1)
%    mi_addsegment(Stator(1,26),Stator(1,27),Stator(1,28),Stator(1,29))
%    mi_selectsegment(Stator(1,36),Stator(1,37))
%    mi_setgroup(1)
   mi_addsegment(Stator(1,32),Stator(1,33),Stator(1,34),Stator(1,35))
   mi_selectsegment(Stator(1,38),Stator(1,39))
   mi_setgroup(1)
   
   mi_addsegment(Stator(1,32),Stator(1,33),Stator(1,14),Stator(1,15))
   mi_selectsegment(Stator(1,14),Stator(1,15))
   mi_setgroup(1)
   mi_addsegment(Stator(1,34),Stator(1,35),Stator(1,16),Stator(1,17))
   mi_selectsegment(Stator(1,16),Stator(1,17))
   mi_setgroup(1)
    mi_addarc(Stator(1,14),Stator(1,15),Stator(1,16),Stator(1,17),1,1)
   mi_selectarcsegment(Stator(1,14),Stator(1,15))
  mi_setgroup(1)
   mi_addarc(Stator(1,3),Stator(1,4),Stator(1,20),Stator(1,21),1,1)
   mi_selectarcsegment(Stator(1,20),Stator(1,21))
  mi_setgroup(1)
%    mi_addarc(Stator(1,9),Stator(1,10),Stator(1,11),Stator(1,12),1,1)
%    mi_selectarcsegment(Stator(1,10),Stator(1,11))
%    mi_setgroup(1)
   mi_addarc(Stator(1,5),Stator(1,6),Stator(1,22),Stator(1,23),1,1)
   mi_selectarcsegment(Stator(1,22),Stator(1,23))
  mi_setgroup(1)
 
  
  mi_selectgroup(1)
 Angle=((b_s+b_t)/Stator_inner_rad)*180/pi ;
 
%  Slots=60;
   mi_copyrotate(0,0,Angle,Slots)
   
   
%    Stator labels

Slabel_inner=(Stator_inner_rad+h_w+0.35*h_s);
Slabel_outer=(Stator_inner_rad+h_w+0.775*h_s);
slot_pitch_labeli=(b_s+b_t)*Slabel_inner/Stator_inner_rad;
slot_pitch_labelo=(b_s+b_t)*Slabel_outer/Stator_inner_rad;
SlabelI(1,1)=0;
SlabelI(1,2)=Slabel_inner;
SlabelI(1,3)=0;
SlabelI(2,1)=0;
SlabelI(2,2)=Slabel_outer;
SlabelI(2,3)=0;

SlabelI(3,1)=(slot_pitch_labeli/Slabel_inner);
SlabelI(3,2)=Slabel_inner*cos(SlabelI(3,1));
SlabelI(3,3)=Slabel_inner*sin(SlabelI(3,1));

SlabelI(4,1)=(slot_pitch_labelo/Slabel_outer);
SlabelI(4,2)=Slabel_outer*cos(SlabelI(4,1));
SlabelI(4,3)=Slabel_outer*sin(SlabelI(4,1));

i=4;
while i<4*Slots+1
    if rem(i,2)<=0
    SlabelI(i+1,1)=SlabelI(i,1)+slot_pitch_labeli/Slabel_inner;
    SlabelI(i+1,2)=Slabel_inner*cos(SlabelI(i+1,1));
    SlabelI(i+1,3)=Slabel_inner*sin(SlabelI(i+1,1));
    else
    SlabelI(i+1,1)=SlabelI(i-2,1)+slot_pitch_labelo/Slabel_outer;
    SlabelI(i+1,2)=Slabel_outer*cos(SlabelI(i+1,1));
    SlabelI(i+1,3)=Slabel_outer*sin(SlabelI(i+1,1));
    end
    i=i+1;
end
for j=1: 4*Slots+1
mi_addblocklabel(SlabelI(j,2),SlabelI(j,3));
 mi_setgroup(10*j);
end

k=1;
while k<4*Slots/2+1
  mi_selectlabel(SlabelI(k,2),SlabelI(k,3)) ;
  mi_selectlabel(SlabelI(k+2,2),SlabelI(k+2,3))
 mi_selectlabel(SlabelI(k+4,2),SlabelI(k+4,3));
 mi_selectlabel(SlabelI(k+6,2),SlabelI(k+6,3));
 mi_selectlabel(SlabelI(k+7,2),SlabelI(k+7,3));
 mi_selectlabel(SlabelI(k+8,2),SlabelI(k+8,3));
 mi_selectlabel(SlabelI(k+9,2),SlabelI(k+9,3));
 mi_selectlabel(SlabelI(k+11,2),SlabelI(k+11,3));
  mi_selectlabel(SlabelI(k+13,2),SlabelI(k+13,3));
  mi_selectlabel(SlabelI(k+15,2),SlabelI(k+15,3));
  if k<10 | k>60 & k<70 | k>120 & k<130
 mi_setblockprop('Stator', 0, 0, 'B+', 0,200*k,N_st);
 mi_setgroup(200*k);
   elseif k>10 & k<20| k>70 & k<80| k>130 & k<140
      mi_setblockprop('Stator', 0, 0, 'A-', 0,200*k+1,N_st);
      mi_setgroup(200*k);
  elseif k>20 & k<30| k>80 & k<90| k>140 & k<150
 mi_setblockprop('Stator', 0, 0, 'C+', 0,200*k+1,N_st);
 mi_setgroup(200*k);
 elseif k>30 & k<40| k>90 & k<100| k>150 & k<160
 mi_setblockprop('Stator', 0, 0, 'B-', 0,200*k+1,N_st);
 mi_setgroup(200*k);
  elseif k>40 & k<50| k>100 & k<110| k>160 & k<170
 mi_setblockprop('Stator', 0, 0, 'A+', 0,200*k+1,N_st);
 mi_setgroup(200*k);
  elseif k>50 & k<60| k>110 & k<120| k>170 & k<180
 mi_setblockprop('Stator', 0, 0, 'C-', 0,200*k+1,N_st);
 mi_setgroup(200*k);
  end
  k=k+10;
 
end


 Wedge_rad_rotor=Rotor_outer_radius-h_w;
 Rotor_yoke_irad=Rotor_outer_radius-h_r;
 Rotor(1,1)=b_ro*0.5;
 Rotor(1,2)=Rotor(1,1)/Rotor_outer_radius;
 Rotor(1,3)=Rotor_outer_radius*cos(Rotor(1,2));
 Rotor(1,4)=Rotor_outer_radius*sin(Rotor(1,2));
  Rotor(1,5)=Rotor_outer_radius*cos(Rotor(1,2));
 Rotor(1,6)=-Rotor_outer_radius*sin(Rotor(1,2));
 Rotor(1,7)=b_r*0.5*Wedge_rad_rotor/Rotor_outer_radius;
 Rotor(1,8)=Rotor(1,7)/Wedge_rad_rotor;
 Rotor(1,9)=Wedge_rad_rotor*cos(Rotor(1,8));
 Rotor(1,10)=Wedge_rad_rotor*sin(Rotor(1,8));
 Rotor(1,11)=Wedge_rad_rotor*cos(Rotor(1,8));
 Rotor(1,12)=-Wedge_rad_rotor*sin(Rotor(1,8));
 Rotor(1,13)=Rotor(1,7)*Rotor_yoke_irad/Rotor_yoke_irad/Wedge_rad_rotor;
 Rotor(1,14)=Rotor_yoke_irad*cos( Rotor(1,13));
 Rotor(1,15)=Rotor_yoke_irad*sin(Rotor(1,13));
 Rotor(1,16)=Rotor_yoke_irad*cos(Rotor(1,13));
 Rotor(1,17)=-Rotor_yoke_irad*sin(Rotor(1,13));
 Rotor(1,18)=b_r*0.5+b_tr*0.5;
 Rotor(1,19)=Rotor(1,18)/Rotor_outer_radius;
 Rotor(1,20)=Rotor_outer_radius*cos(Rotor(1,19));
 Rotor(1,21)=Rotor_outer_radius*sin(Rotor(1,19));
 Rotor(1,22)=Rotor_outer_radius*cos(Rotor(1,19));
 Rotor(1,23)=-Rotor_outer_radius*sin(Rotor(1,19));
 Rotor(1,24)=Wedge_rad_rotor;
 Rotor(1,25)=Rotor(1,7)*Rotor(1,24)/Wedge_rad_rotor/Rotor(1,24);
 Rotor(1,26)=Rotor(1,24)*cos(Rotor(1,25));
 Rotor(1,27)=Rotor(1,24)*sin(Rotor(1,25));
 Rotor(1,28)=Rotor(1,24)*cos(Rotor(1,25));
 Rotor(1,29)=-Rotor(1,24)*sin(Rotor(1,25));
 Rotor(1,30)=Rotor_outer_radius-0.5*h_r;
 Rotor(1,31)=Rotor(1,7)*Rotor(1,30)/Wedge_rad_rotor/Rotor(1,30);
  Rotor(1,32)=Rotor(1,30)*cos(Rotor(1,31));
 Rotor(1,33)=Rotor(1,30)*sin(Rotor(1,31));
 Rotor(1,34)=Rotor(1,30)*cos(Rotor(1,31));
 Rotor(1,35)=-Rotor(1,30)*sin(Rotor(1,31));
 
 Rotor(1,36)=Rotor(1,24);
 Rotor(1,37)=0;
 Rotor(1,38)=Rotor(1,30);
 Rotor(1,39)=0;
 

 
 
 
 mi_addnode(Rotor(1,3),Rotor(1,4))
  mi_selectnode(Rotor(1,3),Rotor(1,4))
       mi_setgroup(2)
 mi_addnode(Rotor(1,5),Rotor(1,6))
 mi_selectnode(Rotor(1,5),Rotor(1,6))
       mi_setgroup(2)
  mi_addnode(Rotor(1,9),Rotor(1,10))
  mi_selectnode(Rotor(1,9),Rotor(1,10))
       mi_setgroup(2)
 mi_addnode(Rotor(1,11),Rotor(1,12))
 mi_selectnode(Rotor(1,11),Rotor(1,12))
       mi_setgroup(2)
  mi_addnode(Rotor(1,14),Rotor(1,15))
  mi_selectnode(Rotor(1,14),Rotor(1,15))
       mi_setgroup(2)
 mi_addnode(Rotor(1,16),Rotor(1,17))
 mi_selectnode(Rotor(1,16),Rotor(1,17))
       mi_setgroup(2)
 mi_addnode(Rotor(1,20),Rotor(1,21))
 mi_selectnode(Rotor(1,20),Rotor(1,21))
       mi_setgroup(2)
 mi_addnode(Rotor(1,22),Rotor(1,23))
 mi_selectnode(Rotor(1,22),Rotor(1,23))
       mi_setgroup(2)
       
        mi_addnode(Rotor(1,26),Rotor(1,27))
 mi_selectnode(Rotor(1,26),Rotor(1,27))
       mi_setgroup(2)
       
       mi_addnode(Rotor(1,28),Rotor(1,29))
 mi_selectnode(Rotor(1,28),Rotor(1,29))
       mi_setgroup(2)
       
         mi_addnode(Rotor(1,32),Rotor(1,33))
 mi_selectnode(Rotor(1,32),Rotor(1,33))
       mi_setgroup(2)
       
        mi_addnode(Rotor(1,34),Rotor(1,35))
 mi_selectnode(Rotor(1,34),Rotor(1,35))
       mi_setgroup(2)
       
       
       
  mi_addsegment(Rotor(1,3),Rotor(1,4),Rotor(1,9),Rotor(1,10))
  mi_selectsegment(Rotor(1,3),Rotor(1,4))
  mi_setgroup(2)
  mi_addsegment(Rotor(1,5),Rotor(1,6),Rotor(1,11),Rotor(1,12))
  mi_selectsegment(Rotor(1,5),Rotor(1,6))
  mi_setgroup(2)
  mi_addsegment(Rotor(1,9),Rotor(1,10),Rotor(1,26),Rotor(1,27))
  mi_selectsegment(Rotor(1,9),Rotor(1,10))
  mi_setgroup(2)
   mi_addsegment(Rotor(1,11),Rotor(1,12),Rotor(1,28),Rotor(1,29))
   mi_selectsegment(Rotor(1,26),Rotor(1,27))
   mi_selectsegment(Rotor(1,28),Rotor(1,29))
   mi_setgroup(2)
   mi_addsegment(Rotor(1,26),Rotor(1,27),Rotor(1,32),Rotor(1,33))
   mi_selectsegment(Rotor(1,32),Rotor(1,33))
   mi_setgroup(2)
   mi_addsegment(Rotor(1,28),Rotor(1,29),Rotor(1,34),Rotor(1,35))
   mi_selectsegment(Rotor(1,34),Rotor(1,35))
   mi_setgroup(2)
   mi_addsegment(Rotor(1,26),Rotor(1,27),Rotor(1,28),Rotor(1,29))
   mi_selectsegment(Rotor(1,36),Rotor(1,37))
   mi_setgroup(2)
   mi_addsegment(Rotor(1,32),Rotor(1,33),Rotor(1,34),Rotor(1,35))
   mi_selectsegment(Rotor(1,38),Rotor(1,39))
   mi_setgroup(2)
   mi_setgroup(2)
   
   mi_addsegment(Rotor(1,32),Rotor(1,33),Rotor(1,14),Rotor(1,15))
   mi_selectsegment(Rotor(1,14),Rotor(1,15))
   mi_setgroup(2)
   mi_addsegment(Rotor(1,34),Rotor(1,35),Rotor(1,16),Rotor(1,17))
   mi_selectsegment(Rotor(1,16),Rotor(1,17))
   mi_setgroup(2)
    mi_addarc(Rotor(1,14),Rotor(1,15),Rotor(1,16),Rotor(1,17),1,1)
   mi_selectarcsegment(Rotor(1,14),Rotor(1,15))
  mi_setgroup(2)
   mi_addarc(Rotor(1,3),Rotor(1,4),Rotor(1,20),Rotor(1,21),1,1)
   mi_selectarcsegment(Rotor(1,20),Rotor(1,21))
  mi_setgroup(2)
   mi_addarc(Rotor(1,5),Rotor(1,6),Rotor(1,22),Rotor(1,23),1,1)
   mi_selectarcsegment(Rotor(1,22),Rotor(1,23))
  mi_setgroup(2)
 
  
  mi_selectgroup(2)
 Angle2=((b_r+b_tr)/Rotor_outer_radius)*180/pi ;
 Rotor_Slots=raw{16,3};
   mi_copyrotate(0,0,Angle2,Rotor_Slots)
   
   
   %    Rotor labels

Rlabel_inner=(Rotor_outer_radius-h_w-0.35*h_r);
Rlabel_outer=(Rotor_outer_radius-h_w-0.75*h_r);
Rslot_pitch_labeli=(b_r+b_tr)*Rlabel_inner/Rotor_outer_radius;
Rslot_pitch_labelo=(b_r+b_tr)*Rlabel_outer/Rotor_outer_radius;
RlabelI(1,1)=0;
RlabelI(1,2)=Rlabel_inner;
RlabelI(1,3)=0;
RlabelI(2,1)=0;
RlabelI(2,2)=Rlabel_outer;
RlabelI(2,3)=0;

RlabelI(3,1)=(Rslot_pitch_labeli/Rlabel_inner);
RlabelI(3,2)=Rlabel_inner*cos(RlabelI(3,1));
RlabelI(3,3)=Rlabel_inner*sin(RlabelI(3,1));

RlabelI(4,1)=(Rslot_pitch_labelo/Rlabel_outer);
RlabelI(4,2)=Rlabel_outer*cos(RlabelI(4,1));
RlabelI(4,3)=Rlabel_outer*sin(RlabelI(4,1));

l=4;
while l<4*Rotor_Slots+1
    if rem(l,2)<=0
    RlabelI(l+1,1)=RlabelI(l,1)+Rslot_pitch_labeli/Rlabel_inner;
    RlabelI(l+1,2)=Rlabel_inner*cos(RlabelI(l+1,1));
    RlabelI(l+1,3)=Rlabel_inner*sin(RlabelI(l+1,1));
    else
    RlabelI(l+1,1)=RlabelI(l-2,1)+Rslot_pitch_labelo/Rlabel_outer;
    RlabelI(l+1,2)=Rlabel_outer*cos(RlabelI(l+1,1));
    RlabelI(l+1,3)=Rlabel_outer*sin(RlabelI(l+1,1));
    end
    l=l+1;
end
for m=1: 4*Rotor_Slots+1
mi_addblocklabel(RlabelI(m,2),RlabelI(m,3));
 mi_setgroup(2);
end
 
 n=1;
  while n<4*Rotor_Slots/2+1
  mi_selectlabel(RlabelI(n+1,2),RlabelI(n+1,3)) ;
  mi_selectlabel(RlabelI(n+3,2),RlabelI(n+3,3))
 mi_selectlabel(RlabelI(n+4,2),RlabelI(n+4,3));
 mi_selectlabel(RlabelI(n+5,2),RlabelI(n+5,3));
 mi_selectlabel(RlabelI(n+6,2),RlabelI(n+6,3));
 mi_selectlabel(RlabelI(n+7,2),RlabelI(n+7,3));
 mi_selectlabel(RlabelI(n+8,2),RlabelI(n+8,3));
 mi_selectlabel(RlabelI(n+10,2),RlabelI(n+10,3));
  if n<8 | n>48 & n<56 | n>96 & n<104| n>144 & n<152
  mi_setblockprop('Rotor', 0, 0, 'A1+', 0,2000,N_rt);
  mi_setgroup(200*n);
    elseif n>8 & n<16 | n>56 & n<64 |n>104 & n<112| n>152 & n<160
       mi_setblockprop('Rotor', 0, 0, 'C1-', 0,4000,N_rt);
       mi_setgroup(200*n);
   elseif n>16 & n<24 | n>64 & n<72|n>112 & n<120| n>160 & n<168
 mi_setblockprop('Rotor', 0, 0, 'B1+', 0,6000,N_rt);
 mi_setgroup(200*n);
  elseif n>24 & n<32 | n>72 & n<80 |n>120 & n<128
  mi_setblockprop('Rotor', 0, 0, 'A1-', 0,8000,N_rt);
  mi_setgroup(200*n);
   elseif n>32 & n<40 | n>80 & n<88 |n>128 & n<136|n>156 & n<164
 mi_setblockprop('Rotor', 0, 0, 'C1+', 0,10000,N_rt);
 mi_setgroup(200*n);
  elseif n>40 & n<48 | n>88 && n<96 |n>136 & n<144
 mi_setblockprop('Rotor', 0, 0, 'B1-', 0,12000,N_rt);
 mi_setgroup(200*n);
   end
  n=n+8;
 
end
Outer_radius=Stator_inner_rad+h_s+h_ys;
Shaft_radius=Rotor_outer_radius-h_r-h_yr;
mi_addnode(Shaft_radius,0);
Iron_rad=Shaft_radius+0.5*h_yr;
Iron_rad_stator=Outer_radius-0.5*h_ys;
Air_gap=Stator_inner_rad-0.5*g;
mi_addblocklabel(Iron_rad*cosd(45),Iron_rad*sind(45));
mi_selectlabel(Iron_rad*cosd(45),Iron_rad*sind(45)) ;
 mi_setblockprop('Iron', 0, 0, 'None', 0,10000,0);
mi_addblocklabel(Iron_rad_stator*cosd(45),Iron_rad_stator*sind(45));
mi_selectlabel(Iron_rad_stator*cosd(45),Iron_rad_stator*sind(45)) ;
 mi_setblockprop('Iron', 0, 0, 'None', 0,10001,0);
mi_addnode(-Shaft_radius,0); 
mi_addnode(Outer_radius,0);
mi_addnode(-Outer_radius,0); 
mi_addarc(-Shaft_radius,0,Shaft_radius,0,180,1)
mi_selectarcsegment(-Shaft_radius,0)
mi_setarcsegmentprop(1,'A=0',0,10000)
mi_addarc(Shaft_radius,0,-Shaft_radius,0,180,1)
mi_selectarcsegment(0,Shaft_radius)
mi_setarcsegmentprop(1,'A=0',0,10000)

mi_addarc(-Outer_radius,0,Outer_radius,0,180,1)
mi_selectarcsegment(0,-Outer_radius)
mi_setarcsegmentprop(1,'A=0',0,10000)
mi_addarc(Outer_radius,0,-Outer_radius,0,180,1)
mi_selectarcsegment(0,Outer_radius)
mi_setarcsegmentprop(1,'A=0',0,10000)
%  
mi_addblocklabel(Air_gap*cosd(45),Air_gap*sind(45));
mi_selectlabel(Air_gap*cosd(45),Air_gap*sind(45)) ;
 mi_setblockprop('Air', 0, 0, 'None', 0,10002,0)
 
 mi_addblocklabel(0,0);
mi_selectlabel(0,0) ;
 mi_setblockprop('Air', 0, 0, 'None', 0,10002,0)
 
 mi_saveas('DFIG_5MW.fem');
 % verify mesh and material properties before analysis
%  mi_analyze
%  mi_loadsolution
 

