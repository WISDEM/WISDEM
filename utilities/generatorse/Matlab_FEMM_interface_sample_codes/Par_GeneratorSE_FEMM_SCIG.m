clear all
close all
clc

addpath('c:\femm42\examples');
openfemm;
newdocument(0)

disp('GeneratorSE Validation for SCIG by FEMM');
disp('Latha Sethuraman and Katherine Dykes')
disp('Copyright (c) NREL. All rights reserved.')
disp(' For queries,please contact : Latha Sethuraman@nrel.gov or Katherine.Dykes @nrel.gov')
disp(' ');


[Parameters,txt,raw] =xlsread('C:\GeneratorSE\src\generatorse\SCIG_5.0_MW.xlsx');% Specify the path of the GeneratorSE output excel files
depth=raw{5,3};
mi_probdef(0, 'meters', 'planar', 1e-8,depth,30);
mi_getmaterial('Pure Iron') 
mi_modifymaterial('Pure Iron',0,'Iron')
mi_addmaterial('Steel',5000,5000,0,0,0,0,0,0,0,0)
mi_addmaterial('Air', 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0);
mi_getmaterial('20 SWG') %fetches the material specified by materialname from
mi_modifymaterial('20 SWG',0,'Stator')
mi_getmaterial('Copper') 
mi_modifymaterial('Copper',0,'Bar_Cu')


mi_addboundprop('A=0', 0, 0, 0, 0, 0, 0, 0, 0, 0)
mi_addboundprop('apbc1', 0, 0, 0, 0, 0, 0, 0, 0, 5)
mi_addboundprop('apbc2', 0, 0, 0, 0, 0, 0, 0, 0, 5)
mi_addboundprop('apbc3', 0, 0, 0, 0, 0, 0, 0, 0, 5)
mi_addboundprop('apbc4', 0, 0, 0, 0, 0, 0, 0, 0, 5)
mi_addboundprop('apbc5', 0, 0, 0, 0, 0, 0, 0, 0, 5)
mi_addboundprop('apbc6', 0, 0, 0, 0, 0, 0, 0, 0, 5)
mi_addboundprop('apbc7', 0, 0, 0, 0, 0, 0, 0, 0, 5)

 

 tau_p=raw{8,3}/1000;
 h_s=raw{10,3}/1000;
 b_s=raw{11,3}/1000;
 h_w=0.005;
 b_so =0.004;
 b_t=raw{13,3}/1000;
 h_ys=raw{14,3}/1000;
 b_ro=0.004;
 h_r=raw{16,3}/1000;
 b_r=raw{17,3}/1000;
 b_tr=raw{18,3}/1000;
 h_yr=raw{19,3}/1000;
 Slots=raw{9,3};
 N_st=2; %turns per coil
 A_s=raw{33,3};
%  N_st=raw{31,3};
 A_r=raw{39,3};
 N_rslots=raw{15,3};
 N_rt=1;
 N_s=round(sqrt(A_s*4/pi)/0.9144);
 N_r=round(sqrt(A_r*4/pi)/0.9144);
 mi_modifymaterial('Stator',12,N_s)
mi_modifymaterial('Rotor',9,4)
mi_modifymaterial('Rotor',12,N_r)
 
 
 Stator_Current=raw{38,3};
 Field_Current=raw{37,3}*0;
 % Circuit properties
 mi_addcircprop('A+', Stator_Current, 1)
mi_addcircprop('A-', -Stator_Current, 1)
mi_addcircprop('B+', Stator_Current*sind(120), 1)
mi_addcircprop('B-', -Stator_Current*sind(120),1)
mi_addcircprop('C+', Stator_Current*sind(240), 1)
mi_addcircprop('C-', -Stator_Current*sind(240), 1)
  I=Field_Current;



  
g=(0.1+0.012*(raw{2,3}*1e6)^(1/3))*0.001;
 Rotor_outer_radius=0.5*raw{4,3}-g;
 Stator_inner_rad=raw{4,3}*0.5;
 Wedge_rad=Stator_inner_rad+h_w;
 Stator_yoke_irad=Wedge_rad+h_s-h_w;
 Stator(1,1)=b_so*0.5;
 Stator(1,2)=Stator(1,1)./Stator_inner_rad;
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
 
  tau_s=tau_p/18;
  mi_selectgroup(1)
 Angle=(b_s+b_t)/Stator_inner_rad*180/pi ;
 
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
a=0;
b=0;
c=0;
d=0;
e=0;
f=0;
k=1;
while k<4*Slots-68
  mi_selectlabel(SlabelI(k+1,2),SlabelI(k+1,3)) ;
  mi_selectlabel(SlabelI(k+3,2),SlabelI(k+3,3));
 mi_selectlabel(SlabelI(k+5,2),SlabelI(k+5,3));
  mi_selectlabel(SlabelI(k+6,2),SlabelI(k+6,3)) ;
 mi_selectlabel(SlabelI(k+8,2),SlabelI(k+8,3)) ;
 mi_selectlabel(SlabelI(k+10,2),SlabelI(k+10,3)) ;
  mi_selectlabel(SlabelI(k+12,2),SlabelI(k+12,3))
 mi_selectlabel(SlabelI(k+14,2),SlabelI(k+14,3));
 mi_selectlabel(SlabelI(k+16,2),SlabelI(k+16,3));
mi_selectlabel(SlabelI(k+67,2),SlabelI(k+67,3));
 mi_selectlabel(SlabelI(k+69,2),SlabelI(k+69,3));
 mi_selectlabel(SlabelI(k+71,2),SlabelI(k+71,3));

 if k<7 | k>67 & k<79 | k>139 & k<151| k>211 & k<223
     
 mi_setblockprop('Stator', 0, 0, 'A+', 0,200*k,N_st);
  mi_setgroup(200*k);
  a=a+1;
   elseif k>=7 & k<19| k>79 & k<91| k>151 & k<163| k>223 & k<235
      mi_setblockprop('Stator', 0, 0, 'C-', 0,200*k+1,N_st);
       mi_setgroup(200*k+1);
        b=b+1;
  elseif k>=19 & k<31| k>91 & k<103| k>163 & k<175| k>235 & k<247
 mi_setblockprop('Stator', 0, 0, 'B+', 0,200*k+2,N_st);
  mi_setgroup(200*k+2);
  c=c+1;
 elseif k>=31 & k<43| k>103 & k<115| k>175 & k<187| k>247 & k<259
 mi_setblockprop('Stator', 0, 0, 'A-', 0,200*k+3,N_st);
  mi_setgroup(200*k+3);
  d=d+1;
  elseif k>=43 & k<55| k>115 & k<127| k>187 & k<199| k>259 & k<271
 mi_setblockprop('Stator', 0, 0, 'C+', 0,200*k+4,N_st);
  mi_setgroup(200*k+4);
e=e+1;
  elseif k>=55 & k<67| k>127 & k<139| k>199 & k<211| k>271 & k<283
 mi_setblockprop('Stator', 0, 0, 'B-', 0,200*k+5,N_st);
  mi_setgroup(200*k+5);
  f=f+1;
  end
  k=k+12;
 
end

Outer_radius=Stator_inner_rad+h_s+h_ys;
Shaft_radius=Rotor_outer_radius-h_r-h_yr;
mi_addnode(Shaft_radius,0);
Iron_rad=Shaft_radius+0.5*h_yr;
Iron_rad_stator=Outer_radius-0.5*h_ys;
mi_addblocklabel(Iron_rad*cosd(45),Iron_rad*sind(45));
mi_addblocklabel(Iron_rad_stator*cosd(45),Iron_rad_stator*sind(45));
%  mi_selectlabel(Iron_rad_stator*cosd(45),Iron_rad_stator*sind(45)) ;
% mi_selectlabel(Iron_rad*cosd(45),Iron_rad*sind(45)) ;
%  mi_setblockprop('Iron', 0, 0, 'None', 0,10000,0);

Air_gap=Stator_inner_rad-0.5*g;
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
mi_addarc(Outer_radius,0,-Outer_radius,0,180,1)
mi_setarcsegmentprop(1,'A=0',0,10000)

 mi_addblocklabel(Air_gap*cosd(45),Air_gap*sind(45));
mi_addblocklabel(0,0);


 

 
 
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
 


 mi_seteditmode('nodes') 
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
       
            
      
       
%          mi_addnode(Rotor(1,32),Rotor(1,33))
%  mi_selectnode(Rotor(1,32),Rotor(1,33))
%        mi_setgroup(2)
%        
%         mi_addnode(Rotor(1,34),Rotor(1,35))
%  mi_selectnode(Rotor(1,34),Rotor(1,35))
%        mi_setgroup(2)
       
      mi_seteditmode('segments') 
       
  mi_addsegment(Rotor(1,3),Rotor(1,4),Rotor(1,9),Rotor(1,10))
  mi_selectsegment(Rotor(1,3),Rotor(1,4))
  mi_setgroup(2)
  mi_addsegment(Rotor(1,5),Rotor(1,6),Rotor(1,11),Rotor(1,12))
  mi_selectsegment(Rotor(1,5),Rotor(1,6))
  mi_setgroup(2)
  mi_addsegment(Rotor(1,9),Rotor(1,10),Rotor(14),Rotor(1,15))
  mi_selectsegment(Rotor(1,14),Rotor(1,15))
  mi_setgroup(2)
   mi_addsegment(Rotor(1,11),Rotor(1,12),Rotor(1,16),Rotor(1,17))
   mi_selectsegment(Rotor(1,16),Rotor(1,17))
   mi_setgroup(2)
    mi_addsegment(Rotor(1,9),Rotor(1,10),Rotor(1,11),Rotor(1,12))
    mi_selectsegment(Rotor(1,36),Rotor(1,37))
   mi_setgroup(2)
%    mi_addsegment(Rotor(1,28),Rotor(1,29),Rotor(1,34),Rotor(1,35))
%    mi_selectsegment(Rotor(1,34),Rotor(1,35))
%    mi_setgroup(2)
%    mi_addsegment(Rotor(1,26),Rotor(1,27),Rotor(1,28),Rotor(1,29))
%    mi_selectsegment(Rotor(1,36),Rotor(1,37))
%    mi_setgroup(2)
%    mi_addsegment(Rotor(1,32),Rotor(1,33),Rotor(1,34),Rotor(1,35))
%    mi_selectsegment(Rotor(1,38),Rotor(1,39))
   
 
   mi_seteditmode('arcsegments')
   mi_addarc(Rotor(1,14),Rotor(1,15),Rotor(1,16),Rotor(1,17),1,1)
   mi_selectarcsegment(Rotor(1,14),Rotor(1,15))
  mi_setgroup(2)
   mi_addarc(Rotor(1,3),Rotor(1,4),Rotor(1,20),Rotor(1,21),1,1)
   mi_selectarcsegment(Rotor(1,20),Rotor(1,21))
  mi_setgroup(2)
   mi_addarc(Rotor(1,5),Rotor(1,6),Rotor(1,22),Rotor(1,23),1,1)
   mi_selectarcsegment(Rotor(1,22),Rotor(1,23))
  mi_setgroup(2)
 
  
 
 Angle2=((b_r+b_tr)/Rotor_outer_radius)*180/pi ;
 Rotor_Slots=raw{15,3};
  mi_selectgroup(2)
   mi_copyrotate(0,0,Angle2,Rotor_Slots-1)
   
  %    Rotor labels

Rlabel=(Rotor_outer_radius-0.5*h_r);
Rslot_pitch_labeli=(b_r+b_tr)*Rlabel/Rotor_outer_radius;
RlabelI(1,1)=0;
RlabelI(1,2)=Rlabel;
RlabelI(1,3)=0;
RlabelI(2,1)=(Rslot_pitch_labeli/Rlabel);
RlabelI(2,2)=Rlabel*cos(RlabelI(2,1));
RlabelI(2,3)=Rlabel*sin(RlabelI(2,1));



 

l=2;
while l<4*Rotor_Slots+1
   
    RlabelI(l+1,1)=RlabelI(l,1)+Rslot_pitch_labeli/Rlabel;
    RlabelI(l+1,2)=Rlabel*cos(RlabelI(l+1,1));
    RlabelI(l+1,3)=Rlabel*sin(RlabelI(l+1,1));
    l=l+1;
end
mi_selectlabel(Air_gap*cosd(45),Air_gap*sind(45)) ;
mi_selectlabel(0,0);
 mi_setblockprop('Air', 0, 0, 'None', 0,10002,0)  

 for m=1: Rotor_Slots+1
 mi_addblocklabel(RlabelI(m,2),RlabelI(m,3));
  mi_setgroup(2);
 end
n=1;
   while n<Rotor_Slots+1
   mi_selectlabel(RlabelI(n,2),RlabelI(n,3)) ;
    mi_setblockprop('Bar_Cu',0, 0, 'None', 0,5,0);
    n=n+1;
   end

 mi_selectarcsegment(Outer_radius,0)
mi_selectarcsegment(0,Outer_radius)
mi_setarcsegmentprop(1,'A=0',0,10000)


 mi_saveas('SCIG_5MW_new.fem');
 
% verify mesh and material properties before analysis
 
%  mi_analyze
%  mi_loadsolution
 
