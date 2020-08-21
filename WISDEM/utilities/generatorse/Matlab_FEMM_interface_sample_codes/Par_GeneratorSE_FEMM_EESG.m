clear all
clc
close all
addpath('c:\femm42\mfiles');

disp('(c) GeneratorSE Validation for EESG by FEMM');
disp('Latha Sethuraman and Katherine Dykes')
disp('Copyright (c) NREL. All rights reserved.')
disp(' For queries contact : Latha Sethuraman@nrel.gov')
disp(' ');

openfemm;
newdocument(0)

[Parameters,txt,raw] =xlsread('C:\SEModels\GeneratorSE\src\generatorse\EESG_5MW.xlsx'); % Specify the path of the GeneratorSE output excel files
depth=raw{18,3};
mi_probdef(0, 'meters', 'planar', 1e-8,depth,30);
mi_getmaterial('NdFeB 40 MGOe') %fetches the material specified by materialname from
mi_modifymaterial('NdFeB 40 MGOe',0,'Magnet')
mi_getmaterial('Pure Iron') 
mi_modifymaterial('Pure Iron',0,'Iron')
mi_addmaterial('Steel',5000,5000,0,0,0,0,0,0,0,0)
mi_addmaterial('Air', 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0);
mi_getmaterial('20 SWG') %fetches the material specified by materialname from
mi_modifymaterial('20 SWG',0,'DC cable')
mi_getmaterial('20 SWG')
mi_modifymaterial('20 SWG',0,'AC cable')


mi_addboundprop('A=0', 0, 0, 0, 0, 0, 0, 0, 0, 0)

 
 
 
 tau_p=raw{20,3}/1000;
 b_p=raw{28,3}/1000;
 h_ps=0.1*tau_p;
 b_pc=0.4*tau_p;
 h_pc=0.6*tau_p;
 A_s=raw{43,3};
 A_r=raw{47,3};
 N_st=raw{42,3};
 N_s=round(sqrt(A_s*4/pi)/0.9144);
 N_r=round(sqrt(A_r*4/pi)/0.9144);
 N_f=round(raw{46,3});
 mi_modifymaterial('AC cable',12,N_s)
mi_modifymaterial('DC cable',9,4)
mi_modifymaterial('DC cable',12,N_r)
 Poles=raw{35,3}*2;
 Pole=zeros(Poles,17);
 A=length(Pole);
  Current=raw{1,3}*1e6/(0.85*3*raw{37,3})*0;
 Field_Current=raw{48,3};
 % Circuit properties
 mi_addcircprop('A1+', Field_Current, 1)
mi_addcircprop('A1-', -1*Field_Current, 1)
 mi_addcircprop('A+', Current, 1)
mi_addcircprop('A-', Current, 1)
mi_addcircprop('B+', Current, 1)
mi_addcircprop('B-', Current,1)
mi_addcircprop('C+', Current, 1)
mi_addcircprop('C-', Current, 1)
 
 
 
 h_yr=raw{26,3}/1000;
 Rotor_outer_radius=raw{17,3}*0.5-0.001*raw{17,3};
 Rotor_yoke_orad=Rotor_outer_radius-h_ps-h_pc;
 Rotor_yoke_irad=Rotor_outer_radius-h_ps-h_pc-h_yr;
 tau_p_new=Rotor_yoke_orad*2*pi/Poles;
 Pole_inner_rad=Rotor_yoke_orad+h_pc;
 Pole_ratio=Pole_inner_rad/Rotor_yoke_orad;
 Pole_ratio2=Rotor_outer_radius/Pole_inner_rad;
 Pole(1,1)=b_pc/2;
 Pole(2,1)=tau_p_new;
 Pole(1,2)=Pole(1,1)/Rotor_yoke_orad;
%
 Pole(1,3)=Rotor_yoke_orad*cos(Pole(1,2));  %Angle 1

 Pole(1,4)=Rotor_yoke_orad*sin(Pole(1,2));

 Pole(1,5)=Rotor_yoke_orad*cos(Pole(1,2));  %Angle 1
 Pole(1,6)=-Rotor_yoke_orad*sin(Pole(1,2)); 
  Pole(1,7)=Pole(1,1)*Pole_ratio;
  Pole(1,8)=Pole(1,7)/Pole_inner_rad;
  Pole(1,9)= Pole_inner_rad*cos(Pole(1,8));
  Pole(1,10)= Pole_inner_rad*sin(Pole(1,8));
  Pole(1,11)= Pole_inner_rad*cos(Pole(1,8));
  Pole(1,12)= -Pole_inner_rad*sin(Pole(1,8));
  Pole(1,13)=b_p*0.5/Pole_inner_rad;
  Pole(1,14)=Pole_inner_rad*cos(Pole(1,13));
  Pole(1,15)=Pole_inner_rad*sin(Pole(1,13));
  Pole(1,16)=Pole_inner_rad*cos(Pole(1,13));
  Pole(1,17)=-Pole_inner_rad*sin(Pole(1,13));
  Pole(1,18)=Pole(1,13)/Rotor_outer_radius;
  
  Pole(1,19)=b_p*0.5*Pole_ratio2/Rotor_outer_radius;
   Pole(1,20)=Rotor_outer_radius*cos(Pole(1,19));
   Pole(1,21)=Rotor_outer_radius*sin(Pole(1,19));
   Pole(1,22)=Rotor_outer_radius*cos(Pole(1,19));
   Pole(1,23)=-Rotor_outer_radius*sin(Pole(1,19));
   Pole(1,24)=tau_p_new*0.5/Rotor_yoke_orad;
   Pole(1,25)=Rotor_yoke_orad*cos(Pole(1,24));
   Pole(1,26)=Rotor_yoke_orad*sin(Pole(1,24));
   Pole(1,27)=Rotor_yoke_orad*cos(Pole(1,24));
   Pole(1,28)=-Rotor_yoke_orad*sin(Pole(1,24));
   Pole(1,29)=tau_p_new*0.5*Pole_inner_rad/Rotor_yoke_orad/Pole_inner_rad;
   Pole(1,30)=Pole_inner_rad*cos(Pole(1,29));
   Pole(1,31)=Pole_inner_rad*sin(Pole(1,29));
   Pole(1,32)=Pole_inner_rad*cos(Pole(1,29));
   Pole(1,33)=-Pole_inner_rad*sin(Pole(1,29));
   Pole(1,34)=Rotor_outer_radius;
   Pole(1,35)=0;
   
   tau_p_new2=(Rotor_yoke_orad+h_pc*0.5)*2*pi/Poles;
   y=0.5*b_p-0.5*b_pc;
   x=0.5*b_pc+0.5*y;
   Label(1,1)=(b_p*0.5)/(Rotor_yoke_orad+h_pc*0.5);% labels
   Label(2,1)=(Label(1,1))+(tau_p_new2-b_p)/(Rotor_yoke_orad+h_pc*0.5);% labels
   Label(1,2)=(Rotor_yoke_orad+h_pc*0.5)*cos(Label(1,1));%;
   Label(2,2)=(Rotor_yoke_orad+h_pc*0.5)*cos(Label(2,1));%;
   Label(1,3)=(Rotor_yoke_orad+h_pc*0.5)*sin(Label(1,1));
   Label(2,3)=(Rotor_yoke_orad+h_pc*0.5)*sin(Label(2,1));
   
   
    for i=1:1
        mi_addnode(Pole(i,34),Pole(i,35))
        mi_selectnode(Pole(i,34),Pole(i,35))
       mi_setgroup(1)
       mi_addnode(Pole(i,3),Pole(i,4))
       mi_selectnode(Pole(i,3),Pole(i,4))
       mi_setgroup(1)
       mi_addnode(Pole(i,5),Pole(i,6))
       mi_selectnode(Pole(i,5),Pole(i,6))
       mi_setgroup(1)
        mi_addnode(Pole(i,9),Pole(i,10))
        mi_selectnode(Pole(i,9),Pole(i,10))
        mi_setgroup(1)
         mi_addnode(Pole(i,11),Pole(i,12))
         mi_selectnode(Pole(i,11),Pole(i,12))
         mi_setgroup(1)
         mi_addnode(Pole(i,14),Pole(i,15))
         mi_selectnode(Pole(i,14),Pole(i,15))
         mi_setgroup(1)
         mi_addnode(Pole(i,16),Pole(i,17))
         mi_selectnode(Pole(i,16),Pole(i,17))
         mi_setgroup(1)
         mi_addnode(Pole(i,20),Pole(i,21))
         mi_selectnode(Pole(i,20),Pole(i,21))
         mi_setgroup(1)
         mi_addnode(Pole(i,22),Pole(i,23))
         mi_selectnode(Pole(i,22),Pole(i,23))
         mi_setgroup(1)
         mi_addnode(Pole(i,25),Pole(i,26))
         mi_addnode(Pole(i,27),Pole(i,28))
         mi_addnode(Pole(i,30),Pole(i,31))
         mi_selectnode(Pole(i,30),Pole(i,31))
         mi_setgroup(1)
         mi_addnode(Pole(i,32),Pole(i,33))
         mi_selectnode(Pole(i,32),Pole(i,33))
         mi_setgroup(1)
   end
   
     for i=1:1
   mi_addsegment(Pole(i,3),Pole(i,4),Pole(i,9),Pole(i,10))
   
   
   mi_addsegment(Pole(i,5),Pole(i,6),Pole(i,11),Pole(i,12))
   mi_addsegment(Pole(i,9),Pole(i,10),Pole(i,14),Pole(i,15))
   mi_addsegment(Pole(i,11),Pole(i,12),Pole(i,16),Pole(i,17))
    mi_addsegment(Pole(i,16),Pole(i,17),Pole(i,22),Pole(i,23))
    mi_addsegment(Pole(i,14),Pole(i,15),Pole(i,20),Pole(i,21))
    mi_addsegment(Pole(i,30),Pole(i,31),Pole(i,25),Pole(i,26))
    mi_addsegment(Pole(i,32),Pole(i,33),Pole(i,27),Pole(i,28))

    mi_selectsegment(Pole(i,3),Pole(i,4))
    mi_selectsegment(Pole(i,5),Pole(i,6))
    mi_selectsegment(Pole(i,14),Pole(i,15))
    mi_selectsegment(Pole(i,16),Pole(i,17))
    mi_selectsegment(Pole(i,20),Pole(i,21))
    mi_selectsegment(Pole(i,22),Pole(i,23))
    mi_selectsegment(Pole(i,30),Pole(i,31))
    mi_selectsegment(Pole(i,32),Pole(i,33))
    mi_setgroup(1)
    
    mi_addarc(Pole(i,30),Pole(i,31),Pole(i,14),Pole(i,15),5,1)
    mi_addarc(Pole(i,16),Pole(i,17),Pole(i,32),Pole(i,33),5,1)
    mi_selectarcsegment(Pole(i,30),Pole(i,31))
     mi_setgroup(1)
     mi_selectarcsegment(Pole(i,16),Pole(i,17))
     mi_setgroup(1)
   mi_addarc(Pole(i,22),Pole(i,23),Pole(i,34),Pole(i,35),1,1)
   mi_selectarcsegment(Pole(i,22),Pole(i,23))
   mi_setgroup(1)
    mi_addarc(Pole(i,34),Pole(i,35),Pole(i,20),Pole(i,21),1,1)
   mi_selectarcsegment(Pole(i,20),Pole(i,21))
   mi_setgroup(1)


   mi_addarc(Pole(i,25),Pole(i,26),Pole(i,3),Pole(i,4),1,1)
   mi_selectarcsegment(Pole(i,25),Pole(i,26))
   mi_setgroup(1)
   mi_addarc(Pole(i,27),Pole(i,28),Pole(i,5),Pole(i,6),1,1)
   mi_selectarcsegment(Pole(i,27),Pole(i,28))
   mi_setgroup(1)
   end
% %
Angle=tau_p_new*180/pi/Rotor_yoke_orad;
   mi_selectgroup(1)
    mi_copyrotate(0,0,Angle,raw{35,3}*2-1)
    
   
   
%     mi_selectnode(Rotor_yoke_orad,0)
%     mi_deleteselectednodes
    k=2;
    while k<Poles*2+1
        if rem(k+1,2)>0
        Label(k+1,1)=(Label(k,1))+(b_p)/(Rotor_yoke_orad+h_pc*0.5);
        else
        Label(k+1,1)=(Label(k,1))+(tau_p_new2-b_p)/(Rotor_yoke_orad+h_pc*0.5);
        end
        Label(k+1,2)=(Rotor_yoke_orad+h_pc*0.5)*cos(Label(k+1,1));
        Label(k+1,3)=(Rotor_yoke_orad+h_pc*0.5)*sin(Label(k+1,1));
        k=k+1;
    end 
      



 

       

%Stator
 Stator_tooth_rad=(raw{17,3})/2;
 h_s=raw{21,3}*0.001;
 Stator_yoke_irad=Stator_tooth_rad+h_s;
 b_t=raw{24,3}*0.001;
 b_s=raw{22,3}*0.001;
 Stator_ratio=Stator_yoke_irad/Stator_tooth_rad;
 b_so=0.004;
 h_w=0.005;


 
  
 
  
  Stator_inner_rad=raw{17,3}*0.5;
 Wedge_rad=Stator_inner_rad+h_w;
 Stator_yoke_irad=Wedge_rad+h_s;
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
 Stator(1,30)=Wedge_rad+0.55*h_s;
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
       
%          mi_addnode(Stator(1,32),Stator(1,33))
%  mi_selectnode(Stator(1,32),Stator(1,33))
%        mi_setgroup(2)
%        
%         mi_addnode(Stator(1,34),Stator(1,35))
%  mi_selectnode(Stator(1,34),Stator(1,35))
%        mi_setgroup(2)
       
       
       
  mi_addsegment(Stator(1,3),Stator(1,4),Stator(1,9),Stator(1,10))
  mi_selectsegment(Stator(1,3),Stator(1,4))
  mi_setgroup(2)
  mi_addsegment(Stator(1,5),Stator(1,6),Stator(1,11),Stator(1,12))
  mi_selectsegment(Stator(1,5),Stator(1,6))
  mi_setgroup(2)
  mi_addsegment(Stator(1,9),Stator(1,10),Stator(1,32),Stator(1,33))
  mi_selectsegment(Stator(1,32),Stator(1,33))
  mi_setgroup(2)
   mi_addsegment(Stator(1,11),Stator(1,12),Stator(1,34),Stator(1,35))
   mi_selectsegment(Stator(1,34),Stator(1,35))
   
   mi_setgroup(2)
   mi_addsegment(Stator(1,9),Stator(1,10),Stator(1,11),Stator(1,12))
   mi_selectsegment(Stator(1,24),0)
   mi_setgroup(2)
     
   
   mi_addsegment(Stator(1,32),Stator(1,33),Stator(1,14),Stator(1,15))
   mi_selectsegment(Stator(1,14),Stator(1,15))
   mi_setgroup(2)
   mi_addsegment(Stator(1,34),Stator(1,35),Stator(1,16),Stator(1,17))
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
 Slots=raw{41,3};
   mi_copyrotate(0,0,Angle,Slots-1)
   
SLabel(1,1)=0;
  SLabel(1,2)=(Stator_inner_rad+0.6*h_s)*cos(SLabel(1,1));
  SLabel(1,3)=(Stator_inner_rad+0.6*h_s)*sin(SLabel(1,1));
  
  
   
  
  
   t=1;
  while t <Slots+1  % circuit labels
  SLabel(t+1,1)=SLabel(t,1)+((b_s+b_t)*(Stator_inner_rad+0.6*h_s)/Stator_inner_rad)/(Stator_inner_rad+0.6*h_s);
  SLabel(t+1,2)=(Stator_inner_rad+0.6*h_s)*cos(SLabel(t+1,1));
  SLabel(t+1,3)=(Stator_inner_rad+0.6*h_s)*sin(SLabel(t+1,1)); 
%   mi_addnode(Stator(t+1,13),Stator(t+1,14));
  
  t=t+1;
  end
  A_plus=0;
   A_minus=0;
    B_plus=0;
     B_minus=0;
    C_plus=0;
     C_minus=0;
 u=1; 
  while u <Slots+1
  mi_addblocklabel(SLabel(u,2),SLabel(u,3));
  mi_addblocklabel(SLabel(u+1,2),SLabel(u+1,3));
  mi_setgroup(4000)
 mi_selectlabel(SLabel(u,2),SLabel(u,3)) ;
mi_selectlabel(SLabel(u+1,2),SLabel(u+1,3)) ;
  mi_setblockprop('AC cable', 0, 0, 'A+', 0,4000,N_st);
  A_plus=A_plus+2;
 mi_addblocklabel(SLabel(u+2,2),SLabel(u+2,3));
  mi_setgroup(5000);
 mi_addblocklabel(SLabel(u+3,2),SLabel(u+3,3));
  mi_setgroup(5000);
 mi_selectlabel(SLabel(u+2,2),SLabel(u+2,3)) ;
mi_selectlabel(SLabel(u+3,2),SLabel(u+3,3)) ;
  mi_setblockprop('AC cable', 0, 0, 'C-', 0,5000,N_st);
  C_minus=C_minus+2;
  mi_addblocklabel(SLabel(u+4,2),SLabel(u+4,3));
  mi_setgroup(6000);
  mi_addblocklabel(SLabel(u+5,2),SLabel(u+5,3));
 mi_setgroup(6000);
mi_selectlabel(SLabel(u+4,2),SLabel(u+4,3)) ;
mi_selectlabel(SLabel(u+5,2),SLabel(u+5,3)) ;
mi_setblockprop('AC cable', 0, 0, 'B+', 0,6000,N_st);
B_plus=B_plus+2;
 mi_addblocklabel(SLabel(u+6,2),SLabel(u+6,3));
 mi_setgroup(7000);
  mi_addblocklabel(SLabel(u+7,2),SLabel(u+7,3));
  mi_setgroup(7000);
mi_selectlabel(SLabel(u+6,2),SLabel(u+6,3)) ;
mi_selectlabel(SLabel(u+7,2),SLabel(u+7,3)) ;

 mi_setblockprop('AC cable', 0, 0, 'A-', 0,7000,N_st);
 A_minus=A_minus+2;
 mi_addblocklabel(SLabel(u+8,2),SLabel(u+8,3));
   mi_setgroup(8000);
  mi_addblocklabel(SLabel(u+9,2),SLabel(u+9,3));
    mi_setgroup(8000);
 mi_selectlabel(SLabel(u+8,2),SLabel(u+8,3)) ;
mi_selectlabel(SLabel(u+9,2),SLabel(u+9,3)) ;
 
 mi_setblockprop('AC cable', 0, 0, 'C+', 0,8000,N_st);
 C_plus=C_plus+2;
 mi_addblocklabel(SLabel(u+10,2),SLabel(u+10,3));
  mi_setgroup(9000);
 mi_addblocklabel(SLabel(u+11,2),SLabel(u+11,3));
   mi_setgroup(9000);
mi_selectlabel(SLabel(u+10,2),SLabel(u+10,3)) ;
mi_selectlabel(SLabel(u+11,2),SLabel(u+11,3)) ;
 mi_setblockprop('AC cable', 0, 0, 'B-', 0,'9000',N_st);
 B_minus=B_minus+2;
 u=u+12;
  end   
  
  
  
  
  
  
  
  
  
  
  
  
  
 
 

h_ys=raw{25,3}*0.001;

Outer_dia=Stator_inner_rad+h_s+h_ys;  

l=1;
   while l<Poles*2   
        mi_addblocklabel(Label(l,2),Label(l,3))
        mi_addblocklabel(Label(l+1,2),Label(l+1,3))
         mi_selectlabel(Label(l,2),Label(l,3))
        mi_selectlabel(Label(l+1,2),Label(l+1,3))
        mi_setgroup(20000)
        mi_selectgroup(20000)
        mi_setblockprop('DC cable', 0, 0, 'A1+', 0,20000,N_f)
        l=l+4;   
   end


  
 mi_addnode(Outer_dia,0);
 mi_addnode(-Outer_dia,0);
 mi_addarc(Outer_dia,0,-Outer_dia,0,180,1); 
 mi_selectarcsegment(0,Outer_dia)
 mi_setarcsegmentprop(0.8257, 'A=0', 0, 0); 
  mi_addarc(-Outer_dia,0,Outer_dia,0,180,1); 
  mi_selectarcsegment(0,-Outer_dia)
   mi_setarcsegmentprop(0.8257, 'A=0', 0, 0); 

mi_addnode(Rotor_yoke_irad,0);
mi_addnode(-Rotor_yoke_irad,0);

mi_addarc(-Rotor_yoke_irad,0,Rotor_yoke_irad,0,180,1);
mi_selectarcsegment(0,Rotor_yoke_irad)
 mi_setarcsegmentprop(0.8257, 'A=0', 0, 0);
mi_addarc(Rotor_yoke_irad,0,-Rotor_yoke_irad,0,180,1);
mi_selectarcsegment(0,Rotor_yoke_irad)
 mi_setarcsegmentprop(0.8257, 'A=0', 0, 0);

 Iron_block=Stator_yoke_irad +raw{25,3}*0.001*0.5;
 mi_addblocklabel( Iron_block*cosd(45),Iron_block*sind(45));
 mi_setgroup(10000);
 mi_selectlabel( Iron_block*cosd(45),Iron_block*sind(45));
  
 mi_setblockprop('Iron', 0, 0, 'None', 0,10000,0);
 
RY=Rotor_yoke_irad+raw{26,3}*0.001*0.5;
mi_addblocklabel( RY*cosd(45),RY*sind(45));
mi_setgroup(10002);
  mi_selectlabel( RY*cosd(45),RY*sind(45));
  mi_setblockprop('Iron', 0, 0, 'None', 0,10002,0);





Air_gap_rad=(raw{17,3})/2-0.001*0.5*(raw{17,3});
mi_addblocklabel( Air_gap_rad*cosd(45),Air_gap_rad*sind(45));
mi_setgroup(10003);
  mi_selectlabel( Air_gap_rad*cosd(45),Air_gap_rad*sind(45));
  
mi_setblockprop('Air', 0, 0, 'None', 0,10003,0);
    
mi_saveas('5MW_EESG_new.fem')
%  mi_analyze
%  mi_loadsolution




r=3;
   while r<Poles*2+1   
        mi_addblocklabel(Label(r,2),Label(r,3))
        mi_addblocklabel(Label(r+1,2),Label(r+1,3))
         mi_selectlabel(Label(r,2),Label(r,3))
        mi_selectlabel(Label(r+1,2),Label(r+1,3))
        mi_setgroup(20001)
        mi_selectgroup(20001)
        mi_setblockprop('DC cable', 0, 0, 'A1-', 0,20001,N_f)
        r=r+4;   
   end

 mi_addblocklabel( 0,0);
mi_setgroup(10001);
  mi_selectlabel( 0,0);
  
mi_setblockprop('Air', 0, 0, 'None', 0,10001,0);




