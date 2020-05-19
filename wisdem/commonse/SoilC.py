#-------------------------------------------------------------------------------
# Name:        SoilC.py
# Purpose:     This module contains the basic Soil Class
#
# Author:      rdamiani
#
# Created:     26/01/2014 - Based on BuildMPtwr.py SoilC and ReadSoilInfo
# Copyright:   (c) rdamiani 2014
# Licence:     <Apache>
#-------------------------------------------------------------------------------
import warnings
from math import * #this is to allow the input file to have mathematical/trig formulas in their param expressions
import numpy as np
import scipy.interpolate as interpolate
#______________________________________________________________________________#
def ReadSoilInfo(SoilInfoFile):
    """This function creates a soil object, filling main params and then \n
        reads in a file containing class data prepared for Matlab processing.\n
        Thus it spits out an augmented version of the object.
         """
         #Now read soil data
    if isinstance(SoilInfoFile,str) and os.path.isfile(SoilInfoFile): #it means we need to read a file
         soil=SoilC() #This creates the object with some default values
         execfile(SoilInfoFile) #This simply includes the file where I have my input deck also available to matlab
         attr1= (attr for attr in dir(soil) if not attr.startswith('_'))
        #Update MP attributes based on input parameters

    else:
         warnings.warn('SoilInfoFile was not found !')

    return soil

#______________________________________________________________________________#
class SoilC():
            #Soil Class
    def __init__(self,  **kwargs):
        """This function creates a soil class.\n
        """
    #start by setting default values in a dictionary fashion
        Pprms={'zbots':-np.array([3.,5.,7.,15.,30.,50.]), 'gammas':np.array([10000,10000,10000,10000,10000,10000]),\
        'cus':np.array([60000,60000,60000,60000,60000,60000]), 'phis':np.array([36.,33.,26.,37.,35.,37.5]),\
        'delta':25.,'sndflg':True, 'plug':False, 'bwtable':True, 'PenderSwtch':False, 'SoilSF':2., 'qu': 200.e3}
        prms=Pprms.copy()
        prms.update(kwargs)
        for key in kwargs:
            if key not in Pprms:
                setattr(self,key,kwargs[key])  #In case user set something else not included in the Pprms
        for key in Pprms:
            #self.key=prms[key] This does not work instead,beats me
            setattr(self,key,prms[key]) #This takes care of updating values without overburdening the MP class with other stuff which belong to MP

def SubgrReact(soilobj,Lp, sndflg=True, bwtable=True):
        """This function returns the coefficient of subgrade reation ks [N/m3].\n
           For Sands, it comes form the API curves as a function of friction angles.\n
           For Clays, it is an average from Bowles (1968).\n
           Note: This assumes that an average value of ks is to be calculated across all fo the layers.\n
           Es(z)=Es0+ks*z. This is an approximation based on Pender's paper and Matlock and Reese (1956-1975).
           INPUTS:\n
           soilobj      -object of class soil.\n
           Lp           -float, length of pile under ground (embedment length) .\n
           sndflg       -boolean, True for sand, Flase for clay.\n
           bwtable      -boolean, True for below water table (always for offshore), False for above.\n"""
        #
        APIphis=np.array([28.,29.,30.,33.,36.,38.,40.,42.5,45.]) #[deg] friction angles for the API ks table
        minphi=min(APIphis)
        maxphi=max(APIphis)
        APIks=np.array([[5., 12.5, 34.375, 60.9375, 93.75, 121.875, 156.25, 181.25, 221.875], \
                         [0., 12.5, 46.875, 92.1875, 159.375, 212.5, 279.6875, 325, 378.125]]) # [lbf/in3] First row for below water table, 2nd for above- SANDS ONLY
        APIks *=271447.1610  #This converts from lbf/in3 to N/m3

        tks=APIks[int(-bwtable)+1,:]
        deltazs=np.hstack((-soilobj.zbots[0],soilobj.zbots-np.roll(soilobj.zbots,-1)))[:-1]
        if min(soilobj.zbots) <= (-Lp):
            idx=np.nonzero( soilobj.zbots <= (-Lp) )[0][0]#first index of zbots exceeding the z of the pile tip
        else:
            idx=len(deltazs)  #In this case we are assuming the soil is constant below the deepeset level known
        if sndflg:
            #f=interpolate.interp1d(APIphis,tks,kind='quadratic',bounds_error=False)  #function containing the interpolation function
            f=interpolate.UnivariateSpline(APIphis,tks,k=2)  #funct
            #idx2= (soilobj.phis > maxphi) #out of bounds points -only for basic interp1d
            #idx3= (soilobj.phis < minphi) #out of bounds points -only for basic interp1d
            f2=f(soilobj.phis)
            #f2[idx2]=tks[-1]  #replicate the value at teh upper bound for those points exceeding it -only for basic interp1d
            #f2[idx3]=tks[0]   #replicate the value at teh lower bound for those points exceeding it -only for basic interp1d
            ks= ( ((f2*deltazs)[0:idx]).sum()+f2[idx-1]*(Lp+soilobj.zbots[idx-1]) )/Lp   #weigthed average of ks [N/m3]
        else: #For clays there is no direct relationship; Bowles suggests 12,000-48,000 depending on soil capacity
            warnings.warn('Clay Coefficient of Subgrade Reaction to be given as input, watch what is getting calculated and inputted.')
            if soilobj.qu<=200.e3 :
                ks=(soilobj.qu-100.e3)/(200.e3-100.e3)*(24000.e3-12000.e3)+12000.e3   #[N/m3]
            elif 200.e3<soilobj.qu<=800.e3:
                ks=(soilobj.qu-200.e3)/(800.e3-200.e3)*(48000.e3-24000.e3)+24000.e3   #[N/m3]
            elif 800.e3<soilobj.qu:
                ks=48000.e3

        return ks/soilobj.SoilSF

def SoilPileStiffness(ks,Dp,Lp,Ep,Gp,Jxx_p,loadZ=0,nus=0.5,PenderSwtch=False,sndflg=True, H=[],M=[],batter=np.nan,psi=-45.*np.pi/180.):
    """This function returns a 6x6 stiffness matrix relative to mudline, assuming a \n
       coefficient of subgrade reation ks [N/m3] linear with depth below mudline.\n
       It uses either Pender's elastic soil medium approximation or the Matlock and Reese (1960)'s  form.\n
       Es(z)=Es0+ks*z. This is an approximation based on Pender's paper and Matlock and Reese (1956-1975).
       INPUTS:\n
       ks           -float, average soil coefficient of subgrade reaction [N/m3].\n
       Dp           -float, pile OD [m].\n
       Lp           -float, pile embedment length [m].\n
       Ep           -float, pile Young's module, [N/m2].\n
       Gp           -float, pile Shear module, [N/m2].\n
       Jxx_p        -float, pile x-section area moment of inertia, [m4].\n
       loadZ        -float, application point above ground level of H and M.
       nus          -float, Poisson's ratio for soil. Default =0.5 from Pender's 1993.
       PenderSwtch  -boolean, True for Pender's version of flexibility coefficients, False for Matlock and Reese(1960)\n
       sndflg       -boolean, True for sand, Flase for clay.\n
       H            -float, Shear at the top of the pile, positive along x: MANDATORY IF PenderSwtch=True \n
       M            -floar, Moment at the top of the pile: MANDATORY IF PenderSwtch=True \n
       batter       -float, 2D batter in the xz plane for the pile: positive batter means tip is to the left of head and H>0 is pointing to the right \n
       psi          -float, [deg] angle of the pile projection on the x,y plane, for a 4 legged jacket it is -45 deg \n"""

    #General Parameters
    LL=Lp/Dp
    EJxx_p=Ep*Jxx_p

    if PenderSwtch:
        if H or M:
            Es_D=ks*Dp
            K=Ep/Es_D
            La=1.3*Dp*K**(0.222) #active length of pile

            #if Lp>=La:  #long(flexible) pile
            fxH=3.2*K**(-.333)/(Es_D*Dp) #CxF
            fxM=ftH=5.*K**(-.556)/(Es_D*Dp**2)  #CxM=CthtF
            ftM=13.6*K**(-0.778)/(Es_D*Dp**3)   #CthtM
            L_Mmax=0.41*La #Location of maximum moment from ground level
            if sndflg:
                fxH=2.14*K**(-0.29)/(Es_D*Dp) #CxF
                fxM=ftH=3.43*K**(-0.53)/(Es_D*Dp**2)  #CxM=CthtF
                ftM=12.16*K**(-0.77)/(Es_D*Dp**3)   #CthtM


            M=M+loadZ*H #account for eccentricity
            f=M/(Dp*H)  #Eccentricity of the load at the pile head
            a=0.6*f
            b=0.17*f**(-0.3)
            IMH=np.min(np.array([8.,a*K**b]))
            Mmax=IMH*Dp*H
            if Lp<= 0.07*Dp*np.sqrt(Ep/Es_D):  #short rigid pile, in which case Es is also considered a constant
                fxH=0.7*LL**(-.33)/(Es_D*Dp) #CxF
                fxM=ftH=0.4*LL**(-0.88)/(Es_D*Dp**2)  #CxM=CthtF
                ftM=0.6*LL**(-1.67)/(Es_D*Dp**3)   #CthtM
            elif Lp<La:  #intermediate length: use 1.25 the calculated values; correct for fxH, but pushing it for the others
                fxH*=1.25 #CxF
                fxM*=1.25 #CxM
                ftH *=1.25  #CxM=CthtF
                ftM *=1.25   #CthtM
                L_Mmax *=1.25
        else:
            warnings.warn('If using Pender''s Method, H and M must be included')
            sys.exit('!!!ABORT: You must speicfy both H and M when using Pender''s method. Check Soil Inputs, in case use PenderSwtch=False.!!!')

    else:  #Use MAtlock and Reese (1960)
        T=(EJxx_p/ks)**(1./5.) #relative stiffness factor for Es=ks*z type, though it should be Es=Es0+ks*z
        fxH=2.43*T**3/EJxx_p #CxF
        fxM=ftH=1.62*T**2/EJxx_p  #CxM=CthtF
        ftM=1.75*T/EJxx_p   #CthtM

    #Invert the flexibility coefficient matrix
    den=fxH*ftM-fxM**2  #determinant of matrix
    Kmat=1./den * np.array([[ftM, -fxM] ,[-ftM , fxH]])  #This is all in a coordinate system fixed with the pile, with z along its axis
    #Add the axial stiffness now from Pender, assuming linear variation of Es with depth
    E_SL=ks*Lp  #Moudulus at tip of pile
    RR=Ep/E_SL
    b=LL/RR
    Kz=1.8*E_SL*Dp*LL**0.55*RR**(-b)

    #Add torsional stiffness, approximating from Guo and Randolph as flexible pile head stiffness with linear trend of Es.
    Gpeq=32*Gp*Jxx_p/(np.pi*Dp**4) #equivalent G for pile
    Lc=Dp/16. * (2*(1+nus)*Gpeq/(ks*Dp))**(1./3) #critical torsional pile length
    Gsc=ks*Lc/(2*(1+nus)) #soil G at critical length
    Kpsipsi=np.pi/16 * np.sqrt(2) * Dp**3 *np.sqrt(Gpeq/Gsc)

    #Assemble a 6x6 matrix to be returned, with all terms positive, since we do care about abs values not actual direction of forces
    Klocal=np.zeros([6,6]) #Initialize pile head stiffness matrix, this is at the mudline
    Klocal[0,0]=Klocal[1,1]=Kmat[0,0]  #Kx=Ky
    Klocal[0,4]=Klocal[4,0]=Kmat[0,1]  #Kx_thetay=Kthetay_x :force along x due to unit rotation about y
    Klocal[2,2]=Kz
    Klocal[1,3]=Klocal[3,1]=Kmat[0,1]  #Ky_thetax=Kthetax_y :force along y due to unit rotation about x
    Klocal[3,3]=Klocal[4,4]=Kmat[1,1]  #Kthetax_thetax=Ktheta_y_thetay
    Klocal[5,5]=Kpsipsi   #torsional stiffness

    Kglobal=Klocal #initialize for vertical piles

    if np.isfinite(batter) and batter:
        al_bat3D=np.arctan(np.sqrt(2.)/batter)
        cpsi=np.cos(psi)  #cos psi
        spsi=np.sin(psi)  #sin psi
        ca3D=np.cos(al_bat3D)  #cos batter 3d angle
        sa3D=np.sin(al_bat3D)  #sin batter 3d angle

        Cl2g=np.zeros([6,6]) #Initialize Transformtation matrix from local to global
        Cl2g[0,0]=Cl2g[3,3]=cpsi*ca3D
        Cl2g[0,1]=Cl2g[3,4]=-spsi
        Cl2g[0,2]=Cl2g[3,5]=-sa3D*cpsi
        Cl2g[1,0]=Cl2g[4,3]=ca3D*spsi
        Cl2g[1,1]=Cl2g[4,4]=cpsi
        Cl2g[1,2]=Cl2g[4,5]=-spsi*sa3D
        Cl2g[2,0]=Cl2g[5,3]=sa3D
        Cl2g[2,2]=Cl2g[5,5]=ca3D

        Kglobal=np.dot(Cl2g,np.dot(Klocal,Cl2g.T))

    return Kglobal
#______________________________________________________________________________#

if __name__ == '__main__':

    sndflg=True
    soil=SoilC(sndflg=True)
    Lp=36. #embedment length of pile
    Dp=1.5 #OD of pile [m]
    tp=0.06 #thickness of pile [m]
    Ep=2.1e11 #Youngs modulus
    Gp=Ep/(1+0.3)/2. #Shear modulus of pile
    Jxx_p=np.pi/64.*(Dp**4-(Dp-2.*tp)**4)
    loadsz=1  #level abover mudline for shear&moment application
    batter=np.nan
    H=100.e3  #[N] shear
    M=150.e3  #[Nm] moment

    PenderSwtch=True

    print('Soil z-bottoms [m]:',soil.zbots,' ;\n','Undrained shear strength [N/m^2]:',soil.cus,' ;\n',\
          'Unit weight  [N/m^3]:', soil.gammas,' ;\n','Friction angles [deg]:', soil.phis,' ;\n',\
          'Pile-soil friction angle [deg]:', soil.delta)

    ksavg=SubgrReact(soil,Lp,sndflg=sndflg)

    print('Coefficient of subgrade reaction [MN/m3]=', SubgrReact(soil,Lp,sndflg=sndflg)/1.e6)

    Kglobal=SoilPileStiffness(ksavg,Dp,Lp,Ep,Gp,Jxx_p,loadZ=0,PenderSwtch=False,H=H,M=M,batter=batter)

    for ii in range(0,6):
        for jj in range(0,6):
            print(('K[{:d},{:d}] = {:10.3e}'.format(ii,jj,Kglobal[ii,jj])))
