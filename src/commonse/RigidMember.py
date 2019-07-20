#-------------------------------------------------------------------------------
# Name:        RigidMember.Py
# Purpose:     Rigid Member Class
#
# Author:      rdamiani
#
# Created:     12/07/2013
# Copyright:   (c) rdamiani 2013
# Licence:     <your licence>
#-------------------------------------------------------------------------------
from commonse.Material import Material

def main():
    pass

if __name__ == '__main__':
    main()

class RigidMember:

    def __init__(self,**kwargs):
        """Rigid member structural virtual properties for FRAME3DD (very high stiffenss\n
        very low density) to simulate TP struts and to connect RNA to top of tower."""
        prms={'name':'~0-weight_rigid','E':10**15,'nu':0.33,'rho':10., 'fy':10**20,'fyc':10**20,'D':1.,'t':0.01,'L':1.,'Kbuck':1.} #SI Units
        prms.update(kwargs) #update in case user put some new params in
        from sys import float_info as CPU
        self.Area=self.Asy=self.Asx=1. #Fix area and shear area to 1 m^2 and then we will increase E and G by 10 OoMs
        self.Jxx=self.Jyy=1. #Again arbitrary set to 1
        self.J0=self.Jyy+self.Jxx
        self.mat=Material(name=prms['name'],E=prms['E'],rho=prms['rho'],nu=prms['nu'],fy=prms['fy'],fyc=prms['fyc']) #rho=CPU.epsilon, E=10**20
        #Auxiliary D and t, L used just as placeholders
        self.D=prms['D']
        self.t=prms['t']
        self.L=prms['L']
        self.Kbuck=prms['Kbuck']