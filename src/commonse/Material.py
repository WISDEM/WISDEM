#-------------------------------------------------------------------------------
# Name:        Material.py
# Purpose: This module contains the material class definition
#
# Author:      rdamiani
#
# Created:     04/11/2013
# Copyright:   (c) rdamiani 2013
# Licence:     APache (2014)
#-------------------------------------------------------------------------------

def main():
    """Material Class"""
    #The main fct is written to test the class
    mat=Material(E=3.5e5,rho=8500.)
    mat1=Material(matname='grout')
    print('Example Returning mat and mat1 as 2 objects')
    return mat,mat1


class Material: #isotropic for the time being
    def __init__(self,**kwargs):

        prms={'matname':'ASTM992_steel','E':2.1e11,'nu':0.33, 'rho':8502., 'fy':345.e6, 'fyc':345.e6, 'fp':0.0, 'relx':0.0} #SI Units
        prms.update(kwargs) #update in case user put some new params in
        for key in prms:  #Initialize material object with these parameters, possibly updated by user' stuff
            setattr(self,key,prms[key])

        #Fill in the rest of the fields user may have skipped

        #Predefined materials
        steel={'matname':'ASTM992_steel',          'E':2.1e11,'nu':0.33,'G':7.895e10,'rho':7805., 'fy':345.e6, 'fyc':345.e6,} #SI Units
        heavysteel={'matname':'ASTM992_steelheavy','E':2.1e11,'nu':0.33,'G':7.895e10,'rho':8741., 'fy':345.e6, 'fyc':345.e6} #SI Units
        grout={'matname':'Grout',                  'E':3.9e10,'nu':0.33,'G':1.466e10,'rho':2500., 'fy':20.68e6,'fyc':20.68e6} #SI Units  TO REVISE FOR GROUT
        concrete={'matname':'Concrete',            'E':24.8e9,'nu':0.18,             'rho':2300,  'fy':0.,     'fyc':27.6e6} #SI Units
        rebar={'matname':'Reinforcement',          'E':200e9, 'nu':0.33,             'rho':7805,  'fy':276.e6, 'fyc':276.e6} #SI Units
        strand={'matname':"Strand",                'E':195e9, 'nu':0.33,             'rho':7805,  'fy':1690.e6, 'fyc':1690.e6, 'fp':1930.e6, 'relx':67} #SI Units;fp=ultimate stress; relx is for Roo, relaxation calculation

        if ((prms['matname'].lower() == 'steel') or (prms['matname'].lower() == 'astm992_steel')  or (prms['matname'].lower() == '')):
            for key in prms:
                if (not(key in kwargs) or (prms[key]==[])) and (key in steel):  #I need to operate on parameters not set by user
                    setattr(self,key,steel[key])

        elif ((prms['matname'].lower() == 'heavysteel') or (prms['matname'].lower() == 'astm992_steelheavy') or (prms['matname'].lower() == 'heavysteel')):
            for key in prms:
                if (not(key in kwargs) or (prms[key]==[])) and (key in heavysteel):  #I need to operate on parameters not set by user
                    setattr(self,key,heavysteel[key])

        elif ( (prms['matname'].lower() == 'grout')):
            for key in prms:
                if (not(key in kwargs) or (prms[key]==[])) and (key in grout):
                    setattr(self,key,grout[key])

        elif prms['matname'].lower() == 'concrete':
            for key in prms:
                if (not(key in kwargs) or (prms[key]==[])) and (key in concrete):
                    setattr(self,key,concrete[key])

        elif ((prms['matname'].lower() == 'rebar') or (prms['matname'].lower() == 'reinforcement')):
            for key in prms:
                if (not(key in kwargs) or (prms[key]==[])) and (key in rebar): #I need to operate on parameters not set by user
                    setattr(self,key,rebar[key])

        elif ((prms['matname'].lower() == 'strand') or (prms['matname'].lower() == 'tendon')):
            for key in prms:
                if (not(key in kwargs) or (prms[key]==[])) and (key in strand):  #I need to operate on parameters not set by user
                    setattr(self,key,strand[key])

        if not(hasattr(self,'G')) or not(self.G):
            self.G=self.E/(2.*(1.+self.nu))  #isotropic
        else: #if G is given then recalculate nu
            self.nu=self.E/(2.*self.G)-1.

if __name__ == '__main__':
    mat,mat1=main()
