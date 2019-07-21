#-------------------------------------------------------------------------------
# Name:        EmbedLength.py
# Purpose:     This module calculates the necessary embedment length for a pile
#              given soil properties and pile head axial force
#
# Author:      rdamiani
#
# Created:     26/01/2014 - Based entirely on BuildMPtwr.py subroutine under PYTHON/MONOPILE
# Copyright:   (c) rdamiani 2014
# Licence:     <Apache licence>
#-------------------------------------------------------------------------------

import numpy as np
import scipy.integrate, warnings
from constants import gravity

#______________________________________________________________________________#
def EmbedLength(Dpile,tpile,rho,Nhead,soil,gravity=gravity):
      """This function calculates the embedment length for a pile, solely based on \n
        soil friction and toe capacity, no lateral stability. It assumes \n
        both internal and external surfaces available, no plug. \n
        It also does a simple minimization of another function. \n
        INPUT     \n
        fsoil       -float(n), friction coeff. values at z below mudline  \n
        Dpile       -float, pile OD [m]                         \n
        tpile       -float, pile thickness [m]                  \n
        Nhead       -float, axial force [N] at the head of the pile, mudline, ABS value, you may use also the max tensile overall if more conservative, it will be thought as pushing down   \n
        soil        -object of class SoilC
        gravity     -optional, m/s^2 vertical acceleration of gravity (absolute value)
        """

        #From API here is a lookup table
      delta_tab=np.array([15.,20.,25.,30.,35.])  # Soil-pile friction angles
      Nq_tab=np.array([8.,12.,20.,40.,50.])     # Nq values
      f_tab=np.array([47.8,67.,81.3,95.7,114.8])*1.e3     # skin friction limiting values [Pa]
      q_tab=np.array([1.9,2.9,4.8,9.6,12.])*1.e6     # end bearing limiting values [Pa]

      def cu(z,soilzs,cusoil):
        """Function for UNdrained SHear Strength at various levels:\n
        INPUT \n
        z           -float(m), m coordinates below mudline (<0) [m]
        soilzs      -float(n), soil layer bottom coordinates [m]
        cusoil -float(n), Undrained shear strength of layers [N/m2]

        """
        cus=scipy.interp(z,soilzs,cusoil)
        return cus

      def p0(z,soilzs,gammas_soil):
        """Function for UNdrained SHear Strength at various levels:\n
        INPUT \n
        z           -float(m), m coordinates below mudline (<0) [m]
        soilzs      -float(n), soil layer bottom coordinates [m]
        cusoil -float(n), Undrained shear strength of layers [N/m2]


        """
        z=np.asarray(z).reshape([1,-1])  #make sure it is an array

        Dzs=np.hstack((-soilzs[0],(soilzs-np.roll(soilzs,-1))[:-1]))
        weights=gammas_soil*Dzs

        p00=np.zeros(z.size)  #initialize
        for ii in np.arange(0,z.size):
            idx=np.nonzero(soilzs>z[ii])[0] #layers above selected z
            if idx.size:
                p00[ii]=weights[idx].sum()+(soilzs[idx[-1]]-z[ii])*gammas_soil[idx[-1]]

        #Check if within the first layer
        idx=np.nonzero(z>soilzs[0])[0]
        p00[idx]=-z[idx]*gammas_soil[0]

        return p00

      def fsoil(z,delta_soil,SF_pile_ex=soil.SoilSF,sndflg=True):
        """INPUT \n
        z         -float(n),  negative z as in below seabed depth at various n levels [m]
        deltasoil   -float, friction angle between pile and soil [deg] \n

        sndflg  -boolean, True if sand; False if clay
        SF_pile_ex  -float, pile capacity safety factor (API RP2A WSD is 1.5-2, old LFRD 1.25) \n
        """
        z=np.asarray(z).reshape([1,-1])
        if sndflg:
            #K_API   -float, Coefficient of lateral earth pressure (0.8 for unplugged, 1 for plugged piles)
            K_API=0.8 #plug=False
            if soil.plug: #plugged
                K_API=1.
            frsoil=np.min((K_API*np.tan(delta_soil*np.pi/180.)*p0(z,soil.zbots,soil.gammas),scipy.interp(soil.delta,delta_tab,f_tab)))

        else:

            p00=np.zeros(z.size) #initialize
            psi=np.zeros(z.size) #initiali
            alpha=np.zeros(z.size) #initiali

            cu0=cu(z,soil.zbots,soil.cus)

            idx=np.nonzero( z != 0)[0]
            #if not p0(z[idx],soil.zbots,soil.gammas):
            #    print "z=",z[idx]
            #if p0(z[idx],soil.zbots,soil.gammas) ==0.:
            #    print z[idx]
            psi[idx]=cu0/p0(z[idx],soil.zbots,soil.gammas)  #psi for API


            idx2=np.nonzero(psi > 1.)[0]
            idx3=np.nonzero(psi !=0.)[0]
            alpha[idx3]=np.minimum( [1.],[0.5* psi[idx3]**(-.5)])

            alpha[idx2]=np.minimum( [1.],[0.5* psi[idx2]**(-.25)])

            frsoil=alpha*cu0

        return frsoil/SF_pile_ex

        #First integrand is N=int_0^z1 (f*D*pi dz)
      def integrand01(z):
        """INPUT
                fsoil    -float(n), friction coefficient at n z levels
                Dpile    -float or (n), Dmp
                tpile    -float or (n), tmp
                plug     -Boolean, whether or not to consider the pile plugged
        """
        frict=fsoil(z,soil.delta,sndflg=soil.sndflg)
        return frict*np.pi*(Dpile+int(not(soil.plug))*(Dpile-2.*tpile))

      def bearCap(z,Atip,SF_pile_ex=soil.SoilSF,sndflg=True):
        """ Ultimate End-Bearing Capacity of pile. \n
        INPUT
            z         -float(n),  negative z as in below seabed depth at various n levels [m]\n
            Atip  -float, pile tip area [m2]  \n
            SF_pile_ex  -float, pile capacity safety factor  \n
            sndflg  -boolean, True if sand; False if clay
            """
        if sndflg:
            bearcap=Atip*np.min((p0(z,soil.zbots,soil.gammas)*scipy.interp(soil.delta,delta_tab,Nq_tab),scipy.interp(soil.delta,delta_tab,q_tab)))
        else:
            bearcap=9.*Atip*cu(z,soil.zbots,soil.cus)
        return bearcap/SF_pile_ex

      # I need to solve an integral equation
      #let us define the function that needs to be minimized to 0
      def minfun(zembd):  #function to minimize
        Atip=np.pi/4. * (Dpile**2. -(Dpile-2.*tpile)**2.)  #x-sect area [m2]
        wght_embd=-Atip*zembd * rho*gravity # weight of the embedded pile
        Qp=bearCap(zembd,Atip,sndflg=soil.sndflg)

        fun2min=scipy.integrate.romberg(integrand01,zembd,0.,divmax=20)    \
        -wght_embd-Nhead #+Qp
        return fun2min[0] #Note not using Qp to simulate pullout also

      #Solve teh equation here

      #print "p0=",p0(-0.00091553,soil.zbots,soil.gammas)
      #print "minfun(-30.)=",minfun(-30.),"fsoil=",fsoil(-30.,soil.delta,sndflg=soil.sndflg) #debug

     # Qp=bearCap(-30.,1.02,sndflg=soil.sndflg); debug
      guess=-15.  #embedment length
      res=scipy.optimize.fsolve(minfun,guess,full_output=1)
      if res[-2] != 1:
            warnings.warn('Embedment Length not found')
      return res[0][0] #zembd
#______________________________________________________________________________#

if __name__ == '__main__':
    main()
