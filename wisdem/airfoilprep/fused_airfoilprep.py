from openmdao.main.api import Component
from openmdao.main.datatypes.api import Float, VarTree, Int

from fusedwind.turbine.airfoil import ModifyAirfoilBase
from fusedwind.turbine.airfoilaero_vt import AirfoilDataVT
from fusedwind.interface import implement_base

from airfoilprep import Polar

@implement_base(ModifyAirfoilBase)
class AirfoilPreppyPolarExtrapolator(Component):
    
    # inputs
    afIn = VarTree(AirfoilDataVT(), iotype='in', desc='tabulated airfoil data')
    cdmax = Float(iotype='in', desc='maximum drag coefficient')
    AR = Float(iotype='in', desc='aspect ratio = (rotor radius / chord_75% radius)\
            if provided, cdmax is computed from AR')
    cdmin = Float(0.001, iotype='in', desc='minimum drag coefficient.  used to prevent \
                                negative values that can sometimes occur\
                                    with this extrapolation method')
    nalpha = Int(15, iotype='in', desc='number of points to add in each segment of Viterna method')
    
    # outputs
    afOut = VarTree(AirfoilDataVT(), iotype='out', desc='tabulated airfoil data')
    
    def execute(self):
        """provides a default behavior (to not modify the airfoil)"""

        self.afOut = self.afIn
        
        # create polar object
        p = Polar(self.afIn.Re,
                  self.afIn.alpha,
                  self.afIn.cl,
                  self.afIn.cd,
                  self.afIn.cm)
        
        if self.AR == 0.0:
            AR = None
        else:
            AR = self.AR
            
        # extrapolate polar
        p_extrap = p.extrapolate(self.cdmax,
                                 AR,
                                 self.cdmin,
                                 self.nalpha)
        
        self.afOut.Re = p_extrap.Re
        self.afOut.alpha = p_extrap.alpha
        self.afOut.cl = p_extrap.cl
        self.afOut.cd = p_extrap.cd
        self.afOut.cm = p_extrap.cm
        
        