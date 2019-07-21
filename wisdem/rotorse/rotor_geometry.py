import numpy as np
import os
# Python 2/3 compatibility:
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import csv
import wisdem.commonse

from openmdao.api import ExplicitComponent, Group, IndepVarComp, ExecComp

from wisdem.commonse.akima import Akima
from wisdem.ccblade.ccblade_component import CCBladeGeometry
from wisdem.ccblade import CCAirfoil
from wisdem.airfoilprep import Airfoil
from precomp import Profile, Orthotropic2DMaterial, CompositeSection, _precomp

from scipy.interpolate import PchipInterpolator

NINPUT = 5
TURBULENCE_CLASS = wisdem.commonse.enum.Enum('A B C')
TURBINE_CLASS = wisdem.commonse.enum.Enum('I II III')
DRIVETRAIN_TYPE = wisdem.commonse.enum.Enum('geared single_stage multi_drive pm_direct_drive')

class ReferenceBlade(object):
    def __init__(self):
        self.name           = None
        self.rating         = None
        self.turbine_class  = None
        self.drivetrainType = None
        self.downwind       = None
        self.nBlades        = None
        
        self.bladeLength   = None
        self.hubFraction   = None
        self.precone       = None
        self.tilt          = None
        
        self.r         = None
        self.r_in      = None
        self.npts      = None
        self.chord     = None
        self.chord_ref = None
        self.theta     = None
        self.precurve  = None
        self.precurveT = None
        self.presweep  = None
        self.airfoils  = None
        
        self.airfoil_files  = None
        self.r_cylinder     = None
        self.r_max_chord    = None
        self.spar_thickness = None
        self.te_thickness   = None
        self.le_location    = None
        
        self.web1 = None
        self.web2 = None
        self.web3 = None
        self.web4 = None
        
        self.sector_idx_strain_spar = None
        self.sector_idx_strain_te   = None

        self.control_Vin  = None
        self.control_Vout = None
        self.control_tsr  = None
        self.control_minOmega = None
        self.control_maxOmega = None
        self.control_maxTS = None
        
    def setRin(self):
        self.r_in = np.r_[0.0, self.r_cylinder, np.linspace(self.r_max_chord, 1.0, NINPUT-2)]
        
    def getAeroPath(self):
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), self.name+'_AFFiles')
    
    def getStructPath(self):
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), self.name+'_PreCompFiles')

    def getAirfoilCoordinates(self):
        data = []
        for a in self.airfoil_files:
            coord = np.loadtxt(a.replace('.dat','.pfl'), skiprows=2)
            data.append(coord)
        return data

    def set_polars(self, thickness, af_thicknesses, af_files, blend=True):
        af_ref = ['']*len(af_thicknesses)
        afinit = CCAirfoil.initFromAerodynFile
        for k, af_file in enumerate(af_files):
            af_ref[k] = afinit(af_file)
        # Get common alpha
        af_alpha = Airfoil.initFromAerodynFile(af_files[0])
        alpha, Re, _, _, _ = af_alpha.createDataGrid()

        if blend:
            # Blend airfoil polars
            self.BlendAirfoils(af_ref, af_thicknesses, alpha, Re, thickness)
        else:
            self.airfoils = ['']*self.npts
            for k, thk in enumerate(thickness):
                af_idx = np.argmin(abs(af_thicknesses - thk))
                self.airfoils[k] = af_ref[af_idx]

    def BlendAirfoils(self, af_ref, af_thicknesses, alpha, Re, thickness):

        af_thicknesses = np.asarray(af_thicknesses)
        thickness    = np.asarray(thickness)


        n_af_ref  = len(af_ref)
        n_aoa     = len(alpha)
        n_span    = len(thickness)

        # error handling for spanwise thickness greater/less than the max/min airfoil thicknesses
        np.place(thickness, thickness>max(af_thicknesses), max(af_thicknesses))
        np.place(thickness, thickness<min(af_thicknesses), min(af_thicknesses))

        # get reference airfoil polars
        cl_ref = np.zeros((n_aoa, n_af_ref))
        cd_ref = np.zeros((n_aoa, n_af_ref))
        cm_ref = np.zeros((n_aoa, n_af_ref))
        for i in range(n_af_ref):
            cl_ref[:,i], cd_ref[:,i], cm_ref[:,i] = af_ref[i].evaluate(alpha*np.pi/180., Re, return_cm=True)
            cl_ref[ 0,i] = 0.
            cl_ref[-1,i] = 0.
            cm_ref[ 0,i] = 0.
            cm_ref[-1,i] = 0.
            cd_ref[-1,i] = cd_ref[0,i]


        # spline selection
        _spline = PchipInterpolator

        # interpolate
        spline_cl = _spline(af_thicknesses, cl_ref, axis=1)
        spline_cd = _spline(af_thicknesses, cd_ref, axis=1)
        spline_cm = _spline(af_thicknesses, cm_ref, axis=1)
        cl = spline_cl(thickness)
        cd = spline_cd(thickness)
        cm = spline_cm(thickness)


        # CCBlade airfoil class instances
        self.airfoils = [None]*n_span
        for i in range(n_span):
            self.airfoils[i] = CCAirfoil(alpha, Re, cl[:,i], cd[:,i], cm[:,i])
            self.airfoils[i].eval_unsteady(alpha, cl[:,i], cd[:,i], cm[:,i])

            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.plot(alpha, cl[:,i])
            # plt.legend()
            # plt.show()

    # def BlendAirfoils(self, af_ref, af_thicknesses, alpha, Re, thickness):

    #     self.airfoils = ['']*self.npts
    #     for k, thk in enumerate(thickness):
    #         # Blend airfoils with exception handling
    #         if thk in af_thicknesses:
    #             af_out = af_ref[np.where(af_thicknesses==thk)[0][0]]
    #             # print thk, thk
    #         elif thk > max(af_thicknesses):
    #             af_out = af_ref[np.argmax(af_thicknesses)]
    #             # print thk, af_thicknesses[np.argmax(af_thicknesses)]
    #         elif thk < min(af_thicknesses):
    #             af_out = af_ref[np.argmin(af_thicknesses)]
    #             # print thk, af_thicknesses[np.argmin(af_thicknesses)]
    #         else:
    #             # Blend airfoils
    #             af1 = max(np.where(af_thicknesses < thk)[0])
    #             af2 = min(np.where(af_thicknesses > thk)[0])
    #             thk1 = af_thicknesses[af1]
    #             thk2 = af_thicknesses[af2]
    #             blend2 = (thk-thk1)/(thk2-thk1)
    #             blend1 = 1-blend2

    #             cl1, cd1 = af_ref[af1].evaluate(alpha*np.pi/180., Re)
    #             cl2, cd2 = af_ref[af2].evaluate(alpha*np.pi/180., Re)

    #             cl = cl1*blend1 + cl2*blend2
    #             cd = cd1*blend1 + cd2*blend2

    #             af_out = CCAirfoil(alpha, Re, cl, cd)

    #             # print '%0.3f\t%d*%0.3f\t%d*%0.3f'%(thk, thk1,blend1, thk2,blend2)
    #             # import matplotlib.pyplot as plt
    #             # plt.figure()
    #             # plt.plot(alpha, cl1, label=str(thk1))
    #             # plt.plot(alpha, cl2, label=str(thk2))
    #             # plt.plot(alpha, cl, label=str(thk))
    #             # plt.legend()
    #             # plt.show()

    #         self.airfoils[k] = af_out

        
class NREL5MW(ReferenceBlade):
    def __init__(self):
        super(NREL5MW, self).__init__()

        # Raw data from https://www.nrel.gov/docs/fy09osti/38060.pdf
        #Node,RNodes,AeroTwst,DRNodes,Chord,Airfoil,Table
        #(-),(m),(deg),(m),(m),(-)
        # raw = StringIO(
        # """1,2.8667,13.308,2.7333,3.542,Cylinder1.dat
        # 2,5.6000,13.308,2.7333,3.854,Cylinder1.dat
        # 3,8.3333,13.308,2.7333,4.167,Cylinder2.dat
        # 4,11.7500,13.308,4.1000,4.557,DU40_A17.dat
        # 5,15.8500,11.480,4.1000,4.652,DU35_A17.dat
        # 6,19.9500,10.162,4.1000,4.458,DU35_A17.dat
        # 7,24.0500,9.011,4.1000,4.249,DU30_A17.dat
        # 8,28.1500,7.795,4.1000,4.007,DU25_A17.dat
        # 9,32.2500,6.544,4.1000,3.748,DU25_A17.dat
        # 10,36.3500,5.361,4.1000,3.502,DU21_A17.dat
        # 11,40.4500,4.188,4.1000,3.256,DU21_A17.dat
        # 12,44.5500,3.125,4.1000,3.010,NACA64_A17.dat
        # 13,48.6500,2.319,4.1000,2.764,NACA64_A17.dat
        # 14,52.7500,1.526,4.1000,2.518,NACA64_A17.dat
        # 15,56.1667,0.863,2.7333,2.313,NACA64_A17.dat
        # 16,58.9000,0.370,2.7333,2.086,NACA64_A17.dat
        # 17,61.6333,0.106,2.7333,1.419,NACA64_A17.dat""")

        raw = StringIO(
        """0.0000000E+00,1.3308000E+01,3.5420000E+00,Cylinder1.dat
        1.3667000E+00,1.3308000E+01,3.5420000E+00,Cylinder1.dat
        4.1000000E+00,1.3308000E+01,3.8540000E+00,Cylinder1.dat
        6.8333000E+00,1.3308000E+01,4.1670000E+00,Cylinder2.dat
        1.0250000E+01,1.3308000E+01,4.5570000E+00,DU40_A17.dat
        1.4350000E+01,1.1480000E+01,4.6520000E+00,DU35_A17.dat
        1.8450000E+01,1.0162000E+01,4.4580000E+00,DU35_A17.dat
        2.2550000E+01,9.0110000E+00,4.2490000E+00,DU30_A17.dat
        2.6650000E+01,7.7950000E+00,4.0070000E+00,DU25_A17.dat
        3.0750000E+01,6.5440000E+00,3.7480000E+00,DU25_A17.dat
        3.4850000E+01,5.3610000E+00,3.5020000E+00,DU21_A17.dat
        3.8950000E+01,4.1880000E+00,3.2560000E+00,DU21_A17.dat
        4.3050000E+01,3.1250000E+00,3.0100000E+00,NACA64_A17.dat
        4.7150000E+01,2.3190000E+00,2.7640000E+00,NACA64_A17.dat
        5.1250000E+01,1.5260000E+00,2.5180000E+00,NACA64_A17.dat
        5.4666700E+01,8.6300000E-01,2.3130000E+00,NACA64_A17.dat
        5.7400000E+01,3.7000000E-01,2.0860000E+00,NACA64_A17.dat
        6.0133300E+01,1.0600000E-01,1.4190000E+00,NACA64_A17.dat
        6.1500000E+01,1.0600000E-01,1.4190000E+00,NACA64_A17.dat""")

        # from Sandia 61.5m blade, Numad
        raw_r_thick = np.array([0.0000000E+00, 3.0000000E-01, 4.0000000E-01, 5.0000000E-01, 6.0000000E-01, 7.0000000E-01, 8.0000000E-01, 1.3667000E+00, 1.5000000E+00, 1.6000000E+00, 4.1000000E+00, 5.5000000E+00, 6.8333000E+00, 9.0000000E+00, 1.0250000E+01, 1.2000000E+01, 1.4350000E+01, 1.7000000E+01, 1.8450000E+01, 2.0500000E+01, 2.2550000E+01, 2.4600000E+01, 2.6650000E+01, 3.0750000E+01, 3.2000000E+01, 3.4850000E+01, 3.7000000E+01, 3.8950000E+01, 4.1000000E+01, 4.2000000E+01, 4.3050000E+01, 4.5000000E+01, 4.7150000E+01, 5.1250000E+01, 5.4666700E+01, 5.7400000E+01, 6.0133300E+01, 6.1500000E+01])
        raw_r_thick = raw_r_thick/raw_r_thick[-1]
        raw_thick = np.array([1.0000000E+00, 1.0000000E+00, 1.0000000E+00, 1.0000000E+00, 1.0000000E+00, 1.0000000E+00, 1.0000000E+00, 1.0000000E+00, 9.8751500E-01, 9.7816802E-01, 7.5439880E-01, 6.4303911E-01, 5.5121360E-01, 4.4117964E-01, 4.0500000E-01, 3.7480511E-01, 3.5000000E-01, 3.3433084E-01, 3.2616543E-01, 3.1401675E-01, 3.0000000E-01, 2.8244827E-01, 2.6403824E-01, 2.3997369E-01, 2.3391742E-01, 2.2014585E-01, 2.0967308E-01, 2.0011990E-01, 1.9007578E-01, 1.8516665E-01, 1.8000000E-01, 1.8000000E-01, 1.8000000E-01, 1.8000000E-01, 1.8000000E-01, 1.8000000E-01, 1.8000000E-01, 1.8000000E-01])
        
        # Name to recover / lookup this info
        self.name     = '5MW'
        self.rating   = 5e6
        self.nBlades  = 3
        self.downwind = False
        self.turbine_class = TURBINE_CLASS['I']
        self.drivetrain    = DRIVETRAIN_TYPE['GEARED']

        self.hubHt  = 90.0
        self.hubFraction = 1.5/61.5
        self.bladeLength = 61.5
        self.precone     = 2.5
        self.tilt        = 5.0
        
        # Analysis grid (old r_str)
        eps = 1e-4
        self.r = np.array([eps, 0.00492790457512, 0.00652942887106, 0.00813095316699,
                           0.00983257273154, 0.0114340970275, 0.0130356213234, 0.02222276,
                           0.024446481932, 0.026048006228, 0.06666667, 0.089508406455,
                           0.11111057, 0.146462614229, 0.16666667, 0.195309105255, 0.23333333,
                           0.276686558545, 0.3, 0.333640766319, 0.36666667, 0.400404310407,
                           0.43333333, 0.5, 0.520818918408, 0.56666667, 0.602196371696,
                           0.63333333, 0.667358391486, 0.683573824984, 0.7, 0.73242031601,
                           0.76666667, 0.83333333, 0.88888943, 0.93333333, 0.97777724, 1.0-eps])
        self.npts = self.r.size
        
        # Blade aero geometry
        raw    = list([row for row in csv.reader(raw)])
        raw_r  = np.array([float(m[0]) for m in raw]) / float(raw[-1][0])
        raw_tw = np.array([float(m[1]) for m in raw])
        raw_c  = np.array([float(m[2]) for m in raw])
        raw_af = [m[-1] for m in raw]

        idx_cylinder = raw_af.index('DU40_A17.dat') - 1
        self.r_cylinder  = raw_r[idx_cylinder]
        self.r_max_chord = raw_r[np.argmax(raw_c)]
        self.setRin()
        
        myspline = Akima(raw_r, raw_tw)
        self.theta, _, _, _ = myspline.interp(self.r_in)
        
        myspline = Akima(raw_r, raw_c)
        self.chord, _, _, _     = myspline.interp(self.r_in)
        self.chord_ref, _, _, _ = myspline.interp(self.r)
        #np.array([3.2612, 3.3100915356, 3.32587052924, 3.34159388653, 3.35823798667, 3.37384375335,
        #3.38939112914, 3.4774055542, 3.49839685, 3.51343645709, 3.87017220335, 4.04645623801, 4.19408216643,
        #                               4.47641008477, 4.55844487985, 4.57383098262, 4.57285771934, 4.51914315648, 4.47677655262, 4.40075650022,
        #                               4.31069949379, 4.20483735936, 4.08985563932, 3.82931757126, 3.74220276467, 3.54415796922, 3.38732428502,
        #                               3.24931446473, 3.23421422609, 3.22701537997, 3.21972125648, 3.08979310611, 2.95152261813, 2.330753331,
        #                               2.05553464181, 1.82577817774, 1.5860853279, 1.4621])
        # TODO: what's the difference?
        #print np.c_[self.chord_ref, chord]


        self.precurve  = np.zeros(self.chord.shape)
        self.precurveT = 0.0
        self.presweep  = np.zeros(self.chord.shape)

        # Spar cap thickness- linear taper from end of cylinder to tip
        spar_str_orig = np.array([0.05, 0.04974449, 0.04973077, 0.04971704, 0.04970245, 0.04968871,
                                  0.04967496, 0.04959602, 0.04957689, 0.04956310, 0.04921172, 0.04901266,
                                  0.04882344, 0.04851176, 0.04833251, 0.04807698, 0.04773559, 0.04749433,
                                  0.04743920, 0.04738432, 0.04728946, 0.04707145, 0.04666403, 0.04495385,
                                  0.04418236, 0.04219110, 0.04038254, 0.03861577, 0.03649927, 0.03542479,
                                  0.03429437, 0.03194315, 0.02928793, 0.02357558, 0.01826438, 0.01365496,
                                  0.00872504, 0.0061398])
        myspline = Akima(self.r, spar_str_orig)
        self.spar_thickness, _, _, _ = myspline.interp(self.r_in)
        
        te_str_orig = np.array([0.1, 0.10082163, 0.10085572, 0.10088880, 0.10092286, 0.10095388,
                                0.10098389, 0.10113668, 0.10116870, 0.10119056, 0.10140956, 0.10124916,
                                0.10090965, 0.09996015, 0.09919790, 0.09784357, 0.09554915, 0.09204126,
                                0.08974946, 0.08603226, 0.08200398, 0.07760656, 0.07314138, 0.06393682,
                                0.06083395, 0.05341039, 0.04736243, 0.04205857, 0.03645787, 0.03391951,
                                0.03146610, 0.02706400, 0.02308701, 0.01642426, 0.01190607, 0.00896844,
                                0.00663243, 0.00569])
        myspline = Akima(self.r, te_str_orig)
        self.te_thickness, _, _, _ = myspline.interp(self.r_in)
        
        self.le_location = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.498, 0.497, 0.465, 0.447, 0.43, 0.411,
                                     0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4,
                                     0.4, 0.4, 0.4, 0.4])


        myspline = Akima(raw_r_thick, raw_thick)
        self.thickness, _, _, _ = myspline.interp(self.r)
         
        # Load airfoil polar files
        afpath = self.getAeroPath()
        af_thicknesses  = np.array([18., 21., 25., 30., 35., 40., 100.])
        airfoil_files = ['NACA64_A17.dat', 'DU21_A17.dat', 'DU25_A17.dat', 'DU30_A17.dat', 'DU35_A17.dat', 'DU40_A17.dat', 'Cylinder1.dat']
        airfoil_files = [os.path.join(afpath, af_file) for af_file in airfoil_files]
        self.set_polars(self.thickness, af_thicknesses, airfoil_files)

        # Now set best guess at airfoil cordinates along span without interpolating like the polar (this is just for plotting)
        self.airfoil_files = ['']*self.npts
        for k in range(self.npts):
            idx = np.argmin( np.abs(raw_r - self.r[k]) )
            self.airfoil_files[k] = os.path.join(afpath, raw_af[idx])

        # Layup info
        self.sector_idx_strain_spar = np.array([2]*self.npts)
        self.sector_idx_strain_te = np.array([3]*self.npts)
        
        self.web1 = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.4114, 0.4102, 0.4094, 0.3876,
                              0.3755, 0.3639, 0.345, 0.3342, 0.3313, 0.3274, 0.323, 0.3206, 0.3172, 0.3138, 0.3104,
                              0.307, 0.3003, 0.2982, 0.2935, 0.2899, 0.2867, 0.2833, 0.2817, 0.2799, 0.2767, 0.2731,
                              0.2664, 0.2607, 0.2562, 0.1886, np.nan])
        self.web2 = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.5886, 0.5868, 0.5854, 0.5508,
                              0.5315, 0.5131, 0.4831, 0.4658, 0.4687, 0.4726, 0.477, 0.4794, 0.4828, 0.4862, 0.4896,
                              0.493, 0.4997, 0.5018, 0.5065, 0.5101, 0.5133, 0.5167, 0.5183, 0.5201, 0.5233, 0.5269,
                              0.5336, 0.5393, 0.5438, 0.6114, np.nan])
        self.web3 = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                              np.nan, np.nan, np.nan, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                              1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
        self.web4 = np.nan * np.ones(self.web3.shape)

        
        # Control
        self.control_Vin      = 3.0
        self.control_Vout     = 25.0
        self.control_minOmega = 6.9
        self.control_maxOmega = 12.1
        self.control_tsr      = 80.0 / 11.4
        self.control_pitch    = 0.0
        self.control_maxTS    = 80.0
        
        
class DTU10MW(ReferenceBlade):
    def __init__(self):
        super(DTU10MW, self).__init__()

        self.name   = '10MW'
        self.rating = 10e6
        self.nBlades  = 3
        self.downwind = False
        self.turbine_class = TURBINE_CLASS['I']
        self.drivetrain    = DRIVETRAIN_TYPE['GEARED']

        self.hubHt  = 119.0
        self.bladeLength = 0.5 * (198.0 - 4.6)
        self.hubFraction = 0.5*4.6 / self.bladeLength
        self.precone     = 4.0
        self.tilt        = 6.0

        # DTU 10MW BLADE PROPS
        #Eta,Chord,Twist,Rel.Thick.,Abs.Thick.,PreBend,Sweep,PitchAx.
        #[-],[mm],[deg],[%],[mm],[mm],[mm],[%]
        raw = StringIO("""0.0000,4600.000,14.500,100.000,4.600,-0.000,0.000,0.500
        0.0176,4601.556,14.037,99.739,4.590,-7.669,0.000,0.500
        0.0313,4601.651,13.737,98.192,4.518,-14.693,0.000,0.500
        0.0420,4609.049,13.545,95.682,4.410,-20.900,0.000,0.498
        0.0507,4636.155,13.413,92.513,4.289,-26.516,0.000,0.489
        0.0579,4667.782,13.312,89.240,4.166,-31.484,0.000,0.480
        0.0642,4698.278,13.223,86.130,4.047,-35.920,0.000,0.473
        0.0700,4729.304,13.139,83.061,3.928,-40.113,0.000,0.465
        0.0758,4763.632,13.056,79.871,3.805,-44.343,0.000,0.459
        0.0821,4807.427,12.969,76.338,3.670,-48.998,0.000,0.453
        0.0893,4869.795,12.874,72.208,3.516,-54.497,0.000,0.445
        0.0980,4953.764,12.752,67.393,3.338,-61.250,0.000,0.435
        0.1087,5063.527,12.565,61.870,3.133,-69.747,0.000,0.423
        0.1224,5218.259,12.223,55.573,2.900,-80.713,0.000,0.408
        0.1400,5444.001,11.600,48.640,2.648,-95.183,0.000,0.389
        0.1610,5688.237,10.674,42.379,2.411,-112.440,0.000,0.369
        0.1841,5937.831,9.329,37.456,2.224,-131.381,0.000,0.351
        0.2093,5987.655,7.980,34.759,2.081,-152.316,0.000,0.339
        0.2366,5990.629,6.724,33.223,1.990,-175.303,0.000,0.331
        0.2660,5896.920,5.879,32.682,1.927,-200.154,0.000,0.331
        0.2973,5704.650,5.222,32.580,1.859,-227.286,0.000,0.332
        0.3305,5463.845,4.668,32.606,1.782,-257.214,0.000,0.341
        0.3653,5176.618,4.131,32.547,1.685,-292.010,0.000,0.353
        0.4015,4857.565,3.614,32.367,1.572,-334.013,0.000,0.368
        0.4387,4520.899,3.085,32.063,1.450,-385.250,0.000,0.387
        0.4766,4179.799,2.496,31.474,1.316,-447.911,0.000,0.407
        0.5148,3842.977,1.843,30.700,1.180,-526.833,0.000,0.427
        0.5529,3519.719,1.161,29.781,1.048,-625.007,0.000,0.447
        0.5905,3217.849,0.472,28.768,0.926,-744.099,0.000,0.465
        0.6273,2942.296,-0.197,27.680,0.814,-887.576,0.000,0.481
        0.6628,2695.475,-0.821,26.556,0.716,-1057.588,0.000,0.495
        0.6969,2477.568,-1.382,25.426,0.630,-1253.153,0.000,0.506
        0.7293,2287.113,-1.870,24.340,0.557,-1474.395,0.000,0.515
        0.7597,2122.247,-2.277,23.397,0.497,-1720.593,0.000,0.522
        0.7882,1980.577,-2.603,22.828,0.452,-1987.436,0.000,0.525
        0.8145,1859.108,-2.850,22.755,0.423,-2273.422,0.000,0.526
        0.8387,1754.739,-3.024,22.673,0.398,-2573.827,0.000,0.526
        0.8608,1665.081,-3.125,22.533,0.375,-2884.584,0.000,0.523
        0.8809,1586.370,-3.160,22.351,0.355,-3202.247,0.000,0.519
        0.8990,1511.990,-3.130,22.145,0.335,-3522.496,0.000,0.513
        0.9153,1434.483,-3.038,21.930,0.315,-3841.371,0.000,0.507
        0.9299,1350.941,-2.883,21.722,0.293,-4154.921,0.000,0.499
        0.9429,1260.917,-2.669,21.533,0.272,-4460.302,0.000,0.492
        0.9544,1166.025,-2.399,21.372,0.249,-4755.279,0.000,0.485
        0.9646,1062.491,-2.085,21.241,0.226,-5037.057,0.000,0.477
        0.9736,957.120,-1.727,21.143,0.202,-5304.189,0.000,0.470
        0.9816,862.535,-1.340,21.073,0.182,-5555.265,0.000,0.464
        0.9885,759.755,-0.927,21.030,0.160,-5789.509,0.000,0.457
        0.9946,540.567,-0.483,21.007,0.114,-6006.941,0.000,0.452
        1.0000,96.200,-0.037,21.000,0.020,-6206.217,0.000,0.446""")

        eps = 1e-4
        self.r = np.array([eps, 0.0204081632653, 0.0408163265306, 0.0612244897959, 0.0816326530612, 0.102040816327,
                           0.122448979592, 0.142857142857, 0.163265306122, 0.183673469388, 0.204081632653, 0.224489795918,
                           0.244897959184, 0.265306122449, 0.285714285714, 0.30612244898, 0.326530612245, 0.34693877551,
                           0.367346938776, 0.387755102041, 0.408163265306, 0.428571428571, 0.448979591837, 0.469387755102,
                           0.489795918367, 0.510204081633, 0.530612244898, 0.551020408163, 0.571428571429, 0.591836734694,
                           0.612244897959, 0.632653061224, 0.65306122449, 0.673469387755, 0.69387755102, 0.714285714286,
                           0.734693877551, 0.755102040816, 0.775510204082, 0.795918367347, 0.816326530612, 0.836734693878,
                           0.857142857143, 0.877551020408, 0.897959183673, 0.918367346939, 0.938775510204, 0.959183673469,
                           0.979591836735, 1.0-eps])
        self.npts = self.r.size
        
        raw     = np.loadtxt(raw, delimiter=',')
        raw_r   = raw[:,0]
        raw_c   = raw[:,1] * 1e-3
        raw_tw  = raw[:,2]
        raw_th  = raw[:,3]
        raw_pre = raw[:,5] * 1e-3
        raw_sw  = raw[:,6] * 1e-3

        idx_cylinder = 5
        self.r_cylinder  = raw_r[idx_cylinder]
        self.r_max_chord = raw_r[np.argmax(raw_c)]
        self.setRin()
        
        myspline = Akima(raw_r, raw_tw)
        self.theta, _, _, _ = myspline.interp(self.r_in)
        
        myspline = Akima(raw_r, raw_c)
        self.chord, _, _, _     = myspline.interp(self.r_in)
        self.chord_ref, _, _, _ = myspline.interp(self.r)
        #self.chord_ref = np.array([5.38, 5.3800643553, 5.38031711143, 5.38780280252, 5.40677951126, 5.48505840079, 5.59326574185,
        #                               5.73141566075, 5.86843135503, 5.99999190341, 6.09904231251, 6.17116486928, 6.19400935481,
        #                               6.20302411962, 6.18309136227, 6.14171800022, 6.07759639166, 5.99796748755, 5.90179584286,
        #                               5.79486385463, 5.67757358533, 5.55267710905, 5.42166648998, 5.28499407124, 5.14373698212,
        #                               4.99871795233, 4.85053210974, 4.70010140244, 4.54830998864, 4.39588552607, 4.24360835244,
        #                               4.09216987991, 3.94187021346, 3.79298624168, 3.64578973915, 3.50055778142, 3.3575837574,
        #                               3.21717725827, 3.07962549378, 2.94515242311, 2.81389912887, 2.68570603823, 2.56087639104,
        #                               2.43927605964, 2.31610551275, 2.18016451487, 2.01720583646, 1.81389861412, 1.50584435653, 0.6])

        myspline = Akima(raw_r, raw_pre)
        self.precurve, _, _, _ = myspline.interp(self.r_in)
        self.precurveT = 0.0

        myspline = Akima(raw_r, raw_sw)
        self.presweep, _, _, _ = myspline.interp(self.r_in)

        # # Spar cap thickness- linear taper from end of cylinder to tip
        # self.spar_thickness = np.linspace(1.44, 0.53, NINPUT)
        # self.te_thickness   = np.linspace(0.8, 0.2, NINPUT)
        t_spar = [0.0320004189281497, 0.032413233514380135, 0.0347998074739571, 0.039211725978893365, 0.04448607137027535, 0.05004958035967985, 
                  0.055773198307797055, 0.060808909220920346, 0.06489796815102368, 0.06729338026826848, 0.07038507637861223, 0.07320149826122159, 
                  0.07505402214311759, 0.07709642418372635, 0.07918050884914904, 0.08103357158490876, 0.08270252680667647, 0.08403234987866959, 
                  0.08515644449040799, 0.08607979218517818, 0.08658210941996525, 0.08659983499339703, 0.08646447909743751, 0.08581626920407662, 
                  0.0848164285567969, 0.08377910964381965, 0.0822367983198804, 0.08070965621865138, 0.07903672500724102, 0.07736379667849971, 
                  0.07580112855098903, 0.07416991429397488, 0.07206050597089943, 0.06946315980483178, 0.0662730627859155, 0.06268615362703936, 
                  0.058682210134965856, 0.054168752160222426, 0.049167791007208306, 0.044039704321394174, 0.0385981511713949, 0.03302340214440563, 
                  0.027470859650715124, 0.02200899556289725, 0.016845764341483292, 0.012236276175915984, 0.008142142767238786, 0.0047736891784759755, 
                  0.0035326892184044827, 0.011810323720413905] # baseline spar cap and TE thickness for DTU10MW, from rotor_structure.ResizeCompositeSection - UPDATE when new composite layups are available
        t_te = [0.04200054984319648, 0.04225117651974834, 0.0449028267651131, 0.05115482875366182, 0.05773200373093057, 0.0646076843283427, 
                0.07157444156721758, 0.07838332015647545, 0.08236702099771726, 0.08636183420245723, 0.08807738710796026, 0.08841695885692735, 
                0.08435661818895519, 0.08227717500375323, 0.07523246366708176, 0.0718785818529033, 0.06482470192539755, 0.061452305196879796, 
                0.054373783400364466, 0.049840546332820826, 0.043967287094035856, 0.04027996631594854, 0.034731762503613456, 0.03165893729347588, 
                0.028640564155395527, 0.026206171585164948, 0.022504329520350853, 0.01841167062556291, 0.01663921489751576, 0.01573381774722061, 
                0.010781562015476103, 0.010435507532911352, 0.009993455758864546, 0.009677378170431624, 0.009324302005793973, 0.008808234511422684,
                0.008254745457758155, 0.007815979771853115, 0.007225157857060406, 0.006580363735558421, 0.005937478092511996, 0.005299415990797145, 
                0.004743698234230719, 0.003942519682115882, 0.003288264073460905, 0.0026828299819035804, 0.0021288202493691083, 0.0016345630556029948, 
                0.001159406221958171, 0.0034522484721209875]
        myspline = Akima(raw_r, t_spar)
        self.spar_thickness, _, _, _ = myspline.interp(self.r_in)
        myspline = Akima(raw_r, t_te)
        self.te_thickness, _, _, _ = myspline.interp(self.r_in)
        
        myspline = Akima(raw_r, raw_th)
        self.thickness, _, _, _ = myspline.interp(self.r)
        self.thickness = np.minimum(100.0, self.thickness)

        # Load airfoil polar files
        afpath = self.getAeroPath()
        # af_thicknesses  = np.array([21.1, 24.1, 27.0, 30.1, 33.0, 36.0, 48.0, 60.0, 72.0, 100.0])
        af_thicknesses  = np.array([24.1, 30.1, 36.0, 48.0, 60.0, 100.0])
        airfoil_files = ['FFA_W3_241.dat', 'FFA_W3_301.dat', 'FFA_W3_360.dat', 'FFA_W3_480.dat', 'FFA_W3_600.dat', 'Cylinder.dat']
        airfoil_files = [os.path.join(afpath, af_file) for af_file in airfoil_files]
        self.set_polars(self.thickness, af_thicknesses, airfoil_files)

        # Now set best guess at airfoil cordinates along span without interpolating like the polar (this is just for plotting)
        self.airfoil_files = ['']*self.npts
        for k in range(self.npts):
            idx_thick       = np.where(self.thickness[k] <= af_thicknesses)[0]
            if idx_thick.size > 0 and idx_thick[0] < af_thicknesses.size-1:
                prefix   = 'FFA_W3_'
                thickStr = str(np.int(10*af_thicknesses[idx_thick[0]]))
            else:
                prefix   = 'Cylinder'
                thickStr = ''
            self.airfoil_files[k] = os.path.join(afpath, prefix + thickStr + '.dat')



        # Structural analysis inputs
        self.le_location = np.array([0.5, 0.499998945239, 0.499990630963, 0.499384561429, 0.497733369567, 0.489487054775,
                                     0.476975219349, 0.458484322766, 0.440125810719, 0.422714559863, 0.407975209714,
                                     0.395449769723, 0.385287280879, 0.376924554763, 0.370088311651, 0.364592902698,
                                     0.3602205136, 0.356780489919, 0.354039530035, 0.351590005932, 0.350233815248, 0.350012355763,
                                     0.349988281626, 0.350000251201, 0.350002561185, 0.350001421895, 0.349997012891, 0.350001029096,
                                     0.350000632518, 0.349999297634, 0.350000264157, 0.350000005654, 0.349999978357, 0.349999995158,
                                     0.350000006591, 0.349999999186, 0.349999998202, 0.350000000551, 0.350000000029, 0.349999999931,
                                     0.35000000004, 0.350000000001, 0.35, 0.350000000001, 0.349999999999, 0.35, 0.35, 0.35, 0.35, 0.35])

        # UPDATE when new composite layup is available
        self.sector_idx_strain_spar = np.array([2]*self.npts)
        self.sector_idx_strain_te = np.array([6]*self.npts)

        self.web1 = np.array([0.446529203227, 0.446642686219, 0.447230977047, 0.449423527671, 0.451384667298, 0.45166085909,
                              0.445821859041, 0.433601957075, 0.414203341702, 0.391111637325, 0.367038887871, 0.344148340044,
                              0.32264263023, 0.303040717673, 0.285780556269, 0.271339581072, 0.261077569528, 0.254987877709,
                              0.250499030835, 0.246801903789, 0.243793928448, 0.242362866767, 0.241169996298, 0.240114471242,
                              0.239138338743, 0.238211240433, 0.237380060299, 0.236625908889, 0.235947619537, 0.235375269498,
                              0.234910524166, 0.234573714458, 0.23437656803, 0.234323591937, 0.234429396513, 0.23469408391,
                              0.235090916602, 0.235639910948, 0.236359205424, 0.237292044985, 0.238468772012, 0.239912928964,
                              0.241676539436, 0.24378663077, 0.246041897214, 0.247824545238, 0.248212620456, 0.247666927859,
                              0.246627910571, 0.154148714864])
        self.web2 = np.array([0.579105947595, 0.579342815032, 0.580624719333, 0.585617777398, 0.5905335998, 0.592757384044,
                              0.587897774807, 0.576668436742, 0.557200875669, 0.532380978251, 0.505531782719, 0.479744314701,
                              0.456216340946, 0.43494475968, 0.416533496674, 0.40143144474, 0.390766314857, 0.384449528527,
                              0.379891695643, 0.376232568599, 0.373403485013, 0.372223966271, 0.371379269973, 0.370759395854,
                              0.370295535294, 0.36996224587, 0.369789327231, 0.36975863937, 0.369855492121, 0.370090120509,
                              0.370456638362, 0.370955636744, 0.371593427472, 0.37236755247, 0.373282787035, 0.374330521266,
                              0.37548259513, 0.376740794894, 0.378108493317, 0.379600043497, 0.381225881051, 0.382999965574,
                              0.384937055128, 0.387054062586, 0.389406784803, 0.392151134201, 0.395577451592, 0.398541075929,
                              0.396140574221, 0.377290889301])
        self.web3 = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.883963623418, 0.882804861283,
                              0.881764724203, 0.880854972132, 0.880063593261, 0.879378185003, 0.878797670205, 0.878308574021,
                              0.87788500707, 0.877514541845, 0.877179551402, 0.876879055, 0.876610936666, 0.876374235852,
                              0.876166483426, 0.875986765133, 0.875832594685, 0.875702298474, 0.875595140019, 0.875508315318,
                              0.8754409845, 0.875392657588, 0.875361122776, 0.875345560917, 0.875342086256, 0.875341086148,
                              0.875340761092, 0.875339510238, 0.875338477803, 0.875337374186, 0.875336204651, 0.87533496247,
                              0.875333692251, 0.875332490358, 0.875331189455, 0.875329175403, 0.875324968372, 0.875312750097,
                              0.875329970752])
        self.web4 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        
        
        # Control
        self.control_Vin      = 4.0
        self.control_Vout     = 25.0
        self.control_minOmega = 6.0
        self.control_maxOmega = 90.0 / self.bladeLength * (60.0/(2.0*np.pi))
        self.control_tsr      = 10.58
        self.control_pitch    = 0.0
        self.control_maxTS    = 90.
        
class TUM3_35MW(ReferenceBlade):
    def __init__(self):
        super(TUM3_35MW, self).__init__()

        self.name   = '3_35MW'
        self.rating = 3.35e6
        self.nBlades  = 3
        self.downwind = False
        self.turbine_class = TURBINE_CLASS['III']
        self.drivetrain    = DRIVETRAIN_TYPE['GEARED']

        self.hubHt  = 110.0
        self.bladeLength = 0.5 * (130.0 - 4.)
        self.hubFraction = 2. / self.bladeLength
        self.precone     = 3.0
        self.tilt        = 5.0

        # DTU 10MW BLADE PROPS
        #Eta,Chord,Twist,Rel.Thick.,Abs.Thick.,PreBend,Sweep,PitchAx.
        #[-],[mm],[deg],[%],[mm],[mm],[mm],[%]
        raw = StringIO("""0.000,2600.0,20.00,100.00,2600.00,0.0,0.000,50.00
        0.010,2600.0,19.87,100.00,2600.00,0.0,0.000,49.65
        0.020,2600.0,19.73,100.00,2600.00,0.0,0.000,49.30
        0.030,2620.6,19.56,99.12,2597.60,0.0,0.000,48.57
        0.040,2680.1,19.39,96.68,2591.05,0.0,0.000,47.16
        0.050,2776.7,19.20,92.96,2581.14,0.0,0.000,45.19
        0.060,2910.6,18.99,88.25,2568.58,0.0,0.000,42.80
        0.070,3032.4,18.76,82.84,2512.03,0.0,0.000,40.78
        0.080,3155.4,18.51,77.01,2430.09,0.0,0.000,38.91
        0.100,3400.0,17.94,65.28,2219.37,0.0,0.000,35.57
        0.120,3618.0,17.07,55.34,2002.05,1.3,0.000,32.93
        0.140,3803.9,15.81,49.44,1880.58,5.5,0.000,30.84
        0.160,3958.7,14.37,45.94,1818.71,10.9,0.000,29.18
        0.180,4083.2,12.95,43.11,1760.48,17.5,0.000,27.85
        0.200,4178.5,11.75,40.96,1711.41,25.3,0.000,26.78
        0.220,4245.5,10.73,39.44,1674.27,34.4,0.000,25.93
        0.240,4285.1,9.71,38.23,1638.27,44.9,0.000,25.27
        0.260,4298.4,8.71,37.23,1600.19,56.7,0.000,24.77
        0.280,4286.7,7.74,36.38,1559.45,69.9,0.000,24.42
        0.300,4252.0,6.81,35.64,1515.33,84.6,0.000,24.19
        0.320,4196.1,5.93,34.96,1466.98,100.8,0.000,24.08
        0.340,4121.1,5.12,34.34,1415.20,118.7,0.000,24.08
        0.360,4028.7,4.39,33.79,1361.24,138.1,0.000,24.18
        0.380,3921.1,3.75,33.29,1305.29,159.3,0.000,24.38
        0.400,3800.0,3.21,32.83,1247.50,182.3,0.000,24.68
        0.420,3667.7,2.76,32.39,1188.13,207.0,0.000,25.08
        0.440,3527.8,2.38,31.97,1127.83,233.9,0.000,25.56
        0.460,3384.0,2.05,31.54,1067.40,262.7,0.000,26.11
        0.480,3240.1,1.77,31.10,1007.61,293.7,0.000,26.72
        0.500,3100.0,1.53,30.62,949.27,326.9,0.000,27.34
        0.520,2966.8,1.32,30.10,893.00,362.5,0.000,27.96
        0.540,2841.2,1.13,29.51,838.52,400.6,0.000,28.55
        0.560,2723.1,0.95,28.86,785.85,441.3,0.000,29.13
        0.580,2612.7,0.78,28.16,735.77,484.7,0.000,29.67
        0.600,2509.8,0.60,27.45,688.91,530.9,0.000,30.16
        0.620,2414.5,0.42,26.75,645.78,580.3,0.000,30.60
        0.640,2326.9,0.27,26.08,606.81,632.8,0.000,30.98
        0.660,2246.8,0.13,25.47,572.33,688.8,0.000,31.28
        0.680,2174.3,0.01,24.96,542.62,748.3,0.000,31.49
        0.700,2109.5,-0.11,24.47,516.14,811.8,0.000,31.59
        0.720,2052.3,-0.22,23.97,491.85,879.1,0.000,31.79
        0.740,2002.8,-0.34,23.46,469.94,951.1,0.000,31.73
        0.760,1960.9,-0.47,22.98,450.52,1027.4,0.000,31.57
        0.780,1926.6,-0.60,22.51,433.73,1109.0,0.000,31.26
        0.800,1900.0,-0.75,22.09,419.69,1195.8,0.000,30.80
        0.820,1879.2,-0.91,21.72,408.10,1288.4,0.000,30.21
        0.840,1854.9,-1.07,21.41,397.12,1387.4,0.000,29.56
        0.860,1816.1,-1.24,21.18,384.61,1493.1,0.000,28.87
        0.880,1751.5,-1.45,21.04,368.48,1606.2,0.000,28.24
        0.900,1650.0,-1.70,21.00,346.50,1727.8,0.000,27.67
        0.920,1500.5,-2.05,21.00,315.11,1858.3,0.000,27.25
        0.940,1291.9,-2.54,21.00,271.31,1998.7,0.000,26.92
        0.960,1013.1,-3.14,21.00,212.74,2150.5,0.000,26.71
        0.980,652.8,-3.84,21.00,137.09,2315.1,0.000,26.44
        1.000,200.0,-4.62,21.00,42.00,2500.0,0.000,25.00""")

        eps = 1e-4
        self.r = np.array([eps, 0.0204081632653, 0.0408163265306, 0.0612244897959, 0.0816326530612, 0.102040816327,
                           0.122448979592, 0.142857142857, 0.163265306122, 0.183673469388, 0.204081632653, 0.224489795918,
                           0.244897959184, 0.265306122449, 0.285714285714, 0.30612244898, 0.326530612245, 0.34693877551,
                           0.367346938776, 0.387755102041, 0.408163265306, 0.428571428571, 0.448979591837, 0.469387755102,
                           0.489795918367, 0.510204081633, 0.530612244898, 0.551020408163, 0.571428571429, 0.591836734694,
                           0.612244897959, 0.632653061224, 0.65306122449, 0.673469387755, 0.69387755102, 0.714285714286,
                           0.734693877551, 0.755102040816, 0.775510204082, 0.795918367347, 0.816326530612, 0.836734693878,
                           0.857142857143, 0.877551020408, 0.897959183673, 0.918367346939, 0.938775510204, 0.959183673469,
                           0.979591836735, 1.0-eps])
        
        # self.r = np.linspace(0, 1.0, 21)
        # self.r[0]=eps
        # self.r[-1]=1. - eps
        
        self.npts = self.r.size

        raw     = np.loadtxt(raw, delimiter=',')
        raw_r   = raw[:,0]
        raw_c   = raw[:,1] * 1e-3
        raw_tw  = raw[:,2]
        raw_th  = raw[:,3]
        raw_pre = raw[:,5] * 1e-3
        raw_sw  = raw[:,6] * 1e-3

        idx_cylinder = 4
        self.r_cylinder  = raw_r[idx_cylinder]
        self.r_max_chord = raw_r[np.argmax(raw_c)]
        self.setRin()
        
        # myspline = Akima(raw_r, raw_tw)
        # self.theta, _, _, _ = myspline.interp(self.r_in)
        
        # myspline = Akima(raw_r, raw_c)
        # self.chord, _, _, _     = myspline.interp(self.r_in)
        # self.chord_ref, _, _, _ = myspline.interp(self.r)

        # myspline = Akima(raw_r, raw_pre)
        # self.precurve, _, _, _ = myspline.interp(self.r_in)
        # self.precurveT = 0.0

        # myspline = Akima(raw_r, raw_sw)
        # self.presweep, _, _, _ = myspline.interp(self.r_in)
        
        
        myspline = PchipInterpolator(raw_r, raw_tw)
        self.theta = myspline(self.r_in)
        
        myspline = PchipInterpolator(raw_r, raw_c)
        self.chord     = myspline(self.r_in)
        self.chord_ref = myspline(self.r)

        myspline = PchipInterpolator(raw_r, raw_pre)
        self.precurve = myspline(self.r_in)
        self.precurve_ref = myspline(self.r)
        self.precurveT = 0.0

        myspline = PchipInterpolator(raw_r, raw_sw)
        self.presweep = myspline(self.r_in)
        
        
        
        
        
        
        
        t_spar = [0.065000, 0.060000, 0.055000, 0.050000, 0.045714, 0.041429, 0.037143, 0.032857, 0.028571, 0.029933, 0.032814,
                  0.035695, 0.038576, 0.041457, 0.044338, 0.048504, 0.052670, 0.056836, 0.061002, 0.065168, 0.062554, 0.059939,
                  0.057325, 0.054710, 0.052096, 0.051623, 0.051150, 0.050676, 0.050203, 0.049730, 0.048233, 0.046736, 0.045239,
                  0.043741, 0.042244, 0.039840, 0.037436, 0.035032, 0.032628, 0.030224, 0.030410, 0.030595, 0.030781, 0.030967,
                  0.031152, 0.025556, 0.019960, 0.014363, 0.008767, 0.003171, 0.003085, 0.003000, 0.001000, 0.001000, 0.001000] 
        t_te = [0.065000, 0.060000, 0.055000, 0.050000, 0.045714, 0.041429, 0.037143, 0.032857, 0.028571, 0.021177, 0.019108,
                0.017039, 0.014970, 0.012901, 0.010831, 0.010429, 0.010028, 0.009626, 0.009224, 0.008822, 0.008656, 0.008491,
                0.008325, 0.008160, 0.007995, 0.008364, 0.008733, 0.009102, 0.009471, 0.009840, 0.008671, 0.007501, 0.006332,
                0.005163, 0.003994, 0.003595, 0.003196, 0.002797, 0.002399, 0.002000, 0.002166, 0.002331, 0.002497, 0.002662,
                0.002828, 0.001075, 0.001056, 0.001037, 0.001019, 0.001000, 0.001000, 0.001000, 0.001000, 0.001000, 0.001000]
        # myspline = Akima(raw_r, t_spar)
        # self.spar_thickness, _, _, _ = myspline.interp(self.r_in)
        # myspline = Akima(raw_r, t_te)
        # self.te_thickness, _, _, _ = myspline.interp(self.r_in)
        
        # myspline = Akima(raw_r, raw_th)
        # self.thickness, _, _, _ = myspline.interp(self.r)
        # self.thickness = np.minimum(100.0, self.thickness)
        
        
        myspline = PchipInterpolator(raw_r, t_spar)
        self.spar_thickness = myspline(self.r_in)
        myspline = PchipInterpolator(raw_r, t_te)
        self.te_thickness = myspline(self.r_in)
        
        myspline = PchipInterpolator(raw_r, raw_th)
        self.thickness = myspline(self.r)
        self.thickness = np.minimum(100.0, self.thickness)
        
        
        
        
        
        # Load airfoil polar files
        afpath = self.getAeroPath()
        # af_thicknesses  = np.array([21.0, 25.0, 30.0, 35.0, 40.0, 50.0, 100.0])
        # airfoil_files_ref = ['DU08-W-210.dat', 'DU91-W2-250.dat', 'DU97-W-300.dat', 'DU00-W2-350.dat', 'FX77-W-400.dat', 'FX77-W-500.dat', 'Cylinder.dat']
        
        af_thicknesses  = np.array([21.0, 25.0, 35.0, 40.0, 100.0])
        airfoil_files_ref = ['DU08-W-210.dat', 'DU91-W2-250.dat', 'DU00-W2-350.dat', 'FX77-W-400.dat', 'Cylinder.dat']
        
        
        airfoil_files_ref = [os.path.join(afpath, af_file) for af_file in airfoil_files_ref]
        # self.set_polars(thickness, af_thicknesses, airfoil_files_ref, blend=False)
        self.set_polars(self.thickness, af_thicknesses, airfoil_files_ref)

        # Now set best guess at airfoil cordinates along span without interpolating like the polar (this is just for plotting)
        self.airfoil_files = ['']*self.npts
        for k in range(self.npts):
            self.airfoil_files[k] = airfoil_files_ref[np.argmin(np.abs(af_thicknesses - self.thickness[k]))]

        # Structural analysis inputs
        raw_le = np.array([0.5000, 0.4965, 0.4930, 0.4857, 0.4716, 0.4519, 0.4280, 0.4078, 0.3891, 0.3557, 0.3293, 0.3084, 0.2918, 0.2785,
                           0.2678, 0.2593, 0.2527, 0.2477, 0.2442, 0.2419, 0.2408, 0.2408, 0.2418, 0.2438, 0.2468, 0.2508, 0.2556, 0.2611,
                           0.2672, 0.2734, 0.2796, 0.2855, 0.2913, 0.2967, 0.3016, 0.3060, 0.3098, 0.3128, 0.3149, 0.3159, 0.3179, 0.3173,
                           0.3157, 0.3126, 0.3080, 0.3021, 0.2956, 0.2887, 0.2824, 0.2767, 0.2725, 0.2692, 0.2671, 0.2644, 0.2500])
        myspline = PchipInterpolator(raw_r, raw_le)
        self.le_location = myspline(self.r)

        # self.sector_idx_strain_spar = np.array([0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0])
        # self.sector_idx_strain_te = np.array([1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 1])
        self.sector_idx_strain_spar = np.array([2]*self.npts)
        self.sector_idx_strain_te = np.array([4]*self.npts)

        self.web1 = np.array([np.nan, np.nan, 0.4701827034215534, 0.4253469916355271, 0.3828130080819686, 0.24746908823626373, 0.22995320599638844, 
                              0.21541572225404393, 0.2038707273438756, 0.1945407600735053, 0.1866181278595582, 0.18086266176650753, 0.176397121415433, 
                              0.1731093503935741, 0.17062331877318185, 0.16876879106649403, 0.16831059135098939, 0.16841699648349387, 0.16943205926557242, 
                              0.17086895268167865, 0.17270249789120876, 0.17581339281765335, 0.1794192889622316, 0.18351850469850114, 0.187358635824116, 
                              0.19119917455467178, 0.19547042052667718, 0.19951734140326338, 0.2034337970373882, 0.20620153182445605, 0.20888234398547278, 
                              0.21160282399969846, 0.21374560522119876, 0.21539795729730044, 0.2150752648952342, 0.21633774360547553, 0.21638653327758073, 
                              0.21554102327464625, 0.2137179463003368, 0.2094600214084868, 0.20516288425364265, 0.20034022024311537, 0.19439236703477922, 
                              0.1876141771165691, 0.1778247533132809, 0.16714558427462486, 0.14989969013495794, 0.26655808283893134, np.nan, np.nan])
        self.web2 = np.array([np.nan, np.nan, 0.4701827034215534, 0.4253469916355271, 0.39088094350964314, 0.4851662989078992, 0.44958875462268477, 
                              0.42441241590306517, 0.404829439598079, 0.3895294492888662, 0.3773553332585585, 0.3688083821270216, 0.36285356603309826, 
                              0.35928022196678755, 0.3576735202551685, 0.3574966158009242, 0.3598837887651105, 0.3639982639018496, 0.369892138375083, 
                              0.3772781406327177, 0.3862087017677361, 0.39751163943876633, 0.4103998659853778, 0.4247511196414795, 0.439663279193677, 
                              0.4551290033447943, 0.47140795507376854, 0.4877847412525208, 0.5042706839596905, 0.5198288909235875, 0.5353390259777657, 
                              0.5508471252450576, 0.5655913969523467, 0.5794847068511754, 0.5908593630110022, 0.6031749942406135, 0.6134208381001381, 
                              0.6216032745337294, 0.6276568358054995, 0.6293882081737915, 0.629980647450311, 0.6305942015894923, 0.6333305336043769, 
                              0.6419175576050282, 0.6591842929704117, 0.6953356290674848, 0.7684174089168752, 0.26977365780451124, np.nan, np.nan])
        self.web3 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.web4 = np.nan * np.ones(self.web3.shape)
        
        
        # Control
        self.control_Vin      = 3.
        self.control_Vout     = 25.0
        self.control_minOmega = 3.77165108498795
        self.control_maxOmega = 11.75298041294
        self.control_tsr      = 8.16326530612245
        self.control_pitch    = 0.803344518293558
        # self.control_maxTS    = 80.
        self.control_maxTS    = 85.
        
class WindPact1_5MW(ReferenceBlade):        
    def __init__(self):
        super(WindPact1_5MW, self).__init__()

        self.name   = '1_5MW'    
        
        self.bladeLength = 33.25
        
        eps = 1e-4
        self.r = np.array([eps, 0.02105, 0.04812, 0.07519, 0.10226, 0.12932, 0.15639, 0.18346, 0.21053, 0.26316, 0.31579,
        0.36842, 0.42105, 0.47368, 0.52632, 0.57895, 0.63158, 0.68421, 0.73684, 0.78947, 0.84211, 0.89474, 0.94737, 1.00000 -eps])
        self.npts = self.r.size
        self.chord_ref = np.array([1.89 ,  1.89 , 2.02 , 2.15 , 2.28 , 2.41 , 2.54 , 2.67 , 2.8 ,  2.672 , 2.544 , 2.416 , 2.288 , 2.16 ,
        2.032 , 1.904 , 1.776 , 1.648 , 1.52 ,  1.391 , 1.262 , 1.133 , 1.004 , 0.875])
        self.le_location = np.array([0.5 , 0.5 , 0.47678571 , 0.45357143 , 0.43035714 , 0.40714286, 0.38392857 , 0.36071429 ,
        0.3375 , 0.3375 , 0.3375 , 0.3375 ,  0.3375 , 0.3375 , 0.3375 , 0.3375 , 0.3375 , 0.3375 , 0.3375 , 0.3375 , 0.3375 , 0.3375 , 0.3375 , 0.3375 ])
        
class SNL13_2MW_00(ReferenceBlade):
    def __init__(self):
        super(SNL13_2MW_00, self).__init__()

        self.name   = '13_2MW_00'    
        
        self.bladeLength = 100.
        
        eps = 1e-4       
        self.r = np.array([eps, 0.00500, 0.00700,  0.00900 ,0.01100 , 0.01300 , 0.02400 , 0.02600 , 0.04700 , 0.06800 , 0.08900 , 0.11400 ,
        0.14600 , 0.16300 , 0.17900 , 0.19500 , 0.22200 , 0.24900 , 0.27600 , 0.35800 , 0.43900 , 0.52000 , 0.60200 , 0.66700 ,
        0.68300 , 0.73200 , 0.76400 , 0.84600 , 0.89400 , 0.94300 , 0.95700 , 0.97200 , 0.98600 , 1.00000 -eps])
        self.npts = self.r.size
        self.chord_ref = np.array([5.694 , 5.694 , 5.694 , 5.694 , 5.694 , 5.694 , 5.792 ,5.811 ,6.058 ,6.304 ,6.551 ,6.835 ,7.215 ,7.404 ,
        7.552 ,7.628 ,7.585 ,7.488 ,7.347 ,6.923 ,6.429 ,5.915 , 5.417 ,5.019 ,4.92  ,4.621 ,4.422 ,3.925 ,3.619 ,2.824 ,
        2.375 ,1.836 ,1.208 ,0.1])
        self.le_location = np.array([0.5, 0.5   ,0.5   ,0.5   ,0.5   ,0.5   ,0.499 ,0.498 ,0.483 ,0.468 ,0.453 ,0.435 ,0.41  ,0.4   ,0.39  ,0.38  ,0.378 ,
        0.377 ,0.375 ,0.375 ,0.375 ,0.375 ,0.375 ,0.375 ,0.375 ,0.375 ,0.375 ,0.375 ,0.375 ,0.375 ,0.375 ,0.375 ,0.375 , 0.375 ])       
        

        
class BAR_00(ReferenceBlade):
    def __init__(self):
        super(BAR_00, self).__init__()

        self.name           = 'BAR_00'    
        
        self.rating         = 5.0e6
        self.nBlades        = 3
        self.downwind       = False
        self.turbine_class  = TURBINE_CLASS['III']
        self.drivetrain     = DRIVETRAIN_TYPE['GEARED']

        self.hubHt     = 140.0
        self.bladeLength    = 100.0
        self.hubFraction    = 3. / self.bladeLength
        self.precone        = 2.5
        self.tilt           = 5.0
        
        eps                 = 1e-4       
        self.r              = np.array([eps, 0.00500, 0.00700,  0.00900 ,0.01100 , 0.01300 , 0.02400 , 0.02600 , 0.04700 , 0.06800 , 0.08900 , 0.09500, 0.10200, 0.11400 , 0.14600 , 0.16300 , 0.17900 , 0.19500 , 0.22200 , 0.24900 , 0.27600 , 0.35800 , 0.43900 , 0.52000 , 0.60200 , 0.66700 , 0.68300 , 0.73200 , 0.76400 , 0.84600 , 0.89400 , 0.94300 , 0.95700 , 0.97200 , 0.98600 , 1.00000 -eps])
        self.npts           = self.r.size        
        self.chord_ref      = np.array([4.500000, 4.505882, 4.508235, 4.510588, 4.512941, 4.515294, 4.551818, 4.560909, 4.656364, 4.779091, 4.901765, 4.930000, 4.9900000, 5.034118, 5.155455, 5.193636, 5.222727, 5.226471, 5.213939, 5.181212, 5.124848, 4.883333, 4.576667, 4.225455, 3.825000, 3.472121, 3.380000, 3.099091, 2.900909, 2.357879, 2.019118, 1.653030, 1.542727, 1.420000, 1.183529, 0.500000])
        
        theta_ref           = np.array([11.130000, 11.130000, 11.130000, 11.130000, 11.130000, 11.130000, 11.130000, 11.130000, 11.130000, 11.130000, 11.130000, 11.130000, 11.130000, 11.130000, 11.130000, 11.130000, 11.130000, 10.837059, 10.186364, 9.572727, 9.006364, 7.504242, 6.240000, 5.132727, 4.147647, 3.444848, 3.280000, 2.804545, 2.502727, 1.783939, 1.382647, 0.987273, 0.874848, 0.756667, 0.551765, 0.000000])
        self.le_location    = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.499, 0.498, 0.483, 0.468, 0.453, 0.445, 0.440, 0.435, 0.41, 0.4, 0.39, 0.38, 0.378, 0.377, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375])
        t_spar      = [0.1051, 0.0891, 0.0851, 0.0841, 0.0831, 0.0881, 0.0941, 0.1001, 0.1101, 0.1201, 0.1191, 0.1291, 0.1141, 0.1141, 0.1141, 0.1141, 0.1141, 0.1141, 0.1141, 0.1141, 0.1141, 0.1141, 0.1141, 0.1141, 0.1141, 0.0991, 0.0891, 0.0791, 0.0691, 0.0591, 0.0491, 0.0191, 0.0191, 0.0191, 0.0191, 0.0191] 
        t_te        = [0.1051, 0.0861, 0.0751, 0.0641, 0.0531, 0.0481, 0.0441, 0.0401, 0.0401, 0.0401, 0.1001, 0.0951, 0.0801, 0.0801, 0.0801, 0.0921, 0.0421, 0.0921, 0.0921, 0.0921, 0.0801, 0.0771, 0.0491, 0.0291, 0.0141, 0.0141, 0.0141, 0.0141, 0.0141, 0.0141, 0.0141, 0.0141, 0.0141, 0.0141, 0.0141, 0.0141]
        self.thickness = np.array([100. , 100.,   99.25, 98.5,  97.75,  97.,   93.1,  92.4051586,   85.02312448,  77.67246576,  70.62348547,  63., 60.0, 57.0,   54.87, 50.47279154, 46.39520394,  42.86, 37.70304653, 34.23, 32.57464363,  29.42125572,  27.,   23.8275728,   20.7582483,   19.,   18.66669067,  18.,   18.,   18.,   18.,   18.,  18.,   18.,   18.,   18.])
        
        # idx_cylinder        = 10
        # self.r_cylinder     = self.r[idx_cylinder]
        # self.r_max_chord    = self.r[np.argmax(self.chord_ref)]
        # self.setRin()
        
        # myspline            = PchipInterpolator(self.r, self.chord_ref)
        # self.chord          = myspline(self.r_in)
               
        # myspline            = PchipInterpolator(self.r, theta_ref)
        # self.theta          = myspline(self.r_in)
        
        # myspline            = PchipInterpolator(self.r, np.zeros_like(self.r))
        # self.precurve       = myspline(self.r_in)
        # self.precurveT      = 0.0
        # self.presweep       = myspline(self.r_in)
        
        
        # myspline    = PchipInterpolator(self.r, t_spar)
        # self.spar_thickness = myspline(self.r_in)
        # myspline    = PchipInterpolator(self.r, t_te)
        # self.te_thickness = myspline(self.r_in)

        # # Load airfoil polar files
        # afpath = self.getAeroPath()
        # af_thicknesses  = np.array([18.0, 19.0, 27.0, 34.23, 42.86, 54.87, 63.00, 97.0 , 100.0])
        # airfoil_files_ref = ['NACA-64-618.dat', 'NACA-64-618_19.dat', 'FB-2700-0230.dat', 'FB-3423-0596.dat', 'FB-4286-0802.dat','FB-5487-1216.dat','FB-6300-1800.dat','SNL-100m-Ellipse97.dat','Cylinder.dat']
        # airfoil_files_ref = [os.path.join(afpath, af_file) for af_file in airfoil_files_ref]
        # self.set_polars(self.thickness, af_thicknesses, airfoil_files_ref)

        # # Now set best guess at airfoil cordinates along span without interpolating like the polar (this is just for plotting)
        # self.airfoil_files = ['']*self.npts
        # for k in range(self.npts):
            # self.airfoil_files[k] = airfoil_files_ref[np.argmin(np.abs(af_thicknesses - self.thickness[k]))]
        
        # self.sector_idx_strain_spar = np.array([1]*self.npts)
        # self.sector_idx_strain_te = np.array([4]*self.npts)
        
        
        # self.web1 = np.nan * np.ones(self.r.shape)
        # self.web2 = np.nan * np.ones(self.web1.shape)
        # self.web3 = np.nan * np.ones(self.web2.shape)
        # self.web4 = np.nan * np.ones(self.web3.shape)
        
        # # Control
        # self.control_Vin      = 3.
        # self.control_Vout     = 25.0
        # self.control_minOmega = 4.0
        # self.control_maxOmega = 8.0
        # self.control_tsr      = 9.66
        # self.control_pitch    = 0.00        
        

class BladeGeometry(ExplicitComponent):
    def initialize(self):
        self.options.declare('RefBlade')
    
    def setup(self):
        self.refBlade = RefBlade = self.options['RefBlade']
        npts = self.refBlade.npts
        
        # variables
        self.add_input('bladeLength', val=0.0, units='m', desc='blade length (if not precurved or swept) otherwise length of blade before curvature')
        self.add_input('r_max_chord', val=0.0, desc='location of max chord on unit radius')
        self.add_input('chord_in', val=np.zeros(NINPUT), units='m', desc='chord at control points')  # defined at hub, then at linearly spaced locations from r_max_chord to tip
        self.add_input('theta_in', val=np.zeros(NINPUT), units='deg', desc='twist at control points')  # defined at linearly spaced locations from r[idx_cylinder] to tip
        self.add_input('precurve_in', val=np.zeros(NINPUT), units='m', desc='precurve at control points')  # defined at same locations at chord, starting at 2nd control point (root must be zero precurve)
        self.add_input('presweep_in', val=np.zeros(NINPUT), units='m', desc='precurve at control points')  # defined at same locations at chord, starting at 2nd control point (root must be zero precurve)
        self.add_input('sparT_in', val=np.zeros(NINPUT), units='m', desc='thickness values of spar cap that linearly vary from non-cylinder position to tip')
        self.add_input('teT_in', val=np.zeros(NINPUT), units='m', desc='thickness values of trailing edge panels that linearly vary from non-cylinder position to tip')

        # parameters
        self.add_input('hubFraction', val=0.0, desc='hub location as fraction of radius')

        # Blade geometry outputs
        self.add_output('Rhub', val=0.0, units='m', desc='dimensional radius of hub')
        self.add_output('Rtip', val=0.0, units='m', desc='dimensional radius of tip')
        self.add_output('r_pts', val=np.zeros(npts), units='m', desc='dimensional aerodynamic grid')
        self.add_output('r_in', val=np.zeros(NINPUT), units='m', desc='Spline control points for inputs')
        self.add_output('max_chord', val=0.0, units='m', desc='maximum chord length')
        self.add_output('chord', val=np.zeros(npts), units='m', desc='chord at airfoil locations')
        self.add_output('theta', val=np.zeros(npts), units='deg', desc='twist at airfoil locations')
        self.add_output('precurve', val=np.zeros(npts), units='m', desc='precurve at airfoil locations')
        self.add_output('presweep', val=np.zeros(npts), units='m', desc='presweep at structural locations')
        self.add_output('sparT', val=np.zeros(npts), units='m', desc='dimensional spar cap thickness distribution')
        self.add_output('teT', val=np.zeros(npts), units='m', desc='dimensional trailing-edge panel thickness distribution')

        self.add_output('hub_diameter', val=0.0, units='m')
        
        self.add_discrete_output('airfoils', val=[], desc='Spanwise coordinates for aerodynamic analysis')
        self.add_output('le_location', val=np.zeros(npts), desc='Leading-edge positions from a reference blade axis (usually blade pitch axis). Locations are normalized by the local chord length. Positive in -x direction for airfoil-aligned coordinate system')
        self.add_output('chord_ref', val=np.zeros(npts), desc='Chord distribution for reference section, thickness of structural layup scaled with reference thickness (fixed t/c)')

        # Blade layup outputs
        self.add_discrete_output('materials', val=np.zeros(npts), desc='material properties of composite materials')
        
        self.add_discrete_output('upperCS', val=np.zeros(npts), desc='list of CompositeSection objections defining the properties for upper surface')
        self.add_discrete_output('lowerCS', val=np.zeros(npts), desc='list of CompositeSection objections defining the properties for lower surface')
        self.add_discrete_output('websCS', val=np.zeros(npts), desc='list of CompositeSection objections defining the properties for shear webs')
        self.add_discrete_output('profile', val=np.zeros(npts), desc='list of CompositeSection profiles')
        
        self.add_discrete_output('sector_idx_strain_spar', val=np.zeros(npts, dtype=np.int_), desc='Index of sector for spar (PreComp definition of sector)')
        self.add_discrete_output('sector_idx_strain_te', val=np.zeros(npts, dtype=np.int_), desc='Index of sector for trailing edge (PreComp definition of sector)')

        self.declare_partials('*', '*', method='fd', form='central', step=1e-6)
        
    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        Rhub = inputs['hubFraction'] * inputs['bladeLength']
        Rtip = Rhub + inputs['bladeLength']

        # make dimensional and evaluate splines
        outputs['Rhub']     = Rhub
        outputs['Rtip']     = Rtip
        outputs['hub_diameter'] = 2.0*Rhub
        outputs['r_pts']    = Rhub + (Rtip-Rhub)*self.refBlade.r
        
        # print(self.refBlade.r_cylinder)
        outputs['r_in']     = Rhub + (Rtip-Rhub)*np.r_[0.0, self.refBlade.r_cylinder, np.linspace(inputs['r_max_chord'], 1.0, NINPUT-2).flatten()]

        # Although the inputs get mirrored to outputs, this is still necessary so that the user can designate the inputs as design variables
        myspline = Akima(outputs['r_in'], inputs['chord_in'])
        outputs['max_chord'], _, _, _ = myspline.interp(Rhub + (Rtip-Rhub)*inputs['r_max_chord'])
        outputs['chord'], _, _, _ = myspline.interp(outputs['r_pts'])

        myspline = Akima(outputs['r_in'], inputs['theta_in'])
        outputs['theta'], _, _, _ = myspline.interp(outputs['r_pts'])

        myspline = Akima(outputs['r_in'], inputs['precurve_in'])
        outputs['precurve'], _, _, _ = myspline.interp(outputs['r_pts'])

        myspline = Akima(outputs['r_in'], inputs['presweep_in'])
        outputs['presweep'], _, _, _ = myspline.interp(outputs['r_pts'])

        myspline = Akima(outputs['r_in'], inputs['sparT_in'])
        outputs['sparT'], _, _, _ = myspline.interp(outputs['r_pts'])
        
        myspline = Akima(outputs['r_in'], inputs['teT_in'])
        outputs['teT'], _, _, _ = myspline.interp(outputs['r_pts'])
        
        # Setup paths
        strucpath = self.refBlade.getStructPath()
        materials = Orthotropic2DMaterial.listFromPreCompFile(os.path.join(strucpath, 'materials.inp'))

        # Now compute some layup properties, independent of which turbine it is
        npts = self.refBlade.npts
        upperCS = [0]*npts
        lowerCS = [0]*npts
        websCS  = [0]*npts
        profile = [0]*npts

        for i in range(npts):
            webLoc = []
            if not np.isnan(self.refBlade.web1[i]): webLoc.append(self.refBlade.web1[i])
            if not np.isnan(self.refBlade.web2[i]): webLoc.append(self.refBlade.web2[i])
            if not np.isnan(self.refBlade.web3[i]): webLoc.append(self.refBlade.web3[i])
            if not np.isnan(self.refBlade.web4[i]): webLoc.append(self.refBlade.web4[i])

            istr = str(i+1) if self.refBlade.name == '5MW' or self.refBlade.name == 'BAR_00'  else str(i)
            upperCS[i], lowerCS[i], websCS[i] = CompositeSection.initFromPreCompLayupFile(os.path.join(strucpath, 'layup_' + istr + '.inp'), webLoc, materials)
            profile[i] = Profile.initFromPreCompFile(os.path.join(strucpath, 'shape_' + istr + '.inp'))

        # Assign outputs
        discrete_outputs['airfoils']               = self.refBlade.airfoils
        outputs['le_location']            = self.refBlade.le_location
        discrete_outputs['upperCS']                = upperCS
        discrete_outputs['lowerCS']                = lowerCS
        discrete_outputs['websCS']                 = websCS
        discrete_outputs['profile']                = profile
        outputs['chord_ref']              = self.refBlade.chord_ref
        discrete_outputs['sector_idx_strain_spar'] = self.refBlade.sector_idx_strain_spar
        discrete_outputs['sector_idx_strain_te']   = self.refBlade.sector_idx_strain_te
        discrete_outputs['materials']              = materials
        
        
class Location(ExplicitComponent):
    def setup(self):
        self.add_input('hubHt', val=0.0, units='m', desc='Tower top hub height')
        self.add_output('wind_zvec', val=np.zeros(1), units='m', desc='Tower top hub height as vector')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        outputs['wind_zvec'] = np.array([ np.float(inputs['hubHt']) ])

    def compute_partials(self, inputs, J):
        J['wind_zvec','hubHt'] = np.ones(1)

        
class TurbineClass(ExplicitComponent):
    def setup(self):
        # parameters
        self.add_discrete_input('turbine_class', val=TURBINE_CLASS['I'], desc='IEC turbine class')

        # outputs should be constant
        self.add_output('V_mean', shape=1, units='m/s', desc='IEC mean wind speed for Rayleigh distribution')
        self.add_output('V_extreme1', shape=1, units='m/s', desc='IEC extreme wind speed at hub height')
        self.add_output('V_extreme50', shape=1, units='m/s', desc='IEC extreme wind speed at hub height')
        self.add_output('V_extreme_full', shape=2, units='m/s', desc='IEC extreme wind speed at hub height')
        
        self.declare_partials('*', '*', method='fd', form='central', step=1e-6)

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        self.turbine_class = discrete_inputs['turbine_class']

        if self.turbine_class == TURBINE_CLASS['I']:
            Vref = 50.0
        elif self.turbine_class == TURBINE_CLASS['II']:
            Vref = 42.5
        elif self.turbine_class == TURBINE_CLASS['III']:
            Vref = 37.5
        elif self.turbine_class == TURBINE_CLASS['IV']:
            Vref = 30.0

        outputs['V_mean'] = 0.2*Vref
        outputs['V_extreme1'] = 0.8*Vref
        outputs['V_extreme50'] = 1.4*Vref
        outputs['V_extreme_full'][0] = 1.4*Vref # for extreme cases TODO: check if other way to do
        outputs['V_extreme_full'][1] = 1.4*Vref



class RotorGeometry(Group):
    def initialize(self):
        self.options.declare('RefBlade')
        self.options.declare('topLevelFlag',default=False)
    
    def setup(self):
        RefBlade = self.options['RefBlade']
        topLevelFlag   = self.options['topLevelFlag']
        # assert isinstance(RefBlade, ReferenceBlade), 'Must pass in either NREL5MW or DTU10MW Reference Blade instance'

        # Independent variables that are unique to TowerSE
        if topLevelFlag:
            geomIndeps = IndepVarComp()
            geomIndeps.add_output('bladeLength', 0.0, units='m')
            geomIndeps.add_output('hubFraction', 0.0)
            geomIndeps.add_output('r_max_chord', 0.0)
            geomIndeps.add_output('chord_in', np.zeros(NINPUT),units='m')
            geomIndeps.add_output('theta_in', np.zeros(NINPUT), units='deg')
            geomIndeps.add_output('precurve_in', np.zeros(NINPUT), units='m')
            geomIndeps.add_output('presweep_in', np.zeros(NINPUT), units='m')
            geomIndeps.add_output('precurveTip', 0.0, units='m')
            geomIndeps.add_output('presweepTip', 0.0, units='m')
            geomIndeps.add_output('precone', 0.0, units='deg')
            geomIndeps.add_output('tilt', 0.0, units='deg')
            geomIndeps.add_output('yaw', 0.0, units='deg')
            geomIndeps.add_discrete_output('nBlades', 3)
            geomIndeps.add_discrete_output('downwind', False)
            geomIndeps.add_discrete_output('turbine_class', val=TURBINE_CLASS['I'], desc='IEC turbine class')
            geomIndeps.add_output('sparT_in', val=np.zeros(NINPUT), units='m', desc='spar cap thickness parameters')
            geomIndeps.add_output('teT_in', val=np.zeros(NINPUT), units='m', desc='trailing-edge thickness parameters')
            self.add_subsystem('geomIndeps', geomIndeps, promotes=['*'])

            
        # --- Rotor Definition ---
        self.add_subsystem('loc', Location(), promotes=['*'])
        self.add_subsystem('turbineclass', TurbineClass(), promotes=['turbine_class'])
        #self.add_subsystem('spline0', BladeGeometry(RefBlade))
        self.add_subsystem('spline', BladeGeometry(RefBlade=RefBlade), promotes=['*'])
        self.add_subsystem('geom', CCBladeGeometry(), promotes=['precone','precurveTip'])

        # connections to turbineclass
        #self.connect('turbine_class', 'turbineclass.turbine_class')

        # connections to spline0
        #self.connect('r_max_chord', 'spline0.r_max_chord')
        #self.connect('chord_in', 'spline0.chord_in')
        #self.connect('theta_in', 'spline0.theta_in')
        #self.connect('precurve_in', 'spline0.precurve_in')
        #self.connect('presweep_in', 'spline0.presweep_in')
        #self.connect('bladeLength', 'spline0.bladeLength')
        #self.connect('hubFraction', 'spline0.hubFraction')
        #self.connect('sparT_in', 'spline0.sparT_in')
        #self.connect('teT_in', 'spline0.teT_in')

        # connections to spline
        #self.connect('r_max_chord', 'spline.r_max_chord')
        #self.connect('chord_in', 'spline.chord_in')
        #self.connect('theta_in', 'spline.theta_in')
        #self.connect('precurve_in', 'spline.precurve_in')
        #self.connect('presweep_in', 'spline.presweep_in')
        #self.connect('bladeLength', 'spline.bladeLength')
        #self.connect('hubFraction', 'spline.hubFraction')
        #self.connect('sparT_in', 'spline.sparT_in')
        #self.connect('teT_in', 'spline.teT_in')

        # connections to geom
        self.connect('Rtip', 'geom.Rtip')
        #self.connect('precone', 'geom.precone')
        #self.connect('precurveTip', 'geom.precurveTip')


if __name__ == "__main__":

    # refBlade = DTU10MW()
    # refBlade = NREL5MW()
    refBlade = TUM3_35MW()
    rotor = RotorGeometry(refBlade)
