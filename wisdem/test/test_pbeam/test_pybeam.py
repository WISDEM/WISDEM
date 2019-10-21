import numpy as np
import numpy.testing as npt
import unittest
import wisdem.pBeam._pBEAM as pb

class TestPyBeam(unittest.TestCase):
    
    def testCantileverDeflection(self):
        # Test data from "Finite Element Structural Analysis", Yang, pg. 145
        E = 2.0
        I = 3.0
        L = 4.0
        p0 = 5.0

        nodes = 2

        Px = np.array([-p0, 0.0])
        Py = Pz = np.zeros(nodes)
        loads = pb.Loads(nodes, Px, Py, Pz)

        tip = pb.TipData()

        base = pb.BaseData(np.ones(6), 1.0)

        z = np.array([0.0, L])
        EIx = EIy = E*I*np.ones(nodes)
        EA = E*np.ones(nodes)
        GJ = rhoA = rhoJ = np.ones(nodes)
        sec = pb.SectionData(nodes, z, EA, EIx, EIy, GJ, rhoA, rhoJ)

        beam = pb.Beam(sec, loads, tip, base)

        (dx, dy, dz, dtheta_x, dtheta_y, dtheta_z) = beam.displacement()

        self.assertAlmostEqual(dx[0], 0.0, 8)
        self.assertAlmostEqual(dx[1], -p0 * L**3.0 / E / I * L / 30.0, 8)
        self.assertAlmostEqual(dtheta_y[0], 0.0, 8)
        self.assertAlmostEqual(dtheta_y[1],-p0 * L**3.0 / E / I * 1 / 24.0, 8)

    def testTaperedDeflections(self):
        # Test data from "Finite Element Structural Analysis", Yang, pg. 180
        E = 2.0
        I = 3.0
        L = 4.0
        P = 5.0

        nodes = 3

        Px = Py = Pz = np.zeros(nodes)
        loads = pb.Loads(nodes, Px, Py, Pz)

        tip = pb.TipData(0.0, np.zeros(3), np.zeros(6), [-P, 0.0, 0.0], np.zeros(3))

        base = pb.BaseData(np.ones(6), 1.0)

        z = np.array([0.0, 0.5*L, L])
        EIx = EIy = E*I*np.array([9.0, 5.0, 1.0])
        EA = E*np.ones(nodes)
        GJ = rhoA = rhoJ = np.ones(nodes)
        sec = pb.SectionData(nodes, z, EA, EIx, EIy, GJ, rhoA, rhoJ)

        beam = pb.Beam(sec, loads, tip, base)

        (dx, dy, dz, dtheta_x, dtheta_y, dtheta_z) = beam.displacement()

        tol = 1e-8
        tol_pct_1 = 0.17;
        tol_pct_2 = 0.77;
        self.assertAlmostEqual(dx[0], 0.0, 8)
        self.assertAlmostEqual(dx[-1], -0.051166*P*L**3/E/I, delta=tol_pct_1)
        self.assertAlmostEqual(dtheta_y[0], 0.0, 8)
        self.assertAlmostEqual(dtheta_y[-1], -0.090668*P*L**2/E/I, delta=tol_pct_2)

    def testFreqFree_FreeBeam_n1(self):
        # Test data from "Consistent Mass Matrix for Distributed Mass Systmes", John Archer,
        # Journal of the Structural Division Proceedings of the American Society of Civil Engineers,
        # pg. 168
        E = 2.0
        I = 3.0
        L = 4.0
        A = 5.0
        rho = 6.0

        n = 1
        nodes = n+1
        
        Px = Py = Pz = np.zeros(nodes)
        loads = pb.Loads(nodes, Px, Py, Pz)

        tip = pb.TipData()

        base = pb.BaseData()
        
        z = L * np.arange(nodes)
        EIx = EIy = E*I*np.ones(nodes)
        EA = GJ = rhoJ = np.ones(nodes)
        rhoA = rho*A*np.ones(nodes)
        sec = pb.SectionData(nodes, z, EA, EIx, EIy, GJ, rhoA, rhoJ)

        beam = pb.Beam(sec, loads, tip, base)

        nFreq = 100
        freq = beam.naturalFrequencies(nFreq)
        
        m = rho * A
        alpha = m * (n*L)**4.0 / (840.0 * E * I)

        tol_pct = 5e-6 * 100
        expect = np.sqrt(0.85714 / alpha) / (2*np.pi)
        self.assertAlmostEqual(freq[1], expect, delta=tol_pct)
        self.assertAlmostEqual(freq[2], expect, delta=tol_pct)
        expect = np.sqrt(10.0 / alpha) / (2*np.pi)
        self.assertAlmostEqual(freq[4], expect, delta=tol_pct)
        self.assertAlmostEqual(freq[5], expect, delta=tol_pct)


    def testFreqFree_FreeBeam_n2(self):
        # Test data from "Consistent Mass Matrix for Distributed Mass Systmes", John Archer,
        # Journal of the Structural Division Proceedings of the American Society of Civil Engineers,
        # pg. 168
        E = 2.0
        I = 3.0
        L = 4.0
        A = 5.0
        rho = 6.0

        n = 2
        nodes = n+1
        
        Px = Py = Pz = np.zeros(nodes)
        loads = pb.Loads(nodes, Px, Py, Pz)

        tip = pb.TipData()

        base = pb.BaseData()
        
        z = L * np.arange(nodes)
        EIx = EIy = E*I*np.ones(nodes)
        EA = GJ = rhoJ = np.ones(nodes)
        rhoA = rho*A*np.ones(nodes)
        sec = pb.SectionData(nodes, z, EA, EIx, EIy, GJ, rhoA, rhoJ)

        beam = pb.Beam(sec, loads, tip, base)

        nFreq = 100
        freq = beam.naturalFrequencies(nFreq)
        
        m = rho * A
        alpha = m * (n*L)**4.0 / (840.0 * E * I)

        tol_pct = 6e-6 * 100
        expect = np.sqrt(0.59858 / alpha) / (2*np.pi)
        self.assertAlmostEqual(freq[1], expect, delta=tol_pct)
        self.assertAlmostEqual(freq[2], expect, delta=tol_pct)
        expect = np.sqrt(5.8629 / alpha) / (2*np.pi)
        self.assertAlmostEqual(freq[5], expect, delta=tol_pct)
        self.assertAlmostEqual(freq[6], expect, delta=tol_pct)
        expect = np.sqrt(36.659 / alpha) / (2*np.pi)
        self.assertAlmostEqual(freq[8], expect, delta=tol_pct)
        self.assertAlmostEqual(freq[9], expect, delta=tol_pct)
        expect = np.sqrt(93.566 / alpha) / (2*np.pi)
        self.assertAlmostEqual(freq[10], expect, delta=tol_pct)
        self.assertAlmostEqual(freq[11], expect, delta=tol_pct)
        

    def testFreqFree_FreeBeam_n3(self):
        # Test data from "Consistent Mass Matrix for Distributed Mass Systmes", John Archer,
        # Journal of the Structural Division Proceedings of the American Society of Civil Engineers,
        # pg. 168
        E = 2.0
        I = 3.0
        L = 4.0
        A = 5.0
        rho = 6.0

        n = 3
        nodes = n+1
        
        Px = Py = Pz = np.zeros(nodes)
        loads = pb.Loads(nodes, Px, Py, Pz)

        tip = pb.TipData()

        base = pb.BaseData()
        
        z = L * np.arange(nodes)
        EIx = EIy = E*I*np.ones(nodes)
        EA = GJ = rhoJ = np.ones(nodes)
        rhoA = rho*A*np.ones(nodes)
        sec = pb.SectionData(nodes, z, EA, EIx, EIy, GJ, rhoA, rhoJ)

        beam = pb.Beam(sec, loads, tip, base)

        nFreq = 100
        freq = beam.naturalFrequencies(nFreq)
        
        m = rho * A
        alpha = m * (n*L)**4.0 / (840.0 * E * I)

        tol_pct = 6e-6 * 100
        expect = np.sqrt(0.59919 / alpha) / (2*np.pi)
        self.assertAlmostEqual(freq[1], expect, delta=tol_pct)
        self.assertAlmostEqual(freq[2], expect, delta=tol_pct)
        expect = np.sqrt(4.5750 / alpha) / (2*np.pi)
        self.assertAlmostEqual(freq[5], expect, delta=tol_pct)
        self.assertAlmostEqual(freq[6], expect, delta=tol_pct)
        expect = np.sqrt(22.010 / alpha) / (2*np.pi)
        self.assertAlmostEqual(freq[8], expect, delta=tol_pct)
        self.assertAlmostEqual(freq[9], expect, delta=tol_pct)
        expect = np.sqrt(70.920 / alpha) / (2*np.pi)
        self.assertAlmostEqual(freq[11], expect, delta=tol_pct)
        self.assertAlmostEqual(freq[12], expect, delta=tol_pct)
        expect = np.sqrt(265.91 / alpha) / (2*np.pi)
        self.assertAlmostEqual(freq[14], expect, delta=tol_pct)
        self.assertAlmostEqual(freq[15], expect, delta=tol_pct)
        expect = np.sqrt(402.40 / alpha) / (2*np.pi)
        self.assertAlmostEqual(freq[16], expect, delta=tol_pct)
        self.assertAlmostEqual(freq[17], expect, delta=tol_pct)

    def testBucklingEuler(self):
        # unit test data from Euler's buckling formula for a clamped/free beam

        E = 2.0
        I = 3.0
        L = 4.0

        n = 3
        nodes = n+1
        
        Px = Py = Pz = np.zeros(nodes)
        loads = pb.Loads(nodes, Px, Py, Pz)

        tip = pb.TipData()

        base = pb.BaseData(np.ones(6), 1.0)
        
        z = L * np.arange(nodes)
        EIx = EIy = E*I*np.ones(nodes)
        EA = GJ = rhoJ = rhoA = np.ones(nodes)
        sec = pb.SectionData(nodes, z, EA, EIx, EIy, GJ, rhoA, rhoJ)

        beam = pb.Beam(sec, loads, tip, base)

        (Pcr_x, Pcr_y) = beam.criticalBucklingLoads()

        Ltotal = n*L
        tol_pct = 0.011
        expect = E * I * (0.5*np.pi/Ltotal)**2.0
        self.assertAlmostEqual(Pcr_x, expect, 2)
        self.assertAlmostEqual(Pcr_y, expect, 2)

    def testShearBendingSimple(self):
        # Test data from "Mechanical of Materials", Gere, 6th ed., pg. 273
        # cantilevered beam with linear distributed load

        L = 10.0
        q0 = 3.0

        n = 1
        nodes = n+1

        tip = pb.TipData()

        base = pb.BaseData(np.ones(6), 1.0)
        
        z = np.arange(nodes, dtype=np.float64) * (L / n)
        EIx = EIy = EA = GJ = rhoJ = rhoA = np.ones(nodes)
        sec = pb.SectionData(nodes, z, EA, EIx, EIy, GJ, rhoA, rhoJ)

        Px = q0*(1 - z/L)
        Py = Pz = np.zeros(nodes)
        loads = pb.Loads(nodes, Px, Py, Pz)

        beam = pb.Beam(sec, loads, tip, base)

        (Vx, Vy, Fz, Mx, My, Tz) = beam.shearAndBending()

        tol_pct = 1e-8
        Vx_expect = np.polyval([q0*L/2.0, -q0*L, q0*L/2.0], [0.0, 1.0])
        My_expect = np.polyval([-q0*L*L/6.0, 3.0*q0*L*L/6.0, -3.0*q0*L*L/6.0, q0*L*L/6.0], [0.0, 1.0])
        npt.assert_almost_equal(Vx, Vx_expect)
        npt.assert_almost_equal(My, My_expect)

    def testShearBendingSimplePt(self):
        # Test data from "Mechanical of Materials", Gere, 6th ed., pg. 288
        # cantilevered beam with two point loads

        L = 10.0
        P1 = 2.0
        P2 = 3.0

        n = 3
        nodes = n+1

        tip = pb.TipData()

        base = pb.BaseData(np.ones(6), 1.0)
        
        z = np.arange(nodes, dtype=np.float64) * (L / n)
        EIx = EIy = EA = GJ = rhoJ = rhoA = np.ones(nodes)
        sec = pb.SectionData(nodes, z, EA, EIx, EIy, GJ, rhoA, rhoJ)

        Px = Py = Pz = Fy_pt = Fz_pt = Mx_pt = My_pt = Mz_pt = np.zeros(nodes)
        Fx_pt = np.zeros(nodes)
        Fx_pt[1] = -P2
        Fx_pt[3] = -P1
        loads = pb.Loads(nodes, Px, Py, Pz, Fx_pt, Fy_pt, Fz_pt, Mx_pt, My_pt, Mz_pt)

        beam = pb.Beam(sec, loads, tip, base)

        (Vx, Vy, Fz, Mx, My, Tz) = beam.shearAndBending()

        tol_pct = 1e-8;
        Vx_expect = np.polyval([0.0, 0.0, -P1-P2], [0.0])
        self.assertAlmostEqual(Vx[0], Vx_expect, 8)

        Vx_expect = np.polyval([0.0, 0.0, -P1], [0.0])
        self.assertAlmostEqual(Vx[1], Vx_expect, 8)
        self.assertAlmostEqual(Vx[2], Vx_expect, 8)

        b = L/3.0
        a = 2.0/3.0*L
        My_expect = np.polyval([0.0, 0.0, -P1*a + P1*L + P2*b, -P1*L - P2*b], [0.0])
        self.assertAlmostEqual(My[0], My_expect, 8)

        My_expect = np.polyval([0.0, 0.0, -0.5*P1*a + P1*a, -P1*a], [0.0])
        self.assertAlmostEqual(My[1], My_expect, 8)

        My_expect = np.polyval([0.0, 0.0, 0.5*P1*a, -0.5*P1*a], [0.0])
        self.assertAlmostEqual(My[2], My_expect, 8)

    def testOtherCalls(self):

        E = 2.0
        I = 3.0
        L = 4.0
        q0 = 5.0
        
        n = 3
        nodes = n+1
        
        tip = pb.TipData()

        base = pb.BaseData(np.ones(6), 1.0)
        
        z = L * np.arange(nodes)
        EIx = EIy = E*I*np.ones(nodes)
        EA = GJ = rhoJ = rhoA = np.ones(nodes)
        sec = pb.SectionData(nodes, z, EA, EIx, EIy, GJ, rhoA, rhoJ)

        Pz = q0*(1 - z/L)
        Py = Px = np.zeros(nodes)
        loads = pb.Loads(nodes, Px, Py, Pz)

        beam = pb.Beam(sec, loads, tip, base)

        badlist = [float('inf'), -float('inf'), float('nan'), 0.0, np.inf, -np.inf, np.nan]
        self.assertNotIn(beam.mass(), badlist)
        self.assertNotIn(beam.outOfPlaneMomentOfInertia(), badlist)

        npts = 10
        xv = yv = np.zeros(npts)
        zv = np.linspace(z[0], z[-1], npts)
        self.assertNotIn( beam.axialStrain(npts, xv, yv, zv).sum(), badlist)

    def testCurveFEM_FixedBeam_n1(self):
        # Test data from "Consistent Mass Matrix for Distributed Mass Systmes", John Archer,
        # Journal of the Structural Division Proceedings of the American Society of Civil Engineers,
        # pg. 168
        E = 2.0
        I = 3.0
        L = 4.0
        A = 5.0
        rho = 6.0
        hubR = 0.0
        omegaRPM = 0.0

        n = 1
        nodes = n+1
        
        z = L * np.arange(nodes)
        EIx = EIy = E*I*np.ones(nodes)
        EA = GJ = rhoJ = np.ones(nodes)
        rhoA = rho*A*np.ones(nodes)
        theta = precurv = presweep = np.zeros(nodes) #np.linspace(0.0, 3.0, nodes)

        mycurve = pb.CurveFEM(omegaRPM, theta, z, precurv, presweep, rhoA, True)
        freq = mycurve.frequencies(EA, EIx, EIy, GJ, rhoJ, nodes)[0]

        #truth = _curvefem.frequencies(omegaRPM, L, hubR, z/L, theta, rhoA, EIx, EIy, GJ, EA, rhoJ, precurv, presweep)
        #print "n1", truth.tolist()
        truth = np.array([0.012572866969753228, 0.012591740426479996, 0.015715395588976708, 0.015715395588976708, 0.02528529867098764, 0.02528529867098764, 0.06883002554331129, 0.06893334807998643, 0.09116741813672731, 0.09116741813672731, 0.1548391375450802, 0.1548391375450802])
        npt.assert_almost_equal(freq, truth)

    def testCurveFEM_FixedBeam_n2(self):
        # Test data from "Consistent Mass Matrix for Distributed Mass Systmes", John Archer,
        # Journal of the Structural Division Proceedings of the American Society of Civil Engineers,
        # pg. 168
        E = 2.0
        I = 3.0
        L = 4.0
        A = 5.0
        rho = 6.0
        hubR = 0.0
        omegaRPM = 0.0

        n = 2
        nodes = n+1
        
        z = L * np.arange(nodes)
        EIx = EIy = E*I*np.ones(nodes)
        EA = GJ = rhoJ = np.ones(nodes)
        rhoA = rho*A*np.ones(nodes)
        theta = precurv = presweep = np.zeros(nodes) #np.linspace(0.0, 3.0, nodes)

        mycurve = pb.CurveFEM(omegaRPM, theta, z, precurv, presweep, rhoA, True)
        freq = mycurve.frequencies(EA, EIx, EIy, GJ, rhoJ, nodes)[0]

        #truth = _curvefem.frequencies(omegaRPM, L, hubR, z/L, theta, rhoA, EIx, EIy, GJ, EA, rhoJ, precurv, presweep)
        #print "n2", truth.tolist()
        truth = np.array([0.003912151523474228, 0.003912151523474228, 0.005852976699216038, 0.012582302518525731, 0.02044674778626943, 0.02471287957376343, 0.02471287957376343, 0.025285550340850595, 0.025285550340850595, 0.032042058241816634, 0.06888168035399123, 0.0835842095344649, 0.0835842095344649, 0.09116764495898616, 0.09116764495898616, 0.11193550172703298, 0.24259762924185935, 0.24259762924185935])
        npt.assert_almost_equal(freq, truth)
        

    def testCurveFEM_FixedBeam_n3(self):
        # Test data from "Consistent Mass Matrix for Distributed Mass Systmes", John Archer,
        # Journal of the Structural Division Proceedings of the American Society of Civil Engineers,
        # pg. 168
        E = 2.0
        I = 3.0
        L = 4.0
        A = 5.0
        rho = 6.0
        hubR = 0.0
        omegaRPM = 0.0

        n = 3
        nodes = n+1
        
        z = L * np.arange(nodes)
        EIx = EIy = E*I*np.ones(nodes)
        EA = GJ = rhoJ = np.ones(nodes)
        rhoA = rho*A*np.ones(nodes)
        theta = precurv = presweep = np.zeros(nodes) #np.linspace(0.0, 3.0, nodes)

        mycurve = pb.CurveFEM(omegaRPM, theta, z, precurv, presweep, rhoA, True)
        freq = mycurve.frequencies(EA, EIx, EIy, GJ, rhoJ, nodes)[0]

        #truth = _curvefem.frequencies(omegaRPM, L, hubR, z/L, theta, rhoA, EIx, EIy, GJ, EA, rhoJ, precurv, presweep)
        #print "n3", truth.tolist()
        truth = np.array([0.0017380701632556908, 0.0017380701632562416, 0.003847212436706601, 0.010926964080897513, 0.010926964080897626, 0.012576854615234626, 0.012587751208210572, 0.02106152327278283, 0.02282612879817582, 0.02528527062288509, 0.02528527062288511, 0.03087567297329799, 0.030875672973298095, 0.06885185586578292, 0.06891150914730364, 0.06953079721389932, 0.06953079721389936, 0.09116748336205782, 0.09116748336205784, 0.12496139758839682, 0.1308572353442806, 0.13085723534428065, 0.2608788373591399, 0.2608788373591401])
        npt.assert_almost_equal(freq, truth)
        

    def testCurveFEM_FixedBeam_n3_withShape(self):
        # Test data from "Consistent Mass Matrix for Distributed Mass Systmes", John Archer,
        # Journal of the Structural Division Proceedings of the American Society of Civil Engineers,
        # pg. 168
        E = 2.0
        I = 3.0
        L = 4.0
        A = 5.0
        rho = 6.0
        hubR = 0.0
        omegaRPM = 12.0

        n = 3
        nodes = n+1
        
        z = L * np.arange(nodes)
        EIx = EIy = E*I*np.ones(nodes)
        EA = GJ = rhoJ = np.ones(nodes)
        rhoA = rho*A*np.ones(nodes)
        theta = precurv = presweep = np.linspace(0.0, 3.0, nodes)

        mycurve = pb.CurveFEM(omegaRPM, theta, z, precurv, presweep, rhoA, True)
        freq = mycurve.frequencies(EA, EIx, EIy, GJ, rhoJ, nodes)[0]

        #truth = _curvefem.frequencies(omegaRPM, L, hubR, z/L, theta, rhoA, EIx, EIy, GJ, EA, rhoJ, precurv, presweep)
        #print "n3 odd", truth.tolist()
        truth = np.array([0.00583962278555471, 0.013747011747362698, 0.019857000185537085, 0.022065457582918686, 0.022508587859798306, 0.06494224803437999, 0.07593365714823054, 0.08099501901331865, 0.08100059834010384, 0.11781473080899654, 0.1973084679918305, 0.44555268080407895, 0.48641244353481683, 0.7481252945270574, 0.7730228919707995, 1.0518193329242682, 1.0696357241743277, 1.4148111654258362, 1.4280996628456801, 1.9080030090517186, 1.9178773250559262, 0.0, 0.0, 0.0])
        npt.assert_almost_equal(freq[truth>0.0], truth[truth>0.0], decimal=5)

        
    def testCurveFEM_FreeBeam_n1(self):
        # Test data from "Consistent Mass Matrix for Distributed Mass Systmes", John Archer,
        # Journal of the Structural Division Proceedings of the American Society of Civil Engineers,
        # pg. 168
        E = 2.0
        I = 3.0
        L = 4.0
        A = 5.0
        rho = 6.0
        hubR = 0.0
        omegaRPM = 0.0

        n = 1
        nodes = n+1
        
        z = L * np.arange(nodes)
        EIx = EIy = E*I*np.ones(nodes)
        EA = GJ = rhoJ = np.ones(nodes)
        rhoA = rho*A*np.ones(nodes)
        theta = precurv = presweep = np.zeros(nodes) #np.linspace(0.0, 3.0, nodes)

        mycurve = pb.CurveFEM(omegaRPM, theta, z, precurv, presweep, rhoA, False)
        freq = mycurve.frequencies(EA, EIx, EIy, GJ, rhoJ, nodes)[0]

        
        m = rho * A
        alpha = m * (n*L)**4.0 / (840.0 * E * I)

        tol_pct = 5e-6 * 100
        expect = np.sqrt(0.85714 / alpha) / (2*np.pi)
        self.assertAlmostEqual(freq[1], expect, delta=tol_pct)
        self.assertAlmostEqual(freq[2], expect, delta=tol_pct)
        expect = np.sqrt(10.0 / alpha) / (2*np.pi)
        self.assertAlmostEqual(freq[4], expect, delta=tol_pct)
        self.assertAlmostEqual(freq[5], expect, delta=tol_pct)


    def testCurveFEM_FreeBeam_n2(self):
        # Test data from "Consistent Mass Matrix for Distributed Mass Systmes", John Archer,
        # Journal of the Structural Division Proceedings of the American Society of Civil Engineers,
        # pg. 168
        E = 2.0
        I = 3.0
        L = 4.0
        A = 5.0
        rho = 6.0
        hubR = 0.0
        omegaRPM = 0.0

        n = 2
        nodes = n+1
        
        z = L * np.arange(nodes)
        EIx = EIy = E*I*np.ones(nodes)
        EA = GJ = rhoJ = np.ones(nodes)
        rhoA = rho*A*np.ones(nodes)
        theta = precurv = presweep = np.zeros(nodes) #np.linspace(0.0, 3.0, nodes)

        mycurve = pb.CurveFEM(omegaRPM, theta, z, precurv, presweep, rhoA, False)
        freq = mycurve.frequencies(EA, EIx, EIy, GJ, rhoJ, nodes)[0]
        
        m = rho * A
        alpha = m * (n*L)**4.0 / (840.0 * E * I)

        tol_pct = 6e-6 * 100
        expect = np.sqrt(0.59858 / alpha) / (2*np.pi)
        self.assertAlmostEqual(freq[1], expect, delta=tol_pct)
        self.assertAlmostEqual(freq[2], expect, delta=tol_pct)
        expect = np.sqrt(5.8629 / alpha) / (2*np.pi)
        self.assertAlmostEqual(freq[5], expect, delta=tol_pct)
        self.assertAlmostEqual(freq[6], expect, delta=tol_pct)
        expect = np.sqrt(36.659 / alpha) / (2*np.pi)
        self.assertAlmostEqual(freq[8], expect, delta=tol_pct)
        self.assertAlmostEqual(freq[9], expect, delta=tol_pct)
        expect = np.sqrt(93.566 / alpha) / (2*np.pi)
        self.assertAlmostEqual(freq[10], expect, delta=tol_pct)
        self.assertAlmostEqual(freq[11], expect, delta=tol_pct)

        

    def testCurveFEM_FreeBeam_n3(self):
        # Test data from "Consistent Mass Matrix for Distributed Mass Systmes", John Archer,
        # Journal of the Structural Division Proceedings of the American Society of Civil Engineers,
        # pg. 168
        E = 2.0
        I = 3.0
        L = 4.0
        A = 5.0
        rho = 6.0
        hubR = 0.0
        omegaRPM = 0.0

        n = 3
        nodes = n+1
        
        z = L * np.arange(nodes)
        EIx = EIy = E*I*np.ones(nodes)
        EA = GJ = rhoJ = np.ones(nodes)
        rhoA = rho*A*np.ones(nodes)
        theta = precurv = presweep = np.zeros(nodes) #np.linspace(0.0, 3.0, nodes)

        mycurve = pb.CurveFEM(omegaRPM, theta, z, precurv, presweep, rhoA, False)
        freq = mycurve.frequencies(EA, EIx, EIy, GJ, rhoJ, nodes)[0]
        
        m = rho * A
        alpha = m * (n*L)**4.0 / (840.0 * E * I)

        tol_pct = 6e-6 * 100
        expect = np.sqrt(0.59919 / alpha) / (2*np.pi)
        self.assertAlmostEqual(freq[1], expect, delta=tol_pct)
        self.assertAlmostEqual(freq[2], expect, delta=tol_pct)
        expect = np.sqrt(4.5750 / alpha) / (2*np.pi)
        self.assertAlmostEqual(freq[5], expect, delta=tol_pct)
        self.assertAlmostEqual(freq[6], expect, delta=tol_pct)
        expect = np.sqrt(22.010 / alpha) / (2*np.pi)
        self.assertAlmostEqual(freq[8], expect, delta=tol_pct)
        self.assertAlmostEqual(freq[9], expect, delta=tol_pct)
        expect = np.sqrt(70.920 / alpha) / (2*np.pi)
        self.assertAlmostEqual(freq[11], expect, delta=tol_pct)
        self.assertAlmostEqual(freq[12], expect, delta=tol_pct)
        expect = np.sqrt(265.91 / alpha) / (2*np.pi)
        self.assertAlmostEqual(freq[14], expect, delta=tol_pct)
        self.assertAlmostEqual(freq[15], expect, delta=tol_pct)
        expect = np.sqrt(402.40 / alpha) / (2*np.pi)
        self.assertAlmostEqual(freq[16], expect, delta=tol_pct)
        self.assertAlmostEqual(freq[17], expect, delta=tol_pct)

        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestPyBeam))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
