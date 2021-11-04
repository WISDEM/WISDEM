import unittest
import numpy as np
import numpy.testing as npt
from wisdem.pyframe3dd import Frame, Options, NodeData, ElementData, ReactionData, StaticLoadCase

class TestBreakdown(unittest.TestCase):
    
    def testAll(self):
        # Nodal data
        ntop = 5
        nbase = 10
        nall = ntop + nbase - 1
        xyz_top = np.c_[np.linspace(0.0, -10, ntop), np.zeros(ntop), 100*np.ones(ntop)]
        xyz_base = np.c_[np.zeros((nbase,2)), np.linspace(0, 100, nbase)]
        xyz_all = np.vstack( (xyz_base, xyz_top[1:,:]) )
        
        zero_top = np.zeros(ntop)
        inode_top = np.arange(ntop, dtype=np.int_)+1
        node_top = NodeData(inode_top, xyz_top[:,0], xyz_top[:,1], xyz_top[:,2], zero_top)
        
        zero_base = np.zeros(nbase)
        inode_base = np.arange(nbase, dtype=np.int_)+1
        node_base = NodeData(inode_base, xyz_base[:,0], xyz_base[:,1], xyz_base[:,2], zero_base)
        
        zero_all = np.zeros(nall)
        inode_all = np.arange(nall, dtype=np.int_)+1
        node_all = NodeData(inode_all, xyz_all[:,0], xyz_all[:,1], xyz_all[:,2], zero_all)

        # Reactions
        rnode = np.array([1], dtype=np.int_)
        Kval = np.array([1], dtype=np.int_)
        rigid = 1
        reactions = ReactionData(rnode, Kval, Kval, Kval, Kval, Kval, Kval, rigid)

        # Element data
        top1 = np.ones(ntop-1)
        base1 = np.ones(nbase-1)
        all1 = np.ones(nall-1)
        ielem_top = np.arange(1, ntop)
        ielem_base = np.arange(1, nbase)
        ielem_all = np.arange(1, nall)
        N1_top = np.arange(ntop-1, dtype=np.int_)+1
        N1_base = np.arange(nbase-1, dtype=np.int_)+1
        N1_all = np.arange(nall-1, dtype=np.int_)+1
        N2_top = N1_top + 1
        N2_base = N1_base + 1
        N2_all = N1_all + 1
        Ax = 75.0
        Asx = Asy = 50.0
        J0 = 200.0
        Ix = Iy = 100.0
        E = 5e6
        G = 5e6
        density = 8e3
        elem_top = ElementData(ielem_top, N1_top, N2_top,
                               Ax*top1, Asx*top1, Asy*top1,
                               J0*top1, Ix*top1, Iy*top1,
                               E*top1, G*top1, 0.0*top1, density*top1)
        elem_base = ElementData(ielem_base, N1_base, N2_base,
                               Ax*base1, Asx*base1, Asy*base1,
                               J0*base1, Ix*base1, Iy*base1,
                               E*base1, G*base1, 0.0*base1, density*base1)
        elem_all = ElementData(ielem_all, N1_all, N2_all,
                               Ax*all1, Asx*all1, Asy*all1,
                               J0*all1, Ix*all1, Iy*all1,
                               E*all1, G*all1, 0.0*all1, density*all1)

        # parameters
        shear = False  # 1: include shear deformation
        geom = False  # 1: include geometric stiffness
        dx = -1.0  # x-axis increment for internal forces
        options = Options(shear, geom, dx)

        frame_top = Frame(node_top, reactions, elem_top, options)
        frame_base = Frame(node_base, reactions, elem_base, options)
        frame_all = Frame(node_all, reactions, elem_all, options)

        # load case
        gx = 0.0
        gy = 0.0
        gz = -9.81
        load_top = StaticLoadCase(gx, gy, gz)
        load_base = StaticLoadCase(gx, gy, gz)
        load_all = StaticLoadCase(gx, gy, gz)

        # pseudo-rotor loads
        nF_top = [inode_top[-1]]
        nF_base = [inode_base[-1]]
        nF_all = [inode_all[-1]]
        F = 1e5
        M = 1e6
        load_top.changePointLoads(nF_top, [F], [F], [F], [M], [M], [M])
        load_all.changePointLoads(nF_all, [F], [F], [F], [M], [M], [M])

        frame_top.addLoadCase(load_top)
        frame_all.addLoadCase(load_all)

        # Added mass
        AN_top = [inode_top[-2]]
        AN_all = [inode_all[-2]]
        EMs = np.array([1e6])
        EMxx = EMyy = EMzz = EMxy = EMxz = EMyz = np.array([0.0])
        rhox = rhoy = rhoz = np.array([0.0])
        addGravityLoad = True
        frame_top.changeExtraNodeMass(AN_top, EMs, EMxx, EMyy, EMzz, EMxy, EMxz, EMyz, rhox, rhoy, rhoz, addGravityLoad)
        frame_all.changeExtraNodeMass(AN_all, EMs, EMxx, EMyy, EMzz, EMxy, EMxz, EMyz, rhox, rhoy, rhoz, addGravityLoad)

        # Run first models
        disp_top, forces_top, rxns_top, _, _, modal_top = frame_top.run()
        disp_all, forces_all, rxns_all, _, _, modal_all = frame_all.run()

        # Transfer loads to base
        load_base.changePointLoads(nF_base,
                                   [-rxns_top.Fx],
                                   [-rxns_top.Fy],
                                   [-rxns_top.Fz],
                                   [-rxns_top.Mxx],
                                   [-rxns_top.Myy],
                                   [-rxns_top.Mzz],
                                   )
        frame_base.addLoadCase(load_base)
        disp_base, forces_base, rxns_base, _, _, modal_base = frame_base.run()

        npt.assert_almost_equal(rxns_all.Fx, rxns_base.Fx, decimal=3)
        npt.assert_almost_equal(rxns_all.Fy, rxns_base.Fy, decimal=3)
        npt.assert_almost_equal(rxns_all.Fz, rxns_base.Fz, decimal=3)
        npt.assert_almost_equal(rxns_all.Mxx, rxns_base.Mxx, decimal=2)
        npt.assert_almost_equal(rxns_all.Myy, rxns_base.Myy, decimal=2)
        npt.assert_almost_equal(rxns_all.Mzz, rxns_base.Mzz, decimal=2)
        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestBreakdown))
    return suite


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
    

