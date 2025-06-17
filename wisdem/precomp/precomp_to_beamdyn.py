import numpy as np

# --------------------------------------------------------------------------------}
# --- Tools to roto-translate 6x6 stiffness and mass matrix
# --------------------------------------------------------------------------------{
class TransformCrossSectionMatrix(object):

    def CrossSectionTranslationMatrix(self, x, y):
        T = np.eye(6)
        T[0,5] = y
        T[1,5] = -x
        T[2,3] = -y
        T[2,4] = x
        return T

    def CrossSectionRotationMatrix(self, alpha):
        c=np.cos(alpha)
        s=np.sin(alpha)
        R1=[[c,s,0],
            [-s,c,0],
            [0,0,1]]
        R=np.vstack((np.hstack((R1, np.zeros((3,3)))),
           np.hstack((np.zeros((3,3)), R1))))
        return R

    def CrossSectionRotoTranslationMatrix(self, M1, x, y, alpha):
        # Rotation
        R = self.CrossSectionRotationMatrix(alpha)
        M2 = R.T @ M1 @ R
        # Translation
        T = self.CrossSectionTranslationMatrix(x, y)
        M3 = T.T @ M2 @ T
        return M3

def pc2bd_K(EA, EIxx, EIyy, EIxy, EA_EIxx, EA_EIyy, EIxx_GJ, EIyy_GJ, EA_GJ, GJ, rhoJ, edge_iner, flap_iner, x_tc, y_tc, kxs = 1., kys = 0.6):
    """
    Given PreComp cross-sectional stiffness inputs, 
    returns 6x6 stiffness matrix at the x=0 y=0
    PreComp defines its outputs with respect to the xe and ye axes,
    which are centered at the elastic center and are rotated by the aerodynamic twist
    The function builds the 6x6 matrix and the elastic center and
    translates it to the origin
    Notes:
        - xe and ye are NOT the principal axes (we have xy cross terms)
        - in the PreComp manual there seems to be some confusion between shear 
          and tension centers. The latter should be the elastic center, but 
          PreComp indicates the former as the elastic center. 
          Nonetheless, the two centers are very close to each other
        - PreComp does not estimate shear terms. These are here estimated
    INPUTS:
        - EA: axial stiffness
        - EIxx: edge stiffness
        - EIyy: flap stiffness
        - EIxy: edge-flap cross term
        - EA_EIxx: axial-edge cross term
        - EA_EIyy: axial-flap cross term
        - EIxx_GJ: edge-torsion cross term
        - EIyy_GJ: flap-torsion cross term
        - EA_GJ: axial-torsion cross-term
        - GJ: torsional stiffness
        - rhoJ: polar moment of inertia, used to estimate shear stiffness terms
        - A: cross sectional area, used to estimate shear stiffness terms
        - x_tc: x coordinate of tension (elastic) center
        - y_tc: y coordinate of tension (elastic) center
        - kxs: stiffness shear along x (default = 1.)
        - kys: stiffness shear along y (default = 0.6)
    """
    G_est = GJ / rhoJ
    E_est_xx = EIxx / edge_iner
    E_est_yy = EIyy / flap_iner
    A_est_xx = EA / E_est_xx
    A_est_yy = EA / E_est_yy
    
    # Stiffness matrix built at the elastic center by PreComp. 
    # Note that shear terms A_est_xx and A_est_yy are flipped 
    # as they match better with the real values from ANBA
    K_sc = np.array([
        [G_est*A_est_yy  , 0.          , 0.      , 0.      , 0.      , 0.     ],
        [0.           , G_est*A_est_xx , 0.      , 0.      , 0.      , 0.     ],
        [0.           , 0.              , EA      , EA_EIxx , EA_EIyy , EA_GJ  ],
        [0.           , 0.              , EA_EIxx , EIxx    , EIxy    , EIxx_GJ],
        [0.           , 0.              , EA_EIyy , EIxy    , EIyy    , EIyy_GJ],
        [0.           , 0.              , EA_GJ   , EIxx_GJ , EIyy_GJ , GJ     ],
    ])
    
    # Translate matrix back to origin
    transform = TransformCrossSectionMatrix()
    T = transform.CrossSectionTranslationMatrix(-x_tc, -y_tc)
    K_ref = T.T @ K_sc @ T 

    return K_ref

def pc2bd_I(rhoA, edge_iner, flap_iner, rhoJ, x_cg, y_cg, Tw_iner, aero_twist):
    """
    Given PreComp cross-sectional inertia inputs, 
    returns 6x6 inertia matrix at the x=0 y=0
    PreComp defines its outputs with respect to the principal 
    inertia axes xg and yg, which are centered at the center of mass
    The function builds the 6x6 matrix at the cg and 
    rototranslates it to the ref axis (chord aligned)
    INPUTS:
        - rhoA: unit mass
        - edge_iner: edgewise mass moment of inertia
        - flap_iner: flapwise mass moment of inertia
        - rhoJ: polar moment of inertia
        - x_cg: x coordinate of mass center
        - y_cg: y coordinate of mass center      
        - Tw_iner: Orientation of the section 
                   principal inertia axes with respect 
                   the blade reference plane (rad)
        - aero_twist: aerodynamic twist (rad)
    """
    
    # Inertia matrix built at the mass center by PreComp

    I_cg = np.array([
        [rhoA , 0.   , 0.   , 0.        , 0.        , 0. ],
        [0.   , rhoA , 0.   , 0.        , 0.        , 0. ],
        [0.   , 0.   , rhoA , 0.        , 0.        , 0. ],
        [0.   , 0.   , 0.   , edge_iner , 0.        , 0. ],
        [0.   , 0.   , 0.   , 0.        , flap_iner , 0. ],
        [0.   , 0.   , 0.   , 0.        , 0.        , rhoJ ],
    ])
    
    transform = TransformCrossSectionMatrix()
    I_ref = transform.CrossSectionRotoTranslationMatrix(
        I_cg, 
        -x_cg, 
        -y_cg, 
        aero_twist - Tw_iner,
        )
    
    # Zero out some terms that must be exactly zero in BeamDyn
    I_ref[0,1] = 0.
    I_ref[0,2] = 0.
    I_ref[1,0] = 0.
    I_ref[1,2] = 0.
    I_ref[2,0] = 0.
    I_ref[2,1] = 0.

    return I_ref