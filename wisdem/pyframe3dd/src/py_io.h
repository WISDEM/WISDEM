/*
S. Andrew Ning
Nov 1, 2013
*/

#include "common.h"
#include <time.h>
#include "microstran/vec3.h"
#include "py_structs.h"

#include <stdio.h>





/**
    Read node coordinate data
*/
int read_node_data (
    Nodes *nodes,   /**node struct            */
    int nN,     /**< number of nodes                */
    vec3 *xyz,  /**< XYZ coordinates of each node       */
    float *rj   /**< rigid radius of each node          */
);

/**
    Read frame element property data
*/
int read_frame_element_data (
    Elements *elements, // element data
    int nN,     /**< number of nodes                */
    int nE,     /**< number of frame elements           */
    vec3 *xyz,  /**< XYZ coordinates of each node       */
    float *rj,  /**< rigid radius of each node          */
    double *L, double *Le,  /**< length of each frame element, effective */
    int *N1, int *N2,   /**< node connectivity          */
    float *Ax, float *Asy, float *Asz,  /**< section areas  */
    float *Jx, float *Iy, float *Iz,    /**< section inertias   */
    float *E, float *G, /**< elastic moduli and shear moduli    */
    float *p,   /**< roll angle of each frame element (radians) */
    float *d    /**< mass density of each frame element     */
);


/**
    Read data controlling certain aspects of the analysis
*/
int read_run_data (
    OtherElementData *other, // struct
    int *shear, /**< 1: include shear deformations, 0: don't    */
    int *geom,  /**< 1: include geometric stiffness, 0: don't   */
    double *exagg_static,/**< factor for static displ. exaggeration */
    float *dx  /**< frame element increment for internal forces*/
);


/**
    Read fixed node displacement boundary conditions
*/
int read_reaction_data(
    Reactions *reactions,  // struct
    int DoF,    /**< number of degrees of freedom       */
    int nN,     /**< number of nodes                */
    int *nR,    /**< number of nodes with reactions     */
    int *q,     /**< q[i]=0: DoF i is fixed, q[i]=1: DoF i is free */
    int *r,     /**< r[i]=1: DoF i is fixed, r[i]=0: DoF i is free */
    int *sumR,  /**< sum of vector R                */
    int verbose, /**< 1: copious screen output; 0: none      */
    int geom,
    float *EKx, float *EKy, float *EKz,  /* extra stiffness */
    float *EKtx, float *EKty, float *EKtz    
);


/**
    read load information data, form un-restrained load vector
*/
int read_and_assemble_loads(
    LoadCase* loadcases, //struct
    int nN,     /**< number of nodes                */
    int nE,     /**< number of frame elements           */
    int nL,     /**< number of load cases           */
    int DoF,    /**< number of degrees of freedom       */
    vec3 *xyz,  /**< XYZ coordinates of each node       */
    double *L, double *Le,  /**< length of each frame element, effective */
    int *N1, int *N2,   /**< node connectivity          */
    float *Ax, float *Asy, float *Asz,  /**< section areas  */
    float *Iy, float *Iz,   /**< section inertias           */
    float *E, float *G, /**< elastic moduli and shear moduli    */
    float *p,   /**< roll angle of each frame element (radians) */
    float *d,  /**< mass density of each frame element      */
    float *gX, /**< gravitational acceleration in global X each load case */
    float *gY, /**< gravitational acceleration in global Y each load case */
    float *gZ, /**< gravitational acceleration in global Z each load case */
    int *r,     /**< r[i]=1: DoF i is fixed, r[i]=0: DoF i is free */
    int shear,  /**< 1: include shear deformations, 0: don't    */
    int *nF,        /**< number of concentrated node loads */
    int *nU,        /**< number of uniformly distributed loads */
    int *nW,        /**< number of trapezoidaly distributed loads */
    int *nP,        /**< number of concentrated point loads */
    int *nT,        /**< number of temperature loads    */
    int *nD,        /**< number of prescribed displacements */
    double **Q,     /**< frame element end forces, every beam */
    double **F_temp,    /**< thermal loads          */
    double **F_mech,    /**< mechanical loads           */
    double *Fo,        /**< thermal loads + mechanical loads   */
    float ***U,     /**< uniformally distributed loads  */
    float ***W,     /**< trapezoidally distributed loads    */
    float ***P,     /**< concentrated point loads       */
    float ***T,     /**< temperature loads          */
    float **Dp,     /**< prescribed displacements at rctns  */
    double ***feF_mech, /**< fixed end forces for mechanical loads */
    double ***feF_temp, /**< fixed end forces for temperature loads */
    int verbose     /**< 1: copious output to screen, 0: none */
);


/**
    read member densities and extra inertial mass data
*/
int read_mass_data(
    DynamicData *dynamic, ExtraInertia *extraInertia, ExtraMass *extraMass, // structs
    int nN, int nE, /**< number of nodes, number of frame elements */
    int *nI,    /**< number of nodes with extra inertia */
    int *nX,    /**< number of elements with extra mass     */
    float *d, float *EMs, /**< density, extra frame element mass    */
    float *NMs, float *NMx, float *NMy, float *NMz, /**< node inertia*/
    double *L,  /**< length of each frame element       */
    float *Ax,  /**< cross section area of each frame element   */
    double *total_mass, /**< total mass of structure and extra mass */
    double *struct_mass,    /**< mass of structural elements    */
    int *nM,    /**< number of modes to find            */
    int *Mmethod,   /**< modal analysis method          */
    int *lump,  /**< 1: use lumped mass matrix, 0: consistent mass */
    double *tol,    /**< convergence tolerance for mode shapes  */
    double *shift,  /**< frequency shift for unrestrained frames    */
    double *exagg_modal, /**< exaggerate modal displacements    */
    int *anim,  /**< list of modes to be graphically animated   */
    float *pan, /**< 1: pan viewpoint during animation, 0: don't */
    int verbose,    /**< 1: copious output to screen, 0: none   */
    int debug   /**< 1: debugging output to screen, 0: none */
);


/**
    read matrix condensation information
*/
int read_condensation_data(
    Condensation *condensation, //struct
    int nN, int nM,     /**< number of nodes, number of modes   */
    int *nC,    /**< number of nodes with condensed DoF's   */
    int *Cdof,  /**< list of DoF's retained in condensed model  */
    int *Cmethod,   /**< matrix conden'n method, static, Guyan, dynamic*/
    int *c,     /**< list of retained degrees of freedom    */
    int *m,     /**< list of retained modes in dynamic condensation */
    int verbose /**< 1: copious output to screen, 0: none   */
);


/**
    save node displacements and member end forces in a text file    9sep08
*/
void write_static_results(
    Displacements* displacements, Forces* forces, ReactionForces* reactionForces, //structs
    Reactions* reactions, int nR,
    int nN, int nE, int nL, int lc, int DoF,
    int *N1, int *N2,
    double *F, double *D, int *r, double **Q,
    double err, int ok
);


/**
    calculate frame element internal forces, Nx, Vy, Vz, Tx, My, Mz
    calculate frame element local displacements, Rx, Dx, Dy, Dz
    write internal forces and local displacements to an output data file
    4jan10
*/
void write_internal_forces(
    InternalForces **internalForces, // array of arrays of structs
    int lc,     /**< load case number               */
    int nL,     /**< number of static load cases        */
    float dx,   /**< increment distance along local x axis      */
    vec3 *xyz,  /**< XYZ locations of each node                */
    double **Q, /**< frame element end forces                   */
    int nN,     /**< number of nodes                           */
    int nE,     /**< number of frame elements                   */
    double *L,  /**< length of each frame element               */
    int *N1, int *N2, /**< node connectivity                       */
    float *Ax,  /**< cross sectional area                       */
    float *Asy, float *Asz, /**< effective shear area               */
    float *Jx,  /**< torsional moment of inertia             */
    float *Iy, float *Iz,   /**< bending moment of inertia          */
    float *E, float *G, /**< elastic and shear modulii          */
    float *p,   /**< roll angle, radians                        */
    float *d,   /**< mass density                               */
    float gX, float gY, float gZ,   /**< gravitational acceleration */
    int nU,     /**< number of uniformly-distributed loads  */
    float **U,  /**< uniformly distributed load data            */
    int nW,     /**< number of trapezoidally-distributed loads  */
    float **W,  /**< trapezoidally distributed load data        */
    int nP,     /**< number of internal point loads     */
    float **P,  /**< internal point load data                   */
    double *D,  /**< node displacements                        */
    int shear,  /**< shear deformation flag                     */
    double error    /**< RMS equilibrium error          */
);


/**
    save modal frequencies and mode shapes          16aug01
*/
void write_modal_results(
    MassResults* massR, ModalResults* modalR, //structs
    int nN, int nE, int nI, int DoF,
    double **M, double *f, double **V,
    double total_mass, double struct_mass,
    int iter, int sumR, int nM,
    double shift, int lump, double tol, int ok
);



/** print a set of dots (periods) */
void dots ( FILE *fp, int n );


/** EVALUATE -  displays a randomly-generated evaluation message.  */
// void evaluate (  float error, float rms_resid, float tol, int geom );

