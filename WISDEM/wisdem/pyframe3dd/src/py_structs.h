/* ----------------
 S. Andrew Ning
 Nov 1, 2013
 Structs used for passing variables in/out from Python (or from other C code)
---------------- */


// --------------
// General Inputs
// --------------

typedef struct {

    int nN;
    int *N;
    double *x, *y, *z, *r;

} Nodes;


typedef struct {

    int nK;
    int* N;
    double *Kx, *Ky, *Kz, *Ktx, *Kty, *Ktz;
    double rigid;

} Reactions;


typedef struct {

    int nE;
    int *EL, *N1, *N2;
    double *Ax, *Asy, *Asz, *Jx, *Iy, *Iz, *E, *G, *roll, *density;

} Elements;


typedef struct {

    int shear, geom;
    double exagg_static, dx;

} OtherElementData;


// --------------
// Load Inputs
// --------------


typedef struct {
    int nF;
    int *N;
    double *Fx, *Fy, *Fz, *Mxx, *Myy, *Mzz;

} PointLoads;


typedef struct {
    int nU;
    int *EL;
    double *Ux, *Uy, *Uz;

} UniformLoads;


typedef struct {
    int nW;
    int *EL;
    double *xx1, *xx2, *wx1, *wx2;
    double *xy1, *xy2, *wy1, *wy2;
    double *xz1, *xz2, *wz1, *wz2;

} TrapezoidalLoads;


typedef struct {
    int nP;
    int *EL;
    double *Px, *Py, *Pz, *x;

} ElementLoads;


typedef struct {
    int nT;
    int *EL;
    double *a, *hy, *hz, *Typ, *Tym, *Tzp, *Tzm;

} TemperatureLoads;


typedef struct {
    int nD;
    int *N;
    double *Dx, *Dy, *Dz, *Dxx, *Dyy, *Dzz;

} PrescribedDisplacements;


typedef struct{

    double gx, gy, gz;
    PointLoads pointLoads;
    UniformLoads uniformLoads;
    TrapezoidalLoads trapezoidalLoads;
    ElementLoads elementLoads;
    TemperatureLoads temperatureLoads;
    PrescribedDisplacements prescribedDisplacements;

} LoadCase;


// --------------
// Dynamic Inputs
// --------------


typedef struct {
    int nM;
    int Mmethod, lump;
    double tol, shift, exagg_modal;

} DynamicData;


typedef struct {
    int nI;
    int *N;
    double *EMs, *EMx, *EMy, *EMz, *EMxy, *EMxz, *EMyz;
    double *rhox, *rhoy, *rhoz;

} ExtraInertia;


typedef struct {
    int nX;
    int *EL;
    double *EMs;

} ExtraMass;


typedef struct {
    int Cmethod, nC;
    int *N;
    double *cx, *cy, *cz, *cxx, *cyy, *czz;
    int *m;

} Condensation;



// --------------
// Static Data Outputs
// --------------


typedef struct{

    int *node;
    double *x, *y, *z, *xrot, *yrot, *zrot;

} Displacements;


typedef struct{

    int *element;
    int *node;
    double *Nx, *Vy, *Vz, *Txx, *Myy, *Mzz;

} Forces;


typedef struct{

    int *node;
    double *Fx, *Fy, *Fz, *Mxx, *Myy, *Mzz;

} ReactionForces;


// --------------
// Internal Force Outputs
// --------------


typedef struct {

    double *x, *Nx, *Vy, *Vz, *Tx, *My, *Mz, *Dx, *Dy, *Dz, *Rx;

} InternalForces;




// --------------
// Modal Outputs
// --------------


typedef struct {
    double *total_mass, *struct_mass;
    int *N;
    double *xmass, *ymass, *zmass, *xinrta, *yinrta, *zinrta;

} MassResults;

typedef struct {
    double *freq;
    double *xmpf, *ympf, *zmpf;
    int *N;
    double *xdsp, *ydsp, *zdsp, *xrot, *yrot, *zrot;

} ModalResults;


