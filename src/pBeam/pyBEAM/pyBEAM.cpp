//
//  pyBeam.cpp
//  pbeam
//
//  Created by Andrew Ning on 2/7/12.
//  Copyright (c) 2012 NREL. All rights reserved.
//

#include "myMath.h"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include "FEAData.h"
#include "Beam.h"
#include "CurveFEM.h"
#include <iostream>

namespace py = pybind11;


//MARK: ---------- WRAPPER FOR TIP DATA ---------------------

struct pyTipData{

  TipData tip;

  pyTipData(){}

  pyTipData(double m, const Vector &cm, const Vector &I, const Vector &F, const Vector &M){

    tip.m = m;

    tip.cm_offsetX = cm[0];
    tip.cm_offsetY = cm[1];
    tip.cm_offsetZ = cm[2];

    tip.Ixx = I[0];
    tip.Iyy = I[1];
    tip.Izz = I[2];
    tip.Ixy = I[3];
    tip.Ixz = I[4];
    tip.Iyz = I[5];

    tip.Fx = F[0];
    tip.Fy = F[1];
    tip.Fz = F[2];

    tip.Mx = M[0];
    tip.My = M[1];
    tip.Mz = M[2];
  }
};



//MARK: ---------- WRAPPER FOR BASE DATA ---------------------

struct pyBaseData{

  BaseData base;

  pyBaseData(){}

  pyBaseData(const Vector &k, double infinity){

    for (int i = 0; i < 6; i++) {
      base.k[i] = k[i];
      base.rigid[i] = (base.k[i] == infinity);
    }
  }

};


//MARK: ---------- WRAPPER FOR STRUCTURAL DATA ---------------------

struct pySectionData {

  SectionData sec;

  pySectionData(int np, const Vector &z_np, const Vector &EA_np,
                const Vector &EIxx_np, const Vector &EIyy_np, const Vector &GJ_np,
                const Vector &rhoA_np, const Vector &rhoJ_np){

    sec.nodes = np;

    sec.z = z_np;
    sec.EA = EA_np;
    sec.EIxx = EIxx_np;
    sec.EIyy = EIyy_np;
    sec.GJ = GJ_np;
    sec.rhoA = rhoA_np;
    sec.rhoJ = rhoJ_np;
  }
};

struct pyPolynomialSectionData {

  PolynomialSectionData sec;

  pyPolynomialSectionData(int nodes, const Vector &z_np, const Vector &nA_np, const Vector &nI_np,
			  const Matrix &EA_np, const Matrix &EIxx_np, const Matrix &EIyy_np,
			  const Matrix &GJ_np, const Matrix &rhoA_np, const Matrix &rhoJ_np){

    sec.nodes = nodes;

    sec.z = z_np;
    sec.EA.resize(sec.nodes-1);
    sec.EIxx.resize(sec.nodes-1);
    sec.EIyy.resize(sec.nodes-1);
    sec.GJ.resize(sec.nodes-1);
    sec.rhoA.resize(sec.nodes-1);
    sec.rhoJ.resize(sec.nodes-1);

    for (int i = 0; i < sec.nodes-1; i++){
      int nA = nA_np[i];
      int nI = nI_np[i];

      sec.EA[i]   = Poly(nA, EA_np.row(i));
      sec.rhoA[i] = Poly(nA, rhoA_np.row(i));
      sec.EIxx[i] = Poly(nI, EIxx_np.row(i));
      sec.EIyy[i] = Poly(nI, EIyy_np.row(i));
      sec.GJ[i]   = Poly(nI, GJ_np.row(i));
      sec.rhoJ[i] = Poly(nI, rhoJ_np.row(i));
    }
  }
};


//MARK: ---------- WRAPPER FOR LOADS DATA ---------------------

struct pyLoads {

  Loads loads;

  pyLoads(int np){

    loads.nodes = np;

    loads.Px.resize(loads.nodes);
    loads.Py.resize(loads.nodes);
    loads.Pz.resize(loads.nodes);
    loads.Fx.resize(loads.nodes);
    loads.Fy.resize(loads.nodes);
    loads.Fz.resize(loads.nodes);
    loads.Mx.resize(loads.nodes);
    loads.My.resize(loads.nodes);
    loads.Mz.resize(loads.nodes);


    loads.Px.setZero();
    loads.Py.setZero();
    loads.Pz.setZero();
    loads.Fx.setZero();
    loads.Fy.setZero();
    loads.Fz.setZero();
    loads.Mx.setZero();
    loads.My.setZero();
    loads.Mz.setZero();
  }

  pyLoads(int np, const Vector &Px_np, const Vector &Py_np,
	  const Vector &Pz_np){

    loads.nodes = np;

    loads.Px = Px_np;
    loads.Py = Py_np;
    loads.Pz = Pz_np;
    loads.Fx.resize(loads.nodes);
    loads.Fy.resize(loads.nodes);
    loads.Fz.resize(loads.nodes);
    loads.Mx.resize(loads.nodes);
    loads.My.resize(loads.nodes);
    loads.Mz.resize(loads.nodes);

    loads.Fx.setZero();
    loads.Fy.setZero();
    loads.Fz.setZero();
    loads.Mx.setZero();
    loads.My.setZero();
    loads.Mz.setZero();

  }

  pyLoads(int np, const Vector &Px_np, const Vector &Py_np, const Vector &Pz_np,
	  const Vector &Fx_np, const Vector &Fy_np, const Vector &Fz_np,
	  const Vector &Mx_np, const Vector &My_np, const Vector &Mz_np){

    loads.nodes = np;

    loads.Px = Px_np;
    loads.Py = Py_np;
    loads.Pz = Pz_np;
    loads.Fx = Fx_np;
    loads.Fy = Fy_np;
    loads.Fz = Fz_np;
    loads.Mx = Mx_np;
    loads.My = My_np;
    loads.Mz = Mz_np;
  }
};


struct pyPolynomialLoads {

  PolynomialLoads loads;

  pyPolynomialLoads(int nodes, const Vector &nP_np, const Matrix &Px_np, const Matrix &Py_np, const Matrix &Pz_np,
		    const Vector &Fx_np, const Vector &Fy_np, const Vector &Fz_np,
		    const Vector &Mx_np, const Vector &My_np, const Vector &Mz_np){

    loads.nodes = nodes;

    loads.Px.resize(loads.nodes-1);
    loads.Py.resize(loads.nodes-1);
    loads.Pz.resize(loads.nodes-1);
    loads.Fx = Fx_np;
    loads.Fy = Fy_np;
    loads.Fz = Fz_np;
    loads.Mx = Mx_np;
    loads.My = My_np;
    loads.Mz = Mz_np;

    for (int i = 0; i < loads.nodes-1; i++){
      int nP = nP_np[i];

      loads.Px[i] = Poly(nP, Px_np.row(i));
      loads.Py[i] = Poly(nP, Py_np.row(i));
      loads.Pz[i] = Poly(nP, Pz_np.row(i));
    }

  }
};


//MARK: ---------- WRAPPER FOR BEAM ---------------------



class pyBEAM {


  Beam *beam;

public:



  // MARK: ---------------- CONSTRUCTORS --------------------------

  // cylindrical shell sections
  pyBEAM(int nodes, const Vector &z_np, const Vector &d_np, const Vector &t_np,
	 const py::object &loads_o, const py::object &mat_o,
	 const py::object &tip_o, const py::object &base_o){

    pyLoads &loads = loads_o.cast<pyLoads&>();
    pyTipData& tip = tip_o.cast<pyTipData&>();
    pyBaseData& base = base_o.cast<pyBaseData&>();
    IsotropicMaterial& mat = mat_o.cast<IsotropicMaterial&>();

    beam = new Beam(z_np, d_np, t_np, loads.loads, mat, tip.tip, base.base);
  }

  //general sections
  pyBEAM(const py::object &section_o, const py::object &loads_o,
	 const py::object &tip_o, const py::object &base_o){
    
    pySectionData& sec = section_o.cast<pySectionData&>();
    pyLoads &loads = loads_o.cast<pyLoads&>();
    pyTipData& tip = tip_o.cast<pyTipData&>();
    pyBaseData& base = base_o.cast<pyBaseData&>();

    beam = new Beam(sec.sec, loads.loads, tip.tip, base.base);
  }

  //general sections as polynomials
  pyBEAM(const py::object &section_o, const py::object &loads_o,
	 const py::object &tip_o, const py::object &base_o, int dummy){

    pyPolynomialSectionData& sec = section_o.cast<pyPolynomialSectionData&>();
    pyPolynomialLoads &loads = loads_o.cast<pyPolynomialLoads&>();
    pyTipData& tip = tip_o.cast<pyTipData&>();
    pyBaseData& base = base_o.cast<pyBaseData&>();

    beam = new Beam(sec.sec, loads.loads, tip.tip, base.base);
  }

  ~pyBEAM(){delete beam;}


  // MARK: ---------------- COMPUTATIONS --------------------------

  /**
     Compute the mass of the structure

     Returns:
     mass

  **/
  double computeMass(){
    return beam->computeMass();
  }


  /**
     Compute the out-of-plane mass moment of inertia of the structure
     int(rho z^2, dV)

     Returns:
     mass moment of inertia

  **/
  double computeOutOfPlaneMomentOfInertia(){
    return beam->computeOutOfPlaneMomentOfInertia();
  }


  /**
     Compute the natural frequencies of the structure

     Arguments:
     n - number of natural frequencies to return (unless n exceeds the total DOF of the structure,
     in which case all computed natural frequencies will be returned.

     Return:
     freq - a numpy array of frequencies (not necessarily of length n as described above).

  **/
  Vector computeNaturalFrequencies(int n){

    Vector vec(n);
    beam->computeNaturalFrequencies(n, vec);

    return vec;
  }


  /**
     Compute the natural frequencies of the structure

     Arguments:
     n - number of natural frequencies to return (unless n exceeds the total DOF of the structure,
     in which case all computed natural frequencies will be returned.

     Return:
     freq - a numpy array of frequencies (not necessarily of length n as described above).

  **/
  py::tuple computeNaturalFrequenciesAndEigenvectors(int n){

    Vector freqs(n);
    Matrix eigenvectors(0, 0);
    beam->computeNaturalFrequencies(n, freqs, eigenvectors);

    return py::make_tuple(freqs, eigenvectors);
  }


  /**
     Compute the displacements of the structure (in global coordinate system)

     Return:
     a tuple containing (x, y, z, theta_x, theta_y, theta_z).
     each entry of the tuple is a numpy array describing the deflections for the given
     degree of freedom at each node.

  **/
  py::tuple computeDisplacement(){

    int nodes = beam->getNumNodes();

    Vector dx(nodes);
    Vector dy(nodes);
    Vector dz(nodes);
    Vector dtx(nodes);
    Vector dty(nodes);
    Vector dtz(nodes);

    beam->computeDisplacement(dx, dy, dz, dtx, dty, dtz);

    return py::make_tuple(dx, dy, dz, dtx, dty, dtz);

  }



  /**
     Estimates the minimum critical buckling loads due to axial loading in addition to any existing input loads.

     Return:
     a tuple (Pcr_x, Pcr_y) where
     Pcr_x - critical buckling load in the x-direction
     Pcr_y - critical buckling load in the y-direction

  **/
  py::tuple computeCriticalBucklingLoads(){

    double Pcr_x, Pcr_y;

    beam->computeMinBucklingLoads(Pcr_x, Pcr_y);

    return py::make_tuple(Pcr_x, Pcr_y);
  }




  /**
     Computes the axial strain along the structure at given locations.

     Arguments:
     Provide numpy array of points to evalute strain at (x,y,z)
     x(i) - x distance from elastic center for point i
     y(i) - y distance from elastic center for point i
     z(i) - z axial location of point i

     Return:
     epsilon_axial - a numpy array containing axial strain at point (x(i), y(i), z(i)) due to axial loads and bi-directional bending

  **/
  Vector computeAxialStrain(int length, Vector &x_np, Vector &y_np, Vector &z_np){
    
    Vector epsilon_axial(length);
    
    beam->computeAxialStrain(x_np, y_np, z_np, epsilon_axial);

    return epsilon_axial;
  }


  
  // in global 3D coordinate system
  py::tuple computeShearAndBending(){

    PolyVec Vx, Vy, Fz, Mx, My, Tz;
    beam->shearAndBending(Vx, Vy, Fz, Mx, My, Tz);

    int n = beam->getNumNodes() - 1;
    int nodes = n + 1;
    
    Vector Vx0(nodes), Vy0(nodes), Fz0(nodes), Mx0(nodes), My0(nodes), Tz0(nodes);

    for(int i = 0; i < n; i++) {
      Vx0[i] = Vx[i].eval(0.0);
      Vy0[i] = Vy[i].eval(0.0);
      Fz0[i] = Fz[i].eval(0.0);
      Mx0[i] = -My[i].eval(0.0);  // translate back to global coordinates
      My0[i] = Mx[i].eval(0.0);  // translate back to global coordinates
      Tz0[i] = Tz[i].eval(0.0);
    }
    Vx0[n] = Vx[n-1].eval(1.0);
    Vy0[n] = Vy[n-1].eval(1.0);
    Fz0[n] = Fz[n-1].eval(1.0);
    Mx0[n] = -My[n-1].eval(1.0);  // translate back to global coordinates
    My0[n] = Mx[n-1].eval(1.0);  // translate back to global coordinates
    Tz0[n] = Tz[n-1].eval(1.0);

    return py::make_tuple(Vx0, Vy0, Fz0, Mx0, My0, Tz0);
  }

};


//MARK: ---------- WRAPPER FOR CURVE FEM ---------------------

class pyCurveFEM{
  CurveFEM *mycurve;
  
public:
  pyCurveFEM(const double omegaRPM, const Vector &StrcTwst, const Vector &BldR,
	     const Vector &PrecrvRef, const Vector &PreswpRef, const Vector &BMassDen, const bool rootFix) {
    mycurve = new CurveFEM(omegaRPM, StrcTwst, BldR, PrecrvRef, PreswpRef, BMassDen, rootFix);
  }
  
  // ~pyCurveFEM(){delete mycurve;}
  
  // Vector compute_frequencies(Vector &ea, Vector &eix, Vector &eiy, Vector &gj, Vector &rhoJ) {
  //   return mycurve->frequencies(ea, eix, eiy, gj, rhoJ);
  // }
  ~pyCurveFEM(){delete mycurve;}  

  py::tuple compute_frequencies(Vector &ea, Vector &eix, Vector &eiy, Vector &gj, Vector &rhoJ, int &n) {
    int ndof;
      ndof = n*6;

    // int nnode;

    // nnode = BldR.size();
    // ndof = 6 * nnode;
    // Vector freqs(ndof);
    // Matrix eig_vec(ndof, ndof);

    Vector freqs(ndof);
    Matrix eig_vec(0, 0);

    mycurve->frequencies(ea, eix, eiy, gj, rhoJ, freqs, eig_vec);

    return py::make_tuple(freqs, eig_vec);
  }
};


  // py::tuple computeNaturalFrequenciesAndEigenvectors(int n){

  //   Vector freqs(n);
  //   Matrix eigenvectors(0, 0);
  //   beam->computeNaturalFrequencies(n, freqs, eigenvectors);

  //   return py::make_tuple(freqs, eigenvectors);
  // }


// MARK: --------- PYTHON MODULE ---------------

PYBIND11_MODULE(_pBEAM, m)
{
  m.doc() = "pBeam python plugin module";

  py::class_<pyBEAM>(m, "Beam")
    .def(py::init<int, Vector, Vector, Vector, py::object, py::object, py::object, py::object>())
    .def(py::init<py::object, py::object, py::object, py::object>())
    .def(py::init<py::object, py::object, py::object, py::object, int>())
    .def("mass", &pyBEAM::computeMass)
    .def("naturalFrequencies", &pyBEAM::computeNaturalFrequencies)
    .def("naturalFrequenciesAndEigenvectors", &pyBEAM::computeNaturalFrequenciesAndEigenvectors)
    .def("displacement", &pyBEAM::computeDisplacement)
    .def("criticalBucklingLoads", &pyBEAM::computeCriticalBucklingLoads)
    .def("axialStrain", &pyBEAM::computeAxialStrain)
    .def("outOfPlaneMomentOfInertia", &pyBEAM::computeOutOfPlaneMomentOfInertia)
    .def("shearAndBending", &pyBEAM::computeShearAndBending)
    ;

  py::class_<pyCurveFEM>(m, "CurveFEM")
    .def(py::init<double, Vector, Vector, Vector, Vector, Vector, bool>())
    .def("frequencies", &pyCurveFEM::compute_frequencies)
    ;

  py::class_<pyTipData>(m, "TipData")
    .def(py::init<double, Vector, Vector, Vector, Vector>())
    .def(py::init<>())
    ;

  py::class_<pyBaseData>(m, "BaseData")
    .def(py::init<Vector, double>())
    .def(py::init<>())
    ;

  py::class_<IsotropicMaterial>(m, "Material")
    .def(py::init<double, double, double>())
    .def_readwrite("E", &IsotropicMaterial::E)
    .def_readwrite("G", &IsotropicMaterial::G)
    .def_readwrite("rho", &IsotropicMaterial::rho)
    ;

  py::class_<pySectionData>(m, "SectionData")
    .def(py::init<int, Vector, Vector, Vector, Vector, Vector, Vector, Vector>());

  py::class_<pyPolynomialSectionData>(m, "PolySectionData")
    .def(py::init<int, Vector, Vector, Vector, Matrix, Matrix, Matrix, Matrix, Matrix, Matrix>());

  py::class_<pyLoads>(m, "Loads")
    .def(py::init<int, Vector, Vector, Vector, Vector, Vector, Vector, Vector, Vector, Vector>())
    .def(py::init<int, Vector, Vector, Vector>())
    .def(py::init<int>())
    ;

  py::class_<pyPolynomialLoads>(m, "PolyLoads")
    .def(py::init<int, Vector, Matrix, Matrix, Matrix, Vector, Vector, Vector, Vector, Vector, Vector>());
}
