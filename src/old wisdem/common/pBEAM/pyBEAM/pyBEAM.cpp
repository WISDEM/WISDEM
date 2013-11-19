//
//  pyBeam.cpp
//  pbeam
//
//  Created by Andrew Ning on 2/7/12.
//  Copyright (c) 2012 NREL. All rights reserved.
//

#include <boost/python.hpp>
#include "myMath.h"
#include "FEAData.h"
#include "Beam.h"


namespace bp = boost::python;
namespace bpn = boost::python::numeric;



//MARK: ---------- WRAPPER FOR TIP DATA ---------------------

struct pyTipData{

    TipData tip;

    pyTipData(){

    }

    pyTipData(double m, const bpn::array &cm, const bpn::array &I, const bpn::array &F, const bpn::array &M){

        tip.m = m;

        tip.cm_offsetX = bp::extract<double>(cm[0]);
        tip.cm_offsetY = bp::extract<double>(cm[1]);
        tip.cm_offsetZ = bp::extract<double>(cm[2]);

        tip.Ixx = bp::extract<double>(I[0]);
        tip.Iyy = bp::extract<double>(I[1]);
        tip.Izz = bp::extract<double>(I[2]);
        tip.Ixy = bp::extract<double>(I[3]);
        tip.Ixz = bp::extract<double>(I[4]);
        tip.Iyz = bp::extract<double>(I[5]);

        tip.Fx = bp::extract<double>(F[0]);
        tip.Fy = bp::extract<double>(F[1]);
        tip.Fz = bp::extract<double>(F[2]);

        tip.Mx = bp::extract<double>(M[0]);
        tip.My = bp::extract<double>(M[1]);
        tip.Mz = bp::extract<double>(M[2]);

    }




};



//MARK: ---------- WRAPPER FOR BASE DATA ---------------------

struct pyBaseData{

    BaseData base;

    // free end
    pyBaseData(){

    }

    pyBaseData(const bpn::array &k, double infinity){

        for (int i = 0; i < 6; i++) {
            base.k[i] = bp::extract<double>(k[i]);
            base.rigid[i] = (base.k[i] == infinity);
        }
    }

};


//MARK: ---------- WRAPPER FOR STRUCTURAL DATA ---------------------

struct pySectionData {

    SectionData sec;

    pySectionData(int np, const bpn::array &z_np, const bpn::array &EA_np,
                const bpn::array &EIxx_np, const bpn::array &EIyy_np, const bpn::array &GJ_np,
                const bpn::array &rhoA_np, const bpn::array &rhoJ_np){

        sec.nodes = np;

        sec.z.resize(sec.nodes);
        sec.EA.resize(sec.nodes);
        sec.EIxx.resize(sec.nodes);
        sec.EIyy.resize(sec.nodes);
        sec.GJ.resize(sec.nodes);
        sec.rhoA.resize(sec.nodes);
        sec.rhoJ.resize(sec.nodes);


        for (int i = 0 ; i < sec.nodes; i++){
            sec.z(i) = bp::extract<double>(z_np[i]);
            sec.EA(i) = bp::extract<double>(EA_np[i]);
            sec.EIxx(i) = bp::extract<double>(EIxx_np[i]);
            sec.EIyy(i) = bp::extract<double>(EIyy_np[i]);
            sec.GJ(i) = bp::extract<double>(GJ_np[i]);
            sec.rhoA(i) = bp::extract<double>(rhoA_np[i]);
            sec.rhoJ(i) = bp::extract<double>(rhoJ_np[i]);
        }



    }
};

struct pyPolynomialSectionData {

    PolynomialSectionData sec;

    pyPolynomialSectionData(int nodes, const bpn::array &z_np, const bpn::array &nA_np, const bpn::array &nI_np,
                            const bp::list &EA_list, const bp::list &EIxx_list, const bp::list &EIyy_list,
                            const bp::list &GJ_list, const bp::list &rhoA_list, const bp::list &rhoJ_list){

        sec.nodes = nodes;

        sec.z.resize(sec.nodes);
        sec.EA.resize(sec.nodes-1);
        sec.EIxx.resize(sec.nodes-1);
        sec.EIyy.resize(sec.nodes-1);
        sec.GJ.resize(sec.nodes-1);
        sec.rhoA.resize(sec.nodes-1);
        sec.rhoJ.resize(sec.nodes-1);


        for (int i = 0; i < sec.nodes; i++){
            sec.z(i) = bp::extract<double>(z_np[i]);
        }


        for (int i = 0; i < sec.nodes-1; i++){
            int nA = bp::extract<double>(nA_np[i]);
            int nI = bp::extract<double>(nI_np[i]);
            double EA_poly[nA];
            double rhoA_poly[nA];

            for (int j = 0; j < nA; j++){
                EA_poly[j] = bp::extract<double>(EA_list[i][j]);
                rhoA_poly[j] = bp::extract<double>(rhoA_list[i][j]);
            }
            sec.EA(i) = Poly(nA, EA_poly);
            sec.rhoA(i) = Poly(nA, rhoA_poly);

            double EIxx_poly[nI];
            double EIyy_poly[nI];
            double GJ_poly[nI];
            double rhoJ_poly[nI];

            for (int j = 0; j < nI; j++){
                EIxx_poly[j] = bp::extract<double>(EIxx_list[i][j]);
                EIyy_poly[j] = bp::extract<double>(EIyy_list[i][j]);
                GJ_poly[j] = bp::extract<double>(GJ_list[i][j]);
                rhoJ_poly[j] = bp::extract<double>(rhoJ_list[i][j]);
            }
            sec.EIxx(i) = Poly(nI, EIxx_poly);
            sec.EIyy(i) = Poly(nI, EIyy_poly);
            sec.GJ(i) = Poly(nI, GJ_poly);
            sec.rhoJ(i) = Poly(nI, rhoJ_poly);

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


        loads.Px.clear();
        loads.Py.clear();
        loads.Pz.clear();
        loads.Fx.clear();
        loads.Fy.clear();
        loads.Fz.clear();
        loads.Mx.clear();
        loads.My.clear();
        loads.Mz.clear();


    }

    pyLoads(int np, const bpn::array &Px_np, const bpn::array &Py_np,
            const bpn::array &Pz_np){

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


        for (int i = 0 ; i < loads.nodes; i++){
            loads.Px(i) = bp::extract<double>(Px_np[i]);
            loads.Py(i) = bp::extract<double>(Py_np[i]);
            loads.Pz(i) = bp::extract<double>(Pz_np[i]);
        }

        loads.Fx.clear();
        loads.Fy.clear();
        loads.Fz.clear();
        loads.Mx.clear();
        loads.My.clear();
        loads.Mz.clear();

    }

    pyLoads(int np, const bpn::array &Px_np, const bpn::array &Py_np, const bpn::array &Pz_np,
            const bpn::array &Fx_np, const bpn::array &Fy_np, const bpn::array &Fz_np,
            const bpn::array &Mx_np, const bpn::array &My_np, const bpn::array &Mz_np){

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


        for (int i = 0 ; i < loads.nodes; i++){
            loads.Px(i) = bp::extract<double>(Px_np[i]);
            loads.Py(i) = bp::extract<double>(Py_np[i]);
            loads.Pz(i) = bp::extract<double>(Pz_np[i]);
            loads.Fx(i) = bp::extract<double>(Fx_np[i]);
            loads.Fy(i) = bp::extract<double>(Fy_np[i]);
            loads.Fz(i) = bp::extract<double>(Fz_np[i]);
            loads.Mx(i) = bp::extract<double>(Mx_np[i]);
            loads.My(i) = bp::extract<double>(My_np[i]);
            loads.Mz(i) = bp::extract<double>(Mz_np[i]);
        }


    }
};


struct pyPolynomialLoads {

    PolynomialLoads loads;

    pyPolynomialLoads(int nodes, const bpn::array &nP_np, const bp::list &Px_list, const bp::list &Py_list, const bp::list &Pz_list,
            const bpn::array &Fx_np, const bpn::array &Fy_np, const bpn::array &Fz_np,
            const bpn::array &Mx_np, const bpn::array &My_np, const bpn::array &Mz_np){

        loads.nodes = nodes;

        loads.Px.resize(loads.nodes-1);
        loads.Py.resize(loads.nodes-1);
        loads.Pz.resize(loads.nodes-1);
        loads.Fx.resize(loads.nodes);
        loads.Fy.resize(loads.nodes);
        loads.Fz.resize(loads.nodes);
        loads.Mx.resize(loads.nodes);
        loads.My.resize(loads.nodes);
        loads.Mz.resize(loads.nodes);


        for (int i = 0; i < loads.nodes; i++){
            loads.Fx(i) = bp::extract<double>(Fx_np[i]);
            loads.Fy(i) = bp::extract<double>(Fy_np[i]);
            loads.Fz(i) = bp::extract<double>(Fz_np[i]);
            loads.Mx(i) = bp::extract<double>(Mx_np[i]);
            loads.My(i) = bp::extract<double>(My_np[i]);
            loads.Mz(i) = bp::extract<double>(Mz_np[i]);
        }

        for (int i = 0; i < loads.nodes-1; i++){
            int nP = bp::extract<double>(nP_np[i]);
            double Px_poly[nP];
            double Py_poly[nP];
            double Pz_poly[nP];

            for (int j = 0; j < nP; j++){
                Px_poly[j] = bp::extract<double>(Px_list[i][j]);
                Py_poly[j] = bp::extract<double>(Py_list[i][j]);
                Pz_poly[j] = bp::extract<double>(Pz_list[i][j]);
            }
            loads.Px(i) = Poly(nP, Px_poly);
            loads.Py(i) = Poly(nP, Py_poly);
            loads.Pz(i) = Poly(nP, Pz_poly);

        }

    }
};


//MARK: ---------- WRAPPER FOR BEAM ---------------------



class pyBEAM {


    Beam *beam;

public:



// MARK: ---------------- CONSTRUCTORS --------------------------

    // cylindrical shell sections
    pyBEAM(int nodes, const bpn::array &z_np, const bpn::array &d_np, const bpn::array &t_np,
           const bp::object &loads_o, const bp::object &mat_o,
           const bp::object &tip_o, const bp::object &base_o){

        Vector z(nodes);
        Vector d(nodes);
        Vector t(nodes);

        for (int i = 0 ; i < nodes; i++){
            z(i) = bp::extract<double>(z_np[i]);
            d(i) = bp::extract<double>(d_np[i]);
            t(i) = bp::extract<double>(t_np[i]);
        }

        pyLoads &loads = bp::extract<pyLoads&>(loads_o);
        IsotropicMaterial& mat = bp::extract<IsotropicMaterial&>(mat_o);
        pyTipData& tip = bp::extract<pyTipData&>(tip_o);
        pyBaseData& base = bp::extract<pyBaseData&>(base_o);

        beam = new Beam(z, d, t, loads.loads, mat, tip.tip, base.base);
    }

    //general sections
    pyBEAM(const bp::object &section_o, const bp::object &loads_o,
           const bp::object &tip_o, const bp::object &base_o){

        pySectionData& sec = bp::extract<pySectionData&>(section_o);
        pyLoads &loads = bp::extract<pyLoads&>(loads_o);
        pyTipData& tip = bp::extract<pyTipData&>(tip_o);
        pyBaseData& base = bp::extract<pyBaseData&>(base_o);

        beam = new Beam(sec.sec, loads.loads, tip.tip, base.base);
    }

    //general sections as polynomials
    pyBEAM(const bp::object &section_o, const bp::object &loads_o,
           const bp::object &tip_o, const bp::object &base_o, int dummy){

        pyPolynomialSectionData& sec = bp::extract<pyPolynomialSectionData&>(section_o);
        pyPolynomialLoads &loads = bp::extract<pyPolynomialLoads&>(loads_o);
        pyTipData& tip = bp::extract<pyTipData&>(tip_o);
        pyBaseData& base = bp::extract<pyBaseData&>(base_o);

        beam = new Beam(sec.sec, loads.loads, tip.tip, base.base);
    }

//    pyBEAM(const pyBEAM &b, int nothing){
//        std::cout << "yo" << std::endl;
//        beam = new Beam(*(b.beam), 1);
//
//    }

    ~pyBEAM(){

        delete beam;
    }




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
    bpn::array computeNaturalFrequencies(int n){

        Vector vec(n);
        beam->computeNaturalFrequencies(n, vec);

        bp::list list;
        for(int i = 0; i < vec.size(); i++)
        {
            list.append(vec(i));
        }

        return bpn::array(list);

    }


    /**
     Compute the natural frequencies of the structure

     Arguments:
     n - number of natural frequencies to return (unless n exceeds the total DOF of the structure,
     in which case all computed natural frequencies will be returned.

     Return:
     freq - a numpy array of frequencies (not necessarily of length n as described above).

     **/
    bp::tuple computeNaturalFrequenciesAndEigenvectors(int n){

        Vector vec(n);
        Matrix mat(0, 0);
        beam->computeNaturalFrequencies(n, vec, mat);

        bp::list freq_list;
        for(int i = 0; i < vec.size(); i++) {
            freq_list.append(vec(i));
        }

        bp::list vec_list;
        for (int j = 0; j < mat.size2(); j++) {

            bp::list inner_list;
            for (int i = 0; i < mat.size1(); i++) {
                inner_list.append(mat(i, j));
            }
            vec_list.append(bpn::array(inner_list));
        }

        return bp::make_tuple(bpn::array(freq_list), bpn::array(vec_list));
    }


    /**
     Compute the displacements of the structure (in global coordinate system)

     Return:
     a tuple containing (x, y, z, theta_x, theta_y, theta_z).
     each entry of the tuple is a numpy array describing the deflections for the given
     degree of freedom at each node.

     **/
    bp::tuple computeDisplacement(){

        int nodes = beam->getNumNodes();

        Vector dx_v(nodes);
        Vector dy_v(nodes);
        Vector dz_v(nodes);
        Vector dtheta_x_v(nodes);
        Vector dtheta_y_v(nodes);
        Vector dtheta_z_v(nodes);

        beam->computeDisplacement(dx_v, dy_v, dz_v, dtheta_x_v, dtheta_y_v, dtheta_z_v);

        bp::list dx, dy, dz, dtx, dty, dtz;

        for(int i = 0; i < nodes; i++)
        {
            dx.append(dx_v(i));
            dy.append(dy_v(i));
            dz.append(dz_v(i));
            dtx.append(dtheta_x_v(i));
            dty.append(dtheta_y_v(i));
            dtz.append(dtheta_z_v(i));
        }

        return bp::make_tuple(bpn::array(dx), bpn::array(dy), bpn::array(dz), bpn::array(dtx), bpn::array(dty), bpn::array(dtz));

    }



    /**
     Estimates the minimum critical buckling loads due to axial loading in addition to any existing input loads.

     Return:
     a tuple (Pcr_x, Pcr_y) where
     Pcr_x - critical buckling load in the x-direction
     Pcr_y - critical buckling load in the y-direction

     **/
    bp::tuple computeCriticalBucklingLoads(){

        double Pcr_x, Pcr_y;

        beam->computeMinBucklingLoads(Pcr_x, Pcr_y);

        return bp::make_tuple(Pcr_x, Pcr_y);
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
    bpn::array computeAxialStrain(int length, const bpn::array &x_np, const bpn::array &y_np,
                                  const bpn::array &z_np){


        Vector x(length);
        Vector y(length);
        Vector z(length);
        // Vector E(length);
        Vector epsilon_axial(length);

        for (int i = 0 ; i < length; i++){
            x(i) = bp::extract<double>(x_np[i]);
            y(i) = bp::extract<double>(y_np[i]);
            z(i) = bp::extract<double>(z_np[i]);
            // E(i) = bp::extract<double>(E_np[i]);
        }

        beam->computeAxialStrain(x, y, z, epsilon_axial);

        bp::list list;
        for(int i = 0; i < epsilon_axial.size(); i++)
        {
            list.append(epsilon_axial(i));
        }

        return bpn::array(list);

    }


    // in global 3D coordinate system
    bp::tuple computeShearAndBending(){

        PolyVec Vx, Vy, Fz, Mx, My, Tz;
        beam->shearAndBending(Vx, Vy, Fz, Mx, My, Tz);


        bp::list Vx0, Vy0, Fz0, Mx0, My0, Tz0;

        int n = beam->getNumNodes() - 1;

        for(int i = 0; i < n; i++)
        {
            Vx0.append(Vx(i).eval(0.0));
            Vy0.append(Vy(i).eval(0.0));
            Fz0.append(Fz(i).eval(0.0));
            Mx0.append(-My(i).eval(0.0));  // translate back to global coordinates
            My0.append(Mx(i).eval(0.0));  // translate back to global coordinates
            Tz0.append(Tz(i).eval(0.0));
        }

        Vx0.append(Vx(n-1).eval(1.0));
        Vy0.append(Vy(n-1).eval(1.0));
        Fz0.append(Fz(n-1).eval(1.0));
        Mx0.append(-My(n-1).eval(1.0));  // translate back to global coordinates
        My0.append(Mx(n-1).eval(1.0));  // translate back to global coordinates
        Tz0.append(Tz(n-1).eval(1.0));


        return bp::make_tuple(bpn::array(Vx0), bpn::array(Vy0), bpn::array(Fz0), bpn::array(Mx0), bpn::array(My0), bpn::array(Tz0));

    }


};




// MARK: --------- BOOST MODULE ---------------


BOOST_PYTHON_MODULE(_pBEAM)
{

    struct beam_pickle : bp::pickle_suite{
        static bp::tuple getinitargs(const pyBEAM& b){

            return bp::make_tuple(b, 1);
        }
    };


    bpn::array::set_module_and_type("numpy", "ndarray");

    bp::class_<pyBEAM>("Beam", bp::init<int, bpn::array, bpn::array, bpn::array,
                       bp::object, bp::object, bp::object, bp::object>())
    .def(bp::init<bp::object, bp::object, bp::object, bp::object>())
    .def(bp::init<bp::object, bp::object, bp::object, bp::object, int>())
//    .def(bp::init<const pyBEAM&, int>())
    .def("mass", &pyBEAM::computeMass)
    .def("naturalFrequencies", &pyBEAM::computeNaturalFrequencies)
    .def("naturalFrequenciesAndEigenvectors", &pyBEAM::computeNaturalFrequenciesAndEigenvectors)
    .def("displacement", &pyBEAM::computeDisplacement)
    .def("criticalBucklingLoads", &pyBEAM::computeCriticalBucklingLoads)
    .def("axialStrain", &pyBEAM::computeAxialStrain)
    .def("outOfPlaneMomentOfInertia", &pyBEAM::computeOutOfPlaneMomentOfInertia)
    .def("shearAndBending", &pyBEAM::computeShearAndBending)
//    .def_pickle(beam_pickle())
    ;

    bp::class_<pyTipData>("TipData", bp::init<double, bpn::array, bpn::array, bpn::array, bpn::array>())
    .def(bp::init<>())
    ;

    bp::class_<pyBaseData>("BaseData", bp::init<bpn::array, double>())
    .def(bp::init<>())
    ;

    bp::class_<IsotropicMaterial>("Material", bp::init<double, double, double>())
    .def_readwrite("E", &IsotropicMaterial::E)
    .def_readwrite("G", &IsotropicMaterial::G)
    .def_readwrite("rho", &IsotropicMaterial::rho)
    ;

    bp::class_<pySectionData>("SectionData", bp::init<int, bpn::array, bpn::array, bpn::array,
        bpn::array, bpn::array, bpn::array, bpn::array>());

    bp::class_<pyPolynomialSectionData>("PolySectionData", bp::init<int, bpn::array, bpn::array,
        bpn::array, bp::list, bp::list, bp::list, bp::list, bp::list, bp::list>());

    bp::class_<pyLoads>("Loads", bp::init<int, bpn::array, bpn::array, bpn::array,
        bpn::array, bpn::array, bpn::array, bpn::array, bpn::array, bpn::array>())
    .def(bp::init<int, bpn::array, bpn::array, bpn::array>())
    .def(bp::init<int>())
    ;

    bp::class_<pyPolynomialLoads>("PolyLoads", bp::init<int, bpn::array, bp::list, bp::list, bp::list,
                        bpn::array, bpn::array, bpn::array, bpn::array, bpn::array, bpn::array>());
}
