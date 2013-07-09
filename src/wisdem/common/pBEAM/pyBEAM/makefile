CXX = clang++
CFLAGS = -O2
PY_FRAMEWORK = /usr/local/Cellar/python/2.7.5/Frameworks/Python.framework/Versions/Current
BOOST_INCL = /usr/local/include
BOOST_LIB = /usr/local/lib
INCL = -I$(BOOST_INCL) -I$(PY_FRAMEWORK)/include/python2.7 -I../pBEAM
LIB = $(BOOST_LIB)/libboost_python-mt.a $(BOOST_LIB)/libboost_system-mt.a $(PY_FRAMEWORK)/Python -framework accelerate
OBJ = ../pBEAM/Poly.o ../pBEAM/myMath.o ../pBEAM/BeamFEA.o ../pBEAM/Beam.o

pybeam: pyBEAM.o
	$(CXX) -dynamiclib $(LIB) pyBeam.o $(OBJ) -o _pBEAM.so

.cpp.o:
	$(CXX) $(INCL) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o _pBEAM.so