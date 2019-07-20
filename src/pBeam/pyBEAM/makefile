CXX = clang++
CFLAGS = -O2 -g -fPIC -std=c++11
INCL = -I../pBEAM -I../include `python-config --includes`
OBJ = ../pBEAM/Poly.o ../pBEAM/myMath.o ../pBEAM/BeamFEA.o ../pBEAM/Beam.o ../pBEAM/CurveFEM.o

pybeam: pyBEAM.o
	$(CXX) $(INCL) $(CFLAGS) -dynamiclib -undefined dynamic_lookup -shared pyBeam.o $(OBJ) -o _pBEAM.so

.cpp.o:
	$(CXX) $(INCL) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o _pBEAM.so
