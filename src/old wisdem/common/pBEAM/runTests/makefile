CXX = clang++
CFLAGS = -O2
INCL = -I../pBEAM
LIB = -lboost_unit_test_framework-mt -llapack
CORE_OBJ = ../pBEAM/Poly.o ../pBEAM/myMath.o ../pBEAM/BeamFEA.o ../pBEAM/Beam.o
OBJ = runTests.o testBeam.o testBeamFEA.o testMath.o testPoly.o


all: $(OBJ)
	$(CXX) $(LIB) $(CORE_OBJ) $(OBJ) -o tests

%.o : %.cpp
	$(CXX) $(CFLAGS) $(INCL) -c $<

clean:
	rm -f *.o tests