CXX = clang++
CFLAGS = -O2 -g -fPIC -std=c++11
INCL = -I ../include
OBJ = Poly.o myMath.o BeamFEA.o Beam.o CurveFEM.o

all: $(OBJ)
	$(CXX) $(CFLAGS) $(INCL) `python-config --includes` -c pyBEAM.cpp -o pyBEAM.o
	$(CXX) $(CFLAGS) $(INCL) `python-config --includes` -dynamiclib -undefined dynamic_lookup -shared pyBEAM.o $(OBJ) -o _pBEAM.so

%.o : %.cpp
	$(CXX) $(CFLAGS) $(INCL) -c $<

clean:
	rm -f *.o _pBEAM.so



