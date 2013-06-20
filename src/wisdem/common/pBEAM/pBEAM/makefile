CXX = clang++
CFLAGS = -O2
INCL = -I./
OBJ = Poly.o myMath.o BeamFEA.o Beam.o
LIB = -framework Accelerate

all: $(OBJ)

%.o : %.cpp
	$(CXX) $(CFLAGS) $(INCL) -c $<

clean:
	rm -f *.o