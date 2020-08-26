CC=clang #gcc
CFLAGS = -c -g -O2 -fPIC

OBS   = coordtrans.o HPGutil.o NRutil.o 
PYOBS = py_main.o py_io.o py_frame3dd.o py_eig.o py_HPGmatrix.o 
EOBS  = main.o frame3dd.o frame3dd_io.o eig.o HPGmatrix.o 

ifeq ($(OS),Windows_NT)
    ARCHFLAGS=-D WIN64
    LIB := _pyframe3dd.dll
    LDFLAGS=-shared
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Linux)
        ARCHFLAGS=-D LINUX
	LIB := _pyframe3dd.so
	LDFLAGS=-shared -Wl,-soname,$(LIB)
    endif
    ifeq ($(UNAME_S),Darwin)
        ARCHFLAGS=-D OSX
	LIB := _pyframe3dd.so
	LDFLAGS=-dynamiclib
    endif
endif

EXE = frame3dd

all: shared exec

%.o: %.c %.h
	$(CC) $(CFLAGS) -o $@ $< 

shared : $(OBS) $(PYOBS)
	$(CC) $(LDFLAGS) -o $(LIB) $(PYOBS) $(OBS)

exec : $(EOBS) $(OBS) 
	$(CC) -o $(EXE) $(EOBS) $(OBS) -lm

clean:
	/bin/rm -rf $(OBS) $(EOBS) $(PYOBS) $(LIB) $(EXE) *~ *.pyc

.PHONY: clean
