
all: py test

py:
	cd pBEAM; make
	cd pyBEAM; make
	mv pyBEAM/_pBEAM.so .

test:
	cd pBEAM; make
	cd runTests; make
	mv runTests/tests .
	./tests

clean:
	cd pBEAM; make clean
	cd pyBEAM; make clean
	cd runTests; make clean
	rm -f _pBEAM.so tests