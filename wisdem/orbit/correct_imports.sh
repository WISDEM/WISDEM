#!/bin/bash

sed -i -s 's/from ORBIT/from wisdem.orbit/g' *.py
sed -i -s 's/from ORBIT/from wisdem.orbit/g' */*.py
sed -i -s 's/from ORBIT/from wisdem.orbit/g' */*/*.py
sed -i -s 's/from ORBIT/from wisdem.orbit/g' */*/*/*.py

sed -i -s 's/import ORBIT/import wisdem.orbit/g' *.py
sed -i -s 's/import ORBIT/import wisdem.orbit/g' */*.py
sed -i -s 's/import ORBIT/import wisdem.orbit/g' */*/*.py
sed -i -s 's/import ORBIT/import wisdem.orbit/g' */*/*/*.py

sed -i -s 's/from tests/from wisdem.test.test_orbit/g' *.py
sed -i -s 's/from tests/from wisdem.test.test_orbit/g' */*.py
sed -i -s 's/from tests/from wisdem.test.test_orbit/g' */*/*.py
sed -i -s 's/from tests/from wisdem.test.test_orbit/g' */*/*/*.py

sed -i -s 's/import tests/import wisdem.test.test_orbit/g' *.py
sed -i -s 's/import tests/import wisdem.test.test_orbit/g' */*.py
sed -i -s 's/import tests/import wisdem.test.test_orbit/g' */*/*.py
sed -i -s 's/import tests/import wisdem.test.test_orbit/g' */*/*/*.py
