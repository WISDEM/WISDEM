#!/bin/bash

find . -name '*.py' | while read file; do
    sed -i -s 's/from ORBIT/from wisdem.orbit/g' $file
    sed -i -s 's/import ORBIT/import wisdem.orbit/g' $file
    sed -i -s 's/from tests/from wisdem.test.test_orbit/g' $file
    sed -i -s 's/import tests/import wisdem.test.test_orbit/g' $file
done

