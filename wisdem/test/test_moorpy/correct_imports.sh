#!/bin/bash

sed -i -s 's/from moorpy/from wisdem.moorpy/g' *.py
sed -i -s 's/import moorpy/import wisdem.moorpy/g' *.py
