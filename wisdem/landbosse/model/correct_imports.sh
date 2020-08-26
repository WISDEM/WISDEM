#!/bin/bash

sed -i -s 's/from landbosse/from wisdem\.landbosse/g' *.py

sed -i -s 's/from \.\./from wisdem\.landbosse\./g' *.py
sed -i -s 's/from \./from wisdem\.landbosse\.model\./g' *.py

sed -i -s 's/import landbosse/import wisdem\.landbosse/g' *.py
sed -i -s 's/import \.\./import wisdem\.landbosse\./g' *.py
sed -i -s 's/import \./import wisdem\.landbosse\.model\./g' *.py
