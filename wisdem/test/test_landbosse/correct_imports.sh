#!/bin/bash

sed -i -s 's/from landbosse/from wisdem.landbosse/g' *.py

sed -i -s 's/import landbosse/import wisdem.landbosse/g' *.py
