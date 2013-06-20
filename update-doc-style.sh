#!/bin/bash

if [[ -n "$1" ]]; then

cp docs-template/Makefile $1
cp docs-template/make.bat $1
cp -R docs-template/exts $1
cp -R docs-template/latex-style $1
cp -R docs-template/nrel-theme $1
mkdir -p $1/scripts
cp -R docs-template/scripts/latex-fix.py $1/scripts/

fi
