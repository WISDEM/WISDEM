#!/bin/bash

function clean {
    rm -f CHANGELOG.md
    rm -f MANIFEST.in
    rm -f MANIFEST
    rm -f setup.py
    rm -rf docs
    rm -rf test
    rm -rf build
    rm -rf *.egg-info
}

clean

mv README.md README_WISDEM.md

if [[ -n "$1" ]]; then

cp $1/README.md .
cp $1/CHANGELOG.md .
cp $1/setup.py .
cp -R $1/docs/_build/html docs
cp $1/docs/UserGuide.pdf docs/
cp MANIFEST.in.template MANIFEST.in

python setup.py sdist --formats=gztar,zip

mv README_WISDEM.md README.md

clean

fi



