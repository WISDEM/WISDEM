#!/bin/bash

function clean {
    rm -f README.md
    rm -f CHANGELOG.md
    rm -f MANIFEST.in
    rm -f MANIFEST
    rm -f setup.py
    rm -rf docs
    rm -f UserGuide.pdf
    rm -rf test
    rm -rf build
    rm -rf *.egg-info
}

clean

if [[ -n "$1" ]]; then

ln -s $1/README.md .
ln -s $1/CHANGELOG.md .
ln -s $1/setup.py .
cp -R $1/docs/_build/html docs
cp $1/docs/UserGuide.pdf docs/
cp MANIFEST.in.template MANIFEST.in

python setup.py sdist --formats=gztar,zip

clean

fi



