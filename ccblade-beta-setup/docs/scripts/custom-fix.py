#!/usr/bin/env python
# encoding: utf-8
"""
custom-fix.py

Created by Andrew Ning on 2013-05-23.
Copyright (c) NREL. All rights reserved.
"""

from tempfile import mkstemp
from shutil import move
from os import remove, close
import re

FLAG = '\chapter{Coordinate System}'


def fixit(path):

    oldfile = open(path)
    handle, temp_path = mkstemp()
    newfile = open(temp_path, 'w')

    for line in oldfile:

        if FLAG in line:
            line = line.replace('chapter', 'chapter*')
            line += '\n\\addcontentsline{toc}{chapter}{Coordinate System}'

        if '\section' in line and '-aligned' in line:
            line = line.replace('section', 'section*')

        if '\includegraphics' in line:
            line = re.sub('\d+\.?\d*in', '3.5in', line)

        if '\includegraphics[width=3.5in]{blade_airfoil.pdf}' in line:
            line = line.replace('3.5in', '4.0in')

        if '\includegraphics[width=3.5in]{azimuth_blade.pdf}' in line:
            line = line.replace('3.5in', '2.25in')

        if '\includegraphics[width=3.5in]{yaw_hub.pdf}' in line:
            line = line.replace('3.5in', '2.5in')


        newfile.write(line)

    oldfile.close()
    newfile.close()
    close(handle)
    remove(path)
    move(temp_path, path)




if __name__ == '__main__':

    import sys
    path = sys.argv[1]
    fixit(path)


