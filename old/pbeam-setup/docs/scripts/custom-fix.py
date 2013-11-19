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

FLAG1 = '\\includegraphics[width=5in]{dynamics.pdf}'
FLAG2 = '\\includegraphics[width=5in]{strain.pdf}'
FLAG3 = '\\includegraphics[width=5in]{dy.pdf}'


def fixit(path):


    oldfile = open(path)
    handle, temp_path = mkstemp()
    newfile = open(temp_path, 'w')

    for line in oldfile:

        if FLAG1 in line:
            line = line.replace('5in', '3in')
        if FLAG2 in line or FLAG3 in line:
            line = line.replace('5in', '4in')

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


