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

FLAG = '[width=5in]{dynamics.pdf}'


def fixit(path):


    oldfile = open(path)
    handle, temp_path = mkstemp()
    newfile = open(temp_path, 'w')

    for line in oldfile:

        if FLAG in line:
            line = line.replace(FLAG, '[width=3in]{dynamics.pdf}')

        newfile.write(line)

    oldfile.close()
    newfile.close()
    close(handle)
    remove(path)
    move(temp_path, path)




if __name__ == '__main__':

    pass

    # import sys
    # path = sys.argv[1]
    # fixit(path)


