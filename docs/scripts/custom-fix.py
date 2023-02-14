#!/usr/bin/env python
# encoding: utf-8
"""
custom-fix.py

Created by Andrew Ning on 2013-05-23.
Copyright (c) NREL. All rights reserved.
"""

from os import close, remove
from shutil import move
from tempfile import mkstemp

FLAG1 = "[width=5in]{distributedAeroLoads.pdf}"
FLAG2 = "[width=5in]{cp.pdf}"
FLAG3 = "Referring to \\citep{ning2013a-simple-soluti}, this parameter controls"


def fixit(path):
    oldfile = open(path)
    handle, temp_path = mkstemp()
    newfile = open(temp_path, "w")

    for line in oldfile:
        # ---------- Your Custom Replacements Go Here ------------------

        # if FLAG1 in line or FLAG2 in line:
        #     line = line.replace('5in', '4in')

        # if FLAG3 in line:
        #     line = line.replace('citep', 'cite')

        # --------------------------------------------------------------

        newfile.write(line)

    oldfile.close()
    newfile.close()
    close(handle)
    remove(path)
    move(temp_path, path)


if __name__ == "__main__":
    import sys

    path = sys.argv[1]
    fixit(path)
