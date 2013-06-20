#!/usr/bin/env python
# encoding: utf-8
"""
latex-fix.py

Created by Andrew Ning on 2013-05-23.
Copyright (c) NREL. All rights reserved.
"""

import re
from tempfile import mkstemp
from shutil import move
from os import remove, close

GET_CAPTION_FLAG = 'TABLE CAPTION::'
INSERT_CAPTION_FLAG = '***PUT CAPTION HERE***'
CITE_FLAG = '\\citep{'


def fixit(path, flag):


    oldfile = open(path)
    handle, temp_path = mkstemp()
    newfile = open(temp_path, 'w')

    for line in oldfile:
        # get rid of left over reference numbers
        line = re.sub('\{\[\}[0-9]+\{\]\}', '', line)

        # put table captions into place
        if GET_CAPTION_FLAG in line:
            caption = line.split(GET_CAPTION_FLAG)[1].lstrip().rstrip()
            line = ''

        if INSERT_CAPTION_FLAG in line:
            line = line.replace(INSERT_CAPTION_FLAG, caption)

        # hack to fix citations that use the name right before
        if flag == '--citefix' and CITE_FLAG in line:
            matches = re.findall('(?:\S+\s)?\S*\\' + CITE_FLAG, line)
            for match in matches:
                if match.split()[0][0].isupper():  # capitalized name before citation
                    line = line.replace(match, '\\cite{')


        newfile.write(line)

    oldfile.close()
    newfile.close()
    close(handle)
    remove(path)
    move(temp_path, path)




if __name__ == '__main__':

    import sys
    path = sys.argv[1]
    flag = None
    if len(sys.argv) > 2:
        flag = sys.argv[2]
    fixit(path, flag)
