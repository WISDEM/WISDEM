#-------------------------------------------------------------------------------
# Name:        GetSheetByName.py
# Purpose:     This module contains a function to find index of sheet in excel file that corresponds to a given name
#
# Author:      rdamiani
#
# Created:     6/01/2014 - Based on stackoverflow
# Copyright:   (c) rdamiani 2014
# Licence:     <Apache>
#-------------------------------------------------------------------------------
import itertools
#______________________________________________________________________________#
def get_sheet_idx(book, name):
    """Get a sheet by name from xlwt.Workbook, a strangely missing method.
    Returns None if no sheet with the given name is present.
    """
    # Note, we have to use exceptions for flow control because the
    # xlwt API is broken and gives us no other choice.
    try:
        for idx in itertools.count():
            sheet = book.get_sheet(idx)
            if sheet.name == name:
                return idx
    except IndexError:
        return None
#______________________________________________________________________________#

if __name__ == '__main__':

    get_sheet_idx('test.xls','testsheet')
