#!/usr/bin/python

"""
csmTurbine.py

Created by George Scott on 2012-08-01.
Copyright (c) NREL. All rights reserved.
"""
import sys, os
import re
import time
from math import *
from socket import gethostname

#------------------------------------------------

class PPITbl:
    ''' a PPITbl object represents a cost table '''
    
    def __init__(self,code="",name="PPITbl"):
        self.cost = [] # empty list
        self.years = []
        self.name = name
        self.code = code
        pass
        
    def add_row(self,year,cost_array):
        self.cost.append(cost_array)
        self.years.append(year) 
        return 1
        
    def getEsc(self,start_yr,start_mon,end_yr,end_mon):
        ''' return cost escalator between two dates (mon==13 is annual value) '''
        start_row = start_yr-self.years[0]
        end_row   = end_yr-self.years[0]
        
        if (len(self.cost[start_row]) < start_mon):
            raise IndexError("Start_mon out of range")
        if (len(self.cost[end_row]) < end_mon):
            print "EM %d > LER %d in table %s" % (end_mon, len(self.cost[end_row]), self.code)
            raise IndexError("End_mon out of range")
            
        try:
            print "SR %d ER %d" % (start_row,end_row)
            print "LSR %d LER %d" % (len(self.cost[start_row]),len(self.cost[end_row]))
            cost_start = self.cost[start_row][start_mon-1]
            cost_end   = self.cost[end_row][end_mon-1]
        except:
            print "Index out of range for table %s %s" % (self.code, self.name)
            return None
        esc = cost_end / cost_start
        print "C[%d][%d] = %.2f  C[%d][%d] = %.2f  Esc = %.4f" % (start_yr,start_mon,cost_start,end_yr,end_mon,cost_end,esc)
        return esc

#------------------------------------------------

def readfile(fname):
    """ This function reads a file and creats an array of PPITbls """
    try:
        infile = open(fname)
    except:
        sys.stdout.write ("Error opening or reading %s\n" % fname)
        return 0
    else:
        sys.stdout.write ("Opened %s\n" % fname)
        
    word_count = 0
    itable = -1
    found_tables = False
    ppitbls = [] # list of PPITbl objects
    iCode   = {}
    
    for line in infile:
        words = line.split("\t")
        if not words:
            continue
            
        if (words[0].startswith("NAICS")):
            found_tables = True
            itable += 1
            iyr = 0
            
            words[2] = words[2].replace(r'"', '')  # strip quotes from name
            ppitbls.append(PPITbl(code=words[1], name=words[2]))  # add a new element to ppitbls
            ippi = len(ppitbls)-1
            print "Created %d %s" % (ippi, ppitbls[ippi].name)
            iCode[words[1]] = ippi  # index of tables by code 
            
        if (found_tables and words[0].startswith("20")): # a year number
            #print "itable %d word %s line %s N(words) %d" % (itable,words[0],line, len(words))
            rvals = []
            i = 1
            while (re.match(r"\d+\.",words[i])):
                rvals.append(float(words[i]))
                i += 1
            #print rvals
            ppitbls[ippi].add_row(int(words[0]), rvals)
            iyr += 1
    
    # Testing
    
    itable = 2 
    print "Table %d %s NYrs %d" % (itable,ppitbls[itable].name,len(ppitbls[itable].years))
    #for iyr in range(len(ppitbls[itable].years)):
    #    print "%4d " % ppitbls[itable].years[iyr],
    #    for imon in range(len(ppitbls[itable].cost[iyr])):
    #        print ppitbls[itable].cost[iyr][imon]
    #    print
    
    for i in range(len(ppitbls)):
        print "%2d %-10s %s" % (i, ppitbls[i].code, ppitbls[i].name)
    
    ppitbls[2].getEsc (2003,1,2011,4)
    
    code = "332991P"
    #code = "abcde"
    if code in iCode:
        print "%2d %s" % (iCode[code], code)
    else:
        print "No table found for code '%s'" % code
    
    # Compute a weighted average of escalators from various tables
    
    sy = 2002; sm = 9;
    ey = 2010; em = 13;       
    codes = ['3272123', '3255204', '332722489', '326150P'] # c[0,1] are wrong in spreadsheet
    cpcts = [ 0.60, 0.23, 0.08, 0.09 ]
    #codes = ['332991P', '3353123', '333612P', '334513']
    #cpcts = [ 0.50, 0.20, 0.20, 0.10 ]
    sum = 0
    for i in range(len(codes)):
        itable = iCode[codes[i]]
        esc = ppitbls[itable].getEsc (sy,sm,ey,em)
        sum += esc * cpcts[i]
    print "Avg Esc = %.4f" % sum
    
    return 1
        
#--------------------- MAIN ------------------

def main():
    print ("Running %s at %s on %s" % (sys.argv[0], time.asctime(), gethostname()) )
    
    print time.asctime()
    
    if (len(sys.argv) == 1):
        print ("\nUSAGE: {0} file\n".format(sys.argv[0]))
        sys.exit()
        
    readfile(sys.argv[1])

if __name__ == "__main__":

    main()
