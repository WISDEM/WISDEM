# csmUtils.py
# 2012 04 02

import sys, os
import time

#------------------------------------------------

def dictRead(fname,debug=0):
    """  function 'dictRead' reads the elements of a dictionary from fname 
         and returns a dictionary
         Input file format:
           Lines starting with '#' are comments and are ignored
           Valid lines have 2 space-delimited fields:
             - vbl name
             - value
           If the vbl name starts with 'i' (lower case I), the value is read as integer.
           Otherwise, it's read as float.
           (No provision for reading text strings as values)
    """
    
    try:
        infile = open(fname)
        lines = infile.read().split("\n")
        infile.close()
        if (debug > 0):
            sys.stdout.write ("dictRead: Opened and read %s\n" % fname)
    except:
        sys.stdout.write ("Error opening or reading %s\n" % fname)
        return None
    else:
        #sys.stdout.write ("Opened %s\n" % fname)
        pass
    
    myDict = {}    
    for line in lines:
        if (line.startswith('#')):
            continue
    
        words = line.split()
        if (len(words) < 1):
            continue
        ival = 1
        if (words[ival] == "=" and len(words) > 2): # some files have "keyword = value"
            ival = 2
        if (words[0].startswith('i')):
            myDict[words[0]] = int(words[ival])
        else:
            myDict[words[0]] = float(words[ival])
        if (debug > 0):
            key = words[0]
            val = str(myDict[key])
            print "{:12} {}".format( key,val )
    
    #if (debug > 0):
    #    sys.stdout.write ("dictRead: Read %s\n\n" % fname)
    return myDict
    
        
#------------------------------------------------

def dictAssign(orig,myDict,key,debug=0):
    """  function 'dictAssign' - if myDict[key] is defined, return that
         otherwise, return orig
    """
    if key in myDict:
        return myDict[key]
    else:
        return orig

#------------------------------------------------------------------

if __name__ == "__main__":

    # simple test of module
    

    def main():
        print ("Running %s at %s" % (sys.argv[0], time.asctime()) )
    
        dict = dictRead(sys.argv[1],debug=1)
        print "dict has {} keys".format(len(dict))
    
    main()
